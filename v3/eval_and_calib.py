# eval_and_calib.py
import os, json, math, random, argparse
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_ID = "intfloat/multilingual-e5-small"
TARGET_KEYS = [
    "az_sin_start","az_cos_start","az_sin_end","az_cos_end",
    "el_sin_start","el_cos_start","el_sin_end","el_cos_end",
    "dist_z","width_start","width_end","wet_norm","gain_norm"
]

# --- embed (freeze) ---
tok = AutoTokenizer.from_pretrained(EMB_ID)
enc = AutoModel.from_pretrained(EMB_ID).to(DEVICE).eval()
@torch.no_grad()
def embed_text(texts):
    inp = tok([f"query: {t}" for t in texts], return_tensors="pt",
              padding=True, truncation=True, max_length=128).to(DEVICE)
    out = enc(**inp); mask = inp["attention_mask"].unsqueeze(-1)
    x = (out.last_hidden_state*mask).sum(1)/mask.sum(1)
    return nn.functional.normalize(x, p=2, dim=-1)

# --- tiny model head (must match training) ---
class Regressor(nn.Module):
    def __init__(self, in_dim=384, out_dim=len(TARGET_KEYS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.net(x)

def load_ckpt(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE)
    model = Regressor(out_dim=len(ck["target_keys"])).to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck

# --- helpers ---
def unitize(s, c, eps=1e-8):
    r = math.sqrt(s*s + c*c) + eps
    return s/r, c/r

def ang_from_sc(s, c):
    s, c = unitize(s, c)
    ang = math.degrees(math.atan2(s, c))
    # wrap to [-180, 180]
    if ang <= -180: ang += 360
    if ang > 180: ang -= 360
    return ang

def inv_dist_from_z(z):
    # [-2..2] -> [0.2..10] m
    z01 = (z + 2.0)/4.0
    return math.exp(z01*(math.log(10.0)-math.log(0.2)) + math.log(0.2))

def clamp01(x): return max(0.0, min(1.0, x))

# --- evaluation on a random slice of the train files (masked MAE/MSE) ---
def eval_random(files, ckpt, nsamp=512):
    # parse jsonl quickly
    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "y" in obj and "mask" in obj:
                    y = {k: float(obj["y"].get(k, 0.0)) for k in TARGET_KEYS}
                    m = {k: float(obj["mask"].get(k, 0.0)) for k in TARGET_KEYS}
                else:
                    # v3 transform (minimal: only keys we know)
                    y, m = {}, {}
                    if "az_sc" in obj and len(obj["az_sc"])==2:
                        s, c = float(obj["az_sc"][0]), float(obj["az_sc"][1])
                        y.update({"az_sin_start":s,"az_cos_start":c,"az_sin_end":s,"az_cos_end":c})
                    if "el_rad" in obj:
                        el = float(obj["el_rad"]); se, ce = math.sin(el), math.cos(el)
                        y.update({"el_sin_start":se,"el_cos_start":ce,"el_sin_end":se,"el_cos_end":ce})
                    if "dist_m" in obj:
                        d = max(0.2, min(10.0, float(obj["dist_m"])))
                        z01 = (math.log(d) - math.log(0.2)) / (math.log(10.0) - math.log(0.2))
                        y["dist_z"] = 4.0*z01 - 2.0
                    if "spread_deg" in obj:
                        w = max(0.0, min(90.0, float(obj["spread_deg"])))/90.0
                        y["width_start"]=w; y["width_end"]=w
                    if "wet_mix" in obj:
                        y["wet_norm"]=clamp01(float(obj["wet_mix"]))
                    if "gain_db" in obj:
                        lo,hi=-18.0,6.0; g=max(lo,min(hi,float(obj["gain_db"])))
                        y["gain_norm"]=(g-lo)/(hi-lo)
                    # mask: provided only
                    for k in TARGET_KEYS:
                        m[k] = 1.0 if k in y else 0.0
                        y.setdefault(k, 0.0)
                rows.append((obj.get("text",""), y, m))
    random.seed(0); random.shuffle(rows)
    rows = rows[:min(nsamp,len(rows))]

    model, ck = load_ckpt(ckpt)
    texts = [r[0] for r in rows]
    y_true = torch.tensor([[r[1][k] for k in TARGET_KEYS] for r in rows], dtype=torch.float32).to(DEVICE)
    m_true = torch.tensor([[r[2][k] for k in TARGET_KEYS] for r in rows], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        x = embed_text(texts)
        y_hat = model(x)

    # masked metrics
    diff = y_hat - y_true
    mse = ((diff**2)*m_true).sum(0)/(m_true.sum(0)+1e-8)
    mae = (diff.abs()*m_true).sum(0)/(m_true.sum(0)+1e-8)

    print("\n=== Masked metrics (random slice) ===")
    for i,k in enumerate(TARGET_KEYS):
        print(f"{k:>15s} | MAE {mae[i].item():.4f} | MSE {mse[i].item():.4f}")
    print(f"Overall (avg over present dims) | MAE {mae[m_true.sum(0)>0].mean().item():.4f} | "
          f"MSE {mse[m_true.sum(0)>0].mean().item():.4f}")

# --- inference + mapping demo ---
def demo(texts, ckpt, engine="spat"):
    model, ck = load_ckpt(ckpt)
    with torch.no_grad():
        x = embed_text(texts)
        y = model(x).cpu().tolist()

    for t, vec in zip(texts, y):
        p = {k:v for k,v in zip(TARGET_KEYS, vec)}
        # unitize sin/cos before angles
        az_s = ang_from_sc(p["az_sin_start"], p["az_cos_start"])
        az_e = ang_from_sc(p["az_sin_end"],   p["az_cos_end"])
        el_s = ang_from_sc(p["el_sin_start"], p["el_cos_start"])
        el_e = ang_from_sc(p["el_sin_end"],   p["el_cos_end"])

        dist_m = inv_dist_from_z(p["dist_z"])
        width_s = clamp01(p["width_start"])*90.0
        width_e = clamp01(p["width_end"])*90.0
        wet    = clamp01(p["wet_norm"])
        gain_db = -18.0 + clamp01(p["gain_norm"])*(6.0+18.0)

        if engine=="spat":
            mapped = {
              "azimuth_start_deg": az_s, "elevation_start_deg": el_s,
              "azimuth_end_deg": az_e,   "elevation_end_deg": el_e,
              "distance_m": dist_m, "spread_start_deg": width_s, "spread_end_deg": width_e,
              "wet": wet, "gain_db": gain_db
            }
        elif engine=="resonance":
            mapped = {"azimuth_deg": az_s, "elevation_deg": el_s,
                      "distance_m": dist_m, "reverb_mix": wet, "gain_db": gain_db}
        else:
            mapped = {"virtual_speaker_az_deg": az_s, "virtual_speaker_el_deg": el_s,
                      "spread_deg": width_s, "wet": wet, "gain_db": gain_db}

        print("\n[TEXT]", t)
        print("[RAW ]", json.dumps(p, ensure_ascii=False))
        print("[MAP ]", json.dumps(mapped, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval","demo"], required=True)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine", default="spat")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--nsamp", type=int, default=512)
    args = ap.parse_args()

    if args.mode=="eval":
        assert args.files, "--files 필요(jsonl들)"
        eval_random(args.files, args.ckpt, nsamp=args.nsamp)
    else:
        assert len(args.text)>0, "--text 문장 필요"
        demo(args.text, args.ckpt, engine=args.engine)
