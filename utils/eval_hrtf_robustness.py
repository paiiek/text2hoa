# -*- coding: utf-8 -*-
# save as: eval_hrtf_robustness.py
import os, json, math, argparse, random, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

EPS=1e-7
random.seed(42); torch.manual_seed(42); np.random.seed(42)

# ---------- util ----------
def prefix_texts(texts):
    out=[]
    for t in texts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

def masked_mean(hs, attn):
    mask = attn.unsqueeze(-1).type_as(hs)
    return (hs*mask).sum(1)/mask.sum(1).clamp_min(1.0)

def angle_err_deg(yhat, ytrue):
    s1,c1, el1 = yhat[:,0],yhat[:,1],yhat[:,2]
    s2,c2, el2 = ytrue[:,0],ytrue[:,1],ytrue[:,2]
    u1 = torch.stack([torch.cos(el1)*c1, torch.cos(el1)*s1, torch.sin(el1)], -1)
    u2 = torch.stack([torch.cos(el2)*c2, torch.cos(el2)*s2, torch.sin(el2)], -1)
    dot = (u1*u2).sum(-1).clamp(-1+EPS,1-EPS)
    return torch.rad2deg(torch.acos(dot))

def az_deg_from_sc(s, c):
    az = math.degrees(math.atan2(float(s), float(c)))
    if az < 0: az += 360.0
    return az

def az12_from_sc(s,c):
    az = az_deg_from_sc(s,c)
    return int(az // 30)

def rows_from_jsonl(path, room_mode="drr"):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x=json.loads(line)
            room = x["room_depth"].get("drr_db") if room_mode=="drr" else x["room_depth"].get("rt60_s")
            if room is None: continue
            y = torch.tensor([
                x["az_sc"][0], x["az_sc"][1], x["el_rad"], x["dist_m"],
                x["spread_deg"], x["wet_mix"], x["gain_db"], room
            ], dtype=torch.float32)
            a12 = az12_from_sc(x["az_sc"][0], x["az_sc"][1])
            rows.append({"text": x["text"], "y": y, "a12": a12})
    return rows

# ---------- model (inference-only) ----------
class T2SModel(nn.Module):
    def __init__(self, enc_name, el_max=1.0472):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(enc_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(enc_name)
        d = self.enc.config.hidden_size
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 768), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(768, 384), nn.ReLU(),
            nn.Linear(384, 8)
        )
        self.EL_MAX = el_max

    def encode(self, texts, device):
        bt=self.tok(prefix_texts(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        bt={k:v.to(device) for k,v in bt.items()}
        hs=self.enc(**bt).last_hidden_state
        h=masked_mean(hs, bt["attention_mask"])
        return F.normalize(self.norm(h), dim=-1)

    @torch.no_grad()
    def forward(self, texts, device):
        h=self.encode(texts, device=device)
        y_raw=self.head(h)
        s=y_raw[:,0]; c=y_raw[:,1]; n=torch.sqrt(s*s+c*c+EPS)
        az0,az1=s/n, c/n
        el=torch.tanh(y_raw[:,2])*self.EL_MAX
        dist=torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread=torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet=torch.sigmoid(y_raw[:,5])
        gain=torch.tanh(y_raw[:,6])*6.0
        room=y_raw[:,7]  # 회귀 원시값(학습 시 변환 적용). 평가에서는 GT와 같은 스케일로 비교하지 않음.
        y=torch.stack([az0,az1,el,dist,spread,wet,gain,room],1)
        return y, h

# ---------- HRTF audio proxy ----------
def load_hrtf_npz(path):
    D=np.load(path, allow_pickle=True)
    az=np.array(D["az_deg"]).astype(float)
    el=np.array(D["el_deg"]).astype(float)
    hl=np.array(D["hrir_l"]).astype(float)
    hr=np.array(D["hrir_r"]).astype(float)
    fs=float(D["fs"]) if "fs" in D else 48000.0
    assert hl.shape==hr.shape and hl.shape[0]==az.shape[0]==el.shape[0]
    return {"az":az, "el":el, "hl":hl, "hr":hr, "fs":fs}

def nearest_hrir(hrtf, az_deg, el_deg):
    # 최소 각거리 방향 선택
    az=hrtf["az"]; el=hrtf["el"]
    # 원형 차이
    da=np.abs(((az - az_deg + 180) % 360) - 180)
    de=np.abs(el - el_deg)
    idx=np.argmin(da + 0.7*de)
    return hrtf["hl"][idx], hrtf["hr"][idx], hrtf["fs"]

def pink_noise(n, rng):
    # Voss-McCartney 간단 근사 대신 white 누적 필터로 근사
    x=rng.standard_normal(n).astype(np.float32)
    # 1/f 근사: 누적 평균 필터
    for k in [16,64,256]:
        x = (x + np.convolve(x, np.ones(k)/k, mode="same"))/2
    x=x/np.max(np.abs(x)+1e-9)
    return x

def convolve_stereo(x, hl, hr):
    L=len(hl); R=len(hr)
    yL=np.convolve(x, hl, mode="full")
    yR=np.convolve(x, hr, mode="full")
    n=min(len(yL), len(yR))
    y=np.stack([yL[:n], yR[:n]], axis=0)
    m=np.max(np.abs(y))+1e-9
    return (y/m).astype(np.float32)

def gcc_phat_itd(y, fs, max_tau=0.0015):
    # y: [2, T]
    x=y[0]; z=y[1]
    n=1
    while n < (len(x)+len(z)): n <<= 1
    X=np.fft.rfft(x, n); Z=np.fft.rfft(z, n)
    R = X*np.conj(Z)
    R = R / (np.abs(R)+1e-12)
    r = np.fft.irfft(R, n)
    max_shift = int(min(max_tau*fs, (n//2)-1))
    r = np.concatenate((r[-max_shift:], r[:max_shift+1]))
    shift = np.argmax(np.abs(r)) - max_shift
    tau = shift / float(fs)
    return tau

def itd_to_az_deg(tau, ear_dist=0.18, c=343.0):
    # tau = (ear_dist/c) * sin(az)
    s = np.clip((c*tau)/max(ear_dist,1e-6), -1.0, 1.0)
    az = math.degrees(math.asin(s))  # [-90,90]
    return az

def wrap180(x):
    y=((x+180)%360)-180
    return y

def audio_based_ae(az_true_deg, az_est_deg):
    # front/back 모호성 고려 → 180 래핑 최소
    diff = abs(wrap180(az_true_deg - az_est_deg))
    diff2 = abs(wrap180(az_true_deg - ((az_est_deg+180)%360)))
    return min(diff, diff2)

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--enc_model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--hrtf_npz", nargs="*", default=[], help="e.g., kemar_hrir.npz cipic_hrir.npz")
    ap.add_argument("--secs", type=float, default=0.6, help="noise duration for proxy")
    ap.add_argument("--out_csv", default="hrtf_robust_summary.csv")
    ap.add_argument("--per_bin_csv", default="")
    args=ap.parse_args()

    dev="cuda:0" if torch.cuda.is_available() else "cpu"
    rows=rows_from_jsonl(args.data, args.room_mode)
    texts=[r["text"] for r in rows]
    Y=torch.stack([r["y"] for r in rows]).to(dev)

    # model
    m=T2SModel(args.enc_model).to(dev)
    sd=torch.load(args.ckpt, map_location="cpu")
    sd=sd.get("state_dict", sd)
    _=m.load_state_dict(sd, strict=False)
    m.eval()

    # inference
    preds=[]
    with torch.no_grad():
        for i in range(0, len(texts), args.bsz):
            yhat,_ = m(texts[i:i+args.bsz], device=dev)
            preds.append(yhat)
    Yhat=torch.cat(preds,0).to(dev)

    # param-based AE
    AE_param = angle_err_deg(Yhat, Y).mean().item()

    # audio-based AE per HRTF
    rng=np.random.default_rng(123)
    results=[{"name":"param_only", "AE_deg": AE_param}]
    if args.hrtf_npz:
        # pre-load HRTFs
        banks=[]
        for p in args.hrtf_npz:
            try:
                banks.append((os.path.basename(p), load_hrtf_npz(p)))
            except Exception as e:
                print(f"[WARN] HRTF load fail: {p} ({e})")
        # generate one shared noise
        n=int((max([b[1]['fs'] for b in banks]) if banks else 48000)*args.secs)
        base_noise = pink_noise(n, rng)
        for name, H in banks:
            ae_list=[]
            for i in range(len(rows)):
                s,c,el = [t.item() for t in Yhat[i,:3]]
                az_pred = az_deg_from_sc(s,c)
                el_pred = math.degrees(float(el))
                hl, hr, fs = nearest_hrir(H, az_pred, el_pred)
                # resample noise if fs differs? (생략: 근사 사용)
                y = convolve_stereo(base_noise, hl, hr)
                tau = gcc_phat_itd(y, fs, max_tau=0.0015)
                az_est = itd_to_az_deg(tau)  # [-90,90]
                az_true = az_deg_from_sc(float(Y[i,0]), float(Y[i,1]))
                ae = audio_based_ae(az_true, az_est)
                ae_list.append(ae)
            results.append({"name":name, "AE_deg": float(np.mean(ae_list))})

    # write summary
    with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["name","AE_deg"])
        w.writeheader(); [w.writerow(r) for r in results]
    print("[HRTF robustness]", results)

    # optional per-bin
    if args.per_bin_csv:
        # az12-bin별 파라미터 AE
        aebins=[[] for _ in range(12)]
        AE_each=angle_err_deg(Yhat, Y).cpu().numpy().tolist()
        for i,r in enumerate(rows):
            aebins[r["a12"]].append(AE_each[i])
        with open(args.per_bin_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["az12_bin","N","AE_mean"])
            for k in range(12):
                arr=aebins[k]; w.writerow([k, len(arr), float(np.mean(arr)) if arr else None])

if __name__=="__main__":
    main()
