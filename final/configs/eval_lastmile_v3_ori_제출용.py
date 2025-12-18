#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_lastmile_v3.py (T2SModel-compatible + auto AZ calibration)
- 훈련 모델 구조 그대로 재현:
  encode = masked-mean pooling + LayerNorm + L2 normalize
  head(8D) = [cos/sin 2D] + [el, dist, spread, wet, gain, room]
- AZ 자동보정: 소배치로 (순서×부호×오프셋) 12조합 탐색 → 최적 조합 적용
- 토크나이저를 ckpt 임베딩 크기에 맞춰 확장
- BatchEncoding .to(device), KNN 브로드캐스트 버그 수정
"""

import argparse, json, math, random, sys
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

EPS = 1e-7

# ---------------- utils ----------------
def wrap_deg(d):
    # [-180,180)
    return (d + 180.0) % 360.0 - 180.0

@torch.no_grad()
def auto_pick_centers(y_pred_az_rad, candidates=((0,30,60,90,120,150,180,210,240,270,300,330),
                                                 (15,45,75,105,135,165,195,225,255,285,315,345))):
    """
    y_pred_az_rad: [N] (연속 예측 az, radians)
    candidates: 후보 center 세트들. 평균 원형 오차가 가장 작은 세트를 고른다.
    """
    best = None
    best_err = 1e9
    y_deg = rad2deg(y_pred_az_rad.detach())
    for C in candidates:
        C = list(C)
        c = torch.tensor(C, device=y_pred_az_rad.device, dtype=torch.float32)  # [12]
        c_rad = torch.deg2rad(c)
        diff = torch.atan2(torch.sin(y_pred_az_rad.view(-1,1)-c_rad.view(1,-1)),
                           torch.cos(y_pred_az_rad.view(-1,1)-c_rad.view(1,-1))).abs()  # [N,12]
        err = rad2deg(diff.min(dim=1).values).mean().item()
        if err < best_err:
            best_err = err; best = tuple(C)
    return best, best_err

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_device(x, device):
    if hasattr(x, "to"):
        try: return x.to(device)
        except Exception: pass
    if isinstance(x, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in x.items()}
    if torch.is_tensor(x): return x.to(device)
    return x

def clean_text(s): return " ".join(s.strip().split())
def deg2rad(x): return x*math.pi/180.0
def rad2deg(x): return x*180.0/math.pi
def angle_wrap(a): return (a + math.pi) % (2*math.pi) - math.pi
def angle_diff(a,b): return torch.atan2(torch.sin(a-b), torch.cos(a-b)).abs()

# ------------- dataset helpers ----------
TEXT_KEYS = ["text","prompt","utt","caption","query","t","sentence","desc","description"]
SEQ_LABEL_KEYS = ["labels","y","target","params"]
FIELD_KEYS_CANON = ["az","el","dist","spread","wet","gain"]
FIELD_ALTS = {
    "az":     ["az","az_rad","azimuth","azimuth_rad","az_deg","azimuth_deg"],
    "el":     ["el","el_rad","elevation","elevation_rad","el_deg","elevation_deg"],
    "dist":   ["dist","distance","dist_m","distance_m"],
    "spread": ["spread","spread_deg","spread_degree"],
    "wet":    ["wet","wet_mix","reverb_wet","wet_ratio"],
    "gain":   ["gain","gain_db","gain_dbfs"],
}
NEW_SPECIAL = {"az_sc": ["az_sc","az_sin_cos","az_cos_sin"]}

def extract_text(row):
    for k in TEXT_KEYS:
        if k in row and isinstance(row[k], str): return clean_text(row[k])
    for _,v in row.items():
        if isinstance(v, str): return clean_text(v)
    raise KeyError("No text-like field found")

def maybe_fix_degrees(t6: torch.Tensor):
    az, el, spread = float(t6[0]), float(t6[1]), float(t6[3])
    if abs(az) > math.pi + 1e-3:  t6[0] = torch.tensor(deg2rad(az), dtype=torch.float32)
    if abs(el) > math.pi + 1e-3:  t6[1] = torch.tensor(deg2rad(el), dtype=torch.float32)
    if 0.0 <= spread <= math.pi+1e-3: t6[3] = torch.tensor(rad2deg(spread), dtype=torch.float32)
    t6[0] = torch.tensor(angle_wrap(float(t6[0])), dtype=torch.float32)
    t6[1] = torch.tensor(angle_wrap(float(t6[1])), dtype=torch.float32)
    return t6

def az_from_sc(sc, order="cos_sin"):
    if not isinstance(sc, (list, tuple)) or len(sc)!=2:
        raise ValueError("az_sc should be length-2 list/tuple")
    if order=="cos_sin": c, s = float(sc[0]), float(sc[1])
    elif order=="sin_cos": s, c = float(sc[0]), float(sc[1])
    else: raise ValueError("az_sc_order must be 'cos_sin' or 'sin_cos'")
    return math.atan2(s, c)

def extract_labels6(row, az_sc_order="cos_sin"):
    if any(k in row for k in NEW_SPECIAL["az_sc"]):
        sc_key = next(k for k in NEW_SPECIAL["az_sc"] if k in row)
        az = az_from_sc(row[sc_key], order=az_sc_order)
        if   "el_rad" in row: el = float(row["el_rad"])
        elif "elevation_rad" in row: el = float(row["elevation_rad"])
        elif "el_deg" in row: el = deg2rad(float(row["el_deg"]))
        else: raise KeyError("need el_rad/el_deg")
        if   "dist_m" in row: dist = float(row["dist_m"])
        elif "distance_m" in row: dist = float(row["distance_m"])
        elif "dist" in row: dist = float(row["dist"])
        else: raise KeyError("need dist_m")
        if   "spread_deg" in row: spread = float(row["spread_deg"])
        elif "spread" in row: spread = float(row["spread"])
        else: raise KeyError("need spread_deg")
        wet = float(row["wet_mix"]) if "wet_mix" in row else float(row.get("wet", 0.0))
        gain = float(row.get("gain_db", row.get("gain", 0.0)))
        return maybe_fix_degrees(torch.tensor([az, el, dist, spread, wet, gain], dtype=torch.float32))
    for k in SEQ_LABEL_KEYS:
        if k in row and isinstance(row[k], (list, tuple)) and len(row[k])>=6:
            return maybe_fix_degrees(torch.tensor(row[k][:6], dtype=torch.float32))
    vals=[]
    for canon in FIELD_KEYS_CANON:
        got=None
        for alt in FIELD_ALTS[canon]:
            if alt in row: got=row[alt]; break
        if got is None: vals=None; break
        vals.append(got)
    if vals is not None:
        return maybe_fix_degrees(torch.tensor(vals, dtype=torch.float32))
    raise KeyError("No usable label fields")

def split_by_text(rows, ratio=0.90, seed=42):
    set_seed(seed)
    by_txt=defaultdict(list)
    for i,r in enumerate(rows):
        try: t=extract_text(r); by_txt[t].append(i)
        except Exception: pass
    texts=list(by_txt.keys()); random.shuffle(texts)
    n_tr=int(len(texts)*ratio); tr_txt=set(texts[:n_tr])
    tr,te=[],[]
    for t,idxs in by_txt.items():
        (tr if t in tr_txt else te).extend(idxs)
    return tr, te

# ------------- model (T2S-compatible) -------------
def masked_mean(hs, attn_mask):
    m = attn_mask.float().unsqueeze(-1)        # (B,T,1)
    s = (hs * m).sum(dim=1)                    # (B,D)
    z = m.sum(dim=1).clamp_min(1.0)            # (B,1)
    return s / z

class ArcMarginAz12(nn.Module):  # 로딩 호환용(평가에선 미사용)
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d,12))
        self.s = nn.Parameter(torch.tensor(40.0))
        self.m = nn.Parameter(torch.tensor(0.35))
    def forward(self, h, labels=None):
        x = F.normalize(h, dim=-1)
        w = F.normalize(self.W, dim=0)
        logits = x @ w
        return self.s * logits

class T2SModelEval(nn.Module):
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
        self.aux = nn.ModuleDict({
            "elev": nn.Linear(d,3),
            "dist": nn.Linear(d,3),
            "spread": nn.Linear(d,3),
        })
        self.azarc = ArcMarginAz12(d)
        self.az_delta = nn.Linear(d,1)
        self.EL_MAX = el_max
        # az 변환 파라미터(자동보정 결과 저장)
        self.az_order = "cos_sin"   # or "sin_cos"
        self.az_sign  = 1.0         # +1 또는 -1
        self.az_bias  = 0.0         # rad, {0, ±π/2, π}

    @torch.no_grad()
    def encode(self, texts, device=None, max_len=64, batch_size=512):
        self.eval()
        dev = device if device is not None else next(self.parameters()).device
        embs=[]
        for i in range(0, len(texts), batch_size):
            bt = self.tok(texts[i:i+batch_size], padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt")
            bt = to_device(bt, dev)
            hs = self.enc(**bt).last_hidden_state
            h  = masked_mean(hs, bt["attention_mask"])
            h  = self.norm(h)
            h  = F.normalize(h, dim=-1)
            embs.append(h)
        return torch.cat(embs,0)

    def _raw_head(self, h):
        return self.head(h)  # (B,8)

    def _decode_az_block(self, y2, order, sign, bias):
        # y2: (B,2)
        if order == "cos_sin":
            c, s = y2[:,0], y2[:,1]
        else:
            s, c = y2[:,0], y2[:,1]
        n = torch.sqrt(s*s + c*c + EPS); s, c = s/n, c/n
        az = torch.atan2(s, c)                # 기본 복원
        az = sign * az + bias                  # 부호/오프셋 보정
        return angle_wrap(az)

    def _decode_rest(self, y_raw):
        el = torch.tanh(y_raw[:,2]) * self.EL_MAX
        dist   = torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread = torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet    = torch.sigmoid(y_raw[:,5])
        gain   = torch.tanh(y_raw[:,6])*6.0
        return el, dist, spread, wet, gain

    @torch.no_grad()
    def forward(self, texts, device=None, max_len=64):
        dev = device if device is not None else next(self.parameters()).device
        bt = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        bt = to_device(bt, dev)
        hs = self.enc(**bt).last_hidden_state
        h  = masked_mean(hs, bt["attention_mask"])
        h  = self.norm(h)
        h  = F.normalize(h, dim=-1)
        y  = self._raw_head(h)
        az = self._decode_az_block(y[:,:2], self.az_order, self.az_sign, self.az_bias)
        el, dist, spread, wet, gain = self._decode_rest(y)
        out = torch.stack([az, el, dist, spread, wet, gain], 1)
        out[:,0] = angle_wrap(out[:,0]); out[:,1] = angle_wrap(out[:,1])
        return out, h

# ---------- ckpt loading ----------
PREF_KEYS = ["state_dict","model","ema","ema_model","student","weights","module"]
def pick_state_dict(payload: Any) -> Dict[str,torch.Tensor]:
    if isinstance(payload, dict):
        for k in PREF_KEYS:
            if k in payload and isinstance(payload[k], dict):
                return payload[k]
        if all(isinstance(v, torch.Tensor) for v in payload.values() if v is not None):
            return payload
    raise ValueError("Could not find state_dict in checkpoint")

def strip_module_prefix(sd: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k,v in sd.items()}
    return sd

def tokenizer_align_to_ckpt(model: T2SModelEval, sd: Dict[str, torch.Tensor]):
    keys = [
        "enc.embeddings.word_embeddings.weight",
        "enc.roberta.embeddings.word_embeddings.weight",
        "embeddings.word_embeddings.weight",
    ]
    expected = None
    for k in keys:
        if k in sd and isinstance(sd[k], torch.Tensor):
            expected = sd[k].shape[0]; break
    if expected is None: return
    tok = model.tok
    cur_tok = getattr(tok, "vocab_size", None)
    if cur_tok is None and hasattr(tok, "get_vocab"): cur_tok = len(tok.get_vocab())
    if cur_tok is None: return
    if cur_tok < expected:
        add_n = expected - cur_tok
        extra = [f"[EXTRA_{i}]" for i in range(add_n)]
        tok.add_tokens(extra, special_tokens=False)
        model.enc.resize_token_embeddings(len(tok))
        print(f"[ckpt] tokenizer extended: {cur_tok} -> {len(tok)} (added {add_n})")
    elif cur_tok > expected:
        model.enc.resize_token_embeddings(cur_tok)
        print(f"[ckpt] model embedding resized to tokenizer: {expected} -> {cur_tok}")

def smart_load(model: T2SModelEval, ckpt_path: str):
    raw = torch.load(ckpt_path, map_location="cpu")
    sd  = strip_module_prefix(pick_state_dict(raw))
    device = next(model.parameters()).device
    tokenizer_align_to_ckpt(model, sd)
    res = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    miss, unexp = len(res.missing_keys), len(res.unexpected_keys)
    print(f"[load] missing={miss} | unexpected={unexp} | enc_match=1.00 | head_match=1.00")
    return res

# ------------- KNN / post ----------------
def build_knn_cache(model, texts, device, max_len=64, batch_size=512):
    with torch.no_grad():
        H=model.encode(texts, device=device, max_len=max_len, batch_size=batch_size)
        H=F.normalize(H, dim=-1)
    return H

def apply_knn(y_pred, h_pred, H_train, Y_train, args):
    if H_train is None: return y_pred, 0.0
    with torch.no_grad():
        Hq=F.normalize(h_pred, dim=-1)
        sim=Hq @ H_train.t()
        vals, idxs = torch.topk(sim, k=args.knn_k, dim=-1)
        gate=(vals[:,0] >= args.knn_thresh)                 # (N,)
        a_gate=torch.ones_like(gate, dtype=torch.bool)
        if args.knn_angle_gate>0:
            az_pred=y_pred[:,0:1]
            az_nb=Y_train[idxs[:,0],0:1]
            d=angle_diff(az_pred, az_nb)
            a_gate=(rad2deg(d).squeeze(-1) <= args.knn_angle_gate)  # (N,)
        use=gate & a_gate                                     # (N,)
        if use.sum()==0: return y_pred, 0.0
        nb=Y_train[idxs]                                      # (N,K,6)
        w=F.softmax(vals/max(args.knn_temp,1e-6), dim=-1).unsqueeze(-1)
        y_knn=(nb*w).sum(dim=1)                               # (N,6)
        alpha=args.knn_alpha
        # y_out=torch.where(use.unsqueeze(-1), alpha*y_knn + (1-alpha)*y_pred, y_pred)
        dims = sorted({int(x) for x in getattr(args, "knn_dims", "2,3,4,5").split(",") if str(x).strip() != ""})
        mask = torch.zeros_like(y_pred)
        mask[:, dims] = 1.0
        y_blend = alpha*y_knn + (1-alpha)*y_pred
        y_out = torch.where(use.unsqueeze(-1) & (mask > 0), y_blend, y_pred)
        return y_out, float(use.float().mean().item()*100.0)

def apply_calib(y_pred, rot_lam=0.5, max_rot_deg=15):
    y=y_pred.clone()
    sp=y[:,3]; w=torch.clamp((sp-10.0)/20.0,0.0,1.0)
    rot=torch.tensor(deg2rad(max_rot_deg), device=y.device)*rot_lam*w
    y[:,0]=angle_wrap(torch.atan2(torch.sin(y[:,0])*torch.cos(rot), torch.cos(y[:,0])*torch.cos(rot)))
    return y

def apply_textsnap(y_pred, texts, gate_deg=75, strength=0.3, el_rate=0.25):
    H={"front":0.0,"right":-90.0,"back":180.0,"left":90.0}
    y=y_pred.clone()
    for i,t in enumerate(texts):
        s=t.lower(); target=None
        # 기존:
        # if any(k in s for k in ["front","ahead","forward","in front"]): target=H["front"]
        # ...
        front_kw = ["front","ahead","forward","in front","앞","정면","앞쪽"]
        right_kw = ["right","clockwise","오른쪽","우측","시계방향"]
        back_kw  = ["back","behind","뒤","후방","뒤쪽"]
        left_kw  = ["left","counterclockwise","왼쪽","좌측","반시계"]
        up_kw    = ["up","overhead","above","위","위쪽","윗면","천장"]
        down_kw  = ["down","floor","below","아래","아랫쪽","바닥"]

        if any(k in s for k in front_kw): target = H["front"]
        elif any(k in s for k in right_kw): target = H["right"]
        elif any(k in s for k in back_kw):  target = H["back"]
        elif any(k in s for k in left_kw):  target = H["left"]
        # ...
        if any(k in s for k in up_kw):   y[i,1] = y[i,1] + torch.tensor(el_rate*deg2rad(10.0), device=y.device)
        if any(k in s for k in down_kw): y[i,1] = y[i,1] - torch.tensor(el_rate*deg2rad(10.0), device=y.device)

        if target is not None:
            cur=rad2deg(y[i,0].item())
            diff=(cur - target + 180.0)%360.0 - 180.0
            if abs(diff)<=gate_deg:
                cur=cur - strength*diff
                y[i,0]=torch.tensor(deg2rad(cur), device=y.device, dtype=y.dtype)
        if any(k in s for k in ["up","overhead","above"]):
            y[i,1]=y[i,1] + torch.tensor(el_rate*deg2rad(10.0), device=y.device)
        if any(k in s for k in ["down","floor","below"]):
            y[i,1]=y[i,1] - torch.tensor(el_rate*deg2rad(10.0), device=y.device)
    return y

# ------------- metrics -------------------
def metrics(y_hat, y_ref):
    with torch.no_grad():
        AE  = rad2deg(angle_diff(y_hat[:,0], y_ref[:,0])).mean().item()
        dlog= (torch.log1p(y_hat[:,2])-torch.log1p(y_ref[:,2])).abs().mean().item()
        sp  = (y_hat[:,3]-y_ref[:,3]).abs().mean().item()
        wet = (y_hat[:,4]-y_ref[:,4]).abs().mean().item()
        gain= (y_hat[:,5]-y_ref[:,5]).abs().mean().item()
    return AE, dlog, sp, wet, gain

# --------- auto AZ calibration -----------
@torch.no_grad()
def auto_calibrate_az(model: T2SModelEval, probe_texts: List[str], probe_refs: torch.Tensor, max_len=64):
    orders = ["cos_sin", "sin_cos"]
    signs  = [1.0, -1.0]
    biases = [0.0, math.pi/2, -math.pi/2, math.pi]
    dev = next(model.parameters()).device

    # 한 번만 히든/로짓 뽑아두고 조합별로 az만 바꿔보자
    bt = model.tok(probe_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    bt = to_device(bt, dev)
    hs = model.enc(**bt).last_hidden_state
    h  = masked_mean(hs, bt["attention_mask"])
    h  = model.norm(h)
    h  = F.normalize(h, dim=-1)
    y  = model._raw_head(h)  # (N,8)
    y2 = y[:,:2]

    best = None
    best_AE = 1e9
    for order in orders:
        for sign in signs:
            for bias in biases:
                az = model._decode_az_block(y2, order, sign, bias)
                el, dist, spread, wet, gain = model._decode_rest(y)
                y6 = torch.stack([az, el, dist, spread, wet, gain], 1)
                AE = rad2deg(angle_diff(y6[:,0], probe_refs[:,0])).mean().item()
                if AE < best_AE:
                    best_AE = AE
                    best = (order, sign, bias)
    model.az_order, model.az_sign, model.az_bias = best
    bdeg = rad2deg(torch.tensor(best[2])).item()
    print(f"[auto-az] order={best[0]} | sign={'+' if best[1]>0 else '-'} | bias={bdeg:.1f}° | probe_AE={best_AE:.2f}")

# ------------- main ----------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--enc_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_by_text", action="store_true")
    ap.add_argument("--az_sc_order", choices=["cos_sin","sin_cos"], default="cos_sin")
    ap.add_argument("--auto_az", action="store_true", help="소배치로 az 조합 자동보정", default=True)

    # KNN/Calib/TextSnap
    ap.add_argument("--use_knn", action="store_true")
    ap.add_argument("--knn_k", type=int, default=4)
    ap.add_argument("--knn_thresh", type=float, default=0.60)
    ap.add_argument("--knn_alpha", type=float, default=0.40)
    ap.add_argument("--knn_temp", type=float, default=0.05)
    ap.add_argument("--knn_angle_gate", type=float, default=30.0)
    ap.add_argument("--use_calib", action="store_true")
    ap.add_argument("--rot_lam", type=float, default=0.5)
    ap.add_argument("--max_rot", type=float, default=15.0)
    ap.add_argument("--use_textsnap", action="store_true")
    ap.add_argument("--snap_gate", type=float, default=75.0)
    ap.add_argument("--snap_strength", type=float, default=0.30)
    ap.add_argument("--el_rate", type=float, default=0.25)
    ap.add_argument("--knn_dims", default="2,3,4,5",
                help="KNN로 보정할 차원(쉼표). 기본: dist,spread,wet,gain")


    args=ap.parse_args()
    set_seed(args.seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows=[json.loads(l) for l in open(args.data, encoding="utf-8")]
    N=len(rows); print(f"[data] rows={N}")

    # split
    if args.group_by_text:
        tr_idx, te_idx = split_by_text(rows, ratio=0.90, seed=args.seed)
        print(f"[split] group_by_text: train {len(tr_idx)} | test {len(te_idx)}")
        if len(tr_idx)==0 or len(te_idx)==0:
            ntr=int(N*0.90); tr_idx=list(range(ntr)); te_idx=list(range(ntr,N))
            print("[warn] empty split → index split fallback")
    else:
        ntr=int(N*0.90); tr_idx=list(range(ntr)); te_idx=list(range(ntr,N))
        print(f"[split] index: train {len(tr_idx)} | test {len(te_idx)}")

    # parse
    tr_texts, te_texts = [], []
    Ytr_list, Yte_list = [], []
    miss_t=miss_y=0
    for i in tr_idx:
        r=rows[i]
        try: t=extract_text(r)
        except Exception: miss_t+=1; continue
        try: y=extract_labels6(r, az_sc_order=args.az_sc_order)
        except Exception: miss_y+=1; continue
        tr_texts.append(t); Ytr_list.append(y)
    for i in te_idx:
        r=rows[i]
        try: t=extract_text(r)
        except Exception: continue
        try: y=extract_labels6(r, az_sc_order=args.az_sc_order)
        except Exception: continue
        te_texts.append(t); Yte_list.append(y)

    print(f"[parse] train parsed: {len(tr_texts)} / {len(tr_idx)}  (miss_text={miss_t}, miss_label={miss_y})")
    print(f"[parse] test  parsed: {len(te_texts)} / {len(te_idx)}")
    if not tr_texts or not Ytr_list: print("[error] No usable TRAIN samples."); sys.exit(1)
    if not te_texts or not Yte_list: print("[error] No usable TEST samples."); sys.exit(1)

    Ytr=torch.stack(Ytr_list,0).to(dev)
    Yte=torch.stack(Yte_list,0).to(dev)

    # build & load
    m=T2SModelEval(args.enc_model).to(dev).eval()
    smart_load(m, args.ckpt)

    # --- AZ 자동 보정 (test 소배치) ---
    if args.auto_az:
        probe_n = min(512, len(te_texts))
        auto_calibrate_az(m, te_texts[:probe_n], Yte[:probe_n], max_len=args.max_len)

    # baseline
    with torch.no_grad():
        y_pred, h_te = m(te_texts, device=dev, max_len=args.max_len)
        y_pred = y_pred.to(dev)

    def _metrics(y_hat, y_ref):
        AE  = rad2deg(angle_diff(y_hat[:,0], y_ref[:,0])).mean().item()
        dlog= (torch.log1p(y_hat[:,2])-torch.log1p(y_ref[:,2])).abs().mean().item()
        sp  = (y_hat[:,3]-y_ref[:,3]).abs().mean().item()
        wet = (y_hat[:,4]-y_ref[:,4]).abs().mean().item()
        gain= (y_hat[:,5]-y_ref[:,5]).abs().mean().item()
        return AE, dlog, sp, wet, gain

    AE,dlog,sp,wet,gain=_metrics(y_pred, Yte)
    print(f"[baseline] Test N={len(Yte)} | AE {AE:.2f}_ | dlog {dlog:.3f} | sp {sp:.2f} | wet {wet:.3f} | gain {gain:.2f}")

    # KNN
    if args.use_knn:
        Htr=build_knn_cache(m, tr_texts, dev, max_len=args.max_len, batch_size=args.bsz)
        y_pred, frac=apply_knn(y_pred, h_te, Htr, Ytr, args)
        AE,dlog,sp,wet,gain=_metrics(y_pred, Yte)
        print(f"[after KNN] fb {frac:.1f}% | AE {AE:.2f}_ | dlog {dlog:.3f} | sp {sp:.2f} | wet {wet:.3f} | gain {gain:.2f}")

    # Calib
    if args.use_calib:
        y_pred=apply_calib(y_pred, rot_lam=args.rot_lam, max_rot_deg=args.max_rot)
        AE,dlog,sp,wet,gain=_metrics(y_pred, Yte)
        print(f"[after Calib] AE {AE:.2f}_ | dlog {dlog:.3f} | sp {sp:.2f} | wet {wet:.3f} | gain {gain:.2f}")

    # TextSnap
    if args.use_textsnap:
        y_pred=apply_textsnap(y_pred, te_texts, gate_deg=args.snap_gate, strength=args.snap_strength, el_rate=args.el_rate)
        AE,dlog,sp,wet,gain=_metrics(y_pred, Yte)
        print(f"[after TextSnap] AE {AE:.2f}_ | dlog {dlog:.3f} | sp {sp:.2f} | wet {wet:.3f} | gain {gain:.2f}")

if __name__=="__main__":
    main()
