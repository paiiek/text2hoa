# -*- coding: utf-8 -*-
# OOM-safe baselines: batched eval + autocast
import os, json, math, argparse, random, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

EPS=1e-7
random.seed(42); torch.manual_seed(42); np.random.seed(42)

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

def load_rows(path, room_mode="drr"):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            x=json.loads(line)
            room = x["room_depth"].get("drr_db") if room_mode=="drr" else x["room_depth"].get("rt60_s")
            if room is None: continue
            y = torch.tensor([
                x["az_sc"][0], x["az_sc"][1], x["el_rad"], x["dist_m"],
                x["spread_deg"], x["wet_mix"], x["gain_db"], room
            ], dtype=torch.float32)
            rows.append({"text": x["text"], "y": y})
    return rows

def rows_to_tensor(rows, device=None):
    Y=torch.stack([r["y"] for r in rows],0)
    return Y.to(device) if device else Y

# ---------- Linear baseline ----------
class TextEncoder(nn.Module):
    def __init__(self, enc_name):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(enc_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(enc_name)
        for p in self.enc.parameters(): p.requires_grad=False
        self.norm = nn.LayerNorm(self.enc.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts, device, max_length=64, amp=True):
        bt=self.tok(prefix_texts(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        bt={k:v.to(device) for k,v in bt.items()}
        use_amp = (amp and device and str(device).startswith("cuda"))
        try:
            ac = torch.cuda.amp.autocast if use_amp else torch.autocast
        except AttributeError:
            ac = torch.cuda.amp.autocast  # older torch fallback
        with torch.cuda.amp.autocast(enabled=use_amp):
            hs=self.enc(**bt).last_hidden_state
        h=masked_mean(hs, bt["attention_mask"])
        return F.normalize(self.norm(h), dim=-1)

class LinearRegressor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 8)

    def forward(self, h):
        y_raw=self.head(h)
        s=y_raw[:,0]; c=y_raw[:,1]; n=torch.sqrt(s*s+c*c+EPS)
        s, c = s/n, c/n
        el=torch.tanh(y_raw[:,2])*(1.0472)
        dist=torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread=torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet=torch.sigmoid(y_raw[:,5])
        gain=torch.tanh(y_raw[:,6])*6.0
        room=y_raw[:,7]
        y=torch.stack([s,c,el,dist,spread,wet,gain,room],1)
        return y

def metrics_batch_accum(m_sums, Yh, Y):
    # 누적 합으로 메모리 절약
    with torch.no_grad():
        AE=angle_err_deg(Yh, Y)
        m_sums["AE_deg"]   += AE.sum().item()
        m_sums["d_log"]    += (torch.log1p(Yh[:,3])-torch.log1p(Y[:,3])).abs().sum().item()
        m_sums["spread"]   += (Yh[:,4]-Y[:,4]).abs().sum().item()
        m_sums["wet"]      += (Yh[:,5]-Y[:,5]).abs().sum().item()
        m_sums["gain"]     += (Yh[:,6]-Y[:,6]).abs().sum().item()
        m_sums["room"]     += (Yh[:,7]-Y[:,7]).abs().sum().item()
        m_sums["N"]        += Y.shape[0]
    return m_sums

def finalize_metrics(m_sums):
    N = max(1, m_sums["N"])
    return {
        "AE_deg":      m_sums["AE_deg"]/N,
        "d_log":       m_sums["d_log"]/N,
        "spread_MAE":  m_sums["spread"]/N,
        "wet_MAE":     m_sums["wet"]/N,
        "gain_MAE":    m_sums["gain"]/N,
        "room_MAE":    m_sums["room"]/N,
    }

def batched_eval(enc, reg, X, Y, device, bsz_eval=256):
    m_sums = {"AE_deg":0.0,"d_log":0.0,"spread":0.0,"wet":0.0,"gain":0.0,"room":0.0,"N":0}
    for i in range(0, len(X), bsz_eval):
        xb = X[i:i+bsz_eval]; Yb = Y[i:i+bsz_eval]
        with torch.no_grad():
            hb = enc.encode(xb, device)
            yh = reg(hb)
        m_sums = metrics_batch_accum(m_sums, yh, Yb)
    return finalize_metrics(m_sums)

def train_linear_baseline(data_path, room_mode, enc_name, out_csv,
                          epochs=2, bsz=256, bsz_eval=256, lr=5e-3, wd=1e-4, seed=42):
    dev="cuda:0" if torch.cuda.is_available() else "cpu"
    rows=load_rows(data_path, room_mode); random.shuffle(rows)
    N=len(rows); ntr=int(0.9*N)
    tr,va = rows[:ntr], rows[ntr:]
    Xtr=[r["text"] for r in tr]; Ytr=rows_to_tensor(tr, dev)
    Xva=[r["text"] for r in va]; Yva=rows_to_tensor(va, dev)

    enc=TextEncoder(enc_name).to(dev)
    d=enc.enc.config.hidden_size
    reg=LinearRegressor(d).to(dev)
    opt=torch.optim.AdamW(reg.parameters(), lr=lr, weight_decay=wd)

    def mk_batches(X, Y, bsz):
        for i in range(0,len(X),bsz):
            yield X[i:i+bsz], Y[i:i+bsz]

    for ep in range(1,epochs+1):
        reg.train(); run=0.0; nb=0
        for xb, yb in mk_batches(Xtr, Ytr, bsz):
            with torch.no_grad(): hb=enc.encode(xb, dev)
            yh=reg(hb)
            loss_mse = F.l1_loss(yh, yb)
            sc = torch.sqrt(yh[:,0]**2 + yh[:,1]**2 + EPS)
            loss_sc = (sc-1.0).abs().mean()
            loss = loss_mse + 0.1*loss_sc
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); nb += 1
        # batched val (OOM-safe)
        reg.eval()
        m=batched_eval(enc, reg, Xva, Yva, dev, bsz_eval=bsz_eval)
        print(f"[Linear ep{ep}] train {run/max(1,nb):.4f} | AE {m['AE_deg']:.2f} dlog {m['d_log']:.3f} "
              f"spread {m['spread_MAE']:.2f} wet {m['wet_MAE']:.3f}")
    # full eval also batched (OOM-safe)
    Yfull = rows_to_tensor(rows, dev)
    mfull = batched_eval(enc, reg, [r["text"] for r in rows], Yfull, dev, bsz_eval=bsz_eval)
    res={"variant":"linear("+enc_name.split('/')[-1]+")", **mfull}
    write_baseline_csv(out_csv, [res])
    return res

# ---------- Rule baseline (아주 단순 비교선) ----------
KR_DIR = {
    "앞":0, "정면":0, "앞쪽":0, "전면":0, "앞에서":0,
    "뒤":180, "뒷":180, "후방":180, "뒤쪽":180, "뒤에서":180,
    "왼":270, "좌":270, "왼쪽":270,
    "오른":90, "우":90, "오른쪽":90,
    "대각 앞오른":45, "앞오른":45, "사선 앞오른":45,
    "대각 앞왼":315, "앞왼":315,
    "대각 뒤오른":135, "뒤오른":135,
    "대각 뒤왼":225, "뒤왼":225,
}
EN_DIR = {
    "front":0, "ahead":0, "in front":0,
    "back":180, "behind":180,
    "left":270, "right":90,
    "front-right":45, "front right":45, "front-left":315, "front left":315,
    "back-right":135, "back right":135, "back-left":225, "back left":225
}
def rule_guess(text):
    tl=text.lower()
    az=None
    for k,deg in EN_DIR.items():
        if k in tl: az=deg; break
    if az is None:
        for k,deg in KR_DIR.items():
            if k in text: az=deg; break
    if az is None: az=0.0
    el=0.0
    if any(t in tl for t in ["overhead","above","over","위", "머리 위"]): el=0.35
    if any(t in tl for t in ["below","floor","바닥","아래"]): el=-0.35
    dist=2.5
    if any(t in tl for t in ["near","close","가까", "근처"]): dist=1.2
    if any(t in tl for t in ["far","distant","멀", "먼"]): dist=4.0
    spread=45.0
    if any(t in tl for t in ["wide","broad","넓", "퍼져"]): spread=90.0
    if any(t in tl for t in ["narrow","tight","좁", "점"]): spread=15.0
    wet=0.35
    if any(t in tl for t in ["reverb","wet","잔향","울림","홀리"]): wet=0.7
    if any(t in tl for t in ["dry","무향","드라이","건조"]): wet=0.1
    gain=0.0
    if any(t in tl for t in ["loud","크게","강하게"]): gain=2.0
    if any(t in tl for t in ["soft","quiet","작게","은은"]): gain=-1.0
    room=2.0 if wet<0.5 else 4.0
    s=math.sin(math.radians(az)); c=math.cos(math.radians(az))
    return [s,c, el, dist, spread, wet, gain, room]

def eval_rule_baseline(data_path, room_mode, out_csv):
    rows=load_rows(data_path, room_mode)
    Y=rows_to_tensor(rows)
    Yh=torch.tensor([rule_guess(r["text"]) for r in rows], dtype=torch.float32)
    m={
        "AE_deg":     angle_err_deg(Yh, Y).mean().item(),
        "d_log":      (torch.log1p(Yh[:,3])-torch.log1p(Y[:,3])).abs().mean().item(),
        "spread_MAE": (Yh[:,4]-Y[:,4]).abs().mean().item(),
        "wet_MAE":    (Yh[:,5]-Y[:,5]).abs().mean().item(),
        "gain_MAE":   (Yh[:,6]-Y[:,6]).abs().mean().item(),
        "room_MAE":   (Yh[:,7]-Y[:,7]).abs().mean().item(),
    }
    res={"variant":"rule_baseline", **m}
    write_baseline_csv(out_csv, [res])
    print("[Rule]", res)
    return res

def write_baseline_csv(path, rows):
    cols=["variant","AE_deg","d_log","spread_MAE","wet_MAE","gain_MAE","room_MAE"]
    write_header= not os.path.exists(path)
    with open(path,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=cols)
        if write_header: w.writeheader()
        for r in rows: w.writerow({k:r[k] for k in cols})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--enc_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--eval_bsz", type=int, default=256)  # NEW: batched eval size
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out_csv", default="baseline_results.csv")
    ap.add_argument("--do_linear", action="store_true")
    ap.add_argument("--do_rule", action="store_true")
    args=ap.parse_args()

    if args.do_linear:
        train_linear_baseline(args.data, args.room_mode, args.enc_model,
                              out_csv=args.out_csv, epochs=args.epochs,
                              bsz=args.bsz, bsz_eval=args.eval_bsz,
                              lr=args.lr, wd=args.wd)
    if args.do_rule:
        eval_rule_baseline(args.data, args.room_mode, out_csv=args.out_csv)
    if not args.do_linear and not args.do_rule:
        print("Specify at least one: --do_linear or --do_rule")

if __name__=="__main__":
    # fragmentation 완화 권장
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True,max_split_size_mb:64")
    main()
