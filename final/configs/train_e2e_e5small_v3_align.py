# -*- coding: utf-8 -*-
# save as: train_e2e_e5small_v3_align.py
import os, json, math, argparse, random
from typing import List

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

random.seed(42); torch.manual_seed(42)
EPS=1e-7

# ---------------- utils ----------------
def prefix_texts(texts: List[str]) -> List[str]:
    out=[]
    for t in texts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

def masked_mean(hs, attn):
    mask = attn.unsqueeze(-1).type_as(hs)
    return (hs*mask).sum(1) / mask.sum(1).clamp_min(1.0)

def angle_err_deg(p, g):
    s1,c1, el1 = p[:,0],p[:,1],p[:,2]
    s2,c2, el2 = g[:,0],g[:,1],g[:,2]
    u1 = torch.stack([torch.cos(el1)*c1, torch.cos(el1)*s1, torch.sin(el1)], dim=-1)
    u2 = torch.stack([torch.cos(el2)*c2, torch.cos(el2)*s2, torch.sin(el2)], dim=-1)
    dot = (u1*u2).sum(-1).clamp(-1+EPS,1-EPS)
    return torch.rad2deg(torch.acos(dot))

def quad_from_sc(s, c):
    az = math.degrees(math.atan2(float(s), float(c)))
    if -45<=az<45: return 0
    if 45<=az<135: return 1
    if -135<=az<-45: return 3
    return 2

def azbin12_from_sc(s, c):
    az = math.degrees(math.atan2(float(s), float(c)))
    if az < 0: az += 360.0
    return int(az // 30.0)

def el_bin(el):  return 0 if el<-0.2 else (2 if el>0.2 else 1)
def dist_bin(d): return 0 if d<1.5 else (1 if d<3.0 else 2)
def spr_bin(s):  return 0 if s<30 else (1 if s<60 else 2)

def load_rows(path, room_mode):
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
            q = quad_from_sc(x["az_sc"][0], x["az_sc"][1])
            a12 = azbin12_from_sc(x["az_sc"][0], x["az_sc"][1])
            eb = el_bin(x["el_rad"]); db = dist_bin(x["dist_m"]); sb = spr_bin(x["spread_deg"])
            rows.append({"text": x["text"], "y": y, "aux": (q,eb,db,sb,a12)})
    return rows

def rows_to_tensor(rows, room_mode):
    return torch.stack([torch.tensor([
        r["y"][0].item(), r["y"][1].item(), r["y"][2].item(), r["y"][3].item(),
        r["y"][4].item(), r["y"][5].item(), r["y"][6].item(), r["y"][7].item()
    ], dtype=torch.float32) for r in rows], 0)

# 12-bin 방위 센터 (rad)
AZ12_CENTERS_DEG = torch.tensor([i*30+15 for i in range(12)], dtype=torch.float32)  # 15,45,75,...,345
AZ12_SIN = torch.sin(torch.deg2rad(AZ12_CENTERS_DEG))
AZ12_COS = torch.cos(torch.deg2rad(AZ12_CENTERS_DEG))

# --------------- ArcFace ----------------
class ArcMarginAz12(nn.Module):
    def __init__(self, d, s=30.0, m=0.25):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, 12))
        nn.init.xavier_uniform_(self.W)
        self.s = s; self.m = m
    def forward(self, h, y=None):
        # 안정성을 위해 autocast off
        try:
            ac = lambda enabled: torch.amp.autocast('cuda', enabled=enabled)
        except TypeError:
            ac = lambda enabled: torch.cuda.amp.autocast(enabled=enabled)
        with ac(False):
            h32 = F.normalize(h.float(), dim=-1)
            W32 = F.normalize(self.W.float(), dim=0)
            logits32 = h32 @ W32
            if y is None:
                return (logits32 * self.s).to(h.dtype)
            logits32 = logits32.clamp(-1+1e-7,1-1e-7)
            theta32 = torch.acos(logits32)
            idx = torch.arange(h32.size(0), device=h32.device)
            target = torch.cos(theta32[idx, y] + self.m)
            logits_m = logits32.clone()
            logits_m[idx, y] = target
            return (logits_m * self.s).to(h.dtype)

# --------------- Model ------------------
class T2SModel(nn.Module):
    def __init__(self, enc_name="intfloat/multilingual-e5-small", out_dim=8, el_max=1.0472):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(enc_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(enc_name)
        d = self.enc.config.hidden_size
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 768), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 384), nn.ReLU(),
            nn.Linear(384, out_dim)
        )
        self.aux = nn.ModuleDict({
            "quad": nn.Linear(d,4),
            "elev": nn.Linear(d,3),
            "dist": nn.Linear(d,3),
            "sprd": nn.Linear(d,3),
        })
        self.azarc = ArcMarginAz12(d, s=30.0, m=0.25)
        self.EL_MAX = el_max

    def set_bitfit(self):
        for p in self.enc.parameters(): p.requires_grad=False
        for n,p in self.enc.named_parameters():
            if n.endswith(".bias") or "LayerNorm" in n or "layer_norm" in n:
                p.requires_grad=True
        print("[BitFit] encoder: bias+LayerNorm trainable")

    def unfreeze_last_n(self, n):
        if n<=0: return
        if not hasattr(self.enc, "encoder") or not hasattr(self.enc.encoder, "layer"):
            print("[WARN] can't unfreeze_last_n on this model structure"); return
        L = len(self.enc.encoder.layer)
        for i in range(L-n, L):
            for p in self.enc.encoder.layer[i].parameters():
                p.requires_grad=True
        print(f"[Unfreeze] last {n} blocks unfrozen.")

    def encode(self, texts, device):
        bt = self.tok(prefix_texts(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        bt = {k:v.to(device) for k,v in bt.items()}
        hs = self.enc(**bt).last_hidden_state
        h  = masked_mean(hs, bt["attention_mask"])
        h  = F.normalize(self.norm(h), dim=-1)
        return h

    def forward(self, texts, az12_labels=None, device=None):
        h = self.encode(texts, device=device)
        y_raw = self.head(h)
        s=y_raw[:,0]; c=y_raw[:,1]; n=torch.sqrt(s*s+c*c+EPS)
        az0=s/n; az1=c/n
        el=torch.tanh(y_raw[:,2])*self.EL_MAX      # ±60°
        dist=torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread=torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet=torch.sigmoid(y_raw[:,5])
        gain=torch.tanh(y_raw[:,6])*6.0
        room=y_raw[:,7]
        y = torch.stack([az0,az1,el,dist,spread,wet,gain,room],1)
        logits = {k:hd(h) for k,hd in self.aux.items()}
        az12_logits = self.azarc(h, az12_labels) if az12_labels is not None else self.azarc(h, None)
        return y, logits, az12_logits, h

# --------------- Losses -----------------
def loss_reg(yhat, y, room_mode="drr", w_dir=40.0, w_other=1.0):
    def to_u(s, c, el):
        x=torch.cos(el)*c; yv=torch.cos(el)*s; z=torch.sin(el)
        return torch.stack([x,yv,z],-1)
    uhat=to_u(yhat[:,0],yhat[:,1],yhat[:,2])
    ugt =to_u(y[:,0],  y[:,1],  y[:,2])
    l_dir=(1.0-(uhat*ugt).sum(-1).clamp(-1+1e-7,1-1e-7)).mean()
    l_el=(yhat[:,2]-y[:,2]).abs().mean()
    l_dist=(torch.log1p(yhat[:,3])-torch.log1p(y[:,3])).abs().mean()
    l_sp=(yhat[:,4]-y[:,4]).abs().mean()
    l_w=(yhat[:,5]-y[:,5]).abs().mean()
    l_g=(yhat[:,6]-y[:,6]).abs().mean()
    room_hat = torch.tanh(yhat[:,7])*12.0 if room_mode=="drr" else torch.sigmoid(yhat[:,7])*(2.5-0.3)+0.3
    l_room=(room_hat - y[:,7]).abs().mean()
    return w_dir*l_dir + w_other*(l_el + l_dist + l_sp + l_w + l_g + l_room), uhat, ugt

def loss_dir_contrast(uhat, ugt, margin=0.15, topk=5, w=1.0):
    uhat = F.normalize(uhat, dim=-1); ugt  = F.normalize(ugt,  dim=-1)
    sim = uhat @ ugt.t()
    pos = sim.diag()
    sim = sim - torch.eye(sim.size(0), device=sim.device)*(1e6)
    k = min(topk, max(1, sim.size(1)-1))
    hard, _ = sim.topk(k=k, dim=1)
    return w * (hard - pos.unsqueeze(1) + margin).clamp_min(0.0).mean()

def loss_aux(base_logits, az12_arc_logits, aux_targets, w_base=0.25, w_az12=2.0):
    q,e,d,s,a12 = aux_targets
    lq=F.cross_entropy(base_logits["quad"], q, label_smoothing=0.05)
    le=F.cross_entropy(base_logits["elev"], e, label_smoothing=0.05)
    ld=F.cross_entropy(base_logits["dist"], d, label_smoothing=0.05)
    ls=F.cross_entropy(base_logits["sprd"], s, label_smoothing=0.05)
    la=F.cross_entropy(az12_arc_logits, a12)
    return w_base*(lq+le+ld+ls) + w_az12*la

# --- NEW: 회귀(s/c) ↔ ArcFace(12-bin) 정렬 손실
def loss_align_sc_with_arc(yhat, az12_logits, temp=2.0, w_align=5.0, device="cuda"):
    # softmax 온도 temp로 12-bin 확률
    p = F.softmax(az12_logits / temp, dim=-1)  # [B,12]
    # 기대값 unit-circle
    s_exp = (p * AZ12_SIN.to(device)).sum(-1)
    c_exp = (p * AZ12_COS.to(device)).sum(-1)
    # 회귀 s,c와 L2 정렬
    s_hat = yhat[:,0]; c_hat = yhat[:,1]
    return w_align * ((s_hat - s_exp)**2 + (c_hat - c_exp)**2).mean()

# --------------- Train ------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--enc_model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_enc", type=float, default=5e-6)
    ap.add_argument("--bitfit", action="store_true")
    ap.add_argument("--unfreeze_last_n", type=int, default=2)
    ap.add_argument("--contrast_k", type=int, default=5)
    ap.add_argument("--contrast_margin", type=float, default=0.15)
    ap.add_argument("--w_contrast", type=float, default=1.0)
    ap.add_argument("--arc_s0", type=float, default=20.0)  # warmup
    ap.add_argument("--arc_m0", type=float, default=0.15)  # warmup
    ap.add_argument("--arc_s1", type=float, default=35.0)  # after 1/3
    ap.add_argument("--arc_m1", type=float, default=0.30)  # after 1/3
    ap.add_argument("--arc_s2", type=float, default=40.0)  # after 2/3
    ap.add_argument("--arc_m2", type=float, default=0.35)  # after 2/3
    ap.add_argument("--align_w", type=float, default=5.0)
    ap.add_argument("--align_temp", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="t2sa_e2e_e5s_align.pt")
    # ... 기존 argparse들 밑에 추가
    ap.add_argument("--load", type=str, default="", help="resume from checkpoint (.pt)")

    args=ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = load_rows(args.data, args.room_mode)
    N=len(rows)

    # split
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    ntr = int(0.9*N)
    tr_idx, va_idx = perm[:ntr], perm[ntr:]
    texts = [r["text"] for r in rows]
    Y = rows_to_tensor(rows, args.room_mode)
    AUX = torch.tensor([[r["aux"][0], r["aux"][1], r["aux"][2], r["aux"][3], r["aux"][4]] for r in rows], dtype=torch.long)

    class TxtSet(torch.utils.data.Dataset):
        def __init__(self, X,Y,A): self.X=X; self.Y=Y; self.A=A
        def __len__(self): return len(self.X)
        def __getitem__(self,i): return self.X[i], self.Y[i], self.A[i]

    Xtr = [texts[i] for i in tr_idx.tolist()]
    Xva = [texts[i] for i in va_idx.tolist()]
    Ytr, Yva = Y[tr_idx], Y[va_idx]
    Atr, Ava = AUX[tr_idx], AUX[va_idx]
    ds_tr = TxtSet(Xtr, Ytr, Atr)
    ds_va = TxtSet(Xva, Yva, Ava)

    tr_loader=DataLoader(ds_tr, batch_size=args.bsz, shuffle=True, drop_last=True, num_workers=0)
    va_loader=DataLoader(ds_va, batch_size=args.bsz, shuffle=False, num_workers=0)

    # model
    m = T2SModel(args.enc_model, el_max=1.0472)
    m = m.to(dev)
    
    # === 여기 추가 ===
    if args.load:
        ckpt = torch.load(args.load, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        ret = m.load_state_dict(sd, strict=False)
        print(f"[resume] loaded: {args.load} | missing={len(ret.missing_keys)} | unexpected={len(ret.unexpected_keys)}")

    if args.bitfit: m.set_bitfit()
    if args.unfreeze_last_n>0: m.unfreeze_last_n(args.unfreeze_last_n)

    # AMP
    try:
        scaler = torch.amp.GradScaler('cuda')
        autocast_ctx = lambda: torch.amp.autocast('cuda')
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=True)

    # optimizer
    enc_params, head_params = [], []
    for n,p in m.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("enc."):
            enc_params.append(p)
        else:
            head_params.append(p)
    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": args.lr_head},
         {"params": enc_params,  "lr": args.lr_enc}],
        weight_decay=0.01
    )
    total_steps = max(1, args.epochs * max(1, len(tr_loader)))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(400, total_steps//10), num_training_steps=total_steps)

    def set_arc(ep, E):
        if ep <= E//3:
            m.azarc.s = args.arc_s0; m.azarc.m = args.arc_m0
        elif ep <= 2*E//3:
            m.azarc.s = args.arc_s1; m.azarc.m = args.arc_m1
        else:
            m.azarc.s = args.arc_s2; m.azarc.m = args.arc_m2

    best_AE = 1e9
    print(f"[split] train {len(ds_tr)} | val {len(ds_va)}")
    for ep in range(1, args.epochs+1):
        set_arc(ep, args.epochs)
        # 전구간 방향가중 높게 유지, other는 초반 0→후반 1
        w_dir = 40.0
        w_other = 0.0 if ep <= 6 else 1.0

        m.train(); run=0.0
        for texts_b, Yb, Ab in tr_loader:
            Yb = Yb.to(dev); Ab=Ab.to(dev)
            texts_b = list(texts_b)
            with autocast_ctx():
                yhat, logits, az12_logits, h = m(texts_b, az12_labels=Ab[:,4], device=dev)
                Lr, uhat, ugt = loss_reg(yhat, Yb, room_mode=args.room_mode, w_dir=w_dir, w_other=w_other)
                Lc = loss_dir_contrast(uhat, ugt, margin=args.contrast_margin, topk=args.contrast_k, w=args.w_contrast)
                La = loss_aux(logits, az12_logits, (Ab[:,0],Ab[:,1],Ab[:,2],Ab[:,3],Ab[:,4]), w_base=0.25, w_az12=2.0)
                Lalign = loss_align_sc_with_arc(yhat, az12_logits, temp=args.align_temp, w_align=args.align_w, device=dev)
                loss = Lr + Lc + La + Lalign
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 0.8)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            run += loss.item()

        # ---- val ----
        m.eval()
        with torch.no_grad():
            Ang=[]; dMAE=[]; spMAE=[]; wMAE=[]; gMAE=[]; rMAE=[]
            for texts_b, Yb, Ab in va_loader:
                Yb=Yb.to(dev); texts_b=list(texts_b)
                yhat, _, _, _ = m(texts_b, az12_labels=None, device=dev)
                room_hat = torch.tanh(yhat[:,7])*12.0 if args.room_mode=="drr" else torch.sigmoid(yhat[:,7])*(2.5-0.3)+0.3
                yh = yhat.clone(); yh[:,7]=room_hat
                Ang.append(angle_err_deg(yh, Yb).cpu())
                dMAE.append((torch.log1p(yh[:,3]) - torch.log1p(Yb[:,3])).abs().cpu())
                spMAE.append((yh[:,4]-Yb[:,4]).abs().cpu())
                wMAE.append((yh[:,5]-Yb[:,5]).abs().cpu())
                gMAE.append((yh[:,6]-Yb[:,6]).abs().cpu())
                rMAE.append((yh[:,7]-Yb[:,7]).abs().cpu())
            def catm(x): return torch.cat(x).mean().item()
            AE = catm(Ang); dlog=catm(dMAE); sp=catm(spMAE); wet=catm(wMAE); gain=catm(gMAE); room=catm(rMAE)
            print(f"epoch {ep:02d}: train {run/len(tr_loader):.4f} | AE {AE:.2f} | dlog {dlog:.3f} | sp {sp:.2f} | wet {wet:.3f} | gain {gain:.2f} | room {room:.2f}")
            if AE < best_AE:
                best_AE = AE
                torch.save({
                    "state_dict": m.state_dict(),
                    "enc_model": args.enc_model,
                    "seed": args.seed,
                    "room_mode": args.room_mode
                }, args.save)
                print("  saved(best AE):", args.save)

if __name__=="__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True,max_split_size_mb:64")
    main()
