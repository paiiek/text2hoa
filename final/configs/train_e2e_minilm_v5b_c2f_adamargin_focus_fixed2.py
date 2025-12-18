# -*- coding: utf-8 -*-
# save as: train_e2e_minilm_v5b_c2f_adamargin_focus.py
import os, json, math, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

# ----------------------------
# Repro & constants
# ----------------------------
random.seed(42); torch.manual_seed(42)
EPS = 1e-7
DELTA_MAX = math.radians(15.0)  # az12 중심 대비 delta 최대치(±15°)

# ----------------------------
# Text preproc
# ----------------------------
def prefix_texts(texts):
    out=[]
    for t in texts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

def masked_mean(hs, attn):
    mask = attn.unsqueeze(-1).type_as(hs)
    return (hs*mask).sum(1)/mask.sum(1).clamp_min(1.0)

# ----------------------------
# Geometry helpers
# ----------------------------
def angle_err_deg(p, g):
    # p,g: [..., 0]=sin, 1=cos, 2=el
    s1,c1, el1 = p[:,0],p[:,1],p[:,2]
    s2,c2, el2 = g[:,0],g[:,1],g[:,2]
    u1 = torch.stack([torch.cos(el1)*c1, torch.cos(el1)*s1, torch.sin(el1)], -1)
    u2 = torch.stack([torch.cos(el2)*c2, torch.cos(el2)*s2, torch.sin(el2)], -1)
    dot = (u1*u2).sum(-1).clamp(-1+EPS,1-EPS)
    return torch.rad2deg(torch.acos(dot))

def quad_from_sc(s, c):
    az = math.degrees(math.atan2(float(s), float(c)))
    if -45<=az<45: return 0  # front
    if 45<=az<135: return 1  # right
    if -135<=az<-45: return 3 # left
    return 2                 # back

def az12_from_sc(s,c):
    az = math.degrees(math.atan2(float(s), float(c)))
    if az < 0: az += 360.0
    return int(az//30)

def el_bin(el):  return 0 if el<-0.2 else (2 if el>0.2 else 1)
def dist_bin(d): return 0 if d<1.5 else (1 if d<3.0 else 2)
def spr_bin(s):  return 0 if s<30 else (1 if s<60 else 2)

# ----------------------------
# Data loading
# ----------------------------
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
            a12 = az12_from_sc(x["az_sc"][0], x["az_sc"][1])
            eb = el_bin(x["el_rad"]); db = dist_bin(x["dist_m"]); sb = spr_bin(x["spread_deg"])
            w = float(x.get("w", 1.0))  # 데이터에 가중치가 있으면 활용
            rows.append({"text": x["text"], "y": y, "aux": (eb,db,sb,a12), "w": w})
    return rows

def rows_to_tensor(rows):
    return torch.stack([r["y"] for r in rows],0)

# ----------------------------
# AZ12 lookups
# ----------------------------
AZ12_CDEG = torch.tensor([i*30+15 for i in range(12)], dtype=torch.float32)
AZ12_SIN  = torch.sin(torch.deg2rad(AZ12_CDEG))
AZ12_COS  = torch.cos(torch.deg2rad(AZ12_CDEG))
AZ12_RAD  = torch.deg2rad(AZ12_CDEG)

# ----------------------------
# ArcMargin head for az12
# ----------------------------
class ArcMarginAz12(nn.Module):
    def __init__(self, d, s=45.0, m_base=0.45):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d,12))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.register_buffer("m_vec", torch.full((12,), float(m_base)))

    @torch.no_grad()
    def set_margin_vec(self, m_vec):
        m_vec = torch.as_tensor(m_vec, dtype=self.m_vec.dtype, device=self.m_vec.device)
        self.m_vec.copy_(m_vec.clamp(0.0, 1.2))

    def forward(self, h, y=None):
        # 안정 위해 autocast Off
        try: ac = lambda enabled: torch.amp.autocast('cuda', enabled=enabled)
        except TypeError: ac = lambda enabled: torch.cuda.amp.autocast(enabled=enabled)
        with ac(False):
            h32 = F.normalize(h.float(), dim=-1)
            W32 = F.normalize(self.W.float(), dim=0)
            logits32 = (h32 @ W32).clamp(-1+1e-7,1-1e-7)
            if y is None:
                return (logits32 * self.s).to(h.dtype)
            idx = torch.arange(h32.size(0), device=h32.device)
            theta32 = torch.acos(logits32)
            m_y = self.m_vec[y]
            target = torch.cos(theta32[idx, y] + m_y)
            logits_m = logits32.clone()
            logits_m[idx, y] = target
            return (logits_m * self.s).to(h.dtype)

# ----------------------------
# Main model
# ----------------------------
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
            nn.Linear(384, 8)  # [sin,cos,el,dist,spread,wet,gain,room(raw)]
        )
        self.aux = nn.ModuleDict({
            "elev": nn.Linear(d,3),
            "dist": nn.Linear(d,3),
            "sprd": nn.Linear(d,3),
        })
        self.azarc = ArcMarginAz12(d, s=45.0, m_base=0.45)
        self.az_delta = nn.Linear(d,1)
        self.EL_MAX = el_max

    def set_bitfit(self):
        for p in self.enc.parameters(): p.requires_grad=False
        for n,p in self.enc.named_parameters():
            if n.endswith(".bias") or "LayerNorm" in n or "layer_norm" in n:
                p.requires_grad=True
        print("[BitFit] encoder: bias+LayerNorm trainable")

    def unfreeze_last_n(self, n):
        if n <= 0:
            return
        # 1) 후보 ModuleList '...layer' 자동 탐색 (BERT/MPNet/DistilBERT/MiniLM 등 호환)
        target = None
        target_name = None
        for name, mod in self.enc.named_modules():
            if isinstance(mod, nn.ModuleList) and name.endswith(".layer") and len(mod) > 0:
                target = mod
                target_name = name
        if target is None:
            print("[WARN] can't find '* .layer' ModuleList; unfreezing encoder biases/LN only.")
            for n,p in self.enc.named_parameters():
                if n.endswith(".bias") or "LayerNorm" in n or "layer_norm" in n:
                    p.requires_grad = True
            return

        # 2) 전체 freeze → 마지막 n개 블록만 unfreeze
        for p in self.enc.parameters():
            p.requires_grad = False
        L = len(target)
        n = min(n, L)
        for i in range(L - n, L):
            for p in target[i].parameters():
                p.requires_grad = True

        # 3) 항상 LN/bias는 학습
        for n,p in self.enc.named_parameters():
            if n.endswith(".bias") or "LayerNorm" in n or "layer_norm" in n:
                p.requires_grad = True
        print(f"[Unfreeze] last {n} transformer blocks from '{target_name}'")


    def encode(self, texts, device):
        bt=self.tok(prefix_texts(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        bt={k:v.to(device) for k,v in bt.items()}
        hs=self.enc(**bt).last_hidden_state
        h=masked_mean(hs, bt["attention_mask"])
        return F.normalize(self.norm(h), dim=-1)

    def forward(self, texts, a12_labels=None, device=None):
        h=self.encode(texts, device=device)
        y_raw=self.head(h)

        # ---- 정규화/클램프 (각도 손실 안정화의 핵심) ----
        s,c = y_raw[:,0], y_raw[:,1]
        n = torch.sqrt(s*s + c*c + EPS)
        az0, az1 = s/n, c/n                       # s/c 단위정규화
        el = torch.tanh(y_raw[:,2]) * self.EL_MAX # ±60°까지 허용하려면 1.0472(=60°)로 조정 가능
        dist = torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread = torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet = torch.sigmoid(y_raw[:,5])
        gain = torch.tanh(y_raw[:,6])*6.0
        room = y_raw[:,7]  # 회귀 원시값(손실에서 변환)

        y = torch.stack([az0,az1,el,dist,spread,wet,gain,room],1)

        logits={k:hd(h) for k,hd in self.aux.items()}
        az12_logits=self.azarc(h, a12_labels) if a12_labels is not None else self.azarc(h, None)
        delta_norm=torch.tanh(self.az_delta(h)).squeeze(1)  # [-1,1] 정규화 델타
        return y, logits, az12_logits, delta_norm, h

# ----------------------------
# Losses
# ----------------------------
def loss_reg(yhat, y, room_mode="drr", w_dir=40.0, w_other=1.0):
    # 3D great-circle 방향 손실 + 나머지 회귀 MAE
    def U(s,c,el): return torch.stack([torch.cos(el)*c, torch.cos(el)*s, torch.sin(el)],-1)
    uhat=U(yhat[:,0],yhat[:,1],yhat[:,2]); ugt=U(y[:,0],y[:,1],y[:,2])
    l_dir=(1.0-(uhat*ugt).sum(-1).clamp(-1+EPS,1-EPS)).mean()  # 각도 손실
    l_el=(yhat[:,2]-y[:,2]).abs().mean()
    l_dist=(torch.log1p(yhat[:,3])-torch.log1p(y[:,3])).abs().mean()
    l_sp=(yhat[:,4]-y[:,4]).abs().mean()
    l_w=(yhat[:,5]-y[:,5]).abs().mean()
    l_g=(yhat[:,6]-y[:,6]).abs().mean()
    # room 스케일 복원 후 MAE
    room_hat = torch.tanh(yhat[:,7])*12.0 if room_mode=="drr" else torch.sigmoid(yhat[:,7])*(2.5-0.3)+0.3
    l_room=(room_hat - y[:,7]).abs().mean()
    return w_dir*l_dir + w_other*(l_el+l_dist+l_sp+l_w+l_g+l_room), uhat, ugt

def loss_dir_contrast(uhat, ugt, margin=0.20, topk=8, w=1.2):
    # 배치 내 hard negatives에 margin 밀어내기
    uhat=F.normalize(uhat,dim=-1); ugt=F.normalize(ugt,dim=-1)
    sim=uhat @ ugt.t()
    pos=sim.diag()
    sim=sim - torch.eye(sim.size(0), device=sim.device)*(1e6)
    k=min(topk, max(1, sim.size(1)-1))
    hard,_=sim.topk(k=k,dim=1)
    return w*(hard - pos.unsqueeze(1) + margin).clamp_min(0).mean()

def circular_ce_two_ring_weighted(logits, y, weights=None, alpha1=0.05, alpha2=0.02):
    # az12 CE with label smoothing & 주변 class 분배(±1, ±2)
    N,C=logits.size()
    logp=F.log_softmax(logits, dim=-1)
    target=torch.zeros_like(logits)
    target.scatter_(1, y.unsqueeze(1), 1.0 - alpha1 - alpha2)
    left1=(y-1)%C; right1=(y+1)%C
    left2=(y-2)%C; right2=(y+2)%C
    ar=torch.arange(N, device=logits.device)
    target[ar,left1]+=alpha1/2; target[ar,right1]+=alpha1/2
    target[ar,left2]+=alpha2/2; target[ar,right2]+=alpha2/2
    nll=-(target*logp).sum(-1)  # [N]
    if weights is None: return nll.mean()
    w = weights.to(logits.dtype)
    return (w*nll).sum() / (w.sum().clamp_min(1.0))

def loss_aux_weighted(base_logits, az12_arc_logits, e3,d3,s3,a12, w_base=0.25, w_az12=2.0,
                      alpha1=0.05, alpha2=0.02, weights=None):
    le=F.cross_entropy(base_logits["elev"], e3, label_smoothing=0.05)
    ld=F.cross_entropy(base_logits["dist"], d3, label_smoothing=0.05)
    ls=F.cross_entropy(base_logits["sprd"], s3, label_smoothing=0.05)
    la=circular_ce_two_ring_weighted(az12_arc_logits, a12, weights=weights, alpha1=alpha1, alpha2=alpha2)
    return w_base*(le+ld+ls) + w_az12*la

def loss_align_sc_with_arc(yhat, az12_logits, temp=1.5, w_align=10.0, device="cuda"):
    p=F.softmax(az12_logits/temp, dim=-1)
    s_exp=(p*AZ12_SIN.to(device)).sum(-1)
    c_exp=(p*AZ12_COS.to(device)).sum(-1)
    return w_align * ((yhat[:,0]-s_exp)**2 + (yhat[:,1]-c_exp)**2).mean()

def loss_delta_weighted(delta_norm_pred, a12, y_true, yhat, weights=None,
                        w_delta=6.0, w_delta_sc=4.0, device="cuda"):
    # 정답 az에 대한 az12 중심 대비 delta 회귀(+s/c 정합)
    az=torch.atan2(y_true[:,0], y_true[:,1])
    centers=AZ12_RAD.to(device)[a12]
    delta=torch.atan2(torch.sin(az-centers), torch.cos(az-centers))
    target=(delta/DELTA_MAX).clamp(-1.0,1.0)
    L_reg=F.smooth_l1_loss(delta_norm_pred, target, reduction="none")  # [N]
    delta_hat=delta_norm_pred*DELTA_MAX
    s_exp=torch.sin(centers+delta_hat)
    c_exp=torch.cos(centers+delta_hat)
    L_sc=((yhat[:,0]-s_exp)**2 + (yhat[:,1]-c_exp)**2)  # [N]
    if weights is not None:
        w=weights.to(L_reg.dtype)
        L_reg=(w*L_reg).sum()/w.sum().clamp_min(1.0)
        L_sc =(w*L_sc ).sum()/w.sum().clamp_min(1.0)
    else:
        L_reg=L_reg.mean(); L_sc=L_sc.mean()
    return w_delta*L_reg + w_delta_sc*L_sc

# ----------------------------
# Adaptive margin & bin weights
# ----------------------------
@torch.no_grad()
def compute_adaptive_margin_vec(model, va_loader, device, base=0.30, mid=0.45, hi=0.65, band=1.0):
    counts=torch.zeros(12, device=device); sums=torch.zeros(12, device=device)
    for texts_b, Yb, Ab in va_loader:
        texts_b=list(texts_b); Yb=Yb.to(device); a12=Ab[:,3].to(device)
        yhat,_,_,_,_=model(texts_b, a12_labels=None, device=device)
        room_hat=torch.tanh(yhat[:,7])*12.0
        yh=yhat.clone(); yh[:,7]=room_hat
        ang=angle_err_deg(yh, Yb).to(device)
        for k in range(12):
            m=(a12==k)
            if m.any():
                counts[k]+=m.sum()
                sums[k]+=ang[m].sum()
    means=torch.where(counts>0, sums/counts.clamp_min(1.0), torch.zeros_like(counts))
    gmean=means[counts>0].mean() if (counts>0).any() else means.mean()
    m_vec=torch.full((12,), mid, device=device)
    m_vec=torch.where(means >= gmean+band, torch.full_like(m_vec, hi), m_vec)
    m_vec=torch.where(means <= gmean-band, torch.full_like(m_vec, base), m_vec)
    return m_vec, means

@torch.no_grad()
def compute_bin_weights(means, gamma=1.0, max_w=2.0):
    g=means[means>0].mean() if (means>0).any() else means.mean()
    w=1.0 + gamma * (means - g) / max(float(g), 1.0)
    w=torch.clamp(w, 1.0, max_w)
    w=torch.where(torch.isfinite(w), w, torch.ones_like(w))
    return w  # [12]

# ----------------------------
# Sampler weights (방향 불균형 보정)
# ----------------------------
def build_sampler_weights(rows_subset, mode="quad", power=1.0, use_json_w=True, clip=5.0):
    """
    mode:
      - 'quad' : front/right/back/left 균등 보정
      - 'az12' : 12분할 균등 보정
    power: inverse frequency ^ power
    use_json_w: json내 'w' 필드(있으면) 곱셈
    clip: 최종 weight 상한
    """
    if mode not in ("quad","az12"):
        return [float(r.get("w",1.0)) for r in rows_subset]

    # count
    counts = {}
    keys = []
    for r in rows_subset:
        s,c = float(r["y"][0]), float(r["y"][1])
        if mode=="quad":
            k = quad_from_sc(s,c)
        else:
            k = az12_from_sc(s,c)
        keys.append(k)
        counts[k] = counts.get(k,0) + 1

    # inv-freq weights
    inv = {}
    for k,v in counts.items():
        inv[k] = (1.0/max(1,v))**power

    # normalize to mean 1.0
    raw = []
    for i,r in enumerate(rows_subset):
        base = inv[keys[i]]
        if use_json_w:
            base *= float(r.get("w",1.0))
        raw.append(base)
    mean_w = sum(raw)/max(1,len(raw))
    ws = [min(clip, w/max(mean_w,1e-8)) for w in raw]
    return ws

# ----------------------------
# Main
# ----------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--enc_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_enc", type=float, default=3e-6)
    ap.add_argument("--bitfit", action="store_true")
    ap.add_argument("--unfreeze_last_n", type=int, default=4)
    # contrast/align/delta
    ap.add_argument("--contrast_k", type=int, default=8)
    ap.add_argument("--contrast_margin", type=float, default=0.20)
    ap.add_argument("--w_contrast", type=float, default=1.2)
    ap.add_argument("--align_w", type=float, default=10.0)
    ap.add_argument("--align_temp", type=float, default=1.5)
    ap.add_argument("--w_delta", type=float, default=6.0)
    ap.add_argument("--w_delta_sc", type=float, default=4.0)
    # 방향 집중 에폭
    ap.add_argument("--dir_focus_epochs", type=int, default=8)
    # adaptive margin
    ap.add_argument("--m_base", type=float, default=0.30)
    ap.add_argument("--m_mid",  type=float, default=0.45)
    ap.add_argument("--m_hi",   type=float, default=0.65)
    ap.add_argument("--m_band", type=float, default=1.0)
    # bin weight params
    ap.add_argument("--gamma_bin_weight", type=float, default=1.0)
    ap.add_argument("--max_bin_weight",   type=float, default=2.0)
    # Sampler
    ap.add_argument("--weighted_sampler", action="store_true")
    ap.add_argument("--sampler_mode", choices=["quad","az12"], default="quad")
    ap.add_argument("--sampler_power", type=float, default=1.0)
    ap.add_argument("--sampler_clip", type=float, default=4.0)
    ap.add_argument("--no_json_w", action="store_true", help="ignore 'w' in json if present")
    # resume/save
    ap.add_argument("--load", default="", help="resume from ckpt")
    ap.add_argument("--save", default="t2sa_e2e_minilm_stage4b_focus.pt")
    ap.add_argument("--seed", type=int, default=42)
    # KNN defaults (메타에 저장하여 eval에서 참고)
    ap.add_argument("--knn_thresh", type=float, default=0.58)
    ap.add_argument("--knn_alpha",  type=float, default=0.10)
    ap.add_argument("--knn_temp",   type=float, default=0.07)
    ap.add_argument("--knn_angle_gate", type=float, default=20.0)
    ap.add_argument("--save_train_embs", action="store_true", help="train 임베딩 저장 (eval KNN용)")
    
    # argparse에 옵션 추가
    ap.add_argument("--boost_bins", default="2,4,7,10")
    ap.add_argument("--boost_margin", type=float, default=0.70)
    ap.add_argument("--aux_base_w", type=float, default=0.10)  # 라스트마일은 aux 약하게

    args=ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows=load_rows(args.data, args.room_mode); N=len(rows)
    g=torch.Generator().manual_seed(args.seed)
    perm=torch.randperm(N, generator=g)
    ntr=int(0.9*N); tr_idx, va_idx = perm[:ntr], perm[ntr:]

    texts=[r["text"] for r in rows]
    Y=rows_to_tensor(rows)
    AUX=torch.tensor([[r["aux"][0],r["aux"][1],r["aux"][2],r["aux"][3]] for r in rows], dtype=torch.long)

    class DS(torch.utils.data.Dataset):
        def __init__(self,X,Y,A): self.X=X; self.Y=Y; self.A=A
        def __len__(self): return len(self.X)
        def __getitem__(self,i): return self.X[i], self.Y[i], self.A[i]

    # Split
    Xtr=[texts[i] for i in tr_idx.tolist()]; Xva=[texts[i] for i in va_idx.tolist()]
    Ytr,Yva=Y[tr_idx],Y[va_idx]; Atr,Ava=AUX[tr_idx],AUX[va_idx]
    rows_tr=[rows[i] for i in tr_idx.tolist()]  # sampler용 원본

    # Weighted sampler (방향 불균형 + json 'w' 반영)
    if args.weighted_sampler:
        w_samples = build_sampler_weights(
            rows_tr,
            mode=args.sampler_mode,
            power=args.sampler_power,
            use_json_w=(not args.no_json_w),
            clip=args.sampler_clip
        )
        sampler=WeightedRandomSampler(w_samples, num_samples=len(w_samples), replacement=True)
        tr_loader=DataLoader(DS(Xtr,Ytr,Atr), batch_size=args.bsz, sampler=sampler, drop_last=True)
        print(f"[sampler] mode={args.sampler_mode} power={args.sampler_power} clip={args.sampler_clip} json_w={not args.no_json_w}")
    else:
        tr_loader=DataLoader(DS(Xtr,Ytr,Atr), batch_size=args.bsz, shuffle=True, drop_last=True)

    va_loader=DataLoader(DS(Xva,Yva,Ava), batch_size=args.bsz, shuffle=False)
    print(f"[split] train {len(Xtr)} | val {len(Xva)}")

    # Model
    m=T2SModel(args.enc_model).to(dev)
    if args.load:
        sd=torch.load(args.load, map_location="cpu")
        sd = sd.get("state_dict", sd)
        ret=m.load_state_dict(sd, strict=False)
        print(f"[resume] loaded: {args.load} | missing={len(ret.missing_keys)} | unexpected={len(ret.unexpected_keys)}")

    if args.bitfit: m.set_bitfit()
    if args.unfreeze_last_n>0: m.unfreeze_last_n(args.unfreeze_last_n)

    enc_params, head_params = [], []
    for n,p in m.named_parameters():
        if not p.requires_grad: continue
        (enc_params if n.startswith("enc.") else head_params).append(p)
    opt=torch.optim.AdamW(
        [{"params": head_params, "lr": args.lr_head},
         {"params": enc_params,  "lr": args.lr_enc}],
        weight_decay=0.01
    )
    total_steps = max(1, args.epochs * max(1, len(tr_loader)))
    sched=get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(400,total_steps//10), num_training_steps=total_steps)

    best=float("inf")

    # 초기 arc-margin
    m.azarc.set_margin_vec(torch.full((12,), args.m_mid, device=dev))
    bin_weights = torch.ones(12, device=dev)

    # ---------------- Training ----------------
    for ep in range(1, args.epochs+1):
        # ---- 적응형 margin/가중 (직전 val 기반) ----
        m.eval()
        with torch.no_grad():
            mvec, means = compute_adaptive_margin_vec(
                m, va_loader, device=dev,
                base=args.m_base, mid=args.m_mid, hi=args.m_hi, band=args.m_band
            )
            # margin/weights 갱신 직후, set_margin_vec 하기 전에 삽입
            bins = [int(i) for i in args.boost_bins.split(",") if str(i).strip()!=""]
            new = 0.7*m.azarc.m_vec + 0.3*mvec
            for b in bins:
                new[b] = torch.maximum(new[b], torch.tensor(args.boost_margin, device=new.device))
            m.azarc.set_margin_vec(new)

            # 약한 EMA
            new = 0.7*m.azarc.m_vec + 0.3*mvec
            m.azarc.set_margin_vec(new)
            bin_weights = compute_bin_weights(means, gamma=args.gamma_bin_weight, max_w=args.max_bin_weight)
        print("[margin]", [f"{v:.2f}" for v in m.azarc.m_vec.tolist()])
        print("[bin-weights]", [f"{w:.2f}" for w in bin_weights.tolist()])

        # 방향 집중 에폭: 초반엔 방향만 강하게(w_other=0)
        w_dir = 40.0
        w_other = 0.0 if ep <= args.dir_focus_epochs else 1.0

        # ---- train ----
        m.train(); run=0.0
        for texts_b, Yb, Ab in tr_loader:
            texts_b=list(texts_b); Yb=Yb.to(dev); Ab=Ab.to(dev)
            e3,d3,s3,a12=Ab[:,0],Ab[:,1],Ab[:,2],Ab[:,3]
            w = bin_weights[a12].detach()  # per-sample bin weight

            yhat, logits, az12_logits, delta_norm, h = m(texts_b, a12_labels=a12, device=dev)
            Lr, uhat, ugt = loss_reg(yhat, Yb, room_mode=args.room_mode, w_dir=w_dir, w_other=w_other)
            Lc = loss_dir_contrast(uhat, ugt, margin=args.contrast_margin, topk=args.contrast_k, w=args.w_contrast)
            La = loss_aux_weighted(logits, az12_logits, e3,d3,s3,a12,
                                   w_base=0.10, w_az12=2.0, alpha1=0.05, alpha2=0.02, weights=w)
            Lalign = loss_align_sc_with_arc(yhat, az12_logits, temp=args.align_temp, w_align=args.align_w, device=dev)
            Ldelta = loss_delta_weighted(delta_norm, a12, Yb, yhat, weights=w,
                                         w_delta=args.w_delta, w_delta_sc=args.w_delta_sc, device=dev)
            loss = Lr + Lc + La + Lalign + Ldelta
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 0.8)
            opt.step(); opt.zero_grad(); sched.step()
            run += loss.item()

        # ---- val ----
        m.eval()
        with torch.no_grad():
            Ang=[]; dMAE=[]; spMAE=[]; wMAE=[]; gMAE=[]; rMAE=[]
            sums=torch.zeros(12); cnts=torch.zeros(12)
            for texts_b, Yb, Ab in va_loader:
                texts_b=list(texts_b); Yb=Yb.to(dev); a12=Ab[:,3].to(dev)
                yhat,_,_,_,_ = m(texts_b, a12_labels=None, device=dev)
                room_hat = torch.tanh(yhat[:,7])*12.0 if args.room_mode=="drr" else torch.sigmoid(yhat[:,7])*(2.5-0.3)+0.3
                yh=yhat.clone(); yh[:,7]=room_hat
                ang=angle_err_deg(yh, Yb).cpu()
                Ang.append(ang)
                dMAE.append((torch.log1p(yh[:,3])-torch.log1p(Yb[:,3])).abs().cpu())
                spMAE.append((yh[:,4]-Yb[:,4]).abs().cpu())
                wMAE.append((yh[:,5]-Yb[:,5]).abs().cpu())
                gMAE.append((yh[:,6]-Yb[:,6]).abs().cpu())
                rMAE.append((yh[:,7]-Yb[:,7]).abs().cpu())
                for k in range(12):
                    msk=(a12.cpu()==k)
                    if msk.any():
                        sums[k]+=ang[msk].sum()
                        cnts[k]+=msk.sum()
            def catm(x): return torch.cat(x).mean().item()
            AE=catm(Ang); dlog=catm(dMAE); sp=catm(spMAE); wet=catm(wMAE); gain=catm(gMAE); room=catm(rMAE)
            means=torch.where(cnts>0, sums/cnts.clamp_min(1.0), torch.zeros_like(cnts))
            worst3=torch.topk(means, k=3).indices.tolist()
            print(f"epoch {ep:02d}: train {run/len(tr_loader):.4f} | AE {AE:.2f} | dlog {dlog:.3f} | sp {sp:.2f} | wet {wet:.3f} | gain {gain:.2f} | room {room:.2f}")
            print("[val AE per-az12 worst3]:", [(i, float(means[i])) for i in worst3])

            if AE < best:
                best = AE
                meta = {
                    "enc_model": args.enc_model,
                    "seed": args.seed,
                    "room_mode": args.room_mode,
                    "knn_defaults": {
                        "k": 3,
                        "thresh": args.knn_thresh,
                        "alpha": args.knn_alpha,
                        "temp": args.knn_temp,
                        "angle_gate": args.knn_angle_gate
                    }
                }
                torch.save({"state_dict": m.state_dict(), "meta": meta}, args.save)
                print("  saved(best AE):", args.save)

    # ---------------- optional: save train embeddings for KNN eval ----------------
    if args.save_train_embs:
        m.eval()
        embs=[]
        with torch.no_grad():
            for i in range(0, len(Xtr), 256):
                h = m.encode(Xtr[i:i+256], dev).cpu()
                embs.append(h)
        if embs:
            H=torch.cat(embs,0)
            outp=os.path.splitext(args.save)[0] + "_train_embs.pt"
            torch.save({"embeddings": H}, outp)
            print("saved train embeddings:", outp)

if __name__=="__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True,max_split_size_mb:64")
    main()
