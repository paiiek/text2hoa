# text2spatial.py — v12t (Az 8-bin + Residual, soft circ mean with temperature, safe keyword margin, AMP-safe)

import os, json, math, random, argparse, contextlib
from collections import Counter
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_ID = "intfloat/multilingual-e5-small"

TARGET_KEYS = [
    "az_sin_start","az_cos_start","az_sin_end","az_cos_end",
    "el_sin_start","el_cos_start","el_sin_end","el_cos_end",
    "dist_z","width_start","width_end","wet_norm","gain_norm"
]

INTENT_KEYS = ["left","right","front","back","up","down","wetter","drier","closer","farther","wider","narrower"]

KO_MAP = {
    "왼":"left","좌":"left","오른":"right","우":"right",
    "앞":"front","전면":"front","뒤":"back","후면":"back",
    "위":"up","위쪽":"up","오버헤드":"up","아래":"down","아랫쪽":"down",
    "웻":"wetter","젖게":"wetter","눅눅":"wetter",
    "드라이":"drier","건조":"drier",
    "가까이":"closer","근접":"closer","앞쪽":"front",
    "멀리":"farther","원거리":"farther","뒤쪽":"back",
    "넓게":"wider","폭넓게":"wider","좁게":"narrower","타이트":"narrower"
}

INTENSIFIERS = {"slightly":0.6,"a bit":0.6,"조금":0.6,"좀":0.6,"약간":0.6,"more":1.0,"더":1.0,"much":1.5,"훨씬":1.5,"매우":1.5}

# =========================
# Embedding (frozen)
# =========================
tok = AutoTokenizer.from_pretrained(EMB_ID)
enc = AutoModel.from_pretrained(EMB_ID).to(DEVICE).eval()

@torch.no_grad()
def embed_text(texts: List[str]) -> torch.Tensor:
    inp = tok([f"query: {t}" for t in texts], return_tensors="pt",
              padding=True, truncation=True, max_length=128).to(DEVICE)
    out = enc(**inp)
    m = inp["attention_mask"].unsqueeze(-1)
    x = (out.last_hidden_state*m).sum(1)/m.sum(1)
    return nn.functional.normalize(x, p=2, dim=-1)  # (B,384)

def extract_intent_feats(texts: List[str]) -> torch.Tensor:
    feats = []
    for t in texts:
        tl = t.lower()
        for ko,en in KO_MAP.items():
            if ko in t: tl += f" {en}"
        v = {k:0.0 for k in INTENT_KEYS}; w = 1.0
        for k,val in INTENSIFIERS.items():
            if k in tl: w = max(w, val)
        for k in v.keys():
            if k in tl: v[k] = w
        feats.append([v[k] for k in INTENT_KEYS])
    return torch.tensor(feats, dtype=torch.float32, device=DEVICE)  # (B,12)

# =========================
# Angle helpers
# =========================
def wrap180(d):
    while d<=-180: d+=360
    while d>180: d-=360
    return d

def angle_from_sc(s, c):
    r=(s*s+c*c+1e-8)**0.5; s,c=s/r,c/r
    a=math.degrees(math.atan2(s,c))
    return wrap180(a)

def label8_from_deg(deg):
    d=(deg+22.5)%360.0
    return int(d//45)

BIN_CENTERS_DEG = [i*45.0 for i in range(8)]
BIN_CENTERS_RAD = [math.radians(d) for d in BIN_CENTERS_DEG]
MAX_DELTA_DEG = 22.5
MAX_DELTA_RAD = math.radians(MAX_DELTA_DEG)

def soft_circ_mean(logits, tau: float = 1.0):
    # temperature-softmax for differentiable, sharpenable circular mean
    p = torch.softmax(logits / max(1e-6, tau), dim=-1)        # (B,8)
    centers = torch.tensor(BIN_CENTERS_RAD, device=logits.device)  # (8,)
    s = torch.sin(centers)                                    # (8,)
    c = torch.cos(centers)
    S = torch.matmul(p, s)                                    # (B,)
    C = torch.matmul(p, c)                                    # (B,)
    return torch.atan2(S, C)                                  # (B,) radians

# =========================
# v3 -> 통합 타깃 변환
# =========================
def v3_to_target(obj: Dict):
    y, m = {}, {}
    if "az_sc" in obj and isinstance(obj["az_sc"], (list,tuple)) and len(obj["az_sc"])==2:
        s,c=float(obj["az_sc"][0]), float(obj["az_sc"][1])
        y.update({"az_sin_start":s,"az_cos_start":c,"az_sin_end":s,"az_cos_end":c})
    if "el_rad" in obj:
        el=float(obj["el_rad"]); se,ce=math.sin(el), math.cos(el)
        y.update({"el_sin_start":se,"el_cos_start":ce,"el_sin_end":se,"el_cos_end":ce})
    if "dist_m" in obj:
        d=max(0.2,min(10.0,float(obj["dist_m"]))); z01=(math.log(d)-math.log(0.2))/(math.log(10.0)-math.log(0.2))
        y["dist_z"]=4.0*z01-2.0
    if "spread_deg" in obj:
        w=max(0.0,min(90.0,float(obj["spread_deg"])))/90.0; y["width_start"]=w; y["width_end"]=w
    if "wet_mix" in obj: y["wet_norm"]=max(0.0,min(1.0,float(obj["wet_mix"])))
    if "gain_db" in obj:
        lo,hi=-18.0,6.0; g=max(lo,min(hi,float(obj["gain_db"]))); y["gain_norm"]=(g-lo)/(hi-lo)
    for k in TARGET_KEYS:
        if k in y: m[k]=1.0
        else: y[k]=0.0; m[k]=0.0
    return y, m

# =========================
# Dataset
# =========================
class Text2SpatialDataset(Dataset):
    def __init__(self, files: List[str]):
        self.items=[]
        for fp in files:
            with open(fp,"r",encoding="utf-8") as f:
                for line in f:
                    o=json.loads(line); text=o.get("text",""); lang=o.get("lang","")
                    if "y" in o and "mask" in o:
                        y=o["y"]; mask=o["mask"]
                        yv=[float(y.get(k,0.0)) for k in TARGET_KEYS]
                        mv=[float(mask.get(k,0.0)) for k in TARGET_KEYS]
                    else:
                        y_conv, m_conv = v3_to_target(o)
                        yv=[y_conv[k] for k in TARGET_KEYS]
                        mv=[m_conv[k] for k in TARGET_KEYS]
                    self.items.append({"text":text,"lang":lang,
                                       "y":torch.tensor(yv,dtype=torch.float32),
                                       "mask":torch.tensor(mv,dtype=torch.float32)})
        random.shuffle(self.items)
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

# =========================
# Model: backbone -> (az bins + residuals) + el angle + scalars
# =========================
class HeadAz(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cls_start = nn.Linear(dim, 8)
        self.cls_end   = nn.Linear(dim, 8)
        self.delta     = nn.Linear(dim, 2)  # [Δstart, Δend] (radians)
    def forward(self, h):
        logits_s = self.cls_start(h)
        logits_e = self.cls_end(h)
        delta    = torch.tanh(self.delta(h)) * MAX_DELTA_RAD
        return logits_s, logits_e, delta

class HeadEl(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.el = nn.Linear(dim, 2)  # start, end radians via tanh
    def forward(self, h):
        return torch.tanh(self.el(h)) * (0.5*math.pi)

class HeadScalars(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 5)  # dist_z, width_start, width_end, wet_norm, gain_norm
    def forward(self, h):
        return self.fc(h)

class Regressor(nn.Module):
    def __init__(self, in_dim=384+len(INTENT_KEYS), out_dim=len(TARGET_KEYS)):
        super().__init__()
        self.backbone=nn.Sequential(
            nn.Linear(in_dim,512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512,256), nn.GELU(), nn.LayerNorm(256)
        )
        self.h_az = HeadAz(256)
        self.h_el = HeadEl(256)
        self.h_sc = HeadScalars(256)

    def forward(self,x, tau: float = 1.0):
        h=self.backbone(x)
        logits_s, logits_e, delta_az = self.h_az(h)               # (B,8),(B,8),(B,2)
        ang_el = self.h_el(h)                                     # (B,2)
        scalars = self.h_sc(h)

        # temperature-soft circular means (분류에도 grad 전달)
        center_s = soft_circ_mean(logits_s, tau=tau)              # (B,)
        center_e = soft_circ_mean(logits_e, tau=tau)              # (B,)
        az_start = center_s + delta_az[:,0]
        az_end   = center_e + delta_az[:,1]
        el_start = ang_el[:,0]; el_end = ang_el[:,1]

        # pack to TARGET_KEYS
        B = x.size(0); out = torch.zeros(B, len(TARGET_KEYS), device=h.device)
        idx = {k:i for i,k in enumerate(TARGET_KEYS)}
        out[:, idx["az_sin_start"]] = torch.sin(az_start)
        out[:, idx["az_cos_start"]] = torch.cos(az_start)
        out[:, idx["az_sin_end"]]   = torch.sin(az_end)
        out[:, idx["az_cos_end"]]   = torch.cos(az_end)
        out[:, idx["el_sin_start"]] = torch.sin(el_start)
        out[:, idx["el_cos_start"]] = torch.cos(el_start)
        out[:, idx["el_sin_end"]]   = torch.sin(el_end)
        out[:, idx["el_cos_end"]]   = torch.cos(el_end)
        out[:, idx["dist_z"]]      = scalars[:,0]
        out[:, idx["width_start"]] = scalars[:,1]
        out[:, idx["width_end"]]   = scalars[:,2]
        out[:, idx["wet_norm"]]    = scalars[:,3]
        out[:, idx["gain_norm"]]   = scalars[:,4]

        return out, logits_s, logits_e, delta_az, ang_el

# =========================
# Losses
# =========================
def circ_loss_from_sc(sp, cp, st, ct):
    ap=torch.atan2(sp,cp); at=torch.atan2(st,ct)
    return 1.0 - torch.cos(ap-at)

def total_loss(pred, logits_s, logits_e, delta_az, ang_el, target, mask, texts,
               w_ce=1.0, w_delta=1.0, w_circ=1.0, w_el=0.5, w_sca=1.0, w_kw=0.3,
               ent_coef: float = 0.0):
    idx={k:i for i,k in enumerate(TARGET_KEYS)}
    B = pred.size(0)

    # true azimuths & labels
    def _ang_from_row(b, s_key, c_key):
        s=float(target[b, idx[s_key]].item()); c=float(target[b, idx[c_key]].item())
        return wrap180(math.degrees(math.atan2(s, c)))
    az_s_true = torch.tensor([_ang_from_row(b,"az_sin_start","az_cos_start") for b in range(B)], device=pred.device)
    az_e_true = torch.tensor([_ang_from_row(b,"az_sin_end","az_cos_end")     for b in range(B)], device=pred.device)
    lab_s = torch.tensor([label8_from_deg((d+360)%360) for d in az_s_true.tolist()], device=pred.device, dtype=torch.long)
    lab_e = torch.tensor([label8_from_deg((d+360)%360) for d in az_e_true.tolist()], device=pred.device, dtype=torch.long)
    centers = torch.tensor(BIN_CENTERS_RAD, device=pred.device)
    cen_s = centers[lab_s]; cen_e = centers[lab_e]

    # residual targets in [-MAX_DELTA, MAX_DELTA]
    def wrap_pi(x):
        while x> math.pi:  x -= 2*math.pi
        while x<=-math.pi: x += 2*math.pi
        return x
    t_delta_s = torch.tensor([max(-MAX_DELTA_RAD, min(MAX_DELTA_RAD, wrap_pi(math.radians(d) - c)))
                              for d,c in zip(az_s_true.tolist(), cen_s.tolist())], device=pred.device)
    t_delta_e = torch.tensor([max(-MAX_DELTA_RAD, min(MAX_DELTA_RAD, wrap_pi(math.radians(d) - c)))
                              for d,c in zip(az_e_true.tolist(), cen_e.tolist())], device=pred.device)

    # CE with smoothing (eps=0.05로 약간 강화)
    def xent_smooth(logits, target_idx, eps=0.05):
        B,C=logits.shape
        with torch.no_grad():
            t=torch.full((B,C), eps/C, device=logits.device)
            t[torch.arange(B), target_idx]+=1.0-eps
        logp=torch.log_softmax(logits, dim=-1)
        return -(t*logp).sum(dim=-1).mean()
    ce = xent_smooth(logits_s, lab_s) + xent_smooth(logits_e, lab_e)

    # residual loss
    delta_loss = torch.nn.functional.smooth_l1_loss(delta_az[:,0], t_delta_s, beta=0.05) + \
                 torch.nn.functional.smooth_l1_loss(delta_az[:,1], t_delta_e, beta=0.05)

    # azimuth circular consistency
    circ=0.0; cnt=0
    for s_key,c_key in [("az_sin_start","az_cos_start"),("az_sin_end","az_cos_end")]:
        si,ci=idx[s_key], idx[c_key]
        l=circ_loss_from_sc(pred[:,si], pred[:,ci], target[:,si], target[:,ci])
        m=torch.clamp(mask[:,si]+mask[:,ci], max=1.0)
        circ+=(l*m).sum(); cnt+=m.sum()
    circ=circ/(cnt+1e-8)

    # elevation circular
    el=0.0; ecnt=0
    for s_key,c_key in [("el_sin_start","el_cos_start"),("el_sin_end","el_cos_end")]:
        si,ci=idx[s_key], idx[c_key]
        l=circ_loss_from_sc(pred[:,si], pred[:,ci], target[:,si], target[:,ci])
        m=torch.clamp(mask[:,si]+mask[:,ci], max=1.0)
        el+=(l*m).sum(); ecnt+=m.sum()
    el=el/(ecnt+1e-8)

    # scalar MSE
    mse=0.0; msum=0.0
    for k in ["dist_z","width_start","width_end","wet_norm","gain_norm"]:
        i=idx[k]; mse += ((pred[:,i]-target[:,i])**2)*mask[:,i]; msum += mask[:,i]
    mse = (mse.sum()/(msum.sum()+1e-8))

    # keyword margin (index-safe)
    pri_bins=[]
    for t in texts:
        tl=t.lower()
        for ko,en in KO_MAP.items():
            if ko in t: tl+=f" {en}"
        cands=[]
        if "front" in tl: cands.append(0.0)
        if "back"  in tl: cands.append(180.0)
        if "left"  in tl: cands.append(+90.0)
        if "right" in tl: cands.append(-90.0)
        if not cands:
            pri_bins.append(None); continue
        sx=sum(math.sin(math.radians(a)) for a in cands)
        cx=sum(math.cos(math.radians(a)) for a in cands)
        ang=wrap180(math.degrees(math.atan2(sx,cx)))
        pri_bins.append(label8_from_deg((ang+360)%360))

    margin=0.35; kw_acc=0.0; mcnt=0
    for logits in (logits_s, logits_e):
        ps=torch.softmax(logits, dim=-1)
        Bps = ps.size(0)
        for b in range(Bps):
            kb = pri_bins[b] if b < len(pri_bins) else None
            if kb is None: continue
            opp=(kb+4)%8
            kw_acc += torch.clamp(margin - (ps[b, kb] - ps[b, opp]), min=0.0)
            mcnt += 1
    kw = (kw_acc/mcnt) if mcnt>0 else torch.tensor(0.0, device=pred.device)

    # delta L2 + logits entropy reg
    delta_l2 = (delta_az[:,0]**2 + delta_az[:,1]**2).mean()
    def entropy_from_logits(logits):
        p = torch.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
    ent_reg = 0.5*(entropy_from_logits(logits_s) + entropy_from_logits(logits_e))

    λ_delta_l2 = 0.05

    loss_total = (w_ce*ce + w_delta*delta_loss + w_circ*circ + w_el*el + w_sca*mse + w_kw*kw
                  + λ_delta_l2*delta_l2 - ent_coef*ent_reg)

    return loss_total, {
        "ce":ce.item(),"delta":delta_loss.item(),"circ":circ.item(),"el":el.item(),
        "mse":mse.item(),"kw":kw.item(),"dl2":delta_l2.item(),"ent":ent_reg.item()
    }

# =========================
# Loader (az-bucket balancing)
# =========================
def build_loader(files, bs=64, num_workers=0):
    ds=Text2SpatialDataset(files)
    idx={k:i for i,k in enumerate(TARGET_KEYS)}
    buckets=[]; counts=Counter()
    for it in ds.items:
        yv=it["y"]; az=angle_from_sc(float(yv[idx["az_sin_start"]]), float(yv[idx["az_cos_start"]]))
        b=label8_from_deg((az+360)%360); buckets.append(b); counts[b]+=1
    w=[]
    for it,b in zip(ds.items, buckets):
        inv=(1.0/max(1,counts[b]))**1.4   # 희소 버킷 강화
        txt=it["text"].lower()
        if any(k in txt for k in ["left","right","front","back","왼","오른","앞","뒤"]): inv*=1.25
        if b==0: inv*=0.85  # front 억제
        w.append(inv)
    w=torch.tensor(w, dtype=torch.double); w=(w/w.sum()).cpu().numpy()
    sampler=WeightedRandomSampler(w, num_samples=len(ds), replacement=True)

    def collate(batch):
        texts=[b["text"] for b in batch]
        y=torch.stack([b["y"] for b in batch]).to(DEVICE)
        m=torch.stack([b["mask"] for b in batch]).to(DEVICE)
        x=embed_text(texts); f=extract_intent_feats(texts); x=torch.cat([x,f],dim=-1)
        return x,y,m,texts

    loader=DataLoader(ds,batch_size=bs,sampler=sampler,num_workers=num_workers,
                      collate_fn=collate,drop_last=False)
    return loader, ds

def load_resume_compat(model: nn.Module, ckpt_path: str):
    ck=torch.load(ckpt_path,map_location=DEVICE)
    missing,unexpected=model.load_state_dict(ck["state_dict"], strict=False)
    print(f"[resume-compat] loaded with strict=False | missing={missing} | unexpected={unexpected}")
    return ck

# =========================
# Train
# =========================
def train(files, epochs=6, bs=64, lr=1e-4, save="text2spatial_head.pt", resume=None, ce_warmup=2):
    loader, ds = build_loader(files, bs)
    model=Regressor().to(DEVICE)

    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.2)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE=="cuda"))
    ac=torch.amp.autocast if DEVICE=="cuda" else contextlib.nullcontext

    if resume and os.path.exists(resume):
        load_resume_compat(model, resume); print(f"[resume] compat-loaded: {resume}")

    best=float("inf")
    for ep in range(1, epochs+1):
        model.train()

        # ---- 스케줄: 가중, 온도, 엔트로피 계수 ----
        if ep <= ce_warmup:
            w = dict(w_ce=1.0, w_delta=1.0, w_circ=0.0, w_el=0.2, w_sca=0.5, w_kw=0.3)
            tau = 0.9
            ent_coef = 0.0
        elif ep == ce_warmup+1:  # full 첫 에폭: CE 더 강, circ 완만, delta 완만
            w = dict(w_ce=0.6, w_delta=0.8, w_circ=0.8, w_el=0.5, w_sca=1.0, w_kw=0.3)
            tau = 0.8
            ent_coef = 0.0
        else:  # 이후: circ↑, CE↓, delta=1.0, tau 더 샤프
            w = dict(w_ce=0.35, w_delta=1.0, w_circ=1.2, w_el=0.5, w_sca=1.0, w_kw=0.3)
            tau = 0.6 if ep == ce_warmup+2 else 0.5
            ent_coef = 0.02

        sum_loss = 0.0; seen = 0

        for x,y,m,texts in loader:
            with ac("cuda", enabled=(DEVICE=="cuda")):
                out, logits_s, logits_e, delta, ang_el = model(x, tau=tau)
                loss, parts = total_loss(out, logits_s, logits_e, delta, ang_el, y, m, texts, ent_coef=ent_coef, **w)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update()
            bs_ = x.size(0); sum_loss += float(loss.item()) * bs_; seen += bs_

        # ---- mini probe (az/el MAE) ----
        model.eval()
        with torch.no_grad():
            idxs=random.sample(range(len(ds)), k=min(256,len(ds)))
            texts_probe=[ds.items[i]["text"] for i in idxs]
            yb=torch.stack([ds.items[i]["y"] for i in idxs]).to(DEVICE)
            mb=torch.stack([ds.items[i]["mask"] for i in idxs]).to(DEVICE)
            xb=embed_text(texts_probe); fb=extract_intent_feats(texts_probe); xb=torch.cat([xb,fb],dim=-1)
            with ac("cuda", enabled=(DEVICE=="cuda")):
                pb, _, _, _, _ = model(xb, tau=0.5)  # 평가 시 샤프하게

            def _ang(s,c):
                r=(s*s+c*c+1e-8)**0.5; s,c=s/r,c/r
                a=math.degrees(math.atan2(s,c)); return wrap180(a)
            idxd={k:i for i,k in enumerate(TARGET_KEYS)}
            def mae_pair(skey, ckey):
                si,ci=idxd[skey], idxd[ckey]
                dif=[]
                for i in range(pb.size(0)):
                    if (mb[i,si]+mb[i,ci])>0:
                        dif.append(abs(_ang(pb[i,si].item(),pb[i,ci].item())-
                                       _ang(yb[i,si].item(),yb[i,ci].item())))
                return sum(dif)/max(1,len(dif))
            az_mae = 0.5*(mae_pair("az_sin_start","az_cos_start")+mae_pair("az_sin_end","az_cos_end"))
            el_mae = 0.5*(mae_pair("el_sin_start","el_cos_start")+mae_pair("el_sin_end","el_cos_end"))

        epoch_loss = sum_loss / max(1, seen)
        print(f"[{ep:02d}] train {epoch_loss:.4f} | azMAE {az_mae:.1f}° | elMAE {el_mae:.1f}° | "
              f"w={w} | tau={tau:.2f} | ent={ent_coef:.2f}")
        if az_mae + 0.5 < best:
            best = az_mae
            torch.save({"state_dict": model.state_dict(), "target_keys": TARGET_KEYS, "emb_id": EMB_ID}, save)
            print("  _ saved(best azMAE):", save)
        sched.step()

# =========================
# Inference & adapter
# =========================
@torch.no_grad()
def infer(texts: List[str], ckpt: str):
    ck=torch.load(ckpt,map_location=DEVICE)
    model=Regressor().to(DEVICE); model.load_state_dict(ck["state_dict"], strict=True); model.eval()
    x=embed_text(texts); f=extract_intent_feats(texts); x=torch.cat([x,f],dim=-1)
    out, *_ = model(x, tau=0.5)  # inference도 샤프하게
    y = out.cpu().tolist()
    return [{k:v for k,v in zip(TARGET_KEYS, row)} for row in y]

def rad2deg(a): return a*180.0/math.pi
def to_angle(s,c):
    r=math.sqrt(s*s+c*c)+1e-8; s,c=s/r,c/r
    return wrap180(rad2deg(math.atan2(s,c)))
def inv_dist_from_z(z):
    z01=(z+2.0)/4.0; return math.exp(z01*(math.log(10.0)-math.log(0.2))+math.log(0.2))
def clamp01(x): return max(0.0,min(1.0,x))
def adapter(params: Dict, engine: str="spat") -> Dict:
    az_s=to_angle(params["az_sin_start"],params["az_cos_start"])
    az_e=to_angle(params["az_sin_end"],params["az_cos_end"])
    el_s=to_angle(params["el_sin_start"],params["el_cos_start"])
    el_e=to_angle(params["el_sin_end"],params["el_cos_end"])
    dist_m=inv_dist_from_z(params["dist_z"])
    w_s=clamp01(params["width_start"])*90.0; w_e=clamp01(params["width_end"])*90.0
    wet=clamp01(params["wet_norm"]); gain_db=-18.0+clamp01(params["gain_norm"])*(6.0+18.0)
    if engine.lower()=="spat":
        return {"spat":{"azimuth_start_deg":az_s,"elevation_start_deg":el_s,"azimuth_end_deg":az_e,"elevation_end_deg":el_e,
                        "distance_m":dist_m,"spread_start_deg":w_s,"spread_end_deg":w_e,"wet":wet,"gain_db":gain_db}}
    if engine.lower()=="resonance":
        return {"resonance":{"azimuth_deg":az_s,"elevation_deg":el_s,"distance_m":dist_m,"reverb_mix":wet,"gain_db":gain_db}}
    return {"raw":params}

# =========================
# CLI
# =========================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["train","infer","demo-map"],required=True)
    ap.add_argument("--files",nargs="*")
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--save",type=str,default="text2spatial_head.pt")
    ap.add_argument("--resume",type=str,default=None)
    ap.add_argument("--ckpt",type=str,default="text2spatial_head.pt")
    ap.add_argument("--text",nargs="*",default=[])
    ap.add_argument("--engine",type=str,default="spat")
    ap.add_argument("--ce_warmup", type=int, default=2)
    args=ap.parse_args()

    if args.mode=="train":
        assert args.files and len(args.files)>0, "--files 필요"
        train(args.files, epochs=args.epochs, bs=args.batch, lr=args.lr,
              save=args.save, resume=args.resume, ce_warmup=args.ce_warmup)

    elif args.mode=="infer":
        assert os.path.exists(args.ckpt)
        outs=infer(args.text,args.ckpt)
        for t,o in zip(args.text,outs):
            print("\n[TEXT]",t); print(json.dumps(o,ensure_ascii=False,indent=2))

    elif args.mode=="demo-map":
        assert os.path.exists(args.ckpt)
        outs=infer(args.text,args.ckpt)
        for t,o in zip(args.text,outs):
            print("\n[TEXT]",t)
            print(json.dumps(adapter(o,engine=args.engine), ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
