# -*- coding: utf-8 -*-
# drop-in replacement for bakeoff_encoders_v1.py
import os, json, math, argparse, random, re
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

random.seed(42); torch.manual_seed(42)
EPS=1e-7; DEG=math.pi/180

def az_deg_from_sc(s,c):
    a=math.degrees(math.atan2(float(s), float(c)))
    return a+360 if a<0 else a
def az12_from_sc(s,c): return int(az_deg_from_sc(s,c)//30)
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
            rows.append({
              "text": x["text"],
              "y": torch.tensor([
                x["az_sc"][0], x["az_sc"][1], x["el_rad"], x["dist_m"],
                x["spread_deg"], x["wet_mix"], x["gain_db"], room
              ], dtype=torch.float32),
              "aux": torch.tensor([
                az12_from_sc(x["az_sc"][0], x["az_sc"][1]),
                el_bin(x["el_rad"]),
                dist_bin(x["dist_m"]),
                spr_bin(x["spread_deg"])
              ], dtype=torch.long)
            })
    return rows

def prefix_texts(texts):
    out=[]
    for t in texts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

def cache_embeddings(data_path, room_mode, enc_name, max_len=64, outfile=None, device="cuda"):
    from transformers import AutoTokenizer, AutoModel
    rows = load_rows(data_path, room_mode)
    texts=[r["text"] for r in rows]; texts=prefix_texts(texts)
    tok = AutoTokenizer.from_pretrained(enc_name, use_fast=True, trust_remote_code=True)
    enc = AutoModel.from_pretrained(enc_name, trust_remote_code=True).to(device)
    enc.eval(); torch.set_grad_enabled(False)
    H=[]
    for i in range(0, len(texts), 256):
        bt = tok(texts[i:i+256], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        hs = enc(**bt).last_hidden_state
        mask = bt["attention_mask"].unsqueeze(-1).type_as(hs)
        h = (hs*mask).sum(1) / mask.sum(1).clamp_min(1.0)
        h = F.normalize(h, dim=-1)
        H.append(h.cpu())
    H=torch.cat(H,0)  # [N,d]
    Y=torch.stack([r["y"] for r in rows],0)         # [N,8]
    AUX=torch.stack([r["aux"] for r in rows],0)     # [N,4]  [az12, el3, d3, s3]
    if outfile is None:
        safe = re.sub(r"[^A-Za-z0-9\-_.]+","_",enc_name)
        outfile=f"cache_{safe}.pt"
    torch.save({"H":H, "Y":Y, "AUX":AUX, "enc":enc_name}, outfile)
    print(f"[cache] {enc_name} -> {outfile}, H={tuple(H.shape)}")
    return outfile

# ---------- ArcFace (autocast off + dtype 일관) ----------
class ArcMarginAz12(nn.Module):
    def __init__(self, d, s=35.0, m=0.35):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d,12))
        nn.init.xavier_uniform_(self.W)
        self.s=s; self.m=m
    def forward(self, h, y=None):
        # autocast 비활성: float32 고정
        try:
            ac = lambda enabled: torch.amp.autocast('cuda', enabled=enabled)
        except TypeError:
            ac = lambda enabled: torch.cuda.amp.autocast(enabled=enabled)
        with ac(False):
            h32 = F.normalize(h.float(), dim=-1)
            W32 = F.normalize(self.W.float(), dim=0)
            logits32 = (h32 @ W32).clamp(-1+1e-7,1-1e-7)
            if y is None:
                return (logits32 * self.s)  # float32
            idx = torch.arange(h32.size(0), device=h32.device)
            theta32 = torch.acos(logits32)
            target = torch.cos(theta32[idx, y] + self.m)
            logits_m = logits32.clone()
            # dtype 안전
            logits_m[idx, y] = target.to(logits_m.dtype)
            return (logits_m * self.s)

class Head(nn.Module):
    def __init__(self, d, el_max=1.0472):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 8)
        )
        self.aux = nn.ModuleDict({
            "elev": nn.Linear(d,3),
            "dist": nn.Linear(d,3),
            "sprd": nn.Linear(d,3),
        })
        self.azarc = ArcMarginAz12(d, s=35.0, m=0.35)
        self.EL_MAX = el_max
    def forward(self, h, a12=None):
        y_raw = self.mlp(h)
        s=y_raw[:,0]; c=y_raw[:,1]; n=torch.sqrt(s*s+c*c+EPS)
        s,c = s/n, c/n
        el=torch.tanh(y_raw[:,2])*self.EL_MAX
        dist=torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread=torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet=torch.sigmoid(y_raw[:,5])
        gain=torch.tanh(y_raw[:,6])*6.0
        room=y_raw[:,7]  # drr raw
        y=torch.stack([s,c,el,dist,spread,wet,gain,room],1)
        logits={k:hd(h) for k,hd in self.aux.items()}
        az12_logits=self.azarc(h, a12) if a12 is not None else self.azarc(h,None)
        return y, logits, az12_logits

AZ12_CDEG=torch.tensor([i*30+15 for i in range(12)], dtype=torch.float32)
AZ12_SIN=torch.sin(torch.deg2rad(AZ12_CDEG)); AZ12_COS=torch.cos(torch.deg2rad(AZ12_CDEG))

def angle_err_deg(yh, Y):
    s1,c1,el1=yh[:,0],yh[:,1],yh[:,2]
    s2,c2,el2=Y[:,0], Y[:,1], Y[:,2]
    u1=torch.stack([torch.cos(el1)*c1, torch.cos(el1)*s1, torch.sin(el1)],-1)
    u2=torch.stack([torch.cos(el2)*c2, torch.cos(el2)*s2, torch.sin(el2)],-1)
    dot=(u1*u2).sum(-1).clamp(-1+EPS,1-EPS)
    return torch.rad2deg(torch.acos(dot))

def loss_reg(yh, Y, w_dir=40.0):
    def u(s,c,el): return torch.stack([torch.cos(el)*c, torch.cos(el)*s, torch.sin(el)],-1)
    l_dir = (1.0-(u(yh[:,0],yh[:,1],yh[:,2]) * u(Y[:,0],Y[:,1],Y[:,2])).sum(-1).clamp(-1+EPS,1-EPS)).mean()
    l_room = (torch.tanh(yh[:,7])*12.0 - Y[:,7]).abs().mean()
    l_misc = (torch.log1p(yh[:,3])-torch.log1p(Y[:,3])).abs().mean() + (yh[:,4]-Y[:,4]).abs().mean() + (yh[:,5]-Y[:,5]).abs().mean() + (yh[:,6]-Y[:,6]).abs().mean()
    return w_dir*l_dir + l_misc + l_room

def loss_aux(logits, arc_logits, a12, e3,d3,s3, w_base=0.25, w_arc=2.0):
    le=F.cross_entropy(logits["elev"], e3, label_smoothing=0.05)
    ld=F.cross_entropy(logits["dist"], d3, label_smoothing=0.05)
    ls=F.cross_entropy(logits["sprd"], s3, label_smoothing=0.05)
    la=F.cross_entropy(arc_logits, a12)
    return w_base*(le+ld+ls) + w_arc*la

def loss_align(yh, arc_logits, temp=1.5, w=8.0, device="cuda"):
    p=F.softmax(arc_logits/temp, dim=-1)
    s_exp=(p*AZ12_SIN.to(device)).sum(-1)
    c_exp=(p*AZ12_COS.to(device)).sum(-1)
    return w*((yh[:,0]-s_exp)**2 + (yh[:,1]-c_exp)**2).mean()

class EmbDataset(Dataset):
    def __init__(self, H,Y,A): self.H=H; self.Y=Y; self.A=A
    def __len__(self): return self.H.size(0)
    def __getitem__(self,i): return self.H[i], self.Y[i], self.A[i]

def train_head_cached(cache_pt, epochs=12, bsz=512, device="cuda"):
    blob=torch.load(cache_pt, map_location="cpu")
    H, Y, AUX = blob["H"], blob["Y"], blob["AUX"]  # AUX: [az12, el3, d3, s3]
    N=H.size(0); idx=torch.randperm(N)
    ntr=int(0.9*N); tr=idx[:ntr]; va=idx[ntr:]
    d=H.size(1); Htr,Hva=H[tr],H[va]; Ytr,Yva=Y[tr],Y[va]; Atr,Ava=AUX[tr],AUX[va]
    ds_tr=EmbDataset(Htr,Ytr,Atr); ds_va=EmbDataset(Hva,Yva,Ava)
    dl_tr=DataLoader(ds_tr,batch_size=bsz,shuffle=True,drop_last=True)
    dl_va=DataLoader(ds_va,batch_size=bsz,shuffle=False)

    m=Head(d).to(device)
    opt=torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)

    # AMP 비활성 (dtype 충돌 방지 + 헤드만이라 속도 영향 미미)
    for ep in range(1,epochs+1):
        m.train(); run=0.0
        for Hb,Yb,Ab in dl_tr:
            Hb=Hb.to(device).float(); Yb=Yb.to(device).float(); Ab=Ab.to(device)
            a12, e3, d3, s3 = Ab[:,0],Ab[:,1],Ab[:,2],Ab[:,3]
            yh, logits, az = m(Hb, a12=a12)
            Lr=loss_reg(yh, Yb, w_dir=40.0)
            La=loss_aux(logits, az, a12,e3,d3,s3, w_base=0.25, w_arc=2.0)
            Lalign=loss_align(yh, az, temp=1.5, w=8.0, device=device)
            loss=Lr+La+Lalign
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 0.8)
            opt.step(); opt.zero_grad()
            run+=loss.item()
        # val
        m.eval(); Ang=[]
        with torch.no_grad():
            for Hb,Yb,_ in dl_va:
                Hb=Hb.to(device).float(); Yb=Yb.to(device).float()
                yh,_,_=m(Hb, a12=None)
                room_hat=torch.tanh(yh[:,7])*12.0
                yh2=yh.clone(); yh2[:,7]=room_hat
                Ang.append(angle_err_deg(yh2, Yb).cpu())
        AE=torch.cat(Ang).mean().item()
        print(f"  epoch {ep:02d} | train {run/len(dl_tr):.4f} | AE {AE:.2f}")
    return AE  # 마지막 값(간단화)

def sanitize(name): return re.sub(r"[^A-Za-z0-9\-_.]+","_",name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--encoders", nargs="+", default=[
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
        "Alibaba-NLP/gte-multilingual-base",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ])
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bsz", type=int, default=512)
    ap.add_argument("--cache_only", action="store_true")
    args=ap.parse_args()

    device="cuda:0" if torch.cuda.is_available() else "cpu"
    results=[]
    for enc in args.encoders:
        cache_pt=f"cache_{sanitize(enc)}.pt"
        if not os.path.exists(cache_pt):
            cache_embeddings(args.data, args.room_mode, enc, args.max_len, cache_pt, device=device)
        else:
            print(f"[cache exists] {cache_pt}")
        if args.cache_only: continue
        print(f"\n=== Train head on {enc} ===")
        best = train_head_cached(cache_pt, epochs=args.epochs, bsz=args.bsz, device=device)
        results.append((enc, best))
    if results:
        results.sort(key=lambda x: x[1])
        print("\n\n=== Leaderboard (lower AE better) ===")
        for enc,ae in results:
            print(f"{ae:6.2f}°  |  {enc}")

if __name__=="__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True,max_split_size_mb:64")
    main()
