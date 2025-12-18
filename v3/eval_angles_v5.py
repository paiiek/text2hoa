# eval_angles_v5.py — supports old(net.*), v5(backbone/reg_head/cls), v7(with align.phi)
import os, json, math, random, sys, torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_ID = "intfloat/multilingual-e5-small"
TARGET_KEYS = ["az_sin_start","az_cos_start","az_sin_end","az_cos_end",
               "el_sin_start","el_cos_start","el_sin_end","el_cos_end",
               "dist_z","width_start","width_end","wet_norm","gain_norm"]

INTENT_KEYS = ["left","right","front","back","up","down","wetter","drier","closer","farther","wider","narrower"]
KO_MAP = {"왼":"left","좌":"left","오른":"right","우":"right","앞":"front","전면":"front","뒤":"back","후면":"back",
          "위":"up","위쪽":"up","오버헤드":"up","아래":"down","아랫쪽":"down","웻":"wetter","젖게":"wetter","눅눅":"wetter",
          "드라이":"drier","건조":"drier","가까이":"closer","근접":"closer","앞쪽":"front","멀리":"farther","원거리":"farther",
          "뒤쪽":"back","넓게":"wider","폭넓게":"wider","좁게":"narrower","타이트":"narrower"}
INTENS = {"slightly":0.6,"a bit":0.6,"조금":0.6,"좀":0.6,"약간":0.6,"more":1.0,"더":1.0,"much":1.5,"훨씬":1.5,"매우":1.5}

# ----- embed + features -----
tok = AutoTokenizer.from_pretrained(EMB_ID)
enc = AutoModel.from_pretrained(EMB_ID).to(DEVICE).eval()

@torch.no_grad()
def embed_text(texts):
    inp = tok([f"query: {t}" for t in texts], return_tensors="pt",
              padding=True, truncation=True, max_length=128).to(DEVICE)
    out = enc(**inp); m = inp["attention_mask"].unsqueeze(-1)
    x = (out.last_hidden_state*m).sum(1)/m.sum(1)
    return nn.functional.normalize(x, p=2, dim=-1)

def extract_feats(texts):
    feats=[]
    for t in texts:
        tl=t.lower()
        for ko,en in KO_MAP.items():
            if ko in t: tl+=f" {en}"
        v={k:0.0 for k in INTENT_KEYS}; w=1.0
        for k,val in INTENS.items():
            if k in tl: w=max(w,val)
        for k in v.keys():
            if k in tl: v[k]=w
        feats.append([v[k] for k in INTENT_KEYS])
    return torch.tensor(feats, dtype=torch.float32, device=DEVICE)

# ----- models -----
class RegressorOld(nn.Module):  # net.* (v3/v4)
    def __init__(self, in_dim=384+len(INTENT_KEYS), out_dim=len(TARGET_KEYS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512,256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256,out_dim)
        )
    def forward(self,x): return self.net(x)

class AngleAlign(nn.Module):
    def __init__(self):
        super().__init__()
        # phi는 체크포인트에서 로드됨(없으면 0)
        self.phi = nn.Parameter(torch.zeros(4))  # az_s, az_e, el_s, el_e (radians)
        self.pairs_idx = None  # run-time에 세팅
    def forward(self, y, idx):
        # y: (B,D) ; idx: dict(key->col)
        self.pairs_idx = self.pairs_idx or [
            (idx["az_sin_start"], idx["az_cos_start"]),
            (idx["az_sin_end"],   idx["az_cos_end"]),
            (idx["el_sin_start"], idx["el_cos_start"]),
            (idx["el_sin_end"],   idx["el_cos_end"]),
        ]
        B,D=y.shape
        cols=[y[:,i:i+1] for i in range(D)]
        for j,(si,ci) in enumerate(self.pairs_idx):
            s=y[:,si]; c=y[:,ci]
            r=torch.sqrt(s*s+c*c+1e-8); s=s/r; c=c/r
            cp=torch.cos(self.phi[j]); sp=torch.sin(self.phi[j])
            s2= cp*s - sp*c
            c2= sp*s + cp*c
            cols[si]=s2.unsqueeze(1).contiguous()
            cols[ci]=c2.unsqueeze(1).contiguous()
        return torch.cat(cols, dim=1)

class RegressorV5Aligned(nn.Module):  # backbone/reg_head[/cls] + align
    def __init__(self, in_dim=384+len(INTENT_KEYS), out_dim=len(TARGET_KEYS)):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim,512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512,256), nn.GELU(), nn.LayerNorm(256)
        )
        self.reg_head = nn.Linear(256, out_dim)
        self.cls_start = nn.Linear(256, 8)
        self.cls_end   = nn.Linear(256, 8)
        self.align = AngleAlign()

    def forward(self,x, apply_align=True):
        h=self.backbone(x)
        y=self.reg_head(h)
        if apply_align:
            idx={k:i for i,k in enumerate(TARGET_KEYS)}
            y=self.align(y, idx)
        return y, self.cls_start(h), self.cls_end(h)

def load_model_auto(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    sd = ck["state_dict"]
    # old vs new 판단
    if any(k.startswith("net.") for k in sd.keys()):
        model = RegressorOld().to(DEVICE)
        model.load_state_dict(sd, strict=True)
        kind = "old"
        return model, kind
    # v5/v7 계열
    model = RegressorV5Aligned().to(DEVICE)
    # align.phi가 없으면 strict=False로 로드(기본 0으로 동작)
    strict_flag = "align.phi" in sd
    model.load_state_dict(sd, strict=strict_flag)
    kind = "v7" if strict_flag else "v5"
    model.eval()
    return model, kind

# ----- angle helpers -----
def ang_deg(s,c):
    r=(s*s+c*c+1e-8)**0.5; s,c=s/r,c/r
    a=math.degrees(math.atan2(s,c))
    while a<=-180: a+=360
    while a>180: a-=360
    return a

@torch.no_grad()
def evaluate(ckpt, files, nsamp=800):
    model,kind = load_model_auto(ckpt)

    rows=[]
    for fp in files:
        with open(fp,'r',encoding='utf-8') as f:
            for line in f:
                o=json.loads(line)
                if "y" in o and "mask" in o:
                    y=o["y"]; m=o["mask"]
                else:
                    y,m={},{}
                    if "az_sc" in o:
                        s,c=o["az_sc"]; y.update({"az_sin_start":s,"az_cos_start":c,"az_sin_end":s,"az_cos_end":c})
                    if "el_rad" in o:
                        el=o["el_rad"]; se,ce=math.sin(el),math.cos(el)
                        y.update({"el_sin_start":se,"el_cos_start":ce,"el_sin_end":se,"el_cos_end":ce})
                    if "dist_m" in o:
                        d=max(0.2,min(10.0,float(o["dist_m"])))
                        z01=(math.log(d)-math.log(0.2))/(math.log(10.0)-math.log(0.2))
                        y["dist_z"]=4*z01-2
                    if "spread_deg" in o:
                        w=max(0.0,min(90.0,float(o["spread_deg"])))/90.0
                        y["width_start"]=w; y["width_end"]=w
                    if "wet_mix" in o: y["wet_norm"]=max(0.0,min(1.0,float(o["wet_mix"])))
                    if "gain_db" in o:
                        lo,hi=-18,6; g=max(lo,min(hi,float(o["gain_db"]))); y["gain_norm"]=(g-lo)/(hi-lo)
                    for k in TARGET_KEYS: m[k]=1.0 if k in y else 0.0; y.setdefault(k,0.0)
                rows.append((o.get("text",""), y, m))
    random.shuffle(rows); rows=rows[:min(nsamp,len(rows))]
    texts=[r[0] for r in rows]
    Y=torch.tensor([[r[1][k] for k in TARGET_KEYS] for r in rows],dtype=torch.float32,device=DEVICE)
    M=torch.tensor([[r[2][k] for k in TARGET_KEYS] for r in rows],dtype=torch.float32,device=DEVICE)

    X=embed_text(texts); F=extract_feats(texts); X=torch.cat([X,F],dim=-1)

    if isinstance(model, RegressorOld):
        P = model(X)
    else:
        P, _, _ = model(X, apply_align=True)

    # 각도 MAE(도)
    idx={k:i for i,k in enumerate(TARGET_KEYS)}
    def angle_mae(pair):
        si,ci=idx[pair[0]],idx[pair[1]]
        diffs=[]
        for i in range(P.size(0)):
            if M[i,si]+M[i,ci] > 0:
                diffs.append(abs(ang_deg(P[i,si].item(),P[i,ci].item()) -
                                 ang_deg(Y[i,si].item(),Y[i,ci].item())))
        return sum(diffs)/max(1,len(diffs))
    az_s = angle_mae(("az_sin_start","az_cos_start"))
    az_e = angle_mae(("az_sin_end","az_cos_end"))
    el_s = angle_mae(("el_sin_start","el_cos_start"))
    el_e = angle_mae(("el_sin_end","el_cos_end"))

    mae = ( (P-Y).abs()*M ).sum(0)/(M.sum(0)+1e-8)
    print(f"[ckpt:{kind}] Angle MAE (deg)  | az_start {az_s:.1f} | az_end {az_e:.1f} | el_start {el_s:.1f} | el_end {el_e:.1f}")
    for k,i in idx.items():
        if "sin" in k or "cos" in k: continue
        print(f"{k:>12s} MAE {mae[i].item():.4f}")

if __name__=="__main__":
    ckpt=sys.argv[1]; files=sys.argv[2:]
    evaluate(ckpt, files)
