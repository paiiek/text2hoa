# estimate_phi.py  (argparse + usage)
import argparse, json, math, random, torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_ID = "intfloat/multilingual-e5-small"
TARGET = ["az_sin_start","az_cos_start","az_sin_end","az_cos_end",
          "el_sin_start","el_cos_start","el_sin_end","el_cos_end",
          "dist_z","width_start","width_end","wet_norm","gain_norm"]
INTENT = ["left","right","front","back","up","down","wetter","drier","closer","farther","wider","narrower"]
KO = {"왼":"left","좌":"left","오른":"right","우":"right","앞":"front","전면":"front","뒤":"back","후면":"back",
      "위":"up","위쪽":"up","오버헤드":"up","아래":"down","아랫쪽":"down","웻":"wetter","젖게":"wetter","눅눅":"wetter",
      "드라이":"drier","건조":"drier","가까이":"closer","근접":"closer","앞쪽":"front","멀리":"farther","원거리":"farther",
      "뒤쪽":"back","넓게":"wider","폭넓게":"wider","좁게":"narrower","타이트":"narrower"}
INTS = {"slightly":0.6,"a bit":0.6,"조금":0.6,"좀":0.6,"약간":0.6,"more":1.0,"더":1.0,"much":1.5,"훨씬":1.5,"매우":1.5}

tok = AutoTokenizer.from_pretrained(EMB_ID)
enc = AutoModel.from_pretrained(EMB_ID).to(DEVICE).eval()

@torch.no_grad()
def embed(texts):
    t = tok([f"query: {x}" for x in texts], return_tensors="pt",
            padding=True, truncation=True, max_length=128).to(DEVICE)
    o = enc(**t); m = t["attention_mask"].unsqueeze(-1)
    x = (o.last_hidden_state*m).sum(1)/m.sum(1)
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def feats(texts):
    vs=[]
    for t in texts:
        tl = t.lower()
        for ko,en in KO.items():
            if ko in t: tl += f" {en}"
        v = {k:0.0 for k in INTENT}; w=1.0
        for k,val in INTS.items():
            if k in tl: w=max(w,val)
        for k in v.keys():
            if k in tl: v[k] = w
        vs.append([v[k] for k in INTENT])
    return torch.tensor(vs, dtype=torch.float32, device=DEVICE)

class RegV5(torch.nn.Module):
    def __init__(self, in_dim=384+len(INTENT), out_dim=len(TARGET)):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(in_dim,512), torch.nn.GELU(), torch.nn.LayerNorm(512),
            torch.nn.Linear(512,256), torch.nn.GELU(), torch.nn.LayerNorm(256))
        self.reg_head = torch.nn.Linear(256,out_dim)
        self.cls_start = torch.nn.Linear(256,8); self.cls_end = torch.nn.Linear(256,8)
    def forward(self,x):
        h=self.backbone(x); return self.reg_head(h)

def load_ck(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE)
    m = RegV5().to(DEVICE)
    m.load_state_dict(ck["state_dict"], strict=False)
    m.eval()
    return m

def ang(s,c):
    r=(s*s+c*c+1e-8)**0.5; s,c=s/r,c/r
    a=math.degrees(math.atan2(s,c))
    while a<=-180: a+=360
    while a>180: a-=360
    return a

def circ_delta(a,b):
    d=a-b
    while d<=-180: d+=360
    while d>180: d-=360
    return d

def main():
    ap = argparse.ArgumentParser(description="Estimate global azimuth offset (deg) between predictions and labels.")
    ap.add_argument("ckpt", help="path to v5/v6 checkpoint (text2spatial_head_*.pt)")
    ap.add_argument("files", nargs="+", help="jsonl dataset files")
    ap.add_argument("--nsamp", type=int, default=1000, help="samples to estimate (random)")
    args = ap.parse_args()

    model = load_ck(args.ckpt)

    rows=[]
    for fp in args.files:
        with open(fp,'r',encoding='utf-8') as f:
            for line in f:
                o = json.loads(line)
                if "y" in o and "mask" in o:
                    y=o["y"]; mask=o["mask"]
                else:
                    y,mask={},{}
                    if "az_sc" in o:
                        s,c=o["az_sc"]; y.update({"az_sin_start":s,"az_cos_start":c,"az_sin_end":s,"az_cos_end":c})
                    if "el_rad" in o:
                        el=o["el_rad"]; se,ce=math.sin(el),math.cos(el)
                        y.update({"el_sin_start":se,"el_cos_start":ce,"el_sin_end":se,"el_cos_end":ce})
                    for k in TARGET: mask[k]=1.0 if k in y else 0.0; y.setdefault(k,0.0)
                rows.append((o.get("text",""), y, mask))
    random.shuffle(rows); rows = rows[:min(args.nsamp, len(rows))]
    texts=[r[0] for r in rows]
    X=embed(texts); F=feats(texts); X=torch.cat([X,F],dim=-1)
    P = model(X).detach().cpu().numpy()
    idx={k:i for i,k in enumerate(TARGET)}

    def est_phi(s_key,c_key):
        num=0.0; den=0.0; cnt=0
        for n,(_,y,_) in enumerate(rows):
            ps,pc=P[n, idx[s_key]], P[n, idx[c_key]]
            ts,tc=y[s_key], y[c_key]
            a=ang(ps,pc); b=ang(ts,tc); d=math.radians(circ_delta(a,b))
            num+=math.sin(d); den+=math.cos(d); cnt+=1
        return math.degrees(math.atan2(num, den))

    phi_s = est_phi("az_sin_start","az_cos_start")
    phi_e = est_phi("az_sin_end","az_cos_end")
    print(f"Estimated azimuth offset (deg): start {phi_s:.2f} | end {phi_e:.2f}")
    print("Hint: use the average (rounded) as --phi_init for retrain with AngleAlign.")

if __name__ == "__main__":
    main()
