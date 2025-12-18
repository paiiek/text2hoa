# -*- coding: utf-8 -*-
# save as: eval_model_metrics_simple.py
import os, json, math, argparse, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from statistics import mean

EPS=1e-7

def prefix_texts(ts):
    out=[]
    for t in ts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

def masked_mean(hs, attn):
    m = attn.unsqueeze(-1).type_as(hs)
    return (hs*m).sum(1)/m.sum(1).clamp_min(1.0)

def angle_err_deg(yh, y):
    s1,c1,el1 = yh[:,0],yh[:,1],yh[:,2]
    s2,c2,el2 = y[:,0],y[:,1],y[:,2]
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

class Model(nn.Module):
    def __init__(self, enc_name, el_max=1.0472):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(enc_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(enc_name)
        d = self.enc.config.hidden_size
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d,768), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(768,384), nn.ReLU(), nn.Linear(384,8))
        self.EL_MAX=el_max

    @torch.no_grad()
    def encode(self, texts, device):
        bt=self.tok(prefix_texts(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        bt={k:v.to(device) for k,v in bt.items()}
        use_amp = device.startswith("cuda")
        import contextlib
        ctx = torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()
        with ctx:
            hs=self.enc(**bt).last_hidden_state
        h=masked_mean(hs, bt["attention_mask"])
        return F.normalize(self.norm(h), dim=-1)

    @torch.no_grad()
    def forward(self, texts, device):
        h=self.encode(texts, device)
        y_raw=self.head(h)
        s=y_raw[:,0]; c=y_raw[:,1]; n=torch.sqrt(s*s+c*c+EPS)
        el=torch.tanh(y_raw[:,2])*self.EL_MAX
        dist=torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread=torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet=torch.sigmoid(y_raw[:,5])
        gain=torch.tanh(y_raw[:,6])*6.0
        room=y_raw[:,7]
        y=torch.stack([s/n, c/n, el, dist, spread, wet, gain, room],1)
        return y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--enc_model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--out_csv", default="baseline_results.csv")
    ap.add_argument("--tag", default="")
    args=ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = load_rows(args.data, args.room_mode)
    texts=[r["text"] for r in rows]
    Y = torch.stack([r["y"] for r in rows]).to(dev)

    m=Model(args.enc_model).to(dev)
    sd=torch.load(args.ckpt, map_location="cpu")
    sd=sd.get("state_dict", sd)
    m.load_state_dict(sd, strict=False)
    m.eval()

    preds=[]
    for i in range(0, len(texts), args.bsz):
        preds.append(m(texts[i:i+args.bsz], device=dev))
    Yh=torch.cat(preds,0)

    AE   = angle_err_deg(Yh, Y).mean().item()
    dlog = (torch.log1p(Yh[:,3])-torch.log1p(Y[:,3])).abs().mean().item()
    sp   = (Yh[:,4]-Y[:,4]).abs().mean().item()
    wet  = (Yh[:,5]-Y[:,5]).abs().mean().item()
    gain = (Yh[:,6]-Y[:,6]).abs().mean().item()
    room = (Yh[:,7]-Y[:,7]).abs().mean().item()

    tag = args.tag if args.tag else f"ours({os.path.basename(args.ckpt)})"
    row = {"variant":tag, "AE_deg":AE, "d_log":dlog, "spread_MAE":sp, "wet_MAE":wet, "gain_MAE":gain, "room_MAE":room}

    write_header = not os.path.exists(args.out_csv)
    import csv
    with open(args.out_csv,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["variant","AE_deg","d_log","spread_MAE","wet_MAE","gain_MAE","room_MAE"])
        if write_header: w.writeheader()
        w.writerow(row)
    print(row)

if __name__=="__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True,max_split_size_mb:64")
    main()
