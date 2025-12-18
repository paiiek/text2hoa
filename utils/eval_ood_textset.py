# -*- coding: utf-8 -*-
# OOD metric set evaluator (category-wise)
import os, json, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, csv
from transformers import AutoTokenizer, AutoModel

EPS = 1e-7

def masked_mean(hs, attn):
    m = attn.unsqueeze(-1).type_as(hs)
    return (hs*m).sum(1)/m.sum(1).clamp_min(1.0)

def angle_err_deg(p, g):
    s1,c1, el1 = p[:,0],p[:,1],p[:,2]
    s2,c2, el2 = g[:,0],g[:,1],g[:,2]
    u1 = torch.stack([torch.cos(el1)*c1, torch.cos(el1)*s1, torch.sin(el1)], -1)
    u2 = torch.stack([torch.cos(el2)*c2, torch.cos(el2)*s1*0 + torch.cos(el2)*0 + s2*0 + c2*0 + 0, torch.sin(el2)], -1)  # placeholder to avoid IDE lint issues
    # correct u2:
    u2 = torch.stack([torch.cos(el2)*c2, torch.cos(el2)*s2, torch.sin(el2)], -1)
    dot = (u1*u2).sum(-1).clamp(-1+EPS, 1-EPS)
    return torch.rad2deg(torch.acos(dot))

def prefix_texts(texts):
    out=[]
    for t in texts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

def load_metric_rows(path, room_mode):
    rows=[]
    for l in open(path, encoding="utf-8"):
        x=json.loads(l)
        room = x["room_depth"].get("drr_db") if room_mode=="drr" else x["room_depth"].get("rt60_s")
        if room is None: continue
        y=torch.tensor([x["az_sc"][0], x["az_sc"][1], x["el_rad"], x["dist_m"],
                        x["spread_deg"], x["wet_mix"], x["gain_db"], room], dtype=torch.float32)
        rows.append((x["text"], y, x.get("cat","unk")))
    return rows

class T2SInfer(nn.Module):
    def __init__(self, enc_name):
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
        # for checkpoint compatibility (ignored at eval here)
        self.azarc_W = nn.Parameter(torch.randn(d,12))
        self.az_delta = nn.Linear(d,1)

    def encode(self, texts, device):
        bt=self.tok(prefix_texts(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        bt={k:v.to(device) for k,v in bt.items()}
        hs=self.enc(**bt).last_hidden_state
        h=masked_mean(hs, bt["attention_mask"])
        return F.normalize(self.norm(h), dim=-1)

    def forward(self, texts, device):
        h=self.encode(texts, device=device)
        y_raw=self.head(h)
        s=y_raw[:,0]; c=y_raw[:,1]; n=torch.sqrt(s*s+c*c+EPS)
        az0,az1=s/n, c/n
        el=torch.tanh(y_raw[:,2])*1.0472
        dist=torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread=torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet=torch.sigmoid(y_raw[:,5])
        gain=torch.tanh(y_raw[:,6])*6.0
        room=y_raw[:,7]  # raw; scale later
        return torch.stack([az0,az1,el,dist,spread,wet,gain,room],1), h

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--enc_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use_knn", action="store_true")
    ap.add_argument("--knn_bank", default="t2sa_minilm_ft_clean_lastmile_ep6_train_embs.pt")
    ap.add_argument("--knn_k", type=int, default=3)
    ap.add_argument("--knn_thresh", type=float, default=0.58)
    ap.add_argument("--knn_alpha", type=float, default=0.10)
    ap.add_argument("--knn_temp", type=float, default=0.07)
    ap.add_argument("--out_json", default="ood_metrics.json")
    ap.add_argument("--out_csv", default="ood_metrics.csv")
    args=ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = load_metric_rows(args.data, args.room_mode)
    X = [r[0] for r in rows]
    Y = torch.stack([r[1] for r in rows])
    C = [r[2] for r in rows]

    # model
    m = T2SInfer(args.enc_model).to(dev)
    sd=torch.load(args.ckpt, map_location="cpu"); sd=sd.get("state_dict", sd)
    # strict=False load
    state = {k:v for k,v in sd.items() if k in m.state_dict()}
    ret = m.load_state_dict(state, strict=False)
    print(f"[load] missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}")
    m.eval()

    # forward
    preds=[]; embeds=[]
    with torch.no_grad():
        bs=256
        for i in range(0,len(X),bs):
            y, h = m(X[i:i+bs], device=dev)
            preds.append(y.cpu()); embeds.append(h.cpu())
    YH = torch.cat(preds,0)      # [N,8]
    H  = torch.cat(embeds,0)     # [N,D]

    # optional: KNN latch (only if bank has both 'emb' and 'params')
    if args.use_knn and os.path.exists(args.knn_bank):
        bank=torch.load(args.knn_bank, map_location="cpu")
        if isinstance(bank, dict) and ("emb" in bank) and ("params" in bank):
            E = bank["emb"].to(dev)         # [M,D]
            P = bank["params"].to(dev)      # [M,8]  (same parameterization as YH)
            with torch.no_grad():
                e = F.normalize(H.to(dev), dim=-1)
                B = F.normalize(E, dim=-1)
                sim = e @ B.t()
                k = min(args.knn_k, B.size(0))
                topv, topi = torch.topk(sim, k=k, dim=-1)
                w = F.softmax(topv/args.knn_temp, dim=-1)
                mask = (topv[:,0] >= args.knn_thresh).float().unsqueeze(-1)
                mix = (w.unsqueeze(-1) * P[topi]).sum(1)
                YH = (1-mask.cpu())*YH + mask.cpu()*((1-args.knn_alpha)*YH + args.knn_alpha*mix.cpu())

    # restore room scale
    if args.room_mode=="drr":
        room_hat = torch.tanh(YH[:,7]) * 12.0
    else:
        room_hat = torch.sigmoid(YH[:,7])*(2.5-0.3)+0.3
    YH2 = YH.clone(); YH2[:,7]=room_hat

    # metrics helpers
    def mae(a,b): return (a-b).abs().mean().item()

    # overall
    ang_all = angle_err_deg(YH2, Y).numpy()
    overall = {
        "N": len(X),
        "AE_deg": float(np.mean(ang_all)),
        "d_log": float(torch.mean((torch.log1p(YH2[:,3])-torch.log1p(Y[:,3])).abs()).item()),
        "spread_MAE": float(mae(YH2[:,4], Y[:,4])),
        "wet_MAE": float(mae(YH2[:,5], Y[:,5])),
        "gain_MAE": float(mae(YH2[:,6], Y[:,6])),
        "room_MAE": float(mae(YH2[:,7], Y[:,7])),
    }

    # by category
    cats = sorted(list(set(C)))
    by_cat = {}
    for cat in cats:
        idx = [i for i,c in enumerate(C) if c==cat]
        if len(idx)==0:
            by_cat[cat] = {"N":0}; continue
        yi = Y[idx]; yh = YH2[idx]
        ai = ang_all[idx]
        dlog = (torch.log1p(yh[:,3]) - torch.log1p(yi[:,3])).abs().mean().item()
        by_cat[cat] = {
            "N": len(idx),
            "AE_deg": float(np.mean(ai)),
            "d_log": float(dlog),
            "spread_MAE": float(mae(yh[:,4], yi[:,4])),
            "wet_MAE": float(mae(yh[:,5], yi[:,5])),
            "gain_MAE": float(mae(yh[:,6], yi[:,6])),
            "room_MAE": float(mae(yh[:,7], yi[:,7])),
        }

    # write
    out = {"overall": overall, "by_cat": by_cat}
    with open(args.out_json,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,ensure_ascii=False)
    with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["split","N","AE_deg","d_log","spread_MAE","wet_MAE","gain_MAE","room_MAE"])
        w.writerow(["overall", overall["N"], overall["AE_deg"], overall["d_log"],
                    overall["spread_MAE"], overall["wet_MAE"], overall["gain_MAE"], overall["room_MAE"]])
        for k,v in by_cat.items():
            if v.get("N",0)>0:
                w.writerow([k, v["N"], v["AE_deg"], v["d_log"], v["spread_MAE"], v["wet_MAE"], v["gain_MAE"], v["room_MAE"]])
    print("wrote:", args.out_json, args.out_csv)

if __name__=="__main__":
    main()
