# -*- coding: utf-8 -*-
# save as: eval_lastmile_v4.py
import os, json, math, argparse, random
from collections import defaultdict
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

EPS=1e-7

# ---------- text cues (일관성 리포트용) ----------
ELEV_UP  = {"머리 위","위쪽","윗쪽","천장","천정","overhead","above","up high","ceiling"}
ELEV_DN  = {"바닥","아래쪽","발치","near the floor","down low","at floor level","from below","underfoot"}
NEAR_W   = {"가까","근처","바로 앞","near","close","intimate"}
FAR_W    = {"멀리","멀리서","먼 곳","아득히","far away","distant"}

def contains(t, keys): 
    tl=t.lower()
    return any(k.lower() in tl for k in keys)

# ---------- geom helpers ----------
def angle_err_deg(p, g):
    s1,c1, el1 = p[:,0],p[:,1],p[:,2]
    s2,c2, el2 = g[:,0],g[:,1],g[:,2]
    u1 = torch.stack([torch.cos(el1)*c1, torch.cos(el1)*s1, torch.sin(el1)], -1)
    u2 = torch.stack([torch.cos(el2)*c2, torch.cos(el2)*s2, torch.sin(el2)], -1)
    dot = (u1*u2).sum(-1).clamp(-1+EPS,1-EPS)
    return torch.rad2deg(torch.acos(dot))

def az12_from_sc(s,c):
    az = math.degrees(math.atan2(float(s), float(c)))
    if az < 0: az += 360.0
    return int(az//30)

# ---------- data ----------
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
            rows.append({"text": x["text"], "lang": x.get("lang","?"), "y": y, "a12": a12})
    return rows

def rows_to_tensor(rows): return torch.stack([r["y"] for r in rows],0)

# ---------- model (infer-only: encoder + pooling만 사용) ----------
def prefix_texts(texts):
    out=[]
    for t in texts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

class InferenceModel(torch.nn.Module):
    def __init__(self, enc_name, ckpt, max_len=96):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(enc_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(enc_name)
        self.max_len = max_len
        sd=torch.load(ckpt, map_location="cpu")
        meta=sd.get("meta", {})
        self.room_mode = meta.get("room_mode","drr")
        self.enc_name = meta.get("enc_model", enc_name)
        state=sd.get("state_dict", sd)
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"[load] missing={len(missing)} | unexpected={len(unexpected)} | enc={self.enc_name}")

    def encode(self, texts, device):
        bt=self.tok(prefix_texts(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        bt={k:v.to(device) for k,v in bt.items()}
        hs=self.enc(**bt).last_hidden_state
        mask = bt["attention_mask"].unsqueeze(-1).type_as(hs)
        h = (hs*mask).sum(1)/mask.sum(1).clamp_min(1.0)
        h = torch.nn.functional.layer_norm(h, (h.size(-1),))
        return F.normalize(h, dim=-1)

# ---------- head-less forward (state_dict에서 MLP 자동 구성) ----------
def head_forward_dynamic(h, state_dict, device):
    w_keys = [k for k in state_dict.keys() if k.startswith("head.") and k.endswith(".weight")]
    if len(w_keys) < 3:
        raise RuntimeError(f"head Linear 레이어(>=3) 탐색 실패: found {w_keys}")
    w_keys = sorted(w_keys, key=lambda k: int(k.split(".")[1]))
    b_keys = [k.replace(".weight",".bias") for k in w_keys]

    x = h
    for wi,bi in zip(w_keys[:-1], b_keys[:-1]):
        W = state_dict[wi].to(device); B = state_dict[bi].to(device)
        x = F.linear(x, W, B)
        x = F.relu(x)
    W = state_dict[w_keys[-1]].to(device); B = state_dict[b_keys[-1]].to(device)
    y_raw = F.linear(x, W, B)

    s,c = y_raw[:,0], y_raw[:,1]
    n = torch.sqrt(s*s + c*c + 1e-8)
    az0, az1 = s/n, c/n
    el = torch.tanh(y_raw[:,2]) * 1.0472
    dist = torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
    spread = torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
    wet = torch.sigmoid(y_raw[:,5])
    gain = torch.tanh(y_raw[:,6])*6.0
    room_raw = y_raw[:,7]
    return torch.stack([az0,az1,el,dist,spread,wet,gain,room_raw],1)

# ---------- KNN fallback ----------
def cosine_knn(q, Kbank, topk=3, temp=0.07):
    sim = q @ Kbank.t()
    vals, idx = torch.topk(sim, k=min(topk, Kbank.size(0)), dim=1)
    w = torch.softmax(vals/temp, dim=1)
    return idx, w, vals

def apply_knn_blend(yhat, y_knn, alpha=0.10):
    out = yhat.clone()
    out[:,0:3] = (1.0-alpha)*yhat[:,0:3] + alpha*y_knn[:,0:3]
    return out

# ---------- main eval ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--enc_model", required=True)
    ap.add_argument("--group_by_text", action="store_true")
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    # KNN
    ap.add_argument("--use_knn", action="store_true")
    ap.add_argument("--train_data", default=None)
    ap.add_argument("--train_embs", default="")
    ap.add_argument("--knn_k", type=int, default=3)
    ap.add_argument("--knn_thresh", type=float, default=0.58)
    ap.add_argument("--knn_alpha", type=float, default=0.10)
    ap.add_argument("--knn_temp", type=float, default=0.07)
    ap.add_argument("--knn_angle_gate", type=float, default=20.0)

    # outputs
    ap.add_argument("--save_preds", default="preds.jsonl")
    ap.add_argument("--save_metrics", default="metrics.json")
    args=ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    dev="cuda:0" if torch.cuda.is_available() else "cpu"

    rows = load_rows(args.data, args.room_mode)
    texts=[r["text"] for r in rows]; Y=rows_to_tensor(rows)
    langs=[r["lang"] for r in rows]; A12=torch.tensor([r["a12"] for r in rows], dtype=torch.long)
    N=len(rows)

    # split
    if args.group_by_text:
        seen = {}
        for t in texts: seen.setdefault(t,0); seen[t]+=1
        uniq = list(seen.keys()); random.Random(args.seed).shuffle(uniq)
        cut=int(0.9*len(uniq))
        tr_set=set(uniq[:cut]); te_set=set(uniq[cut:])
        tr_idx=[i for i,t in enumerate(texts) if t in tr_set]
        te_idx=[i for i,t in enumerate(texts) if t in te_set]
    else:
        idx=list(range(N)); random.Random(args.seed).shuffle(idx)
        cut=int(0.9*N); tr_idx, te_idx = idx[:cut], idx[cut:]
    print(f"[split] group_by_text: train {len(tr_idx)} | test {len(te_idx)}")

    Xtr=[texts[i] for i in tr_idx]; Xte=[texts[i] for i in te_idx]
    Ytr,Yte=Y[tr_idx],Y[te_idx]
    A12tr,A12te=A12[tr_idx],A12[te_idx]
    Ltr=[langs[i] for i in tr_idx]; Lte=[langs[i] for i in te_idx]

    sd = torch.load(args.ckpt, map_location="cpu")
    state=sd.get("state_dict", sd)

    m = InferenceModel(args.enc_model, args.ckpt, max_len=args.max_len).to(dev)
    m.eval()

    def enc_texts(xx):
        H=[]
        with torch.no_grad():
            for i in range(0, len(xx), args.bsz):
                H.append(m.encode(xx[i:i+args.bsz], dev).cpu())
        return torch.cat(H,0)

    if args.use_knn:
        train_data = args.train_data or args.data
        assert train_data==args.data, "KNN 일관성 위해 --train_data는 --data와 동일 권장"
        Htr=None
        if args.train_embs and os.path.exists(args.train_embs):
            blob=torch.load(args.train_embs, map_location="cpu")
            Htr=blob.get("embeddings", None)
            if Htr is not None and Htr.size(0)!=len(Xtr):
                print(f"[warn] train_embs rows {Htr.size(0)} != Xtr {len(Xtr)}. 재계산")
                Htr=None
        if Htr is None:
            Htr=enc_texts(Xtr)
        Htr=F.normalize(Htr, dim=-1)

    def predict_batch(xx):
        with torch.no_grad():
            H=m.encode(xx, dev)
            yhat=head_forward_dynamic(H, state, dev)
        return yhat, H

    preds=[]
    with torch.no_grad():
        Ang=[]; dMAE=[]; spMAE=[]; wMAE=[]; gMAE=[]; rMAE=[]
        per_bin_sum=torch.zeros(12); per_bin_cnt=torch.zeros(12)
        lang_err=defaultdict(list)

        # keyword consistency (GT & PRED 둘 다 계산)
        stats = {"gt": {"up":[0,0],"down":[0,0],"near":[0,0],"far":[0,0]},
                 "pred":{"up":[0,0],"down":[0,0],"near":[0,0],"far":[0,0]}}
        def acc(stat, key, hit):
            stat[key][1]+=1; stat[key][0]+=int(hit)

        for i0 in range(0, len(Xte), args.bsz):
            xs=Xte[i0:i0+args.bsz]
            yb=Yte[i0:i0+args.bsz].to(dev)
            a12=A12te[i0:i0+args.bsz]
            yhat, H = predict_batch(xs)

            # KNN
            knn_used=[False]*len(xs)
            if args.use_knn:
                idx, w, sims = cosine_knn(H.cpu(), Htr, topk=args.knn_k, temp=args.knn_temp)
                sim_max = sims[:,0]
                msk = (sim_max >= args.knn_thresh)
                if msk.any():
                    y_knn=[]
                    for j,(row,ww) in enumerate(zip(idx[msk], w[msk])):
                        ys = Ytr[row]  # [k,8]
                        ydir = torch.stack([ys[:,0], ys[:,1], ys[:,2]],1)
                        ydir_hat = (ww.unsqueeze(1)*ydir).sum(0)
                        rest = yhat[msk][j].clone()
                        rest[:3] = ydir_hat
                        y_knn.append(rest)
                    y_knn = torch.stack(y_knn,0).to(dev)
                    if args.knn_angle_gate>=0:
                        def U(s,c,el): return torch.stack([torch.cos(el)*c, torch.cos(el)*s, torch.sin(el)],-1)
                        yh = yhat[msk]; uk = U(y_knn[:,0], y_knn[:,1], y_knn[:,2]); up = U(yh[:,0], yh[:,1], yh[:,2])
                        d = torch.rad2deg(torch.acos((uk*up).sum(-1).clamp(-1+EPS,1-EPS)))
                        gate = (d <= args.knn_angle_gate)
                        yhat[msk] = torch.where(gate.unsqueeze(1), apply_knn_blend(yh, y_knn, alpha=args.knn_alpha), yh)
                        ki = torch.nonzero(msk, as_tuple=False).squeeze(1)
                        # FIX: 배치 인덱스 그대로 사용 (i0 빼지 않음)
                        for gi, ok in zip(ki.tolist(), gate.tolist()):
                            if 0 <= gi < len(knn_used):
                                knn_used[gi] = bool(ok)
                    else:
                        yhat[msk] = apply_knn_blend(yhat[msk], y_knn, alpha=args.knn_alpha)
                        ki = torch.nonzero(msk, as_tuple=False).squeeze(1)
                        for gi in ki.tolist():
                            if 0 <= gi < len(knn_used):
                                knn_used[gi] = True

            # room rescale
            room_hat = torch.tanh(yhat[:,7])*12.0 if args.room_mode=="drr" else torch.sigmoid(yhat[:,7])*(2.5-0.3)+0.3
            yh=yhat.clone(); yh[:,7]=room_hat

            ang=angle_err_deg(yh, yb).cpu()
            Ang.append(ang)
            dMAE.append((torch.log1p(yh[:,3])-torch.log1p(yb[:,3])).abs().cpu())
            spMAE.append((yh[:,4]-yb[:,4]).abs().cpu())
            wMAE.append((yh[:,5]-yb[:,5]).abs().cpu())
            gMAE.append((yh[:,6]-yb[:,6]).abs().cpu())
            rMAE.append((yh[:,7]-yb[:,7]).abs().cpu())

            # per-az12
            a_pred = torch.tensor([az12_from_sc(float(s), float(c)) for s,c in zip(yh[:,0].cpu(), yh[:,1].cpu())])
            for k in range(12):
                msk_k=(a12.cpu()==k)
                if msk_k.any():
                    per_bin_sum[k]+=ang[msk_k].sum()
                    per_bin_cnt[k]+=msk_k.sum()

            # lang split + kw consistency
            for j,t in enumerate(xs):
                lang=Lte[i0+j]; lang_err[lang].append(float(ang[j]))
                # GT 기준
                gt_el=float(yb[j,2]); gt_d=float(yb[j,3])
                if contains(t, ELEV_UP):  acc(stats["gt"],   "up",   gt_el>0)
                if contains(t, ELEV_DN):  acc(stats["gt"],   "down", gt_el<0)
                if contains(t, NEAR_W):   acc(stats["gt"],   "near", gt_d<1.5)
                if contains(t, FAR_W):    acc(stats["gt"],   "far",  gt_d>=3.0)
                # Pred 기준
                pr_el=float(yh[j,2]); pr_d=float(yh[j,3])
                if contains(t, ELEV_UP):  acc(stats["pred"], "up",   pr_el>0)
                if contains(t, ELEV_DN):  acc(stats["pred"], "down", pr_el<0)
                if contains(t, NEAR_W):   acc(stats["pred"], "near", pr_d<1.5)
                if contains(t, FAR_W):    acc(stats["pred"], "far",  pr_d>=3.0)

            # save preds
            for j in range(len(xs)):
                yt = yb[j].cpu(); yp = yh[j].cpu()
                preds.append({
                    "text": xs[j],
                    "lang": Lte[i0+j],
                    "a12_true": int(a12[j].cpu().item()),
                    "a12_pred": int(a_pred[j].cpu().item()),
                    "y_true": [float(x) for x in yt.tolist()],
                    "y_pred": [float(x) for x in yp.tolist()],
                    "ae_deg": float(ang[j].item()),
                    "knn_used": knn_used[j]
                })

        def catm(x): return torch.cat(x).mean().item()
        AE=catm(Ang); dlog=catm(dMAE); sp=catm(spMAE); wet=catm(wMAE); gain=catm(gMAE); room=catm(rMAE)
        means=torch.where(per_bin_cnt>0, per_bin_sum/per_bin_cnt.clamp_min(1.0), torch.zeros_like(per_bin_sum))
        worst3=torch.topk(means, k=3).indices.tolist()

        lang_metrics={k: sum(v)/max(1,len(v)) for k,v in lang_err.items()}
        def pct(hit_tot): 
            hit, tot = hit_tot
            return round(100.0*hit/max(1,tot),1)
        kw_report={
            "GT":   {"up%":pct(stats["gt"]["up"]), "down%":pct(stats["gt"]["down"]), "near%":pct(stats["gt"]["near"]), "far%":pct(stats["gt"]["far"])},
            "PRED": {"up%":pct(stats["pred"]["up"]), "down%":pct(stats["pred"]["down"]), "near%":pct(stats["pred"]["near"]), "far%":pct(stats["pred"]["far"])},
        }

        metrics={
            "N_test": len(Xte),
            "AE": AE, "dlog": dlog, "spread_mae": sp, "wet_mae": wet, "gain_mae": gain, "room_mae": room,
            "per_az12_mean": [float(means[i]) for i in range(12)],
            "worst3_bins": worst3,
            "lang_AE": lang_metrics,
            "use_knn": args.use_knn, "knn": {"k":args.knn_k,"thresh":args.knn_thresh,"alpha":args.knn_alpha,"temp":args.knn_temp,"angle_gate":args.knn_angle_gate},
            "group_by_text": args.group_by_text,
            "ckpt": args.ckpt, "enc_model": args.enc_model,
            "kw_consistency": kw_report
        }
        with open(args.save_metrics,"w",encoding="utf-8") as f:
            json.dump(metrics,f,ensure_ascii=False,indent=2)
        with open(args.save_preds,"w",encoding="utf-8") as f:
            for p in preds: f.write(json.dumps(p,ensure_ascii=False)+"\n")

        print(f"[RESULT] Test N={len(Xte)} | AE {AE:.2f} | dlog {dlog:.3f} | sp {sp:.2f} | wet {wet:.3f} | gain {gain:.2f} | room {room:.2f}")
        print("[worst az12 3]:", [(i, float(means[i])) for i in worst3])
        print("[lang AE]:", {k: round(v,2) for k,v in lang_metrics.items()})
        print("[kw consistency]:", kw_report)

if __name__=="__main__":
    main()
