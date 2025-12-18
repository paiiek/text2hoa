# -*- coding: utf-8 -*-
# save as: eval_ensemble_calib_v1.py
import os, json, math, argparse, random
from collections import defaultdict
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

EPS=1e-7
AZ12_CDEG = torch.tensor([i*30+15 for i in range(12)], dtype=torch.float32)
AZ12_SIN  = torch.sin(torch.deg2rad(AZ12_CDEG))
AZ12_COS  = torch.cos(torch.deg2rad(AZ12_CDEG))

ELEV_UP  = {"머리 위","위쪽","윗쪽","천장","천정","overhead","above","up high","ceiling"}
ELEV_DN  = {"바닥","아래쪽","발치","near the floor","down low","at floor level","from below","underfoot"}
NEAR_W   = {"가까","근처","바로 앞","near","close","intimate","nearby"}
FAR_W    = {"멀리","멀리서","먼 곳","아득히","far away","distant","from afar"}

def contains(t, keys):
    tl=t.lower()
    return any(k.lower() in tl for k in keys)

def U(s,c,el): return torch.stack([torch.cos(el)*c, torch.cos(el)*s, torch.sin(el)], -1)

def angle_err_deg(p, g):
    u1 = U(p[:,0],p[:,1],p[:,2]); u2 = U(g[:,0],g[:,1],g[:,2])
    dot = (u1*u2).sum(-1).clamp(-1+EPS,1-EPS)
    return torch.rad2deg(torch.acos(dot))

def az12_from_sc(s,c):
    az = math.degrees(math.atan2(float(s), float(c)))
    if az < 0: az += 360.0
    return int(az//30)

def prefix_texts(texts):
    out=[]
    for t in texts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

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
        print(f"[load] ckpt={os.path.basename(ckpt)} | missing={len(missing)} | unexpected={len(unexpected)}")

    def encode(self, texts, device):
        bt=self.tok(prefix_texts(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        bt={k:v.to(device) for k,v in bt.items()}
        hs=self.enc(**bt).last_hidden_state
        mask = bt["attention_mask"].unsqueeze(-1).type_as(hs)
        h = (hs*mask).sum(1)/mask.sum(1).clamp_min(1.0)
        h = torch.nn.functional.layer_norm(h, (h.size(-1),))
        return F.normalize(h, dim=-1)

def head_forward_dynamic(h, state_dict, device):
    w_keys = [k for k in state_dict.keys() if k.startswith("head.") and k.endswith(".weight")]
    if len(w_keys) < 3:
        raise RuntimeError(f"head Linear 레이어(>=3) 탐색 실패: found {w_keys}")
    w_keys = sorted(w_keys, key=lambda k: int(k.split(".")[1]))
    b_keys = [k.replace(".weight",".bias") for k in w_keys]

    x = h
    for wi,bi in zip(w_keys[:-1], b_keys[:-1]):
        W = state_dict[wi].to(device); B = state_dict[bi].to(device)
        x = F.linear(x, W, B); x = F.relu(x)
    W = state_dict[w_keys[-1]].to(device); B = state_dict[b_keys[-1]].to(device)
    y_raw = F.linear(x, W, B)

    s,c = y_raw[:,0], y_raw[:,1]; n = torch.sqrt(s*s + c*c + 1e-8)
    az0, az1 = s/n, c/n
    el = torch.tanh(y_raw[:,2]) * 1.0472
    dist = torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
    spread = torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
    wet = torch.sigmoid(y_raw[:,5])
    gain = torch.tanh(y_raw[:,6])*6.0
    room_raw = y_raw[:,7]
    return torch.stack([az0,az1,el,dist,spread,wet,gain,room_raw],1)

def predict_with_model(model, state, texts, bsz, device):
    outs=[]
    with torch.no_grad():
        for i in range(0, len(texts), bsz):
            H = model.encode(texts[i:i+bsz], device)
            outs.append(head_forward_dynamic(H, state, device).cpu())
    return torch.cat(outs,0)

def ensemble_y(list_y, weights=None):
    # list_y: list of [N,8] tensors; weights: list of floats
    if weights is None: weights=[1.0]*len(list_y)
    w = torch.tensor(weights, dtype=torch.float32).view(-1,1)
    Ys = torch.stack(list_y,0)       # [M,N,8]
    # direction: convert to unit vectors then weighted mean
    s=Ys[:,:,0]; c=Ys[:,:,1]; el=Ys[:,:,2]
    u = U(s, c, el)                  # [M,N,3]
    u = F.normalize(u, dim=-1)
    u_mean = (w[:,:,None]*u).sum(0) / max(w.sum().item(), 1e-8)
    u_mean = F.normalize(u_mean, dim=-1)
    # back to (s,c,el)
    x,y,z=u_mean[:,0],u_mean[:,1],u_mean[:,2]
    az=torch.atan2(y,x); s_out=torch.sin(az); c_out=torch.cos(az); el_out=torch.asin(z).clamp(-1.0472,1.0472)
    # scalars (dist,spread,wet,gain,room_raw): weighted mean
    rest = (w[:,:,None]*Ys[:,:,3:]).sum(0)/max(w.sum().item(),1e-8)
    return torch.cat([s_out.unsqueeze(1), c_out.unsqueeze(1), el_out.unsqueeze(1), rest], dim=1)

@torch.no_grad()
def build_lang_predbin_protos(models, states, Xtr, Ytr, Ltr, bsz, device):
    # 1) ensemble prediction on train
    preds=[]
    for m,st in zip(models, states):
        preds.append(predict_with_model(m, st, Xtr, bsz, device))
    y_ens = ensemble_y(preds)  # [N,8]
    a12_pred = torch.tensor([az12_from_sc(float(s), float(c)) for s,c in zip(y_ens[:,0], y_ens[:,1])])

    # 2) lang-specific prototypes
    u_true = U(Ytr[:,0], Ytr[:,1], Ytr[:,2]); u_true = F.normalize(u_true, dim=-1)
    proto = {"ko":{"u":torch.zeros(12,3), "el":torch.zeros(12), "cnt":torch.zeros(12)},
             "en":{"u":torch.zeros(12,3), "el":torch.zeros(12), "cnt":torch.zeros(12)},
             "all":{"u":torch.zeros(12,3), "el":torch.zeros(12), "cnt":torch.zeros(12)}}
    langs = ["ko","en"]
    for k in range(12):
        m_all = (a12_pred==k)
        if m_all.any():
            uu = u_true[m_all]
            proto["all"]["u"][k]  = F.normalize(uu.mean(0, keepdim=True), dim=-1).squeeze(0)
            proto["all"]["el"][k] = Ytr[m_all,2].mean()
            proto["all"]["cnt"][k]= m_all.sum()
        else:
            cen = math.radians(15 + 30*k)
            proto["all"]["u"][k]  = torch.tensor([math.cos(cen), math.sin(cen), 0.0])
            proto["all"]["el"][k] = 0.0
        for lg in langs:
            m = m_all.clone()
            # lang mask
            idxs = [i for i,l in enumerate(Ltr) if l==lg]
            if len(idxs)>0:
                mask_lang = torch.zeros_like(m_all, dtype=torch.bool)
                mask_lang[idxs] = True
                m = m_all & mask_lang
            if m.any():
                uu = u_true[m]
                proto[lg]["u"][k]  = F.normalize(uu.mean(0, keepdim=True), dim=-1).squeeze(0)
                proto[lg]["el"][k] = Ytr[m,2].mean()
                proto[lg]["cnt"][k]= m.sum()
            else:
                proto[lg]["u"][k]  = proto["all"]["u"][k]
                proto[lg]["el"][k] = proto["all"]["el"][k]
    # move to device
    for k in proto.keys():
        for key in ("u","el","cnt"):
            proto[k][key] = proto[k][key].to(device)
    print("[lang-predbin proto] ko/en counts (sum):",
          int(proto["ko"]["cnt"].sum().item()), "/", int(proto["en"]["cnt"].sum().item()))
    return proto

def snap_lang_predbin(yh, langs, proto, alpha=0.25, gate_deg=10.0, boost_bins=None, boost_alpha=0.35):
    # yh: [B,8]; langs: list[str]; proto: dict from build_lang_predbin_protos
    device=yh.device
    a12_pred = torch.tensor([az12_from_sc(float(s), float(c)) for s,c in zip(yh[:,0].cpu(), yh[:,1].cpu())], device=device)
    u_pred = U(yh[:,0], yh[:,1], yh[:,2]); u_pred = F.normalize(u_pred, dim=-1)
    u_new = u_pred.clone(); el_new=yh[:,2].clone()
    for i in range(yh.size(0)):
        lg = langs[i] if langs[i] in ("ko","en") else "all"
        k  = int(a12_pred[i].item())
        u_ref  = proto[lg]["u"][k]; el_ref = proto[lg]["el"][k]
        dot = (u_pred[i]*u_ref).sum().clamp(-1+EPS,1-EPS)
        ddeg = float(torch.rad2deg(torch.acos(dot)).item())
        a = alpha
        if boost_bins and (k in boost_bins): a = max(a, boost_alpha)
        if ddeg > gate_deg and a>0:
            u_new[i] = F.normalize((1-a)*u_pred[i] + a*u_ref, dim=-1)
            el_new[i]= (1-0.5*a)*el_new[i] + (0.5*a)*el_ref
    x,y,z = u_new[:,0],u_new[:,1],u_new[:,2]
    az = torch.atan2(y, x)
    s = torch.sin(az); c = torch.cos(az)
    out = yh.clone(); out[:,0]=s; out[:,1]=c; out[:,2]=el_new
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--ckpts", required=True, help="comma-separated ckpt paths (>=1)")
    ap.add_argument("--weights", default="", help="comma-separated weights (same length as ckpts)")
    ap.add_argument("--enc_model", required=True)
    ap.add_argument("--group_by_text", action="store_true")
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    # KNN (optional, uses first model only)
    ap.add_argument("--use_knn", action="store_true")
    ap.add_argument("--knn_k", type=int, default=3)
    ap.add_argument("--knn_thresh", type=float, default=0.58)
    ap.add_argument("--knn_alpha", type=float, default=0.10)
    ap.add_argument("--knn_temp", type=float, default=0.07)
    ap.add_argument("--knn_angle_gate", type=float, default=20.0)

    # language-aware pred-bin calibration
    ap.add_argument("--predbin_alpha", type=float, default=0.25)
    ap.add_argument("--predbin_gate",  type=float, default=10.0)
    ap.add_argument("--boost_bins", default="2,4,7")
    ap.add_argument("--boost_alpha", type=float, default=0.35)

    ap.add_argument("--save_preds", default="preds_ens.jsonl")
    ap.add_argument("--save_metrics", default="metrics_ens.json")
    args=ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    dev="cuda:0" if torch.cuda.is_available() else "cpu"

    ckpts=[s.strip() for s in args.ckpts.split(",") if s.strip()]
    if args.weights:
        ws=[float(x) for x in args.weights.split(",")]
        assert len(ws)==len(ckpts), "--weights length must match --ckpts"
    else:
        ws=[1.0]*len(ckpts)

    # data split
    rows = load_rows(args.data, args.room_mode)
    texts=[r["text"] for r in rows]; Y=rows_to_tensor(rows)
    langs=[r["lang"] for r in rows]; A12=torch.tensor([r["a12"] for r in rows], dtype=torch.long)
    N=len(rows)
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
    A12te=A12[te_idx]
    Ltr=[langs[i] for i in tr_idx]; Lte=[langs[i] for i in te_idx]

    # build models
    models=[]; states=[]
    for ck in ckpts:
        m=InferenceModel(args.enc_model, ck, max_len=args.max_len).to(dev); m.eval()
        sd=torch.load(ck, map_location="cpu"); st=sd.get("state_dict", sd)
        models.append(m); states.append(st)

    # lang-aware pred-bin prototypes (from ensemble on train)
    proto = build_lang_predbin_protos(models, states, Xtr, Ytr, Ltr, args.bsz, dev)

    # KNN 준비 (옵션, 첫 모델 기준)
    if args.use_knn:
        with torch.no_grad():
            Htr=[]
            for i in range(0, len(Xtr), args.bsz):
                Htr.append(models[0].encode(Xtr[i:i+args.bsz], dev).cpu())
            Htr=torch.cat(Htr,0); Htr=F.normalize(Htr, dim=-1)
        def cosine_knn(q, Kbank, topk=3, temp=0.07):
            sim = q @ Kbank.t(); vals, idx = torch.topk(sim, k=min(topk, Kbank.size(0)), dim=1)
            w = torch.softmax(vals/temp, dim=1); return idx, w, vals

    preds=[]
    with torch.no_grad():
        Ang=[]; dMAE=[]; spMAE=[]; wMAE=[]; gMAE=[]; rMAE=[]
        per_bin_sum=torch.zeros(12); per_bin_cnt=torch.zeros(12)
        lang_err=defaultdict(list)

        for i0 in range(0, len(Xte), args.bsz):
            xs=Xte[i0:i0+args.bsz]; Lb=Lte[i0:i0+args.bsz]
            y_list=[]
            for m,st in zip(models, states):
                y_list.append(predict_with_model(m, st, xs, args.bsz, dev))
            yhat = ensemble_y(y_list, weights=ws).to(dev)

            # KNN (optional) using model[0]
            if args.use_knn:
                Hq=models[0].encode(xs, dev).cpu()
                idx, w, sims = cosine_knn(Hq, Htr, topk=args.knn_k, temp=args.knn_temp)
                sim_max = sims[:,0]; msk = (sim_max >= args.knn_thresh)
                if msk.any():
                    y_knn=[]
                    for j,(row,ww) in enumerate(zip(idx[msk], w[msk])):
                        ys = Ytr[row]  # [k,8]
                        ydir = torch.stack([ys[:,0], ys[:,1], ys[:,2]],1)
                        ydir_hat = (ww.unsqueeze(1)*ydir).sum(0)
                        rest = yhat[msk][j].clone(); rest[:3] = ydir_hat
                        y_knn.append(rest)
                    y_knn = torch.stack(y_knn,0).to(dev)
                    if args.knn_angle_gate>=0:
                        def Uloc(s,c,el): return torch.stack([torch.cos(el)*c, torch.cos(el)*s, torch.sin(el)],-1)
                        yh = yhat[msk]; uk = Uloc(y_knn[:,0], y_knn[:,1], y_knn[:,2]); up = Uloc(yh[:,0], yh[:,1], yh[:,2])
                        d = torch.rad2deg(torch.acos((uk*up).sum(-1).clamp(-1+EPS,1-EPS)))
                        gate = (d <= args.knn_angle_gate)
                        yhat[msk] = torch.where(gate.unsqueeze(1), (1.0-args.knn_alpha)*yh + args.knn_alpha*y_knn, yh)
                    else:
                        yhat[msk] = (1.0-args.knn_alpha)*yhat[msk] + args.knn_alpha*y_knn

            # lang-aware pred-bin snap (+boost for hard bins)
            boost_bins=set(int(x) for x in args.boost_bins.split(",") if x.strip())
            yhat = snap_lang_predbin(yhat, Lb, proto,
                                     alpha=args.predbin_alpha,
                                     gate_deg=args.predbin_gate,
                                     boost_bins=boost_bins,
                                     boost_alpha=args.boost_alpha)

            # room rescale
            room_hat = torch.tanh(yhat[:,7])*12.0 if args.room_mode=="drr" else torch.sigmoid(yhat[:,7])*(2.5-0.3)+0.3
            yh=yhat.clone(); yh[:,7]=room_hat
            yb=Yte[i0:i0+args.bsz].to(dev)

            ang=angle_err_deg(yh, yb).cpu()
            Ang.append(ang)
            dMAE.append((torch.log1p(yh[:,3])-torch.log1p(yb[:,3])).abs().cpu())
            spMAE.append((yh[:,4]-yb[:,4]).abs().cpu())
            wMAE.append((yh[:,5]-yb[:,5]).abs().cpu())
            gMAE.append((yh[:,6]-yb[:,6]).abs().cpu())
            rMAE.append((yh[:,7]-yb[:,7]).abs().cpu())

            a12=A12te[i0:i0+args.bsz]
            for k in range(12):
                msk_k=(a12.cpu()==k)
                if msk_k.any():
                    per_bin_sum[k]+=ang[msk_k].sum()
                    per_bin_cnt[k]+=msk_k.sum()

            for j in range(len(xs)):
                lang=Lb[j]; lang_err[lang].append(float(ang[j]))
                preds.append({
                    "text": xs[j],
                    "lang": lang,
                    "y_true": [float(x) for x in yb[j].cpu().tolist()],
                    "y_pred": [float(x) for x in yh[j].cpu().tolist()],
                    "ae_deg": float(ang[j].item()),
                })

        def catm(x): return torch.cat(x).mean().item()
        AE=catm(Ang); dlog=catm(dMAE); sp=catm(spMAE); wet=catm(wMAE); gain=catm(gMAE); room=catm(rMAE)
        means=torch.where(per_bin_cnt>0, per_bin_sum/per_bin_cnt.clamp_min(1.0), torch.zeros_like(per_bin_sum))
        worst3=torch.topk(means, k=3).indices.tolist()
        lang_metrics={k: sum(v)/max(1,len(v)) for k,v in lang_err.items()}

        metrics={
            "N_test": len(Xte),
            "AE": AE, "dlog": dlog, "spread_mae": sp, "wet_mae": wet, "gain_mae": gain, "room_mae": room,
            "per_az12_mean": [float(means[i]) for i in range(12)],
            "worst3_bins": worst3,
            "lang_AE": lang_metrics,
            "ckpts": [os.path.basename(c) for c in ckpts], "weights": ws,
            "predbin": {"alpha": args.predbin_alpha, "gate": args.predbin_gate, "boost_bins": list(boost_bins), "boost_alpha": args.boost_alpha},
            "use_knn": args.use_knn, "knn": {"k":args.knn_k,"thresh":args.knn_thresh,"alpha":args.knn_alpha,"temp":args.knn_temp,"angle_gate":args.knn_angle_gate},
            "group_by_text": args.group_by_text,
        }
        with open(args.save_metrics,"w",encoding="utf-8") as f:
            json.dump(metrics,f,ensure_ascii=False,indent=2)
        with open(args.save_preds,"w",encoding="utf-8") as f:
            for p in preds: f.write(json.dumps(p,ensure_ascii=False)+"\n")

        print(f"[RESULT] Test N={len(Xte)} | AE {AE:.2f} | dlog {dlog:.3f} | sp {sp:.2f} | wet {wet:.3f} | gain {gain:.2f} | room {room:.2f}")
        print("[worst az12 3]:", [(i, float(means[i])) for i in worst3])
        print("[lang AE]:", {k: round(v,2) for k,v in lang_metrics.items()})

if __name__=="__main__":
    main()
