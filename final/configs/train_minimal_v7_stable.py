# -*- coding: utf-8 -*-
# save as: train_minimal_v7_stable.py
import os, json, math, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from typing import List

ENC = "intfloat/multilingual-e5-large"
TOK = AutoTokenizer.from_pretrained(ENC, use_fast=True)

random.seed(0); torch.manual_seed(0)

# ----------------- utils -----------------
def angle_err_deg(p, g):
    s1,c1, el1 = p[:,0],p[:,1],p[:,2]
    s2,c2, el2 = g[:,0],g[:,1],g[:,2]
    u1 = torch.stack([torch.cos(el1)*c1, torch.cos(el1)*s1, torch.sin(el1)], dim=-1)
    u2 = torch.stack([torch.cos(el2)*c2, torch.cos(el2)*s2, torch.sin(el2)], dim=-1)
    dot = (u1*u2).sum(-1).clamp(-0.999999,0.999999)
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

# ----------------- dataset -----------------
class DS(Dataset):
    def __init__(self, path, room_mode):
        self.rows=[]; self.room_mode=room_mode
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
                self.rows.append({"text": x["text"], "y": y, "aux": (q,eb,db,sb,a12)})
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def _prefix_texts(texts: List[str]) -> List[str]:
    out=[]
    for t in texts:
        tl=t.lstrip().lower()
        out.append("query: "+t if not (tl.startswith("query:") or tl.startswith("passage:") or tl.startswith("spatial:")) else t)
    return out

def collate_batch(batch):
    texts=_prefix_texts([b["text"] for b in batch])
    be = TOK(texts, padding=True, truncation=True, return_tensors="pt")
    toks = {"input_ids": be["input_ids"], "attention_mask": be["attention_mask"]}
    if "token_type_ids" in be: toks["token_type_ids"] = be["token_type_ids"]
    y=torch.stack([b["y"] for b in batch], dim=0)
    aux = tuple(torch.tensor([b["aux"][k] for b in batch], dtype=torch.long) for k in range(5))  # (q,eb,db,sb,a12)
    return {"tokens": toks, "y": y, "aux": aux}

# ----------------- model -----------------
class Head8(nn.Module):
    def __init__(self, d, out=8):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,768), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768,384), nn.ReLU(),
            nn.Linear(384,out)
        )
    def forward(self,x): return self.net(x)

class AuxHeads(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qhead=nn.Linear(d,4); self.ehead=nn.Linear(d,3)
        self.dhead=nn.Linear(d,3); self.shead=nn.Linear(d,3)
        self.a12 =nn.Linear(d,12)
    def forward(self,h):
        return {"quad":self.qhead(h),"elev":self.ehead(h),
                "dist":self.dhead(h),"sprd":self.shead(h),"az12":self.a12(h)}

class Model(nn.Module):
    def __init__(self, room_mode:str):
        super().__init__()
        self.room_mode=room_mode
        self.enc=AutoModel.from_pretrained(ENC)
        d=self.enc.config.hidden_size
        self.head=Head8(d,8)
        self.aux = AuxHeads(d)
        self.register_buffer("_dev_anchor", torch.empty(0))
    def encode(self, input_ids, attention_mask, token_type_ids=None):
        dev=self._dev_anchor.device
        input_ids=input_ids.to(dev, non_blocking=True)
        attention_mask=attention_mask.to(dev, non_blocking=True)
        if token_type_ids is not None:
            token_type_ids=token_type_ids.to(dev, non_blocking=True)
            hs=self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        else:
            hs=self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(hs)
        return (hs*mask).sum(1)/mask.sum(1).clamp_min(1.0)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        h=self.encode(input_ids, attention_mask, token_type_ids)
        y_raw=self.head(h)
        s=y_raw[:,0]; c=y_raw[:,1]; n=torch.sqrt(s*s+c*c+1e-8)
        az0=s/n; az1=c/n
        el=torch.tanh(y_raw[:,2])*0.7854
        dist=torch.sigmoid(y_raw[:,3])*(6.0-0.6)+0.6
        spread=torch.sigmoid(y_raw[:,4])*(120.0-5.0)+5.0
        wet=torch.sigmoid(y_raw[:,5])
        gain=torch.tanh(y_raw[:,6])*6.0
        if self.room_mode=="drr": room=torch.tanh(y_raw[:,7])*12.0
        else: room=torch.sigmoid(y_raw[:,7])*(2.5-0.3)+0.3
        y=torch.stack([az0,az1,el,dist,spread,wet,gain,room],1)
        aux_logits = self.aux(h if any(p.requires_grad for p in self.enc.parameters()) else h.detach())
        return y, h.detach(), aux_logits

# ----------------- losses -----------------
def loss_reg(yhat, y, w_dir=10.0, w_other=1.0):
    def to_u(s, c, el):
        x=torch.cos(el)*c; yv=torch.cos(el)*s; z=torch.sin(el)
        return torch.stack([x,yv,z],-1)
    uhat=to_u(yhat[:,0],yhat[:,1],yhat[:,2])
    ugt =to_u(y[:,0],  y[:,1],  y[:,2])
    l_dir=(1.0-(uhat*ugt).sum(-1).clamp(-0.999999,0.999999)).mean()
    l_el=(yhat[:,2]-y[:,2]).abs().mean()
    l_dist=(torch.log1p(yhat[:,3])-torch.log1p(y[:,3])).abs().mean()
    l_sp=(yhat[:,4]-y[:,4]).abs().mean()
    l_w=(yhat[:,5]-y[:,5]).abs().mean()
    l_g=(yhat[:,6]-y[:,6]).abs().mean()
    l_room=(yhat[:,7]-y[:,7]).abs().mean()
    return w_dir*l_dir + w_other*(l_el + l_dist + l_sp + l_w + l_g + l_room)

def loss_aux(aux_logits, aux_targets, w_base=0.40, w_az12=2.0):
    q,e,d,s,a12 = aux_targets
    lq=F.cross_entropy(aux_logits["quad"], q, label_smoothing=0.05)
    le=F.cross_entropy(aux_logits["elev"], e, label_smoothing=0.05)
    ld=F.cross_entropy(aux_logits["dist"], d, label_smoothing=0.05)
    ls=F.cross_entropy(aux_logits["sprd"], s, label_smoothing=0.05)
    la=F.cross_entropy(aux_logits["az12"], a12, label_smoothing=0.05)
    return w_base*(lq+le+ld+ls) + w_az12*la

# ----------------- main -----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--save", default="t2sa_minimal_v7.pt")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=2e-4)
    ap.add_argument("--lr_enc", type=float, default=5e-6)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--weighted_sampler", action="store_true")
    args=ap.parse_args()

    # data
    ds=DS(args.data, args.room_mode)
    n=int(len(ds)*0.9)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n, len(ds)-n], generator=torch.Generator().manual_seed(1))

    if args.weighted_sampler:
        quad_counts=[0,0,0,0]
        for idx in train_ds.indices: quad_counts[ ds.rows[idx]["aux"][0] ] += 1
        inv=[1.0/max(1,c) for c in quad_counts]
        weights=[inv[ds.rows[idx]["aux"][0]] for idx in train_ds.indices]
        sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        tr_loader=DataLoader(train_ds, batch_size=args.bsz, sampler=sampler, drop_last=True, collate_fn=collate_batch)
    else:
        tr_loader=DataLoader(train_ds, batch_size=args.bsz, shuffle=True, drop_last=True, collate_fn=collate_batch)
    val_loader=DataLoader(val_ds, batch_size=args.bsz, collate_fn=collate_batch)

    dev="cuda:0" if torch.cuda.is_available() else "cpu"
    model=Model(room_mode=args.room_mode).to(dev)
    # 전층 즉시 언프리즈 (소LR)
    for p in model.enc.parameters(): p.requires_grad=True

    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

    # optimizer (한 번만 생성: 모멘텀/스케줄 유지)
    def group_params(m):
        enc_params=[]; head_params=[]; aux_params=[]
        for n,p in (m.module if isinstance(m, nn.DataParallel) else m).named_parameters():
            if not p.requires_grad: continue
            if   n.startswith("enc."):  enc_params.append(p)
            elif n.startswith("head."): head_params.append(p)
            elif n.startswith("aux."):  aux_params.append(p)
        return enc_params, head_params, aux_params

    enc_params, head_params, aux_params = group_params(model)
    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": args.lr_head},
         {"params": aux_params,  "lr": args.lr_head},
         {"params": enc_params,  "lr": args.lr_enc}],
        weight_decay=0.01
    )
    total_steps = args.epochs * len(tr_loader) // max(1,args.accum)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(100,total_steps//20), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    def ramp(ep, E=6):
        t=min(1.0, ep/float(E))
        return t

    best=1e9
    for ep in range(1, args.epochs+1):
        model.train(); running=0.0
        t=ramp(ep, E=6)
        w_dir  = 10.0 + (25.0-10.0)*t
        w_other= 0.0 + (1.0-0.0)*t
        w_base = 0.0 + (0.40-0.0)*t
        w_az12 = 0.5 + (2.0-0.5)*t

        for step, batch in enumerate(tr_loader):
            toks=batch["tokens"]; y=batch["y"].to(dev)
            aux_t = tuple(t.to(dev) for t in batch["aux"])
            ids = toks["input_ids"]; attn = toks["attention_mask"]; tt = toks.get("token_type_ids", None)
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    yhat, h, aux_logits = model(ids, attn, tt)
                    Lr = loss_reg(yhat, y, w_dir=w_dir, w_other=w_other)
                    La = loss_aux(aux_logits, aux_t, w_base=w_base, w_az12=w_az12)
                    loss = (Lr + La) / args.accum
                scaler.scale(loss).backward()
                if (step+1)%args.accum==0:
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            else:
                yhat, h, aux_logits = model(ids, attn, tt)
                Lr = loss_reg(yhat, y, w_dir=w_dir, w_other=w_other)
                La = loss_aux(aux_logits, aux_t, w_base=w_base, w_az12=w_az12)
                loss = (Lr + La) / args.accum
                loss.backward()
                if (step+1)%args.accum==0:
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step(); opt.zero_grad(); sched.step()
            running += loss.item()*args.accum

        # ---- val ----
        model.eval(); vloss=0.0; Ang=[]; dMAE=[]; spMAE=[]; wMAE=[]; gMAE=[]; rMAE=[]
        with torch.no_grad():
            for batch in val_loader:
                toks=batch["tokens"]; y=batch["y"].to(dev)
                aux_t = tuple(t.to(dev) for t in batch["aux"])
                ids = toks["input_ids"]; attn = toks["attention_mask"]; tt = toks.get("token_type_ids", None)
                yhat, h, aux_logits = model(ids, attn, tt)
                vloss += (loss_reg(yhat,y, w_dir=w_dir, w_other=w_other) + loss_aux(aux_logits, aux_t, w_base=w_base, w_az12=w_az12)).item()
                Ang.append(angle_err_deg(yhat,y).cpu())
                dMAE.append((torch.log1p(yhat[:,3]) - torch.log1p(y[:,3])).abs().cpu())
                spMAE.append((yhat[:,4]-y[:,4]).abs().cpu())
                wMAE.append((yhat[:,5]-y[:,5]).abs().cpu())
                gMAE.append((yhat[:,6]-y[:,6]).abs().cpu())
                rMAE.append((yhat[:,7]-y[:,7]).abs().cpu())
        def catmean(x): return torch.cat(x).mean().item() if len(x)>0 else float("nan")
        tr = running/len(tr_loader); va = vloss/len(val_loader)
        print(f"epoch {ep:02d}: train {tr:.4f}  val {va:.4f}  "
              f"AE {catmean(Ang):.2f} | dlog {catmean(dMAE):.3f} | sp {catmean(spMAE):.2f} "
              f"| wet {catmean(wMAE):.3f} | gain {catmean(gMAE):.2f} | room {catmean(rMAE):.2f}")
        if va<best:
            best=va
            torch.save((model.module if isinstance(model, nn.DataParallel) else model).state_dict(), args.save)
            print("  saved:", args.save)

    # ---- save train embeddings (for eval KNN fallback) ----
    m = model.module if isinstance(model, nn.DataParallel) else model
    embs=[]
    with torch.no_grad():
        for i in range(0, len(train_ds), 64):
            texts=[ ds.rows[train_ds.indices[j]]["text"] for j in range(i, min(len(train_ds), i+64)) ]
            texts=_prefix_texts(texts)
            be = TOK(texts, padding=True, truncation=True, return_tensors="pt")
            ids, attn = be["input_ids"], be["attention_mask"]
            tt = be.get("token_type_ids", None)
            _, h, _ = m(ids.to(dev), attn.to(dev), tt.to(dev) if tt is not None else None)
            embs.append(h.cpu())
    if embs:
        embs=torch.cat(embs, dim=0)
        outp=os.path.splitext(args.save)[0]+"_train_embs.pt"
        torch.save({"embeddings":embs}, outp)
        print("saved train embeddings:", outp)

if __name__=="__main__":
    main()
