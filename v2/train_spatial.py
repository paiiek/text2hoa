# train_spatial.py
import argparse, json, math, os, random, sys, traceback
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16}

# ----- Optional: LoRA -----
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

TARGET_KEYS = [
    "az_sin_start","az_cos_start","az_sin_end","az_cos_end",
    "el_sin_start","el_cos_start","el_sin_end","el_cos_end",
    "dist_z","width_start","width_end","wet_norm","gain_norm"
]

# ---------------- Utilities ----------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ceil_to_mul(x, mul):
    if mul <= 1: return x
    return (x + mul - 1) // mul * mul

def re_has_hangul(s: str) -> bool:
    import re
    return re.search(r"[가-힣]", s) is not None

# ---------------- Text Augmentation ----------------
# 방향성/공간감 키워드 확장 (KO/EN)
KO_SYNS = {
    "와이드": ["넓게","퍼지게","광폭으로","폭넓게"],
    "좁게": ["타이트하게","좁혀서","좁은 느낌으로","조밀하게"],
    "가까이": ["가깝게","붙어서","근접하게","바짝"],
    "멀리": ["멀찍이","먼거리로","떨어져서","멀게"],
    "리버브": ["잔향","울림","홀감","공명"],
    "선명": ["또렷","명확","깨끗","청명"],
    "왼쪽": ["좌측","왼편","왼쪽으로","좌로"],
    "오른쪽": ["우측","오른편","오른쪽으로","우로"],
    "정면": ["앞쪽","앞으로","전방","정면으로"],
    "후면": ["뒤쪽","뒤로","후방"],
    "대각선": ["사선","비스듬히","대각으로"],
    "시계방향": ["시계 방향","오른쪽 회전"],
    "반시계방향": ["반시계 방향","왼쪽 회전"]
}
EN_SYNS = {
    "wide": ["broad","spread-out","wide-open","widened"],
    "narrow": ["tight","focused","thin","pinched"],
    "near": ["close","nearby","up-close","adjacent"],
    "far": ["distant","farther","far-away","remote"],
    "reverb": ["ambience","roomy","echo","resonance"],
    "clear": ["crisp","clean","distinct","pristine"],
    "left": ["to the left","left side","leftward","pan left"],
    "right": ["to the right","right side","rightward","pan right"],
    "front": ["in front","frontward","forward","towards front"],
    "rear": ["back","backward","rear side","towards back"],
    "diagonal": ["on a diagonal","slanted","oblique"],
    "clockwise": ["cw","rotate clockwise","turn clockwise"],
    "counterclockwise": ["ccw","counter-clockwise","turn counterclockwise"]
}
KO_ADVS = ["아주","매우","상당히","꽤","살짝","조금","강하게","약간","더","덜"]
EN_ADVS = ["very","highly","quite","fairly","slightly","a bit","strongly","slightly more","less"]

def maybe_augment(text: str, y: Dict[str,float], prob: float) -> Tuple[str, Dict[str,float]]:
    if random.random() > prob:
        return text, y
    t = text
    # 동의어 교체 + 강조어 전치 삽입
    if re_has_hangul(t):
        for k, cands in KO_SYNS.items():
            if k in t and random.random() < 0.6:
                t = t.replace(k, random.choice(cands))
        if random.random() < 0.5:
            t = f"{random.choice(KO_ADVS)} " + t
    else:
        low = t.lower()
        for k, cands in EN_SYNS.items():
            if k in low and random.random() < 0.6:
                # 간단 치환 (소문자 기준)
                t = t.replace(k, random.choice(cands))
        if random.random() < 0.5:
            t = f"{random.choice(EN_ADVS)} " + t

    # 라벨 동조: wet/gain/width_end에 작은 jitter
    y = dict(y)
    def jitter(val, lo=0.0, hi=1.0, amp=0.12):
        d = (random.random()*2-1)*amp
        return max(lo, min(hi, val + d))
    y["wet_norm"]   = jitter(y["wet_norm"], 0.0, 1.0, amp=0.12 if random.random()<0.5 else 0.07)
    y["gain_norm"]  = jitter(y["gain_norm"], 0.0, 1.0, amp=0.12 if random.random()<0.5 else 0.07)
    y["width_end"]  = jitter(y["width_end"], 0.0, 1.0, amp=0.15 if random.random()<0.5 else 0.08)
    return t, y

def rotate_sin_cos(sinv, cosv, deg):
    rad = math.radians(deg)
    s, c = math.sin(rad), math.cos(rad)
    # [cos θ -sin θ; sin θ cos θ] · [cos; sin] → (cos', sin')
    cosn = c * cosv - s * sinv
    sinn = s * cosv + c * sinv
    return sinn, cosn

def maybe_small_az_rotate(text: str, y: Dict[str,float], p: float) -> Tuple[str, Dict[str,float]]:
    """±[5,15]° 소각 회전. start/end 동일 회전 + 텍스트 힌트 주석."""
    if random.random() > p:
        return text, y
    delta = random.choice([1,-1]) * random.uniform(5.0, 15.0)
    y = dict(y)
    # start
    y["az_sin_start"], y["az_cos_start"] = rotate_sin_cos(y["az_sin_start"], y["az_cos_start"], delta)
    # end
    y["az_sin_end"],   y["az_cos_end"]   = rotate_sin_cos(y["az_sin_end"],   y["az_cos_end"],   delta)
    # 텍스트 힌트(아주 약하게)
    hint_ko = "약간 왼쪽" if delta<0 else "약간 오른쪽"
    hint_en = "slightly left" if delta<0 else "slightly right"
    if re_has_hangul(text):
        text = f"{hint_ko} | {text}"
    else:
        text = f"{hint_en} | {text}"
    return text, y

# ---------------- Dataset / Collate ----------------
class T2S(Dataset):
    def __init__(self, path, tok_name, max_len=96, augment_prob=0.0, az_small_rotate_prob=0.15):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = AutoTokenizer.from_pretrained(tok_name)
        self.max_len = max_len
        self.augment_prob = augment_prob
        self.az_small_rotate_prob = az_small_rotate_prob
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        ex = self.rows[i]
        text = ex["text"]
        y = {k: float(ex["y"][k]) for k in TARGET_KEYS}
        if self.augment_prob > 0.0:
            text, y = maybe_augment(text, y, self.augment_prob)
            text, y = maybe_small_az_rotate(text, y, p=self.az_small_rotate_prob)
        enc = self.tok(text, truncation=True, max_length=self.max_len)
        yvec = [y[k] for k in TARGET_KEYS]
        m = [float(ex["mask"][k]) for k in TARGET_KEYS]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "y": yvec, "mask": m}

class Collate:
    def __init__(self, pad_id, pad_to_mul=1):
        self.pad_id = pad_id; self.pad_to_mul = pad_to_mul
    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        L = ceil_to_mul(L, self.pad_to_mul)
        ids, attn = [], []
        for b in batch:
            pad = L - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id]*pad)
            attn.append(b["attention_mask"] + [0]*pad)
        y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
        m = torch.tensor([b["mask"] for b in batch], dtype=torch.float32)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn), "y": y, "mask": m}

# ---------------- Model / Head ----------------
class Head(torch.nn.Module):
    def __init__(self, hdim, dist_sigmoid=False, az_bins=24):
        super().__init__()
        self.fc1 = torch.nn.Linear(hdim, 256)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 13)           # regression head
        self.az_bins = az_bins
        self.fc_az_cls = torch.nn.Linear(256, 2*az_bins)  # az_start/az_end classification (aux)
        self.dist_sigmoid = dist_sigmoid
    def forward(self, x):
        h = self.act(self.fc1(x))
        o = self.fc2(h)
        # angles: 4×(sin,cos) → unit-norm
        sc = o[:, 0:8].view(-1, 4, 2)
        sc = sc / (sc.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        s  = sc.reshape(-1, 8)
        # dist
        dist = o[:, 8:9]
        if self.dist_sigmoid:
            dist = torch.sigmoid(dist)
        # others
        width = torch.sigmoid(o[:, 9:11])
        wet   = torch.sigmoid(o[:, 11:12])
        gain  = torch.sigmoid(o[:, 12:13])
        reg_out = torch.cat([s, dist, width, wet, gain], dim=-1)
        # aux logits
        az_logits = self.fc_az_cls(h)  # (B, 2*az_bins)
        return reg_out, az_logits

class Model(torch.nn.Module):
    def __init__(self, enc_name, grad_ckpt=True, lora_r=0, lora_alpha=16, lora_dropout=0.05,
                 param_dtype="fp32", dist_sigmoid=False, az_bins=24):
        super().__init__()
        dtype = DTYPE_MAP.get(param_dtype, torch.float32)
        # NOTE: new HF uses `dtype` (torch_dtype deprecated)
        self.enc = AutoModel.from_pretrained(enc_name, dtype=dtype, low_cpu_mem_usage=True)
        if lora_r and HAS_PEFT:
            cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_r, lora_alpha=lora_alpha,
                             lora_dropout=lora_dropout,
                             target_modules=["query","key","value","q_proj","k_proj","v_proj"])
            self.enc = get_peft_model(self.enc, cfg)
        self.head = Head(self.enc.config.hidden_size, dist_sigmoid=dist_sigmoid, az_bins=az_bins).to(dtype)
        self.grad_ckpt_enabled = False
        self._want_grad_ckpt = grad_ckpt
    def set_grad_ckpt(self, flag: bool):
        if hasattr(self.enc, "gradient_checkpointing_enable"):
            if flag and not self.grad_ckpt_enabled:
                self.enc.gradient_checkpointing_enable(); self.grad_ckpt_enabled = True
            elif not flag and self.grad_ckpt_enabled:
                self.enc.gradient_checkpointing_disable(); self.grad_ckpt_enabled = False
    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)  # (reg_out, az_logits)

# ---------------- Loss ----------------
def _smooth_one_hot(ncls: int, target_idx: torch.Tensor, eps: float):
    # target_idx: (B,) long
    B = target_idx.size(0)
    out = torch.full((B, ncls), eps / (ncls - 1), device=target_idx.device, dtype=torch.float32)
    out.scatter_(1, target_idx.view(-1,1), 1.0 - eps)
    return out  # (B,ncls)

def _ce_or_focal_with_smoothing(logits: torch.Tensor, target_idx: torch.Tensor,
                                smooth_eps: float=0.1, focal_gamma: float=0.0):
    # logits: (B,C)
    if smooth_eps > 0:
        soft = _smooth_one_hot(logits.size(1), target_idx, smooth_eps)
        logp = torch.log_softmax(logits, dim=-1)
        if focal_gamma > 0:
            p = torch.softmax(logits, dim=-1)
            mod = (1 - p).clamp(min=1e-6).pow(focal_gamma)
            loss = -(soft * mod * logp).sum(dim=-1)
        else:
            loss = -(soft * logp).sum(dim=-1)
        return loss.mean()
    else:
        if focal_gamma > 0:
            # focal CE (hard labels)
            logp = torch.log_softmax(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            pt = torch.gather(p, 1, target_idx.view(-1,1)).squeeze(1).clamp(min=1e-6)
            mod = (1 - pt).pow(focal_gamma)
            nll = torch.nn.functional.nll_loss(logp, target_idx, reduction='none')
            return (mod * nll).mean()
        else:
            return torch.nn.functional.cross_entropy(logits, target_idx)

def masked_loss_cosonly_with_cls(pred, cls_logits, target, mask,
                                 loss_type="huber", huber_delta=1.0,
                                 angle_lambda=1.3, cls_lambda=0.3, nbin=36,
                                 az_smooth=0.1, cls_focal_gamma=0.0):
    """
    pred: (B,13) with first 8 = 4×(sin,cos) unit-norm
    cls_logits: (B, 2*nbin) for az_start/az_end bins
    """
    # (1) angles cosine-only
    ps = pred[:, 0:8].view(-1,4,2)
    ts = target[:, 0:8].view(-1,4,2)
    am = mask[:, 0:8].view(-1,4,2)[:, :, 0]  # (B,4)
    dot = (ps * ts).sum(dim=-1).clamp(-1+1e-6, 1-1e-6)  # (B,4)
    angle_loss = (1.0 - dot) * am
    angle_loss = angle_loss.sum() / am.sum().clamp(min=1)

    # (2) rest regression (8:)
    rest_pred, rest_true, rest_mask = pred[:, 8:], target[:, 8:], mask[:, 8:]
    if loss_type == "huber":
        abs_err = (rest_pred - rest_true).abs()
        rest = torch.where(abs_err <= huber_delta, 0.5*abs_err**2,
                           huber_delta*(abs_err - 0.5*huber_delta))
    else:
        rest = (rest_pred - rest_true)**2
    rest_loss = rest.sum() / rest_mask.sum().clamp(min=1)

    # (3) aux classification with smoothing/focal
    st_logits, en_logits = cls_logits[:, :nbin], cls_logits[:, nbin:nbin*2]
    sin_st = target[:, 0]; cos_st = target[:, 1]
    sin_en = target[:, 2]; cos_en = target[:, 3]
    deg_st = torch.rad2deg(torch.atan2(sin_st, cos_st)) % 360
    deg_en = torch.rad2deg(torch.atan2(sin_en, cos_en)) % 360
    bin_st = torch.div(deg_st, 360/nbin, rounding_mode='floor').long().clamp(0, nbin-1)
    bin_en = torch.div(deg_en, 360/nbin, rounding_mode='floor').long().clamp(0, nbin-1)

    ce_st = _ce_or_focal_with_smoothing(st_logits, bin_st, smooth_eps=az_smooth, focal_gamma=cls_focal_gamma)
    ce_en = _ce_or_focal_with_smoothing(en_logits, bin_en, smooth_eps=az_smooth, focal_gamma=cls_focal_gamma)
    cls_loss = 0.5*(ce_st + ce_en)

    return angle_loss*angle_lambda + rest_loss + cls_loss*cls_lambda

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--stats", required=True)  # kept for compatibility
    ap.add_argument("--encoder", default="xlm-roberta-base")

    # Training schedule / optimization
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--accum_steps", type=int, default=3)
    ap.add_argument("--lr_enc", type=float, default=5e-6)
    ap.add_argument("--lr_head", type=float, default=2e-3)     # 가속
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--freeze_epochs", type=int, default=0)

    # Augmentation
    ap.add_argument("--augment_prob", type=float, default=0.35)
    ap.add_argument("--az_small_rotate_prob", type=float, default=0.15)

    # Loss & heads
    ap.add_argument("--loss", choices=["mse","huber"], default="huber")
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--angle_lambda", type=float, default=1.3)
    ap.add_argument("--az_bins", type=int, default=36)         # 10° 간격
    ap.add_argument("--cls_lambda", type=float, default=0.3)
    ap.add_argument("--az_smooth", type=float, default=0.1, help="label smoothing for az bin CE")
    ap.add_argument("--cls_focal_gamma", type=float, default=1.5, help=">0이면 focal CE 사용")

    # Memory / infra
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--pad_to_mul", type=int, default=8)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--param_dtype", choices=["fp32","fp16"], default="fp32")
    ap.add_argument("--dist_sigmoid", type=int, default=0, help="1 if dist_z label in [0,1]")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # I/O
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ----- DDP -----
    use_ddp = torch.distributed.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rank = torch.distributed.get_rank() if use_ddp else 0
    is_main = (rank == 0)
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed + (rank if use_ddp else 0))

    # ----- Data -----
    tok_name = args.encoder
    train_ds = T2S(args.train, tok_name, max_len=args.max_len,
                   augment_prob=args.augment_prob, az_small_rotate_prob=args.az_small_rotate_prob)
    valid_ds = T2S(args.valid, tok_name, max_len=args.max_len, augment_prob=0.0, az_small_rotate_prob=0.0)
    collate = Collate(pad_id=train_ds.tok.pad_token_id, pad_to_mul=args.pad_to_mul)

    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None; valid_sampler = None

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                          sampler=train_sampler, collate_fn=collate, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=max(8,args.batch_size), shuffle=False,
                          sampler=valid_sampler, collate_fn=collate, pin_memory=True)

    # ----- Model -----
    model = Model(
        args.encoder,
        grad_ckpt=bool(args.grad_ckpt),
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        param_dtype=args.param_dtype,
        dist_sigmoid=bool(args.dist_sigmoid),
        az_bins=args.az_bins
    ).to(device)

    # freeze encoder BEFORE DDP (if freeze_epochs > 0)
    if args.freeze_epochs > 0:
        for p in model.enc.parameters(): p.requires_grad = False

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    # ----- Optim / Sched -----
    enc = (model.module.enc if use_ddp else model.enc)
    head = (model.module.head if use_ddp else model.head)
    opt = torch.optim.AdamW([
        {"params": [p for p in enc.parameters()], "lr": args.lr_enc, "weight_decay": args.weight_decay},
        {"params": list(head.parameters()), "lr": args.lr_head, "weight_decay": args.weight_decay},
    ])
    total_steps = args.epochs * max(1, len(train_dl)) // max(1, args.accum_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    # AMP/Scaler: 파라미터가 fp32일 때만 사용 (fp16 파라미터와 충돌 방지)
    amp_enabled = (args.param_dtype == "fp32")
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    if is_main:
        n_all = sum(p.numel() for p in (model.module if use_ddp else model).parameters())
        n_train = sum(p.numel() for p in (model.module if use_ddp else model).parameters() if p.requires_grad)
        print(f"[rank{rank}] params total={n_all/1e6:.2f}M trainable={n_train/1e6:.2f}M LoRA={args.lora_r}", flush=True)
        print(f"[rank{rank}] train={len(train_ds)} valid={len(valid_ds)} bs={args.batch_size} accum={args.accum_steps} max_len={args.max_len}", flush=True)

    best_val = float("inf"); step = 0
    try:
        for epoch in range(1, args.epochs+1):
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # unfreeze after freeze_epochs
            if epoch > args.freeze_epochs:
                mm = model.module if use_ddp else model
                for p in mm.enc.parameters(): p.requires_grad = True
                mm.set_grad_ckpt(bool(args.grad_ckpt))

            model.train()
            running = 0.0; count = 0
            opt.zero_grad(set_to_none=True)

            for b in train_dl:
                for k in ("input_ids","attention_mask","y","mask"):
                    b[k]=b[k].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    pred, az_cls = model(b["input_ids"], b["attention_mask"])
                    loss = masked_loss_cosonly_with_cls(
                        pred, az_cls, b["y"], b["mask"],
                        loss_type=args.loss, huber_delta=args.huber_delta,
                        angle_lambda=args.angle_lambda, cls_lambda=args.cls_lambda,
                        nbin=args.az_bins, az_smooth=args.az_smooth, cls_focal_gamma=args.cls_focal_gamma
                    )

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    raise RuntimeError("NaN/Inf loss")

                if amp_enabled:
                    scaler.scale(loss/args.accum_steps).backward()
                else:
                    (loss/args.accum_steps).backward()

                if (step+1) % args.accum_steps == 0:
                    if amp_enabled:
                        scaler.step(opt); scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True); sched.step()
                running += loss.item()*b["y"].size(0); count += b["y"].size(0); step+=1

            tr_loss = running / max(1,count)
            if use_ddp:
                tl = torch.tensor([tr_loss*count, count], dtype=torch.float64, device=device)
                torch.distributed.all_reduce(tl, op=torch.distributed.ReduceOp.SUM)
                tr_loss = (tl[0] / torch.clamp(tl[1], min=1)).item()

            # eval
            model.eval(); v_running=0; v_count=0
            with torch.no_grad():
                for b in valid_dl:
                    for k in ("input_ids","attention_mask","y","mask"):
                        b[k]=b[k].to(device, non_blocking=True)
                    pred, az_cls = model(b["input_ids"], b["attention_mask"])
                    vloss = masked_loss_cosonly_with_cls(
                        pred, az_cls, b["y"], b["mask"],
                        loss_type=args.loss, huber_delta=args.huber_delta,
                        angle_lambda=args.angle_lambda, cls_lambda=args.cls_lambda,
                        nbin=args.az_bins, az_smooth=args.az_smooth, cls_focal_gamma=args.cls_focal_gamma
                    )
                    v_running += vloss.item()*b["y"].size(0); v_count+=b["y"].size(0)
            val_loss = v_running / max(1,v_count)
            if use_ddp:
                vl = torch.tensor([val_loss*v_count, v_count], dtype=torch.float64, device=device)
                torch.distributed.all_reduce(vl, op=torch.distributed.ReduceOp.SUM)
                val_loss = (vl[0] / torch.clamp(vl[1], min=1)).item()

            if is_main:
                print(f"[epoch {epoch}] train {tr_loss:.4f}  valid {val_loss:.4f}", flush=True)

            if is_main and val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "state_dict": (model.module.state_dict() if use_ddp else model.state_dict()),
                    "encoder": args.encoder,
                    "target_keys": TARGET_KEYS,
                    "best_val": best_val,
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
                print(f"[epoch {epoch}] saved best.pt (val {best_val:.4f})", flush=True)

        if use_ddp:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

    except Exception as e:
        print(f"\n[rank{rank}] !!! EXCEPTION !!! {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        if use_ddp:
            try: torch.distributed.destroy_process_group()
            except Exception: pass
        raise

if __name__ == "__main__":
    main()
