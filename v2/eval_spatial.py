# eval_spatial.py
import argparse, json, math, os, csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

TARGET_KEYS_DEFAULT = [
    "az_sin_start","az_cos_start","az_sin_end","az_cos_end",
    "el_sin_start","el_cos_start","el_sin_end","el_cos_end",
    "dist_z","width_start","width_end","wet_norm","gain_norm"
]

# Optional LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

def to_angle(sinv, cosv):
    return math.atan2(float(sinv), float(cosv))  # [-pi, pi]

def circ_err_deg(a, b):
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return abs(d) * 180.0 / math.pi

class EvalDS(Dataset):
    def __init__(self, path, tok_name, max_len=64):
        import json
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = AutoTokenizer.from_pretrained(tok_name)
        self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        ex = self.rows[i]
        enc = self.tok(ex["text"], truncation=True, max_length=self.max_len)
        y = [float(ex["y"][k]) for k in TARGET_KEYS_DEFAULT]
        return {"text": ex["text"], "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "y": y, "raw": ex}

class Collate:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids, attn = [], []
        for b in batch:
            pad = L - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id]*pad)
            attn.append(b["attention_mask"] + [0]*pad)
        y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
        return {"text":[b["text"] for b in batch], "raw":[b["raw"] for b in batch],
                "input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn), "y": y}

class Head(torch.nn.Module):
    def __init__(self, hdim, dist_sigmoid=False, az_bins=24):
        super().__init__()
        self.fc1 = torch.nn.Linear(hdim, 256)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 13)
        self.fc_az_cls = torch.nn.Linear(256, 2*az_bins)  # not used at eval
        self.dist_sigmoid = dist_sigmoid
    def forward(self, x):
        h = self.act(self.fc1(x))
        o = self.fc2(h)
        sc = o[:, 0:8].view(-1,4,2)
        sc = sc / (sc.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        s  = sc.reshape(-1,8)
        dist = o[:, 8:9]
        if self.dist_sigmoid: dist = torch.sigmoid(dist)
        width = torch.sigmoid(o[:, 9:11])
        wet   = torch.sigmoid(o[:, 11:12])
        gain  = torch.sigmoid(o[:, 12:13])
        reg_out = torch.cat([s, dist, width, wet, gain], dim=-1)
        logits = self.fc_az_cls(h)
        return reg_out, logits

class Model(torch.nn.Module):
    def __init__(self, enc_name, lora_r=0, lora_alpha=16, lora_dropout=0.05, dist_sigmoid=False, az_bins=24):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc_name)
        if lora_r and HAS_PEFT:
            cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_r, lora_alpha=lora_alpha,
                             lora_dropout=lora_dropout,
                             target_modules=["query","key","value","q_proj","k_proj","v_proj"])
            self.enc = get_peft_model(self.enc, cfg)
        self.head = Head(self.enc.config.hidden_size, dist_sigmoid=dist_sigmoid, az_bins=az_bins)
    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)

def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--report_json", default=None)
    ap.add_argument("--dump_unitnorm", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    enc_name = args.encoder or ckpt.get("encoder") or "xlm-roberta-base"
    target_keys = ckpt.get("target_keys", TARGET_KEYS_DEFAULT)
    assert len(target_keys) == 13, f"unexpected target_keys length {len(target_keys)}"

    cargs = ckpt.get("args", {})
    lora_r = int(cargs.get("lora_r", 0))
    lora_alpha = int(cargs.get("lora_alpha", 16))
    lora_dropout = float(cargs.get("lora_dropout", 0.05))
    dist_sigmoid = bool(int(cargs.get("dist_sigmoid", 0)))
    az_bins = int(cargs.get("az_bins", 24))
    if lora_r > 0 and not HAS_PEFT:
        print("[eval][warn] CKPT indicates LoRA but 'peft' not installed. Evaluating WITHOUT LoRA.")

    ds = EvalDS(args.data, enc_name, max_len=args.max_len)
    collate = Collate(pad_id=ds.tok.pad_token_id)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(enc_name, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                  dist_sigmoid=dist_sigmoid, az_bins=az_bins).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[eval] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    all_rows = []
    with torch.no_grad():
        for b in dl:
            for k in ("input_ids","attention_mask"): b[k] = b[k].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                pred, _ = model(b["input_ids"], b["attention_mask"])
                pred = pred.float().cpu()
            ys_true = []
            for ex in b["raw"]:
                ys_true.append([float(ex["y"][k]) for k in target_keys])
            ys_true = torch.tensor(ys_true, dtype=torch.float32)

            for i in range(pred.size(0)):
                row = {"text": b["text"][i]}
                for j,k in enumerate(target_keys):
                    row[f"y_true_{k}"] = float(ys_true[i,j])
                    row[f"y_pred_{k}"] = float(pred[i,j])
                all_rows.append(row)

    # CSV 저장
    out_path = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = ["text"] + sum([[f"y_true_{k}", f"y_pred_{k}"] for k in target_keys], [])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in all_rows: w.writerow(r)
    print(f"[eval] Wrote CSV: {out_path} (rows={len(all_rows)})")

    # Metrics
    def col(k): return np.array([r[k] for r in all_rows], dtype=float)

    az_st = [circ_err_deg(to_angle(r[f"y_true_az_sin_start"], r[f"y_true_az_cos_start"]),
                          to_angle(r[f"y_pred_az_sin_start"], r[f"y_pred_az_cos_start"])) for r in all_rows]
    az_en = [circ_err_deg(to_angle(r[f"y_true_az_sin_end"], r[f"y_true_az_cos_end"]),
                          to_angle(r[f"y_pred_az_sin_end"], r[f"y_pred_az_cos_end"])) for r in all_rows]
    el_st = [circ_err_deg(to_angle(r[f"y_true_el_sin_start"], r[f"y_true_el_cos_start"]),
                          to_angle(r[f"y_pred_el_sin_start"], r[f"y_pred_el_cos_start"])) for r in all_rows]
    el_en = [circ_err_deg(to_angle(r[f"y_true_el_sin_end"], r[f"y_true_el_cos_end"]),
                          to_angle(r[f"y_pred_el_sin_end"], r[f"y_pred_el_cos_end"])) for r in all_rows]

    print("== Angular MAE (deg) ==")
    import numpy as np
    print(f"Az start: {float(np.mean(az_st)):.2f}")
    print(f"Az end  : {float(np.mean(az_en)):.2f}")
    print(f"El start: {float(np.mean(el_st)):.2f}")
    print(f"El end  : {float(np.mean(el_en)):.2f}")

    print("\n== Other targets MAE ==")
    metrics = {}
    for k in ["dist_z","width_start","width_end","wet_norm","gain_norm"]:
        t = col(f"y_true_{k}"); p = col(f"y_pred_{k}")
        mae_k = float(np.mean(np.abs(t-p))); metrics[k] = mae_k
        print(f"{k:12s}: {mae_k:.4f}")

    metrics.update({
        "az_start_deg": float(np.mean(az_st)),
        "az_end_deg": float(np.mean(az_en)),
        "el_start_deg": float(np.mean(el_st)),
        "el_end_deg": float(np.mean(el_en)),
        "rows": len(all_rows),
        "encoder": enc_name,
        "lora_r": lora_r,
        "dist_sigmoid": dist_sigmoid,
        "az_bins": az_bins
    })

    # unit-norm 진단
    if args.dump_unitnorm:
        def unit_stats(name_s, name_c):
            s = col(name_s); c = col(name_c)
            u = s*s + c*c
            return float(u.mean()), float(u.var())
        for (ns, nc, lbl) in [
            ("y_pred_az_sin_start","y_pred_az_cos_start","pred az start"),
            ("y_pred_az_sin_end","y_pred_az_cos_end","pred az end"),
            ("y_pred_el_sin_start","y_pred_el_cos_start","pred el start"),
            ("y_pred_el_sin_end","y_pred_el_cos_end","pred el end"),
        ]:
            mu, var = unit_stats(ns, nc)
            print(f"[unit] {lbl}: mean={mu:.3f} var={var:.3f}")

    if args.report_json:
        rpath = os.path.abspath(args.report_json)
        os.makedirs(os.path.dirname(rpath) or ".", exist_ok=True)
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[eval] Wrote metrics JSON: {rpath}")

if __name__ == "__main__":
    main()
