#!/usr/bin/env python3
import json, math, re, random, csv, argparse, os
from pathlib import Path

LEFT   = r"(왼|좌|left)"
RIGHT  = r"(오른|우|right)"
UP     = r"(위|상|over|above|up(?!date))"
DOWN   = r"(아래|하|below|down)"
NEAR   = r"(가까|근접|near|close(r)?\b)"
FAR    = r"(멀|원거리|far|distant)"
WIDE   = r"(와이드|넓|퍼지|wide|spread(ed)?)"
NARROW = r"(좁|타이트|narrow|tight)"
REV_MORE = r"(리버브|잔향|울림|reverb|wet|hall|cathedral)"
REV_LESS = r"(드라이|직접음|dry|dead|studio\s?dry)"
LOUD   = r"(크게|볼륨\s?업|점점\s?커져|loud|amplif(y|ied)|increase|fade\s?in)"
QUIET  = r"(작게|볼륨\s?다운|점점\s?작아져|quiet|attenuat(ed|e)|decrease|fade\s?out)"

def has(pat, text): 
    return re.search(pat, text, flags=re.IGNORECASE) is not None

def clamp(x, lo, hi): 
    return lo if x<lo else hi if x>hi else x

def sincos(x):
    import math
    return math.sin(x), math.cos(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delta_el", type=float, default=0.35)
    ap.add_argument("--delta_width", type=float, default=0.30)
    args = ap.parse_args()
    random.seed(args.seed)

    SRC = args.src
    OUT_PREFIX = args.out_prefix
    TRAIN = OUT_PREFIX + "_train.jsonl"
    VALID = OUT_PREFIX + "_valid.jsonl"
    TEST  = OUT_PREFIX + "_test.jsonl"
    STATS = OUT_PREFIX + "_stats.json"
    QC    = OUT_PREFIX + "_qc_report.csv"

    # load
    rows = []
    with open(SRC, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    assert len(rows)>0, "empty input"

    # gather stats
    import math, statistics
    dist_list, spread_list, gain_db_list = [], [], []
    for ex in rows:
        params = ex.get("params") or ex.get("spatial") or ex
        dist_list.append(float(params.get("dist_m",1.0)))
        spread_list.append(float(params.get("spread_deg",30.0)))
        gain_db_list.append(float(params.get("gain_db",-3.0)))
    import numpy as np
    LOGD_MEAN = float(np.mean([math.log(max(d,1e-6)) for d in dist_list]))
    LOGD_STD = float(np.std([math.log(max(d,1e-6)) for d in dist_list], ddof=0) or 1.0)
    GAIN_LIN_MIN = float(min(10**(g/20.0) for g in gain_db_list))
    GAIN_LIN_MAX = float(max(10**(g/20.0) for g in gain_db_list))
    DIST_MED = float(statistics.median(dist_list))
    SPREAD_MED = float(statistics.median(spread_list))

    converted, qc_rows = [], []
    for i, ex in enumerate(rows):
        text = ex.get("text","")
        lang = ex.get("lang") or ("ko" if re.search(r"[가-힣]", text) else "en")
        params = ex.get("params") or ex.get("spatial") or ex

        az_sc = params.get("az_sc", 0.0)
        el_rad = float(params.get("el_rad", 0.0))
        dist_m = float(params.get("dist_m", 1.0))
        spread_deg = float(params.get("spread_deg", 30.0))
        wet_mix = float(params.get("wet_mix", 0.2))
        gain_db = float(params.get("gain_db", -3.0))

        # az rad
        if isinstance(az_sc, list) and len(az_sc)>=2:
            az_start = float(az_sc[0]) * math.pi
            az_end   = float(az_sc[1]) * math.pi
        else:
            az_val = float(az_sc) * math.pi
            az_start = az_end = az_val

        # el start/end
        if isinstance(params.get("el_sc"), list) and len(params["el_sc"])>=2:
            el_start = float(params["el_sc"][0]); el_end = float(params["el_sc"][-1])
        else:
            el_start = float(el_rad)
            if has(UP, text) and not has(DOWN, text):
                el_end = clamp(el_start + args.delta_el, -math.pi/2, math.pi/2)
            elif has(DOWN, text) and not has(UP, text):
                el_end = clamp(el_start - args.delta_el, -math.pi/2, math.pi/2)
            else:
                el_end = el_start

        width_start = clamp(spread_deg/120.0, 0.0, 1.0)
        if has(WIDE, text) and not has(NARROW, text):
            width_end = clamp(width_start + args.delta_width, 0.0, 1.0)
        elif has(NARROW, text) and not has(WIDE, text):
            width_end = clamp(width_start - args.delta_width, 0.0, 1.0)
        else:
            width_end = width_start

        dist_z = (math.log(max(dist_m,1e-6)) - LOGD_MEAN) / (LOGD_STD if LOGD_STD>1e-6 else 1.0)
        wet_norm = max(0.0, min(1.0, wet_mix))

        gain_lin = 10**(gain_db/20.0)
        gain_norm = (gain_lin - GAIN_LIN_MIN) / (GAIN_LIN_MAX - GAIN_LIN_MIN + 1e-8)

        az_sin_start, az_cos_start = sincos(az_start)
        az_sin_end,   az_cos_end   = sincos(az_end)
        el_sin_start, el_cos_start = sincos(el_start)
        el_sin_end,   el_cos_end   = sincos(el_end)

        mask = {
            "az_sin_start": 1.0 if (has(LEFT,text) or has(RIGHT,text)) else 0.5,
            "az_cos_start": 1.0 if (has(LEFT,text) or has(RIGHT,text)) else 0.5,
            "az_sin_end":   1.0 if (has(LEFT,text) or has(RIGHT,text)) else 0.5,
            "az_cos_end":   1.0 if (has(LEFT,text) or has(RIGHT,text)) else 0.5,
            "el_sin_start": 1.0 if (has(UP,text) or has(DOWN,text)) else 0.5,
            "el_cos_start": 1.0 if (has(UP,text) or has(DOWN,text)) else 0.5,
            "el_sin_end":   1.0 if (has(UP,text) or has(DOWN,text)) else 0.5,
            "el_cos_end":   1.0 if (has(UP,text) or has(DOWN,text)) else 0.5,
            "dist_z":       1.0 if (has(NEAR,text) or has(FAR,text)) else 0.5,
            "width_start":  1.0 if (has(WIDE,text) or has(NARROW,text)) else 0.5,
            "width_end":    1.0 if (has(WIDE,text) or has(NARROW,text)) else 0.5,
            "wet_norm":     1.0 if (has(REV_MORE,text) or has(REV_LESS,text)) else 0.5,
            "gain_norm":    1.0 if (has(LOUD,text) or has(QUIET,text)) else 0.5,
        }

        y = {
            "az_sin_start": az_sin_start, "az_cos_start": az_cos_start,
            "az_sin_end": az_sin_end,     "az_cos_end": az_cos_end,
            "el_sin_start": el_sin_start, "el_cos_start": el_cos_start,
            "el_sin_end": el_sin_end,     "el_cos_end": el_cos_end,
            "dist_z": dist_z,
            "width_start": width_start, "width_end": width_end,
            "wet_norm": wet_norm, "gain_norm": gain_norm
        }
        converted.append({"text": text, "lang": lang, "y": y, "mask": mask, "meta": {"src_idx": i}})

        warn = []
        # quick QC relative to original fields
        az_s = (az_sc[0] if isinstance(az_sc,list) else az_sc)
        if has(RIGHT, text) and not has(LEFT,text) and not (az_s>0): warn.append("az_start sign!=right")
        if has(LEFT, text) and not has(RIGHT,text) and not (az_s<0): warn.append("az_start sign!=left")
        if has(UP,text) and not has(DOWN,text) and el_rad <= 0: warn.append("el_start sign!=up")
        if has(DOWN,text) and not has(UP,text) and el_rad >= 0: warn.append("el_start sign!=down")
        if has(NEAR,text) and dist_m >= DIST_MED: warn.append("near but dist>=median")
        if has(FAR,text) and dist_m <= DIST_MED: warn.append("far but dist<=median")
        if has(WIDE,text) and spread_deg <= SPREAD_MED: warn.append("wide but spread<=median")
        if has(NARROW,text) and spread_deg >= SPREAD_MED: warn.append("narrow but spread>=median")
        qc_rows.append({"idx": i, "lang": lang, "text_snippet": text[:120].replace("\n"," "), "warnings": "; ".join(warn)})

    # split
    import numpy as np
    idxs = list(range(len(converted)))
    rng = random.Random(args.seed)
    rng.shuffle(idxs)
    n = len(idxs); n_tr = int(0.8*n); n_va=int(0.1*n)
    train_idx = idxs[:n_tr]; valid_idx = idxs[n_tr:n_tr+n_va]; test_idx = idxs[n_tr+n_va:]

    def wjsonl(pth, subset):
        with open(pth, "w", encoding="utf-8") as f:
            for i in subset:
                f.write(json.dumps(converted[i], ensure_ascii=False)+"\n")

    wjsonl(TRAIN, train_idx)
    wjsonl(VALID, valid_idx)
    wjsonl(TEST,  test_idx)

    stats = {
        "source": SRC, "N": len(rows),
        "splits": {"train": len(train_idx), "valid": len(valid_idx), "test": len(test_idx)},
        "normalization": {
            "log_dist": {"mean": LOGD_MEAN, "std": LOGD_STD},
            "gain_lin": {"min": GAIN_LIN_MIN, "max": GAIN_LIN_MAX},
            "width_scale_deg": 120.0,
            "azimuth_scale": "az_sc * pi (rad)",
            "synth_delta": {"elevation_rad": args.delta_el, "width": args.delta_width}
        }
    }
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(QC, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx","lang","text_snippet","warnings"])
        writer.writeheader()
        writer.writerows(qc_rows)

    print("Wrote:", TRAIN, VALID, TEST, STATS, QC)

if __name__=="__main__":
    main()
