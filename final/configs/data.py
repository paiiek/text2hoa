"""
텍스트→스페이셜 오디오 파라미터 학습을 위한 데이터 재정비/검증/분할 스크립트
- 입력: JSONL (예: /mnt/data/text2spatial_v3_rebalanced.jsonl)
  각 줄에 최소한 아래 필드가 존재한다고 가정:
    - text (str)
    - lang (ko|en)  # 없으면 텍스트로 추정
    - params 또는 최상위에: az_sc([-1,1] 또는 [start,end]), el_rad(rad), dist_m(m), spread_deg(°), wet_mix(0–1), gain_db(dB)
- 출력:
    /mnt/data/text2spatial_v4_train.jsonl
    /mnt/data/text2spatial_v4_valid.jsonl
    /mnt/data/text2spatial_v4_test.jsonl
    /mnt/data/text2spatial_v4_stats.json
    /mnt/data/text2spatial_v4_qc_report.csv
"""

import json, math, re, random, csv, os
from pathlib import Path
from statistics import median, mean, pstdev

# ---------- 경로 ----------
SRC = "/workspace/mmhoa/text2hoa/text2spatial_v3_rebalanced.jsonl"
OUT_PREFIX = "/workspace/mmhoa/text2hoa/text2spatial_v4"
TRAIN = OUT_PREFIX + "_train.jsonl"
VALID = OUT_PREFIX + "_valid.jsonl"
TEST  = OUT_PREFIX + "_test.jsonl"
STATS = OUT_PREFIX + "_stats.json"
QC    = OUT_PREFIX + "_qc_report.csv"

# ---------- 로드 ----------
def getv(d, key, default=None):
    return (d.get(key) if isinstance(d, dict) else None) if d is not None else default

def load_jsonl(path):
    out=[]
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                out.append(json.loads(ln))
            except:
                pass
    return out

samples = load_jsonl(SRC)
assert len(samples)>0, f"입력 비어있음: {SRC}"

# ---------- 기본 통계 ----------
dist_list, spread_list, wet_list, gain_db_list, az_s_list, el_list = [],[],[],[],[],[]
langs = {"ko":0,"en":0}
for ex in samples:
    params = getv(ex,"params") or getv(ex,"spatial") or ex
    dist_m = float(getv(params,"dist_m", 1.0))
    spread_deg = float(getv(params,"spread_deg", 30.0))
    wet = float(getv(params,"wet_mix", 0.2))
    gain_db = float(getv(params,"gain_db", -3.0))
    az_sc = getv(params,"az_sc", 0.0)
    el_rad = float(getv(params,"el_rad", 0.0))

    dist_list.append(dist_m)
    spread_list.append(spread_deg)
    wet_list.append(wet)
    gain_db_list.append(gain_db)
    if isinstance(az_sc, list) and len(az_sc)>0: az_s_list.append(float(az_sc[0]))
    elif isinstance(az_sc, (int,float)): az_s_list.append(float(az_sc))
    el_list.append(el_rad)

    lang = getv(ex,"lang") or ("ko" if re.search(r"[가-힣]", getv(ex,"text","")) else "en")
    langs[lang] = langs.get(lang,0)+1

# 파생 통계
import numpy as np
logd = [math.log(x) for x in dist_list]
LOGD_MEAN = float(np.mean(logd))
LOGD_STD  = float(np.std(logd, ddof=0)) or 1.0
GAIN_LIN_MIN = float(min(10**(db/20.0) for db in gain_db_list))
GAIN_LIN_MAX = float(max(10**(db/20.0) for db in gain_db_list))
DIST_MED = float(median(dist_list))
SPREAD_MED = float(median(spread_list))

# ---------- 키워드(ko/en) ----------
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

def has(pat, text): return re.search(pat, text, flags=re.IGNORECASE) is not None
def clamp(x, lo, hi): return lo if x<lo else hi if x>hi else x
def sincos(x): return math.sin(x), math.cos(x)

# 합성 강도(필요시 조절)
DELTA_EL = 0.35       # 고도 변화 ~20°
DELTA_WIDTH = 0.30    # 폭 변화 30%

# ---------- 변환 + QC ----------
converted, qc_rows = [], []
for i, ex in enumerate(samples):
    text = getv(ex,"text","")
    lang = getv(ex,"lang") or ("ko" if re.search(r"[가-힣]", text) else "en")
    params = getv(ex,"params") or getv(ex,"spatial") or ex

    az_sc = getv(params,"az_sc", 0.0)
    el_rad = float(getv(params,"el_rad", 0.0))
    dist_m = float(getv(params,"dist_m", 1.0))
    spread_deg = float(getv(params,"spread_deg", 30.0))
    wet_mix = float(getv(params,"wet_mix", 0.2))
    gain_db = float(getv(params,"gain_db", -3.0))

    # Azimuth start/end (rad)
    if isinstance(az_sc, list) and len(az_sc)>=2:
        az_start = float(az_sc[0]) * math.pi
        az_end   = float(az_sc[1]) * math.pi
    else:
        az_val = float(az_sc) * math.pi
        az_start = az_end = az_val

    # Elevation start/end (rad)
    if isinstance(getv(params,"el_sc", None), list) and len(getv(params,"el_sc"))>=2:
        el_start = float(getv(params,"el_sc")[0]); el_end = float(getv(params,"el_sc")[-1])
    else:
        el_start = float(el_rad)
        if has(UP, text) and not has(DOWN, text):
            el_end = clamp(el_start + DELTA_EL, -math.pi/2, math.pi/2)
        elif has(DOWN, text) and not has(UP, text):
            el_end = clamp(el_start - DELTA_EL, -math.pi/2, math.pi/2)
        else:
            el_end = el_start

    # Width start/end (0–1)
    width_start = clamp(spread_deg/120.0, 0.0, 1.0)
    if has(WIDE, text) and not has(NARROW, text):
        width_end = clamp(width_start + DELTA_WIDTH, 0.0, 1.0)
    elif has(NARROW, text) and not has(WIDE, text):
        width_end = clamp(width_start - DELTA_WIDTH, 0.0, 1.0)
    else:
        width_end = width_start

    # Distance (log-z)
    dist_z = (math.log(max(dist_m, 1e-6)) - LOGD_MEAN) / (LOGD_STD if LOGD_STD>1e-6 else 1.0)

    # Wet (0–1) + 텍스트 보정
    wet_norm = clamp(wet_mix, 0.0, 1.0)
    if has(REV_MORE, text) and not has(REV_LESS, text):
        wet_norm = clamp(wet_norm + 0.15, 0.0, 1.0)
    elif has(REV_LESS, text) and not has(REV_MORE, text):
        wet_norm = clamp(wet_norm - 0.15, 0.0, 1.0)

    # Gain (dB → linear → 0–1) + 텍스트 보정
    gain_lin = 10**(gain_db/20.0)
    gain_norm = (gain_lin - GAIN_LIN_MIN) / (GAIN_LIN_MAX - GAIN_LIN_MIN + 1e-8)
    if has(LOUD, text) and not has(QUIET, text):
        gain_norm = clamp(gain_norm + 0.1, 0.0, 1.0)
    elif has(QUIET, text) and not has(LOUD, text):
        gain_norm = clamp(gain_norm - 0.1, 0.0, 1.0)

    # sin/cos 표현
    az_sin_start, az_cos_start = sincos(az_start)
    az_sin_end,   az_cos_end   = sincos(az_end)
    el_sin_start, el_cos_start = sincos(el_start)
    el_sin_end,   el_cos_end   = sincos(el_end)

    # 마스크(문장 명시=1.0, 아니면=0.5)
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

    # --- QC 규칙: 텍스트 vs 원 라벨 불일치 경고 ---
    warn = []
    # 좌/우
    if has(RIGHT, text) and not has(LEFT, text):
        az_s = (az_sc[0] if isinstance(az_sc, list) else az_sc)
        if not (az_s > 0): warn.append("az_start sign!=right")
    if has(LEFT, text) and not has(RIGHT, text):
        az_s = (az_sc[0] if isinstance(az_sc, list) else az_sc)
        if not (az_s < 0): warn.append("az_start sign!=left")
    # 위/아래
    if has(UP, text) and not has(DOWN, text) and el_rad <= 0: warn.append("el_start sign!=up")
    if has(DOWN, text) and not has(UP, text) and el_rad >= 0: warn.append("el_start sign!=down")
    # 거리
    if has(NEAR, text) and dist_m >= DIST_MED: warn.append("near but dist>=median")
    if has(FAR, text) and dist_m <= DIST_MED: warn.append("far but dist<=median")
    # 폭
    if has(WIDE, text) and spread_deg <= SPREAD_MED: warn.append("wide but spread<=median")
    if has(NARROW, text) and spread_deg >= SPREAD_MED: warn.append("narrow but spread>=median")

    qc_rows.append({"idx": i, "lang": lang, "text_snippet": text[:120].replace("\n"," "), "warnings": "; ".join(warn)})

# ---------- 분할(80/10/10) ----------
random.seed(42)
idxs = list(range(len(converted)))
random.shuffle(idxs)
n = len(idxs)
n_tr = int(0.8*n); n_va = int(0.1*n)
train_idx = idxs[:n_tr]; valid_idx = idxs[n_tr:n_tr+n_va]; test_idx = idxs[n_tr+n_va:]

def write_jsonl(path, subset):
    with open(path, "w", encoding="utf-8") as f:
        for i in subset:
            f.write(json.dumps(converted[i], ensure_ascii=False)+"\n")

write_jsonl(TRAIN, train_idx)
write_jsonl(VALID, valid_idx)
write_jsonl(TEST,  test_idx)

# ---------- 통계/메타 ----------
stats = {
    "source": SRC,
    "N": len(samples),
    "splits": {"train": len(train_idx), "valid": len(valid_idx), "test": len(test_idx)},
    "normalization": {
        "log_dist": {"mean": LOGD_MEAN, "std": LOGD_STD},
        "gain_lin": {"min": GAIN_LIN_MIN, "max": GAIN_LIN_MAX},
        "width_scale_deg": 120.0,
        "azimuth_scale": "az_sc * pi (rad)",
        "elevation": "el_rad as-is (rad)",
        "synth_delta": {"elevation_rad": DELTA_EL, "width": DELTA_WIDTH}
    },
    "medians_for_qc": {"dist_m": DIST_MED, "spread_deg": SPREAD_MED},
    "lang_counts": langs
}
with open(STATS, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# ---------- QC CSV ----------
with open(QC, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["idx","lang","text_snippet","warnings"])
    writer.writeheader()
    for r in qc_rows:
        writer.writerow(r)

print("done:", TRAIN, VALID, TEST, STATS, QC)
