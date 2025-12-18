#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_text2spatial.py
- 공개 감정 코퍼스(GoEmotions/EmoBank/KOTE)와 로컬 텍스트를 모아 한 덩어리로 구성
- v3 YAML 스키마에 맞춰 '프라이어' 타깃(weak labels) 생성
- train/val/test 스플릿 및 JSONL 내보내기
- 하루 안에 학습 가능한 경량 파이프라인에 맞춰 설계됨

Usage (예시):
python prepare_text2spatial.py \
  --out_dir out_data \
  --sources go_emotions emobank kote templates \
  --local_glob "/path/to/*.csv" "/path/to/*.jsonl" \
  --max_per_source 8000 \
  --min_len 5 --max_len 400 \
  --lang both \
  --dedup \
  --train_ratio 0.9 --val_ratio 0.05 --test_ratio 0.05 \
  --yaml /mnt/data/pro_params_v3.yaml

필요 패키지:
  pip install datasets pandas pyyaml transformers torch
"""
import os, re, sys, glob, json, math, random, argparse
from typing import List, Dict, Any
import pandas as pd

# HuggingFace datasets는 선택적 - 없는 경우 해당 소스만 건너뜀
try:
    from datasets import load_dataset, Dataset, concatenate_datasets
except Exception:
    load_dataset = None
    Dataset = None
    concatenate_datasets = None

# -------------------------
# 유틸
# -------------------------
def set_seed(seed:int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def is_korean(s:str)->bool:
    return bool(re.search(r"[가-힣]", s))

def is_english(s:str)->bool:
    return bool(re.search(r"[A-Za-z]", s))

def normalize_text(s:str)->str:
    return re.sub(r"\s+", " ", s.strip())

def stratified_split(items:List[Dict[str,Any]], train_ratio:float, val_ratio:float, test_ratio:float, seed:int=42):
    assert abs(train_ratio+val_ratio+test_ratio - 1.0) < 1e-6, "split ratios must sum to 1"
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n*train_ratio)
    n_val = int(n*val_ratio)
    train = items[:n_train]
    val   = items[n_train:n_train+n_val]
    test  = items[n_train+n_val:]
    return train, val, test

def save_jsonl(rows:List[Dict[str,Any]], path:str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[save] {path} ({len(rows)} rows)")

# -------------------------
# v3 YAML 로드
# -------------------------
import yaml
def load_v3_yaml(yaml_path:str):
    with open(yaml_path, "r") as f:
        conf = yaml.safe_load(f)
    # 중복 파라미터는 마지막 정의를 채택
    seen = {}
    for p in conf["params"]:
        seen[p["name"]] = p
    params = list(seen.values())
    return {
        "K_POS": conf["K"]["pos"],
        "K_FX": conf["K"]["fx"],
        "params": params
    }

def steps_for(p, K_POS, K_FX):
    return K_POS if p["K"]=="pos" else K_FX

# 공통 스케일(sigmoid→[lo,hi])
def sigmoid_scale(x, lo, hi):
    import math
    s = 1/(1+math.exp(-x))
    return lo + (hi-lo)*s

# -------------------------
# 간단 프라이어 생성기 (v3에 맞춤)
# - 규칙/키워드 중심 (NRC-VAD 등 외부 자원 없이)
# - 필요시 --lexicon_json 경로로 사용자 사전 연결 가능 (키워드->파라미터 바이어스)
# -------------------------
DEFAULT_TEMPLATES = [
    "속삭임이 등 뒤에서 시작해 오른쪽으로 스친다. 잔향이 길게 남는다.",
    "머리 위에서 종이 울리고 점점 멀어진다.",
    "왼쪽 뒤에서 달려와 정면을 지나간다, 휘익.",
    "문 너머에서 웅웅거리는 소리, 답답하게 들린다.",
    "성당 홀처럼 긴 잔향, 따뜻하고 부드럽다.",
    "아래쪽에서 시작해 위로 치솟는다, 도플러가 크게 느껴진다.",
    "정면에서 또렷하게 들리고, 포커스가 좁다.",
]

def priors_from_texts(texts:List[str], schema:Dict[str,Any], user_lexicon:Dict[str,Any]=None):
    K_POS, K_FX = schema["K_POS"], schema["K_FX"]
    params = schema["params"]

    def apply_rules(name, lo, hi, K, t):
        # 기본값: 중앙
        base = [ (lo+hi)/2.0 for _ in range(K) ]
        tl = t.lower()

        # 공간/차폐/직접성
        if name=="occlusion" and any(k in t for k in ["뒤", "등 뒤", "벽 너머", "문 너머", "behind", "muffled"]):
            base = [ max(base[i], hi*0.7) for i in range(K) ]
        if name=="direct_alpha" and any(k in t for k in ["또렷", "명료", "정면", "앞", "focus", "sharp"]):
            base = [ max(base[i], lo + (hi-lo)*0.75) for i in range(K) ]
        if name=="direct_focus_deg" and any(k in t for k in ["왼", "left"]):
            base = [ -60.0 for _ in range(K) ]
        if name=="direct_focus_deg" and any(k in t for k in ["오", "right"]):
            base = [  60.0 for _ in range(K) ]

        # 머리 회전
        if name=="head_yaw_deg" and ("왼" in t or "left" in tl):
            base = [ -30.0 for _ in range(K) ]
        if name=="head_yaw_deg" and ("오" in t or "right" in tl):
            base = [  30.0 for _ in range(K) ]
        if name=="head_pitch_deg" and any(k in t for k in ["위", "overhead", "위쪽"]):
            base = [  15.0 for _ in range(K) ]
        if name=="head_pitch_deg" and any(k in t for k in ["아래", "below"]):
            base = [ -10.0 for _ in range(K) ]

        # 움직임/도플러
        if name=="doppler_depth" and any(k in t for k in ["다가오", "지나가", "swoosh", "whizz", "train"]):
            base = [ max(base[i], lo + (hi-lo)*0.8) for i in range(K) ]

        # 감정(arousal proxy) -> 톤/컴프/젖음/확산
        arousal_hot = any(k in t for k in ["비명", "절규", "격앙", "분노", "scream", "panic"])
        if arousal_hot:
            if name=="eq_h_gain_db":
                base = [ min(hi, 6.0) for _ in range(K) ]
            if name=="comp_ratio":
                base = [ max(base[i], lo + (hi-lo)*0.7) for i in range(K) ]
            if name=="comp_attack_ms":
                base = [ lo + (hi-lo)*0.2 for _ in range(K) ]
            if name=="wet_mix":
                base = [ max(base[i], lo + (hi-lo)*0.65) for i in range(K) ]
            if name=="spread_deg":
                base = [ max(base[i], lo + (hi-lo)*0.6) for i in range(K) ]

        # 장소/잔향
        if any(k in tl for k in ["hall","cathedral","gym","성당","체육관","큰 방","동굴"]):
            if name=="rt60_s":
                base = [ max(base[i], min(hi, 1.8)) for i in range(K) ]
            if name=="er_gain":
                base = [ max(base[i], lo + (hi-lo)*0.7) for i in range(K) ]

        # 사용자 정의 렉시콘 반영(옵션): {"rules": [{"if_contains": ["키워드"...], "add": {"param": +delta or value}}]}
        if user_lexicon and "rules" in user_lexicon:
            for r in user_lexicon["rules"]:
                cond = False
                for kw in r.get("if_contains", []):
                    if kw.lower() in tl or kw in t:
                        cond = True; break
                if cond:
                    if "set" in r:
                        if name in r["set"]:
                            val = r["set"][name]
                            base = [ float(val) for _ in range(K) ]
                    if "add" in r:
                        if name in r["add"]:
                            delta = float(r["add"][name])
                            base = [ min(hi, max(lo, base[i] + delta)) for i in range(K) ]
        return base

    priors = []
    for t in texts:
        item = {}
        for p in params:
            name = p["name"]
            K = steps_for(p, K_POS, K_FX)
            lo = p.get("lo", 0.0)
            hi = p.get("hi", 1.0)
            item[name] = apply_rules(name, lo, hi, K, t)
        priors.append(item)
    return priors

# -------------------------
# 데이터 로더들
# -------------------------
def load_go_emotions(max_per_source:int=None, seed:int=42):
    if load_dataset is None: 
        print("[WARN] datasets 미설치로 go_emotions 건너뜀"); return []
    try:
        ds = load_dataset("go_emotions")["train"]
        # 텍스트 컬럼 이름 정규화
        if "text" not in ds.column_names:
            ds = ds.rename_column("comment_text", "text") if "comment_text" in ds.column_names else ds
        if "text" not in ds.column_names:
            raise RuntimeError("go_emotions: text column not found")
        if max_per_source:
            ds = ds.shuffle(seed=seed).select(range(min(max_per_source, len(ds))))
        return [{"text": str(x)} for x in ds["text"]]
    except Exception as e:
        print("[WARN] go_emotions 실패:", e)
        return []

def load_emobank(max_per_source:int=None, seed:int=42):
    if load_dataset is None: 
        print("[WARN] datasets 미설치로 emobank 건너뜀"); return []
    tried = ["emobank", "emobank/emobank"]
    for name in tried:
        try:
            ds = load_dataset(name)["train"]
            cols = ds.column_names
            text_col = "text" if "text" in cols else ("sentence" if "sentence" in cols else None)
            if not text_col: 
                continue
            ds = ds.rename_column(text_col, "text")
            if max_per_source:
                ds = ds.shuffle(seed=seed).select(range(min(max_per_source, len(ds))))
            return [{"text": str(x)} for x in ds["text"]]
        except Exception as e:
            print(f"[WARN] {name} 실패:", e)
    return []

def load_kote(max_per_source:int=None, seed:int=42):
    if load_dataset is None: 
        print("[WARN] datasets 미설치로 KOTE 건너뜀"); return []
    tried = ["kote", "kote-for-emotion", "kote/kote"]
    for name in tried:
        try:
            ds = load_dataset(name)["train"]
            cols = ds.column_names
            cand = [c for c in ["text","sentence","utterance","comment"] if c in cols]
            if not cand: 
                continue
            ds = ds.rename_column(cand[0], "text")
            if max_per_source:
                ds = ds.shuffle(seed=seed).select(range(min(max_per_source, len(ds))))
            return [{"text": str(x)} for x in ds["text"]]
        except Exception as e:
            print(f"[WARN] {name} 실패:", e)
    return []

def load_local_files(globs:List[str]):
    rows = []
    for pattern in globs:
        for fp in glob.glob(pattern):
            try:
                if fp.endswith(".csv"):
                    df = pd.read_csv(fp)
                elif fp.endswith(".tsv"):
                    df = pd.read_csv(fp, sep="\t")
                elif fp.endswith(".jsonl"):
                    with open(fp, "r", encoding="utf-8") as f:
                        df = pd.DataFrame([json.loads(x) for x in f])
                elif fp.endswith(".json"):
                    df = pd.read_json(fp, lines=False)
                else:
                    print("[INFO] unsupported file:", fp)
                    continue
                if "text" in df.columns:
                    for x in df["text"].astype(str).tolist():
                        rows.append({"text": x})
                else:
                    print("[WARN] 'text' 컬럼이 없어 건너뜀:", fp)
            except Exception as e:
                print("[WARN] 로컬 파일 로드 실패:", fp, e)
    return rows

def load_templates(extra_templates_path:str=None):
    rows = [{"text": t} for t in DEFAULT_TEMPLATES]
    if extra_templates_path and os.path.isfile(extra_templates_path):
        try:
            with open(extra_templates_path, "r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if line:
                        rows.append({"text": line})
        except Exception as e:
            print("[WARN] 템플릿 파일 로드 실패:", e)
    return rows

# -------------------------
# 메인 파이프라인
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--sources", nargs="+", default=["go_emotions","emobank","kote","templates"],
                    help="사용할 소스 선택: go_emotions emobank kote local templates")
    ap.add_argument("--local_glob", nargs="*", default=[], help="로컬 파일 글롭 패턴들 (csv/tsv/json/jsonl, 열이름 'text')")
    ap.add_argument("--extra_templates", type=str, default=None, help="한 줄당 한 문장인 텍스트 파일 경로")
    ap.add_argument("--max_per_source", type=int, default=None, help="각 소스별 최대 샘플 수")
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=500)
    ap.add_argument("--lang", type=str, choices=["ko","en","both"], default="both", help="간단 언어 필터(한글 포함/영문 포함)")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument("--yaml", type=str, default="/mnt/data/pro_params_v3.yaml", help="v3 YAML 경로")
    ap.add_argument("--lexicon_json", type=str, default=None, help="사용자 정의 키워드-파라미터 룰(JSON)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 소스별 로드
    pool = []
    if "go_emotions" in args.sources:
        pool += load_go_emotions(args.max_per_source, args.seed)
    if "emobank" in args.sources:
        pool += load_emobank(args.max_per_source, args.seed)
    if "kote" in args.sources:
        pool += load_kote(args.max_per_source, args.seed)
    if "local" in args.sources and args.local_glob:
        pool += load_local_files(args.local_glob)
    if "templates" in args.sources:
        pool += load_templates(args.extra_templates)

    # 2) 정제/필터/중복제거
    texts = []
    seen = set()
    for r in pool:
        if not isinstance(r, dict) or "text" not in r: 
            continue
        t = normalize_text(str(r["text"]))
        if len(t) < args.min_len or len(t) > args.max_len:
            continue
        if args.lang=="ko" and not is_korean(t):
            continue
        if args.lang=="en" and not is_english(t):
            continue
        if args.dedup:
            if t in seen: 
                continue
            seen.add(t)
        texts.append(t)

    print(f"[info] collected texts: {len(texts)}")
    assert len(texts) > 0, "수집된 텍스트가 없습니다. sources/local_glob/filters를 확인하세요."

    # 3) YAML 스키마 & 사용자 렉시콘 로드
    schema = load_v3_yaml(args.yaml)
    user_lex = None
    if args.lexicon_json and os.path.isfile(args.lexicon_json):
        with open(args.lexicon_json, "r", encoding="utf-8") as f:
            user_lex = json.load(f)

    # 4) 프라이어 타깃 생성 (weak labels)
    print("[info] generating priors (weak labels)...")
    priors = priors_from_texts(texts, schema, user_lexicon=user_lex)

    rows = []
    for t, pri in zip(texts, priors):
        item = {"text": t}
        item.update(pri)  # 각 파라미터 이름: [K] 시퀀스
        rows.append(item)

    # 5) 스플릿 & 저장
    train, val, test = stratified_split(rows, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
    save_jsonl(train, os.path.join(args.out_dir, "train_weak.jsonl"))
    save_jsonl(val,   os.path.join(args.out_dir, "val_weak.jsonl"))
    save_jsonl(test,  os.path.join(args.out_dir, "test_weak.jsonl"))

    # 6) 메타 저장(재현성)
    meta = {
        "args": vars(args),
        "n_total": len(rows),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "yaml": args.yaml,
        "params": [p["name"] for p in schema["params"]],
        "K_POS": schema["K_POS"],
        "K_FX": schema["K_FX"]
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[done] dataset prepared.")

if __name__ == "__main__":
    main()



