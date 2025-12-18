# -*- coding: utf-8 -*-
# responses.csv 형식:
# subject_id,prompt_id,system,file,fit,clarity,naturalness,preference(optional)
# 각 점수는 1..5 정수. 캐치트라이얼은 prompt_id가 같은 항목이 2개 이상.
import csv, argparse, statistics, math
from collections import defaultdict

def mean_ci(vals, alpha=0.05):
    n=len(vals)
    if n==0: return float('nan'), float('nan')
    m=statistics.mean(vals)
    if n==1: return m, 0.0
    sd=statistics.pstdev(vals) if n>1 else 0.0
    # 근사로 1.96 사용(표본 작으면 t-분포 권장이나 간단히)
    ci=1.96*sd/(n**0.5)
    return m, ci

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--resp", default="responses.csv")
    ap.add_argument("--out", default="mos_summary.csv")
    args=ap.parse_args()

    rows=[]
    with open(args.resp, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            for k in ["fit","clarity","naturalness","preference"]:
                if k in row and row[k]!="":
                    row[k]=int(row[k])
            rows.append(row)

    # 일관성(캐치) 검사: 동일 prompt_id, 동일 file 이 두 번 이상이면 차이 계산
    by_sub_prompt_file = defaultdict(list)
    for r in rows:
        key=(r["subject_id"], r["prompt_id"], r["file"])
        by_sub_prompt_file[key].append(r)
    inconsistencies=[]
    for key, lst in by_sub_prompt_file.items():
        if len(lst)>=2:
            d = {}
            for k in ["fit","clarity","naturalness"]:
                if k in lst[0] and lst[0][k]!="" and k in lst[1] and lst[1][k]!="":
                    d[k]=abs(lst[0][k]-lst[1][k])
            inconsistencies.append((key, d))
    if inconsistencies:
        print("[info] duplicates(consistency check) found:", len(inconsistencies))

    # 시스템/프롬프트 별 MOS
    by_sys = defaultdict(lambda: defaultdict(list))
    by_prompt_sys = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sys=r["system"]; pid=r["prompt_id"]
        for k in ["fit","clarity","naturalness"]:
            if k in r and isinstance(r[k], int):
                by_sys[sys][k].append(r[k])
                by_prompt_sys[pid+"__"+sys][k].append(r[k])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["level","id","metric","MOS","±95%CI","N"])
        # 시스템 레벨
        for sys, d in by_sys.items():
            for met, vals in d.items():
                m, ci = mean_ci(vals)
                w.writerow(["system", sys, met, f"{m:.3f}", f"{ci:.3f}", len(vals)])
        # 프롬프트×시스템 레벨
        for key, d in by_prompt_sys.items():
            for met, vals in d.items():
                m, ci = mean_ci(vals)
                w.writerow(["prompt×system", key, met, f"{m:.3f}", f"{ci:.3f}", len(vals)])
    print("wrote:", args.out)

if __name__=="__main__": main()
