#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, os, csv
from statistics import mean, pstdev

PAT_BASE = re.compile(
    r"\[baseline\].*?AE\s+([0-9.]+)_\s*\|\s*dlog\s+([0-9.]+)\s*\|\s*sp\s+([0-9.]+)\s*\|\s*wet\s+([0-9.]+)\s*\|\s*gain\s+([0-9.]+)"
)
PAT_KNN = re.compile(
    r"\[after KNN\]\s*fb\s+([0-9.]+)%\s*\|\s*AE\s+([0-9.]+)_\s*\|\s*dlog\s+([0-9.]+)\s*\|\s*sp\s+([0-9.]+)\s*\|\s*wet\s+([0-9.]+)\s*\|\s*gain\s+([0-9.]+)"
)
PAT_CAL = re.compile(
    r"\[after Calib\]\s*AE\s+([0-9.]+)_\s*\|\s*dlog\s+([0-9.]+)\s*\|\s*sp\s+([0-9.]+)\s*\|\s*wet\s+([0-9.]+)\s*\|\s*gain\s+([0-9.]+)"
)
PAT_SNAP = re.compile(
    r"\[after TextSnap\]\s*AE\s+([0-9.]+)_\s*\|\s*dlog\s+([0-9.]+)\s*\|\s*sp\s+([0-9.]+)\s*\|\s*wet\s+([0-9.]+)\s*\|\s*gain\s+([0-9.]+)"
)
PAT_HEADER = re.compile(
    r"\[data\].*?\n.*?\[split\].*?\n.*?\[parse\].*?\n.*?\[parse\].*?\n\[(?:ckpt|load)\].*?\n\[(?:load|auto-az)\].*?",
    re.S
)

def parse_file(path):
    with open(path, "r") as f:
        txt = f.read()
    row = {"file": os.path.basename(path)}
    # Baseline
    mb = PAT_BASE.search(txt)
    if mb:
        row.update({
            "AE_base": float(mb.group(1)),
            "dlog_base": float(mb.group(2)),
            "sp_base": float(mb.group(3)),
            "wet_base": float(mb.group(4)),
            "gain_base": float(mb.group(5)),
        })
    # KNN
    mk = PAT_KNN.search(txt)
    if mk:
        row.update({
            "fb": float(mk.group(1)),
            "AE_knn": float(mk.group(2)),
            "dlog_knn": float(mk.group(3)),
            "sp_knn": float(mk.group(4)),
            "wet_knn": float(mk.group(5)),
            "gain_knn": float(mk.group(6)),
        })
    # Calib
    mc = PAT_CAL.search(txt)
    if mc:
        row.update({
            "AE_cal": float(mc.group(1)),
            "dlog_cal": float(mc.group(2)),
            "sp_cal": float(mc.group(3)),
            "wet_cal": float(mc.group(4)),
            "gain_cal": float(mc.group(5)),
        })
    # TextSnap
    ms = PAT_SNAP.search(txt)
    if ms:
        row.update({
            "AE_snap": float(ms.group(1)),
            "dlog_snap": float(ms.group(2)),
            "sp_snap": float(ms.group(3)),
            "wet_snap": float(ms.group(4)),
            "gain_snap": float(ms.group(5)),
        })
    return row

def fmt_mu_sd(vals, nd=2):
    if not vals: return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    mu = mean(vals); sd = pstdev(vals)
    return f"{mu:.{nd}f} ± {sd:.{nd}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="one or more eval log files")
    ap.add_argument("--out_csv", default="ablation_summary.csv")
    ap.add_argument("--out_md", default="ablation_summary.md")
    args = ap.parse_args()

    rows = [parse_file(p) for p in args.logs]

    # save CSV
    fields = [
        "file",
        "AE_base","dlog_base","sp_base","wet_base","gain_base",
        "fb","AE_knn","dlog_knn","sp_knn","wet_knn","gain_knn",
        "AE_cal","dlog_cal","sp_cal","wet_cal","gain_cal",
        "AE_snap","dlog_snap","sp_snap","wet_snap","gain_snap"
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[saved] {args.out_csv}")

    # aggregate (전체 로그 평균)
    def collect(k): return [r[k] for r in rows if k in r]
    md = []
    md.append("| Split | AE | dlog | spread | wet | gain | fb% |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    # baseline
    md.append("| Baseline | {AE} | {dlog} | {sp} | {wet} | {gain} | - |".format(
        AE=fmt_mu_sd(collect("AE_base")),
        dlog=fmt_mu_sd(collect("dlog_base"), 3),
        sp=fmt_mu_sd(collect("sp_base")),
        wet=fmt_mu_sd(collect("wet_base"), 3),
        gain=fmt_mu_sd(collect("gain_base")),
    ))
    # after KNN
    if collect("AE_knn"):
        md.append("| After KNN | {AE} | {dlog} | {sp} | {wet} | {gain} | {fb} |".format(
            AE=fmt_mu_sd(collect("AE_knn")),
            dlog=fmt_mu_sd(collect("dlog_knn"), 3),
            sp=fmt_mu_sd(collect("sp_knn")),
            wet=fmt_mu_sd(collect("wet_knn"), 3),
            gain=fmt_mu_sd(collect("gain_knn")),
            fb=fmt_mu_sd(collect("fb")),
        ))
    # after Calib
    if collect("AE_cal"):
        md.append("| After Calib | {AE} | {dlog} | {sp} | {wet} | {gain} | - |".format(
            AE=fmt_mu_sd(collect("AE_cal")),
            dlog=fmt_mu_sd(collect("dlog_cal"), 3),
            sp=fmt_mu_sd(collect("sp_cal")),
            wet=fmt_mu_sd(collect("wet_cal"), 3),
            gain=fmt_mu_sd(collect("gain_cal")),
        ))
    # after TextSnap
    if collect("AE_snap"):
        md.append("| After TextSnap | {AE} | {dlog} | {sp} | {wet} | {gain} | - |".format(
            AE=fmt_mu_sd(collect("AE_snap")),
            dlog=fmt_mu_sd(collect("dlog_snap"), 3),
            sp=fmt_mu_sd(collect("sp_snap")),
            wet=fmt_mu_sd(collect("wet_snap"), 3),
            gain=fmt_mu_sd(collect("gain_snap")),
        ))

    with open(args.out_md, "w") as f:
        f.write("\n".join(md))
    print(f"[saved] {args.out_md}")
    print("\n==== Markdown Preview ====\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
