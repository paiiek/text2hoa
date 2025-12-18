# -*- coding: utf-8 -*-
import csv, argparse, os
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    rows=[]; 
    with open(path, newline='', encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="figs_ablation")
    args=ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows=load_csv(args.csv)
    xs=[r["variant"] for r in rows]
    AE=[float(r["AE_deg"]) for r in rows]
    fig = plt.figure(figsize=(8,4))
    idx=np.arange(len(xs))
    plt.bar(idx, AE)
    plt.xticks(idx, xs, rotation=30, ha="right")
    plt.ylabel("AE (deg)"); plt.title("Ablation on validation")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir,"ablation_ae.png"), dpi=200)
    plt.close(fig)

    # LaTeX table
    with open(os.path.join(args.outdir,"ablation_table.tex"),"w",encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrl}\n\\toprule\n")
        f.write("Variant & AE$\\downarrow$ & d$_{log}\\downarrow$ & Notes\\\\\\midrule\n")
        for r in rows:
            f.write(f"{r['variant']} & {float(r['AE_deg']):.2f} & {float(r['d_log']):.3f} & {r.get('notes','')}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

if __name__=="__main__":
    main()
