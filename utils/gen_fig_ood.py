# -*- coding: utf-8 -*-
# Plot OOD category bars and emit LaTeX table from ood_metrics.csv
import csv, argparse, os
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    rows=[]
    with open(path, newline='', encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def to_float(r, key): 
    try: return float(r[key])
    except: return None

def barplot(xs, ys, title, ylabel, outpath):
    fig = plt.figure(figsize=(7,4))
    idx = np.arange(len(xs))
    plt.bar(idx, ys)
    plt.xticks(idx, xs, rotation=20)
    plt.ylabel(ylabel); plt.title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_tex(rows, outpath):
    keys = ["split","N","AE_deg","d_log","spread_MAE","wet_MAE","gain_MAE","room_MAE"]
    with open(outpath,"w",encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        f.write("Split & N & AE$\\downarrow$ & d$_{log}\\downarrow$ & Spread$\\downarrow$ & Wet$\\downarrow$ & Gain$\\downarrow$ & Room$\\downarrow$\\\\\\midrule\n")
        for r in rows:
            f.write(f"{r['split']} & {r['N']} & {float(r['AE_deg']):.2f} & {float(r['d_log']):.3f} & {float(r['spread_MAE']):.2f} & {float(r['wet_MAE']):.3f} & {float(r['gain_MAE']):.2f} & {float(r['room_MAE']):.2f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="figs_ood")
    args=ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows=load_csv(args.csv)
    cats=[r["split"] for r in rows if r["split"]!="overall"]
    AE=[to_float(r,"AE_deg") for r in rows if r["split"]!="overall"]
    barplot(cats, AE, "OOD AE by category", "AE (deg)", os.path.join(args.outdir,"ood_ae.png"))

    save_tex(rows, os.path.join(args.outdir,"ood_table.tex"))

if __name__=="__main__":
    main()
