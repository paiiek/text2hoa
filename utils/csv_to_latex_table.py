# -*- coding: utf-8 -*-
# save as: csv_to_latex_table.py
import csv, argparse

ap=argparse.ArgumentParser()
ap.add_argument("--csv", default="baseline_results.csv")
ap.add_argument("--caption", default="Overall metrics on full dataset.")
ap.add_argument("--label", default="tab:baseline")
args=ap.parse_args()

rows=[]
with open(args.csv,encoding="utf-8") as f:
    r=csv.DictReader(f)
    for x in r: rows.append(x)

cols=[("variant","Method"),("AE_deg","AE↓"),("d_log","d_log↓"),
      ("spread_MAE","Spread MAE↓"),("wet_MAE","Wet MAE↓"),
      ("gain_MAE","Gain MAE↓"),("room_MAE","Room MAE↓")]

print("\\begin{table}[t]")
print("\\centering")
print("\\begin{tabular}{lrrrrrr}")
print("\\toprule")
print(" & ".join([c[1] for c in cols])+" \\\\ \\midrule")
for x in rows:
    line=[]
    for k,_ in cols:
        v = x[k]
        if k!="variant":
            try: v = f"{float(v):.2f}"
            except: pass
        line.append(v)
    print(" & ".join(line)+" \\\\")
print("\\bottomrule")
print("\\caption{"+args.caption+"}")
print("\\label{"+args.label+"}")
print("\\end{tabular}")
print("\\end{table}")
