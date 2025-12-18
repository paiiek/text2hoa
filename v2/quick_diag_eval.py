
# Quick diagnostics on icassp_test_preds.csv
import csv, math, numpy as np, statistics as st, sys

csv_path = sys.argv[1] if len(sys.argv)>1 else "icassp_test_preds.csv"
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

def to_f(k): return [float(r[k]) for r in rows]

def circ(a, b):
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return abs(d) * 180.0 / math.pi

def to_ang(s, c):
    return [math.atan2(float(si), float(co)) for si,co in zip(to_f(s), to_f(c))]

for pair in [("az","start"),("az","end"),("el","start"),("el","end")]:
    base = f"{pair[0]}_{pair[1]}"
    gt = to_ang(f"y_true_{pair[0]}_sin_{pair[1]}", f"y_true_{pair[0]}_cos_{pair[1]}")
    pdv= to_ang(f"y_pred_{pair[0]}_sin_{pair[1]}", f"y_pred_{pair[0]}_cos_{pair[1]}")
    err = [circ(a,b) for a,b in zip(gt,pdv)]
    print(f"{base} deg MAE: {st.mean(err):.2f} (N={len(err)})")

# Norm check for predicted sin^2+cos^2
import numpy as np
def norm_stats(prefix):
    s = np.array(to_f(f"y_pred_{prefix}_sin_start")); c = np.array(to_f(f"y_pred_{prefix}_cos_start"))
    rad = s*s + c*c
    return float(rad.mean()), float(rad.std())
for which in ["az","el"]:
    m,sd = norm_stats(which)
    print(f"pred {which} start sin^2+cos^2 mean={m:.3f} std={sd:.3f}")
