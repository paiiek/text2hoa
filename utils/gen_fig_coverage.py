# -*- coding: utf-8 -*-
# Create coverage figures & LaTeX table from text2spatial jsonl
import json, math, argparse, os
import numpy as np
import matplotlib.pyplot as plt

def az12_from_sc(s, c):
    az = math.degrees(math.atan2(s, c))
    if az < 0: az += 360.0
    return int(az // 30)

def load_rows(paths, room_mode):
    rows=[]
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                x=json.loads(line)
                rd=x.get("room_depth", {})
                room = rd.get("drr_db") if room_mode=="drr" else rd.get("rt60_s")
                if room is None: continue
                rows.append({
                    "text": x["text"],
                    "s": float(x["az_sc"][0]),
                    "c": float(x["az_sc"][1]),
                    "el": float(x["el_rad"]),
                    "dist": float(x["dist_m"]),
                    "spread": float(x["spread_deg"]),
                    "wet": float(x["wet_mix"]),
                    "gain": float(x["gain_db"]),
                    "room": float(room)
                })
    return rows

def bins_and_counts(rows):
    az_bins=np.zeros(12, dtype=int)
    el_bins=[0,0,0]    # low<-0.2, mid[-0.2,0.2], high>0.2
    d_bins =[0,0,0]    # <1.5, [1.5,3), >=3
    sp_bins=[0,0,0]    # <30, [30,60), >=60
    wet_bins=[0,0,0]   # tertiles by value
    wets=[r["wet"] for r in rows]
    if len(wets)>0:
        q1, q2 = np.quantile(wets, [1/3, 2/3])
    else:
        q1=q2=0.33
    for r in rows:
        az_bins[az12_from_sc(r["s"], r["c"])]+=1
        el = r["el"]
        el_bins[0 if el<-0.2 else (2 if el>0.2 else 1)] += 1
        d  = r["dist"]
        d_bins[0 if d<1.5 else (1 if d<3.0 else 2)] += 1
        sp = r["spread"]
        sp_bins[0 if sp<30 else (1 if sp<60 else 2)] += 1
        w  = r["wet"]
        wet_bins[0 if w<q1 else (1 if w<q2 else 2)] += 1
    return az_bins, el_bins, d_bins, sp_bins, wet_bins, (q1,q2)

def save_polar_rose(az_bins, outpath):
    # centers at 15°,45°,...(12 bins), width 30°
    theta = np.deg2rad(np.arange(12)*30 + 15)
    width = np.deg2rad(30*np.ones(12))
    r = az_bins.astype(float)
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, r, width=width, align='center')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Azimuth coverage (12 bins)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_hist(vals, title, xlabel, outpath, bins=30):
    fig = plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel("count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_tex_table(el_bins, d_bins, sp_bins, wet_bins, outpath):
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Category & Bin1 & Bin2 & Bin3 & Total\\\\\\midrule\n")
        f.write(f"Elevation(low/mid/high) & {el_bins[0]} & {el_bins[1]} & {el_bins[2]} & {sum(el_bins)}\\\\\n")
        f.write(f"Distance(0.6-1.5/1.5-3/3-6m) & {d_bins[0]} & {d_bins[1]} & {d_bins[2]} & {sum(d_bins)}\\\\\n")
        f.write(f"Spread(5-30/30-60/60-120°) & {sp_bins[0]} & {sp_bins[1]} & {sp_bins[2]} & {sum(sp_bins)}\\\\\n")
        f.write(f"Wet(low/mid/high tertiles) & {wet_bins[0]} & {wet_bins[1]} & {wet_bins[2]} & {sum(wet_bins)}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--outdir", default="figs_cov")
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows=load_rows(args.data, args.room_mode)
    az, elb, db, spb, wb, wet_q = bins_and_counts(rows)
    save_polar_rose(az, os.path.join(args.outdir,"az_rose.png"))

    save_hist([math.degrees(math.atan2(r["s"], r["c"])) % 360 for r in rows], "Azimuth (deg)", "deg", os.path.join(args.outdir,"az_hist.png"))
    save_hist([r["el"] for r in rows], "Elevation (rad)", "rad", os.path.join(args.outdir,"el_hist.png"))
    save_hist([r["dist"] for r in rows], "Distance (m)", "m", os.path.join(args.outdir,"dist_hist.png"))
    save_hist([r["spread"] for r in rows], "Spread (deg)", "deg", os.path.join(args.outdir,"spread_hist.png"))
    save_hist([r["wet"] for r in rows], "Wet mix", "", os.path.join(args.outdir,"wet_hist.png"))

    save_tex_table(elb, db, spb, wb, os.path.join(args.outdir,"coverage_bins.tex"))
    with open(os.path.join(args.outdir,"coverage_summary.json"),"w",encoding="utf-8") as f:
        json.dump({"N":len(rows), "az12":az.tolist(), "el_bins":elb, "dist_bins":db, "spread_bins":spb, "wet_bins":wb, "wet_tertiles":wet_q}, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    main()
