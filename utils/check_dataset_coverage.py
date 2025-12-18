# save as: check_dataset_coverage.py
import json, argparse, math
from collections import Counter, defaultdict

def bin_stats(vals, edges):
    bins=[0]* (len(edges)-1)
    for v in vals:
        for i in range(len(edges)-1):
            if edges[i] <= v < edges[i+1]: bins[i]+=1; break
    return bins

ap=argparse.ArgumentParser()
ap.add_argument("--data", required=True)
args=ap.parse_args()

rows=[json.loads(l) for l in open(args.data,encoding="utf-8")]
langs=Counter(r["lang"] for r in rows)
print("langs:", dict(langs), "N=",len(rows))

# az quadrant
def quad(s,c):
    import math
    az=math.degrees(math.atan2(s,c))
    if -45<=az<45: return "front"
    if 45<=az<135: return "right"
    if -135<=az<-45: return "left"
    return "back"
q=Counter(quad(*r["az_sc"]) for r in rows)
print("az quads:", dict(q))

els=[r["el_rad"] for r in rows]
print("el bins(low,mid,high):", bin_stats(els, [-10, -0.2, 0.2, 10]))

dists=[r["dist_m"] for r in rows]
print("dist bins(0.6-1.5,1.5-3,3-6):", bin_stats(dists, [0,1.5,3,10]))

spreads=[r["spread_deg"] for r in rows]
print("spread bins(5-30,30-60,60-120):", bin_stats(spreads, [0,30,60,120]))

wets=[r["wet_mix"] for r in rows]
print("wet bins(0-0.33,0.33-0.66,0.66-1):", bin_stats(wets, [0,0.33,0.66,1.01]))
