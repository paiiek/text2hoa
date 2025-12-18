# save as: quick_stats_v2.py
import json, math, argparse
from collections import defaultdict

def az_deg(s,c):
    a = math.degrees(math.atan2(s,c))
    return a+360 if a<0 else a
def az12(a): return int(az_deg(a[0],a[1])//30)

def main(path):
    rows=[json.loads(l) for l in open(path,encoding="utf-8")]
    N=len(rows)
    by_text=defaultdict(int)
    bins=[0]*12
    azset=set(); el_min=1e9; el_max=-1e9
    for x in rows:
        by_text[x["text"]]+=1
        bins[az12(x["az_sc"])]+=1
        azset.add(round(az_deg(*x["az_sc"]),3))
        el=x["el_rad"]; el_min=min(el_min,el); el_max=max(el_max,el)
    print(f"N={N}")
    print("unique texts:", len(by_text))
    print("avg dup per text:", round(N/max(1,len(by_text)),2))
    print("az unique count:", len(azset))
    print("az 12-bin:", bins)
    print(f"el range: {el_min:.3f} ~ {el_max:.3f} rad (~ {math.degrees(el_min):.1f}° ~ {math.degrees(el_max):.1f}°)")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args=ap.parse_args()
    main(args.data)
