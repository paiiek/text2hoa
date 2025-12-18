#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_rulelex_eval.py
- 매우 단순한 한/영 키워드 기반 파라미터 추정(az/el/dist/spread/wet/gain)
- 고정 split에서 test 성능(AE/dlog/MAE들) 산출
"""
import argparse, json, math

EPS=1e-8
def ang_wrap_rad(a):
    # [-pi, pi]
    while a>math.pi: a-=2*math.pi
    while a<-math.pi: a+=2*math.pi
    return a

def ae_deg_from_sc(gt_cos, gt_sin, pr_cos, pr_sin):
    gt = math.atan2(gt_sin, gt_cos)
    pr = math.atan2(pr_sin, pr_cos)
    d = abs(ang_wrap_rad(gt-pr))
    if d>math.pi: d=2*math.pi-d
    return d*180.0/math.pi

# ---------- 키워드 사전(축약) ----------
K_LEFT   = ["왼","좌","left","port"]
K_RIGHT  = ["오른","우","right","starboard"]
K_FRONT  = ["정면","앞","front","ahead"]
K_BACK   = ["뒤","후","back","behind","rear"]
K_FL     = ["왼앞","좌전","front-left","front left","front-left"]
K_FR     = ["오른앞","우전","front-right","front right","front-right"]
K_BL     = ["왼뒤","좌후","back-left","back left","back-left"]
K_BR     = ["오른뒤","우후","back-right","back right","back-right"]

K_UP     = ["위","상","천장","overhead","above","ceiling","up"]
K_DOWN   = ["아래","하","바닥","below","down","floor","ground"]

K_NEAR   = ["가까","근접","near","close","close-in","nearby"]
K_MID    = ["중간","중거리","mid","medium","moderate"]
K_FAR    = ["멀","원거리","far","distant"]

K_WIDE   = ["넓","퍼져","둘러싸","감싸","surround","wide","broad","diffuse"]
K_NARROW = ["좁","또렷","집중","타이트","narrow","focused","tight"]
K_WET    = ["잔향","울리","리버브","reverb","wet","echo","hall"]
K_DRY    = ["건조","드라이","dry","dead","anechoic"]
K_LOUD   = ["크게","시끄럽","충만","loud","strong","powerful"]
K_SOFT   = ["작게","조용","부드럽","soft","quiet","gentle"]
K_MUFF   = ["먹먹","차폐","muffled","occluded","behind a door","leaks"]

def any_kw(text, keys):
    t=text.lower()
    return any(k in t for k in keys)

def heur_text2params(text):
    # --- azimuth (deg) ---
    # 기본 front(0°)로 두고 키워드로 보정
    az_deg = 0.0
    # 조합 우선
    if any_kw(text,K_FL): az_deg=-45.0
    elif any_kw(text,K_FR): az_deg= 45.0
    elif any_kw(text,K_BL): az_deg=-135.0
    elif any_kw(text,K_BR): az_deg= 135.0
    else:
        if any_kw(text,K_LEFT): az_deg -= 90.0
        if any_kw(text,K_RIGHT): az_deg += 90.0
        if any_kw(text,K_BACK): az_deg = 180.0 if az_deg==0 else (180.0 if abs(az_deg)<1e-3 else az_deg+0.0)
        if any_kw(text,K_FRONT) and abs(az_deg)>1e-3:
            # front+left/right 조합이면 front 영향은 무시
            pass

    # --- elevation (rad) ---
    el_rad = 0.0
    if any_kw(text,K_UP):   el_rad += math.radians(30.0)
    if any_kw(text,K_DOWN): el_rad -= math.radians(20.0)

    # --- distance (m) ---
    if any_kw(text,K_NEAR): dist=0.9
    elif any_kw(text,K_FAR): dist=4.0
    else: dist=2.0
    if any_kw(text,K_MID): dist=2.0

    # --- spread (deg) ---
    if any_kw(text,K_WIDE): spread=90.0
    elif any_kw(text,K_NARROW): spread=15.0
    else: spread=45.0

    # --- wet (0..1) ---
    wet = 0.15
    if any_kw(text,K_WET): wet = 0.45
    if any_kw(text,K_DRY): wet = 0.05
    if any_kw(text,K_MUFF): wet = max(wet, 0.30)

    # --- gain (dB) ---
    gain = 1.0
    if any_kw(text,K_LOUD): gain = 2.0
    if any_kw(text,K_SOFT): gain = 0.5

    # sc
    az_rad = math.radians(az_deg)
    az_sc = (math.cos(az_rad), math.sin(az_rad))
    return az_sc, el_rad, dist, spread, wet, gain

def load_rows(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for l in f:
            if l.strip():
                rows.append(json.loads(l))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="metrics_rulelex.json")
    args=ap.parse_args()

    rows=load_rows(args.data)
    split=json.load(open(args.split,"r",encoding="utf-8"))
    te_idx=split["test_idx"]

    ae=[]; dlog=[]; m_sp=[]; m_wet=[]; m_gain=[]
    for i in te_idx:
        r=rows[i]
        text=r["text"]
        pr_sc, pr_el, pr_dist, pr_sp, pr_wet, pr_gain = heur_text2params(text)

        gt_sc=r["az_sc"] # [cos,sin]
        gt_el=float(r["el_rad"])
        gt_dist=float(r["dist_m"])
        gt_sp=float(r["spread_deg"])
        gt_wet=float(r["wet_mix"])
        gt_gain=float(r["gain_db"])

        ae.append(ae_deg_from_sc(gt_sc[0],gt_sc[1], pr_sc[0],pr_sc[1]))
        dlog.append(abs(math.log(gt_dist+EPS)-math.log(pr_dist+EPS)))
        m_sp.append(abs(gt_sp-pr_sp))
        m_wet.append(abs(gt_wet-pr_wet))
        m_gain.append(abs(gt_gain-pr_gain))

    M=lambda x: sum(x)/max(1,len(x))
    metrics={"N":len(te_idx), "AE":M(ae), "dlog":M(dlog),
             "spread":M(m_sp), "wet":M(m_wet), "gain":M(m_gain)}
    print(metrics)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)
    print(f"[save] {args.out}")

if __name__=="__main__":
    main()
