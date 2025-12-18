# -*- coding: utf-8 -*-
# make OOD test sets: metricable(라벨有) + qualitative(라벨無)
# 출력: ood_metric_v1.jsonl, ood_qual_v1.jsonl
import json, math, random, argparse

random.seed(1234)

def sc_from_deg(deg):
    r = math.radians(deg % 360.0)
    return math.sin(r), math.cos(r)

def clamp(x,a,b): return max(a, min(b, x))

def u(a,b): return random.uniform(a,b)

def mk_item(text, az_deg, el_rad, dist_m, spread_deg, wet, gain_db, drr_db, lang, cat, w=1.0):
    s,c = sc_from_deg(az_deg)
    return {
        "text": text, "lang": lang, "cat": cat, "w": w,
        "az_sc": [s, c],
        "el_rad": float(el_rad),
        "dist_m": float(dist_m),
        "spread_deg": float(spread_deg),
        "wet_mix": float(wet),
        "gain_db": float(gain_db),
        "room_depth": {"drr_db": float(drr_db)}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_metric", default="ood_metric_v1.jsonl")
    ap.add_argument("--out_qual", default="ood_qual_v1.jsonl")
    ap.add_argument("--n_each", type=int, default=20, help="각 카테고리당 개수")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    random.seed(args.seed)

    metric_rows = []
    qual_rows = []

    # ---- 1) Long (긴 문장) ----
    KO_LONG = [
      "정면보다 약간 오른쪽에서, 귀 높이보다 살짝 위쪽으로, 매우 좁은 폭으로 또렷하게 고정되어 들려",
      "왼쪽 앞 대각에서 중간 거리로 다가오다가, 방 중앙에 이르면 서서히 퍼지며 잔향이 늘어나",
      "머리 위 약간 앞쪽에서 시작해, 아래로 천천히 떨어지면서 건조하게 사라져"
    ]
    EN_LONG = [
      "From slightly to the right of the front and just above ear level, fixed and razor-thin in focus.",
      "Approaching from the front-left diagonal at mid distance, then broadening near the center with increasing reverb.",
      "Starting overhead slightly in front, slowly descending while remaining dry and fading out."
    ]
    for _ in range(args.n_each):
        t = random.choice(KO_LONG); lang="ko"; cat="long"
        if "오른쪽" in t:
            az = u(15, 45); el = u(0.05, 0.18); spread = u(5, 15); dist = u(1.0, 2.0); wet = u(0.05, 0.20)
        elif "왼쪽 앞" in t:
            az = u(315, 345); el = u(-0.05, 0.10); spread = u(35, 65); dist = u(1.5, 3.0); wet = u(0.25, 0.55)
        else: # overhead fall
            az = u(350, 10); el = u(0.15, 0.30); spread = u(10, 30); dist = u(1.0, 2.0); wet = u(0.05, 0.20)
        metric_rows.append(mk_item(t, az, el, dist, spread, wet, u(-1.5,1.5), u(-4,4), lang, cat))

        t = random.choice(EN_LONG); lang="en"
        if "right of the front" in t:
            az = u(15, 45); el = u(0.05, 0.18); spread = u(5, 15); dist = u(1.0, 2.0); wet = u(0.05, 0.20)
        elif "front-left diagonal" in t:
            az = u(315, 345); el = u(-0.05, 0.10); spread = u(35, 65); dist = u(1.5, 3.0); wet = u(0.25, 0.55)
        else:
            az = u(350, 10); el = u(0.15, 0.30); spread = u(10, 30); dist = u(1.0, 2.0); wet = u(0.05, 0.20)
        metric_rows.append(mk_item(t, az, el, dist, spread, wet, u(-1.5,1.5), u(-4,4), lang, cat))

    # ---- 2) Metaphor(은유) ----
    KO_META = [
      "비처럼 위에서 내려 꽂히는 느낌", "커튼 뒤에서 새어 나오는 소리",
      "구름처럼 뒤편에서 퍼져 감싸는 느낌", "스포트라이트처럼 정면에서 가늘게"
    ]
    EN_META = [
      "Raining down from above.", "Leaking through a curtain behind.",
      "Cloud-like diffusion enveloping from the back.", "Razor-thin spotlight from the front."
    ]
    for _ in range(args.n_each):
        t = random.choice(KO_META); lang="ko"; cat="metaphor"
        if "비처럼" in t:        az=u(340,20); el=u(0.12,0.28); spread=u(15,35); wet=u(0.25,0.55); dist=u(1.2,2.5)
        elif "커튼" in t:        az=u(120,150); el=u(-0.05,0.05); spread=u(20,45); wet=u(0.30,0.60); dist=u(1.5,3.0)
        elif "구름" in t:        az=u(150,210); el=u(-0.05,0.10); spread=u(45,80); wet=u(0.30,0.60); dist=u(2.0,3.5)
        else:                    az=u(350,10);  el=u(-0.05,0.05); spread=u(5,12);  wet=u(0.05,0.20); dist=u(1.0,2.0)
        metric_rows.append(mk_item(t, az, el, dist, spread, wet, u(-2,2), u(-6,2), lang, cat))

        t = random.choice(EN_META); lang="en"
        if "Raining" in t:       az=u(340,20); el=u(0.12,0.28); spread=u(15,35); wet=u(0.25,0.55); dist=u(1.2,2.5)
        elif "curtain" in t:     az=u(120,150); el=u(-0.05,0.05); spread=u(20,45); wet=u(0.30,0.60); dist=u(1.5,3.0)
        elif "Cloud-like" in t:  az=u(150,210); el=u(-0.05,0.10); spread=u(45,80); wet=u(0.30,0.60); dist=u(2.0,3.5)
        else:                    az=u(350,10);  el=u(-0.05,0.05); spread=u(5,12);  wet=u(0.05,0.20); dist=u(1.0,2.0)
        metric_rows.append(mk_item(t, az, el, dist, spread, wet, u(-2,2), u(-6,2), lang, cat))

    # ---- 3) Numeric mix(수치 힌트 섞임) ----
    KO_NUM = [
      "정면에서 약 30도 오른쪽, 귀 높이, 폭은 좁게, 리버브 20%",
      "뒤-왼쪽 대각(약 225도), 약간 높은 고도, 중간 폭, 웻 45%"
    ]
    EN_NUM = [
      "Front with ~30° right bias, ear-level, narrow width, ~20% wet.",
      "Back-left diagonal (~225°), slightly elevated, medium spread, 45% wet."
    ]
    for _ in range(args.n_each):
        t=random.choice(KO_NUM); lang="ko"; cat="numeric"
        if "30도" in t:          az=u(20,40); el=u(-0.02,0.05); spread=u(5,20);  wet=u(0.18,0.25); dist=u(1.0,2.0)
        else:                    az=u(215,235);el=u(0.05,0.15); spread=u(30,60); wet=u(0.40,0.50); dist=u(1.5,3.0)
        metric_rows.append(mk_item(t, az, el, dist, spread, wet, u(-1,1), u(-5,3), lang, cat))

        t=random.choice(EN_NUM); lang="en"
        if "30°" in t:           az=u(20,40); el=u(-0.02,0.05); spread=u(5,20);  wet=u(0.18,0.25); dist=u(1.0,2.0)
        else:                    az=u(215,235);el=u(0.05,0.15); spread=u(30,60); wet=u(0.40,0.50); dist=u(1.5,3.0)
        metric_rows.append(mk_item(t, az, el, dist, spread, wet, u(-1,1), u(-5,3), lang, cat))

    # ---- 4) Paraphrase(의미 유지 재표현) ----
    KO_PARA = [
      "정면 가까이에서 또렷하게", "앞쪽 근거리에서 선명하게 고정",
      "뒤-오른쪽 대각에서 존재감 있게, 약간 퍼져"
    ]
    EN_PARA = [
      "Clear and upfront at close range.",
      "Fixed and crisp from the front near-field.",
      "Present at the back-right diagonal with slight diffusion."
    ]
    for _ in range(args.n_each):
        t=random.choice(KO_PARA); lang="ko"; cat="paraphrase"
        if "정면" in t or "앞쪽" in t:
            az=u(350,10); el=u(-0.05,0.05); spread=u(5,18); dist=u(0.8,1.6); wet=u(0.05,0.20)
        else:
            az=u(125,145); el=u(-0.05,0.08); spread=u(20,40); dist=u(1.2,2.5); wet=u(0.15,0.35)
        metric_rows.append(mk_item(t, az, el, dist, spread, wet, u(-1.5,1.5), u(-4,4), lang, cat))

        t=random.choice(EN_PARA); lang="en"
        if "front" in t or "upfront" in t or "near-field" in t:
            az=u(350,10); el=u(-0.05,0.05); spread=u(5,18); dist=u(0.8,1.6); wet=u(0.05,0.20)
        else:
            az=u(125,145); el=u(-0.05,0.08); spread=u(20,40); dist=u(1.2,2.5); wet=u(0.15,0.35)
        metric_rows.append(mk_item(t, az, el, dist, spread, wet, u(-1.5,1.5), u(-4,4), lang, cat))

    # ---- Qual-only(모순·애매) ----
    QUAL = [
      ("넓지만 아주 타이트하게 동시에", "ko"),
      ("뒤와 앞에서 동시에 들려", "ko"),
      ("Wide yet extremely tight at the same time.", "en"),
      ("Both in front and behind simultaneously.", "en"),
      ("지그재그로 빠르게이면서 느리게 이동", "ko"),
      ("Rapid and slow motion at once, zig-zag path.", "en"),
    ]
    for t,lang in QUAL:
        qual_rows.append({"text":t, "lang":lang, "cat":"contradiction_or_ambiguous"})

    with open(args.out_metric,"w",encoding="utf-8") as f:
        for r in metric_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(args.out_qual,"w",encoding="utf-8") as f:
        for r in qual_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print("wrote", len(metric_rows), "→", args.out_metric)
    print("wrote", len(qual_rows), "→", args.out_qual)

if __name__=="__main__": main()
