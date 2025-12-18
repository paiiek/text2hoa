# save as: augment_text_v3_labelaware.py
import json, re, random, argparse, math
random.seed(777)
DEG=math.pi/180

# --- 언어 판별 (간단) ---
def is_ko(s): return any('\uac00' <= ch <= '\ud7a3' for ch in s)

# --- 안전 동의어 사전 (방향은 기본적으로 보존) ---
KO_VERBS = [
    ("다가오", ["가까워지","근접하","접근하"]),  # 다가오다 원형 포함
    ("맴돌", ["빙빙돌","선회하","돌아다니"]), 
    ("지나가", ["스쳐가","통과하","흘러가"]),
    ("퍼지", ["번지","확산되","넓어지"]),
    ("머무르", ["정지하","그대로있","머물"]),
    ("모이", ["집중되","수렴하","모여들"]),
]
KO_ADVS = [
    ("천천히", ["서서히","느리게","완만하게"]),
    ("빠르게", ["재빨리","급히","신속하게"]),
    ("부드럽게", ["은은하게","잔잔하게","매끄럽게"]),
    ("선명하게", ["또렷하게","뚜렷하게","명확하게"]),
    ("희미하게", ["옅게","약하게","점차적으로"]),
    ("넓게", ["크게","폭넓게","주위를 감싸듯"]),
    ("좁게", ["타이트하게","집중적으로","포커스되게"]),
    ("건조하게", ["드라이하게","메마르게"]),
]
KO_DIR_SAFE = [
    # 안전: 의미 동일(방향 불변)
    ("왼쪽", ["좌측","왼편"]), ("오른쪽", ["우측","오른편"]),
    ("정면", ["앞쪽","전면"]), ("뒤쪽", ["후면","뒤편"]),
    ("머리 위", ["천장 쪽","상부"]), ("바닥", ["아래쪽","하부"])
]

EN_VERBS = [
    ("approach", ["come closer","move nearer","draw closer"]),
    ("circle", ["orbit","loop around","move around"]),
    ("pass", ["sweep past","glide by","move past"]),
    ("spread", ["diffuse","fan out","radiate"]),
    ("linger", ["stay","remain","hover"]),
    ("gather", ["converge","focus","cluster"]),
]
EN_ADVS = [
    ("slowly", ["gradually","gently","steadily"]),
    ("quickly", ["rapidly","swiftly","briskly"]),
    ("softly", ["smoothly","quietly","subtly"]),
    ("clearly", ["distinctly","sharply"]),
    ("faintly", ["dimly","lightly","softly"]),
    ("widely", ["broadly","expansively","all around"]),
    ("narrowly", ["tightly","focused","concentrated"]),
    ("dry", ["unreverberant","plain","dryly"]),
]
EN_DIR_SAFE = [
    ("from the left", ["on the left","at the left side"]),
    ("from the right", ["on the right","at the right side"]),
    ("in front", ["at the front","ahead"]),
    ("from behind", ["at the back","behind"]),
    ("overhead", ["above","from above"]),
    ("near the floor", ["below","down low"]),
]

# 라벨 힌트(옵션) – 수치에 맞춰 보조 수식어 추가
def hint_phrases_ko(el, dist, spread, wet):
    out=[]
    if el>0.35: out.append(random.choice(["머리 위에서","상부에서"]))
    if el<-0.35: out.append(random.choice(["바닥 쪽에서","하부에서"]))
    if dist<1.2: out.append(random.choice(["가까이에서","바로 근처에서"]))
    if dist>4.0: out.append(random.choice(["멀리서","먼 거리에서"]))
    if spread>80: out.append(random.choice(["넓게","주위를 감싸듯"]))
    if spread<20: out.append(random.choice(["좁게","집중적으로"]))
    if wet>0.7: out.append(random.choice(["잔향감 있게","울림이 크게"]))
    if wet<0.2: out.append(random.choice(["건조하게","드라이하게"]))
    return list(dict.fromkeys(out))  # 중복 제거

def hint_phrases_en(el, dist, spread, wet):
    out=[]
    if el>0.35: out.append(random.choice(["overhead","from above"]))
    if el<-0.35: out.append(random.choice(["near the floor","from below"]))
    if dist<1.2: out.append(random.choice(["nearby","very close"]))
    if dist>4.0: out.append(random.choice(["from far away","at a distance"]))
    if spread>80: out.append(random.choice(["widely","surrounding"]))
    if spread<20: out.append(random.choice(["narrowly","focused"]))
    if wet>0.7: out.append(random.choice(["reverberant","echoey"]))
    if wet<0.2: out.append(random.choice(["dry","unreverberant"]))
    return list(dict.fromkeys(out))

def replace_from_table(text, table, safe=True):
    t=text
    for src, cands in table:
        if safe and src not in t: continue
        if random.random()<0.5:  # 50% 확률로 교체 (과도한 치환 방지)
            repl=random.choice(cands)
            t = re.sub(rf"\b{re.escape(src)}\b", repl, t)
    return t

def ko_augment(txt):
    t=txt
    # 동사/부사 치환(방향어는 기본 보존)
    for src,cands in KO_VERBS:
        if random.random()<0.6:
            t=re.sub(src, random.choice(cands), t)
    for src,cands in KO_ADVS:
        if src in t and random.random()<0.6:
            t=t.replace(src, random.choice(cands))
    return t

def en_augment(txt):
    t=txt
    # 기본 소문자/대소문자 보존은 단순화
    for src,cands in EN_VERBS:
        if re.search(rf"\b{src}\w*\b", t, flags=re.I) and random.random()<0.6:
            t=re.sub(rf"\b{src}\w*\b", random.choice(cands), t, flags=re.I)
    for src,cands in EN_ADVS:
        if re.search(rf"\b{src}\b", t, flags=re.I) and random.random()<0.6:
            t=re.sub(rf"\b{src}\b", random.choice(cands), t, flags=re.I)
    return t

def dir_safe_replace(txt, full_dir_syn=False):
    if not full_dir_syn: return txt
    t=txt
    # 같은 방향 내 동의어(의미 동일)
    for src,cands in KO_DIR_SAFE:
        if src in t and random.random()<0.5:
            t=t.replace(src, random.choice(cands))
    for src,cands in EN_DIR_SAFE:
        if re.search(re.escape(src), t, flags=re.I) and random.random()<0.5:
            t=re.sub(re.escape(src), random.choice(cands), t, flags=re.I)
    return t

def build_label_hint(txt, el, dist, spread, wet):
    if is_ko(txt):
        hints=hint_phrases_ko(el,dist,spread,wet)
        if not hints: return ""
        # 문장 말미에 자연스럽게 덧붙이기
        join = " / " if random.random()<0.3 else ", "
        return (" " if not txt.endswith((".", "!", "…")) else "") + join.join(hints)
    else:
        hints=hint_phrases_en(el,dist,spread,wet)
        if not hints: return ""
        join = " / " if random.random()<0.3 else ", "
        return (" " if not txt.endswith((".", "!", "…")) else "") + join.join(hints)

def uniqify(seq):
    seen=set(); out=[]
    for s in seq:
        k=s.strip()
        if k not in seen:
            seen.add(k); out.append(s)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--per_row", type=int, default=1, help="행당 생성할 파라프레이즈 수 (1이면 2배)")
    ap.add_argument("--full_dir_syn", action="store_true", help="방향 단어도 같은 의미로 치환 허용")
    ap.add_argument("--add_label_hints", action="store_true", help="라벨 기반 보조 수식어 추가")
    ap.add_argument("--max_rows", type=int, default=None, help="상한(디버그용)")
    args=ap.parse_args()

    rows=[json.loads(l) for l in open(args.infile,encoding="utf-8")]
    out=[]
    for i,r in enumerate(rows):
        if args.max_rows and i>=args.max_rows: break
        out.append(r)  # 원본 유지
        base_txt=r["text"]
        el=r["el_rad"]; dist=r["dist_m"]; spread=r["spread_deg"]; wet=r["wet_mix"]

        cands=[]
        # 1) 안전한 동의어/패턴 치환
        if is_ko(base_txt):
            for _ in range(args.per_row*2):  # 여유 생성 후 중복 제거
                t=ko_augment(base_txt)
                t=dir_safe_replace(t, full_dir_syn=args.full_dir_syn)
                if args.add_label_hints and random.random()<0.75:
                    t = t + build_label_hint(t, el, dist, spread, wet)
                cands.append(t)
        else:
            for _ in range(args.per_row*2):
                t=en_augment(base_txt)
                t=dir_safe_replace(t, full_dir_syn=args.full_dir_syn)
                if args.add_label_hints and random.random()<0.75:
                    t = t + build_label_hint(t, el, dist, spread, wet)
                # 약한 어순 변형
                if random.random()<0.3:
                    if "," in t: 
                        parts=[p.strip() for p in t.split(",")]
                        random.shuffle(parts)
                        t=", ".join([p for p in parts if p])
                cands.append(t)

        cands=uniqify([c for c in cands if len(c.strip())>0 and c.strip()!=base_txt])
        for t in cands[:args.per_row]:
            rr=dict(r); rr["text"]=t
            out.append(rr)

    with open(args.outfile,"w",encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False)+"\n")
    print(f"wrote {len(out)} to {args.outfile}")

if __name__=="__main__":
    main()
