# -*- coding: utf-8 -*-
# save as: make_dataset_v3_no_subject.py
"""
Text→Spatial Params dataset generator (no subject labels).
Outputs JSONL rows with fields:
  text, lang, az_sc[sin,cos], el_rad, dist_m, spread_deg, wet_mix,
  room_depth{drr_db|rt60_s}, gain_db

Coverage goals:
- Direction (front/back/left/right + diagonals), elevation (up/overhead/down/floor)
- Distance (near/close vs far/distant), clarity (clear vs faint)
- Spread/surround (diffuse/wide/wrap), dry/wet
- Occlusion/material (door/wall/curtain/glass), environment size (hall/cathedral/studio/outdoor)
- Motion intents: circle/orbit/spiral, fall, stay/hover, zigzag/weave, sweep/cross-pan,
  converge/diverge, expand/contract, sway/oscillate, bounce, arc
- Stage/FOH language: stage left/right, house left/right, center/mono, stereo wide
"""

import json, random, argparse, math

# ==== Rebalance priors (default) ====
DIR_PRIOR   = [0.25, 0.25, 0.25, 0.25]   # [front, right, back, left]
ELEV_PRIOR  = [0.30, 0.40, 0.30]         # [low, mid, high]
DIST_PRIOR  = [0.33, 0.34, 0.33]         # [near, mid, far]
SPRD_PRIOR  = [0.30, 0.40, 0.30]         # [low(5-30), mid(30-60), high(60-120)]

def _sample_idx(probs):
    r = random.random(); acc = 0.0
    for i,p in enumerate(probs):
        acc += p
        if r <= acc: return i
    return len(probs)-1


# --------------------------
# 1) Base templates (KO/EN)
# --------------------------
KO_BASE = [
    # 방향/거리/선명도
    "정면 가까이에서 또렷해져", "정면에서 멀어지며 희미해져",
    "왼쪽에서 서서히 가까워져", "오른쪽에서 서서히 멀어져",
    "뒤쪽에서 천천히 다가와", "뒤쪽에서 멀리서 옅게 들려",
    # 대각
    "왼쪽 뒤에서 낮게 울려", "오른쪽 뒤에서 빠르게 스쳐가",
    "왼쪽 앞에서 은은하게 번져", "오른쪽 앞에서 점차 선명해져",
    # 고도
    "머리 위에서 부드럽게 맴돌아", "천장 쪽에서 넓게 퍼져",
    "바닥 근처에서 건조하게 지나가", "아래쪽에서 위로 서서히 올라와",
    # 둘러쌈/확산
    "사방에서 둘러싸듯 번져", "주위를 감싸듯 넓게 퍼져",
    # 차폐/재질
    "문 너머에서 답답하게 새어나와", "두꺼운 벽 뒤에서 먹먹하게 들려",
    "커튼 뒤에서 둔탁하게 번져", "유리창 너머에서 차갑게 울려",
    # 가로질름/패스
    "왼쪽에서 오른쪽으로 가로질러", "뒤에서 앞을 스쳐 지나가",
    # ===== 모션 특화 =====
    "원형으로 머리 위를 한 바퀴 돌아",                   # circular
    "위에서 아래로 곧게 떨어져",                         # falling
    "그 자리에 머무르며 은은하게 지속돼",                 # staying
    "지그재그로 좌우를 오가",                            # zigzag
    "주위를 크게 한 바퀴 돌며 가까워져",                  # circular + near
    "천천히 모여들어 한 점으로",                         # converge
    "먼 곳에서 퍼졌다가 점차 모여들어",                    # diverge→converge
    "위에서 아래로 나선형으로 내려와",                    # spiral down
    "아래에서 위로 나선형으로 올라가",                    # spiral up
    "좌우로 흔들리며 오고 가",                           # sway/oscillate
    "위아래로 튀듯이 움직여",                            # bounce
    "넓게 퍼졌다가 천천히 좁혀져",                        # expand→contract
    "왼쪽 앞에서 오른쪽 뒤로 호를 그리며 지나가"            # arc pass
]

EN_BASE = [
    # direction/distance/clarity
    "becomes clear close in front", "drifts farther ahead and fades",
    "approaches slowly from the left", "recedes slowly to the right",
    "approaches slowly from the back", "from the back, faint and distant",
    # diagonals
    "low rumble from the back-left", "passes swiftly from the back-right",
    "softly spreading from the front-left", "growing distinct at the front-right",
    # elevation
    "circles softly overhead", "spreads broadly from above",
    "passes dry near the floor", "rises slowly from below",
    # surround/spread
    "diffuses as if surrounding", "widens as if wrapping around",
    # occlusion/material
    "leaks muffled from behind a door", "damped behind a thick wall",
    "smears behind a curtain", "glassy and distant behind a window",
    # cross-pass
    "cuts across left to right", "sweeps forward from behind",
    # ===== motion-specific =====
    "orbits all around overhead",                 # circular
    "falls straight down from above",             # falling
    "stays in place softly",                      # staying
    "moves in a zigzag left and right",           # zigzag
    "circles widely and comes closer",            # circular + near
    "converges gradually into a single point",    # converge
    "spreads out first, then gathers back",       # diverge→converge
    "spirals downward from above",                # spiral down
    "spirals upward from below",                  # spiral up
    "sways side to side as it moves",             # sway
    "bounces up and down lightly",                # bounce
    "expands wide then narrows down",             # expand→contract
    "arcs from front-left to back-right"          # arc pass
]

# Stage/FOH user language (treated as additional templates)
KO_BASE += [
    "무대 왼쪽에서 중앙으로 천천히 이동해",  # stage left → left
    "무대 오른쪽에서 바깥으로 퍼져",          # stage right → right
    "하우스 왼쪽에서 둘러싸듯 넓어져",        # house left → audience left
    "센터에서 모노로 또렷하게",               # center / mono / clear
    "스테레오로 아주 넓게 퍼져"               # stereo wide
]
EN_BASE += [
    "from stage left moving to center slowly",
    "from stage right spreading outward",
    "from house left widening around",
    "centered mono and distinct",
    "stereo image spreads very wide"
]

# --------------------------
# 2) Synonym expansion maps
# --------------------------
KO_SYNS = {
    # speed/change
    "천천히": ["서서히","느리게"], "빠르게": ["급히","재빨리","휙"],
    "점차": ["차츰","조금씩","서서히"], "스쳐": ["휙","스치듯","순간적으로"],
    # clarity/intensity
    "또렷": ["선명","뚜렷","확실"], "희미": ["옅","약하","어렴풋"],
    # spread/surround
    "넓게": ["크게","광범위하게","와이드하게"], "둘러싸": ["감싸","주위를 채우"],
    "번져": ["퍼져","확장되","퍼지듯"], "좁혀": ["모아지","좁아지","수축되"],
    # occlusion/material
    "먹먹": ["답답","둔탁","막힌"], "문 너머": ["문 뒤","문 건너편"],
    "벽 뒤": ["두꺼운 벽 뒤","벽 건너편"], "커튼": ["천 뒤","두꺼운 천"],
    "유리": ["유리창","창문"],
    # elevation
    "머리 위": ["윗쪽","위쪽","머리 바로 위"], "천장": ["천정","위 천장"],
    "바닥": ["아래쪽","발치","바닥쪽"],
    # direction
    "왼쪽": ["좌측","왼편"], "오른쪽": ["우측","오른편"],
    "정면": ["앞쪽","앞으로"], "뒤쪽": ["후방","뒤로"],
    # motion
    "원형": ["원을 그리","둥글게 돌","한 바퀴 돌"],
    "떨어져": ["낙하","내려와","아래로 내려가"],
    "머무르": ["정지하","고정되","자리 지켜","가만히"],
    "지그재그": ["좌우로 흔들리","스윙하","지그재그로"],
    "모여들": ["수렴하","한 점으로 모이","집중되"],
    "나선형": ["스파이럴","회오리처럼"]  # spiral hint
}

EN_SYNS = {
    # speed/change
    "slowly": ["gradually","gently"], "quickly": ["rapidly","swiftly"],
    "passes": ["sweeps","glides","slides"], "fades": ["dims","weakens","ebbs"],
    "becomes clear": ["grows distinct","sharpens","gets clearer"],
    # spread/surround
    "spreads": ["widens","broadens","expands"], "diffuses": ["disperses","scatters"],
    "surrounding": ["around","encircling","all around","wrapping around"],
    "narrows": ["contracts","tightens"],
    # occlusion/material
    "muffled": ["damped","occluded","blocked"], "behind a door": ["through a door","past a door"],
    "behind a wall": ["through a wall","past a wall"], "behind a curtain": ["through a curtain"],
    "behind a window": ["through glass","past a window"],
    # elevation
    "overhead": ["above","up high"], "near the floor": ["down low","at floor level"],
    "from below": ["rising up","low and rising"],
    # direction
    "from the left": ["leftward","off to the left"], "to the right": ["rightward"],
    "from the back": ["from behind","backward"], "in front": ["ahead","forward"],
    # motion
    "orbits": ["circles","revolves","goes around","swirls"],
    "falls": ["drops","descends"],
    "stays": ["remains","holds still","lingers","hovers"],
    "zigzag": ["zig-zag","in a zigzag","weaves"],
    "converges": ["gathers","focuses","narrows down"],
    "spirals": ["helical","corkscrews"]
}

# --------------------------
# 3) Detection keyword sets (KO/EN, mixed)
# --------------------------
def _any(t, keys): return any(k in t for k in keys)

# directions
DIR_RIGHT   = {"오른쪽","우측","오른편","right","to the right","from the right","stage right","house right"}
DIR_LEFT    = {"왼쪽","좌측","왼편","left","from the left","stage left","house left"}
DIR_BACK    = {"뒤","후방","뒤쪽","behind","from the back","from behind","backward"}
DIR_FRONT   = {"정면","앞","앞쪽","in front","ahead","forward","center"}
DIAG_BL     = {"왼쪽 뒤","뒤 왼쪽","back-left"}
DIAG_BR     = {"오른쪽 뒤","뒤 오른쪽","back-right"}
CENTERED    = {"센터","centered","mono","center"}  # center/mono hint

# elevation
ELEV_UP     = {"머리 위","위쪽","윗쪽","천장","천정","overhead","above","up high","ceiling"}
ELEV_DOWN   = {"바닥","아래쪽","발치","near the floor","down low","at floor level","from below","underfoot"}

# distance / clarity
NEAR_KEYS   = {"가까이","근처","바로 앞","near","close","intimate"}
CLEAR_KEYS  = {"또렷","선명","뚜렷","clear","distinct","sharp"}
# FAINT_KEYS 확장 (파일 상단 집합 정의 부분)
FAINT_KEYS  = {
    "희미","옅","약하","faint","dim","weak","fade","fades","weakens",
    "distant","far away","멀리","멀리서","먼 곳","아득히"
}


# spread/surround
SPREAD_KEYS = {"넓게","와이드","둘러싸","감싸","퍼져","번져","surround","around","diffuse","spreads","widens","broadens","expands","wrapping"}

# occlusion/material
OCCL_KEYS   = {"문 너머","문 뒤","벽 뒤","두꺼운 벽","커튼","천 뒤","유리","유리창",
               "behind a door","behind a wall","behind a curtain","behind a window","occluded","muffled","damped","blocked"}

# dryness
DRY_KEYS    = {"건조","드라이","dry","plain","unreverberant"}

# stage/FOH hints
FOH_LEFT    = {"stage left","house left"}
FOH_RIGHT   = {"stage right","house right"}

# environments
ENV_BIG     = {"홀","대성당","교회","강당","체육관","터널","동굴","지하철","아트리움",
               "hall","cathedral","church","auditorium","gym","tunnel","cave","subway","atrium"}
ENV_SMALL   = {"스튜디오","부스","작은 방","방음실","클로젯",
               "studio","booth","small room","isolation booth","closet"}
ENV_OUTDOOR = {"야외","밖","노천","outdoor","outside","open air"}

# motions
M_CIRCLE    = {"원형","원을 그리","둥글게","한 바퀴","orbits","circles","revolves","goes around","swirls"}
M_SPIRAL    = {"나선형","스파이럴","회오리","spiral","helical","corkscrew"}
M_FALL      = {"떨어져","낙하","내려와","falls","drops","descends"}
M_STAY      = {"머무르","정지","고정","자리 지켜","stays","remains","holds still","lingers","hovers"}
M_ZIGZAG    = {"지그재그","zigzag","zig-zag","weaves"}
M_SWAY      = {"흔들리","스윙","sways","oscillates","wobbles"}
M_SWEEP     = {"스쳐","가로질러","지나가","cuts across","sweeps","passes","glides","slides"}
M_CONVERGE  = {"모여들","수렴","한 점으로","집중","converges","gathers","focuses","narrows down"}
M_DIVERGE   = {"퍼졌다가","흩어지","spreads out","scatters"}
M_EXPAND    = {"넓어지","확장되","expands","widens","broadens"}
M_CONTRACT  = {"좁혀","수축","narrows","contracts","tightens"}
M_BOUNCE    = {"튀듯","튀어","bounces","hops","bounds"}
M_ARC       = {"호를 그리","arc","arcs"}

# --------------------------
# 4) helpers
# --------------------------
def randf(a, b): return random.uniform(a, b)

def compute_room(room_mode:str, wet:float, near:bool, big_env:bool, small_env:bool, outdoor:bool, occluded:bool):
    """
    Returns dict: {"drr_db": ..} or {"rt60_s": ..}
    Heuristics:
      - near/clear → DRR+, small RT60; far/occluded/big → DRR-, big RT60
      - outdoor → low RT60 or DRR+
    """
    if room_mode == "drr":
        base = 0.0
        if near: base += randf(+4, +10)
        if big_env: base += randf(-10, -4)
        if small_env: base += randf(+2, +8)
        if outdoor: base += randf(+2, +8)
        if occluded: base += randf(-6, -2)
        # wet correlation: more wet → slightly lower DRR
        base += (wet - 0.5) * -6.0
        return {"drr_db": max(-12.0, min(12.0, base))}
    else:
        # rt60 baseline
        base = randf(0.5, 1.4)
        if near: base -= randf(0.1, 0.4)
        if big_env: base += randf(0.6, 1.0)
        if small_env: base -= randf(0.2, 0.4)
        if outdoor: base -= randf(0.2, 0.5)
        if occluded: base += randf(0.1, 0.4)
        base = max(0.3, min(2.5, base))
        return {"rt60_s": round(base, 2)}

# --------------------------
# 5) label mapping (KO/EN)
# --------------------------
def map_common(text:str):
    # Direction
    has_dir = False
    s, c = 0.0, 1.0
    if _any(text, DIAG_BL): s, c = -0.707107, -0.707107; has_dir=True
    elif _any(text, DIAG_BR): s, c =  0.707107, -0.707107; has_dir=True
    elif _any(text, DIR_RIGHT) or _any(text, FOH_RIGHT): s, c = 1.0, 0.0; has_dir=True
    elif _any(text, DIR_LEFT)  or _any(text, FOH_LEFT):  s, c = -1.0, 0.0; has_dir=True
    elif _any(text, DIR_BACK):  s, c = 0.0, -1.0; has_dir=True
    elif _any(text, CENTERED):  s, c = 0.0, 1.0; has_dir=True  # center front

    # if direction ambiguous → sample quadrant by prior
    if not has_dir:
        q = _sample_idx(DIR_PRIOR)  # 0:F,1:R,2:B,3:L
        if   q==0: s,c = 0.0, 1.0
        elif q==1: s,c = 1.0, 0.0
        elif q==2: s,c = 0.0,-1.0
        else:      s,c =-1.0, 0.0

    # Elevation
    has_el = False
    el = 0.0
    if _any(text, ELEV_UP):   el = randf(0.6, 1.2); has_el=True
    if _any(text, ELEV_DOWN): el = randf(-0.8, -0.4); has_el=True
    if not has_el:
        ebin = _sample_idx(ELEV_PRIOR)  # 0:low 1:mid 2:high
        if   ebin==0: el = randf(-0.7,-0.3)
        elif ebin==1: el = 0.0
        else:         el = randf(0.6, 1.0)

    # Distance
    has_dist_cue = _any(text, NEAR_KEYS) or _any(text, CLEAR_KEYS) or _any(text, FAINT_KEYS)
    is_near = _any(text, NEAR_KEYS) or _any(text, CLEAR_KEYS)
    if is_near: dist = randf(0.6, 1.5)
    elif _any(text, FAINT_KEYS): dist = randf(3.0, 6.0)
    else:
        dbin = _sample_idx(DIST_PRIOR)  # 0:near 1:mid 2:far
        dist = randf(0.6,1.5) if dbin==0 else (randf(1.5,3.0) if dbin==1 else randf(3.0,6.0))

    # Spread/Wet/Gain
    # === 기존 spread 계산 블록을 아래로 교체 ===
    has_spread_cue = _any(text, SPREAD_KEYS)
    if has_spread_cue:
        # 확산 키워드가 있어도 mid 70%, low 20%, high 10%
        z = random.random()
        if z < 0.70:
            spread = randf(30,60)     # mid
        elif z < 0.90:
            spread = randf(8,30)      # low
        else:
            spread = randf(60,95)     # high (상한 ↓)
    else:
        sbin = _sample_idx(SPRD_PRIOR)  # 0:low 1:mid 2:high
        if   sbin==0: spread = randf(8,30)
        elif sbin==1: spread = randf(30,60)
        else:         spread = randf(60,95)  # 상한을 95°로 낮춤

    # (옵션) 글로벌 리밸런서: high가 과하면 일부를 mid로 다운시프트
    if spread > 60 and random.random() < 0.35:
        spread = randf(30,60)

        
    wet    = 0.8 if has_spread_cue or _any(text, ELEV_UP) else randf(0.1, 0.6)
    if _any(text, DRY_KEYS): wet = min(wet, randf(0.05, 0.25))
    gain   = randf(-10, -3) if _any(text, FAINT_KEYS) else randf(-6, 0)

    # Environment flags for room
    big_env   = _any(text, ENV_BIG)
    small_env = _any(text, ENV_SMALL)
    outdoor   = _any(text, ENV_OUTDOOR)
    occluded  = _any(text, OCCL_KEYS)   # << 오타 수정: OCCL_KEYS

    return (s, c, el, dist, spread, wet, gain, is_near, big_env, small_env, outdoor, occluded)

def apply_motion_adjustments(text:str, s,c, el, dist, spread, wet, gain):
    # Circle/orbit → elevate if none, spread↑, wet↑
    if _any(text, M_CIRCLE):
        if abs(el) < 1e-6: el = randf(0.5, 1.0)
        spread = max(spread, randf(55, 85))   # (기존 80~115 → 55~85)
        wet    = max(wet, randf(0.5, 0.85))


    # Spiral → like circle but directional el drift
    if _any(text, M_SPIRAL):
        spread = max(spread, randf(60, 90))   # (기존 85~115 → 60~90)
        wet    = max(wet, randf(0.5, 0.85))
        # push el slightly up or down if not specified
        if "down" in text or "아래" in text:
            el = randf(-0.7, -0.2)
        elif "up" in text or "위" in text:
            el = randf(0.5, 1.0)
        else:
            el = el if abs(el)>0 else randf(0.2, 0.8) * (1 if random.random()<0.5 else -1)

    # Fall
    if _any(text, M_FALL):
        el = randf(-0.7, -0.2)
        wet = max(wet, randf(0.35, 0.7))

    # Stay/hover
    if _any(text, M_STAY):
        spread = min(spread, randf(8, 20))
        wet = min(wet, randf(0.1, 0.4))
        dist = min(dist, randf(0.6, 2.0))

    # Zigzag/weave → side emphasis, spread↑, wet mid
    if _any(text, M_ZIGZAG):
        spread = max(spread, randf(45, 80))   # (기존 70~110 → 45~80)
        if random.random() < 0.5: s, c = 1.0, 0.0
        else: s, c = -1.0, 0.0
        wet = max(wet, randf(0.25, 0.65))

    # Sway/oscillate
    if _any(text, M_SWAY):
        spread = max(spread, randf(35, 75))   # (기존 60~100 → 35~75)
        wet = max(wet, randf(0.25, 0.6))


    # Sweep / cross-pan
    if _any(text, M_SWEEP):
        # choose an entry/exit side, single-frame snapshot picks one side
        if "left to right" in text or "왼쪽에서 오른쪽" in text:
            s, c = -1.0, 0.0
        elif "right to left" in text or "오른쪽에서 왼쪽" in text:
            s, c = 1.0, 0.0
        spread = max(spread, randf(40, 90))

    # Converge / Diverge / Expand / Contract
    if _any(text, M_CONVERGE):
        spread = min(spread, randf(8, 18))
        wet = min(wet, randf(0.1, 0.35))
        dist = min(dist, randf(0.6, 2.5))
        gain = max(gain, randf(-4, -1))
    if _any(text, M_DIVERGE) or _any(text, M_EXPAND):
        spread = max(spread, randf(55, 90))   # (기존 80~115 → 55~90)
    if _any(text, M_CONTRACT):
        spread = min(spread, randf(8, 20))

    # Bounce
    if _any(text, M_BOUNCE):
        # snapshot: choose mid el, keep wet moderate
        el = randf(-0.2, 0.4)
        wet = max(wet, randf(0.25, 0.6))

    # Arc
    if _any(text, M_ARC):
        spread = max(spread, randf(40, 90))
        wet = max(wet, randf(0.3, 0.7))

    # FOH stereo hints
    if "스테레오" in text or "stereo" in text:
        spread = max(spread, randf(80, 115))
    if "모노" in text or "mono" in text:
        spread = min(spread, randf(5, 15))

    return s, c, el, dist, spread, wet, gain

def ko_to_labels(text):
    s, c, el, dist, spread, wet, gain, is_near, big_env, small_env, outdoor, occluded = map_common(text)
    s, c, el, dist, spread, wet, gain = apply_motion_adjustments(text, s,c,el,dist,spread,wet,gain)
    return (s,c, el, dist, spread, wet, gain, is_near, big_env, small_env, outdoor, occluded)

def en_to_labels(text):
    s, c, el, dist, spread, wet, gain, is_near, big_env, small_env, outdoor, occluded = map_common(text)
    s, c, el, dist, spread, wet, gain = apply_motion_adjustments(text, s,c,el,dist,spread,wet,gain)
    return (s,c, el, dist, spread, wet, gain, is_near, big_env, small_env, outdoor, occluded)

# --------------------------
# 6) expansion via synonyms
# --------------------------
def expand(base, syns, mult):
    out=[]
    for b in base:
        cand=[b]
        for k,vs in syns.items():
            if k in b:
                for v in vs: cand.append(b.replace(k, v))
        random.shuffle(cand)
        out.extend(cand[:mult])
    return out

# --------------------------
# 7) main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="text2spatial_v3_aug.jsonl")
    ap.add_argument("--ko_mult", type=int, default=12)
    ap.add_argument("--en_mult", type=int, default=12)
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr", help="room_depth single axis")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--prior_dir",  default="0.25,0.25,0.25,0.25",
                help="front,right,back,left priors for ambiguous texts")
    ap.add_argument("--prior_el",   default="0.30,0.40,0.30",
                    help="low,mid,high elevation priors for ambiguous")
    ap.add_argument("--prior_dist", default="0.33,0.34,0.33",
                    help="near,mid,far priors for ambiguous")
    ap.add_argument("--prior_spread", default="0.30,0.40,0.30",
                    help="low(5-30),mid(30-60),high(60-120) priors for ambiguous")

    args = ap.parse_args()
    
    # parse priors into globals
    global DIR_PRIOR, ELEV_PRIOR, DIST_PRIOR, SPRD_PRIOR
    DIR_PRIOR  = [float(x) for x in args.prior_dir.split(",")]
    ELEV_PRIOR = [float(x) for x in args.prior_el.split(",")]
    DIST_PRIOR = [float(x) for x in args.prior_dist.split(",")]
    SPRD_PRIOR = [float(x) for x in args.prior_spread.split(",")]
    # normalize (safety)
    def _norm(v): s=sum(v); return [x/s for x in v]
    DIR_PRIOR, ELEV_PRIOR, DIST_PRIOR, SPRD_PRIOR = map(_norm, [DIR_PRIOR,ELEV_PRIOR,DIST_PRIOR,SPRD_PRIOR])

    random.seed(args.seed)

    ko_texts = expand(KO_BASE, KO_SYNS, args.ko_mult)
    en_texts = expand(EN_BASE, EN_SYNS, args.en_mult)

    rows=[]
    # KO
    for txt in ko_texts:
        s,c,el,dist,spread,wet,gain,is_near,big_env,small_env,outdoor,occluded = ko_to_labels(txt)
        room = compute_room(args.room_mode, wet, is_near, big_env, small_env, outdoor, occluded)
        rows.append({
            "text": txt.strip(), "lang":"ko",
            "az_sc": [round(s,6), round(c,6)], "el_rad": round(el,6),
            "dist_m": round(dist,3), "spread_deg": round(spread,1),
            "wet_mix": round(wet,3), "room_depth": room,
            "gain_db": round(gain,2)
        })
    # EN
    for txt in en_texts:
        s,c,el,dist,spread,wet,gain,is_near,big_env,small_env,outdoor,occluded = en_to_labels(txt)
        room = compute_room(args.room_mode, wet, is_near, big_env, small_env, outdoor, occluded)
        rows.append({
            "text": txt.strip(), "lang":"en",
            "az_sc": [round(s,6), round(c,6)], "el_rad": round(el,6),
            "dist_m": round(dist,3), "spread_deg": round(spread,1),
            "wet_mix": round(wet,3), "room_depth": room,
            "gain_db": round(gain,2)
        })

    random.shuffle(rows)
    with open(args.out,"w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    print("wrote", len(rows), "lines to", args.out)

if __name__ == "__main__":
    main()
