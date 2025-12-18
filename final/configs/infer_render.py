#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_render.py — per-waypoint re-infer + param curves + true stereo-follow reverb + scalers
"""

import os, re, math, argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoTokenizer, AutoModel

EPS = 1e-8

# ---------- utils ----------
def rad2deg(x): return x*180.0/math.pi
def deg2rad(x): return x*math.pi/180.0
def hardclip(x): return torch.clamp(x, -1.0, 1.0)
def db_to_lin(db): return 10.0**(db/20.0)
def angle_wrap_deg(a): return (a + 180.0) % 360.0 - 180.0

def match_len_pad(a, b):
    Ta, Tb = a.shape[-1], b.shape[-1]
    if Ta == Tb: return a, b
    if Ta < Tb:  a = F.pad(a, (0, Tb - Ta))
    else:        b = F.pad(b, (0, Ta - Tb))
    return a, b

def fractional_delay(x, delay_samp):
    T = x.shape[-1]
    n = torch.arange(T, device=x.device).float()
    src = n - delay_samp
    n0 = torch.floor(src).long()
    a  = (src - n0.float()).clamp(0,1)
    x0 = torch.where((n0>=0)&(n0<T), x[..., n0.clamp(0,T-1)], torch.zeros_like(x))
    x1 = torch.where((n0+1>=0)&(n0+1<T), x[..., (n0+1).clamp(0,T-1)], torch.zeros_like(x))
    return (1-a)*x0 + a*x1

def fft_convolve(x, h):
    h = h.to(x.device, dtype=x.dtype)
    C, T = x.shape
    L = h.shape[-1]
    N = 1
    while N < T + L - 1: N <<= 1
    X = torch.fft.rfft(x, n=N)
    H = torch.fft.rfft(h.expand(C, -1), n=N)
    Y = X * H
    y = torch.fft.irfft(Y, n=N)[..., :T + L - 1]
    return y

def synth_room_ir_stereo(fs, rt60=0.6, pre_delay_ms=10.0, drr_db=0.0, decor_ms=3.0, device="cpu", dtype=torch.float32):
    preL = int(round((pre_delay_ms - decor_ms*0.5)*1e-3*fs))
    preR = int(round((pre_delay_ms + decor_ms*0.5)*1e-3*fs))
    preL = max(0, preL); preR = max(0, preR)

    T60 = max(rt60, 0.1)
    dur = min(3.0, T60*1.5)
    N   = int(dur*fs)

    tailL = torch.randn(N, device=device, dtype=dtype).unsqueeze(0)
    tailR = torch.randn(N, device=device, dtype=dtype).unsqueeze(0)
    t = torch.arange(N, device=device, dtype=dtype)/fs
    decay = torch.exp(-6.91*t/T60)
    tailL *= decay; tailR *= decay

    kernel = torch.tensor([[[-0.2,0.6,0.6,-0.2]]], device=device, dtype=dtype)
    tailL = F.conv1d(tailL.unsqueeze(0), kernel, padding=0).squeeze(0)
    tailR = F.conv1d(tailR.unsqueeze(0), kernel, padding=0).squeeze(0)

    er_amp = [0.6, 0.4, 0.3]
    er_offL_ms = [7.0, 11.0, 17.0]
    er_offR_ms = [8.5, 13.0, 19.0]
    max_pre = max(preL, preR)
    padL = torch.zeros(1, max_pre+1, device=device, dtype=dtype)
    padR = torch.zeros(1, max_pre+1, device=device, dtype=dtype)
    padL[0, preL] = db_to_lin(drr_db); padR[0, preR] = db_to_lin(drr_db)
    for a, oL, oR in zip(er_amp, er_offL_ms, er_offR_ms):
        nL = int(round((pre_delay_ms + oL)*1e-3*fs))
        nR = int(round((pre_delay_ms + oR)*1e-3*fs))
        if nL >= padL.shape[-1]:
            padL = F.pad(padL, (0, nL - padL.shape[-1] + 1))
        if nR >= padR.shape[-1]:
            padR = F.pad(padR, (0, nR - padR.shape[-1] + 1))
        padL[0, nL] += a*0.25
        padR[0, nR] += a*0.25

    irL = torch.cat([padL, tailL], dim=1)
    irR = torch.cat([padR, tailR], dim=1)
    L = max(irL.shape[-1], irR.shape[-1])
    if irL.shape[-1] < L: irL = F.pad(irL, (0, L - irL.shape[-1]))
    if irR.shape[-1] < L: irR = F.pad(irR, (0, L - irR.shape[-1]))
    ir = torch.cat([irL, irR], dim=0)
    ir = ir / (ir.abs().amax()+1e-6)
    return ir

def distance_attenuation(dist_m, ref=1.0):
    return ref / (ref + dist_m)

def cosine_ease(x):
    return 0.5 - 0.5*torch.cos(torch.clamp(x,0,1)*math.pi)

def hann_smooth(sig, fs, ms=30.0):
    win = max(5, int(round(ms*1e-3*fs)))
    if win % 2 == 0: win += 1
    w = torch.hann_window(win, device=sig.device, dtype=sig.dtype)
    w = w / w.sum()
    pad = win//2
    s = F.pad(sig.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect')
    y = F.conv1d(s, w.view(1,1,-1)).squeeze()
    return y[:sig.numel()]

# ---------- model ----------
class T2SModelEval(nn.Module):
    def __init__(self, enc_name, el_max=1.0472):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(enc_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(enc_name)
        d = self.enc.config.hidden_size
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 768), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(768, 384), nn.ReLU(),
            nn.Linear(384, 8)
        )
        self.EL_MAX = el_max
        self.az_order="cos_sin"; self.az_sign=1.0; self.az_bias=0.0

    def _raw_head(self, h): return self.head(h)
    def _decode_az_block(self, y2, order, sign, bias):
        if order=="cos_sin": c,s = y2[...,0], y2[...,1]
        else:                s,c = y2[...,0], y2[...,1]
        n = torch.sqrt(s*s + c*c + EPS); s,c = s/n, c/n
        az = torch.atan2(s,c)
        az = sign*az + bias
        return torch.remainder(az+math.pi, 2*math.pi)-math.pi
    def _decode_rest(self, y):
        el = torch.tanh(y[...,2])*self.EL_MAX
        dist   = torch.sigmoid(y[...,3])*(6.0-0.6)+0.6
        spread = torch.sigmoid(y[...,4])*(120.0-5.0)+5.0
        wet    = torch.sigmoid(y[...,5])
        gain   = torch.tanh(y[...,6])*6.0
        return el, dist, spread, wet, gain
    @torch.no_grad()
    def forward_texts(self, texts, device="cpu", max_len=64):
        bt = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        bt = {k:v.to(device) for k,v in bt.items()}
        hs = self.enc(**bt).last_hidden_state
        attn = bt["attention_mask"].float().unsqueeze(-1)
        h = (hs*attn).sum(1) / attn.sum(1).clamp_min(1.0)
        h = self.norm(h); h = F.normalize(h, dim=-1)
        y = self._raw_head(h)
        az = self._decode_az_block(y[...,:2], self.az_order, self.az_sign, self.az_bias)
        el, dist, spread, wet, gain = self._decode_rest(y)
        out = torch.stack([az, el, dist, spread, wet, gain], dim=-1)
        return out, h

def pick_state_dict(payload):
    if isinstance(payload, dict):
        for k in ["state_dict","model","ema","ema_model","student","weights","module"]:
            if k in payload and isinstance(payload[k], dict): return payload[k]
        if all(isinstance(v, torch.Tensor) for v in payload.values() if v is not None): return payload
    raise ValueError("bad ckpt")

def strip_module(sd): return { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }

def tokenizer_align_to_ckpt(model, sd):
    keys = ["enc.embeddings.word_embeddings.weight","enc.roberta.embeddings.word_embeddings.weight","embeddings.word_embeddings.weight"]
    expected=None
    for k in keys:
        if k in sd and isinstance(sd[k], torch.Tensor): expected=sd[k].shape[0]; break
    if expected is None: return
    tok=model.tok
    cur=getattr(tok,"vocab_size",None)
    if cur is None and hasattr(tok,"get_vocab"): cur=len(tok.get_vocab())
    if cur is None: return
    if cur < expected:
        extra=[f"[EXTRA_{i}]" for i in range(expected-cur)]
        tok.add_tokens(extra, special_tokens=False)
        model.enc.resize_token_embeddings(len(tok))
        print(f"[ckpt] tokenizer extended: {cur} -> {len(tok)}")

def smart_load(model, ckpt_path, device):
    raw = torch.load(ckpt_path, map_location="cpu")
    sd  = strip_module(pick_state_dict(raw))
    tokenizer_align_to_ckpt(model, sd)
    res = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"[load] missing={len(res.missing_keys)} | unexpected={len(res.unexpected_keys)}")
    return res

# ---------- motion parsing ----------
DIR_KEYWORDS = {
    "정면":{"az":0},"앞":{"az":0},"앞쪽":{"az":0},"앞으로":{"az":0},"가운데":{"az":0},"중앙":{"az":0},"센터":{"az":0},
    "오른쪽":{"az":+90},"우":{"az":+90},"우측":{"az":+90},"오른쪽으로":{"az":+90},
    "왼쪽":{"az":-90},"좌":{"az":-90},"좌측":{"az":-90},"왼쪽으로":{"az":-90},
    "뒤":{"az":180},"뒤로":{"az":180},"후방":{"az":180},"뒤쪽":{"az":180},
    "위":{"el":+45},"위쪽":{"el":+45},"천장":{"el":+60},"위로":{"el":+45},
    "아래":{"el":-45},"아랫쪽":{"el":-45},"바닥":{"el":-60},"아래로":{"el":-45},
    "front":{"az":0},"right":{"az":+90},"left":{"az":-90},"back":{"az":180},"up":{"el":+45},"down":{"el":-45},
    "center":{"az":0},"centre":{"az":0}
}
TIME_PAT = re.compile(r'(?P<t1>\d+(?:\.\d+)?)\s*(?:초|s)?\s*(?:-|~|to|–|—)\s*(?P<t2>\d+(?:\.\d+)?)\s*(?:초|s)?', re.IGNORECASE)
DIR_TOK = re.compile(r'(가운데|중앙|센터|왼쪽|좌|left|오른쪽|우|right|위|up|아래|down|앞|앞으로|front|뒤|뒤로|back|center|centre)')

def extract_degs(token):
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:deg|도|°)', token)
    return float(m.group(1)) if m else None

def token_to_pose(tok: str):
    tok = tok.strip()
    pose = {}
    deg = extract_degs(tok)
    for k,v in DIR_KEYWORDS.items():
        if k in tok:
            pose.update(v)
    if ('오른쪽' in tok or 'right' in tok) and deg is not None: pose['az'] = +abs(deg)
    if ('왼쪽'   in tok or 'left'  in tok) and deg is not None: pose['az'] = -abs(deg)
    if ('앞'     in tok or 'front' in tok) and deg is not None: pose['az'] = angle_wrap_deg(deg)
    if ('뒤'     in tok or 'back'  in tok) and deg is not None: pose['az'] = angle_wrap_deg(deg if deg is not None else 180)
    if ('위'     in tok or 'up'    in tok) and deg is not None: pose['el'] = +abs(deg)
    if ('아래'   in tok or 'down'  in tok) and deg is not None: pose['el'] = -abs(deg)
    return pose

def text_to_waypoints(text: str, total_dur: float, model_az_deg: float, model_el_deg: float):
    s = text.strip().lower()
    strong_pairs = []
    pair_pat = re.compile(
        r'(왼쪽|좌|left|오른쪽|우|right|위|up|아래|down|앞|앞으로|front|뒤|뒤로|back|가운데|중앙|센터|center|centre)\s*에서\s*(?:아주|매우|정말|끝까지)?\s*(왼쪽|좌|left|오른쪽|우|right|위|up|아래|down|앞|앞으로|front|뒤|뒤로|back|가운데|중앙|센터|center|centre)\s*로'
    )
    for m in pair_pat.finditer(s):
        strong_pairs.append((m.group(1), m.group(2)))

    dir_seq = [m.group(1) for m in DIR_TOK.finditer(s)]
    spans = []
    for m in TIME_PAT.finditer(s):
        t1 = float(m.group('t1')); t2 = float(m.group('t2'))
        if t2 < t1: t1, t2 = t2, t1
        spans.append((t1, t2))
    spans = sorted(spans)

    name2pose = {
        "가운데":{"az":0},"중앙":{"az":0},"센터":{"az":0},"center":{"az":0},"centre":{"az":0},
        "앞":{"az":0},"앞으로":{"az":0},"front":{"az":0},
        "왼쪽":{"az":-90},"좌":{"az":-90},"left":{"az":-90},
        "오른쪽":{"az":+90},"우":{"az":+90},"right":{"az":+90},
        "뒤":{"az":180},"뒤로":{"az":180},"후방":{"az":180},"back":{"az":180},
        "위":{"el":+45},"위로":{"el":+45},"up":{"el":+45},
        "아래":{"el":-45},"아래로":{"el":-45},"down":{"el":-45},
    }

    targets=[]; hints=[]
    if strong_pairs:
        for a,b in strong_pairs:
            if a in name2pose: targets.append(name2pose[a]); hints.append(a)
            if b in name2pose: targets.append(name2pose[b]); hints.append(b)
    elif dir_seq:
        for tok in dir_seq:
            if tok in name2pose:
                targets.append(name2pose[tok]); hints.append(tok)
    else:
        parts = re.split(r'(?:→|->|,|，|그리고|및|그리고\s*나서|가다가|towards|to)', s)
        parts = [p.strip() for p in parts if p.strip()]
        for p in parts:
            pose = token_to_pose(p)
            if pose: targets.append(pose); hints.append(p)

    waypoints=[]
    if spans:
        cur_az = model_az_deg; cur_el = model_el_deg
        t0 = max(0.0, spans[0][0])
        waypoints.append( (t0, {'az':cur_az,'el':cur_el}, "시작") )
        n = max(len(spans), len(targets))
        for i in range(n):
            t1, t2 = spans[i] if i < len(spans) else (waypoints[-1][0], total_dur)
            tgt = targets[i] if i < len(targets) else {}
            if 'az' not in tgt: tgt['az'] = cur_az
            if 'el' not in tgt: tgt['el'] = cur_el
            hint = hints[i] if i < len(hints) else "유지"
            waypoints.append( (t2, {'az':tgt['az'],'el':tgt['el']}, hint) )
            cur_az, cur_el = tgt['az'], tgt['el']
        if waypoints[-1][0] < total_dur:
            waypoints.append( (total_dur, {'az':cur_az,'el':cur_el}, "끝") )
    elif targets:
        first = targets[0]
        if 'az' not in first: first['az'] = model_az_deg
        if 'el' not in first: first['el'] = model_el_deg
        rest = targets[1:]; rest_h = hints[1:] if len(hints)>1 else []
        K = len(rest) + 1
        times = [i*total_dur/(K) for i in range(K+1)]
        waypoints.append( (0.0, {'az':first['az'],'el':first['el']}, hints[0] if hints else "시작") )
        cur_az, cur_el = first['az'], first['el']
        for i, tgt in enumerate(rest, start=1):
            if 'az' not in tgt: tgt['az'] = cur_az
            if 'el' not in tgt: tgt['el'] = cur_el
            hint = rest_h[i-1] if i-1 < len(rest_h) else "이동"
            waypoints.append( (times[i], {'az':tgt['az'],'el':tgt['el']}, hint) )
            cur_az, cur_el = tgt['az'], tgt['el']
        waypoints.append( (times[-1], {'az':cur_az,'el':cur_el}, "끝") )
    else:
        waypoints = [(0.0, {'az':model_az_deg,'el':model_el_deg}, "시작"),
                     (total_dur, {'az':model_az_deg,'el':model_el_deg}, "끝")]
    # clamp & dedupe
    cleaned=[]; last_t=-1
    for t,pose,h in waypoints:
        t = max(0.0, min(total_dur, float(t)))
        if t < last_t: continue
        az = angle_wrap_deg(float(pose.get('az', model_az_deg)))
        el = max(-85.0, min(85.0, float(pose.get('el', model_el_deg))))
        cleaned.append((t, {'az': az, 'el': el}, h)); last_t = t
    uniq=[]
    for i,(t,p,h) in enumerate(cleaned):
        if i>0 and abs(t - cleaned[i-1][0]) < 1e-6: continue
        uniq.append((t,p,h))
    return uniq

# ---------- per-waypoint RE-INFER ----------
def build_per_knot_text(base_text: str, hint: str):
    hint = hint.strip()
    if not hint or hint in ("유지","끝","시작"): return base_text
    return f"{base_text} (현재 위치/방향 힌트: {hint}) (pose hint: {hint})"

@torch.no_grad()
def infer_params_for_text(model, device, text):
    y,_ = model.forward_texts([text], device=device)
    az,el,dist,spread,wet,gain = [float(v) for v in y[0].tolist()]
    return {"az_deg": rad2deg(az), "el_deg": rad2deg(el),
            "dist_m": dist, "spread_deg": spread, "wet": wet, "gain_db": gain}

def build_param_curves_from_knots(knots, T, fs):
    device = torch.device("cpu")
    t_axis = torch.arange(T, device=device)/fs
    ts  = torch.tensor([k["t_sec"] for k in knots], device=device)
    azs = torch.tensor([k["az_deg"] for k in knots], device=device)
    els = torch.tensor([k["el_deg"] for k in knots], device=device)
    dss = torch.tensor([k["dist_m"] for k in knots], device=device)
    spr = torch.tensor([k["spread_deg"] for k in knots], device=device)
    wet = torch.tensor([k["wet"] for k in knots], device=device)
    gn  = torch.tensor([k["gain_db"] for k in knots], device=device)

    az_curve = torch.empty(T, device=device)
    el_curve = torch.empty(T, device=device)
    ds_curve = torch.empty(T, device=device)
    sp_curve = torch.empty(T, device=device)
    wt_curve = torch.empty(T, device=device)
    gn_curve = torch.empty(T, device=device)

    for i in range(len(ts)-1):
        t0 = ts[i].item(); t1 = ts[i+1].item()
        idx = (t_axis >= t0) & (t_axis <= t1 if i==len(ts)-2 else t_axis < t1)
        if not idx.any(): continue
        frac = cosine_ease((t_axis[idx]-t0)/max(t1-t0, 1e-6))
        a0, a1 = azs[i].item(), azs[i+1].item()
        da = angle_wrap_deg(a1 - a0)
        az_curve[idx] = angle_wrap_deg(a0 + da*frac)
        el_curve[idx] = els[i] + (els[i+1]-els[i]) * frac
        ds_curve[idx] = dss[i] + (dss[i+1]-dss[i]) * frac
        sp_curve[idx] = spr[i] + (spr[i+1]-spr[i]) * frac
        wt_curve[idx] = wet[i] + (wet[i+1]-wet[i]) * frac
        gn_curve[idx] = gn[i]  + (gn[i+1] -gn[i])  * frac

    # endpoints
    for c, arr in [(az_curve,azs),(el_curve,els),(ds_curve,dss),(sp_curve,spr),(wt_curve,wet),(gn_curve,gn)]:
        c[0]=arr[0]; c[-1]=arr[-1]

    # smooth ~30ms
    az_curve = hann_smooth(az_curve, fs, 30.0)
    el_curve = hann_smooth(el_curve, fs, 30.0)
    ds_curve = hann_smooth(ds_curve, fs, 30.0)
    sp_curve = hann_smooth(sp_curve, fs, 30.0)
    wt_curve = hann_smooth(wt_curve, fs, 30.0)
    gn_curve = hann_smooth(gn_curve, fs, 30.0)

    return az_curve, el_curve, ds_curve, sp_curve, wt_curve, gn_curve

# ---------- renderers (stereo with TRUE stereo-follow reverb) ----------
def render_stereo_motion_paramcurves(x_mono, fs,
                                     az_curve_deg, el_curve_deg,
                                     dist_curve, wet_curve, gain_curve, spread_curve,
                                     itd_ms=1.0, room_mode="drr"):
    T = x_mono.shape[-1]
    base = x_mono.squeeze(0)
    device = base.device
    dtype  = base.dtype

    # --- 팬닝: spread를 게인에 섞지 말고 100% 위치기반으로 ---
    azr = torch.deg2rad(az_curve_deg.to(device))
    p  = (azr.clamp(-math.pi/2, math.pi/2) + math.pi/2) * 0.5
    gL = torch.cos(p)              # equal-power
    gR = torch.sin(p)

    # ITD (Haas)
    max_itd = max(0.0, float(itd_ms))/1000.0
    itd_samp = (max_itd*torch.sin(azr)) * fs

    # TRUE stereo-follow reverb (리버브는 좌/우 각각 만듦)
    ir_st = synth_room_ir_stereo(fs, rt60=(0.8 if room_mode=="drr" else 0.55),
                                 pre_delay_ms=12.0, drr_db=0.0, device=device, dtype=dtype)
    wet_L_base = fft_convolve(x_mono, ir_st[0:1]).squeeze(0)[:T]
    wet_R_base = fft_convolve(x_mono, ir_st[1:2]).squeeze(0)[:T]

    # spread는 "소스 폭"으로만 사용 → 리버브 쪽 크로스피드에만 반영(선택)
    # 값이 클수록 리버브를 더 좌우로 퍼뜨림
    spread_w = torch.clamp(spread_curve.to(device) / 120.0, 0.0, 1.0)

    frame, hop = 1024, 512
    win = torch.hann_window(frame, periodic=True, device=device, dtype=dtype)
    outL = torch.zeros(T+frame, device=device, dtype=dtype)
    outR = torch.zeros(T+frame, device=device, dtype=dtype)

    for i in range(0, T, hop):
        s=i; e=min(i+frame, T); w=win[:e-s]
        gL_b = gL[s:e].mean(); gR_b = gR[s:e].mean()
        d_b  = itd_samp[s:e].mean().item()
        dist_b = dist_curve[s:e].mean().item()
        wet_b  = float(wet_curve[s:e].mean().item())
        gain_b = gain_curve[s:e].mean().item()
        att = distance_attenuation(dist_b)
        g_all = db_to_lin(gain_b)*att

        seg = base[s:e]
        # ITD on dry
        if d_b >= 0:
            Ld = fractional_delay(seg.unsqueeze(0),  d_b).squeeze(0)
            Rd = seg
        else:
            Ld = seg
            Rd = fractional_delay(seg.unsqueeze(0), -d_b).squeeze(0)
        Ld = Ld * gL_b * g_all * w
        Rd = Rd * gR_b * g_all * w

        # stereo wet follow + spread 기반 리버브 크로스토크(마스킹 완화)
        segWL = wet_L_base[s:e]; segWR = wet_R_base[s:e]
        if d_b >= 0:
            Lw = fractional_delay(segWL.unsqueeze(0),  d_b).squeeze(0)
            Rw = segWR
        else:
            Lw = segWL
            Rw = fractional_delay(segWR.unsqueeze(0), -d_b).squeeze(0)

        # 퍼짐(spread)으로 리버브만 살짝 좌우 블렌드
        sw = float(spread_w[s:e].mean().item())
        # sw=0: 그대로, sw=1: 리버브가 양 채널 반반 섞임(폭 넓게)
        Lw = (1.0-sw)*Lw + sw*0.5*(Lw+Rw)
        Rw = (1.0-sw)*Rw + sw*0.5*(Lw+Rw)

        outL[s:s+Ld.numel()] += (1.0-wet_b)*Ld + wet_b*(Lw*w)
        outR[s:s+Rd.numel()] += (1.0-wet_b)*Rd + wet_b*(Rw*w)

    y = torch.stack([outL[:T], outR[:T]], dim=0)
    return hardclip(y / (y.abs().amax()+1e-6))


# ---------- binaural (ILD+ITD + stereo-follow reverb) ----------
def _highshelf_block(block, fs, gain_db, fc=4000.0):
    if abs(gain_db) < 1e-6: return block
    A  = 10**(gain_db/40.0)
    w0 = 2*math.pi*fc/fs
    alpha = math.sin(w0)/math.sqrt(2)
    cosw0 = math.cos(w0)
    b0 =    A*((A+1)+ (A-1)*cosw0 + 2*alpha)
    b1 = -2*A*((A-1)+ (A+1)*cosw0)
    b2 =    A*((A+1)+ (A-1)*cosw0 - 2*alpha)
    a0 =       (A+1) - (A-1)*cosw0 + 2*alpha
    a1 =  2* ( (A-1) - (A+1)*cosw0 )
    a2 =       (A+1) - (A-1)*cosw0 - 2*alpha
    b0/=a0; b1/=a0; b2/=a0; a1/=a0; a2/=a0
    y = torch.zeros_like(block)
    x1=x2=y1=y2=0.0
    for n in range(block.numel()):
        x = float(block[n])
        yv = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[n] = yv
        x2,x1 = x1,x
        y2,y1 = y1,yv
    return y

def render_binaural_motion_paramcurves(x_mono, fs,
                                       az_curve_deg, el_curve_deg,
                                       dist_curve, wet_curve, gain_curve, spread_curve,
                                       itd_ms=1.0, ild_db_max=12.0, room_mode="drr"):
    T = x_mono.shape[-1]
    base = x_mono.squeeze(0)
    device = base.device
    dtype  = base.dtype

    azr  = torch.deg2rad(az_curve_deg.to(device))
    elc  = el_curve_deg.to(device)

    # --- 팬닝: ILD/ITD만으로 위치, spread는 게인에 섞지 않음 ---
    ild_db = float(ild_db_max)*torch.sin(azr.abs())
    g_near = db_to_lin(+ild_db/2)
    g_far  = db_to_lin(-ild_db/2)
    sign   = torch.sign(azr)

    max_itd = max(0.0, float(itd_ms))/1000.0
    itd_samp = (max_itd*torch.sin(azr)) * fs

    ir_st = synth_room_ir_stereo(fs, rt60=(0.7 if room_mode=="drr" else 0.55),
                                 pre_delay_ms=10.0, drr_db=0.0, device=device, dtype=dtype)
    wet_L_base = fft_convolve(x_mono, ir_st[0:1]).squeeze(0)[:T]
    wet_R_base = fft_convolve(x_mono, ir_st[1:2]).squeeze(0)[:T]

    # spread는 리버브 크로스토크에만 반영
    spread_w = torch.clamp(spread_curve.to(device) / 120.0, 0.0, 1.0)

    frame, hop = 1024, 512
    win = torch.hann_window(frame, periodic=True, device=device, dtype=dtype)
    outL = torch.zeros(T+frame, device=device, dtype=dtype)
    outR = torch.zeros(T+frame, device=device, dtype=dtype)

    for i in range(0, T, hop):
        s=i; e=min(i+frame, T); w=win[:e-s]
        sign_b = sign[s:e].mean()
        ild_n  = g_near[s:e].mean()
        ild_f  = g_far[s:e].mean()
        d_b    = itd_samp[s:e].mean().item()
        az_b   = rad2deg(azr[s:e].mean().item())
        el_b   = elc[s:e].mean().item()
        dist_b = dist_curve[s:e].mean().item()
        wet_b  = float(wet_curve[s:e].mean().item())
        gain_b = gain_curve[s:e].mean().item()
        sw     = float(spread_w[s:e].mean().item())

        att = distance_attenuation(dist_b)
        g_all = db_to_lin(gain_b)*att

        seg = base[s:e].clone()
        if abs(az_b) > 135: seg = _highshelf_block(seg, fs, -8.0, fc=3500.0)
        if el_b > +10.0:    seg = _highshelf_block(seg, fs, +4.0, fc=6000.0)
        elif el_b < -10.0:  seg = _highshelf_block(seg, fs, -4.0, fc=6000.0)

        if d_b >= 0:
            Ld = fractional_delay(seg.unsqueeze(0),  d_b).squeeze(0)
            Rd = seg
        else:
            Ld = seg
            Rd = fractional_delay(seg.unsqueeze(0), -d_b).squeeze(0)

        if sign_b >= 0:
            L = (Ld*ild_f*g_all) * w
            R = (Rd*ild_n*g_all) * w
        else:
            L = (Ld*ild_n*g_all) * w
            R = (Rd*ild_f*g_all) * w

        segWL = wet_L_base[s:e]; segWR = wet_R_base[s:e]
        if d_b >= 0:
            Lw = fractional_delay(segWL.unsqueeze(0),  d_b).squeeze(0)
            Rw = segWR
        else:
            Lw = segWL
            Rw = fractional_delay(segWR.unsqueeze(0), -d_b).squeeze(0)

        # 리버브만 spread로 폭 넓힘
        Lw = (1.0-sw)*Lw + sw*0.5*(Lw+Rw)
        Rw = (1.0-sw)*Rw + sw*0.5*(Lw+Rw)

        L += Lw * w * wet_b
        R += Rw * w * wet_b

        outL[s:s+L.numel()] += L
        outR[s:s+R.numel()] += R

    y = torch.stack([outL[:T], outR[:T]], dim=0)
    return hardclip(y / (y.abs().amax()+1e-6))


def encode_foa_motion(x_mono, az_curve_deg, el_curve_deg):
    az = torch.deg2rad(az_curve_deg); el=torch.deg2rad(el_curve_deg)
    sW=1/math.sqrt(2); sV=math.sqrt(3/2)
    base = x_mono.squeeze(0)
    W = sW*base
    Y = sV*base*torch.sin(az)*torch.cos(el)
    Z = sV*base*torch.sin(el)
    X = sV*base*torch.cos(az)*torch.cos(el)
    return torch.stack([W,Y,Z,X], dim=0)

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--enc_model", required=True)
    ap.add_argument("--wav_in", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_dir", default="renders")
    ap.add_argument("--room_mode", choices=["drr","rt60"], default="drr")
    ap.add_argument("--stereo", action="store_true")
    ap.add_argument("--binaural", action="store_true")
    ap.add_argument("--hoa", action="store_true")
    ap.add_argument("--itd_ms", type=float, default=1.2)
    ap.add_argument("--ild_db", type=float, default=12.0)
    ap.add_argument("--scale_dist",   type=float, default=2.0)
    ap.add_argument("--scale_spread", type=float, default=2.0)
    ap.add_argument("--scale_gain",   type=float, default=2.0)
    ap.add_argument("--scale_wet",    type=float, default=2.0)
    ap.add_argument("--max_len", type=int, default=64)
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    x, fs = torchaudio.load(args.wav_in)
    x = x.mean(0, keepdim=True)
    x = x / (x.abs().amax()+1e-6)
    T = x.shape[-1]; total_dur = T/fs

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=T2SModelEval(args.enc_model).to(device).eval()
    smart_load(m, args.ckpt, device)

    with torch.no_grad():
        y,_=m.forward_texts([args.text], device=device, max_len=args.max_len)
    az,el,dist,spread,wet,gain=[float(v) for v in y[0].tolist()]
    base_params = {"az_deg": rad2deg(az), "el_deg": rad2deg(el),
                   "dist_m": dist, "spread_deg": spread, "wet": wet, "gain_db": gain}

    text_lc = args.text.lower()
    if any(k in text_lc for k in ["드라이","건조","dry","less reverb","no reverb"]): base_params["wet"] = 0.05
    if any(k in text_lc for k in ["웻","젖게","reverb heavy","more reverb","roomy"]): base_params["wet"] = max(base_params["wet"], 0.45)

    wayp = text_to_waypoints(args.text, total_dur, base_params["az_deg"], base_params["el_deg"])

    # Per-waypoint re-infer
    knots=[]
    for (t_sec, pose, hint) in wayp:
        pred = infer_params_for_text(m, device, build_per_knot_text(args.text, hint))
        pred["az_deg"] = pose["az"]; pred["el_deg"]=pose["el"]
        if any(k in text_lc for k in ["드라이","건조","dry"]): pred["wet"] = min(pred["wet"], 0.08)
        if any(k in text_lc for k in ["웻","젖게","roomy"]):   pred["wet"] = max(pred["wet"], 0.45)
        pred["t_sec"]=float(t_sec)
        knots.append(pred)

    if knots[0]["t_sec"] > 0.0:
        k0 = knots[0].copy(); k0["t_sec"]=0.0; knots.insert(0,k0)
    if knots[-1]["t_sec"] < total_dur:
        kN = knots[-1].copy(); kN["t_sec"]=total_dur; knots.append(kN)

    az_curve, el_curve, dist_curve, spread_curve, wet_curve, gain_curve = \
        build_param_curves_from_knots(knots, T, fs)

    # amplify deltas (around median)
    def amplify(curve, scale):
        if scale==1.0: return curve
        base = torch.median(curve)
        return base + (curve - base) * scale
    dist_curve   = amplify(dist_curve,   args.scale_dist)
    spread_curve = amplify(spread_curve, args.scale_spread)
    gain_curve   = amplify(gain_curve,   args.scale_gain)
    wet_curve    = torch.clamp(amplify(wet_curve, args.scale_wet), 0.0, 1.0)

    # Render
    if args.stereo:
        y_st = render_stereo_motion_paramcurves(x.to(device), fs,
                                                az_curve, el_curve,
                                                dist_curve, wet_curve, gain_curve, spread_curve,
                                                itd_ms=args.itd_ms, room_mode=args.room_mode)
        torchaudio.save(os.path.join(args.out_dir,"out_stereo.wav"), y_st.cpu(), fs)
        print("[write] out_stereo.wav")

    if args.binaural:
        y_bi = render_binaural_motion_paramcurves(x.to(device), fs,
                                                  az_curve, el_curve,
                                                  dist_curve, wet_curve, gain_curve, spread_curve,
                                                  itd_ms=args.itd_ms, ild_db_max=args.ild_db, room_mode=args.room_mode)
        torchaudio.save(os.path.join(args.out_dir,"out_binaural.wav"), y_bi.cpu(), fs)
        print("[write] out_binaural.wav")

    if args.hoa:
        foa = encode_foa_motion(x.to(device), az_curve.to(device), el_curve.to(device))
        torchaudio.save(os.path.join(args.out_dir,"out_foa.wav"), foa.cpu(), fs, channels_first=True)
        print("[write] out_foa.wav")

    # curve stats
    def stats(t):
        return {"min": float(t.min()), "max": float(t.max()),
                "median": float(torch.median(t)), "p95": float(torch.quantile(t, torch.tensor(0.95)))}

    meta = {
        "text": args.text,
        "base": base_params,
        "waypoints": wayp,
        "param_knots": knots,
        "curve_stats": {
            "az_deg": stats(az_curve), "el_deg": stats(el_curve),
            "dist_m": stats(dist_curve), "spread_deg": stats(spread_curve),
            "wet": stats(wet_curve), "gain_db": stats(gain_curve)
        },
        "scalers": {
            "scale_dist": args.scale_dist, "scale_spread": args.scale_spread,
            "scale_gain": args.scale_gain, "scale_wet": args.scale_wet
        }
    }
    with open(os.path.join(args.out_dir,"predicted_params.json"),"w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[meta] predicted_params.json")
    print("[curve] dist_m", meta["curve_stats"]["dist_m"])
    print("[curve] spread_deg", meta["curve_stats"]["spread_deg"])
    print("[curve] wet", meta["curve_stats"]["wet"])
    print("[curve] gain_db", meta["curve_stats"]["gain_db"])

if __name__=="__main__":
    main()
