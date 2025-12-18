import math, csv, statistics as st
import numpy as np

CSV = "icassp_test_preds.csv"

def to_angle(sinv, cosv):
    return math.atan2(float(sinv), float(cosv))  # [-pi, pi]

def circ_err_deg(a, b):
    d = (a - b + math.pi) % (2*math.pi) - math.pi  # [-pi, pi]
    return abs(d) * 180.0 / math.pi

def mae(xs, ys): return float(np.mean([abs(x-y) for x,y in zip(xs,ys)]))

# 누적 버킷
az_start_deg, az_end_deg, el_start_deg, el_end_deg = [], [], [], []
dist_z_t, dist_z_p = [], []
width_s_t, width_s_p, width_e_t, width_e_p = [], [], [], []
wet_t, wet_p, gain_t, gain_p = [], [], [], []

with open(CSV, newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        # angles: reconstruct
        az_ts = to_angle(r["y_true_az_sin_start"], r["y_true_az_cos_start"])
        az_ps = to_angle(r["y_pred_az_sin_start"], r["y_pred_az_cos_start"])
        az_te = to_angle(r["y_true_az_sin_end"],   r["y_true_az_cos_end"])
        az_pe = to_angle(r["y_pred_az_sin_end"],   r["y_pred_az_cos_end"])
        el_ts = to_angle(r["y_true_el_sin_start"], r["y_true_el_cos_start"])
        el_ps = to_angle(r["y_pred_el_sin_start"], r["y_pred_el_cos_start"])
        el_te = to_angle(r["y_true_el_sin_end"],   r["y_true_el_cos_end"])
        el_pe = to_angle(r["y_pred_el_sin_end"],   r["y_pred_el_cos_end"])

        az_start_deg.append(circ_err_deg(az_ts, az_ps))
        az_end_deg.append(circ_err_deg(az_te, az_pe))
        el_start_deg.append(circ_err_deg(el_ts, el_ps))
        el_end_deg.append(circ_err_deg(el_te, el_pe))

        dist_z_t.append(float(r["y_true_dist_z"]));        dist_z_p.append(float(r["y_pred_dist_z"]))
        width_s_t.append(float(r["y_true_width_start"]));  width_s_p.append(float(r["y_pred_width_start"]))
        width_e_t.append(float(r["y_true_width_end"]));    width_e_p.append(float(r["y_pred_width_end"]))
        wet_t.append(float(r["y_true_wet_norm"]));         wet_p.append(float(r["y_pred_wet_norm"]))
        gain_t.append(float(r["y_true_gain_norm"]));       gain_p.append(float(r["y_pred_gain_norm"]))

print("== Angular MAE (deg) ==")
print(f"Az start: {st.mean(az_start_deg):.2f}")
print(f"Az end  : {st.mean(az_end_deg):.2f}")
print(f"El start: {st.mean(el_start_deg):.2f}")
print(f"El end  : {st.mean(el_end_deg):.2f}")

print("\n== Other targets MAE ==")
print(f"dist_z      : {mae(dist_z_t, dist_z_p):.4f}")
print(f"width_start : {mae(width_s_t, width_s_p):.4f}")
print(f"width_end   : {mae(width_e_t, width_e_p):.4f}")
print(f"wet_norm    : {mae(wet_t, wet_p):.4f}")
print(f"gain_norm   : {mae(gain_t, gain_p):.4f}")

# 가중 총점(논문용): 각도는 deg 평균, 나머지는 0–1 스케일의 MAE → 그룹 가중합
w_angles, w_others = 2.0, 1.0  # 권장 가중
ang_mean = st.mean([st.mean(az_start_deg), st.mean(az_end_deg), st.mean(el_start_deg), st.mean(el_end_deg)])
oth_mean = st.mean([mae(dist_z_t, dist_z_p), mae(width_s_t,width_s_p), mae(width_e_t,width_e_p), mae(wet_t,wet_p), mae(gain_t,gain_p)])
score = w_angles*ang_mean + w_others*oth_mean
print(f"\nComposite score (lower better): {score:.3f}")
