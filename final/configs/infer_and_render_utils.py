# save as: infer_and_render_utils.py
import math, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

ENC="intfloat/multilingual-e5-large"

class MinimalInfer(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.tok=AutoTokenizer.from_pretrained(ENC)
        self.enc=AutoModel.from_pretrained(ENC)
        d=self.enc.config.hidden_size
        self.head=nn.Sequential(nn.Linear(d,768), nn.ReLU(),
                                nn.Linear(768,384), nn.ReLU(),
                                nn.Linear(384,8))
        self.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    def forward(self, texts):
        toks=self.tok(texts, padding=True, truncation=True, return_tensors="pt")
        h=self.enc(**toks).last_hidden_state[:,0,:]
        y=self.head(h)
        s,c=y[:,0], y[:,1]; n=torch.sqrt(s*s+c*c+1e-8); y[:,0],y[:,1]=s/n,c/n
        y[:,2]=torch.tanh(y[:,2])*0.7854
        y[:,3]=torch.sigmoid(y[:,3])*(6.0-0.6)+0.6
        y[:,4]=torch.sigmoid(y[:,4])*(120.0-5.0)+5.0
        y[:,5]=torch.sigmoid(y[:,5])
        y[:,6]=torch.tanh(y[:,6])*6.0
        return y  # 8D

def to_binaural_params(y8, room_mode="drr"):
    s,c,el,dist,spread,wet,gain,room = [v.item() for v in y8]
    az_deg = math.degrees(math.atan2(s,c))
    out = {
        "az_deg": az_deg,
        "el_rad": el,
        "dist_m": dist,
        "spread_deg": spread,
        "wet_mix": wet,
        "gain_db": gain
    }
    out["drr_db" if room_mode=="drr" else "rt60_s"] = room
    # ▼▼▼ Post-render 처리(학습 제외) 지점 ▼▼▼
    out.update({
        "post_head_ypr": (0.0,0.0,0.0),     # 디코더에서 헤드 회전 적용
        "post_eq": None,                    # 톤/EQ는 후단에서
        "post_comp": None,                  # 컴프/리미터 후단
        "post_reverb_char": "hall"          # early/late 분배 등 캐릭터
    })
    return out

def acn_sn3d_sh(az, el, N=3):
    # 간결한 real SH(ACN/SN3D). 실제 제품에선 검증된 라이브러리 사용 권장.
    Y=[]
    # n=0
    Y += [1.0]
    # n=1
    Y += [math.sin(el), math.cos(el)*math.sin(az), math.cos(el)*math.cos(az)]
    # n=2 (간단 구성)
    ce, se = math.cos(el), math.sin(el)
    Y += [math.sqrt(3)*se*ce, math.sqrt(3)*math.sin(2*az)*(ce**2)/2,
          (3*(ce**2)-1)/2, math.sqrt(3)*math.cos(2*az)*(ce**2)/2, -math.sqrt(3)*se*ce]
    # n=3 (간단 구성)
    Y += [
        0.5*(5*se**2-1)*se,
        0.5*math.sqrt(6)*math.sin(az)*(5*ce**3-3*ce),
        0.5*math.sqrt(10)*(5*ce**3-3*ce),
        0.5*math.sqrt(6)*math.cos(az)*(5*ce**3-3*ce),
        -0.5*(5*se**2-1)*se,
        -math.sqrt(6)*math.sin(az)*se*(5*ce**2-1)/2,
        -math.sqrt(10)*se*(5*ce**2-1)/2,
        -math.sqrt(6)*math.cos(az)*se*(5*ce**2-1)/2
    ]
    return np.array(Y[:(N+1)**2], dtype=float)

def to_hoa3_coeffs(y8, room_mode="drr"):
    s,c,el,dist,spread,wet,gain,room = [v.item() for v in y8]
    az = math.atan2(s,c)
    Y = acn_sn3d_sh(az, el, N=3)  # 16ch
    # spread → 고차 모드 감쇠 (확산↑일수록 고차 감소)
    width = max(0.0, min(1.0, (spread-5.0)/(120.0-5.0)))
    scales = [1.0, 1.0-0.3*width, 1.0-0.6*width, 1.0-0.9*width]
    out=[]; idx=0
    for n in range(0,4):
        for m in range(-n, n+1):
            out.append(Y[idx]*scales[n]); idx+=1
    meta = {"wet_mix":wet, "gain_db":gain, ("drr_db" if room_mode=="drr" else "rt60_s"):room}
    # ▼▼▼ Post: head 회전 / EQ / ER 캐릭터 등은 렌더러에서 ▼▼▼
    return np.array(out), meta
