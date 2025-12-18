# save as: infer_text2spatial_api.py
import torch, json
from transformers import AutoTokenizer, AutoModel

class T2SInfer:
    def __init__(self, ckpt, enc_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(enc_name)
        self.enc = AutoModel.from_pretrained(enc_name).to(self.device).eval()
        self.m = torch.load(ckpt, map_location=self.device)
        if hasattr(self.m, "_modules"): self.m.eval()

    @torch.no_grad()
    def encode(self, texts, max_len=64):
        toks = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        h = self.enc(**toks).last_hidden_state[:,0,:]
        return h

    @torch.no_grad()
    def predict(self, texts):
        if isinstance(texts,str): texts=[texts]
        h = self.encode(texts)
        yhat, *_ = self.m(h, device=self.device) if callable(getattr(self.m,"forward",None)) else self.m(h)
        # clamp & format
        s, c = yhat[:,0].cpu().tolist(), yhat[:,1].cpu().tolist()
        el = yhat[:,2].clamp(-1.0472,1.0472).cpu().tolist()
        dist = yhat[:,3].clamp(0.6,6.0).cpu().tolist()
        spr = yhat[:,4].clamp(5.0,120.0).cpu().tolist()
        wet = yhat[:,5].sigmoid().clamp(0,1).cpu().tolist() if yhat.shape[1]>5 else [0.2]*len(texts)
        gain = yhat[:,6].clamp(-6,6).cpu().tolist() if yhat.shape[1]>6 else [0.0]*len(texts)
        return [
            {"az_sc":[s[i],c[i]], "el_rad":el[i], "dist_m":dist[i], "spread_deg":spr[i],
             "wet_mix":wet[i], "gain_db":gain[i]}
            for i in range(len(texts))
        ]

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True, nargs="+")
    args=ap.parse_args()
    inf=T2SInfer(args.ckpt)
    out=inf.predict(args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))



# python infer_text2spatial_api.py \
#   --ckpt t2sa_e2e_minilm_stage4f_lastmilefocus.pt  \
#   --text "앞오른쪽에서 빠르게 스쳐가" "approaches between right and back, softly"

