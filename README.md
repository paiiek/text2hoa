# Text-to-Spatial Audio Parameter Regression

This repository contains the implementation for the ICASSP 2026 paper:
**"Natural Language to Spatial Audio Parameters: Lightweight Deterministic Rendering for Creative Authoring"**

Authors: Seungryeol Paik, Kyogu Lee
Affiliation: Seoul National University
Demo page: https://paiiek.github.io/mmhoa-demo/

---

## Overview

This work presents a lightweight regression model that maps natural language descriptions to spatial audio parameters (azimuth, elevation, distance, spread, reverberation, gain). The model supports Korean and English inputs and achieves 33.2° angular error on out-of-distribution test sets.

Key features:
- Multilingual sentence encoder (MiniLM-L12-v2) with BitFit fine-tuning
- Angular-margin loss with adaptive per-bin adjustment
- Deterministic parameter-based evaluation (renderer-agnostic)
- Cross-HRTF robustness validation
- DAW integration via OSC protocol

---

## Repository Structure

```
text2hoa/
├── README.md                   # This file
├── final/                      # Production-ready code and models
│   ├── models/                 # Best trained models
│   │   ├── t2sa_e2e_minilm_stage4f_lastmilefocus.pt  # Main model (33.2° AE)
│   │   ├── t2sa_minilm_ft_clean_lastmile_ep6.pt      # Alternative checkpoint
│   │   └── t2sa_e2e_e5s_align.pt                     # E5-small variant
│   ├── datasets/               # Final train/valid/test splits
│   │   ├── text2spatial_v4_train.jsonl
│   │   ├── text2spatial_v4_valid.jsonl
│   │   ├── text2spatial_v4_test.jsonl
│   │   └── tiny.jsonl          # Small test set
│   └── configs/                # Training and inference scripts
│       ├── train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py
│       ├── eval_lastmile_v4.py
│       ├── infer_text2spatial_api.py
│       └── data.py
├── utils/                      # Utility scripts
│   ├── baseline_rulelex_eval.py          # Rule-based baseline (150-term lexicon)
│   ├── bakeoff_encoders_v1_fix.py        # Encoder comparison
│   ├── eval_hrtf_robustness.py           # Cross-HRTF testing
│   ├── eval_ood_textset.py               # OOD evaluation
│   └── make_icassp_tables.py             # Paper tables
├── docs/                       # Results and documentation
│   ├── lexicon_complete.json   # 150-term spatial audio lexicon (KO/EN)
│   └── results/                # Evaluation metrics (JSON/CSV)
├── renderer/                   # Spatial audio rendering backend
│   └── hrtf/kemar.sofa         # HRTF data (KEMAR)
└── v2/, v3/, archive/          # Experimental variants
```

---

## Installation

```bash
pip install torch transformers sofa-tools librosa soundfile pydub
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+

---

## Quick Start

### Inference

```bash
cd final/configs

# Single text input
python infer_text2spatial_api.py \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --text "오른쪽 앞에서 조금씩 선명해져"

# Multiple inputs (Korean + English)
python infer_text2spatial_api.py \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --text "앞오른쪽에서 빠르게 스쳐가" "approaches between right and back, softly"
```

Output format:
```json
{
  "azimuth_deg": 45.2,
  "elevation_deg": 0.0,
  "distance_m": 1.8,
  "spread_deg": 22.5,
  "wet_mix": 0.18,
  "gain_db": -3.2,
  "room_drr_db": 8.5
}
```

### Rendering

```bash
# Binaural rendering with HRTF
python infer_render.py \
  --text "왼쪽 뒤에서 천천히 다가와" \
  --mono_audio input.wav \
  --output output_binaural.wav \
  --hrtf ../renderer/hrtf/kemar.sofa
```

---

## Dataset

Total samples: 17,151 (after augmentation)
- Train: 15,000
- Validation: 1,700
- Test: 451 (OOD)
- Languages: Korean (59%), English (41%)

Spatial parameters:
- Azimuth: sin/cos representation (360°)
- Elevation: radians (±60°)
- Distance: 0.6–6.0 m (log-scaled)
- Spread: 5–120°
- Reverberation: wet mix (0–1)
- Gain: dB
- Room: DRR or RT60

Data format (JSONL):
```json
{
  "text": "오른쪽 앞에서 조금씩 선명해져",
  "lang": "ko",
  "az_sc": [0.707, 0.707],
  "el_rad": 0.0,
  "dist_m": 1.217,
  "spread_deg": 18.2,
  "wet_mix": 0.23,
  "room_depth": {"drr_db": 8.68},
  "gain_db": -4.98
}
```

---

## Model Architecture

Encoder: `paraphrase-multilingual-MiniLM-L12-v2`
- Fine-tuned with BitFit (bias + LayerNorm only)
- Last 2 layers unfrozen

Regression Head:
```
Linear(384 → 768) → ReLU → Dropout(0.1)
→ Linear(768 → 384) → ReLU
→ Linear(384 → 8)  # [sin(az), cos(az), el, dist, spread, wet, gain, room]
```

Training objectives:
1. Angular-margin loss (ArcFace for 12-bin azimuth classification)
2. MAE for continuous parameters (elevation, distance, spread, wet, gain, room)
3. Adaptive margins (per-bin adjustment based on validation error)
4. Directional contrast loss (metric learning for spatial embeddings)
5. KNN adjustment (nearest-neighbor smoothing)

---

## Results

Main paper results (test set):
- Angular Error: 33.2°
- Distance MAE: 0.264 (log-scale)
- Spread MAE: 13.9°
- Wet MAE: 0.176
- Gain MAE: 1.09 dB

Ablation study:
| Configuration | Angular Error (°) |
|--------------|------------------|
| Full model | 33.2 |
| w/o Angular-margin | 41.0 |
| w/o Adaptive margins | 38.7 |
| w/o Directional focus | 36.8 |
| w/o KNN adjustment | 37.5 |
| E5 encoder | 38.2 |

Baselines:
- Rule-based (150-term lexicon): 71.0°
- Linear (SBERT): 61.8°
- Linear (E5): 76.8°
- Diff-SAGe (ICASSP 2025): ~38° (waveform-based generative model)

MOS evaluation (25 participants, 5-point scale):
- Overall preference: 4.02 ± 0.64
- Localization clarity: 4.28 ± 0.60
- Text-spatial fit: 4.12 ± 0.63
- Naturalness: 3.96 ± 0.64

---

## Evaluation

```bash
cd final/configs

# Standard evaluation
python eval_lastmile_v4.py \
  --data ../datasets/text2spatial_v4_test.jsonl \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt

# OOD robustness
cd ../../utils
python eval_ood_textset.py \
  --ckpt ../final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt

# Cross-HRTF robustness
python eval_hrtf_robustness.py \
  --cipic_path /path/to/cipic.sofa \
  --kemar_path ../renderer/hrtf/kemar.sofa
```

---

## Training

```bash
cd final/configs

# Main training (reproduces paper results)
python train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py \
  --data ../datasets/text2spatial_v4_train.jsonl \
  --epochs 16 --bsz 96 --lr_head 3e-4 --lr_enc 5e-6 \
  --save ../models/my_model.pt

# E5-small variant
python train_e2e_e5small_v3_align.py \
  --data ../datasets/text2spatial_v4_train.jsonl \
  --epochs 28 --bsz 128 --bitfit --unfreeze_last_n 2
```

Training time:
- Single V100 GPU: ~4 hours (16 epochs)
- Batch size 96 with gradient accumulation

---

## Rendering Backend

Supported formats:
- Stereo panning (intensity-based)
- Binaural (HRTF convolution with SOFA datasets)
- First-Order Ambisonics (B-format output)

OSC Export:
Parameters can be streamed to SpatRevolution or other DAWs via OSC protocol for real-time control.

---

## Reproducibility

All results in the paper can be reproduced using:
- Model: `final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt`
- Dataset: `final/datasets/text2spatial_v4_test.jsonl`
- Evaluation: `final/configs/eval_lastmile_v4.py`

Random seeds are fixed in training scripts for deterministic results.

---

## Citation

```bibtex
@inproceedings{paik2026text2spatial,
  title={Natural Language to Spatial Audio Parameters: Lightweight Deterministic Rendering for Creative Authoring},
  author={Paik, Seungryeol and Lee, Kyogu},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

---

## Links

- Demo page: https://paiiek.github.io/mmhoa-demo/
- Paper: [To be added after publication]
- HRTF datasets:
  - CIPIC: https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/
  - KEMAR: https://sound.media.mit.edu/resources/KEMAR.html

---

## Contact

For questions or collaboration:
- Seungryeol Paik: paiiek@snu.ac.kr
- Kyogu Lee: kglee@snu.ac.kr

---

## License

This code is released for academic research purposes. Commercial use requires permission from the authors.

---

## Acknowledgments

- HRTF datasets: CIPIC, KEMAR (MIT Media Lab)
- Pre-trained encoders: Sentence-Transformers, E5 (Microsoft)
- Supported by Seoul National University AI Institute
