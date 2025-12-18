# Text-to-Spatial Audio: Implementation Guide

**Paper:** "Natural Language to Spatial Audio Parameters: Lightweight Deterministic Rendering for Creative Authoring"
**Authors:** Seungryeol Paik, Kyogu Lee (Seoul National University)
**Conference:** ICASSP 2026

This guide provides **step-by-step instructions** to reproduce the paper results and implement the text-to-spatial audio system.

---

## 📋 Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Model Training](#3-model-training)
4. [Evaluation](#4-evaluation)
5. [Inference & Rendering](#5-inference--rendering)
6. [Reproducing Paper Results](#6-reproducing-paper-results)
7. [Advanced Usage](#7-advanced-usage)

---

## 1. Environment Setup

### 1.1 System Requirements

- **OS:** Linux (Ubuntu 20.04+) or macOS
- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 2080 or better)
- **CPU:** 16+ cores recommended for rendering
- **RAM:** 32GB+
- **Storage:** 50GB free space

### 1.2 Install Dependencies

```bash
# Clone repository
cd /home/seung/mmhoa/text2hoa

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers==4.30.0
pip install librosa==0.10.0
pip install soundfile==0.12.0
pip install pydub==0.25.0
pip install scipy==1.10.0
pip install tqdm==4.65.0
pip install matplotlib==3.7.0
pip install pandas==2.0.0

# Install SOFA tools for HRTF processing
pip install python-sofa==0.2.0
```

### 1.3 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 2. Dataset Preparation

### 2.1 Dataset Format

The model expects JSONL files with the following structure:

```json
{
  "text": "오른쪽 앞에서 조금씩 선명해져",
  "lang": "ko",
  "az_sc": [1.0, 0.0],
  "el_rad": 0.0,
  "dist_m": 1.217,
  "spread_deg": 18.2,
  "wet_mix": 0.23,
  "room_depth": {"drr_db": 8.68},
  "gain_db": -4.98
}
```

**Field descriptions:**
- `text`: Natural language description (Korean or English)
- `lang`: Language code (`"ko"` or `"en"`)
- `az_sc`: Azimuth as `[sin(θ), cos(θ)]` where θ ∈ [-π, π]
- `el_rad`: Elevation in radians (±π/3 for ±60°)
- `dist_m`: Distance in meters (0.6–6.0m)
- `spread_deg`: Sound spread in degrees (5–120°)
- `wet_mix`: Reverberation ratio (0–1)
- `room_depth.drr_db`: Direct-to-Reverberant Ratio in dB
- `gain_db`: Gain in dB

### 2.2 Use Provided Dataset

The paper uses the pre-processed v4 dataset:

```bash
ls -lh final/datasets/
# text2spatial_v4_train.jsonl  (1,092 samples)
# text2spatial_v4_valid.jsonl  (136 samples)
# text2spatial_v4_test.jsonl   (138 samples)
```

### 2.3 Create Custom Dataset (Optional)

```bash
cd utils

# Generate synthetic dataset
python make_dataset_v3_no_subject.py \
  --ko_mult 100 \
  --en_mult 100 \
  --room_mode drr \
  --out custom_dataset.jsonl

# Check dataset statistics
python check_dataset_coverage.py \
  --data custom_dataset.jsonl

# Move to final datasets
mv custom_dataset.jsonl ../final/datasets/
```

### 2.4 Dataset Augmentation

```bash
cd utils

# Label-aware text augmentation
python augment_text_v3_labelaware.py \
  --input ../final/datasets/text2spatial_v4_train.jsonl \
  --output ../final/datasets/text2spatial_v4_train_augmented.jsonl \
  --multiplier 2
```

---

## 3. Model Training

### 3.1 Quick Start: Reproduce Paper Model

**Main model (33.2° angular error):**

```bash
cd final/configs

# Train with paper settings
python train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py \
  --data ../datasets/text2spatial_v4_train.jsonl \
  --epochs 16 \
  --bsz 96 \
  --lr_head 3e-4 \
  --lr_enc 5e-6 \
  --save ../models/my_model.pt
```

**Expected training time:** ~4 hours on 2x RTX 2080 Ti

### 3.2 Training Configuration Explained

The main training script implements:

1. **Encoder:** `paraphrase-multilingual-MiniLM-L12-v2`
   - BitFit fine-tuning (bias + LayerNorm only)
   - Last 2 layers unfrozen

2. **Loss Functions:**
   - **Angular-margin loss** (ArcFace): 12-bin azimuth classification
   - **MAE loss:** Continuous parameters (elevation, distance, spread, wet, gain, room)
   - **Adaptive margins:** Per-bin adjustment based on validation error
   - **Directional contrast loss:** Metric learning for spatial embeddings
   - **Alignment loss:** Regression ↔ ArcFace alignment

3. **Training Strategy:**
   - Coarse-to-fine (C2F): Start with easier bins, progressively refine
   - Directional focus epochs: Emphasize under-represented directions
   - AMP (Automatic Mixed Precision): Faster training with fp16

### 3.3 Monitor Training

```bash
# Training will output logs like:
# epoch 01: train 2.1234 | AE 45.2° | dlog 0.35 | sp 18.5 | wet 0.22 | gain 1.5 | room 2.3
# epoch 16: train 0.8456 | AE 33.2° | dlog 0.26 | sp 13.9 | wet 0.17 | gain 1.0 | room 1.9
#   saved(best AE): ../models/my_model.pt
```

**Metrics:**
- `AE`: Angular error (degrees) → **target: <35°**
- `dlog`: Distance error (log-scaled MAE)
- `sp`: Spread error (degrees)
- `wet`: Wetness error (0-1)
- `gain`: Gain error (dB)
- `room`: Room descriptor error

### 3.4 Alternative: E5-Small Encoder

For better multilingual performance (slightly higher error but more robust):

```bash
cd final/configs

python train_e2e_e5small_v3_align.py \
  --data ../datasets/text2spatial_v4_train.jsonl \
  --room_mode drr \
  --epochs 28 \
  --bsz 128 \
  --lr_head 3e-4 \
  --lr_enc 5e-6 \
  --bitfit \
  --unfreeze_last_n 2 \
  --save ../models/e5_model.pt
```

### 3.5 Stable Baseline Training

For a simpler baseline without advanced techniques:

```bash
cd final/configs

python train_minimal_v7_stable.py \
  --data ../datasets/text2spatial_v4_train.jsonl \
  --room_mode drr \
  --epochs 30 \
  --bsz 32 \
  --lr 2e-5 \
  --save ../models/baseline_model.pt
```

---

## 4. Evaluation

### 4.1 Standard Evaluation

```bash
cd final/configs

# Evaluate on test set
python eval_lastmile_v4.py \
  --data ../datasets/text2spatial_v4_test.jsonl \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --out_metrics eval_results.json

# Results will show:
# N_test: 138
# AE: 33.2°
# dlog: 0.265
# spread_mae: 13.9
# wet_mae: 0.176
# gain_mae: 1.09
```

### 4.2 Out-of-Domain (OOD) Evaluation

Test robustness on challenging inputs:

```bash
cd utils

# Create OOD test sets
python make_ood_textset_v1.py \
  --output ../final/datasets/ood_test.jsonl

# Evaluate
python eval_ood_textset.py \
  --ckpt ../final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --data ../final/datasets/ood_test.jsonl

# Expected results (from paper):
# Numeric instructions: 19.9° AE (good)
# Long sentences: >90° AE (challenging)
# Metaphorical: >95° AE (very challenging)
# Paraphrases: >90° AE (challenging)
```

### 4.3 Cross-HRTF Robustness

Verify model works across different HRTFs:

```bash
cd utils

python eval_hrtf_robustness.py \
  --ckpt ../final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --cipic_hrtf /path/to/cipic.sofa \
  --kemar_hrtf ../renderer/hrtf/kemar.sofa

# Expected: ~33° for both CIPIC and KEMAR
```

### 4.4 Baseline Comparison

Reproduce Table 2 from the paper:

```bash
cd utils

# Run all baselines
python run_baselines_linear_and_rule.py \
  --data ../final/datasets/text2spatial_v4_test.jsonl

# Results:
# Rule-based (keyword): 71.0° AE
# Linear (SBERT): 61.8° AE
# Linear (E5): 76.8° AE
# Proposed: 33.2° AE
```

### 4.5 Encoder Bakeoff

Compare different encoders (Table 3 ablation):

```bash
cd utils

python bakeoff_encoders_v1_fix.py \
  --data ../final/datasets/text2spatial_v4_train.jsonl \
  --epochs 16

# Tests: MiniLM, E5-small, E5-base, GTE-multilingual
```

---

## 5. Inference & Rendering

### 5.1 Simple Inference (Parameter Prediction)

```bash
cd final/configs

# Single text input
python infer_text2spatial_api.py \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --text "오른쪽 앞에서 천천히 다가와"

# Output:
# [
#   {
#     "az_sc": [0.707, 0.707],  # sin, cos
#     "el_rad": 0.0,
#     "dist_m": 2.5,
#     "spread_deg": 25.0,
#     "wet_mix": 0.3,
#     "gain_db": -2.0
#   }
# ]
```

### 5.2 Batch Inference (Multiple Texts)

```bash
# Korean + English batch
python infer_text2spatial_api.py \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --text \
    "왼쪽 뒤에서 멀리 울려퍼져" \
    "approaches from right, getting louder" \
    "앞쪽 가까이에서 타이트하게"
```

### 5.3 Binaural Rendering

Render mono audio to spatial binaural:

```bash
cd final/configs

# Render with HRTF
python infer_render.py \
  --text "오른쪽 위에서 천천히 다가와" \
  --mono_audio input.wav \
  --output output_binaural.wav \
  --hrtf ../../renderer/hrtf/kemar.sofa \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --format binaural

# Check output
ffplay output_binaural.wav  # Use headphones!
```

### 5.4 Stereo Rendering

```bash
python infer_render.py \
  --text "왼쪽에서 넓게 퍼져" \
  --mono_audio input.wav \
  --output output_stereo.wav \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --format stereo
```

### 5.5 FOA (First-Order Ambisonics) Rendering

```bash
python infer_render.py \
  --text "360도로 회전하며 멀어져" \
  --mono_audio input.wav \
  --output output_foa.wav \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --format foa
```

### 5.6 Python API Usage

```python
import torch
from infer_text2spatial_api import T2SInfer

# Load model
inferencer = T2SInfer(
    ckpt="final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt",
    enc_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Single prediction
texts = ["오른쪽 앞에서 조금씩 선명해져"]
params = inferencer.predict(texts)

print(params[0])
# {
#   'az_sc': [0.8, 0.6],
#   'el_rad': 0.0,
#   'dist_m': 1.5,
#   'spread_deg': 20.0,
#   'wet_mix': 0.25,
#   'gain_db': -3.0
# }
```

---

## 6. Reproducing Paper Results

### 6.1 Main Results (Table 2)

```bash
cd final/configs

# Reproduce 33.2° angular error
python eval_lastmile_v4.py \
  --data ../datasets/text2spatial_v4_test.jsonl \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt

# Expected output:
# N_test: 138
# AE: 33.2°
# dlog: 0.265
# spread_mae: 13.9
# wet_mae: 0.176
# gain_mae: 1.09
# room_mae: 1.97
```

### 6.2 Ablation Study (Table 3)

```bash
cd utils

# This script trains 5 ablation models:
# 1. Full model
# 2. -- Angular-margin loss
# 3. -- Adaptive margins
# 4. -- Directional focus
# 5. -- KNN adjustment
python gen_fig_ablation.py \
  --data ../final/datasets/text2spatial_v4_train.jsonl \
  --output_table ablation_results.csv

# Results:
# Full model: 33.2°
# -- Angular-margin: 41.0°
# -- Adaptive margins: 38.7°
# -- Directional focus: 36.8°
# -- KNN adjustment: 37.5°
```

### 6.3 OOD Evaluation (Section 5.4)

```bash
cd utils

python eval_ood_textset.py \
  --ckpt ../final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt

# Expected:
# Numeric: 19.9° AE
# Long: >90° AE
# Metaphorical: >95° AE
# Paraphrased: >90° AE
```

### 6.4 Generate Paper Figures

```bash
cd utils

# Figure: Dataset coverage
python gen_fig_coverage.py \
  --data ../final/datasets/text2spatial_v4_train.jsonl \
  --output ../docs/figures/fig_coverage.png

# Figure: OOD results
python gen_fig_ood.py \
  --metrics ../docs/results/ood_metrics.json \
  --output ../docs/figures/fig_ood.png

# Figure: Ablation comparison
python gen_fig_ablation.py \
  --output ../docs/figures/fig_ablation.png
```

### 6.5 Generate Paper Tables

```bash
cd utils

# Generate all ICASSP tables
python make_icassp_tables.py \
  --metrics_dir ../docs/results \
  --output_dir ../docs/tables

# Output:
# table_main_results.tex
# table_ablation.tex
# table_ood.tex
# table_baselines.tex
```

---

## 7. Advanced Usage

### 7.1 OSC Export for DAWs

Export parameters to SpatRevolution or other OSC-compatible tools:

```python
from infer_text2spatial_api import T2SInfer
from pythonosc import udp_client

# Setup OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

# Predict parameters
inferencer = T2SInfer(ckpt="final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt")
params = inferencer.predict(["오른쪽에서 다가와"])[0]

# Convert to azimuth/elevation in degrees
import math
az_deg = math.degrees(math.atan2(params['az_sc'][0], params['az_sc'][1]))
el_deg = math.degrees(params['el_rad'])

# Send OSC messages
osc_client.send_message("/source/azimuth", az_deg)
osc_client.send_message("/source/elevation", el_deg)
osc_client.send_message("/source/distance", params['dist_m'])
osc_client.send_message("/source/spread", params['spread_deg'])
```

### 7.2 Real-time Streaming (Experimental)

```python
import pyaudio
import numpy as np

# Not real-time yet (1-2 min per 1-min audio)
# Future work: optimize for <100ms latency
```

### 7.3 Custom HRTF Integration

```bash
# Download custom HRTF (SOFA format)
wget https://example.com/custom_hrtf.sofa -O renderer/hrtf/custom.sofa

# Use in rendering
python final/configs/infer_render.py \
  --text "..." \
  --mono_audio input.wav \
  --hrtf renderer/hrtf/custom.sofa \
  --output output.wav
```

### 7.4 Multi-Source Spatialization

```python
# Predict parameters for multiple sources
texts = [
    "왼쪽 앞에서 가까이",
    "오른쪽 뒤에서 멀리",
    "정중앙에서 타이트하게"
]
params_list = inferencer.predict(texts)

# Render each source and mix
# (See infer_and_render_utils.py for implementation)
```

### 7.5 Fine-tuning on Custom Data

```bash
cd final/configs

# Prepare your dataset in JSONL format
# Then fine-tune:
python train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py \
  --data my_custom_dataset.jsonl \
  --load ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --epochs 5 \
  --lr_head 1e-4 \
  --lr_enc 1e-6 \
  --save ../models/finetuned_model.pt
```

---

## 🎯 Quick Reference

### Reproduce Paper (Complete Pipeline)

```bash
# 1. Train model
cd final/configs
python train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py \
  --data ../datasets/text2spatial_v4_train.jsonl \
  --epochs 16 --bsz 96 --save my_model.pt

# 2. Evaluate
python eval_lastmile_v4.py \
  --data ../datasets/text2spatial_v4_test.jsonl \
  --ckpt my_model.pt

# 3. Inference
python infer_text2spatial_api.py \
  --ckpt my_model.pt \
  --text "오른쪽에서 천천히"

# 4. Render
python infer_render.py \
  --text "왼쪽 뒤에서 넓게" \
  --mono_audio test.wav \
  --output spatialized.wav \
  --ckpt my_model.pt
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Reduce batch size
python train_*.py --bsz 32  # instead of 96

# Or use gradient accumulation
python train_*.py --bsz 32 --accum 3  # effective bsz = 96
```

### Issue: Slow Training

```bash
# Use AMP (Automatic Mixed Precision)
# Already enabled in train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py

# Or use fewer workers
python train_*.py --num_workers 0
```

### Issue: ModuleNotFoundError

```bash
# Ensure working directory is correct
cd /home/seung/mmhoa/text2hoa/final/configs
export PYTHONPATH=/home/seung/mmhoa/text2hoa:$PYTHONPATH
```

### Issue: HRTF File Not Found

```bash
# Check path
ls -lh ../../renderer/hrtf/kemar.sofa

# If missing, use provided HRTF or download CIPIC
wget https://www.ece.ucdavis.edu/cipic/data/cipic.sofa
```

---

## 📊 Expected Performance

| Metric | Paper Result | Expected Range |
|--------|-------------|----------------|
| Angular Error (AE) | 33.2° | 32-35° |
| Distance MAE (log) | 0.265 | 0.25-0.28 |
| Spread MAE | 13.9° | 13-15° |
| Wet MAE | 0.176 | 0.17-0.19 |
| Gain MAE | 1.09 dB | 1.0-1.2 dB |
| **Training Time** | 4 hours | 3-5 hours (2x RTX 2080) |
| **Inference Time** | ~50ms/sample | 40-60ms (GPU) |
| **Rendering Time** | 1-2 min/min | CPU-dependent |

---

## 📚 Additional Resources

- **Paper:** [ICASSP 2026 Proceedings - To be published]
- **Demo:** https://paiiek.github.io/mmhoa-demo/
- **HRTF Databases:**
  - CIPIC: https://www.ece.ucdavis.edu/cipic/
  - KEMAR: https://sound.media.mit.edu/resources/KEMAR.html
- **Pretrained Models:**
  - MiniLM: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - E5: https://huggingface.co/intfloat/multilingual-e5-small

---

## 📧 Support

For implementation questions:
- **Email:** paiiek@snu.ac.kr
- **Issues:** [GitHub Issues - To be created]

For paper-related questions:
- **Authors:** Seungryeol Paik, Kyogu Lee
- **Affiliation:** Seoul National University, AI Institute

---

## ✅ Checklist: Paper Reproduction

- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] Dataset downloaded (text2spatial_v4)
- [ ] Model trained (16 epochs, ~4 hours)
- [ ] Evaluation shows AE < 35°
- [ ] Inference works on sample texts
- [ ] Binaural rendering produces output
- [ ] Ablation study reproduced
- [ ] OOD evaluation completed
- [ ] Paper figures generated

**Expected time to full reproduction:** 1-2 days (including training)

---

**Last updated:** 2025-12-18
**Version:** 1.0 (ICASSP 2026 submission)
