# Text2HOA Repository Structure (Cleaned)

**Last updated:** 2025-12-18
**Status:** ✅ Ready for GitHub release

---

## 📂 Directory Tree

```
text2hoa/
├── 📄 README.md                    # Main documentation
├── 📄 CLEANUP_SUMMARY.md           # Cleanup process details
├── 📄 STRUCTURE.md                 # This file
├── 📄 commands.txt                 # Quick reference commands
│
├── 📁 final/                       # 🎯 Production-ready code
│   ├── models/                     # (3 files, 1.3GB)
│   │   ├── t2sa_e2e_minilm_stage4f_lastmilefocus.pt   # 🏆 MAIN MODEL (33.2° AE)
│   │   ├── t2sa_minilm_ft_clean_lastmile_ep6.pt       # Alternative
│   │   └── t2sa_e2e_e5s_align.pt                      # E5-small variant
│   ├── datasets/                   # (6 files, 1.1MB)
│   │   ├── text2spatial_v4_train.jsonl
│   │   ├── text2spatial_v4_valid.jsonl
│   │   ├── text2spatial_v4_test.jsonl
│   │   ├── text2spatial_v4_stats.json
│   │   ├── text2spatial_v4_qc_report.csv
│   │   └── tiny.jsonl
│   └── configs/                    # (10 Python files)
│       ├── train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py
│       ├── train_e2e_e5small_v3_align.py
│       ├── train_minimal_v7_stable.py
│       ├── eval_lastmile_v4.py
│       ├── eval_lastmile_v4cal.py
│       ├── eval_lastmile_v3_ori_제출용.py
│       ├── infer_text2spatial_api.py
│       ├── infer_render.py
│       ├── infer_and_render_utils.py
│       └── data.py
│
├── 📁 utils/                       # 🔧 Helper scripts (20 files)
│   ├── Dataset creation
│   │   ├── make_dataset_v3_no_subject.py
│   │   ├── augment_text_v3_labelaware.py
│   │   └── make_ood_textset_v1.py
│   ├── Evaluation
│   │   ├── eval_ood_textset.py
│   │   ├── eval_hrtf_robustness.py
│   │   ├── eval_ensemble_calib_v1.py
│   │   └── eval_model_metrics_simple.py
│   ├── Baselines
│   │   ├── bakeoff_encoders_v1_fix.py
│   │   ├── baseline_rulelex_eval.py
│   │   └── run_baselines_linear_and_rule.py
│   ├── Figures & Tables
│   │   ├── gen_fig_ablation.py
│   │   ├── gen_fig_coverage.py
│   │   ├── gen_fig_ood.py
│   │   ├── gen_fig_pipeline.py
│   │   ├── make_icassp_tables.py
│   │   └── csv_to_latex_table.py
│   └── Analysis
│       ├── check_dataset_coverage.py
│       ├── quick_stats_v2.py
│       ├── analyze_mos.py
│       └── summarize_eval_logs.py
│
├── 📁 docs/                        # 📊 Results & documentation
│   ├── results/                    # All metrics and tables
│   │   ├── metrics_final.json
│   │   ├── metrics_all.json
│   │   ├── metrics_base.json
│   │   ├── ood_metrics.json
│   │   ├── baseline_results.csv
│   │   ├── icassp_summary_methods.csv
│   │   ├── icassp_perbin_all.csv
│   │   ├── hrtf_robust_summary.csv
│   │   └── [15+ more metric files]
│   ├── figures/                    # Paper figures (empty, ready for use)
│   └── make_report_md.py.          # Report generator
│
├── 📁 cache/                       # 🗄️ Pre-computed embeddings (73MB)
│   ├── cache_sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2.pt
│   ├── cache_intfloat_multilingual-e5-small.pt
│   ├── cache_intfloat_multilingual-e5-base.pt
│   ├── cache_Alibaba-NLP_gte-multilingual-base.pt
│   ├── cache_e5base.pt
│   └── cache_e5small.pt
│
├── 📁 renderer/                    # 🎵 Spatial audio rendering (4.9GB)
│   ├── hrtf/kemar.sofa
│   ├── mos_questions_*/            # MOS test stimuli (multiple instruments)
│   ├── demo_out_*/                 # Demo outputs
│   └── [rendering scripts]
│
├── 📁 v2/                          # 🧪 LoRA experiments (9.4GB)
│   ├── icassp_run1/, icassp_run2/
│   ├── icassp_run_lora8*/
│   ├── config_spatial.json
│   ├── train_spatial.py
│   └── prepare_spatial_dataset.py
│
├── 📁 v3/                          # 🧪 Latest experiments (20MB)
│
├── 📁 emotion/                     # 😊 Emotion-based spatial audio (13MB)
│   ├── train_weak.jsonl
│   ├── prepare_text2spatial.py
│   └── pro_params_v3.yaml
│
└── 📁 archive/                     # 📦 Historical experiments (~28GB)
    ├── intermediate_models/        # Old checkpoints (stages 1-4e, ablations)
    ├── intermediate_datasets/      # Dataset variants (parts, augmented versions)
    ├── intermediate_scripts/       # Training variants (v1-v6, repair scripts)
    └── old_eval/                   # Previous evaluation scripts
```

---

## 📊 Size Breakdown

| Folder | Size | Description |
|--------|------|-------------|
| `final/` | **1.3GB** | Production models + datasets |
| `cache/` | 73MB | Pre-computed embeddings |
| `docs/` | <1MB | Results, metrics, figures |
| `utils/` | <1MB | Helper scripts |
| `renderer/` | 4.9GB | Audio backend + MOS tests |
| `v2/` | 9.4GB | LoRA experiments |
| `v3/` | 20MB | Latest experiments |
| `emotion/` | 13MB | Emotion project |
| `archive/` | **~28GB** | Historical experiments |
| **Total** | **~44GB** | |

---

## 🎯 Quick Access

### **Run Inference**
```bash
cd final/configs
python infer_text2spatial_api.py \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --text "오른쪽 앞에서 천천히 다가와"
```

### **Evaluate on Test Set**
```bash
cd final/configs
python eval_lastmile_v4.py \
  --data ../datasets/text2spatial_v4_test.jsonl \
  --ckpt ../models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt
```

### **Train New Model**
```bash
cd final/configs
python train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py \
  --data ../datasets/text2spatial_v4_train.jsonl \
  --epochs 16 --bsz 96 --save my_model.pt
```

### **Generate Paper Figures**
```bash
cd utils
python gen_fig_ablation.py
python make_icassp_tables.py
```

---

## 📝 File Categories

### **Production Code** (`final/`)
- ✅ Best trained models
- ✅ Clean train/valid/test splits
- ✅ Main training & evaluation scripts
- ✅ Inference API

### **Utilities** (`utils/`)
- 🔧 Dataset generation & augmentation
- 📊 Evaluation scripts (OOD, HRTF robustness)
- 📈 Figure & table generation
- 🔍 Analysis tools

### **Documentation** (`docs/`)
- 📊 All evaluation metrics (JSON, CSV)
- 🖼️ Paper figures (ready for generation)
- 📝 Summary reports

### **Archive** (`archive/`)
- 🗄️ Old training scripts (v1-v6)
- 🗄️ Intermediate checkpoints
- 🗄️ Dataset variants
- ⚠️ **Not for production use**

---

## ⚠️ Important Notes

### **Main Paper Model**
```
final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt
```
This achieved **33.2° angular error** in the ICASSP 2026 paper.

### **Dataset Split**
- Train: 1,092 samples
- Valid: 136 samples
- Test: 138 samples
- **Total (after augmentation):** ~17,000 samples

### **Nothing Was Deleted**
All files were **moved** to organized folders:
- Critical → `final/`
- Helpers → `utils/`
- Results → `docs/`
- Historical → `archive/`

---

## 🚀 Next Steps for GitHub

### 1. **Create `.gitignore`**
```gitignore
# Large files (use Git LFS)
*.pt
!final/models/*.pt

# Cache
cache/

# Archives
archive/

# Renderer outputs
renderer/demo_out_*
renderer/mos_questions_*

# Python
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

### 2. **Create `requirements.txt`**
```txt
torch>=2.0.0
transformers>=4.30.0
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
numpy>=1.24.0
scipy>=1.10.0
tqdm>=4.65.0
```

### 3. **Add LICENSE**
Choose:
- MIT License (permissive)
- Apache 2.0 (patent protection)
- CC BY-NC 4.0 (academic use only)

### 4. **Host Models Externally**
Options:
- Hugging Face Hub (recommended)
- Zenodo (for reproducibility)
- Google Drive (quick setup)

Update README with download links.

### 5. **Create Demo Notebook**
`demo.ipynb` with:
- Installation instructions
- Inference walkthrough
- Audio rendering examples
- Visualization of predictions

---

## 📧 Maintainers

- **Seungryeol Paik** (paiiek@snu.ac.kr)
- **Kyogu Lee** (kglee@snu.ac.kr)

Seoul National University, AI Institute

---

## ✅ Cleanup Checklist

- [x] All scripts categorized (final vs archive)
- [x] Best models in `final/models/`
- [x] Clean datasets in `final/datasets/`
- [x] Utilities organized by function
- [x] Results moved to `docs/results/`
- [x] Cache files in dedicated folder
- [x] README.md comprehensive
- [x] No critical files deleted
- [x] Root directory clean (4 files only)
- [x] Ready for GitHub release

---

**Status:** ✨ Repository is clean and organized!
