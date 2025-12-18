# Text2HOA Folder Cleanup Summary

**Date:** 2025-12-18
**Original size:** 44GB
**Cleaned structure:** Organized into `final/`, `utils/`, `archive/`

---

## ✅ What Was Done

### 1. **Created New Structure**
```
text2hoa/
├── final/           # Production-ready code
├── utils/           # Helper scripts
├── archive/         # Historical experiments
├── renderer/        # Spatial audio backend (unchanged)
├── v2/, v3/         # Latest experiments (unchanged)
└── emotion/         # Emotion project (unchanged)
```

### 2. **Moved to `final/models/` (Best Models)**
- ✅ `t2sa_e2e_minilm_stage4f_lastmilefocus.pt` - **Main paper model (33.2° AE)**
- ✅ `t2sa_minilm_ft_clean_lastmile_ep6.pt` - Alternative checkpoint
- ✅ `t2sa_e2e_e5s_align.pt` - E5-small variant

### 3. **Moved to `final/datasets/` (Final Data Splits)**
- ✅ `text2spatial_v4_train.jsonl` (1,092 samples)
- ✅ `text2spatial_v4_valid.jsonl` (136 samples)
- ✅ `text2spatial_v4_test.jsonl` (138 samples)
- ✅ `tiny.jsonl` (test set)

### 4. **Moved to `final/configs/` (Production Scripts)**
- ✅ `train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py` - Main training
- ✅ `train_e2e_e5small_v3_align.py` - E5 training
- ✅ `train_minimal_v7_stable.py` - Stable baseline
- ✅ `eval_lastmile_v4.py` - Main evaluation
- ✅ `eval_lastmile_v4cal.py` - Calibrated evaluation
- ✅ `infer_text2spatial_api.py` - Inference API
- ✅ `infer_render.py` - Rendering pipeline
- ✅ `infer_and_render_utils.py` - Rendering utilities
- ✅ `data.py` - Data processing

### 5. **Moved to `utils/` (Helper Scripts)**

#### Dataset Creation & Analysis
- `make_dataset_v3_no_subject.py` - Dataset generation
- `augment_text_v3_labelaware.py` - Text augmentation
- `make_ood_textset_v1.py` - OOD test set creation
- `check_dataset_coverage.py` - Dataset statistics
- `quick_stats_v2.py` - Quick statistics

#### Evaluation & Baselines
- `eval_ood_textset.py` - OOD evaluation
- `eval_hrtf_robustness.py` - Cross-HRTF testing
- `eval_ensemble_calib_v1.py` - Ensemble methods
- `eval_model_metrics_simple.py` - Simple metrics
- `bakeoff_encoders_v1_fix.py` - Encoder comparison
- `baseline_rulelex_eval.py` - Rule-based baseline
- `run_baselines_linear_and_rule.py` - Linear baselines

#### Figure & Table Generation
- `gen_fig_ablation.py` - Ablation figures
- `gen_fig_coverage.py` - Coverage visualization
- `gen_fig_ood.py` - OOD figures
- `gen_fig_pipeline.py` - Pipeline diagram
- `make_icassp_tables.py` - Paper tables
- `csv_to_latex_table.py` - LaTeX conversion

#### Miscellaneous
- `analyze_mos.py` - MOS analysis
- `summarize_eval_logs.py` - Log summarization

### 6. **Moved to `archive/intermediate_scripts/`**

#### Old Training Scripts (v1-v6)
- `train_minimal_v3_amp_dp.py`
- `train_minimal_v4_aux_bitfit.py`
- `train_minimal_v4_aux_bitfit_2.py`
- `train_minimal_v5_dircos.py`
- `train_minimal_v6_azbins.py`
- `train_minimal_v6p_dircurr.py`
- `train_minimal_v7m_memsafe.py`

#### E2E Training Variants
- `train_e2e_e5small_v1.py`
- `train_e2e_e5small_v2_nodp.py`
- `train_head_cached_v1.py` through `v4`
- `train_e2e_minilm_v4_c2f.py`
- `train_e2e_minilm_v5_c2f_adamargin.py`
- `train_e2e_minilm_v5b_c2f_adamargin_focus.py`
- `train_e2e_minilm_v5b_c2f_adamargin_focus_fixed.py`
- `train_e2e_minilm_v5b_c2f_adamargin_focus_fixed3.py`

#### Dataset Repair & Augmentation
- `repair_dataset_v1.py`, `v2_expand.py`
- `rebalance_az12_exact.py`
- `rebalance_labels.py`
- `make_fixed_split.py`
- `make_diagonal_boost.py`
- `make_diag_right_boost.py`
- `make_targeted_aug_diag_v1.py` through `v3`
- `prep_boosted4_clean_and_weights.py`
- `augment_worstbins_v1.py`
- `audit_dataset_v1.py`

### 7. **Moved to `archive/old_eval/`**
- `eval_minimal_v3.py` through `v7`
- `eval_lastmile_v1.py` through `v3`
- `eval_lastmile_v3_fixed.py`
- `eval_lastmile_v4p.py`
- `eval_lastmile_v4cal_align.py`
- `eval_head_cached_v1.py`

### 8. **Moved to `archive/intermediate_models/`**

#### Minimal Models
- `t2sa_minimal_v4.pt` through `v6.pt` (+ train_embs)

#### Head-only Models
- `t2sa_headonly.pt` through `v4hy.pt`

#### E5 Models
- `t2sa_e2e_e5s_aug.pt`
- `t2sa_e5large_ft_clean.pt` (2.1GB)
- `t2sa_e5large_ft_diagR_stage3.pt` (+ train_embs)

#### MiniLM Stages
- `t2sa_e2e_minilm_stage1.pt` through `stage4g_binpush.pt`
- All intermediate stage models (stage4e, stage4e_lasttouch, etc.)

#### Fine-tuned Variants
- `t2sa_minilm_ft_clean.pt`
- `t2sa_minilm_ft_clean_diagboost.pt`
- `t2sa_minilm_ft_clean_lastmile.pt`
- `t2sa_minilm_ft_clean_lastmile_ep10.pt`
- `t2sa_minilm_lastmile_ft.pt`
- All associated train_embs files

#### Ablation Models
- `t2sa_minilm_ood_ablate.pt`
- `t2sa_minilm_ood_ablate2.pt`
- `t2sa_minilm_scratch_clean.pt`

### 9. **Moved to `archive/intermediate_datasets/`**

#### Dataset Parts
- `part0.jsonl` through `part_15.jsonl`
- `en_add2.jsonl`

#### Augmented Versions
- `text2spatial_v3_aug.jsonl`
- `text2spatial_v3_rebalanced.jsonl`
- `text2spatial_v3_fixed.jsonl`
- `text2spatial_v3_balanced*.jsonl`
- `text2spatial_v3_boosted4.jsonl`
- `text2spatial_pool.jsonl`

#### Diagonal Boost Variants
- `diag_boost_*.jsonl` (1036, 147, 14710, 579)
- `diag_right_boost.jsonl`
- `clean_plus_diagR.jsonl`

#### OOD Sets
- `ood_metric_v1.jsonl`
- `ood_qual_v1.jsonl`

#### Predictions
- `preds_proto.jsonl`
- `preds_all.jsonl`
- `preds_ens_pb.jsonl`
- `preds_knn56_open.jsonl`

---

## 🔒 What Was Kept in Root

### Essential Files
- `README.md` - **NEW** comprehensive documentation
- `CLEANUP_SUMMARY.md` - **NEW** this file
- `commands.txt` - Example commands
- `text2spatial_v4_stats.json` - Normalization statistics
- `metrics_*.json` - Evaluation results
- `baseline_results.csv` - Baseline comparison

### Active Scripts (will move to final/ if needed)
- `data.py`
- `eval_lastmile_v3_ori_제출용.py` - Submission version
- `eval_lastmile_v4.py`
- `eval_lastmile_v4cal.py`
- `infer_*.py` (3 files)
- `train_e2e_e5small_v3_align.py`
- `train_e2e_minilm_v5b_c2f_adamargin_focus_fixed2.py`
- `train_minimal_v7_stable.py`

### Cache Files (kept for efficiency)
- `cache_Alibaba-NLP_gte-multilingual-base.pt` (23MB)
- `cache_e5base.pt` (2.1MB)
- `cache_e5small.pt` (1.1MB)
- `cache_intfloat_multilingual-e5-base.pt` (23MB)
- `cache_intfloat_multilingual-e5-small.pt` (12MB)
- `cache_sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2.pt` (12MB)

### Unchanged Folders
- `renderer/` - Spatial audio backend (4.9GB)
- `v2/` - LoRA experiments (9.4GB)
- `v3/` - Latest experiments (20MB)
- `emotion/` - Emotion project (13MB)

---

## 📊 Disk Space Impact

**Before cleanup:**
- Total: ~44GB
- Root files: ~30GB (models + datasets)

**After cleanup:**
- `final/`: ~1.3GB (3 models) + ~1MB (datasets)
- `archive/`: ~28GB (old models + datasets + scripts)
- Root: ~100MB (cache + configs + docs)
- Unchanged: ~14GB (renderer, v2, v3, emotion)

**Space saved in root:** ~29GB → Moved to organized `archive/`

---

## ⚠️ Important Notes

### **NOTHING WAS DELETED**
All files were **moved** to organized folders. Critical files are in:
- `final/` - For immediate use
- `archive/` - For historical reference

### **Recovery Instructions**
If you need an archived file:
```bash
# Find it
find archive/ -name "filename.pt"

# Copy back to root if needed
cp archive/intermediate_models/t2sa_minimal_v4.pt .
```

### **Main Paper Model**
```bash
final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt
```
This is the model that achieved **33.2° angular error** in the paper.

### **Recommended .gitignore**
```gitignore
# Large model checkpoints (use Git LFS or external storage)
*.pt
!final/models/*.pt  # Exception: keep final models

# Cache files
cache_*.pt

# Large datasets
archive/

# Renderer outputs
renderer/demo_out_*
renderer/mos_questions_*

# Python
__pycache__/
*.pyc
.venv/
```

---

## 📝 Next Steps for GitHub Release

1. **Create `.gitignore`** (see above)
2. **Add LICENSE file** (MIT or Academic Use Only)
3. **Host models externally:**
   - Hugging Face Hub
   - Google Drive
   - Zenodo (for reproducibility)
4. **Update README.md** with download links
5. **Create `requirements.txt`:**
   ```
   torch>=2.0.0
   transformers>=4.30.0
   sofa-tools>=0.2.0
   librosa>=0.10.0
   soundfile>=0.12.0
   pydub>=0.25.0
   ```

6. **Add example notebook:**
   - `demo.ipynb` with inference walkthrough
   - Audio samples with spatial rendering

7. **Tag final release:**
   ```bash
   git tag -a v1.0-icassp2026 -m "ICASSP 2026 submission"
   ```

---

## 🎯 Quick Commands

```bash
# Run inference with best model
python final/configs/infer_text2spatial_api.py \
  --ckpt final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt \
  --text "오른쪽 앞에서 조금씩 선명해져"

# Evaluate on test set
python final/configs/eval_lastmile_v4.py \
  --data final/datasets/text2spatial_v4_test.jsonl \
  --ckpt final/models/t2sa_e2e_minilm_stage4f_lastmilefocus.pt

# Generate paper figures
python utils/gen_fig_ablation.py
python utils/make_icassp_tables.py
```

---

## ✅ Verification Checklist

- [x] All training scripts categorized (archive vs final)
- [x] Best models identified and copied to final/
- [x] Final datasets (v4) separated from intermediate versions
- [x] Utility scripts organized by function
- [x] README.md created with comprehensive docs
- [x] No files deleted (only moved/organized)
- [x] Cache files preserved for efficiency
- [x] Renderer folder unchanged (contains HRTF data)
- [x] v2/, v3/, emotion/ folders unchanged

---

**Cleanup completed successfully! ✨**

The repository is now ready for GitHub release with clear structure and documentation.
