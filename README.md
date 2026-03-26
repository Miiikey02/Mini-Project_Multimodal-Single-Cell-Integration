# Multiome Single-Cell Prediction
### NeurIPS 2022 — Open Problems in Single-Cell Multimodal Integration
*PhD application mini-project for Dr. Boxiang Liu's lab, NUS*

---

## Task
Predict RNA expression (23K genes) from chromatin accessibility (228K ATAC-seq peaks)
in single hematopoietic stem and progenitor cells (HSPCs).

**Key challenge:** Generalize to Day 10 — a time point never seen during training.

## Data Splits

| Set | Donors | Days |
|-----|--------|------|
| Train | 13176, 31800, 32606 | 2, 3, 4, 7 |
| Public test | 27678 | 2, 3, 7 |
| Private test | 27678 | **10 (unseen)** |

## Metric
Mean Rowwise Pearson Correlation — Pearson r per cell, averaged across cells.

## Structure
```
data/
  multiome/            # train_multi_inputs.h5, train_multi_targets.h5, test_multi_inputs.h5
  citeseq/             # train/test cite h5 files (not used in this project)
  metadata.csv, evaluation_ids.csv, sample_submission.csv
notebooks/             # Numbered exploration and experiment scripts
src/
  data_utils.py        # H5 loading, metadata, sparsity stats
  evaluate.py          # Pearson metric
  preprocessing.py     # LSI, feature selection (Phase 2)
  models.py            # Ridge, MLP (Phase 3–4)
  train.py             # Training loops (Phase 4)
config.py              # All paths and constants
outputs/
  figures/             # Plots saved by scripts
  models/              # Checkpoints
  predictions/         # Submission CSVs
report/
```

## Quickstart
```bash
pip install -r requirements.txt

# Data lives in ./data/ (multiome/ and citeseq/ subdirs)
python notebooks/01_data_exploration.py
python notebooks/02_preprocessing.py
python notebooks/03_baseline_model.py
```

## 14-Day Timeline
| Days | Goal |
|------|------|
| 1–2  | EDA, data inspection |
| 3–4  | LSI preprocessing, feature selection |
| 5–6  | Ridge regression baseline |
| 7–10 | MLP development + tuning |
| 11–12| Analysis, comparison to benchmarks |
| 13–14| Report writing |
