# Synthetic-to-Real Cerebral Microbleed Detection

This repository provides a reproducible pipeline for studying synthetic-to-real transfer in cerebral microbleed (CMB) detection on SWAN MRI. We investigate whether synthetically generated CMBs can provide effective supervision for training models that detect real CMBs.

────────────────────────────────────

## 💡 Synthetic-to-Real CMB Detection

![Overview](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/blob/main/figures/synthetic_supervised_CMB_detection_overview.png)

────────────────────────────────────

## 📊 Results

### Synthetic-to-Real Transfer

| Training Data | Mean Dice ↑ | Lesion Sensitivity ↑ | FP/Case ↓ |
|---|---:|---:|---:|
| Real (MONAI R78) | **0.688** | **0.854** | **1.135** |
| Synthetic (MONAI S75) | 0.576 | 0.749 | 1.423 |

Synthetic-only training retained **87.7% of real-trained lesion sensitivity** without using real CMB annotations for training or validation.

### Architecture Dependency

| Framework | Training | Dice ↑ | Sensitivity ↑ | FP/Case ↓ |
|---|---|---:|---:|---:|
| MONAI 3D U-Net | Real | **0.688** | **0.854** | 1.135 |
| MONAI 3D U-Net | Synthetic | 0.576 | 0.749 | 1.423 |
| nnU-Net | Real | **0.777** | **0.860** | 0.500 |
| nnU-Net | Synthetic | 0.240 | 0.269 | **0.192** |

Synthetic-to-real transfer was strongly **architecture-dependent**.

![Sample Prediction](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/blob/main/figures/MONAI_sample_982880_R3664297_SWAN_rank_01_slice_058.png)

> See the paper for complete experiments, confidence intervals, case-level metrics, and lesion-size-stratified results.

────────────────────────────────────

## 🚀 Getting Started
### Data Availability

Clinical imaging data are not publicly distributed due to privacy and institutional restrictions. For potential research collaborations, please contact the corresponding author, Dr. Wei-Chun Wang (017141@tool.caaumed.org.tw).

### Repository Structure

| Stage | Directory | What it does | Documentation |
|-------|-----------|--------------|---------------|
| 1 | preprocessing/ | Prepare SWAN MRI and brain masks | [README](./preprocessing/README.md) |
| 2 | synthesis/ | Generate synthetic CMBs | [README](./synthesis/README.md) |
| 3 | analysis/ | Characterize real and synthetic CMBs | [README](./analysis/README.md) |
| 4 | detection/ | Train, infer, and ensemble detectors | [README](./detection/README.md) |
| 5 | evaluation/ | Evaluate predictions and bootstrap CIs | [README](./evaluation/README.md) |

## 📄 Paper & Citation
...
