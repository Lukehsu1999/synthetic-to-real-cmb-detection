# Synthetic-to-Real Cerebral Microbleed Detection

This repository provides a reproducible pipeline for studying synthetic-to-real transfer in cerebral microbleed (CMB) detection on SWAN MRI. We investigate whether synthetically generated CMBs can provide effective supervision for training models that detect real CMBs.

The clinical imaging data used in this study are not publicly distributed with this repository due to data privacy and institutional restrictions. Researchers interested in collaboration or data access may contact the corresponding author, Dr. Wei-Chun Wang (017141@tool.caaumed.org.tw), to discuss potential research collaborations.

────────────────────────────────────

## 💡 Synthetic-to-Real CMB Detection

![Overview](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/blob/main/figures/synthetic_supervised_CMB_detection_overview.png)

Synthetic CMBs are inserted into healthy brain scans to create paired synthetic images and segmentation labels. These pairs are used to train a CMB detector, which is then evaluated on real CMB images to measure how well synthetic supervision transfers to real-world detection.

────────────────────────────────────

## 🔬 Pipeline & Repository

| Stage | Directory | What it does | Documentation |
|-------|-----------|--------------|---------------|
| 1 | preprocessing/ | Prepare SWAN MRI and brain masks | [README](./preprocessing/README.md) |
| 2 | synthesis/ | Generate synthetic CMBs | [README](./synthesis/README.md) |
| 3 | analysis/ | Characterize real and synthetic CMBs | [README](./analysis/README.md) |
| 4 | detection/ | Train, infer, and ensemble detectors | [README](./detection/README.md) |
| 5 | evaluation/ | Evaluate predictions and bootstrap CIs | [README](./evaluation/README.md) |

────────────────────────────────────

## 📊 Results
### Key Findings
- **Synthetic-only training transferred to real CMB detection.** With the MONAI 3D U-Net, synthetic-only training retained **87.7% of the lesion sensitivity** achieved by real-data training.
- **Real data remained more effective.** Real-trained models consistently outperformed synthetic-trained models across the evaluated training-set sizes.
- **Synthetic-to-real transfer was architecture-dependent.** MONAI retained substantial performance under synthetic-only training, whereas nnU-Net showed a much larger performance drop.

### Synthetic-to-Real Transfer

| Training Data | Mean Dice ↑ | Lesion Sensitivity ↑ | FP/Case ↓ |
|---|---:|---:|---:|
| Real (MONAI R78) | **0.688** | **0.854** | **1.135** |
| Synthetic (MONAI S75) | 0.576 | 0.749 | 1.423 |

Synthetic-only training retained **87.7% of the lesion sensitivity** of real-data training despite using no real CMB annotations during training or validation.

### Transfer Depends on Detection Framework

| Framework | Training Data | Mean Dice ↑ | Lesion Sensitivity ↑ | FP/Case ↓ |
|---|---|---:|---:|---:|
| MONAI 3D U-Net | Real (R78) | **0.688** | **0.854** | 1.135 |
| MONAI 3D U-Net | Synthetic (S75) | 0.576 | 0.749 | 1.423 |
| nnU-Net | Real (R78) | **0.777** | **0.860** | 0.500 |
| nnU-Net | Synthetic (S75) | 0.240 | 0.269 | **0.192** |


> For complete experiments, 95% bootstrap confidence intervals, case-level metrics, and lesion-size-stratified results, see the paper.
> For more sample images, checked out our website.

![Sample Prediction](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/blob/main/figures/MONAI_sample_982880_R3664297_SWAN_rank_01_slice_058.png)

## 🚀 Getting Started
...

## 📄 Paper & Citation
...