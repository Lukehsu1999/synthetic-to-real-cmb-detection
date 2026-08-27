# Synthetic-to-Real Cerebral Microbleed Detection

This repository provides a reproducible pipeline for studying synthetic-to-real transfer in cerebral microbleed (CMB) detection on SWAN MRI. We investigate whether synthetically generated CMBs can provide effective supervision for training models that detect real CMBs.

────────────────────────────────────

## 💡 Synthetic-to-Real CMB Detection

![Overview](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/blob/main/figures/synthetic_supervised_CMB_detection_overview.png)

Synthetic CMBs are inserted into healthy brain scans to create paired synthetic images and segmentation labels. These pairs are used to train a CMB detector, which is then evaluated on real CMB images to measure how well synthetic supervision transfers to real-world detection.

────────────────────────────────────

## 🔬 Pipeline & Repository

[ WORKFLOW ILLUSTRATION ]

| Stage | Directory | What it does | Documentation |
|-------|-----------|--------------|---------------|
| 1 | preprocessing/ | Prepare SWAN MRI and brain masks | [README](./preprocessing/README.md) |
| 2 | synthesis/ | Generate synthetic CMBs | README |
| 3 | analysis/ | Characterize real and synthetic CMBs | README |
| 4 | detection/ | Train, infer, and ensemble detectors | README |
| 5 | evaluation/ | Evaluate predictions and bootstrap CIs | README |

────────────────────────────────────

## 📊 Results
...

## 🚀 Getting Started
...

## 📄 Paper & Citation
...