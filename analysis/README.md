# Synthetic vs. Real CMB Analysis

This folder contains scripts and derived statistics for comparing **real
and synthetic cerebral microbleeds (CMBs)**.

## Real vs. Synthetic Comparison Results

### Distribution Comparison
![Distribution Comparison](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/blob/main/figures/real_vs_synthetic_characteristics_composite.png)

### ECDF Comparison
![ECDF Comparison](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/blob/main/figures/real_vs_synthetic_ecdf_composite.png)

### IQR & KS-D Test

Values are reported as median (IQR). The Kolmogorov–Smirnov statistic (*D*) 
quantifies the maximum difference between the empirical cumulative 
distributions of real and synthetic lesions.

| Characteristic | Real CMB | Synthetic CMB | KS *D* | *p*-value |
|---|---:|---:|---:|---:|
| Lesion volume (mm³) | 22.86 (14.73–38.46) | 23.52 (12.79–37.58) | 0.08 | 0.084 |
| Elongation ratio | 1.59 (1.40–1.83) | 1.67 (1.41–1.95) | 0.14 | 3.16 × 10⁻⁴ |
| Median lesion intensity | 54.00 (29.00–83.25) | 58.00 (35.00–85.50) | 0.08 | 0.068 |
| Background-to-lesion contrast | 126.00 (100.00–151.00) | 112.00 (94.00–130.00) | 0.22 | 1.50 × 10⁻¹⁰ |

## Contents

``` text
analysis/
├── stats/
│   ├── Realmicrobleed_profile_stats.xlsx
│   ├── Realmicrobleed_shape_stats.xlsx
│   ├── Syntheticmicrobleed_profile_stats.xlsx
│   └── Syntheticmicrobleed_shape_stats.xlsx
├── analyze_lesion_profiles.py
├── analyze_lesion_shapes.py
└── compare_real_synthetic_distributions.ipynb
```

-   **`analyze_lesion_shapes.py`** --- Extracts lesion-level shape
    statistics from CMB masks.
-   **`analyze_lesion_profiles.py`** --- Extracts lesion intensity and
    surrounding intensity-profile statistics from SWAN images and masks.
-   **`compare_real_synthetic_distributions.ipynb`** --- Compares real
    and synthetic CMB characteristics using distribution plots and
    Kolmogorov--Smirnov (KS) statistics.
-   **`stats/`** --- Derived lesion-level statistics used for
    real-vs-synthetic comparison.

## Example Usage

``` bash
python analyze_lesion_shapes.py \
    --mask_dir /path/to/masks \
    --output_csv /path/to/shape_stats.csv
```

``` bash
python analyze_lesion_profiles.py \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --output_csv /path/to/profile_stats.csv
```