# Synthetic-to-Real CMB Analysis

This folder contains scripts and derived statistics for comparing **real
and synthetic cerebral microbleeds (CMBs)**.

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

## Real vs. Synthetic Comparison

```{=html}
<!-- TODO: Add distribution comparison plots -->
```
### Shape comparison

> *Figure placeholder --- real vs. synthetic lesion shape distributions*

### Intensity profile comparison

> *Figure placeholder --- real vs. synthetic intensity/profile
> distributions*
