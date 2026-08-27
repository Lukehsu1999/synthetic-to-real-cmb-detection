# Evaluation

Scripts for evaluating CMB detection predictions and estimating confidence intervals.

## Workflow

```text
Prediction masks + ground truth
        ↓
evaluate_predictions.py
        ↓
per-case metrics CSV
        ↓
bootstrap_evaluation_ci.py
```

- **`evaluate_predictions.py`** — Computes voxel-, lesion-, and case-level metrics from prediction and ground-truth NIfTI masks.
- **`bootstrap_evaluation_ci.py`** — Computes patient-level bootstrap confidence intervals from the per-case evaluation CSV.
- **`evaluate_sensitivity_by_lesion_size.py`** — Evaluates lesion sensitivity by GT lesion volume with case-bootstrap confidence intervals.

## Example Usage

### 1. Evaluate predictions

```bash
python evaluate_predictions.py \
  --gt_dir /path/to/labelsTest \
  --ensemble_prediction_dir /path/to/predictions \
  --output_csv /path/to/results/per_case_metrics.csv \
  --min_voxels 15
```

### 2. Calculate confidence intervals

```bash
python bootstrap_evaluation_ci.py \
  --input_csv /path/to/results/per_case_metrics.csv \
  --output_csv /path/to/results/bootstrap_ci.csv \
  --n_bootstrap 10000 \
  --seed 42
```

### 3. Evaluate sensitivity by lesion size

```bash
python evaluate_sensitivity_by_lesion_size.py \
  --gt_dir /path/to/labelsTest \
  --pred_dir /path/to/predictions \
  --output_dir /path/to/results/lesion_size \
  --connectivity 26 \
  --bootstrap_replicates 10000
```

Run any script with `--help` for all available options.
