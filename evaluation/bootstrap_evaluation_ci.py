#!/usr/bin/env python3
"""
Estimate 95% confidence intervals from the per-case evaluation CSV.

Bootstrap unit
--------------
The patient/case is the resampling unit.

Metric definitions
------------------
1. Mean Dice:
   Mean of case-wise Dice over CMB-positive cases only.

2. Lesion sensitivity:
   Pooled detected GT lesions / pooled total GT lesions over CMB-positive cases.

3. FP per case:
   Total false-positive lesions / total cases over all cases.
   Positive and negative cases are resampled separately to preserve the
   original test-set composition.

Optional additional metrics are also reported:
- case sensitivity
- case specificity
- case precision
- case F1
- lesion precision
- lesion F1

Example
-------
python bootstrap_evaluation_ci.py \
    --input_csv /path/to/staple_per_case_metrics.csv \
    --output_csv /path/to/staple_bootstrap_ci.csv \
    --n_bootstrap 10000 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def percentile_interval(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Return a two-sided percentile bootstrap interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan"), float("nan")

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lower, upper


def mean_positive_dice(df: pd.DataFrame) -> float:
    positive = df[df["is_cmb_positive"] == 1]
    return float(positive["dice"].mean()) if len(positive) else float("nan")


def pooled_lesion_sensitivity(df: pd.DataFrame) -> float:
    positive = df[df["is_cmb_positive"] == 1]
    detected = float(positive["detected_gt_lesions"].sum())
    total_gt = float(positive["gt_lesions"].sum())
    return safe_div(detected, total_gt)


def pooled_lesion_precision(df: pd.DataFrame) -> float:
    matched = float(df["matched_pred_lesions"].sum())
    total_pred = float(df["pred_lesions"].sum())
    return safe_div(matched, total_pred)


def pooled_lesion_f1(df: pd.DataFrame) -> float:
    sensitivity = pooled_lesion_sensitivity(df)
    precision = pooled_lesion_precision(df)

    if (
        np.isfinite(sensitivity)
        and np.isfinite(precision)
        and sensitivity + precision > 0
    ):
        return float(2 * sensitivity * precision / (sensitivity + precision))
    return float("nan")


def fp_per_case(df: pd.DataFrame) -> float:
    return safe_div(float(df["fp_lesions"].sum()), float(len(df)))


def case_sensitivity(df: pd.DataFrame) -> float:
    tp = float(df["case_tp"].sum())
    fn = float(df["case_fn"].sum())
    return safe_div(tp, tp + fn)


def case_specificity(df: pd.DataFrame) -> float:
    tn = float(df["case_tn"].sum())
    fp = float(df["case_fp"].sum())
    return safe_div(tn, tn + fp)


def case_precision(df: pd.DataFrame) -> float:
    tp = float(df["case_tp"].sum())
    fp = float(df["case_fp"].sum())
    return safe_div(tp, tp + fp)


def case_f1(df: pd.DataFrame) -> float:
    tp = float(df["case_tp"].sum())
    fp = float(df["case_fp"].sum())
    fn = float(df["case_fn"].sum())
    return safe_div(2 * tp, 2 * tp + fp + fn)


METRICS: Dict[str, Callable[[pd.DataFrame], float]] = {
    "mean_dice_positive_cases": mean_positive_dice,
    "lesion_sensitivity_pooled": pooled_lesion_sensitivity,
    "fp_lesions_per_case_all": fp_per_case,
    "lesion_precision_pooled": pooled_lesion_precision,
    "lesion_f1_pooled": pooled_lesion_f1,
    "case_sensitivity": case_sensitivity,
    "case_specificity": case_specificity,
    "case_precision": case_precision,
    "case_f1": case_f1,
}


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "is_cmb_positive",
        "dice",
        "detected_gt_lesions",
        "gt_lesions",
        "matched_pred_lesions",
        "pred_lesions",
        "fp_lesions",
        "case_tp",
        "case_fp",
        "case_tn",
        "case_fn",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n  "
            + "\n  ".join(missing)
        )

    positive_values = set(df["is_cmb_positive"].dropna().astype(int).unique())
    if not positive_values.issubset({0, 1}):
        raise ValueError(
            "is_cmb_positive must contain only 0 and 1."
        )


def stratified_patient_bootstrap(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Resample positive and negative patients separately with replacement.

    This preserves the original number of positive and negative cases in every
    bootstrap replicate.
    """
    positive_indices = rng.integers(
        0,
        len(positive_df),
        size=len(positive_df),
    )
    negative_indices = rng.integers(
        0,
        len(negative_df),
        size=len(negative_df),
    )

    positive_sample = positive_df.iloc[positive_indices]
    negative_sample = negative_df.iloc[negative_indices]

    return pd.concat(
        [positive_sample, negative_sample],
        axis=0,
        ignore_index=True,
    )


def bootstrap_confidence_intervals(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Calculate point estimates and percentile bootstrap confidence intervals.

    Parameters
    ----------
    input_csv:
        Per-case CSV created by evaluate_ensemble_predictions.py.

    output_csv:
        Optional destination for a one-row-per-metric summary CSV.

    n_bootstrap:
        Number of patient-level bootstrap replicates.

    confidence_level:
        Confidence level, e.g. 0.95 for a 95% CI.

    seed:
        Random seed for reproducibility.
    """
    input_csv = Path(input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap should be at least 100.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    df = pd.read_csv(input_csv)
    validate_columns(df)

    positive_df = df[df["is_cmb_positive"] == 1].reset_index(drop=True)
    negative_df = df[df["is_cmb_positive"] == 0].reset_index(drop=True)

    if positive_df.empty:
        raise ValueError("No CMB-positive cases found.")
    if negative_df.empty:
        raise ValueError("No CMB-negative cases found.")

    rng = np.random.default_rng(seed)

    bootstrap_values: Dict[str, List[float]] = {
        metric_name: [] for metric_name in METRICS
    }

    for _ in tqdm(range(n_bootstrap)):
        sample_df = stratified_patient_bootstrap(
            positive_df,
            negative_df,
            rng,
        )

        for metric_name, metric_fn in METRICS.items():
            bootstrap_values[metric_name].append(metric_fn(sample_df))

    rows = []

    for metric_name, metric_fn in METRICS.items():
        point_estimate = metric_fn(df)
        lower, upper = percentile_interval(
            bootstrap_values[metric_name],
            confidence_level=confidence_level,
        )

        rows.append(
            {
                "metric": metric_name,
                "point_estimate": point_estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence_level,
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "num_cases": len(df),
                "num_positive_cases": len(positive_df),
                "num_negative_cases": len(negative_df),
            }
        )

    results = pd.DataFrame(rows)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv, index=False)
        print(f"[Done] Saved bootstrap CI CSV: {output_csv}")

    print(
        f"[Dataset] total={len(df)}, "
        f"positive={len(positive_df)}, "
        f"negative={len(negative_df)}"
    )
    print(
        f"[Bootstrap] replicates={n_bootstrap}, "
        f"confidence={confidence_level:.1%}, "
        f"seed={seed}"
    )

    for _, row in results.iterrows():
        print(
            f"{row['metric']}: "
            f"{row['point_estimate']:.4f} "
            f"({confidence_level:.0%} CI "
            f"{row['ci_lower']:.4f}-{row['ci_upper']:.4f})"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate stratified patient-level bootstrap confidence intervals "
            "from the per-case ensemble evaluation CSV."
        )
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    bootstrap_confidence_intervals(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
