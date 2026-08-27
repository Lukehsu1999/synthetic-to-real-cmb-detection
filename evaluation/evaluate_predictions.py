#!/usr/bin/env python3
"""
Evaluate ensembled binary 3D NIfTI predictions against ground-truth masks.

The output is one CSV row per patient, containing the patient-level statistics
needed for later confidence-interval estimation:

Voxel level
-----------
- voxel_tp, voxel_fp, voxel_fn, voxel_tn
- dice

Lesion level
------------
- gt_lesions
- detected_gt_lesions
- missed_gt_lesions
- pred_lesions
- matched_pred_lesions
- fp_lesions
- lesion_sensitivity
- lesion_precision
- lesion_f1

Case level
----------
- is_cmb_positive: based directly on whether the raw GT mask is non-empty
- pred_case_positive: based on whether at least one predicted component remains
  after min_voxels filtering
- case_tp, case_fp, case_tn, case_fn

Example
-------
python evaluate_predictions.py \
    --gt_dir /path/to/labelsTest \
    --ensemble_prediction_dir /path/to/staple_predictions \
    --output_csv /path/to/staple_per_case_metrics.csv \
    --min_voxels 15
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label
from tqdm import tqdm


def safe_div(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when denominator is zero."""
    return float(numerator / denominator) if denominator > 0 else float("nan")


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def remove_nifti_extension(filename: str) -> str:
    if filename.lower().endswith(".nii.gz"):
        return filename[:-7]
    if filename.lower().endswith(".nii"):
        return filename[:-4]
    return filename


def strip_postfix(filename: str, postfix: str = "") -> str:
    case_id = remove_nifti_extension(filename)
    if postfix and case_id.endswith(postfix):
        case_id = case_id[: -len(postfix)]
    return case_id


def load_binary_mask(path: Path, threshold: float = 0.5) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI image and return a Boolean mask plus its image object."""
    image = nib.load(str(path))
    mask = np.asarray(image.dataobj) > threshold
    return mask, image


def validate_geometry(
    gt_image: nib.Nifti1Image,
    pred_image: nib.Nifti1Image,
    gt_path: Path,
    pred_path: Path,
    affine_tolerance: float = 1e-5,
) -> None:
    """Require matching array shape and approximately matching affine."""
    if gt_image.shape != pred_image.shape:
        raise ValueError(
            f"Shape mismatch for:\n"
            f"  GT:   {gt_path} -> {gt_image.shape}\n"
            f"  Pred: {pred_path} -> {pred_image.shape}"
        )

    if not np.allclose(
        gt_image.affine,
        pred_image.affine,
        atol=affine_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            f"Affine mismatch for:\n"
            f"  GT:   {gt_path}\n"
            f"  Pred: {pred_path}"
        )


def voxel_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    """Calculate voxel-level confusion counts and Dice."""
    gt = gt.astype(bool, copy=False)
    pred = pred.astype(bool, copy=False)

    voxel_tp = int(np.logical_and(gt, pred).sum())
    voxel_fp = int(np.logical_and(~gt, pred).sum())
    voxel_fn = int(np.logical_and(gt, ~pred).sum())
    voxel_tn = int(np.logical_and(~gt, ~pred).sum())

    dice_denominator = 2 * voxel_tp + voxel_fp + voxel_fn
    dice = (
        1.0
        if dice_denominator == 0
        else float((2 * voxel_tp) / dice_denominator)
    )

    return {
        "gt_voxels": int(gt.sum()),
        "pred_voxels": int(pred.sum()),
        "voxel_tp": voxel_tp,
        "voxel_fp": voxel_fp,
        "voxel_fn": voxel_fn,
        "voxel_tn": voxel_tn,
        "dice": dice,
    }


def _valid_components(
    mask: np.ndarray,
    min_voxels: int,
    connectivity: int,
) -> Tuple[np.ndarray, Sequence[int], np.ndarray]:
    """
    Label components and retain only components with at least min_voxels.

    Returns
    -------
    labeled:
        Integer connected-component label image.
    valid_ids:
        IDs of retained components.
    kept_mask:
        Boolean union of all retained components.
    """
    structure = generate_binary_structure(
        rank=mask.ndim,
        connectivity=connectivity,
    )
    labeled, component_count = label(mask.astype(bool), structure=structure)

    if component_count == 0:
        return labeled, [], np.zeros_like(mask, dtype=bool)

    component_sizes = np.bincount(labeled.ravel())
    valid_ids = [
        component_id
        for component_id in range(1, component_count + 1)
        if int(component_sizes[component_id]) >= min_voxels
    ]

    kept_mask = np.isin(labeled, valid_ids) if valid_ids else np.zeros_like(
        mask,
        dtype=bool,
    )
    return labeled, valid_ids, kept_mask


def lesion_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    min_voxels: int = 15,
    connectivity: int = 3,
) -> Dict[str, Any]:
    """
    Calculate connected-component lesion-detection metrics.

    Matching rule
    -------------
    - GT and predicted components smaller than min_voxels are excluded.
    - A retained GT lesion is detected if it overlaps at least one retained
      predicted component.
    - A retained predicted component is a false positive if it overlaps no
      retained GT component.

    This preserves the matching convention used in the original evaluator.
    """
    gt_labeled, gt_ids, gt_kept = _valid_components(
        gt,
        min_voxels=min_voxels,
        connectivity=connectivity,
    )
    pred_labeled, pred_ids, pred_kept = _valid_components(
        pred,
        min_voxels=min_voxels,
        connectivity=connectivity,
    )

    detected_gt_lesions = sum(
        bool(np.logical_and(gt_labeled == component_id, pred_kept).any())
        for component_id in gt_ids
    )
    missed_gt_lesions = len(gt_ids) - detected_gt_lesions

    matched_pred_lesions = sum(
        bool(np.logical_and(pred_labeled == component_id, gt_kept).any())
        for component_id in pred_ids
    )
    fp_lesions = len(pred_ids) - matched_pred_lesions

    lesion_sensitivity = safe_div(detected_gt_lesions, len(gt_ids))
    lesion_precision = safe_div(matched_pred_lesions, len(pred_ids))

    if (
        np.isfinite(lesion_sensitivity)
        and np.isfinite(lesion_precision)
        and lesion_sensitivity + lesion_precision > 0
    ):
        lesion_f1 = float(
            2
            * lesion_sensitivity
            * lesion_precision
            / (lesion_sensitivity + lesion_precision)
        )
    else:
        lesion_f1 = float("nan")

    return {
        "gt_lesions": int(len(gt_ids)),
        "detected_gt_lesions": int(detected_gt_lesions),
        "missed_gt_lesions": int(missed_gt_lesions),
        "pred_lesions": int(len(pred_ids)),
        "matched_pred_lesions": int(matched_pred_lesions),
        "fp_lesions": int(fp_lesions),
        "lesion_sensitivity": lesion_sensitivity,
        "lesion_precision": lesion_precision,
        "lesion_f1": lesion_f1,
    }


def case_metrics(
    is_cmb_positive: bool,
    pred_lesions: int,
) -> Dict[str, int]:
    """
    Calculate one-case classification indicators.

    is_cmb_positive is based on the raw GT mask being non-empty, as requested.
    Prediction positivity is based on at least one retained predicted lesion.
    """
    pred_case_positive = pred_lesions > 0

    return {
        "is_cmb_positive": int(is_cmb_positive),
        "pred_case_positive": int(pred_case_positive),
        "case_tp": int(is_cmb_positive and pred_case_positive),
        "case_fp": int((not is_cmb_positive) and pred_case_positive),
        "case_tn": int((not is_cmb_positive) and (not pred_case_positive)),
        "case_fn": int(is_cmb_positive and (not pred_case_positive)),
    }


def evaluate_ensemble_predictions(
    gt_dir: str | Path,
    ensemble_prediction_dir: str | Path,
    output_csv: str | Path,
    *,
    min_voxels: int = 15,
    connectivity: int = 3,
    threshold: float = 0.5,
    gt_postfix: str = "",
    pred_postfix: str = "",
    strict: bool = True,
) -> List[Dict[str, Any]]:
    """
    Evaluate ensembled NIfTI predictions and save one per-case CSV.

    Parameters
    ----------
    gt_dir:
        Directory containing ground-truth NIfTI masks.

    ensemble_prediction_dir:
        Directory containing ensembled binary prediction NIfTIs.

    output_csv:
        Destination path for the per-case CSV.

    min_voxels:
        Minimum connected-component size retained for lesion- and case-level
        evaluation. Default: 15.

    connectivity:
        scipy.ndimage 3D connectivity: 1, 2, or 3. Default: 3 (26-connected).

    threshold:
        Threshold used when reading masks. This only converts stored arrays to
        Boolean values; it does not alter already-binary predictions.

    gt_postfix / pred_postfix:
        Optional filename suffixes removed before matching cases.

    strict:
        If True, fail on missing predictions, duplicate case IDs, or geometry
        mismatches. If False, warn and skip invalid cases.

    Returns
    -------
    List of per-case metric dictionaries.
    """
    gt_dir = Path(gt_dir)
    ensemble_prediction_dir = Path(ensemble_prediction_dir)
    output_csv = Path(output_csv)

    if not gt_dir.is_dir():
        raise NotADirectoryError(f"GT directory not found: {gt_dir}")
    if not ensemble_prediction_dir.is_dir():
        raise NotADirectoryError(
            f"Prediction directory not found: {ensemble_prediction_dir}"
        )
    if min_voxels < 1:
        raise ValueError("min_voxels must be at least 1.")
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2, or 3.")

    gt_files = sorted(path for path in gt_dir.iterdir() if is_nifti(path))
    pred_files = sorted(
        path for path in ensemble_prediction_dir.iterdir() if is_nifti(path)
    )

    if not gt_files:
        raise FileNotFoundError(f"No NIfTI files found in: {gt_dir}")
    if not pred_files:
        raise FileNotFoundError(
            f"No NIfTI files found in: {ensemble_prediction_dir}"
        )

    pred_map: Dict[str, Path] = {}
    for pred_path in pred_files:
        case_id = strip_postfix(pred_path.name, pred_postfix)
        if case_id in pred_map:
            raise ValueError(
                f"Duplicate prediction case ID '{case_id}':\n"
                f"  {pred_map[case_id]}\n"
                f"  {pred_path}"
            )
        pred_map[case_id] = pred_path

    rows: List[Dict[str, Any]] = []

    for gt_path in tqdm(gt_files, desc="Evaluating ensemble"):
        case_id = strip_postfix(gt_path.name, gt_postfix)
        pred_path = pred_map.get(case_id)

        if pred_path is None:
            message = (
                f"No prediction matched GT case '{case_id}' "
                f"from file '{gt_path.name}'."
            )
            if strict:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}")
            continue

        try:
            gt_mask, gt_image = load_binary_mask(gt_path, threshold=threshold)
            pred_mask, pred_image = load_binary_mask(
                pred_path,
                threshold=threshold,
            )
            validate_geometry(
                gt_image,
                pred_image,
                gt_path,
                pred_path,
            )
        except Exception as exc:
            if strict:
                raise
            print(f"[WARN] Skipping '{case_id}': {exc}")
            continue

        voxel = voxel_metrics(gt_mask, pred_mask)
        lesions = lesion_metrics(
            gt_mask,
            pred_mask,
            min_voxels=min_voxels,
            connectivity=connectivity,
        )

        # This is intentionally based on the unfiltered GT mask.
        is_cmb_positive = bool(gt_mask.any())
        case = case_metrics(
            is_cmb_positive=is_cmb_positive,
            pred_lesions=lesions["pred_lesions"],
        )

        row = {
            "case_id": case_id,
            "gt_file": gt_path.name,
            "pred_file": pred_path.name,
            **case,
            **voxel,
            **lesions,
        }
        rows.append(row)

        print(
            f"[OK] {case_id}: "
            f"positive={case['is_cmb_positive']}, "
            f"dice={voxel['dice']:.4f}, "
            f"detected={lesions['detected_gt_lesions']}/"
            f"{lesions['gt_lesions']}, "
            f"fp_lesions={lesions['fp_lesions']}"
        )

    if not rows:
        raise RuntimeError("No valid matched cases were evaluated.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    positive_count = sum(row["is_cmb_positive"] for row in rows)
    negative_count = len(rows) - positive_count

    print(f"\n[Done] Saved per-case CSV: {output_csv}")
    print(
        f"[Cases] total={len(rows)}, "
        f"CMB-positive={positive_count}, "
        f"CMB-negative={negative_count}"
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ensembled binary NIfTI predictions and save per-case "
            "voxel-, lesion-, and case-level statistics."
        )
    )
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--ensemble_prediction_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--min_voxels", type=int, default=15)
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2, 3),
        default=3,
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gt_postfix", default="")
    parser.add_argument("--pred_postfix", default="")
    parser.add_argument(
        "--skip_invalid",
        action="store_true",
        help="Warn and skip missing or invalid cases instead of failing.",
    )

    args = parser.parse_args()

    evaluate_ensemble_predictions(
        gt_dir=args.gt_dir,
        ensemble_prediction_dir=args.ensemble_prediction_dir,
        output_csv=args.output_csv,
        min_voxels=args.min_voxels,
        connectivity=args.connectivity,
        threshold=args.threshold,
        gt_postfix=args.gt_postfix,
        pred_postfix=args.pred_postfix,
        strict=not args.skip_invalid,
    )


if __name__ == "__main__":
    main()
