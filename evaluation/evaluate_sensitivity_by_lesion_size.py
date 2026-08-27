#!/usr/bin/env python3

import argparse
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label


VOLUME_GROUP_ORDER = ["<15", "15-25", "25-40", ">40"]
SUMMARY_GROUP_ORDER = VOLUME_GROUP_ORDER + ["Overall"]


def get_volume_group(volume_mm3: float) -> str:
    """Assign a lesion to a volume category in mm³."""
    if volume_mm3 < 15:
        return "<15"
    if volume_mm3 < 25:
        return "15-25"
    if volume_mm3 < 40:
        return "25-40"
    return ">40"


def strip_nii_extension(filename: str) -> str:
    """Remove either .nii.gz or .nii from a filename."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def find_nifti_files(directory: str) -> list[str]:
    """Find .nii.gz and .nii files directly inside a directory."""
    nii_gz_files = glob.glob(os.path.join(directory, "*.nii.gz"))
    nii_files = glob.glob(os.path.join(directory, "*.nii"))

    return sorted(set(nii_gz_files + nii_files))


def build_prediction_lookup(pred_dir: str) -> dict[str, str]:
    """
    Build a lookup from filename stem to prediction path.

    Examples:
        case001.nii.gz -> case001
        case001.nii    -> case001
    """
    prediction_files = find_nifti_files(pred_dir)
    lookup = {}

    for path in prediction_files:
        stem = strip_nii_extension(os.path.basename(path))

        if stem in lookup:
            raise RuntimeError(
                f"Duplicate prediction stem detected: '{stem}'\n"
                f"Files:\n"
                f"  {lookup[stem]}\n"
                f"  {path}"
            )

        lookup[stem] = path

    return lookup


def get_connectivity_structure(connectivity: int) -> np.ndarray:
    """Create a 3D connected-component structure."""
    if connectivity == 6:
        structure = np.zeros((3, 3, 3), dtype=np.uint8)

        structure[1, 1, 1] = 1
        structure[0, 1, 1] = 1
        structure[2, 1, 1] = 1
        structure[1, 0, 1] = 1
        structure[1, 2, 1] = 1
        structure[1, 1, 0] = 1
        structure[1, 1, 2] = 1

        return structure

    if connectivity == 18:
        structure = np.ones((3, 3, 3), dtype=np.uint8)

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    manhattan_distance = (
                        abs(x - 1)
                        + abs(y - 1)
                        + abs(z - 1)
                    )

                    if manhattan_distance > 2:
                        structure[x, y, z] = 0

        return structure

    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=np.uint8)

    raise ValueError(
        f"Unsupported connectivity: {connectivity}. "
        "Choose 6, 18, or 26."
    )


def evaluate_case(
    gt_path: str,
    pred_path: str,
    connectivity: int = 26,
    min_gt_voxels: int = 1,
) -> list[dict]:
    """
    Evaluate whether each GT lesion overlaps predicted foreground.

    A GT lesion is considered detected when at least one predicted voxel
    overlaps the lesion.
    """
    gt_img = nib.load(gt_path)
    pred_img = nib.load(pred_path)

    gt = np.asarray(gt_img.dataobj) > 0
    pred = np.asarray(pred_img.dataobj) > 0

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch for case '{os.path.basename(gt_path)}':\n"
            f"  GT shape:   {gt.shape}\n"
            f"  Pred shape: {pred.shape}"
        )

    if not np.allclose(
        gt_img.affine,
        pred_img.affine,
        atol=1e-4,
    ):
        print(
            "  Warning: affine mismatch for "
            f"{os.path.basename(gt_path)}"
        )

    voxel_spacing = gt_img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(voxel_spacing))

    structure = get_connectivity_structure(connectivity)

    labeled_gt, num_gt_components = label(
        gt,
        structure=structure,
    )

    case_id = strip_nii_extension(os.path.basename(gt_path))

    rows = []
    retained_lesion_id = 0

    for component_id in range(1, num_gt_components + 1):
        lesion_mask = labeled_gt == component_id
        voxel_count = int(np.count_nonzero(lesion_mask))

        if voxel_count < min_gt_voxels:
            continue

        retained_lesion_id += 1

        volume_mm3 = voxel_count * voxel_volume_mm3

        overlap_voxels = int(
            np.count_nonzero(
                np.logical_and(lesion_mask, pred)
            )
        )

        detected = overlap_voxels > 0

        coordinates = np.argwhere(lesion_mask)
        centroid_voxel = coordinates.mean(axis=0)

        rows.append(
            {
                "case_id": case_id,
                "lesion_id": retained_lesion_id,
                "original_component_id": component_id,
                "volume_voxels": voxel_count,
                "voxel_volume_mm3": voxel_volume_mm3,
                "volume_mm3": volume_mm3,
                "volume_group": get_volume_group(volume_mm3),
                "detected": int(detected),
                "overlap_voxels": overlap_voxels,
                "centroid_x_voxel": float(centroid_voxel[0]),
                "centroid_y_voxel": float(centroid_voxel[1]),
                "centroid_z_voxel": float(centroid_voxel[2]),
                "gt_filename": os.path.basename(gt_path),
                "prediction_filename": os.path.basename(pred_path),
            }
        )

    return rows


def calculate_point_estimate_summary(
    lesion_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate observed stratified and overall sensitivities."""
    rows = []

    for group_name in VOLUME_GROUP_ORDER:
        group_df = lesion_df[
            lesion_df["volume_group"] == group_name
        ]

        gt_lesions = int(len(group_df))
        detected_lesions = int(group_df["detected"].sum())
        missed_lesions = gt_lesions - detected_lesions

        sensitivity = (
            detected_lesions / gt_lesions
            if gt_lesions > 0
            else np.nan
        )

        rows.append(
            {
                "volume_group": group_name,
                "gt_lesions": gt_lesions,
                "detected_lesions": detected_lesions,
                "missed_lesions": missed_lesions,
                "sensitivity": sensitivity,
            }
        )

    overall_gt = int(len(lesion_df))
    overall_detected = int(lesion_df["detected"].sum())

    rows.append(
        {
            "volume_group": "Overall",
            "gt_lesions": overall_gt,
            "detected_lesions": overall_detected,
            "missed_lesions": overall_gt - overall_detected,
            "sensitivity": (
                overall_detected / overall_gt
                if overall_gt > 0
                else np.nan
            ),
        }
    )

    return pd.DataFrame(rows)


def bootstrap_case_level_sensitivity(
    lesion_df: pd.DataFrame,
    all_case_ids: list[str],
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap sensitivity by sampling cases with replacement.

    All evaluated cases are sampled, including cases containing no GT lesions.
    Lesions from a case sampled multiple times receive the same multiplicity.
    """
    if replicates <= 0:
        raise ValueError(
            "--bootstrap_replicates must be greater than zero."
        )

    if not 0 < confidence < 100:
        raise ValueError(
            "--confidence must be between 0 and 100."
        )

    case_ids = np.asarray(all_case_ids, dtype=object)

    if len(case_ids) == 0:
        raise ValueError(
            "No evaluated case IDs were available for bootstrapping."
        )

    rng = np.random.default_rng(seed)

    # Store each case's lesion table once for efficient lookup.
    lesions_by_case = {
        case_id: lesion_df[
            lesion_df["case_id"] == case_id
        ]
        for case_id in case_ids
    }

    replicate_rows = []

    for replicate_index in range(replicates):
        sampled_case_ids = rng.choice(
            case_ids,
            size=len(case_ids),
            replace=True,
        )

        # Count how many times each case was sampled.
        sampled_case_counts = pd.Series(
            sampled_case_ids
        ).value_counts()

        weighted_case_frames = []

        for case_id, multiplicity in sampled_case_counts.items():
            case_df = lesions_by_case[case_id]

            # Negative cases have no lesions and therefore contribute no
            # lesion rows, but remain part of the case-level resampling.
            if case_df.empty:
                continue

            repeated_case_df = pd.concat(
                [case_df] * int(multiplicity),
                ignore_index=True,
            )

            weighted_case_frames.append(repeated_case_df)

        if weighted_case_frames:
            bootstrap_df = pd.concat(
                weighted_case_frames,
                ignore_index=True,
            )
        else:
            bootstrap_df = lesion_df.iloc[0:0].copy()

        for group_name in VOLUME_GROUP_ORDER:
            group_df = bootstrap_df[
                bootstrap_df["volume_group"] == group_name
            ]

            gt_lesions = int(len(group_df))
            detected_lesions = int(group_df["detected"].sum())

            sensitivity = (
                detected_lesions / gt_lesions
                if gt_lesions > 0
                else np.nan
            )

            replicate_rows.append(
                {
                    "replicate": replicate_index + 1,
                    "volume_group": group_name,
                    "gt_lesions": gt_lesions,
                    "detected_lesions": detected_lesions,
                    "sensitivity": sensitivity,
                }
            )

        overall_gt = int(len(bootstrap_df))
        overall_detected = int(
            bootstrap_df["detected"].sum()
        )

        replicate_rows.append(
            {
                "replicate": replicate_index + 1,
                "volume_group": "Overall",
                "gt_lesions": overall_gt,
                "detected_lesions": overall_detected,
                "sensitivity": (
                    overall_detected / overall_gt
                    if overall_gt > 0
                    else np.nan
                ),
            }
        )

    replicate_df = pd.DataFrame(replicate_rows)

    alpha = (100.0 - confidence) / 2.0
    lower_percentile = alpha
    upper_percentile = 100.0 - alpha

    ci_rows = []

    for group_name in SUMMARY_GROUP_ORDER:
        values = replicate_df.loc[
            replicate_df["volume_group"] == group_name,
            "sensitivity",
        ].dropna()

        if values.empty:
            ci_lower = np.nan
            ci_upper = np.nan
            valid_replicates = 0
        else:
            ci_lower = float(
                np.percentile(values, lower_percentile)
            )
            ci_upper = float(
                np.percentile(values, upper_percentile)
            )
            valid_replicates = int(len(values))

        ci_rows.append(
            {
                "volume_group": group_name,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "valid_bootstrap_replicates": valid_replicates,
            }
        )

    ci_df = pd.DataFrame(ci_rows)

    return ci_df, replicate_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate lesion sensitivity stratified by GT lesion "
            "volume, with case-level bootstrap confidence intervals."
        )
    )

    parser.add_argument(
        "--gt_dir",
        required=True,
        help="Directory containing GT NIfTI masks.",
    )

    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing prediction NIfTI masks.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory in which output CSV files will be saved.",
    )

    parser.add_argument(
        "--prefix",
        default="",
        help=(
            "Optional prefix for output filenames, for example "
            "'MONAI_Synth75'."
        ),
    )

    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[6, 18, 26],
        default=26,
        help="Connectivity used to define GT lesions. Default: 26.",
    )

    parser.add_argument(
        "--min_gt_voxels",
        type=int,
        default=1,
        help=(
            "Ignore GT connected components smaller than this number "
            "of voxels. Default: 1."
        ),
    )

    parser.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=10000,
        help=(
            "Number of case-level bootstrap replicates. "
            "Default: 10000."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=95.0,
        help="Confidence interval percentage. Default: 95.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrapping. Default: 42.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.gt_dir):
        raise NotADirectoryError(
            f"GT directory does not exist: {args.gt_dir}"
        )

    if not os.path.isdir(args.pred_dir):
        raise NotADirectoryError(
            f"Prediction directory does not exist: {args.pred_dir}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    gt_files = find_nifti_files(args.gt_dir)
    prediction_lookup = build_prediction_lookup(args.pred_dir)

    print(f"GT directory: {args.gt_dir}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Found {len(gt_files)} GT files")
    print(f"Found {len(prediction_lookup)} prediction files")

    if not gt_files:
        raise FileNotFoundError(
            f"No .nii or .nii.gz files found in: {args.gt_dir}"
        )

    if not prediction_lookup:
        raise FileNotFoundError(
            f"No .nii or .nii.gz files found in: {args.pred_dir}"
        )

    all_rows = []
    evaluated_case_ids = []
    missing_predictions = []
    failed_cases = []

    for index, gt_path in enumerate(gt_files, start=1):
        gt_filename = os.path.basename(gt_path)
        gt_stem = strip_nii_extension(gt_filename)

        pred_path = prediction_lookup.get(gt_stem)

        if pred_path is None:
            missing_predictions.append(gt_filename)

            print(
                f"[{index}/{len(gt_files)}] Missing prediction: "
                f"{gt_filename}"
            )

            continue

        print(
            f"[{index}/{len(gt_files)}] Evaluating: "
            f"{gt_filename}"
        )

        try:
            case_rows = evaluate_case(
                gt_path=gt_path,
                pred_path=pred_path,
                connectivity=args.connectivity,
                min_gt_voxels=args.min_gt_voxels,
            )

        except Exception as error:
            failed_cases.append((gt_filename, str(error)))
            print(f"  Failed: {error}")
            continue

        evaluated_case_ids.append(gt_stem)
        all_rows.extend(case_rows)

        print(f"  Retained GT lesions: {len(case_rows)}")

    print("\nEvaluation summary")
    print("------------------")
    print(f"Matched cases: {len(evaluated_case_ids)}")
    print(f"Missing predictions: {len(missing_predictions)}")
    print(f"Failed cases: {len(failed_cases)}")
    print(f"Total retained GT lesions: {len(all_rows)}")

    if missing_predictions:
        print("\nMissing prediction filenames:")

        for filename in missing_predictions:
            print(f"  {filename}")

    if failed_cases:
        print("\nFailed cases:")

        for filename, error in failed_cases:
            print(f"  {filename}: {error}")

    columns = [
        "case_id",
        "lesion_id",
        "original_component_id",
        "volume_voxels",
        "voxel_volume_mm3",
        "volume_mm3",
        "volume_group",
        "detected",
        "overlap_voxels",
        "centroid_x_voxel",
        "centroid_y_voxel",
        "centroid_z_voxel",
        "gt_filename",
        "prediction_filename",
    ]

    lesion_df = pd.DataFrame(all_rows, columns=columns)

    if lesion_df.empty:
        raise RuntimeError(
            "\nNo GT lesions were evaluated.\n"
            "Possible causes:\n"
            "1. GT and prediction filenames do not match.\n"
            "2. GT masks contain no nonzero voxels.\n"
            "3. --min_gt_voxels is too large.\n"
            "4. All matched cases failed.\n"
        )

    filename_prefix = (
        f"{args.prefix}_"
        if args.prefix
        else ""
    )

    per_lesion_output_path = os.path.join(
        args.output_dir,
        f"{filename_prefix}per_lesion_detection.csv",
    )

    summary_output_path = os.path.join(
        args.output_dir,
        f"{filename_prefix}stratified_sensitivity.csv",
    )

    bootstrap_output_path = os.path.join(
        args.output_dir,
        f"{filename_prefix}"
        "bootstrap_sensitivity_replicates.csv",
    )

    lesion_df.to_csv(
        per_lesion_output_path,
        index=False,
    )

    print(
        "\nBootstrap configuration\n"
        "-----------------------"
    )
    print(f"Sampling unit: case")
    print(f"Cases per replicate: {len(evaluated_case_ids)}")
    print(f"Replicates: {args.bootstrap_replicates}")
    print(f"Confidence: {args.confidence:.1f}%")
    print(f"Seed: {args.seed}")

    ci_df, bootstrap_df = bootstrap_case_level_sensitivity(
        lesion_df=lesion_df,
        all_case_ids=evaluated_case_ids,
        replicates=args.bootstrap_replicates,
        confidence=args.confidence,
        seed=args.seed,
    )

    point_summary_df = calculate_point_estimate_summary(
        lesion_df
    )

    final_summary_df = point_summary_df.merge(
        ci_df,
        on="volume_group",
        how="left",
    )

    final_summary_df["sensitivity_ci"] = (
        final_summary_df.apply(
            lambda row: (
                f"{row['sensitivity']:.4f} "
                f"({row['ci_lower']:.4f}-"
                f"{row['ci_upper']:.4f})"
            )
            if pd.notna(row["sensitivity"])
            and pd.notna(row["ci_lower"])
            and pd.notna(row["ci_upper"])
            else "NA",
            axis=1,
        )
    )

    final_summary_df.to_csv(
        summary_output_path,
        index=False,
    )

    bootstrap_df.to_csv(
        bootstrap_output_path,
        index=False,
    )

    print(
        "\nStratified lesion sensitivity "
        f"({args.confidence:.1f}% case-bootstrap CI)"
    )
    print("------------------------------------------------")

    report_columns = [
        "volume_group",
        "gt_lesions",
        "detected_lesions",
        "missed_lesions",
        "sensitivity",
        "ci_lower",
        "ci_upper",
    ]

    print(
        final_summary_df[report_columns].to_string(
            index=False,
            formatters={
                "sensitivity": lambda value: f"{value:.6f}",
                "ci_lower": lambda value: f"{value:.6f}",
                "ci_upper": lambda value: f"{value:.6f}",
            },
        )
    )

    print("\nFormatted sensitivity with CI")
    print("-----------------------------")

    print(
        final_summary_df[
            ["volume_group", "sensitivity_ci"]
        ].to_string(index=False)
    )

    print("\nSaved output files:")
    print(f"  {per_lesion_output_path}")
    print(f"  {summary_output_path}")
    print(f"  {bootstrap_output_path}")


if __name__ == "__main__":
    main()
    
"""
python evaluate_sensitivity_by_lesion_size.py \
    --gt_dir "/media/volume1/Luke/Microbleed_Data/MixedNormalizedTestSet/labelsTest" \
    --pred_dir "/media/volume1/Luke/Microbleed_Data/MixedNormalizedTestSet/model_predictions/nnUNet_Synth75/nnUNet_Synth75_Ensemble" \
    --output_dir "/media/volume1/Luke/Microbleed_Data/MixedNormalizedTestSet/lesion_metrics" \
    --prefix "nnUNet_Synth75" \
    --connectivity 26 \
    --min_gt_voxels 1 \
    --bootstrap_replicates 10000 \
    --confidence 95 \
    --seed 42
"""