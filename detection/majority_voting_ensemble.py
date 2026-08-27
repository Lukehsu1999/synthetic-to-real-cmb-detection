#!/usr/bin/env python3

"""
Ensemble binarized NIfTI segmentation predictions using voxel-wise majority
voting.

For five prediction directories, the default majority threshold requires at
least three folds to predict a voxel as foreground.

Example
-------
python majority_voting_ensemble.py \
    --prediction-dirs \
        /path/to/fold_0_predictions \
        /path/to/fold_1_predictions \
        /path/to/fold_2_predictions \
        /path/to/fold_3_predictions \
        /path/to/fold_4_predictions \
    --output-dir /path/to/majority_vote_predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import SimpleITK as sitk
from tqdm import tqdm


def majority_voting_ensemble_nifti_directories(
    prediction_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    minimum_votes: int | None = None,
    output_suffix: str = "",
    overwrite: bool = False,
) -> None:
    """
    Ensemble binarized NIfTI predictions using voxel-wise majority voting.

    All prediction directories must contain matching filenames. Input masks
    must contain only background value 0 and foreground value 1.

    Parameters
    ----------
    prediction_dirs:
        Directories containing predictions from individual folds or models.
    output_dir:
        Directory in which ensemble predictions will be saved.
    minimum_votes:
        Number of foreground votes required for a voxel to be included in the
        final mask. If None, strict majority voting is used:

            floor(number_of_models / 2) + 1

        Examples:
            5 models -> 3 votes
            4 models -> 3 votes
            3 models -> 2 votes
    output_suffix:
        Optional suffix inserted before the NIfTI extension.
    overwrite:
        Whether existing output files should be overwritten.
    """
    prediction_dirs = [
        Path(path).expanduser().resolve()
        for path in prediction_dirs
    ]
    output_dir = Path(output_dir).expanduser().resolve()

    number_of_models = len(prediction_dirs)

    if number_of_models < 2:
        raise ValueError(
            "Majority voting requires at least two prediction directories."
        )

    if minimum_votes is None:
        minimum_votes = number_of_models // 2 + 1

    if not 1 <= minimum_votes <= number_of_models:
        raise ValueError(
            "--minimum-votes must be between 1 and the number of "
            f"prediction directories ({number_of_models})."
        )

    for directory in prediction_dirs:
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Prediction directory does not exist: {directory}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    reference_files = find_nifti_files(prediction_dirs[0])

    if not reference_files:
        raise FileNotFoundError(
            f"No NIfTI files were found in: {prediction_dirs[0]}"
        )

    validate_directory_filenames(
        prediction_dirs=prediction_dirs,
        reference_files=reference_files,
    )

    print(f"Found {len(reference_files)} cases.")
    print(f"Using {number_of_models} prediction directories.")
    print(f"Minimum foreground votes: {minimum_votes}")
    print(f"Output directory: {output_dir}")

    saved_count = 0
    skipped_count = 0

    progress_bar = tqdm(
        reference_files,
        desc="Majority voting",
        unit="case",
    )

    for reference_path in progress_bar:
        filename = reference_path.name
        output_name = add_nifti_suffix(filename, output_suffix)
        output_path = output_dir / output_name

        if output_path.exists() and not overwrite:
            skipped_count += 1
            progress_bar.set_postfix(
                saved=saved_count,
                skipped=skipped_count,
            )
            continue

        fold_paths = [
            directory / filename
            for directory in prediction_dirs
        ]

        images = [
            sitk.ReadImage(str(path))
            for path in fold_paths
        ]

        validate_matching_geometry(images, fold_paths)
        validate_binary_masks(images, fold_paths)

        binary_images = [
            sitk.Cast(image, sitk.sitkUInt8)
            for image in images
        ]

        # Use a wider integer type to avoid overflow if many models are used.
        vote_sum = sitk.Cast(
            binary_images[0],
            sitk.sitkUInt16,
        )

        for image in binary_images[1:]:
            vote_sum = vote_sum + sitk.Cast(
                image,
                sitk.sitkUInt16,
            )

        ensemble_binary = sitk.Cast(
            vote_sum >= minimum_votes,
            sitk.sitkUInt8,
        )

        ensemble_binary.CopyInformation(images[0])

        sitk.WriteImage(
            ensemble_binary,
            str(output_path),
            useCompression=True,
        )

        saved_count += 1
        progress_bar.set_postfix(
            saved=saved_count,
            skipped=skipped_count,
        )

    print("\nMajority-voting ensembling complete.")
    print(f"Saved:   {saved_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output:  {output_dir}")


def find_nifti_files(directory: Path) -> list[Path]:
    """
    Return all .nii and .nii.gz files in a directory.
    """
    files = {
        path.name: path
        for pattern in ("*.nii", "*.nii.gz")
        for path in directory.glob(pattern)
        if path.is_file()
    }

    return [
        files[filename]
        for filename in sorted(files)
    ]


def validate_directory_filenames(
    prediction_dirs: Sequence[Path],
    reference_files: Sequence[Path],
) -> None:
    """
    Ensure that every prediction directory contains all filenames found in the
    first directory.
    """
    reference_filenames = {
        path.name
        for path in reference_files
    }

    for directory in prediction_dirs[1:]:
        current_filenames = {
            path.name
            for path in find_nifti_files(directory)
        }

        missing_filenames = sorted(
            reference_filenames - current_filenames
        )

        if missing_filenames:
            preview = "\n".join(
                f"  {filename}"
                for filename in missing_filenames[:20]
            )

            additional_count = len(missing_filenames) - 20
            additional_text = (
                f"\n  ... and {additional_count} more"
                if additional_count > 0
                else ""
            )

            raise FileNotFoundError(
                f"Directory is missing "
                f"{len(missing_filenames)} predictions:\n"
                f"{directory}\n"
                f"{preview}"
                f"{additional_text}"
            )


def validate_binary_masks(
    images: Sequence[sitk.Image],
    paths: Sequence[Path],
) -> None:
    """
    Verify that every prediction contains only values 0 and 1.
    """
    for image, path in zip(images, paths):
        statistics = sitk.StatisticsImageFilter()
        statistics.Execute(image)

        minimum = statistics.GetMinimum()
        maximum = statistics.GetMaximum()

        if minimum < 0 or maximum > 1:
            raise ValueError(
                f"Prediction is not binary: {path}\n"
                f"Observed value range: [{minimum}, {maximum}]"
            )

        nonzero_mask = sitk.Cast(
            image != 0,
            sitk.sitkUInt8,
        )
        foreground_mask = sitk.Cast(
            image == 1,
            sitk.sitkUInt8,
        )

        nonzero_statistics = sitk.StatisticsImageFilter()
        nonzero_statistics.Execute(nonzero_mask)

        foreground_statistics = sitk.StatisticsImageFilter()
        foreground_statistics.Execute(foreground_mask)

        nonzero_count = int(nonzero_statistics.GetSum())
        foreground_count = int(foreground_statistics.GetSum())

        if nonzero_count != foreground_count:
            raise ValueError(
                f"Prediction contains values other than 0 and 1: {path}"
            )


def validate_matching_geometry(
    images: Sequence[sitk.Image],
    paths: Sequence[Path],
    *,
    tolerance: float = 1e-5,
) -> None:
    """
    Ensure that all predictions use the same voxel grid and spatial metadata.
    """
    reference = images[0]
    reference_path = paths[0]

    for image, path in zip(images[1:], paths[1:]):
        if image.GetDimension() != reference.GetDimension():
            raise ValueError(
                "Image-dimension mismatch:\n"
                f"  Reference: {reference_path} "
                f"-> {reference.GetDimension()}D\n"
                f"  Current:   {path} "
                f"-> {image.GetDimension()}D"
            )

        if image.GetSize() != reference.GetSize():
            raise ValueError(
                "Image-size mismatch:\n"
                f"  Reference: {reference_path} "
                f"-> {reference.GetSize()}\n"
                f"  Current:   {path} "
                f"-> {image.GetSize()}"
            )

        comparisons = {
            "spacing": (
                reference.GetSpacing(),
                image.GetSpacing(),
            ),
            "origin": (
                reference.GetOrigin(),
                image.GetOrigin(),
            ),
            "direction": (
                reference.GetDirection(),
                image.GetDirection(),
            ),
        }

        for field_name, (
            reference_values,
            current_values,
        ) in comparisons.items():
            if any(
                abs(reference_value - current_value) > tolerance
                for reference_value, current_value in zip(
                    reference_values,
                    current_values,
                )
            ):
                raise ValueError(
                    f"{field_name.capitalize()} mismatch:\n"
                    f"  Reference: {reference_path} "
                    f"-> {reference_values}\n"
                    f"  Current:   {path} "
                    f"-> {current_values}"
                )


def add_nifti_suffix(
    filename: str,
    suffix: str,
) -> str:
    """
    Insert a suffix before the .nii or .nii.gz extension.
    """
    if not suffix:
        return filename

    if filename.endswith(".nii.gz"):
        return f"{filename[:-7]}{suffix}.nii.gz"

    if filename.endswith(".nii"):
        return f"{filename[:-4]}{suffix}.nii"

    raise ValueError(
        f"Unsupported NIfTI filename: {filename}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble binarized NIfTI segmentation predictions from multiple "
            "directories using voxel-wise majority voting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prediction-dirs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Two or more directories containing matching binarized NIfTI "
            "prediction files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory in which ensemble predictions will be saved."
        ),
    )

    parser.add_argument(
        "--minimum-votes",
        type=int,
        default=None,
        help=(
            "Minimum number of foreground votes required. By default, a "
            "strict majority is used. For five folds, the default is 3."
        ),
    )

    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help=(
            "Optional suffix inserted before the NIfTI extension, such as "
            "'_majority'."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite ensemble predictions that already exist."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    try:
        majority_voting_ensemble_nifti_directories(
            prediction_dirs=args.prediction_dirs,
            output_dir=args.output_dir,
            minimum_votes=args.minimum_votes,
            output_suffix=args.output_suffix,
            overwrite=args.overwrite,
        )
    except (
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
    ) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()