#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi
from tqdm import tqdm


def nifti_stem(p: Path) -> str:
    if p.name.endswith(".nii.gz"):
        return p.name[:-7]
    if p.name.endswith(".nii"):
        return p.name[:-4]
    return p.stem


def find_image(label_path: Path, image_dir: Path) -> Path | None:
    stem = nifti_stem(label_path)
    for name in [f"{stem}_0000.nii.gz", f"{stem}_0000.nii", f"{stem}.nii.gz", f"{stem}.nii"]:
        p = image_dir / name
        if p.exists():
            return p
    return None


def get_structure(connectivity: int):
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndi.generate_binary_structure(3, 2)
    if connectivity == 26:
        return ndi.generate_binary_structure(3, 3)
    raise ValueError("connectivity must be 6, 18, or 26")


def crop_around_mask(mask: np.ndarray, margin_vox: int = 12):
    coords = np.argwhere(mask)
    mins = np.maximum(coords.min(axis=0) - margin_vox, 0)
    maxs = np.minimum(coords.max(axis=0) + margin_vox + 1, mask.shape)
    return tuple(slice(mins[d], maxs[d]) for d in range(3))


def robust_stats(x: np.ndarray, prefix: str) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p05": float(np.percentile(x, 5)),
        f"{prefix}_p25": float(np.percentile(x, 25)),
        f"{prefix}_p75": float(np.percentile(x, 75)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
    }


def radial_profile(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    brain_mask: np.ndarray,
    spacing,
    max_distance_mm: float = 5.0,
    bin_width_mm: float = 0.5,
):
    dist = ndi.distance_transform_edt(~lesion_mask, sampling=spacing)

    lesion_vals = image[lesion_mask]
    lesion_median = float(np.median(lesion_vals))

    bg_mask = (dist > 1.0) & (dist <= 3.0) & brain_mask
    bg_vals = image[bg_mask]

    bg_median = float(np.median(bg_vals)) if bg_vals.size > 0 else np.nan
    bg_std = float(np.std(bg_vals)) if bg_vals.size > 1 else np.nan

    contrast_abs = bg_median - lesion_median if np.isfinite(bg_median) else np.nan
    contrast_z = contrast_abs / bg_std if np.isfinite(bg_std) and bg_std > 1e-8 else np.nan

    out = {
        "local_bg_median_1_3mm": bg_median,
        "local_bg_std_1_3mm": bg_std,
        "dark_contrast_abs_bg_minus_lesion": contrast_abs,
        "dark_contrast_zscore": contrast_z,
    }

    centers = []
    medians = []

    bins = np.arange(0, max_distance_mm + bin_width_mm, bin_width_mm)

    for lo, hi in zip(bins[:-1], bins[1:]):
        shell = (dist > lo) & (dist <= hi) & brain_mask
        vals = image[shell]

        center = (lo + hi) / 2
        centers.append(center)

        if vals.size > 0:
            mean = float(np.mean(vals))
            median = float(np.median(vals))
            n = int(vals.size)
        else:
            mean = np.nan
            median = np.nan
            n = 0

        medians.append(median)

        key = f"profile_{center:.2f}mm"
        out[f"{key}_mean"] = mean
        out[f"{key}_median"] = median
        out[f"{key}_n"] = n

    centers = np.asarray(centers)
    medians = np.asarray(medians)

    valid_0_2 = np.isfinite(medians) & (centers <= 2.0)

    if valid_0_2.sum() >= 2:
        out["boundary_gradient_0_2mm"] = float(
            np.polyfit(centers[valid_0_2], medians[valid_0_2], deg=1)[0]
        )
    else:
        out["boundary_gradient_0_2mm"] = np.nan

    if np.isfinite(bg_median):
        target = lesion_median + 0.5 * (bg_median - lesion_median)
        valid = np.isfinite(medians)
        candidates = centers[valid & (medians >= target)]
        out["half_recovery_distance_mm"] = float(candidates[0]) if candidates.size > 0 else np.nan
    else:
        out["half_recovery_distance_mm"] = np.nan

    return out


def analyze_case(image_path: Path, label_path: Path, connectivity: int, margin_vox: int):
    img = nib.load(str(image_path))
    lab = nib.load(str(label_path))

    image = img.get_fdata().astype(np.float32)
    label = lab.get_fdata()

    spacing = img.header.get_zooms()[:3]

    all_lesions = label == 1
    brain_mask = (image != 0) | all_lesions

    labeled, n_comp = ndi.label(all_lesions, structure=get_structure(connectivity))

    rows = []

    for comp_id in range(1, n_comp + 1):
        lesion_full = labeled == comp_id
        crop = crop_around_mask(lesion_full, margin_vox=margin_vox)

        image_crop = image[crop]
        lesion_crop = lesion_full[crop]
        brain_crop = brain_mask[crop]

        lesion_vals = image_crop[lesion_crop]

        row = {
            "case_id": nifti_stem(label_path),
            "image_file": image_path.name,
            "label_file": label_path.name,
            "component_id": comp_id,
            "voxel_count": int(lesion_crop.sum()),
            "spacing_x_mm": float(spacing[0]),
            "spacing_y_mm": float(spacing[1]),
            "spacing_z_mm": float(spacing[2]),
        }

        row.update(robust_stats(lesion_vals, "lesion_intensity"))

        row.update(
            radial_profile(
                image=image_crop,
                lesion_mask=lesion_crop,
                brain_mask=brain_crop,
                spacing=spacing,
                max_distance_mm=5.0,
                bin_width_mm=0.5,
            )
        )

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    parser.add_argument("--margin_vox", type=int, default=12)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_csv = Path(args.output_csv)

    label_paths = sorted(list(mask_dir.glob("*.nii")) + list(mask_dir.glob("*.nii.gz")))

    all_rows = []

    for label_path in tqdm(label_paths):
        image_path = find_image(label_path, image_dir)

        if image_path is None:
            print(f"[WARN] no image found for {label_path.name}")
            continue

        try:
            rows = analyze_case(
                image_path=image_path,
                label_path=label_path,
                connectivity=args.connectivity,
                margin_vox=args.margin_vox,
            )
            all_rows.extend(rows)
            print(f"[OK] {label_path.name}: {len(rows)} lesion(s)")
        except Exception as e:
            print(f"[ERROR] {label_path.name}: {e}")

    df = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n=== Summary ===")
    print(f"Total lesions: {len(df)}")

    if len(df) > 0:
        print("\nLesion median intensity:")
        print(df["lesion_intensity_median"].describe())

        print("\nDark contrast, bg - lesion:")
        print(df["dark_contrast_abs_bg_minus_lesion"].describe())

        print("\nBoundary gradient 0-2mm:")
        print(df["boundary_gradient_0_2mm"].describe())

        print("\nHalf recovery distance:")
        print(df["half_recovery_distance_mm"].describe())

    print(f"\nSaved CSV to: {output_csv}")


if __name__ == "__main__":
    main()
    
"""
python analyze_lesion_profiles.py \
  --image_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset301_NormalizedNewTrainSet/imagesTr \
  --mask_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset301_NormalizedNewTrainSet/labelsTr \
  --output_csv /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset301_NormalizedNewTrainSet/stats/microbleed_profile_stats.csv \
  --connectivity 26
"""
