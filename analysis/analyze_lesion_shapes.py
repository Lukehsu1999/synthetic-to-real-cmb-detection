#!/usr/bin/env python3
"""
Compute 3D shape statistics for true microbleed lesions in NIfTI masks.

Assumptions:
- Only label value == 1 is treated as microbleed
- Each connected component of label 1 is treated as one lesion
- Shape analysis is performed in physical space (mm), not just voxel space

Usage:
    python analyze_lesion_shapes.py \
        --mask_dir /path/to/masks \
        --output_csv /path/to/microbleed_shape_stats.csv

Optional:
    --connectivity 26
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi
from tqdm import tqdm


def nifti_stem(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def get_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    elif connectivity == 18:
        # scipy only directly gives 6 or 26 via rank=1/2/3 logic for 3D,
        # so we use full 26-connectivity for 18 approximation if needed.
        # You can customize this later if strict 18-connectivity matters.
        return ndi.generate_binary_structure(3, 2)
    elif connectivity == 26:
        return ndi.generate_binary_structure(3, 3)
    else:
        raise ValueError("connectivity must be one of: 6, 18, 26")


def equivalent_sphere_diameter_mm(volume_mm3: float) -> float:
    """Diameter of a sphere with the same volume."""
    if volume_mm3 <= 0:
        return 0.0
    radius = ((3.0 * volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return 2.0 * radius


def classify_shape(a: float, b: float, c: float) -> tuple[str, str]:
    """
    a >= b >= c are principal axis lengths in mm.

    Returns:
        shape_class: spherical / mildly_oval / stretched_oval / elongated
        subtype: near_spherical / prolate_like / oblate_like / general_ellipsoid
    """
    eps = 1e-8
    ac = a / max(c, eps)
    ab = a / max(b, eps)
    bc = b / max(c, eps)

    # coarse global class
    if ac < 1.3:
        shape_class = "spherical"
    elif ac < 1.8:
        shape_class = "mildly_oval"
    elif ac < 2.5:
        shape_class = "stretched_oval"
    else:
        shape_class = "elongated"

    # subtype
    if ac < 1.3:
        subtype = "near_spherical"
    elif ab > 1.2 and bc < 1.2:
        subtype = "prolate_like"      # cigar-like: a >> b ≈ c
    elif ab < 1.2 and bc > 1.2:
        subtype = "oblate_like"       # pancake-like: a ≈ b >> c
    else:
        subtype = "general_ellipsoid"

    return shape_class, subtype


def pca_axes_mm_and_vectors(coords_mm: np.ndarray):
    """
    PCA on lesion coordinates in physical mm space.

    Returns:
        axis_lengths_mm: (a, b, c) sorted descending
        eigvecs_sorted:  (3, 3), columns are the unit eigenvectors
                        corresponding to a, b, c
    """
    n = coords_mm.shape[0]
    if n <= 1:
        axis_lengths = (0.0, 0.0, 0.0)
        eigvecs = np.eye(3, dtype=np.float64)
        return axis_lengths, eigvecs

    centered = coords_mm - coords_mm.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)

    # eigh gives both eigenvalues and eigenvectors for symmetric matrices
    eigvals, eigvecs = np.linalg.eigh(cov)

    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    stds = np.sqrt(eigvals)
    axis_lengths = 4.0 * stds  # same descriptive span as before

    return (
        float(axis_lengths[0]),
        float(axis_lengths[1]),
        float(axis_lengths[2]),
    ), eigvecs


def voxel_indices_to_mm(ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel indices (N, 3) -> world coordinates in mm using affine.
    """
    ijk_h = np.concatenate([ijk, np.ones((ijk.shape[0], 1), dtype=np.float64)], axis=1)
    xyz_h = ijk_h @ affine.T
    return xyz_h[:, :3]


def bbox_extents_mm(ijk: np.ndarray, zooms: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Compute bbox extents in mm from voxel index ranges.
    """
    mins = ijk.min(axis=0)
    maxs = ijk.max(axis=0)
    spans_vox = (maxs - mins + 1).astype(np.float64)
    spans_mm = spans_vox * np.array(zooms, dtype=np.float64)
    return float(spans_mm[0]), float(spans_mm[1]), float(spans_mm[2])


def analyze_one_mask(mask_path: Path, connectivity: int) -> list[dict]:
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    affine = img.affine
    zooms = img.header.get_zooms()[:3]

    # only label == 1 is true microbleed
    binary = (data == 1)

    structure = get_structure(connectivity)
    labeled, n_comp = ndi.label(binary, structure=structure)

    rows = []
    voxel_volume_mm3 = float(np.prod(zooms))

    for comp_id in range(1, n_comp + 1):
        ijk = np.argwhere(labeled == comp_id)
        if ijk.shape[0] == 0:
            continue

        voxel_count = int(ijk.shape[0])
        volume_mm3 = voxel_count * voxel_volume_mm3
        eq_diameter_mm = equivalent_sphere_diameter_mm(volume_mm3)

        extent_x_mm, extent_y_mm, extent_z_mm = bbox_extents_mm(ijk, zooms)

        coords_mm = voxel_indices_to_mm(ijk.astype(np.float64), affine)
        (axis_a_mm, axis_b_mm, axis_c_mm), eigvecs = pca_axes_mm_and_vectors(coords_mm)

        major_vec = eigvecs[:, 0]   # unit vector of major axis
        inter_vec = eigvecs[:, 1]
        minor_vec = eigvecs[:, 2]

        vx, vy, vz = major_vec

        # alignment with z-axis
        major_align_z_abs = abs(float(vz))
        major_angle_to_z_deg = float(np.degrees(np.arccos(np.clip(major_align_z_abs, -1.0, 1.0))))

        # in-plane angle of major axis projection
        xy_norm = float(np.sqrt(vx**2 + vy**2))
        if xy_norm < 1e-8:
            major_angle_xy_deg = np.nan
        else:
            major_angle_xy_deg = float(np.degrees(np.arctan2(vy, vx)))

        elongation_ac = axis_a_mm / axis_c_mm if axis_c_mm > 1e-8 else np.nan
        elongation_ab = axis_a_mm / axis_b_mm if axis_b_mm > 1e-8 else np.nan
        elongation_bc = axis_b_mm / axis_c_mm if axis_c_mm > 1e-8 else np.nan

        shape_class, subtype = classify_shape(axis_a_mm, axis_b_mm, axis_c_mm)

        center_voxel = ijk.mean(axis=0)
        center_mm = voxel_indices_to_mm(center_voxel[None, :], affine)[0]

        rows.append({
            "mask_file": mask_path.name,
            "case_id": nifti_stem(mask_path),
            "component_id": comp_id,

            "voxel_count": voxel_count,
            "voxel_volume_mm3": voxel_volume_mm3,
            "volume_mm3": volume_mm3,
            "equivalent_diameter_mm": eq_diameter_mm,

            "spacing_x_mm": float(zooms[0]),
            "spacing_y_mm": float(zooms[1]),
            "spacing_z_mm": float(zooms[2]),

            "bbox_x_mm": extent_x_mm,
            "bbox_y_mm": extent_y_mm,
            "bbox_z_mm": extent_z_mm,

            "axis_major_mm": axis_a_mm,
            "axis_intermediate_mm": axis_b_mm,
            "axis_minor_mm": axis_c_mm,

            "elongation_major_minor": elongation_ac,
            "elongation_major_intermediate": elongation_ab,
            "elongation_intermediate_minor": elongation_bc,
            
            "major_vec_x": float(vx),
            "major_vec_y": float(vy),
            "major_vec_z": float(vz),

            "intermediate_vec_x": float(inter_vec[0]),
            "intermediate_vec_y": float(inter_vec[1]),
            "intermediate_vec_z": float(inter_vec[2]),

            "minor_vec_x": float(minor_vec[0]),
            "minor_vec_y": float(minor_vec[1]),
            "minor_vec_z": float(minor_vec[2]),

            "major_align_z_abs": major_align_z_abs,
            "major_angle_to_z_deg": major_angle_to_z_deg,
            "major_angle_xy_deg": major_angle_xy_deg,

            "shape_class": shape_class,
            "shape_subtype": subtype,

            "center_i": float(center_voxel[0]),
            "center_j": float(center_voxel[1]),
            "center_k": float(center_voxel[2]),

            "center_x_mm": float(center_mm[0]),
            "center_y_mm": float(center_mm[1]),
            "center_z_mm": float(center_mm[2]),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", type=str, required=True, help="Folder of mask NIfTI files")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save lesion-wise CSV")
    parser.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26], help="Connected-component connectivity")
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    mask_paths = sorted(list(mask_dir.glob("*.nii")) + list(mask_dir.glob("*.nii.gz")))

    all_rows = []
    for p in tqdm(mask_paths):
        try:
            rows = analyze_one_mask(p, connectivity=args.connectivity)
            all_rows.extend(rows)
            print(f"[OK] {p.name}: found {len(rows)} lesion(s)")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        print("No lesions found with label == 1.")
    else:
        df = df.sort_values(by=["case_id", "component_id"]).reset_index(drop=True)
        print("\n=== Summary ===")
        print(f"Total lesions: {len(df)}")
        print(df["shape_class"].value_counts(dropna=False))
        print("\nEquivalent diameter (mm):")
        print(df["equivalent_diameter_mm"].describe())
        print("\nElongation major/minor:")
        print(df["elongation_major_minor"].describe())

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV to: {output_csv}")


if __name__ == "__main__":
    main()

