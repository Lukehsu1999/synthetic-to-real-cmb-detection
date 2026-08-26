#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from tqdm import tqdm


def load_nifti(path: Path):
    nii = nib.load(str(path))
    data = nii.get_fdata().astype(np.float32)
    return data, nii.affine, nii.header


def save_nifti(data, affine, header, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data, affine, header=header)
    nib.save(out, str(path))


def nifti_stem(p: Path) -> str:
    if p.name.endswith(".nii.gz"):
        return p.name[:-7]
    if p.name.endswith(".nii"):
        return p.name[:-4]
    return p.stem


def find_corresponding_mask(image_path: Path, mask_dir: Path) -> Path | None:
    stem = nifti_stem(image_path)

    # handle nnUNet-style image names
    if stem.endswith("_0000"):
        stem_no_channel = stem[:-5]
    else:
        stem_no_channel = stem

    candidates = [
        mask_dir / f"{stem}.nii.gz",
        mask_dir / f"{stem}.nii",
        mask_dir / f"{stem_no_channel}.nii.gz",
        mask_dir / f"{stem_no_channel}.nii",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def get_spacing(header):
    z = header.get_zooms()[:3]
    return float(z[0]), float(z[1]), float(z[2])


def equivalent_diameter_to_volume_mm3(d_mm: float) -> float:
    r = d_mm / 2.0
    return (4.0 / 3.0) * np.pi * r**3


def radii_from_volume_and_ratio(volume_mm3: float, ratio: float):
    """
    Ellipsoid radii:
        rx = ry = r
        rz = ratio * r

    Volume = 4/3 pi rx ry rz
           = 4/3 pi ratio r^3
    """
    r = ((3.0 * volume_mm3) / (4.0 * np.pi * ratio)) ** (1.0 / 3.0)
    return float(r), float(r), float(ratio * r)


def sample_mcb_params():
    """
    Based on your true MCB stats:
    - equivalent diameter median ≈ 3.52 mm
    - IQR ≈ 3.04–4.19 mm
    - elongation median ≈ 1.59
    - contrast median ≈ 126
    """

    diameter_mm = np.random.normal(loc=3.6, scale=0.8)
    diameter_mm = float(np.clip(diameter_mm, 2.0, 6.0))

    ratio = np.random.normal(loc=1.6, scale=0.35)
    ratio = float(np.clip(ratio, 1.0, 2.5))

    contrast = np.random.normal(loc=160.0, scale=30.0)
    contrast = float(np.clip(contrast, 90.0, 230.0))

    edge_sigma_mm = np.random.normal(loc=0.3, scale=0.1)
    edge_sigma_mm = float(np.clip(edge_sigma_mm, 0.15, 0.6))

    strength = np.random.normal(loc=0.95, scale=0.05)
    strength = float(np.clip(strength, 0.85, 1.0))

    return diameter_mm, ratio, contrast, edge_sigma_mm, strength


def random_rotation_matrix():
    """
    Random 3D rotation matrix.
    """
    q = np.random.normal(size=4)
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def make_soft_ellipsoid_alpha_crop(
    crop_shape,
    center_crop,
    radii_mm,
    spacing_mm,
    edge_sigma_mm,
    rotation_matrix,
):
    """
    Create soft rotated ellipsoid alpha inside cropped ROI.
    """
    sx, sy, sz = spacing_mm
    rx, ry, rz = radii_mm

    grid = np.indices(crop_shape).astype(np.float32)
    x = (grid[0] - center_crop[0]) * sx
    y = (grid[1] - center_crop[1]) * sy
    z = (grid[2] - center_crop[2]) * sz

    coords = np.stack([x, y, z], axis=0).reshape(3, -1)

    # rotate coordinates into ellipsoid frame
    coords_rot = rotation_matrix.T @ coords

    xr = coords_rot[0].reshape(crop_shape)
    yr = coords_rot[1].reshape(crop_shape)
    zr = coords_rot[2].reshape(crop_shape)

    dist = np.sqrt(
        (xr / rx) ** 2 +
        (yr / ry) ** 2 +
        (zr / rz) ** 2
    )

    signed_mm = (dist - 1.0) * np.mean(radii_mm)

    alpha = 1.0 / (1.0 + np.exp(signed_mm / max(edge_sigma_mm, 1e-6)))
    return alpha.astype(np.float32)


def crop_slices_around_center(shape, center, radius_vox):
    center = np.asarray(center)
    radius_vox = np.asarray(radius_vox)

    mins = np.maximum(np.floor(center - radius_vox).astype(int), 0)
    maxs = np.minimum(np.ceil(center + radius_vox).astype(int) + 1, shape)

    return tuple(slice(mins[d], maxs[d]) for d in range(3)), mins


def sample_valid_center(brain_mask, margin_vox=8):
    safe = brain_mask.copy()

    if margin_vox > 0:
        safe = ndi.binary_erosion(
            safe,
            structure=ndi.generate_binary_structure(3, 1),
            iterations=margin_vox,
        )

    coords = np.argwhere(safe)

    if len(coords) == 0:
        coords = np.argwhere(brain_mask)

    if len(coords) == 0:
        raise ValueError("brain mask is empty")

    return tuple(coords[np.random.randint(len(coords))])


def get_local_background_median(image, brain_mask, center, spacing, inner_mm=3.0, outer_mm=6.0):
    sx, sy, sz = spacing
    radius_vox = int(np.ceil(outer_mm / min(spacing))) + 2

    crop, mins = crop_slices_around_center(
        image.shape,
        center,
        radius_vox=[radius_vox, radius_vox, radius_vox],
    )

    image_crop = image[crop]
    brain_crop = brain_mask[crop]

    cc = np.asarray(center) - mins

    grid = np.indices(image_crop.shape).astype(np.float32)
    dx = (grid[0] - cc[0]) * sx
    dy = (grid[1] - cc[1]) * sy
    dz = (grid[2] - cc[2]) * sz
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    shell = (dist >= inner_mm) & (dist <= outer_mm) & brain_crop

    vals = image_crop[shell]

    if vals.size < 10:
        vals = image_crop[brain_crop]

    if vals.size == 0:
        return float(np.median(image))

    return float(np.median(vals))


def insert_one_mcb(image, lesion_mask, brain_mask, spacing):
    diameter_mm, ratio, contrast, edge_sigma_mm, strength = sample_mcb_params()

    volume_mm3 = equivalent_diameter_to_volume_mm3(diameter_mm)
    radii_mm = radii_from_volume_and_ratio(volume_mm3, ratio)

    max_radius_mm = max(radii_mm) + 3 * edge_sigma_mm
    radius_vox = [
        max_radius_mm / spacing[0] + 3,
        max_radius_mm / spacing[1] + 3,
        max_radius_mm / spacing[2] + 3,
    ]

    center = sample_valid_center(brain_mask, margin_vox=8)

    crop, mins = crop_slices_around_center(
        image.shape,
        center,
        radius_vox=radius_vox,
    )

    image_crop = image[crop]
    brain_crop = brain_mask[crop]

    center_crop = np.asarray(center) - mins

    local_bg = get_local_background_median(
        image=image,
        brain_mask=brain_mask,
        center=center,
        spacing=spacing,
    )

    dark_value = np.clip(local_bg - contrast, 0, 60)

    R = random_rotation_matrix()

    alpha = make_soft_ellipsoid_alpha_crop(
        crop_shape=image_crop.shape,
        center_crop=center_crop,
        radii_mm=radii_mm,
        spacing_mm=spacing,
        edge_sigma_mm=edge_sigma_mm,
        rotation_matrix=R,
    )

    alpha = alpha * brain_crop.astype(np.float32)

    a = np.clip(alpha, 0, 1) * strength

    new_crop = image_crop * (1 - a) + dark_value * a

    image[crop] = new_crop

    # binary GT mask
    lesion_binary = (alpha > 0.5) & brain_crop
    lesion_mask[crop] = np.maximum(lesion_mask[crop], lesion_binary.astype(np.uint8))

    return image, lesion_mask


def process_one_case(
    image_path,
    brainmask_path,
    output_image_dir,
    output_mask_dir,
    num_lesions,
):
    image, affine, header = load_nifti(image_path)
    brainmask, _, _ = load_nifti(brainmask_path)

    brain_mask = brainmask > 0

    spacing = get_spacing(header)

    synth = image.copy()
    lesion_mask = np.zeros(image.shape, dtype=np.uint8)

    for _ in range(num_lesions):
        synth, lesion_mask = insert_one_mcb(
            image=synth,
            lesion_mask=lesion_mask,
            brain_mask=brain_mask,
            spacing=spacing,
        )

    stem = nifti_stem(image_path)
    stem = stem.replace("_0000", "")

    out_img = output_image_dir / f"{stem}_0000.nii.gz"
    out_mask = output_mask_dir / f"{stem}.nii.gz"

    save_nifti(synth.astype(np.float32), affine, header, out_img)
    save_nifti(lesion_mask.astype(np.uint8), affine, header, out_mask)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--healthy_dir", type=Path, required=True)
    parser.add_argument("--brainmask_dir", type=Path, required=True)

    parser.add_argument("--output_image_dir", type=Path, required=True)
    parser.add_argument("--output_mask_dir", type=Path, required=True)

    parser.add_argument("--num_lesions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    image_paths = sorted(
        list(args.healthy_dir.glob("*.nii")) +
        list(args.healthy_dir.glob("*.nii.gz"))
    )

    for image_path in tqdm(image_paths):
        brainmask_path = find_corresponding_mask(image_path, args.brainmask_dir)

        if brainmask_path is None:
            print(f"[WARN] no brainmask found for {image_path.name}")
            continue

        try:
            process_one_case(
                image_path=image_path,
                brainmask_path=brainmask_path,
                output_image_dir=args.output_image_dir,
                output_mask_dir=args.output_mask_dir,
                num_lesions=args.num_lesions,
            )
            print(f"[OK] {image_path.name}")
        except Exception as e:
            print(f"[ERROR] {image_path.name}: {e}")


if __name__ == "__main__":
    main()

"""
python add_3d_synth_mcb_batch_401.py \
  --healthy_dir /media/volume1/Luke/Microbleed_Data/DataSource/HealthyBrain/canonical_3D/train_set \
  --brainmask_dir /media/volume1/Luke/Microbleed_Data/DataSource/HealthyBrain/canonical_3D/cleaned_brainmask \
  --output_image_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset501_SimpleSynth/imagesSynth \
  --output_mask_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset501_SimpleSynth/labelsSynth\
  --num_lesions 5 \
  --seed 42
"""