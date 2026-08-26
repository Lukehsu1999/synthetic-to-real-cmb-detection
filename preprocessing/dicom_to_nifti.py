import os
import numpy as np
import nibabel as nib
from pydicom.filereader import dcmread
from nibabel.orientations import aff2axcodes

def build_affine_from_dicom(dcms_sorted):
    ds0 = dcms_sorted[0]
    iop = np.array([float(x) for x in ds0.ImageOrientationPatient])  # [r_x,r_y,r_z,c_x,c_y,c_z]
    row_cos = iop[0:3]
    col_cos = iop[3:6]
    slc_cos = np.cross(row_cos, col_cos)

    row_spacing, col_spacing = [float(x) for x in ds0.PixelSpacing]

    # sort along true slice normal using IPP projection
    ipps = np.array([[float(x) for x in d.ImagePositionPatient] for d in dcms_sorted])
    proj = ipps @ slc_cos
    order = np.argsort(proj)
    dcms_sorted = [dcms_sorted[i] for i in order]
    ipps = ipps[order]
    proj = proj[order]

    dz = np.median(np.diff(proj)) if len(dcms_sorted) >= 2 else float(getattr(ds0, "SliceThickness", 1.0))

    # LPS affine (columns correspond to i,j,k axes of data array)
    A_lps = np.eye(4, dtype=float)
    A_lps[0:3, 0] = row_cos * row_spacing
    A_lps[0:3, 1] = col_cos * col_spacing
    A_lps[0:3, 2] = slc_cos * dz
    A_lps[0:3, 3] = ipps[0]  # origin at first slice (LPS)

    # Convert LPS -> RAS
    lps2ras = np.diag([-1, -1, 1, 1])
    A_ras = lps2ras @ A_lps
    return A_ras, dcms_sorted

def dicom_to_nifti_ax_swan_fast(dicom_folder, output_path, target_series=("AX SWAN FAST", "3D Ax SWAN Fast")):
    all_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder)
                 if f.lower().endswith('.dcm') or '.' not in f]

    # group by SeriesDescription
    series = {}
    for f in all_files:
        try:
            ds = dcmread(f, stop_before_pixels=True)
            desc = getattr(ds, "SeriesDescription", "Unknown")
            series.setdefault(desc, []).append(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    # choose series by case-insensitive match against candidates
    chosen_key = None
    keys_lower = {k.lower(): k for k in series.keys()}
    for cand in target_series:
        if cand.lower() in keys_lower:
            chosen_key = keys_lower[cand.lower()]
            break
    if chosen_key is None:
        raise ValueError(f"No series found with description in {target_series}. "
                         f"Available: {list(series.keys())}")

    selected_paths = series[chosen_key]
    dcms = [dcmread(p) for p in selected_paths]

    affine, dcms_sorted = build_affine_from_dicom(dcms)

    # stack as (rows, cols, slices)
    data = np.stack([d.pixel_array for d in dcms_sorted], axis=-1).astype(np.int16)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, output_path)

    print(f"✅ NIfTI saved to: {output_path}")
    print("Orientation codes:", aff2axcodes(affine))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert AX SWAN FAST/3D Ax SWAN Fast DICOM series to NIfTI with correct geometry."
    )
    parser.add_argument("dicom_folder", help="Path to folder containing DICOM files")
    parser.add_argument("output_path", help="Output NIfTI file path (e.g., output.nii.gz)")
    parser.add_argument(
        "--series",
        action="append",
        help=("SeriesDescription candidate to match (case-insensitive). "
              "Use multiple times to provide several candidates. "
              "Defaults to: 'AX SWAN FAST' and '3D Ax SWAN Fast'.")
    )
    args = parser.parse_args()

    targets = tuple(args.series) if args.series else ("AX SWAN FAST", "3D Ax SWAN Fast")
    dicom_to_nifti_ax_swan_fast(args.dicom_folder, args.output_path, target_series=targets)

"""
python dicom_to_nifti.py \
  /path/to/DICOM_folder \
  /path/to/output.nii.gz \
  --series "{series description, e.g., 'AX SWAN FAST'}"

"""
