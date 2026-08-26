# Preprocessing README
![Preprocessing Flow](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/figures/preprocessing_flow.png)

## Steps
### 1. dicom_to_nifti.py
Transform DICOM into Nifti, need to specify which series

### 2. skull_stripping_preprocess.ipynb
Everything else: from training a skull-stripping nnUNet for SWAN, checking the prediction mask, to percentile-based normalization.