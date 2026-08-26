# Microbleed Detection
For cerebral microbleed (CMB) detection, we use a configurable MONAI 3D U-Net as the primary detection framework, with the self-configuring nnU-Net as a complementary framework for evaluating synthetic-to-real transfer across different segmentation architectures and training pipelines.

# MONAI CMB Detection

Lightweight MONAI-based 3D U-Net training and inference scripts for cerebral microbleed (CMB) segmentation.

## Structure

```text
monai/
├── configs/
│   └── sample_train_config.yaml
├── other_training_scripts/
├── train.py
└── infer.py
```

## Data Format

Training images and labels should be NIfTI files (`.nii` or `.nii.gz`).

The scripts expect nnU-Net-style image naming:

```text
imagesTr/
├── case001_0000.nii.gz
└── case002_0000.nii.gz

labelsTr/
├── case001.nii.gz
└── case002.nii.gz
```

## Training

Training is config-driven. **Do not modify `train.py` for individual experiments.** Instead, copy the sample config and create a new YAML file under `configs/`:

```bash
cp configs/sample_train_config.yaml configs/my_experiment.yaml
```

Edit the new config to specify data paths, model settings, sampling strategy, and training parameters, then run:

```bash
python train.py --config configs/my_experiment.yaml
```

Common settings include ROI size, positive/negative patch sampling, model architecture, learning rate, number of epochs, and K-fold split (`num_folds`, `fold_index`).

Each run is saved under:

```text
<output_root>/<exp_name>/
```

with the resolved config and checkpoints including:

```text
best_model.pt
best_single_dice_model.pt
last_model.pt
config_resolved.yaml
```

Training metrics are logged with Weights & Biases.

## Inference

Run inference using a training config and checkpoint:

```bash
python infer.py \
  --config configs/sample_train_config.yaml \
  --checkpoint runs/<exp_name>/best_model.pt \
  --image_dir /path/to/imagesTs \
  --output_dir /path/to/predictions
```

Sliding-window inference supports Gaussian or constant blending and optional mirror test-time augmentation:

```bash
python infer.py \
  --config configs/sample_train_config.yaml \
  --checkpoint runs/<exp_name>/best_model.pt \
  --image_dir /path/to/imagesTs \
  --output_dir /path/to/predictions \
  --sw_mode gaussian \
  --use_tta
```

Predictions are saved as NIfTI files in the original image space.

## Dependencies

Core dependencies include Python, PyTorch, MONAI, NumPy, NiBabel, PyYAML, and Weights & Biases.

## Supporting Scripts
In /other_training_scripts

# nnUNet
Checkout nnUNet_Microbleed_Segmentation.ipynb for instructions setting up and training a nnUNet for Microbleed Segmentation