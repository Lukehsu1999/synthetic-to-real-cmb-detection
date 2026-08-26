# infer_from_config.py

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import argparse
from copy import deepcopy

import yaml
import torch
import numpy as np
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    EnsureTyped,
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference


# -----------------------------
# Config helpers
# -----------------------------

def default_config():
    return {
        "paths": {
            "image_dir": None,       # training image dir; can be reused if --image_dir not passed
            "label_dir": None,
            "output_dir": "./runs/real_mcb_3d_binary_unet",  # training run dir
            "infer_image_dir": None,
            "infer_output_dir": None,
            "checkpoint": None,
        },
        "data": {
            "roi_size": [96, 96, 32],
            "sw_batch_size": 2,
            "sw_overlap": 0.5,
            "sw_mode": "gaussian",  # closer to nnU-Net; use "constant" for old MONAI behavior
        },
        "model": {
            "name": "monai_unet",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [16, 32, 64],
            "strides": [2, 2],
            "num_res_units": 1,
        },
        "training": {
            "threshold": 0.5,
        },
        "inference": {
            "threshold": None,       # if None, falls back to training.threshold
            "use_tta": False,        # mirror TTA over spatial axes
            "tta_axes": [0, 1, 2],
            "save_probabilities": False,
            "prob_suffix": "_prob.nii.gz",
            "pred_suffix": None,     # if None, keeps fname.replace('_0000', '')
        },
    }


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path):
    cfg = default_config()
    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(cfg, user_cfg)


def apply_cli_overrides(cfg, args):
    if args.image_dir is not None:
        cfg["paths"]["infer_image_dir"] = args.image_dir
    if args.output_dir is not None:
        cfg["paths"]["infer_output_dir"] = args.output_dir
    if args.checkpoint is not None:
        cfg["paths"]["checkpoint"] = args.checkpoint
    if args.roi_size is not None:
        cfg["data"]["roi_size"] = args.roi_size
    if args.sw_batch_size is not None:
        cfg["data"]["sw_batch_size"] = args.sw_batch_size
    if args.sw_overlap is not None:
        cfg["data"]["sw_overlap"] = args.sw_overlap
    if args.sw_mode is not None:
        cfg["data"]["sw_mode"] = args.sw_mode
    if args.threshold is not None:
        cfg["inference"]["threshold"] = args.threshold
    if args.use_tta:
        cfg["inference"]["use_tta"] = True
    if args.no_tta:
        cfg["inference"]["use_tta"] = False
    return cfg


def resolve_checkpoint_path(cfg):
    paths = cfg["paths"]
    if paths.get("checkpoint"):
        return paths["checkpoint"]

    run_dir = paths.get("output_dir")
    if not run_dir:
        raise ValueError("Cannot resolve checkpoint: set paths.checkpoint or paths.output_dir.")

    candidates = [
        os.path.join(run_dir, "best_model.pt"),
        os.path.join(run_dir, "best_single_dice_model.pt"),
        os.path.join(run_dir, "last_model.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "No checkpoint found. Tried:\n" + "\n".join(candidates) +
        "\nSet paths.checkpoint in YAML or pass --checkpoint."
    )


def resolve_image_dir(cfg):
    paths = cfg["paths"]
    image_dir = paths.get("infer_image_dir") or paths.get("image_dir")
    if not image_dir:
        raise ValueError("Missing inference image dir. Set paths.infer_image_dir or pass --image_dir.")
    return image_dir


def resolve_output_dir(cfg, checkpoint_path):
    paths = cfg["paths"]
    if paths.get("infer_output_dir"):
        return paths["infer_output_dir"]

    run_dir = paths.get("output_dir") or os.path.dirname(checkpoint_path)
    return os.path.join(run_dir, "model_predictions")


def validate_model_config(model_cfg):
    if model_cfg.get("name", "monai_unet") != "monai_unet":
        raise ValueError(f"Unsupported model.name={model_cfg.get('name')}. Currently supported: monai_unet")
    channels = model_cfg["channels"]
    strides = model_cfg["strides"]
    if len(strides) != len(channels) - 1:
        raise ValueError(
            f"MONAI UNet requires len(strides) == len(channels)-1. "
            f"Got channels={channels}, strides={strides}."
        )


def build_model(model_cfg):
    validate_model_config(model_cfg)
    return UNet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        channels=tuple(model_cfg["channels"]),
        strides=tuple(model_cfg["strides"]),
        num_res_units=int(model_cfg.get("num_res_units", 1)),
    )


# -----------------------------
# Inference helpers
# -----------------------------

def get_infer_transform():
    # Intentionally no CropForegroundd: preserve original image shape for direct NIfTI export.
    # Intentionally no Orientationd: this mirrors your old inference script and avoids saving
    # a reoriented array with the original affine/header.
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"]),
    ])


def predict_logits_sliding_window(image, model, roi_size, sw_batch_size, sw_overlap, sw_mode):
    return sliding_window_inference(
        inputs=image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=sw_overlap,
        mode=sw_mode,
    )


def predict_logits_with_tta(image, model, roi_size, sw_batch_size, sw_overlap, sw_mode, tta_axes):
    # image shape: [B, C, X, Y, Z]. User-facing axes are spatial axes 0/1/2.
    spatial_dims = [2 + int(a) for a in tta_axes]

    flips = [()]
    for ax in spatial_dims:
        flips += [prev + (ax,) for prev in flips.copy()]

    logits_sum = None
    for flip_dims in flips:
        x = torch.flip(image, dims=flip_dims) if flip_dims else image
        logits = predict_logits_sliding_window(x, model, roi_size, sw_batch_size, sw_overlap, sw_mode)
        if flip_dims:
            logits = torch.flip(logits, dims=flip_dims)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    return logits_sum / float(len(flips))


def output_name_from_input(fname, pred_suffix=None):
    base = fname.replace("_0000", "")
    if pred_suffix is None:
        return base

    if base.endswith(".nii.gz"):
        stem = base[:-7]
        return stem + pred_suffix + ".nii.gz"
    if base.endswith(".nii"):
        stem = base[:-4]
        return stem + pred_suffix + ".nii"
    return base + pred_suffix


def save_nifti_like(array_np, reference_nifti, out_path, dtype):
    out = nib.Nifti1Image(array_np.astype(dtype), affine=reference_nifti.affine, header=reference_nifti.header)
    out.set_data_dtype(dtype)
    nib.save(out, out_path)


# -----------------------------
# Main
# -----------------------------

def main(cfg):
    checkpoint_path = resolve_checkpoint_path(cfg)
    image_dir = resolve_image_dir(cfg)
    output_dir = resolve_output_dir(cfg, checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Prefer the config saved inside checkpoint because it is guaranteed to match the weights.
    # If unavailable, use the YAML config.
    if isinstance(ckpt, dict) and "config" in ckpt and "model" in ckpt["config"]:
        model_cfg = deepcopy(ckpt["config"]["model"])
        print("Using model architecture from checkpoint['config']['model'].")
    else:
        model_cfg = deepcopy(cfg["model"])
        print("Using model architecture from YAML config.")

    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    threshold = infer_cfg["threshold"]
    if threshold is None:
        threshold = cfg.get("training", {}).get("threshold", 0.5)

    roi_size = tuple(data_cfg["roi_size"])
    sw_batch_size = int(data_cfg.get("sw_batch_size", 1))
    sw_overlap = float(data_cfg.get("sw_overlap", 0.5))
    sw_mode = data_cfg.get("sw_mode", "gaussian")
    if sw_mode not in ("constant", "gaussian"):
        raise ValueError(f"data.sw_mode must be 'constant' or 'gaussian', got {sw_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image dir: {image_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Model: channels={model_cfg['channels']}, strides={model_cfg['strides']}, num_res_units={model_cfg.get('num_res_units', 1)}")
    print(f"Inference: roi_size={roi_size}, overlap={sw_overlap}, mode={sw_mode}, threshold={threshold}, TTA={infer_cfg.get('use_tta', False)}")

    model = build_model(model_cfg).to(device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
    if len(image_paths) == 0:
        raise RuntimeError(f"No .nii/.nii.gz images found in {image_dir}")
    print(f"Found {len(image_paths)} images.")

    infer_transform = get_infer_transform()

    with torch.no_grad():
        for image_path in image_paths:
            fname = os.path.basename(image_path)
            print(f"Inferencing: {fname}")
            reference_nifti = nib.load(image_path)

            data = infer_transform({"image": image_path})
            image = data["image"].unsqueeze(0).to(device)

            if infer_cfg.get("use_tta", False):
                logits = predict_logits_with_tta(
                    image=image,
                    model=model,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    sw_overlap=sw_overlap,
                    sw_mode=sw_mode,
                    tta_axes=infer_cfg.get("tta_axes", [0, 1, 2]),
                )
            else:
                logits = predict_logits_sliding_window(
                    image=image,
                    model=model,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    sw_overlap=sw_overlap,
                    sw_mode=sw_mode,
                )

            probs = torch.sigmoid(logits)
            pred = (probs > float(threshold)).float()
            pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)

            if pred_np.shape != reference_nifti.shape:
                raise RuntimeError(
                    f"Prediction shape {pred_np.shape} != original shape {reference_nifti.shape}. "
                    "Do not save misaligned prediction."
                )

            out_name = output_name_from_input(fname, infer_cfg.get("pred_suffix", None))
            out_path = os.path.join(output_dir, out_name)
            save_nifti_like(pred_np, reference_nifti, out_path, np.uint8)
            print(f"Saved prediction: {out_path}")

            if infer_cfg.get("save_probabilities", False):
                prob_np = probs[0, 0].cpu().numpy().astype(np.float32)
                prob_name = output_name_from_input(fname, infer_cfg.get("prob_suffix", "_prob.nii.gz"))
                prob_path = os.path.join(output_dir, prob_name)
                save_nifti_like(prob_np, reference_nifti, prob_path, np.float32)
                print(f"Saved probability: {prob_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file used for training.")

    # Optional overrides for inference.
    parser.add_argument("--image_dir", type=str, default=None, help="Inference image dir. Overrides paths.infer_image_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Prediction output dir. Overrides paths.infer_output_dir.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Overrides paths.checkpoint.")
    parser.add_argument("--roi_size", type=int, nargs=3, default=None)
    parser.add_argument("--sw_batch_size", type=int, default=None)
    parser.add_argument("--sw_overlap", type=float, default=None)
    parser.add_argument("--sw_mode", type=str, default=None, choices=["constant", "gaussian"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--use_tta", action="store_true", help="Enable mirror TTA over inference. Can also be set in YAML.")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA even if enabled in YAML.")

    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    main(cfg)
"""
Default: gaussian + mirror TTA
CUDA_VISIBLE_DEVICES=1 python infer_from_config.py \
  --config ./configs/vanilla_light_config.yaml \
  --checkpoint ./runs/vanilla_light_64x64x32_pos1neg1/best_model.pt \
  --image_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset302_NormalizedNewTestSet/imagesTest \
  --output_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset302_NormalizedNewTestSet/model_predictions/vanilla_light_64x64x32_pos1neg1_gaussian_tta \
  --sw_mode gaussian \
  --use_tta
"""
"""
Example 1: constant
python infer_from_config.py \
  --config ./configs/vanilla_light_config.yaml \
  --checkpoint ./runs/real_only_unet3d_yaml_pos4neg1/best_model.pt \
  --image_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset302_NormalizedNewTestSet/imagesTest \
  --output_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset302_NormalizedNewTestSet/model_predictions/real_only_unet3d_yaml_pos4neg1_constant \
  --sw_mode constant
  
Example 2: gaussian
python infer_from_config.py \
  --config ./configs/vanilla_light_config.yaml \
  --checkpoint ./runs/real_only_unet3d_yaml_pos4neg1/best_model.pt \
  --image_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset302_NormalizedNewTestSet/imagesTest \
  --output_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset302_NormalizedNewTestSet/model_predictions/real_only_unet3d_yaml_pos4neg1_gaussian \
  --sw_mode gaussian
  
Example 3: gaussian + mirror TTA
CUDA_VISIBLE_DEVICES=1 python infer_from_config.py \
  --config ./configs/vanilla_light_config.yaml \
  --checkpoint ./runs/real_only_unet3d_yaml_pos4neg1/best_model.pt \
  --image_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset302_NormalizedNewTestSet/imagesTest \
  --output_dir /media/volume1/Luke/Microbleed_nnUNet_Data/Dataset302_NormalizedNewTestSet/model_predictions/real_only_unet3d_yaml_pos4neg1_gaussian_tta \
  --sw_mode gaussian \
  --use_tta
"""