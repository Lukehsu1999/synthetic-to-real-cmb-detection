# train_monai_3d_unet_real_mcb_yaml.py

# train_monai_3d_unet_real_mcb_yaml.py

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import random
import argparse
from datetime import datetime
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import wandb
import yaml

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism


# -----------------------------
# Config helpers
# -----------------------------

def default_config():
    return {
        "paths": {
            "image_dir": None,
            "label_dir": None,
            # Do not set output_dir manually.
            # It will be resolved as: output_root / experiment.exp_name
            "output_root": "/home/t101977/microbleed-detection/experiments/3d_seg/runs",
        },
        "experiment": {
            "exp_name": None,
            "wandb_project": "microbleed-3d",
            "wandb_mode": "online",
            "wandb_watch": False,
        },
        "data": {
            "val_ratio": 0.2,
            "seed": 42,
            "training_seed": 43,
            "roi_size": [96, 96, 32],
            "batch_size": 1,
            "sw_batch_size": 2,
            "sw_overlap": 0.5,
            "num_workers": 4,
            "num_folds": 1,
            "fold_index": 0,
        },
        "sampling": {
            "pos": 4,
            "neg": 1,
            "num_samples": 8,
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
            "max_epochs": 100,
            "validate_every_epoch": True,
            "val_interval": 1,
            "log_interval": 20,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "dice_smooth": 1.0,
            "threshold": 0.5,
            "best_ma_window": 5,
            "save_single_best_also": True,
        },
        "scheduler": {
            # Minimal stabilizing intervention.
            # Options: "none", "cosine", "reduce_on_plateau".
            "name": "cosine",
            "eta_min": 1e-5,
            # Used only by reduce_on_plateau.
            "mode": "max",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-5,
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
    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, user_cfg)
    return cfg


def flatten_config(cfg):
    flat = {}
    for section, values in cfg.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[f"{section}.{k}"] = v
        else:
            flat[section] = values
    return flat


def apply_cli_overrides(cfg, args):
    # Keep the script convenient: common paths can still be overridden by CLI.
    if args.image_dir is not None:
        cfg["paths"]["image_dir"] = args.image_dir
    if args.label_dir is not None:
        cfg["paths"]["label_dir"] = args.label_dir
    if args.output_root is not None:
        cfg["paths"]["output_root"] = args.output_root
    if args.exp_name is not None:
        cfg["experiment"]["exp_name"] = args.exp_name
    if args.wandb_mode is not None:
        cfg["experiment"]["wandb_mode"] = args.wandb_mode
    if getattr(args, "num_folds", None) is not None:
        cfg["data"]["num_folds"] = args.num_folds
    if getattr(args, "fold_index", None) is not None:
        cfg["data"]["fold_index"] = args.fold_index
    return cfg


def validate_config(cfg):
    if not cfg["paths"]["image_dir"]:
        raise ValueError("Missing paths.image_dir. Set it in YAML or pass --image_dir.")
    if not cfg["paths"]["label_dir"]:
        raise ValueError("Missing paths.label_dir. Set it in YAML or pass --label_dir.")

    channels = cfg["model"]["channels"]
    strides = cfg["model"]["strides"]
    if len(strides) != len(channels) - 1:
        raise ValueError(
            f"MONAI UNet requires len(strides) == len(channels)-1. "
            f"Got channels={channels}, strides={strides}."
        )

    if cfg["sampling"]["pos"] < 0 or cfg["sampling"]["neg"] < 0:
        raise ValueError("sampling.pos and sampling.neg must be non-negative.")
    if cfg["sampling"]["pos"] + cfg["sampling"]["neg"] <= 0:
        raise ValueError("sampling.pos + sampling.neg must be > 0.")

    if cfg["training"].get("validate_every_epoch", True):
        cfg["training"]["val_interval"] = 1

    if cfg["training"]["best_ma_window"] <= 0:
        raise ValueError("training.best_ma_window must be >= 1.")

    scheduler_name = cfg.get("scheduler", {}).get("name", "none")
    if scheduler_name not in ["none", "cosine", "reduce_on_plateau"]:
        raise ValueError(
            "scheduler.name must be one of: none, cosine, reduce_on_plateau. "
            f"Got {scheduler_name}."
        )

    return cfg


def resolve_experiment_paths(cfg):
    """
    Resolve experiment name and output directory in one place.

    New convention:
        paths.output_root / experiment.exp_name

    This prevents multiple runs from accidentally writing into the same folder.
    """
    exp_cfg = cfg["experiment"]
    paths = cfg["paths"]

    if exp_cfg.get("exp_name") is None:
        exp_cfg["exp_name"] = datetime.now().strftime("real_only_unet3d_binary_%Y%m%d_%H%M%S")

    output_root = paths.get("output_root", "./runs")
    paths["output_dir"] = os.path.join(output_root, exp_cfg["exp_name"])

    return cfg


def save_resolved_config(cfg):
    """
    Save the exact resolved config into the run folder so each checkpoint folder
    is self-describing.
    """
    out_path = os.path.join(cfg["paths"]["output_dir"], "config_resolved.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved resolved config: {out_path}")


# -----------------------------
# Data/model helpers
# -----------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def build_file_list(image_dir, label_dir):
    print("[DEBUG] Image DIR:", image_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
    data = []

    for image_path in image_paths:
        fname = os.path.basename(image_path)
        label_fname = fname.replace("_0000", "")
        label_path = os.path.join(label_dir, label_fname)

        if os.path.exists(label_path):
            data.append({"image": image_path, "label": label_path})

    print(f"Found {len(data)} valid image-label pairs.")
    if len(data) == 0:
        raise RuntimeError("No valid image-label pairs found.")

    return data


def train_val_split(data, val_ratio=0.2, seed=42, num_folds=1, fold_index=0):
    """
    Deterministic train/val split.

    If num_folds <= 1, keep the old random val_ratio split.
    If num_folds > 1, use a true K-fold split over the training dataset:
      - shuffle once with `seed`
      - fold_index is used as validation fold
      - all remaining folds are training cases
    """
    rng = random.Random(seed)
    data = data.copy()
    rng.shuffle(data)

    if num_folds is None or int(num_folds) <= 1:
        n_val = max(1, int(len(data) * val_ratio))
        val_data = data[:n_val]
        train_data = data[n_val:]
        print(f"Using random split: val_ratio={val_ratio}, seed={seed}")
    else:
        num_folds = int(num_folds)
        fold_index = int(fold_index)
        if num_folds < 2:
            raise ValueError("data.num_folds must be >= 2 for K-fold training.")
        if not (0 <= fold_index < num_folds):
            raise ValueError(f"data.fold_index must be in [0, {num_folds - 1}], got {fold_index}.")
        folds = np.array_split(np.arange(len(data)), num_folds)
        val_idx = set(int(i) for i in folds[fold_index])
        val_data = [d for i, d in enumerate(data) if i in val_idx]
        train_data = [d for i, d in enumerate(data) if i not in val_idx]
        print(f"Using K-fold split: fold={fold_index}/{num_folds}, seed={seed}")

    print(f"Train cases: {len(train_data)}")
    print(f"Val cases: {len(val_data)}")

    return train_data, val_data


def get_transforms(roi_size, pos, neg, num_samples):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms


def build_model(model_cfg):
    name = model_cfg.get("name", "monai_unet")
    if name != "monai_unet":
        raise ValueError(f"Unsupported model.name={name}. Currently supported: monai_unet")

    return UNet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        channels=tuple(model_cfg["channels"]),
        strides=tuple(model_cfg["strides"]),
        num_res_units=int(model_cfg.get("num_res_units", 1)),
    )


def build_scheduler(optimizer, scheduler_cfg, train_cfg):
    """
    Small, conservative LR scheduling options.

    cosine:
      Smoothly decays LR from training.lr to eta_min across max_epochs.
      This is the simplest stabilizer and does not depend on noisy tiny-val Dice.

    reduce_on_plateau:
      Reduces LR when validation Dice moving average stops improving.
      More adaptive, but can be noisy with very small validation folds.
    """
    scheduler_cfg = scheduler_cfg or {"name": "none"}
    name = scheduler_cfg.get("name", "none")

    if name == "none":
        return None

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg["max_epochs"]),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-5)),
        )

    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "max"),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 10)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
        )

    raise ValueError(f"Unsupported scheduler.name={name}")


def setup_wandb(cfg):
    exp_cfg = cfg["experiment"]

    wandb.init(
        project=exp_cfg["wandb_project"],
        name=exp_cfg["exp_name"],
        config=flatten_config(cfg),
        mode=exp_cfg["wandb_mode"],
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train_step/*", step_metric="global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")


def save_checkpoint(path, epoch, model, optimizer, cfg, metrics, tag, scheduler=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "metrics": metrics,
        "selection_tag": tag,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


# -----------------------------
# Training
# -----------------------------

def main(cfg):
    cfg = validate_config(cfg)
    cfg = resolve_experiment_paths(cfg)

    paths = cfg["paths"]
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    samp_cfg = cfg["sampling"]
    train_cfg = cfg["training"]
    scheduler_cfg = cfg.get("scheduler", {"name": "none"})

    training_seed = data_cfg.get("training_seed")
    if training_seed is None:
        training_seed = data_cfg["seed"]
    training_seed = int(training_seed)

    # Keep data.seed fixed for the train/validation split.
    # Change only training_seed to rerun the same fold with different
    # model initialization, augmentation, patch sampling, and loader order.
    seed_everything(training_seed)
    os.makedirs(paths["output_dir"], exist_ok=True)
    save_resolved_config(cfg)

    setup_wandb(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Split seed: {data_cfg['seed']}")
    print(f"Training seed: {training_seed}")
    print(f"Output dir: {paths['output_dir']}")
    print(f"Config: channels={cfg['model']['channels']}, strides={cfg['model']['strides']}, "
          f"pos:neg={samp_cfg['pos']}:{samp_cfg['neg']}, "
          f"best_ma_window={train_cfg['best_ma_window']}, "
          f"scheduler={scheduler_cfg.get('name', 'none')}")

    data = build_file_list(paths["image_dir"], paths["label_dir"])
    train_data, val_data = train_val_split(
        data,
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg["seed"],
        num_folds=data_cfg.get("num_folds", 1),
        fold_index=data_cfg.get("fold_index", 0),
    )

    wandb.config.update({
        "num_total_cases": len(data),
        "num_train_cases": len(train_data),
        "num_val_cases": len(val_data),
        "num_folds": data_cfg.get("num_folds", 1),
        "fold_index": data_cfg.get("fold_index", 0),
        "split_seed": data_cfg["seed"],
        "training_seed": training_seed,
    }, allow_val_change=True)

    roi_size = tuple(data_cfg["roi_size"])

    train_transforms, val_transforms = get_transforms(
        roi_size=roi_size,
        pos=samp_cfg["pos"],
        neg=samp_cfg["neg"],
        num_samples=samp_cfg["num_samples"],
    )

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({"model_parameters": n_params}, allow_val_change=True)

    if exp_cfg["wandb_watch"]:
        wandb.watch(model, log="gradients", log_freq=100)

    loss_fn = DiceLoss(
        sigmoid=True,
        smooth_nr=train_cfg["dice_smooth"],
        smooth_dr=train_cfg["dice_smooth"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_cfg=scheduler_cfg,
        train_cfg=train_cfg,
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean")

    best_ma_dice = -1.0
    best_single_dice = -1.0
    best_model_path = os.path.join(paths["output_dir"], "best_model.pt")
    best_single_model_path = os.path.join(paths["output_dir"], "best_single_dice_model.pt")
    last_model_path = os.path.join(paths["output_dir"], "last_model.pt")

    val_dice_window = deque(maxlen=train_cfg["best_ma_window"])
    global_step = 0

    print("Start training...")

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        model.train()

        epoch_loss = 0.0
        epoch_pos_voxels = 0.0
        epoch_positive_patch_count = 0
        step = 0

        for batch in train_loader:
            step += 1
            global_step += 1

            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()
            labels = (labels > 0).float()

            pos_voxels = labels.sum().item()
            epoch_pos_voxels += pos_voxels
            if pos_voxels > 0:
                epoch_positive_patch_count += 1

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if global_step % train_cfg["log_interval"] == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs > train_cfg["threshold"]).float()

                wandb.log({
                    "global_step": global_step,
                    "train_step/loss": loss.item(),
                    "train_step/positive_voxels": pos_voxels,
                    "train_step/pred_pos_ratio": preds.mean().item(),
                    "train_step/prob_mean": probs.mean().item(),
                    "train_step/prob_max": probs.max().item(),
                })

        epoch_loss /= max(step, 1)
        mean_pos_voxels = epoch_pos_voxels / max(step, 1)
        positive_patch_ratio = epoch_positive_patch_count / max(step, 1)

        print(
            f"[Epoch {epoch:03d}/{train_cfg['max_epochs']}] "
            f"Train loss: {epoch_loss:.4f} | "
            f"Mean pos vox/batch: {mean_pos_voxels:.1f} | "
            f"Positive patch ratio: {positive_patch_ratio:.3f}"
        )

        wandb.log({
            "epoch": epoch,
            "train/loss": epoch_loss,
            "train/mean_positive_voxels_per_batch": mean_pos_voxels,
            "train/patches_with_positive_label_ratio": positive_patch_ratio,
            "lr": optimizer.param_groups[0]["lr"],
        })

        should_validate = (epoch % train_cfg["val_interval"] == 0)
        metrics = {
            "train_loss": epoch_loss,
            "best_ma_dice": best_ma_dice,
            "best_single_dice": best_single_dice,
        }

        if should_validate:
            model.eval()
            dice_metric.reset()

            val_loss = 0.0
            val_cases = 0
            val_pred_voxels = 0.0
            val_gt_voxels = 0.0
            val_prob_mean = 0.0
            val_prob_max = 0.0
            val_pred_pos_ratio = 0.0
            val_gt_pos_ratio = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_cases += 1
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device).float()
                    val_labels = (val_labels > 0).float()

                    logits = sliding_window_inference(
                        inputs=val_images,
                        roi_size=roi_size,
                        sw_batch_size=data_cfg["sw_batch_size"],
                        predictor=model,
                        overlap=data_cfg["sw_overlap"],
                    )

                    loss = loss_fn(logits, val_labels)
                    val_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    preds = (probs > train_cfg["threshold"]).float()
                    dice_metric(y_pred=preds, y=val_labels)

                    val_pred_voxels += preds.sum().item()
                    val_gt_voxels += val_labels.sum().item()
                    val_prob_mean += probs.mean().item()
                    val_prob_max += probs.max().item()
                    val_pred_pos_ratio += preds.mean().item()
                    val_gt_pos_ratio += val_labels.mean().item()

                mean_val_loss = val_loss / max(val_cases, 1)
                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()

                mean_pred_voxels = val_pred_voxels / max(val_cases, 1)
                mean_gt_voxels = val_gt_voxels / max(val_cases, 1)
                mean_prob_mean = val_prob_mean / max(val_cases, 1)
                mean_prob_max = val_prob_max / max(val_cases, 1)
                mean_pred_pos_ratio = val_pred_pos_ratio / max(val_cases, 1)
                mean_gt_pos_ratio = val_gt_pos_ratio / max(val_cases, 1)

            val_dice_window.append(mean_dice)
            ma_dice = float(np.mean(val_dice_window))
            ma_window_used = len(val_dice_window)

            metrics.update({
                "val_loss": mean_val_loss,
                "val_dice": mean_dice,
                "val_dice_ma": ma_dice,
                "val_dice_ma_window_used": ma_window_used,
                "mean_pred_voxels": mean_pred_voxels,
                "mean_gt_voxels": mean_gt_voxels,
                "mean_prob_mean": mean_prob_mean,
                "mean_prob_max": mean_prob_max,
                "mean_pred_pos_ratio": mean_pred_pos_ratio,
                "mean_gt_pos_ratio": mean_gt_pos_ratio,
                "best_ma_dice": best_ma_dice,
                "best_single_dice": best_single_dice,
            })

            print(
                f"[Epoch {epoch:03d}] "
                f"Val loss: {mean_val_loss:.4f} | "
                f"Val Dice: {mean_dice:.4f} | "
                f"MA Dice({ma_window_used}/{train_cfg['best_ma_window']}): {ma_dice:.4f} | "
                f"Pred vox/case: {mean_pred_voxels:.1f} | "
                f"GT vox/case: {mean_gt_voxels:.1f} | "
                f"Prob max: {mean_prob_max:.4f}"
            )

            wandb.log({
                "epoch": epoch,
                "val/loss": mean_val_loss,
                "val/dice": mean_dice,
                "val/dice_moving_average": ma_dice,
                "val/dice_ma_window_used": ma_window_used,
                "val/mean_predicted_positive_voxels": mean_pred_voxels,
                "val/mean_groundtruth_positive_voxels": mean_gt_voxels,
                "val/prob_mean": mean_prob_mean,
                "val/prob_max": mean_prob_max,
                "val/pred_pos_ratio": mean_pred_pos_ratio,
                "val/gt_pos_ratio": mean_gt_pos_ratio,
            })

            # Main best checkpoint: chosen by moving average, not single-epoch spike.
            if ma_dice > best_ma_dice:
                best_ma_dice = ma_dice
                metrics["best_ma_dice"] = best_ma_dice
                save_checkpoint(
                    best_model_path,
                    epoch,
                    model,
                    optimizer,
                    deepcopy(cfg),
                    metrics,
                    tag=f"best_val_dice_moving_average_window_{train_cfg['best_ma_window']}",
                    scheduler=scheduler,
                )
                wandb.run.summary["best_ma_dice"] = best_ma_dice
                wandb.run.summary["best_ma_epoch"] = epoch
                print(f"Saved best moving-average model: {best_model_path}")

            # Optional diagnostic checkpoint: the old behavior, separated by name.
            if train_cfg.get("save_single_best_also", True) and mean_dice > best_single_dice:
                best_single_dice = mean_dice
                metrics["best_single_dice"] = best_single_dice
                save_checkpoint(
                    best_single_model_path,
                    epoch,
                    model,
                    optimizer,
                    deepcopy(cfg),
                    metrics,
                    tag="best_single_epoch_val_dice",
                    scheduler=scheduler,
                )
                wandb.run.summary["best_single_dice"] = best_single_dice
                wandb.run.summary["best_single_epoch"] = epoch
                print(f"Saved best single-epoch Dice model: {best_single_model_path}")

        # Step LR scheduler once per epoch.
        # For ReduceLROnPlateau, use the moving-average validation Dice because
        # single-fold validation Dice can be extremely spiky in this project.
        if scheduler is not None:
            if scheduler_cfg.get("name") == "reduce_on_plateau":
                if should_validate and "val_dice_ma" in metrics:
                    scheduler.step(metrics["val_dice_ma"])
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch:03d}] LR after scheduler step: {current_lr:.6g}")
            wandb.log({"epoch": epoch, "lr": current_lr})

        save_checkpoint(
            last_model_path,
            epoch,
            model,
            optimizer,
            deepcopy(cfg),
            metrics,
            tag="last_model",
            scheduler=scheduler,
        )

    print("Training complete.")
    print(f"Best moving-average validation Dice: {best_ma_dice:.4f}")
    if train_cfg.get("save_single_best_also", True):
        print(f"Best single-epoch validation Dice: {best_single_dice:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")

    # Optional convenient overrides.
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root folder for all runs. Final output_dir = output_root / experiment.exp_name.")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--num_folds", type=int, default=None, help="Number of K folds. Use 1 to keep old random val split.")
    parser.add_argument("--fold_index", type=int, default=None, help="Validation fold index, 0-based.")

    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    main(cfg)

"""
Trying to save: Pick75Var1, fold 1
CUDA_VISIBLE_DEVICES=0 python experiments/3d_seg/train_with_config_kfold_scheduler_two_seeds.py \
    --config experiments/3d_seg/configs/mixed600Real40Synth40/mixed600Real40Synth40_light_96x96x48_pos2neg1_config.yaml \
    --num_folds 5 \
    --fold_index 4
"""
"""
Trying to save: Pick40Var1, fold 4
CUDA_VISIBLE_DEVICES=0 python experiments/3d_seg/train_with_config_kfold_scheduler.py \
    --config experiments/3d_seg/configs/synth500PoCPick40Var1/synth500PoCPick40Var1_light_96x96x48_pos2neg1_300epochs_config.yaml \
    --num_folds 5 \
    --fold_index 4
"""
"""
Trying to save: Pick20Var1, fold 2: Results: no enlightement
CUDA_VISIBLE_DEVICES=1 python experiments/3d_seg/train_with_config_kfold_scheduler.py \
    --config experiments/3d_seg/configs/synth500PoCPick20Var1/synth500PoCPick20Var1_light_96x96x48_pos2neg1_config.yaml \
    --num_folds 5 \
    --fold_index 2
"""
"""
Trying to save: Pick20Var1, fold 2: Results: no enlightement
CUDA_VISIBLE_DEVICES=1 python experiments/3d_seg/train_with_config_kfold_scheduler.py \
    --config experiments/3d_seg/configs/synth500PoCPick20Var1/synth500PoCPick20Var1_light_96x96x48_pos2neg1_300epochs_config.yaml \
    --num_folds 5 \
    --fold_index 2
"""
"""
Trying to save: Pick20Var1, fold 2: Results: no enlightement
CUDA_VISIBLE_DEVICES=0 python experiments/3d_seg/train_with_config_kfold_scheduler.py \
    --config experiments/3d_seg/configs/synth500PoCPick20Var1/synth500PoCPick20Var1_light_96x96x48_pos2neg1_300epochs_config.yaml \
    --num_folds 5 \
    --fold_index 2
"""
"""
Pick20Var1, fold 2: Results: 
CUDA_VISIBLE_DEVICES=1 python experiments/3d_seg/train_with_config_kfold_scheduler.py \
    --config experiments/3d_seg/configs/synth500PoCPick20Var1/synth500PoCPick20Var1_light_96x96x48_pos2neg1_config.yaml \
    --num_folds 5 \
    --fold_index 0
"""