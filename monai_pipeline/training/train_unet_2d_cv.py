"""
============================================================
5-FOLD CROSS-VALIDATION TRAINING (MONAI, 2D U-NET)
============================================================

This script trains a 2D U-Net for binary whole-tumor segmentation.

- 5-fold cross-validation (patient-level split)
- 2D patch training from 3D MRI volumes
- 4 input channels (T1, T1c, T2, FLAIR)
- 1 output channel (whole tumor mask)
- Saves best model per fold + metrics JSON

If you run the module and see NOTHING printed, the most common reason is:
- missing `if __name__ == "__main__": main()`

This file includes that block and also prints immediately when it starts.
"""
from __future__ import annotations

# This should print immediately when the script begins executing
print(">>> train_unet_2d_cv.py started")

import os
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# IMPORTANT:
# - autocast is used for mixed precision (AMP)
# - GradScaler prevents numeric underflow when using AMP
# - Your PyTorch version may support different GradScaler signatures,
#   so we handle both safely (see scaler block below).
from torch.amp import autocast, GradScaler

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

from monai_pipeline.config.paths import OUTPUTS_DIR
from monai_pipeline.data.dataloader_cv import create_loaders_for_fold_2d


# 1) Training configuration

@dataclass
class TrainConfig:
    # Cross-validation
    k_folds: int = 5
    seed: int = 42

    # Patch sampling
    patch_size: tuple[int, int] = (128, 128)
    num_samples_per_volume: int = 4  # patches per patient per iteration

    # Data loading
    batch_size: int = 2
    num_workers: int = 2
    cache_rate: float = 0.2

    # Optimization
    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Behavior
    validate_every: int = 1
    amp: bool = True

    # Output folder name inside outputs/
    outputs_subdir: str = "unet2d_cv"


# 2) Model

def make_model(device: torch.device) -> nn.Module:
    """
    Build a 2D U-Net.

    The model expects:
      input:  (B, 4, H, W)  -> 4 MRI channels
      output: (B, 1, H, W)  -> 1 binary mask channel

    We use MONAI's UNet implementation:
    - channels controls how many feature maps at each level
    - strides controls downsampling between levels
    - norm="BATCH" adds BatchNorm layers for training stability
    """
    model = UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="BATCH",
    ).to(device)
    return model


# 3) Validation metric (patch-level Dice)

def dice_on_loader(model: nn.Module, loader, device: torch.device) -> float:
    """
    Compute mean Dice score on the validation loader.

    NOTE:
    - This computes patch-level Dice, because training/validation uses patches.
    - Full-volume Dice can be added later using sliding-window inference
      (different evaluation pipeline).

    Why threshold=0.5?
    - Model outputs probabilities after sigmoid.
    - We convert probabilities to binary mask using 0.5 cutoff.
    """
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean")

    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)

    with torch.no_grad():
        for batch in loader:
            # IMPORTANT:
            # MONAI's RandCropByPosNegLabeld can return a LIST of dicts.
            # Your dataloader earlier showed "batch is a list".
            batches = batch if isinstance(batch, list) else [batch]

            for b in batches:
                images = b["image"].to(device)  # (B,4,H,W,1)
                labels = b["label"].to(device)  # (B,1,H,W,1)

                # Convert from "2.5D shape" (H,W,1) to true 2D tensors (H,W)
                images = images.squeeze(-1)  # -> (B,4,H,W)
                labels = labels.squeeze(-1)  # -> (B,1,H,W)

                logits = model(images)
                probs = torch.sigmoid(logits)

                preds = post_pred(probs)
                labs = post_label(labels)

                dice_metric(y_pred=preds, y=labs)

        score = float(dice_metric.aggregate().item())
        dice_metric.reset()

    return score


# 4) Train one fold

def train_one_fold(cfg: TrainConfig, fold: int, device: torch.device) -> Dict:
    """
    Train a model for ONE CV fold and return a summary dictionary.

    Each fold:
    - builds its own train/val loaders
    - trains for cfg.max_epochs
    - evaluates every cfg.validate_every epochs
    - saves the best model based on validation Dice
    """
    # Make randomness reproducible per fold
    set_determinism(seed=cfg.seed + fold)

    # Output paths for this fold
    fold_dir = os.path.join(OUTPUTS_DIR, cfg.outputs_subdir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    model_path = os.path.join(fold_dir, "best_model.pt")
    metrics_path = os.path.join(fold_dir, "metrics.json")

    # Build train/val loaders for this fold
    train_loader, val_loader = create_loaders_for_fold_2d(
        fold_index=fold,
        k=cfg.k_folds,
        seed=cfg.seed,
        patch_size=cfg.patch_size,
        num_samples_per_volume=cfg.num_samples_per_volume,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cache_rate=cfg.cache_rate,
        cache_workers=None,  # OS-safe auto selection inside dataloader_cv.py
    )

    # Create model + optimization objects
    model = make_model(device)

    # DiceLoss(sigmoid=True) means:
    # - the loss internally applies sigmoid to logits
    # - you do NOT manually sigmoid before passing to loss
    loss_fn = DiceLoss(sigmoid=True, include_background=True)

    # AdamW:
    # - Adam optimizer + "decoupled" weight decay (better regularization behavior)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # -------------------------------------------------
    # Mixed precision scaler (AMP)
    #
    # Why we need this:
    # - AMP uses float16 in many ops -> faster, less VRAM
    # - But float16 gradients can underflow (become zeros)
    # - GradScaler multiplies the loss to keep gradients in a safe numeric range
    #
    # IMPORTANT:
    # Different PyTorch builds support different GradScaler signatures.
    # Some accept: GradScaler("cuda", enabled=...)
    # Older ones accept: torch.cuda.amp.GradScaler(enabled=...)
    #
    # This block supports BOTH (no change of functionality, just compatibility).
    # -------------------------------------------------
    scaler = None  # helps some editors/linters
    try:
        # Newer / preferred API (many PyTorch 2.x builds)
        scaler = GradScaler(
            "cuda",
            enabled=(cfg.amp and device.type == "cuda"),
        )
    except TypeError:
        # Older fallback (still works; may show deprecation warning)
        from torch.cuda.amp import GradScaler as CudaGradScaler
        scaler = CudaGradScaler(
            enabled=(cfg.amp and device.type == "cuda")
        )

    best_val_dice = -1.0
    history: List[Dict] = []

    print(f"\n===== Fold {fold} / {cfg.k_folds - 1} =====")
    print("Training samples (patch dataset length):", len(train_loader.dataset))
    print("Validation samples (patch dataset length):", len(val_loader.dataset))
    print("Device:", device)

    start_time = time.time()

    # -------------------------
    # Epoch loop
    # -------------------------
    for epoch in range(1, cfg.max_epochs + 1):
        print(f"\n[Fold {fold}] Epoch {epoch}/{cfg.max_epochs} started")

        model.train()
        epoch_loss = 0.0
        steps = 0

        # IMPORTANT:
        # Each "batch" might be a LIST of dictionaries (MONAI cropping behavior).
        # So we iterate accordingly.
        for batch in train_loader:
            batches = batch if isinstance(batch, list) else [batch]

            for b in batches:
                images = b["image"].to(device)  # (B,4,H,W,1)
                labels = b["label"].to(device)  # (B,1,H,W,1)

                images = images.squeeze(-1)     # (B,4,H,W)
                labels = labels.squeeze(-1)     # (B,1,H,W)

                optimizer.zero_grad(set_to_none=True)

                # autocast:
                # - only enabled if cfg.amp is True and GPU is used
                # - runs many ops in float16 for speed/VRAM reduction
                with autocast(
                    device_type="cuda",
                    enabled=(cfg.amp and device.type == "cuda"),
                ):
                    logits = model(images)
                    loss = loss_fn(logits, labels)

                # AMP step:
                # - scale(loss) to avoid underflow
                # - backward on scaled loss
                # - step optimizer using scaled gradients
                # - update scale factor dynamically
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += float(loss.item())
                steps += 1

        epoch_loss /= max(steps, 1)

        # 6) Validation
        val_dice: Optional[float] = None
        if epoch % cfg.validate_every == 0:
            val_dice = dice_on_loader(model, val_loader, device)

            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), model_path)
                print(f"[Fold {fold}] The best model saved (val_dice={best_val_dice:.4f})")

        print(
            f"[Fold {fold}] Epoch {epoch}/{cfg.max_epochs} done | "
            f"train_loss={epoch_loss:.4f} | val_dice={val_dice if val_dice is not None else 'NA'} | "
            f"best={best_val_dice:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_dice": val_dice,
                "best_val_dice_so_far": best_val_dice,
            }
        )

    elapsed = time.time() - start_time

    print(f"\n===== Fold {fold} finished =====")
    print(f"Best val Dice: {best_val_dice:.4f}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Best model path: {model_path}")

    summary = {
        "fold": fold,
        "best_val_dice": best_val_dice,
        "model_path": model_path,
        "elapsed_seconds": elapsed,
        "config": cfg.__dict__,
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# 6) Run all folds

def main():
    """
    Entry point:
    - selects device (GPU if available)
    - trains all folds
    - writes final cv_summary.json
    """
    cfg = TrainConfig()

    # Decide whether to use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Device selected:", device)

    out_dir = os.path.join(OUTPUTS_DIR, cfg.outputs_subdir)
    os.makedirs(out_dir, exist_ok=True)

    all_summaries: List[Dict] = []

    for fold in range(cfg.k_folds):
        summary = train_one_fold(cfg, fold, device)
        all_summaries.append(summary)

    # Aggregate CV results
    best_dices = [s["best_val_dice"] for s in all_summaries]
    mean_dice = float(sum(best_dices) / len(best_dices))
    std_dice = float(torch.tensor(best_dices).std(unbiased=False).item())

    report = {
        "k_folds": cfg.k_folds,
        "best_dice_per_fold": best_dices,
        "mean_best_dice": mean_dice,
        "std_best_dice": std_dice,
        "device": str(device),
    }

    report_path = os.path.join(out_dir, "cv_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n===== CV SUMMARY =====")
    print("Best Dice per fold:", best_dices)
    print(f"Mean best Dice: {mean_dice:.4f}")
    print(f"Std best Dice:  {std_dice:.4f}")
    print("Saved summary:", report_path)


# 7) IMPORTANT: This is what makes the script actually run

if __name__ == "__main__":
    main()
