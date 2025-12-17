"""
# 1) FULL-VOLUME INFERENCE + FULL-VOLUME DICE (5-FOLD CV)

What this script does (in simple terms):
- The script has already trained a 2D U-Net using 2D patches.
- Now the script evaluates the model on FULL 3D patient volumes.

Because the model is 2D:
- The script runs inference slice-by-slice through the 3D volume (along Z).
- On each slice, the script uses sliding-window inference in 2D.
- Then the script stacks all predicted slices back into a full 3D predicted mask.

Then the script compares:
- GT mask (manual expert annotation) vs predicted mask (model output) using FULL-VOLUME Dice per patient.

Outputs:
- CSV: per-patient dice scores across folds
- JSON: per-fold mean dice + overall mean dice

This step does NOT retrain anything.
It only evaluates the already-trained models more correctly.
"""

from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    CropForegroundd,
    NormalizeIntensityd,
    LambdaD,
)
from monai.inferers import SlidingWindowInferer


# 1) Import the project paths

# The script reuses the same path resolver that was already created.
from monai_pipeline.config.paths import (
    OUTPUTS_DIR,
    UCSF_PDGM_DATASET_DIR,
)

# The script uses the dataset JSON created earlier by build_dataset.py
DATASET_JSON = os.path.join(OUTPUTS_DIR, "dataset_final_index.json")

# The script uses the training output folder (where fold_0/best_model.pt etc. are)
# NOTE: adjust if the training folder name differs.
TRAIN_OUT_DIR = os.path.join(OUTPUTS_DIR, "unet2d_cv")


# 2) Convert mask to binary

def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """
    Convert any non-zero label to 1 (whole tumor), else 0.
    This makes the evaluation robust even if the dataset mask uses other encodings.
    """
    return (mask > 0).astype(np.uint8)


# 3) Full-volume evaluation transforms

def get_full_volume_eval_transforms(
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Transforms for evaluation (FULL VOLUME):

    IMPORTANT:
    - The script does NOT random crop.
    - The script does NOT center crop to patch size.
    - The script keeps the full 3D volume so it can compute full-volume Dice.

    The script still:
    - The script loads images/mask
    - The script standardizes orientation + spacing
    - The script crops foreground (optional but speeds inference)
    - The script normalizes intensities
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # image -> (4,H,W,Z), label -> (1,H,W,Z)

            Orientationd(keys=["image", "label"], axcodes="RAS"),

            # Resample both image + label into a consistent voxel spacing
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),

            # Convert label to binary whole tumor
            LambdaD(keys="label", func=_mask_to_binary),

            # Optional: remove big empty background areas (faster inference)
            CropForegroundd(keys=["image", "label"], source_key="label"),

            # Normalize MRI intensities (helps stability)
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            EnsureTyped(keys=["image", "label"]),
        ]
    )


# 4) Build the same model as training

def make_model(device: torch.device) -> nn.Module:
    """
    Must match the training architecture exactly.
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


# 5) Load dataset JSON list

def load_dataset_list(json_path: str) -> List[Dict]:
    """
    Reads outputs/dataset_final_index.json created earlier.
    Each item should contain:
      - patient_id
      - image: [t1, t1c, t2, flair] paths
      - label: mask path
      - label_hgg (optional for this script)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Dataset JSON is empty or not a list: {json_path}")

    return data


# 6) Create fold splits (patient-level)

def make_patient_folds(patient_ids: List[str], k: int, seed: int) -> List[List[str]]:
    """
    Returns folds as a list of lists of patient_ids.

    The script tries to use sklearn KFold if installed (common),
    else fallback to a deterministic numpy shuffle + chunking.

    Why do the script need patient-level folds?
    - The script must never split slices/patches from the same patient into train+val.
    - The split must be reproducible (seed).
    """
    unique_ids = sorted(list(set(patient_ids)))

    # Try sklearn (very standard)
    try:
        from sklearn.model_selection import KFold  # type: ignore

        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds: List[List[str]] = []
        for _train_idx, val_idx in kf.split(unique_ids):
            folds.append([unique_ids[i] for i in val_idx])
        return folds

    except Exception:
        # Fallback: manual shuffle + split
        rng = np.random.RandomState(seed)
        ids = unique_ids[:]
        rng.shuffle(ids)
        folds = [list(x) for x in np.array_split(ids, k)]
        return folds


# 7) Full-volume inference for one patient

@torch.no_grad()
def predict_full_volume_2d_slice_by_slice(
    model: nn.Module,
    image_4ch: torch.Tensor,
    roi_hw: Tuple[int, int],
    device: torch.device,
    sw_batch_size: int = 4,
) -> torch.Tensor:
    """
    Run full-volume inference using a 2D model.

    Input:
      image_4ch: tensor shape (4, H, W, Z)
    Output:
      pred_mask: tensor shape (1, H, W, Z) with values {0,1}

    How it works:
    - For each slice z:
        - take (4, H, W) slice
        - run SlidingWindowInferer in 2D (roi_hw)
        - get probability map (1, H, W)
    - stack all slices back into (1, H, W, Z)
    - threshold at 0.5
    """
    model.eval()

    # Sliding window inferer for 2D slices
    inferer = SlidingWindowInferer(
        roi_size=roi_hw,         # patch size in (H,W)
        sw_batch_size=sw_batch_size,
        overlap=0.25,            # overlap smooths boundaries; good default
        mode="gaussian",         # smooth blending in overlaps
    )

    C, H, W, Z = image_4ch.shape
    pred_slices = []

    for z in range(Z):
        # slice shape: (4, H, W)
        x2d = image_4ch[:, :, :, z]

        # add batch dimension -> (1, 4, H, W)
        x2d = x2d.unsqueeze(0).to(device)

        # model output logits -> (1, 1, H, W)
        logits = inferer(x2d, model)

        # convert to probability with sigmoid
        probs = torch.sigmoid(logits)

        # remove batch -> (1, H, W)
        probs = probs.squeeze(0)

        pred_slices.append(probs.cpu())

    # Stack back to volume: list of (1,H,W) -> (1,H,W,Z)
    pred_prob_vol = torch.stack(pred_slices, dim=-1)

    # Threshold to binary mask
    pred_bin_vol = (pred_prob_vol > 0.5).to(torch.uint8)

    return pred_bin_vol

# 8) Compute Dice score for binary masks

def dice_binary(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute Dice score for binary masks.

    pred, gt should be same shape, binary (0/1).
    Dice = 2*|A∩B| / (|A| + |B|)
    """
    pred = pred.to(torch.float32)
    gt = gt.to(torch.float32)

    inter = torch.sum(pred * gt).item()
    denom = torch.sum(pred).item() + torch.sum(gt).item()

    # Avoid divide-by-zero; if both empty, define dice=1
    if denom == 0:
        return 1.0

    return float((2.0 * inter) / denom)


# 9) Main evaluation loop

@dataclass
class EvalConfig:
    k_folds: int = 5
    seed: int = 42

    # Must match the training patch size (or you can increase it later)
    roi_hw: Tuple[int, int] = (128, 128)

    # Spacing used in preprocessing
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Sliding window micro-batch size (trade speed vs VRAM)
    sw_batch_size: int = 4

# 10) Main function

def main():
    print("=== Full-volume Dice evaluation started ===")
    print("Dataset JSON:", DATASET_JSON)
    print("Training output dir:", TRAIN_OUT_DIR)
    print("Dataset root:", UCSF_PDGM_DATASET_DIR)

    cfg = EvalConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset list (paths + patient_id)
    dataset_list = load_dataset_list(DATASET_JSON)

    # Extract patient IDs
    patient_ids = [d["patient_id"] for d in dataset_list]
    folds = make_patient_folds(patient_ids, k=cfg.k_folds, seed=cfg.seed)

    # Build fast lookup: patient_id -> dataset item
    by_id = {d["patient_id"]: d for d in dataset_list}

    # Prepare transforms
    xform = get_full_volume_eval_transforms(pixdim=cfg.pixdim)

    # Output files
    out_csv = os.path.join(TRAIN_OUT_DIR, "full_volume_dice_per_patient.csv")
    out_json = os.path.join(TRAIN_OUT_DIR, "full_volume_dice_summary.json")

    os.makedirs(TRAIN_OUT_DIR, exist_ok=True)

    all_rows = []
    fold_means = []

    for fold_idx in range(cfg.k_folds):
        print(f"\n--- Evaluating fold {fold_idx}/{cfg.k_folds - 1} ---")

        # Load model weights from this fold
        model_path = os.path.join(TRAIN_OUT_DIR, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")

        model = make_model(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        val_ids = folds[fold_idx]
        dices = []

        for pid in val_ids:
            item = by_id[pid]

            # MONAI expects: {"image": [4 paths], "label": mask_path}
            sample = {"image": item["image"], "label": item["label"]}

            # Apply transforms -> get tensors
            sample_t = xform(sample)

            # image: (4,H,W,Z), label: (1,H,W,Z)
            img = sample_t["image"]
            gt = sample_t["label"].to(torch.uint8)

            # Predict full volume (binary)
            pred = predict_full_volume_2d_slice_by_slice(
                model=model,
                image_4ch=img,
                roi_hw=cfg.roi_hw,
                device=device,
                sw_batch_size=cfg.sw_batch_size,
            )

            # Dice on full volume
            dsc = dice_binary(pred, gt)
            dices.append(dsc)

            all_rows.append({
                "fold": fold_idx,
                "patient_id": pid,
                "dice_full_volume": f"{dsc:.6f}",
            })

        fold_mean = float(np.mean(dices)) if len(dices) else 0.0
        fold_means.append(fold_mean)
        print(f"Fold {fold_idx} mean full-volume Dice: {fold_mean:.4f}  (n={len(dices)})")

    overall_mean = float(np.mean(fold_means))
    overall_std = float(np.std(fold_means, ddof=0))

    summary = {
        "k_folds": cfg.k_folds,
        "seed": cfg.seed,
        "roi_hw": cfg.roi_hw,
        "pixdim": cfg.pixdim,
        "sw_batch_size": cfg.sw_batch_size,
        "fold_mean_full_volume_dice": fold_means,
        "mean_full_volume_dice_over_folds": overall_mean,
        "std_full_volume_dice_over_folds": overall_std,
        "device": str(device),
    }

    # Write CSV

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "patient_id", "dice_full_volume"])
        writer.writeheader()
        writer.writerows(all_rows)

    # Write JSON

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 11) Print summary

    print("\n=== Full-volume Dice evaluation is complete ===")
    print("Saved per-patient CSV:", out_csv)
    print("Saved summary JSON:", out_json)
    print("Fold means:", [round(x, 4) for x in fold_means])
    print(f"Overall mean (over folds): {overall_mean:.4f}")
    print(f"Overall std  (over folds): {overall_std:.4f}")


if __name__ == "__main__":
    main()
