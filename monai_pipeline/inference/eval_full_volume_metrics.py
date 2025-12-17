"""
FULL-VOLUME METRICS (5-FOLD CV)  

This script computes FULL-VOLUME segmentation metrics per patient.

Metrics reported:
- Dice
- Precision
- Recall / Sensitivity
- Specificity

IMPORTANT:
- It uses the already-trained best_model.pt per fold.
- It runs full-volume inference (slice-by-slice, because the model is 2D),
  then computes metrics on the reconstructed 3D predicted mask.

Outputs:
- CSV per patient: full_volume_metrics_per_patient.csv
- JSON summary per fold + overall: full_volume_metrics_summary.json

Note about overwriting:
- The CSV/JSON are opened with "w", so rerunning will overwrite them.
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

from monai_pipeline.config.paths import OUTPUTS_DIR, UCSF_PDGM_DATASET_DIR

# Dataset JSON created earlier by your build_dataset.py
DATASET_JSON = os.path.join(OUTPUTS_DIR, "dataset_final_index.json")

# Folder where your fold_0/best_model.pt ... fold_4/best_model.pt live
TRAIN_OUT_DIR = os.path.join(OUTPUTS_DIR, "unet2d_cv")


# 1) Convert mask to binary

def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert any non-zero label to 1 (whole tumor), else 0."""
    return (mask > 0).astype(np.uint8)


# 2) Full-volume evaluation transforms

def get_full_volume_eval_transforms(
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Evaluation transforms:
    - load image+label
    - orient to RAS
    - resample to consistent spacing
    - binarize label
    - crop to foreground (faster)
    - normalize intensities
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            LambdaD(keys="label", func=_mask_to_binary),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )



# 3) Model (must match the training)

def make_model(device: torch.device) -> nn.Module:
    """Same UNet definition as training; do not change unless you retrain."""
    return UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="BATCH",
    ).to(device)



# 4) Load the dataset JSON list

def load_dataset_list(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset JSON is empty or invalid: {json_path}")
    return data


# 5) Create CV folds by patient_id

def make_patient_folds(patient_ids: List[str], k: int, seed: int) -> List[List[str]]:
    unique_ids = sorted(set(patient_ids))

    # Prefer sklearn if installed (recommended)
    try:
        from sklearn.model_selection import KFold  # type: ignore

        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds: List[List[str]] = []
        for _tr, va in kf.split(unique_ids):
            folds.append([unique_ids[i] for i in va])
        return folds

    except Exception:
        # Fallback: deterministic shuffle + chunking
        rng = np.random.RandomState(seed)
        ids = list(unique_ids)
        rng.shuffle(ids)
        return [list(x) for x in np.array_split(ids, k)]



# 6) Full-volume inference (2D model slice-by-slice)

@torch.no_grad()
def predict_full_volume_2d_slice_by_slice(
    model: nn.Module,
    image_4ch: torch.Tensor,              # (4, H, W, Z)
    roi_hw: Tuple[int, int],
    device: torch.device,
    sw_batch_size: int = 4,
    overlap: float = 0.25,
) -> torch.Tensor:
    """
    Returns binary predicted mask volume: (1, H, W, Z) with values {0,1}.

    Because the model is 2D, we:
    - loop over Z slices
    - run 2D sliding window inference on each slice
    - stack predictions back to a 3D volume
    """
    model.eval()

    inferer = SlidingWindowInferer(
        roi_size=roi_hw,         # (H, W)
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
    )

    C, H, W, Z = image_4ch.shape
    pred_slices = []

    for z in range(Z):
        # (4,H,W) -> (1,4,H,W)
        x2d = image_4ch[:, :, :, z].unsqueeze(0).to(device)

        # (1,1,H,W) logits
        logits = inferer(x2d, model)

        # sigmoid -> probabilities
        probs = torch.sigmoid(logits).squeeze(0)  # (1,H,W)

        pred_slices.append(probs.cpu())

    # Stack back: list of (1,H,W) -> (1,H,W,Z)
    prob_vol = torch.stack(pred_slices, dim=-1)

    # Threshold to binary (0/1)
    pred_bin = (prob_vol > 0.5).to(torch.uint8)

    return pred_bin


# 7) Metrics from confusion counts

def confusion_counts(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN for binary masks.

    pred, gt: uint8 tensors with same shape (1,H,W,Z) (or without the channel dim).
    """
    pred = pred.bool()
    gt = gt.bool()

    tp = int(torch.logical_and(pred, gt).sum().item())
    fp = int(torch.logical_and(pred, torch.logical_not(gt)).sum().item())
    fn = int(torch.logical_and(torch.logical_not(pred), gt).sum().item())
    tn = int(torch.logical_and(torch.logical_not(pred), torch.logical_not(gt)).sum().item())
    return tp, fp, fn, tn


def safe_div(num: float, den: float) -> float:
    """Safe division to avoid crashing when denominator is 0."""
    return float(num / den) if den != 0 else 0.0


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    """Dice = 2TP / (2TP + FP + FN)"""
    return safe_div(2 * tp, (2 * tp + fp + fn))


def precision_from_counts(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP)"""
    return safe_div(tp, (tp + fp))


def recall_from_counts(tp: int, fn: int) -> float:
    """Recall/Sensitivity = TP / (TP + FN)"""
    return safe_div(tp, (tp + fn))


def specificity_from_counts(tn: int, fp: int) -> float:
    """Specificity = TN / (TN + FP)"""
    return safe_div(tn, (tn + fp))


# 8) Main function

@dataclass
class EvalConfig:
    k_folds: int = 5
    seed: int = 42

    # IMPORTANT: keep roi_hw equal to (or larger than) the patch size used in training
    roi_hw: Tuple[int, int] = (128, 128)

    # Spacing used in preprocessing
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Sliding window micro-batch size (trade speed vs VRAM)
    sw_batch_size: int = 4


def main():
    print("=== Full-volume metrics evaluation started ===")
    print("Dataset JSON:", DATASET_JSON)
    print("Training output dir:", TRAIN_OUT_DIR)
    print("Dataset root:", UCSF_PDGM_DATASET_DIR)

    cfg = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load dataset list (patient_id + image paths + label path)
    dataset_list = load_dataset_list(DATASET_JSON)

    # 2) Create folds
    patient_ids = [d["patient_id"] for d in dataset_list]
    folds = make_patient_folds(patient_ids, k=cfg.k_folds, seed=cfg.seed)

    # 3) Fast lookup
    by_id = {d["patient_id"]: d for d in dataset_list}

    # 4) Evaluation transforms
    xform = get_full_volume_eval_transforms(pixdim=cfg.pixdim)

    # 5) Output paths (reruns overwrite these files automatically)
    out_csv = os.path.join(TRAIN_OUT_DIR, "full_volume_metrics_per_patient.csv")
    out_json = os.path.join(TRAIN_OUT_DIR, "full_volume_metrics_summary.json")
    os.makedirs(TRAIN_OUT_DIR, exist_ok=True)

    per_patient_rows: List[Dict] = []
    fold_summaries: List[Dict] = []

    for fold_idx in range(cfg.k_folds):
        print(f"\n--- Fold {fold_idx}/{cfg.k_folds - 1} ---")

        # Load fold model
        model_path = os.path.join(TRAIN_OUT_DIR, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")

        model = make_model(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Validation patients for this fold
        val_ids = folds[fold_idx]

        # Collect metrics for fold-level averaging
        fold_metrics = {
            "dice": [],
            "precision": [],
            "recall": [],
            "specificity": [],
        }

        for pid in val_ids:
            item = by_id[pid]

            # MONAI expects dict input
            sample = {"image": item["image"], "label": item["label"]}

            # Apply transforms -> tensors
            sample_t = xform(sample)
            img = sample_t["image"]                   # (4,H,W,Z)
            gt = sample_t["label"].to(torch.uint8)    # (1,H,W,Z)

            # Predict full 3D mask
            pred = predict_full_volume_2d_slice_by_slice(
                model=model,
                image_4ch=img,
                roi_hw=cfg.roi_hw,
                device=device,
                sw_batch_size=cfg.sw_batch_size,
            )

            # Confusion counts -> metrics
            tp, fp, fn, tn = confusion_counts(pred, gt)

            dice = dice_from_counts(tp, fp, fn)
            precision = precision_from_counts(tp, fp)
            recall = recall_from_counts(tp, fn)
            spec = specificity_from_counts(tn, fp)

            # Save per-patient row
            per_patient_rows.append(
                {
                    "fold": fold_idx,
                    "patient_id": pid,
                    "dice": f"{dice:.6f}",
                    "precision": f"{precision:.6f}",
                    "recall_sensitivity": f"{recall:.6f}",
                    "specificity": f"{spec:.6f}",
                }
            )

            # Add to fold lists
            fold_metrics["dice"].append(dice)
            fold_metrics["precision"].append(precision)
            fold_metrics["recall"].append(recall)
            fold_metrics["specificity"].append(spec)

        # Helper: fold means/stats
        def mean(x): return float(np.mean(x)) if len(x) else 0.0
        def std(x): return float(np.std(x, ddof=0)) if len(x) else 0.0

        fold_summary = {
            "fold": fold_idx,
            "n_patients": len(val_ids),

            "mean_dice": mean(fold_metrics["dice"]),
            "std_dice_patients_within_fold": std(fold_metrics["dice"]),

            "mean_precision": mean(fold_metrics["precision"]),
            "std_precision_patients_within_fold": std(fold_metrics["precision"]),

            "mean_recall_sensitivity": mean(fold_metrics["recall"]),
            "std_recall_patients_within_fold": std(fold_metrics["recall"]),

            "mean_specificity": mean(fold_metrics["specificity"]),
            "std_specificity_patients_within_fold": std(fold_metrics["specificity"]),
        }

        fold_summaries.append(fold_summary)

        print("Fold mean Dice:", round(fold_summary["mean_dice"], 4))
        print("Fold mean Prec:", round(fold_summary["mean_precision"], 4))
        print("Fold mean Rec :", round(fold_summary["mean_recall_sensitivity"], 4))
        print("Fold mean Spec:", round(fold_summary["mean_specificity"], 4))

    # 9) Overall summary across folds
    # (mean/std of the fold means)

    fold_mean_dice = [f["mean_dice"] for f in fold_summaries]
    fold_mean_precision = [f["mean_precision"] for f in fold_summaries]
    fold_mean_recall = [f["mean_recall_sensitivity"] for f in fold_summaries]
    fold_mean_specificity = [f["mean_specificity"] for f in fold_summaries]

    overall = {
        "k_folds": cfg.k_folds,
        "seed": cfg.seed,
        "roi_hw": cfg.roi_hw,
        "pixdim": cfg.pixdim,
        "sw_batch_size": cfg.sw_batch_size,
        "device": str(device),

        # Fold summaries kept for transparency
        "fold_summaries": fold_summaries,

        # Overall (mean/std across folds for EACH metric)
        "mean_dice_over_folds": float(np.mean(fold_mean_dice)),
        "std_dice_over_folds": float(np.std(fold_mean_dice, ddof=0)),

        "mean_precision_over_folds": float(np.mean(fold_mean_precision)),
        "std_precision_over_folds": float(np.std(fold_mean_precision, ddof=0)),

        "mean_recall_sensitivity_over_folds": float(np.mean(fold_mean_recall)),
        "std_recall_sensitivity_over_folds": float(np.std(fold_mean_recall, ddof=0)),

        "mean_specificity_over_folds": float(np.mean(fold_mean_specificity)),
        "std_specificity_over_folds": float(np.std(fold_mean_specificity, ddof=0)),
    }

    # 10) Write CSV (overwrite on rerun)
    fieldnames = [
        "fold",
        "patient_id",
        "dice",
        "precision",
        "recall_sensitivity",
        "specificity",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_patient_rows)

    # 11) Write JSON with overall metrics (overwrite on rerun)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n=== Full-volume metrics evaluation is complete ===")
    print("Saved per-patient CSV :", out_csv)
    print("Saved summary JSON    :", out_json)

    print("\n=== Overall mean ± std across folds ===")
    print(f"Dice:       {overall['mean_dice_over_folds']:.4f} ± {overall['std_dice_over_folds']:.4f}")
    print(f"Precision:  {overall['mean_precision_over_folds']:.4f} ± {overall['std_precision_over_folds']:.4f}")
    print(f"Recall:     {overall['mean_recall_sensitivity_over_folds']:.4f} ± {overall['std_recall_sensitivity_over_folds']:.4f}")
    print(f"Specificity:{overall['mean_specificity_over_folds']:.4f} ± {overall['std_specificity_over_folds']:.4f}")


if __name__ == "__main__":
    main()
