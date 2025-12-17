"""
GT vs Predicted Overlays (Ranked Selection Per Fold)

Selection rule per fold (total = 5 patients):
- 1 BEST   = highest full-volume Dice
- 1 WORST  = lowest full-volume Dice
- 3 MEDIAN = closest to the fold median Dice

Why this is good:
- Best case shows model capability
- Worst case shows failure modes (important for thesis discussion)
- Median cases show "typical" performance (not cherry-picked)

Inputs:
- outputs/unet2d_cv/full_volume_metrics_per_patient.csv
- outputs/dataset_final_index.json
- outputs/unet2d_cv/fold_X/best_model.pt

Outputs:
- outputs/unet2d_cv/overlays_ranked/fold_X/<category>_<patient_id>/

Note:
- Rerunning overwrites the same PNG names.
"""

from __future__ import annotations

import os
import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

from monai_pipeline.config.paths import OUTPUTS_DIR

DATASET_JSON = os.path.join(OUTPUTS_DIR, "dataset_final_index.json")
TRAIN_OUT_DIR = os.path.join(OUTPUTS_DIR, "unet2d_cv")
METRICS_CSV = os.path.join(TRAIN_OUT_DIR, "full_volume_metrics_per_patient.csv")
OVERLAYS_DIR = os.path.join(TRAIN_OUT_DIR, "overlays_ranked")


# 1) Convert mask to binary

def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert any non-zero label to 1 (whole tumor), else 0."""
    return (mask > 0).astype(np.uint8)


# 2) Eval-style transforms

def get_full_volume_eval_transforms(
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Keep preprocessing consistent with evaluation."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            LambdaD(keys="label", func=_mask_to_binary),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


# 3) Model (same as the training)

def make_model(device: torch.device) -> nn.Module:
    return UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="BATCH",
    ).to(device)


# 4) Load the dataset list

def load_dataset_list(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset JSON is empty or invalid: {json_path}")
    return data


# 5) Full-volume inference (2D slice-by-slice)

@torch.no_grad()
def predict_full_volume_2d_slice_by_slice(
    model: nn.Module,
    image_4ch: torch.Tensor,              # (4, H, W, Z)
    roi_hw: Tuple[int, int],
    device: torch.device,
    sw_batch_size: int = 4,
    overlap: float = 0.25,
) -> torch.Tensor:
    """Returns binary predicted mask volume: (1, H, W, Z) with values {0,1}."""
    model.eval()

    inferer = SlidingWindowInferer(
        roi_size=roi_hw, sw_batch_size=sw_batch_size, overlap=overlap, mode="gaussian"
    )

    C, H, W, Z = image_4ch.shape
    pred_slices = []

    for z in range(Z):
        x2d = image_4ch[:, :, :, z].unsqueeze(0).to(device)  # (1,4,H,W)
        logits = inferer(x2d, model)                          # (1,1,H,W)
        probs = torch.sigmoid(logits).squeeze(0)              # (1,H,W)
        pred_slices.append(probs.cpu())

    prob_vol = torch.stack(pred_slices, dim=-1)  # (1,H,W,Z)
    pred_bin = (prob_vol > 0.5).to(torch.uint8)
    return pred_bin


# 6) Choose informative slice

def choose_best_slice(gt: np.ndarray, pred: np.ndarray) -> int:
    """
    Choose slice with largest tumor area.
    Priority:
    1) max GT tumor area
    2) if GT empty, max Pred area
    3) fallback = middle slice
    """
    if gt.ndim == 4:
        gt = gt[0]
    if pred.ndim == 4:
        pred = pred[0]

    z = gt.shape[-1]
    gt_areas = gt.reshape(-1, z).sum(axis=0)
    if gt_areas.max() > 0:
        return int(gt_areas.argmax())

    pred_areas = pred.reshape(-1, z).sum(axis=0)
    if pred_areas.max() > 0:
        return int(pred_areas.argmax())

    return z // 2


# 7) Save overlay figure

def save_overlay_figure(
    out_path: str,
    img2d: np.ndarray,
    gt2d: np.ndarray,
    pred2d: np.ndarray,
    title: str,
):
    """
    Saves a 2x2 panel:
    - MRI
    - MRI + GT contour
    - MRI + Pred contour
    - MRI + GT (solid) + Pred (dashed)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(img2d, cmap="gray")
    axes[0, 0].set_title("MRI slice")

    axes[0, 1].imshow(img2d, cmap="gray")
    axes[0, 1].contour(gt2d, levels=[0.5], linewidths=2)
    axes[0, 1].set_title("GT mask contour")

    axes[1, 0].imshow(img2d, cmap="gray")
    axes[1, 0].contour(pred2d, levels=[0.5], linewidths=2)
    axes[1, 0].set_title("Predicted mask contour")

    axes[1, 1].imshow(img2d, cmap="gray")
    axes[1, 1].contour(gt2d, levels=[0.5], linewidths=2)
    axes[1, 1].contour(pred2d, levels=[0.5], linewidths=2, linestyles="--")
    axes[1, 1].set_title("GT (solid) + Pred (dashed)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# 8) Load metrics and select patients

def read_metrics_csv(csv_path: str) -> List[Dict]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Metrics CSV not found:\n{csv_path}\n\n"
            "Make sure you ran the full-volume metrics evaluation first."
        )

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Parse dice as float (CSV stores it as string)
            r["dice"] = float(r["dice"])
            r["fold"] = int(r["fold"])
            rows.append(r)
    return rows


def select_best_worst_medians(rows_fold: List[Dict], n_medians: int = 3) -> List[Tuple[str, str, float]]:
    """
    Returns list of tuples: (category, patient_id, dice)
    category is one of: best, worst, median1, median2, median3
    """
    if not rows_fold:
        return []

    # Sort by dice ascending
    rows_sorted = sorted(rows_fold, key=lambda x: x["dice"])
    worst = rows_sorted[0]
    best = rows_sorted[-1]

    # Median selection: pick items closest to median dice
    dices = np.array([r["dice"] for r in rows_sorted], dtype=np.float32)
    med = float(np.median(dices))

    # Exclude best/worst from median candidates (avoid duplicates)
    candidates = [r for r in rows_sorted if r["patient_id"] not in {best["patient_id"], worst["patient_id"]}]
    candidates_sorted = sorted(candidates, key=lambda r: abs(r["dice"] - med))

    medians = candidates_sorted[:n_medians]

    selected: List[Tuple[str, str, float]] = [
        ("worst", worst["patient_id"], float(worst["dice"])),
        ("best", best["patient_id"], float(best["dice"])),
    ]

    for i, r in enumerate(medians, start=1):
        selected.append((f"median{i}", r["patient_id"], float(r["dice"])))

    return selected


# 9) Config

@dataclass
class OverlayConfig:
    k_folds: int = 5
    seed: int = 42
    roi_hw: Tuple[int, int] = (128, 128)
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    sw_batch_size: int = 4

    # Background modality for visualization:
    # 0=T1, 1=T1c, 2=T2, 3=FLAIR
    background_channel: int = 1  # T1c


def main():
    cfg = OverlayConfig()

    print("=== Ranked overlays started ===")
    print("Metrics CSV:", METRICS_CSV)
    print("Dataset JSON:", DATASET_JSON)
    print("Output dir :", OVERLAYS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load metrics CSV
    metrics_rows = read_metrics_csv(METRICS_CSV)

    # Load dataset JSON (for paths)
    dataset_list = load_dataset_list(DATASET_JSON)
    by_id = {d["patient_id"]: d for d in dataset_list}

    # Preprocessing
    xform = get_full_volume_eval_transforms(pixdim=cfg.pixdim)

    os.makedirs(OVERLAYS_DIR, exist_ok=True)

    for fold_idx in range(cfg.k_folds):
        print(f"\n--- Fold {fold_idx}/{cfg.k_folds - 1} ---")

        # Filter metrics to this fold
        rows_fold = [r for r in metrics_rows if r["fold"] == fold_idx]
        if not rows_fold:
            print(f"WARNING: no metrics rows found for fold {fold_idx}, skipping.")
            continue

        selected = select_best_worst_medians(rows_fold, n_medians=3)

        # Load fold model
        model_path = os.path.join(TRAIN_OUT_DIR, f"fold_{fold_idx}", "best_model.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")

        model = make_model(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        fold_out_dir = os.path.join(OVERLAYS_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_out_dir, exist_ok=True)

        for category, pid, dice in selected:
            if pid not in by_id:
                print(f"WARNING: patient {pid} not found in dataset JSON, skipping.")
                continue

            item = by_id[pid]
            sample = {"image": item["image"], "label": item["label"]}

            # Apply preprocessing
            sample_t = xform(sample)
            img = sample_t["image"]                     # (4,H,W,Z)
            gt = sample_t["label"].to(torch.uint8)      # (1,H,W,Z)

            # Predict full volume
            pred = predict_full_volume_2d_slice_by_slice(
                model=model,
                image_4ch=img,
                roi_hw=cfg.roi_hw,
                device=device,
                sw_batch_size=cfg.sw_batch_size,
            )

            # Convert to numpy for plotting
            img_np = img.cpu().numpy()
            gt_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()

            # Choose slice
            z_idx = choose_best_slice(gt_np, pred_np)

            ch = cfg.background_channel
            img2d = img_np[ch, :, :, z_idx]
            gt2d = gt_np[0, :, :, z_idx]
            pred2d = pred_np[0, :, :, z_idx]

            # Output folder per patient/category
            patient_dir = os.path.join(fold_out_dir, f"{category}_{pid}")
            os.makedirs(patient_dir, exist_ok=True)

            out_png = os.path.join(patient_dir, f"{category}_{pid}_dice{dice:.3f}_z{z_idx:03d}.png")
            title = f"{pid} | fold {fold_idx} | {category} | Dice={dice:.3f} | z={z_idx} | bg_ch={ch}"

            save_overlay_figure(out_png, img2d, gt2d, pred2d, title)
            print(f"Saved {category}: {out_png}")

    print("\n=== Done. Ranked overlays saved under: ===")
    print(OVERLAYS_DIR)


if __name__ == "__main__":
    main()
