"""
FEATURE EXTRACTION FROM DL-PREDICTED MASKS (5-FOLD)

Goal:
- For each fold:
  - load the DL segmentation model for that fold
  - run inference on that fold's VALIDATION patients
  - get a predicted whole-tumor mask
  - extract features using the predicted mask
- Save one combined CSV:
    outputs/ml_predmask_features/features_from_predmask.csv

Why:
- This simulates the real end-to-end pipeline:
    MRI -> DL segmentation -> ML classification (HGG/LGG)

Notes:
- This does not retrain DL.
- This does not change the dataset splits: the fold assignment is deterministic.
"""

from __future__ import annotations

import os
import json
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sklearn.model_selection import KFold

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    DivisiblePadd,
)
from monai.networks.nets import UNet

from monai_pipeline.config.paths import (
    PROJECT_ROOT,
    UCSF_PDGM_DATASET_DIR,
    PATIENT_INDEX_FINAL_CSV,
    OUTPUTS_DIR,
)

# 1) Config

@dataclass
class PredMaskFeatureConfig:
    k_folds: int = 5
    seed: int = 42

    # Same preprocessing spacing as used in the MONAI pipeline
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # 2D inference window size (must match training patch size)
    roi_hw: Tuple[int, int] = (128, 128)

    # If True, saves predicted masks as .npy for debugging (optional)
    save_pred_masks_npy: bool = False


# 2) Model (same as your training)

def make_unet_2d(device: torch.device) -> nn.Module:
    """
    2D U-Net:
      input:  (B, 4, H, W)
      output: (B, 1, H, W)
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


# 3) Preprocessing transforms (full volume, no random crop)

def get_infer_preprocess(cfg: PredMaskFeatureConfig) -> Compose:
    """
    IMPORTANT:
    - No RandCrop or augmentation here.
    - We want stable full-volume inference.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=cfg.pixdim,
                mode=("bilinear", "nearest"),
            ),

            # Crop around the tumor region to reduce background
            CropForegroundd(keys=["image", "label"], source_key="label"),

            # U-Net downsamples 4 times -> H and W must be divisible by 16.
            # Pad both image and label so shapes stay aligned.
            DivisiblePadd(keys=["image", "label"], k=16),

            # Normalize intensity after spatial ops
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )



# 4) Simple “radiomics-style” features (stable + reproducible)

# This matches the goal: features from mask + intensities.
# (It may not be identical to pyradiomics, but it is reproducible)
def _safe_stats(x: np.ndarray) -> Dict[str, float]:
    """Compute robust stats for a 1D array; return NaNs if empty."""
    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }

# 5) Extract features from predicted mask

def extract_features_from_predmask(
    image_4ch: np.ndarray,   # (4, H, W, Z)
    pred_mask: np.ndarray,   # (H, W, Z) binary {0,1}
) -> Dict[str, float]:
    """
    Extract a reproducible feature set from the predicted mask.

    Features include:
    - shape/size features from the mask
    - intensity stats per modality inside the mask
    """
    feats: Dict[str, float] = {}

    mask = (pred_mask > 0).astype(np.uint8)
    voxels = int(mask.sum())
    feats["tumor_voxels"] = float(voxels)
    feats["tumor_voxel_fraction"] = float(voxels / mask.size)

    # Bounding box features
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        # no tumor predicted -> fill zeros/NaNs safely
        feats["bbox_h"] = 0.0
        feats["bbox_w"] = 0.0
        feats["bbox_z"] = 0.0
        feats["tumor_slices"] = 0.0
        feats["tumor_centroid_h"] = np.nan
        feats["tumor_centroid_w"] = np.nan
        feats["tumor_centroid_z"] = np.nan
    else:
        (h0, w0, z0) = coords.min(axis=0)
        (h1, w1, z1) = coords.max(axis=0)
        feats["bbox_h"] = float(h1 - h0 + 1)
        feats["bbox_w"] = float(w1 - w0 + 1)
        feats["bbox_z"] = float(z1 - z0 + 1)

        feats["tumor_slices"] = float(len(np.unique(coords[:, 2])))

        centroid = coords.mean(axis=0)
        feats["tumor_centroid_h"] = float(centroid[0])
        feats["tumor_centroid_w"] = float(centroid[1])
        feats["tumor_centroid_z"] = float(centroid[2])

    # Intensity stats per modality inside tumor
    for c, name in enumerate(["T1", "T1c", "T2", "FLAIR"]):
        vol = image_4ch[c]  # (H, W, Z)
        inside = vol[mask > 0].astype(np.float32)

        stats = _safe_stats(inside)
        for k, v in stats.items():
            feats[f"{name}_{k}"] = v

    return feats


# 6) 2D slice-by-slice inference function

@torch.no_grad()
def predict_mask_full_volume_2d(
    model: nn.Module,
    image_4ch: torch.Tensor,  # (1, 4, H, W, Z)
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run slice-by-slice 2D inference over Z.

    Returns:
      pred_mask: (H, W, Z) uint8
    """
    model.eval()

    # Move to device
    image_4ch = image_4ch.to(device)

    _, _, H, W, Z = image_4ch.shape
    out = np.zeros((H, W, Z), dtype=np.uint8)

    for z in range(Z):
        # slice tensor: (1,4,H,W)
        x = image_4ch[..., z]  # (1,4,H,W)
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).to(torch.uint8)  # (1,1,H,W)
        out[..., z] = pred.squeeze(0).squeeze(0).detach().cpu().numpy()

    return out


# 7) Main function

def main():
    cfg = PredMaskFeatureConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("OS:", os.name)
    print("Project root:", PROJECT_ROOT)
    print("Dataset root:", UCSF_PDGM_DATASET_DIR)
    print("Device:", device)

    # Read training-ready CSV (patients + file names + label_hgg)
    df = pd.read_csv(PATIENT_INDEX_FINAL_CSV)
    df = df.reset_index(drop=True)

    # Deterministic fold assignment from patient list
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    fold_id = np.full(len(df), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(kf.split(df)):
        fold_id[val_idx] = fold
    df["fold"] = fold_id

    # Output folders
    out_dir = os.path.join(OUTPUTS_DIR, "ml_predmask_features")
    os.makedirs(out_dir, exist_ok=True)

    debug_masks_dir = os.path.join(out_dir, "pred_masks_npy")
    if cfg.save_pred_masks_npy:
        os.makedirs(debug_masks_dir, exist_ok=True)

    preprocess = get_infer_preprocess(cfg)

    all_rows: List[Dict] = []

    for fold in range(cfg.k_folds):
        print(f"\n===== Fold {fold} inference on VAL patients =====")

        # Load the DL model for this fold
        dl_model_path = os.path.join(
            OUTPUTS_DIR, "unet2d_cv", f"fold_{fold}", "best_model.pt"
        )
        if not os.path.isfile(dl_model_path):
            raise FileNotFoundError(f"Missing DL model for fold {fold}:\n{dl_model_path}")

        model = make_unet_2d(device)
        model.load_state_dict(torch.load(dl_model_path, map_location=device))

        # Only evaluate on that fold’s val set
        val_df = df[df["fold"] == fold].copy()
        print("Patients:", len(val_df))

        for _, r in val_df.iterrows():
            patient_id = r["patient_id"]
            y = int(r["label_hgg"])

            # Build file paths from filenames stored in CSV
            patient_dir = os.path.join(UCSF_PDGM_DATASET_DIR, patient_id)
            img_paths = [
                os.path.join(patient_dir, r["t1_file"]),
                os.path.join(patient_dir, r["t1c_file"]),
                os.path.join(patient_dir, r["t2_file"]),
                os.path.join(patient_dir, r["flair_file"]),
            ]
            lbl_path = os.path.join(patient_dir, r["mask_file"])

            sample = {"image": img_paths, "label": lbl_path}
            sample = preprocess(sample)

            # MONAI gives:
            # image: (4, H, W, Z)
            # label: (1, H, W, Z)
            img = sample["image"]
            lbl = sample["label"]

            # Convert to torch for inference
            # image_4ch: (1,4,H,W,Z)
            image_4ch = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)

            # Predict mask (H,W,Z)
            pred_mask = predict_mask_full_volume_2d(model, image_4ch, device=device)

            # Extract features from predicted mask (use numpy image)
            img_np = np.asarray(img)  # (4,H,W,Z)
            feats = extract_features_from_predmask(img_np, pred_mask)

            out_row = {
                "patient_id": patient_id,
                "fold": fold,
                "label_hgg": y,
                **feats,
            }
            all_rows.append(out_row)

            if cfg.save_pred_masks_npy:
                np.save(os.path.join(debug_masks_dir, f"{patient_id}_fold{fold}.npy"), pred_mask)

    out_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(out_dir, "features_from_predmask.csv")
    out_df.to_csv(out_csv, index=False)

    print("\n=== Pred-mask feature extraction done ===")
    print("Saved:", out_csv)
    print("Rows:", len(out_df))
    print("Columns:", len(out_df.columns))


if __name__ == "__main__":
    main()
