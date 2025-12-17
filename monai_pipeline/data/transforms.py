"""
MONAI transforms for:
- 2D U-Net segmentation
- 4-channel multi-modal MRI input (T1, T1c, T2, FLAIR)
- Binary whole-tumor mask (tumor = 1, background = 0)
- On-the-fly 2D patch sampling (no patches saved to disk)

Key idea:
- We load 3D volumes but train a 2D network by sampling 2D patches (H, W)
  from random slices using RandCropByPosNegLabeld.
"""

from __future__ import annotations

import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    SpatialPadd,
    CenterSpatialCropd,

    # 🔹 foreground / normalization
    CropForegroundd,
    NormalizeIntensityd,

    # 🔹 random patch sampling
    RandCropByPosNegLabeld,

    # 🔹 data augmentation
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,

    # 🔹 utility
    LambdaD,
)


# 1) Convert mask to binary whole tumor

def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """
    Convert any non-zero label to 1 (whole tumor), else 0.

    Works whether the mask labels are:
      - {0,1}
      - {0,1,2,4} (BraTS-style)
      - other multi-class encodings

    Output dtype is uint8 to save memory.
    """
    return (mask > 0).astype(np.uint8)


# 2) Main transform builders

def get_train_transforms_2d(
    patch_size: tuple[int, int] = (128, 128),
    num_samples_per_volume: int = 4,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Training transforms for 2D U-Net segmentation.

    Parameters
    ----------
    patch_size:
        2D patch size (H, W).
    num_samples_per_volume:
        How many random 2D patches to sample from each 3D volume per epoch iteration.
        More samples = more variety but slower.
    pixdim:
        Target voxel spacing (x, y, z) in mm. (Standardizes resolution.)

    Returns
    -------
    Compose
        MONAI transform pipeline for training.
    """

    keys_img = "image"
    keys_lbl = "label"

    return Compose(
        [
            # Load NIfTI from disk
            # "image" is a list of 4 paths -> MONAI loads them and stacks later
            LoadImaged(keys=[keys_img, keys_lbl]),

            # Ensure channel-first
            # - image becomes (4, X, Y, Z)
            # - label becomes (1, X, Y, Z)
            EnsureChannelFirstd(keys=[keys_img, keys_lbl]),

            # Standardize orientation to RAS (common neuroimaging convention)
            Orientationd(keys=[keys_img, keys_lbl], axcodes="RAS"),

            # Resample to common voxel spacing
            # - images: linear interpolation
            # - labels: nearest-neighbor (so mask labels don't get fractional)
            Spacingd(
                keys=[keys_img, keys_lbl],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),

            # Convert mask to binary whole tumor
            LambdaD(keys=keys_lbl, func=_mask_to_binary),

            # Crop around non-zero region (foreground)
            # This removes a lot of empty background and speeds up training.
            CropForegroundd(keys=[keys_img, keys_lbl], source_key=keys_lbl),

            # Intensity normalization (per channel)
            # MRI intensities are not standardized like CT; this helps training stability.
            NormalizeIntensityd(keys=keys_img, nonzero=True, channel_wise=True),

            # On-the-fly patch sampling (2D patches)
            # RandCropByPosNegLabeld samples around tumor (pos) and background (neg).
            # spatial_size is (H, W, 1) to effectively take 2D slices.
            RandCropByPosNegLabeld(
                keys=[keys_img, keys_lbl],
                label_key=keys_lbl,
                spatial_size=(patch_size[0], patch_size[1], 1),
                pos=1,
                neg=1,
                num_samples=num_samples_per_volume,
                image_key=keys_img,
                image_threshold=0,
                allow_smaller=True,
            ),

            # IMPORTANT:
            # Even with allow_smaller=True, some samples can still be smaller than
            # patch_size due to cropping near borders or small volumes.
            # SpatialPadd ensures every output patch is exactly (H, W, 1),
            # so the DataLoader can stack them into batches.
            SpatialPadd(
                keys=[keys_img, keys_lbl],
                spatial_size=(patch_size[0], patch_size[1], 1),
                mode=("constant", "constant"),
            ),

            # 3) Simple augmentations (safe for MRI)
            RandFlipd(keys=[keys_img, keys_lbl], spatial_axis=0, prob=0.5),
            RandFlipd(keys=[keys_img, keys_lbl], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=[keys_img, keys_lbl], prob=0.3, max_k=3),
            RandShiftIntensityd(keys=keys_img, offsets=0.1, prob=0.3),

            # 4) Ensure tensors / correct dtype for PyTorch training
            EnsureTyped(keys=[keys_img, keys_lbl]),
        ]
    )


def get_val_transforms_2d(
    patch_size: tuple[int, int] = (128, 128),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Validation must return SAME-SHAPE tensors so DataLoader can stack them.

    We:
      - load images + label
      - standardize orientation/spacing
      - convert label to binary (so it matches training!)
      - intensity normalize (same approach as training for consistency)
      - pad up to patch size (so crop never fails)
      - center-crop a fixed patch (deterministic)
    """
    keys_img = "image"
    keys_lbl = "label"

    roi = (patch_size[0], patch_size[1], 1)  # (H, W, depth=1)

    return Compose(
        [
            # Load NIfTI from disk
            LoadImaged(keys=[keys_img, keys_lbl]),

            # Ensure channel-first
            # - image becomes (4, X, Y, Z)
            # - label becomes (1, X, Y, Z)
            EnsureChannelFirstd(keys=[keys_img, keys_lbl]),

            # Standardize orientation to RAS (same as training)
            Orientationd(keys=[keys_img, keys_lbl], axcodes="RAS"),

            # Resample to common voxel spacing (same as training)
            Spacingd(
                keys=[keys_img, keys_lbl],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),

            # Convert mask to binary whole tumor (IMPORTANT: match training)
            LambdaD(keys=keys_lbl, func=_mask_to_binary),

            # Intensity normalization (same as training for fair evaluation)
            NormalizeIntensityd(keys=keys_img, nonzero=True, channel_wise=True),

            # Ensure volume is at least roi size so the next crop always works
            SpatialPadd(keys=[keys_img, keys_lbl], spatial_size=roi),

            # Deterministic crop so every output is exactly (H,W,1)
            CenterSpatialCropd(keys=[keys_img, keys_lbl], roi_size=roi),

            # Ensure tensors / correct dtype for PyTorch metrics
            EnsureTyped(keys=[keys_img, keys_lbl]),
        ]
    )
