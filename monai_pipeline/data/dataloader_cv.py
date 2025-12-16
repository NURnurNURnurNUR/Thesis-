"""
Create MONAI + PyTorch DataLoaders for 5-fold cross-validation.

Why this file exists:
- Training needs batches of tensors (images + masks).
- MONAI handles medical imaging loading + transforms.
- 5-fold CV gives robust evaluation and avoids reliance on a single split.

Important OS note:
- On Windows, multiprocessing inside CacheDataset can deadlock/hang.
- Therefore we automatically set CacheDataset(num_workers=0) on Windows.
- On Linux/macOS we allow multiprocessing for faster caching.

How to run a debugging test:
  python -m monai_pipeline.data.dataloader_cv
"""

from __future__ import annotations

import os
import platform
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from monai.data import CacheDataset

from monai_pipeline.config.paths import OUTPUTS_DIR
from monai_pipeline.data.transforms import get_train_transforms_2d, get_val_transforms_2d
from monai_pipeline.data.cv_splits import get_fold_samples

# 1) Create (train_loader, val_loader) for a specific CV fold

def create_loaders_for_fold_2d(
    fold_index: int,
    k: int = 5,
    seed: int = 42,
    dataset_json_path: str | None = None,
    patch_size: Tuple[int, int] = (128, 128),
    num_samples_per_volume: int = 4,
    batch_size: int = 2,
    num_workers: int = 2,
    cache_rate: float = 0.2,
    cache_workers: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create (train_loader, val_loader) for a specific cross-validation fold.

    Parameters
    ----------
    fold_index:
        Which fold to use as validation (0..k-1).
    k:
        Number of folds (default 5).
    seed:
        Seed for reproducible fold splitting.
    dataset_json_path:
        Path to JSON list produced by build_dataset.py.
        Default: outputs/dataset_final_index.json
    patch_size:
        2D patch size (H, W).
    num_samples_per_volume:
        How many random patches to sample from each volume per epoch iteration.
        (Used by RandCropByPosNegLabeld in your train transforms.)
    batch_size:
        Batch size of patches (NOT patients).
        Start small for RTX 2080 8GB: 2 is a safe default.
    num_workers:
        Workers used by PyTorch DataLoader for loading/transforming.
        On Windows, start with 0 or 2. On Linux, 4+ can help.
    cache_rate:
        Fraction of dataset cached in RAM for speed.
        With 16GB RAM, 0.1–0.3 is usually safe.
    cache_workers:
        Workers used internally by CacheDataset while filling cache.
        If None:
          - Windows -> 0 (stability)
          - Linux/macOS -> num_workers (speed)

    Returns
    -------
    train_loader, val_loader
    """

    
    # 2) Resolve dataset JSON path

    if dataset_json_path is None:
        dataset_json_path = os.path.join(OUTPUTS_DIR, "dataset_final_index.json")

    # 3) OS-safe CacheDataset worker selection
    # Windows often hangs with multiprocessing in CacheDataset
    # We set cache_workers=0 on Windows for stability
    is_windows = platform.system().lower().startswith("win")
    if cache_workers is None:
        cache_workers = 0 if is_windows else num_workers

    # 4) Get train/val samples for this fold (patient-level split)

    train_samples, val_samples = get_fold_samples(
        dataset_json_path=dataset_json_path,
        fold_index=fold_index,
        k=k,
        seed=seed,
    )

    # 5) Build transforms

    train_tfms = get_train_transforms_2d(
        patch_size=patch_size,
        num_samples_per_volume=num_samples_per_volume,
    )
    val_tfms = get_val_transforms_2d(
        patch_size=patch_size,
    )

    # 6) Build MONAI CacheDatasets

    # CacheDataset caches deterministic steps (loading, spacing, orientation, etc.)
    # This speeds training significantly for NIfTI files.
    train_ds = CacheDataset(
        data=train_samples,
        transform=train_tfms,
        cache_rate=cache_rate,
        num_workers=cache_workers,  # OS-safe
    )

    val_ds = CacheDataset(
        data=val_samples,
        transform=val_tfms,
        cache_rate=cache_rate,
        num_workers=cache_workers,  # OS-safe
    )

    # 7) Build PyTorch DataLoaders

    # DataLoader handles:
    # - batching
    # - shuffling (train only)
    # - parallel loading/transforming (num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 8) Print summary (useful for debugging + thesis transparency)

    print("=== CV DataLoader summary ===")
    print(f"OS: {platform.system()}  |  is_windows={is_windows}")
    print(f"Fold: {fold_index}/{k-1} (seed={seed})")
    print(f"Train patients: {len(train_samples)}")
    print(f"Val patients:   {len(val_samples)}")
    print(f"Patch size: {patch_size} | patches/volume: {num_samples_per_volume}")
    print(f"Batch size: {batch_size}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Cache rate: {cache_rate}")
    print(f"CacheDataset workers (cache_workers): {cache_workers}")
    print("=============================")

    return train_loader, val_loader


# 9) Debugging test

# This code runs only when you execute this file directly:
#   python -m monai_pipeline.data.dataloader_cv
#
# It does not run when imported by training scripts.
if __name__ == "__main__":
    train_loader, val_loader = create_loaders_for_fold_2d(
        fold_index=0,
        k=5,
        seed=42,
        patch_size=(128, 128),
        num_samples_per_volume=4,
        batch_size=2,     # safe default for RTX 2080 8GB
        num_workers=2,    # DataLoader workers
        cache_rate=0.2,
        cache_workers=None,  # None -> auto OS-safe choice
    )

    # Pull one batch from the training loader to confirm that everything works
    batch = next(iter(train_loader))
    
    # RandCropByPosNegLabeld with num_samples>1 can produce list outputs.
    # If we got a list, inspect the first element.
    if isinstance(batch, list):
        print(f"Batch is a list with length: {len(batch)}")
        sample0 = batch[0]
        print("First element type:", type(sample0))
        if isinstance(sample0, dict):
            print("Keys:", sample0.keys())
            print("image shape:", tuple(sample0["image"].shape))
            print("label shape:", tuple(sample0["label"].shape))
        else:
            print("First element is not a dict; cannot show keys.")
    else:
        # Normal case: dict batch
        print("Batch type:", type(batch))
        print("Batch keys:", batch.keys())
        print("image shape:", tuple(batch["image"].shape))
        print("label shape:", tuple(batch["label"].shape))

    print("=======================")
