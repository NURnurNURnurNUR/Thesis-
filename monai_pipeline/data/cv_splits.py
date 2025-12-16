"""
Patient-level K-fold split utilities for this project.

Why this file exists:
- Cross-validation (CV) must split by PATIENT, not by slices/patches.
- This prevents "data leakage" (same patient in train and val).
- We want deterministic folds so results are reproducible.

What it does:
- Loads the dataset list from outputs/dataset_final_index.json
- Generates K folds (default K=5) using a fixed random seed
- Returns (train_samples, val_samples) for a requested fold_index
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Tuple

import numpy as np

from monai_pipeline.config.paths import OUTPUTS_DIR

# 1) Load the dataset list (list of dicts) from a JSON file

def load_samples_from_json(json_path: str) -> List[Dict]:
    """
    Load the dataset list (list of dicts) from a JSON file.

    Each dict should include at least:
      - patient_id
      - image (list of paths)
      - label (mask path)
      - label_hgg (0/1)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Dataset JSON is empty or not a list.")

    # Ensure that patient_id exists
    if "patient_id" not in samples[0]:
        raise ValueError("Dataset JSON samples do not contain 'patient_id' key.")

    return samples


# 2) Create k fold index arrays for n samples

def make_kfold_indices(
    n: int,
    k: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> List[np.ndarray]:
    """
    Create k fold index arrays for n samples.

    Parameters
    ----------
    n:
        Number of total samples (patients).
    k:
        Number of folds.
    seed:
        Random seed for reproducibility.
    shuffle:
        Whether to shuffle before splitting.

    Returns
    -------
    folds:
        List of numpy arrays, each array contains indices for that fold's validation set.
    """
    if k < 2:
        raise ValueError("k must be >= 2 for cross-validation.")
    if n < k:
        raise ValueError(f"Not enough samples (n={n}) for k={k} folds.")

    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    # Split indices into k nearly-equal chunks (folds)
    folds = np.array_split(indices, k)
    return folds


# 3) Return (train_samples, val_samples) for a given fold

def get_fold_samples(
    dataset_json_path: str | None = None,
    fold_index: int = 0,
    k: int = 5,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (train_samples, val_samples) for a given fold.

    Fold definition:
    - val_samples = fold[fold_index]
    - train_samples = all other folds combined

    Parameters
    ----------
    dataset_json_path:
        Path to outputs/dataset_final_index.json. If None, uses default OUTPUTS_DIR.
    fold_index:
        Which fold to use as validation (0..k-1).
    k:
        Number of folds.
    seed:
        Seed controlling fold shuffling (reproducible).

    Returns
    -------
    train_samples, val_samples
    """
    if dataset_json_path is None:
        dataset_json_path = os.path.join(OUTPUTS_DIR, "dataset_final_index.json")

    samples = load_samples_from_json(dataset_json_path)
    n = len(samples)

    folds = make_kfold_indices(n=n, k=k, seed=seed, shuffle=True)

    if fold_index < 0 or fold_index >= k:
        raise ValueError(f"fold_index must be in [0, {k-1}], got {fold_index}")

    val_idx = folds[fold_index]
    train_idx = np.concatenate([folds[i] for i in range(k) if i != fold_index])

    val_samples = [samples[i] for i in val_idx.tolist()]
    train_samples = [samples[i] for i in train_idx.tolist()]

    return train_samples, val_samples

# 4) Test: print fold sizes

if __name__ == "__main__":
    for fold in range(5):
        tr, va = get_fold_samples(fold_index=fold, k=5, seed=42)
        print(f"Fold {fold}: train={len(tr)}, val={len(va)}")
