"""
ML Feature Extraction from UCSF-PDGM (GT masks)

This script builds a tabular dataset for an ML classifier (HGG vs LGG)
using features extracted from:
- 4 MRI modalities (T1, T1c, T2, FLAIR)
- tumor mask (whole tumor; binarized)

INPUT:
  indices/patient_index_final.csv
  Columns expected (from the pipeline):
    - patient_id
    - t1_file, t1c_file, t2_file, flair_file, mask_file
    - label_hgg
    - (optional) who_cns_grade, diagnosis_text, etc.

OUTPUT:
  outputs/ml_features/features_from_gtmask.csv

Notes:
- Uses GT masks -> best for a clean ML baseline.
- Later, to mimic real pipeline, we can extract features from *predicted* masks
  (ideally out-of-fold predictions to avoid data leakage).
"""

from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib

# 1) Config

@dataclass
class FeatureConfig:
    # Paths are repo-relative to keep it GitHub-friendly
    project_root: str
    index_csv: str
    output_csv: str

    # Tumor mask processing
    mask_threshold: float = 0.0   # whole tumor: mask > 0
    ring_dilation_vox: int = 3    # ring thickness around tumor (in voxels)

    # Intensity stats percentiles
    percentiles: Tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)

    # Safety
    min_tumor_voxels: int = 10    # skip if tumor is too tiny (noise / empty)


# 2) Convert strings to floats

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# 3) Load NIfTI and return image as float32 numpy array (no header returned)

def load_nii(path: str) -> np.ndarray:
    """
    Load NIfTI and return image as float32 numpy array (no header returned).
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data

# 4) Convert mask to binary

def binarize_mask(mask: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """
    Convert mask to binary: everything > thr becomes 1.
    """
    return (mask > thr).astype(np.uint8)

# 5) Bounding box of nonzero voxels

def bbox_3d(binary: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Return bounding box of nonzero voxels as:
      (zmin, zmax, ymin, ymax, xmin, xmax)  with max exclusive
    """
    coords = np.argwhere(binary > 0)
    if coords.size == 0:
        return None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1
    return int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)


# 6) Compute voxel volume in mm^3 from NIfTI header zooms

def compute_voxel_volume_mm3(nifti_path: str) -> float:
    """
    Compute voxel volume in mm^3 from NIfTI header zooms.
    """
    img = nib.load(nifti_path)
    zooms = img.header.get_zooms()[:3]  # (x,y,z) or (dx,dy,dz)
    return float(zooms[0] * zooms[1] * zooms[2])


# 7) Build a simple ring around the tumor

def ring_mask(binary: np.ndarray, dilate_vox: int) -> np.ndarray:
    """
    Build a simple ring around the tumor:
      ring = dilate(tumor) - tumor

    Implemented using a cheap max-filter style dilation without scipy.
    This is not perfect morphology, but is stable and dependency-free.

    binary shape: (Z, Y, X)
    """
    if dilate_vox <= 0:
        return np.zeros_like(binary, dtype=np.uint8)

    z, y, x = binary.shape
    out = np.zeros_like(binary, dtype=np.uint8)

    # For each shift within +-dilate_vox, OR accumulate
    # Complexity is okay for small dilations (like 3).
    for dz in range(-dilate_vox, dilate_vox + 1):
        for dy in range(-dilate_vox, dilate_vox + 1):
            for dx in range(-dilate_vox, dilate_vox + 1):
                # skip far corners to approximate a ball-ish structuring element
                if abs(dz) + abs(dy) + abs(dx) > dilate_vox:
                    continue

                z0 = max(0, 0 + dz)
                y0 = max(0, 0 + dy)
                x0 = max(0, 0 + dx)

                z1 = min(z, z + dz)
                y1 = min(y, y + dy)
                x1 = min(x, x + dx)

                src_z0 = max(0, 0 - dz)
                src_y0 = max(0, 0 - dy)
                src_x0 = max(0, 0 - dx)

                src_z1 = src_z0 + (z1 - z0)
                src_y1 = src_y0 + (y1 - y0)
                src_x1 = src_x0 + (x1 - x0)

                out[z0:z1, y0:y1, x0:x1] |= binary[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]

    ring = (out > 0).astype(np.uint8)
    ring = np.clip(ring - binary, 0, 1).astype(np.uint8)
    return ring


# 8) Compute robust intensity stats on 1D array

def intensity_stats(values: np.ndarray, percentiles: Tuple[int, ...]) -> Dict[str, float]:
    """
    Compute robust intensity stats on 1D array.
    """
    values = values.astype(np.float32)
    if values.size == 0:
        # return NaNs for consistency
        stats = {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
        for p in percentiles:
            stats[f"p{p}"] = float("nan")
        return stats

    stats = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    perc = np.percentile(values, percentiles).astype(np.float32)
    for p, v in zip(percentiles, perc):
        stats[f"p{p}"] = float(v)
    return stats


# 9) Feature extraction per patient

def extract_features_for_patient(
    patient_dir: str,
    patient_id: str,
    files: Dict[str, str],
    label_hgg: int,
    cfg: FeatureConfig,
) -> Optional[Dict[str, object]]:
    """
    Extract one row of features.
    files keys: t1_file, t1c_file, t2_file, flair_file, mask_file
    """
    # Build full paths
    t1_path = os.path.join(patient_dir, files["t1_file"])
    t1c_path = os.path.join(patient_dir, files["t1c_file"])
    t2_path = os.path.join(patient_dir, files["t2_file"])
    flair_path = os.path.join(patient_dir, files["flair_file"])
    mask_path = os.path.join(patient_dir, files["mask_file"])

    # Load volumes
    t1 = load_nii(t1_path)
    t1c = load_nii(t1c_path)
    t2 = load_nii(t2_path)
    flair = load_nii(flair_path)
    mask_raw = load_nii(mask_path)

    # Ensure all shapes match 
    if not (t1.shape == t1c.shape == t2.shape == flair.shape == mask_raw.shape):
        # Skip to avoid garbage features
        return None

    # Convert to (Z,Y,X) if needed
    # Nibabel typically returns (X,Y,Z). Your pipeline may be (X,Y,Z).
    # We'll standardize to (Z,Y,X) for bbox/slice logic:
    # If it's 3D, assume (X,Y,Z) and transpose to (Z,Y,X).
    if t1.ndim != 3:
        return None

    t1_zyx = np.transpose(t1, (2, 1, 0))
    t1c_zyx = np.transpose(t1c, (2, 1, 0))
    t2_zyx = np.transpose(t2, (2, 1, 0))
    flair_zyx = np.transpose(flair, (2, 1, 0))
    mask_zyx = np.transpose(mask_raw, (2, 1, 0))

    tumor = binarize_mask(mask_zyx, thr=cfg.mask_threshold)
    tumor_vox = int(tumor.sum())
    if tumor_vox < cfg.min_tumor_voxels:
        return None

    # Physical volume (mm^3) from header spacing
    voxel_vol_mm3 = compute_voxel_volume_mm3(mask_path)
    tumor_vol_mm3 = tumor_vox * voxel_vol_mm3

    # BBox features
    bb = bbox_3d(tumor)
    if bb is None:
        return None
    zmin, zmax, ymin, ymax, xmin, xmax = bb
    bbox_dz = zmax - zmin
    bbox_dy = ymax - ymin
    bbox_dx = xmax - xmin
    bbox_vol_vox = bbox_dz * bbox_dy * bbox_dx

    # Slice-based area distribution
    slice_areas = tumor.sum(axis=(1, 2)).astype(np.int32)  # per Z slice
    slice_nonzero = np.count_nonzero(slice_areas)
    slice_area_mean = float(np.mean(slice_areas[slice_areas > 0]))
    slice_area_max = int(np.max(slice_areas))
    slice_area_median = float(np.median(slice_areas[slice_areas > 0]))

    # Ring around tumor (context)
    ring = ring_mask(tumor, cfg.ring_dilation_vox)
    ring_vox = int(ring.sum())

    # Intensity stats inside tumor and in ring, per modality
    def stats_for_mod(vol_zyx: np.ndarray, region: np.ndarray) -> Dict[str, float]:
        vals = vol_zyx[region > 0]
        # Remove extreme NaNs if any (shouldn't happen normally)
        vals = vals[np.isfinite(vals)]
        return intensity_stats(vals, cfg.percentiles)

    tumor_t1 = stats_for_mod(t1_zyx, tumor)
    tumor_t1c = stats_for_mod(t1c_zyx, tumor)
    tumor_t2 = stats_for_mod(t2_zyx, tumor)
    tumor_flair = stats_for_mod(flair_zyx, tumor)

    ring_t1 = stats_for_mod(t1_zyx, ring) if ring_vox > 0 else intensity_stats(np.array([], dtype=np.float32), cfg.percentiles)
    ring_t1c = stats_for_mod(t1c_zyx, ring) if ring_vox > 0 else intensity_stats(np.array([], dtype=np.float32), cfg.percentiles)
    ring_t2 = stats_for_mod(t2_zyx, ring) if ring_vox > 0 else intensity_stats(np.array([], dtype=np.float32), cfg.percentiles)
    ring_flair = stats_for_mod(flair_zyx, ring) if ring_vox > 0 else intensity_stats(np.array([], dtype=np.float32), cfg.percentiles)

    # Simple “contrast” features: tumor mean - ring mean (per modality)
    contrast_t1 = tumor_t1["mean"] - ring_t1["mean"]
    contrast_t1c = tumor_t1c["mean"] - ring_t1c["mean"]
    contrast_t2 = tumor_t2["mean"] - ring_t2["mean"]
    contrast_flair = tumor_flair["mean"] - ring_flair["mean"]

    # 10) Build output row

    row: Dict[str, object] = {
        "patient_id": patient_id,
        "label_hgg": int(label_hgg),

        # Tumor geometry
        "tumor_voxels": tumor_vox,
        "tumor_volume_mm3": float(tumor_vol_mm3),
        "bbox_dx": int(bbox_dx),
        "bbox_dy": int(bbox_dy),
        "bbox_dz": int(bbox_dz),
        "bbox_volume_vox": int(bbox_vol_vox),

        # Slice distribution
        "tumor_slices_nonzero": int(slice_nonzero),
        "slice_area_mean": float(slice_area_mean),
        "slice_area_median": float(slice_area_median),
        "slice_area_max": int(slice_area_max),

        # Ring
        "ring_voxels": int(ring_vox),
        "ring_dilation_vox": int(cfg.ring_dilation_vox),

        # Simple contrast (tumor minus ring)
        "contrast_t1_mean_minus_ring": float(contrast_t1),
        "contrast_t1c_mean_minus_ring": float(contrast_t1c),
        "contrast_t2_mean_minus_ring": float(contrast_t2),
        "contrast_flair_mean_minus_ring": float(contrast_flair),
    }

    def add_stats(prefix: str, stats: Dict[str, float]):
        for k, v in stats.items():
            row[f"{prefix}_{k}"] = float(v)

    # Add tumor intensity stats
    add_stats("tumor_t1", tumor_t1)
    add_stats("tumor_t1c", tumor_t1c)
    add_stats("tumor_t2", tumor_t2)
    add_stats("tumor_flair", tumor_flair)

    # Add ring intensity stats
    add_stats("ring_t1", ring_t1)
    add_stats("ring_t1c", ring_t1c)
    add_stats("ring_t2", ring_t2)
    add_stats("ring_flair", ring_flair)

    return row


# 11) Main runner

def main():
    # PROJECT_ROOT is 2 levels up from this file: thesis/ml_pipeline/features/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))

    index_csv = os.path.join(project_root, "indices", "patient_index_final.csv")

    out_dir = os.path.join(project_root, "outputs", "ml_features")
    os.makedirs(out_dir, exist_ok=True)

    output_csv = os.path.join(out_dir, "features_from_gtmask.csv")

    cfg = FeatureConfig(
        project_root=project_root,
        index_csv=index_csv,
        output_csv=output_csv,
    )

    print("PROJECT_ROOT:", cfg.project_root)
    print("INDEX CSV:", cfg.index_csv)
    print("OUTPUT CSV:", cfg.output_csv)

    # Read index CSV
    rows: List[Dict[str, str]] = []
    with open(cfg.index_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError("patient_index_final.csv is empty.")

    required = {"patient_id", "t1_file", "t1c_file", "t2_file", "flair_file", "mask_file", "label_hgg"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns in patient_index_final.csv: {sorted(missing)}")

    # Infer dataset root from the monai config structure:
    # We already have: thesis/UCSF-PDGM/UCSF-PDGM-v5/<patient_id>/
    # We support override via env var if someone stores data elsewhere.
    env_dataset = os.environ.get("UCSF_PDGM_DATASET")
    if env_dataset and os.path.isdir(env_dataset):
        dataset_root = env_dataset
    else:
        dataset_root = os.path.join(project_root, "UCSF-PDGM", "UCSF-PDGM-v5")

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            "Could not find dataset root.\n"
            f"Tried: {dataset_root}\n"
            "Fix:\n"
            "  - put dataset at <project_root>/UCSF-PDGM/UCSF-PDGM-v5\n"
            "  - OR set env var UCSF_PDGM_DATASET to the dataset path"
        )

    print("DATASET_ROOT:", dataset_root)

    # 12) Extract features
    
    out_rows: List[Dict[str, object]] = []
    skipped = 0

    for i, r in enumerate(rows, start=1):
        pid = r["patient_id"]
        patient_dir = os.path.join(dataset_root, pid)

        if not os.path.isdir(patient_dir):
            skipped += 1
            continue

        files = {
            "t1_file": r["t1_file"],
            "t1c_file": r["t1c_file"],
            "t2_file": r["t2_file"],
            "flair_file": r["flair_file"],
            "mask_file": r["mask_file"],
        }

        label_hgg = int(float(r["label_hgg"]))  # robust if stored as "1" or "1.0"

        try:
            feat = extract_features_for_patient(
                patient_dir=patient_dir,
                patient_id=pid,
                files=files,
                label_hgg=label_hgg,
                cfg=cfg,
            )
        except Exception as e:
            # If one patient has unexpected corruption, skip but continue
            feat = None

        if feat is None:
            skipped += 1
        else:
            out_rows.append(feat)

        if i % 25 == 0:
            print(f"Processed {i}/{len(rows)} ... extracted={len(out_rows)} skipped={skipped}")

    if not out_rows:
        raise RuntimeError("No features extracted. Something is wrong with paths or masks.")

    # 13) Write output CSV
    
    # Use union of keys to be safe
    all_keys = sorted({k for row in out_rows for k in row.keys()})

    with open(cfg.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(out_rows)

    print("\n=== Feature extraction done ===")
    print("Extracted rows:", len(out_rows))
    print("Skipped rows:", skipped)
    print("Saved:", cfg.output_csv)


if __name__ == "__main__":
    main()
