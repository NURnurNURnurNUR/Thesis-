"""
Build a MONAI-style dataset list (list of dictionaries) from:
  indices/patient_index_final.csv

Why this exists:
- MONAI expects data as a list of dicts with file paths.
- We keep it reproducible by using the "final index" CSV.

Output:
- A Python list of dict samples (for training scripts)
- Writes a JSON file to outputs/ for transparency
run like this python -m monai_pipeline.data.build_dataset since acts like module
"""

from __future__ import annotations

import os
import csv
import json
from typing import List, Dict, Tuple

# 1) Import central paths 

from monai_pipeline.config.paths import (
    UCSF_PDGM_DATASET_DIR,
    PATIENT_INDEX_FINAL_CSV,
    OUTPUTS_DIR,
)

# 2) Build samples from the final index

def build_samples_from_final_index(
    final_csv_path: str = PATIENT_INDEX_FINAL_CSV,
    dataset_dir: str = UCSF_PDGM_DATASET_DIR,
) -> List[Dict]:
    """
    Read patient_index_final.csv and build MONAI sample dictionaries.

    Each sample contains:
      - patient_id (folder name)
      - image: list of 4 modality file paths [T1, T1c, T2, FLAIR]
      - label: mask file path
      - label_hgg: 0 or 1 (classification label)
      - who_cns_grade: string/int (optional, for reporting)

    Returns:
      samples: list of dicts
    """

    samples: List[Dict] = []

    with open(final_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"patient_id", "t1_file", "t1c_file", "t2_file", "flair_file", "mask_file", "label_hgg"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "final CSV is missing required columns:\n"
                f"{sorted(missing)}\n\n"
                "Make sure all previous indexing steps were completed successfully.\n"
            )

        for row in reader:
            patient_id = row["patient_id"].strip()
            patient_dir = os.path.join(dataset_dir, patient_id)

            # Build full paths to each modality (4 channels)
            t1_path = os.path.join(patient_dir, row["t1_file"].strip())
            t1c_path = os.path.join(patient_dir, row["t1c_file"].strip())
            t2_path = os.path.join(patient_dir, row["t2_file"].strip())
            flair_path = os.path.join(patient_dir, row["flair_file"].strip())

            # Label (tumor segmentation mask)
            mask_path = os.path.join(patient_dir, row["mask_file"].strip())

            # Classification label (HGG=1, LGG=0). Keep as int for PyTorch.
            label_hgg_str = str(row["label_hgg"]).strip()
            label_hgg = int(label_hgg_str) if label_hgg_str != "" else -1

            # Metadata fields 
            who_grade = row.get("who_cns_grade", "")

            sample = {
                "patient_id": patient_id,
                # MONAI can load multi-channel if a list is passed
                "image": [t1_path, t1c_path, t2_path, flair_path],
                "label": mask_path,
                # for the classifier later
                "label_hgg": label_hgg,
                "who_cns_grade": who_grade,
            }

            samples.append(sample)

    return samples


# 3) Verify samples exist

def verify_samples_exist(samples: List[Dict], max_checks: int = 10) -> Tuple[int, List[str]]:
    """
    Verify that files exist on disk for a few samples.

    Returns:
      missing_count: number of missing file paths found (within checked subset)
      missing_examples: list of up to a few missing file paths
    """
    missing_count = 0
    missing_examples: List[str] = []

    for i, s in enumerate(samples[:max_checks]):
        # Check all modality files
        for p in s["image"]:
            if not os.path.isfile(p):
                missing_count += 1
                if len(missing_examples) < 20:
                    missing_examples.append(p)

        # Check mask file
        if not os.path.isfile(s["label"]):
            missing_count += 1
            if len(missing_examples) < 20:
                missing_examples.append(s["label"])

    return missing_count, missing_examples

# 4) Save samples to JSON

def save_samples_json(samples: List[Dict], out_path: str) -> None:
    """
    Save the dataset list to JSON for reproducibility / debugging.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)


if __name__ == "__main__":
    samples = build_samples_from_final_index()

    print(f"Loaded training-ready patients from: {PATIENT_INDEX_FINAL_CSV}")
    print(f"Dataset root: {UCSF_PDGM_DATASET_DIR}")
    print(f"Total samples: {len(samples)}")

    missing_count, missing_examples = verify_samples_exist(samples, max_checks=10)
    print(f"Check if the first 10 samples are missing paths found: {missing_count}")
    if missing_examples:
        print("Examples of missing paths:")
        for p in missing_examples[:10]:
            print("  -", p)

    # Write JSON for transparency
    out_json = os.path.join(OUTPUTS_DIR, "dataset_final_index.json")
    save_samples_json(samples, out_json)
    print(f"Saved JSON dataset list: {out_json}")
