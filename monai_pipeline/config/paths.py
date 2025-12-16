"""
Central place for all important paths in this thesis project.

Why this file exists:
- To avoid hardcoding absolute paths like C:\\Users\\...
- To make the project reproducible on GitHub for other users
- So that all MONAI scripts import paths from one place

How it works:
1) It determines PROJECT_ROOT (the "thesis/" folder).
2) It finds the UCSF-PDGM dataset folder:
   - First tries environment variable UCSF_PDGM_DATASET (recommended for users)
   - Otherwise tries common folder names inside the project root
   - Supports a one-level nested folder (like ".../PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5/")
3) It defines standard project folders:
   - indices/, metadata/, outputs/
"""

from __future__ import annotations

import os
from typing import Optional, List


# 1) Project root

# This file is in: thesis/monai/config/paths.py
# We go up two levels to reach: thesis/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# 2) Helper functions

def _contains_patient_folders(folder: str) -> bool:
    # Return True if folder contains UCSF-PDGM patient folders directly
    try:
        entries = [
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ]
    except FileNotFoundError:
        return False

    return any(e.startswith("UCSF-PDGM-") and e.endswith("_nifti") for e in entries)


def _find_patient_root(start_folder: str) -> Optional[str]:
    """
    Try to return the folder that directly contains patient folders

    We check:
    A) start_folder contains patient folders directly
    B) one nested folder level contains patient folders

    Returns:
      - Path string if found
      - None if not found
    """
    if not os.path.isdir(start_folder):
        return None

    # Case A: patient folders directly inside start_folder
    if _contains_patient_folders(start_folder):
        return start_folder

    # Case B: one folder deeper
    subfolders = [
        d for d in os.listdir(start_folder)
        if os.path.isdir(os.path.join(start_folder, d))
    ]
    for sub in subfolders:
        candidate = os.path.join(start_folder, sub)
        if _contains_patient_folders(candidate):
            return candidate

    return None


def resolve_ucsf_pdgm_dataset_dir(project_root: str) -> str:
    """
    Resolve the dataset folder that directly contains UCSF-PDGM patient folders

    Priority:
      1) Environment variable UCSF_PDGM_DATASET (recommended for other users)
      2) Common dataset folder names inside the repository (project_root)

    This is designed to support the original TCIA-style folder name:
      "PKG - UCSF-PDGM Version 5"
    and also cleaner names like:
      "UCSF-PDGM"
    """
    # 1) Environment variable override (best for other people's machines)
    env_path = os.environ.get("UCSF_PDGM_DATASET")
    if env_path:
        if not os.path.isdir(env_path):
            raise FileNotFoundError(
                f"UCSF_PDGM_DATASET is set but folder does not exist:\n{env_path}"
            )
        # env_path must be either the patient-root itself, or a parent that contains it
        found = _find_patient_root(env_path)
        if found is not None:
            return found
        raise FileNotFoundError(
            "UCSF_PDGM_DATASET is set, but no patient folders were found inside it.\n"
            "Make sure it points to a folder containing UCSF-PDGM-XXXX_nifti folders "
            "or a parent folder one level above them.\n"
            f"Given: {env_path}"
        )

    # 2) Search common folder names inside the project root
    common_names: List[str] = [
        "PKG - UCSF-PDGM Version 5",  # original TCIA download name
        "UCSF-PDGM",
        "UCSF-PDGM-v5",
        "UCSF-PDGM Version 5",
    ]

    for name in common_names:
        start = os.path.join(project_root, name)
        found = _find_patient_root(start)
        if found is not None:
            return found

    # If nothing worked, raise an error that tells the user what to do
    raise FileNotFoundError(
        "Could not locate the UCSF-PDGM dataset folder.\n\n"
        "Fix options:\n"
        "  A) Put the dataset inside the project root with one of these names:\n"
        f"     {common_names}\n"
        "     (patient folders can be directly inside or one nested folder deeper)\n"
        "  B) OR set environment variable UCSF_PDGM_DATASET to the dataset path.\n\n"
        "Example (PowerShell):\n"
        "  $env:UCSF_PDGM_DATASET='D:\\Datasets\\UCSF-PDGM-v5'\n"
    )

# 3) Standard project folders

INDICES_DIR = os.path.join(PROJECT_ROOT, "indices")
METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create outputs folders if they don't exist (safe to call repeatedly)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "figures"), exist_ok=True)


# 4) Key files used by MONAI stage

# Final index created by the Step 4 script
PATIENT_INDEX_FINAL_CSV = os.path.join(INDICES_DIR, "patient_index_final.csv")

# Clinical metadata (used earlier; still useful for analysis)
CLINICAL_METADATA_CSV = os.path.join(METADATA_DIR, "UCSF-PDGM-metadata_v5.csv")


# 5) Dataset directory (resolved once)

UCSF_PDGM_DATASET_DIR = resolve_ucsf_pdgm_dataset_dir(PROJECT_ROOT)


# 6) Debug print when running this script directly

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("UCSF_PDGM_DATASET_DIR:", UCSF_PDGM_DATASET_DIR)
    print("PATIENT_INDEX_FINAL_CSV:", PATIENT_INDEX_FINAL_CSV)
    print("CLINICAL_METADATA_CSV:", CLINICAL_METADATA_CSV)
    print("OUTPUTS_DIR:", OUTPUTS_DIR)
