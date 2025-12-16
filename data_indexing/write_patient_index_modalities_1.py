# Step 1.2
# This script:
# 1) Calls audit_modalities() to scan the dataset directory
# 2) Builds a patient-level index table (one row per patient)
# 3) Saves it as indices/patient_index_modalities.csv

# This is part of the "add columns" approach:
#   modalities -> geometry -> labels -> final

# - Dataset path can be overridden via environment variable
# - If dataset is inside the repo, it is auto-detected


import os
import sys
import csv
from typing import Optional

# 1) Path setup

# This script is in: thesis/data_indexing/
# PROJECT_ROOT becomes: thesis/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Make sure Python can import modules from the project root
# This lets us import: data_audit.audit_modalities
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_audit.audit_modalities_1 import audit_modalities


# 2) Dataset directory resolver
def resolve_dataset_dir(project_root: str) -> str:
    """
    Find the folder that directly contains patient folders like:
      UCSF-PDGM-0004_nifti, UCSF-PDGM-0005_nifti, ...

    Priority:
      1) Use environment variable UCSF_PDGM_DATASET if set.
      2) Otherwise, search common dataset folder names inside the project root.
      3) If needed, go one nested level deeper to find patient folders.
    """

    # 1) Environment variable override (best for users storing data elsewhere)
    env_path = os.environ.get("UCSF_PDGM_DATASET")
    if env_path:
        if not os.path.isdir(env_path):
            raise FileNotFoundError(
                f"UCSF_PDGM_DATASET is set but folder does not exist:\n{env_path}"
            )
        return env_path

    # Helper: check if a folder contains patient folders
    def contains_patient_folders(folder: str) -> bool:
        try:
            entries = [
                d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))
            ]
        except FileNotFoundError:
            return False

        return any(e.startswith("UCSF-PDGM-") and e.endswith("_nifti") for e in entries)

    # Helper: try folder, and if needed, search one level deeper
    def find_patient_root(start_folder: str) -> Optional[str]:
        if not os.path.isdir(start_folder):
            return None

        # Case A: patient folders are directly inside start_folder
        if contains_patient_folders(start_folder):
            return start_folder

        # Case B: patient folders are one level deeper
        subfolders = [
            d for d in os.listdir(start_folder)
            if os.path.isdir(os.path.join(start_folder, d))
        ]
        for sub in subfolders:
            candidate = os.path.join(start_folder, sub)
            if contains_patient_folders(candidate):
                return candidate

        return None

    # 2) Search common dataset folder names inside the project root
    # Add/keep names you expect users to have after downloading.
    common_names = [
        "PKG - UCSF-PDGM Version 5",  # original TCIA download name (your choice)
        "UCSF-PDGM",                  # cleaned name (also supported)
        "UCSF-PDGM-v5",               # sometimes users rename to this
        "UCSF-PDGM Version 5",
    ]

    for name in common_names:
        start = os.path.join(project_root, name)
        found = find_patient_root(start)
        if found is not None:
            return found

    # 3) If nothing worked, give a clear error
    raise FileNotFoundError(
        "Could not locate UCSF-PDGM dataset folder.\n\n"
        "Fix options:\n"
        "  A) Keep the dataset folder inside the project root with one of these names:\n"
        f"     {common_names}\n"
        "     (patient folders may be directly inside, or one nested level deeper)\n"
        "  B) OR set environment variable UCSF_PDGM_DATASET to the dataset path.\n\n"
        "Example (PowerShell):\n"
        "  $env:UCSF_PDGM_DATASET='D:\\Datasets\\UCSF-PDGM-v5'\n"
    )




# 3) Configuration

DATASET_DIR = resolve_dataset_dir(PROJECT_ROOT)

# All generated CSV indices are stored in: thesis/indices/
INDICES_DIR = os.path.join(PROJECT_ROOT, "indices")
os.makedirs(INDICES_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(INDICES_DIR, "patient_index_modalities.csv")


# 4) Run modality audit

# audit_modalities scans each patient folder and detects which modalities exist
# It returns two values:
#   per_patient: list of dicts, one per patient
#   summary:     overall counts of modality combinations (not used here)

per_patient, _summary = audit_modalities(DATASET_DIR)


# 5) Write output CSV

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "patient_id",
            "has_T1",
            "has_T1c",
            "has_T2",
            "has_FLAIR",
            "has_all4",
            "has_core3",
             # extra useful columns:
            "t1_file",
            "t1c_file",
            "t2_file",
            "flair_file",
        ],
    )

    # Write header row
    writer.writeheader()

    # Write one row per patient
    for p in per_patient:
        has_all4 = p["has_T1"] and p["has_T1c"] and p["has_T2"] and p["has_FLAIR"]
        has_core3 = p["has_T1c"] and p["has_T2"] and p["has_FLAIR"]

        writer.writerow(
            {
                "patient_id": p["patient_id"],
                "has_T1": int(p["has_T1"]),
                "has_T1c": int(p["has_T1c"]),
                "has_T2": int(p["has_T2"]),
                "has_FLAIR": int(p["has_FLAIR"]),
                "has_all4": int(has_all4),
                "has_core3": int(has_core3),
                # filenames (empty string if missing)
                "t1_file": p.get("t1_file", ""),
                "t1c_file": p.get("t1c_file", ""),
                "t2_file": p.get("t2_file", ""),
                "flair_file": p.get("flair_file", ""),
            }
        )

# 6) Print confirmation

print(f"Patient index modalities created: {OUTPUT_CSV}")
print(f"Total patients indexed: {len(per_patient)}")
print("DATASET_DIR resolved to:", DATASET_DIR)
print("Example entries:", os.listdir(DATASET_DIR)[:5])

