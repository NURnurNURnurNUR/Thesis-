# Step 1.1 (AUDIT): Modality availability audit for UCSF-PDGM
# Purpose:
#   Scan patient folders and determine which MRI modalities exist:
#     - T1
#     - T1c
#     - T2
#     - FLAIR
# Why this file is a "module":
#   - It is meant to be imported by other scripts
#   - It should not run heavy scanning automatically when imported
# Outputs:
#   - per_patient: list of dicts (one per patient)
#   - summary: Counter with counts for each modality combination
# Improvements vs your version:
#   - Robust dataset path resolver (supports default TCIA folder name)
#   - Records which exact filenames were found (helpful for debugging)


import os
from collections import Counter
from typing import List, Dict, Tuple


# Helper: Resolve dataset directory in a robust, reproducible way

def resolve_ucsf_pdgm_dataset_dir(project_root: str) -> str:
    """
    Try to locate the UCSF-PDGM dataset directory.

    Priority:
      1) Environment variable UCSF_PDGM_DATASET (user override)
      2) Common folder names under project_root
      3) Allow one nested folder level (e.g. .../PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5)

    Returns:
      A path that directly contains patient folders:
        UCSF-PDGM-0004_nifti, UCSF-PDGM-0005_nifti, ...
    """
    # 1) Explicit override
    env_path = os.environ.get("UCSF_PDGM_DATASET")
    if env_path:
        if not os.path.isdir(env_path):
            raise FileNotFoundError(f"UCSF_PDGM_DATASET is set but does not exist:\n{env_path}")
        return env_path

    # Helper: detect patient folders inside a folder
    def contains_patient_folders(folder: str) -> bool:
        try:
            entries = [
                d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))
            ]
        except FileNotFoundError:
            return False
        return any(e.startswith("UCSF-PDGM-") and e.endswith("_nifti") for e in entries)

    # Helper: look in folder, then one level deeper
    def find_patient_root(start_folder: str) -> str | None:
        if not os.path.isdir(start_folder):
            return None

        if contains_patient_folders(start_folder):
            return start_folder

        subfolders = [
            d for d in os.listdir(start_folder)
            if os.path.isdir(os.path.join(start_folder, d))
        ]
        for sub in subfolders:
            candidate = os.path.join(start_folder, sub)
            if contains_patient_folders(candidate):
                return candidate

        return None

    # 2) Common folder names people may keep after downloading
    common_names = [
        "PKG - UCSF-PDGM Version 5", 
        "UCSF-PDGM",
        "UCSF-PDGM-v5",
        "UCSF-PDGM Version 5",
    ]

    for name in common_names:
        start = os.path.join(project_root, name)
        found = find_patient_root(start)
        if found:
            return found

    raise FileNotFoundError(
        "Could not locate UCSF-PDGM dataset.\n\n"
        "Fix options:\n"
        "  A) Put the dataset folder inside the project root using a common name like:\n"
        f"     {common_names}\n"
        "  B) OR set environment variable UCSF_PDGM_DATASET to your dataset path.\n\n"
        "Example (PowerShell):\n"
        "  $env:UCSF_PDGM_DATASET='D:\\Datasets\\UCSF-PDGM-v5'\n"
    )


# Main audit function
def audit_modalities(dataset_dir: str) -> Tuple[List[Dict], Counter]:
    """
    Scan all patient directories and record MRI modality availability.

    Parameters
    ----------
    dataset_dir : str
        Path to the directory that contains patient folders.

    Returns
    -------
    per_patient : list of dict
        One dict per patient:
          - patient_id
          - has_T1, has_T1c, has_T2, has_FLAIR (bool)
          - t1_file, t1c_file, t2_file, flair_file (filename or "")
    summary : Counter
        Counts how many patients fall into each modality combination tuple:
          (has_T1, has_T1c, has_T2, has_FLAIR)
    """

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist:\n{dataset_dir}")

    summary = Counter()
    per_patient = []

    # Iterate through patient folders
    for patient in sorted(os.listdir(dataset_dir)):
        patient_dir = os.path.join(dataset_dir, patient)

        # Skip non-folders
        if not os.path.isdir(patient_dir):
            continue

        files = os.listdir(patient_dir)

        # The exact filename found for each modality
        # This is helpful later and avoids guessing
        t1_file = next((f for f in files if f.endswith("_T1.nii.gz") or f.endswith("_T1_bias.nii.gz")), "")
        t1c_file = next((f for f in files if f.endswith("_T1c.nii.gz") or f.endswith("_T1c_bias.nii.gz")), "")
        t2_file = next((f for f in files if f.endswith("_T2.nii.gz") or f.endswith("_T2_bias.nii.gz")), "")
        flair_file = next((f for f in files if f.endswith("_FLAIR.nii.gz") or f.endswith("_FLAIR_bias.nii.gz")), "")

        has_T1 = t1_file != ""
        has_T1c = t1c_file != ""
        has_T2 = t2_file != ""
        has_FLAIR = flair_file != ""

        per_patient.append(
            {
                "patient_id": patient,
                "has_T1": has_T1,
                "has_T1c": has_T1c,
                "has_T2": has_T2,
                "has_FLAIR": has_FLAIR,
                # filenames (empty string means missing)
                "t1_file": t1_file,
                "t1c_file": t1c_file,
                "t2_file": t2_file,
                "flair_file": flair_file,
            }
        )

        summary[(has_T1, has_T1c, has_T2, has_FLAIR)] += 1

    return per_patient, summary


# Optional pretty printer
def print_summary(per_patient: List[Dict], summary: Counter) -> None:
    """
    Print a human-readable summary of modality availability.
    """
    print("=== Modality availability summary ===")
    print("Format: T1, T1c, T2, FLAIR -> number of patients\n")

    for (t1, t1c, t2, flair), count in summary.items():
        print(f"T1={t1}, T1c={t1c}, T2={t2}, FLAIR={flair} : {count} patients")

    all_4 = sum(
        1 for p in per_patient
        if p["has_T1"] and p["has_T1c"] and p["has_T2"] and p["has_FLAIR"]
    )

    all_3 = sum(
        1 for p in per_patient
        if p["has_T1c"] and p["has_T2"] and p["has_FLAIR"]
    )

    print("\n=== Decision-support counts ===")
    print("Patients with all 4 modalities (T1+T1c+T2+FLAIR):", all_4)
    print("Patients with 3 modalities (T1c+T2+FLAIR):", all_3)
    print("Total patient folders scanned:", len(per_patient))


# Run only if executed directly (not when imported)
if __name__ == "__main__":
    # Project root = one level above this file's folder (thesis/)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Resolve dataset dir automatically (or via env var)
    DATASET_DIR = resolve_ucsf_pdgm_dataset_dir(PROJECT_ROOT)

    patients, summary = audit_modalities(DATASET_DIR)
    print("DATASET_DIR resolved to:", DATASET_DIR)
    print("Example entries:", os.listdir(DATASET_DIR)[:5])
    print_summary(patients, summary)
