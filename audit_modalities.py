# Step 1 
# This module scans patient folders and determines which MRI  modalities (T1, T1c, T2, FLAIR) are available for each patient
# It is designed to be IMPORTED by other scripts, not to automatically print or save anything when imported


import os
from collections import Counter


def audit_modalities(dataset_dir):
    """
    Scan all patient directories and record MRI modality availability.

    Parameters
    ----------
    dataset_dir : str
        Path to the directory that contains all patient folders.

    Returns
    -------
    per_patient : list of dict
        Each dict represents one patient and stores modality flags.
    summary : Counter
        Counts how many patients fall into each modality combination.
    """

    # Counter to summarize how many patients have each combination of modalities (T1, T1c, T2, FLAIR)
    summary = Counter()

    # List that will store one dictionary per patient
    per_patient = []

    # Iterate through all patient folders in the dataset directory
    for patient in sorted(os.listdir(dataset_dir)):
        patient_dir = os.path.join(dataset_dir, patient)

        # Skip non-directory entries
        if not os.path.isdir(patient_dir):
            continue

        # List all files inside the patient folder
        files = os.listdir(patient_dir)

        # Check for each modality if it exists
        has_T1 = any(
            f.endswith("_T1.nii.gz") or f.endswith("_T1_bias.nii.gz")
            for f in files
        )

        has_T1c = any(
            f.endswith("_T1c.nii.gz") or f.endswith("_T1c_bias.nii.gz")
            for f in files
        )

        has_T2 = any(
            f.endswith("_T2.nii.gz") or f.endswith("_T2_bias.nii.gz")
            for f in files
        )

        has_FLAIR = any(
            f.endswith("_FLAIR.nii.gz") or f.endswith("_FLAIR_bias.nii.gz")
            for f in files
        )

        # Store per-patient modality availability
        per_patient.append({
            "patient_id": patient,
            "has_T1": has_T1,
            "has_T1c": has_T1c,
            "has_T2": has_T2,
            "has_FLAIR": has_FLAIR,
        })

        # Update the summary counter
        summary[(has_T1, has_T1c, has_T2, has_FLAIR)] += 1

    return per_patient, summary


def print_summary(per_patient, summary):
    """
    Print a human-readable summary of modality availability.

    This function is optional and should be called explicitly.
    """

    print("=== Modality availability summary ===")
    print("Format: T1, T1c, T2, FLAIR -> number of patients\n")

    for (t1, t1c, t2, flair), count in summary.items():
        print(f"T1={t1}, T1c={t1c}, T2={t2}, FLAIR={flair} : {count} patients")

    # Count how many patients have all four modalities
    all_4 = sum(
        1 for p in per_patient
        if p["has_T1"] and p["has_T1c"] and p["has_T2"] and p["has_FLAIR"]
    )

    # Count how many patients have the three (core) modalities
    all_3 = sum(
        1 for p in per_patient
        if p["has_T1c"] and p["has_T2"] and p["has_FLAIR"]
    )

    print("\n=== Decision-support counts ===")
    print("Patients with all 4 modalities (T1+T1c+T2+FLAIR):", all_4)
    print("Patients with 3 modalities (T1c+T2+FLAIR):", all_3)
    print("Total patient folders scanned:", len(per_patient))


# This block runs ONLY if this file is executed directly
# It will NOT run when the module is imported elsewhere

if __name__ == "__main__":
    DATASET_DIR = r"C:\Users\Nurma\Desktop\thesis\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5"
    patients, summary = audit_modalities(DATASET_DIR)
    print_summary(patients, summary)
