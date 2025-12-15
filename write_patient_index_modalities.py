# This script:
# 1) Calls audit_modalities() to scan the dataset
# 2) Builds a patient-level index table
# 3) Saves it as a CSV file for later pipeline steps
#
# This is a step of the "add columns" approach
import csv
from audit_modalities import audit_modalities

# 1) Configuration

# Path to the dataset directory that contains all patient folders
DATASET_DIR = r"C:\Users\Nurma\Desktop\thesis\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5"

# Output CSV file name
OUTPUT_CSV = "patient_index_modalities.csv"

# 2) Run modality audit

# Call the audit function.
# This returns:
#   per_patient : list of dictionaries (one per patient)
#   summary     : modality combination counts (not used here)
per_patient, _summary = audit_modalities(DATASET_DIR)


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    # Define CSV column names explicitly
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
        ]
    )

    # Write header row
    writer.writeheader()

    # Write one row per patient
    for p in per_patient:
        # Patient has all four modalities
        has_all4 = (
            p["has_T1"] and
            p["has_T1c"] and
            p["has_T2"] and
            p["has_FLAIR"]
        )

        # Patient has the three core modalities
        has_core3 = (
            p["has_T1c"] and
            p["has_T2"] and
            p["has_FLAIR"]
        )

        # Write row to CSV
        writer.writerow({
            "patient_id": p["patient_id"],
            "has_T1": int(p["has_T1"]),
            "has_T1c": int(p["has_T1c"]),
            "has_T2": int(p["has_T2"]),
            "has_FLAIR": int(p["has_FLAIR"]),
            "has_all4": int(has_all4),
            "has_core3": int(has_core3),
        })


# 3) Print final confirmation message
print(f" Patient index modalities created: {OUTPUT_CSV}")
print(f" Total patients indexed: {len(per_patient)}")
