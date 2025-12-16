# Step 3.2
# This script adds HGG/LGG labels to the patient index
#
# This script:
# 1) Reads the patient index geometry CSV
# 2) Reads the clinical metadata CSV
# 3) Matches patients by ID
# 4) Adds WHO grade + HGG/LGG label columns
# 5) Writes a new patient index labels CSV


import os
import sys
import csv

# 1) Path setup

# This script is in: thesis/data_indexing/
# PROJECT_ROOT becomes: thesis/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Make sure Python can import modules from the project root
# This lets Python find: data_audit.audit_labels_3
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_audit.audit_labels_3 import build_clinical_lookup, normalize_ucsf_pdgm_id


# 2) Path configuration

INDICES_DIR = os.path.join(PROJECT_ROOT, "indices")
METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")

# Input: geometry index from previous step
INPUT_CSV = os.path.join(INDICES_DIR, "patient_index_geometry.csv")

# Clinical metadata CSV downloaded from dataset website (put it in thesis/metadata/)
CLINICAL_CSV = os.path.join(METADATA_DIR, "UCSF-PDGM-metadata_v5.csv")

# Output: labels-added index
OUTPUT_CSV = os.path.join(INDICES_DIR, "patient_index_labels.csv")


# 3) Clinical column names

CLINICAL_ID_COL = "ID"
CLINICAL_GRADE_COL = "WHO CNS Grade"
CLINICAL_DX_COL = "Final pathologic diagnosis (WHO 2021)"

# 4) Build lookup table from clinical CSV

clinical_lookup = build_clinical_lookup(
    clinical_csv_path=CLINICAL_CSV,
    id_col=CLINICAL_ID_COL,
    grade_col=CLINICAL_GRADE_COL,
    dx_col=CLINICAL_DX_COL
)

print(f"Loaded clinical records: {len(clinical_lookup)}")

# 5) Read the patient index geometry CSV 

rows = []
with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Loaded patient index rows: {len(rows)}")

# 6) Merge labels into patient index rows

out_rows = []
missing_in_clinical = 0
usable_after_labels = 0

for r in rows:
    folder_id = r.get("patient_id", "")
    norm_id = normalize_ucsf_pdgm_id(folder_id)
    clinical = clinical_lookup.get(norm_id)

    r["ucsf_id_used_for_label"] = norm_id


    if clinical is None:
        # If the patient cannot be found in the clinical CSV, it cannot be labeled
        r["who_cns_grade"] = ""
        r["label_hgg"] = ""
        r["label_ok"] = "0"
        r["label_notes"] = "missing_in_clinical_csv"
        r["diagnosis_text"] = ""
        missing_in_clinical += 1
    else:
        # Copy clinical-derived fields into the row
        r["who_cns_grade"] = clinical["who_cns_grade"]
        r["label_hgg"] = clinical["label_hgg"]
        r["label_ok"] = str(clinical["label_ok"])
        r["label_notes"] = clinical["label_notes"]
        r["diagnosis_text"] = clinical["diagnosis_text"]

        # Count label-usable patients (label_ok == 1)
        if str(clinical["label_ok"]) == "1":
            usable_after_labels += 1

    out_rows.append(r)

# 7) Write the patient index labels CSV

fieldnames = list(out_rows[0].keys()) if out_rows else []

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

# 8) Print summary

print(f"\n Patient index labels created: {OUTPUT_CSV}")
print(f"Saved updated index: {OUTPUT_CSV}")
print(f"Patients in index: {len(out_rows)}")
print(f"Missing from clinical CSV: {missing_in_clinical}")
print(f"Patients with valid grade label (label_ok==1): {usable_after_labels}")

# Computes how many patients are usable for the final training set
# Typically, it is required that:
#   geometry_ok == 1 and label_ok == 1
final_usable = sum(
    1 for r in out_rows 
    if r.get("geometry_ok") == "1" and r.get("label_ok") == "1"
    )
print(f"Patients usable for training (geometry_ok == 1 AND label_ok == 1): {final_usable}")
