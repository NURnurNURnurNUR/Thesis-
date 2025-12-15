# This script updates the patient index with geometry checks
#
# This script:
# 1) Reads the patient index modalities CSV 
# 2) Filters to patients with complete four modalities input
# 3) Runs geometry audit for each patient folder
# 4) Writes a new patient index geometry CSV with added columns
#
# IMPORTANT:
# This script does not delete any data from disk.
# This script only marks which patients are usable for training.


import csv
import os
from audit_geometry import audit_patient_geometry

# 1) Configuration

# Directory that contains all patient folders (same as before)
DATASET_DIR = r"C:\Users\Nurma\Desktop\thesis\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5"

# Input CSV produced by the modality audit
INPUT_CSV = "patient_index_modalities.csv"

# Output CSV for this geometry audit
OUTPUT_CSV = "patient_index_geometry.csv"

# 2) Read the patient index modalities CSV

rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Loaded {len(rows)} patients from {INPUT_CSV}")

# 2) Filter to patients with complete modalities
# Because the deep learning model expects consistent four modalities input
# (T1, T1c, T2, FLAIR), we only audit geometry for has_all4 == 1.

rows_all4 = [r for r in rows if r["has_all4"] == "1"]
print(f"Geometry audit will run on {len(rows_all4)} patients with has_all4 == 1")

# 3) Run geometry audit for each patient

out_rows = []
problem_count = 0

for r in rows_all4:
    patient_id = r["patient_id"]

    # Build the filesystem path to this patient's folder
    patient_dir = os.path.join(DATASET_DIR, patient_id)

    # Run audit for this patient folder.
    # Returns a dictionary of new columns (has_mask, shape_ok, etc.)
    geom = audit_patient_geometry(patient_dir)

    # Combine:
    # - existing modalities columns
    # - new geometry audit columns
    out_row = dict(r)
    out_row.update(geom)

    # Count how many patients are problematic (geometry_ok == 0 or missing files)
    if out_row.get("geometry_ok", 0) in ["0", 0]:
        problem_count += 1

    out_rows.append(out_row)


# 4) Write geometry audit CSV

# fieldnames define the column order.
# Take keys from the first row (all rows have same keys)
fieldnames = list(out_rows[0].keys()) if out_rows else []

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)


# 5) Print final summary

print(f"\n Geometry audit complete.")
print(f"Saved updated index: {OUTPUT_CSV}")
print(f"Patients processed: {len(out_rows)}")
print(f"Patients flagged (geometry_ok == 0 or missing files): {problem_count}")
