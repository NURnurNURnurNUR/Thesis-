# Step 4 (final)
# This script creates a final "training-ready" patient index CSV
#
# Goal:
#   The patients have the following indices:
#     - modality availability (Step 1)
#     - geometry checks + mask existence (Step 2)
#     - grade labels (HGG/LGG) (Step 3)
#
# Now we create a final CSV that contains ONLY patients that are
# safe to use for training 
#
# It will be filltered following these rules:
#   1) has_all4 == 1      -> ensures consistent 4-channel input
#   2) geometry_ok == 1   -> ensures modalities + mask align
#   3) label_ok == 1      -> ensures grade label is usable



import os
import csv


# 1) Configuration

# This script is in: thesis/data_indexing/
# PROJECT_ROOT becomes: thesis/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Indices folder for all CSV outputs
INDICES_DIR = os.path.join(PROJECT_ROOT, "indices")

# Input CSV from step 3 (labels added)
INPUT_CSV = os.path.join(INDICES_DIR, "patient_index_labels.csv")

# Output CSV for step 4 (final filtered)
OUTPUT_CSV = os.path.join(INDICES_DIR, "patient_index_final.csv")


# 2) Read the input CSV

rows = []
with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    # Save the column order (fieldnames) for writing later
    input_columns = reader.fieldnames

    # Read each row (each row represents a patient)
    for r in reader:
        rows.append(r)

print(f"Loaded rows from {INPUT_CSV}: {len(rows)} patients")


# 3) Filter rows using the inclusion criteria

# Each value in CSV is read as a string. For example:
#   "1" or "0"
#
# So we compare using string values "1" / "0"

final_rows = []

# The counts are collected for transparency
count_all4 = 0
count_geometry_ok = 0
count_label_ok = 0

for r in rows:
    # Rule 1: Has all 4 modalities (T1, T1c, T2, FLAIR)
    # This ensures the model always receives 4 input channels
    if r.get("has_all4", "0") != "1":
        continue
    count_all4 += 1

    # Rule 2: Geometry OK (modalities + tumor mask aligned)
    # This ensures the tumor mask overlays correctly on the MRIs
    if r.get("geometry_ok", "0") != "1":
        continue
    count_geometry_ok += 1

    # Rule 3: Label OK (HGG/LGG label usable)
    # This ensures the WHO grade was present and mapped correctly
    if r.get("label_ok", "0") != "1":
        continue
    count_label_ok += 1

    # If the patient passes all rules, keep the row in the final CSV
    final_rows.append(r)

# 4) Print summary
print("\n Final training-ready patients:")
print(f" - Patients with has_all4 == 1: {count_all4}")
print(f" - Patients with geometry_ok == 1 (after has_all4 filter): {count_geometry_ok}")
print(f" - Patients with label_ok == 1 (after previous filters): {count_label_ok}")
print(f" - Final training-ready patients: {len(final_rows)}")



# 5) Write the final CSV

# Write the same columns as the input CSV, but only for the filtered subset

# Keeping all columns is useful because it keeps file names
# (t1_file, mask_file) and labels (label_hgg) in one place

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=input_columns)
    writer.writeheader()
    writer.writerows(final_rows)

print(f"\nSaved final training-ready patient index CSV: {OUTPUT_CSV}")