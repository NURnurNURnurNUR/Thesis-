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
import sys

# 1) Path setup

# This script is in: thesis/data_indexing/
# PROJECT_ROOT becomes: thesis/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Make sure Python can import modules from the project root
# This lets Python find: data_audit.audit_geometry
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_audit.audit_geometry_2 import audit_patient_geometry

# 2) Dataset directory resolver

def resolve_dataset_dir(project_root: str) -> str:
    env_path = os.environ.get("UCSF_PDGM_DATASET")
    if env_path:
        if not os.path.isdir(env_path):
            raise FileNotFoundError(f"UCSF_PDGM_DATASET does not exist:\n{env_path}")
        return env_path

    common_names = [
        "PKG - UCSF-PDGM Version 5",
        "UCSF-PDGM",
        "UCSF-PDGM-v5",
        "UCSF-PDGM Version 5",
    ]

    for name in common_names:
        base = os.path.join(project_root, name)
        if not os.path.isdir(base):
            continue

        # case A: patient folders directly inside
        entries = os.listdir(base)
        if any(e.startswith("UCSF-PDGM-") and e.endswith("_nifti") for e in entries):
            return base

        # case B: one level deeper
        for sub in entries:
            candidate = os.path.join(base, sub)
            if os.path.isdir(candidate):
                sub_entries = os.listdir(candidate)
                if any(e.startswith("UCSF-PDGM-") and e.endswith("_nifti") for e in sub_entries):
                    return candidate

    raise FileNotFoundError(
        "Could not locate UCSF-PDGM dataset.\n"
        "Either place it inside the project root or set UCSF_PDGM_DATASET."
    )


def find_mask_filename(patient_dir: str) -> str:
    """Find mask filename in patient folder (returns '' if missing)."""
    for f in os.listdir(patient_dir):
        if f.endswith("_tumor_segmentation.nii.gz"):
            return f
    return ""


# 3) Configuration

DATASET_DIR = resolve_dataset_dir(PROJECT_ROOT)

INDICES_DIR = os.path.join(PROJECT_ROOT, "indices")
INPUT_CSV = os.path.join(INDICES_DIR, "patient_index_modalities.csv")
OUTPUT_CSV = os.path.join(INDICES_DIR, "patient_index_geometry.csv")

# 4) Read the patient index modalities CSV

rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Loaded {len(rows)} patients from {INPUT_CSV}")

# 5) Filter to patients with complete modalities
# Because the deep learning model expects consistent four modalities input
# (T1, T1c, T2, FLAIR), we only audit geometry for has_all4 == 1.

rows_all4 = [r for r in rows if r["has_all4"] == "1"]
print(f"Geometry audit will run on {len(rows_all4)} patients with has_all4 == 1")

# 6) Run geometry audit for each patient

out_rows = []
problem_count = 0

for r in rows_all4:
    patient_id = r["patient_id"]
    # Build the filesystem path to this patient's folder
    patient_dir = os.path.join(DATASET_DIR, patient_id)

    # Pull modality filenames from CSV 
    t1_file = r.get("t1_file", "")
    t1c_file = r.get("t1c_file", "")
    t2_file = r.get("t2_file", "")
    flair_file = r.get("flair_file", "")

    # Mask filename: either already present, or found by quick scan
    mask_file = r.get("mask_file", "")
    if not mask_file:
        mask_file = find_mask_filename(patient_dir)

    geom = audit_patient_geometry(
        patient_dir=patient_dir,
        t1_file=t1_file,
        t1c_file=t1c_file,
        t2_file=t2_file,
        flair_file=flair_file,
        mask_file=mask_file,
)

    # Combine:
    # - existing modalities columns
    # - new geometry audit columns
    out_row = dict(r)
    out_row.update(geom)

    # Count how many patients are problematic (geometry_ok == 0 or missing files)
    if out_row.get("geometry_ok", 0) in ["0", 0]:
        problem_count += 1

    out_rows.append(out_row)


# 7) Write geometry audit CSV

# fieldnames define the column order.
# Take keys from the first row (all rows have same keys)
fieldnames = list(out_rows[0].keys()) if out_rows else []

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)


# 8) Print final summary

print(f"\n Geometry audit complete.")
print(f"Saved updated index: {OUTPUT_CSV}")
print(f"Patients processed: {len(out_rows)}")
print(f"Patients flagged (geometry_ok == 0 or missing files): {problem_count}")
print(f"DATASET_DIR used: {DATASET_DIR}")