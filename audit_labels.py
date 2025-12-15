# Step 3

# Purpose:
#   Read UCSF-PDGM clinical metadata (CSV) and build a lookup table to assign HGG/LGG labels to patients.
#
# Why this is needed:
#   - The segmentation masks exist in NIfTI folders
#   - The glioma grade labels exist in the clinical CSV
#   - We must merge them safely and reproducibly

import csv


def normalize_ucsf_pdgm_id(pid: str) -> str:
    """
    Convert different UCSF-PDGM ID formats into a standard form.

    Examples:
      'UCSF-PDGM-004'        -> 'UCSF-PDGM-0004'
      'UCSF-PDGM-14'         -> 'UCSF-PDGM-0014'
      'UCSF-PDGM-0004'       -> 'UCSF-PDGM-0004'
      'UCSF-PDGM-0004_nifti' -> 'UCSF-PDGM-0004'
    """
    if pid is None:
        return ""

    s = str(pid).strip()

    # Remove folder suffix if present
    if s.endswith("_nifti"):
        s = s[:-6]

    # Expect something like UCSF-PDGM-<number>
    if not s.startswith("UCSF-PDGM-"):
        return s  # return as-is if unexpected format

    prefix = "UCSF-PDGM-"
    num_part = s[len(prefix):].strip()

    # If there are extra characters, keep only digits
    digits = "".join(ch for ch in num_part if ch.isdigit())
    if digits == "":
        return ""

    # Pad to 4 digits
    digits_4 = digits.zfill(4)

    return prefix + digits_4


def safe_int(value):
    """
    Convert a value to int if possible; otherwise return None.

    Clinical CSVs sometimes store numbers as:
      - '4'
      - 4
      - '4.0'
      - blank / 'NA' / 'unknown'
    This function makes parsing robust.
    """
    if value is None:
        return None

    s = str(value).strip()
    if s == "" or s.lower() in {"na", "nan", "none", "unknown"}:
        return None

    # Handle values like "4.0"
    try:
        return int(float(s))
    except ValueError:
        return None


def grade_to_hgg_label(who_grade):
    """
    Map WHO CNS Grade to HGG/LGG label.

    Standard mapping:
      Grade 2 -> LGG (0)
      Grade 3 -> HGG (1)
      Grade 4 -> HGG (1)

    Returns:
      (label, label_ok, note)
    """
    if who_grade is None:
        return None, 0, "missing_grade"

    if who_grade == 2:
        return 0, 1, ""  # LGG
    if who_grade in (3, 4):
        return 1, 1, ""  # HGG

    # If grade is something unexpected, we mark it unusable.
    return None, 0, f"unexpected_grade:{who_grade}"


def build_clinical_lookup(clinical_csv_path = "UCSF-PDGM-metadata_v5.csv", id_col = "ID", grade_col = "WHO CNS Grade", dx_col = "Final pathologic diagnosis (WHO 2021)"):
    """
    Read the clinical metadata CSV and build a dictionary:
      lookup[patient_id] -> {who_grade, label_hgg, diagnosis_text, ...}

    Parameters:
      clinical_csv_path : str
        Path to UCSF-PDGM clinical CSV.
      id_col : str
        Name of the patient ID column in the clinical CSV.
      grade_col : str
        Name of WHO grade column.
      dx_col : str
        Name of diagnosis text column.

    Returns:
      lookup : dict
        Keys are normalized patient IDs (e.g., UCSF-PDGM-0004)
        Values are dictionaries with grade/label/diagnosis info.
    """
    lookup = {}

    with open(clinical_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        # Loop over each row in the clinical CSV
        for row in reader:
            raw_id = row.get(id_col, "")
            pid = normalize_ucsf_pdgm_id(raw_id)


            if pid == "":
                continue  # skip rows with no ID

            # Parse WHO grade
            who_grade = safe_int(row.get(grade_col))

            # Convert WHO grade to HGG/LGG labels
            label_hgg, label_ok, note = grade_to_hgg_label(who_grade)

            # Diagnosis text can be useful for reporting or filtering later
            diagnosis_text = (row.get(dx_col) or "").strip()

            # Store in the lookup table
            lookup[pid] = {
                "who_cns_grade": who_grade if who_grade is not None else "",
                "label_hgg": label_hgg if label_hgg is not None else "",
                "label_ok": label_ok,
                "label_notes": note,
                "diagnosis_text": diagnosis_text,
            }

    return lookup
