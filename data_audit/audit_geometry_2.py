# Step 2.1
# This module checks whether, for ONE patient, the MRI modalities (T1, T1c, T2, FLAIR) and the tumor mask are geometrically aligned

# WHY THIS MATTERS:
# - The tumor segmentation mask must overlay the tumor in the MRI.
# - If geometry is inconsistent, the model learns incorrect pixel-to-label mappings and training becomes meaningless.

import os
import nibabel as nib

# Tolerances for comparing floating-point geometry values.
# These values are small because differences come from rounding or file precision, not real misalignment

# Voxel spacing is in millimeters (mm). The tiny differences are allowed 
SPACING_TOL = 1e-3

# Affine matrix values represent orientation/position. Also floats.
AFFINE_TOL = 1e-3


def nearly_equal(a, b, tol):
    """
    Compare two numbers and return True if they are close enough.

    We use this because spacing/affine values are floats and
    floating-point numbers often differ by tiny rounding errors.
    """
    return abs(float(a) - float(b)) <= tol


def tuples_close(t1, t2, tol):
    """
    Compare two tuples element-by-element using nearly_equal().

    Example:
      spacing of T1c might be (1.0, 1.0, 1.0)
      spacing of T2  might be (1.0, 1.0, 0.9999999)

    We consider them equal if differences are within tolerance.
    """
    if t1 is None or t2 is None:
        return False
    if len(t1) != len(t2):
        return False

    # Check each component (x, y, z spacing)
    return all(nearly_equal(x, y, tol) for x, y in zip(t1, t2))


def affine_close(A, B, tol):
    """
    Compare two 4x4 affine matrices element-by-element.

    WHAT IS AN AFFINE?
    - A 4x4 matrix that maps voxel coordinates (i,j,k) to real-world
      coordinates (x,y,z) in millimeters.
    - If the affine differs, the volumes may be shifted/rotated,
      meaning mask and MRI may not align.

    We return True if every element is close within tolerance.
    """
    for r in range(4):
        for c in range(4):
            if not nearly_equal(A[r][c], B[r][c], tol):
                return False
    return True




def find_mask(patient_dir):
    """
    Find the tumor segmentation mask NIfTI file.

    UCSF-PDGM typically names it:
      *_tumor_segmentation.nii.gz

    Returns:
      Full path to mask file, or None if not found.
    """
    files = os.listdir(patient_dir)
    for f in files:
        if f.endswith("_tumor_segmentation.nii.gz"):
            return os.path.join(patient_dir, f)
    return None


def read_geometry(nifti_path):
    """
    Read geometry information from a NIfTI file.

    We read:
      - shape: number of voxels in each dimension (X, Y, Z)
      - zooms: voxel spacing in mm (dx, dy, dz)
      - affine: 4x4 matrix mapping voxels to world coordinates

    IMPORTANT:
    We do NOT load the full image array into memory here.
    nibabel loads header metadata efficiently; this keeps it fast.
    """
    img = nib.load(nifti_path)

 
    shape = img.shape[:3]

    # zooms/spacings: e.g., (1.0, 1.0, 1.0)
    # We take only first 3 values (x,y,z). Some NIfTI include time dim.
    zooms = img.header.get_zooms()[:3]

    # affine: 4x4 matrix
    affine = img.affine

    return shape, zooms, affine


def audit_patient_geometry(
    patient_dir,
    t1_file,
    t1c_file,
    t2_file,
    flair_file,
    mask_file,
):
    """
    Perform geometry checks for one patient folder.

    Returns a dictionary with:
      - file names chosen for each modality
      - whether mask exists
      - shape_ok / spacing_ok / affine_ok flags
      - geometry_ok (1 only if all checks pass)
      - notes explaining what went wrong if something fails

    This dictionary is designed to be added as new columns to your
    patient index CSV in the next step.
    """

    # Default result structure.
    # We initialize everything so even failure cases produce a row.
    result = {
        "has_mask": 0,
        "shape_ok": 0,
        "spacing_ok": 0,
        "affine_ok": 0,
        "geometry_ok": 0,
        "notes": "",
        # Store file names (helpful for debugging)
        "t1_file": t1_file or "",
        "t1c_file": t1c_file or "",
        "t2_file": t2_file or "",
        "flair_file": flair_file or "",
        "mask_file": mask_file or "",
    }

   # Build full paths from patient_dir + filenames
    t1_path = os.path.join(patient_dir, t1_file) if t1_file else None
    t1c_path = os.path.join(patient_dir, t1c_file) if t1c_file else None
    t2_path = os.path.join(patient_dir, t2_file) if t2_file else None
    flair_path = os.path.join(patient_dir, flair_file) if flair_file else None
    mask_path = os.path.join(patient_dir, mask_file) if mask_file else None

    # Check required files exist (strict: all4 + mask)
    missing = []
    if not t1_path or not os.path.isfile(t1_path):
        missing.append("T1")
    if not t1c_path or not os.path.isfile(t1c_path):
        missing.append("T1c")
    if not t2_path or not os.path.isfile(t2_path):
        missing.append("T2")
    if not flair_path or not os.path.isfile(flair_path):
        missing.append("FLAIR")
    if not mask_path or not os.path.isfile(mask_path):
        missing.append("MASK")

    # If anything is missing, we cannot do geometry checks.
    # Return early with "missing:..." note.
    if missing:
        result["notes"] = "missing:" + ",".join(missing)
        return result

    # Mask exists if we reached here
    result["has_mask"] = 1
    
    # Geometry checks
    # We use T1c as a reference modality (common in glioma work).
    # Everything else should match T1c geometry if data is aligned.
  
    ref_shape, ref_zooms, ref_affine = read_geometry(t1c_path)

    # Read geometry for all modalities and the mask
    shapes = {}
    zooms = {}
    affines = {}

    for name, path in [
        ("T1", t1_path),
        ("T1c", t1c_path),
        ("T2", t2_path),
        ("FLAIR", flair_path),
        ("MASK", mask_path),
    ]:
        s, z, a = read_geometry(path)
        shapes[name] = s
        zooms[name] = z
        affines[name] = a

    # 1) Shape check:
    # All files should have exactly the same voxel grid size.
    # If shapes differ, masks won't overlay correctly.
    result["shape_ok"] = int(all(shapes[k] == ref_shape for k in shapes))

    # 2) Spacing check:
    # All files should have same voxel spacing in mm.
    # Small differences may occur due to rounding -> tolerance.
    result["spacing_ok"] = int(all(tuples_close(zooms[k], ref_zooms, SPACING_TOL) for k in zooms))

    # 3) Affine check:
    # Affine encodes orientation and position in space.
    # It must match for correct alignment.
    result["affine_ok"] = int(all(affine_close(affines[k], ref_affine, AFFINE_TOL) for k in affines))

    # geometry_ok is True only if all checks pass
    result["geometry_ok"] = int(
        result["shape_ok"] and result["spacing_ok"] and result["affine_ok"]
    )

    # If geometry is not ok, write a helpful "mismatch" note
    if not result["geometry_ok"]:
        problems = []
        if not result["shape_ok"]:
            problems.append("shape")
        if not result["spacing_ok"]:
            problems.append("spacing")
        if not result["affine_ok"]:
            problems.append("affine")
        result["notes"] = "mismatch:" + ",".join(problems)

    return result
