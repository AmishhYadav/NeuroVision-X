"""Resamples a predicted mask from atlas space back into one modality's native geometry.

Milestone 4, Phase E, task E6 follow-up: `neurovision.reporting.dicom_seg`'s own module
docstring names this exact gap. After E2 (`neurovision.data.clinical_preprocess`) a
predicted mask lives in ATLAS space (SRI24, 240x240x155, 1 mm isotropic), not in the
pixel geometry of the source DICOM series the study came from. `write_dicom_seg` correctly
REFUSES to write a SEG object when the mask's geometry does not match the source series --
this module builds the missing piece that lets a caller close that gap: resampling the
mask back through E2's saved inverse registration transform into a chosen modality's
*native* (pre-E2) geometry, which is what `write_dicom_seg`'s geometry check actually
wants to see.

**This module does NOT wire itself into `dicom_seg.py` or the live job pipeline.** That is
separate, later work -- `dicom_seg.py` is untouched. It also does not un-crop a mask from
the research pipeline's cropped frame (`neurovision.data.preprocessing.preprocess_case`'s
nonzero-bbox crop) back to full atlas-space shape -- `resample_mask_to_source` takes a
precondition, not a responsibility, that its `mask` argument already has the FULL,
uncropped atlas-space shape (matching `atlas_affine`). A caller whose mask came from
`segment_case` (which operates on the research pipeline's CROPPED frame) must undo that
crop using the cropped case's own saved `bbox`/`original_shape` metadata before calling
this function; doing that un-cropping is out of scope here.

**Why atlas space, not the research pipeline's cropped frame.** E2's
`AtlasCentricPreprocessor` writes every modality onto the SRI24 atlas grid *before* the
research path's own crop-to-nonzero-bbox ever runs (`preprocess_case` is what performs
that crop, on E2's output). So the frame this function resamples FROM is the uncropped
atlas grid E2 produces -- exactly the frame `atlas_affine` (read off one of E2's own
atlas-registered output NIfTIs) describes.

**`brainles_preprocessing.transform.Transform` does the actual resampling.** Its own
docstring: "Common use case is to apply inverse transformations to transform e.g.
segmentations in atlas space back to native space." `Transform(transformations_dir)` reads
back the fitted ANTs registration transforms `clinical_preprocess.run_plan` already saves
for every modality role (`PreprocessResult.transformations_dir`, one subdirectory per role,
named after that role's own `modality_name` -- confirmed against
`brainles_preprocessing.preprocessor.atlas_centric_preprocessor.AtlasCentricPreprocessor
.run`, which iterates `self.all_modalities` and writes into
`save_dir_transformations / modality.modality_name`, and `clinical_preprocess.run_plan`
constructs every `CenterModality`/`Modality` with `modality_name` set to exactly this
project's own role strings, `plan.center_role` and each of `plan.moving_roles`). `.apply(
..., inverse=True)` runs the inverse transform, i.e. atlas space -> that modality's native
space, using `target_modality_img` (the modality's ORIGINAL, pre-E2 NIfTI) purely to define
the output voxel grid.

**Why `nibabel` and `Transform` are imported inside the function body, never at module
scope.** Following the exact convention `dicom_seg.py`'s and `clinical_preprocess.py`'s own
module docstrings document: `brainles_preprocessing` lives only in `.venv-clinical`
(`requirements-clinical.txt`), never in this project's main `.venv`. A module that imports
it at module scope would break every caller running under the main `.venv` just by being
imported. `nibabel` is listed as a main-`.venv` dependency too, but is kept lazy here
anyway so this module's only "requires .venv-clinical" surface is the one function that
actually needs it, matching the shape of every other clinical module in this codebase.

**Why the output is validated, not trusted.** `interpolator="genericLabel"` is ANTs' label-
preserving interpolation scheme (it is designed never to blend discrete label values), but
CLAUDE.md's own traps record more than one dependency call here that silently produced a
plausible-looking, wrong result. So this function checks, before returning: (a) the
resampled mask's shape exactly matches the target native image's shape (a mismatch means
the resample did something other than land on that image's grid), and (b) every value in
the result rounds to one of `{0, 1, 2, 3}` within a small floating-point tolerance (a
`genericLabel` result that does NOT round cleanly to a known class value means it was not
actually label-preserving, and this function refuses to guess which case that is).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from neurovision.utils.io import ensure_dir

__all__ = ["resample_mask_to_source"]

logger = logging.getLogger(__name__)

# The only values a class map produced by
# neurovision.reporting.dicom_seg.classes_from_regions (or the underlying
# neurovision.inference.postprocess.regions_to_classes) can ever contain:
# 0 = background, 1 = necrotic/non-enhancing core, 2 = edema, 3 = enhancing
# tumour. Anything else surviving the resample is evidence of a real bug,
# not a legitimate label.
_VALID_CLASS_VALUES = frozenset({0, 1, 2, 3})

# genericLabel interpolation is designed to preserve discrete label values
# exactly, so a value that lands even slightly off an integer after loading
# a saved NIfTI back is ordinary floating-point round-trip noise (NIfTI
# intensities decode through nibabel as float64), not a real fractional
# value. This tolerance is generous relative to float32 storage precision at
# label magnitudes of 0-3, while still catching genuine blending (which
# would show up as e.g. 1.5, far outside this tolerance).
_ROUND_TOLERANCE = 1e-3


def resample_mask_to_source(
    mask: np.ndarray,
    atlas_affine: np.ndarray,
    transformations_dir: Path,
    target_role: str,
    target_native_path: Path,
    out_dir: Path,
) -> Path:
    """Resamples an atlas-space class map back into one modality's native (pre-E2) geometry.

    Precondition: `mask` must already be in the FULL, uncropped atlas-space voxel grid --
    the shape and frame E2 (`clinical_preprocess.run_plan`) produces before the research
    pipeline's `preprocess_case` ever crops it, and the frame `atlas_affine` describes. A
    mask from the research/cropped frame (e.g. straight off `segment_case`) must be
    un-cropped back to that full atlas-space shape by the caller first -- this function does
    not detect or handle a cropped input. Passing one anyway produces a silently wrong
    resample: ANTs' resampling always writes its output onto the *target* image's own grid,
    regardless of the moving (here, cropped) image's shape, so the shape check below offers
    close to no protection against this specific precondition violation -- a cropped input
    would typically still come back with a shape that matches. This precondition is enforced
    by the caller only; nothing in this function checks for it.

    Args:
        mask: `(D, H, W)` array, values in `{0, 1, 2, 3}`, on the SRI24 atlas-space voxel
            grid (240x240x155, 1 mm isotropic).
        atlas_affine: The 4x4 NIfTI affine of the atlas-space grid `mask` lives on. Read off
            one of E2's own atlas-registered output NIfTIs by the caller (e.g.
            `nibabel.load(preprocess_result.outputs[some_role]).affine`) -- this function
            takes it as a plain argument and never computes or hardcodes an atlas affine
            itself.
        transformations_dir: `PreprocessResult.transformations_dir` from the same E2 run
            that produced `atlas_affine`'s modality.
        target_role: Which of this project's modality role strings (see
            `neurovision.data.dicom_ingest.ROLES`, i.e. one of `"t1"`, `"t1ce"`, `"t2"`,
            `"flair"`) to resample into. Must have its own subdirectory under
            `transformations_dir`.
        target_native_path: Path to that role's ORIGINAL, pre-E2 NIfTI (e.g.
            `IngestResult.paths[target_role]`) -- the file whose geometry is the actual
            source DICOM series' geometry. Defines the output voxel grid.
        out_dir: Directory to write intermediate and output files into. Created if missing.

    Returns:
        Path to the resampled, validated mask NIfTI, in `target_native_path`'s exact voxel
        grid (shape, spacing, orientation, affine), values in `{0, 1, 2, 3}`.

    Raises:
        FileNotFoundError: If `transformations_dir` has no subdirectory for `target_role` --
            names the missing directory and lists the roles that DO have one.
        RuntimeError: If the resampled mask's shape does not match
            `target_native_path`'s shape, or if any of its values do not round cleanly to
            `{0, 1, 2, 3}` -- either way, a result that must not be allowed to reach
            `write_dicom_seg` looking plausible.
    """
    # Lazy imports: brainles_preprocessing lives only in .venv-clinical (see the
    # module docstring). nibabel is kept lazy too, so importing this module never
    # requires anything beyond numpy, matching this codebase's other clinical modules.
    import nibabel as nib
    from brainles_preprocessing.transform import Transform

    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(
            f"resample_mask_to_source: expected a (D, H, W) mask, got shape {mask.shape}."
        )

    transformations_dir = Path(transformations_dir)
    target_native_path = Path(target_native_path)
    out_dir = ensure_dir(out_dir)

    role_dir = transformations_dir / target_role
    if not role_dir.is_dir():
        available_roles = (
            sorted(p.name for p in transformations_dir.iterdir() if p.is_dir())
            if transformations_dir.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"resample_mask_to_source: no transformations directory for role {target_role!r} "
            f"at {role_dir}. Role(s) with a transformations directory under "
            f"{transformations_dir}: {available_roles}."
        )

    # Write the atlas-space mask to disk: Transform.apply (like every brainles-preprocessing
    # entry point) works on NIfTI file paths, not in-memory arrays.
    atlas_mask_path = out_dir / "atlas_space_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.asarray(atlas_affine)), str(atlas_mask_path))

    resampled_path = out_dir / f"{target_role}_native_mask.nii.gz"
    log_path = out_dir / f"{target_role}_resample.log"

    Transform(transformations_dir).apply(
        target_modality_name=target_role,
        target_modality_img=str(target_native_path),
        moving_image=str(atlas_mask_path),
        output_img_path=str(resampled_path),
        log_file_path=str(log_path),
        interpolator="genericLabel",
        inverse=True,
    )

    resampled_img = nib.load(str(resampled_path))
    resampled_data = np.asarray(resampled_img.dataobj)

    target_shape = tuple(nib.load(str(target_native_path)).shape)
    if resampled_data.shape != target_shape:
        raise RuntimeError(
            f"resample_mask_to_source: resampled mask shape {resampled_data.shape} does not "
            f"match target native image shape {target_shape} ({target_native_path}) -- the "
            "resample did not land on the expected grid and must not be trusted."
        )

    rounded = np.rint(resampled_data)
    max_deviation = float(np.max(np.abs(resampled_data - rounded))) if resampled_data.size else 0.0
    if max_deviation > _ROUND_TOLERANCE:
        raise RuntimeError(
            f"resample_mask_to_source: resampled mask at {resampled_path} has values that do "
            f"not round cleanly to an integer (max deviation from the nearest integer: "
            f"{max_deviation:.4g}, tolerance {_ROUND_TOLERANCE}) after 'genericLabel' "
            "interpolation, which is designed to preserve discrete label values exactly. "
            "This indicates the resample blended labels instead of preserving them and must "
            "not be trusted."
        )

    class_map = rounded.astype(np.uint8)
    invalid_values = sorted(set(np.unique(class_map).tolist()) - _VALID_CLASS_VALUES)
    if invalid_values:
        raise RuntimeError(
            f"resample_mask_to_source: resampled mask at {resampled_path} contains class "
            f"value(s) outside {sorted(_VALID_CLASS_VALUES)}: {invalid_values}."
        )

    # Overwrite the raw Transform output with the validated, clean uint8 class map --
    # `resampled_path` is the one path this function promises, and it should carry exactly
    # what was validated above, not the float64 array `Transform.apply` originally wrote.
    # Built from the affine alone (not `resampled_img.header`): passing that header along
    # would keep ITS float64 data-dtype field, silently upcasting `class_map` back to
    # float64 on save instead of writing the clean uint8 array this function just validated.
    nib.save(nib.Nifti1Image(class_map, resampled_img.affine), str(resampled_path))

    logger.info(
        "resample_mask_to_source: resampled atlas-space mask into %r native geometry at %s",
        target_role,
        resampled_path,
    )
    return resampled_path
