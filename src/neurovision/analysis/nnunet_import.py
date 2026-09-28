"""Imports nnU-Net's raw NIfTI predictions into our scoring path (Gate A).

Gate A (`docs/research_docs/preregistrations/preregistration_strong_baseline.md`, Amendment 1 item
8 "Scoring convention") compares our model against nnU-Net. nnU-Net writes
one hard-label NIfTI per case (e.g. `BraTS2021_01417.nii.gz`) on the
ORIGINAL raw BraTS 2021 grid (240x240x155) -- not on our cropped grid, and
not in our label convention. Before any such prediction can be scored
through `neurovision.analysis.replay.replay_case` -- the same path every
other arm in this project is scored through -- it must be converted:

    nnU-Net labels -> our label convention -> our cropped geometry
    -> +/-20 pseudo-logits on region channels (ET, TC, WT)
    -> `replay_case` with the project default post-processing chain

## The label convention swap

Ours (`neurovision.data.preprocessing.remap_labels`): 0 background,
1 NCR/NET, 2 ED, 3 ET. nnU-Net's, as WE set it in our own exporter
(`scripts/export_nnunet_dataset.py`, `4->3, 2->1, 1->2`): nnU-Net label 1 is
OUR label 2 (ED), nnU-Net label 2 is OUR label 1 (NCR). 0 and 3 already
agree. So importing a prediction back is the exact inverse of that export
remap: swap 1<->2, leave 0 and 3 alone. `nnunet_to_project_labels` is that
swap, and because swapping two values is its own inverse, the SAME swap
also converts OUR convention to nnU-Net's convention (used by
`roundtrip_self_test` to build a synthetic nnU-Net-style file).

## Why `import_prediction` asserts the reoriented affine equals `meta["affine"]`

A prediction written on the wrong grid -- most dangerously, a left-right
mirrored one -- must never be scored as if it were correct. Trap 3 in
`docs/research_docs/lessons.md`: brain-mask Dice actually scores *higher* on a mirrored
volume (0.9416 vs 0.9394), so a coarse sanity check on the output content
cannot catch this. The only thing that can is checking GEOMETRY, before any
scoring happens: read the NIfTI's own affine, reorient it to the project's
`target_axcodes` the same way `preprocess_case` did for the ground truth,
and require the result to land on the exact same affine and shape as the
matching case's `meta.json`. A header that claims the wrong orientation (or
a genuinely mirrored volume with a self-consistent-looking header) is
exactly what this check exists to catch before the wrong number ever reaches
a results table.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neurovision.analysis.real_dicom_scoring import region_masks
from neurovision.analysis.replay import replay_case
from neurovision.data.preprocessing import crop_to_bbox, reorient_to_axcodes
from neurovision.inference.postprocess import uncrop_to_original
from neurovision.metrics import REGION_NAMES

__all__ = [
    "nnunet_to_project_labels",
    "import_prediction",
    "labels_to_pseudo_logits",
    "score_imported_case",
    "roundtrip_self_test",
]

logger = logging.getLogger(__name__)

# nnU-Net's own labels, as OUR exporter (scripts/export_nnunet_dataset.py)
# writes them: 0 background, 1 = ED, 2 = NCR/NET, 3 = ET. Anything outside
# this set means the file did not come from our nnU-Net pipeline.
_VALID_NNUNET_LABELS = {0, 1, 2, 3}


def nnunet_to_project_labels(pred: np.ndarray) -> np.ndarray:
    """Converts nnU-Net's label convention to ours (and back -- it is self-inverse).

    nnU-Net label 1 (our exporter's ED) becomes our label 2 (ED); nnU-Net
    label 2 (our exporter's NCR) becomes our label 1 (NCR). 0 (background)
    and 3 (ET) are unchanged in both conventions.

    Args:
        pred: Integer label array of any shape, values in `{0, 1, 2, 3}`.

    Returns:
        `uint8` array, same shape, with 1 and 2 swapped.

    Raises:
        ValueError: If `pred` contains any value outside `{0, 1, 2, 3}`.
    """
    pred = np.asarray(pred)
    unique_vals = np.unique(pred)
    bad_vals = sorted(int(v) for v in unique_vals if int(v) not in _VALID_NNUNET_LABELS)
    if bad_vals:
        raise ValueError(
            f"nnunet_to_project_labels: unexpected label value(s) {bad_vals}; "
            f"expected values in {sorted(_VALID_NNUNET_LABELS)}."
        )

    out = np.asarray(pred, dtype=np.uint8).copy()
    swapped = out.copy()
    swapped[out == 1] = 2
    swapped[out == 2] = 1
    return swapped


def import_prediction(
    nifti_path: str | Path,
    meta: Mapping[str, Any],
    target_axcodes: tuple[str, str, str],
    *,
    atol: float = 1e-3,
) -> np.ndarray:
    """Loads one nnU-Net NIfTI prediction into our cropped geometry and label convention.

    Pipeline: load the NIfTI with its own affine -> reorient to
    `target_axcodes` (flips/transposes only, never resamples -- see
    `neurovision.data.preprocessing.reorient_to_axcodes`) -> assert the
    result matches this case's `meta.json` exactly (the geometry gate, see
    module docstring) -> crop to `meta["bbox"]` -> swap nnU-Net's label
    convention to ours.

    Args:
        nifti_path: Path to nnU-Net's hard-label prediction NIfTI for one
            case, on the ORIGINAL (uncropped) raw grid.
        meta: The matching case's `meta.json`, as a dict. Needs `affine`
            (4x4, the ORIGINAL grid's affine after the project's own
            reorientation), `original_shape` (3 ints), and `bbox` (three
            `[start, end)` pairs).
        target_axcodes: Voxel axis codes every case was normalized to during
            preprocessing, e.g. `("L", "P", "S")` -- must match
            `cfg.data.preprocessing.target_axcodes`, never hardcoded by the
            caller of this module.
        atol: Absolute tolerance for the affine-equality check.

    Returns:
        `(D, H, W)` uint8 labels, values in `{0, 1, 2, 3}`, in OUR
        convention, cropped to the same geometry as that case's
        `label.npy`.

    Raises:
        ValueError: If the reoriented affine does not match `meta["affine"]`
            within `atol`, or the reoriented shape does not match
            `meta["original_shape"]` -- either means this prediction and
            `meta` describe different grids (e.g. a mirrored or otherwise
            mis-oriented file), and it must not be scored.
    """
    nifti_path = Path(nifti_path)
    img = nib.load(str(nifti_path))
    raw = np.asanyarray(img.dataobj)
    source_affine = np.asarray(img.affine, dtype=np.float64)
    target = tuple(str(c) for c in target_axcodes)

    reoriented, updated_affine = reorient_to_axcodes(raw, source_affine, target)

    expected_affine = np.asarray(meta["affine"], dtype=np.float64)
    if not np.allclose(updated_affine, expected_affine, atol=atol):
        raise ValueError(
            f"import_prediction: reoriented affine for {nifti_path} does not match "
            f"meta['affine'] within atol={atol}. This is the geometry gate that catches a "
            "prediction written on the wrong grid (e.g. mirrored); a brain-mask Dice check "
            "cannot catch a mirror (docs/research_docs/lessons.md "
            "trap 3), so this affine equality is the "
            f"actual gate.\nreoriented affine:\n{updated_affine}\n"
            f"expected affine (meta['affine']):\n{expected_affine}"
        )

    expected_shape = tuple(int(s) for s in meta["original_shape"])
    if tuple(int(s) for s in reoriented.shape) != expected_shape:
        raise ValueError(
            f"import_prediction: reoriented shape {tuple(reoriented.shape)} for {nifti_path} "
            f"does not match meta['original_shape'] {expected_shape}. The prediction and this "
            "case's meta.json describe different grids."
        )

    cropped = crop_to_bbox(reoriented, meta["bbox"])
    return nnunet_to_project_labels(cropped)


def labels_to_pseudo_logits(labels: np.ndarray, magnitude: float = 20.0) -> np.ndarray:
    """Turns hard labels into saturating +/-`magnitude` logits on the 3 region channels.

    `replay_case` (and every other scored arm in this project) consumes
    logits, not hard labels, so nnU-Net's discrete prediction is expressed
    as logits so far from 0 that sigmoid + the default threshold 0.5
    recovers the exact same hard mask -- this is a format conversion, not
    an approximation. Region definitions come from
    `neurovision.analysis.real_dicom_scoring.region_masks` (ET = {3},
    TC = {1, 3}, WT = {1, 2, 3}), so this can never silently disagree with
    the project's own region convention.

    Args:
        labels: `(D, H, W)` integer labels, values in `{0, 1, 2, 3}`, in OUR
            convention.
        magnitude: The saturating logit value used inside a region's mask;
            `-magnitude` is used outside it.

    Returns:
        `(3, D, H, W)` float32 array, channel order `(ET, TC, WT)` (matches
        `neurovision.metrics.REGION_NAMES`).
    """
    masks = region_masks(labels)
    stacked = np.stack([masks[name] for name in REGION_NAMES], axis=0)  # (3, D, H, W) bool
    return np.where(stacked, magnitude, -magnitude).astype(np.float32)


def score_imported_case(
    labels_pred: np.ndarray,
    label_gt: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    lesionwise: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Scores one imported nnU-Net prediction through the project's own metric path.

    Thin wrapper: `labels_to_pseudo_logits` then
    `neurovision.analysis.replay.replay_case` with the project default
    post-processing chain (`postprocess_cfg=None`) and threshold 0.5 --
    identical to how every other arm in this project is scored, so an
    nnU-Net row is directly comparable.

    Args:
        labels_pred: `(D, H, W)` integer prediction, OUR convention, cropped
            geometry (e.g. `import_prediction`'s return value).
        label_gt: `(D, H, W)` integer ground truth, OUR convention, same
            cropped geometry as `labels_pred`.
        spacing: Voxel spacing in mm, `(dz, dy, dx)`, from the case's
            `meta.json`.
        lesionwise: `None` (default) to skip lesion-wise metrics, or a
            mapping of keyword arguments forwarded to `replay_case`'s own
            `lesionwise` argument.

    Returns:
        The flat metric dict `replay_case` returns (`dice_*`, `hd95_*`, and
        the lesion-wise columns when `lesionwise` is not `None`).
    """
    logits = labels_to_pseudo_logits(labels_pred)
    return replay_case(
        logits,
        label_gt,
        threshold=0.5,
        postprocess_cfg=None,
        spacing=spacing,
        lesionwise=lesionwise,
    )


def roundtrip_self_test(
    label_cropped: np.ndarray,
    meta: Mapping[str, Any],
    target_axcodes: tuple[str, str, str],
    tmp_dir: str | Path,
) -> None:
    """Pre-registered gate: proves the encode/decode round trip is lossless before any real score.

    Takes a real case's own cropped ground-truth label, uncrops it,
    converts it to nnU-Net's convention (the SAME 1<->2 swap
    `nnunet_to_project_labels` performs, applied inline here rather than by
    calling that function -- so a test can independently corrupt the decode
    step, inside `import_prediction`, without also corrupting this encode
    step), writes it as a NIfTI using `meta["affine"]`, then runs
    `import_prediction` on that file and requires the result to equal the
    original cropped label exactly. Nothing from a real nnU-Net run should
    be scored until this passes for that case.

    Args:
        label_cropped: `(D, H, W)` cropped ground-truth label, OUR
            convention. Must contain BOTH label 1 (NCR) and label 2 (ED) --
            a case with only one of them cannot expose a missed swap,
            because swapping a value that is not present is a no-op.
        meta: The matching case's `meta.json` (`bbox`, `original_shape`,
            `affine`).
        target_axcodes: Voxel axis codes cases were normalized to, e.g.
            `("L", "P", "S")`.
        tmp_dir: Directory to write the one temporary NIfTI this function
            creates. The only file write this module performs.

    Raises:
        ValueError: If `label_cropped` does not contain both label 1 and
            label 2.
        AssertionError: If the decoded round trip does not exactly equal
            `label_cropped` -- the encode/decode pipeline (a missed label
            swap, or an orientation mismatch) is broken.
    """
    label_cropped = np.asarray(label_cropped)
    present = {int(v) for v in np.unique(label_cropped)}
    if not {1, 2}.issubset(present):
        raise ValueError(
            "roundtrip_self_test: label_cropped must contain both label 1 (NCR) and label 2 "
            "(ED) -- a case without both cannot detect a missed 1<->2 swap, which is exactly "
            "the bug this self-test exists to catch."
        )

    full = uncrop_to_original(label_cropped, meta["bbox"], meta["original_shape"])

    # OUR -> nnU-Net convention: the same 1<->2 swap, written out inline
    # (not via nnunet_to_project_labels) -- see the docstring above for why.
    nnunet_full = full.copy()
    nnunet_full[full == 1] = 2
    nnunet_full[full == 2] = 1

    affine = np.asarray(meta["affine"], dtype=np.float64)
    img = nib.Nifti1Image(nnunet_full.astype(np.int16), affine)
    tmp_path = Path(tmp_dir) / "roundtrip_self_test.nii.gz"
    nib.save(img, str(tmp_path))

    decoded = import_prediction(tmp_path, meta, target_axcodes)
    if not np.array_equal(decoded, label_cropped):
        raise AssertionError(
            "roundtrip_self_test: the decoded prediction does not equal the original cropped "
            "label -- the encode/decode round trip is broken (a missed label swap, or an "
            "orientation mismatch). No real nnU-Net prediction should be scored until this "
            "self-test passes for this case."
        )
