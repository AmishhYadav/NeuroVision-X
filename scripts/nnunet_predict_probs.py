"""Runs the Gate A nnU-Net on one cohort and saves region LOGITS in our eval-directory format.

Contract: `docs/research_docs/preregistrations/preregistration_conformal_nnunet.md`
(read it first; it is never edited from here). The conformal bound is fitted on
per-voxel probabilities, and nnU-Net normally only writes hard labels. This
script runs `nnUNetv2_predict_from_modelfolder --save_probabilities` on a
cohort's PREPROCESSED arrays and converts what nnU-Net writes into

    <out_eval_dir>/logits/<case_id>.npy    fp16, (3, D, H, W), channels (ET, TC, WT)
    <out_eval_dir>/per_case_metrics.csv    default post-processing chain, voxel metrics
    <out_eval_dir>/summary.csv
    <out_eval_dir>/eval_config.yaml        the resolved config (post-processing chain)
    <out_eval_dir>/nnunet_probs_summary.json

which is exactly what `scripts/evaluate.py` writes for our own models, so
`scripts/conformal.py` and `scripts/replay_logits.py` run on it unmodified.
Lesion-wise metrics are off (panoptica is not installed on Kaggle).

`nnunetv2` is never imported here. It is called as a subprocess only, so this
module (and its tests) run in the main `.venv`, on CPU, with no nnunetv2 at all.

## What was verified in the installed nnunetv2==2.8.1 source (`.venv-analysis`)

- CLI flags: `nnunetv2/inference/predict_from_raw_data.py::
  predict_entry_point_modelfolder` -- `-i -o -m -f -chk -device`, `--disable_tta`,
  `--save_probabilities`. Reused through `nnunet_predict_session.build_predict_command`.
- Output files: `nnunetv2/inference/export_prediction.py::export_prediction_from_logits`
  writes, per case, `<case>.npz` with the SINGLE key `probabilities`
  (`np.savez_compressed(..., probabilities=...)`), `<case>.pkl` (properties) and the
  label map `<case><file_ending>`.
- Probabilities are sigmoids (region mode): `utilities/label_handling/label_handling.py::
  LabelManager.__init__` picks `torch.sigmoid` when any label is a list;
  `apply_inference_nonlin` applies it. Values are float32 in [0, 1].
- Channel order = the order of `dataset.json["labels"]` with background removed
  (`LabelManager._get_regions`). Ours: (whole tumour, tumour core, enhancing tumour).
  The label map is built by `convert_probabilities_to_segmentation`: for channel `i`,
  `segmentation[p_i > 0.5] = regions_class_order[i]` in channel order, later channels
  overwriting earlier ones -> WT writes 1, TC writes 2, ET writes 3.
- Axis order of the .npz: `convert_predicted_logits_to_segmentation_with_correct_shape`
  puts the probabilities back onto the INPUT image's grid (reverts nnU-Net's crop and
  resampling, then `transpose(0, *(i + 1 for i in transpose_backward))`). nnU-Net reads
  images with SimpleITK (`imageio/simpleitk_reader_writer.py::SimpleITKIO.read_images`,
  `sitk.GetArrayFromImage`), whose array axes are (z, y, x) -- the REVERSE of
  nibabel's (x, y, z). So for an input NIfTI whose nibabel array is (D, H, W), the
  probabilities come back as `(3, W, H, D)`. Verified empirically (SimpleITK in
  `.venv-analysis`): `GetArrayFromImage(read(nib_file)) == nib_array.transpose(2, 1, 0)`,
  and a label map written back with `sitk.GetImageFromArray` reads with nibabel as
  `(D, H, W)` again. So: probabilities need `transpose(0, 3, 2, 1)`; the exported
  label map, read with nibabel, is already `(D, H, W)` and needs no transform.
- `SimpleITKIO.write_seg` copies spacing/origin/direction from the input, so the label
  map lands on the same grid as the NIfTI we wrote.

## The input-writing decision

Gate A trained on raw BraTS 2021 NIfTIs read by SimpleITK. nnU-Net never reorients
(only `SimpleITKIOWithReorient` does, and the plans do not name it), so the network saw
the raw ARRAY layout: BraTS 2021 raw affine `diag(-1, -1, 1)` = axis codes LPS.
`preprocess_case` reorients every cohort to LPS (a no-op for BraTS 2021, flips for the
RAS-shipped BraTS-Africa), z-scores, then crops to the brain bounding box. So
`image.npy[c]` is already the training layout, cropped. We write it as a nibabel NIfTI
with the array UNCHANGED `(D, H, W)` and the affine
`meta["affine"]` (the reoriented FULL-grid affine) shifted by the crop start:
`new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ bbox_start`, so voxel (0, 0, 0) of
the crop sits where voxel `bbox_start` of the full grid sat. The header spacing then
comes from that affine. Channels follow `scripts/export_nnunet_dataset.py::_CHANNEL_ORDER`
(`t1, t1ce, t2, flair` -> `_0000.._0003`), which is also the order of `image.npy`
(`preprocessing._MODALITY_ROLES`, `BratsCase.modality_paths`). Because the input is
already cropped, the output needs no un-cropping.

This route differs from Gate A's raw-file route (nnU-Net z-scores a second time over
nonzero voxels, and our fp16 storage rounds the values), which is why the
pre-registration puts a bridge check on the test split before any number is read.

## Per-case checks

Thresholding the converted logits at 0.5 must reproduce nnU-Net's own exported label map.
The comparison rebuilds a label map from the thresholded regions with nnU-Net's own
overwrite rule (WT -> 1, TC -> 2, ET -> 3) and compares REGIONS derived from it
(WT = label > 0, TC = label in {2, 3}, ET = label == 3) with the same regions derived
from nnU-Net's label map. Rebuilding first matters: nnU-Net's sigmoid heads are not
forced to be nested, and a voxel with ET > 0.5 but TC <= 0.5 is labelled 3 by nnU-Net
(so TC = True there) while its raw TC channel says False. Comparing the raw channels
would charge that legitimate disagreement to the tolerance. A flipped axis or a swapped
channel still fails by thousands of voxels.

## Safety against partial or stale output

`scripts/conformal.py` builds its case list from a glob of `logits/*.npy` and only warns
if `per_case_metrics.csv` is missing. So `out_eval_dir` is wiped at the start, cases are
processed in chunks (inputs -> nnU-Net -> convert, check, score, delete; keeps Kaggle's
disk small), the mean WT Dice of the first few cases is checked against `label.npy` (the
label check compares nnU-Net with itself and cannot see a frame or modality-order error),
and the summary files, `eval_config.yaml` and a `COMPLETE` marker are written LAST, only
after the logits of exactly the predicted cases exist. A directory without `COMPLETE` is
not a result.

Example usage (one cohort; on Kaggle):

    python scripts/nnunet_predict_probs.py \\
        analysis.nnunet_probs.model_search_root=/kaggle/input \\
        analysis.nnunet_probs.prep_dir=/kaggle/input/neurovision-brats-ssa-prep \\
        analysis.nnunet_probs.splits_path=configs/data/splits_ssa.yaml \\
        analysis.nnunet_probs.split=test \\
        analysis.nnunet_probs.work_dir=/kaggle/working/nnunet_work \\
        analysis.nnunet_probs.out_eval_dir=/kaggle/working/eval_ssa_nnunet
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.replay import replay_case
from neurovision.data.dataset import load_splits
from neurovision.metrics import REGION_NAMES
from neurovision.metrics.segmentation import MetricAggregator
from neurovision.utils.io import ensure_dir, read_json, read_yaml, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def _load_predict_session() -> ModuleType:
    """Loads `scripts/nnunet_predict_session.py` so its helpers are reused, not copied.

    `scripts/` is not a package, so this uses the same `importlib` pattern
    `nnunet_predict_session.py` itself uses to load `nnunet_session.py`.

    Returns:
        The executed `nnunet_predict_session` module.
    """
    path = Path(__file__).resolve().parent / "nnunet_predict_session.py"
    spec = importlib.util.spec_from_file_location("nnunet_predict_session", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_predict_session = _load_predict_session()
find_model_folder = _predict_session.find_model_folder
check_checkpoint_identity = _predict_session.check_checkpoint_identity
read_file_ending = _predict_session.read_file_ending
build_predict_command = _predict_session.build_predict_command
read_checkpoint_meta = _predict_session.read_checkpoint_meta
_nnunetv2_version = _predict_session._nnunetv2_version
_require_cuda_or_raise = _predict_session._require_cuda_or_raise

# The only checkpoint the pre-registration allows (checkpoint_best is selected on
# training cases in fold "all", so it is not a fair signal).
_ALLOWED_CHECKPOINT = "checkpoint_final.pth"

# One input channel per modality, in the order scripts/export_nnunet_dataset.py
# gave nnU-Net during training: t1, t1ce, t2, flair -> _0000 .. _0003. image.npy
# stores its channels in the same order (preprocessing._MODALITY_ROLES).
_N_CHANNELS = 4

# The voxel-axis convention nnU-Net trained on: raw BraTS 2021 has affine
# diag(-1, -1, 1), i.e. axis codes LPS. nnU-Net never reorients, so a cohort whose
# preprocessed arrays are in any other layout would be a silent mirror/transpose.
_TRAINING_AXCODES = ("L", "P", "S")

# Readers whose internal array axes are (z, y, x) with NO reorientation, which is what
# our axis reasoning assumes. Gate A's real plans.json says "NibabelIO".
#   SimpleITKIO: imageio/simpleitk_reader_writer.py::read_images (sitk.GetArrayFromImage).
#   NibabelIO:   imageio/nibabel_reader_writer.py::read_images line 55,
#                `get_fdata().transpose((2, 1, 0))` "to be consistent with the way SimpleITk
#                reads images"; write_seg line 95 reverts it. Same axes, same label-map layout.
# NibabelIOWithReorient (same file, line 100+) reorients to RAS first, which would break
# our geometry, so it is refused along with anything else.
_ALLOWED_READERS = ("SimpleITKIO", "NibabelIO")

# dataset.json "labels" of a region-mode BraTS model, in channel order. Verified
# against LabelManager._get_regions (channel order = labels order, background dropped).
_EXPECTED_REGIONS = ([1, 2, 3], [2, 3], [3])  # whole tumour, tumour core, enhancing tumour
_EXPECTED_CLASS_ORDER = [1, 2, 3]

# nnU-Net's channel index for each of OUR regions (ET, TC, WT).
_NNUNET_CHANNEL = {"ET": 2, "TC": 1, "WT": 0}

# The single key export_prediction_from_logits stores the array under.
_NPZ_KEY = "probabilities"

# Clip bound for p before the logit. Computed in float32: in fp16, 1 - 1e-6 rounds
# to exactly 1.0, log(0) -> inf (CLAUDE.md trap #2).
_EPS = 1.0e-6

_INPUT_SUBDIR = "input"
_RAW_SUBDIR = "raw_out"


# ---------------------------------------------------------------------------
# 1. Checks on the model folder
# ---------------------------------------------------------------------------


def check_region_mode(dataset_json: dict[str, Any]) -> None:
    """Requires the model to be the BraTS region-mode one (WT, TC, ET sigmoids).

    Args:
        dataset_json: The model folder's parsed `dataset.json`.

    Raises:
        ValueError: If `labels` is not exactly background 0 plus whole tumour
            [1,2,3], tumour core [2,3], enhancing tumour [3] (in that order, which is
            nnU-Net's channel order), or `regions_class_order` is not [1,2,3].
    """
    labels = dataset_json.get("labels")
    if not isinstance(labels, dict) or labels.get("background") != 0:
        raise ValueError(f"dataset.json 'labels' must be a dict with background == 0: {labels!r}.")
    foreground = [v for k, v in labels.items() if k != "background"]
    got = tuple(list(v) if isinstance(v, (list, tuple)) else v for v in foreground)
    if got != _EXPECTED_REGIONS:
        raise ValueError(
            "This script only supports the region-mode BraTS model: dataset.json labels must "
            f"be background 0 then {list(_EXPECTED_REGIONS)} (whole tumour, tumour core, "
            f"enhancing tumour, in that order); got {got}."
        )
    if list(dataset_json.get("regions_class_order") or []) != _EXPECTED_CLASS_ORDER:
        raise ValueError(
            f"dataset.json regions_class_order must be {_EXPECTED_CLASS_ORDER}; got "
            f"{dataset_json.get('regions_class_order')!r}."
        )


def check_reader_writer(model_folder: Path) -> None:
    """Requires a plain (non-reorienting) reader, which the axis reasoning above depends on.

    Args:
        model_folder: The trainer dir, holding `plans.json`.

    Raises:
        ValueError: If `plans.json` names a reader other than `SimpleITKIO` or
            `NibabelIO` (e.g. `NibabelIOWithReorient`). A missing file or key only logs a
            warning (the per-case label check still guards).
    """
    plans_path = model_folder / "plans.json"
    plans = read_json(plans_path) if plans_path.is_file() else {}
    reader = plans.get("image_reader_writer")
    if reader is None:
        logger.warning(
            "nnunet_predict_probs: %s has no 'image_reader_writer'; assuming one of %s. The "
            "per-case label check will catch a wrong axis order.",
            plans_path,
            _ALLOWED_READERS,
        )
    elif reader not in _ALLOWED_READERS:
        raise ValueError(
            f"plans.json image_reader_writer is {reader!r}; the axis handling here is only "
            f"derived for {_ALLOWED_READERS}."
        )


# ---------------------------------------------------------------------------
# 2. Checks on the preprocessed cohort
# ---------------------------------------------------------------------------


def check_prep_complete(prep_dir: Path, case_ids: Sequence[str]) -> None:
    """Requires image.npy, label.npy and meta.json for every case (no partial cohort).

    Args:
        prep_dir: Root of the preprocessed tree.
        case_ids: Every case the split names.

    Raises:
        FileNotFoundError: Naming every case with a missing file.
    """
    problems = []
    for case_id in case_ids:
        missing = [
            name
            for name in ("image.npy", "label.npy", "meta.json")
            if not (prep_dir / case_id / name).is_file()
        ]
        if missing:
            problems.append(f"{case_id}: {missing}")
    if problems:
        raise FileNotFoundError(
            f"{len(problems)} case(s) under {prep_dir} lack preprocessed files; refusing to "
            f"predict a partial cohort: {problems}."
        )


def check_case_geometry(prep_dir: Path, case_ids: Sequence[str]) -> None:
    """Checks array shapes, crop box and voxel-axis convention of every case, cheaply.

    Args:
        prep_dir: Root of the preprocessed tree.
        case_ids: Cases to check.

    Raises:
        ValueError: Naming the first offending case, if image.npy is not
            `(4, D, H, W)`, its `(D, H, W)` differs from label.npy or
            `meta["cropped_shape"]`, the bbox disagrees with the cropped shape, or the
            affine's axis codes are not LPS (nnU-Net's training layout).
    """
    for case_id in case_ids:
        case_dir = prep_dir / case_id
        meta = read_json(case_dir / "meta.json")
        # mmap: read only the .npy headers, not the volumes.
        image_shape = np.load(case_dir / "image.npy", mmap_mode="r").shape
        label_shape = np.load(case_dir / "label.npy", mmap_mode="r").shape
        cropped = tuple(int(s) for s in meta["cropped_shape"])
        bbox_shape = tuple(int(end) - int(start) for start, end in meta["bbox"])
        if len(image_shape) != 4 or image_shape[0] != _N_CHANNELS:
            raise ValueError(f"{case_id}: image.npy shape {image_shape} is not (4, D, H, W).")
        if not (tuple(image_shape[1:]) == tuple(label_shape) == cropped == bbox_shape):
            raise ValueError(
                f"{case_id}: image {image_shape[1:]}, label {label_shape}, meta cropped_shape "
                f"{cropped} and bbox extent {bbox_shape} must all agree."
            )
        axcodes = nib.aff2axcodes(np.asarray(meta["affine"], dtype=np.float64))
        if tuple(axcodes) != _TRAINING_AXCODES:
            raise ValueError(
                f"{case_id}: affine axis codes are {tuple(axcodes)}, but nnU-Net trained on "
                f"{_TRAINING_AXCODES}. It never reorients, so this cohort would be silently "
                "mirrored or transposed."
            )


# ---------------------------------------------------------------------------
# 3. Writing nnU-Net's inputs
# ---------------------------------------------------------------------------


def crop_affine(meta: dict[str, Any]) -> np.ndarray:
    """The affine of the CROPPED array: the full-grid affine shifted by the crop start.

    Args:
        meta: A case's `meta.json` (`affine` = reoriented full-grid affine, `bbox`).

    Returns:
        `(4, 4)` float64 affine such that cropped voxel (0, 0, 0) has the world position
        full-grid voxel `bbox_start` had.
    """
    affine = np.asarray(meta["affine"], dtype=np.float64).copy()
    start = np.array([int(b[0]) for b in meta["bbox"]], dtype=np.float64)
    affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ start
    return affine


def write_nnunet_inputs(
    prep_dir: Path, case_ids: Sequence[str], input_dir: Path, file_ending: str
) -> None:
    """Writes `<case>_000{0..3}<file_ending>` float32 NIfTIs from each case's image.npy.

    The array is written UNCHANGED `(D, H, W)` (see the module docstring for why).

    Args:
        prep_dir: Root of the preprocessed tree.
        case_ids: Cases to write.
        input_dir: Created fresh by the caller; nnU-Net's `-i` folder.
        file_ending: `.nii` or `.nii.gz` (read from the model's dataset.json).

    Raises:
        ValueError: If `file_ending` is not one nibabel writes.
    """
    if file_ending not in (".nii", ".nii.gz"):
        raise ValueError(f"file_ending {file_ending!r} is not .nii or .nii.gz.")
    for case_id in case_ids:
        meta = read_json(prep_dir / case_id / "meta.json")
        affine = crop_affine(meta)
        # Header spacing comes from the affine's column lengths. meta["spacing"] was read
        # BEFORE reorientation, so only warn (axis order may legitimately differ).
        zooms = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.allclose(sorted(zooms), sorted(float(s) for s in meta["spacing"]), atol=1e-3):
            logger.warning(
                "%s: affine voxel size %s differs from meta spacing %s.",
                case_id,
                zooms,
                meta["spacing"],
            )
        image = np.load(prep_dir / case_id / "image.npy")  # (4, D, H, W) fp16
        for channel in range(_N_CHANNELS):
            volume = image[channel].astype(np.float32)  # (D, H, W)
            nib.save(
                nib.Nifti1Image(volume, affine),
                str(input_dir / f"{case_id}_{channel:04d}{file_ending}"),
            )


# ---------------------------------------------------------------------------
# 4. Running nnU-Net
# ---------------------------------------------------------------------------


def run_prediction(cmd: list[str]) -> float:
    """Runs the nnU-Net command and returns its wall time in seconds.

    Args:
        cmd: The full argv list.

    Returns:
        Wall-clock seconds.

    Raises:
        subprocess.CalledProcessError: If nnU-Net exits non-zero.
    """
    logger.info("nnunet_predict_probs: command: %s", " ".join(cmd))
    start = time.monotonic()
    # Not captured: nnU-Net's progress lines stream straight into the Kaggle log.
    subprocess.run(cmd, check=True)
    return time.monotonic() - start


# ---------------------------------------------------------------------------
# 5. Converting one case
# ---------------------------------------------------------------------------


def probabilities_to_logits(probs: np.ndarray) -> np.ndarray:
    """Turns probabilities into fp16 logits without ever doing the maths in fp16.

    Args:
        probs: Probabilities in [0, 1], any shape.

    Returns:
        fp16 array, same shape, `log(p / (1 - p))` of `p` clipped to `[1e-6, 1 - 1e-6]`
        (about +-13.8).

    Raises:
        ValueError: If any result is not finite (e.g. NaN probabilities).
    """
    p = np.asarray(probs, dtype=np.float32)  # float32 BEFORE clipping: see _EPS
    p = np.clip(p, np.float32(_EPS), np.float32(1.0) - np.float32(_EPS))
    logits = (np.log(p) - np.log1p(-p)).astype(np.float16)  # cast LAST
    if not np.isfinite(logits).all():
        raise ValueError("probabilities_to_logits: non-finite logit (NaN probabilities?).")
    return logits


def regions_from_nnunet_labels(labels: np.ndarray) -> np.ndarray:
    """Derives the (ET, TC, WT) boolean regions from an nnU-Net label map.

    nnU-Net's labels here: 1 = ED, 2 = NCR, 3 = ET.

    Args:
        labels: `(D, H, W)` integer label map, values in {0, 1, 2, 3}.

    Returns:
        `(3, D, H, W)` bool array, channel order `REGION_NAMES` (ET, TC, WT).
    """
    et = labels == 3
    tc = (labels == 2) | (labels == 3)
    wt = labels > 0
    return np.stack([et, tc, wt], axis=0)


def label_check(logits: np.ndarray, label_map: np.ndarray) -> dict[str, float]:
    """Compares regions from our logits with regions from nnU-Net's own label map.

    Args:
        logits: `(3, D, H, W)` fp16 logits, channels (ET, TC, WT).
        label_map: `(D, H, W)` nnU-Net label map on the same grid.

    Returns:
        Per region name, the fraction of the volume's voxels on which the two disagree.
    """
    probs = torch.sigmoid(torch.from_numpy(logits.astype(np.float32)))
    ours = (probs >= 0.5).numpy()  # (3, D, H, W) bool, ET/TC/WT
    # Rebuild nnU-Net's label map with ITS overwrite rule (see module docstring).
    rebuilt = np.zeros(label_map.shape, dtype=np.uint8)
    rebuilt[ours[REGION_NAMES.index("WT")]] = 1
    rebuilt[ours[REGION_NAMES.index("TC")]] = 2
    rebuilt[ours[REGION_NAMES.index("ET")]] = 3
    got = regions_from_nnunet_labels(rebuilt)
    want = regions_from_nnunet_labels(label_map)
    return {name: float((got[i] != want[i]).mean()) for i, name in enumerate(REGION_NAMES)}


def convert_case(
    case_id: str, raw_dir: Path, label_shape: tuple[int, ...], file_ending: str, tol: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Converts one case's nnU-Net outputs into fp16 (ET, TC, WT) logits on our grid.

    Args:
        case_id: The case.
        raw_dir: nnU-Net's output folder.
        label_shape: `(D, H, W)` of this case's label.npy.
        file_ending: nnU-Net's label-map file ending.
        tol: Largest allowed per-region mismatch fraction.

    Returns:
        `(logits, mismatch)`: `(3, D, H, W)` float16, and the per-region mismatch fractions.

    Raises:
        KeyError: If the .npz has no `probabilities` key.
        ValueError: If shapes disagree with the label, or a region's mismatch exceeds `tol`
            (names the case and the regions).
    """
    with np.load(raw_dir / f"{case_id}.npz") as archive:
        if _NPZ_KEY not in archive.files:
            raise KeyError(f"{case_id}.npz has keys {archive.files}, expected {_NPZ_KEY!r}.")
        probs = archive[_NPZ_KEY]
    # nnU-Net's array axes are SimpleITK's (W, H, D); ours are (D, H, W).
    expected = (len(_NNUNET_CHANNEL), *label_shape[::-1])
    if probs.shape != expected:
        raise ValueError(f"{case_id}: probabilities shape {probs.shape}, expected {expected}.")
    probs = probs.transpose(0, 3, 2, 1)  # (3, D, H, W), channels still (WT, TC, ET)
    ordered = np.stack([probs[_NNUNET_CHANNEL[name]] for name in REGION_NAMES], axis=0)
    logits = probabilities_to_logits(ordered)
    if logits.shape != (3, *label_shape):
        raise ValueError(f"{case_id}: logits shape {logits.shape}, label shape {label_shape}.")

    label_map = np.asarray(nib.load(str(raw_dir / f"{case_id}{file_ending}")).dataobj)
    if label_map.shape != tuple(label_shape):
        raise ValueError(f"{case_id}: nnU-Net label map {label_map.shape} vs {label_shape}.")
    if not np.isin(label_map, (0, 1, 2, 3)).all():
        raise ValueError(f"{case_id}: nnU-Net label map has values outside 0..3.")
    mismatch = label_check(logits, label_map)
    bad = {name: frac for name, frac in mismatch.items() if frac > tol}
    if bad:
        raise ValueError(
            f"{case_id}: converted logits do not reproduce nnU-Net's own label map "
            f"(mismatch fraction per region {bad}, tolerance {tol}). The axis order or the "
            "channel order is wrong."
        )
    return logits, mismatch


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------


def _require(value: Any, key: str, example: str) -> Any:
    """Raises a ValueError naming a config key left at its null default."""
    if value is None:
        raise ValueError(f"analysis.nnunet_probs.{key} is not set. Example: '{example}'.")
    return value


def log_use_mask_for_norm(model_folder: Path) -> list[bool] | None:
    """Logs the plans' `use_mask_for_norm`, warning if any channel does not use it.

    Our preprocessed arrays are already z-scored over nonzero voxels with exact zeros
    outside the brain; this route assumes nnU-Net re-normalises over that same nonzero
    mask (`use_mask_for_norm` true), as it did in Gate A.

    Args:
        model_folder: The trainer dir, holding `plans.json`.

    Returns:
        The `configurations.3d_fullres.use_mask_for_norm` list, or None if absent.
    """
    plans_path = model_folder / "plans.json"
    plans = read_json(plans_path) if plans_path.is_file() else {}
    flags = ((plans.get("configurations") or {}).get("3d_fullres") or {}).get("use_mask_for_norm")
    logger.info("nnunet_predict_probs: plans 3d_fullres use_mask_for_norm = %s", flags)
    if flags is None or not all(bool(f) for f in flags):
        logger.warning(
            "nnunet_predict_probs: use_mask_for_norm is %s, not all true; the preprocessed-"
            "array route assumes nnU-Net z-scores over the nonzero mask.",
            flags,
        )
    return None if flags is None else [bool(f) for f in flags]


def check_postprocess_is_default(cfg: DictConfig) -> None:
    """Refuses a config whose post-processing chain differs from the project default.

    `replay_case(postprocess_cfg=None)` always applies `configs/inference/default.yaml`'s
    chain, so `eval_config.yaml` must never record a different one.

    Args:
        cfg: The full composed config.

    Raises:
        ValueError: Naming every `inference.postprocess` key that differs.
    """
    default = read_yaml(Path(_CONFIG_DIR) / "inference" / "default.yaml")["postprocess"]
    got = OmegaConf.to_container(cfg.inference.postprocess, resolve=True)
    diff = {
        k: (got.get(k), default.get(k))
        for k in set(got) | set(default)
        if got.get(k) != default.get(k)
    }
    if diff:
        raise ValueError(
            "inference.postprocess differs from configs/inference/default.yaml (key: (config, "
            f"default)) {diff}. Scoring always uses the default chain, so eval_config.yaml "
            "must not record anything else."
        )


def _check_safe_to_wipe(out_dir: Path, others: Sequence[Path]) -> None:
    """Refuses to delete `out_dir` if it is, or contains, one of the inputs.

    Args:
        out_dir: The eval directory about to be removed and recreated.
        others: Paths that must survive (prep dir, work dir, model root, split file).

    Raises:
        ValueError: If any of `others` is `out_dir` itself or lives inside it.
    """
    root = out_dir.resolve()
    for other in others:
        resolved = other.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(
                f"out_eval_dir {out_dir} is, or contains, {other}; refusing to wipe it."
            )


def _check_dice_sanity(dice_wt: Sequence[float], minimum: float | None, after_n: int) -> None:
    """Logs the running mean WT Dice; raises if the first `after_n` cases are too low.

    The label check compares nnU-Net with itself, so a modality-order or frame error
    against `label.npy` would pass it. A real tumour model scores far above 0.5 WT Dice.

    Args:
        dice_wt: WT Dice of the cases scored so far.
        minimum: Smallest allowed mean, or None to disable the raise.
        after_n: The check fires once this many cases are scored.

    Raises:
        ValueError: If the mean WT Dice of the first `after_n` cases is below `minimum`.
    """
    mean = float(np.nanmean(dice_wt))
    logger.info(
        "nnunet_predict_probs: running mean dice_WT = %.4f over %d case(s)", mean, len(dice_wt)
    )
    if minimum is not None and len(dice_wt) == after_n and mean < minimum:
        raise ValueError(
            f"Mean dice_WT over the first {after_n} cases is {mean:.4f} < {minimum}. The "
            "predictions are not aligned with label.npy (modality order or frame error?)."
        )


def run_nnunet_probs(cfg: DictConfig) -> dict[str, Any]:
    """Predicts one cohort with probabilities and writes the eval directory.

    Cases are processed in chunks (write inputs -> nnU-Net -> convert, check, score,
    delete) so Kaggle's disk never holds a whole cohort of inputs and probabilities.
    `out_eval_dir` is wiped first, and its summary files, `eval_config.yaml` and the
    `COMPLETE` marker are written LAST, only after every case's logits exist, because
    `scripts/conformal.py` builds its case list from a glob of `logits/*.npy`.

    Args:
        cfg: The full composed Hydra config (`cfg.analysis.nnunet_probs`).

    Returns:
        The dict written to `nnunet_probs_summary.json`.

    Raises:
        ValueError: A null required key, a bad config value, a non-region model, a
            non-default post-processing chain, bad cohort geometry, a failed label check
            or a failed Dice sanity check.
        FileNotFoundError: A missing model folder, checkpoint, or preprocessed file.
        RuntimeError: A checkpoint identity mismatch, no usable CUDA, or an incomplete
            set of logits at the end.
        subprocess.CalledProcessError: nnU-Net exited non-zero.
    """
    pc = cfg.analysis.nnunet_probs
    search_root = Path(str(_require(pc.model_search_root, "model_search_root", "/kaggle/input")))
    out_dir = Path(str(_require(pc.out_eval_dir, "out_eval_dir", "outputs/eval_ssa_nnunet")))
    work_dir = Path(str(_require(pc.work_dir, "work_dir", "/kaggle/working/nnunet_work")))
    device = str(pc.device)
    if device not in ("cuda", "cpu"):
        raise ValueError(f"analysis.nnunet_probs.device must be 'cuda' or 'cpu', got {device!r}.")
    checkpoint_name = str(pc.checkpoint_name)
    if checkpoint_name != _ALLOWED_CHECKPOINT:
        raise ValueError(
            f"analysis.nnunet_probs.checkpoint_name must be {_ALLOWED_CHECKPOINT!r} "
            f"(pre-registered); got {checkpoint_name!r}. checkpoint_best is never used."
        )
    case_limit = None if pc.case_limit is None else int(pc.case_limit)
    if case_limit is not None and case_limit < 1:
        raise ValueError(
            f"analysis.nnunet_probs.case_limit must be >= 1 or null, got {case_limit}."
        )
    chunk_size = int(pc.chunk_size)
    if chunk_size < 1:
        raise ValueError(f"analysis.nnunet_probs.chunk_size must be >= 1, got {chunk_size}.")
    sanity_after_n = int(pc.sanity_after_n)
    if sanity_after_n < 1:
        raise ValueError(
            f"analysis.nnunet_probs.sanity_after_n must be >= 1, got {sanity_after_n}."
        )
    min_dice = None if pc.min_dice_wt_sanity is None else float(pc.min_dice_wt_sanity)
    tol = float(pc.label_check_tol)
    check_postprocess_is_default(cfg)

    # --- 1. Model folder: checkpoint identity, file ending, region mode ---
    model_folder = find_model_folder(search_root)
    meta = read_checkpoint_meta(model_folder / "fold_all" / checkpoint_name)
    check_checkpoint_identity(meta, int(pc.expect_epoch))
    file_ending = read_file_ending(model_folder)
    if file_ending not in (".nii", ".nii.gz"):
        raise ValueError(f"model file_ending {file_ending!r} is not .nii or .nii.gz.")
    check_region_mode(read_json(model_folder / "dataset.json"))
    check_reader_writer(model_folder)
    log_use_mask_for_norm(model_folder)

    # --- 2. Case list and cohort checks (case_limit truncates AFTER the full check) ---
    prep_dir = Path(str(pc.prep_dir))
    splits_path = Path(str(pc.splits_path))
    splits = load_splits(str(splits_path))
    split = str(pc.split)
    if split not in splits:
        raise ValueError(f"analysis.nnunet_probs.split={split!r} not in {sorted(splits)}.")
    case_ids = list(splits[split])
    check_prep_complete(prep_dir, case_ids)
    if case_limit is not None:
        case_ids = case_ids[:case_limit]
    if not case_ids:
        raise ValueError(f"Split {split!r} has no cases.")
    check_case_geometry(prep_dir, case_ids)
    _check_safe_to_wipe(out_dir, [prep_dir, work_dir, search_root, splits_path])

    # --- 3. Last preflight: a usable GPU, before anything is written ---
    if device == "cuda":
        _require_cuda_or_raise()

    # --- 4. Fresh output dir: stale or partial logits must never be consumed ---
    if out_dir.exists():
        logger.info("nnunet_predict_probs: removing existing %s", out_dir)
        shutil.rmtree(out_dir)
    logits_dir = ensure_dir(out_dir / "logits")
    input_dir, raw_dir = work_dir / _INPUT_SUBDIR, work_dir / _RAW_SUBDIR

    cmd = build_predict_command(input_dir, raw_dir, model_folder, device, bool(pc.disable_tta))
    cmd.append("--save_probabilities")
    aggregator = MetricAggregator()
    max_mismatch = {name: 0.0 for name in REGION_NAMES}
    dice_wt: list[float] = []
    chunk_walls: list[float] = []
    chunks = [case_ids[i : i + chunk_size] for i in range(0, len(case_ids), chunk_size)]

    for k, chunk in enumerate(chunks, start=1):
        # Scratch folders are ours: recreated fresh per chunk.
        for scratch in (input_dir, raw_dir):
            shutil.rmtree(scratch, ignore_errors=True)
            ensure_dir(scratch)
        write_nnunet_inputs(prep_dir, chunk, input_dir, file_ending)
        chunk_wall = run_prediction(cmd)
        chunk_walls.append(chunk_wall)
        logger.info(
            "nnunet_predict_probs: chunk %d/%d, %d case(s), %.1f s (%.2f s/case)",
            k,
            len(chunks),
            len(chunk),
            chunk_wall,
            chunk_wall / len(chunk),
        )
        missing = [
            c
            for c in chunk
            if not ((raw_dir / f"{c}.npz").is_file() and (raw_dir / f"{c}{file_ending}").is_file())
        ]
        if missing:
            raise FileNotFoundError(
                f"nnU-Net wrote no .npz/label map for {len(missing)} case(s): {missing}."
            )

        # Convert, check, save, score and clean up, one case at a time.
        for case_id in chunk:
            case_dir = prep_dir / case_id
            label = np.load(case_dir / "label.npy")
            spacing = tuple(float(s) for s in read_json(case_dir / "meta.json")["spacing"])
            logits, mismatch = convert_case(case_id, raw_dir, label.shape, file_ending, tol)
            for name, frac in mismatch.items():
                max_mismatch[name] = max(max_mismatch[name], frac)
            np.save(logits_dir / f"{case_id}.npy", logits)
            # Score the fp16 logits we just saved, cast to float32 exactly as
            # replay.load_case_logits does, so a later replay reproduces these numbers.
            metrics = replay_case(
                logits.astype(np.float32),
                label,
                threshold=0.5,
                postprocess_cfg=None,  # project default chain
                spacing=spacing,
                lesionwise=None,
            )
            aggregator.update(case_id, metrics)
            dice_wt.append(float(metrics["dice_WT"]))
            _check_dice_sanity(dice_wt, min_dice, sanity_after_n)
            # Disk on Kaggle is limited: drop everything made for this case.
            for leftover in (
                raw_dir / f"{case_id}.npz",
                raw_dir / f"{case_id}.pkl",
                raw_dir / f"{case_id}{file_ending}",
                *(input_dir / f"{case_id}_{ch:04d}{file_ending}" for ch in range(_N_CHANNELS)),
            ):
                leftover.unlink(missing_ok=True)

    if len(dice_wt) < sanity_after_n:  # a tiny run never reached the in-loop check
        _check_dice_sanity(dice_wt, min_dice, len(dice_wt))

    # --- 5. Completeness, then the summary files LAST, then the marker ---
    found = {p.stem for p in logits_dir.glob("*.npy")}
    if found != set(case_ids):
        raise RuntimeError(
            f"logits/ does not hold exactly the predicted cases: missing "
            f"{sorted(set(case_ids) - found)}, extra {sorted(found - set(case_ids))}."
        )
    wall_s = float(sum(chunk_walls))
    s_per_case = wall_s / len(case_ids)
    per_case = aggregator.per_case()
    per_case.to_csv(out_dir / "per_case_metrics.csv")
    aggregator.summary().to_csv(out_dir / "summary.csv")
    summary = {
        "model_folder": str(model_folder),
        "checkpoint_epoch": meta["current_epoch"],
        "nnunetv2_version": _nnunetv2_version(),
        "command": cmd,
        "split": split,
        "disable_tta": bool(pc.disable_tta),
        "device": device,
        "wall_s": wall_s,
        "s_per_case": s_per_case,
        "chunk_size": chunk_size,
        "chunk_wall_s": chunk_walls,
        "n_cases": len(case_ids),
        "label_check_max_mismatch": max_mismatch,
        "label_check_tol": tol,
    }
    (out_dir / "eval_config.yaml").write_text(
        _eval_config_yaml(cfg, model_folder, checkpoint_name, meta["current_epoch"]),
        encoding="utf-8",
    )
    write_json(summary, out_dir / "nnunet_probs_summary.json")
    (out_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    _print_summary(per_case, out_dir, s_per_case)
    return summary


def _eval_config_yaml(cfg: DictConfig, model_folder: Path, checkpoint_name: str, epoch: int) -> str:
    """The resolved full config plus a block saying which model really made the logits.

    The full config is kept because `rebuild_predictions.py` reads
    `inference.postprocess` from it.

    Args:
        cfg: The full composed config.
        model_folder: The nnU-Net model folder used.
        checkpoint_name: The checkpoint file name.
        epoch: The checkpoint's epoch.

    Returns:
        YAML text.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    container["nnunet_probs_provenance"] = {
        "model": "nnunet_v2_3dfullres_gatea",
        "model_folder": str(model_folder),
        "checkpoint": checkpoint_name,
        "checkpoint_epoch": epoch,
        "disable_tta": bool(cfg.analysis.nnunet_probs.disable_tta),
        "note": "model/experiment blocks above are the project defaults and do NOT describe "
        "the model that produced these logits",
    }
    return OmegaConf.to_yaml(OmegaConf.create(container))


def _print_summary(table: pd.DataFrame, out_dir: Path, s_per_case: float) -> None:
    """Prints a compact end-of-run block (print, like score_nnunet.py's summary)."""
    lines = [
        "=" * 70,
        f"nnunet_predict_probs: out_eval_dir={out_dir}",
        "=" * 70,
        f"  cases: {len(table)}",
    ]
    for metric in ("dice_ET", "dice_TC", "dice_WT"):
        lines.append(f"  {metric}: mean={float(table[metric].mean()):.4f}")
    lines.append(f"  s/case: {s_per_case:.2f}")
    print("\n".join(lines))


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs nnU-Net with probabilities on one cohort, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_nnunet_probs(cfg)


if __name__ == "__main__":
    main()
