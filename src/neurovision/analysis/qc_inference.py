"""Shared QC-inference building blocks: logits -> the `(image, mask, entropy)` volume.

`scripts/train_qc.py` (Phase C3) and `scripts/validate_qc.py` (its
not-yet-written sibling) both need to turn a case's SAVED LOGITS
(`<eval_dir>/logits/<case_id>.npy`) into the packed volume `SegQC`
consumes. `scripts/` is not an importable package (`pyproject.toml`'s
`packages.find` only looks under `src/`), so a second script cannot import
this logic from the first without a `spec_from_file_location` hack. This
module is the shared home for it -- a faithful copy of
`scripts/train_qc.py`'s `entropy_from_logits`, `_resize_packed`,
`_CaseArrays` and `_load_case_arrays`, plus one new function
(`pack_sample`) that assembles the same three-channel stack
`QCPairsDataset.__getitem__` builds today, so a second driver does not have
to re-derive that assembly by hand.

This module is a pure extraction: it contains no training loop, no
`Dataset`, no sampler. `scripts/train_qc.py` still owns those, and (per a
separate follow-up task) will come to depend on the functions here instead
of defining its own private copies.

## The central design decision: downsample, never crop

Dice is a WHOLE-VOLUME quantity. A packed `(image, mask, entropy)` volume
is therefore resized as a WHOLE to `target_shape` (default 64^3) instead of
cropped: trilinear for the image and entropy channels (continuous,
physically meaningful to blend), NEAREST for the mask channel (trilinear on
a 0/1 mask produces fractional "mask" values like 0.37, which is not a mask
-- see `resize_packed`). The whole case stays in view, and the Dice target
stays exactly correct, because it is read straight from a FULL-RESOLUTION
computation (`neurovision.data.qc_pairs.generate_pairs` /
`generate_one_pair`) and never recomputed after resizing.

## The frozen entropy channel

The entropy channel is computed ONCE per case, from the case's saved
logits, and does NOT change when a mask-degradation step damages the mask
channel for a given sample. This is deliberate, not a bug: at deployment
the QC model sees the model's OWN entropy map alongside SOME mask (its own
prediction, already fixed by the time QC runs), so training and validation
data must reflect that the entropy map is a fixed observation independent
of which particular way a given pair happens to have damaged the mask.
Part of the QC model's job is learning to notice when a mask and an
entropy map DISAGREE (e.g. a mask that looks confident and complete over a
region the segmentation model itself was uncertain about) -- collapsing
that signal by recomputing entropy from a degraded mask would erase
exactly the thing a second, independent QC model is supposed to add.

## Why the Dice target is computed at full resolution

See `neurovision.data.qc_pairs.generate_pairs`: whatever regression target
a caller pairs with a sample from `load_case_arrays` / `pack_sample` must
come from that full-resolution computation, taken as-is, never recomputed
from the resized mask and label -- a crop or a downsampled mask no longer
describes what a whole-case Dice number means (see the module docstring's
"central design decision" section above).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from neurovision.inference.postprocess import postprocess_logits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entropy, in nats, from raw logits
# ---------------------------------------------------------------------------


def entropy_from_logits(logits: Tensor) -> Tensor:
    """Per-voxel Bernoulli predictive entropy, in NATS, computed from logits.

    `H(p) = p * softplus(-z) + (1 - p) * softplus(z)`, where `p =
    sigmoid(z)`. Computed from LOGITS, never from a clamped probability: an
    `eps` clamp sized for fp32 (`p.clamp(eps, 1 - eps)` with `eps=1e-6`) is a
    complete no-op in fp16, whose own epsilon is ~9.8e-4 -- `1.0 - 1e-6`
    rounds to exactly `1.0`, so `log(1 - p)` becomes `log(0)` -> `-inf`, and
    `0 * -inf` is `NaN`. That exact bug cost this project 10.5 GPU-hours on a
    real training run (see CLAUDE.md's traps list, `docs/lessons.md`, and
    `neurovision.models.fusion.adaptive_fusion.BranchAmbiguity`, whose fix
    this mirrors). `softplus` is finite for any finite input, so a saturated
    logit gives `0 * finite = 0` -- the correct entropy of a certain
    prediction -- instead of `NaN`.

    Deliberately left in NATS here, NOT normalised to `[0, 1]` by `ln 2` the
    way `neurovision.analysis.detection._entropy_from_logits` and
    `BranchAmbiguity` do: this entropy is only ever a model INPUT feature in
    this module, never a bounded quantity reported in a table, so there is
    no reason to rescale it.

    Args:
        logits: Raw (pre-sigmoid) logits, any shape.

    Returns:
        Entropy in nats, same shape as `logits`, finite everywhere a finite
        logit was given.
    """
    p = torch.sigmoid(logits)
    return p * F.softplus(-logits) + (1.0 - p) * F.softplus(logits)


# ---------------------------------------------------------------------------
# Resizing: trilinear for continuous channels, nearest for the mask
# ---------------------------------------------------------------------------


def resize_packed(packed: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
    """Resizes one packed `(3, D, H, W)` sample to `target_shape`.

    Channel 0 (image) and channel 2 (entropy) are continuous, physically
    meaningful quantities and use TRILINEAR interpolation. Channel 1 (mask)
    is resized with NEAREST-neighbour: trilinear on a binary 0/1 mask would
    blend neighbouring voxels into fractional values like 0.37 at every
    downsampled position, which is not a mask and is not a meaningful
    "probability" either (a degraded mask's output is a hard binary
    decision, not a calibrated probability) -- see the module docstring's
    "central design decision" section for why resizing happens at all
    instead of cropping.

    Args:
        packed: `(3, D, H, W)`, channel order `(image, mask, entropy)`.
        target_shape: `(D', H', W')` to resize to.

    Returns:
        `(3, *target_shape)`, float32.
    """
    batched = packed.unsqueeze(0)  # (1, 3, D, H, W)
    image_entropy = batched[:, (0, 2), ...]  # (1, 2, D, H, W)
    mask = batched[:, 1:2, ...]  # (1, 1, D, H, W)

    image_entropy_resized = F.interpolate(
        image_entropy, size=target_shape, mode="trilinear", align_corners=False
    )
    mask_resized = F.interpolate(mask, size=target_shape, mode="nearest")

    out = torch.empty((3, *target_shape), dtype=torch.float32)
    out[0] = image_entropy_resized[0, 0]
    out[1] = mask_resized[0, 0]
    out[2] = image_entropy_resized[0, 1]
    return out


# ---------------------------------------------------------------------------
# Per-case array loading
# ---------------------------------------------------------------------------


@dataclass
class CaseArrays:
    """One case's arrays, loaded once from disk.

    Attributes:
        pred_mask: The DEPLOYED prediction -- `postprocess_logits` run on
            the case's saved logits at the project default threshold and
            post-processing chain. `(3, D, H, W)`, `uint8`, channel order
            `(ET, TC, WT)`. This, never the label, is what a mask-
            degradation step damages (master plan section 2, principle 3:
            downstream models train on PREDICTED masks, never ground-truth
            masks -- see `neurovision.data.qc_pairs`'s module docstring for
            the full argument).
        label: Ground truth as an integer class map, `(D, H, W)`, values in
            `{0, 1, 2, 3}`. Used ONLY to compute the Dice regression target
            (e.g. via `neurovision.data.qc_pairs.generate_pairs` /
            `generate_one_pair`, which expand it internally).
        image_modality: One MRI modality's voxel values, `(D, H, W)`,
            `float32` -- `cfg.analysis.qc.modality_index` selects which of
            the 4 preprocessed modalities.
        entropy: Per-voxel Bernoulli predictive entropy in nats, `(3, D, H,
            W)`, computed once from the case's raw logits. Meant to stay
            FROZEN across every degraded variant of this case's mask -- see
            the module docstring's "frozen entropy channel" section.
    """

    pred_mask: np.ndarray
    label: np.ndarray
    image_modality: np.ndarray
    entropy: np.ndarray


def load_case_arrays(cfg: Any, eval_dir: Path, prep_dir: Path, case_id: str) -> CaseArrays:
    """Loads and derives everything a QC dataset/driver needs for one case.

    Args:
        cfg: The full composed Hydra config (`postprocess_logits` reads
            `cfg.inference.postprocess`; `cfg.analysis.qc.modality_index`
            selects the image channel).
        eval_dir: A `scripts/evaluate.py` output directory holding
            `logits/<case_id>.npy`.
        prep_dir: Root of the preprocessed BraTS data, holding
            `<case_id>/{image.npy,label.npy}`.
        case_id: The case identifier.

    Returns:
        A populated `CaseArrays`.

    Raises:
        FileNotFoundError: Any of the three source `.npy` files is missing.
        ValueError: The loaded image/label/logits arrays disagree on spatial
            shape -- almost always means one of the three came from a
            different preprocessing run or a different geometry (e.g.
            `predictions/`'s ORIGINAL uncropped shape instead of `logits/`'s
            CROPPED one).
    """
    logits_path = eval_dir / "logits" / f"{case_id}.npy"
    if not logits_path.is_file():
        raise FileNotFoundError(f"load_case_arrays({case_id!r}): {logits_path} does not exist.")
    logits = torch.from_numpy(np.load(logits_path).astype(np.float32))  # (3, D, H, W)

    label_path = prep_dir / case_id / "label.npy"
    if not label_path.is_file():
        raise FileNotFoundError(f"load_case_arrays({case_id!r}): {label_path} does not exist.")
    label = np.load(label_path).astype(np.int64)  # (D, H, W)

    image_path = prep_dir / case_id / "image.npy"
    if not image_path.is_file():
        raise FileNotFoundError(f"load_case_arrays({case_id!r}): {image_path} does not exist.")
    image = np.load(image_path).astype(np.float32)  # (4, D, H, W)

    if logits.shape[1:] != label.shape:
        raise ValueError(
            f"load_case_arrays({case_id!r}): logits spatial shape {tuple(logits.shape[1:])} "
            f"from {logits_path} disagrees with label shape {tuple(label.shape)} from "
            f"{label_path}. This usually means the two came from different preprocessing runs."
        )
    if image.shape[1:] != label.shape:
        raise ValueError(
            f"load_case_arrays({case_id!r}): image spatial shape {tuple(image.shape[1:])} from "
            f"{image_path} disagrees with label shape {tuple(label.shape)} from {label_path}."
        )

    pred_mask = postprocess_logits(logits.unsqueeze(0), cfg)[0]  # (3, D, H, W)
    entropy = entropy_from_logits(logits)  # (3, D, H, W), nats

    modality_index = int(cfg.analysis.qc.modality_index)
    if modality_index < 0 or modality_index >= image.shape[0]:
        raise IndexError(
            f"load_case_arrays({case_id!r}): analysis.qc.modality_index={modality_index} is out "
            f"of range for {image_path}, which has {image.shape[0]} modality channel(s)."
        )
    image_modality = image[modality_index]  # (D, H, W)

    return CaseArrays(
        pred_mask=pred_mask.numpy().astype(np.uint8),
        label=label,
        image_modality=image_modality,
        entropy=entropy.numpy(),
    )


# ---------------------------------------------------------------------------
# Assembling one packed sample from a case's arrays plus a region's mask
# ---------------------------------------------------------------------------


def pack_sample(
    arrays: CaseArrays,
    mask: np.ndarray,
    region_channel: int,
    target_shape: tuple[int, int, int],
) -> Tensor:
    """Stacks `(image, mask, entropy)` for one region and resizes to `target_shape`.

    Mirrors `scripts/train_qc.py`'s `QCPairsDataset.__getitem__` exactly:
    the image channel is `arrays.image_modality` (unchanged by the region
    or by any mask degradation), the mask channel is `mask[region_channel]`
    -- `mask` is a REGION-STACK `(3, D, H, W)` array, e.g. one degraded
    pair's `.mask` from `neurovision.data.qc_pairs.generate_one_pair` /
    `generate_pairs`, NOT a pre-indexed `(D, H, W)` array -- and the entropy
    channel is `arrays.entropy[region_channel]`, the FROZEN entropy for that
    same region (see the module docstring's "frozen entropy channel"
    section: entropy is read from `arrays`, never recomputed from `mask`).

    Args:
        arrays: One case's arrays, from `load_case_arrays`.
        mask: `(3, D, H, W)` region-stack mask, channel order `(ET, TC,
            WT)` -- indexed by `region_channel` here, same as
            `arrays.pred_mask` and `arrays.entropy`.
        region_channel: Which of the 3 region channels to pack (0=ET,
            1=TC, 2=WT, per `neurovision.data.transforms.REGION_NAMES`).
        target_shape: `(D', H', W')` the packed volume is resized to.

    Returns:
        `(3, *target_shape)`, float32, channel order `(image, mask,
        entropy)`. Valid (all-zero channel 1) even when `mask[region_channel]`
        is entirely background -- no special-casing is needed since stacking
        and resizing an all-zero array raises nothing.
    """
    image_channel = torch.from_numpy(arrays.image_modality)
    mask_channel = torch.from_numpy(mask[region_channel].astype(np.float32))
    entropy_channel = torch.from_numpy(arrays.entropy[region_channel])

    packed = torch.stack([image_channel, mask_channel, entropy_channel], dim=0)  # (3, D, H, W)
    return resize_packed(packed, target_shape)
