"""Hydra entry point that trains SegQC, Phase C3 of the master plan.

`neurovision.data.qc_pairs.generate_pairs` and `neurovision.models.qc.SegQC`
are built and unit tested already; nothing trains the model they exist for.
This script is that trainer.

## The binding principle (restated -- read `neurovision/data/qc_pairs.py`'s
module docstring for the full argument)

Master plan section 2, principle 3: **downstream models train on PREDICTED
masks, never ground-truth masks.** The masks this script degrades come from
a checkpoint's own SAVED LOGITS (`<eval_dir>/logits/<case_id>.npy`), never
from `label.npy`. The label is used for exactly one thing -- computing the
Dice regression target inside `generate_pairs`.

## The central design decision: downsample, never crop

Dice is a WHOLE-VOLUME quantity. `generate_pairs` computes one Dice number
per region for an entire case's mask against an entire case's label -- that
number describes the case, not any sub-region of it. If this dataset
trained on random crops of a packed volume, each crop would carry the
FULL-VOLUME Dice as its target even though the crop itself might sit
entirely inside a correctly-segmented area of an otherwise badly-wrong
case (or vice versa) -- the crop's label would not describe what the crop
shows. This is not a minor approximation; it silently teaches the model to
associate arbitrary sub-volumes with a number that has nothing to do with
their own content.

So every packed `(image, mask, entropy)` volume is resized as a WHOLE to
`target_shape` (default 64^3) instead: trilinear for the image and entropy
channels (continuous, physically meaningful to blend), NEAREST for the mask
channel (trilinear on a 0/1 mask produces fractional "mask" values like
0.37, which is not a mask -- see `_resize_packed`). The whole case stays in
view, and the Dice target stays exactly correct, because it is read
straight from `generate_pairs`'s FULL-RESOLUTION computation and never
recomputed after resizing.

## The frozen entropy channel

The entropy channel is computed ONCE per case, from the case's saved
logits, and does NOT change when `qc_pairs.degrade_mask` damages the mask
channel for a given training pair. This is deliberate, not a bug: at
deployment the QC model sees the model's OWN entropy map alongside SOME
mask (its own prediction, already fixed by the time QC runs), so training
data must reflect that the entropy map is a fixed observation independent
of which particular way this pair happens to have damaged the mask. Part of
the QC model's job is learning to notice when a mask and an entropy map
DISAGREE (e.g. a mask that looks confident and complete over a region the
segmentation model itself was uncertain about) -- collapsing that signal by
recomputing entropy from the degraded mask would erase exactly the thing a
second, independent QC model is supposed to add.

## Why the sigmoid lives in the training loop, not the model

`SegQC.forward` returns a raw logit (see `neurovision.models.qc`'s module
docstring for why: every model in this project keeps the nonlinearity out
of `forward` so there is exactly one place per script it gets applied,
rather than risking two call sites silently disagreeing about whether a
number is a logit or a probability). `predicted_dice()` applies the sigmoid;
this script calls it once, right before computing the loss, and nowhere
else.

## Why `__getitem__` calls `generate_one_pair`, not `generate_pairs`

`__getitem__` needs exactly ONE per-region pair. Calling
`generate_pairs(..., specs=[spec], per_region=True)` and keeping only the
matching pair would compute -- and throw away -- the whole-mask pair and
every OTHER region's pair too, on the hot path of a CPU training loop.
`neurovision.data.qc_pairs.generate_one_pair` (added alongside this script)
returns the SAME pair `generate_pairs` would, at the same generator state,
without scoring the three pairs nobody reads -- see its own docstring for
why it still has to perform the SAME `degrade_mask` calls in the SAME
order as `generate_pairs` (to keep `generator`'s state in sync for
stochastic spec kinds), and only skips the wasted Dice computation.

## Case-grouped shuffling, not full shuffling and not no shuffling

Two competing needs, both real: (1) `QCPairsDataset`'s single-case cache
(`_case_arrays`) only pays off when consecutive samples share a case, and
(2) a batch made of several degradations of ONE case has highly correlated
gradients, and training on the exact same case ORDER every epoch is close
to training on a sorted dataset -- a real optimisation problem, not a
theoretical one. Plain `shuffle=True` solves (2) and destroys (1) (every
sample would need a fresh disk read); plain `shuffle=False` solves (1) and
leaves (2) unsolved.

`CaseGroupedSampler` gets both: it permutes the ORDER OF CASES independently
every epoch, while keeping each case's own `per_case` samples contiguous
within that order. The cache still loads each case once per epoch; batches
still draw from a shuffled sequence of cases. Its permutation is seeded from
`cfg.seed` and the epoch number (via `set_epoch`, the same convention
`torch.utils.data.DistributedSampler` uses) -- `run_training` calls
`set_epoch` with the training loop's own `epoch` variable, which after a
resume is the RESTORED epoch, never a counter that restarts at 0. Getting
that wrong would make a resumed run silently repeat an earlier epoch's exact
case order.

## Checkpointing

`neurovision.training.checkpoint.save_checkpoint` / `load_checkpoint` are
reused as-is -- they already give atomic writes, full RNG-state resume, and
the `find_resume_checkpoint` convention every other training entry point in
this project uses. Nothing here needed a second checkpoint format: there is
no scheduler and no AMP scaler (see below), so the extra fields those
functions accept are simply left at their `None` defaults.

## W&B and AMP are deliberately omitted

This trains on CPU only -- SegQC is small (well under 5M parameters,
64^3 inputs) and the training set (a few thousand pairs at most) fits the
"Mac is a correctness harness, not a compute device" rule with room to
spare. AMP is CUDA-only in this project (`neurovision.utils.device.
amp_enabled`) and buys nothing on CPU; W&B is skipped to keep this script
runnable with zero network access during tests and casual local runs. Both
could be added later without touching `QCPairsDataset` or the loss.

Example usage (the `model=segqc` override is REQUIRED -- the root config's
default model group is `unet3d`, which has none of the keys
`neurovision.models.qc.build_segqc` reads):

    python scripts/train_qc.py model=segqc \\
        analysis.qc.train_eval_dir=outputs/neurovision/eval_val \\
        analysis.qc.heldout_eval_dir=outputs/neurovision/eval_test
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

from neurovision.data.qc_pairs import DEFAULT_SPECS, DegradationSpec, generate_one_pair
from neurovision.data.transforms import REGION_NAMES
from neurovision.inference.postprocess import postprocess_logits
from neurovision.models.qc import build_segqc, predicted_dice
from neurovision.training.checkpoint import (
    find_resume_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


# ---------------------------------------------------------------------------
# Directory resolution (fit-on-val / hold-out-test separation)
# ---------------------------------------------------------------------------


def resolve_dirs(cfg: DictConfig) -> tuple[Path, Path, Path, Path]:
    """Resolves and validates `analysis.qc`'s two eval directories.

    Mirrors `scripts/calibrate.py`'s `resolve_eval_dirs`: the QC model
    trains on `train_eval_dir` and is scored, every epoch, against
    `heldout_eval_dir` -- and the two must not resolve to the same
    directory, or "held out" would just mean "the training split again",
    making every epoch's Spearman/MAE number meaningless.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `(train_eval_dir, train_prep_dir, heldout_eval_dir, heldout_prep_dir)`.

    Raises:
        ValueError: Either eval_dir key is `None`, or the two resolve (via
            `Path.resolve()`) to the same directory.
        FileNotFoundError: Either resolved eval_dir does not exist.
    """
    qc_cfg = cfg.analysis.qc
    train_eval_raw = qc_cfg.train_eval_dir
    heldout_eval_raw = qc_cfg.heldout_eval_dir

    missing = []
    if train_eval_raw is None:
        missing.append("analysis.qc.train_eval_dir")
    if heldout_eval_raw is None:
        missing.append("analysis.qc.heldout_eval_dir")
    if missing:
        raise ValueError(
            f"{' and '.join(missing)} must be set. scripts/train_qc.py needs two eval "
            "directories: one to TRAIN the QC model on (the VAL split's eval_dir) and one to "
            "hold out for the per-epoch Spearman/MAE report (the TEST split's eval_dir). "
            "Example:\n"
            "  python scripts/train_qc.py model=segqc "
            "analysis.qc.train_eval_dir=outputs/<experiment>/eval_val "
            "analysis.qc.heldout_eval_dir=outputs/<experiment>/eval_test"
        )

    train_eval_dir = Path(train_eval_raw)
    heldout_eval_dir = Path(heldout_eval_raw)

    if not train_eval_dir.is_dir():
        raise FileNotFoundError(
            f"analysis.qc.train_eval_dir does not exist: {train_eval_dir.resolve()}"
        )
    if not heldout_eval_dir.is_dir():
        raise FileNotFoundError(
            f"analysis.qc.heldout_eval_dir does not exist: {heldout_eval_dir.resolve()}"
        )

    if train_eval_dir.resolve() == heldout_eval_dir.resolve():
        raise ValueError(
            f"analysis.qc.train_eval_dir and analysis.qc.heldout_eval_dir both resolve to "
            f"{train_eval_dir.resolve()}. Training the QC model on the same split its "
            "Spearman/MAE is reported against every epoch would make that report meaningless "
            "-- point train_eval_dir at the VAL split's eval_dir and heldout_eval_dir at the "
            "TEST split's."
        )

    train_prep_dir = Path(qc_cfg.train_prep_dir)
    heldout_prep_dir = Path(qc_cfg.heldout_prep_dir)

    return train_eval_dir, train_prep_dir, heldout_eval_dir, heldout_prep_dir


def _shared_case_ids(eval_dir: Path, prep_dir: Path, max_cases: int | None) -> list[str]:
    """Case ids with BOTH saved logits under `eval_dir` and a case dir under `prep_dir`.

    Args:
        eval_dir: A `scripts/evaluate.py` output directory.
        prep_dir: Root of the preprocessed BraTS data.
        max_cases: If not `None`, truncates the sorted id list to this many
            entries -- `analysis.qc.max_cases`, a deterministic subsetting
            knob (not a random subsample), so the first real run can be
            small.

    Returns:
        Sorted, deduplicated list of shared case ids.

    Raises:
        FileNotFoundError: `<eval_dir>/logits` does not exist.
        ValueError: No case id is present in both `<eval_dir>/logits` and
            `prep_dir`.
    """
    logits_dir = eval_dir / "logits"
    if not logits_dir.is_dir():
        raise FileNotFoundError(
            f"{logits_dir} does not exist. scripts/train_qc.py needs saved logits -- re-run "
            "scripts/evaluate.py with inference.evaluation.save_logits=true against this "
            "eval_dir."
        )

    logits_ids = {p.stem for p in logits_dir.glob("*.npy")}
    prep_ids = {p.name for p in prep_dir.iterdir() if p.is_dir()} if prep_dir.is_dir() else set()
    shared = sorted(logits_ids & prep_ids)
    if not shared:
        raise ValueError(
            f"No case id is present in both {logits_dir} and {prep_dir}; nothing to train on."
        )

    if max_cases is not None:
        shared = shared[: int(max_cases)]
    return shared


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
    this script, never a bounded quantity reported in a table, so there is
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


def _resize_packed(packed: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
    """Resizes one packed `(3, D, H, W)` sample to `target_shape`.

    Channel 0 (image) and channel 2 (entropy) are continuous, physically
    meaningful quantities and use TRILINEAR interpolation. Channel 1 (mask)
    is resized with NEAREST-neighbour: trilinear on a binary 0/1 mask would
    blend neighbouring voxels into fractional values like 0.37 at every
    downsampled position, which is not a mask and is not a meaningful
    "probability" either (`degrade_mask`'s output is a hard binary decision,
    not a calibrated probability) -- see the module docstring's "central
    design decision" section for why resizing happens at all instead of
    cropping.

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
# Per-case array loading (cached by QCPairsDataset)
# ---------------------------------------------------------------------------


@dataclass
class _CaseArrays:
    """One case's arrays, loaded once and cached across consecutive `__getitem__` calls.

    Attributes:
        pred_mask: The DEPLOYED prediction -- `postprocess_logits` run on
            the case's saved logits at the project default threshold and
            post-processing chain. `(3, D, H, W)`, `uint8`, channel order
            `(ET, TC, WT)`. This, never the label, is what `degrade_mask`
            damages (see the module docstring's binding principle).
        label: Ground truth as an integer class map, `(D, H, W)`, values in
            `{0, 1, 2, 3}`. Passed straight to `generate_one_pair`, which
            expands it internally -- used ONLY to compute the Dice
            regression target.
        image_modality: One MRI modality's voxel values, `(D, H, W)`,
            `float32` -- `analysis.qc.modality_index` selects which of the
            4 preprocessed modalities.
        entropy: Per-voxel Bernoulli predictive entropy in nats, `(3, D, H,
            W)`, computed once from the case's raw logits. FROZEN across
            every degradation of this case -- see the module docstring's
            "frozen entropy channel" section.
    """

    pred_mask: np.ndarray
    label: np.ndarray
    image_modality: np.ndarray
    entropy: np.ndarray


def _load_case_arrays(cfg: Any, eval_dir: Path, prep_dir: Path, case_id: str) -> _CaseArrays:
    """Loads and derives everything `QCPairsDataset` needs for one case.

    Args:
        cfg: The full composed Hydra config (`postprocess_logits` reads
            `cfg.inference.postprocess`; `analysis.qc.modality_index`
            selects the image channel).
        eval_dir: A `scripts/evaluate.py` output directory holding
            `logits/<case_id>.npy`.
        prep_dir: Root of the preprocessed BraTS data, holding
            `<case_id>/{image.npy,label.npy}`.
        case_id: The case identifier.

    Returns:
        A populated `_CaseArrays`.

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
        raise FileNotFoundError(f"_load_case_arrays({case_id!r}): {logits_path} does not exist.")
    logits = torch.from_numpy(np.load(logits_path).astype(np.float32))  # (3, D, H, W)

    label_path = prep_dir / case_id / "label.npy"
    if not label_path.is_file():
        raise FileNotFoundError(f"_load_case_arrays({case_id!r}): {label_path} does not exist.")
    label = np.load(label_path).astype(np.int64)  # (D, H, W)

    image_path = prep_dir / case_id / "image.npy"
    if not image_path.is_file():
        raise FileNotFoundError(f"_load_case_arrays({case_id!r}): {image_path} does not exist.")
    image = np.load(image_path).astype(np.float32)  # (4, D, H, W)

    if logits.shape[1:] != label.shape:
        raise ValueError(
            f"_load_case_arrays({case_id!r}): logits spatial shape {tuple(logits.shape[1:])} "
            f"from {logits_path} disagrees with label shape {tuple(label.shape)} from "
            f"{label_path}. This usually means the two came from different preprocessing runs."
        )
    if image.shape[1:] != label.shape:
        raise ValueError(
            f"_load_case_arrays({case_id!r}): image spatial shape {tuple(image.shape[1:])} from "
            f"{image_path} disagrees with label shape {tuple(label.shape)} from {label_path}."
        )

    pred_mask = postprocess_logits(logits.unsqueeze(0), cfg)[0]  # (3, D, H, W)
    entropy = entropy_from_logits(logits)  # (3, D, H, W), nats

    modality_index = int(cfg.analysis.qc.modality_index)
    image_modality = image[modality_index]  # (D, H, W)

    return _CaseArrays(
        pred_mask=pred_mask.numpy().astype(np.uint8),
        label=label,
        image_modality=image_modality,
        entropy=entropy.numpy(),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class QCPairsDataset(Dataset):
    """Streams `(packed volume, target Dice)` training pairs for `SegQC`.

    A pair is a FULL VOLUME, not a small tensor -- 187 val cases x roughly
    the whole of `DEFAULT_SPECS` is far more data than fits in RAM or on
    disk at once, so nothing here is precomputed or materialised up front.
    `__len__` is computed from case count x number of specs x number of
    regions alone (all fixed, known without touching a file); `__getitem__`
    does the actual work.

    Indexing order (why consecutive indices share a case): index `i` decodes
    as `case_pos, spec_idx, region_pos = decode(i)`, with `region_pos`
    varying FASTEST, then `spec_idx`, then `case_pos` -- so every run of
    `len(specs) * len(regions)` consecutive indices belongs to exactly one
    case. `_case_arrays` caches the single most-recently-loaded case's
    arrays (`pred_mask`, `label`, `image_modality`, `entropy` -- everything
    that costs a disk read plus a `postprocess_logits` pass to produce), so
    a pass that visits one case's indices contiguously loads and
    post-processes each case's logits ONCE per epoch rather than once per
    training pair. `CaseGroupedSampler` (below) is what gives the training
    loop that access pattern while still shuffling case ORDER every epoch --
    see the module docstring's "case-grouped shuffling" section for why a
    plain sequential or fully-shuffled DataLoader is each wrong for a
    different reason.

    See the module docstring for why this resizes whole volumes instead of
    cropping, and why the entropy channel is frozen across a case's
    degraded pairs.
    """

    def __init__(
        self,
        cfg: Any,
        eval_dir: str | Path,
        prep_dir: str | Path,
        *,
        specs: Sequence[DegradationSpec] | None = None,
    ) -> None:
        """Builds the case/spec/region index. Touches the filesystem only to list case ids.

        Args:
            cfg: The full composed Hydra config, exposing `cfg.analysis.qc`
                and `cfg.inference.postprocess`.
            eval_dir: A `scripts/evaluate.py` output directory holding
                `logits/*.npy`.
            prep_dir: Root of the preprocessed BraTS data.
            specs: Degradations to draw from. Defaults to
                `neurovision.data.qc_pairs.DEFAULT_SPECS`. Not
                config-driven (`analysis.qc` has no `specs` key) -- tests
                pass a short list here to keep runtime tiny; a real run
                uses the library default.

        Raises:
            FileNotFoundError: `<eval_dir>/logits` does not exist.
            ValueError: No case id is shared between `eval_dir` and
                `prep_dir`, or `analysis.qc.regions` names something outside
                `neurovision.data.transforms.REGION_NAMES`.
        """
        qc_cfg = cfg.analysis.qc
        self._cfg = cfg
        self._eval_dir = Path(eval_dir)
        self._prep_dir = Path(prep_dir)
        self._specs: tuple[DegradationSpec, ...] = (
            tuple(specs) if specs is not None else DEFAULT_SPECS
        )
        self._target_shape: tuple[int, int, int] = tuple(int(v) for v in qc_cfg.target_shape)
        self._seed = int(cfg.seed)

        region_names = [str(r) for r in qc_cfg.regions]
        for name in region_names:
            if name not in REGION_NAMES:
                raise ValueError(
                    f"analysis.qc.regions entry {name!r} is not one of {REGION_NAMES}."
                )
        self._region_channels: list[int] = [REGION_NAMES.index(name) for name in region_names]

        self._case_ids = _shared_case_ids(self._eval_dir, self._prep_dir, qc_cfg.max_cases)
        self._per_case = len(self._specs) * len(self._region_channels)

        # The single-entry cache _case_arrays relies on -- see class docstring.
        self._cached_case_id: str | None = None
        self._cached_arrays: _CaseArrays | None = None

        logger.info(
            "QCPairsDataset(%s): %d case(s), %d spec(s) x %d region(s) = %d sample(s)/case, "
            "%d sample(s) total.",
            self._eval_dir,
            len(self._case_ids),
            len(self._specs),
            len(self._region_channels),
            self._per_case,
            len(self),
        )

    def __len__(self) -> int:
        return len(self._case_ids) * self._per_case

    @property
    def num_cases(self) -> int:
        """Number of cases in this dataset -- what `CaseGroupedSampler` permutes."""
        return len(self._case_ids)

    @property
    def per_case(self) -> int:
        """Samples per case (`len(specs) * len(regions)`) -- what `CaseGroupedSampler`
        keeps contiguous within its permuted case order."""
        return self._per_case

    def _decode_index(self, index: int) -> tuple[int, int, int]:
        """Splits a flat index into `(case_pos, spec_idx, region_pos)`. See class docstring."""
        case_pos, remainder = divmod(index, self._per_case)
        spec_idx, region_pos = divmod(remainder, len(self._region_channels))
        return case_pos, spec_idx, region_pos

    def _case_arrays(self, case_id: str) -> _CaseArrays:
        """Returns `case_id`'s arrays, loading from disk only on a cache miss."""
        if self._cached_case_id != case_id:
            self._cached_arrays = _load_case_arrays(
                self._cfg, self._eval_dir, self._prep_dir, case_id
            )
            self._cached_case_id = case_id
        assert self._cached_arrays is not None  # just set above, or hit on a prior call
        return self._cached_arrays

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Builds one training pair.

        Args:
            index: Flat index in `[0, len(self))`.

        Returns:
            `(sample, target)`. `sample` is `(3, *target_shape)` float32,
            channel order `(image, mask, entropy)`. `target` is a 0-D
            float32 tensor, the region's Dice against the label, taken
            AS-IS from `generate_one_pair`'s full-resolution computation --
            never recomputed after resizing.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} out of range for dataset of length {len(self)}.")

        case_pos, spec_idx, region_pos = self._decode_index(index)
        case_id = self._case_ids[case_pos]
        region_channel = self._region_channels[region_pos]
        spec = self._specs[spec_idx]

        arrays = self._case_arrays(case_id)

        # An explicit generator seeded from (cfg.seed, index) -- never a
        # global np.random call -- so the SAME index always reproduces the
        # SAME degradation, regardless of access order or how many other
        # samples were drawn first. generate_one_pair returns exactly the
        # per-region pair generate_pairs(..., specs=[spec], per_region=True)
        # would, at this generator state, without computing the whole-mask
        # pair's or any other region's Dice -- see the module docstring's
        # "why generate_one_pair" section.
        generator = np.random.default_rng([self._seed, index])
        pair = generate_one_pair(
            arrays.pred_mask, arrays.label, spec, region_channel, generator=generator
        )

        image_channel = torch.from_numpy(arrays.image_modality)
        mask_channel = torch.from_numpy(pair.mask[region_channel].astype(np.float32))
        entropy_channel = torch.from_numpy(arrays.entropy[region_channel])

        packed = torch.stack([image_channel, mask_channel, entropy_channel], dim=0)
        sample = _resize_packed(packed, self._target_shape)

        target = torch.tensor(float(pair.dice[region_channel]), dtype=torch.float32)
        return sample, target


# ---------------------------------------------------------------------------
# Sampler: shuffle case ORDER per epoch, keep each case's pairs contiguous
# ---------------------------------------------------------------------------


class CaseGroupedSampler(Sampler[int]):
    """Shuffles CASE ORDER every epoch, while keeping each case's own pairs contiguous.

    See the module docstring's "case-grouped shuffling" section for why
    neither `shuffle=True` nor `shuffle=False` alone is right for training
    on a `QCPairsDataset`: plain sequential order trains on the same fixed
    case order every epoch (batches are many correlated degradations of one
    volume); plain full shuffling defeats `QCPairsDataset`'s single-case
    cache. This sampler gets both -- the case order is a fresh permutation
    every epoch, but indices `[case_pos * per_case, (case_pos + 1) *
    per_case)` are always emitted together, in that fixed internal order,
    so a case's samples still arrive as one contiguous run.

    Must be paired with `DataLoader(..., shuffle=False, sampler=...)` --
    PyTorch's `DataLoader` raises if both `shuffle=True` and a `sampler` are
    given, since they are two different ways of choosing the same thing.
    """

    def __init__(self, num_cases: int, per_case: int, seed: int) -> None:
        """Builds the sampler. Does no shuffling yet -- see `set_epoch`.

        Args:
            num_cases: Number of cases (`QCPairsDataset.num_cases`).
            per_case: Samples per case (`QCPairsDataset.per_case`).
            seed: `cfg.seed`. Combined with the epoch number (`set_epoch`)
                to derive a reproducible, resume-safe per-epoch case
                permutation -- never a global RNG call.
        """
        self._num_cases = num_cases
        self._per_case = per_case
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch this sampler's next `__iter__` call permutes for.

        MUST be called with the run's ACTUAL epoch number, including after a
        resume -- where that number is the RESTORED epoch, never a counter
        that restarts at 0. `run_training` calls this once per iteration of
        its epoch loop, using that loop's own `epoch` variable, which is
        already correct across a resume for exactly this reason (it comes
        from `range(start_epoch, total_epochs)`, and `start_epoch` comes
        from the checkpoint). Calling this with the wrong epoch would make a
        resumed run silently repeat an earlier epoch's exact case order --
        the same "training on a fixed order" problem this sampler exists to
        avoid, recurring on a cycle instead of disappearing.

        Args:
            epoch: The epoch about to run.
        """
        self._epoch = epoch

    def __iter__(self):
        # A fresh generator every call (never a stored, mutating one) so
        # calling __iter__ twice without an intervening set_epoch (e.g. a
        # DataLoader with persistent_workers, or a caller inspecting the
        # sampler) reproduces the SAME order rather than advancing further.
        generator = np.random.default_rng([self._seed, self._epoch])
        case_order = generator.permutation(self._num_cases)
        for case_pos in case_order:
            start = int(case_pos) * self._per_case
            yield from range(start, start + self._per_case)

    def __len__(self) -> int:
        return self._num_cases * self._per_case


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int,
) -> tuple[float, int]:
    """Runs one training epoch.

    Args:
        model: `SegQC` (or any module with the same `forward` contract).
        loader: Yields `(x, target)` batches from a `QCPairsDataset`.
        optimizer: Already built from `model.parameters()`.
        device: Where `model` lives.
        global_step: Optimizer-step counter carried in from the previous
            epoch (or a resumed checkpoint), incremented once per batch.

    Returns:
        `(mean_loss, global_step)` -- the mean per-sample MSE loss over the
        epoch, and the updated step counter.
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, target in loader:
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(x)
        # Sigmoid applied HERE, not inside SegQC.forward -- see the module
        # docstring's "why the sigmoid lives in the training loop" section.
        pred = predicted_dice(logits)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        global_step += 1

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        n_samples += batch_size

    mean_loss = total_loss / n_samples if n_samples else float("nan")
    return mean_loss, global_step


def evaluate_heldout(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Scores `model` on a held-out `QCPairsDataset`, with no gradient.

    Args:
        model: `SegQC` (or any module with the same `forward` contract).
        loader: Yields `(x, target)` batches from the held-out
            `QCPairsDataset`.
        device: Where `model` lives.

    Returns:
        `{"mae": ..., "spearman": ...}`. `spearman` -- the headline metric;
        the master plan's Gate C bar is `rho > 0.7` in distribution -- is
        `NaN` when fewer than 2 samples were scored (`scipy.stats.spearmanr`
        is undefined below that).
    """
    model.eval()
    all_preds: list[Tensor] = []
    all_targets: list[Tensor] = []
    with torch.no_grad():
        for x, target in loader:
            x = x.to(device)
            logits = model(x)
            pred = predicted_dice(logits)
            all_preds.append(pred.detach().cpu())
            all_targets.append(target)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mae = float((preds - targets).abs().mean().item())

    if preds.numel() >= 2:
        rho, _ = spearmanr(preds.numpy(), targets.numpy())
        rho = float(rho)
    else:
        rho = float("nan")

    return {"mae": mae, "spearman": rho}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _print_summary(history: pd.DataFrame, out_dir: Path) -> None:
    """Prints (not logs -- see `scripts/evaluate.py`'s identical convention) a compact summary."""
    lines = [
        "=" * 70,
        f"QC training summary -- {len(history)} epoch(s), out_dir={out_dir}",
        "=" * 70,
    ]
    if not history.empty:
        last = history.iloc[-1]
        lines.append(
            f"  final (epoch {int(last['epoch'])}): train_loss={last['train_loss']:.4f} "
            f"heldout_mae={last['mae']:.4f} heldout_spearman={last['spearman']:.4f}"
        )
        if history["spearman"].notna().any():
            best = history.loc[history["spearman"].idxmax()]
            lines.append(
                f"  best heldout spearman: {best['spearman']:.4f} at epoch {int(best['epoch'])}"
            )
    # print only, not logger.info as well -- matches scripts/evaluate.py's
    # _log_and_print_summary and scripts/calibrate.py's _print_summary.
    print("\n".join(lines))


def run_training(cfg: DictConfig) -> dict[str, Any]:
    """Orchestrates the full QC training run: data, model, loop, checkpoint, report.

    Supports full resume: if `<analysis.qc.out_dir>/last.pt` already exists,
    training picks back up from the epoch after the one it saved, with
    model/optimizer/RNG state restored (see
    `neurovision.training.checkpoint`).

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A dict mapping a short name to the `Path` each output was written
        to, so tests can assert on what was produced.
    """
    qc_cfg = cfg.analysis.qc
    train_eval_dir, train_prep_dir, heldout_eval_dir, heldout_prep_dir = resolve_dirs(cfg)
    out_dir = ensure_dir(qc_cfg.out_dir)
    device = get_device(cfg)

    train_dataset = QCPairsDataset(cfg, train_eval_dir, train_prep_dir)
    heldout_dataset = QCPairsDataset(cfg, heldout_eval_dir, heldout_prep_dir)

    # CaseGroupedSampler, not shuffle=True and not shuffle=False -- see the
    # module docstring's "case-grouped shuffling" section. shuffle=False is
    # required here because `sampler` is given: DataLoader raises if both
    # shuffle=True and a sampler are passed together.
    train_sampler = CaseGroupedSampler(
        train_dataset.num_cases, train_dataset.per_case, int(cfg.seed)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(qc_cfg.batch_size),
        shuffle=False,
        sampler=train_sampler,
        num_workers=int(qc_cfg.num_workers),
    )
    # The held-out split is never trained on, so it has no gradient-
    # correlation problem to solve -- plain sequential order is fine and
    # still gets the full benefit of QCPairsDataset's single-case cache.
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=int(qc_cfg.batch_size),
        shuffle=False,
        num_workers=int(qc_cfg.num_workers),
    )

    model = build_segqc(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(qc_cfg.lr))

    start_epoch = 0
    global_step = 0
    best_spearman = float("-inf")
    resume_path = find_resume_checkpoint(out_dir)
    if resume_path is not None:
        resume_state = load_checkpoint(resume_path, model, optimizer, map_location=str(device))
        start_epoch = resume_state.start_epoch
        global_step = resume_state.global_step
        best_spearman = resume_state.best_metric
        logger.info("run_training: resuming from %s at epoch %d.", resume_path, start_epoch)

    total_epochs = int(qc_cfg.epochs)
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, total_epochs):
        # epoch here is ALREADY resume-safe: on a fresh run it starts at 0,
        # on a resumed run it starts at the RESTORED epoch (start_epoch,
        # from the checkpoint) -- never a counter that restarts at 0. See
        # CaseGroupedSampler.set_epoch's docstring for why that matters.
        train_sampler.set_epoch(epoch)
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, global_step
        )
        metrics = evaluate_heldout(model, heldout_loader, device)
        logger.info(
            "run_training: epoch %d/%d -- train_loss=%.4f heldout_mae=%.4f "
            "heldout_spearman=%.4f",
            epoch,
            total_epochs - 1,
            train_loss,
            metrics["mae"],
            metrics["spearman"],
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        is_best = (not math.isnan(metrics["spearman"])) and metrics["spearman"] > best_spearman
        if is_best:
            best_spearman = metrics["spearman"]

        save_checkpoint(
            out_dir,
            model,
            optimizer,
            epoch,
            global_step,
            best_metric=best_spearman,
            best_metric_name="spearman",
            best_metric_mode="max",
            cfg=cfg,
            is_best=is_best,
        )

    history_df = pd.DataFrame(history)
    history_path = out_dir / "history.csv"
    history_df.to_csv(history_path, index=False)

    config_path = out_dir / "qc_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    _print_summary(history_df, out_dir)

    return {
        "history_csv": history_path,
        "qc_config_yaml": config_path,
        "checkpoint_dir": out_dir,
    }


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs QC training, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
            Must include `model=segqc` -- see this module's docstring.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_training(cfg)


if __name__ == "__main__":
    main()
