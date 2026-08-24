"""Generates (degraded mask, true Dice) training pairs for the segmentation QC model.

Phase C of the master plan trains a *second*, independently-trained network
that looks at (image, predicted mask, uncertainty map) and predicts the Dice
score that mask would have scored -- with no ground truth available at
inference time. That QC model needs training examples that span a wide range
of quality, not just the ~0.87 Dice the real segmentation model happens to
produce. This module is the generator: it takes ONE case's predicted mask,
damages it in several realistic ways, and scores each damaged copy against
the ground-truth label to make a (degraded mask, true Dice) pair.

**The binding design principle (master plan section 2, principle 3):
downstream models train on PREDICTED masks, never ground-truth masks.**
Training on damaged ground truth and then deploying on the segmentation
model's own output is train/serve skew: degraded ground truth has smooth,
anatomically plausible boundaries, while a real prediction has the specific
ragged failure modes of *this* network (spurious small lesions, an eroded
enhancing-tumor core, a registration-shifted boundary). A QC model trained on
the wrong kind of damage would learn to recognise the wrong kind of failure.

So `degrade_mask`, the function that does the damaging, structurally CANNOT
see the label -- it has no parameter for it at all (there is a test that
introspects the signature to guarantee this stays true). The label is used
for exactly one thing, in `generate_pairs`: computing the true Dice that
becomes the regression target, via `neurovision.metrics.segmentation.dice_score`
so it inherits this project's `ignore_empty=False` empty-region convention
rather than a second, silently-disagreeing definition of Dice.

**The intended ceiling this implies:** every degradation here (`erode`,
`dilate`, `drop_component`, `shift`, `speckle`) is chosen to make a mask
WORSE, or, for `identity`, leave it unchanged -- never to systematically
improve it. For the WHOLE-MASK, ALL-THREE-REGIONS case (`region_index=None`,
which is what `generate_pairs` uses for its non-`per_region` pairs) this
holds up empirically: shrinking, growing, dropping lesions, shifting or
adding noise to an already-decent prediction essentially never accidentally
produces a better match to the true boundary than the prediction already
was. So in practice the achievable Dice range for a case is
`[0, dice(pred_mask, label)]`, not `[0, 1]`, and a QC model trained on this
data will almost never see an example better than the segmentation model's
own output for that lesion -- worth remembering when the regression targets
top out below 1.0.

**This is NOT a mathematical guarantee, and the `per_region` pairs can
violate it.** `dice_score` is bounded above by 1.0 regardless of geometry, so
nothing here can exceed a PERFECT match -- but it CAN exceed the
*undegraded prediction's own* Dice for a single region, if that region
happened to be imperfectly registered and the degradation accidentally
improves the alignment. Confirmed empirically: `shift` applied to a single
region can translate it exactly onto the true boundary if the prediction's
own misalignment happens to be an integer number of voxels along one axis --
scoring a HIGHER Dice for that one region than the undegraded prediction did.
`dilate` on a single region that was a clean, centered under-segmentation of
the true label can do the same by growing to fill the gap. Both are edge
cases of otherwise-designed-to-be-worsening operations landing, by
coincidence of geometry, exactly on a better alignment; they are not treated
specially here, both because a real prediction's failure modes are not this
kind of clean, correctable offset, and because a QC model seeing an
occasional "this small edit made it slightly better" example alongside a
huge majority of worsening ones is a realistic distribution, not a bug to
suppress.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

from neurovision.data.transforms import REGION_NAMES
from neurovision.inference.postprocess import enforce_nesting
from neurovision.metrics.segmentation import classes_to_regions, dice_score

__all__ = [
    "DEGRADATION_KINDS",
    "DegradationSpec",
    "QCPair",
    "degrade_mask",
    "generate_pairs",
    "generate_one_pair",
    "DEFAULT_SPECS",
]

logger = logging.getLogger(__name__)

# The six ways this module knows how to damage a mask. Each mimics a failure
# mode this project's models are known to actually make (see the per-function
# docstrings below for the specific evidence), rather than being an arbitrary
# choice of image-processing operations.
DEGRADATION_KINDS: tuple[str, ...] = (
    "identity",
    "erode",
    "dilate",
    "drop_component",
    "shift",
    "speckle",
)


@dataclass(frozen=True)
class DegradationSpec:
    """One degradation to apply: which kind, and how strong.

    Attributes:
        kind: One of `DEGRADATION_KINDS`.
        magnitude: Meaning depends on `kind` -- iterations for
            `erode`/`dilate`, a fraction in `(0, 1]` of components for
            `drop_component`, voxels for `shift`, a fraction of the mask's
            own voxel count for `speckle`. Unused (must be `0.0`) for
            `identity`.
    """

    kind: str
    magnitude: float

    def __post_init__(self) -> None:
        """Validates `kind` and `magnitude` against the meaning documented above.

        Raises:
            ValueError: If `kind` is not in `DEGRADATION_KINDS`, if
                `magnitude` is not finite, or if `magnitude` is out of the
                range that `kind` requires (see class docstring).
        """
        if self.kind not in DEGRADATION_KINDS:
            raise ValueError(
                f"DegradationSpec.kind must be one of {DEGRADATION_KINDS}, got {self.kind!r}."
            )
        if not math.isfinite(self.magnitude):
            raise ValueError(f"DegradationSpec.magnitude must be finite, got {self.magnitude!r}.")

        if self.kind == "identity":
            # identity ignores magnitude entirely; requiring exactly 0.0 here
            # means a caller who accidentally sets a nonzero magnitude on an
            # "identity" spec finds out immediately, instead of the value
            # being silently swallowed inside degrade_mask.
            if self.magnitude != 0.0:
                raise ValueError(
                    f"DegradationSpec.kind='identity' does not use magnitude; got "
                    f"{self.magnitude!r}, must be 0.0."
                )
        elif self.kind == "drop_component":
            if not (0.0 < self.magnitude <= 1.0):
                raise ValueError(
                    "DegradationSpec.kind='drop_component' requires magnitude in (0, 1], got "
                    f"{self.magnitude!r}."
                )
        else:  # erode, dilate, shift, speckle
            if self.magnitude <= 0.0:
                raise ValueError(
                    f"DegradationSpec.kind={self.kind!r} requires magnitude > 0, got "
                    f"{self.magnitude!r}."
                )


@dataclass(frozen=True)
class QCPair:
    """One training example for the QC model: a damaged mask and its true Dice.

    Attributes:
        mask: The degraded mask, `(3, D, H, W)` `uint8`, channel order
            `(ET, TC, WT)`.
        dice: Per-region Dice of `mask` against the case's ground-truth
            label, one float per entry of `REGION_NAMES`, computed by
            `neurovision.metrics.segmentation.dice_score`.
        spec: The `DegradationSpec` that produced `mask`.
        region_index: Which region channel was degraded (`0`=ET, `1`=TC,
            `2`=WT), or `None` if all three were degraded together.
    """

    mask: np.ndarray
    dice: tuple[float, ...]
    spec: DegradationSpec
    region_index: int | None


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Normalizes a `(3, D, H, W)` mask (uint8/bool/float) to binary `uint8`.

    Args:
        mask: Region mask, any dtype, values interpreted as foreground if
            `> 0.5`.

    Returns:
        `uint8` array of 0/1 values, same shape as `mask`.

    Raises:
        ValueError: If `mask` is not 4-D with exactly 3 leading channels.
    """
    arr = np.asarray(mask)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(f"Expected a (3, D, H, W) region mask, got shape {arr.shape}.")
    return (arr > 0.5).astype(np.uint8)


def _round_iterations(magnitude: float) -> int:
    """Rounds a magnitude to an iteration count of at least 1."""
    return max(1, round(magnitude))


def _erode_channel(channel: np.ndarray, magnitude: float) -> np.ndarray:
    """Binary erosion: simulates under-segmentation (the model shrinks a lesion).

    A model unsure about a lesion's true extent tends to retreat toward its
    most confident, innermost voxels -- erosion mimics that shrinkage.
    `magnitude` iterations, rounded, at least 1.
    """
    iterations = _round_iterations(magnitude)
    eroded = ndimage.binary_erosion(channel.astype(bool), iterations=iterations)
    return eroded.astype(np.uint8)


def _dilate_channel(channel: np.ndarray, magnitude: float) -> np.ndarray:
    """Binary dilation: simulates over-segmentation (the model over-calls a lesion).

    The mirror image of erosion -- a model that over-extends a confident core
    into surrounding, actually-healthy tissue. `magnitude` iterations,
    rounded, at least 1.
    """
    iterations = _round_iterations(magnitude)
    dilated = ndimage.binary_dilation(channel.astype(bool), iterations=iterations)
    return dilated.astype(np.uint8)


def _drop_components(
    channel: np.ndarray, magnitude: float, generator: np.random.Generator
) -> np.ndarray:
    """Removes a fraction of a channel's connected components.

    Simulates a spurious/missed whole lesion, not a partial one -- this
    project's models are already measured to emit 0.32-0.98 spurious lesions
    per case (note 41), the exact failure mode lesion-wise Dice exists to
    catch. `magnitude` is the fraction of components to remove, rounded to a
    count of at least 1 (if any components exist). Components are chosen with
    `generator`, and the single LARGEST component is protected unless the
    fraction leaves no other choice (e.g. `magnitude=1.0`, or a mask that is
    only one component to begin with) -- dropping only small/medium lesions
    most of the time is the realistic failure; dropping the dominant lesion
    is the rarer, more catastrophic one.
    """
    labeled, n_components = ndimage.label(channel)
    if n_components == 0:
        return channel.copy()  # nothing to drop

    sizes = ndimage.sum(channel, labeled, index=np.arange(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    other_labels = [lbl for lbl in range(1, n_components + 1) if lbl != largest_label]

    num_to_drop = max(1, round(magnitude * n_components))
    num_to_drop = min(num_to_drop, n_components)

    if num_to_drop <= len(other_labels):
        chosen = generator.choice(
            np.array(other_labels, dtype=np.int64), size=num_to_drop, replace=False
        )
    else:
        # Not enough non-largest components to satisfy the requested count --
        # the fraction forces dropping the largest one too.
        chosen = np.array(other_labels + [largest_label], dtype=np.int64)

    out = channel.copy()
    for lbl in chosen:
        out[labeled == lbl] = 0
    return out


def _shift_channel(
    channel: np.ndarray, magnitude: float, generator: np.random.Generator
) -> np.ndarray:
    """Translates a channel along one randomly-chosen axis.

    Simulates a registration error: the whole predicted lesion sits at a
    consistent offset from where it should be, rather than having a wrong
    shape. `magnitude` voxels, rounded to at least 1; axis and sign chosen
    with `generator`. Uses nearest-neighbour resampling (`order=0`) so the
    output stays exactly binary, and `mode="constant", cval=0` so voxels
    shifted out of the volume are dropped rather than wrapping around.
    """
    amount = _round_iterations(magnitude)
    axis = int(generator.integers(0, 3))
    sign = 1 if int(generator.integers(0, 2)) == 0 else -1
    shift_vec = [0, 0, 0]
    shift_vec[axis] = sign * amount
    shifted = ndimage.shift(
        channel.astype(np.float32), shift=shift_vec, order=0, mode="constant", cval=0.0
    )
    return (shifted > 0.5).astype(np.uint8)


def _speckle_channel(
    channel: np.ndarray, magnitude: float, generator: np.random.Generator
) -> np.ndarray:
    """Adds isolated false-positive voxels outside the current mask.

    Simulates the speckle noise `postprocess_logits`'s `min_component_size`
    filter normally removes -- this generates training data for a QC model
    that has to judge a mask WITHOUT that filter having necessarily run, or
    with speckle the filter's `min_size` threshold happened to miss.
    `magnitude` is a fraction of the channel's OWN current voxel count (so an
    empty channel legitimately gets no speckle added -- there is nothing to
    take a fraction of; this is a real limitation of "fraction of own voxel
    count" and is left as-is rather than special-cased). New voxel locations
    are drawn from the background with `generator`, without replacement.
    """
    current_count = int(channel.sum())
    num_new = round(magnitude * current_count)
    out = channel.copy()
    if num_new <= 0:
        return out

    background = np.argwhere(channel == 0)
    if background.shape[0] == 0:
        return out  # mask already fills the whole volume

    num_new = min(num_new, background.shape[0])
    chosen = generator.choice(background.shape[0], size=num_new, replace=False)
    coords = background[chosen]
    out[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return out


def _apply_single_channel(
    channel: np.ndarray, spec: DegradationSpec, generator: np.random.Generator
) -> np.ndarray:
    """Dispatches one channel's worth of damage to the kind-specific helper."""
    if spec.kind == "erode":
        return _erode_channel(channel, spec.magnitude)
    if spec.kind == "dilate":
        return _dilate_channel(channel, spec.magnitude)
    if spec.kind == "drop_component":
        return _drop_components(channel, spec.magnitude, generator)
    if spec.kind == "shift":
        return _shift_channel(channel, spec.magnitude, generator)
    if spec.kind == "speckle":
        return _speckle_channel(channel, spec.magnitude, generator)
    # Unreachable: DegradationSpec.__post_init__ already restricts kind to
    # DEGRADATION_KINDS, and "identity" is handled before this is called.
    raise AssertionError(f"unhandled degradation kind {spec.kind!r}")


def degrade_mask(
    mask: np.ndarray,
    spec: DegradationSpec,
    *,
    generator: np.random.Generator,
    region_index: int | None = None,
) -> np.ndarray:
    """Damages a predicted mask according to `spec`. Takes NO label argument.

    This is the structural guarantee behind principle 3: the function that
    manufactures degraded training inputs cannot see ground truth, because
    it has no parameter to pass it through.

    Args:
        mask: The model's PREDICTED region mask (never ground truth),
            `(3, D, H, W)`, channel order `(ET, TC, WT)`. Accepts
            uint8/bool/float; values are normalised to binary internally.
        spec: Which degradation to apply and how strong.
        generator: Source of randomness for the stochastic kinds
            (`drop_component`, `shift`, `speckle`). Never a module-level
            `np.random` call, so the same generator state reproduces the
            same output.
        region_index: `0`/`1`/`2` to degrade only ET/TC/WT; `None` (default)
            to degrade all three regions.

            When `None`, each channel is damaged independently and then
            `neurovision.inference.postprocess.enforce_nesting` is applied,
            so the result always satisfies `ET subset-of TC subset-of WT` --
            this project's existing nesting-repair logic is reused rather
            than reimplemented here.

            When a single region is given, ONLY that channel changes and
            nesting is deliberately NOT repaired: a real predicted mask can
            legitimately violate nesting (see
            `neurovision.inference.postprocess.remove_small_components`'s
            own docstring for why), and the QC model this data trains must
            see that violation as a signal of a bad mask, not have it
            silently cleaned up before training even starts.

    Returns:
        `uint8` array, shape `(3, D, H, W)`, values in `{0, 1}`.

    Raises:
        ValueError: If `mask` is not `(3, D, H, W)`, or `region_index` is
            not `None`/`0`/`1`/`2`.
    """
    regions = _binarize_mask(mask)
    if region_index is not None and region_index not in (0, 1, 2):
        raise ValueError(f"region_index must be None, 0, 1 or 2, got {region_index!r}.")

    if spec.kind == "identity":
        return regions.copy()

    if region_index is not None:
        out = regions.copy()
        out[region_index] = _apply_single_channel(regions[region_index], spec, generator)
        return out

    degraded = np.stack(
        [_apply_single_channel(regions[c], spec, generator) for c in range(3)], axis=0
    )
    nested = enforce_nesting(torch.from_numpy(degraded.astype(np.float32)))
    return (nested.numpy() > 0.5).astype(np.uint8)


def _expand_label(label: np.ndarray) -> np.ndarray:
    """Turns a class map OR an already-expanded region stack into a binary region stack.

    Args:
        label: Either an integer class map `(D, H, W)` with values in
            `{0, 1, 2, 3}`, or an already-expanded region mask
            `(3, D, H, W)`.

    Returns:
        `uint8` array, shape `(3, D, H, W)`, values in `{0, 1}`.

    Raises:
        ValueError: If `label` is neither a 3-D class map nor a `(3, D, H, W)`
            region stack.
    """
    arr = np.asarray(label)
    if arr.ndim == 3:
        # classes_to_regions is the single source of truth for the class ->
        # region algebra (it mirrors the training transform exactly) --
        # re-deriving ET/TC/WT from raw class values here would risk a
        # second, silently-disagreeing definition.
        regions_t = classes_to_regions(torch.from_numpy(arr.astype(np.int64)))
        return (regions_t[0].numpy() > 0.5).astype(np.uint8)
    if arr.ndim == 4:
        return _binarize_mask(arr)
    raise ValueError(
        f"label must be a (D, H, W) class map or a (3, D, H, W) region stack, got shape "
        f"{arr.shape}."
    )


def _dice_tuple(pred_regions: np.ndarray, label_regions: np.ndarray) -> tuple[float, ...]:
    """Computes per-region Dice via the project's `dice_score`, never a hand-rolled formula."""
    pred_t = torch.from_numpy(pred_regions.astype(np.float32)).unsqueeze(0)  # (1, 3, D, H, W)
    label_t = torch.from_numpy(label_regions.astype(np.float32)).unsqueeze(0)
    dice = dice_score(pred_t, label_t, ignore_empty=False)[0]  # (3,)
    return tuple(float(v) for v in dice.tolist())


def generate_pairs(
    pred_mask: np.ndarray,
    label: np.ndarray,
    *,
    generator: np.random.Generator,
    specs: Sequence[DegradationSpec] | None = None,
    per_region: bool = True,
) -> list[QCPair]:
    """Builds a set of (degraded mask, true Dice) training pairs for one case.

    For every spec in `specs`, degrades `pred_mask` (all three regions
    together) and scores the result against `label`. If `per_region` is
    True, additionally produces one pair per (spec, region) with ONLY that
    region degraded -- this is what teaches the QC model to attribute error
    to a specific region rather than only judging the mask as a whole.

    Args:
        pred_mask: The model's PREDICTED region mask, `(3, D, H, W)`,
            channel order `(ET, TC, WT)`.
        label: Ground truth, either a `(D, H, W)` integer class map or an
            already-expanded `(3, D, H, W)` region mask. Used ONLY to
            compute the Dice regression target -- never passed to
            `degrade_mask`.
        generator: Source of randomness, forwarded to every stochastic
            degradation.
        specs: Degradations to apply. Defaults to `DEFAULT_SPECS`.
        per_region: If True (default), also generate single-region-degraded
            pairs for each spec.

    Returns:
        A list of `QCPair`. Length is `len(specs)` if `per_region` is False,
        else `len(specs) * 4` (one all-regions pair plus one per-region pair
        for each spec).
    """
    resolved_specs = specs if specs is not None else DEFAULT_SPECS
    pred_regions = _binarize_mask(pred_mask)
    label_regions = _expand_label(label)

    pairs: list[QCPair] = []
    for spec in resolved_specs:
        degraded = degrade_mask(pred_regions, spec, generator=generator, region_index=None)
        pairs.append(
            QCPair(
                mask=degraded,
                dice=_dice_tuple(degraded, label_regions),
                spec=spec,
                region_index=None,
            )
        )
        if per_region:
            for region_index in range(len(REGION_NAMES)):
                degraded_r = degrade_mask(
                    pred_regions, spec, generator=generator, region_index=region_index
                )
                pairs.append(
                    QCPair(
                        mask=degraded_r,
                        dice=_dice_tuple(degraded_r, label_regions),
                        spec=spec,
                        region_index=region_index,
                    )
                )

    logger.info(
        "generate_pairs: built %d QC pairs from %d specs (per_region=%s).",
        len(pairs),
        len(resolved_specs),
        per_region,
    )
    return pairs


def generate_one_pair(
    pred_mask: np.ndarray,
    label: np.ndarray,
    spec: DegradationSpec,
    region_index: int,
    *,
    generator: np.random.Generator,
) -> QCPair:
    """Builds exactly the ONE per-region pair `generate_pairs` would build, without
    scoring the three pairs a caller who only wants one region never reads.

    Added for `scripts/train_qc.py`'s `QCPairsDataset`, whose `__getitem__`
    packs and trains on exactly one region at a time. Calling
    `generate_pairs(pred_mask, label, generator=generator, specs=[spec],
    per_region=True)` there and keeping only the pair matching one
    `region_index` computes -- and immediately discards -- the whole-mask
    pair and every OTHER region's pair, on the hot path of a CPU training
    loop. This function returns the SAME pair `generate_pairs` would, at
    the same generator state, without that waste.

    **Why this cannot simply call `degrade_mask(pred_mask, spec,
    generator=generator, region_index=region_index)` directly, skipping the
    other three entirely:** `generate_pairs`, for one spec with
    `per_region=True`, degrades the WHOLE mask first (its `region_index=
    None` pair -- always built, regardless of `per_region`), then region 0,
    then region 1, then region 2, ALL against the SAME `generator`,
    consuming its random draws sequentially in that order. For a
    stochastic `spec.kind` (`drop_component`, `shift`, `speckle` --
    `degrade_mask`'s own docstring documents which kinds touch `generator`
    at all; `identity`/`erode`/`dilate` never do), the exact result at
    `region_index` therefore depends on how many draws the whole-mask
    degrade and every earlier region already consumed. Skipping those calls
    would leave `generator` in a DIFFERENT state than `generate_pairs`
    would have reached, and the two would silently stop agreeing --
    exactly the kind of drift `docs/lessons.md` exists to catalogue. So
    this function performs the SAME `degrade_mask` calls, in the SAME
    order, for the SAME reason (consuming `generator` identically): only
    `_dice_tuple` -- the Dice computation, never the degradation itself --
    is skipped for the discarded whole-mask pair and every region before
    `region_index`.

    Args:
        pred_mask: The model's PREDICTED region mask, `(3, D, H, W)`,
            channel order `(ET, TC, WT)`. Never ground truth -- see this
            module's docstring for the binding principle.
        label: Ground truth, either a `(D, H, W)` integer class map or an
            already-expanded `(3, D, H, W)` region mask. Used ONLY to
            compute the returned pair's Dice.
        spec: The single degradation to apply.
        region_index: Which region channel to degrade -- `0`, `1`, or `2`
            for ET/TC/WT. Unlike `degrade_mask`'s own `region_index`,
            `None` is not accepted: `generate_pairs` always scores the
            whole-mask pair regardless of `per_region`, so there is no
            wasted work this function would save for that case -- call
            `degrade_mask` directly instead.
        generator: Source of randomness, in the SAME state a fresh call to
            `generate_pairs(..., specs=[spec], per_region=True)` would have
            started from -- e.g. a generator just constructed from a fixed
            seed. The result matches `generate_pairs`'s corresponding pair
            only if this precondition holds; it is the caller's
            responsibility, exactly as it already is for `generate_pairs`.

    Returns:
        The `QCPair` for `region_index`, identical to the matching entry of
        `generate_pairs(pred_mask, label, generator=<same-state generator>,
        specs=[spec], per_region=True)`.

    Raises:
        ValueError: `region_index` is not `0`, `1`, or `2`.
    """
    if region_index not in (0, 1, 2):
        raise ValueError(
            f"generate_one_pair: region_index must be 0, 1 or 2, got {region_index!r}. "
            "(None is not accepted -- see this function's docstring.)"
        )

    pred_regions = _binarize_mask(pred_mask)
    label_regions = _expand_label(label)

    # Whole-mask degrade, then every region strictly BEFORE region_index --
    # run for generator continuity ONLY. Their outputs and Dice are never
    # computed; see the docstring for why they cannot simply be skipped.
    degrade_mask(pred_regions, spec, generator=generator, region_index=None)
    for earlier_region in range(region_index):
        degrade_mask(pred_regions, spec, generator=generator, region_index=earlier_region)

    degraded = degrade_mask(pred_regions, spec, generator=generator, region_index=region_index)
    return QCPair(
        mask=degraded,
        dice=_dice_tuple(degraded, label_regions),
        spec=spec,
        region_index=region_index,
    )


# Magnitudes chosen to span mild to severe damage for each kind, at roughly
# a doubling progression (1, 2, 4, 8) so the achievable Dice range (see the
# module docstring's ceiling note) gets covered without needing many more
# specs than this:
#   - identity: exactly one spec. Not padding -- without an undamaged
#     example, the QC model would never see a genuinely good mask during
#     training and would systematically under-predict Dice right at the
#     operating point that matters most (the model's actual output quality).
#   - erode / dilate: 1/2/4/8 iterations. 1 iteration barely nibbles the
#     boundary; 8 is enough to delete a small lesion (e.g. most ET cores)
#     entirely, giving Dice all the way down toward 0.
#   - drop_component: 0.25/0.5/0.75/1.0 (a fraction of components). 0.25
#     drops a single small stray lesion; 1.0 drops everything, the
#     worst-case "model found nothing" pair.
#   - shift: 1/2/4/8 voxels. 1 voxel is sub-clinical registration jitter;
#     8 voxels is a large, clearly-wrong offset for a typical BraTS lesion.
#   - speckle: 0.1/0.25/0.5/1.0 (fraction of the mask's own voxel count
#     added as noise). 1.0 doubles the foreground voxel count with
#     scattered false positives, which is a lot of noise but still far
#     short of BraTS's largest false-positive outliers.
DEFAULT_SPECS: tuple[DegradationSpec, ...] = (
    DegradationSpec("identity", 0.0),
    DegradationSpec("erode", 1.0),
    DegradationSpec("erode", 2.0),
    DegradationSpec("erode", 4.0),
    DegradationSpec("erode", 8.0),
    DegradationSpec("dilate", 1.0),
    DegradationSpec("dilate", 2.0),
    DegradationSpec("dilate", 4.0),
    DegradationSpec("dilate", 8.0),
    DegradationSpec("drop_component", 0.25),
    DegradationSpec("drop_component", 0.5),
    DegradationSpec("drop_component", 0.75),
    DegradationSpec("drop_component", 1.0),
    DegradationSpec("shift", 1.0),
    DegradationSpec("shift", 2.0),
    DegradationSpec("shift", 4.0),
    DegradationSpec("shift", 8.0),
    DegradationSpec("speckle", 0.1),
    DegradationSpec("speckle", 0.25),
    DegradationSpec("speckle", 0.5),
    DegradationSpec("speckle", 1.0),
)
