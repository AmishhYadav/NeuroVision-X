"""Faithfulness metrics for attribution heatmaps: deletion, insertion, and localization.

A Grad-CAM / Integrated-Gradients / attention-rollout heatmap is a claim about which voxels
the model relied on. That claim is unfalsifiable if all anyone ever does is look at it -- a
heatmap that happens to glow over the tumor could be doing so for reasons that have nothing to
do with the model's actual reliance on those voxels (tumors are salient, high-contrast regions,
and plenty of "explanations" are really just edge detectors). This module tests the claim
directly, by PERTURBING the input according to the heatmap's own ranking and watching whether
the model's behaviour changes the way the heatmap predicts it should.

Two families of test:

- **Deletion / insertion curves** (`deletion_curve`, `insertion_curve`). Rank voxels by
  |attribution|, then either progressively DELETE the most-attributed voxels (replacing them
  with an uninformative fill and watching the prediction degrade) or progressively INSERT them
  into an otherwise-blank volume (watching the prediction recover). A faithful attribution
  degrades the prediction fast under deletion and recovers it fast under insertion; an
  attribution that is just noise does neither reliably.

- **Localization** (`pointing_game`, `attribution_mass_ratio`). Does the attribution actually
  sit on the tumor, compared against the null hypothesis of "the tumor is just a large fraction
  of the volume, so anything vaguely central wins"?

`compare_methods` assembles both families, plus an automatically-added `"random"` null row,
into one comparison table across however many attribution methods a caller supplies.

## Decision 1 -- primary score is Dice against the model's OWN prediction, not ground truth

Every curve function computes and returns BOTH: `dice_vs_prediction` (Dice between the
perturbed-input prediction and the ORIGINAL, unperturbed prediction) and
`dice_vs_ground_truth` (Dice between the perturbed-input prediction and the real label, if
supplied). `dice_vs_prediction` is the PRIMARY number and drives `auc_vs_prediction`, which is
what `compare_methods` reports. The reason: this module is testing attribution faithfulness,
not model accuracy. A model that segments a case badly would drag a ground-truth-referenced
curve down for reasons that have nothing to do with whether the heatmap identifies the voxels
the model actually used -- that confound has to be kept out of the headline number.
`dice_vs_ground_truth` is still computed and returned, because it is the more clinically
legible number ("does perturbing the important voxels start looking like a wrong diagnosis"),
and a reader building a figure should see it alongside the primary column. Document which
column is which wherever this module's output is used -- they measure different things.

## Decision 2 -- the 100% point is a known-degenerate point of THIS architecture

Deleting every voxel down to a constant fill value produces a spatially CONSTANT input volume.
This project's model is full of normalization layers (GroupNorm throughout the CNN
encoder/decoder, LayerNorm inside Swin -- see `neurovision.explainability.integrated_gradients`'s
top-of-file docstring for the measured numbers on the exact same architecture), and a spatially
constant input collapses every one of those layers' per-group standard deviation to exactly 0,
so their output is decided entirely by the epsilon floor in the denominator -- not by anything
resembling the model's normal, trained behaviour. Measured on the production model in that other
module: an all-zeros input recovers a completeness ratio of 0.002 against 0.993 for a noise
input. The 100% point of a deletion curve (and, by the same argument, the 0% point of an
insertion curve, which starts from the identical all-fill volume) sits on that same singularity.

Consequence: `deletion_curve` and `insertion_curve` default to `include_endpoint=False`, which
EXCLUDES the degenerate endpoint from the reported AUC only -- the point is still computed and
still present in the returned arrays, so it stays visible to anyone plotting the curve, it just
does not get to secretly dominate the summary statistic with an architectural artifact rather
than a faithfulness signal.

## Decision 3 -- `pointing_game` needs the null, or the number is meaningless

The target score both `deletion_curve`/`insertion_curve` and the callers of `pointing_game` are
built against is a SUM OF LOGITS OVER THE MODEL'S OWN PREDICTED TUMOR REGION. Every attribution
method computes a gradient/activation quantity that is already concentrated on that region
almost by construction -- Grad-CAM's activations, IG's attributions, and attention weights are
all downstream of a forward pass that produced that very prediction. So "the attribution's
argmax voxel lands inside the tumor 97% of the time" is not evidence of anything if the tumor
is, say, 15% of a cropped patch: a method that pointed at a uniformly random voxel would still
land inside the tumor 15% of the time. `pointing_game` therefore ALWAYS returns
`gt_volume_fraction` (the hit rate a uniformly random point would get) alongside `hit`, and
`ratio = hit / gt_volume_fraction` -- report `ratio`, never the bare hit rate. Same lesson as
the random-referral baseline in `neurovision.uncertainty.risk_coverage.random_curve`: a metric
without its null baseline invites exactly the wrong reading.

## Decision 4 -- `attribution_mass_ratio` is the recommended headline localization number

`pointing_game`'s hit/miss is a single argmax point and throws away almost everything a dense 3D
attribution map contains -- two attribution maps that differ everywhere except at one voxel
score identically. `attribution_mass_ratio` instead asks what FRACTION of the map's total
|attribution| mass falls inside the ground-truth region, and compares that fraction against
`gt_volume_fraction` the same way: `ratio = mass_inside / gt_volume_fraction`. A ratio of
EXACTLY 1.0 means the attribution is distributed exactly as a spatially uniform map would be --
i.e. no localization whatsoever, the tumor gets attribution mass in proportion to its size and
nothing more. This is the better statistic and should be presented as the headline localization
number in any figure or table; `pointing_game` is the familiar-but-weaker companion, useful
mainly because readers already know what a "pointing game" is.

## Decision 5 -- attention rollout is NOT target-specific, and the table must say so

`neurovision.explainability.gradcam.grad_cam` and
`neurovision.explainability.integrated_gradients.integrated_gradients` both explain a CHOSEN
region channel -- their target score is built from `logits[:, region_index]`. Swin's attention
weights carry no such dependency: `neurovision.explainability.attention_rollout.attention_rollout`
produces the identical map regardless of which region (ET/TC/WT) a caller is asking about,
because the attention itself was computed upstream of any region-specific head. Comparing
rollout head-to-head against Grad-CAM/IG on a region-specific faithfulness curve is therefore
apples-to-oranges, and rollout will likely score worse for a reason that has nothing to do with
"it is a worse explanation" -- it is answering a different, coarser question ("what did the
transformer attend to at all") rather than the one the curve is built to test ("what drove THIS
region's prediction"). `compare_methods` takes a `target_specific: Mapping[str, bool]` argument
(default `True` for every supplied method, `False` for the automatically-added `"random"` row --
see Decision 3, a random map does not depend on the target either) and always carries a
`target_specific` boolean column in its output table, so a figure caption can say so rather than
present the comparison as apples-to-apples.

## Decision 6 -- resolution is a confound in deletion curves, and the table says so too

A coarse attribution map (e.g. Grad-CAM taken at a stride-8 layer, upsampled to voxel
resolution) deletes large contiguous BLOBS of voxels at any given fraction; a voxel-resolution
map (IG, or a fine-layer Grad-CAM) deletes SCATTERED individual voxels. Scattered deletion can
disrupt a convolutional network more than blob deletion for reasons that have nothing to do with
whether either map correctly identified important voxels -- a well-known critique of
deletion-style faithfulness metrics in the literature. This module does not attempt to correct
for it (there is no agreed-upon correction), it only makes the confound visible: `compare_methods`
takes a `native_resolution: Mapping[str, str]` argument (free-text, e.g. `"voxel"` or
`"stride 8 (upsampled)"`, supplied per method by the caller) and carries it through as a column,
so a reader can weigh a deletion-AUC difference against the resolution difference that might be
producing it.

## Cost -- this is figure-generation tooling, not an evaluation-loop metric

Measured on the production `NeuroVisionX` (34,904,981 params, CPU, per CLAUDE.md): a forward
pass alone is 2.9s at 64^3 and 9.2s at 96^3 (forward+backward, which this module never needs, is
5.5s / 17.9s). One curve at `n_points=11` is roughly `11 * 2.9s ~= 32s` at 64^3, and a full
`compare_methods` call over three attribution methods plus the automatic `"random"` null runs 8
curves (4 methods x deletion + insertion) -- roughly 4-5 minutes at 64^3. This is a tool for
producing a handful of published figures on a handful of hand-picked cases, not something to run
across a 189-case evaluation split.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from neurovision.metrics.segmentation import binarize, dice_score

logger = logging.getLogger(__name__)

__all__ = [
    "FaithfulnessCurve",
    "rank_voxels_by_attribution",
    "make_fill",
    "deletion_curve",
    "insertion_curve",
    "pointing_game",
    "attribution_mass_ratio",
    "random_attribution_like",
    "compare_methods",
]

_VALID_FILLS: tuple[str, ...] = ("zero", "noise", "mean")


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of `y` over `x`, tolerant of the numpy trapz/trapezoid rename.

    Mirrors `neurovision.uncertainty.risk_coverage._trapz`: `numpy.trapz` was removed in favour
    of `numpy.trapezoid` in recent numpy releases, so this tries the new name first and falls
    back to the old one rather than hardcoding either.
    """
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def _to_batched_5d(image: Tensor, name: str) -> Tensor:
    """Normalizes an input volume to `(1, C, D, H, W)`, unsqueezing a missing batch axis.

    Args:
        image: `(1, C, D, H, W)` or `(C, D, H, W)`.
        name: Used in error messages.

    Returns:
        `(1, C, D, H, W)`.

    Raises:
        ValueError: `image` is not 4-D or 5-D, or has a batch size other than 1.
    """
    if image.ndim == 4:
        image = image.unsqueeze(0)
    elif image.ndim != 5:
        raise ValueError(f"{name} must be (1, C, D, H, W) or (C, D, H, W), got ndim={image.ndim}.")
    if image.shape[0] != 1:
        raise ValueError(
            f"{name} supports a single-volume batch only, got batch size {image.shape[0]}."
        )
    return image


def _reduce_to_3d(x: Tensor, name: str) -> Tensor:
    """Squeezes any number of leading size-1 axes down to a plain `(D, H, W)` tensor.

    Args:
        x: `(D, H, W)`, or that shape with any number of leading size-1 axes
            (e.g. `(1, 1, D, H, W)`).
        name: Used in error messages.

    Returns:
        `(D, H, W)`.

    Raises:
        ValueError: A leading axis has size != 1, or the tensor never reduces to 3-D.
    """
    t = torch.as_tensor(x).detach()
    while t.ndim > 3:
        if t.shape[0] != 1:
            raise ValueError(
                f"{name} has unexpected shape {tuple(x.shape)}; expected (D, H, W) or that "
                "shape with leading size-1 axes only."
            )
        t = t.squeeze(0)
    if t.ndim != 3:
        raise ValueError(f"{name} must reduce to (D, H, W); got shape {tuple(x.shape)}.")
    return t


def _extract_region_channel(
    tensor: Tensor, region_index: int, spatial_shape: tuple[int, int, int], name: str
) -> Tensor:
    """Pulls one region channel out of a `(B, C, D, H, W)`-style tensor as `(1, 1, D, H, W)`.

    A bare `(D, H, W)` input is accepted and taken to be an ALREADY-EXTRACTED single
    region mask, with `region_index` ignored for it. That case is unambiguous (the three
    layouts differ in ndim, so nothing has to be guessed from a channel count) and it is
    the shape a caller naturally reaches for -- a thresholded prediction channel, or a
    synthetic mask built with `torch.meshgrid`, is `(D, H, W)`. Rejecting it forced the
    caller to `unsqueeze` for no reason, which is exactly the kind of friction that turns
    into a mis-shaped tensor somewhere else.

    Args:
        tensor: `(1, C, D, H, W)`, `(C, D, H, W)`, or `(D, H, W)` (already one region).
        region_index: Channel to extract. Ignored for a `(D, H, W)` input.
        spatial_shape: The `(D, H, W)` shape `tensor` must match.
        name: Used in error messages.

    Returns:
        Float32 `(1, 1, D, H, W)`.

    Raises:
        ValueError: Bad ndim/batch size, `region_index` out of range, or spatial shape mismatch.
    """
    t = tensor
    if t.ndim == 3:
        # Already a single region mask; add the batch and channel axes it is missing.
        t = t.unsqueeze(0).unsqueeze(0)
        region_index = 0
    elif t.ndim == 4:
        t = t.unsqueeze(0)
    if t.ndim != 5:
        raise ValueError(
            f"{name} must be (1, C, D, H, W), (C, D, H, W), or (D, H, W) for an "
            f"already-extracted single region, got ndim={tensor.ndim}."
        )
    if t.shape[0] != 1:
        raise ValueError(
            f"{name} supports a single-volume batch only, got batch size {t.shape[0]}."
        )
    if not (0 <= region_index < t.shape[1]):
        raise ValueError(
            f"region_index ({region_index}) is out of range for {name} with {t.shape[1]} channels."
        )
    if tuple(t.shape[2:]) != tuple(spatial_shape):
        raise ValueError(
            f"{name} spatial shape {tuple(t.shape[2:])} does not match the expected spatial "
            f"shape {tuple(spatial_shape)}."
        )
    return t[:, region_index : region_index + 1].to(dtype=torch.float32)


def rank_voxels_by_attribution(attribution: Tensor, mask: Tensor | None = None) -> Tensor:
    """Ranks every spatial voxel by attribution MAGNITUDE, descending.

    Attribution is ranked by `.abs()`, not by signed value: a voxel with a large NEGATIVE
    attribution (strong evidence AGAINST the target region) is just as much "a voxel the model
    relied on" as one with a large positive attribution, and treating it as unimportant (as a
    signed, ascending-from-most-positive ranking would) throws away exactly the evidence a
    faithfulness curve needs to perturb first.

    Ties are broken by a STABLE sort, which preserves each tied voxel's original (flattened
    `(D, H, W)`) order. This matters concretely: a ReLU'd Grad-CAM map has huge ties at exactly
    0 (everywhere the CAM found no positive evidence), and an unstable sort would make the
    resulting deletion/insertion order -- and therefore the whole curve -- depend on
    `torch.argsort`'s internal implementation rather than on the attribution itself.

    Args:
        attribution: `(1, 1, D, H, W)` or `(D, H, W)`.
        mask: Optional boolean-like tensor, same spatial shape convention as `attribution`
            (leading size-1 axes squeezed away). When given, only voxels where `mask` is True
            are included in the returned ranking -- masked-OUT voxels are dropped entirely, not
            merely ranked last. `None` (the default) ranks every voxel.

    Returns:
        A 1-D `LongTensor` of flat indices into the flattened `(D, H, W)` volume, sorted by
        `|attribution|` descending. Length `D * H * W` when `mask is None`, else the number of
        True entries in `mask`.

    Raises:
        ValueError: `attribution` (or `mask`) does not reduce to `(D, H, W)`.
    """
    attr = _reduce_to_3d(attribution, "rank_voxels_by_attribution attribution")
    flat = attr.abs().reshape(-1)
    # stable=True: see the ties paragraph above -- reproducibility across torch versions and
    # across repeated calls depends on this, not just on the values themselves.
    order = torch.argsort(flat, descending=True, stable=True)

    if mask is not None:
        mask_3d = _reduce_to_3d(mask, "rank_voxels_by_attribution mask")
        if tuple(mask_3d.shape) != tuple(attr.shape):
            raise ValueError(
                f"rank_voxels_by_attribution: mask shape {tuple(mask_3d.shape)} does not match "
                f"attribution shape {tuple(attr.shape)}."
            )
        mask_flat = mask_3d.reshape(-1).to(torch.bool)
        order = order[mask_flat[order]]

    return order


def make_fill(image: Tensor, fill: str, generator: torch.Generator | None = None) -> Tensor:
    """Builds the replacement volume `deletion_curve`/`insertion_curve` perturb voxels toward.

    `"zero"` is the standard, and default, choice here for a reason specific to this project's
    preprocessing: `neurovision.data.preprocessing.normalize_nonzero` z-scores each modality
    over its own nonzero (brain-tissue) region, so a voxel value of exactly 0 is, by
    construction, the MEAN intensity of normal brain tissue -- not the black, out-of-distribution
    value it would be on a natural image. This makes `"zero"` a meaningful "remove the
    information here, replace it with typical tissue" fill rather than an arbitrary constant.

    That said, see this module's top-of-file docstring, Decision 2: a FULLY deleted (or, for
    insertion, not-yet-restored) volume is spatially CONSTANT regardless of which fill value was
    used, and a spatially constant input is degenerate for this architecture's normalization
    layers no matter what constant it is. `"zero"` does not avoid that degeneracy -- nothing
    can, for a 100%-deleted volume -- it only makes the PARTIALLY-perturbed points on the curve
    (which is what the curve's AUC is actually built from) a meaningful fill rather than a
    strange one.

    Args:
        image: `(1, C, D, H, W)` or `(C, D, H, W)`.
        fill: One of `"zero"` (all zeros -- see above), `"noise"` (`torch.randn` drawn from
            `generator`, same shape/dtype as `image`), or `"mean"` (each channel's own spatial
            mean, broadcast back to the full volume).
        generator: Required when `fill == "noise"`. Ignored otherwise. No default and no use of
            the global RNG (CLAUDE.md: randomness only through an explicitly passed generator).

    Returns:
        A tensor the same shape and dtype as `image`.

    Raises:
        ValueError: `fill` is not one of `_VALID_FILLS`, or `fill == "noise"` and `generator`
            is `None`.
    """
    if fill not in _VALID_FILLS:
        raise ValueError(f"make_fill: unknown fill {fill!r}; expected one of {_VALID_FILLS}.")

    if fill == "zero":
        return torch.zeros_like(image)

    if fill == "noise":
        if generator is None:
            raise ValueError(
                "make_fill: generator is required when fill='noise' (CLAUDE.md: randomness "
                "only through an explicitly passed generator, no global RNG fallback)."
            )
        return torch.randn(
            tuple(image.shape), generator=generator, dtype=image.dtype, device=image.device
        )

    # fill == "mean": per-channel spatial mean, broadcast back to the full volume.
    spatial_dims = tuple(range(image.ndim - 3, image.ndim))
    channel_mean = image.mean(dim=spatial_dims, keepdim=True)
    return channel_mean.expand_as(image).clone()


@dataclass
class FaithfulnessCurve:
    """One deletion or insertion curve. See module docstring, Decision 1, for the two Dice
    columns' meaning, and Decision 2 for why the degenerate endpoint is excluded from the AUC
    by default while still being present in the arrays.

    Attributes:
        fractions: `(n_points,)`, ascending, `linspace(0, 1, n_points)`. The fraction of voxels
            perturbed so far (deleted, for `mode="deletion"`; restored, for `mode="insertion"`).
        dice_vs_prediction: `(n_points,)`. Dice between the perturbed-input prediction and the
            model's ORIGINAL (unperturbed) prediction, for `region_index`. THE PRIMARY COLUMN --
            see Decision 1. A model that becomes wildly worse under perturbation for a faithful
            attribution shows a curve that drops fast (deletion) or rises fast (insertion).
        dice_vs_ground_truth: `(n_points,)`, or `None` if no `ground_truth` was supplied. Dice
            between the perturbed-input prediction and the real label. More clinically legible,
            but confounded by the model's own accuracy -- see Decision 1.
        auc_vs_prediction: Area under `dice_vs_prediction` (`np.trapezoid` over `fractions`),
            respecting `include_endpoint`. For `mode="deletion"`, LOWER is better (the
            prediction collapses fast once important voxels are gone). For `mode="insertion"`,
            HIGHER is better (the prediction recovers fast once important voxels are back).
        auc_vs_ground_truth: Same, computed from `dice_vs_ground_truth`; `None` if that column
            is `None`.
        mode: `"deletion"` or `"insertion"`.
        include_endpoint: Whether the degenerate endpoint (`fractions[-1] == 1.0` for deletion,
            `fractions[0] == 0.0` for insertion) was included in the two AUC values above. The
            point itself is ALWAYS present in the arrays regardless of this flag.
        n_points: `len(fractions)`.
    """

    fractions: np.ndarray
    dice_vs_prediction: np.ndarray
    dice_vs_ground_truth: np.ndarray | None
    auc_vs_prediction: float
    auc_vs_ground_truth: float | None
    mode: str
    include_endpoint: bool
    n_points: int


def _compute_auc(
    fractions: np.ndarray, values: np.ndarray, mode: str, include_endpoint: bool
) -> float:
    """AUC of `values` over `fractions`, optionally dropping the degenerate endpoint.

    See this module's top-of-file docstring, Decision 2. Deletion drops the LAST point
    (`fractions[-1] == 1.0`, everything deleted); insertion drops the FIRST point
    (`fractions[0] == 0.0`, nothing yet restored) -- both are the all-fill, spatially constant
    volume.

    Args:
        fractions: `(n_points,)`, ascending.
        values: `(n_points,)`, same length.
        mode: `"deletion"` or `"insertion"`.
        include_endpoint: If True, no point is dropped.

    Returns:
        `np.trapezoid(values, fractions)` over whichever points are included.
    """
    if include_endpoint or fractions.size <= 1:
        return _trapz(values, fractions)
    if mode == "deletion":
        return _trapz(values[:-1], fractions[:-1])
    return _trapz(values[1:], fractions[1:])


def _run_curve(
    mode: str,
    model: nn.Module,
    image: Tensor,
    attribution: Tensor,
    region_index: int,
    ground_truth: Tensor | None,
    n_points: int,
    fill: str,
    generator: torch.Generator | None,
    threshold: float,
    include_endpoint: bool,
) -> FaithfulnessCurve:
    """Shared implementation behind `deletion_curve` and `insertion_curve`.

    See those two functions' docstrings for the algorithm; this only exists to avoid
    duplicating it, since the two modes differ only in which side of `torch.where` the fill
    value sits on.
    """
    if model.training:
        raise ValueError(
            f"{mode}_curve requires model.training is False (eval mode). Call model.eval() "
            "yourself first -- this function deliberately does not flip your model's mode for "
            "you, matching every other function in neurovision.explainability (see "
            "neurovision.explainability.gradcam.grad_cam's docstring for the full reasoning: "
            "NeuroVisionX.forward only guarantees a single logits Tensor back in eval mode)."
        )
    if n_points < 2:
        raise ValueError(f"{mode}_curve: n_points must be >= 2, got {n_points}.")

    image_5d = _to_batched_5d(image, f"{mode}_curve image")
    spatial_shape = tuple(image_5d.shape[2:])

    attr_3d = _reduce_to_3d(attribution, f"{mode}_curve attribution")
    if tuple(attr_3d.shape) != spatial_shape:
        raise ValueError(
            f"{mode}_curve: attribution spatial shape {tuple(attr_3d.shape)} does not match "
            f"image spatial shape {spatial_shape}."
        )

    with torch.no_grad():
        logits = model(image_5d)
    if not isinstance(logits, Tensor):
        raise ValueError(
            f"{mode}_curve: model(image) returned a {type(logits).__name__}, not a Tensor. "
            "This means the model was not actually on the plain-Tensor eval path -- see "
            "NeuroVisionX.forward's docstring for its three possible return types (Tensor / "
            "list[Tensor] / MultiTaskOutput)."
        )
    num_channels = logits.shape[1]
    if not (0 <= region_index < num_channels):
        raise ValueError(
            f"region_index ({region_index}) is out of range for model output with "
            f"{num_channels} channels."
        )
    with torch.no_grad():
        original_pred_full = binarize(logits, threshold=threshold)
    original_pred_region = original_pred_full[:, region_index : region_index + 1]

    gt_region: Tensor | None = None
    if ground_truth is not None:
        gt_region = _extract_region_channel(
            ground_truth, region_index, spatial_shape, f"{mode}_curve ground_truth"
        )

    # Built ONCE and reused across every fraction -- for fill="noise" this is what makes the
    # curve a well-defined function of a single fill draw rather than a fresh (and therefore
    # incomparable, point to point) draw at every step.
    fill_volume = make_fill(image_5d, fill, generator=generator)

    order = rank_voxels_by_attribution(attr_3d)  # (D*H*W,), descending by |attribution|
    total_voxels = order.numel()

    fractions = np.linspace(0.0, 1.0, n_points)
    dice_pred = np.empty(n_points, dtype=np.float64)
    dice_gt = np.empty(n_points, dtype=np.float64) if gt_region is not None else None

    for i, f in enumerate(fractions):
        k = int(round(float(f) * total_voxels))
        mask_flat = torch.zeros(total_voxels, dtype=torch.bool)
        if k > 0:
            mask_flat[order[:k]] = True
        mask = mask_flat.view(1, 1, *spatial_shape)  # broadcasts across every input channel

        if mode == "deletion":
            # Top-ranked (most-attributed) voxels are REPLACED with fill; everything else stays
            # the real image. At f=0 (k=0) mask is all-False, so this is the real image
            # unchanged -- the deletion_curve docstring's sanity anchor.
            perturbed = torch.where(mask, fill_volume, image_5d)
        else:
            # Top-ranked voxels are RESTORED to their real value; everything else stays fill.
            # At f=1 (k=total_voxels) mask is all-True, so this is the real image restored in
            # full -- the insertion_curve docstring's sanity anchor.
            perturbed = torch.where(mask, image_5d, fill_volume)

        with torch.no_grad():
            perturbed_logits = model(perturbed)
            perturbed_pred_region = binarize(perturbed_logits, threshold=threshold)[
                :, region_index : region_index + 1
            ]

        dice_pred[i] = float(
            dice_score(perturbed_pred_region, original_pred_region, ignore_empty=False)[0, 0]
        )
        if gt_region is not None:
            dice_gt[i] = float(
                dice_score(perturbed_pred_region, gt_region, ignore_empty=False)[0, 0]
            )

    auc_pred = _compute_auc(fractions, dice_pred, mode, include_endpoint)
    auc_gt = (
        _compute_auc(fractions, dice_gt, mode, include_endpoint) if dice_gt is not None else None
    )

    return FaithfulnessCurve(
        fractions=fractions,
        dice_vs_prediction=dice_pred,
        dice_vs_ground_truth=dice_gt,
        auc_vs_prediction=auc_pred,
        auc_vs_ground_truth=auc_gt,
        mode=mode,
        include_endpoint=include_endpoint,
        n_points=n_points,
    )


def deletion_curve(
    model: nn.Module,
    image: Tensor,
    attribution: Tensor,
    region_index: int = 0,
    ground_truth: Tensor | None = None,
    n_points: int = 11,
    fill: str = "zero",
    generator: torch.Generator | None = None,
    threshold: float = 0.5,
    include_endpoint: bool = False,
) -> FaithfulnessCurve:
    """Progressively deletes the most-attributed voxels and tracks how the prediction degrades.

    Algorithm: rank every spatial voxel of `attribution` by `|attribution|` descending
    (`rank_voxels_by_attribution`). For each fraction `f` in `linspace(0, 1, n_points)`, replace
    the top `f` fraction of ranked voxels -- across ALL input channels at those spatial
    positions, since a heatmap ranks LOCATIONS, not individual modality channels -- with a fill
    value (`make_fill`), run `model` on the result under `no_grad`, threshold the logits at
    `threshold`, and score Dice for `region_index` against (a) the model's ORIGINAL prediction
    on the unperturbed `image` and (b) `ground_truth`, if given. See this module's top-of-file
    docstring, Decision 1, for why (a) is the primary column.

    LOWER `auc_vs_prediction` is better here: a faithful attribution should make the prediction
    collapse quickly as the voxels it claims are important disappear, which shows up as a curve
    that drops toward 0 early and therefore has a small area under it.

    Args:
        model: The segmentation model, already in eval mode (`model.eval()`) -- see `Raises`.
        image: `(1, C, D, H, W)` or `(C, D, H, W)`.
        attribution: `(1, 1, D, H, W)` or `(D, H, W)`, spatial shape matching `image`'s.
        region_index: Output channel to score, e.g. 0/1/2 for ET/TC/WT.
        ground_truth: Optional `(1, C, D, H, W)` or `(C, D, H, W)` binary region tensor (e.g.
            `neurovision.metrics.segmentation.classes_to_regions`'s output). `None` (the
            default) skips `dice_vs_ground_truth` entirely (left `None` on the result).
        n_points: Number of fractions sampled, `linspace(0, 1, n_points)`. Must be `>= 2`.
        fill: Passed to `make_fill` -- `"zero"` (default), `"noise"`, or `"mean"`.
        generator: Required when `fill == "noise"`; ignored otherwise.
        threshold: Sigmoid-probability threshold used to binarize every prediction (original,
            per-fraction, and -- implicitly, since `ground_truth` is assumed already binary --
            nowhere else).
        include_endpoint: If False (the default), the `f == 1.0` point (the fully-deleted,
            spatially constant volume -- see Decision 2) is excluded from `auc_vs_prediction`
            and `auc_vs_ground_truth`, but is still present in the returned arrays.

    Returns:
        A `FaithfulnessCurve` with `mode="deletion"`.

    Raises:
        ValueError: `model.training` is True; `n_points < 2`; `image`/`attribution` shapes are
            invalid or disagree; `region_index` is out of range for the model's output; `fill`
            is unknown, or `fill == "noise"` with no `generator`; `model(image)` does not return
            a plain `Tensor` (see `NeuroVisionX.forward`'s three possible return types).
    """
    return _run_curve(
        "deletion",
        model,
        image,
        attribution,
        region_index,
        ground_truth,
        n_points,
        fill,
        generator,
        threshold,
        include_endpoint,
    )


def insertion_curve(
    model: nn.Module,
    image: Tensor,
    attribution: Tensor,
    region_index: int = 0,
    ground_truth: Tensor | None = None,
    n_points: int = 11,
    fill: str = "zero",
    generator: torch.Generator | None = None,
    threshold: float = 0.5,
    include_endpoint: bool = False,
) -> FaithfulnessCurve:
    """The mirror of `deletion_curve`: starts from an all-fill volume and RESTORES voxels.

    Same ranking and fraction grid as `deletion_curve`. At fraction `f`, the top `f` fraction of
    ranked voxels are restored to their real `image` value (across all input channels at those
    spatial positions); every other voxel stays at the fill value. `f == 0` is therefore the
    fully-fill, spatially constant volume (the SAME degenerate point `deletion_curve` reaches at
    `f == 1`, see this module's top-of-file docstring, Decision 2), and `f == 1` is the real
    `image` restored in full.

    HIGHER `auc_vs_prediction` is better here (the mirror of `deletion_curve`'s convention): a
    faithful attribution should recover the original prediction quickly as the voxels it claims
    are important come back, which shows up as a curve that rises toward 1 early and therefore
    has a large area under it. `include_endpoint=False` (the default) excludes the `f == 0`
    point from the AUC for the same degeneracy reason `deletion_curve` excludes `f == 1`.

    Args, Returns, Raises: identical to `deletion_curve`, with `mode="insertion"` on the result.
    """
    return _run_curve(
        "insertion",
        model,
        image,
        attribution,
        region_index,
        ground_truth,
        n_points,
        fill,
        generator,
        threshold,
        include_endpoint,
    )


def pointing_game(attribution: Tensor, ground_truth: Tensor) -> dict[str, float]:
    """The classic "does the single most-attributed voxel land on the tumor" localization check.

    See this module's top-of-file docstring, Decision 3, for why the bare hit rate is not
    reportable on its own: `ratio` (hit divided by the null hit rate a uniformly random point
    would achieve) is the number to report. A single case's `hit` is binary (0.0 or 1.0) and
    only the MEAN over many cases is a meaningful localization statistic -- do not read anything
    into one case's result on its own.

    Args:
        attribution: `(1, 1, D, H, W)` or `(D, H, W)`.
        ground_truth: Boolean-like, same spatial shape convention as `attribution`.

    Returns:
        `{"hit": 0.0 or 1.0, "gt_volume_fraction": float, "ratio": float, "n_gt_voxels": int}`.
        `gt_volume_fraction = n_gt_voxels / total_voxels`, the hit probability of a uniformly
        random point. `ratio = hit / gt_volume_fraction`; `float("nan")` (with a warning
        logged) if `ground_truth` is entirely empty, since the ratio is undefined there.

    Raises:
        ValueError: `attribution`/`ground_truth` do not reduce to `(D, H, W)`, or their shapes
            disagree.
    """
    attr = _reduce_to_3d(attribution, "pointing_game attribution")
    gt = _reduce_to_3d(ground_truth, "pointing_game ground_truth").to(torch.bool)
    if tuple(attr.shape) != tuple(gt.shape):
        raise ValueError(
            f"pointing_game: attribution shape {tuple(attr.shape)} does not match "
            f"ground_truth shape {tuple(gt.shape)}."
        )

    total_voxels = attr.numel()
    n_gt_voxels = int(gt.sum().item())
    gt_volume_fraction = n_gt_voxels / total_voxels

    argmax_idx = int(torch.argmax(attr.abs().reshape(-1)).item())
    hit = float(gt.reshape(-1)[argmax_idx].item())

    if gt_volume_fraction == 0.0:
        logger.warning(
            "pointing_game: ground_truth is entirely empty; ratio is undefined and reported "
            "as NaN."
        )
        ratio = float("nan")
    else:
        ratio = hit / gt_volume_fraction

    return {
        "hit": hit,
        "gt_volume_fraction": gt_volume_fraction,
        "ratio": ratio,
        "n_gt_voxels": n_gt_voxels,
    }


def attribution_mass_ratio(attribution: Tensor, ground_truth: Tensor) -> dict[str, float]:
    """The RECOMMENDED headline localization statistic -- see Decision 4 above.

    Compares the FRACTION of `|attribution|`'s total mass that falls inside `ground_truth`
    against `gt_volume_fraction`, the fraction a spatially uniform attribution map would put
    there by construction. `ratio == 1.0` means exactly that -- no localization whatsoever, the
    attribution is distributed in proportion to the tumor's size and nothing more. `ratio > 1.0`
    means genuine concentration inside the tumor.

    Args:
        attribution: `(1, 1, D, H, W)` or `(D, H, W)`.
        ground_truth: Boolean-like, same spatial shape convention as `attribution`.

    Returns:
        `{"mass_inside": float, "gt_volume_fraction": float, "ratio": float}`.
        `mass_inside = sum(|attribution| inside ground_truth) / sum(|attribution|)`, `float("nan")`
        (with a warning logged) if the total attribution mass is exactly 0. `ratio =
        mass_inside / gt_volume_fraction`, `float("nan")` (with a warning logged) if
        `ground_truth` is entirely empty.

    Raises:
        ValueError: `attribution`/`ground_truth` do not reduce to `(D, H, W)`, or their shapes
            disagree.
    """
    attr = _reduce_to_3d(attribution, "attribution_mass_ratio attribution").abs()
    gt = _reduce_to_3d(ground_truth, "attribution_mass_ratio ground_truth").to(torch.bool)
    if tuple(attr.shape) != tuple(gt.shape):
        raise ValueError(
            f"attribution_mass_ratio: attribution shape {tuple(attr.shape)} does not match "
            f"ground_truth shape {tuple(gt.shape)}."
        )

    total_voxels = attr.numel()
    n_gt_voxels = int(gt.sum().item())
    gt_volume_fraction = n_gt_voxels / total_voxels

    total_mass = float(attr.sum().item())
    if total_mass == 0.0:
        logger.warning(
            "attribution_mass_ratio: total |attribution| mass is exactly 0; mass_inside (and "
            "therefore ratio) is undefined and reported as NaN."
        )
        mass_inside = float("nan")
    else:
        mass_inside = float(attr[gt].sum().item()) / total_mass

    if gt_volume_fraction == 0.0:
        logger.warning(
            "attribution_mass_ratio: ground_truth is entirely empty; ratio is undefined and "
            "reported as NaN."
        )
        ratio = float("nan")
    else:
        ratio = mass_inside / gt_volume_fraction

    return {
        "mass_inside": mass_inside,
        "gt_volume_fraction": gt_volume_fraction,
        "ratio": ratio,
    }


def random_attribution_like(attribution: Tensor, generator: torch.Generator) -> Tensor:
    """A uniform-random attribution map, same shape as `attribution` -- the null baseline.

    Used by `compare_methods` to build its automatically-added `"random"` row -- see this
    module's top-of-file docstring, Decision 3: without this row, none of the other methods'
    numbers are interpretable.

    Args:
        attribution: Any shape; only `.shape` (and floating-point-ness, for the dtype) is used.
        generator: Required (no default, no global-RNG fallback -- CLAUDE.md).

    Returns:
        `torch.rand(attribution.shape, generator=generator)`, values in `[0, 1)`, dtype
        float32 (or `attribution.dtype` if it is already floating-point).

    Raises:
        ValueError: `generator` is `None`.
    """
    if generator is None:
        raise ValueError(
            "random_attribution_like requires an explicit torch.Generator -- no default and no "
            "use of the global RNG (CLAUDE.md: randomness only through an explicitly passed "
            "generator)."
        )
    dtype = attribution.dtype if attribution.dtype.is_floating_point else torch.float32
    return torch.rand(tuple(attribution.shape), generator=generator, dtype=dtype)


def compare_methods(
    model: nn.Module,
    image: Tensor,
    attributions: Mapping[str, Tensor],
    ground_truth: Tensor,
    region_index: int = 0,
    n_points: int = 11,
    fill: str = "zero",
    generator: torch.Generator | None = None,
    target_specific: Mapping[str, bool] | None = None,
    native_resolution: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Runs deletion, insertion, and both localization metrics over every supplied method.

    The `"random"` row is added automatically (via `random_attribution_like`, using the first
    supplied attribution's shape as a template) and is NOT optional garnish -- per this module's
    top-of-file docstring, Decision 3, it is the only thing that makes the other rows
    interpretable. A method that fails to beat it on `insertion_minus_deletion` has not been
    shown to explain anything: that column is exactly the gap between "the prediction recovers
    fast when the claimed-important voxels come back" and "the prediction collapses fast when
    they are removed", and a method indistinguishable from uniform noise on both counts has
    that gap indistinguishable from the random row's too.

    Args:
        model: The segmentation model, already in eval mode.
        image: `(1, C, D, H, W)` or `(C, D, H, W)`.
        attributions: `{method_name: attribution}`, each `(1, 1, D, H, W)` or `(D, H, W)`,
            spatial shape matching `image`'s. Must be non-empty.
        ground_truth: `(1, C, D, H, W)` or `(C, D, H, W)` binary region tensor.
        region_index: Output channel every method is compared on.
        n_points: Forwarded to `deletion_curve`/`insertion_curve`.
        fill: Forwarded to `deletion_curve`/`insertion_curve`/`make_fill`.
        generator: REQUIRED (not optional here, even though `fill="zero"` needs none) -- the
            automatically-added `"random"` row always needs one to draw its uniform map.
        target_specific: Optional `{method_name: bool}` -- see this module's top-of-file
            docstring, Decision 5. Any method not named here defaults to `True`; the
            automatically-added `"random"` row defaults to `False` (a random map does not
            depend on the target region either), unless explicitly overridden here.
        native_resolution: Optional `{method_name: str}`, free text (e.g. `"voxel"`,
            `"stride 8 (upsampled)"`) -- see Decision 6. Any method not named here (including
            `"random"`, which is genuinely drawn at voxel resolution) defaults to `"voxel"`.

    Returns:
        A `DataFrame` indexed by method name (every key of `attributions`, plus `"random"`),
        columns `deletion_auc`, `insertion_auc`, `insertion_minus_deletion`
        (`insertion_auc - deletion_auc`), `pointing_hit`, `pointing_ratio`, `mass_ratio`,
        `target_specific`, `native_resolution`, `n_points`.

    Raises:
        ValueError: `attributions` is empty; `generator` is `None`; or propagated from
            `deletion_curve`/`insertion_curve`/`pointing_game`/`attribution_mass_ratio` (bad
            shapes, `model.training` True, `region_index` out of range, ...).
    """
    if not attributions:
        raise ValueError("compare_methods: attributions is empty; need at least one method.")
    if generator is None:
        raise ValueError(
            "compare_methods requires an explicit torch.Generator: the automatically-added "
            "'random' null-baseline row (see this module's top-of-file docstring, Decision 3) "
            "needs one to draw its uniform map, and this project takes randomness only from an "
            "explicitly seeded generator (CLAUDE.md)."
        )

    target_specific_map = dict(target_specific) if target_specific is not None else {}
    native_resolution_map = dict(native_resolution) if native_resolution is not None else {}

    image_5d = _to_batched_5d(image, "compare_methods image")
    spatial_shape = tuple(image_5d.shape[2:])
    gt_region = _extract_region_channel(
        ground_truth, region_index, spatial_shape, "compare_methods ground_truth"
    )

    methods: dict[str, Tensor] = dict(attributions)
    first_attribution = next(iter(attributions.values()))
    methods["random"] = random_attribution_like(first_attribution, generator)

    logger.info(
        "compare_methods: comparing %d method(s) (%s) at n_points=%d -- see this module's "
        "top-of-file docstring for the measured per-curve cost; this is figure-generation "
        "tooling, not something to run across an evaluation split.",
        len(methods),
        sorted(methods),
        n_points,
    )

    rows: dict[str, dict[str, object]] = {}
    for name, attr in methods.items():
        deletion = deletion_curve(
            model,
            image_5d,
            attr,
            region_index=region_index,
            ground_truth=ground_truth,
            n_points=n_points,
            fill=fill,
            generator=generator,
        )
        insertion = insertion_curve(
            model,
            image_5d,
            attr,
            region_index=region_index,
            ground_truth=ground_truth,
            n_points=n_points,
            fill=fill,
            generator=generator,
        )
        pg = pointing_game(attr, gt_region)
        mr = attribution_mass_ratio(attr, gt_region)

        default_target_specific = name != "random"  # see Decision 3 / Decision 5
        rows[name] = {
            "deletion_auc": deletion.auc_vs_prediction,
            "insertion_auc": insertion.auc_vs_prediction,
            "insertion_minus_deletion": insertion.auc_vs_prediction - deletion.auc_vs_prediction,
            "pointing_hit": pg["hit"],
            "pointing_ratio": pg["ratio"],
            "mass_ratio": mr["ratio"],
            "target_specific": target_specific_map.get(name, default_target_specific),
            "native_resolution": native_resolution_map.get(name, "voxel"),
            "n_points": n_points,
        }

    return pd.DataFrame.from_dict(rows, orient="index")
