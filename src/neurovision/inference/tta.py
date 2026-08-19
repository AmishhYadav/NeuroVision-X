"""Test-time flip augmentation (TTA).

Runs the model on the input volume and on every requested flipped copy of
it, un-flips each prediction back into the original orientation, and
averages the resulting PROBABILITIES. This buys a small, well-documented
Dice gain in the segmentation literature and -- the reason it belongs in
this project specifically -- it is a second, essentially free calibration
lever: averaging several slightly-disagreeing views of the same volume
softens an overconfident single pass, and the spread ACROSS those views is
itself a usable (if partial) uncertainty signal, at the cost of a few extra
sliding-window passes rather than a GPU-hour of training.

## Why only flips, and only these flips

`configs/data/*` trains with `flip_prob: 0.5` applied INDEPENDENTLY on each
of the 3 spatial axes (see `neurovision.data.transforms`), so the model was
explicitly trained to be invariant to axis flips and to nothing else. Flip
TTA therefore tests exactly the invariance the model was trained to have.
Rotations or rescaling are NOT part of that training augmentation, so
running TTA over rotated/scaled copies would probe invariances the model
was never asked to learn -- the augmented predictions would disagree for a
reason that has nothing to do with genuine predictive uncertainty, and
averaging them in would not reliably improve Dice the way flip TTA does.
This module implements flips only, deliberately.

## Structural mirror of `neurovision.inference.mc_dropout`

This module solves a near-identical shape of problem to
`mc_dropout_predict` -- N forward passes over the same volume, reduced to
one averaged prediction plus a spread statistic -- and copies its solution
on purpose:

- A small dataclass output (`TTAOutput`, vs. `MCDropoutOutput`).
- Two running accumulators sized like ONE pass (`sum_prob`, `sum_sq_prob`),
  never an N-deep stack of full-volume tensors. At N=8 (`FLIP_AXES_8`) on a
  median BraTS 2021 cropped volume (~137x171x140, 3 channels, fp32), a stack
  would hold ~8 x 39 MB = ~312 MB for no benefit; the two accumulators used
  here cost ~78 MB total REGARDLESS of N, exactly as `mc_dropout.py`'s
  docstring works out for its own two accumulators.
- Averaging PROBABILITIES, never logits: `mean(sigmoid(z_i))` is the
  correct Monte-Carlo-style estimate of the predictive mean over
  augmentations, `sigmoid(mean(z_i))` is a different (generally
  smaller-magnitude) quantity because `sigmoid` is nonlinear. Exactly the
  same reasoning, and the same downstream hazard, as
  `MCDropoutOutput.mean_prob` -- `TTAOutput.mean_prob` also breaks the
  "everything in `neurovision.inference` is logits" convention for the same
  necessary reason, and also needs
  `neurovision.inference.mc_dropout.logits_from_mean_prob` (reused here
  rather than duplicated) before it can be handed to
  `neurovision.inference.postprocess.postprocess_logits`, which applies its
  OWN sigmoid and would otherwise double-sigmoid the values.

## The one way this module must NOT be unified with `mc_dropout.py`

TTA is deliberately DETERMINISTIC. `sliding_window_predict` is called here
with its default `set_eval=True` -- unlike `mc_dropout_predict`, this
module never wants dropout active, and must never be changed to enable it.
Do not "simplify" the two modules into one code path that toggles
dropout on or off; they answer different questions (spread across
augmented VIEWS of one deterministic model, vs. spread across stochastic
FORWARD PASSES of a fixed view) and conflating them would make a caller's
uncertainty source ambiguous -- exactly the trap CLAUDE.md records for the
demo's `X-Uncertainty-Kind` header.

## No randomness anywhere in this module

`flip_combinations` enumerates a fixed, finite set of axis-flip
combinations and `tta_predict` iterates over them in a fixed order; nothing
here draws from any RNG, seeded or otherwise. Unlike
`mc_dropout_predict`, there is no seed parameter and no RNG state to
save/restore.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from neurovision.inference.sliding_window import build_inferer, sliding_window_predict
from neurovision.utils.device import get_device

logger = logging.getLogger(__name__)


def flip_combinations(axes: Sequence[int] = (0, 1, 2)) -> tuple[tuple[int, ...], ...]:
    """All `2 ** len(axes)` flip combinations over `axes`, identity first.

    Each combination is a tuple of the SPATIAL axes (not tensor dimensions,
    see `tta_predict`'s docstring) to flip together -- `()` means "no flip"
    (the identity/original pass), `(0,)` means "flip axis 0 only", `(0, 1)`
    means "flip both axis 0 and axis 1 simultaneously", and so on through
    every subset of `axes`.

    Args:
        axes: The spatial axes to enumerate flip combinations over. Default
            `(0, 1, 2)` -- all 3 spatial axes (D, H, W).

    Returns:
        A tuple of `2 ** len(axes)` unique tuples, each a sorted subset of
        `axes`, with the empty tuple `()` FIRST (so the identity pass is
        always run and always ordered first, which is what makes it the
        natural pass to check outputs against in tests and in
        `progress` callbacks).
    """
    axes = tuple(axes)
    combos: list[tuple[int, ...]] = []
    for r in range(len(axes) + 1):
        combos.extend(itertools.combinations(axes, r))
    return tuple(combos)


# The full 8-way flip family over all 3 spatial axes -- the default
# `tta_predict` uses when `flips=None`, matching the training augmentation's
# `flip_prob` applied independently per axis.
FLIP_AXES_8: tuple[tuple[int, ...], ...] = flip_combinations((0, 1, 2))


@dataclass
class TTAOutput:
    """The result of one `tta_predict` call.

    Both tensor fields are UNBATCHED (`(C, D, H, W)`, no leading batch
    dimension), because `tta_predict` operates on a single case at a time --
    see its docstring for why a batch of more than one case is refused
    rather than silently handled.

    Attributes:
        mean_prob: The mean sigmoid PROBABILITY across all augmented
            passes, shape `(C, D, H, W)` float32, values in `[0, 1]`. These
            are PROBABILITIES, not logits -- see this module's top-of-file
            docstring before feeding this anywhere that expects logits; use
            `neurovision.inference.mc_dropout.logits_from_mean_prob` first.
        std_prob: The per-voxel POPULATION standard deviation of the
            per-augmentation probabilities (divided by `num_augmentations`,
            not `num_augmentations - 1` -- see `tta_predict`'s docstring for
            why). Same shape as `mean_prob`. Exactly `0.0` everywhere when
            `num_augmentations == 1` (nothing to vary against). A genuinely
            flip-equivariant model also gives (near) `0.0` everywhere,
            regardless of `num_augmentations`; a model whose predictions
            depend on absolute spatial position gives a nonzero map that
            doubles as a rough, free uncertainty signal.
        num_augmentations: How many flip passes were averaged.
        flips: The exact flip combinations used, in the order they were run
            (spatial axis tuples, as returned by `flip_combinations` /
            `FLIP_AXES_8`, or whatever was passed via `tta_predict`'s
            `flips` argument).
    """

    mean_prob: Tensor
    std_prob: Tensor
    num_augmentations: int
    flips: tuple[tuple[int, ...], ...]


def tta_predict(
    model: nn.Module,
    image: Tensor,
    cfg: DictConfig,
    *,
    flips: Sequence[Sequence[int]] | None = None,
    device: torch.device | None = None,
    inferer: Any | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> TTAOutput:
    """Sliding-window prediction averaged over flip augmentations.

    Runs one `sliding_window_predict` call per requested flip combination:
    flips the input, runs the model, and flips the resulting logits back
    (see "The un-flip" below) before converting to probability and folding
    it into two running accumulators. Cost is exactly
    `len(flips)` x one plain `sliding_window_predict` call --
    `FLIP_AXES_8` (the default) means 8x the cost of a single deterministic
    pass, the same order of magnitude as a modest MC-dropout `num_samples`.

    Axis convention -- READ THIS BEFORE CHANGING `flips`. `image` is
    `(B, C, D, H, W)` (batch size must be 1, see below) or `(C, D, H, W)`.
    Every flip axis in `flips` / `FLIP_AXES_8` is a SPATIAL axis index
    (`0` = D, `1` = H, `2` = W), never a raw tensor dimension. This function
    converts spatial axis `a` to the batched tensor's dimension `a + 2`
    internally (`0 -> dim 2`, `1 -> dim 3`, `2 -> dim 4`) before calling
    `torch.flip`. Passing raw tensor dimensions instead (e.g. flipping dim 1,
    the CHANNEL axis, by mistake) would silently reorder the input's
    modalities -- MRI channels get shuffled, not reflected -- and produce a
    plausible-looking but wrong prediction with no shape error anywhere,
    the same class of bug CLAUDE.md records for `_window_partition`.

    The un-flip is the SAME flip. A flip along a given set of axes is its
    own inverse (flipping twice returns the original), so
    `torch.flip(torch.flip(x, dims=d), dims=d) == x` for any `d`. Applying
    the IDENTICAL `tensor_dims` tuple to the model's output that was applied
    to its input is therefore exactly the operation that restores spatial
    alignment -- not a separately-derived "inverse flip". This function
    reuses one local variable, `tensor_dims`, for both the forward flip and
    the un-flip, specifically so this cannot drift into two different axis
    sets by accident. `TTAOutput`'s identity-model round-trip test in
    `tests/test_tta.py` is the one that would fail first if this ever did.

    Average PROBABILITIES, never logits -- see this module's top-of-file
    docstring. `prob = sigmoid(unflipped_logits)` is computed once per pass
    and only probability tensors are accumulated.

    Memory. Two running float32 accumulators (`sum_prob`, `sum_sq_prob`)
    sized like ONE pass, exactly mirroring
    `neurovision.inference.mc_dropout.mc_dropout_predict`. Never a
    `len(flips)`-deep stack of full-volume tensors -- see this module's
    top-of-file docstring for the measured memory cost this avoids.

    Eval mode. Calls `sliding_window_predict` with its default
    `set_eval=True` on every pass -- TTA is deterministic and must NEVER
    enable dropout. Do not pass `set_eval=False` here or "unify" this
    function with `mc_dropout_predict`; see this module's top-of-file
    docstring for why that would make a caller's uncertainty source
    ambiguous.

    `std_prob` uses the POPULATION standard deviation (divide the summed
    squared deviations by `num_augmentations`, not `num_augmentations - 1`).
    The `len(flips)` augmented passes run here are not a random sample drawn
    from some larger population whose variance Bessel's correction would be
    estimating -- they are the entire, fixed, finite population of views
    this call actually produced -- so the uncorrected (population) estimator
    is the correct one. It also has the convenient side effect of being
    exactly `0.0`, not undefined, at `num_augmentations == 1`.

    No randomness. This function draws from no RNG, seeded or otherwise;
    `flips` is a fixed, ordered sequence and every pass is deterministic
    given `model` and `image`.

    Args:
        model: The segmentation model. Assumed already on `device`; its
            train/eval mode is set by `sliding_window_predict` itself
            (`set_eval=True`, the default there).
        image: Input volume, shape `(B, in_channels, D, H, W)` with `B == 1`,
            or unbatched `(in_channels, D, H, W)` -- either is accepted,
            since `TTAOutput` never carries a batch dimension. A batch of
            more than one case raises rather than being silently averaged
            or silently reshaped; run this function once per case.
        cfg: The full composed Hydra config, exposing
            `cfg.inference.sliding_window` (read by `sliding_window_predict`
            / `build_inferer`) and `cfg.device` (read by `get_device` when
            `device` is not given explicitly).
        flips: The flip combinations to average over, each a sequence of
            spatial axis indices in `{0, 1, 2}` (see "Axis convention"
            above). `None` (the default) uses `FLIP_AXES_8`, all 8
            combinations over all 3 spatial axes -- matching the project's
            `flip_prob` training augmentation, which is applied
            independently per axis. Must contain at least one entry (`[()]`
            is valid and means "run only the identity pass").
        device: Device to run inference on. `None` (the default) resolves
            it once via `neurovision.utils.device.get_device(cfg)`, exactly
            like `scripts/evaluate.py` and `scripts/extract_gates.py` do
            before calling into `neurovision.inference`. An explicit
            `torch.device` overrides that resolution.
        inferer: A pre-built `SlidingWindowInferer` to reuse across all
            passes, or `None` (the default) to build one via
            `build_inferer(cfg)` once, before the loop -- same reasoning as
            `mc_dropout_predict`'s `inferer` parameter: building it does no
            real work, but reusing one object avoids repeating that
            construction `len(flips)` times.
        progress: Optional callback invoked as `progress(completed, total)`
            immediately after each pass finishes, so a caller looping over
            an evaluation split can report progress without this function
            depending on any particular progress-bar library. Called
            exactly `len(flips)` times, the last call always
            `progress(len(flips), len(flips))`.

    Returns:
        A `TTAOutput`. See that dataclass's docstring for each field's
        shape, units, and range.

    Raises:
        ValueError: If `flips` is given and empty, if `image` is not 4-D or
            5-D, or if `image` is 5-D with a batch size other than 1.
    """
    flip_list: tuple[tuple[int, ...], ...] = (
        FLIP_AXES_8 if flips is None else tuple(tuple(f) for f in flips)
    )
    num_augmentations = len(flip_list)
    if num_augmentations < 1:
        raise ValueError(
            "tta_predict: flips must contain at least one flip combination "
            "(pass [()] for identity-only)."
        )

    if device is None:
        device = get_device(cfg)

    if image.ndim == 4:
        batched_image = image.unsqueeze(0)
    elif image.ndim == 5:
        if image.shape[0] != 1:
            raise ValueError(
                f"tta_predict expects a single case (batch size 1); got batch size "
                f"{image.shape[0]}. TTAOutput carries no batch dimension by design -- "
                "call this function once per case."
            )
        batched_image = image
    else:
        raise ValueError(
            f"tta_predict expects a (C, D, H, W) or (1, C, D, H, W) tensor, got shape "
            f"{tuple(image.shape)}."
        )

    if inferer is None:
        inferer = build_inferer(cfg)  # built once, reused across all passes below

    logger.info(
        "tta_predict: num_augmentations=%d flips=%s input_shape=%s",
        num_augmentations,
        flip_list,
        tuple(batched_image.shape[2:]),
    )

    # Two running float32 accumulators, shaped like ONE pass's output --
    # never a stack of all num_augmentations passes. See this module's
    # top-of-file docstring for the measured memory this avoids.
    sum_prob: Tensor | None = None
    sum_sq_prob: Tensor | None = None

    for completed, flip_axes in enumerate(flip_list, start=1):
        # Spatial axes -> batched tensor dims: axis 0/1/2 (D/H/W) is tensor
        # dim 2/3/4 of a (B, C, D, H, W) tensor. See the docstring above --
        # this is the one line a wrong offset here would silently corrupt.
        tensor_dims = tuple(axis + 2 for axis in flip_axes)

        flipped_input = (
            torch.flip(batched_image, dims=tensor_dims) if tensor_dims else batched_image
        )

        logits = sliding_window_predict(
            model,
            flipped_input,
            cfg,
            device,
            inferer=inferer,
            # Default True: TTA must stay deterministic. See docstring.
            set_eval=True,
        )

        # The un-flip IS the forward flip -- a flip is its own inverse, so
        # re-applying the SAME tensor_dims restores alignment with the
        # original, unflipped input. Do not derive a separate "inverse".
        unflipped_logits = torch.flip(logits, dims=tensor_dims) if tensor_dims else logits

        prob = torch.sigmoid(unflipped_logits)

        if sum_prob is None:
            # Allocated lazily from the first pass's own output, so shape and
            # device are inherited rather than guessed -- mirrors
            # mc_dropout_predict, and respects
            # cfg.inference.sliding_window.output_device: "cpu" the same way.
            sum_prob = torch.zeros_like(prob)
            sum_sq_prob = torch.zeros_like(prob)
        sum_prob += prob
        sum_sq_prob += prob * prob

        if progress is not None:
            progress(completed, num_augmentations)

    assert sum_prob is not None and sum_sq_prob is not None  # num_augmentations >= 1, checked above

    mean_prob = sum_prob / num_augmentations
    # Population variance: E[X^2] - E[X]^2, divided by N not N-1 -- see the
    # docstring's "std_prob uses the POPULATION standard deviation" section.
    variance = (sum_sq_prob / num_augmentations) - mean_prob * mean_prob
    # Clamp away from a tiny negative float-rounding artifact (mathematically
    # variance is >= 0; at num_augmentations == 1 this is also what makes
    # the result EXACTLY 0.0 rather than a possibly-negative near-zero).
    variance = variance.clamp(min=0.0)
    std_prob = torch.sqrt(variance)

    # Drop the batch dimension this function added (or that the caller
    # already had, at size 1) -- TTAOutput is unbatched by design.
    mean_prob = mean_prob.squeeze(0)
    std_prob = std_prob.squeeze(0)

    return TTAOutput(
        mean_prob=mean_prob,
        std_prob=std_prob,
        num_augmentations=num_augmentations,
        flips=flip_list,
    )
