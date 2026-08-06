"""Calibration measurement for the BraTS region-overlap setup.

This project's headline claim is reliability, not raw Dice (see CLAUDE.md:
"competitive Dice with substantially better calibration and boundary
accuracy"). This module is the measurement that claim rests on: given
per-voxel predicted PROBABILITIES and binary ground truth for the three
nested regions (ET, TC, WT), it computes Expected/Maximum Calibration Error
(ECE/MCE), Brier score, reliability-diagram data, and a fitted temperature
scaling.

## The convention: probability vs frequency, not confidence vs accuracy

The classic classification-calibration setup bins `max(p, 1-p)` (the
confidence in whichever class was predicted) against whether that prediction
was correct. That form does not fit here: the three output channels are
INDEPENDENT sigmoids over OVERLAPPING regions (a voxel can legitimately be
foreground in all three), so there is no single "predicted class" to be
confident about. Instead, everything in this module bins the raw predicted
PROBABILITY of a channel against the OBSERVED POSITIVE FREQUENCY of that
channel -- e.g. "voxels this model called ~70% likely to be enhancing tumor
are, in fact, enhancing tumor about 70% of the time." This form is also the
one that is mathematically coherent with Brier score (a proper scoring rule
over probabilities, not over confidences) and with temperature scaling
(which rescales a probability, not a confidence). Every metric here is
reported PER REGION and never pooled across regions -- matching
`neurovision.inference.mc_dropout`, which reports its uncertainty
decomposition the same way and for the same reason: ET is the smallest,
hardest region and the one this project's calibration claim leans on most,
so collapsing it into a whole-tumor average would throw away the exact
breakdown a reader needs.

## Which reporting mask to use

`union_foreground_mask` (predicted-positive UNION ground-truth-positive) is
CIRCULAR: a voxel with `p < threshold` can only join it via `label > 0`, so
`P(label=1 | p < threshold, in mask)` is exactly 1 by construction, not a
measurement -- measured to inflate reported ECE by 41-57% and to bias a
fitted temperature by roughly 3-4x on a real 189-case split. It is kept in
this module as a DIAGNOSTIC only, and it warns on every call. Use
`predicted_foreground_mask` (the model's own predicted-positive voxels,
label-free) or `brain_mask` (the nonzero-intensity brain region, also
label-free) to compute any number that gets reported. See each function's
own docstring for the full account.

## CUDA hazard (read before touching device/dtype handling here)

CLAUDE.md records three separate CUDA-only faults that shipped past a green
CPU-only test suite in this project (a metrics device mismatch in
`evaluate.py`, a CuPy-only HD95 code path, and an RNG-restore bug that only
fires on a CUDA resume). The Mac is a correctness harness for LOGIC here, not
for device placement -- no local test can catch a device bug if every tensor
already happens to be on CPU. `scripts/evaluate.py` may hand this module
CUDA fp16 tensors (predicted probabilities straight off a sliding-window
inference call, or `MCDropoutOutput.mean_prob`), so every free function below
immediately detaches and moves its `prob`/`label`/`mask` inputs to
`(dtype=torch.float32, device="cpu")`. float32 rather than the input's native
dtype: an ECE computed by binning millions of fp16 values would round enough
low-order bits away at the 15-bin resolution used here to matter, and this
module accumulates over up to ~10^9 voxels in `CalibrationAccumulator`, where
fp16 (max exact integer ~2048) would silently lose count precision.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from neurovision.data.transforms import REGION_NAMES

__all__ = [
    "DEFAULT_N_BINS",
    "bin_edges",
    "reliability_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "union_foreground_mask",
    "predicted_foreground_mask",
    "brain_mask",
    "subsample_voxels",
    "TemperatureResult",
    "fit_temperature",
    "apply_temperature",
    "CalibrationAccumulator",
]

logger = logging.getLogger(__name__)

DEFAULT_N_BINS = 15


# ---------------------------------------------------------------------------
# Shared input handling
# ---------------------------------------------------------------------------


def _to_cpu_float32(x: Tensor | np.ndarray) -> Tensor:
    """Detaches and moves `x` to `(dtype=torch.float32, device="cpu")`.

    See this module's top-of-file docstring for why this cast is mandatory
    rather than defensive: evaluation may hand this module CUDA fp16
    tensors, and this is the one place in the pipeline responsible for never
    letting a device/precision bug through silently.
    """
    return torch.as_tensor(x).detach().to(dtype=torch.float32, device="cpu")


def _validate_prob_label(
    a: Tensor, b: Tensor, check_prob_range: bool, name_a: str = "prob", name_b: str = "label"
) -> None:
    """Shared shape/range/binary validation for the (prob, label) and (logits, labels) pairs.

    Args:
        a: First tensor (`prob` or `logits`), already cast to float32 CPU.
        b: Second tensor (`label` or `labels`), same convention.
        check_prob_range: If True, `a` must contain only values in `[0, 1]`.
            False for `fit_temperature`, whose first argument is LOGITS, not
            probabilities, and can legitimately be any real number.
        name_a: Name used for `a` in error messages.
        name_b: Name used for `b` in error messages.

    Raises:
        ValueError: Shape mismatch, zero elements, `a` outside `[0, 1]`
            (only when `check_prob_range`), or `b` not binary.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got {tuple(a.shape)} and "
            f"{tuple(b.shape)}."
        )
    if a.numel() == 0:
        raise ValueError(
            f"{name_a}/{name_b} have zero elements; nothing to compute a calibration metric over."
        )
    if check_prob_range:
        p_min = float(a.min())
        p_max = float(a.max())
        if p_min < 0.0 or p_max > 1.0:
            raise ValueError(
                f"{name_a} must contain values in [0, 1], got min={p_min} max={p_max}. This "
                "usually means raw logits, or a doubly-sigmoided probability (e.g. "
                "MCDropoutOutput.mean_prob passed through postprocess_logits' internal sigmoid "
                "a second time), were passed instead of plain probabilities."
            )
    is_binary = bool(torch.all((b == 0.0) | (b == 1.0)))
    if not is_binary:
        b_min = float(b.min())
        b_max = float(b.max())
        raise ValueError(
            f"{name_b} must be binary (0 or 1 only), got values in [{b_min}, {b_max}] that are "
            "not exactly 0 or 1."
        )


def _prepare_flat(
    prob: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
    mask: Tensor | np.ndarray | None = None,
    check_prob_range: bool = True,
) -> tuple[Tensor, Tensor]:
    """Casts, validates, flattens, and (optionally) masks a (prob, label) pair.

    Validation runs on the FULL, unmasked tensors -- a mask that happens to
    select zero elements is a normal, expected event (a region can be
    entirely absent from a BraTS case) and must not raise; see the "Empty
    mask" note on the public functions below. Only a genuinely empty INPUT
    (before masking) is an error.

    Args:
        prob: Predicted probabilities, any shape, values expected in `[0, 1]`.
        label: Binary ground truth, same shape as `prob`.
        mask: Optional boolean-like array/tensor broadcastable to `prob`'s
            shape. `True`/nonzero selects a voxel for inclusion.
        check_prob_range: Forwarded to `_validate_prob_label`.

    Returns:
        `(prob_flat, label_flat)`, 1-D float32 CPU tensors, masked if `mask`
        was given. May have zero elements if `mask` selected nothing.

    Raises:
        ValueError: See `_validate_prob_label`.
    """
    prob_t = _to_cpu_float32(prob)
    label_t = _to_cpu_float32(label)
    _validate_prob_label(prob_t, label_t, check_prob_range=check_prob_range)

    prob_flat = prob_t.reshape(-1)
    label_flat = label_t.reshape(-1)

    if mask is None:
        return prob_flat, label_flat

    mask_t = torch.as_tensor(mask).detach().to(dtype=torch.bool, device="cpu")
    mask_t = torch.broadcast_to(mask_t, prob_t.shape).reshape(-1)
    return prob_flat[mask_t], label_flat[mask_t]


# ---------------------------------------------------------------------------
# Binning primitives, shared by the free functions AND CalibrationAccumulator
# so both code paths compute a bin exactly the same way -- this is what makes
# the streaming vs. one-shot agreement test in tests/test_calibration.py hold
# to floating-point precision rather than just "close".
# ---------------------------------------------------------------------------


def bin_edges(n_bins: int = DEFAULT_N_BINS) -> Tensor:
    """Equal-width bin edges spanning `[0, 1]`.

    Args:
        n_bins: Number of bins.

    Returns:
        Float32 tensor of shape `(n_bins + 1,)`: `edges[i]`, `edges[i+1]`
        bound bin `i`.

    Raises:
        ValueError: If `n_bins < 1`.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")
    return torch.linspace(0.0, 1.0, steps=n_bins + 1, dtype=torch.float32)


def _bin_stats(
    prob_flat: Tensor, label_flat: Tensor, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assigns voxels to equal-width bins and returns per-bin sums.

    Assignment is `idx = clamp(floor(p * n_bins), 0, n_bins - 1)`, so
    `p == 1.0` lands in the LAST bin rather than one-past-the-end. This is
    not a theoretical edge case: fp16 rounding during preprocessing/AMP
    inference produces exact 1.0 (and exact 0.0) probabilities routinely, so
    an off-by-one here would silently drop real voxels from every bin table.

    Args:
        prob_flat: 1-D float32 tensor, already validated and (optionally)
            masked. May be empty -- callers must check `numel() == 0`
            themselves before calling this (an empty input here would
            produce all-zero bins, which is a valid but different case from
            "not yet checked").
        label_flat: 1-D float32 tensor, same shape as `prob_flat`.
        n_bins: Number of equal-width bins.

    Returns:
        Three float64 numpy arrays of shape `(n_bins,)`: per-bin voxel
        count, sum of probabilities, sum of labels.

    Raises:
        ValueError: If `n_bins < 1`.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")
    idx = torch.clamp(torch.floor(prob_flat * n_bins).long(), 0, n_bins - 1).numpy()
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_prob = np.bincount(idx, weights=prob_flat.numpy().astype(np.float64), minlength=n_bins)
    sum_label = np.bincount(idx, weights=label_flat.numpy().astype(np.float64), minlength=n_bins)
    return count, sum_prob, sum_label


def _bin_means(
    count: np.ndarray, sum_prob: np.ndarray, sum_label: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin `(mean_prob, mean_label, gap)`, NaN for empty bins.

    `gap = mean_label - mean_prob` is SIGNED: negative means the model was
    overconfident (predicted probability higher than the observed
    frequency), positive means underconfident. The paper needs the sign,
    not just `|gap|` -- e.g. "this model is systematically overconfident on
    ET" is a stronger and more actionable statement than "this model's ET
    calibration is off by 0.08".
    """
    mean_prob = np.full_like(count, np.nan, dtype=np.float64)
    mean_label = np.full_like(count, np.nan, dtype=np.float64)
    nonzero = count > 0
    mean_prob[nonzero] = sum_prob[nonzero] / count[nonzero]
    mean_label[nonzero] = sum_label[nonzero] / count[nonzero]
    gap = mean_label - mean_prob
    return mean_prob, mean_label, gap


def _ece_from_bins(count: np.ndarray, gap: np.ndarray) -> float:
    """`sum_b (count_b / N) * |gap_b|` over non-empty bins. NaN if none qualify."""
    nonzero = count > 0
    if not np.any(nonzero):
        return float("nan")
    total = float(count[nonzero].sum())
    return float(np.sum((count[nonzero] / total) * np.abs(gap[nonzero])))


def _mce_from_bins(count: np.ndarray, gap: np.ndarray, min_count: int) -> float:
    """`max_b |gap_b|` over bins with `count_b >= min_count`. NaN if none qualify."""
    qualifying = count >= min_count
    if not np.any(qualifying):
        return float("nan")
    return float(np.max(np.abs(gap[qualifying])))


def _sum_squared_error(prob_flat: Tensor, label_flat: Tensor) -> float:
    """`sum((prob - label) ** 2)`, accumulated in float64.

    Squaring and summing in float32 (`prob_flat`/`label_flat`'s native
    dtype) is precise enough for any one call, but `CalibrationAccumulator`
    calls this once per case and sums the results across a whole split --
    and `brier_score` itself is called once over a full concatenated split
    in the streaming-vs-one-shot regression test. Casting to float64 here
    is what makes those two code paths agree to a tight tolerance rather
    than drifting apart by float32 rounding that depends on how the same
    voxels happened to be chunked.
    """
    diff = (prob_flat - label_flat).to(torch.float64)
    return float(torch.sum(diff * diff))


def _nanmean_or_nan(values: Sequence[float]) -> float:
    """`np.nanmean`, but returns NaN (silently) instead of warning on an all-NaN input."""
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _reliability_frame(
    lower: np.ndarray,
    upper: np.ndarray,
    count: np.ndarray,
    mean_prob: np.ndarray,
    mean_label: np.ndarray,
    gap: np.ndarray,
) -> pd.DataFrame:
    """Assembles the standard reliability-diagram DataFrame, columns in the documented order."""
    return pd.DataFrame(
        {
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count.astype(np.int64),
            "mean_prob": mean_prob,
            "mean_label": mean_label,
            "gap": gap,
        }
    )


# ---------------------------------------------------------------------------
# Public free functions
# ---------------------------------------------------------------------------


def reliability_curve(
    prob: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    mask: Tensor | np.ndarray | None = None,
) -> pd.DataFrame:
    """Bins predicted probability against observed positive frequency.

    See this module's top-of-file docstring for the probability-vs-frequency
    convention (not confidence-vs-accuracy).

    Empty mask: if `mask` selects zero voxels (a region legitimately absent
    from a case -- normal in BraTS, must not abort a whole-set evaluation
    loop), this logs a warning and returns an EMPTY-BUT-WELL-FORMED table:
    all `n_bins` rows present, `count = 0`, `mean_prob`/`mean_label`/`gap`
    NaN. A table with silently missing bins would misrepresent the
    reliability diagram it feeds.

    Args:
        prob: Predicted probabilities, any shape, values in `[0, 1]`.
        label: Binary ground truth, same shape as `prob`.
        n_bins: Number of equal-width bins over `[0, 1]`.
        mask: Optional boolean-like array/tensor broadcastable to `prob`'s
            shape; only `True`/nonzero voxels are included.

    Returns:
        DataFrame with exactly `n_bins` rows (every bin present, including
        empty ones), columns `bin_lower, bin_upper, count, mean_prob,
        mean_label, gap` in that order. `gap = mean_label - mean_prob`
        (signed: negative means overconfident).

    Raises:
        ValueError: `prob`/`label` shape mismatch, zero elements, `prob`
            outside `[0, 1]`, or `label` not binary. NOT raised when `mask`
            (as opposed to the raw input) selects zero elements.
    """
    edges = bin_edges(n_bins)
    lower = edges[:-1].numpy()
    upper = edges[1:].numpy()

    prob_flat, label_flat = _prepare_flat(prob, label, mask=mask)

    if prob_flat.numel() == 0:
        logger.warning(
            "reliability_curve: mask selected zero voxels; returning an empty-but-well-formed "
            "%d-bin table (all counts 0). Normal when a region is entirely absent from a case.",
            n_bins,
        )
        zeros = np.zeros(n_bins, dtype=np.float64)
        nans = np.full(n_bins, np.nan)
        return _reliability_frame(lower, upper, zeros, nans, nans, nans)

    count, sum_prob, sum_label = _bin_stats(prob_flat, label_flat, n_bins)
    mean_prob, mean_label, gap = _bin_means(count, sum_prob, sum_label)
    return _reliability_frame(lower, upper, count, mean_prob, mean_label, gap)


def expected_calibration_error(
    prob: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    mask: Tensor | np.ndarray | None = None,
) -> float:
    """`sum_b (count_b / N) * |gap_b|` over non-empty bins.

    Args:
        prob: Predicted probabilities, any shape, values in `[0, 1]`.
        label: Binary ground truth, same shape as `prob`.
        n_bins: Number of equal-width bins.
        mask: Optional boolean-like mask, see `reliability_curve`.

    Returns:
        A plain Python float. NaN if `mask` selects zero voxels (logged as
        a warning, not raised -- see `reliability_curve`).

    Raises:
        ValueError: See `reliability_curve`.
    """
    prob_flat, label_flat = _prepare_flat(prob, label, mask=mask)
    if prob_flat.numel() == 0:
        logger.warning("expected_calibration_error: mask selected zero voxels; returning NaN.")
        return float("nan")
    count, sum_prob, sum_label = _bin_stats(prob_flat, label_flat, n_bins)
    _, _, gap = _bin_means(count, sum_prob, sum_label)
    return _ece_from_bins(count, gap)


def maximum_calibration_error(
    prob: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    mask: Tensor | np.ndarray | None = None,
    min_count: int = 1,
) -> float:
    """`max_b |gap_b|` over bins with at least `min_count` voxels.

    MCE is dominated by whichever sparse bin happens to be noisiest -- a bin
    with a single voxel that happens to disagree with its own probability by
    chance produces `|gap| = 1.0`, swamping every well-populated bin. Raise
    `min_count` for reporting (this project's own use is with a value well
    above 1); the default of 1 matches the mathematical definition of MCE
    exactly, for callers who want that instead.

    Args:
        prob: Predicted probabilities, any shape, values in `[0, 1]`.
        label: Binary ground truth, same shape as `prob`.
        n_bins: Number of equal-width bins.
        mask: Optional boolean-like mask, see `reliability_curve`.
        min_count: Minimum voxel count for a bin to be eligible.

    Returns:
        A plain Python float. NaN if no bin has `count >= min_count`
        (including the `mask`-selects-zero-voxels case).

    Raises:
        ValueError: See `reliability_curve`.
    """
    prob_flat, label_flat = _prepare_flat(prob, label, mask=mask)
    if prob_flat.numel() == 0:
        logger.warning("maximum_calibration_error: mask selected zero voxels; returning NaN.")
        return float("nan")
    count, sum_prob, sum_label = _bin_stats(prob_flat, label_flat, n_bins)
    _, _, gap = _bin_means(count, sum_prob, sum_label)
    return _mce_from_bins(count, gap, min_count)


def brier_score(
    prob: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
    mask: Tensor | np.ndarray | None = None,
) -> float:
    """`mean((prob - label) ** 2)`, computed directly from the values (not from bins).

    Deliberately NOT derived from `reliability_curve`'s bins -- binning
    first would make this an approximation of the Brier score rather than
    the Brier score itself.

    Args:
        prob: Predicted probabilities, any shape, values in `[0, 1]`.
        label: Binary ground truth, same shape as `prob`.
        mask: Optional boolean-like mask, see `reliability_curve`.

    Returns:
        A plain Python float. NaN if `mask` selects zero voxels.

    Raises:
        ValueError: Shape mismatch, zero elements, `prob` outside `[0, 1]`,
            or `label` not binary.
    """
    prob_flat, label_flat = _prepare_flat(prob, label, mask=mask)
    if prob_flat.numel() == 0:
        logger.warning("brier_score: mask selected zero voxels; returning NaN.")
        return float("nan")
    return _sum_squared_error(prob_flat, label_flat) / prob_flat.numel()


def union_foreground_mask(
    prob: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
    threshold: float = 0.5,
) -> Tensor:
    """DIAGNOSTIC-ONLY mask: predicted-positive UNION ground-truth-positive. CIRCULAR -- read this.

    ## The circularity, and why it is not a theoretical nitpick

    A voxel with `p < threshold` can only enter this mask by having
    `label > 0` -- that is the only door in for it. So among the
    sub-threshold voxels this mask selects, `P(label = 1 | p < threshold, in
    mask)` is exactly 1 BY CONSTRUCTION, regardless of whether the model is
    any good. Confirmed on a real 189-case evaluation: every populated
    reliability bin below the threshold showed `mean_label == 1.000000`,
    for every region. That is arithmetic, not a measurement, and it is not a
    small effect: it was measured to contribute 41% (ET), 50% (TC), and 57%
    (WT) of the TOTAL reported ECE, it pushed a fitted temperature to
    `[4.58, 4.75, 3.71]` (a segmentation net normally needs 1.1-2.0), and
    after applying that temperature the WT ECE got WORSE (0.0730 -> 0.0776)
    -- because the fit was optimizing NLL against a contaminated
    distribution, not against the model's real behaviour.

    **Do not use this mask to compute a reported ECE/MCE/Brier number, and
    do not use it to fit a temperature that will be reported.** Use
    `predicted_foreground_mask` (label-free, cannot be circular by
    construction) or `brain_mask` instead. This function stays in the
    module because it is still a legitimate DIAGNOSTIC -- e.g. quantifying
    how much a false-negative-inclusive mask changes the picture -- never
    as the basis for a paper claim. Every call logs a WARNING naming this,
    so an accidental reporting use is visible in the run log rather than
    silently producing a wrong number.

    An earlier version of this docstring justified the mask by analogy to
    Dice: "Dice is computed against the ground truth too." That analogy is
    FALSE and is corrected here rather than repeated. Dice uses the label
    as the OBJECT of comparison -- prediction vs. label, a symmetric
    overlap measure that does not change which voxels are compared based on
    their own outcome. This mask instead uses the label to SELECT WHICH
    VOXELS ARE MEASURED, and then measures how well probability predicts
    that same label on the selected set -- selection on the dependent
    variable, a different and invalid maneuver that Dice does not commit.

    ## Why a mask is needed at all (this part of the original reasoning still holds)

    Roughly 99% of a BraTS volume is background the model is trivially
    certain about, so a whole-volume ECE is an average over voxels that no
    model gets wrong. Measured on a synthetic model that predicts `p = 0.99`
    on foreground while being right only 60% of the time -- catastrophic
    overconfidence -- whole-volume ECE is 0.0049 and foreground ECE is
    0.3896, an 80x difference. Reporting the whole-volume number would make
    every model, including a broken one, look perfectly calibrated. This is
    still true; it just does not license computing the restricted number
    with a mask that bakes the label into which voxels get measured. See
    `predicted_foreground_mask` and `brain_mask` for the label-free
    alternatives that solve the same 99%-background problem without this
    mask's circularity.

    Args:
        prob: Predicted probabilities, `(C, D, H, W)` or `(1, C, D, H, W)` or
            any shape matching `label`, values in `[0, 1]`.
        label: Binary ground truth, same shape as `prob`.
        threshold: Probability at or above which a voxel counts as predicted
            positive.

    Returns:
        A boolean CPU tensor, same shape as `prob`, True where the voxel is
        predicted positive OR labelled positive. Per-region (one channel per
        region), so it is accepted directly as `CalibrationAccumulator`'s
        per-region `mask` -- but see the warning above about what it must
        never be used for.

    Raises:
        ValueError: Shape mismatch, zero elements, `prob` outside `[0, 1]`,
            or `label` not binary.
    """
    prob_t = _to_cpu_float32(prob)
    label_t = _to_cpu_float32(label)
    _validate_prob_label(prob_t, label_t, check_prob_range=True)
    logger.warning(
        "union_foreground_mask: this mask is CIRCULAR for calibration reporting -- a voxel "
        "with p < threshold can only be included via label > 0, so its sub-threshold bins have "
        "mean_label == 1.0 by construction, not as a measurement. Do NOT use this to compute a "
        "reported ECE/MCE/Brier or to fit a reported temperature; it is diagnostic-only. Use "
        "predicted_foreground_mask or brain_mask for reporting instead. See this function's "
        "docstring for the full account."
    )
    return (prob_t >= threshold) | (label_t > 0.5)


def predicted_foreground_mask(
    prob: Tensor | np.ndarray,
    threshold: float = 0.5,
) -> Tensor:
    """The model's own predicted-positive voxels. Label-free, so it cannot be circular.

    `label` is not even a parameter here -- this mask depends only on
    `prob`, so `P(label = 1 | in mask)` computed over it is a genuine
    measurement of the model rather than an identity forced by the mask's
    own definition. That is the bug fixed in `union_foreground_mask`; see
    its docstring for the measured damage it caused.

    Known blind spot, and its mitigation. By construction this mask
    EXCLUDES every false negative -- a voxel where the model confidently
    said background (`p < threshold`) but tumour was actually present.
    Those voxels never enter this mask, so a calibration number computed
    only over it says nothing about how well-calibrated the model is on its
    confident misses. This is not left uncovered: those exact voxels are
    what `neurovision.metrics.boundary.boundary_stratified_errors`'s
    `bfnr_*` columns measure (false-negative rate stratified by distance to
    the ground-truth boundary) -- the blind spot is covered by a DIFFERENT
    analysis, rather than papered over by folding the label back into this
    mask, which is exactly what made `union_foreground_mask` circular in
    the first place.

    Args:
        prob: Predicted probabilities, any shape, values in `[0, 1]`.
        threshold: Probability at or above which a voxel counts as predicted
            positive. Matches `cfg.inference.postprocess.threshold`; pass the
            config value rather than relying on this default if the two ever
            diverge, or the reported calibration would cover a different
            voxel set than the reported Dice.

    Returns:
        A boolean CPU tensor, same shape as `prob`, True where
        `prob >= threshold`.

    Raises:
        ValueError: `prob` has zero elements, or contains a value outside
            `[0, 1]`.
    """
    prob_t = _to_cpu_float32(prob)
    if prob_t.numel() == 0:
        raise ValueError("prob has zero elements; nothing to compute a mask over.")
    p_min = float(prob_t.min())
    p_max = float(prob_t.max())
    if p_min < 0.0 or p_max > 1.0:
        raise ValueError(f"prob must contain values in [0, 1], got min={p_min} max={p_max}.")
    return prob_t >= threshold


def brain_mask(
    image: Tensor | np.ndarray | None = None,
    mask: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Selects the nonzero-intensity brain region of a preprocessed MRI volume. Label-free.

    Preprocessing (`neurovision.data.preprocessing.normalize_nonzero`)
    z-scores each modality over its OWN nonzero voxels and crops to the
    union nonzero bounding box, so brain INTERIORS are routinely NEGATIVE
    after normalization and exact zero is the marker for air, not for a
    low-intensity voxel. This mask is therefore derived from `image != 0`,
    unioned across the modality/channel axis -- **never `image > 0`**,
    which would silently drop every negative-valued (below-mean) brain
    voxel and misreport the mask as covering only half the brain.

    Args:
        image: The preprocessed MRI volume, shape `(C, D, H, W)` (one
            channel per modality, typically 4 for T1/T1CE/T2/FLAIR). The
            returned mask is the union of `image[c] != 0` over `c`, i.e. a
            voxel counts as brain if ANY modality has signal there.
        mask: An already-computed boolean mask. Use this when constructing
            the mask from `image` is impractical at the call site (e.g. the
            image was not loaded); it takes priority over `image` when both
            are given, and is returned as-is (cast to bool, on CPU).

    Returns:
        A boolean CPU tensor. Shape `(D, H, W)` when derived from `image`
        (the channel axis is reduced away by the union); whatever shape
        `mask` was given as when `mask` is used instead.

    Raises:
        ValueError: Neither `image` nor `mask` was given. There is
            deliberately NO silent fallback to a whole-volume all-True mask
            here -- returning "everything is brain" without saying so would
            defeat the purpose of a brain-restricted reporting mask with
            nothing failing anywhere to reveal it. Pass one of the two
            arguments explicitly.
    """
    if mask is not None:
        return torch.as_tensor(mask).detach().to(dtype=torch.bool, device="cpu")
    if image is None:
        raise ValueError(
            "brain_mask needs either `image` (the preprocessed (C, D, H, W) MRI volume, to "
            "derive the mask from `image != 0`) or `mask` (an already-computed boolean array). "
            "Falling back to a whole-volume all-True mask silently would defeat the purpose of "
            "a brain-restricted reporting mask."
        )
    image_t = _to_cpu_float32(image)
    return (image_t != 0.0).any(dim=0)


def subsample_voxels(
    prob: Tensor | np.ndarray,
    label: Tensor | np.ndarray,
    n_samples: int,
    generator: torch.Generator,
    mask: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """Uniformly subsamples voxels without replacement, for memory-safe temperature fitting.

    `fit_temperature` needs its `logits` held in memory as one dense tensor
    (LBFGS is not a streaming optimizer), and a 189-case test split at
    roughly 3.3M voxels per case per channel does not fit. Subsampling a
    fixed number of voxels per case and concatenating across cases is the
    memory-safe route to a temperature fit.

    Args:
        prob: Predicted probabilities, any shape, values in `[0, 1]`.
        label: Binary ground truth, same shape as `prob`.
        n_samples: Number of voxels to draw.
        generator: An explicit `torch.Generator` -- required, no default and
            no use of the global RNG, so a caller looping over an entire
            split gets a reproducible, non-interfering draw per case.
        mask: Optional boolean-like mask, see `reliability_curve`; sampling
            is restricted to the masked (flattened) voxels.

    Returns:
        `(prob_sample, label_sample)`, 1-D float32 CPU tensors of length
        `n_samples`, or fewer if fewer voxels than that are available (ALL
        available voxels are returned in that case -- never padded, never
        an error).

    Raises:
        ValueError: `prob`/`label` shape mismatch, zero elements (before
            masking), `prob` outside `[0, 1]`, or `label` not binary.
    """
    prob_flat, label_flat = _prepare_flat(prob, label, mask=mask)
    n_available = prob_flat.numel()

    if n_available < n_samples:
        logger.debug(
            "subsample_voxels: only %d voxels available, fewer than the %d requested; "
            "returning all of them.",
            n_available,
            n_samples,
        )
        return prob_flat, label_flat

    perm = torch.randperm(n_available, generator=generator)[:n_samples]
    return prob_flat[perm], label_flat[perm]


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


@dataclass
class TemperatureResult:
    """The result of one `fit_temperature` call.

    Attributes:
        temperature: The fitted temperature. Shape `(C,)` when
            `per_channel=True` (`(1,)` if the input was 1-D), shape `()`
            (a 0-D scalar tensor) when `per_channel=False`.
        nll_before: Mean BCE (negative log-likelihood) at `T = 1`.
        nll_after: Mean BCE at the fitted temperature.
        converged: `nll_after <= nll_before`. If False, the optimizer
            diverged and the fitted temperature must NOT be used.
    """

    temperature: Tensor
    nll_before: float
    nll_after: float
    converged: bool


def fit_temperature(
    logits: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    per_channel: bool = True,
    max_iter: int = 100,
    lr: float = 0.01,
) -> TemperatureResult:
    """Fits a temperature `T` minimizing `BCEWithLogitsLoss(logits / T, labels)`.

    IMPORTANT -- fit on VALIDATION, apply to TEST. Fitting and reporting the
    calibration number on the same split makes that number meaningless: `T`
    would be fit to that split's own noise, and this project's calibration
    claim is exactly the number this function's caller is not allowed to
    launder that way. Fit once on the validation split, freeze `T`, and
    apply it (via `apply_temperature`) when reporting the test split.

    IMPORTANT -- mask consistently. `logits`/`labels` here are expected to
    already be restricted to whatever voxels the reported calibration number
    covers (typically via `subsample_voxels(..., mask=...)` using the SAME
    mask used for reporting). Fitting on unmasked whole-volume voxels lets
    the ~99%-background majority of a BraTS volume decide `T`, which has
    nothing to do with the foreground calibration this project reports.

    `T = exp(log_T)` -- parametrized through a log so positivity is
    STRUCTURAL, not enforced by a clamp. A clamped-at-zero optimizer step
    would divide `logits` by zero and return an all-NaN result with no
    error raised anywhere; `exp` can never reach zero or go negative, so
    there is no boundary for an optimizer step to cross.

    Optimizer: `torch.optim.LBFGS` on a single `log_T` vector. When
    `per_channel`, all channels are fit JOINTLY in one `(C,)`-shaped `log_T`
    over the SUMMED per-channel loss -- mathematically equivalent to `C`
    independent per-channel fits (the channels' losses are additively
    separable, so a joint minimum is exactly the tuple of per-channel
    minima), done in one LBFGS run instead of `C`.

    Args:
        logits: Raw (pre-sigmoid) model output, shape `(N,)` or `(N, C)`.
            NOTE: these are LOGITS, so the `[0, 1]` range check other
            functions in this module apply does NOT apply here.
        labels: Binary ground truth, same shape as `logits`.
        per_channel: If True (default) and `logits` is 2-D, fits one `T`
            per channel. If `logits` is 1-D, treated as a single channel
            regardless of this flag, and the result is shape `(1,)`.
        max_iter: `LBFGS`'s `max_iter` (number of optimizer iterations
            within the single `step` call below).
        lr: `LBFGS`'s learning rate.

    Returns:
        A `TemperatureResult`. `converged` is False (with a WARNING logged)
        when `nll_after > nll_before` -- treat the fitted `T` as unusable in
        that case.

    Raises:
        ValueError: `logits`/`labels` shape mismatch, zero elements, labels
            not binary, or `logits.ndim` not in `{1, 2}`.
    """
    logits_t = _to_cpu_float32(logits)
    labels_t = _to_cpu_float32(labels)
    _validate_prob_label(
        logits_t, labels_t, check_prob_range=False, name_a="logits", name_b="labels"
    )

    if logits_t.ndim not in (1, 2):
        raise ValueError(
            "fit_temperature expects logits shaped (N,) or (N, C), got ndim="
            f"{logits_t.ndim} (shape {tuple(logits_t.shape)})."
        )

    is_1d = logits_t.ndim == 1
    logits_2d = logits_t.unsqueeze(1) if is_1d else logits_t  # (N, C)
    labels_2d = labels_t.unsqueeze(1) if is_1d else labels_t
    num_channels = logits_2d.shape[1]
    n_temps = num_channels if per_channel else 1

    bce = nn.BCEWithLogitsLoss()

    def _nll(log_t_vec: Tensor) -> Tensor:
        temperature = torch.exp(log_t_vec)  # structural positivity, see docstring above
        temp_view = temperature.view(1, num_channels) if per_channel else temperature.view(1, 1)
        return bce(logits_2d / temp_view, labels_2d)

    with torch.no_grad():
        nll_before = float(_nll(torch.zeros(n_temps)))

    log_t = torch.zeros(n_temps, dtype=torch.float32, requires_grad=True)
    # line_search_fn="strong_wolfe": without it, LBFGS's default fixed-step-size
    # behaviour is very sensitive to `lr` under this log-parametrization (measured:
    # lr=0.01 without a line search stalls at T~1.49 on a case whose true answer is
    # T=2.0; the same problem converges to T~2.01 with strong_wolfe at that same lr).
    # The line search adapts the step size per-iteration instead of trusting `lr`
    # directly, which is what makes the fit robust to `lr` rather than requiring the
    # caller to hand-tune it per dataset.
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = _nll(log_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        nll_after = float(_nll(log_t))
        temperature = torch.exp(log_t.detach())
    if not per_channel:
        temperature = temperature.squeeze(0)  # shape () -- one shared scalar temperature

    converged = nll_after <= nll_before
    if not converged:
        logger.warning(
            "fit_temperature: nll_after (%.6f) > nll_before (%.6f) -- the optimizer diverged. "
            "Do not use this fitted temperature.",
            nll_after,
            nll_before,
        )

    return TemperatureResult(
        temperature=temperature, nll_before=nll_before, nll_after=nll_after, converged=converged
    )


def apply_temperature(logits: Tensor, temperature: Tensor) -> Tensor:
    """Returns `logits / temperature`, broadcasting along the channel axis.

    Temperature scaling is STRICTLY MONOTONE (dividing by a positive
    constant never changes the sign of a logit, hence never changes which
    side of the `p = 0.5` threshold a voxel falls on), so it changes NO
    binary prediction and therefore NO Dice or HD95 -- it only moves the
    reported PROBABILITIES. That is the entire point of the technique. It
    is also a built-in correctness check: a temperature-scaled row in a
    results table showing a DIFFERENT Dice from its unscaled row means
    there is a bug somewhere in the pipeline, not a genuine effect of
    temperature scaling.

    Args:
        logits: Raw model output, shape `(B, C, D, H, W)` or `(N, C)`.
        temperature: A 0-D (scalar, shared across all channels) or 1-D
            `(C,)` (per-channel) tensor, e.g. `TemperatureResult.temperature`.

    Returns:
        `logits / temperature`, same shape and device as `logits`.

    Raises:
        ValueError: `temperature` has more than 1 dimension, or its
            per-channel size disagrees with `logits`'s channel axis.
    """
    temp = torch.as_tensor(temperature).to(dtype=logits.dtype, device=logits.device)

    if temp.ndim == 0:
        return logits / temp
    if temp.ndim != 1:
        raise ValueError(
            f"temperature must be 0-D (scalar) or 1-D (per-channel), got shape {tuple(temp.shape)}."
        )

    num_channels = temp.shape[0]
    if logits.ndim == 5:
        if logits.shape[1] != num_channels:
            raise ValueError(
                f"apply_temperature: logits has {logits.shape[1]} channels but temperature has "
                f"{num_channels} entries. Sizes must agree along the channel axis (dim=1)."
            )
        temp_view = temp.view(1, num_channels, 1, 1, 1)
    elif logits.ndim == 2:
        if logits.shape[1] != num_channels:
            raise ValueError(
                f"apply_temperature: logits has {logits.shape[1]} channels but temperature has "
                f"{num_channels} entries. Sizes must agree along the channel axis (dim=1)."
            )
        temp_view = temp.view(1, num_channels)
    else:
        raise ValueError(
            "apply_temperature expects logits shaped (B, C, D, H, W) or (N, C), got ndim="
            f"{logits.ndim} (shape {tuple(logits.shape)})."
        )

    return logits / temp_view


# ---------------------------------------------------------------------------
# Streaming accumulator
# ---------------------------------------------------------------------------


class CalibrationAccumulator:
    """Collects calibration statistics across a validation/test set, one case at a time.

    Mirrors `neurovision.metrics.segmentation.MetricAggregator`'s ergonomics:
    call `add_case` once per case during evaluation, then read `per_case()`
    for a per-case table (an uncertainty-vs-error scatter needs per-case
    ECE) and `summary()` for the pooled, set-level numbers a report or W&B
    log wants.
    """

    def __init__(
        self, n_bins: int = DEFAULT_N_BINS, region_names: Sequence[str] = REGION_NAMES
    ) -> None:
        """Initializes an empty accumulator.

        Args:
            n_bins: Number of equal-width bins over `[0, 1]`, used for every
                region and for both the streaming and per-case metrics.
            region_names: Region name per channel, in channel order.
                Defaults to `REGION_NAMES` (`("ET", "TC", "WT")`).
        """
        self.n_bins = n_bins
        self.region_names = list(region_names)
        self.reset()

    def add_case(
        self,
        case_id: str,
        prob: Tensor | np.ndarray,
        label: Tensor | np.ndarray,
        mask: Tensor | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Folds one case into the running set-level bins and returns its OWN metrics.

        Args:
            case_id: Unique identifier for the case (e.g. BraTS case name).
            prob: Predicted probabilities for ONE case, shape `(C, D, H, W)`
                or `(1, C, D, H, W)`, `C == len(self.region_names)`.
            label: Binary ground truth, same shape convention as `prob`.
            mask: Optional boolean-like mask. Either per-region
                (`(C, D, H, W)`/`(1, C, D, H, W)`, sliced alongside `prob`)
                or shared across regions (e.g. `(D, H, W)`, broadcast to
                each region in turn) -- distinguished by whether its
                leading (post-batch) axis equals `len(self.region_names)`.

        Returns:
            A flat dict: `ece_<REGION>`, `mce_<REGION>`, `brier_<REGION>`
            for each region, plus NaN-skipping `ece_mean`, `brier_mean`.
            This dict is stored and is exactly what `per_case()` returns.
            If a region has zero unmasked voxels for this case, its three
            metrics are NaN (logged as a warning) rather than raising --
            normal when a region is entirely absent from a case.

        Raises:
            ValueError: `case_id` was already added; `prob`'s channel count
                does not equal `len(self.region_names)`; or shape/range/
                binary validation fails (see `_validate_prob_label`).
        """
        if case_id in self._case_ids:
            raise ValueError(
                f"case_id {case_id!r} was already added to this CalibrationAccumulator. A "
                "duplicate would silently overwrite the earlier case's calibration metrics."
            )

        prob_t = _to_cpu_float32(prob)
        label_t = _to_cpu_float32(label)
        if prob_t.ndim == 5:
            if prob_t.shape[0] != 1:
                raise ValueError(
                    "CalibrationAccumulator.add_case is for a single case (batch size 1), got "
                    f"prob batch {prob_t.shape[0]}."
                )
            prob_t = prob_t[0]
        if label_t.ndim == 5:
            if label_t.shape[0] != 1:
                raise ValueError(
                    "CalibrationAccumulator.add_case is for a single case (batch size 1), got "
                    f"label batch {label_t.shape[0]}."
                )
            label_t = label_t[0]
        if prob_t.ndim != 4 or label_t.ndim != 4:
            raise ValueError(
                "add_case expects prob/label shaped (C, D, H, W) or (1, C, D, H, W), got prob "
                f"{tuple(prob_t.shape)} and label {tuple(label_t.shape)}."
            )
        if prob_t.shape[0] != len(self.region_names):
            raise ValueError(
                f"add_case: prob has {prob_t.shape[0]} channels but region_names has "
                f"{len(self.region_names)} entries {tuple(self.region_names)}. One channel per "
                "region name is required."
            )
        if prob_t.shape != label_t.shape:
            raise ValueError(
                f"prob and label must have the same shape, got {tuple(prob_t.shape)} and "
                f"{tuple(label_t.shape)}."
            )

        mask_t: Tensor | None = None
        if mask is not None:
            mask_t = torch.as_tensor(mask).detach().to(dtype=torch.bool, device="cpu")
            if mask_t.ndim == 5:
                mask_t = mask_t[0]
            # A per-region mask (leading axis == number of regions) is sliced alongside
            # prob/label below; anything else is treated as shared across every region
            # and left for torch.broadcast_to (inside _prepare_flat) to expand.
        per_region_mask = (
            mask_t is not None and mask_t.ndim == 4 and mask_t.shape[0] == len(self.region_names)
        )

        case_metrics: dict[str, float] = {}
        for i, region in enumerate(self.region_names):
            region_mask = mask_t[i] if per_region_mask else mask_t
            prob_flat, label_flat = _prepare_flat(prob_t[i], label_t[i], mask=region_mask)

            if prob_flat.numel() == 0:
                logger.warning(
                    "CalibrationAccumulator.add_case(%s): region %s has zero unmasked voxels; "
                    "its metrics for this case are NaN.",
                    case_id,
                    region,
                )
                case_metrics[f"ece_{region}"] = float("nan")
                case_metrics[f"mce_{region}"] = float("nan")
                case_metrics[f"brier_{region}"] = float("nan")
                continue

            count, sum_prob, sum_label = _bin_stats(prob_flat, label_flat, self.n_bins)
            _, _, gap = _bin_means(count, sum_prob, sum_label)
            case_sum_sq_error = _sum_squared_error(prob_flat, label_flat)
            case_metrics[f"ece_{region}"] = _ece_from_bins(count, gap)
            case_metrics[f"mce_{region}"] = _mce_from_bins(count, gap, min_count=1)
            case_metrics[f"brier_{region}"] = case_sum_sq_error / prob_flat.numel()

            # Fold into the running SET-LEVEL accumulators. float64, not float32: these
            # accumulate over up to ~10^9 voxels across a full evaluation split, and
            # float32 loses low-order bits badly at that magnitude.
            self._bin_count[region] += count
            self._bin_sum_prob[region] += sum_prob
            self._bin_sum_label[region] += sum_label
            self._sum_sq_error[region] += case_sum_sq_error
            self._n_total[region] += int(prob_flat.numel())

        case_metrics["ece_mean"] = _nanmean_or_nan(
            [case_metrics[f"ece_{r}"] for r in self.region_names]
        )
        case_metrics["brier_mean"] = _nanmean_or_nan(
            [case_metrics[f"brier_{r}"] for r in self.region_names]
        )

        self._case_ids.append(case_id)
        self._case_records.append(case_metrics)
        return case_metrics

    def reliability(self, region: str) -> pd.DataFrame:
        """The SET-LEVEL (pooled over every case added so far) reliability table for `region`.

        Args:
            region: One of `self.region_names`.

        Returns:
            Same columns as `reliability_curve`: `bin_lower, bin_upper,
            count, mean_prob, mean_label, gap`.

        Raises:
            ValueError: `region` is not in `self.region_names`.
        """
        if region not in self.region_names:
            raise ValueError(
                f"Unknown region {region!r}; valid regions are {tuple(self.region_names)}."
            )
        edges = bin_edges(self.n_bins)
        lower = edges[:-1].numpy()
        upper = edges[1:].numpy()
        count = self._bin_count[region]
        mean_prob, mean_label, gap = _bin_means(
            count, self._bin_sum_prob[region], self._bin_sum_label[region]
        )
        return _reliability_frame(lower, upper, count, mean_prob, mean_label, gap)

    def per_case(self) -> pd.DataFrame:
        """Returns every stored case's OWN metrics as a table.

        Returns:
            DataFrame indexed by `case_id`, columns `ece_<REGION>`,
            `mce_<REGION>`, `brier_<REGION>` per region plus `ece_mean`,
            `brier_mean`. Empty (but valid) DataFrame if no cases were
            added.
        """
        if not self._case_records:
            return pd.DataFrame()
        return pd.DataFrame(self._case_records, index=pd.Index(self._case_ids, name="case_id"))

    def summary(self) -> pd.DataFrame:
        """Summarizes calibration across every stored case, POOLED over all voxels.

        NOT the mean of `per_case()`'s columns, and the two are genuinely
        different quantities: this pooled summary weights every VOXEL
        equally (so a case with a large tumor dominates it), while
        `per_case().mean()` weights every CASE equally. Both are legitimate
        things to report; the paper must state which one it uses.

        Returns:
            DataFrame indexed by metric name (`ece_<REGION>`, `mce_<REGION>`,
            `brier_<REGION>`, `ece_mean`, `brier_mean`, `n_cases`,
            `n_voxels`) with a single `value` column. `n_voxels` is the
            total number of (voxel, region) pairs folded in across every
            region -- i.e. each voxel is counted once per region it
            contributed to, since each region is a separate binary
            calibration problem over the same spatial grid. Empty (but
            valid) DataFrame if no cases were added.
        """
        if not self._case_ids:
            return pd.DataFrame()

        rows: dict[str, float] = {}
        ece_values: list[float] = []
        brier_values: list[float] = []
        for region in self.region_names:
            count = self._bin_count[region]
            mean_prob, mean_label, gap = _bin_means(
                count, self._bin_sum_prob[region], self._bin_sum_label[region]
            )
            del mean_prob, mean_label  # only gap is needed for ECE/MCE
            ece = _ece_from_bins(count, gap)
            mce = _mce_from_bins(count, gap, min_count=1)
            n_total = self._n_total[region]
            brier = float(self._sum_sq_error[region] / n_total) if n_total > 0 else float("nan")

            rows[f"ece_{region}"] = ece
            rows[f"mce_{region}"] = mce
            rows[f"brier_{region}"] = brier
            ece_values.append(ece)
            brier_values.append(brier)

        rows["ece_mean"] = _nanmean_or_nan(ece_values)
        rows["brier_mean"] = _nanmean_or_nan(brier_values)

        summary_df = pd.DataFrame({"value": pd.Series(rows)})
        summary_df.loc["n_cases"] = float(len(self._case_ids))
        summary_df.loc["n_voxels"] = float(sum(self._n_total[r] for r in self.region_names))
        return summary_df

    def reset(self) -> None:
        """Clears all stored cases and running bin accumulators."""
        self._bin_count: dict[str, np.ndarray] = {
            r: np.zeros(self.n_bins, dtype=np.float64) for r in self.region_names
        }
        self._bin_sum_prob: dict[str, np.ndarray] = {
            r: np.zeros(self.n_bins, dtype=np.float64) for r in self.region_names
        }
        self._bin_sum_label: dict[str, np.ndarray] = {
            r: np.zeros(self.n_bins, dtype=np.float64) for r in self.region_names
        }
        self._sum_sq_error: dict[str, float] = dict.fromkeys(self.region_names, 0.0)
        self._n_total: dict[str, int] = dict.fromkeys(self.region_names, 0)
        self._case_ids: list[str] = []
        self._case_records: list[dict[str, float]] = []

    def __len__(self) -> int:
        """Returns the number of cases stored so far."""
        return len(self._case_ids)
