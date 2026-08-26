"""Conformal risk control for the segmentation mask's miss rate.

## What this module answers

The model's predicted mask at the usual `p >= 0.5` threshold is a POINT
ESTIMATE: it is the network's single best guess at the tumour boundary, and
nothing about it says how much of the true tumour that guess could plausibly
be missing. This module answers a narrower, more useful question: **at what
probability threshold `tau` does the CONSERVATIVE mask `{v : p(v) >= tau}`
miss, ON AVERAGE, no more than an `alpha` fraction of the true tumour?**
Lowering `tau` below 0.5 only ever grows the mask (it can only include MORE
voxels, never fewer), so it is a dial that trades mask size for a
false-negative guarantee, and this module picks the least conservative
setting of that dial that still keeps the promise.

## The guarantee, and what it does NOT say

Given a calibration set of `n` cases assumed EXCHANGEABLE with whatever case
the mask is later applied to (informally: the calibration cases and the
future case could have been drawn in any order without changing anything
about the setup -- no information distinguishes "calibration" from "future"
except which one happened to be measured first), `fit_threshold` picks a
threshold `tau_hat` such that

    E[ L_test(tau_hat) ] <= alpha

where `L_test` is the false-negative rate (see below) on a FRESH case drawn
from the same exchangeable population. Two things about this that are easy
to misread:

- **It is a bound on an EXPECTATION, over the random draw of calibration AND
  test data -- not a promise about any one case.** A specific future case
  can still miss far more than `alpha` of its tumour; the guarantee is that
  such cases are rare enough, and the good cases good enough, that the
  AVERAGE stays under `alpha`. This is exactly analogous to a 95% confidence
  interval: 95% of intervals built this way contain the truth, but any one
  interval either does or does not, full stop.
- **It says nothing about the mask being good, small, or clinically
  useful.** A model that is confidently wrong everywhere still gets a valid
  guarantee -- `fit_threshold` will simply be forced to pick a very
  permissive (low) `tau`, producing a very large, very inflated mask, in
  order to keep the false-negative rate down. THIS IS BY DESIGN: the
  guarantee is a property of the CALIBRATION PROCEDURE (Angelopoulos, Bates,
  Fisch, Lei, Schuster, *Conformal Risk Control*, ICLR 2024), not of the
  model, and it holds for an arbitrarily bad model precisely because the
  procedure adapts `tau` to however bad the model turns out to be on the
  calibration set. That adaptation is also why `band_inflation` below is not
  optional decoration: a "safe" mask that is bought by predicting the whole
  brain is worthless, and that function is what would expose it.

## The loss, and why threshold direction matters

For case `i`, predicted probability map `p_i`, ground truth `G_i`, and
threshold `tau`:

    L_i(tau) = | G_i \\ M_i(tau) | / | G_i |,   M_i(tau) = {v : p_i(v) >= tau}

the fraction of the true tumour the conservative mask MISSES -- a clinician's
question ("how much of the tumour could this mask have missed?"), bounded in
`[0, 1]` with no need to clip, and NON-DECREASING in `tau`: lowering `tau`
can only add voxels to `M_i(tau)`, so it can only remove voxels from the
false-negative set, never add to it. That monotonicity is exactly what the
theorem needs; the ICLR paper states it for a loss that DECREASES as its
threshold parameter increases and takes an infimum over the feasible set.
Our `tau` runs the opposite way (bigger `tau` means a stingier mask means
MORE misses, i.e. HIGHER loss), so the same rule becomes a MAXIMUM over the
feasible set instead of a minimum -- see `fit_threshold` for the exact
selection rule, and note the feasible set is a down-set in `tau` (every
`tau` below a feasible one is also feasible, because lowering `tau` can only
lower the risk), which is what makes "take the largest feasible `tau`" a
well-defined, unambiguous choice.

## Empty ground truth is 0/0, not 0 and not 1

When `|G_i| = 0` (no tumour of this region present in this case -- routine
for ET, where roughly 2.6% of BraTS cases have none), `L_i(tau)` is
undefined. The two tempting conventions -- score it 0 ("nothing missed") or
score it 1 ("everything missed") -- are each defensible and each would move
the calibrated `tau` and the reported risk in a different direction. This
module picks neither: such cases are EXCLUDED from every mean and the
excluded count is always reported alongside, so a reader can see exactly how
many cases were dropped and decide for themselves whether that matters.

## Why the expensive step (`case_loss_curve`) happens exactly once per case

`CaseLossCurve` stores, for a fixed grid of thresholds, three arrays of a
few hundred integers each: the ground-truth voxel count and, per threshold,
the false-negative voxel count and the mask voxel count. That is the
SUFFICIENT STATISTIC for everything this module computes -- the miss rate at
any grid threshold, the mask-inflation ratio at any grid threshold,
recalibration at any `alpha` -- all reduce to arithmetic over these small
arrays. The one pass over full-resolution voxel data (computing a
comparison at every threshold in the grid, for a volume that can hold
~10^7 voxels) happens once per case, in `case_loss_curve`; every later
analysis in this module, including refitting at a different `alpha` or
re-deriving the risk on a different subset of cases, is then cheap array
arithmetic over stored integers rather than a re-scan of the volume.

## Why the default grid is dense near zero

The interesting thresholds for a CONSERVATIVE mask are almost always far
below the ordinary `0.5` operating point -- the pre-registration's own
feasibility probe found a case still missing 39% of its tumour core at
`tau = 0.01`. A uniform grid over `[0, 1]` would place almost no points in
`[0, 0.1]`, resolving exactly the region the fitted threshold is likely to
land in about as coarsely as possible. `DEFAULT_THRESHOLDS` therefore
spends 31 points on a log-spaced sweep of `[1e-4, 1e-1]` (four decades near
zero) and a further 35 points on an ordinary linear sweep of `[0.1, 0.95]`
covering the usual operating range, including `0.5` exactly (see
`mask_inflation`'s default reference threshold, and the grid-construction
test that pins this down).

## Randomness

This module draws from no random-number generator anywhere -- `fit_threshold`
is a deterministic function of the input loss curves and `alpha`. (The
pre-registration's bootstrap confidence intervals around the REALISED risk
live in `neurovision.analysis.statistics`, seeded there, not here.)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import Tensor

__all__ = [
    "DEFAULT_THRESHOLDS",
    "CaseLossCurve",
    "case_loss_curve",
    "load_curves_npz",
    "ConformalFit",
    "fit_threshold",
    "realised_risk",
    "band_inflation",
]

logger = logging.getLogger(__name__)


def _build_default_thresholds() -> tuple[float, ...]:
    """Builds the fixed conformal threshold grid. See module docstring for why it is shaped this
    way.

    Returns:
        A sorted, strictly increasing `tuple[float, ...]`: 31 log-spaced
        points over `[1e-4, 1e-1]` unioned with 35 linear points over
        `[0.1, 0.95]` (which includes `0.5` exactly), rounded to 12 decimals
        before de-duplicating so floating-point noise does not produce
        near-duplicate grid points.
    """
    fine = np.logspace(-4, -1, 31)  # dense where a conservative mask actually lives
    operating = np.linspace(0.1, 0.95, 35)  # the ordinary point-estimate range
    grid = np.unique(np.round(np.concatenate([fine, operating]), 12))
    return tuple(float(t) for t in grid)


DEFAULT_THRESHOLDS: tuple[float, ...] = _build_default_thresholds()


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """Normalizes a tensor-or-array entry point to a CPU numpy array.

    Args:
        x: Input, possibly a CUDA tensor.

    Returns:
        A numpy array with the same values as `x`. Never assumes CUDA is
        present or absent -- a `Tensor` is always detached and moved to CPU
        first; a plain `np.ndarray` is passed through `np.asarray`.
    """
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _threshold_index(thresholds: np.ndarray, value: float, tol: float = 1e-9) -> int:
    """Finds the index of `value` in `thresholds`, exactly (to `tol`).

    Args:
        thresholds: 1-D array of grid thresholds.
        value: The threshold to look up.
        tol: Maximum allowed absolute difference to count as a match.

    Returns:
        The index of the matching entry.

    Raises:
        ValueError: No entry within `tol` of `value` exists. The message
            names the nearest available value.
    """
    diffs = np.abs(thresholds - value)
    idx = int(np.argmin(diffs))
    if diffs[idx] > tol:
        raise ValueError(
            f"{value} is not in the threshold grid; nearest available value is "
            f"{thresholds[idx]!r}."
        )
    return idx


# ---------------------------------------------------------------------------
# CaseLossCurve: the sufficient statistic, computed once per case
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseLossCurve:
    """The per-case, per-region false-negative and mask-size curve over a fixed threshold grid.

    See the module docstring's "Why the expensive step happens exactly once
    per case" section -- everything downstream (`miss_rate`, `mask_inflation`,
    `fit_threshold`, `realised_risk`, `band_inflation`) derives from these
    three small arrays rather than re-scanning the volume.

    Attributes:
        case_id: Identifier for the case this curve was computed from.
        region: The region name (e.g. `"ET"`, `"TC"`, `"WT"`) this curve
            covers. One `CaseLossCurve` per (case, region) pair.
        gt_voxels: Total ground-truth positive voxel count, `|G_i|`.
        thresholds: The threshold grid this curve was evaluated on, sorted
            strictly increasing.
        fn_voxels: False-negative voxel count at each threshold in
            `thresholds`, same length and order.
        mask_voxels: Predicted-mask voxel count at each threshold in
            `thresholds`, same length and order.
    """

    case_id: str
    region: str
    gt_voxels: int
    thresholds: tuple[float, ...]
    fn_voxels: tuple[int, ...]
    mask_voxels: tuple[int, ...]

    @property
    def empty_gt(self) -> bool:
        """True when this case/region has no ground-truth-positive voxels at all.

        See the module docstring's "Empty ground truth is 0/0" section --
        `miss_rate` is undefined (not 0, not 1) for such cases, and every
        downstream mean excludes them and counts how many were excluded.
        """
        return self.gt_voxels == 0

    def miss_rate(self) -> np.ndarray:
        """The false-negative rate `fn_voxels / gt_voxels` at every grid threshold.

        Returns:
            A float64 array of shape `(T,)`, `T = len(self.thresholds)`.
            All-NaN when `self.empty_gt` (the loss is genuinely undefined,
            0/0, at every threshold -- never silently scored as 0 or 1).
        """
        if self.empty_gt:
            return np.full(len(self.thresholds), np.nan, dtype=np.float64)
        fn = np.asarray(self.fn_voxels, dtype=np.float64)
        return fn / self.gt_voxels

    def mask_inflation(self, reference_threshold: float = 0.5) -> np.ndarray:
        """The mask-size ratio `mask_voxels(tau) / mask_voxels(reference_threshold)`.

        Args:
            reference_threshold: The threshold whose mask size is the
                denominator -- must be present in `self.thresholds` exactly
                (to floating-point tolerance).

        Returns:
            A float64 array of shape `(T,)`. NaN at every entry when the
            reference mask itself is empty (0/0) -- callers that need to
            skip such cases (e.g. `band_inflation`) do so by checking for
            NaN, rather than this method raising.

        Raises:
            ValueError: `reference_threshold` is not in `self.thresholds`.
                The message names the nearest available value.
        """
        thresholds_arr = np.asarray(self.thresholds, dtype=np.float64)
        ref_idx = _threshold_index(thresholds_arr, reference_threshold)
        ref_voxels = self.mask_voxels[ref_idx]
        mask = np.asarray(self.mask_voxels, dtype=np.float64)
        if ref_voxels == 0:
            return np.full(len(self.thresholds), np.nan, dtype=np.float64)
        return mask / ref_voxels


def case_loss_curve(
    prob: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    *,
    case_id: str,
    region: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> CaseLossCurve:
    """Computes the sufficient-statistic loss curve for one case and one region.

    Args:
        prob: Per-voxel predicted probabilities (already sigmoid-ed) for ONE
            case and ONE region, any shape, any float dtype, any device.
        target: Binary ground truth, same shape as `prob`.
        case_id: Identifier for this case, stored on the result.
        region: Region name (e.g. `"ET"`, `"TC"`, `"WT"`), stored on the
            result.
        thresholds: The threshold grid to evaluate at, strictly increasing.
            Defaults to `DEFAULT_THRESHOLDS`.

    Returns:
        A `CaseLossCurve` holding `gt_voxels`, and `fn_voxels`/`mask_voxels`
        at every threshold in `thresholds`.

    Raises:
        ValueError: `prob` and `target` shapes disagree, `thresholds` is
            empty, or `thresholds` is not strictly increasing.
    """
    prob_np = _to_numpy(prob)
    target_np = _to_numpy(target)
    if prob_np.shape != target_np.shape:
        raise ValueError(
            f"prob and target must have the same shape, got {prob_np.shape} and {target_np.shape}."
        )

    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    if thresholds_arr.size == 0:
        raise ValueError("thresholds must not be empty.")
    if not np.all(np.diff(thresholds_arr) > 0):
        raise ValueError("thresholds must be strictly increasing.")

    target_bool = target_np.astype(bool)
    gt_voxels = int(target_bool.sum())

    fn_voxels: list[int] = []
    mask_voxels: list[int] = []
    for tau in thresholds_arr:
        mask = prob_np >= tau
        mask_voxels.append(int(mask.sum()))
        fn_voxels.append(int(np.logical_and(target_bool, ~mask).sum()))

    return CaseLossCurve(
        case_id=case_id,
        region=region,
        gt_voxels=gt_voxels,
        thresholds=tuple(float(t) for t in thresholds_arr),
        fn_voxels=tuple(fn_voxels),
        mask_voxels=tuple(mask_voxels),
    )


# ---------------------------------------------------------------------------
# Reloading a cached curves.npz -- a public reader, because scripts/ is not
# an importable package
# ---------------------------------------------------------------------------


def load_curves_npz(path: Path, regions: Sequence[str]) -> dict[str, list[CaseLossCurve]]:
    """Reloads a `curves.npz` cache (written by `scripts/conformal.py`) into `CaseLossCurve`s.

    `scripts/conformal.py::_write_curves_npz` is the only writer of this file format, and
    that script keeps its own private, underscore-prefixed reader
    (`_load_curves_npz`) for its own use. `scripts/` is not an importable package
    (`pyproject.toml`'s `packages.find` only looks under `src/`), so a second caller that
    also needs to read a `curves.npz` -- today, `neurovision.analysis.
    gatekeeper_calibration`, which reads a previously-extracted conformal run's curves to
    build the gatekeeper's calibration table -- cannot import that private function
    across the script boundary, the same problem `neurovision.analysis.qc_inference`'s
    module docstring already describes and solves for the QC case. This function is that
    same fix, applied here: a small, PUBLIC reader living next to `CaseLossCurve` itself,
    so a second caller reads exactly the same format through exactly the same logic
    instead of maintaining a second, drifting copy of it.

    Args:
        path: Path to a `curves.npz` file, in EXACTLY the layout
            `scripts/conformal.py::_write_curves_npz` writes: a shared `"thresholds"`
            float64 array, plus, for every region `R` in `regions`, `f"{R}__case_ids"`,
            `f"{R}__gt_voxels"` (int64), `f"{R}__fn_voxels"` (int64, shape `(n_cases,
            n_thresholds)`) and `f"{R}__mask_voxels"` (int64, same shape).
        regions: Region names to read back, e.g. `("WT", "TC")`. Must match (or be a
            subset of) the regions the file was written with -- a region absent from
            the file raises `KeyError` via the missing `.npz` array lookup.

    Returns:
        `{region: [CaseLossCurve, ...]}`, one `CaseLossCurve` per case saved under that
        region, in the same order they were saved, all sharing the file's threshold
        grid.
    """
    data = np.load(path)
    thresholds = tuple(float(t) for t in data["thresholds"])
    curves: dict[str, list[CaseLossCurve]] = {}
    for region in regions:
        case_ids = data[f"{region}__case_ids"]
        gt = data[f"{region}__gt_voxels"]
        fn = data[f"{region}__fn_voxels"]
        mask = data[f"{region}__mask_voxels"]
        curves[region] = [
            CaseLossCurve(
                case_id=str(case_ids[i]),
                region=region,
                gt_voxels=int(gt[i]),
                thresholds=thresholds,
                fn_voxels=tuple(int(v) for v in fn[i]),
                mask_voxels=tuple(int(v) for v in mask[i]),
            )
            for i in range(len(case_ids))
        ]
    return curves


# ---------------------------------------------------------------------------
# fit_threshold: the conformal risk control procedure itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformalFit:
    """The result of one `fit_threshold` call.

    Attributes:
        alpha: The target risk level this fit was run at.
        region: The region these curves cover (all curves must share one).
        n_calibration: Number of curves actually used (non-empty-GT).
        n_excluded_empty: Number of curves dropped for `empty_gt`.
        feasible: Whether any grid threshold satisfies the conformal
            inequality at this `alpha`. False is a legitimate, expected
            outcome -- see the module and pre-registration for why.
        threshold: The largest feasible threshold, or `None` if infeasible.
        calibrated_risk: `risk_curve` evaluated at `threshold`, or `None`
            if infeasible.
        min_achievable_risk: The adjusted risk at the SMALLEST grid
            threshold -- reported unconditionally, feasible or not, so an
            infeasible result still carries how far off it was.
        thresholds: The shared threshold grid the curves were evaluated on.
        risk_curve: Mean miss rate (over the used curves) at each threshold
            in `thresholds`, same order.
    """

    alpha: float
    region: str
    n_calibration: int
    n_excluded_empty: int
    feasible: bool
    threshold: float | None
    calibrated_risk: float | None
    min_achievable_risk: float
    thresholds: tuple[float, ...]
    risk_curve: tuple[float, ...]


def fit_threshold(
    curves: Sequence[CaseLossCurve],
    alpha: float,
    *,
    bound: float = 1.0,
    monotonicity_tol: float = 1e-9,
) -> ConformalFit:
    """Selects the conformal risk control threshold `tau_hat` at level `alpha`.

    Implements
    `tau_hat = max { tau in grid : (n * R_hat(tau) + bound) / (n + 1) <= alpha }`,
    where `R_hat(tau)` is the mean miss rate over the non-empty-GT curves.
    Because `R_hat` is non-decreasing in `tau` (see module docstring), the
    adjusted quantity is too, so the feasible set is a down-set in `tau` and
    "the largest feasible `tau`" is unambiguous -- found here by taking the
    highest-index grid point that still satisfies the inequality.

    Args:
        curves: One `CaseLossCurve` per calibration case, all for the same
            region and evaluated on the identical threshold grid.
        alpha: Target risk level, must be in `(0, 1]`.
        bound: The loss's known upper bound, `B` in the formula above.
            Defaults to `1.0` (the miss rate is a fraction, so it is always
            in `[0, 1]`).
        monotonicity_tol: Tolerance for the monotonicity falsifier check
            below -- a diff more negative than `-monotonicity_tol` is
            treated as a genuine violation rather than floating-point noise.

    Returns:
        A `ConformalFit`. See its docstring for what `feasible=False` means
        and why it is not an error.

    Raises:
        ValueError: `curves` is empty; all curves are `empty_gt`; the
            curves do not all share an identical `thresholds` tuple; the
            curves do not all share the same `region`; `alpha` is not in
            `(0, 1]`; or the mean risk curve is not non-decreasing in
            `tau` beyond `monotonicity_tol` (the pre-registration's
            monotonicity falsifier -- a violation means the loss or the
            threshold direction is wired backwards, and must be reported
            as an implementation bug, not an analysis finding).
    """
    if len(curves) == 0:
        raise ValueError("curves must not be empty.")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")

    thresholds = curves[0].thresholds
    region = curves[0].region
    for c in curves:
        if c.thresholds != thresholds:
            raise ValueError(
                "all curves must share an identical thresholds tuple; got a mismatch involving "
                f"case_id={c.case_id!r}."
            )
        if c.region != region:
            raise ValueError(
                f"all curves must share the same region; expected {region!r}, got {c.region!r} "
                f"for case_id={c.case_id!r}."
            )

    used = [c for c in curves if not c.empty_gt]
    n_excluded_empty = len(curves) - len(used)
    n_calibration = len(used)
    if n_calibration == 0:
        raise ValueError(
            f"all {len(curves)} curves have empty ground truth for region {region!r}; there is "
            "nothing to calibrate a threshold on."
        )

    miss_rates = np.stack([c.miss_rate() for c in used], axis=0)  # (n_calibration, T)
    risk_curve = miss_rates.mean(axis=0)

    diffs = np.diff(risk_curve)
    bad_indices = np.where(diffs < -monotonicity_tol)[0]
    if bad_indices.size > 0:
        bad_idx = int(bad_indices[0])
        raise ValueError(
            "risk_curve is not non-decreasing in threshold -- the monotonicity falsifier failed "
            f"at index {bad_idx}: risk_curve[{bad_idx}]={risk_curve[bad_idx]!r} > "
            f"risk_curve[{bad_idx + 1}]={risk_curve[bad_idx + 1]!r}. This means the loss or the "
            "threshold direction is wired backwards; see the pre-registration's falsifier list."
        )

    n = n_calibration
    adjusted = (n * risk_curve + bound) / (n + 1)
    min_achievable_risk = float(adjusted[0])

    feasible_mask = adjusted <= alpha
    if np.any(feasible_mask):
        idx = int(np.max(np.where(feasible_mask)[0]))
        threshold: float | None = float(thresholds[idx])
        calibrated_risk: float | None = float(risk_curve[idx])
        feasible = True
    else:
        threshold = None
        calibrated_risk = None
        feasible = False
        logger.warning(
            "fit_threshold: infeasible at alpha=%.4f for region %s (n=%d) -- even the smallest "
            "grid threshold has adjusted risk %.4f > alpha. Reporting infeasible, not a "
            "silently clipped threshold; see min_achievable_risk.",
            alpha,
            region,
            n,
            min_achievable_risk,
        )

    return ConformalFit(
        alpha=alpha,
        region=region,
        n_calibration=n_calibration,
        n_excluded_empty=n_excluded_empty,
        feasible=feasible,
        threshold=threshold,
        calibrated_risk=calibrated_risk,
        min_achievable_risk=min_achievable_risk,
        thresholds=thresholds,
        risk_curve=tuple(float(x) for x in risk_curve),
    )


# ---------------------------------------------------------------------------
# Reporting: realised risk and the mandatory mask-inflation cost
# ---------------------------------------------------------------------------


def realised_risk(
    curves: Sequence[CaseLossCurve],
    threshold: float,
) -> dict[str, float]:
    """Mean miss rate at a fixed `threshold`, over an evaluation (or held-out) set of curves.

    No bootstrap or confidence interval here -- that lives in
    `neurovision.analysis.statistics`, which already owns resampling and
    must not have it duplicated.

    Args:
        curves: One `CaseLossCurve` per case, all sharing the identical
            threshold grid that `threshold` must be a member of.
        threshold: The threshold to evaluate the miss rate at (typically a
            `ConformalFit.threshold` fitted elsewhere and applied frozen).

    Returns:
        `{"mean_miss_rate": float, "n": float, "n_excluded_empty": float}`
        -- the mean over non-empty-GT curves, and how many were excluded.
        `mean_miss_rate` is NaN if every curve was empty-GT.

    Raises:
        ValueError: `curves` is empty, the curves do not share an identical
            thresholds tuple, or `threshold` is not in that grid.
    """
    if len(curves) == 0:
        raise ValueError("curves must not be empty.")
    thresholds = curves[0].thresholds
    for c in curves:
        if c.thresholds != thresholds:
            raise ValueError(
                "all curves must share an identical thresholds tuple; got a mismatch involving "
                f"case_id={c.case_id!r}."
            )
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    idx = _threshold_index(thresholds_arr, threshold)

    values: list[float] = []
    n_excluded_empty = 0
    for c in curves:
        if c.empty_gt:
            n_excluded_empty += 1
            continue
        values.append(float(c.miss_rate()[idx]))

    n = len(values)
    mean_miss_rate = float(np.mean(values)) if n > 0 else float("nan")

    return {
        "mean_miss_rate": mean_miss_rate,
        "n": float(n),
        "n_excluded_empty": float(n_excluded_empty),
    }


def band_inflation(
    curves: Sequence[CaseLossCurve],
    threshold: float,
    *,
    reference_threshold: float = 0.5,
) -> dict[str, float]:
    """The mandatory cost-of-the-guarantee table: how much the conservative mask inflates.

    See the module docstring: a bound bought by predicting the whole brain
    is worthless, and this is the check that prevents reporting one as a
    success. Cases whose reference (`reference_threshold`) mask is empty
    are skipped (the ratio is 0/0) rather than propagated as NaN into the
    mean.

    Args:
        curves: One `CaseLossCurve` per case, all sharing the identical
            threshold grid that `threshold` and `reference_threshold` must
            each be a member of.
        threshold: The conservative threshold (typically the fitted
            `ConformalFit.threshold`) whose mask size is the numerator.
        reference_threshold: The point-estimate threshold whose mask size
            is the denominator. Defaults to `0.5`.

    Returns:
        `{"mean_inflation": float, "median_inflation": float, "n": float,
        "n_skipped": float}` over cases with a non-empty reference mask.
        Both statistics are NaN if every case was skipped.

    Raises:
        ValueError: `curves` is empty, the curves do not share an identical
            thresholds tuple, or `threshold`/`reference_threshold` is not
            in that grid.
    """
    if len(curves) == 0:
        raise ValueError("curves must not be empty.")
    thresholds = curves[0].thresholds
    for c in curves:
        if c.thresholds != thresholds:
            raise ValueError(
                "all curves must share an identical thresholds tuple; got a mismatch involving "
                f"case_id={c.case_id!r}."
            )
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    idx = _threshold_index(thresholds_arr, threshold)
    # Validate reference_threshold is in the grid up front so a bad value fails fast rather
    # than surfacing only once every single case happens to have an empty reference mask.
    _threshold_index(thresholds_arr, reference_threshold)

    inflations: list[float] = []
    n_skipped = 0
    for c in curves:
        value = float(c.mask_inflation(reference_threshold=reference_threshold)[idx])
        if np.isnan(value):
            n_skipped += 1
            continue
        inflations.append(value)

    n = len(inflations)
    if n == 0:
        mean_inflation = float("nan")
        median_inflation = float("nan")
    else:
        mean_inflation = float(np.mean(inflations))
        median_inflation = float(np.median(inflations))

    return {
        "mean_inflation": mean_inflation,
        "median_inflation": median_inflation,
        "n": float(n),
        "n_skipped": float(n_skipped),
    }
