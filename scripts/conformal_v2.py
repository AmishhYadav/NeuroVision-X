"""Hydra driver for conformal v2: CRC + RCPS, three models, three regions, with the mask cost.

Specification: `docs/research_docs/preregistrations/preregistration_conformal_v2.md` (read it
first; it is the contract). Statistical building blocks live in
`neurovision.uncertainty.conformal` (v2 API at the end of that file); the v1 driver
`scripts/conformal.py` supplies curve extraction and is imported, never copied or changed.

Steps, in order (each is a small function so it can be tested alone):

1. Curves: per model and cohort, loss curves on the EXTENDED threshold grid (from saved logits),
   or, for a model without logits, the saved v1 curves (WT/TC only, v1 grid, raw mask).
   Anything that cannot be measured is written to `not_measured.csv` as
   "not_measured_needs_W2.6", never fabricated.
2. Additivity check (runs before any v2 number is written, raises `RuntimeError` on a mismatch):
   v2 curves restricted to the v1 grid must reproduce v1's `fit.json` and `realised_risk.csv`.
3. Fits on BraTS val: CRC and RCPS per model x region x alpha -> `fits.csv`.
4. Frozen application of each val threshold to test / SSA / PED -> `realised.csv`
   (bootstrap CI, verdict, mask cost).
5. Re-splits (in distribution): pool val + test, many random halvings, fit on one half and read
   realised risk on the other -> `resplits.csv`, `resplits_per_split.csv.gz`.
6. Deployed post-processing (logit models only): the fitted threshold is also pushed through
   `postprocess_logits` -> `postprocessed.csv`.
7. `run_meta.json`.

NOTE on step 6: tau-hat is fitted on RAW curves only, because the conformal theorem needs the
risk to be monotone in tau and the small-component filter is not. The post-processed numbers
are therefore DESCRIPTIVE (what the deployed mask does at that threshold), not guaranteed.

CPU only, one case's logits in memory at a time. Run (after the protocol is externally
registered):

    python scripts/conformal_v2.py
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import logging
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.replay import _load_label_and_spacing, load_case_logits
from neurovision.analysis.statistics import paired_bootstrap_ci
from neurovision.inference.postprocess import postprocess_logits
from neurovision.metrics.segmentation import classes_to_regions
from neurovision.uncertainty.conformal import (
    DEFAULT_THRESHOLDS,
    EXTENDED_THRESHOLDS,
    CaseLossCurve,
    fit_threshold,
    fit_threshold_crc,
    fit_threshold_rcps,
    load_curves_npz,
    mask_quality,
    realised_risk,
    realised_risk_with,
    wsr_upper_bound,
)
from neurovision.utils.io import ensure_dir, read_json, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
_SCRIPTS_DIR = Path(__file__).resolve().parent

COHORTS: tuple[str, ...] = ("val", "test", "ssa", "ped")
APPLY_COHORTS: tuple[str, ...] = ("test", "ssa", "ped")  # val is the calibration set
GUARANTEES: tuple[str, ...] = ("crc", "rcps")
REGION_ORDER: tuple[str, ...] = ("ET", "TC", "WT")  # channel order of the saved logits
_V1_REGIONS: tuple[str, ...] = ("WT", "TC")  # the regions v1 fitted (ET was excluded)
_NOT_MEASURED = "not_measured_needs_W2.6"
_LOG_EVERY = 25
_ADDITIVITY_TOL = 1e-9
_PP_LOGGER = "neurovision.inference.postprocess"  # logs one INFO line per call; far too chatty


# ---------------------------------------------------------------------------
# Importing the v1 driver (scripts/ is not a package)
# ---------------------------------------------------------------------------


def _load_v1_driver() -> Any:
    """Loads `scripts/conformal.py` by path, reusing an already-loaded copy if present."""
    name = "conformal_script"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / "conformal.py")
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError("conformal_v2: cannot import scripts/conformal.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V1 = _load_v1_driver()
extract_curves = _V1.extract_curves  # cached curve extraction, v1's own code
resolve_curves_dir = _V1.resolve_curves_dir


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolves a config path against `repo_root` (absolute paths pass through unchanged)."""
    return Path(repo_root) / Path(str(path))


def _grid_index(thresholds: Sequence[float], value: float, tol: float = 1e-9) -> int:
    """Index of `value` in a threshold grid, exactly to `tol`; raises `ValueError` if absent."""
    arr = np.asarray(thresholds, dtype=np.float64)
    idx = int(np.argmin(np.abs(arr - value)))
    if abs(arr[idx] - value) > tol:
        raise ValueError(f"{value} is not in the threshold grid (nearest {arr[idx]!r}).")
    return idx


def _is_close(a: float | None, b: float | None, tol: float = _ADDITIVITY_TOL) -> bool:
    """True if both are None, or both are numbers within `tol`."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def _nan_to_none(x: float | None) -> float | None:
    """Maps NaN to None (so JSON stays valid)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(x)


@contextlib.contextmanager
def _quiet_logger(name: str) -> Iterator[None]:
    """Temporarily raises a logger to WARNING so a per-case INFO line does not flood the log."""
    target = logging.getLogger(name)
    old = target.level
    target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        target.setLevel(old)


def restrict_curves(
    curves: Sequence[CaseLossCurve], thresholds: Sequence[float] = DEFAULT_THRESHOLDS
) -> list[CaseLossCurve]:
    """Restricts curves to a sub-grid by picking the matching columns.

    Args:
        curves: Curves sharing one grid that contains every value in `thresholds`.
        thresholds: The sub-grid (default: the v1 grid).

    Returns:
        New `CaseLossCurve`s on the sub-grid, same order.

    Raises:
        ValueError: A sub-grid threshold is missing from the curves' grid.
    """
    if not curves:
        return []
    idx = [_grid_index(curves[0].thresholds, t) for t in thresholds]
    sub = tuple(float(curves[0].thresholds[i]) for i in idx)
    return [
        CaseLossCurve(
            case_id=c.case_id,
            region=c.region,
            gt_voxels=c.gt_voxels,
            thresholds=sub,
            fn_voxels=tuple(c.fn_voxels[i] for i in idx),
            mask_voxels=tuple(c.mask_voxels[i] for i in idx),
        )
        for c in curves
    ]


def loss_matrix(curves: Sequence[CaseLossCurve], empty: str) -> np.ndarray:
    """Per-case miss rates at every threshold: shape `(n_cases, T)`.

    Rows of empty-reference cases are all-NaN under `empty="exclude"` and all-zero under
    `"zero"`.
    """
    return np.stack([c.miss_rate_with(empty) for c in curves], axis=0)


# ---------------------------------------------------------------------------
# Step 1: curves
# ---------------------------------------------------------------------------


@dataclass
class ModelCurves:
    """Everything the later steps need to know about one model.

    Attributes:
        name: Model key from the config.
        curves: `{cohort: {region: [CaseLossCurve, ...]}}`.
        grid: `"extended"` or `"v1"`.
        has_logits: Whether post-processing can run (needs saved logits).
        eval_dirs: `{cohort: Path}` for logit models, else empty.
        regions: Regions actually measured.
        not_measured: Rows for `not_measured.csv`.
    """

    name: str
    curves: dict[str, dict[str, list[CaseLossCurve]]]
    grid: str
    has_logits: bool
    eval_dirs: dict[str, Path] = field(default_factory=dict)
    regions: list[str] = field(default_factory=list)
    not_measured: list[dict[str, str]] = field(default_factory=list)


def load_model_curves(
    name: str,
    model_cfg: Mapping[str, Any],
    block: Mapping[str, Any],
    out_dir: Path,
    repo_root: Path,
) -> ModelCurves:
    """Loads (or extracts and caches) one model's curves for all four cohorts.

    A model with `eval_dirs` is extracted from its saved logits on `EXTENDED_THRESHOLDS`
    (cached by `extract_curves`' manifest). A model with `curves` is read from the saved v1
    `curves.npz` files, only for the regions present in them.

    Args:
        name: Model key.
        model_cfg: That model's config entry (`eval_dirs` or `curves`).
        block: The `conformal_v2` config block.
        out_dir: The run's output directory.
        repo_root: Root against which relative config paths resolve.

    Returns:
        A `ModelCurves`.
    """
    regions = [str(r) for r in block["regions"]]
    prep_dirs = {c: _resolve(repo_root, block["prep_dirs"][c]) for c in COHORTS}
    curves: dict[str, dict[str, list[CaseLossCurve]]] = {}

    if "eval_dirs" in model_cfg:
        eval_dirs = {c: _resolve(repo_root, model_cfg["eval_dirs"][c]) for c in COHORTS}
        for cohort in COHORTS:
            logger.info("curves: %s / %s (extended grid) from %s", name, cohort, eval_dirs[cohort])
            curves[cohort] = extract_curves(
                eval_dirs[cohort],
                prep_dirs[cohort],
                out_dir / "curves" / name,
                regions=regions,
                thresholds=EXTENDED_THRESHOLDS,
            )
        return ModelCurves(name, curves, "extended", True, eval_dirs, regions)

    # Curves-only model: read whichever configured regions the file actually holds.
    present: list[str] = []
    for cohort in COHORTS:
        path = _resolve(repo_root, model_cfg["curves"][cohort])
        with np.load(path) as data:
            here = [r for r in regions if f"{r}__case_ids" in data.files]
        present = here if not present else [r for r in present if r in here]
        curves[cohort] = load_curves_npz(path, here)
    missing = [r for r in regions if r not in present]
    not_measured = [
        {"model": name, "item": f"region:{r}", "status": _NOT_MEASURED} for r in missing
    ] + [
        {"model": name, "item": "extended_grid", "status": _NOT_MEASURED},
        {"model": name, "item": "postprocessed", "status": _NOT_MEASURED},
    ]
    for cohort in COHORTS:
        curves[cohort] = {r: curves[cohort][r] for r in present}
    return ModelCurves(name, curves, "v1", False, {}, present, not_measured)


# ---------------------------------------------------------------------------
# Step 2: additivity check
# ---------------------------------------------------------------------------


def check_additivity(
    model: ModelCurves,
    v1_dir: Path,
    alphas: Sequence[float],
    regions: Sequence[str] = _V1_REGIONS,
) -> list[dict[str, Any]]:
    """Compares v2 curves restricted to the v1 grid against v1's saved fit and realised risk.

    Args:
        model: A logit model (extended grid) with a v1 conformal directory.
        v1_dir: v1's output dir holding `fit.json` and `realised_risk.csv`.
        alphas: Alpha levels to compare.
        regions: Regions v1 fitted (WT and TC).

    Returns:
        One dict per comparison with `ok` (bool); nothing is raised here so the caller can
        write every comparison to disk before failing.
    """
    fit_json = read_json(Path(v1_dir) / "fit.json")
    realised = pd.read_csv(Path(v1_dir) / "realised_risk.csv")
    realised["_name"] = realised["apply_dir"].map(lambda p: Path(str(p)).name)
    rows: list[dict[str, Any]] = []

    restricted = {
        cohort: {r: restrict_curves(model.curves[cohort][r]) for r in regions} for cohort in COHORTS
    }
    for region in regions:
        for alpha in alphas:
            v1 = fit_json[f"{region}__alpha_{alpha}"]
            fit = fit_threshold(restricted["val"][region], alpha)
            ok = (
                fit.feasible == bool(v1["feasible"])
                and _is_close(fit.threshold, v1["threshold"])
                and _is_close(fit.calibrated_risk, v1["calibrated_risk"])
            )
            rows.append(
                {
                    "model": model.name,
                    "kind": "fit",
                    "cohort": "val",
                    "region": region,
                    "alpha": alpha,
                    "v1_threshold": v1["threshold"],
                    "v2_threshold": fit.threshold,
                    "v1_value": v1["calibrated_risk"],
                    "v2_value": fit.calibrated_risk,
                    "ok": bool(ok),
                }
            )
            if not fit.feasible or fit.threshold is None:
                continue  # v1 wrote no realised-risk row for an infeasible fit
            for cohort in APPLY_COHORTS:
                eval_name = model.eval_dirs[cohort].name
                match = realised[
                    (realised["_name"] == eval_name)
                    & (realised["region"] == region)
                    & np.isclose(realised["alpha"], alpha)
                ]
                v2_mean = realised_risk(restricted[cohort][region], fit.threshold)["mean_miss_rate"]
                v1_mean = float(match["mean_miss_rate"].iloc[0]) if len(match) == 1 else None
                rows.append(
                    {
                        "model": model.name,
                        "kind": "realised_risk",
                        "cohort": cohort,
                        "region": region,
                        "alpha": alpha,
                        "v1_threshold": float(match["threshold"].iloc[0]) if len(match) else None,
                        "v2_threshold": fit.threshold,
                        "v1_value": v1_mean,
                        "v2_value": v2_mean,
                        "ok": bool(
                            len(match) == 1
                            and _is_close(v1_mean, v2_mean)
                            and _is_close(float(match["threshold"].iloc[0]), fit.threshold)
                        ),
                    }
                )
    return rows


def raise_on_additivity_failure(rows: Sequence[Mapping[str, Any]]) -> None:
    """Raises `RuntimeError` naming the first comparison with `ok == False`."""
    for row in rows:
        if not row["ok"]:
            raise RuntimeError(
                "conformal_v2 additivity check FAILED -- v2 curves restricted to the v1 grid do "
                f"not reproduce v1: model={row['model']} kind={row['kind']} "
                f"cohort={row['cohort']} region={row['region']} alpha={row['alpha']} "
                f"(v1 threshold={row['v1_threshold']}, v2 threshold={row['v2_threshold']}, "
                f"v1 value={row['v1_value']}, v2 value={row['v2_value']}). Nothing downstream "
                "is valid until this is resolved."
            )


# ---------------------------------------------------------------------------
# Step 3: fits on val
# ---------------------------------------------------------------------------


def fit_cell_rows(
    model: str,
    region: str,
    curves: Sequence[CaseLossCurve],
    *,
    alphas: Sequence[float],
    delta: float,
    order_seed: int,
    empty: str,
    grid: str,
) -> list[dict[str, Any]]:
    """CRC and RCPS fits for one model x region, one row per alpha x guarantee.

    Args:
        model: Model key (for the row).
        region: Region name.
        curves: Calibration (val) curves for that region.
        alphas: Alpha levels.
        delta: RCPS confidence level.
        order_seed: RCPS case-permutation seed.
        empty: `"zero"` or `"exclude"`.
        grid: `"extended"` or `"v1"` (recorded in the row).

    Returns:
        Row dicts; `threshold` is None for an infeasible fit.
    """
    ceiling = max(curves[0].thresholds)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        crc = fit_threshold_crc(curves, alpha, empty=empty)
        rcps = fit_threshold_rcps(curves, alpha, delta, order_seed=order_seed, empty=empty)
        base = {"model": model, "region": region, "alpha": alpha, "empty": empty, "grid": grid}
        rows.append(
            {
                **base,
                "guarantee": "crc",
                "feasible": crc.feasible,
                "threshold": crc.threshold,
                "calibrated_risk": crc.calibrated_risk,
                "ucb_at_threshold": float("nan"),
                "min_achievable": crc.min_achievable_risk,
                "min_ucb": float("nan"),
                "n_calibration": crc.n_calibration,
                "n_excluded_empty": crc.n_excluded_empty,
                "at_grid_ceiling": crc.threshold is not None and crc.threshold == ceiling,
            }
        )
        rows.append(
            {
                **base,
                "guarantee": "rcps",
                "feasible": rcps.feasible,
                "threshold": rcps.threshold,
                "calibrated_risk": rcps.calibrated_risk,
                "ucb_at_threshold": rcps.ucb_at_threshold,
                "min_achievable": float("nan"),
                "min_ucb": rcps.min_ucb,
                "n_calibration": rcps.n_calibration,
                "n_excluded_empty": rcps.n_excluded_empty,
                "at_grid_ceiling": rcps.threshold is not None and rcps.threshold == ceiling,
            }
        )
    return rows


def threshold_lookup(fits: pd.DataFrame) -> dict[tuple[str, str, float, str], float | None]:
    """`{(model, region, alpha, guarantee): tau_hat or None}` from `fits.csv`'s table."""
    out: dict[tuple[str, str, float, str], float | None] = {}
    for r in fits.itertuples(index=False):
        out[(r.model, r.region, float(r.alpha), r.guarantee)] = (
            None if pd.isna(r.threshold) else float(r.threshold)
        )
    return out


# ---------------------------------------------------------------------------
# Step 4: frozen application (risk, CI, verdict, mask cost)
# ---------------------------------------------------------------------------


def cell_verdict(cohort: str, point: float, ci_lo: float, ci_hi: float, alpha: float) -> str:
    """Verdict for one frozen cell.

    `test` (in distribution; v1's B1 rule): "holds" if the point estimate is <= alpha, else
    "exceeds". `ssa` / `ped` (v1's B2 rule): "VIOLATED" if the CI lower bound > alpha, "holds"
    if the upper bound <= alpha, else "inconclusive" (also when the CI is undefined).
    """
    if cohort == "test":
        return "holds" if point <= alpha else "exceeds"
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "inconclusive"
    if ci_lo > alpha:
        return "VIOLATED"
    if ci_hi <= alpha:
        return "holds"
    return "inconclusive"


def _nan_summary(values: np.ndarray) -> tuple[float, float, int]:
    """(mean, median, n_nan) ignoring NaNs; mean and median are NaN if nothing is left."""
    finite = values[~np.isnan(values)]
    n_nan = int(values.size - finite.size)
    if finite.size == 0:
        return float("nan"), float("nan"), n_nan
    return float(finite.mean()), float(np.median(finite)), n_nan


def mask_cost_stats(curves: Sequence[CaseLossCurve], threshold: float) -> dict[str, float]:
    """Mean and median precision, Dice and volume ratio at `threshold`, plus NaN counts.

    Uses `mask_quality` (so the empty-reference conventions are the protocol's); NaNs (e.g.
    precision when the mask is empty, volume ratio when the reference is empty) are excluded
    from the mean/median and counted.
    """
    q = mask_quality(curves, threshold)
    out: dict[str, float] = {}
    for key in ("precision", "dice", "volume_ratio"):
        mean, median, n_nan = _nan_summary(q[key])
        out[f"{key}_mean"] = mean
        out[f"{key}_median"] = median
        out[f"{key}_n_nan"] = float(n_nan)
    return out


def evaluate_cell(
    curves: Sequence[CaseLossCurve],
    threshold: float,
    *,
    empty: str,
    n_boot: int,
    generator: np.random.Generator,
) -> dict[str, float]:
    """Realised risk, bootstrap 95% CI and mask cost of one frozen threshold.

    The CI reuses v1's recipe: `paired_bootstrap_ci(values, zeros)` is a single-sample
    bootstrap of the mean of `values` (the per-case miss rates under `empty`).

    Args:
        curves: Evaluation curves containing `threshold` in their grid.
        threshold: The frozen tau-hat.
        empty: Empty-reference convention.
        n_boot: Bootstrap resamples.
        generator: Seeded generator, advanced by the call.

    Returns:
        Dict with `mean_miss_rate`, `n`, `n_excluded_empty`, `ci_lo`, `ci_hi` and mask-cost keys.
    """
    risk = realised_risk_with(curves, threshold, empty=empty)
    idx = _grid_index(curves[0].thresholds, threshold)
    values = np.asarray(
        [float(c.miss_rate_with(empty)[idx]) for c in curves if empty == "zero" or not c.empty_gt],
        dtype=np.float64,
    )
    if values.size >= 2:
        boot = paired_bootstrap_ci(
            values, np.zeros_like(values), n_boot=n_boot, generator=generator
        )
        lo, hi = boot.lo, boot.hi
    else:
        lo = hi = float("nan")
    return {
        "mean_miss_rate": risk["mean_miss_rate"],
        "n": risk["n"],
        "n_excluded_empty": risk["n_excluded_empty"],
        "ci_lo": lo,
        "ci_hi": hi,
        **mask_cost_stats(curves, threshold),
    }


def build_realised_table(
    models: Sequence[ModelCurves],
    fits: pd.DataFrame,
    block: Mapping[str, Any],
) -> pd.DataFrame:
    """Applies every fitted val threshold, frozen, to test / SSA / PED.

    An infeasible fit gives a row with an empty threshold and verdict "infeasible".
    """
    generator = np.random.default_rng(int(block["boot_seed"]))
    n_boot = int(block["n_boot"])
    lookup = threshold_lookup(fits)
    rows: list[dict[str, Any]] = []
    for m in models:
        for cohort in APPLY_COHORTS:
            for region in m.regions:
                empty = str(block["empty"][region])
                for alpha in block["alphas"]:
                    for g in GUARANTEES:
                        tau = lookup[(m.name, region, float(alpha), g)]
                        base = {
                            "model": m.name,
                            "cohort": cohort,
                            "region": region,
                            "alpha": float(alpha),
                            "guarantee": g,
                            "empty": empty,
                            "grid": m.grid,
                        }
                        if tau is None:
                            rows.append(
                                {**base, "threshold": float("nan"), "verdict": "infeasible"}
                            )
                            continue
                        res = evaluate_cell(
                            m.curves[cohort][region],
                            tau,
                            empty=empty,
                            n_boot=n_boot,
                            generator=generator,
                        )
                        verdict = cell_verdict(
                            cohort, res["mean_miss_rate"], res["ci_lo"], res["ci_hi"], float(alpha)
                        )
                        rows.append({**base, "threshold": tau, **res, "verdict": verdict})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: re-splits
# ---------------------------------------------------------------------------


def make_perms(n_pool: int, n_resplits: int, seed: int) -> np.ndarray:
    """`n_resplits` permutations of `range(n_pool)` from ONE `default_rng(seed)` stream."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n_pool) for _ in range(n_resplits)], axis=0)


def rcps_indices(
    ordered: np.ndarray, alphas: Sequence[float], delta: float
) -> dict[float, int | None]:
    """RCPS threshold index for every alpha from ONE pass of upper bounds.

    Same rule as `fit_threshold_rcps` (largest tau such that the UCB is < alpha at every grid
    point up to it), but the UCB at each grid point is computed once and shared by all alphas,
    and the scan stops at the first point whose UCB is >= the largest alpha (every alpha has
    failed by then).

    Args:
        ordered: Losses in the permuted case order, shape `(n_cases, T)`.
        alphas: Alpha levels.
        delta: RCPS confidence level.

    Returns:
        `{alpha: grid index or None if infeasible}`.
    """
    n_thresholds = ordered.shape[1]
    max_alpha = max(alphas)
    first_fail = {a: n_thresholds for a in alphas}  # index of first failing tau, per alpha
    pending = set(alphas)
    for t in range(n_thresholds):
        ucb = wsr_upper_bound(ordered[:, t], delta)
        for a in list(pending):
            if ucb >= a:
                first_fail[a] = t
                pending.discard(a)
        if ucb >= max_alpha or not pending:
            break
    return {a: (first_fail[a] - 1 if first_fail[a] > 0 else None) for a in alphas}


def resplit_cell(
    losses: np.ndarray,
    perms: np.ndarray,
    alphas: Sequence[float],
    delta: float,
    order_seed: int,
) -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]:
    """Fits CRC and RCPS on one half of each permutation and reads risk on the other half.

    `losses` rows must be sorted by case id (as `fit_threshold_rcps` sorts by id before
    permuting); an all-NaN row marks an excluded empty-reference case.

    Args:
        losses: Pooled per-case losses, shape `(N, T)`.
        perms: Permutations, shape `(S, N)`; calibration = first `N // 2` entries of each,
            evaluation = the rest.
        alphas: Alpha levels.
        delta: RCPS confidence level.
        order_seed: RCPS case-order seed.

    Returns:
        `{(guarantee, alpha): (tau_index (S,) int with -1 = infeasible, realised (S,) float with
        NaN = infeasible)}`.
    """
    n_splits, n_pool = perms.shape
    half = n_pool // 2
    out = {
        (g, a): (np.full(n_splits, -1, dtype=int), np.full(n_splits, np.nan))
        for g in GUARANTEES
        for a in alphas
    }
    case_orders: dict[int, np.ndarray] = {}  # permutation of n_used cases, cached by n_used
    for s in range(n_splits):
        calib = losses[np.sort(perms[s, :half])]  # id order preserved, as the library does
        calib = calib[~np.isnan(calib[:, 0])]  # drop excluded (empty-reference) cases
        evaluation = losses[perms[s, half:]]
        evaluation = evaluation[~np.isnan(evaluation[:, 0])]
        if calib.shape[0] == 0 or evaluation.shape[0] == 0:
            continue
        n = calib.shape[0]
        adjusted = (n * calib.mean(axis=0) + 1.0) / (n + 1)  # CRC rule, bound = 1
        if n not in case_orders:
            case_orders[n] = np.random.default_rng(order_seed).permutation(n)
        rcps_idx = rcps_indices(calib[case_orders[n]], alphas, delta)
        eval_risk = evaluation.mean(axis=0)
        for a in alphas:
            feasible = np.where(adjusted <= a)[0]
            if feasible.size:
                idx = int(feasible.max())
                out[("crc", a)][0][s] = idx
                out[("crc", a)][1][s] = eval_risk[idx]
            r_idx = rcps_idx[a]
            if r_idx is not None:
                out[("rcps", a)][0][s] = r_idx
                out[("rcps", a)][1][s] = eval_risk[r_idx]
    return out


def summarise_resplits(
    realised: np.ndarray, alpha: float, guarantee: str, delta: float
) -> dict[str, Any]:
    """Per-cell re-split summary and the protocol's V1 verdict.

    CRC holds if mean realised risk <= alpha + 2 * MCSE(mean); RCPS holds if
    P(realised > alpha) <= delta + 2 * MCSE(P); otherwise FAILS. Infeasible splits (NaN) are
    counted, not scored as risk 0.
    """
    n_splits = int(realised.size)
    ok = realised[~np.isnan(realised)]
    n_feasible = int(ok.size)
    out: dict[str, Any] = {
        "n_resplits": n_splits,
        "n_feasible": n_feasible,
        "n_infeasible": n_splits - n_feasible,
    }
    if n_feasible == 0:
        out.update(
            dict.fromkeys(
                ("mean_realised", "mcse_mean", "p_exceed", "mcse_p", "q05", "q50", "q95"),
                float("nan"),
            )
        )
        out["verdict"] = "infeasible"
        return out
    mean = float(ok.mean())
    mcse_mean = float(ok.std(ddof=1) / np.sqrt(n_feasible)) if n_feasible > 1 else float("nan")
    p = float(np.mean(ok > alpha))
    mcse_p = float(np.sqrt(p * (1.0 - p) / n_feasible))
    q05, q50, q95 = (float(q) for q in np.quantile(ok, [0.05, 0.5, 0.95]))
    if guarantee == "crc":
        holds = mean <= alpha + 2.0 * (0.0 if np.isnan(mcse_mean) else mcse_mean)
    else:
        holds = p <= delta + 2.0 * mcse_p
    out.update(
        {
            "mean_realised": mean,
            "mcse_mean": mcse_mean,
            "p_exceed": p,
            "mcse_p": mcse_p,
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "verdict": "holds" if holds else "FAILS",
        }
    )
    return out


def build_resplit_tables(
    models: Sequence[ModelCurves], block: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-split experiment over val + test pooled, for every model x region.

    Returns:
        `(summary, per_split)` tables (`resplits.csv`, `resplits_per_split.csv.gz`).

    Raises:
        AssertionError: val and test share a case id (the pool would double count a case).
    """
    alphas = [float(a) for a in block["alphas"]]
    delta = float(block["delta"])
    summary_rows: list[dict[str, Any]] = []
    split_frames: list[pd.DataFrame] = []
    perm_cache: dict[int, np.ndarray] = {}
    for m in models:
        for region in m.regions:
            empty = str(block["empty"][region])
            pool = sorted(
                m.curves["val"][region] + m.curves["test"][region], key=lambda c: c.case_id
            )
            ids = [c.case_id for c in pool]
            if len(set(ids)) != len(ids):
                raise AssertionError(
                    f"resplits: val and test share case ids for model {m.name!r}, region "
                    f"{region!r}; the pooled set must be disjoint."
                )
            n_pool = len(pool)
            if n_pool not in perm_cache:
                perm_cache[n_pool] = make_perms(
                    n_pool, int(block["n_resplits"]), int(block["resplit_seed"])
                )
            perms = perm_cache[n_pool]
            logger.info(
                "resplits: %s / %s (pool %d, %d splits)", m.name, region, n_pool, len(perms)
            )
            result = resplit_cell(
                loss_matrix(pool, empty), perms, alphas, delta, int(block["rcps_order_seed"])
            )
            grid = np.asarray(pool[0].thresholds)
            for (g, alpha), (tau_idx, realised) in result.items():
                summary_rows.append(
                    {
                        "model": m.name,
                        "region": region,
                        "alpha": alpha,
                        "guarantee": g,
                        "empty": empty,
                        "delta": delta if g == "rcps" else float("nan"),
                        **summarise_resplits(realised, alpha, g, delta),
                    }
                )
                split_frames.append(
                    pd.DataFrame(
                        {
                            "model": m.name,
                            "region": region,
                            "alpha": alpha,
                            "guarantee": g,
                            "split": np.arange(realised.size),
                            "threshold": np.where(
                                tau_idx >= 0, grid[np.maximum(tau_idx, 0)], np.nan
                            ),
                            "realised": realised,
                        }
                    )
                )
    return pd.DataFrame(summary_rows), pd.concat(split_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 6: deployed post-processing
# ---------------------------------------------------------------------------


def _tau_key(tau: float) -> float:
    return round(float(tau), 12)


def postprocess_counts(
    eval_dir: Path,
    prep_dir: Path,
    case_ids: Sequence[str],
    taus: Sequence[float],
    regions: Sequence[str],
    full_cfg: Any,
) -> dict[float, dict[str, list[CaseLossCurve]]]:
    """Runs the deployed post-processing at each tau and records per-region voxel counts.

    One case's logits are loaded at a time. For each tau, `postprocess_logits` runs on the
    `(1, 3, D, H, W)` logits with `inference.postprocess.threshold = tau` (every other
    deployed setting is read from `full_cfg`), and false-negative, mask and reference voxel
    counts are taken per region. Each result is stored as a `CaseLossCurve` on the one-point
    grid `(tau,)`, so the same risk / bootstrap / mask-cost code applies unchanged.

    Args:
        eval_dir: Eval directory with saved logits.
        prep_dir: Preprocessed root holding `label.npy`.
        case_ids: Cases to process.
        taus: Distinct thresholds.
        regions: Regions to record.
        full_cfg: Full composed config (only `inference.postprocess` is read).

    Returns:
        `{tau: {region: [CaseLossCurve per case]}}`.
    """
    channel = {r: REGION_ORDER.index(r) for r in regions}
    pp_cfgs = {}
    for tau in taus:
        cfg_tau = copy.deepcopy(full_cfg)
        cfg_tau.inference.postprocess.threshold = float(tau)
        pp_cfgs[tau] = cfg_tau
    out: dict[float, dict[str, list[CaseLossCurve]]] = {
        tau: {r: [] for r in regions} for tau in taus
    }
    with _quiet_logger(_PP_LOGGER):
        for i, case_id in enumerate(case_ids, start=1):
            loaded = _load_label_and_spacing(prep_dir, case_id)
            if loaded is None:
                continue
            label, _ = loaded
            logits = torch.from_numpy(load_case_logits(eval_dir, case_id))[None]  # (1,3,D,H,W)
            target = classes_to_regions(torch.from_numpy(label))[0] > 0.5  # (3, D, H, W) bool
            for tau in taus:
                pred = postprocess_logits(logits, pp_cfgs[tau])[0] > 0.5  # (3, D, H, W) bool
                for r in regions:
                    c = channel[r]
                    out[tau][r].append(
                        CaseLossCurve(
                            case_id=case_id,
                            region=r,
                            gt_voxels=int(target[c].sum()),
                            thresholds=(float(tau),),
                            fn_voxels=(int((target[c] & ~pred[c]).sum()),),
                            mask_voxels=(int(pred[c].sum()),),
                        )
                    )
            if i % _LOG_EVERY == 0:
                logger.info("postprocess: %d/%d cases from %s", i, len(case_ids), eval_dir)
    return out


def _cached_postprocess_counts(
    cache_path: Path,
    eval_dir: Path,
    prep_dir: Path,
    case_ids: Sequence[str],
    taus: Sequence[float],
    regions: Sequence[str],
    full_cfg: Any,
) -> dict[float, dict[str, list[CaseLossCurve]]]:
    """`postprocess_counts` with a small JSON cache, so a killed multi-hour run can resume.

    The cache is reused only if eval dir, case ids, taus, regions and the whole
    `inference.postprocess` block match exactly.
    """
    key = {
        "eval_dir": str(Path(eval_dir).resolve()),
        "case_ids": list(case_ids),
        "taus": [_tau_key(t) for t in taus],
        "regions": list(regions),
        "postprocess": OmegaConf.to_container(full_cfg.inference.postprocess, resolve=True),
    }
    if cache_path.is_file():
        cached = read_json(cache_path)
        if cached.get("key") == key:
            logger.info("postprocess: reusing cached counts %s", cache_path)
            return {
                float(tau): {
                    r: [
                        CaseLossCurve(cid, r, gt, (float(tau),), (fn,), (mk,))
                        for cid, gt, fn, mk in zip(case_ids, *cached["counts"][i][r], strict=True)
                    ]
                    for r in regions
                }
                for i, tau in enumerate(taus)
            }
    counts = postprocess_counts(eval_dir, prep_dir, case_ids, taus, regions, full_cfg)
    payload = [
        {
            r: [
                [c.gt_voxels for c in counts[tau][r]],
                [c.fn_voxels[0] for c in counts[tau][r]],
                [c.mask_voxels[0] for c in counts[tau][r]],
            ]
            for r in regions
        }
        for tau in taus
    ]
    write_json({"key": key, "counts": payload}, cache_path)
    return counts


def build_postprocessed_table(
    models: Sequence[ModelCurves],
    fits: pd.DataFrame,
    realised: pd.DataFrame,
    block: Mapping[str, Any],
    full_cfg: Any,
    out_dir: Path,
    repo_root: Path,
) -> pd.DataFrame:
    """Realised risk and mask cost of the post-processed mask at each cell's own tau-hat.

    Only models with saved logits take part. tau-hat stays the RAW-curve value; the deployed
    post-processing is applied afterwards, so these numbers are descriptive.
    """
    lookup = threshold_lookup(fits)
    raw = {
        (r.model, r.cohort, r.region, float(r.alpha), r.guarantee): r.mean_miss_rate
        for r in realised.itertuples(index=False)
        if r.verdict != "infeasible"
    }
    prep_dirs = {c: _resolve(repo_root, block["prep_dirs"][c]) for c in COHORTS}
    generator = np.random.default_rng(int(block["boot_seed"]))
    n_boot = int(block["n_boot"])
    rows: list[dict[str, Any]] = []
    for m in models:
        if not m.has_logits:
            continue
        taus = sorted(
            {tau for (name, *_), tau in lookup.items() if name == m.name and tau is not None}
        )
        if not taus:
            continue
        for cohort in APPLY_COHORTS:
            case_ids = [c.case_id for c in m.curves[cohort][m.regions[0]]]
            counts = _cached_postprocess_counts(
                out_dir / "postprocessed_counts" / f"{m.name}__{cohort}.json",
                m.eval_dirs[cohort],
                prep_dirs[cohort],
                case_ids,
                taus,
                m.regions,
                full_cfg,
            )
            by_tau = {_tau_key(t): counts[t] for t in taus}
            for region in m.regions:
                empty = str(block["empty"][region])
                for alpha in block["alphas"]:
                    for g in GUARANTEES:
                        tau = lookup[(m.name, region, float(alpha), g)]
                        if tau is None:
                            continue
                        curves = by_tau[_tau_key(tau)][region]
                        res = evaluate_cell(
                            curves, tau, empty=empty, n_boot=n_boot, generator=generator
                        )
                        raw_risk = raw[(m.name, cohort, region, float(alpha), g)]
                        rows.append(
                            {
                                "model": m.name,
                                "cohort": cohort,
                                "region": region,
                                "alpha": float(alpha),
                                "guarantee": g,
                                "empty": empty,
                                "threshold": tau,
                                "raw_mean_miss_rate": raw_risk,
                                **res,
                                "delta_risk_vs_raw": res["mean_miss_rate"] - raw_risk,
                                "verdict": cell_verdict(
                                    cohort,
                                    res["mean_miss_rate"],
                                    res["ci_lo"],
                                    res["ci_hi"],
                                    float(alpha),
                                ),
                            }
                        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_v2(block_cfg: Any, full_cfg: Any, repo_root: Path) -> dict[str, Path]:
    """Runs the whole conformal v2 analysis (Hydra-free core).

    Args:
        block_cfg: The `analysis.conformal_v2` config block (DictConfig or mapping).
        full_cfg: The full composed config (only `inference.postprocess` is read, for step 6).
        repo_root: Root against which relative config paths resolve.

    Returns:
        `{short name: Path}` of each file written.

    Raises:
        RuntimeError: The additivity check failed (nothing after it is written).
        AssertionError: val and test case ids overlap.
    """
    block: dict[str, Any] = (
        OmegaConf.to_container(block_cfg, resolve=True)
        if isinstance(block_cfg, DictConfig)
        else dict(block_cfg)
    )
    repo_root = Path(repo_root)
    out_dir = ensure_dir(_resolve(repo_root, block["out_dir"]))
    alphas = [float(a) for a in block["alphas"]]
    delta = float(block["delta"])
    timings: dict[str, float] = {}
    paths: dict[str, Path] = {}

    def timed(step: str, t0: float) -> None:
        timings[step] = round(time.perf_counter() - t0, 3)
        logger.info("conformal_v2: step '%s' took %.1f s", step, timings[step])

    # 1. curves
    t0 = time.perf_counter()
    models = [
        load_model_curves(name, mcfg, block, out_dir, repo_root)
        for name, mcfg in block["models"].items()
    ]
    not_measured = [row for m in models for row in m.not_measured]
    paths["not_measured_csv"] = out_dir / "not_measured.csv"
    pd.DataFrame(not_measured, columns=["model", "item", "status"]).to_csv(
        paths["not_measured_csv"], index=False
    )
    timed("curves", t0)

    # 2. additivity -- written in full, then raised on, before any v2 number exists
    t0 = time.perf_counter()
    additivity: list[dict[str, Any]] = []
    for m in models:
        v1 = block["models"][m.name].get("v1_conformal_dir")
        if v1 is not None:
            additivity += check_additivity(m, _resolve(repo_root, v1), alphas)
    paths["additivity_json"] = out_dir / "additivity.json"
    write_json(
        {
            "n_comparisons": len(additivity),
            "all_ok": all(r["ok"] for r in additivity),
            "rows": additivity,
        },
        paths["additivity_json"],
    )
    raise_on_additivity_failure(additivity)
    timed("additivity", t0)

    # 3. fits
    t0 = time.perf_counter()
    fit_rows: list[dict[str, Any]] = []
    for m in models:
        for region in m.regions:
            fit_rows += fit_cell_rows(
                m.name,
                region,
                m.curves["val"][region],
                alphas=alphas,
                delta=delta,
                order_seed=int(block["rcps_order_seed"]),
                empty=str(block["empty"][region]),
                grid=m.grid,
            )
    fits = pd.DataFrame(fit_rows)
    paths["fits_csv"] = out_dir / "fits.csv"
    fits.to_csv(paths["fits_csv"], index=False)
    timed("fits", t0)

    # 4. frozen application
    t0 = time.perf_counter()
    realised = build_realised_table(models, fits, block)
    paths["realised_csv"] = out_dir / "realised.csv"
    realised.to_csv(paths["realised_csv"], index=False)
    timed("realised", t0)

    # 5. re-splits
    t0 = time.perf_counter()
    summary, per_split = build_resplit_tables(models, block)
    paths["resplits_csv"] = out_dir / "resplits.csv"
    paths["resplits_per_split"] = out_dir / "resplits_per_split.csv.gz"
    summary.to_csv(paths["resplits_csv"], index=False)
    per_split.to_csv(paths["resplits_per_split"], index=False)
    timed("resplits", t0)

    # 6. deployed post-processing
    if bool(block["postprocess"]):
        t0 = time.perf_counter()
        post = build_postprocessed_table(
            models, fits, realised, block, full_cfg, out_dir, repo_root
        )
        paths["postprocessed_csv"] = out_dir / "postprocessed.csv"
        post.to_csv(paths["postprocessed_csv"], index=False)
        timed("postprocessed", t0)

    # 7. run meta
    paths["run_meta_json"] = out_dir / "run_meta.json"
    write_json(
        {
            "config": block,
            "grid_sizes": {"v1": len(DEFAULT_THRESHOLDS), "extended": len(EXTENDED_THRESHOLDS)},
            "model_grids": {m.name: m.grid for m in models},
            "n_cases": {
                m.name: {c: len(m.curves[c][m.regions[0]]) for c in COHORTS} for m in models
            },
            "wall_time_s": timings,
            "not_measured": not_measured,
        },
        paths["run_meta_json"],
    )
    return paths


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs conformal v2 per `cfg.analysis.conformal_v2`.

    Example:

        python scripts/conformal_v2.py

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_v2(cfg.analysis.conformal_v2, cfg, Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    main()
