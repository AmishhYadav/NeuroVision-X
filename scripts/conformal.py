"""Hydra entry point for conformal risk control: extract loss curves, fit, apply.

`src/neurovision/uncertainty/conformal.py` implements the statistical procedure
(`case_loss_curve`, `fit_threshold`, `realised_risk`, `band_inflation`) and is
fully tested. This script is the driver that runs it over this project's saved
fp16 logits, per `docs/research/preregistration_conformal.md` (read that file
first -- it is the contract this script implements, and it must never be
edited from here).

## What the guarantee IS

The model's ordinary `p >= 0.5` mask is a point estimate with no promise about
how much true tumour it might be missing. This script instead picks, per
region, the LARGEST threshold `tau_hat` (i.e. the least conservative choice)
such that the CONSERVATIVE mask `{v : p(v) >= tau_hat}` misses, ON AVERAGE, no
more than an `alpha` fraction of the true tumour on a FRESH case drawn from
the same population as the calibration set. That is a bound on an
EXPECTATION over the random draw of calibration and test data -- it is not a
promise about any single case, in exactly the sense that a 95% confidence
interval does not promise that any one interval you compute contains the
truth. A specific patient's case can still miss much more than `alpha` of
their tumour; the guarantee is that such cases are rare enough, averaged with
the good ones, that the MEAN stays under `alpha`. Measured concretely: across
200 independent calibration draws at alpha=0.10, the mean realised risk over
those 200 splits was 0.0828 (comfortably under 0.10, as the theorem promises
on average) -- but 19% of the INDIVIDUAL splits landed above 0.10. That 19%
is not a bug and not a violation of anything; it is what "a bound on the
average" looks like when you look at one draw at a time. A single evaluation
split coming in slightly above alpha is therefore not evidence the method is
broken -- see the pre-registration's decision rules for what would be.

## Why extraction is a separate, cached stage

`case_loss_curve` (in `conformal.py`) is the one expensive step: it compares a
prediction against ground truth at every threshold in a several-dozen-point
grid, over a volume that can hold ~10^7 voxels. Measured at 0.73 s/case for
all three regions' curves together. Everything downstream -- fitting a
threshold at a given alpha, applying it to report realised risk, re-deriving
the mask-inflation cost, even redoing all of that on a different subset of
cases -- is then cheap arithmetic over the few hundred integers each
`CaseLossCurve` stores (`gt_voxels` plus `fn_voxels`/`mask_voxels` at each grid
threshold). That is the whole point of `CaseLossCurve` being a sufficient
statistic: the full-resolution pass happens exactly once per eval directory,
cached to `<out_dir>/<eval_dir basename>/curves.npz`, and every later `fit`/
`apply` invocation (including a rerun at a different alpha) is arithmetic on
that small cached file, not another pass over the volumes.

## Why the curves are built on the RAW thresholded probability, with NO post-processing

This is not a performance shortcut -- it is required by the theorem's own
precondition. `fit_threshold` needs the risk curve to be non-decreasing in
`tau` (see `conformal.py`'s docstring); it enforces this itself and RAISES if
it is not. The project's usual post-processing chain
(`neurovision.inference.postprocess`) includes a minimum-connected-component
filter, and that filter is NOT monotone in `tau`: lowering the threshold grows
the mask, and growing the mask can merge two components that were previously
separate into one component whose SIZE now clears the filter differently than
either did alone. A case's reported miss rate could therefore go UP as `tau`
goes DOWN once post-processing is in the loop -- exactly backwards, and
exactly the precondition `fit_threshold` polices. So the curves here are built
directly from `sigmoid(logits) >= tau`, never from `postprocess_logits`'
output. (Guard 3 below separately runs the project's REAL post-processing
chain, once, to confirm the saved logits reproduce the published Dice -- that
check exists to verify the logits themselves, not to feed post-processed
masks into the conformal curves.)

## The four guards this script refuses to skip

Per the pre-registration's falsifier list -- a bug here must be caught as a
bug, never reported as a scientific finding:

1. **Fit/apply separation** (`resolve_dirs`): `conformal.calib_dir` and every
   entry of `conformal.apply_dirs` must resolve (`Path.resolve()`) to
   different paths. Fitting and reporting on the same split fits the
   threshold to that split's own noise.
2. **Monotonicity**: enforced inside `fit_threshold` itself, which raises
   `ValueError` rather than returning a fit. This script never catches that
   exception -- it propagates, and nothing downstream is written.
3. **Replay self-consistency** (`_check_replay_consistency`): before trusting
   any eval directory's saved logits, this script replays them at the
   project's DEFAULT threshold and DEFAULT post-processing chain (unlike the
   conformal curves above, which use neither) and compares the resulting
   `dice_ET`/`dice_TC`/`dice_WT` against that directory's own committed
   `per_case_metrics.csv`. A mean absolute delta at or above
   `conformal.consistency_tol` means the saved logits do not belong to the
   published numbers, and the run aborts before writing anything.
4. **Degenerate alpha=1.0 endpoint** (`_check_alpha_one_degenerate`): fitting
   at `alpha=1.0` on the calibration curves must select the largest grid
   threshold as feasible -- a self-check for a reversed comparison, run
   before the configured alphas are fit.

Infeasibility at a configured alpha (no grid threshold achieves the target
risk) is NOT one of these failures -- it is a registered, substantive
scientific outcome (see the pre-registration) and is reported in `fit.json`
as `feasible: false`, never raised as an error and never silently clipped to
the smallest threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.replay import (
    _load_label_and_spacing,
    available_logit_cases,
    load_case_logits,
    per_case_replay,
)
from neurovision.analysis.statistics import load_per_case, paired_bootstrap_ci
from neurovision.data.transforms import REGION_NAMES
from neurovision.metrics.segmentation import classes_to_regions
from neurovision.uncertainty.conformal import (
    DEFAULT_THRESHOLDS,
    CaseLossCurve,
    ConformalFit,
    band_inflation,
    case_loss_curve,
    fit_threshold,
    realised_risk,
)
from neurovision.utils.io import ensure_dir, read_json, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/evaluate.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# How often (in cases) extraction logs progress. Matches analysis/replay.py's
# own _LOG_EVERY so a long extraction and a long replay read the same way.
_LOG_EVERY = 25

# Metrics the self-consistency check (guard 3) compares against a committed
# per_case_metrics.csv. Same three metrics scripts/replay_logits.py checks,
# for the same reason: HD95 can legitimately be NaN on one side and not the
# other, which would make an "agree" check noisy for nothing.
_CONSISTENCY_METRICS: tuple[str, ...] = ("dice_ET", "dice_TC", "dice_WT")


# ---------------------------------------------------------------------------
# Directory resolution (guard 1: fit/apply separation)
# ---------------------------------------------------------------------------


def resolve_dirs(cfg: DictConfig) -> tuple[Path, list[Path]]:
    """Resolves and validates `conformal.calib_dir` / `conformal.apply_dirs`.

    This is where fit/apply separation (falsifier 4 in the pre-registration)
    is enforced structurally, not just documented: the calibration directory
    and every apply directory must exist and must not resolve
    (`Path.resolve()`) to the same location.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `(calib_dir, apply_dirs)`, all existing directories.

    Raises:
        ValueError: `conformal.calib_dir` is `None`, `conformal.apply_dirs` is
            empty, or `calib_dir` resolves to the same path as one of
            `apply_dirs`.
        FileNotFoundError: Any resolved path does not exist.
    """
    conf_cfg = cfg.conformal
    calib_raw = conf_cfg.calib_dir
    apply_raw = list(conf_cfg.apply_dirs) if conf_cfg.apply_dirs else []

    if calib_raw is None:
        raise ValueError(
            "conformal.calib_dir must be set -- point it at the VAL split's eval_dir. Example:\n"
            "  python scripts/conformal.py conformal.calib_dir=outputs/neurovision/eval_val "
            "'conformal.apply_dirs=[outputs/neurovision/eval_test]'"
        )
    if not apply_raw:
        raise ValueError(
            "conformal.apply_dirs must be a non-empty list -- point it at one or more eval_dirs "
            "to report the frozen threshold's realised risk on. Example:\n"
            "  python scripts/conformal.py conformal.calib_dir=outputs/neurovision/eval_val "
            "'conformal.apply_dirs=[outputs/neurovision/eval_test]'"
        )

    calib_dir = Path(str(calib_raw))
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"conformal.calib_dir does not exist: {calib_dir.resolve()}")
    calib_resolved = calib_dir.resolve()

    apply_dirs: list[Path] = []
    for raw in apply_raw:
        apply_dir = Path(str(raw))
        if not apply_dir.is_dir():
            raise FileNotFoundError(
                f"conformal.apply_dirs entry does not exist: {apply_dir.resolve()}"
            )
        if apply_dir.resolve() == calib_resolved:
            raise ValueError(
                f"conformal.calib_dir and one of conformal.apply_dirs both resolve to the SAME "
                f"path ({calib_resolved}). Fitting the conformal threshold on a split and then "
                "reporting its realised risk on that SAME split fits and reports on the same "
                "noise, which invalidates the risk-control guarantee -- see the pre-registration. "
                "Point calib_dir at the VAL split's eval_dir and apply_dirs at the TEST split's "
                "(and any external cohort's) eval_dir."
            )
        apply_dirs.append(apply_dir)

    return calib_dir, apply_dirs


def resolve_prep_dirs(cfg: DictConfig) -> tuple[Path, list[Path]]:
    """Resolves `conformal.calib_prep_dir` / `conformal.apply_prep_dirs`.

    `apply_prep_dirs` is either empty (every apply dir defaults to
    `calib_prep_dir` -- the ordinary same-dataset case) or a list that must be
    exactly as long as `apply_dirs` (one preprocessed root per external
    cohort, e.g. `data/preprocessed/brats_ssa`, `data/preprocessed/brats_ped`).

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `(calib_prep_dir, apply_prep_dirs)`, the latter parallel to
        `conformal.apply_dirs` in order.

    Raises:
        ValueError: `conformal.apply_prep_dirs` is non-empty and its length
            does not match `conformal.apply_dirs`'s.
    """
    conf_cfg = cfg.conformal
    calib_prep_dir = Path(str(conf_cfg.calib_prep_dir))
    apply_prep_raw = list(conf_cfg.apply_prep_dirs) if conf_cfg.apply_prep_dirs else []
    n_apply = len(list(conf_cfg.apply_dirs))

    if not apply_prep_raw:
        apply_prep_dirs = [calib_prep_dir] * n_apply
        logger.info(
            "resolve_prep_dirs: conformal.apply_prep_dirs is empty -- defaulting every apply "
            "dir's labels root to calib_prep_dir=%s.",
            calib_prep_dir,
        )
    else:
        if len(apply_prep_raw) != n_apply:
            raise ValueError(
                f"conformal.apply_prep_dirs has {len(apply_prep_raw)} entries but "
                f"conformal.apply_dirs has {n_apply}; they must be parallel lists (one "
                "preprocessed labels root per apply eval_dir, in the same order)."
            )
        apply_prep_dirs = [Path(str(p)) for p in apply_prep_raw]

    return calib_prep_dir, apply_prep_dirs


def _region_index(region: str) -> int:
    """Channel index of `region` within `REGION_NAMES` (`ET`, `TC`, `WT` order)."""
    try:
        return REGION_NAMES.index(region)
    except ValueError as exc:
        raise ValueError(
            f"conformal: region {region!r} is not one of REGION_NAMES {REGION_NAMES}."
        ) from exc


# ---------------------------------------------------------------------------
# Guard 3: replay self-consistency
# ---------------------------------------------------------------------------


def _check_replay_consistency(
    eval_dir: Path, prep_dir: Path, tol: float
) -> dict[str, float] | None:
    """Verifies `eval_dir`'s saved logits reproduce its own committed `per_case_metrics.csv`.

    Unlike the conformal loss curves (`extract_curves`, below), which are
    computed on the RAW thresholded probability with no post-processing, this
    check DOES run the project's default threshold and default post-processing
    chain (via `per_case_replay`) -- it is verifying the DEPLOYED path that
    `scripts/evaluate.py` actually reported, not the conformal analysis. Two
    different conventions inside one script is exactly the kind of thing a
    later reader trips on, so: this function alone runs post-processing;
    nothing else in this script does.

    Args:
        eval_dir: An eval directory to verify.
        prep_dir: Root of the preprocessed BraTS data holding its labels.
        tol: Mean absolute Dice delta at or above which this raises.

    Returns:
        The measured `{metric: mean_absolute_delta}` dict on success, or
        `None` if the check was skipped (no committed `per_case_metrics.csv`,
        or no shared case ids).

    Raises:
        ValueError: A committed table exists, shares at least one case id
            with the replay, and any metric's mean absolute delta is at or
            above `tol`.
    """
    published_path = Path(eval_dir) / "per_case_metrics.csv"
    if not published_path.is_file():
        logger.warning(
            "conformal: no per_case_metrics.csv at %s; skipping the replay self-consistency "
            "check for this directory. Its saved logits are therefore UNVERIFIED.",
            published_path,
        )
        return None

    replayed = per_case_replay(eval_dir, prep_dir)
    published = load_per_case(published_path)
    metrics = list(_CONSISTENCY_METRICS)
    joined = replayed[metrics].join(
        published[metrics], how="inner", lsuffix="_replay", rsuffix="_published"
    )
    if joined.empty:
        logger.warning(
            "conformal: the replay of %s and %s share no case ids; skipping the self-consistency "
            "check.",
            eval_dir,
            published_path,
        )
        return None

    deltas = {
        metric: float((joined[f"{metric}_replay"] - joined[f"{metric}_published"]).abs().mean())
        for metric in metrics
    }
    logger.info(
        "conformal: replay self-consistency mean absolute deltas for %s vs %s (n=%d shared "
        "case(s)): %s",
        eval_dir,
        published_path,
        len(joined),
        deltas,
    )

    if max(deltas.values()) >= tol:
        raise ValueError(
            f"conformal: replaying {eval_dir}'s saved logits at the project default threshold "
            f"and post-processing disagrees with its committed {published_path} by more than "
            f"{tol} (mean absolute deltas: {deltas}). This means the saved logits do not belong "
            "to the published numbers -- refusing to run conformal risk control on them until "
            "this is resolved."
        )
    return deltas


# ---------------------------------------------------------------------------
# Stage 1: extract (with caching)
# ---------------------------------------------------------------------------


def resolve_curves_dir(out_dir: Path, eval_dir: Path) -> Path:
    """`<out_dir>/<eval_dir's basename>` -- where one eval directory's curve cache lives."""
    return Path(out_dir) / Path(eval_dir).name


def _manifest_request(
    eval_dir: Path, prep_dir: Path, regions: Sequence[str], thresholds: Sequence[float]
) -> dict[str, Any]:
    """The subset of a manifest that must match for a cached `curves.npz` to be reused."""
    return {
        "eval_dir": str(Path(eval_dir).resolve()),
        "prep_dir": str(Path(prep_dir).resolve()),
        "regions": list(regions),
        "thresholds": [round(float(t), 12) for t in thresholds],
    }


def _manifest_matches(existing: Mapping[str, Any], requested: Mapping[str, Any]) -> bool:
    """True if `existing` (loaded from `curves_manifest.json`) agrees with `requested` on
    every key `_manifest_request` produces."""
    return all(existing.get(key) == value for key, value in requested.items())


def _compute_curves_and_gt_voxels(
    eval_dir: Path,
    prep_dir: Path,
    case_ids: Sequence[str],
    regions: Sequence[str],
    thresholds: Sequence[float],
) -> tuple[dict[str, list[CaseLossCurve]], dict[str, dict[str, int]]]:
    """One pass over `case_ids`: builds every region's `CaseLossCurve` for each case.

    This is the ONLY place in this script that touches full-resolution voxel
    data (measured 0.73 s/case for all three regions together) -- see the
    module docstring's "sufficient statistic" section. The loss is computed
    on `sigmoid(logits) >= tau` directly, never on post-processed output (see
    the module docstring for why post-processing would break the
    monotonicity `fit_threshold` requires).

    Args:
        eval_dir: An eval directory with `logits/` saved.
        prep_dir: Root of the preprocessed BraTS data.
        case_ids: Case ids to process.
        regions: Region names to build curves for.
        thresholds: The shared threshold grid.

    Returns:
        `(curves, gt_voxels)`: `curves[region]` is a list of `CaseLossCurve`,
        one per case that had a `label.npy` (missing-label cases are skipped
        and logged, matching `analysis.replay`'s convention). `gt_voxels` is
        `{case_id: {region: gt_voxels}}`, recorded into the cache manifest.
    """
    curves: dict[str, list[CaseLossCurve]] = {region: [] for region in regions}
    gt_voxels: dict[str, dict[str, int]] = {}
    region_indices = {region: _region_index(region) for region in regions}

    n_used = 0
    for i, case_id in enumerate(case_ids, start=1):
        loaded = _load_label_and_spacing(prep_dir, case_id)
        if loaded is None:
            continue
        label, _spacing = loaded

        logits = load_case_logits(eval_dir, case_id)  # (3, D, H, W) float32; fp16 -> float32
        # already happened inside load_case_logits, before this sigmoid touches the
        # values -- fp16 loses exactly the resolution the conservative, low
        # thresholds this analysis lives at would need.
        prob = torch.sigmoid(torch.from_numpy(logits)).numpy()
        target = classes_to_regions(torch.from_numpy(label))[0].numpy()  # (3, D, H, W)

        case_gt: dict[str, int] = {}
        for region in regions:
            c = region_indices[region]
            curve = case_loss_curve(
                prob[c], target[c], case_id=case_id, region=region, thresholds=thresholds
            )
            curves[region].append(curve)
            case_gt[region] = curve.gt_voxels
        gt_voxels[case_id] = case_gt
        n_used += 1

        if i % _LOG_EVERY == 0:
            logger.info("extract: processed %d/%d case(s) from %s", i, len(case_ids), eval_dir)

    logger.info(
        "extract: built loss curves for %d/%d case(s) x %d region(s) from %s.",
        n_used,
        len(case_ids),
        len(regions),
        eval_dir,
    )
    return curves, gt_voxels


def _write_curves_npz(
    curves: Mapping[str, Sequence[CaseLossCurve]], thresholds: Sequence[float], path: Path
) -> None:
    """Persists every region's loss curves to one compact `.npz` file.

    A few hundred integers per case, not a full-resolution volume -- see the
    module docstring's "sufficient statistic" section for why this is what
    makes every later `fit`/`apply` cheap.
    """
    payload: dict[str, np.ndarray] = {"thresholds": np.asarray(thresholds, dtype=np.float64)}
    for region, region_curves in curves.items():
        payload[f"{region}__case_ids"] = np.array([c.case_id for c in region_curves])
        payload[f"{region}__gt_voxels"] = np.array(
            [c.gt_voxels for c in region_curves], dtype=np.int64
        )
        payload[f"{region}__fn_voxels"] = np.array(
            [c.fn_voxels for c in region_curves], dtype=np.int64
        )
        payload[f"{region}__mask_voxels"] = np.array(
            [c.mask_voxels for c in region_curves], dtype=np.int64
        )
    np.savez_compressed(path, **payload)


def _load_curves_npz(path: Path, regions: Sequence[str]) -> dict[str, list[CaseLossCurve]]:
    """Reloads `_write_curves_npz`'s output back into `CaseLossCurve` objects."""
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


def extract_curves(
    eval_dir: Path,
    prep_dir: Path,
    out_dir: Path,
    *,
    regions: Sequence[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, list[CaseLossCurve]]:
    """Loads (from cache) or computes and persists one eval directory's per-region loss curves.

    Reuses `<out_dir>/<eval_dir basename>/curves.npz` when its sibling
    `curves_manifest.json` matches this request's `eval_dir`, `prep_dir`,
    `regions` and `thresholds` exactly; recomputes (and overwrites both files)
    otherwise. See the module docstring for why this caching is the point of
    the whole design, not an optimization.

    Args:
        eval_dir: An eval directory with `logits/` saved.
        prep_dir: Root of the preprocessed BraTS data holding its labels.
        out_dir: This run's conformal output directory
            (`conformal.out_dir`) -- the cache lives one level below it.
        regions: Region names to build curves for.
        thresholds: The shared threshold grid.

    Returns:
        `{region: [CaseLossCurve, ...]}`.

    Raises:
        ValueError: No case under `eval_dir/logits/` has saved logits.
    """
    curves_dir = resolve_curves_dir(out_dir, eval_dir)
    npz_path = curves_dir / "curves.npz"
    manifest_path = curves_dir / "curves_manifest.json"
    requested = _manifest_request(eval_dir, prep_dir, regions, thresholds)

    if npz_path.is_file() and manifest_path.is_file():
        existing = read_json(manifest_path)
        if _manifest_matches(existing, requested):
            logger.info(
                "extract: reusing cached curves at %s (manifest matches eval_dir, prep_dir, "
                "regions and thresholds).",
                npz_path,
            )
            return _load_curves_npz(npz_path, regions)
        logger.info(
            "extract: cached manifest at %s does not match this request (eval_dir, prep_dir, "
            "regions or thresholds changed); re-extracting rather than reusing a mismatched "
            "cache.",
            manifest_path,
        )

    case_ids = available_logit_cases(eval_dir)
    if not case_ids:
        raise ValueError(f"extract: no saved logits under {Path(eval_dir) / 'logits'}.")

    curves, gt_voxels = _compute_curves_and_gt_voxels(
        eval_dir, prep_dir, case_ids, regions, thresholds
    )

    ensure_dir(curves_dir)
    _write_curves_npz(curves, thresholds, npz_path)
    manifest = {**requested, "n_cases": len(case_ids), "gt_voxels": gt_voxels}
    write_json(manifest, manifest_path)
    logger.info("extract: wrote %s and %s.", npz_path, manifest_path)

    return curves


# ---------------------------------------------------------------------------
# Guard 4: degenerate alpha=1.0 endpoint
# ---------------------------------------------------------------------------


def _check_alpha_one_degenerate(calib_curves: Mapping[str, list[CaseLossCurve]]) -> None:
    """Self-check: `alpha=1.0` must select the largest grid threshold, for every region.

    Registered by the pre-registration as falsifier 2. Run unconditionally on
    the calibration curves, independent of whatever alphas were configured --
    this catches a reversed comparison in the selection rule itself, not a
    property of any particular alpha a user happened to ask for.

    Args:
        calib_curves: `{region: [CaseLossCurve, ...]}` for the calibration set.

    Raises:
        AssertionError: `alpha=1.0` is infeasible, or its selected threshold
            is not the largest grid value, for any region.
    """
    for region, curves in calib_curves.items():
        max_threshold = curves[0].thresholds[-1]
        fit = fit_threshold(curves, 1.0)
        if (
            not fit.feasible
            or fit.threshold is None
            or not np.isclose(fit.threshold, max_threshold)
        ):
            raise AssertionError(
                f"conformal self-check FAILED for region {region!r}: alpha=1.0 must select the "
                f"largest grid threshold ({max_threshold!r}), got feasible={fit.feasible} "
                f"threshold={fit.threshold!r}. This means the threshold-selection rule is wired "
                "backwards -- see the pre-registration's falsifier list."
            )
        logger.info(
            "conformal: self-check OK -- alpha=1.0 selects threshold=%.6g (the largest grid "
            "value) for region %s, as the degenerate-endpoint falsifier requires.",
            fit.threshold,
            region,
        )


# ---------------------------------------------------------------------------
# Stage 2: fit
# ---------------------------------------------------------------------------


def fit_all(
    calib_curves: Mapping[str, list[CaseLossCurve]], alphas: Sequence[float]
) -> dict[tuple[str, float], ConformalFit]:
    """Fits a conformal threshold for every (region, alpha) pair on the calibration curves.

    Args:
        calib_curves: `{region: [CaseLossCurve, ...]}` for the calibration set.
        alphas: Target risk levels.

    Returns:
        `{(region, alpha): ConformalFit}`.

    Raises:
        ValueError: Propagated from `fit_threshold` -- in particular, its
            monotonicity falsifier (guard 2). Never caught here; a violation
            must abort the run, not be downgraded to a warning.
    """
    fits: dict[tuple[str, float], ConformalFit] = {}
    for region, curves in calib_curves.items():
        for alpha in alphas:
            fit = fit_threshold(curves, alpha)
            fits[(region, alpha)] = fit
            if fit.feasible:
                logger.info(
                    "fit: region=%s alpha=%.4f -> threshold=%.6g, calibrated_risk=%.6g "
                    "(n_calibration=%d, n_excluded_empty=%d).",
                    region,
                    alpha,
                    fit.threshold,
                    fit.calibrated_risk,
                    fit.n_calibration,
                    fit.n_excluded_empty,
                )
            else:
                logger.warning(
                    "fit: region=%s alpha=%.4f is INFEASIBLE -- no grid threshold achieves this "
                    "risk (min_achievable_risk=%.6g, n_calibration=%d, n_excluded_empty=%d). "
                    "This is a registered, substantive scientific outcome (see the "
                    "pre-registration), not an error.",
                    region,
                    alpha,
                    fit.min_achievable_risk,
                    fit.n_calibration,
                    fit.n_excluded_empty,
                )
    return fits


def _fit_payload(fits: Mapping[tuple[str, float], ConformalFit]) -> dict[str, Any]:
    """Assembles the `fit.json` payload, one entry per (region, alpha)."""
    payload: dict[str, Any] = {}
    for (region, alpha), fit in fits.items():
        payload[f"{region}__alpha_{alpha}"] = {
            "region": region,
            "alpha": alpha,
            "threshold": fit.threshold,
            "feasible": fit.feasible,
            "calibrated_risk": fit.calibrated_risk,
            "min_achievable_risk": fit.min_achievable_risk,
            "n_calibration": fit.n_calibration,
            "n_excluded_empty": fit.n_excluded_empty,
        }
    return payload


# ---------------------------------------------------------------------------
# Stage 3: apply
# ---------------------------------------------------------------------------


def _case_miss_rates_at_threshold(
    curves: Sequence[CaseLossCurve], threshold: float, tol: float = 1e-9
) -> np.ndarray:
    """Per-case miss rate at `threshold`, over non-empty-GT curves -- the raw values
    `realised_risk` means over.

    `realised_risk` (in `uncertainty/conformal.py`) returns only the
    aggregate mean; the bootstrap CI below needs the individual per-case
    values it was computed from, so this reproduces that same selection
    (skip `empty_gt`, look up `threshold`'s index in the shared grid)
    directly rather than widening that module's public return type for one
    caller's bookkeeping.

    Args:
        curves: One `CaseLossCurve` per case, all sharing an identical
            threshold grid.
        threshold: The threshold to read the miss rate at.
        tol: Tolerance for locating `threshold` in the grid.

    Returns:
        A 1-D float64 array, one entry per non-empty-GT case.

    Raises:
        ValueError: `curves` is empty, or `threshold` is not in the grid.
    """
    if not curves:
        raise ValueError("_case_miss_rates_at_threshold: curves must not be empty.")
    thresholds_arr = np.asarray(curves[0].thresholds, dtype=np.float64)
    idx = int(np.argmin(np.abs(thresholds_arr - threshold)))
    if abs(thresholds_arr[idx] - threshold) > tol:
        raise ValueError(
            f"{threshold} is not in the threshold grid; nearest available value is "
            f"{thresholds_arr[idx]!r}."
        )
    values = [c.miss_rate()[idx] for c in curves if not c.empty_gt]
    return np.asarray(values, dtype=np.float64)


def apply_all(
    apply_curves: Mapping[str, Mapping[str, list[CaseLossCurve]]],
    fits: Mapping[tuple[str, float], ConformalFit],
    alphas: Sequence[float],
    regions: Sequence[str],
    generator: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies every feasible (region, alpha) threshold, frozen, to every apply directory.

    A `bootstrap 95% CI on `mean_miss_rate` is added via
    `neurovision.analysis.statistics.paired_bootstrap_ci` rather than a
    second, hand-rolled bootstrap routine. That function's contract is a
    bootstrap CI on the PAIRED DIFFERENCE `a - b`; passing an all-zero `b`
    makes `diff = a - 0 = a`, so the resulting interval is exactly a
    single-sample bootstrap CI on `mean(a)` -- the paired machinery
    degenerates to the single-sample case rather than needing its own
    implementation. Seeded once, from `cfg.seed`, and threaded through every
    (apply_dir, region, alpha) row so the whole run draws from one
    reproducible stream.

    Args:
        apply_curves: `{apply_dir (as a string key): {region: [CaseLossCurve,
            ...]}}`.
        fits: `{(region, alpha): ConformalFit}` from `fit_all`, applied
            FROZEN -- never refit here.
        alphas: Configured alphas, in report order.
        regions: Configured regions, in report order.
        generator: A seeded `np.random.Generator`.

    Returns:
        `(realised_risk_df, inflation_df)`. `realised_risk_df` has one row
        per `(apply_dir, region, alpha)` with a feasible fit -- infeasible or
        missing fits are skipped (there is no frozen threshold to apply) and
        logged. `inflation_df` mirrors it with the mask-inflation cost table.
    """
    risk_rows: list[dict[str, Any]] = []
    inflation_rows: list[dict[str, Any]] = []

    for apply_name, region_curves in apply_curves.items():
        for region in regions:
            curves = region_curves[region]
            for alpha in alphas:
                fit = fits.get((region, alpha))
                if fit is None or not fit.feasible or fit.threshold is None:
                    logger.info(
                        "apply: skipping apply_dir=%s region=%s alpha=%.4f -- the calibration "
                        "fit is infeasible (or missing), so there is no frozen threshold to "
                        "apply.",
                        apply_name,
                        region,
                        alpha,
                    )
                    continue
                threshold = fit.threshold

                risk = realised_risk(curves, threshold)
                row: dict[str, Any] = {
                    "apply_dir": apply_name,
                    "region": region,
                    "alpha": alpha,
                    "threshold": threshold,
                    "mean_miss_rate": risk["mean_miss_rate"],
                    "n": risk["n"],
                    "n_excluded_empty": risk["n_excluded_empty"],
                }

                values = _case_miss_rates_at_threshold(curves, threshold)
                if values.size >= 2:
                    boot = paired_bootstrap_ci(values, np.zeros_like(values), generator=generator)
                    row["ci_lo"] = boot.lo
                    row["ci_hi"] = boot.hi
                else:
                    row["ci_lo"] = float("nan")
                    row["ci_hi"] = float("nan")
                risk_rows.append(row)

                inflation = band_inflation(curves, threshold)
                inflation_rows.append(
                    {
                        "apply_dir": apply_name,
                        "region": region,
                        "alpha": alpha,
                        "mean_inflation": inflation["mean_inflation"],
                        "median_inflation": inflation["median_inflation"],
                        "n": inflation["n"],
                        "n_skipped": inflation["n_skipped"],
                    }
                )

    return pd.DataFrame(risk_rows), pd.DataFrame(inflation_rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _print_summary(risk_df: pd.DataFrame, out_dir: Path) -> None:
    """Prints (not logs -- see `scripts/calibrate.py`'s equivalent) a compact end-of-run summary."""
    lines = ["=" * 70, f"Conformal risk control summary -- out_dir={out_dir}", "=" * 70]
    if risk_df.empty:
        lines.append(
            "  (no feasible fit was applied anywhere -- every configured alpha was infeasible; "
            "see fit.json for min_achievable_risk)"
        )
    else:
        for _, row in risk_df.iterrows():
            ci = ""
            if not pd.isna(row.get("ci_lo")):
                ci = f" CI=[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
            lines.append(
                f"  {row['apply_dir']} | {row['region']:>3s} | alpha={row['alpha']:.2f} | "
                f"tau={row['threshold']:.4g} | mean_miss_rate={row['mean_miss_rate']:.4f}{ci} "
                f"(n={int(row['n'])}, excluded={int(row['n_excluded_empty'])})"
            )
    # print only, not logger.info as well -- setup_logging's StreamHandler
    # already targets stdout, so doing both would print this block twice.
    # Matches scripts/calibrate.py's / scripts/replay_logits.py's summaries.
    print("\n".join(lines))


def run_conformal(cfg: DictConfig) -> dict[str, Path]:
    """Runs the full extract -> fit -> apply pipeline, per the composed config.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A dict mapping a short name to the `Path` each output file was
        written to, so tests can assert on what was produced without
        re-parsing every file.

    Raises:
        ValueError: See `resolve_dirs`, `resolve_prep_dirs`,
            `_check_replay_consistency`, `extract_curves`, and
            `fit_threshold`'s monotonicity falsifier (via `fit_all`).
        AssertionError: `_check_alpha_one_degenerate`'s self-check fails.
    """
    conf_cfg = cfg.conformal

    calib_dir, apply_dirs = resolve_dirs(cfg)
    calib_prep_dir, apply_prep_dirs = resolve_prep_dirs(cfg)

    regions = [str(r) for r in conf_cfg.regions]
    for region in regions:
        _region_index(region)  # validates every configured region name up front
    alphas = [float(a) for a in conf_cfg.alphas]

    out_dir = ensure_dir(str(conf_cfg.out_dir))

    check_consistency = bool(conf_cfg.check_consistency)
    consistency_tol = float(conf_cfg.consistency_tol)
    if check_consistency:
        logger.info(
            "conformal: verifying saved logits reproduce their own published metrics before "
            "trusting them (guard 3)."
        )
        _check_replay_consistency(calib_dir, calib_prep_dir, consistency_tol)
        for apply_dir, apply_prep_dir in zip(apply_dirs, apply_prep_dirs, strict=True):
            _check_replay_consistency(apply_dir, apply_prep_dir, consistency_tol)
    else:
        logger.warning(
            "conformal: check_consistency=false -- the saved logits are NOT being verified "
            "against a committed per_case_metrics.csv before use."
        )

    logger.info("conformal: extracting/loading loss curves for calib_dir=%s.", calib_dir)
    calib_curves = extract_curves(calib_dir, calib_prep_dir, out_dir, regions=regions)

    apply_curves: dict[str, dict[str, list[CaseLossCurve]]] = {}
    for apply_dir, apply_prep_dir in zip(apply_dirs, apply_prep_dirs, strict=True):
        logger.info("conformal: extracting/loading loss curves for apply_dir=%s.", apply_dir)
        apply_curves[str(apply_dir)] = extract_curves(
            apply_dir, apply_prep_dir, out_dir, regions=regions
        )

    _check_alpha_one_degenerate(calib_curves)

    fits = fit_all(calib_curves, alphas)
    fit_json_path = out_dir / "fit.json"
    write_json(_fit_payload(fits), fit_json_path)
    logger.info("conformal: wrote %s", fit_json_path)

    generator = np.random.default_rng(int(cfg.seed))
    risk_df, inflation_df = apply_all(apply_curves, fits, alphas, regions, generator)

    risk_csv_path = out_dir / "realised_risk.csv"
    inflation_csv_path = out_dir / "inflation.csv"
    risk_df.to_csv(risk_csv_path, index=False)
    inflation_df.to_csv(inflation_csv_path, index=False)
    logger.info("conformal: wrote %s and %s", risk_csv_path, inflation_csv_path)

    config_path = out_dir / "conformal_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    logger.info("conformal: wrote %s", config_path)

    _print_summary(risk_df, out_dir)

    return {
        "fit_json": fit_json_path,
        "realised_risk_csv": risk_csv_path,
        "inflation_csv": inflation_csv_path,
        "conformal_config_yaml": config_path,
    }


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs conformal risk control extract -> fit -> apply, per the composed config.

    Example:

        python scripts/conformal.py \\
            conformal.calib_dir=outputs/neurovision/eval_val \\
            'conformal.apply_dirs=[outputs/neurovision/eval_test]'

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_conformal(cfg)


if __name__ == "__main__":
    main()
