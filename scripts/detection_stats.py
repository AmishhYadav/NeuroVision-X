"""Gate 1 -- does inter-branch disagreement carry failure-detection signal entropy lacks?

This is the driver for the pre-registered decision in
`docs/research/preregistration_ambiguity.md` -- READ THAT FILE FIRST, its
"Decision rule" table is reproduced (verbatim, not paraphrased) in
`build_verdict`'s docstring below. It runs no model, loads no checkpoint,
needs no GPU: every input is a cache `scripts/extract_ambiguity.py` and
`scripts/evaluate.py` have already written to disk.

## Why entropy is the comparator, not an afterthought

Every segmentation model -- including a plain U-Net -- emits a per-voxel
predictive entropy for free. If inter-branch disagreement merely reproduces
that entropy, it is a duplicate of something a single-encoder baseline can
already compute, and the claim motivating this whole pivot ("only a
dual-encoder model can produce this") is false regardless of how strong the
RAW correlation with error looks. So every reported quantity here is
INCREMENTAL over entropy: `neurovision.analysis.detection.partial_spearman`
at the case level, `neurovision.analysis.detection.residualised_auroc` at
the voxel level. Neither raw correlation nor raw AUROC is ever reported as
if it settled anything on its own.

## Two endpoints, fixed before any p-value is examined

- **Case level** (`case_level_table`): partial Spearman correlation between
  a per-case disagreement scalar and per-case Dice, controlling for a
  per-case entropy scalar. Answers "does knowing disagreement, on top of
  entropy, tell you anything about how well this case will segment".
- **Voxel level** (`voxel_level_table`): residualised AUROC for predicting
  per-voxel prediction error from disagreement, after removing whatever a
  linear fit on entropy's own rank already explains. Answers "does
  disagreement flag WHERE a case fails, beyond what entropy already flags".

Both endpoints are run on all three cohorts (in-distribution BraTS test,
plus the two external cohorts SSA and PED) but the pre-registered PASS
condition only ever looks at the external ones -- a detector that only works
in-distribution does not address the failure that motivated it (out-of-
distribution transfer reversing, see CLAUDE.md's "external validation on
BraTS-Africa came back negative" note).

## The label-free sampling mask -- read before touching `voxel_level_table`

This project has already shipped a bug where a reporting mask was built
FROM the ground-truth label (`union_foreground_mask` in
`neurovision.uncertainty.calibration`): it manufactured 41-57% of a reported
ECE and passed 984 tests, because the code did exactly what it said. The
voxel-level sampling mask here (`_predicted_dilated_mask`) is built from the
model's OWN prediction alone -- never the label -- for the same reason.
`tests/test_detection_stats.py` pins this with a test that swaps the label
array entirely between two runs and asserts the sampled voxels do not move.

## Multiplicity

The pre-registered family is 2 endpoints x 3 cohorts = 6 tests, Holm-
corrected once (`build_family_table`), from the case-level partial-
correlation p-value and the voxel-level "ANY region" residualised-AUROC
p-value -- fixed before any p-value in this run is looked at.

Example usage:

    python scripts/detection_stats.py
    python scripts/detection_stats.py analysis.detection.voxel.enabled=false
"""

from __future__ import annotations

import glob
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import distance_transform_edt

from neurovision.analysis.detection import (
    REGION_NAMES,
    _entropy_from_logits,
    case_entropy_scalars,
    partial_spearman_ci,
    residualised_auroc,
    spearman,
)
from neurovision.analysis.replay import (
    _apply_postprocess_steps,
    _binarize_regions,
    _load_label_and_spacing,
    _resolve_postprocess_cfg,
    load_case_logits,
)
from neurovision.analysis.statistics import holm_bonferroni
from neurovision.metrics.segmentation import classes_to_regions
from neurovision.utils.io import ensure_dir, write_json, write_yaml
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/extract_ambiguity.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Which cohort names count as "external" for the pre-registered pass
# condition. Matches the preregistration's Data section exactly.
_EXTERNAL_COHORTS: tuple[str, ...] = ("ssa", "ped")

# The pre-registered pass-condition thresholds, stated here rather than read
# from config -- pre-registration rule 2 is "the thresholds are not adjusted
# after seeing data", so they must not be a config field a later run could
# quietly override.
_PASS_RHO_THRESHOLD = 0.20
_PASS_AUROC_THRESHOLD = 0.60

# Regions the voxel-level endpoint reports, plus the pooled "ANY" variant
# that feeds the pre-registered family (see build_family_table).
_VOXEL_GROUPS: tuple[str, ...] = (*REGION_NAMES, "ANY")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _expand_shard_dirs(entries: Sequence[Any]) -> list[Path]:
    """Resolves `ambiguity_dirs` entries, expanding any that hold a glob.

    A cohort is extracted in chunks -- `scripts/extract_ambiguity_serial.py`
    writes one directory per process so that a killed run loses only the
    chunk in flight -- so the number of shard directories is not known when
    this config is written. A glob entry keeps the config stable across
    however many chunks a cohort ends up needing.

    Expansion deliberately keeps only directories that actually hold an
    `ambiguity_summary.csv`: an aborted launch can leave an empty directory
    matching the pattern, and that is not a shard, it is debris. A LITERAL
    entry is never filtered this way -- naming a specific directory that
    turns out to have no summary is a user error, and `load_cohort` must
    still raise on it rather than silently dropping a cohort slice.

    Args:
        entries: Raw `ambiguity_dirs` values, each a path or a glob pattern.

    Returns:
        Shard directories in a stable order: patterns expand in sorted
        order, and entries keep their configured order.

    Raises:
        ValueError: If `entries` is empty, or a glob pattern matches no
            directory containing an `ambiguity_summary.csv`.
    """
    raw = [str(entry) for entry in entries]
    if not raw:
        raise ValueError("load_cohort: cohort_cfg.ambiguity_dirs is empty.")

    resolved: list[Path] = []
    for entry in raw:
        if not any(char in entry for char in "*?["):
            resolved.append(Path(entry))
            continue
        matched = sorted(
            path
            for path in (Path(hit) for hit in glob.glob(entry))
            if path.is_dir() and (path / "ambiguity_summary.csv").is_file()
        )
        if not matched:
            raise ValueError(
                f"load_cohort: the pattern {entry!r} matched no directory holding an "
                "ambiguity_summary.csv. Either the cohort has not been extracted yet "
                "(run scripts/extract_ambiguity_serial.py), or the pattern is wrong."
            )
        resolved.extend(matched)
    return resolved


def load_cohort(cohort_cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Concatenates a cohort's ambiguity-extraction shards into one table and one npz map.

    Args:
        cohort_cfg: One entry of `cfg.analysis.detection.cohorts` (a
            `DictConfig` or plain mapping), exposing `ambiguity_dirs` --
            one or more shard directories, each written by one
            `scripts/extract_ambiguity.py` run over a disjoint slice of the
            same cohort. An entry containing a glob character is expanded
            (see `_expand_shard_dirs`), which is what lets the resumable
            serial driver keep adding chunk directories without anyone
            having to edit this config between runs.

    Returns:
        `(summary, npz_paths)`. `summary` is every shard's
        `ambiguity_summary.csv` concatenated, indexed by `case_id`.
        `npz_paths` maps `case_id -> <shard_dir>/<case_id>.npz`.

    Raises:
        ValueError: If any `case_id` appears in more than one shard
            directory, naming the offending case id(s) -- shards must be
            disjoint, or a pooled statistic over the cohort would double-
            count that case.
        FileNotFoundError: If a shard directory has no
            `ambiguity_summary.csv`.
    """
    shard_dirs = _expand_shard_dirs(cohort_cfg["ambiguity_dirs"])

    case_to_shards: dict[str, list[Path]] = {}
    frames: list[pd.DataFrame] = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "ambiguity_summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"load_cohort: no ambiguity_summary.csv at {summary_path}.")
        frame = pd.read_csv(summary_path, index_col="case_id")
        frames.append(frame)
        for case_id in frame.index:
            case_to_shards.setdefault(str(case_id), []).append(shard_dir)

    duplicated = {cid: dirs for cid, dirs in case_to_shards.items() if len(dirs) > 1}
    if duplicated:
        raise ValueError(
            "load_cohort: the following case_id(s) appear in more than one shard directory -- "
            f"shards must be disjoint, or the pooled statistic double-counts them: {duplicated}."
        )

    summary = pd.concat(frames).rename_axis("case_id")
    npz_paths = {cid: dirs[0] / f"{cid}.npz" for cid, dirs in case_to_shards.items()}
    return summary, npz_paths


def entropy_table(
    eval_dir: str | Path, case_ids: Sequence[str], cache_path: str | Path
) -> pd.DataFrame:
    """Computes (or reuses) the single-pass predictive-entropy baseline for a set of cases.

    This IS the comparator the whole gate is about, so it is computed from
    the already-SAVED `logits/` in `eval_dir` -- no model run, no label --
    via `neurovision.analysis.detection.case_entropy_scalars`.

    Args:
        eval_dir: A `scripts/evaluate.py` output directory with
            `logits/` saved (`cfg.inference.evaluation.save_logits=true`).
        case_ids: Cases to compute entropy scalars for.
        cache_path: Where to read/write the cached table. Reused (and
            logged as such, never silently) when every requested case_id is
            already present; otherwise recomputed for the full requested set
            and the cache is overwritten.

    Returns:
        A `DataFrame` indexed by `case_id`, one column per
        `case_entropy_scalars` key, restricted to `case_ids` in the reuse
        path (a superset cache is not returned as-is).
    """
    cache_path = Path(cache_path)
    case_ids = [str(c) for c in case_ids]

    if cache_path.is_file():
        cached = pd.read_csv(cache_path, index_col="case_id")
        missing = [c for c in case_ids if c not in cached.index]
        if not missing:
            logger.info(
                "entropy_table: reusing cached entropy table at %s (%d case(s) already present, "
                "no logits re-read).",
                cache_path.resolve(),
                len(case_ids),
            )
            return cached.loc[case_ids]
        logger.info(
            "entropy_table: cache at %s is missing %d/%d requested case(s); recomputing the "
            "whole table.",
            cache_path.resolve(),
            len(missing),
            len(case_ids),
        )

    rows: dict[str, dict[str, float]] = {}
    for case_id in case_ids:
        logits = load_case_logits(eval_dir, case_id)
        rows[case_id] = case_entropy_scalars(logits)
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("case_id")

    ensure_dir(cache_path.parent)
    table.to_csv(cache_path)
    return table


def _join_case_tables(
    ambiguity_summary: pd.DataFrame,
    entropy: pd.DataFrame,
    per_case_metrics: pd.DataFrame,
    cohort_name: str,
) -> pd.DataFrame:
    """Inner-joins the three per-case tables the case-level endpoint needs.

    Args:
        ambiguity_summary: Indexed by `case_id`, from `load_cohort`.
        entropy: Indexed by `case_id`, from `entropy_table`.
        per_case_metrics: Indexed by `case_id`, a `per_case_metrics.csv` as
            written by `scripts/evaluate.py`.
        cohort_name: Used only in the log line and the error message.

    Returns:
        The inner join on `case_id`.

    Raises:
        ValueError: Fewer than 10 cases survive the join -- a silently tiny
            n is how this project has been burned before (see the
            calibration-mask and eloquence-layer entries in CLAUDE.md).
    """
    all_ids = set(ambiguity_summary.index) | set(entropy.index) | set(per_case_metrics.index)
    joined = ambiguity_summary.join(entropy, how="inner").join(per_case_metrics, how="inner")
    n_dropped = len(all_ids) - len(joined)
    logger.info(
        "Cohort %r: case-level inner join kept %d/%d case(s) (%d dropped).",
        cohort_name,
        len(joined),
        len(all_ids),
        n_dropped,
    )
    if len(joined) < 10:
        raise ValueError(
            f"Cohort {cohort_name!r}: only {len(joined)} case(s) survive the inner join across "
            "the ambiguity summary, the entropy table and per_case_metrics.csv -- refusing to "
            "compute a case-level statistic from this few cases."
        )
    return joined


# ---------------------------------------------------------------------------
# Shared bootstrap p-value
# ---------------------------------------------------------------------------


def _bootstrap_two_sided_p(replicates: np.ndarray, n_boot: int, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for a replicate distribution against `null`.

    Shared by the case-level partial-correlation endpoint (`null=0.0`) and
    the voxel-level residualised-AUROC endpoint (`null=0.5`, the value a
    detector with zero incremental information collapses to).

    Args:
        replicates: Bootstrap replicate statistics, possibly containing NaN
            for degenerate resamples -- dropped before the p-value is
            computed.
        n_boot: Number of replicates originally drawn. Used only to set the
            smallest reportable p-value (`1 / n_boot`), so a finite
            bootstrap never reports an exact 0.
        null: The null-hypothesis value the statistic is centred against.

    Returns:
        `2 * min(P(replicate <= null), P(replicate >= null))`, clipped to
        `[1 / n_boot, 1.0]`. NaN if every replicate is non-finite.
    """
    valid = replicates[np.isfinite(replicates)]
    if valid.size == 0:
        return float("nan")
    p_low = float(np.mean(valid <= null))
    p_high = float(np.mean(valid >= null))
    p_boot = 2.0 * min(p_low, p_high)
    return float(np.clip(p_boot, 1.0 / n_boot, 1.0))


# ---------------------------------------------------------------------------
# Case-level endpoint
# ---------------------------------------------------------------------------


def case_level_table(
    cohort_name: str,
    joined: pd.DataFrame,
    cfg: DictConfig,
    generator: np.random.Generator,
) -> pd.DataFrame:
    """Primary endpoint A: partial Spearman rho, disagreement vs. Dice, controlling for entropy.

    Args:
        cohort_name: Recorded in the `cohort` column.
        joined: One row per case, columns including
            `cfg.case.score_column`, `cfg.case.control_column`,
            `cfg.case.metric_column` -- see `_join_case_tables`.
        cfg: `cfg.analysis.detection` (exposes `case`, `bootstrap`).
        generator: An explicit `np.random.Generator` used for the bootstrap
            CI, shared across cohorts and endpoints in one run so the whole
            run is reproducible from one seed.

    Returns:
        A one-row `DataFrame`: `cohort, n, score_column, control_column,
        metric_column, rho_score, rho_control, rho_partial, ci_lo, ci_hi,
        contains_zero, p_boot, n_boot`. `rho_score` / `rho_control` are the
        RAW (non-partial) correlations, reported so a reader can see how
        much of `rho_partial` was already visible without controlling for
        entropy. `p_boot` is a two-sided bootstrap p-value for
        `rho_partial` against a null of 0, from the same replicate
        distribution the CI is built from (`_bootstrap_two_sided_p`).
    """
    case_cfg = cfg.case
    score_col = str(case_cfg.score_column)
    control_col = str(case_cfg.control_column)
    metric_col = str(case_cfg.metric_column)

    score = joined[score_col].to_numpy(dtype=np.float64)
    control = joined[control_col].to_numpy(dtype=np.float64)
    metric = joined[metric_col].to_numpy(dtype=np.float64)

    n_boot = int(cfg.bootstrap.n_boot)
    ci = float(cfg.bootstrap.ci)

    result, replicates = partial_spearman_ci(
        score, metric, control, generator=generator, n_boot=n_boot, ci=ci, return_replicates=True
    )
    p_boot = _bootstrap_two_sided_p(replicates, n_boot, null=0.0)

    row = {
        "cohort": cohort_name,
        "n": int(result.n),
        "score_column": score_col,
        "control_column": control_col,
        "metric_column": metric_col,
        "rho_score": spearman(score, metric),
        "rho_control": spearman(control, metric),
        "rho_partial": result.point,
        "ci_lo": result.lo,
        "ci_hi": result.hi,
        "contains_zero": bool(result.contains_zero),
        "p_boot": p_boot,
        "n_boot": n_boot,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Voxel-level endpoint
# ---------------------------------------------------------------------------


def _predicted_dilated_mask(
    pred_wt: np.ndarray,
    spacing: tuple[float, float, float],
    dilation_mm: float,
    cohort_name: str,
    case_id: str,
) -> np.ndarray:
    """Label-free voxel-sampling mask: within `dilation_mm` of the PREDICTED whole-tumour mask.

    Built from the prediction alone, never the ground-truth label -- see
    this module's top-of-file docstring for why that distinction is
    load-bearing. Uses `scipy.ndimage.distance_transform_edt` with no
    `sampling=` argument, so the returned distance is in VOXEL units at an
    assumed isotropic grid; dividing `dilation_mm` by the SMALLEST spacing
    axis (rather than resampling per-axis) gives the more permissive
    (larger) bound on the coarser axes, which keeps this one easy-to-audit
    line rather than a per-axis anisotropic dilation.

    Args:
        pred_wt: Predicted whole-tumour boolean mask, `(D, H, W)`.
        spacing: Voxel spacing in mm, `(D, H, W)` order, from `meta.json`.
        dilation_mm: Dilation radius in millimetres.
        cohort_name: Used only in the empty-mask fallback warning.
        case_id: Used only in the empty-mask fallback warning.

    Returns:
        Boolean mask, same shape as `pred_wt`. Falls back to an all-True
        mask (the whole volume) when the predicted WT is empty -- a case
        the model missed entirely still needs voxels to sample from.
    """
    if not pred_wt.any():
        logger.warning(
            "voxel_level_table: cohort %r case %r has an empty predicted WT mask; falling back "
            "to the whole volume for voxel sampling.",
            cohort_name,
            case_id,
        )
        return np.ones_like(pred_wt, dtype=bool)
    distance_voxels = distance_transform_edt(~pred_wt)
    return distance_voxels <= (dilation_mm / min(spacing))


def _process_voxel_case(
    case_id: str,
    npz_path: Path,
    prep_dir: Path,
    mask_mode: str,
    dilation_mm: float,
    max_voxels: int,
    generator: np.random.Generator,
    cohort_name: str,
) -> dict[str, dict[str, np.ndarray]] | None:
    """Loads one case's saved ambiguity maps, samples a label-free voxel subset, and scores error.

    Args:
        case_id: The case to process.
        npz_path: `<shard_dir>/<case_id>.npz`, holding fp16
            `disagreement`, `entropy_cnn`, `entropy_swin`, `logits`
            (each `(3, D, H, W)`), as written by
            `scripts/extract_ambiguity.py`.
        prep_dir: Root of the preprocessed tree (for the label and spacing).
        mask_mode: `"predicted_dilated"` or `"all"`.
        dilation_mm: Forwarded to `_predicted_dilated_mask`.
        max_voxels: Maximum voxels drawn from the mask.
        generator: An explicit `np.random.Generator`.
        cohort_name: Used only in log messages.

    Returns:
        `None` if the case has no label on disk, or its sampling mask is
        empty. Otherwise a dict keyed by `"ET"`, `"TC"`, `"WT"`, `"ANY"`,
        each mapping to `{"score": ..., "control": ..., "positive": ...}`
        -- disagreement, single-pass entropy (computed from `logits` with
        `neurovision.analysis.detection._entropy_from_logits`, the same
        formula the fusion module itself uses under fp16), and a boolean
        prediction-error indicator, all at the SAME `max_voxels`-sized
        sample of voxel positions. `"ANY"` uses the region-MEAN score/
        control maps and `positive.any(axis=0)` -- one row that pools
        across ET/TC/WT for the family-level endpoint.

    Raises:
        ValueError: `mask_mode` is neither `"predicted_dilated"` nor
            `"all"`.
    """
    with np.load(npz_path) as data:
        disagreement = data["disagreement"].astype(np.float32)  # (3, D, H, W)
        logits = data["logits"].astype(np.float32)  # (3, D, H, W)
    entropy = _entropy_from_logits(logits)  # (3, D, H, W)

    loaded = _load_label_and_spacing(prep_dir, case_id)
    if loaded is None:
        return None
    label, spacing = loaded

    logits_t = torch.from_numpy(logits).unsqueeze(0)  # (1, 3, D, H, W)
    pp_cfg = _resolve_postprocess_cfg(None)  # project default post-processing chain
    regions_t = _binarize_regions(logits_t, threshold=0.5)
    regions_t = _apply_postprocess_steps(regions_t, pp_cfg)
    pred = regions_t[0].numpy().astype(bool)  # (3, D, H, W)

    label_t = torch.as_tensor(np.asarray(label))
    target = classes_to_regions(label_t)[0].numpy().astype(bool)  # (3, D, H, W)

    positive = pred != target  # (3, D, H, W), per-voxel per-region error

    if mask_mode == "predicted_dilated":
        mask = _predicted_dilated_mask(pred[2], spacing, dilation_mm, cohort_name, case_id)
    elif mask_mode == "all":
        mask = np.ones(pred.shape[1:], dtype=bool)
    else:
        raise ValueError(
            f"voxel_level_table: unknown analysis.detection.voxel.mask={mask_mode!r}; expected "
            "'predicted_dilated' or 'all'."
        )

    flat_idx = np.flatnonzero(mask.reshape(-1))
    if flat_idx.size == 0:
        logger.warning(
            "voxel_level_table: cohort %r case %r sampling mask is empty; skipping.",
            cohort_name,
            case_id,
        )
        return None

    n_draw = min(max_voxels, flat_idx.size)
    drawn = generator.choice(flat_idx, size=n_draw, replace=False)

    out: dict[str, dict[str, np.ndarray]] = {}
    for i, region in enumerate(REGION_NAMES):
        out[region] = {
            "score": disagreement[i].reshape(-1)[drawn],
            "control": entropy[i].reshape(-1)[drawn],
            "positive": positive[i].reshape(-1)[drawn],
        }
    out["ANY"] = {
        "score": disagreement.mean(axis=0).reshape(-1)[drawn],
        "control": entropy.mean(axis=0).reshape(-1)[drawn],
        "positive": positive.any(axis=0).reshape(-1)[drawn],
    }
    return out


def _cluster_bootstrap_residual_auroc(
    score: np.ndarray,
    control: np.ndarray,
    positive: np.ndarray,
    case_ids: np.ndarray,
    generator: np.random.Generator,
    n_boot: int,
    ci: float,
) -> tuple[float, float, float, np.ndarray]:
    """Cluster bootstrap CI for `auroc_residual`, resampling CASE indices, not voxels.

    Voxels within one case are spatially correlated (neighbouring voxels
    share nearly the same disagreement and entropy value), so resampling
    voxels independently would understate the true sampling variability of
    the reported interval -- the same reasoning
    `neurovision.analysis.statistics.paired_bootstrap_ci` resamples case
    indices rather than raw per-case scores. Each replicate draws cases
    WITH replacement and concatenates every already-sampled voxel belonging
    to each drawn case.

    Args:
        score: Pooled voxel-level scores, one entry per sampled voxel.
        control: Pooled voxel-level control values, same length as `score`.
        positive: Pooled voxel-level boolean error indicator, same length.
        case_ids: The case id each sampled voxel came from, same length.
        generator: An explicit `np.random.Generator`.
        n_boot: Number of bootstrap replicates.
        ci: Confidence level, e.g. `0.95`.

    Returns:
        `(point, lo, hi, replicates)`. `point` is `auroc_residual` on the
        un-resampled pooled sample. `replicates` is the raw
        (possibly-NaN-containing) length-`n_boot` array. `lo`/`hi` are NaN
        if every replicate is non-finite.
    """
    unique_cases = np.unique(case_ids)
    n = unique_cases.size
    case_to_idx = {c: np.flatnonzero(case_ids == c) for c in unique_cases}

    point = residualised_auroc(score, control, positive)["auroc_residual"]

    replicates = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        drawn_cases = generator.choice(unique_cases, size=n, replace=True)
        idx = np.concatenate([case_to_idx[c] for c in drawn_cases])
        replicates[b] = residualised_auroc(score[idx], control[idx], positive[idx])[
            "auroc_residual"
        ]

    valid = replicates[np.isfinite(replicates)]
    if valid.size == 0:
        return point, float("nan"), float("nan"), replicates
    alpha = 1.0 - ci
    lo, hi = (
        float(v) for v in np.percentile(valid, [100.0 * alpha / 2.0, 100.0 - 100.0 * alpha / 2.0])
    )
    return point, lo, hi, replicates


def voxel_level_table(
    cohort_name: str,
    npz_paths: Mapping[str, Path],
    prep_dir: str | Path,
    cfg: DictConfig,
    generator: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Primary endpoint B: residualised AUROC for per-voxel prediction error.

    Args:
        cohort_name: Recorded in the `cohort` column.
        npz_paths: `case_id -> <shard_dir>/<case_id>.npz`, from
            `load_cohort`.
        prep_dir: Root of the preprocessed tree (for labels and spacing).
        cfg: `cfg.analysis.detection` (exposes `voxel`, `bootstrap`).
        generator: An explicit `np.random.Generator`, shared with
            `case_level_table` for one run's reproducibility.

    Returns:
        `(table, diagnostics)`. `table` has one row per group in
        `("ET", "TC", "WT", "ANY")`: `cohort, region, n_cases, n_voxels,
        auroc_score, auroc_control, auroc_residual, resid_ci_lo,
        resid_ci_hi, p_boot`. Empty (with `diagnostics = {"enabled":
        False}`) when `cfg.voxel.enabled` is false. `diagnostics` records
        what the run actually did (case counts, mask settings) for the
        provenance dump -- it is not part of the pre-registered endpoint
        itself.

    Raises:
        ValueError: Zero cases survive loading (every case had no label or
            an empty sampling mask), or `cfg.voxel.mask` names neither
            `"predicted_dilated"` nor `"all"` (raised from
            `_process_voxel_case`).
    """
    voxel_cfg = cfg.voxel
    prep_dir = Path(prep_dir)
    if not bool(voxel_cfg.enabled):
        logger.info(
            "Cohort %r: voxel-level endpoint disabled (analysis.detection.voxel.enabled=false).",
            cohort_name,
        )
        return pd.DataFrame(), {"enabled": False}

    case_ids = sorted(npz_paths)
    max_cases = voxel_cfg.max_cases
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    max_voxels = int(voxel_cfg.max_voxels_per_case)
    mask_mode = str(voxel_cfg.mask)
    dilation_mm = float(voxel_cfg.dilation_mm)

    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        group: {"score": [], "control": [], "positive": [], "case": []} for group in _VOXEL_GROUPS
    }
    n_cases_used = 0
    n_cases_skipped = 0

    for case_id in case_ids:
        result = _process_voxel_case(
            case_id,
            npz_paths[case_id],
            prep_dir,
            mask_mode,
            dilation_mm,
            max_voxels,
            generator,
            cohort_name,
        )
        if result is None:
            n_cases_skipped += 1
            continue
        n_cases_used += 1
        for group, arrays in result.items():
            n = arrays["score"].size
            pooled[group]["score"].append(arrays["score"])
            pooled[group]["control"].append(arrays["control"])
            pooled[group]["positive"].append(arrays["positive"])
            pooled[group]["case"].append(np.full(n, case_id, dtype=object))

    if n_cases_used == 0:
        raise ValueError(
            f"voxel_level_table: cohort {cohort_name!r} produced zero usable cases (no label, or "
            "an empty sampling mask, on every candidate case)."
        )

    n_boot_voxel = min(int(cfg.bootstrap.n_boot), 2000)
    ci = float(cfg.bootstrap.ci)

    rows: list[dict[str, Any]] = []
    for group in _VOXEL_GROUPS:
        score = np.concatenate(pooled[group]["score"])
        control = np.concatenate(pooled[group]["control"])
        positive = np.concatenate(pooled[group]["positive"])
        case_arr = np.concatenate(pooled[group]["case"])

        base = residualised_auroc(score, control, positive)
        point, lo, hi, replicates = _cluster_bootstrap_residual_auroc(
            score, control, positive, case_arr, generator, n_boot_voxel, ci
        )
        p_boot = _bootstrap_two_sided_p(replicates, n_boot_voxel, null=0.5)

        rows.append(
            {
                "cohort": cohort_name,
                "region": group,
                "n_cases": n_cases_used,
                "n_voxels": int(score.size),
                "auroc_score": base["auroc_score"],
                "auroc_control": base["auroc_control"],
                "auroc_residual": point,
                "resid_ci_lo": lo,
                "resid_ci_hi": hi,
                "p_boot": p_boot,
            }
        )

    table = pd.DataFrame(rows)
    diagnostics = {
        "enabled": True,
        "mask": mask_mode,
        "dilation_mm": dilation_mm,
        "max_voxels_per_case": max_voxels,
        "n_boot_voxel": n_boot_voxel,
        "n_cases_available": len(case_ids),
        "n_cases_used": n_cases_used,
        "n_cases_skipped": n_cases_skipped,
    }
    return table, diagnostics


# ---------------------------------------------------------------------------
# Family correction and verdict
# ---------------------------------------------------------------------------


def build_family_table(
    case_rows: Mapping[str, Mapping[str, Any]],
    voxel_any_rows: Mapping[str, Mapping[str, Any]],
    alpha: float,
) -> pd.DataFrame:
    """Builds and Holm-corrects the pre-registered 6-test family (2 endpoints x 3 cohorts).

    Args:
        case_rows: `cohort_name -> case_level_table`'s one row (as a dict
            or `Series`), for every cohort that finished.
        voxel_any_rows: `cohort_name -> voxel_level_table`'s `"ANY"`-region
            row, for every cohort that finished.
        alpha: Family-wise significance level, passed straight to
            `neurovision.analysis.statistics.holm_bonferroni`.

    Returns:
        Columns `cohort, endpoint, statistic, p_raw, p_holm, reject`.
        Ordered as every cohort's `"case"` row followed by every cohort's
        `"voxel_any"` row (cohort names sorted within each block) -- with
        all three cohorts present this is exactly 2 x 3 = 6 rows, matching
        the pre-registration's declared family. A cohort present in only
        one of `case_rows` / `voxel_any_rows` is dropped from the family
        entirely rather than contributing a lone row: Holm's family must be
        fixed before any p-value is examined, and a family whose size
        depends on which endpoint happened to finish for which cohort
        contradicts that.

    Raises:
        ValueError: No cohort has both endpoints available.
    """
    common = sorted(set(case_rows) & set(voxel_any_rows))
    if not common:
        raise ValueError(
            "build_family_table: no cohort has both the case-level and voxel-level endpoint "
            "available; nothing to correct."
        )

    records: list[dict[str, Any]] = []
    for name in common:
        records.append(
            {
                "cohort": name,
                "endpoint": "case",
                "statistic": "rho_partial",
                "p_raw": float(case_rows[name]["p_boot"]),
            }
        )
    for name in common:
        records.append(
            {
                "cohort": name,
                "endpoint": "voxel_any",
                "statistic": "auroc_residual",
                "p_raw": float(voxel_any_rows[name]["p_boot"]),
            }
        )

    pvals = [r["p_raw"] for r in records]
    adjusted, reject = holm_bonferroni(pvals, alpha=alpha)
    for record, adj, rej in zip(records, adjusted, reject):
        record["p_holm"] = float(adj)
        record["reject"] = bool(rej)
    return pd.DataFrame(records)


def build_verdict(
    case_rows: Mapping[str, Mapping[str, Any]],
    voxel_any_rows: Mapping[str, Mapping[str, Any]],
    *,
    rho_threshold: float = _PASS_RHO_THRESHOLD,
    auroc_threshold: float = _PASS_AUROC_THRESHOLD,
    external_cohorts: Sequence[str] = _EXTERNAL_COHORTS,
) -> dict[str, Any]:
    """Applies the pre-registered Gate 1 decision rule, verbatim.

    From `docs/research/preregistration_ambiguity.md`'s "Decision rule"
    table:

    - **pass**: on at least one EXTERNAL cohort (`ssa` or `ped`),
      `abs(rho_partial) >= 0.20` with a CI excluding zero, AND that
      cohort's voxel-level residualised AUROC (`"ANY"` region) `>= 0.60`
      with its CI excluding 0.5.
    - **partial**: some endpoint's CI excludes its null (0 for
      `rho_partial`, 0.5 for `auroc_residual`) but its magnitude falls
      below the threshold above. Checked over every cohort with both
      endpoints available, not only the external ones -- the
      preregistration restricts the EXTERNAL-cohort requirement explicitly
      to the pass condition; the partial outcome ("efficiency, not
      superiority") is not narrowed the same way.
    - **fail**: otherwise.

    Args:
        case_rows: `cohort_name -> case_level_table`'s one row, for every
            cohort that finished (needs `rho_partial`, `contains_zero`).
        voxel_any_rows: `cohort_name -> voxel_level_table`'s `"ANY"`-region
            row, for every cohort that finished (needs `auroc_residual`,
            `resid_ci_lo`, `resid_ci_hi`).
        rho_threshold: The pre-registered case-level magnitude threshold.
            An argument only so tests can probe the boundary directly --
            never read from config or overridden at the CLI: rule 2 of the
            pre-registration is "the thresholds are not adjusted after
            seeing data".
        auroc_threshold: The pre-registered voxel-level magnitude
            threshold, same caveat.
        external_cohorts: Which cohort names count as "external" for the
            pass condition.

    Returns:
        `{"verdict": "pass" | "partial" | "fail", "thresholds": {...},
        "external_cohorts": [...], "passed_cohorts": [...],
        "partial_cohorts": [...], "per_cohort": {...},
        "preregistration": "docs/research/preregistration_ambiguity.md"}`.
    """
    common = sorted(set(case_rows) & set(voxel_any_rows))
    per_cohort: dict[str, dict[str, Any]] = {}
    passed: list[str] = []
    partial_hit: list[str] = []

    for name in common:
        c = case_rows[name]
        v = voxel_any_rows[name]

        rho = float(c["rho_partial"])
        rho_ci_excludes_zero = not bool(c["contains_zero"])

        auroc_residual = float(v["auroc_residual"])
        resid_lo = float(v["resid_ci_lo"])
        resid_hi = float(v["resid_ci_hi"])
        auroc_ci_excludes_half = not (resid_lo <= 0.5 <= resid_hi)

        meets_threshold = abs(rho) >= rho_threshold and auroc_residual >= auroc_threshold
        clears_ci = rho_ci_excludes_zero and auroc_ci_excludes_half
        is_external = name in external_cohorts

        per_cohort[name] = {
            "external": is_external,
            "rho_partial": rho,
            "rho_ci_excludes_zero": rho_ci_excludes_zero,
            "auroc_residual": auroc_residual,
            "auroc_ci_excludes_half": auroc_ci_excludes_half,
            "meets_threshold": meets_threshold,
        }

        if is_external and meets_threshold and clears_ci:
            passed.append(name)
        if rho_ci_excludes_zero or auroc_ci_excludes_half:
            partial_hit.append(name)

    if passed:
        verdict = "pass"
    elif partial_hit:
        verdict = "partial"
    else:
        verdict = "fail"

    return {
        "verdict": verdict,
        "thresholds": {"rho_partial_abs": rho_threshold, "auroc_residual": auroc_threshold},
        "external_cohorts": list(external_cohorts),
        "passed_cohorts": passed,
        "partial_cohorts": partial_hit,
        "per_cohort": per_cohort,
        "preregistration": "docs/research/preregistration_ambiguity.md",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _log_and_print_summary(
    case_level_df: pd.DataFrame,
    voxel_level_df: pd.DataFrame,
    family_df: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    """Logs and prints a compact end-of-run summary.

    Args:
        case_level_df: Every processed cohort's `case_level_table` row.
        voxel_level_df: Every processed cohort's `voxel_level_table` rows
            (all regions), may be empty if the voxel endpoint was disabled.
        family_df: `build_family_table`'s output.
        verdict: `build_verdict`'s output.
    """
    lines = [
        "=" * 70,
        "Gate 1 -- inter-branch disagreement as a failure detector",
        "=" * 70,
        f"  cohorts processed: {sorted(case_level_df['cohort'].unique().tolist())}",
        f"  verdict: {verdict['verdict']!r}",
    ]
    for _, row in case_level_df.iterrows():
        lines.append(
            f"    [{row['cohort']}] case-level rho_partial={row['rho_partial']:.3f} "
            f"CI=({row['ci_lo']:.3f}, {row['ci_hi']:.3f}) p_boot={row['p_boot']:.4f}"
        )
    if not voxel_level_df.empty:
        for _, row in voxel_level_df.loc[voxel_level_df["region"] == "ANY"].iterrows():
            lines.append(
                f"    [{row['cohort']}] voxel-level (ANY) "
                f"auroc_residual={row['auroc_residual']:.3f} "
                f"CI=({row['resid_ci_lo']:.3f}, {row['resid_ci_hi']:.3f}) "
                f"p_boot={row['p_boot']:.4f}"
            )
    n_reject = int(family_df["reject"].sum()) if not family_df.empty else 0
    lines.append(f"  Holm family: {len(family_df)} test(s), {n_reject} rejected")

    # print only, not logger.info as well -- setup_logging's StreamHandler
    # already targets stdout, so doing both would print this block twice.
    # Matches scripts/evaluate.py's / scripts/extract_ambiguity.py's summary.
    print("\n".join(lines))


def run_detection(cfg: DictConfig) -> Path:
    """Runs Gate 1 over every configured cohort that is ready, and writes every output.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The path of `detection_family.csv`.

    Raises:
        ValueError: No cohort was ready, or (from `build_family_table`) no
            cohort has both endpoints available.
    """
    detection_cfg = cfg.analysis.detection
    out_dir = ensure_dir(str(detection_cfg.out_dir))
    generator = np.random.default_rng(int(detection_cfg.seed))

    case_frames: list[pd.DataFrame] = []
    voxel_tables: list[pd.DataFrame] = []
    case_rows: dict[str, dict[str, Any]] = {}
    voxel_any_rows: dict[str, dict[str, Any]] = {}
    cohort_provenance: dict[str, Any] = {}

    for cohort_cfg in detection_cfg.cohorts:
        name = str(cohort_cfg.name)
        ambiguity_dirs = [Path(str(d)) for d in cohort_cfg.ambiguity_dirs]
        eval_dir = Path(str(cohort_cfg.eval_dir))
        prep_dir = Path(str(cohort_cfg.prep_dir))

        missing_ambiguity = [d for d in ambiguity_dirs if not d.is_dir()]
        per_case_metrics_path = eval_dir / "per_case_metrics.csv"
        if missing_ambiguity or not per_case_metrics_path.is_file():
            # Cohorts finish extracting at different times -- a partial run
            # must still produce every cohort that IS ready, not raise
            # because one is not.
            logger.warning(
                "Cohort %r not ready yet, skipping. Missing ambiguity dir(s): %s. "
                "per_case_metrics.csv present: %s.",
                name,
                [str(d) for d in missing_ambiguity],
                per_case_metrics_path.is_file(),
            )
            continue

        logger.info("Cohort %r: loading ambiguity shard(s) %s", name, ambiguity_dirs)
        ambiguity_summary, npz_paths = load_cohort(cohort_cfg)
        per_case_metrics = pd.read_csv(per_case_metrics_path, index_col="case_id")

        cache_path = out_dir / f"entropy_cache_{name}.csv"
        entropy = entropy_table(eval_dir, list(ambiguity_summary.index), cache_path)

        joined = _join_case_tables(ambiguity_summary, entropy, per_case_metrics, name)
        case_row_df = case_level_table(name, joined, detection_cfg, generator)
        case_frames.append(case_row_df)
        case_rows[name] = case_row_df.iloc[0].to_dict()

        voxel_table, voxel_diag = voxel_level_table(
            name, npz_paths, prep_dir, detection_cfg, generator
        )
        if not voxel_table.empty:
            voxel_tables.append(voxel_table)
            voxel_any_rows[name] = voxel_table.loc[voxel_table["region"] == "ANY"].iloc[0].to_dict()

        cohort_provenance[name] = {
            "ambiguity_dirs": [str(d.resolve()) for d in ambiguity_dirs],
            "eval_dir": str(eval_dir.resolve()),
            "prep_dir": str(prep_dir.resolve()),
            "n_cases_joined": int(len(joined)),
            "voxel": voxel_diag,
        }

    if not case_rows:
        raise ValueError(
            "run_detection: no cohort was ready (every cohort is missing its ambiguity_dirs or "
            "its eval_dir's per_case_metrics.csv) -- nothing to compute."
        )

    case_level_df = pd.concat(case_frames, ignore_index=True)
    case_level_df.to_csv(out_dir / "detection_case_level.csv", index=False)

    voxel_level_df = pd.concat(voxel_tables, ignore_index=True) if voxel_tables else pd.DataFrame()
    voxel_level_df.to_csv(out_dir / "detection_voxel_level.csv", index=False)

    family_df = build_family_table(case_rows, voxel_any_rows, alpha=float(detection_cfg.alpha))
    family_df.to_csv(out_dir / "detection_family.csv", index=False)

    verdict = build_verdict(case_rows, voxel_any_rows)
    write_json(verdict, out_dir / "detection_verdict.json")

    # Provenance that only ever lived in a terminal log cannot be traced
    # months later -- the same reason localize_config.yaml / ambiguity_config
    # .yaml / burden_config.yaml exist.
    config_record = OmegaConf.to_container(detection_cfg, resolve=True)
    config_record["cohort_provenance"] = cohort_provenance
    write_yaml(config_record, out_dir / "detection_config.yaml")

    _log_and_print_summary(case_level_df, voxel_level_df, family_df, verdict)

    return out_dir / "detection_family.csv"


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs Gate 1 over every configured cohort that is ready, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    run_detection(cfg)


if __name__ == "__main__":
    main()
