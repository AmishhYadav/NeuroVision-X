"""Hydra entry point for Phase G, the end-to-end error budget.

`docs/research/error_budget_protocol.md` fixes every definition this script uses --
read it FIRST, before touching anything here. `neurovision.analysis.error_budget` is
the pure-arithmetic half (no file I/O, no model, no torch); this script is exactly the
other half: it reads artifacts already on disk (saved logits, the trained QC model's
predictions, the frozen conformal fit, the deployed gate's frozen thresholds), calls
that module's functions, and writes the protocol's output files. It fits no threshold,
degrades no mask, and runs no segmentation model -- the only model this script runs is
the already-trained `SegQC` regressor, once per cohort, via
`neurovision.analysis.gatekeeper_calibration.qc_predicted_dice_table` (the same
identity path `scripts/calibrate_gatekeeper.py` already uses).

Example usage (the `model=segqc` override is REQUIRED -- see below):

    python scripts/error_budget.py model=segqc

## Why `model=segqc` is required

Every cohort's `predicted_dice_<R>` signal comes from
`neurovision.analysis.gatekeeper_calibration.qc_predicted_dice_table`, which builds
and loads a trained `SegQC` checkpoint via `neurovision.models.qc.build_segqc(cfg)`.
The root config's default model group is `unet3d`, which has none of the keys
`build_segqc` reads -- this is the SAME requirement `scripts/calibrate_gatekeeper.py`
and `scripts/train_qc.py` already state, for the same reason. `run_error_budget`
checks `cfg.model.name` and raises a clear `ValueError` before any cohort is
processed, rather than letting the first cohort fail partway through with a
confusing attribute error.

## Nothing here is refitted

The gate's `Thresholds` come from `neurovision.inference.gatekeeper.load_thresholds`
only -- the deployed, frozen `outputs/gatekeeper/thresholds.json` -- never
recalibrated. The conformal threshold each cohort's `conformal_band_<R>` and
`miss_rate_<R>` are read at is `neurovision.analysis.gatekeeper_calibration.
resolve_fitted_thresholds`'s value from `cfg.clinical.gatekeeper.conformal_dir`'s
`fit.json`, fitted once on val, long before this script ever runs. The ONE place this
script refits anything is `coverage_curve`'s own quantile sweep (G3), which the
protocol explicitly asks for, on the frozen val `calibration_table.csv`, at a
pre-registered quantile grid -- see `neurovision.analysis.error_budget.coverage_curve`'s
own docstring for why that is a description of the deployed operating point, not a new
calibration.

## CPU only, every path from config

`qc_predicted_dice_table` is already hard-pinned to CPU internally (see its own
module docstring) -- this calibration-adjacent step runs on the Mac, never the GPU
cluster (`CLAUDE.md`'s machine split). No path in this file is hardcoded: every
input and output location comes from `cfg.analysis.error_budget`,
`cfg.clinical.gatekeeper` or `cfg.analysis.qc_validate.checkpoint`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis import error_budget, gatekeeper_calibration
from neurovision.analysis.statistics import load_per_case
from neurovision.inference.gatekeeper import load_thresholds
from neurovision.uncertainty.conformal import CaseLossCurve, load_curves_npz
from neurovision.utils.io import ensure_dir, write_yaml
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Tolerance for the exact-match threshold-grid lookup below -- matches
# neurovision.uncertainty.conformal's own private `_threshold_index` default,
# since `_miss_rate_at_threshold` mirrors that helper's semantics exactly.
_THRESHOLD_TOL = 1e-9


# ---------------------------------------------------------------------------
# cfg.model must be the segqc group
# ---------------------------------------------------------------------------


def _require_segqc_model(cfg: DictConfig) -> None:
    """Raises early if `cfg.model` is not composed as the `segqc` model group.

    Every cohort's `predicted_dice_<R>` signal is built by
    `qc_predicted_dice_table`, which builds a `SegQC` via
    `neurovision.models.qc.build_segqc(cfg)` -- that function reads
    `cfg.model.in_channels`, `cfg.model.widths` etc, keys the default `unet3d`
    model group does not have. Failing loudly here, before any cohort's
    signals are built, is cheaper than a confusing attribute error partway
    through the first cohort -- the same reasoning
    `scripts/calibrate_gatekeeper.py`'s module docstring gives for the
    identical requirement.

    Args:
        cfg: The full composed Hydra config.

    Raises:
        ValueError: `cfg.model.name` is not `"segqc"`.
    """
    name = cfg.model.get("name", None)
    if str(name) != "segqc":
        raise ValueError(
            "error_budget: cfg.model must be composed as the segqc model group (got "
            f"cfg.model.name={name!r}). Run with `model=segqc`, e.g. "
            "`python scripts/error_budget.py model=segqc` -- the predicted_dice signal "
            "is built via neurovision.models.qc.build_segqc(cfg), which reads keys the "
            "default unet3d model group does not have."
        )


# ---------------------------------------------------------------------------
# Cohort resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """One Phase G cohort's resolved paths.

    Attributes:
        name: Cohort name, e.g. `"test"`, `"ssa"`, `"ped"`.
        eval_dir: A `scripts/evaluate.py` output directory holding
            `logits/*.npy` and `per_case_metrics.csv`.
        prep_dir: Root of the preprocessed BraTS data for this cohort.
        conformal_curves: Path to this cohort's already-extracted
            `curves.npz` (`scripts/conformal.py`'s own output).
    """

    name: str
    eval_dir: Path
    prep_dir: Path
    conformal_curves: Path


def resolve_cohorts(cfg: DictConfig) -> list[CohortSpec]:
    """Resolves and validates every configured cohort, before any case is processed.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        One `CohortSpec` per entry of `cfg.analysis.error_budget.cohorts`, in
        the configured order.

    Raises:
        FileNotFoundError: A cohort's `eval_dir` has no `logits/` subdirectory,
            or its `conformal_curves` file does not exist. Raised before any
            per-case work starts, for any cohort -- mirrors
            `scripts/validate_qc.py::resolve_cohorts`.
    """
    cohorts: list[CohortSpec] = []
    for entry in cfg.analysis.error_budget.cohorts:
        name = str(entry.name)
        eval_dir = Path(str(entry.eval_dir))
        prep_dir = Path(str(entry.prep_dir))
        conformal_curves = Path(str(entry.conformal_curves))

        logits_dir = eval_dir / "logits"
        if not logits_dir.is_dir():
            raise FileNotFoundError(
                f"error_budget: cohort {name!r}'s eval_dir has no logits/ directory: "
                f"{logits_dir}. Re-run scripts/evaluate.py with "
                "inference.evaluation.save_logits=true against this eval_dir."
            )
        if not conformal_curves.is_file():
            raise FileNotFoundError(
                f"error_budget: cohort {name!r} has no conformal_curves file at "
                f"{conformal_curves}. Re-run scripts/conformal.py against this cohort first."
            )
        cohorts.append(
            CohortSpec(
                name=name, eval_dir=eval_dir, prep_dir=prep_dir, conformal_curves=conformal_curves
            )
        )
    return cohorts


def _case_ids(eval_dir: Path, max_cases: int | None, cohort_name: str) -> list[str]:
    """Case ids with both saved logits and a row in `per_case_metrics.csv`.

    Mirrors `scripts/validate_qc.py::_shared_case_ids`'s pattern (log dropped
    ids, truncate deterministically with `max_cases`), but intersects against
    `per_case_metrics.csv`'s own `case_id` index instead of a `prep_dir` case
    listing: this script's signals already reach into `prep_dir` internally
    (via `qc_predicted_dice_table`), so the one thing worth checking here is
    whether a case has both saved logits AND a published Dice to build the
    usability label from.

    Args:
        eval_dir: A `scripts/evaluate.py` output directory.
        max_cases: If not `None`, truncates the sorted id list to this many
            entries -- a deterministic subsetting knob, not a random subsample.
        cohort_name: Used only in log messages.

    Returns:
        Sorted, deduplicated list of shared case ids.

    Raises:
        ValueError: No case id is shared between `eval_dir/logits` and
            `eval_dir/per_case_metrics.csv`.
    """
    logits_dir = eval_dir / "logits"
    logits_ids = {p.stem for p in logits_dir.glob("*.npy")}
    metrics = load_per_case(eval_dir / "per_case_metrics.csv")
    metrics_ids = {str(c) for c in metrics.index}
    shared = sorted(logits_ids & metrics_ids)

    excluded = logits_ids ^ metrics_ids  # present in exactly one of the two sources
    if excluded:
        logger.warning(
            "error_budget: cohort %r excluded %d case id(s) present in only one of "
            "<eval_dir>/logits and per_case_metrics.csv -- not scored.",
            cohort_name,
            len(excluded),
        )

    if not shared:
        raise ValueError(
            f"error_budget: cohort {cohort_name!r} has no case id shared between "
            f"{logits_dir} and per_case_metrics.csv; nothing to score."
        )

    if max_cases is not None:
        shared = shared[: int(max_cases)]
    logger.info("error_budget: cohort %s -- %d case id(s) to score.", cohort_name, len(shared))
    return shared


# ---------------------------------------------------------------------------
# Signal assembly: conformal band width / miss rate, reshaped to columns
# ---------------------------------------------------------------------------


def _reshape_per_region_case_values(
    values_by_region: Mapping[str, Mapping[str, float]], column_prefix: str
) -> pd.DataFrame:
    """Reshapes `{region: {case_id: value}}` into one row per `case_id`, columns `<prefix>_<R>`.

    Mirrors `neurovision.analysis.gatekeeper_calibration.build_gatekeeper_calibration_table`'s
    own `conformal_rows` reshape of `case_conformal_band_widths`'s output -- copied
    here because this script needs the identical reshape for two different value
    dicts (conformal band width and, separately, miss rate).

    Args:
        values_by_region: `{region: {case_id: value}}`, e.g. from
            `case_conformal_band_widths` or a per-case miss-rate lookup.
        column_prefix: The column-name prefix, e.g. `"conformal_band"` or
            `"miss_rate"` -- the resulting column is `f"{column_prefix}_{region}"`.

    Returns:
        One row per `case_id` seen in any region, with a `f"{column_prefix}_{R}"`
        column per region in `values_by_region` (missing for a case/region pair
        with no entry -- left as a genuinely absent value, i.e. NaN once read
        back through pandas, not fabricated here).
    """
    rows: dict[str, dict[str, Any]] = {}
    for region, per_case in values_by_region.items():
        for case_id, value in per_case.items():
            rows.setdefault(case_id, {"case_id": case_id})[f"{column_prefix}_{region}"] = value
    return pd.DataFrame(list(rows.values()))


def _miss_rate_at_threshold(
    curve: CaseLossCurve, threshold: float, tol: float = _THRESHOLD_TOL
) -> float:
    """The per-case conformal miss rate at `threshold`'s exact position in `curve.thresholds`.

    Mirrors `neurovision.uncertainty.conformal`'s private `_threshold_index`
    lookup (exact match to `tol`, nearest-value error message) rather than
    importing it -- that name is private for a reason, and this script needs
    the identical exact-match-with-tolerance semantics for exactly one column,
    not the rest of that module's surface.

    Args:
        curve: The `CaseLossCurve` to read `threshold`'s position from.
        threshold: The grid value to look up (a fitted conformal threshold).
        tol: Maximum allowed absolute difference to count as a match.

    Returns:
        `curve.miss_rate()[idx]` -- NaN when `curve.empty_gt`, since
        `CaseLossCurve.miss_rate()` is already an all-NaN array in that case
        (the loss is genuinely undefined, 0/0; see that method's docstring).

    Raises:
        ValueError: No entry of `curve.thresholds` is within `tol` of
            `threshold`. Names the nearest available value.
    """
    thresholds = np.asarray(curve.thresholds, dtype=np.float64)
    diffs = np.abs(thresholds - threshold)
    idx = int(np.argmin(diffs))
    if diffs[idx] > tol:
        raise ValueError(
            f"_miss_rate_at_threshold: {threshold} is not in the threshold grid for case "
            f"{curve.case_id!r} region {curve.region!r}; nearest available value is "
            f"{thresholds[idx]!r}."
        )
    return float(curve.miss_rate()[idx])


def _miss_rate_table(
    curves_by_region: Mapping[str, Sequence[CaseLossCurve]],
    fitted_thresholds: Mapping[str, float],
    regions: Sequence[str],
) -> pd.DataFrame:
    """Builds the per-case conformal miss-rate table, one `miss_rate_<R>` column per region.

    Args:
        curves_by_region: `{region: [CaseLossCurve, ...]}`, e.g. from `load_curves_npz`.
        fitted_thresholds: `{region: threshold}`, e.g. from `resolve_fitted_thresholds`.
            Must have an entry for every key of `curves_by_region`.
        regions: Region names to build a column for.

    Returns:
        One row per case_id seen in any region's curves, with a `miss_rate_<R>`
        column per region (see `_miss_rate_at_threshold` for what each value means).
    """
    values_by_region: dict[str, dict[str, float]] = {}
    for region in regions:
        threshold = fitted_thresholds[region]
        values_by_region[region] = {
            curve.case_id: _miss_rate_at_threshold(curve, threshold)
            for curve in curves_by_region[region]
        }
    return _reshape_per_region_case_values(values_by_region, "miss_rate")


def _inner_join_signals(
    dice_table: pd.DataFrame,
    conformal_table: pd.DataFrame,
    ood_table: pd.DataFrame,
    cohort_name: str,
) -> pd.DataFrame:
    """Inner-joins the three per-case signal tables on `case_id`, logging any dropped ids.

    Mirrors `neurovision.analysis.gatekeeper_calibration._inner_join_calibration_tables`'s
    own drop-and-log discipline, applied to one cohort's signal tables instead of
    the val calibration table.

    Args:
        dice_table: A frame with `case_id` plus `predicted_dice_<R>` columns.
        conformal_table: A frame with `case_id` plus `conformal_band_<R>` columns.
        ood_table: A frame with `case_id` and `ood_score` columns.
        cohort_name: Used only in log/error messages.

    Returns:
        The inner join of all three on `case_id`.

    Raises:
        ValueError: The join produces zero rows.
    """
    all_ids = (
        set(dice_table["case_id"]) | set(conformal_table["case_id"]) | set(ood_table["case_id"])
    )
    merged = dice_table.merge(conformal_table, on="case_id", how="inner").merge(
        ood_table, on="case_id", how="inner"
    )

    n_dropped = len(all_ids) - len(merged)
    if n_dropped > 0:
        logger.warning(
            "error_budget: cohort %r dropped %d case id(s) present in only some of the "
            "predicted-Dice (n=%d), conformal-band (n=%d) and ood-score (n=%d) tables; "
            "%d case(s) survive.",
            cohort_name,
            n_dropped,
            len(dice_table),
            len(conformal_table),
            len(ood_table),
            len(merged),
        )

    if merged.empty:
        raise ValueError(
            f"error_budget: cohort {cohort_name!r}'s predicted-Dice, conformal-band and "
            "ood-score tables share no case_id at all; nothing to score."
        )
    return merged


# ---------------------------------------------------------------------------
# bar_sweep must include usable_bar
# ---------------------------------------------------------------------------


def _ensure_bar_included(bar_sweep: Sequence[float], usable_bar: float) -> list[float]:
    """Returns `bar_sweep` as a sorted list, guaranteed to include `usable_bar`.

    `per_case_<cohort>.csv`'s own `usable`/`cell` columns, and the taxonomy and
    coverage-curve tables, are all built at `usable_bar` -- so `usable_bar` must
    always be one of `summary.csv`'s rows, never a value the reader has to
    interpolate between two swept points.

    Args:
        bar_sweep: The configured sensitivity-sweep bars.
        usable_bar: The bar the headline `usable` column is defined at.

    Returns:
        `bar_sweep` as a sorted `list[float]`, de-duplicated, with `usable_bar`
        included exactly once whether or not it was already present.
    """
    bars = {float(b) for b in bar_sweep}
    bars.add(float(usable_bar))
    return sorted(bars)


# ---------------------------------------------------------------------------
# stage_reliability.csv rows
# ---------------------------------------------------------------------------


def _stage_reliability_rows(
    summary_row: Mapping[str, Any], regions: Sequence[str], alpha: float
) -> list[dict[str, Any]]:
    """Reshapes one cohort's `usable_bar` `summarise_cohort` row into long `stage_reliability` rows.

    Five stages, per the protocol's own pipeline picture: `segmentation`
    (`p_usable`, the no-gate rate), `conformal` (one `realised_risk_<R>_all` row
    per region present, plus one row naming the target `alpha` itself so a reader
    can compare the two without cross-referencing another file), `gate`
    (`p_accepted`, `p_usable_given_accepted` -- what the gate buys), `end_to_end`
    (`p_accepted_and_usable`), and `input_qc` (`not_applicable_curated_cohort`,
    NaN -- see the protocol's "Input QC is not applicable to these cohorts").

    Args:
        summary_row: One row from `neurovision.analysis.error_budget.summarise_cohort`,
            at `usable_bar` -- must have `cohort`, `n`, `p_usable`, `p_accepted`,
            `p_usable_given_accepted`, `p_accepted_and_usable` (each optionally with
            `<key>_lo`/`<key>_hi` siblings), and zero or more `realised_risk_<R>_all`
            keys.
        regions: Region names whose `realised_risk_<R>_all` becomes a `conformal`
            stage row, in this order, when present in `summary_row`.
        alpha: The conformal risk level the gate's `conformal_band` signal was
            fitted at -- reported as its own row so it sits beside the realised
            risk it should be compared against.

    Returns:
        One row per `(stage, quantity)`: `cohort`, `stage`, `quantity`, `value`,
        `lo`, `hi`, `n`. `lo`/`hi` are NaN wherever `summary_row` has no matching
        `<quantity>_lo`/`<quantity>_hi` keys (e.g. the `alpha` and `input_qc` rows,
        which carry no bootstrap interval at all).
    """
    cohort = summary_row["cohort"]
    n = summary_row["n"]
    nan = float("nan")

    def _row(stage: str, quantity: str, value: Any) -> dict[str, Any]:
        return {
            "cohort": cohort,
            "stage": stage,
            "quantity": quantity,
            "value": value,
            "lo": summary_row.get(f"{quantity}_lo", nan),
            "hi": summary_row.get(f"{quantity}_hi", nan),
            "n": n,
        }

    rows: list[dict[str, Any]] = [_row("segmentation", "p_usable", summary_row["p_usable"])]

    for region in regions:
        key = f"realised_risk_{region}_all"
        if key in summary_row:
            rows.append(_row("conformal", key, summary_row[key]))
    rows.append(
        {
            "cohort": cohort,
            "stage": "conformal",
            "quantity": "alpha",
            "value": alpha,
            "lo": nan,
            "hi": nan,
            "n": n,
        }
    )

    rows.append(_row("gate", "p_accepted", summary_row["p_accepted"]))
    rows.append(_row("gate", "p_usable_given_accepted", summary_row["p_usable_given_accepted"]))

    rows.append(_row("end_to_end", "p_accepted_and_usable", summary_row["p_accepted_and_usable"]))

    rows.append(
        {
            "cohort": cohort,
            "stage": "input_qc",
            "quantity": "not_applicable_curated_cohort",
            "value": nan,
            "lo": nan,
            "hi": nan,
            "n": n,
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(
    summary_table: pd.DataFrame,
    usable_bar: float,
    alpha: float,
    regions: Sequence[str],
    out_dir: Path,
) -> None:
    """Prints (not logs -- see `scripts/validate_qc.py`'s identical convention) a compact summary.

    Restricted to `summary_table`'s rows at `bar == usable_bar` -- the other
    swept bars are in `summary.csv`, not repeated on the console.
    """
    lines = [
        "=" * 78,
        f"Phase G error budget summary -- out_dir={out_dir}, usable_bar={usable_bar}",
        "=" * 78,
    ]
    at_bar = summary_table[summary_table["bar"] == usable_bar]
    for _, row in at_bar.iterrows():

        def _ci(prefix: str, row: pd.Series = row) -> str:
            return f"{row[prefix]:.4f} [{row[f'{prefix}_lo']:.4f}, {row[f'{prefix}_hi']:.4f}]"

        lines.append(f"  {row['cohort']:>10s} | n={int(row['n']):4d}")
        lines.append(f"    p_accepted={_ci('p_accepted')}  p_usable={_ci('p_usable')}")
        lines.append(
            f"    p_usable_given_accepted={_ci('p_usable_given_accepted')}  "
            f"p_accepted_and_usable={_ci('p_accepted_and_usable')}"
        )
        lines.append(
            f"    cells: correct_accept={int(row['n_correct_accept'])} "
            f"silent_failure={int(row['n_silent_failure'])} "
            f"over_refusal={int(row['n_over_refusal'])} "
            f"correct_refusal={int(row['n_correct_refusal'])}"
        )
        for region in regions:
            key = f"realised_risk_{region}_all"
            if key in row.index:
                lines.append(
                    f"    realised_risk_{region}: all={_ci(key)}  "
                    f"accepted={_ci(f'realised_risk_{region}_accepted')}  alpha={alpha}"
                )
    lines.append("-" * 78)
    # print only, not logger.info as well -- matches scripts/validate_qc.py's /
    # scripts/calibrate_gatekeeper.py's summary convention.
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_error_budget(cfg: DictConfig) -> dict[str, Path]:
    """Runs Phase G end to end: per-cohort signals -> gate decisions -> the error-budget tables.

    See `docs/research/error_budget_protocol.md` for the fixed definitions this
    function implements, and `neurovision.analysis.error_budget`'s own module
    docstring for the pure-arithmetic half this script assembles inputs for.

    Args:
        cfg: The full composed Hydra config. `cfg.model` must already be composed
            as the `segqc` model group (`model=segqc` on the command line) -- see
            this module's docstring. Reads `cfg.analysis.error_budget` (this
            driver's own config block), `cfg.clinical.gatekeeper.{regions,
            conformal_dir,conformal_alpha,caution_quantile,thresholds}`, and
            `cfg.analysis.qc_validate.checkpoint`.

    Returns:
        A dict mapping a short name to the `Path` each output file was written
        to: one `"per_case_<cohort>"` entry per cohort, plus `"summary_csv"`,
        `"taxonomy_csv"`, `"coverage_curve_csv"`, `"stage_reliability_csv"`,
        `"error_budget_config_yaml"`.

    Raises:
        ValueError: `cfg.model` is not composed as the `segqc` group; a
            cohort's signal tables share no `case_id` at all; or
            `cfg.clinical.gatekeeper.thresholds` is unset (nothing frozen to
            judge cases against -- see `load_thresholds`).
        FileNotFoundError: A cohort's `eval_dir/logits`, its `conformal_curves`
            file, the deployed gate's conformal `fit.json`, or the QC
            checkpoint are missing.
    """
    _require_segqc_model(cfg)

    eb_cfg = cfg.analysis.error_budget
    gk_cfg = cfg.clinical.gatekeeper

    checkpoint = Path(str(cfg.analysis.qc_validate.checkpoint))
    regions = [str(r) for r in gk_cfg.regions]
    usable_regions = [str(r) for r in eb_cfg.usable_regions]
    usable_bar = float(eb_cfg.usable_bar)
    max_cases = eb_cfg.max_cases
    conformal_alpha = float(gk_cfg.conformal_alpha)

    # The deployed gate's frozen thresholds -- loaded once, judged against for
    # every case in every cohort. Never recalibrated in this script (see the
    # module docstring's "Nothing here is refitted" section).
    thresholds = load_thresholds(cfg)
    if thresholds is None:
        raise ValueError(
            "error_budget: cfg.clinical.gatekeeper.thresholds is None -- the deployed gate "
            "has no frozen thresholds to judge cases against. Run "
            "`python scripts/calibrate_gatekeeper.py model=segqc` first, then point "
            "clinical.gatekeeper.thresholds at the written thresholds.json."
        )

    conformal_dir = Path(str(gk_cfg.conformal_dir))
    fit_path = conformal_dir / "fit.json"
    if not fit_path.is_file():
        raise FileNotFoundError(f"error_budget: no fit.json at {fit_path}.")
    fit_payload = json.loads(fit_path.read_text())
    # One fitted threshold per region, shared by every cohort -- the val fit is
    # frozen, not re-derived per cohort.
    fitted_thresholds = gatekeeper_calibration.resolve_fitted_thresholds(
        fit_payload, regions, conformal_alpha
    )

    cohorts = resolve_cohorts(cfg)

    generator = np.random.default_rng(int(eb_cfg.seed))
    n_boot = int(eb_cfg.n_boot)
    ci = float(eb_cfg.ci)
    bar_sweep = _ensure_bar_included(list(eb_cfg.bar_sweep), usable_bar)

    out_dir = ensure_dir(str(eb_cfg.out_dir))
    output_paths: dict[str, Path] = {}

    cohort_signals: dict[str, pd.DataFrame] = {}
    cohort_usable: dict[str, pd.Series] = {}
    cohort_dice_tc: dict[str, pd.Series] = {}
    per_case_at_usable_bar: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    for cohort in cohorts:
        case_ids = _case_ids(cohort.eval_dir, max_cases, cohort.name)

        # --- Step 2: the three signal tables, inner-joined -----------------
        dice_table = gatekeeper_calibration.qc_predicted_dice_table(
            cfg, checkpoint, cohort.eval_dir, cohort.prep_dir, regions, case_ids
        )
        curves_by_region = load_curves_npz(cohort.conformal_curves, regions)
        band_widths = gatekeeper_calibration.case_conformal_band_widths(
            curves_by_region, fitted_thresholds
        )
        conformal_table = _reshape_per_region_case_values(band_widths, "conformal_band")
        ood_table = gatekeeper_calibration.ood_score_table(cohort.eval_dir, case_ids)

        signals = _inner_join_signals(dice_table, conformal_table, ood_table, cohort.name)
        signals = signals[signals["case_id"].isin(case_ids)].reset_index(drop=True)

        miss_rate_table = _miss_rate_table(curves_by_region, fitted_thresholds, regions)

        # --- Step 3: decisions, through run_gatekeeper unmodified ----------
        decisions = error_budget.decide_cases(cfg, signals, thresholds, regions=regions)

        # --- Step 4: ground-truth Dice, this cohort's scored cases only ----
        metrics = load_per_case(cohort.eval_dir / "per_case_metrics.csv")
        metrics = metrics.loc[metrics.index.isin(case_ids), ["dice_ET", "dice_TC", "dice_WT"]]
        metrics = metrics.reset_index()

        # --- Step 5: signals join decisions join metrics join miss rates ---
        per_case = (
            signals.merge(miss_rate_table, on="case_id", how="left")
            .merge(decisions, on="case_id", how="inner")
            .merge(metrics, on="case_id", how="inner")
        )

        usable = error_budget.label_usable(per_case, regions=usable_regions, bar=usable_bar)
        per_case_usable_bar = per_case.copy()
        per_case_usable_bar["usable"] = usable
        per_case_usable_bar["cell"] = error_budget.outcome_cells(
            per_case_usable_bar["accepted"], per_case_usable_bar["usable"]
        )
        # case_id first, everything else in the order it was assembled.
        ordered_columns = ["case_id"] + [c for c in per_case_usable_bar.columns if c != "case_id"]
        per_case_usable_bar = per_case_usable_bar[ordered_columns]

        per_case_path = out_dir / f"per_case_{cohort.name}.csv"
        per_case_usable_bar.to_csv(per_case_path, index=False)
        output_paths[f"per_case_{cohort.name}"] = per_case_path
        logger.info("error_budget: wrote %s (%d case(s)).", per_case_path, len(per_case_usable_bar))

        per_case_at_usable_bar[cohort.name] = per_case_usable_bar
        cohort_signals[cohort.name] = signals
        cohort_usable[cohort.name] = per_case_usable_bar.set_index("case_id")["usable"]
        cohort_dice_tc[cohort.name] = per_case.set_index("case_id")["dice_TC"]

        # --- Step 6: the bar sweep, one summarise_cohort row per bar -------
        for bar in bar_sweep:
            per_case_at_bar = per_case.copy()
            per_case_at_bar["usable"] = error_budget.label_usable(
                per_case_at_bar, regions=usable_regions, bar=bar
            )
            row = error_budget.summarise_cohort(
                per_case_at_bar,
                cohort=cohort.name,
                bar=bar,
                n_boot=n_boot,
                ci=ci,
                generator=generator,
                alpha=conformal_alpha,
            )
            summary_rows.append(row)
            if bar == usable_bar:
                stage_rows.extend(_stage_reliability_rows(row, usable_regions, conformal_alpha))

    # --- Step 6 (cont.): summary.csv ---------------------------------------
    summary_table = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary_table.to_csv(summary_path, index=False)
    output_paths["summary_csv"] = summary_path
    logger.info("error_budget: wrote %s.", summary_path)

    # --- Step 7: taxonomy.csv, coverage_curve.csv ---------------------------
    taxonomy = error_budget.taxonomy_table(per_case_at_usable_bar)
    taxonomy_path = out_dir / "taxonomy.csv"
    taxonomy.to_csv(taxonomy_path, index=False)
    output_paths["taxonomy_csv"] = taxonomy_path
    logger.info("error_budget: wrote %s.", taxonomy_path)

    calibration_table = pd.read_csv(str(eb_cfg.calibration_table))
    # calibrate_thresholds requires 0 < refuse_quantile < 1 -- see
    # coverage_curve's own docstring note on q=0.0. The protocol's grid
    # nominally starts at 0.0 ("the gate refuses only at/below the minimum"),
    # but that function deliberately does not special-case it: this is the
    # one place a configured value is dropped before use, logged so it is
    # never silently missing from coverage_curve.csv.
    raw_refuse_quantiles = [float(q) for q in eb_cfg.refuse_quantiles]
    swept_refuse_quantiles = [q for q in raw_refuse_quantiles if 0.0 < q < 1.0]
    dropped = [q for q in raw_refuse_quantiles if not (0.0 < q < 1.0)]
    if dropped:
        logger.info(
            "error_budget: dropping refuse_quantile(s) %s before coverage_curve -- "
            "calibrate_thresholds requires 0 < refuse_quantile < 1; the coverage curve "
            "is described from the smallest quantile actually run (see "
            "neurovision.analysis.error_budget.coverage_curve's docstring).",
            dropped,
        )

    coverage = error_budget.coverage_curve(
        cfg,
        calibration_table,
        cohort_signals,
        cohort_usable,
        cohort_dice_tc,
        regions=regions,
        refuse_quantiles=swept_refuse_quantiles,
        caution_quantile=float(gk_cfg.caution_quantile),
    )
    coverage_path = out_dir / "coverage_curve.csv"
    coverage.to_csv(coverage_path, index=False)
    output_paths["coverage_curve_csv"] = coverage_path
    logger.info("error_budget: wrote %s.", coverage_path)

    # --- Step 8: stage_reliability.csv --------------------------------------
    stage_table = pd.DataFrame(
        stage_rows, columns=["cohort", "stage", "quantity", "value", "lo", "hi", "n"]
    )
    stage_path = out_dir / "stage_reliability.csv"
    stage_table.to_csv(stage_path, index=False)
    output_paths["stage_reliability_csv"] = stage_path
    logger.info("error_budget: wrote %s.", stage_path)

    # --- Step 9: error_budget_config.yaml -----------------------------------
    config_payload: dict[str, Any] = dict(OmegaConf.to_container(eb_cfg, resolve=True))
    config_payload["gatekeeper"] = {
        "regions": regions,
        "conformal_dir": str(conformal_dir),
        "conformal_alpha": conformal_alpha,
        "caution_quantile": float(gk_cfg.caution_quantile),
        "thresholds_path": str(gk_cfg.thresholds),
    }
    config_path = out_dir / "error_budget_config.yaml"
    write_yaml(config_payload, config_path)
    output_paths["error_budget_config_yaml"] = config_path
    logger.info("error_budget: wrote %s.", config_path)

    _print_summary(summary_table, usable_bar, conformal_alpha, usable_regions, out_dir)

    return output_paths


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the Phase G error budget, per the composed config.

    Example:

        python scripts/error_budget.py model=segqc

    (`model=segqc` is REQUIRED -- see this module's docstring; every other
    knob it needs already lives at `cfg.analysis.error_budget` /
    `cfg.clinical.gatekeeper` -- see `configs/analysis/default.yaml` and
    `configs/clinical/default.yaml`.)

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_error_budget(cfg)


if __name__ == "__main__":
    main()
