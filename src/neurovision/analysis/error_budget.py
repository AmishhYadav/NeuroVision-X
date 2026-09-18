"""Phase G, the end-to-end error budget: pure analysis on top of the frozen gatekeeper.

`docs/research/error_budget_protocol.md` is the fixed pre-registration this module
implements — read it before touching anything here. It answers one question the
project's own principle states but has never measured: the master plan's principle 2
says "five stages at 95% each is 77% end to end"; every stage has been measured in
isolation, nobody has measured what a study gets *after* passing through all of them,
and what the refusal gate buys at the pipeline level over no gate at all.

**This module does no file I/O, runs no model, and imports no torch.** It is pure
`numpy`/`pandas` arithmetic over tables a separate driver script assembles from
artifacts already on disk (saved logits, the QC model's predictions, the conformal
fit, `outputs/gatekeeper/thresholds.json`). That split mirrors
`neurovision.inference.gatekeeper` itself, whose own docstring makes the same
promise for the same reason: a function that reads a file and a function that
does arithmetic fail in different ways, and mixing them makes both harder to test
on synthetic data.

**Every gate decision here goes through `neurovision.inference.gatekeeper.run_gatekeeper`
unmodified**, with an explicit, frozen `Thresholds` — this module never lets
`run_gatekeeper` fall back to loading thresholds from `cfg` (see `decide_cases`), because
the protocol's whole point is that "nothing is refitted here" except at the one place
(`coverage_curve`) the protocol explicitly asks for a refit on the frozen val calibration
table, at quantiles picked in advance.

**Column naming matches `neurovision.analysis.gatekeeper_calibration`'s own convention**,
so the same signal tables that module builds for calibration can be reused here directly:
`case_id`, `predicted_dice_<R>`, `conformal_band_<R>`, and an optional `ood_score`, for
every region `R`.

**The usability bar is a segmentation-quality statement, never a gate input.** `dice_WT`
and `dice_TC` come from ground truth and are read only by `label_usable` — never by
anything that also touches `decide_cases` or a `judge_*` call — so a bug here can never
leak a label into the label-free gate the way CLAUDE.md's calibration-mask trap did.

No CUDA, no torch, CPU only. Randomness enters only through the `np.random.Generator`
callers pass into `bootstrap_rate_ci` / `summarise_cohort` — never a module-level seed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from neurovision.inference.gatekeeper import (
    Decision,
    GateSignals,
    Thresholds,
    calibrate_thresholds,
    run_gatekeeper,
)
from neurovision.inference.input_qc import InputQCReport, Severity

logger = logging.getLogger(__name__)

# A case is "accepted" by the pipeline if the gate lets it through at all, with or
# without a caution banner -- only REFUSE withholds a mask from the reader. This is
# the protocol's own definition ("Gate decision" section), restated here as one
# frozen constant so every function in this module agrees with it structurally
# rather than by convention.
ACCEPTED_DECISIONS: frozenset[Decision] = frozenset(
    {Decision.PROCEED, Decision.PROCEED_WITH_CAUTION}
)

# The G5 taxonomy's four outcome cells, in the protocol's own 2x2 order (accepted x
# usable). Fixed here, not derived from `outcome_cells`' branches, so a table built
# from it (e.g. `taxonomy_table`) can always report all four even when a cohort has
# zero cases in one of them.
CELL_NAMES: tuple[str, ...] = (
    "correct_accept",
    "silent_failure",
    "over_refusal",
    "correct_refusal",
)


def curated_input_qc_report() -> InputQCReport:
    """The pass-by-construction `InputQCReport` for a curated challenge cohort.

    The protocol's "Cohorts and inputs" section is explicit that input QC "is not
    applicable to these cohorts": BraTS test, SSA and PED are already curated,
    skull-stripped, co-registered NIfTI volumes -- exactly the state input QC exists
    to verify a real DICOM upload has reached before segmentation. There is nothing
    for input QC to check on data that is already in that state, so rather than
    fabricate findings or skip the signal (which `judge_input_qc` treats as a REFUSE,
    since a missing report is indistinguishable from a check that never ran), the
    protocol scores it as a pass by construction: `Severity.OK`, no findings. Input
    QC's real-world behaviour is reported separately, as the n=3 TCIA fixture
    anecdote in note 45 -- never blended into this rate.

    Returns:
        `InputQCReport(verdict=Severity.OK, findings=())`.
    """
    return InputQCReport(verdict=Severity.OK, findings=())


def decide_cases(
    cfg: Any,
    signals: pd.DataFrame,
    thresholds: Thresholds,
    *,
    regions: Sequence[str],
    input_qc: InputQCReport | None = None,
) -> pd.DataFrame:
    """Runs `run_gatekeeper` once per row of `signals`, with an explicit frozen `Thresholds`.

    `thresholds` is always passed straight through to `run_gatekeeper`, which then never
    calls `load_thresholds(cfg)` internally -- this is what "nothing is refitted here"
    (the protocol's own words) means at the code level: the only thresholds a case is
    ever judged against are the ones the caller hands in.

    Args:
        cfg: The root config, exposing `cfg.clinical.gatekeeper.enabled_signals` and
            `cfg.clinical.gatekeeper.regions` -- read by `run_gatekeeper` itself, not
            by this function. Must agree with `regions` below, since `run_gatekeeper`
            reads regions from `cfg`, not from this function's `regions` argument.
        signals: One row per case, with columns `case_id`, `predicted_dice_<R>` and
            `conformal_band_<R>` for every `R` in `regions`, and optionally `ood_score`.
        thresholds: The frozen, calibrated `Thresholds` to judge every case against.
        regions: The regions to pull `predicted_dice_<R>` / `conformal_band_<R>` values
            for, e.g. `("WT", "TC")`.
        input_qc: The `InputQCReport` to use for every row, or `None` (the default) to
            use `curated_input_qc_report()` -- the protocol's stand-in for a curated
            cohort with no real input-QC signal of its own.

    Returns:
        A `DataFrame`, one row per input row (order preserved), with columns:
        `case_id`, `decision` (the `Decision` enum's `.value` string), `accepted`
        (bool, `decision` in `ACCEPTED_DECISIONS`), `refusing_signals` (semicolon-
        joined names of every signal whose verdict was REFUSE, `''` if none), and
        `cautioning_signals` (same, for PROCEED_WITH_CAUTION).

    Raises:
        KeyError: A required `predicted_dice_<R>` or `conformal_band_<R>` column
            (for a region in `regions`) is absent from `signals`.
    """
    report = input_qc if input_qc is not None else curated_input_qc_report()

    rows: list[dict[str, Any]] = []
    for _, row in signals.iterrows():
        predicted_dice = {r: row[f"predicted_dice_{r}"] for r in regions}
        conformal_band = {r: row[f"conformal_band_{r}"] for r in regions}
        # A NaN or missing value is passed through as-is -- run_gatekeeper's own
        # judge_* functions decide what a NaN/None signal means (REFUSE, via
        # `_is_bad_number`); pre-filtering it here would silently duplicate that
        # policy in a second place.
        case_signals = GateSignals(
            input_qc=report,
            predicted_dice=predicted_dice,
            conformal_band=conformal_band,
            ood_score=row.get("ood_score"),
        )
        decision = run_gatekeeper(cfg, case_signals, thresholds)
        refusing = ";".join(v.signal for v in decision.verdicts if v.decision is Decision.REFUSE)
        cautioning = ";".join(
            v.signal for v in decision.verdicts if v.decision is Decision.PROCEED_WITH_CAUTION
        )
        rows.append(
            {
                "case_id": row["case_id"],
                "decision": decision.decision.value,
                "accepted": decision.decision in ACCEPTED_DECISIONS,
                "refusing_signals": refusing,
                "cautioning_signals": cautioning,
            }
        )

    result = pd.DataFrame(
        rows, columns=["case_id", "decision", "accepted", "refusing_signals", "cautioning_signals"]
    )
    logger.info("decide_cases: %d/%d case(s) accepted.", int(result["accepted"].sum()), len(result))
    return result


def label_usable(
    metrics: pd.DataFrame,
    *,
    regions: Sequence[str] = ("WT", "TC"),
    bar: float = 0.7,
) -> pd.Series:
    """The per-case usability label: `dice_<R> >= bar` for EVERY region in `regions`.

    This is the protocol's "usable segmentation" bar, read from ground truth -- the
    one place in this module that is allowed to touch a label, and deliberately kept
    out of everything gate-related (`decide_cases`, `run_gatekeeper`) so a label can
    never leak into a decision meant to be label-free.

    Args:
        metrics: A `DataFrame` with (at least) `dice_<R>` columns for every `R` in
            `regions`, one row per case.
        regions: The regions the bar applies to, e.g. `("WT", "TC")` -- ET is excluded
            by the protocol because 2.6% of BraTS 2021 cases have no enhancing tumour
            and the metric is undefined on them.
        bar: The minimum Dice each region must reach, e.g. `0.7`.

    Returns:
        A boolean `Series`, aligned to `metrics.index`: `True` only if every required
        region's Dice is `>= bar`. A NaN Dice in any required region makes the case
        `False` -- an undefined score is not a usable score, and NaN compared with
        `>=` is already `False` on both sides, so this is the safe default, not an
        extra check bolted on.

    Raises:
        ValueError: A required `dice_<R>` column is absent from `metrics`, naming it.
    """
    missing = [f"dice_{r}" for r in regions if f"dice_{r}" not in metrics.columns]
    if missing:
        raise ValueError(f"label_usable: missing column(s) {missing} in `metrics`.")

    usable = pd.Series(True, index=metrics.index)
    for region in regions:
        # `col >= bar` is already False for NaN under IEEE comparison semantics, so
        # NaN falls out of the accumulated AND on its own -- no separate `notna()`
        # branch is needed, but the case is called out explicitly here (and in the
        # docstring) because a silently-False NaN is exactly the shape of trap
        # CLAUDE.md's lessons warn about elsewhere in this codebase.
        usable &= metrics[f"dice_{region}"] >= bar
    return usable


def outcome_cells(accepted: pd.Series, usable: pd.Series) -> pd.Series:
    """The G5 taxonomy cell for every case: one of `CELL_NAMES`, from `accepted` x `usable`.

    Args:
        accepted: Boolean `Series`, e.g. `decide_cases(...)["accepted"]`.
        usable: Boolean `Series`, e.g. `label_usable(...)`, on the SAME index as `accepted`.

    Returns:
        A `Series` of dtype `category` (categories `CELL_NAMES`), aligned to
        `accepted.index`: `"correct_accept"` (accepted & usable), `"silent_failure"`
        (accepted & ~usable -- the number that matters), `"over_refusal"` (~accepted &
        usable), `"correct_refusal"` (~accepted & ~usable).

    Raises:
        ValueError: `accepted.index` and `usable.index` are not exactly equal.
    """
    if not accepted.index.equals(usable.index):
        raise ValueError(
            "outcome_cells: `accepted` and `usable` must share exactly the same index."
        )

    accepted_bool = accepted.astype(bool)
    usable_bool = usable.astype(bool)
    cell = pd.Series(index=accepted.index, dtype=object)
    cell[accepted_bool & usable_bool] = "correct_accept"
    cell[accepted_bool & ~usable_bool] = "silent_failure"
    cell[~accepted_bool & usable_bool] = "over_refusal"
    cell[~accepted_bool & ~usable_bool] = "correct_refusal"
    return cell.astype(pd.CategoricalDtype(categories=CELL_NAMES))


def bootstrap_rate_ci(
    values: np.ndarray | pd.Series,
    *,
    n_boot: int,
    ci: float,
    generator: np.random.Generator,
) -> tuple[float, float, float]:
    """A percentile, case-resampled bootstrap CI for the mean of a 0/1 or float array.

    Args:
        values: A 1-D array-like of numbers (e.g. 0/1 indicators, or continuous
            values like a per-case miss rate). NaNs are dropped before resampling.
        n_boot: Number of bootstrap resamples.
        ci: The central confidence level, e.g. `0.95` for a 95% interval (2.5th /
            97.5th percentile).
        generator: The seeded `np.random.Generator` to draw resample indices from --
            this module never seeds its own randomness (see the module docstring).

    Returns:
        `(point, lo, hi)`: the sample mean, and the `ci`-level percentile bootstrap
        interval around it. `(nan, nan, nan)` if `values` has zero non-NaN entries.
        `(v, v, v)` if it has exactly one -- a single point has no resampling
        variance to estimate, so this is reported directly rather than asking the
        RNG to resample a length-1 array `n_boot` times for a CI that would be a
        point mass at `v` anyway.

    Raises:
        Nothing -- degenerate inputs are handled explicitly, per the module's
        edge-case discipline, rather than left to fail inside `np.percentile`.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.shape[0]
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(arr.mean())
    if n == 1:
        return (point, point, point)

    resample_idx = generator.integers(0, n, size=(n_boot, n))
    boot_means = arr[resample_idx].mean(axis=1)
    tail = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100.0 * tail))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - tail)))
    return (point, lo, hi)


def _nanmean_safe(values: np.ndarray) -> float:
    """`np.nanmean`, but `nan` (not a `RuntimeWarning`) for an empty or all-NaN array."""
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def summarise_cohort(
    per_case: pd.DataFrame,
    *,
    cohort: str,
    bar: float,
    n_boot: int,
    ci: float,
    generator: np.random.Generator,
    alpha: float | None = None,
) -> dict[str, Any]:
    """One cohort's G2/G5 summary row: gate rates, end-to-end success, and the taxonomy counts.

    Args:
        per_case: One row per case, columns `accepted` (bool), `usable` (bool),
            `dice_WT`, `dice_TC`, `dice_ET`, and optionally one `miss_rate_<R>`
            column per region with a fitted conformal threshold (the per-case
            realised miss rate at that threshold; may contain NaN).
        cohort: The cohort's name, carried into the output row for a later concat.
        bar: The usability bar used to build `per_case["usable"]` -- carried through
            only for bookkeeping (this function does not itself apply a bar).
        n_boot: Bootstrap resamples, passed to every `bootstrap_rate_ci` call.
        ci: Confidence level, passed to every `bootstrap_rate_ci` call.
        generator: The seeded `np.random.Generator` shared across every bootstrap in
            this row.
        alpha: The conformal risk level the `miss_rate_<R>` columns were fitted at,
            e.g. `0.10`. `frac_case_risk_le_alpha_<R>_*` keys are only emitted when
            this is not `None` -- see the protocol's "Guarantee met" definition,
            which is explicit that the per-case fraction is descriptive, not the
            guarantee itself.

    Returns:
        A flat `dict`: `cohort`, `bar`, `n`, `p_accepted(+_lo/_hi)`,
        `p_usable(+_lo/_hi)` (the no-gate rate), `p_usable_given_accepted(+_lo/_hi)`
        (NaN with no exception if nothing was accepted -- `bootstrap_rate_ci` already
        returns `(nan, nan, nan)` for an empty array), `p_accepted_and_usable(+_lo/_hi)`
        (end-to-end success), `n_correct_accept`, `n_silent_failure`, `n_over_refusal`,
        `n_correct_refusal`, `dice_<R>_all_mean` / `dice_<R>_accepted_mean` for
        `R in ("WT", "TC", "ET")` (via `nanmean`), and, for every `miss_rate_<R>`
        column present in `per_case`: `realised_risk_<R>_all(+_lo/_hi)`,
        `realised_risk_<R>_accepted(+_lo/_hi)`, and (only if `alpha is not None`)
        `frac_case_risk_le_alpha_<R>_all`, `frac_case_risk_le_alpha_<R>_accepted`.
    """
    n = len(per_case)
    accepted = per_case["accepted"].to_numpy(dtype=bool)
    usable = per_case["usable"].to_numpy(dtype=bool)

    result: dict[str, Any] = {"cohort": cohort, "bar": bar, "n": n}

    def _add(prefix: str, values: np.ndarray) -> None:
        point, lo, hi = bootstrap_rate_ci(values, n_boot=n_boot, ci=ci, generator=generator)
        result[prefix] = point
        result[f"{prefix}_lo"] = lo
        result[f"{prefix}_hi"] = hi

    _add("p_accepted", accepted.astype(float))
    _add("p_usable", usable.astype(float))
    _add("p_accepted_and_usable", (accepted & usable).astype(float))
    # Restricting to accepted rows BEFORE bootstrapping is what makes this "what the
    # gate buys" rather than a second copy of p_accepted_and_usable: an empty
    # restriction (nothing accepted) is handled for free by bootstrap_rate_ci's own
    # n==0 branch, so no separate guard is needed here.
    _add("p_usable_given_accepted", usable[accepted].astype(float))

    cells = outcome_cells(per_case["accepted"], per_case["usable"])
    counts = cells.value_counts()
    for name in CELL_NAMES:
        result[f"n_{name}"] = int(counts.get(name, 0))

    for region in ("WT", "TC", "ET"):
        values = per_case[f"dice_{region}"].to_numpy(dtype=float)
        result[f"dice_{region}_all_mean"] = _nanmean_safe(values)
        result[f"dice_{region}_accepted_mean"] = _nanmean_safe(values[accepted])

    miss_rate_columns = [c for c in per_case.columns if c.startswith("miss_rate_")]
    for column in miss_rate_columns:
        region = column[len("miss_rate_") :]
        values = per_case[column].to_numpy(dtype=float)
        _add(f"realised_risk_{region}_all", values)
        _add(f"realised_risk_{region}_accepted", values[accepted])

        if alpha is not None:
            valid_all = per_case[column].dropna()
            result[f"frac_case_risk_le_alpha_{region}_all"] = (
                float((valid_all <= alpha).mean()) if len(valid_all) else float("nan")
            )
            valid_accepted = per_case.loc[per_case["accepted"], column].dropna()
            result[f"frac_case_risk_le_alpha_{region}_accepted"] = (
                float((valid_accepted <= alpha).mean()) if len(valid_accepted) else float("nan")
            )

    return result


def coverage_curve(
    cfg: Any,
    calibration_table: pd.DataFrame,
    cohort_signals: Mapping[str, pd.DataFrame],
    cohort_usable: Mapping[str, pd.Series],
    cohort_dice_tc: Mapping[str, pd.Series],
    *,
    regions: Sequence[str],
    refuse_quantiles: Sequence[float],
    caution_quantile: float,
) -> pd.DataFrame:
    """The G3 pipeline-level risk-coverage curve: re-fit quantile thresholds, apply to every cohort.

    For each `q` in `refuse_quantiles`, `calibrate_thresholds` re-fits on
    `calibration_table` (the frozen val calibration table -- see the protocol's G3
    definition) at `refuse_quantile=q` and a `caution_quantile` computed as:
    `caution_quantile` itself when `q < caution_quantile` (the normal, deployed case),
    otherwise `min(q + 0.01, 1.0)`. **This column has no effect on anything
    `coverage_curve` reports.** `accepted` (and therefore `coverage`,
    `p_usable_given_accepted`, `mean_dice_tc_accepted` and `n_silent_failure`) is
    PROCEED-or-PROCEED_WITH_CAUTION -- a cautioned case is still accepted, so where
    exactly the CAUTION cut sits does not move any number this function returns. The
    `+ 0.01` nudge exists purely so `calibrate_thresholds`' own structural requirement
    (`refuse_quantile` strictly less than `caution_quantile`) is satisfied once `q`
    reaches or passes the deployed caution quantile -- it is bookkeeping to let the
    fit succeed, not a second calibrated operating point, and the actual value used is
    reported back in `caution_quantile_used` precisely so a reader can see it moved for
    this reason alone.

    Note on `q=0.0`: `calibrate_thresholds` requires `0.0 < refuse_quantile < 1.0` and
    raises `ValueError` on exactly `0.0` (a REFUSE-below-the-minimum quantile is not a
    real cut point under `pandas`' quantile interpolation at `q=0`). The protocol's
    grid nominally starts at `0.0` ("the gate refuses only at/below the minimum"), but
    this function does not special-case it: callers must supply `0.01` as the lowest
    quantile actually run, and the resulting curve is described from there.

    Args:
        cfg: The root config, passed straight through to `decide_cases` /
            `run_gatekeeper`; must expose `cfg.clinical.gatekeeper.enabled_signals`
            and `.regions` (the latter must agree with `regions` below).
        calibration_table: The frozen val calibration table `calibrate_thresholds`
            fits from -- columns `predicted_dice_<R>`, `conformal_band_<R>` per
            region, and `ood_score`.
        cohort_signals: `{cohort_name: signals_df}`, each `signals_df` shaped like
            `decide_cases`' own `signals` argument (`case_id`, per-region signal
            columns, optional `ood_score`).
        cohort_usable: `{cohort_name: usable_series}`, each a boolean `Series`
            (e.g. from `label_usable`) indexed by `case_id`.
        cohort_dice_tc: `{cohort_name: dice_tc_series}`, each a `Series` of `dice_TC`
            indexed by `case_id`.
        regions: The regions to fit and judge, e.g. `("WT", "TC")`.
        refuse_quantiles: The REFUSE-quantile grid to sweep, e.g.
            `(0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)`. Every value in this
            grid is supported, including ones at or beyond `caution_quantile` -- see
            above.
        caution_quantile: The deployed CAUTION quantile (e.g. `0.10`), used as-is
            while `q < caution_quantile`, and only nudged (to `min(q + 0.01, 1.0)`,
            reported in `caution_quantile_used`) once `q` would otherwise collide
            with or exceed it. Never affects any reported number -- see above.

    Returns:
        One row per `(cohort, q)`: `cohort`, `refuse_quantile`, `caution_quantile_used`,
        `n`, `n_accepted`, `coverage` (`n_accepted / n`), `p_usable_given_accepted`
        (NaN if `n_accepted == 0`), `mean_dice_tc_accepted` (`nanmean`, NaN if none
        accepted), `n_silent_failure`.

    Raises:
        ValueError: A cohort's `signals` `case_id` has no matching entry in that
            cohort's `cohort_usable` Series -- an unlabelled case cannot be scored
            for usability, so it is refused to run rather than silently dropped.
    """
    rows: list[dict[str, Any]] = []
    for q in refuse_quantiles:
        # caution_quantile only ever gates the PROCEED/PROCEED_WITH_CAUTION split,
        # never accepted-vs-refused, so it is free to move purely to keep the fit
        # legal once q reaches the deployed caution baseline -- see the docstring.
        effective_caution = caution_quantile if q < caution_quantile else min(q + 0.01, 1.0)
        thresholds = calibrate_thresholds(
            calibration_table,
            regions=regions,
            caution_quantile=effective_caution,
            refuse_quantile=q,
        )
        for cohort, signals in cohort_signals.items():
            usable = cohort_usable[cohort]
            dice_tc = cohort_dice_tc[cohort]
            case_ids = list(signals["case_id"])
            missing = [cid for cid in case_ids if cid not in usable.index]
            if missing:
                raise ValueError(
                    f"coverage_curve: cohort {cohort!r} has case_id(s) with no usable "
                    f"label: {missing[:5]}{'...' if len(missing) > 5 else ''}."
                )
            usable_aligned = usable.reindex(case_ids).to_numpy(dtype=bool)
            dice_aligned = dice_tc.reindex(case_ids).to_numpy(dtype=float)

            decided = decide_cases(cfg, signals, thresholds, regions=regions)
            accepted_mask = decided["accepted"].to_numpy(dtype=bool)

            n = len(case_ids)
            n_accepted = int(accepted_mask.sum())
            coverage = float(n_accepted) / n if n else float("nan")
            if n_accepted == 0:
                p_usable_given_accepted = float("nan")
                mean_dice_tc_accepted = float("nan")
            else:
                p_usable_given_accepted = float(usable_aligned[accepted_mask].mean())
                mean_dice_tc_accepted = _nanmean_safe(dice_aligned[accepted_mask])

            cells = outcome_cells(
                pd.Series(accepted_mask, index=range(n)),
                pd.Series(usable_aligned, index=range(n)),
            )
            n_silent_failure = int((cells == "silent_failure").sum())

            rows.append(
                {
                    "cohort": cohort,
                    "refuse_quantile": q,
                    "caution_quantile_used": effective_caution,
                    "n": n,
                    "n_accepted": n_accepted,
                    "coverage": coverage,
                    "p_usable_given_accepted": p_usable_given_accepted,
                    "mean_dice_tc_accepted": mean_dice_tc_accepted,
                    "n_silent_failure": n_silent_failure,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "cohort",
            "refuse_quantile",
            "caution_quantile_used",
            "n",
            "n_accepted",
            "coverage",
            "p_usable_given_accepted",
            "mean_dice_tc_accepted",
            "n_silent_failure",
        ],
    )


def taxonomy_table(per_case_by_cohort: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The G5 taxonomy as a long table: every `(cohort, cell)` broken down by refusing signal.

    Args:
        per_case_by_cohort: `{cohort_name: per_case_df}`, each `per_case_df` holding
            (at least) a `cell` column (values from `CELL_NAMES`, e.g. from
            `outcome_cells`) and a `refusing_signals` column (e.g. from
            `decide_cases`, `''` for a case that was not refused).

    Returns:
        A long `DataFrame`: `cohort`, `cell`, `refusing_signals` (`''` for the
        accepted cells `correct_accept` / `silent_failure`, since nothing refused
        them), `n`, `frac_of_cohort` (`n / len(per_case_df)`). Every `(cohort, cell)`
        combination appears at least once, even at `n=0` (with `refusing_signals=''`)
        if that cell had no cases in a cohort -- so a reader scanning the table never
        has to infer a missing cell means zero, and `frac_of_cohort` always sums to
        `1.0` per cohort.

    Raises:
        ValueError: A cohort's `DataFrame` is missing `cell` or `refusing_signals`.
    """
    rows: list[dict[str, Any]] = []
    for cohort, df in per_case_by_cohort.items():
        if "cell" not in df.columns or "refusing_signals" not in df.columns:
            raise ValueError(
                f"taxonomy_table: cohort {cohort!r} is missing 'cell' or "
                "'refusing_signals' -- both are required."
            )
        n_total = len(df)
        # Cast to plain str before grouping: `cell` may be the pandas `category`
        # dtype `outcome_cells` returns, and grouping directly on a categorical
        # column (with dropna=False) would fabricate a zero-row group for every
        # unused category x refusing_signals combination, not just every unused
        # cell -- far noisier than the "one filler row per missing cell" this
        # function promises.
        cell_col = df["cell"].astype(str)
        signal_col = df["refusing_signals"].astype(str)
        grouped = (
            pd.DataFrame({"cell": cell_col, "refusing_signals": signal_col})
            .groupby(["cell", "refusing_signals"])
            .size()
        )

        seen_cells: set[str] = set()
        for (cell, refusing_signals), count in grouped.items():
            rows.append(
                {
                    "cohort": cohort,
                    "cell": cell,
                    "refusing_signals": refusing_signals,
                    "n": int(count),
                    "frac_of_cohort": float(count) / n_total if n_total else float("nan"),
                }
            )
            seen_cells.add(cell)

        for cell in CELL_NAMES:
            if cell in seen_cells:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "cell": cell,
                    "refusing_signals": "",
                    "n": 0,
                    "frac_of_cohort": 0.0 if n_total else float("nan"),
                }
            )

    return pd.DataFrame(rows, columns=["cohort", "cell", "refusing_signals", "n", "frac_of_cohort"])
