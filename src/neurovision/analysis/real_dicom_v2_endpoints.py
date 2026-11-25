"""Real-DICOM v2 endpoints (MasterPlan W3.4, protocol Amendment 1): the statistics half.

Pure functions over one per-case DataFrame. No file I/O, no pipeline, no Hydra: every
threshold, margin, bootstrap size and seed is an ARGUMENT (the caller reads them from
config), because the protocol fixes them before any carried-reference Dice exists and a
default here would be a second, silent place to change them.

Per-case input schema (one row per study; NaN where not applicable)::

    case_id                      str
    status                       "done" | "refused" | "failed"   (deployed run)
    refusing_stage               str | None   e.g. "input_qc (pre-preprocessing)", "gatekeeper"
    thick_slice_refusal          bool  refused at input QC for spacing and/or anisotropy
    accepted                     bool  accepted by the deployed gate
    gate_decision                str   PROCEED / CAUTION / REFUSE
    atlas_dice_{ET,TC,WT}        Dice vs BraTS's own atlas-space label (as P1.2)
    carried_dice_{ET,TC,WT}      Dice vs the label carried into OUR atlas space
    carry_qc_dice                brain-mask Dice, carried BraTS vs ours (registration screen)
    research_dice_{ET,TC,WT}     Dice of the research path (no clinical front end)
    -- only for thick-slice refusals, from the second run with those refusals disabled --
    override_status, override_accepted, override_gate_decision,
    override_atlas_dice_{ET,TC,WT}, override_carried_dice_{ET,TC,WT}, override_carry_qc_dice

Logic from `scripts/validate_real_dicom.py` (`label_usable`, `completion_counts`,
`usable_rate_stats`, `front_end_cost`) is RE-EXPRESSED here, not imported (library code
does not import from `scripts/`), so v2's atlas-space column is computed the same way as
P1.2's. The statistics helpers `paired_bootstrap_ci`, `wilcoxon_signed_rank` and
`bootstrap_rate_ci` are reused from `neurovision.analysis`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neurovision.analysis.error_budget import bootstrap_rate_ci
from neurovision.analysis.statistics import paired_bootstrap_ci, wilcoxon_signed_rank

logger = logging.getLogger(__name__)

_REGIONS = ("ET", "TC", "WT")
_CI = 0.95  # fixed by the protocol: every interval here is 95%

# The stage string of a refusal that happens AFTER segmentation, so a mask exists.
# Compared with `==`, never `in`: other stage names contain similar substrings.
_GATEKEEPER_STAGE = "gatekeeper"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "case_id",
    "status",
    "refusing_stage",
    "thick_slice_refusal",
    "accepted",
    "gate_decision",
    *(f"atlas_dice_{r}" for r in _REGIONS),
    *(f"carried_dice_{r}" for r in _REGIONS),
    "carry_qc_dice",
    *(f"research_dice_{r}" for r in _REGIONS),
    "override_status",
    "override_accepted",
    "override_gate_decision",
    *(f"override_atlas_dice_{r}" for r in _REGIONS),
    *(f"override_carried_dice_{r}" for r in _REGIONS),
    "override_carry_qc_dice",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame) -> None:
    """Raises ValueError listing every schema column missing from `df`."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"real_dicom_v2_endpoints: per-case frame is missing columns: {missing}")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """A column as float (strings / None become NaN)."""
    return pd.to_numeric(df[col], errors="coerce")


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    """A column as strict bool; NaN / None count as False."""
    return df[col].fillna(False).astype(bool)


def usable(dice_wt: pd.Series, dice_tc: pd.Series, threshold: float) -> pd.Series:
    """Usable-segmentation label: `WT >= threshold AND TC >= threshold`.

    Args:
        dice_wt: Whole-tumour Dice per case.
        dice_tc: Tumour-core Dice per case, same index.
        threshold: The Dice bar (e.g. 0.7), from config.

    Returns:
        Boolean Series. NaN compares False under `>=`, so an unscored case is
        not usable (it is never silently treated as good).
    """
    wt = pd.to_numeric(dice_wt, errors="coerce")
    tc = pd.to_numeric(dice_tc, errors="coerce")
    return (wt >= threshold) & (tc >= threshold)


def _has_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose DEPLOYED run produced a mask: done, or refused by the gatekeeper.

    An input-QC refusal happens before segmentation, so there is nothing to score.
    """
    status = df["status"].astype(str)
    stage = df["refusing_stage"].astype(str)
    return (status == "done") | ((status == "refused") & (stage == _GATEKEEPER_STAGE))


def _clopper_pearson(k: int, n: int, conf: float) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided binomial interval.

    Chosen over a normal approximation because n is small (~36) and the fraction may
    sit near 0 or 1, where the normal interval is wrong and can leave [0, 1]. The
    exact interval inverts the binomial test through Beta quantiles and always
    guarantees at least `conf` coverage.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def _paired_median_summary(
    carried: np.ndarray, research: np.ndarray, *, n_boot: int, generator: np.random.Generator
) -> dict[str, Any]:
    """Median / mean of (carried - research), bootstrap CI of the median, Wilcoxon p.

    The MEDIAN is the registered statistic: a few cases where a front end fails
    badly (Dice near 0) would drag a mean far from what a typical study sees.
    The percentile bootstrap is used because it makes no distributional
    assumption about the differences and is the method the other project
    endpoints use. The Wilcoxon p is descriptive only (no verdict reads it).
    """
    ok = ~(np.isnan(carried) | np.isnan(research))
    a, b = carried[ok], research[ok]
    n = int(a.size)
    out: dict[str, Any] = {
        "n": n,
        "median_diff": float("nan"),
        "median_diff_lo": float("nan"),
        "median_diff_hi": float("nan"),
        "mean_diff": float("nan"),
        "wilcoxon_pvalue": float("nan"),
    }
    if n < 2:
        logger.warning("paired summary: only %d paired case(s); no CI or test reported.", n)
        if n == 1:
            out["median_diff"] = out["mean_diff"] = float(a[0] - b[0])
        return out
    boot = paired_bootstrap_ci(
        a, b, n_boot=n_boot, ci=_CI, generator=generator, method="percentile", statistic="median"
    )
    out["median_diff"] = boot.point
    out["median_diff_lo"] = boot.lo
    out["median_diff_hi"] = boot.hi
    out["mean_diff"] = float(np.mean(a - b))
    out["wilcoxon_pvalue"] = wilcoxon_signed_rank(a, b).pvalue
    return out


# ---------------------------------------------------------------------------
# E1: front-end cost after removing registration disagreement (primary)
# ---------------------------------------------------------------------------


def e1_front_end_cost(
    df: pd.DataFrame,
    *,
    margin: float,
    carry_qc_threshold: float,
    n_boot: int,
    seed: int,
    regions: Sequence[str] = _REGIONS,
) -> dict[str, Any]:
    """E1: paired (carried-reference - research-path) Dice, deployed outcome has a mask.

    Population: status `done`, or `refused` at the `gatekeeper` stage (a mask exists);
    carry QC passed (`carry_qc_dice >= carry_qc_threshold`); carried WT not NaN.
    Input-QC refusals are out (no mask). Studies failing carry QC are excluded and listed.

    Args:
        df: Per-case frame (module docstring).
        margin: Equivalence margin on the WT median difference (e.g. 0.05).
        carry_qc_threshold: Carry-QC brain-mask Dice bar (e.g. 0.90).
        n_boot: Bootstrap replicates.
        seed: Seed for `numpy.random.default_rng` (one generator, used region by region).
        regions: Regions to report; the verdict always reads WT.

    Returns:
        JSON-friendly dict: `per_region` (list of dicts with n, median/mean diff,
        median CI, Wilcoxon p), `verdict` (SMALL if CI lower > -margin; LARGE if CI
        upper < -margin; else INCONCLUSIVE), `n_population`, `n_excluded_carry_qc`,
        `excluded_carry_qc_case_ids`.
    """
    _require_columns(df)
    has_mask = _has_mask(df)
    qc_ok = _num(df, "carry_qc_dice") >= carry_qc_threshold  # NaN -> False
    excluded = df[has_mask & ~qc_ok]
    pop = df[has_mask & qc_ok & _num(df, "carried_dice_WT").notna()]

    generator = np.random.default_rng(seed)
    per_region = []
    for region in regions:
        summary = _paired_median_summary(
            _num(pop, f"carried_dice_{region}").to_numpy(),
            _num(pop, f"research_dice_{region}").to_numpy(),
            n_boot=n_boot,
            generator=generator,
        )
        per_region.append({"region": region, **summary})

    wt = next((r for r in per_region if r["region"] == "WT"), None)
    verdict = "INCONCLUSIVE"
    if wt is not None and not np.isnan(wt["median_diff_lo"]):
        if wt["median_diff_lo"] > -margin:
            verdict = "SMALL"
        elif wt["median_diff_hi"] < -margin:
            verdict = "LARGE"

    return {
        "endpoint": "E1",
        "margin": margin,
        "carry_qc_threshold": carry_qc_threshold,
        "n_population": int(len(pop)),
        "n_excluded_carry_qc": int(len(excluded)),
        "excluded_carry_qc_case_ids": [str(c) for c in excluded["case_id"]],
        "per_region": per_region,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# E2: was the thick-slice refusal right? (primary)
# ---------------------------------------------------------------------------


def _fraction_block(k: int, n: int, conf: float) -> dict[str, Any]:
    lo, hi = _clopper_pearson(k, n, conf)
    return {
        "k": int(k),
        "n": int(n),
        "fraction": (k / n) if n else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
    }


def e2_refusal_check(
    df: pd.DataFrame,
    *,
    usable_threshold: float,
    carry_qc_threshold: float,
    conf: float = 0.95,
) -> dict[str, Any]:
    """E2: of the thick-slice refusals, how many were actually usable when overridden?

    Population: `thick_slice_refusal` rows. Usable = override-run WT and TC Dice vs the
    CARRIED reference >= `usable_threshold`. Rows with `override_carry_qc_dice` below
    `carry_qc_threshold` (or NaN carry QC / NaN carried Dice, e.g. the override run
    failed) are reported separately, counted, and excluded from the fraction.

    Args:
        df: Per-case frame.
        usable_threshold: Dice bar for usable (e.g. 0.7).
        carry_qc_threshold: Carry-QC bar (e.g. 0.90).
        conf: Confidence level of the Clopper-Pearson interval.

    Returns:
        Dict with `carried` (k, n, fraction, ci_lo, ci_hi), `n_thick`,
        `n_excluded`, `excluded_case_ids`, `atlas_descriptive` (same fraction vs the
        atlas-space reference, over thick rows whose override run was scored there),
        and `verdict`: REFUSES USABLE STUDIES if the lower bound > 0.5;
        REFUSAL JUSTIFIED if the upper bound < 0.5; else INCONCLUSIVE.
    """
    _require_columns(df)
    thick = df[_flag(df, "thick_slice_refusal")]

    carried_wt = _num(thick, "override_carried_dice_WT")
    carried_tc = _num(thick, "override_carried_dice_TC")
    ok = (
        (_num(thick, "override_carry_qc_dice") >= carry_qc_threshold)
        & carried_wt.notna()
        & carried_tc.notna()
    )
    use = usable(carried_wt, carried_tc, usable_threshold)
    carried_block = _fraction_block(int((use & ok).sum()), int(ok.sum()), conf)

    # Descriptive: the same question against BraTS's own atlas-space label. No carry
    # QC applies there (nothing was carried); an unscored override run is skipped.
    a_wt = _num(thick, "override_atlas_dice_WT")
    a_tc = _num(thick, "override_atlas_dice_TC")
    a_scored = a_wt.notna() & a_tc.notna()
    a_use = usable(a_wt, a_tc, usable_threshold)
    atlas_block = _fraction_block(int((a_use & a_scored).sum()), int(a_scored.sum()), conf)

    # Verdict reads the exact interval against the 0.5 coin-flip line: "most thick-slice
    # studies were usable" (refusing was wrong) vs "most were not" (refusing was right).
    verdict = "INCONCLUSIVE"
    if carried_block["n"] > 0:
        if carried_block["ci_lo"] > 0.5:
            verdict = "REFUSES USABLE STUDIES"
        elif carried_block["ci_hi"] < 0.5:
            verdict = "REFUSAL JUSTIFIED"

    return {
        "endpoint": "E2",
        "usable_threshold": usable_threshold,
        "carry_qc_threshold": carry_qc_threshold,
        "conf": conf,
        "n_thick": int(len(thick)),
        "n_excluded": int((~ok).sum()),
        "excluded_case_ids": [str(c) for c in thick.loc[~ok, "case_id"]],
        "carried": carried_block,
        "atlas_descriptive": atlas_block,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# E3 / E4: end-to-end tables
# ---------------------------------------------------------------------------


def _outcome_frame(
    df: pd.DataFrame, column: str, *, usable_threshold: float, carry_qc_threshold: float
) -> pd.DataFrame:
    """One row per study with the outcome a table column talks about.

    Output columns: `case_id`, `completion` (label), `accepted`, `has_mask`, `usable`,
    `carry_failed` (mask exists but carry QC failed; always False for `atlas`),
    `dice_{R}` (the reference Dice for this column) and `research_{R}`.

    `column`:
      * `atlas`      deployed outcome, atlas-space Dice (as P1.2).
      * `carried`    deployed outcome, carried Dice.
      * `overridden` thick-slice refusals replaced by their overridden run
                     (`override_*`), carried Dice; everything else deployed.
    """
    if column not in ("atlas", "carried", "overridden"):
        raise ValueError(f"unknown column {column!r}")

    out = pd.DataFrame({"case_id": df["case_id"].astype(str).to_numpy()}, index=df.index)
    status = df["status"].astype(str)
    stage = df["refusing_stage"].astype(str)
    completion = pd.Series(
        np.where(
            status == "refused",
            "refused (" + stage + ")",
            status,
        ),
        index=df.index,
    )
    accepted = _flag(df, "accepted")
    has_mask = _has_mask(df)
    prefix = "atlas_dice_" if column == "atlas" else "carried_dice_"
    dice = {r: _num(df, f"{prefix}{r}") for r in _REGIONS}
    carry_qc = _num(df, "carry_qc_dice")

    if column == "overridden":
        swap = _flag(df, "thick_slice_refusal")
        o_status = df["override_status"].astype(str)
        completion = completion.where(
            ~swap,
            np.where(o_status == "refused", "refused (after override)", o_status),
        )
        accepted = accepted.where(~swap, _flag(df, "override_accepted"))
        # An override run has a mask iff it ran to a scored result.
        o_mask = o_status.isin(["done", "refused"]) & _num(df, "override_carried_dice_WT").notna()
        has_mask = has_mask.where(~swap, o_mask)
        for r in _REGIONS:
            dice[r] = dice[r].where(~swap, _num(df, f"override_carried_dice_{r}"))
        carry_qc = carry_qc.where(~swap, _num(df, "override_carry_qc_dice"))

    out["completion"] = completion.to_numpy()
    out["accepted"] = accepted.to_numpy()
    out["has_mask"] = has_mask.to_numpy()
    out["usable"] = usable(dice["WT"], dice["TC"], usable_threshold).to_numpy()
    for r in _REGIONS:
        out[f"dice_{r}"] = dice[r].to_numpy()
        out[f"research_{r}"] = _num(df, f"research_dice_{r}").to_numpy()
    if column == "atlas":
        out["carry_failed"] = False
    else:
        # A study with a mask whose carry QC is below the bar (or missing) has an
        # untrustworthy carried label: excluded from usability fractions, never
        # counted as a silent failure. Studies with no mask are never excluded.
        out["carry_failed"] = (has_mask & ~(carry_qc >= carry_qc_threshold)).to_numpy()
    return out


def _rates(out: pd.DataFrame, *, n_boot: int, generator: np.random.Generator) -> dict[str, Any]:
    """End-to-end rates with bootstrap CIs (P1.2's `usable_rate_stats`, re-expressed).

    Denominator: every study except carry-QC failures. Studies refused before
    segmentation stay in (they are not accepted, so they count against
    P(accepted AND usable) -- nothing is hidden by refusing).
    """
    keep = out[~out["carry_failed"].astype(bool)]
    accepted = keep["accepted"].astype(bool).to_numpy()
    use = keep["usable"].astype(bool).to_numpy()

    p_au = bootstrap_rate_ci(
        (accepted & use).astype(float), n_boot=n_boot, ci=_CI, generator=generator
    )
    p_uga = bootstrap_rate_ci(
        use[accepted].astype(float), n_boot=n_boot, ci=_CI, generator=generator
    )
    p_sf = bootstrap_rate_ci(
        (accepted & ~use).astype(float), n_boot=n_boot, ci=_CI, generator=generator
    )
    return {
        "n": int(len(keep)),
        "n_accepted": int(accepted.sum()),
        "n_carry_qc_excluded": int(out["carry_failed"].sum()),
        "p_accepted_and_usable": p_au,
        "p_usable_given_accepted": p_uga,
        "p_silent_failure": p_sf,
    }


def _front_end_rows(
    out: pd.DataFrame, *, n_boot: int, generator: np.random.Generator
) -> dict[str, dict[str, Any]]:
    """Per-region median (reference Dice - research Dice) over studies with a mask."""
    pop = out[out["has_mask"].astype(bool) & ~out["carry_failed"].astype(bool)]
    result = {}
    for r in _REGIONS:
        result[r] = _paired_median_summary(
            pop[f"dice_{r}"].to_numpy(dtype=float),
            pop[f"research_{r}"].to_numpy(dtype=float),
            n_boot=n_boot,
            generator=generator,
        )
    return result


def e3_table(
    df: pd.DataFrame,
    *,
    usable_threshold: float,
    carry_qc_threshold: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """E3, "Table XV v2": the end-to-end table in three reference columns.

    Columns: `atlas` (deployed outcome, atlas-space Dice, as P1.2), `carried`
    (deployed outcome, carried Dice; carry-QC failures excluded from the usability
    fractions and from front-end cost, count in row `carry_qc_excluded`), `overridden`
    (thick-slice refusals replaced by their overridden run's outcome and acceptance,
    carried Dice). Descriptive; bootstrap CIs are percentile, case-resampled.

    Args:
        df: Per-case frame.
        usable_threshold: Dice bar for usable.
        carry_qc_threshold: Carry-QC bar.
        n_boot: Bootstrap replicates.
        seed: Seed for `numpy.random.default_rng` (a fresh generator per column, so
            adding or reordering columns never changes another column's numbers).

    Returns:
        Tidy long frame with columns `row, column, value, ci_lo, ci_hi, n`. Rows:
        `completion: <label>` (value = fraction of ALL studies, n = total, CI
        bootstrap), `carry_qc_excluded` (value = count, n = total),
        `p_accepted_and_usable`, `p_usable_given_accepted` (n = accepted studies),
        `p_silent_failure` (accepted AND unusable), and
        `front_end_cost_median_{ET,TC,WT}` (value = median, CI bootstrap, n = pairs).
    """
    _require_columns(df)
    records: list[dict[str, Any]] = []

    for column in ("atlas", "carried", "overridden"):
        out = _outcome_frame(
            df, column, usable_threshold=usable_threshold, carry_qc_threshold=carry_qc_threshold
        )
        generator = np.random.default_rng(seed)
        total = len(out)

        # Completion: one row per label, nothing excluded (carry QC does not change
        # whether the pipeline completed).
        for label in sorted(out["completion"].unique()):
            indicator = (out["completion"] == label).to_numpy(dtype=float)
            point, lo, hi = bootstrap_rate_ci(indicator, n_boot=n_boot, ci=_CI, generator=generator)
            records.append(
                {
                    "row": f"completion: {label}",
                    "column": column,
                    "value": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n": total,
                }
            )

        rates = _rates(out, n_boot=n_boot, generator=generator)
        records.append(
            {
                "row": "carry_qc_excluded",
                "column": column,
                "value": float(rates["n_carry_qc_excluded"]),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n": total,
            }
        )
        for name in ("p_accepted_and_usable", "p_usable_given_accepted", "p_silent_failure"):
            point, lo, hi = rates[name]
            n_den = rates["n_accepted"] if name == "p_usable_given_accepted" else rates["n"]
            records.append(
                {
                    "row": name,
                    "column": column,
                    "value": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n": n_den,
                }
            )

        for region, summ in _front_end_rows(out, n_boot=n_boot, generator=generator).items():
            records.append(
                {
                    "row": f"front_end_cost_median_{region}",
                    "column": column,
                    "value": summ["median_diff"],
                    "ci_lo": summ["median_diff_lo"],
                    "ci_hi": summ["median_diff_hi"],
                    "n": summ["n"],
                }
            )

    return pd.DataFrame(records, columns=["row", "column", "value", "ci_lo", "ci_hi", "n"])


def e4_caution_variant(
    df: pd.DataFrame,
    *,
    usable_threshold: float,
    carry_qc_threshold: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """E4: the post-hoc variant where thick-slice refusals become CAUTION.

    Thick-slice refusals are not refused: they are run as CAUTION, i.e. accepted iff
    their overridden run was accepted by the gate (`override_accepted`); every other
    study keeps its deployed outcome. End-to-end numbers on all rows, carried
    reference. (This coincides numerically with E3's `overridden` column by
    construction; the separate function exists so the post-hoc label and the
    registered rule travel with the numbers.)

    This is ALWAYS computed. Whether it is REPORTED depends on E2 (the protocol
    reports it only if E2 reads REFUSES USABLE STUDIES); the caller decides.

    Args:
        df: Per-case frame.
        usable_threshold: Dice bar for usable.
        carry_qc_threshold: Carry-QC bar.
        n_boot: Bootstrap replicates.
        seed: Seed for `numpy.random.default_rng`.

    Returns:
        Dict with `label` ("post-hoc variant"), `n_thick_slice`,
        `n_thick_slice_accepted`, and the rates (`n`, `n_accepted`,
        `n_carry_qc_excluded`, each `p_*` as `(point, lo, hi)`).
    """
    _require_columns(df)
    out = _outcome_frame(
        df, "overridden", usable_threshold=usable_threshold, carry_qc_threshold=carry_qc_threshold
    )
    generator = np.random.default_rng(seed)
    thick = _flag(df, "thick_slice_refusal")
    return {
        "endpoint": "E4",
        "label": "post-hoc variant",
        "reference": "carried",
        "n_thick_slice": int(thick.sum()),
        "n_thick_slice_accepted": int((thick & _flag(df, "override_accepted")).sum()),
        **_rates(out, n_boot=n_boot, generator=generator),
    }


# ---------------------------------------------------------------------------
# Bridge check: do the re-run masks reproduce the stored P1.2 masks?
# ---------------------------------------------------------------------------


def bridge_check(rerun_wt_dice: pd.Series, *, min_dice: float) -> dict[str, Any]:
    """Per-study WT Dice between the v2 re-run mask and the stored P1.2 mask.

    Args:
        rerun_wt_dice: Series indexed by case_id, WT Dice(re-run, stored).
        min_dice: Reproduction bar (the protocol says 0.99), inclusive: a study at
            exactly `min_dice` passes. NaN (no comparable mask) counts as below.

    Returns:
        `{"n", "min_dice", "per_study", "below_min_dice" (case_ids), "all_pass"}`;
        `all_pass` is False for an empty input (nothing was verified).
    """
    values = pd.to_numeric(rerun_wt_dice, errors="coerce")
    below = values.index[~(values >= min_dice)]
    return {
        "n": int(len(values)),
        "min_dice": min_dice,
        "per_study": {str(k): float(v) for k, v in values.items()},
        "below_min_dice": [str(c) for c in below],
        "all_pass": bool(len(values) > 0 and len(below) == 0),
    }
