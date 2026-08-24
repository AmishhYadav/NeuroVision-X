"""Gate C -- does the trained SegQC model beat free predictive entropy at spotting a bad mask?

This is the statistics half of Phase C4 (per-cohort validation) and C5 (the
silent-failure test), backing the decision pre-registered in
`docs/research/preregistration_qc.md` -- READ THAT FILE FIRST. Nothing here
may reinterpret its endpoints, its family, or its decision rule; every
function below implements one paragraph of that document, named in its
docstring.

## Why entropy is the comparator

`SegQC` (`src/neurovision/models/qc.py`) is a second, independently-trained
network that regresses a mask's own Dice with no ground truth available. The
literature says that is achievable; the open question this project has
learned to ask three times now (note 39, note 44) is whether a trained
signal beats what is already free. Here the free alternative is mean
predicted-foreground entropy, `ent_mean_fg_R`, already cached from saved
logits at zero extra compute. The **primary** endpoint is therefore always
the DIFFERENCE, `delta_auroc = auroc_qc - auroc_ent`, never `auroc_qc` on
its own -- an AUROC the free baseline also reaches is not a result.

## Falsification before any endpoint

`falsification_check` is the discipline that runs before a single endpoint
is trusted: the QC pipeline reconstructs a case's identity-pair Dice from
saved logits, entirely independently of the evaluation pipeline that wrote
`per_case_metrics.csv`. If the two disagree by more than `tol`, one of them
is scoring a mask nobody ever evaluated, and the pre-registration says to
raise, not report.

## Pairing, always

`cell_endpoints`'s bootstrap resamples CASE INDICES, and both `AUROC_QC`
and `AUROC_ent` are recomputed on the SAME resampled indices in every
replicate. They are two scores for one set of cases; an unpaired interval
(resampling each independently) inflates the variance of their difference
and would report a falsely wide -- or, worse, falsely narrow in the wrong
direction -- confidence interval on `delta_auroc`. See
`neurovision.analysis.statistics.paired_bootstrap_ci`'s module docstring
for the same argument made about model-vs-model comparisons.

No torch import here on purpose -- this module runs no model, only numpy,
pandas and scipy on tables and arrays already computed elsewhere.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

# Reaching for detection.py's private `_pairwise_complete` deliberately: this
# project has already been bitten by two independent implementations of one
# quantity (AUROC, Spearman, Holm) drifting apart, and a second "drop any
# case with a NaN on any of three arrays" helper would be exactly that risk
# for no benefit -- the two modules already sit side by side in this package.
from neurovision.analysis.detection import _pairwise_complete, auroc, spearman
from neurovision.analysis.statistics import holm_bonferroni

__all__ = [
    "CellEndpoints",
    "falsification_check",
    "cell_endpoints",
    "endpoints_table",
    "mark_family",
    "gate_c_verdict",
    "silent_failure_table",
]

logger = logging.getLogger(__name__)

ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class CellEndpoints:
    """Every endpoint for one cohort x region cell, per `preregistration_qc.md`.

    Attributes:
        cohort: Cohort name, e.g. `"test"`, `"ssa"`, `"ped"`.
        region: Region name, e.g. `"ET"`, `"TC"`, `"WT"`.
        n: Cases scored, after dropping any case with a NaN on true Dice,
            predicted Dice, or entropy (pairwise-complete across all three).
        n_positive: Cases with `true_dice < bad_dice_threshold` -- the
            "bad case" event both AUROCs detect.
        auroc_qc: AUROC of the event using score `-qc_pred` (a LOWER
            predicted Dice should indicate a WORSE case).
        auroc_ent: AUROC of the same event using score `+entropy` (a HIGHER
            entropy should indicate a worse case).
        delta_auroc: `auroc_qc - auroc_ent`. The PRIMARY endpoint -- not
            `auroc_qc` alone, which the free baseline may also reach.
        delta_ci_lo: Lower bound of the paired-bootstrap percentile CI on
            `delta_auroc`.
        delta_ci_hi: Upper bound of that CI.
        p_bootstrap: Two-sided paired-bootstrap p-value for `delta_auroc !=
            0`, clipped into `[1 / n_valid_replicates, 1.0]` -- a bootstrap
            p-value can never be reported as exactly 0 from a finite number
            of replicates.
        spearman_qc: Spearman(`qc_pred`, `true_dice`). Expected POSITIVE --
            a better-predicted Dice should track a better true Dice.
        spearman_ci_lo: Lower bound of the paired-bootstrap percentile CI on
            `spearman_qc`.
        spearman_ci_hi: Upper bound of that CI.
        spearman_ent: Spearman(`entropy`, `true_dice`). Expected NEGATIVE
            (higher entropy, lower Dice) and reported with NO sign flip --
            this is the raw correlation, not a detection score, so its sign
            is informative and must not be normalised away.
        mae: `mean(|qc_pred - true_dice|)`.
        bias: `mean(qc_pred - true_dice)`. SIGN MATTERS: a POSITIVE bias
            means the QC model OVER-STATES mask quality -- the dangerous
            direction, because it makes a bad mask look fine.
        n_valid_replicates: Number of the `n_boot` bootstrap replicates
            whose `delta_auroc` was defined (not every resample of a small
            or class-imbalanced cell yields both AUROCs). `0` means every
            CI and p-value field above is NaN.
    """

    cohort: str
    region: str
    n: int
    n_positive: int
    auroc_qc: float
    auroc_ent: float
    delta_auroc: float
    delta_ci_lo: float
    delta_ci_hi: float
    p_bootstrap: float
    spearman_qc: float
    spearman_ci_lo: float
    spearman_ci_hi: float
    spearman_ent: float
    mae: float
    bias: float
    n_valid_replicates: int


def falsification_check(
    table: pd.DataFrame,
    published: pd.DataFrame,
    regions: Sequence[str],
    tol: float,
) -> pd.DataFrame:
    """Checks the QC pipeline's own Dice reconstruction against the published evaluation numbers.

    `table` holds, for the undegraded prediction, the Dice the QC pair
    generator reconstructs from saved logits (`true_dice_<region>`).
    `published` is that cohort's already-written `per_case_metrics.csv`
    (`dice_<region>`). These are the SAME quantity computed through two
    independent paths, so per the pre-registration's "falsification check"
    section, a disagreement means one path reconstructs a different mask
    than it claims to.

    Args:
        table: Indexed by `case_id`, with a `true_dice_<region>` column for
            every entry of `regions`.
        published: That cohort's `per_case_metrics.csv`, indexed by
            `case_id`, with a `dice_<region>` column for every entry of
            `regions`.
        regions: Region names to check, e.g. `("ET", "TC", "WT")`.
        tol: Maximum allowed median absolute difference, per region.

    Returns:
        A tidy `DataFrame`, one row per region: `region, n, median_abs_diff,
        max_abs_diff, n_over_tol`.

    Raises:
        ValueError: The shared `case_id` set between `table` and
            `published` is empty; a required column is missing from either
            frame; or any region's `median_abs_diff` exceeds `tol` -- named
            explicitly in the message, along with the offending value.
    """
    common = table.index.intersection(published.index)
    if len(common) == 0:
        raise ValueError(
            "falsification_check: table and published share no case_id at all -- nothing to "
            "compare. This usually means one of the two frames was loaded for the wrong cohort."
        )

    rows: list[dict[str, Any]] = []
    for region in regions:
        col_true = f"true_dice_{region}"
        col_pub = f"dice_{region}"
        if col_true not in table.columns:
            raise ValueError(f"falsification_check: {col_true!r} is missing from table.")
        if col_pub not in published.columns:
            raise ValueError(f"falsification_check: {col_pub!r} is missing from published.")

        recon = table.loc[common, col_true].to_numpy(dtype=np.float64)
        pub = published.loc[common, col_pub].to_numpy(dtype=np.float64)
        abs_diff = np.abs(recon - pub)

        rows.append(
            {
                "region": region,
                "n": int(len(common)),
                "median_abs_diff": float(np.median(abs_diff)),
                "max_abs_diff": float(np.max(abs_diff)),
                "n_over_tol": int(np.sum(abs_diff > tol)),
            }
        )

    result = pd.DataFrame(rows)
    failing = result[result["median_abs_diff"] > tol]
    if not failing.empty:
        row = failing.iloc[0]
        raise ValueError(
            f"falsification_check: region {row['region']!r} has median_abs_diff="
            f"{row['median_abs_diff']:.6g} > tol={tol:.6g}. true_dice_{row['region']} (the QC "
            "pipeline's own reconstruction of the undegraded prediction's Dice, read from saved "
            f"logits) and dice_{row['region']} (the already-published evaluation-pipeline number "
            "in per_case_metrics.csv) are the SAME quantity computed through TWO INDEPENDENT "
            "PATHS, so this disagreement means one path reconstructs a different mask than it "
            "claims to -- a different threshold, a different post-processing rule, or a geometry "
            "mismatch. No Gate C endpoint may be reported until this is resolved."
        )
    return result


def cell_endpoints(
    true_dice: ArrayLike,
    qc_pred: ArrayLike,
    entropy: ArrayLike,
    *,
    cohort: str,
    region: str,
    bad_dice_threshold: float,
    n_boot: int,
    ci: float,
    seed: int,
) -> CellEndpoints:
    """Computes every Gate C endpoint for one cohort x region cell.

    Cases with a NaN on any of `true_dice`, `qc_pred`, `entropy` are dropped
    PAIRWISE-COMPLETE across all three together (via
    `neurovision.analysis.detection._pairwise_complete`), never per-array --
    masking each independently would correlate different subsets of cases
    against each other. This is also why an all-NaN `entropy` (e.g. a region
    whose predicted foreground is empty in every case) leaves `n == 0`
    rather than silently substituting 0.0: an empty prediction and a
    confident one are different states.

    The bootstrap resamples CASE INDICES, and `AUROC_QC` / `AUROC_ent` are
    recomputed on the SAME resampled indices in every replicate -- see the
    module docstring's "Pairing, always" section.

    Args:
        true_dice: `(n,)` per-case true Dice for this cohort x region.
        qc_pred: `(n,)` per-case QC-predicted Dice, same case order.
        entropy: `(n,)` per-case `ent_mean_fg_<region>`, same case order.
        cohort: Cohort name, recorded in the result.
        region: Region name, recorded in the result.
        bad_dice_threshold: The positive-event threshold; the pre-
            registration fixes this at `0.7`.
        n_boot: Number of bootstrap replicates.
        ci: Confidence level, e.g. `0.95`.
        seed: Seeds `numpy.random.default_rng(seed)` -- the ONLY source of
            randomness used here, never the global `numpy.random`.

    Returns:
        A `CellEndpoints`. If zero cases survive pairwise-complete NaN
        removal, every field is NaN except `cohort`, `region`, `n=0`,
        `n_positive=0`, `n_valid_replicates=0` -- returned rather than
        raised, so a structurally-empty cell (e.g. every case's predicted
        foreground was empty) still produces a row in the endpoints table.
        If `n > 0` but every bootstrap replicate's `delta_auroc` is
        undefined (e.g. too few positive cases for any resample to contain
        both classes), every CI and p-value field is NaN while the point
        estimates (`auroc_qc`, `auroc_ent`, `delta_auroc`, `spearman_qc`,
        `spearman_ent`, `mae`, `bias`) are still computed from the
        un-resampled data.
    """
    true_clean, qc_clean, ent_clean = _pairwise_complete(true_dice, qc_pred, entropy)
    n = int(true_clean.size)

    if n == 0:
        logger.warning(
            "cell_endpoints(%s, %s): zero cases survive pairwise-complete NaN removal across "
            "true_dice/qc_pred/entropy -- returning an all-NaN cell.",
            cohort,
            region,
        )
        return CellEndpoints(
            cohort=cohort,
            region=region,
            n=0,
            n_positive=0,
            auroc_qc=float("nan"),
            auroc_ent=float("nan"),
            delta_auroc=float("nan"),
            delta_ci_lo=float("nan"),
            delta_ci_hi=float("nan"),
            p_bootstrap=float("nan"),
            spearman_qc=float("nan"),
            spearman_ci_lo=float("nan"),
            spearman_ci_hi=float("nan"),
            spearman_ent=float("nan"),
            mae=float("nan"),
            bias=float("nan"),
            n_valid_replicates=0,
        )

    positive = true_clean < bad_dice_threshold
    n_positive = int(positive.sum())

    # Lower predicted Dice => worse case, so the QC score is NEGATED.
    # Higher entropy => worse case, so the entropy score is used as-is.
    auroc_qc = auroc(-qc_clean, positive)
    auroc_ent = auroc(ent_clean, positive)
    delta_auroc = auroc_qc - auroc_ent  # NaN propagates if either side is NaN

    spearman_qc = spearman(qc_clean, true_clean)
    # Deliberately NOT sign-flipped -- see CellEndpoints.spearman_ent's docstring.
    spearman_ent = spearman(ent_clean, true_clean)

    mae = float(np.mean(np.abs(qc_clean - true_clean)))
    bias = float(np.mean(qc_clean - true_clean))

    generator = np.random.default_rng(seed)
    idx = generator.integers(0, n, size=(n_boot, n))  # resample CASE INDICES, preserves pairing

    delta_replicates = np.empty(n_boot, dtype=np.float64)
    spearman_replicates = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        rows = idx[b]
        true_b = true_clean[rows]
        qc_b = qc_clean[rows]
        ent_b = ent_clean[rows]
        positive_b = true_b < bad_dice_threshold

        # Same resampled case set feeds BOTH AUROCs -- this is what makes
        # the interval on delta_auroc a paired one.
        auroc_qc_b = auroc(-qc_b, positive_b)
        auroc_ent_b = auroc(ent_b, positive_b)
        delta_replicates[b] = auroc_qc_b - auroc_ent_b
        spearman_replicates[b] = spearman(qc_b, true_b)

    valid_delta = delta_replicates[np.isfinite(delta_replicates)]
    n_valid_replicates = int(valid_delta.size)

    if n_valid_replicates == 0:
        logger.warning(
            "cell_endpoints(%s, %s): every one of %d bootstrap replicates gave an undefined "
            "delta_auroc (likely too few positive cases for any resample to contain both "
            "classes) -- CI and p-value fields are NaN.",
            cohort,
            region,
            n_boot,
        )
        delta_ci_lo = delta_ci_hi = float("nan")
        p_bootstrap = float("nan")
        spearman_ci_lo = spearman_ci_hi = float("nan")
    else:
        alpha = 1.0 - ci
        lower_pct = 100.0 * alpha / 2.0
        upper_pct = 100.0 - lower_pct

        delta_ci_lo, delta_ci_hi = (
            float(v) for v in np.percentile(valid_delta, [lower_pct, upper_pct])
        )

        p_low = float(np.mean(valid_delta <= 0.0))
        p_high = float(np.mean(valid_delta >= 0.0))
        p_bootstrap = float(np.clip(2.0 * min(p_low, p_high), 1.0 / n_valid_replicates, 1.0))

        valid_spearman = spearman_replicates[np.isfinite(spearman_replicates)]
        if valid_spearman.size > 0:
            spearman_ci_lo, spearman_ci_hi = (
                float(v) for v in np.percentile(valid_spearman, [lower_pct, upper_pct])
            )
        else:
            spearman_ci_lo = spearman_ci_hi = float("nan")

    return CellEndpoints(
        cohort=cohort,
        region=region,
        n=n,
        n_positive=n_positive,
        auroc_qc=auroc_qc,
        auroc_ent=auroc_ent,
        delta_auroc=delta_auroc,
        delta_ci_lo=delta_ci_lo,
        delta_ci_hi=delta_ci_hi,
        p_bootstrap=p_bootstrap,
        spearman_qc=spearman_qc,
        spearman_ci_lo=spearman_ci_lo,
        spearman_ci_hi=spearman_ci_hi,
        spearman_ent=spearman_ent,
        mae=mae,
        bias=bias,
        n_valid_replicates=n_valid_replicates,
    )


def endpoints_table(cells: Sequence[CellEndpoints]) -> pd.DataFrame:
    """Stacks a sequence of `CellEndpoints` into one tidy table.

    Args:
        cells: One `CellEndpoints` per cohort x region cell.

    Returns:
        A `DataFrame`, one row per cell, columns matching `CellEndpoints`'
        field order.
    """
    return pd.DataFrame.from_records([asdict(cell) for cell in cells])


def mark_family(
    table: pd.DataFrame,
    *,
    in_distribution_cohort: str,
    min_positives: int,
) -> pd.DataFrame:
    """Marks the pre-registered gate family and Holm-corrects it.

    Per `preregistration_qc.md`'s "which cells enter the gate family"
    section: a cell is in the family iff its cohort is NOT the
    in-distribution one AND it has at least `min_positives` positive cases.
    Every row is kept -- only the family is corrected; a row outside it
    keeps `p_holm = NaN` rather than being dropped, so the full table stays
    inspectable.

    Args:
        table: An `endpoints_table` output (or anything with `cohort`,
            `n_positive`, `p_bootstrap` columns).
        in_distribution_cohort: The cohort name excluded from the family
            regardless of its positive count, e.g. `"test"`.
        min_positives: Minimum `n_positive` required to enter the family,
            e.g. `5`.

    Returns:
        A copy of `table` with two new columns: `in_family` (bool) and
        `p_holm` (float, NaN outside the family). If no row is `in_family`,
        `p_holm` is NaN for every row and `holm_bonferroni` is never called
        (it raises on an empty family; here that must not happen).
    """
    out = table.copy()
    out["in_family"] = (out["cohort"] != in_distribution_cohort) & (
        out["n_positive"] >= min_positives
    )
    out["p_holm"] = float("nan")

    family_mask = out["in_family"].to_numpy()
    if family_mask.any():
        p_raw = out.loc[family_mask, "p_bootstrap"].to_numpy(dtype=np.float64)
        adjusted, _ = holm_bonferroni(p_raw)
        out.loc[family_mask, "p_holm"] = adjusted
    else:
        logger.warning("mark_family: no row is in_family -- p_holm is NaN for every row.")

    return out


def gate_c_verdict(
    table: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Applies the pre-registered Gate C decision rule, verbatim.

    From `preregistration_qc.md`'s "decision rule" section: Gate C fires
    POSITIVE if, in at least one `in_family` row, `delta_auroc > 0` AND
    `p_holm < alpha` AND `delta_ci_lo > 0`. Otherwise NEGATIVE. There is no
    third outcome.

    Args:
        table: A `mark_family` output (needs `in_family`, `cohort`,
            `region`, `delta_auroc`, `delta_ci_lo`, `delta_ci_hi`,
            `p_holm`).
        alpha: Significance threshold `p_holm` is compared against. The
            pre-registration fixes this at `0.05`.

    Returns:
        A JSON-serialisable dict: `{"verdict": "POSITIVE"|"NEGATIVE",
        "alpha": float, "family_size": int, "firing_cells": [{"cohort":
        str, "region": str, "delta_auroc": float, "delta_ci_lo": float,
        "delta_ci_hi": float, "p_holm": float}, ...],
        "n_family_cells_with_positive_delta": int}`. Every value is a plain
        Python type (no numpy scalar, no `numpy.bool_`) so `json.dumps`
        never fails on it.

    Raises:
        KeyError: `table` is missing the `in_family` column -- `mark_family`
            must run first.
    """
    if "in_family" not in table.columns:
        raise KeyError("gate_c_verdict: table has no 'in_family' column; run mark_family first.")

    family = table[table["in_family"]]
    firing = family[
        (family["delta_auroc"] > 0) & (family["p_holm"] < alpha) & (family["delta_ci_lo"] > 0)
    ]

    firing_cells = [
        {
            "cohort": str(row["cohort"]),
            "region": str(row["region"]),
            "delta_auroc": float(row["delta_auroc"]),
            "delta_ci_lo": float(row["delta_ci_lo"]),
            "delta_ci_hi": float(row["delta_ci_hi"]),
            "p_holm": float(row["p_holm"]),
        }
        for _, row in firing.iterrows()
    ]

    verdict = "POSITIVE" if firing_cells else "NEGATIVE"
    if verdict == "NEGATIVE":
        logger.info("gate_c_verdict: NEGATIVE -- no in_family cell met all three conditions.")

    return {
        "verdict": verdict,
        "alpha": float(alpha),
        "family_size": int(len(family)),
        "firing_cells": firing_cells,
        "n_family_cells_with_positive_delta": int((family["delta_auroc"] > 0).sum()),
    }


def silent_failure_table(
    table: pd.DataFrame,
    *,
    in_distribution_cohort: str,
) -> pd.DataFrame:
    """Builds the C5 silent-failure endpoints: does the QC model degrade quietly under shift?

    Per `preregistration_qc.md`'s C5 section: for each shifted cohort, the
    change in Spearman and in signed bias from the in-distribution cohort,
    plus the pre-registered directional flag (bias is predicted to be MORE
    POSITIVE on PED than in-distribution -- i.e. the QC model over-estimates
    Dice worst where the segmentation model is worst).

    Args:
        table: An `endpoints_table` output (needs `cohort`, `region`,
            `spearman_qc`, `bias`, `mae`).
        in_distribution_cohort: The cohort every delta is measured against,
            e.g. `"test"`.

    Returns:
        A `DataFrame`, one row per cohort x region: `cohort, region,
        spearman_qc, bias, mae, delta_spearman_vs_in_distribution,
        delta_bias_vs_in_distribution, bias_more_positive_than_in_distribution`.
        In-distribution rows get `NaN` deltas and `False` for the flag,
        since there is no external comparison for them to make.
    """
    out = table[["cohort", "region", "spearman_qc", "bias", "mae"]].copy()

    in_dist = table[table["cohort"] == in_distribution_cohort].set_index("region")

    delta_spearman: list[float] = []
    delta_bias: list[float] = []
    more_positive: list[bool] = []
    for _, row in table.iterrows():
        region = row["region"]
        if row["cohort"] == in_distribution_cohort or region not in in_dist.index:
            delta_spearman.append(float("nan"))
            delta_bias.append(float("nan"))
            more_positive.append(False)
            continue

        ref = in_dist.loc[region]
        delta_spearman.append(float(row["spearman_qc"] - ref["spearman_qc"]))
        delta_bias.append(float(row["bias"] - ref["bias"]))
        more_positive.append(bool(row["bias"] > ref["bias"]))

    out["delta_spearman_vs_in_distribution"] = delta_spearman
    out["delta_bias_vs_in_distribution"] = delta_bias
    out["bias_more_positive_than_in_distribution"] = more_positive
    return out
