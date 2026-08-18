"""Measures how often a segmentation error changes the generated report (Phase 5).

`scripts/report.py` (not yet written) will produce one JSON report per case
from ground truth and, separately, one per case from a model's prediction --
both through `neurovision.reporting.report.build_report`. Dice alone cannot
say whether those two reports actually agree: `docs/experiments.md` note 18
and the entry in CLAUDE.md under "Report agreement is NOT monotonic in
Dice" record a real case where a LOWER-Dice model produced the BETTER
report. This module is the instrument that turns "agree" into a number --
one row of metrics per case, comparing a ground-truth-derived report against
a prediction-derived one for the same case.

The output table is built to be fed straight into
`neurovision.analysis.statistics.compare_models`: one row per `case_id`, one
numeric column per metric, and every column name uses a prefix already in
`KNOWN_METRIC_DIRECTIONS` (`agree_`, `jaccard_`, `match_`, `precision_`,
`recall_`, `abserr_`, `relerr_`) so the existing bootstrap/Wilcoxon/Holm
machinery works on it unchanged. Two model runs' agreement tables can also be
compared to each other with `compare_models` directly (e.g. "does
`neurovision`'s report agree with ground truth more often than the
baseline's does") -- that comparison IS what connects this pipeline to the
project's headline +0.0267 ET Dice result.

This module runs no model and reads no NIfTI: it only reads the JSON files
`neurovision.reporting.report.write_report` already wrote. It has no
dependency on the deep-learning stack -- see
`tests/test_report_agreement.py::test_module_does_not_import_the_deep_learning_stack`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path

import pandas as pd

__all__ = [
    "REPORT_AGREEMENT_VERSION",
    "load_report",
    "structure_set",
    "top_structure",
    "jaccard",
    "compare_reports",
    "agreement_table",
]

logger = logging.getLogger(__name__)

REPORT_AGREEMENT_VERSION: int = 1


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _is_missing(value: object) -> bool:
    """True for `None` or a non-finite float -- the two "no measurement" shapes a report carries."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def _abs_diff(gt_value: object, pred_value: object) -> float:
    """`|pred - gt|`, or NaN if either side is missing (see `_is_missing`)."""
    if _is_missing(gt_value) or _is_missing(pred_value):
        return float("nan")
    return abs(float(pred_value) - float(gt_value))  # type: ignore[arg-type]


def _rel_err(gt_value: object, pred_value: object) -> float:
    """`|pred - gt| / gt`, or NaN if either side is missing OR the ground truth is exactly 0.

    Never returns `inf`. 2.6% of BraTS 2021 cases have zero enhancing tumour
    (see CLAUDE.md), and a relative error against a zero denominator is
    undefined, not enormous -- an `inf` here would dominate every mean and
    every bootstrap replicate it lands in.
    """
    if _is_missing(gt_value) or _is_missing(pred_value):
        return float("nan")
    gt_float = float(gt_value)  # type: ignore[arg-type]
    if gt_float == 0.0:
        return float("nan")
    return abs(float(pred_value) - gt_float) / gt_float  # type: ignore[arg-type]


def _bool_agree(gt_value: object, pred_value: object) -> float:
    """`1.0`/`0.0` on equality, or NaN if either side is `None`."""
    if gt_value is None or pred_value is None:
        return float("nan")
    return 1.0 if gt_value == pred_value else 0.0


def _frac_of_structure_key(row: Mapping) -> float:
    """Sort key for picking the top structure row: `frac_of_structure`, missing/non-finite last."""
    value = row.get("frac_of_structure")
    try:
        as_float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("-inf")
    return as_float if math.isfinite(as_float) else float("-inf")


def _burden_value(burden: Mapping, block: str, key: str) -> object:
    """Reads `burden[block][key]`, returning `None` when the block or key is absent."""
    return burden.get(block, {}).get(key)


# --------------------------------------------------------------------------- #
# IO and per-report readers
# --------------------------------------------------------------------------- #


def load_report(path: str | Path) -> dict:
    """Reads one `build_report`-shaped JSON file.

    Args:
        path: Path to a `<case_id>.json` file written by
            `neurovision.reporting.report.write_report`.

    Returns:
        The parsed report dict.
    """
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def structure_set(report: Mapping, *, eloquent_only: bool = False) -> set[str]:
    """The set of structure names a report names.

    Args:
        report: A `build_report`-shaped dict.
        eloquent_only: If `True`, reads `eloquence.involved` (the structures
            a published eloquence classification also names). If `False`
            (default), reads `anatomy.structures` (every reported structure,
            up to the run's `top_n`).

    Returns:
        The set of non-empty `"structure"` values across the chosen list.
        Empty if the list itself is absent or empty.
    """
    block = "eloquence" if eloquent_only else "anatomy"
    field = "involved" if eloquent_only else "structures"
    rows = report.get(block, {}).get(field) or []
    return {row["structure"] for row in rows if row.get("structure")}


def top_structure(report: Mapping) -> str | None:
    """The name of `anatomy.structures`'s highest-`frac_of_structure` row.

    Args:
        report: A `build_report`-shaped dict.

    Returns:
        The top structure's name, or `None` if `anatomy.structures` is
        absent or empty.
    """
    rows = report.get("anatomy", {}).get("structures") or []
    if not rows:
        return None
    return max(rows, key=_frac_of_structure_key).get("structure")


def jaccard(a: AbstractSet[str], b: AbstractSet[str]) -> float:
    """`|a & b| / |a | b|`, defined as `1.0` when both sets are empty.

    Two reports that both name no structure agree completely; `0/0` must
    not become NaN (it would silently drop from every downstream mean) and
    must not become `0.0` either (that would read as maximal disagreement).

    Args:
        a: First structure-name set.
        b: Second structure-name set.

    Returns:
        A value in `[0.0, 1.0]`.
    """
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


# --------------------------------------------------------------------------- #
# compare_reports
# --------------------------------------------------------------------------- #


def compare_reports(gt: Mapping, pred: Mapping) -> dict[str, float]:
    """Compares one ground-truth-derived report against one prediction-derived report.

    Args:
        gt: A `build_report`-shaped dict built from the ground-truth
            segmentation.
        pred: A `build_report`-shaped dict for the SAME case, built from a
            model's prediction.

    Returns:
        A flat dict of floats (booleans emitted as `1.0`/`0.0`, never Python
        `bool`), one entry per metric: `jaccard_structures`,
        `precision_structures`, `recall_structures`, `match_top_structure`,
        `abserr_n_structures`, `agree_eloquent_any`,
        `jaccard_eloquent_structures`, `agree_near_eloquent`,
        `abserr_distance_mm`, `relerr_vol_ET`, `relerr_vol_TC`,
        `relerr_vol_WT`, `abserr_frac_enhancing_of_wt`,
        `abserr_n_components_WT`, `agree_multifocal_WT`,
        `agree_dominant_side_WT`. Every value not explicitly a raise-worthy
        structural problem (see below) that cannot be computed from an
        absent optional value is NaN for that metric only -- never a
        substituted zero, which would enter a population mean as a real
        observation.

    Raises:
        ValueError: `gt["case_id"] != pred["case_id"]` (comparing two
            different cases would produce a plausible-looking but
            meaningless row); `gt["report_version"] != pred["report_version"]`
            (schema drift means the fields being compared may not be the
            same fields); or either report is missing one of the required
            top-level blocks `anatomy` / `eloquence` / `burden`.
    """
    gt_case = gt["case_id"]
    pred_case = pred["case_id"]
    if gt_case != pred_case:
        raise ValueError(
            f"compare_reports: case_id mismatch -- gt is {gt_case!r}, pred is {pred_case!r}. "
            "Comparing two different cases would produce a plausible-looking but meaningless row."
        )

    gt_version = gt["report_version"]
    pred_version = pred["report_version"]
    if gt_version != pred_version:
        raise ValueError(
            f"compare_reports: report_version mismatch for case {gt_case!r} -- gt is "
            f"{gt_version!r}, pred is {pred_version!r}. Schema drift means the fields being "
            "compared may not be the same fields."
        )

    for name, report in (("gt", gt), ("pred", pred)):
        for key in ("anatomy", "eloquence", "burden"):
            if key not in report:
                raise ValueError(
                    f"compare_reports: {name} report for case {gt_case!r} is missing the "
                    f"required key {key!r}. Refusing to fill in a structurally wrong report "
                    "with NaN."
                )

    # --- Structures (anatomy.structures) --------------------------------- #
    gt_structs = structure_set(gt)
    pred_structs = structure_set(pred)
    intersection = gt_structs & pred_structs

    jaccard_structures = jaccard(gt_structs, pred_structs)
    precision_structures = len(intersection) / len(pred_structs) if pred_structs else float("nan")
    recall_structures = len(intersection) / len(gt_structs) if gt_structs else float("nan")
    if gt_structs and pred_structs:
        match_top_structure = 1.0 if top_structure(gt) == top_structure(pred) else 0.0
    else:
        match_top_structure = float("nan")
    abserr_n_structures = _abs_diff(
        gt["anatomy"].get("n_structures_involved"), pred["anatomy"].get("n_structures_involved")
    )

    # --- Eloquence --------------------------------------------------------- #
    gt_elo = gt["eloquence"]
    pred_elo = pred["eloquence"]
    gt_has_eloquent = bool(gt_elo.get("involved"))
    pred_has_eloquent = bool(pred_elo.get("involved"))
    agree_eloquent_any = 1.0 if gt_has_eloquent == pred_has_eloquent else 0.0
    jaccard_eloquent_structures = jaccard(
        structure_set(gt, eloquent_only=True), structure_set(pred, eloquent_only=True)
    )
    agree_near_eloquent = _bool_agree(gt_elo.get("near_eloquent"), pred_elo.get("near_eloquent"))
    abserr_distance_mm = _abs_diff(gt_elo.get("distance_mm"), pred_elo.get("distance_mm"))

    # --- Burden ------------------------------------------------------------ #
    gt_burden = gt["burden"]
    pred_burden = pred["burden"]

    relerr_vol_ET = _rel_err(
        _burden_value(gt_burden, "volumes", "vol_ET_mm3"),
        _burden_value(pred_burden, "volumes", "vol_ET_mm3"),
    )
    relerr_vol_TC = _rel_err(
        _burden_value(gt_burden, "volumes", "vol_TC_mm3"),
        _burden_value(pred_burden, "volumes", "vol_TC_mm3"),
    )
    relerr_vol_WT = _rel_err(
        _burden_value(gt_burden, "volumes", "vol_WT_mm3"),
        _burden_value(pred_burden, "volumes", "vol_WT_mm3"),
    )
    abserr_frac_enhancing_of_wt = _abs_diff(
        _burden_value(gt_burden, "fractions", "frac_enhancing_of_wt"),
        _burden_value(pred_burden, "fractions", "frac_enhancing_of_wt"),
    )
    gt_n_components_wt = _burden_value(gt_burden, "multifocality", "n_components_WT")
    pred_n_components_wt = _burden_value(pred_burden, "multifocality", "n_components_WT")
    abserr_n_components_WT = _abs_diff(gt_n_components_wt, pred_n_components_wt)
    # All three models over-report multifocality against ground truth
    # (30.7-40.7% vs a true 22.8%, docs/experiments.md note 18), which makes
    # this the least reliable field in the burden profile and the one most
    # worth measuring here rather than the component COUNT (agree_* is about
    # the multifocal/unifocal call, not the count -- see abserr_n_components
    # for the count itself).
    if _is_missing(gt_n_components_wt) or _is_missing(pred_n_components_wt):
        agree_multifocal_WT = float("nan")
    else:
        agree_multifocal_WT = (
            1.0 if (gt_n_components_wt >= 2) == (pred_n_components_wt >= 2) else 0.0  # type: ignore[operator]
        )
    agree_dominant_side_WT = _bool_agree(
        _burden_value(gt_burden, "laterality", "dominant_side_WT"),
        _burden_value(pred_burden, "laterality", "dominant_side_WT"),
    )

    return {
        "jaccard_structures": jaccard_structures,
        "precision_structures": precision_structures,
        "recall_structures": recall_structures,
        "match_top_structure": match_top_structure,
        "abserr_n_structures": abserr_n_structures,
        "agree_eloquent_any": agree_eloquent_any,
        "jaccard_eloquent_structures": jaccard_eloquent_structures,
        "agree_near_eloquent": agree_near_eloquent,
        "abserr_distance_mm": abserr_distance_mm,
        "relerr_vol_ET": relerr_vol_ET,
        "relerr_vol_TC": relerr_vol_TC,
        "relerr_vol_WT": relerr_vol_WT,
        "abserr_frac_enhancing_of_wt": abserr_frac_enhancing_of_wt,
        "abserr_n_components_WT": abserr_n_components_WT,
        "agree_multifocal_WT": agree_multifocal_WT,
        "agree_dominant_side_WT": agree_dominant_side_WT,
    }


# --------------------------------------------------------------------------- #
# agreement_table
# --------------------------------------------------------------------------- #

_NON_CASE_STEMS: frozenset[str] = frozenset({"report_manifest"})


def _resolve_case_ids(gt_dir: Path, pred_dir: Path) -> list[str]:
    """Sorted intersection of the two directories' report stems, warning about any mismatch."""
    gt_cases = {p.stem for p in gt_dir.glob("*.json")} - _NON_CASE_STEMS
    pred_cases = {p.stem for p in pred_dir.glob("*.json")} - _NON_CASE_STEMS
    only_gt = gt_cases - pred_cases
    only_pred = pred_cases - gt_cases
    if only_gt or only_pred:
        logger.warning(
            "agreement_table: %d case(s) present only in gt_dir, %d present only in pred_dir -- "
            "using the %d-case intersection.",
            len(only_gt),
            len(only_pred),
            len(gt_cases & pred_cases),
        )
    return sorted(gt_cases & pred_cases)


def agreement_table(
    gt_dir: str | Path, pred_dir: str | Path, *, cases: Sequence[str] | None = None
) -> pd.DataFrame:
    """Builds one report-agreement row per case.

    Args:
        gt_dir: Directory of ground-truth-derived reports, `<case_id>.json`
            per case.
        pred_dir: Directory of prediction-derived reports, same naming.
        cases: Explicit case ids to compare. If `None` (default), uses the
            sorted intersection of the two directories' `*.json` stems
            (ignoring `report_manifest` and any non-JSON file), logging at
            WARNING the count present in only one of them.

    Returns:
        A `DataFrame` indexed by `case_id` (sorted), one numeric column per
        `compare_reports` metric -- directly consumable by
        `neurovision.analysis.statistics.compare_models`.

    Raises:
        ValueError: The resolved case set is empty.
        RuntimeError: Every resolved case failed to compare (each failure is
            logged at ERROR with its traceback and does not stop the run).
    """
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)

    case_list = list(cases) if cases is not None else _resolve_case_ids(gt_path, pred_path)
    if not case_list:
        raise ValueError("agreement_table: no cases to compare -- the resolved case set is empty.")

    rows: dict[str, dict[str, float]] = {}
    n_failed = 0
    for case_id in case_list:
        try:
            gt_report = load_report(gt_path / f"{case_id}.json")
            pred_report = load_report(pred_path / f"{case_id}.json")
            rows[case_id] = compare_reports(gt_report, pred_report)
        except Exception:
            n_failed += 1
            logger.error("agreement_table: failed to compare case %r.", case_id, exc_info=True)

    if not rows:
        raise RuntimeError(
            f"agreement_table: every one of {len(case_list)} case(s) failed to compare; see the "
            "logged errors above."
        )

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "case_id"
    table = table.sort_index()

    logger.info(
        "agreement_table: %d/%d case(s) compared (%d failed). median jaccard_structures=%.4g, "
        "mean agree_near_eloquent=%.4g.",
        len(rows),
        len(case_list),
        n_failed,
        table["jaccard_structures"].median(),
        table["agree_near_eloquent"].mean(),
    )
    return table
