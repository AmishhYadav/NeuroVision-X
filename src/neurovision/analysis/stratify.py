"""Segmentation performance stratified by ground-truth tumour size.

Measured fact motivating this module: BraTS-Africa tumours are roughly 1.7x
larger than BraTS 2021 ones (WT median 163,749 vs 89,335 voxels). Larger
tumours are generally easier to segment, so a raw cross-cohort Dice
comparison confounds "domain shift" with "tumour size" -- a cohort could
score higher for no reason other than its tumours being bigger. This module
makes performance readable AS A FUNCTION OF tumour size (`stratified_summary`
/ `stratified_comparison`) and supports restricting a cross-cohort
comparison to a size range both cohorts actually populate
(`overlapping_volume_range` / `volume_matched_subset`), so a claim of
"cohort A is harder" is not really a claim of "cohort A's tumours are
smaller".

It deliberately has no dependency on torch, so it (and anything that imports
it) stays importable in an environment with no deep-learning stack at all --
the same reasoning as `neurovision.visualization.figures` and
`neurovision.anatomy.burden`.

Ground-truth region definitions mirror
`neurovision.metrics.segmentation.classes_to_regions` exactly (read that
function before changing this one): ET = {3}, TC = {1, 3}, WT = {1, 2, 3}.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from neurovision.analysis.statistics import compare_models

__all__ = [
    "ground_truth_volumes",
    "quantile_bin_edges",
    "assign_bins",
    "stratified_summary",
    "stratified_comparison",
    "overlapping_volume_range",
    "volume_matched_subset",
]

logger = logging.getLogger(__name__)

# Mirrors neurovision.metrics.segmentation._NECROTIC_CORE / _EDEMA /
# _ENHANCING_TUMOR. Kept as local constants rather than importing that
# module's private names, for the same reason classes_to_regions itself is
# not imported into other analysis modules: this file must stay import-safe
# with no torch installed, and neurovision.metrics.segmentation imports torch.
_NECROTIC_CORE = 1
_EDEMA = 2
_ENHANCING_TUMOR = 3

_REGION_CLASSES: dict[str, set[int]] = {
    "ET": {_ENHANCING_TUMOR},
    "TC": {_NECROTIC_CORE, _ENHANCING_TUMOR},
    "WT": {_NECROTIC_CORE, _EDEMA, _ENHANCING_TUMOR},
}


def _with_case_id_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sets `case_id` as the index if the index is unnamed and that column exists.

    Mirrors `neurovision.analysis.statistics._with_case_id_index` exactly,
    so a per-case table can be passed to either module interchangeably.
    """
    if df.index.name is None and "case_id" in df.columns:
        df = df.set_index("case_id")
    return df


def ground_truth_volumes(
    prep_dir: str | Path,
    case_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Computes ground-truth region volumes for preprocessed cases.

    Reads `<prep_dir>/<case_id>/label.npy` (uint8 class map, values in
    `{0, 1, 2, 3}`, shape `(D, H, W)`) and `<prep_dir>/<case_id>/meta.json`
    for `spacing`, exactly the layout `neurovision.data.preprocessing`
    writes.

    Args:
        prep_dir: Root directory of preprocessed cases.
        case_ids: Case identifiers to read. If `None`, every subdirectory of
            `prep_dir` that has a `label.npy` is used.

    Returns:
        A `DataFrame` indexed by `case_id` with integer voxel-count columns
        `vol_ET_vox`, `vol_TC_vox`, `vol_WT_vox` and millimetre-cubed
        columns `vol_ET_mm3`, `vol_TC_mm3`, `vol_WT_mm3` (voxel count times
        the product of `spacing`).

    Raises:
        FileNotFoundError: `prep_dir` does not exist.
    """
    prep_dir = Path(prep_dir)
    if not prep_dir.exists():
        raise FileNotFoundError(f"ground_truth_volumes: no directory at {prep_dir}.")

    if case_ids is None:
        resolved_ids = sorted(
            p.name for p in prep_dir.iterdir() if p.is_dir() and (p / "label.npy").is_file()
        )
    else:
        resolved_ids = list(case_ids)

    records: list[dict[str, object]] = []
    for case_id in resolved_ids:
        case_dir = prep_dir / case_id
        label_path = case_dir / "label.npy"
        if not label_path.is_file():
            logger.warning(
                "ground_truth_volumes: no label.npy for case %r at %s -- skipping (unlabeled "
                "cases legitimately exist).",
                case_id,
                case_dir,
            )
            continue

        label = np.load(label_path)  # (D, H, W) uint8
        meta_path = case_dir / "meta.json"
        with meta_path.open("r") as f:
            meta = json.load(f)
        spacing = tuple(float(s) for s in meta["spacing"])
        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

        record: dict[str, object] = {"case_id": case_id}
        for region, classes in _REGION_CLASSES.items():
            vox = int(np.isin(label, list(classes)).sum())
            record[f"vol_{region}_vox"] = vox
            record[f"vol_{region}_mm3"] = vox * voxel_volume_mm3
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "vol_ET_vox",
                "vol_TC_vox",
                "vol_WT_vox",
                "vol_ET_mm3",
                "vol_TC_mm3",
                "vol_WT_mm3",
            ]
        ).rename_axis("case_id")

    return pd.DataFrame(records).set_index("case_id")


def quantile_bin_edges(values: Sequence[float], n_bins: int = 4) -> list[float]:
    """Computes bin edges at equal-count quantiles of `values`.

    Args:
        values: Values to bin, e.g. a ground-truth volume column.
        n_bins: Number of requested bins.

    Returns:
        `n_bins + 1` edges, ascending, with the first edge equal to
        `min(values)` and the last edge `float("inf")` (so `assign_bins`'s
        half-open convention always has an upper bound for the final bin).
        Interior edges from `np.quantile` are deduplicated when many tied
        values collapse two quantiles onto the same point, which yields
        fewer than `n_bins` bins -- logged as a warning when that happens.

    Raises:
        ValueError: `n_bins < 1` or `values` is empty.
    """
    if n_bins < 1:
        raise ValueError(f"quantile_bin_edges: n_bins must be >= 1, got {n_bins}.")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("quantile_bin_edges: values is empty.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = np.quantile(arr, quantiles)
    raw_edges[-1] = np.inf

    # Deduplicate interior edges (a run of identical quantiles from
    # tied/duplicate-heavy input) while always keeping the first (min) and
    # last (inf) edge.
    edges: list[float] = [float(raw_edges[0])]
    for e in raw_edges[1:]:
        if float(e) != edges[-1]:
            edges.append(float(e))

    n_actual_bins = len(edges) - 1
    if n_actual_bins < n_bins:
        logger.warning(
            "quantile_bin_edges: requested %d bins but only %d distinct edge(s) resulted after "
            "deduplication (heavily duplicate-valued input) -- %d bin(s) produced.",
            n_bins,
            len(edges),
            n_actual_bins,
        )

    return edges


def assign_bins(values: Sequence[float], edges: Sequence[float]) -> list[str]:
    """Labels each value with the half-open bin `[lo, hi)` it falls in.

    Mirrors the half-open convention of
    `neurovision.metrics.boundary.boundary_band_masks` / `band_label`, so
    bins tile `[edges[0], edges[-1])` exactly once with no value double
    counted at a shared edge.

    Args:
        values: Values to bin, e.g. a ground-truth volume column.
        edges: Ascending bin edges, e.g. from `quantile_bin_edges`. Must
            have at least 2 entries.

    Returns:
        One label per value, format `f"{lo:.0f}-{hi:.0f}"`, with the final
        bin rendered `f"{lo:.0f}-inf"` when `edges[-1]` is `float("inf")`.

    Raises:
        ValueError: `edges` has fewer than 2 entries, is not strictly
            ascending, or a value falls below `edges[0]` (meaning the edges
            came from a different cohort than the one being binned).
    """
    edges_arr = np.asarray(edges, dtype=np.float64)
    if edges_arr.size < 2:
        raise ValueError(f"assign_bins: edges must have at least 2 entries, got {edges_arr.size}.")
    if np.any(np.diff(edges_arr) <= 0):
        raise ValueError(
            f"assign_bins: edges must be strictly ascending, got {edges_arr.tolist()}."
        )

    def _fmt(v: float) -> str:
        return "inf" if v == float("inf") else f"{v:.0f}"

    labels = [f"{_fmt(edges_arr[i])}-{_fmt(edges_arr[i + 1])}" for i in range(edges_arr.size - 1)]

    result: list[str] = []
    for v in values:
        v = float(v)
        if v < edges_arr[0]:
            raise ValueError(
                f"assign_bins: value {v} is below the first edge {edges_arr[0]} -- this means "
                "the edges came from a different cohort than the one being binned."
            )
        # Half-open [lo, hi): searchsorted with side="right" puts a value
        # exactly equal to an interior edge into the UPPER bin.
        idx = int(np.searchsorted(edges_arr, v, side="right")) - 1
        idx = min(idx, len(labels) - 1)  # a value == the last (inf) edge cannot occur
        result.append(labels[idx])
    return result


def _join_metrics_and_volumes(
    per_case: pd.DataFrame, volumes: pd.DataFrame, volume_col: str
) -> pd.DataFrame:
    """Inner-joins a per-case metric table to a volumes table on case id."""
    per_case_idx = _with_case_id_index(per_case)
    volumes_idx = _with_case_id_index(volumes)
    joined = per_case_idx.join(volumes_idx[[volume_col]], how="inner")
    return joined


def stratified_summary(
    per_case: pd.DataFrame,
    volumes: pd.DataFrame,
    metric_cols: Sequence[str],
    volume_col: str = "vol_WT_vox",
    n_bins: int = 4,
    edges: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Summarizes metrics per ground-truth-volume bin.

    Args:
        per_case: Per-case metric table, indexed by `case_id` (or with a
            `case_id` column and an unnamed index).
        volumes: Ground-truth volume table, same convention, e.g. from
            `ground_truth_volumes`.
        metric_cols: Metric column names in `per_case` to summarize.
        volume_col: Column of `volumes` to bin on.
        n_bins: Forwarded to `quantile_bin_edges` when `edges` is `None`.
        edges: Explicit bin edges. If given, `n_bins` is ignored.

    Returns:
        A `DataFrame`, one row per bin, ORDERED NUMERICALLY by `bin_lo` (a
        lexicographic sort of the bin label would put e.g. `"10000-inf"`
        before `"2000-5000"`), with columns `bin`, `bin_lo`, `bin_hi`, `n`,
        and per metric `<metric>_mean`, `<metric>_median`,
        `<metric>_n_missing`.
    """
    joined = _join_metrics_and_volumes(per_case, volumes, volume_col)
    resolved_edges = (
        list(edges)
        if edges is not None
        else quantile_bin_edges(joined[volume_col].to_numpy(), n_bins=n_bins)
    )
    bins = assign_bins(joined[volume_col].to_numpy(), resolved_edges)
    joined = joined.assign(bin=bins)

    bin_bounds: dict[str, tuple[float, float]] = {}
    for i in range(len(resolved_edges) - 1):
        lo, hi = resolved_edges[i], resolved_edges[i + 1]

        def _fmt(v: float) -> str:
            return "inf" if v == float("inf") else f"{v:.0f}"

        bin_bounds[f"{_fmt(lo)}-{_fmt(hi)}"] = (lo, hi)

    records: list[dict[str, object]] = []
    for bin_label, group in joined.groupby("bin"):
        lo, hi = bin_bounds[bin_label]
        record: dict[str, object] = {
            "bin": bin_label,
            "bin_lo": lo,
            "bin_hi": hi,
            "n": len(group),
        }
        for metric in metric_cols:
            values = group[metric]
            record[f"{metric}_mean"] = float(values.mean(skipna=True))
            record[f"{metric}_median"] = float(values.median(skipna=True))
            record[f"{metric}_n_missing"] = int(values.isna().sum())
        records.append(record)

    result = pd.DataFrame(records).sort_values("bin_lo").reset_index(drop=True)
    return result


def stratified_comparison(
    per_case_a: pd.DataFrame,
    per_case_b: pd.DataFrame,
    volumes: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    generator: np.random.Generator,
    name_a: str = "a",
    name_b: str = "b",
    volume_col: str = "vol_WT_vox",
    n_bins: int = 4,
    edges: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Runs a paired A-vs-B `compare_models` comparison separately within each volume bin.

    Reuses `neurovision.analysis.statistics.compare_models` for every
    statistical computation -- this function only partitions cases into
    bins and concatenates the per-bin results, so bootstrap CIs, Wilcoxon
    tests, effect sizes and Holm correction all come from that module
    unchanged.

    Args:
        per_case_a: Per-case metric table for model/cohort A.
        per_case_b: Per-case metric table for model/cohort B.
        volumes: Ground-truth volume table used to assign bins. Cases are
            binned by THEIR OWN volume, so `volumes` should hold an entry
            for every case_id appearing in `per_case_a`/`per_case_b` (a case
            missing from `volumes` is simply absent from every bin, the
            same way a case missing from one of `per_case_a`/`per_case_b`
            is dropped by `compare_models`' index intersection).
        metric_cols: Forwarded to `compare_models` as `metrics`.
        generator: Forwarded to `compare_models` for every bin -- REQUIRED,
            no default and no use of the global RNG.
        name_a: Forwarded to `compare_models`.
        name_b: Forwarded to `compare_models`.
        volume_col: Column of `volumes` to bin on.
        n_bins: Forwarded to `quantile_bin_edges` when `edges` is `None`.
            Bin edges are computed ONCE, over every case's volume (before
            splitting into A/B), so both models are compared within
            identical bin boundaries.
        edges: Explicit bin edges. If given, `n_bins` is ignored.

    Returns:
        The concatenation of one `compare_models` call per bin, each with a
        `bin` column (the bin label) and a `holm_family` column set to the
        SAME bin label -- Holm correction happens INSIDE each per-bin
        `compare_models` call, across that bin's metrics only, never across
        bins. A bin with fewer than 5 paired cases (cases present in both
        `per_case_a` and `per_case_b`, intersected against `volumes`) is
        skipped with a `logging.warning` and does not appear in the output.
    """
    volumes_idx = _with_case_id_index(volumes)
    resolved_edges = (
        list(edges)
        if edges is not None
        else quantile_bin_edges(volumes_idx[volume_col].to_numpy(), n_bins=n_bins)
    )
    bin_labels = assign_bins(volumes_idx[volume_col].to_numpy(), resolved_edges)
    case_to_bin = dict(zip(volumes_idx.index, bin_labels, strict=True))

    a_idx = _with_case_id_index(per_case_a)
    b_idx = _with_case_id_index(per_case_b)
    paired_cases = a_idx.index.intersection(b_idx.index).intersection(volumes_idx.index)

    bin_bounds: dict[str, tuple[float, float]] = {}
    for i in range(len(resolved_edges) - 1):
        lo, hi = resolved_edges[i], resolved_edges[i + 1]

        def _fmt(v: float) -> str:
            return "inf" if v == float("inf") else f"{v:.0f}"

        bin_bounds[f"{_fmt(lo)}-{_fmt(hi)}"] = (lo, hi)

    # Group the paired case ids by bin.
    cases_by_bin: dict[str, list[str]] = {label: [] for label in bin_bounds}
    for case_id in paired_cases:
        cases_by_bin[case_to_bin[case_id]].append(case_id)

    tables: list[pd.DataFrame] = []
    for bin_label, bounds in sorted(bin_bounds.items(), key=lambda kv: kv[1][0]):
        cases = cases_by_bin[bin_label]
        if len(cases) < 5:
            logger.warning(
                "stratified_comparison: bin %r has only %d paired case(s) (< 5) -- skipping. "
                "A bootstrap CI over this few cases is not meaningful.",
                bin_label,
                len(cases),
            )
            continue
        table = compare_models(
            a_idx.loc[cases],
            b_idx.loc[cases],
            generator=generator,
            metrics=metric_cols,
            name_a=name_a,
            name_b=name_b,
        )
        table = table.reset_index()
        table["bin"] = bin_label
        table["holm_family"] = bin_label
        tables.append(table)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def overlapping_volume_range(
    volumes_a: pd.DataFrame,
    volumes_b: pd.DataFrame,
    volume_col: str = "vol_WT_vox",
    quantile: float = 0.05,
) -> tuple[float, float]:
    """Finds the volume interval where two cohorts genuinely overlap in size.

    Args:
        volumes_a: Ground-truth volume table for cohort A.
        volumes_b: Ground-truth volume table for cohort B.
        volume_col: Column to compare.
        quantile: Trims `quantile` from each tail of each cohort before
            intersecting, so one outlier case cannot define the whole
            matched range.

    Returns:
        `(lo, hi)` where `lo = max(quantile(a), quantile(b))` and
        `hi = min(quantile(a, 1 - quantile), quantile(b, 1 - quantile))`.

    Raises:
        ValueError: The resulting interval is empty (`lo >= hi`), meaning
            the two cohorts do not overlap in size at all (at this
            trimming).
    """
    a_vals = _with_case_id_index(volumes_a)[volume_col].to_numpy(dtype=np.float64)
    b_vals = _with_case_id_index(volumes_b)[volume_col].to_numpy(dtype=np.float64)

    a_lo, a_hi = np.quantile(a_vals, [quantile, 1.0 - quantile])
    b_lo, b_hi = np.quantile(b_vals, [quantile, 1.0 - quantile])

    lo = float(max(a_lo, b_lo))
    hi = float(min(a_hi, b_hi))

    if lo >= hi:
        raise ValueError(
            f"overlapping_volume_range: cohorts do not overlap in size at quantile={quantile} -- "
            f"got an empty interval [{lo}, {hi}]. Cohort A range [{a_lo:.4g}, {a_hi:.4g}], "
            f"cohort B range [{b_lo:.4g}, {b_hi:.4g}]."
        )
    return lo, hi


def volume_matched_subset(
    volumes: pd.DataFrame,
    lo: float,
    hi: float,
    volume_col: str = "vol_WT_vox",
) -> list[str]:
    """Selects case ids whose volume lies in the closed interval `[lo, hi]`.

    Args:
        volumes: Ground-truth volume table, e.g. from `ground_truth_volumes`.
        lo: Lower bound, inclusive.
        hi: Upper bound, inclusive.
        volume_col: Column to filter on.

    Returns:
        Case ids (the index of `volumes`) whose `volume_col` value satisfies
        `lo <= v <= hi`, in the same order they appear in `volumes`.
    """
    volumes_idx = _with_case_id_index(volumes)
    mask = (volumes_idx[volume_col] >= lo) & (volumes_idx[volume_col] <= hi)
    return list(volumes_idx.index[mask])
