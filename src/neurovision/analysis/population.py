"""Population-level anatomical statistics across a whole split (Phase 5).

`docs/research/interpretable_pipeline_plan.md` names "population-level
anatomical statistics across all 1,251 cases as a headline figure -- it is
both a validation and a genuinely interesting result." This module is the
aggregation layer that turns the two per-case tables Phase 1 already writes
into population-level summaries:

    anatomy.csv         -- long, one row per (case_id, region, structure),
                            written by `scripts/localize.py` via
                            `neurovision.anatomy.localize.localize_case`.
    anatomy_summary.csv -- one row per case, written by the same script via
                            `neurovision.anatomy.localize.summarize_case`
                            (plus the two fields `scripts/localize.py`
                            overwrites itself: `distance_to_eloquent_mm`,
                            `near_eloquent`, and `frac_of_tumour_retained`).

Read `neurovision.anatomy.localize`'s module docstring before this one: the
`frac_of_tumour` / `frac_of_structure` distinction it establishes is the same
distinction this module aggregates, and conflating them here would be the
same mistake at population scale instead of per-case.

This is pure `pandas` table arithmetic: no model, no atlas, no NIfTI I/O, and
no dependency on the deep-learning stack or on plotting -- see
`tests/test_population.py::test_module_does_not_import_plotting_or_deep_learning_stack`.
Figures and the Hydra driver that will call this module live elsewhere and
are out of scope here.
"""

from __future__ import annotations

import logging

import pandas as pd

__all__ = [
    "structure_involvement_frequency",
    "lobe_burden_distribution",
    "eloquence_rates",
    "laterality_distribution",
    "summarize_population",
]

logger = logging.getLogger(__name__)

# Must match `neurovision.anatomy.localize`'s private `_UNLABELLED_NAME`
# exactly. Not exported by that module (it is an implementation detail of its
# own table), so this module names it again -- the same situation
# `scripts/localize.py`'s own `_UNLABELLED_STRUCTURE_NAME` is in, and for the
# same reason: this module has to recognise the row without importing a
# private symbol.
_UNLABELLED_NAME = "unlabelled"

# The label this module gives the "unlabelled" structure's bucket in
# `lobe_burden_distribution`'s `lobe` column. `localize_mask` stores that
# row's `lobe` field as `""` (it has no atlas lobe), so grouping by the raw
# `lobe` column would silently fold it into an unlabelled-looking `""`
# bucket next to any genuinely blank lobe value. Renaming it to a value that
# cannot collide with a real AAL lobe name keeps it a legible, distinct row.
_UNLABELLED_LOBE = "unlabelled"

# Rate-valued fields `eloquence_rates` checks for saturation. Deliberately
# NOT every key in the returned dict: `median_distance_to_eloquent_mm` is a
# millimetre distance and `median_n_structures_involved` /
# `median_frac_unlabelled` are medians of counts/fractions, none of which are
# themselves "the fraction of cases for which X is true" -- the thing
# `degenerate_fields` is about surfacing.
_RATE_FIELDS: tuple[str, ...] = ("frac_any_eloquent", "frac_near_eloquent", "frac_distance_zero")


# --------------------------------------------------------------------------- #
# Shared validation
# --------------------------------------------------------------------------- #


def _raise_if_empty(df: pd.DataFrame, caller: str) -> None:
    """Raises when `df` has no rows -- a population statistic over zero cases is not a result."""
    if df.empty:
        raise ValueError(
            f"{caller}: the input frame is empty. A population statistic computed over zero "
            "cases is not a result."
        )


def _filter_by_region(df: pd.DataFrame, region: str, *, caller: str) -> pd.DataFrame:
    """Restricts `df` to one region's rows, when `df` carries a `region` column at all.

    Never aggregates silently across ET, TC and WT: a structure "involved" in
    a case would then be counted once per region it was measured in, and the
    resulting frequency could exceed 1.0 with nothing failing. When `df` has
    no `region` column, it is assumed to already be scoped to a single region
    by the caller (e.g. a table pre-filtered before being passed in) and is
    returned unchanged.

    Args:
        df: An `anatomy.csv`-shaped frame, or a subset of one.
        region: The region to keep, e.g. `"WT"`.
        caller: The public function name, for the error message.

    Returns:
        `df` restricted to `df["region"] == region`, or `df` unchanged if it
        has no `region` column.

    Raises:
        ValueError: If `df` has a `region` column but no row matches
            `region`, naming the requested region and the ones actually
            present.
    """
    if "region" not in df.columns:
        return df
    available = sorted(str(r) for r in df["region"].unique())
    scoped = df[df["region"] == region]
    if scoped.empty:
        raise ValueError(
            f"{caller}: region {region!r} is not present in the input. Available region(s): "
            f"{available}."
        )
    return scoped


# --------------------------------------------------------------------------- #
# Structure-level frequency
# --------------------------------------------------------------------------- #


def structure_involvement_frequency(
    anatomy: pd.DataFrame,
    *,
    region: str = "WT",
    min_frac_of_structure: float = 0.05,
    exclude_unlabelled: bool = True,
) -> pd.DataFrame:
    """How often, and how badly, each atlas structure is involved across a population.

    A structure counts as "involved" in a case when that case's
    `frac_of_structure` for it reaches `min_frac_of_structure` -- the same
    quantity `neurovision.anatomy.localize` insists is not interchangeable
    with `frac_of_tumour`: a structure can hold a tiny share of the tumour's
    own volume while a large share of the structure itself is destroyed, and
    that is the quantity a population-level "how often is this structure
    affected" statistic should threshold on, not tumour share.

    Args:
        anatomy: An `anatomy.csv`-shaped long table (`case_id`, optionally
            `region`, `structure`, `laterality`, `lobe`, `eloquence`,
            `frac_of_tumour`, `frac_of_structure`).
        region: Region to restrict to; see `_filter_by_region`.
        min_frac_of_structure: The `frac_of_structure` threshold a case must
            reach for a structure to count as involved in it.
        exclude_unlabelled: Drops the `"unlabelled"` row before aggregating.
            That row exists because AAL parcellates grey matter only, so
            roughly a third of a real glioma matches no atlas structure --
            it is the population-scale form of the same coverage gap
            `neurovision.anatomy.localize` documents per case, and it is
            what makes `frac_of_tumour` sum to 1.0 within a case. It is not
            a structure, so it is excluded by default; set this `False` to
            see its own involvement rate rather than lose it silently.

    Returns:
        One row per structure actually present in the (region-filtered,
        optionally unlabelled-excluded) input, columns `structure`,
        `laterality`, `lobe`, `eloquence`, `n_cases_involved`,
        `frac_cases_involved`, `median_frac_of_structure`,
        `median_frac_of_tumour`, `n_cases`. `frac_cases_involved` divides by
        `n_cases`, the TOTAL distinct `case_id` count in the region-filtered
        input -- never by the number of cases in which the structure happens
        to appear, which would make every structure in the table look
        involved in 100% of cases. `median_frac_of_structure` /
        `median_frac_of_tumour` are computed over involved cases only, and
        are `NaN` when a structure has none. `n_cases` repeats on every row
        so the denominator never has to be recovered from elsewhere. Sorted
        by `frac_cases_involved` descending, then `structure` ascending.

    Raises:
        ValueError: If `anatomy` is empty, if `region` is not present in it
            (see `_filter_by_region`), or if a structure's `lobe` or
            `eloquence` is not constant across the cases it appears in --
            that would mean two different knowledge-base runs were
            concatenated into one table, and every count below would
            silently mix them.
    """
    _raise_if_empty(anatomy, "structure_involvement_frequency")
    scoped = _filter_by_region(anatomy, region, caller="structure_involvement_frequency")
    n_cases = int(scoped["case_id"].nunique())

    working = scoped
    if exclude_unlabelled:
        working = working[working["structure"] != _UNLABELLED_NAME]

    for col in ("lobe", "eloquence"):
        distinct_per_structure = working.groupby("structure")[col].nunique()
        offenders = distinct_per_structure[distinct_per_structure > 1]
        if not offenders.empty:
            bad_structure = str(offenders.index[0])
            values = sorted(
                str(v) for v in working.loc[working["structure"] == bad_structure, col].unique()
            )
            raise ValueError(
                f"structure_involvement_frequency: structure {bad_structure!r} has more than "
                f"one distinct {col!r} value across cases: {values}. This usually means two "
                "different knowledge-base runs were concatenated into one table."
            )

    involved = working[working["frac_of_structure"] >= min_frac_of_structure]

    rows: list[dict[str, object]] = []
    for structure, group in working.groupby("structure"):
        involved_group = involved[involved["structure"] == structure]
        n_cases_involved = int(involved_group["case_id"].nunique())
        rows.append(
            {
                "structure": structure,
                "laterality": group["laterality"].iloc[0],
                "lobe": group["lobe"].iloc[0],
                "eloquence": group["eloquence"].iloc[0],
                "n_cases_involved": n_cases_involved,
                "frac_cases_involved": n_cases_involved / n_cases,
                "median_frac_of_structure": (
                    float(involved_group["frac_of_structure"].median())
                    if not involved_group.empty
                    else float("nan")
                ),
                "median_frac_of_tumour": (
                    float(involved_group["frac_of_tumour"].median())
                    if not involved_group.empty
                    else float("nan")
                ),
                "n_cases": n_cases,
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(
        by=["frac_cases_involved", "structure"], ascending=[False, True]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Lobe burden
# --------------------------------------------------------------------------- #


def lobe_burden_distribution(anatomy: pd.DataFrame, *, region: str = "WT") -> pd.DataFrame:
    """Population-level share of tumour burden landing in each AAL lobe.

    The `"unlabelled"` row is INCLUDED, deliberately and always -- unlike
    `structure_involvement_frequency`, which excludes it by default because
    it is not a structure. Here it represents real tumour volume (roughly a
    third of a glioma, on real data) that the atlas has no lobe for, and a
    lobe chart that silently omitted it would misrepresent the distribution
    by implying the visible lobes account for the whole tumour. It is
    surfaced under the pseudo-lobe `"unlabelled"` (its own `lobe` field in
    `anatomy.csv` is `""`, which this function relabels so it cannot be
    confused with, or silently merged into, a genuinely blank value). Shares
    are NEVER renormalised after including it: if the labelled lobes sum to
    0.7 of the tumour, this function reports 0.7, not 1.0.

    Args:
        anatomy: An `anatomy.csv`-shaped long table.
        region: Region to restrict to; see `_filter_by_region`.

    Returns:
        One row per lobe (including `"unlabelled"`), columns `lobe`,
        `total_frac_of_tumour` (summed `frac_of_tumour` over all rows of
        that lobe, divided by `n_cases` -- the MEAN share of a tumour that
        lands there), `median_frac_of_tumour_per_case` (per case, the
        within-lobe `frac_of_tumour` is summed first -- 0.0 for a case with
        no involvement in that lobe -- and the median is then taken across
        ALL `n_cases` cases; this is a different statistic from the mean
        above and both are reported), `n_cases_involved`,
        `frac_cases_involved`, `n_cases`. Sorted by `total_frac_of_tumour`
        descending.

    Raises:
        ValueError: If `anatomy` is empty or `region` is absent from it.
    """
    _raise_if_empty(anatomy, "lobe_burden_distribution")
    scoped = _filter_by_region(anatomy, region, caller="lobe_burden_distribution")
    n_cases = int(scoped["case_id"].nunique())
    all_case_ids = scoped["case_id"].unique()

    working = scoped.copy()
    working["lobe"] = working["lobe"].where(
        working["structure"] != _UNLABELLED_NAME, _UNLABELLED_LOBE
    )

    rows: list[dict[str, object]] = []
    for lobe, group in working.groupby("lobe"):
        per_case = (
            group.groupby("case_id")["frac_of_tumour"].sum().reindex(all_case_ids, fill_value=0.0)
        )
        n_cases_involved = int((per_case > 0).sum())
        rows.append(
            {
                "lobe": lobe,
                "total_frac_of_tumour": float(group["frac_of_tumour"].sum()) / n_cases,
                "median_frac_of_tumour_per_case": float(per_case.median()),
                "n_cases_involved": n_cases_involved,
                "frac_cases_involved": n_cases_involved / n_cases,
                "n_cases": n_cases,
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(by="total_frac_of_tumour", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Eloquence
# --------------------------------------------------------------------------- #


def eloquence_rates(summary: pd.DataFrame) -> dict[str, float]:
    """Population-level eloquence-involvement rates from `anatomy_summary.csv`.

    Args:
        summary: An `anatomy_summary.csv`-shaped frame, one row per case,
            carrying `n_eloquent_structures`, `near_eloquent`,
            `distance_to_eloquent_mm`, `n_structures_involved` and
            `frac_unlabelled` (the fields `scripts/localize.py`'s
            `localize_one` writes, via `summarize_case` plus its own
            overwritten `distance_to_eloquent_mm` / `near_eloquent`).

    Returns:
        A dict with `n_cases`, `frac_any_eloquent` (share of cases with
        `n_eloquent_structures > 0`), `frac_near_eloquent` (share with
        `near_eloquent`), `median_distance_to_eloquent_mm`,
        `frac_distance_zero` (share with `distance_to_eloquent_mm == 0`),
        `median_n_structures_involved`, `median_frac_unlabelled`, and
        `degenerate_fields`: a list naming every one of
        `frac_any_eloquent` / `frac_near_eloquent` / `frac_distance_zero`
        that is exactly `0.0` or exactly `1.0` across the whole population.
        Measured on the real 189-case test split, `near_eloquent` is `True`
        for 100% of cases and `distance_to_eloquent_mm` is exactly `0` for
        99.5% of them -- essentially every glioma in this cohort directly
        touches a structure on the Sawaya list, so those two fields carry no
        per-case discriminating information. That is a real, reportable
        finding about the eloquence layer, not a 100% success rate, and this
        field exists so a reader does not mistake saturation for agreement.

    Raises:
        ValueError: If `summary` is empty.
    """
    _raise_if_empty(summary, "eloquence_rates")

    n_cases = int(len(summary))
    # Typed as dict[str, float] to match the module's public signature, even
    # though n_cases is stored as an int and degenerate_fields as a list --
    # see the Returns docstring above for what each key actually holds.
    rates: dict[str, object] = {
        "n_cases": n_cases,
        "frac_any_eloquent": float((summary["n_eloquent_structures"] > 0).mean()),
        "frac_near_eloquent": float(summary["near_eloquent"].mean()),
        "median_distance_to_eloquent_mm": float(summary["distance_to_eloquent_mm"].median()),
        "frac_distance_zero": float((summary["distance_to_eloquent_mm"] == 0).mean()),
        "median_n_structures_involved": float(summary["n_structures_involved"].median()),
        "median_frac_unlabelled": float(summary["frac_unlabelled"].median()),
    }

    degenerate = [field for field in _RATE_FIELDS if rates[field] in (0.0, 1.0)]
    rates["degenerate_fields"] = degenerate
    return rates  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Laterality
# --------------------------------------------------------------------------- #


def laterality_distribution(anatomy: pd.DataFrame, *, region: str = "WT") -> pd.DataFrame:
    """Population-level tumour share by laterality (`L` / `R` / `midline`).

    The population-scale sanity check that would catch a left-right flipped
    atlas: a real cohort should show `L` and `R` sharing tumour burden
    roughly symmetrically, and a systematic imbalance is exactly the signal
    `docs/research/phase0_atlas_findings.md` used to find the atlas's own
    laterality bug (see CLAUDE.md's "brain-mask Dice cannot detect a
    left-right flipped atlas" entry) -- this is that same class of check, run
    on tumour location instead of the atlas's own anatomy.

    Args:
        anatomy: An `anatomy.csv`-shaped long table.
        region: Region to restrict to; see `_filter_by_region`.

    Returns:
        One row per `laterality` value present, columns `laterality`,
        `mean_frac_of_tumour`, `median_frac_of_tumour` (both computed per
        case first -- summing `frac_of_tumour` within the laterality, 0.0
        for a case with none -- then averaged/medianed across all `n_cases`
        cases), `n_cases_involved`, `frac_cases_involved`, `n_cases`. Sorted
        by `laterality` ascending.

    Raises:
        ValueError: If `anatomy` is empty or `region` is absent from it.
    """
    _raise_if_empty(anatomy, "laterality_distribution")
    scoped = _filter_by_region(anatomy, region, caller="laterality_distribution")
    n_cases = int(scoped["case_id"].nunique())
    all_case_ids = scoped["case_id"].unique()

    rows: list[dict[str, object]] = []
    for laterality, group in scoped.groupby("laterality"):
        per_case = (
            group.groupby("case_id")["frac_of_tumour"].sum().reindex(all_case_ids, fill_value=0.0)
        )
        n_cases_involved = int((per_case > 0).sum())
        rows.append(
            {
                "laterality": laterality,
                "mean_frac_of_tumour": float(per_case.mean()),
                "median_frac_of_tumour": float(per_case.median()),
                "n_cases_involved": n_cases_involved,
                "frac_cases_involved": n_cases_involved / n_cases,
                "n_cases": n_cases,
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(by="laterality").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Everything together
# --------------------------------------------------------------------------- #


def summarize_population(
    anatomy: pd.DataFrame, summary: pd.DataFrame, *, region: str = "WT"
) -> dict[str, object]:
    """Runs the four population statistics and returns them as one bundle.

    Adds no computation of its own beyond reading off the total case count --
    it is a convenience assembly of `structure_involvement_frequency`,
    `lobe_burden_distribution`, `laterality_distribution` and
    `eloquence_rates`, so a caller (the Hydra driver or a figure-generation
    notebook) has one call site instead of four.

    Args:
        anatomy: An `anatomy.csv`-shaped long table, passed to the three
            structure/lobe/laterality functions.
        summary: An `anatomy_summary.csv`-shaped frame, passed to
            `eloquence_rates`.
        region: Region to restrict `anatomy` to; passed through to all three
            region-aware functions. `eloquence_rates` has no `region`
            parameter -- `summarize_case` already scopes `anatomy_summary.csv`
            to WT internally, see its docstring.

    Returns:
        `{"n_cases": int, "region": str, "structures": DataFrame,
        "lobes": DataFrame, "laterality": DataFrame, "eloquence": dict}`.
        `n_cases` is the region-filtered `anatomy`'s total distinct
        `case_id` count -- the same denominator `structures` / `lobes` /
        `laterality` each already repeat on every one of their own rows.

    Raises:
        ValueError: Whatever the four underlying functions raise -- an empty
            input, `region` absent from `anatomy`, or an inconstant
            `lobe`/`eloquence` value for some structure.
    """
    _raise_if_empty(anatomy, "summarize_population")
    scoped = _filter_by_region(anatomy, region, caller="summarize_population")
    n_cases = int(scoped["case_id"].nunique())

    return {
        "n_cases": n_cases,
        "region": region,
        "structures": structure_involvement_frequency(anatomy, region=region),
        "lobes": lobe_burden_distribution(anatomy, region=region),
        "laterality": laterality_distribution(anatomy, region=region),
        "eloquence": eloquence_rates(summary),
    }
