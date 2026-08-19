"""Tests for `neurovision.analysis.population`.

All fixtures are hand-built here, small enough that every returned number is
hand-checkable -- no file under `outputs/` is ever read, matching
`neurovision.anatomy.localize`'s and `neurovision.analysis.statistics`'s own
test conventions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neurovision.analysis import population
from neurovision.analysis.population import (
    eloquence_rates,
    laterality_distribution,
    lobe_burden_distribution,
    structure_involvement_frequency,
    summarize_population,
)

# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


def _anatomy_row(
    case_id: str,
    structure: str,
    *,
    region: str = "WT",
    laterality: str = "L",
    lobe: str = "frontal",
    eloquence: str = "unclassified",
    frac_of_tumour: float = 0.1,
    frac_of_structure: float = 0.1,
) -> dict[str, object]:
    """One `anatomy.csv`-shaped row, with sensible defaults for fields a test does not vary."""
    return {
        "case_id": case_id,
        "region": region,
        "structure": structure,
        "laterality": laterality,
        "lobe": lobe,
        "eloquence": eloquence,
        "matched_term": "",
        "n_voxels": 100,
        "volume_mm3": 100.0,
        "frac_of_tumour": frac_of_tumour,
        "frac_of_structure": frac_of_structure,
    }


def _anatomy_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _summary_row(
    case_id: str,
    *,
    n_eloquent_structures: int = 0,
    near_eloquent: bool = False,
    distance_to_eloquent_mm: float = float("nan"),
    n_structures_involved: int = 1,
    frac_unlabelled: float = 0.0,
) -> dict[str, object]:
    """One `anatomy_summary.csv`-shaped row, with sensible defaults."""
    return {
        "case_id": case_id,
        "n_structures_involved": n_structures_involved,
        "top_structure": "Frontal_L",
        "top_frac_of_structure": 0.5,
        "most_displaced_structure": "Frontal_L",
        "frac_unlabelled": frac_unlabelled,
        "n_eloquent_structures": n_eloquent_structures,
        "eloquent_frac_of_tumour": 0.0,
        "distance_to_eloquent_mm": distance_to_eloquent_mm,
        "dominant_lobe": "frontal",
        "coverage_line": "",
        "near_eloquent": near_eloquent,
        "frac_of_tumour_retained": 1.0,
    }


def _summary_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. frac_cases_involved uses the TOTAL case count, not the appearance count
# --------------------------------------------------------------------------- #


def test_frac_cases_involved_uses_total_case_count_as_denominator() -> None:
    rows = [
        _anatomy_row("case1", "Frontal_L", frac_of_structure=0.5, frac_of_tumour=0.3),
        _anatomy_row("case2", "Frontal_L", frac_of_structure=0.6, frac_of_tumour=0.4),
        _anatomy_row(
            "case3",
            "Parietal_R",
            lobe="parietal",
            eloquence="eloquent",
            frac_of_structure=0.2,
            frac_of_tumour=0.5,
        ),
        _anatomy_row(
            "case4",
            "Parietal_R",
            lobe="parietal",
            eloquence="eloquent",
            frac_of_structure=0.3,
            frac_of_tumour=0.6,
        ),
        _anatomy_row(
            "case5",
            "Parietal_R",
            lobe="parietal",
            eloquence="eloquent",
            frac_of_structure=0.1,
            frac_of_tumour=0.2,
        ),
    ]
    df = _anatomy_df(rows)

    table = structure_involvement_frequency(df)

    frontal = table[table["structure"] == "Frontal_L"].iloc[0]
    # Frontal_L appears in 2 of 5 TOTAL cases -- the denominator is 5, the
    # split's total case count, not 2 (the number of rows Frontal_L has).
    assert frontal["n_cases_involved"] == 2
    assert frontal["n_cases"] == 5
    assert frontal["frac_cases_involved"] == pytest.approx(0.4)

    parietal = table[table["structure"] == "Parietal_R"].iloc[0]
    assert parietal["n_cases_involved"] == 3
    assert parietal["frac_cases_involved"] == pytest.approx(0.6)

    # Sorted by frac_cases_involved descending.
    assert list(table["structure"]) == ["Parietal_R", "Frontal_L"]


# --------------------------------------------------------------------------- #
# 2. min_frac_of_structure excludes a case below threshold
# --------------------------------------------------------------------------- #


def test_min_frac_of_structure_excludes_case_below_threshold() -> None:
    rows = [
        _anatomy_row("case_a", "Frontal_L", frac_of_structure=0.5, frac_of_tumour=0.3),
        _anatomy_row("case_b", "Frontal_L", frac_of_structure=0.6, frac_of_tumour=0.4),
        _anatomy_row("case_c", "Frontal_L", frac_of_structure=0.02, frac_of_tumour=0.01),
    ]
    df = _anatomy_df(rows)

    default_threshold = structure_involvement_frequency(df, min_frac_of_structure=0.05)
    row = default_threshold[default_threshold["structure"] == "Frontal_L"].iloc[0]
    assert row["n_cases_involved"] == 2
    assert row["frac_cases_involved"] == pytest.approx(2 / 3)
    assert row["median_frac_of_structure"] == pytest.approx(0.55)
    assert row["median_frac_of_tumour"] == pytest.approx(0.35)

    lower_threshold = structure_involvement_frequency(df, min_frac_of_structure=0.01)
    row_lower = lower_threshold[lower_threshold["structure"] == "Frontal_L"].iloc[0]
    assert row_lower["n_cases_involved"] == 3
    assert row_lower["frac_cases_involved"] == pytest.approx(1.0)
    assert row_lower["median_frac_of_structure"] == pytest.approx(0.5)
    assert row_lower["median_frac_of_tumour"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# 3. Medians are over involved cases only
# --------------------------------------------------------------------------- #


def test_medians_are_over_involved_cases_only() -> None:
    rows = [
        _anatomy_row("case1", "X", frac_of_structure=0.9, frac_of_tumour=0.1),
        _anatomy_row("case2", "X", frac_of_structure=0.9, frac_of_tumour=0.2),
        _anatomy_row("case3", "X", frac_of_structure=0.9, frac_of_tumour=0.3),
        # Below the default 0.05 threshold: must not contribute to the median.
        _anatomy_row("case4", "X", frac_of_structure=0.01, frac_of_tumour=0.99),
    ]
    df = _anatomy_df(rows)

    table = structure_involvement_frequency(df)
    row = table[table["structure"] == "X"].iloc[0]

    assert row["n_cases_involved"] == 3
    # median(0.1, 0.2, 0.3) = 0.2. Wrongly including case4's 0.99 would give
    # median(0.1, 0.2, 0.3, 0.99) = 0.25 instead.
    assert row["median_frac_of_tumour"] == pytest.approx(0.2)
    assert row["median_frac_of_structure"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# 4. Inconstant lobe/eloquence raises, naming the structure
# --------------------------------------------------------------------------- #


def test_inconstant_lobe_raises_naming_the_structure() -> None:
    rows = [
        _anatomy_row("case1", "Y", lobe="frontal", frac_of_structure=0.3),
        _anatomy_row("case2", "Y", lobe="parietal", frac_of_structure=0.3),
    ]
    df = _anatomy_df(rows)

    with pytest.raises(ValueError, match="'Y'"):
        structure_involvement_frequency(df)


def test_inconstant_eloquence_raises_naming_the_structure() -> None:
    rows = [
        _anatomy_row("case1", "Z", eloquence="eloquent", frac_of_structure=0.3),
        _anatomy_row("case2", "Z", eloquence="unclassified", frac_of_structure=0.3),
    ]
    df = _anatomy_df(rows)

    with pytest.raises(ValueError, match="'Z'"):
        structure_involvement_frequency(df)


# --------------------------------------------------------------------------- #
# 5. exclude_unlabelled both ways; lobe_burden_distribution keeps it regardless
# --------------------------------------------------------------------------- #


def test_exclude_unlabelled_both_ways() -> None:
    rows = [
        _anatomy_row("case1", "Frontal_L", frac_of_structure=0.5, frac_of_tumour=0.6),
        _anatomy_row(
            "case1",
            "unlabelled",
            laterality="midline",
            lobe="",
            eloquence="",
            frac_of_structure=float("nan"),
            frac_of_tumour=0.4,
        ),
    ]
    df = _anatomy_df(rows)

    excluded = structure_involvement_frequency(df, exclude_unlabelled=True)
    assert "unlabelled" not in set(excluded["structure"])

    included = structure_involvement_frequency(df, exclude_unlabelled=False)
    assert "unlabelled" in set(included["structure"])
    unlabelled_row = included[included["structure"] == "unlabelled"].iloc[0]
    # frac_of_structure is NaN for the unlabelled row, so it can never clear
    # the involvement threshold -- NaN comparisons are always False.
    assert unlabelled_row["n_cases_involved"] == 0

    lobes = lobe_burden_distribution(df)
    assert "unlabelled" in set(lobes["lobe"])


# --------------------------------------------------------------------------- #
# 6. Lobe shares are NOT renormalised after unlabelled is included
# --------------------------------------------------------------------------- #


def test_lobe_shares_are_not_renormalised() -> None:
    rows = [
        _anatomy_row("case1", "Frontal_L", lobe="frontal", frac_of_tumour=0.7),
        _anatomy_row(
            "case1",
            "unlabelled",
            laterality="midline",
            lobe="",
            eloquence="",
            frac_of_structure=float("nan"),
            frac_of_tumour=0.3,
        ),
    ]
    df = _anatomy_df(rows)

    lobes = lobe_burden_distribution(df)

    frontal_share = lobes.loc[lobes["lobe"] == "frontal", "total_frac_of_tumour"].iloc[0]
    unlabelled_share = lobes.loc[lobes["lobe"] == "unlabelled", "total_frac_of_tumour"].iloc[0]

    # Must report 0.7, not renormalise to 1.0 now that unlabelled is present.
    assert frontal_share == pytest.approx(0.7)
    assert unlabelled_share == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# 7. degenerate_fields lists a saturated rate and only a saturated rate
# --------------------------------------------------------------------------- #


def test_degenerate_fields_lists_saturated_rate() -> None:
    all_near = _summary_df(
        [
            _summary_row(
                "c1", near_eloquent=True, n_eloquent_structures=1, distance_to_eloquent_mm=0.0
            ),
            _summary_row(
                "c2", near_eloquent=True, n_eloquent_structures=1, distance_to_eloquent_mm=0.0
            ),
            _summary_row(
                "c3", near_eloquent=True, n_eloquent_structures=1, distance_to_eloquent_mm=0.0
            ),
        ]
    )
    rates_all_near = eloquence_rates(all_near)
    assert rates_all_near["frac_near_eloquent"] == pytest.approx(1.0)
    assert "frac_near_eloquent" in rates_all_near["degenerate_fields"]

    one_not_near = _summary_df(
        [
            _summary_row(
                "c1", near_eloquent=True, n_eloquent_structures=1, distance_to_eloquent_mm=0.0
            ),
            _summary_row(
                "c2", near_eloquent=True, n_eloquent_structures=1, distance_to_eloquent_mm=0.0
            ),
            _summary_row(
                "c3", near_eloquent=False, n_eloquent_structures=0, distance_to_eloquent_mm=5.0
            ),
        ]
    )
    rates_mixed = eloquence_rates(one_not_near)
    assert rates_mixed["frac_near_eloquent"] == pytest.approx(2 / 3)
    assert "frac_near_eloquent" not in rates_mixed["degenerate_fields"]


# --------------------------------------------------------------------------- #
# 8. Requesting an absent region raises, naming the available regions
# --------------------------------------------------------------------------- #


def test_absent_region_raises_naming_available_regions() -> None:
    rows = [
        _anatomy_row("case1", "Frontal_L", region="ET", frac_of_structure=0.5),
        _anatomy_row("case1", "Frontal_L", region="TC", frac_of_structure=0.5),
    ]
    df = _anatomy_df(rows)

    with pytest.raises(ValueError, match="WT"):
        structure_involvement_frequency(df, region="WT")

    with pytest.raises(ValueError, match="ET"):
        structure_involvement_frequency(df, region="WT")


# --------------------------------------------------------------------------- #
# 9. The region filter is genuinely applied
# --------------------------------------------------------------------------- #


def test_region_filter_matches_pre_filtered_table() -> None:
    rows = [
        _anatomy_row("case1", "Frontal_L", region="ET", frac_of_structure=0.9, frac_of_tumour=0.9),
        _anatomy_row("case1", "Frontal_L", region="WT", frac_of_structure=0.4, frac_of_tumour=0.5),
        _anatomy_row(
            "case2",
            "Parietal_R",
            region="WT",
            lobe="parietal",
            eloquence="eloquent",
            frac_of_structure=0.3,
            frac_of_tumour=0.6,
        ),
        _anatomy_row(
            "case2",
            "Parietal_R",
            region="TC",
            lobe="parietal",
            eloquence="eloquent",
            frac_of_structure=0.1,
            frac_of_tumour=0.2,
        ),
    ]
    multi_region = _anatomy_df(rows)
    wt_only_with_column = multi_region[multi_region["region"] == "WT"].reset_index(drop=True)
    wt_only_no_column = wt_only_with_column.drop(columns=["region"])

    from_multi = structure_involvement_frequency(multi_region, region="WT")
    from_wt_with_column = structure_involvement_frequency(wt_only_with_column, region="WT")
    from_wt_no_column = structure_involvement_frequency(wt_only_no_column, region="WT")

    pd.testing.assert_frame_equal(from_multi, from_wt_with_column)
    pd.testing.assert_frame_equal(from_multi, from_wt_no_column)


# --------------------------------------------------------------------------- #
# 10. An empty frame raises
# --------------------------------------------------------------------------- #


def test_empty_anatomy_frame_raises() -> None:
    empty = pd.DataFrame(
        columns=[
            "case_id",
            "region",
            "structure",
            "laterality",
            "lobe",
            "eloquence",
            "matched_term",
            "n_voxels",
            "volume_mm3",
            "frac_of_tumour",
            "frac_of_structure",
        ]
    )
    with pytest.raises(ValueError):
        structure_involvement_frequency(empty)
    with pytest.raises(ValueError):
        lobe_burden_distribution(empty)
    with pytest.raises(ValueError):
        laterality_distribution(empty)
    with pytest.raises(ValueError):
        summarize_population(empty, _summary_df([_summary_row("c1")]))


def test_empty_summary_frame_raises() -> None:
    empty_summary = pd.DataFrame(
        columns=[
            "case_id",
            "n_structures_involved",
            "top_structure",
            "top_frac_of_structure",
            "most_displaced_structure",
            "frac_unlabelled",
            "n_eloquent_structures",
            "eloquent_frac_of_tumour",
            "distance_to_eloquent_mm",
            "dominant_lobe",
            "coverage_line",
            "near_eloquent",
            "frac_of_tumour_retained",
        ]
    )
    with pytest.raises(ValueError):
        eloquence_rates(empty_summary)


# --------------------------------------------------------------------------- #
# 11. No deep-learning stack, no plotting stack
# --------------------------------------------------------------------------- #


def test_module_does_not_import_plotting_or_deep_learning_stack() -> None:
    """Keeps this module importable with no torch/MONAI/matplotlib/scipy installed.

    Checked against the source text rather than `sys.modules`, because
    pytest has almost certainly already imported some of that stack for
    another test file in the suite -- see
    `test_localize.py::test_localize_module_does_not_import_the_deep_learning_stack`
    and `figures.py`'s equivalent guard against importing torch.
    """
    source = Path(population.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "monai" not in source.lower()
    assert "matplotlib" not in source
    assert "import scipy" not in source
    assert "from scipy" not in source


# --------------------------------------------------------------------------- #
# Additional coverage: laterality_distribution and summarize_population
# --------------------------------------------------------------------------- #


def test_laterality_distribution_computes_mean_median_and_involvement() -> None:
    rows = [
        _anatomy_row("case1", "Frontal_L", laterality="L", frac_of_tumour=0.6),
        _anatomy_row("case1", "Frontal_R", laterality="R", frac_of_tumour=0.4),
        _anatomy_row("case2", "Frontal_L", laterality="L", frac_of_tumour=1.0),
        # case3 has no laterality rows at all in this fixture; still counts
        # toward n_cases via the other cases' presence, and contributes a 0.0
        # to L's and R's per-case series.
        _anatomy_row(
            "case3",
            "unlabelled",
            laterality="midline",
            lobe="",
            eloquence="",
            frac_of_structure=float("nan"),
            frac_of_tumour=1.0,
        ),
    ]
    df = _anatomy_df(rows)

    table = laterality_distribution(df)

    left = table[table["laterality"] == "L"].iloc[0]
    # Per-case L share: case1=0.6, case2=1.0, case3=0.0 (no L row) -> mean 0.5333..
    assert left["mean_frac_of_tumour"] == pytest.approx((0.6 + 1.0 + 0.0) / 3)
    assert left["n_cases_involved"] == 2
    assert left["frac_cases_involved"] == pytest.approx(2 / 3)

    # case3's only row is the `unlabelled` pseudo-structure, which localize.py
    # gives a "midline" placeholder laterality. It is reported under its own
    # `unlabelled` laterality, never as midline -- see
    # test_laterality_reports_unlabelled_separately_and_never_as_midline.
    assert table[table["laterality"] == "midline"].empty
    unlabelled = table[table["laterality"] == "unlabelled"].iloc[0]
    assert unlabelled["n_cases_involved"] == 1
    assert unlabelled["frac_cases_involved"] == pytest.approx(1 / 3)


def test_summarize_population_assembles_all_four() -> None:
    anatomy_rows = [
        _anatomy_row("case1", "Frontal_L", frac_of_structure=0.5, frac_of_tumour=0.6),
        _anatomy_row(
            "case2",
            "Parietal_R",
            lobe="parietal",
            eloquence="eloquent",
            frac_of_structure=0.3,
            frac_of_tumour=0.7,
        ),
    ]
    anatomy_df = _anatomy_df(anatomy_rows)
    summary_df = _summary_df(
        [
            _summary_row("case1", near_eloquent=False, n_eloquent_structures=0),
            _summary_row(
                "case2", near_eloquent=True, n_eloquent_structures=1, distance_to_eloquent_mm=0.0
            ),
        ]
    )

    bundle = summarize_population(anatomy_df, summary_df, region="WT")

    assert bundle["n_cases"] == 2
    assert bundle["region"] == "WT"
    assert isinstance(bundle["structures"], pd.DataFrame)
    assert isinstance(bundle["lobes"], pd.DataFrame)
    assert isinstance(bundle["laterality"], pd.DataFrame)
    assert isinstance(bundle["eloquence"], dict)
    assert bundle["eloquence"]["n_cases"] == 2


def test_laterality_reports_unlabelled_separately_and_never_as_midline() -> None:
    """The unlabelled pseudo-structure carries a `midline` placeholder laterality. Folding it
    in made the midline bucket read 34.8% of tumour volume across all 1251 BraTS 2021 cases,
    against ~0.2% for the actual midline structures -- a table that summed to 1.0, looked
    entirely reasonable, and was wrong by two orders of magnitude."""
    anatomy = pd.DataFrame(
        {
            "case_id": ["c1", "c1", "c1", "c2", "c2"],
            "region": ["WT"] * 5,
            "structure": ["Frontal_L", "CorpusCallosum", "unlabelled", "Frontal_R", "unlabelled"],
            "laterality": ["L", "midline", "midline", "R", "midline"],
            "lobe": ["frontal", "callosum", "unlabelled", "frontal", "unlabelled"],
            "eloquence": ["unclassified"] * 5,
            "frac_of_tumour": [0.6, 0.05, 0.35, 0.7, 0.3],
            "frac_of_structure": [0.5, 0.1, float("nan"), 0.5, float("nan")],
        }
    )

    table = laterality_distribution(anatomy).set_index("laterality")

    assert "unlabelled" in table.index
    # The genuine midline structure contributes 0.05 in one case of two.
    assert table.loc["midline", "mean_frac_of_tumour"] == pytest.approx(0.025)
    assert table.loc["unlabelled", "mean_frac_of_tumour"] == pytest.approx(0.325)
    # Nothing is dropped or renormalised: the four rows still account for the whole tumour.
    assert table["mean_frac_of_tumour"].sum() == pytest.approx(1.0)
