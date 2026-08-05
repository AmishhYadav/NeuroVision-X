"""Tests for `neurovision.analysis.statistics`.

CPU only, tiny hand-built / analytically-derived arrays, whole file well
under 5 seconds. Numeric expectations are hand-computed in the test itself
wherever the spec calls for it; a handful of cases can only reasonably be
pinned by running the code once and asserting the documented CONVENTION
(e.g. "an identical-differences bootstrap collapses to a point interval"),
and those are called out in a comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurovision.analysis.statistics import (
    DEFAULT_ALPHA,
    EffectSize,
    WilcoxonResult,
    compare_models,
    format_comparison,
    holm_bonferroni,
    load_per_case,
    metric_direction,
    paired_bootstrap_ci,
    paired_effect_size,
    wilcoxon_signed_rank,
)

# ---------------------------------------------------------------------------
# 1. holm_bonferroni
# ---------------------------------------------------------------------------


def test_holm_bonferroni_hand_computed() -> None:
    adjusted, reject = holm_bonferroni([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    np.testing.assert_array_equal(reject, [True, False, False])


def test_holm_bonferroni_enforces_monotonicity() -> None:
    # sorted ascending: 0.005, 0.2, 0.21 -> raw adjusted 0.015, 0.4, 0.21
    # (multipliers 3, 2, 1). Without the running-max fix, the largest raw
    # p-value (0.21) would adjust to LESS than the middle one (0.4) -- an
    # adjusted p-value that is not monotone in the raw p-value, which is
    # nonsensical for a step-down procedure.
    adjusted, _ = holm_bonferroni([0.005, 0.2, 0.21])
    np.testing.assert_allclose(adjusted, [0.015, 0.4, 0.4])
    assert adjusted[2] >= adjusted[1]


def test_holm_bonferroni_preserves_input_order() -> None:
    # Same p-values as the first test, permuted -- adjusted values must
    # follow the permutation, not the sorted order.
    adjusted, reject = holm_bonferroni([0.04, 0.01, 0.03])
    np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])
    np.testing.assert_array_equal(reject, [False, True, False])


def test_holm_bonferroni_single_pvalue_unchanged() -> None:
    adjusted, reject = holm_bonferroni([0.03])
    np.testing.assert_allclose(adjusted, [0.03])
    assert reject[0] == (0.03 <= DEFAULT_ALPHA)


def test_holm_bonferroni_raises_on_bad_input() -> None:
    with pytest.raises(ValueError):
        holm_bonferroni([])
    with pytest.raises(ValueError):
        holm_bonferroni([0.5, 1.5])
    with pytest.raises(ValueError):
        holm_bonferroni([0.5, float("nan")])


# ---------------------------------------------------------------------------
# 2 & 3. paired_bootstrap_ci: identical arrays / constant offset
# ---------------------------------------------------------------------------


def test_bootstrap_ci_identical_arrays_collapses_to_zero() -> None:
    a = np.array([0.8, 0.85, 0.9, 0.75, 0.95])
    gen = np.random.default_rng(0)
    result = paired_bootstrap_ci(a, a, n_boot=500, generator=gen)
    assert result.point == 0.0
    assert result.lo == 0.0
    assert result.hi == 0.0
    assert result.contains_zero is True


def test_bootstrap_ci_constant_offset_is_tight_and_excludes_zero() -> None:
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65])
    b = a + 0.1
    gen = np.random.default_rng(0)
    result = paired_bootstrap_ci(a, b, n_boot=1000, generator=gen)
    # diff = a - b is EXACTLY -0.1 for every case, so every bootstrap
    # replicate (a resample of a constant array) is also exactly -0.1.
    assert result.point == pytest.approx(-0.1)
    assert result.lo == pytest.approx(-0.1)
    assert result.hi == pytest.approx(-0.1)
    assert result.contains_zero is False


# ---------------------------------------------------------------------------
# 4. Reproducibility
# ---------------------------------------------------------------------------


def test_bootstrap_ci_reproducible_with_same_seed() -> None:
    a = np.array([0.7, 0.8, 0.6, 0.9, 0.75, 0.85, 0.65, 0.95, 0.55, 0.72])
    b = np.array([0.6, 0.7, 0.65, 0.8, 0.7, 0.9, 0.6, 0.85, 0.5, 0.68])

    r1 = paired_bootstrap_ci(a, b, n_boot=500, generator=np.random.default_rng(0))
    r2 = paired_bootstrap_ci(a, b, n_boot=500, generator=np.random.default_rng(0))
    assert r1.lo == r2.lo
    assert r1.hi == r2.hi

    r3 = paired_bootstrap_ci(a, b, n_boot=500, generator=np.random.default_rng(123))
    # A different seed gives a different draw ...
    assert (r3.lo, r3.hi) != (r1.lo, r1.hi)
    # ... but a "close" interval, since it is the same underlying data.
    assert abs(r3.lo - r1.lo) < 0.05
    assert abs(r3.hi - r1.hi) < 0.05


def test_bootstrap_ci_requires_explicit_generator() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1, 0.2], [0.3, 0.4], generator=None)


# ---------------------------------------------------------------------------
# 5. Pairing is preserved (the anti-independent-resampling test)
# ---------------------------------------------------------------------------


def test_bootstrap_ci_preserves_pairing_despite_wide_raw_magnitudes() -> None:
    # Per-case differences are all exactly +0.05, but the raw values swing
    # from 0.5 to 200 across cases. An UNPAIRED bootstrap (resampling `a`
    # and `b` independently) would produce a wide interval dominated by
    # that magnitude spread; a correctly PAIRED bootstrap only ever sees
    # the constant +0.05 difference and must stay tight around it.
    a = np.array([1.0, 50.0, 3.0, 200.0, 0.5, 75.0, 12.0])
    b = a - 0.05

    gen = np.random.default_rng(0)
    result = paired_bootstrap_ci(a, b, n_boot=1000, generator=gen)

    assert result.point == pytest.approx(0.05)
    assert (result.hi - result.lo) < 0.01


# ---------------------------------------------------------------------------
# 6. BCa
# ---------------------------------------------------------------------------


def test_bca_runs_and_is_close_to_percentile_on_symmetric_data() -> None:
    rng_data = np.random.default_rng(1)
    a = rng_data.normal(0.7, 0.05, size=30)
    diff = rng_data.normal(0.05, 0.02, size=30)
    b = a - diff

    bca = paired_bootstrap_ci(
        a, b, n_boot=2000, ci=0.95, generator=np.random.default_rng(2), method="bca"
    )
    percentile = paired_bootstrap_ci(
        a, b, n_boot=2000, ci=0.95, generator=np.random.default_rng(2), method="percentile"
    )

    assert bca.method == "bca"
    assert np.isfinite(bca.lo)
    assert np.isfinite(bca.hi)
    assert abs(bca.lo - percentile.lo) < 0.03
    assert abs(bca.hi - percentile.hi) < 0.03


def test_bca_degenerate_all_identical_falls_back_without_raising() -> None:
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    gen = np.random.default_rng(0)
    result = paired_bootstrap_ci(a, a, n_boot=300, generator=gen, method="bca")
    assert result.method == "percentile (bca degenerate)"
    assert result.lo == 0.0
    assert result.hi == 0.0


def test_paired_bootstrap_ci_rejects_bad_statistic_and_method() -> None:
    a, b = np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])
    gen = np.random.default_rng(0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b, generator=gen, statistic="mode")
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b, generator=gen, method="jackknife")


# ---------------------------------------------------------------------------
# 7. wilcoxon_signed_rank
# ---------------------------------------------------------------------------


def test_wilcoxon_all_positive_differences() -> None:
    a = np.array([0.9, 0.85, 0.8, 0.95, 0.99])
    b = a - 0.05
    result = wilcoxon_signed_rank(a, b)
    assert isinstance(result, WilcoxonResult)
    assert result.rank_biserial == pytest.approx(1.0)
    assert result.pvalue < 0.1


def test_wilcoxon_all_zero_differences_does_not_raise() -> None:
    a = np.array([0.5, 0.6, 0.7, 0.8])
    result = wilcoxon_signed_rank(a, a)
    assert result.pvalue == 1.0
    assert result.statistic == 0.0
    assert result.rank_biserial == 0.0


def test_wilcoxon_n_zero_counted_correctly() -> None:
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    b = np.array([0.5, 0.6, 0.9, 0.8, 0.7])
    # diff = [0, 0, -0.2, 0, 0.2] -> 3 exact ties
    result = wilcoxon_signed_rank(a, b)
    assert result.n == 5
    assert result.n_zero == 3
    assert result.n_effective == 2  # zero_method="wilcox" drops the ties


def test_wilcoxon_sign_convention_flips_with_swap() -> None:
    a = np.array([0.9, 0.85, 0.8, 0.95, 0.99])
    b = a - 0.05
    forward = wilcoxon_signed_rank(a, b)
    backward = wilcoxon_signed_rank(b, a)
    assert forward.rank_biserial == pytest.approx(1.0)
    assert backward.rank_biserial == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 8. paired_effect_size
# ---------------------------------------------------------------------------


def test_effect_size_hand_computed() -> None:
    diffs = np.array([1.0, 2.0, 3.0, 4.0])
    zeros = np.zeros_like(diffs)
    result = paired_effect_size(diffs, zeros)
    assert isinstance(result, EffectSize)
    assert result.mean_diff == pytest.approx(2.5)
    assert result.sd_diff == pytest.approx(1.2909944, abs=1e-6)
    assert result.cohens_dz == pytest.approx(1.936492, abs=1e-5)
    assert result.hedges_g < result.cohens_dz
    assert result.hedges_g == pytest.approx(1.936492 * (1 - 3 / 11), abs=1e-5)


def _diff_pair_with_exact_dz(target_dz: float) -> np.ndarray:
    """Two-point diff array whose Cohen's dz is exactly `target_dz`.

    For n=2, sample std (ddof=1) of [mean + g, mean - g] is `g * sqrt(2)`.
    Fixing `g = sqrt(2) / 2` makes that std exactly 1, so
    `dz = mean / std = mean = target_dz`.
    """
    half_gap = np.sqrt(2.0) / 2.0
    return np.array([target_dz + half_gap, target_dz - half_gap])


@pytest.mark.parametrize(
    ("target_dz", "expected_magnitude"),
    [
        (0.1999, "negligible"),
        # Just above each cutoff rather than exactly on it: the two-point
        # construction's dz is only exact to within a couple of float64 ULP
        # (measured -- exactly 0.8 comes back as 0.7999999999999998), so
        # landing exactly on a "<" boundary is not reliable. A few 1e-4
        # margin comfortably clears that ULP-level noise while still
        # pinning that strict "<" is used (0.2000... is NOT "negligible").
        (0.2001, "small"),
        (0.4999, "small"),
        (0.5001, "medium"),
        (0.7999, "medium"),
        (0.8001, "large"),
    ],
)
def test_effect_size_magnitude_boundaries(target_dz: float, expected_magnitude: str) -> None:
    diffs = _diff_pair_with_exact_dz(target_dz)
    result = paired_effect_size(diffs, np.zeros_like(diffs))
    assert result.cohens_dz == pytest.approx(target_dz, abs=1e-9)
    assert result.magnitude == expected_magnitude


def test_effect_size_zero_sd_zero_mean() -> None:
    diffs = np.array([0.0, 0.0, 0.0])
    result = paired_effect_size(diffs, np.zeros_like(diffs))
    assert result.sd_diff == 0.0
    assert result.cohens_dz == 0.0
    assert result.magnitude == "negligible"


def test_effect_size_zero_sd_nonzero_mean() -> None:
    # 0.5 rather than 0.1: 0.1 is not exactly representable in float64, and
    # summing three copies and dividing back down leaves a ~1.7e-17 residual
    # std (measured) instead of an exact 0.0. 0.5 is an exact power of two
    # and round-trips exactly.
    diffs = np.array([0.5, 0.5, 0.5])
    result = paired_effect_size(diffs, np.zeros_like(diffs))
    assert result.sd_diff == 0.0
    assert result.cohens_dz == float("inf")
    assert result.magnitude == "large"

    neg_result = paired_effect_size(-diffs, np.zeros_like(diffs))
    assert neg_result.cohens_dz == float("-inf")


# ---------------------------------------------------------------------------
# 9. NaN handling
# ---------------------------------------------------------------------------


def test_nan_pair_dropped_and_n_reflects_it() -> None:
    a = np.array([0.8, 0.9, np.nan, 0.7, 0.85])
    b = np.array([0.7, 0.85, 0.6, np.nan, 0.8])
    # Two pairs have a NaN on one side (index 2 and index 3) -> 3 usable.
    result = paired_effect_size(a, b)
    assert result.n == 3


def test_compare_models_reports_n_missing() -> None:
    case_ids = [f"case_{i}" for i in range(6)]
    a = pd.DataFrame(
        {"case_id": case_ids, "dice_ET": [0.8, 0.9, np.nan, 0.7, 0.85, 0.6]}
    ).set_index("case_id")
    b = pd.DataFrame(
        {"case_id": case_ids, "dice_ET": [0.7, 0.85, 0.6, np.nan, 0.8, 0.5]}
    ).set_index("case_id")

    table = compare_models(a, b, generator=np.random.default_rng(0), n_boot=300)
    assert table.loc["dice_ET", "n"] == 4
    assert table.loc["dice_ET", "n_missing"] == 2


# ---------------------------------------------------------------------------
# 10. metric_direction
# ---------------------------------------------------------------------------


def test_metric_direction_known_prefixes() -> None:
    assert metric_direction("dice_ET") is True
    assert metric_direction("hd95_WT") is False


def test_metric_direction_unknown_raises() -> None:
    with pytest.raises(ValueError):
        metric_direction("frobnicate_score")


def test_metric_direction_override_wins() -> None:
    assert metric_direction("frobnicate_score", overrides={"frobnicate_score": True}) is True
    # Override wins even against a known prefix.
    assert metric_direction("dice_ET", overrides={"dice_ET": False}) is False


# ---------------------------------------------------------------------------
# 11. compare_models
# ---------------------------------------------------------------------------


def _hd95_tables(n: int = 30, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two case tables where A's hd95_WT is reliably ~5mm LOWER (better) than B's."""
    rng = np.random.default_rng(seed)
    case_ids = [f"case_{i:03d}" for i in range(n)]
    hd95_a = rng.normal(3.5, 0.3, size=n)
    hd95_b = hd95_a + rng.normal(5.0, 0.2, size=n)  # B is worse by ~5mm, consistently
    a = pd.DataFrame({"case_id": case_ids, "hd95_WT": hd95_a}).set_index("case_id")
    b = pd.DataFrame({"case_id": case_ids, "hd95_WT": hd95_b}).set_index("case_id")
    return a, b


def test_compare_models_case_id_alignment_is_order_invariant() -> None:
    a, b = _hd95_tables()
    b_shuffled = b.sample(frac=1.0, random_state=7)  # same rows, different order

    table_unshuffled = compare_models(a, b, generator=np.random.default_rng(0), n_boot=300)
    table_shuffled = compare_models(a, b_shuffled, generator=np.random.default_rng(0), n_boot=300)

    pd.testing.assert_frame_equal(table_unshuffled, table_shuffled)


def test_compare_models_partial_case_overlap_uses_intersection_only(caplog) -> None:
    a, b = _hd95_tables(n=30)
    b_missing_some = b.drop(index=b.index[:5])  # 5 cases only in a
    a_with_extra = a.copy()
    a_with_extra.loc["case_extra"] = {"hd95_WT": 4.0}  # 1 case only in a_with_extra

    with caplog.at_level("WARNING"):
        table = compare_models(
            a_with_extra, b_missing_some, generator=np.random.default_rng(0), n_boot=300
        )
    assert table.loc["hd95_WT", "n"] == 25  # 30 - 5 missing from b, extra case not in b either
    assert any("case sets differ" in r.message for r in caplog.records)


def test_compare_models_hd95_direction_and_verdict() -> None:
    a, b = _hd95_tables()
    table = compare_models(a, b, generator=np.random.default_rng(0), n_boot=500)
    row = table.loc["hd95_WT"]
    assert row["higher_is_better"] == False  # noqa: E712
    assert row["improvement"] > 0  # A has LOWER (better) hd95, so improvement is positive
    assert row["improvement_lo"] <= row["improvement_hi"]
    assert row["verdict"] == "better"


def test_compare_models_identical_tables_are_all_inconclusive() -> None:
    a, _ = _hd95_tables()
    a["dice_ET"] = np.linspace(0.7, 0.95, len(a))
    b = a.copy()

    table = compare_models(a, b, generator=np.random.default_rng(0), n_boot=300)
    assert (table["verdict"] == "inconclusive").all()


def test_compare_models_practical_threshold_gives_negligible() -> None:
    a, b = _hd95_tables()
    table = compare_models(
        a, b, generator=np.random.default_rng(0), n_boot=500, practical_threshold=100.0
    )
    assert table.loc["hd95_WT", "verdict"] == "negligible"


def test_compare_models_column_order() -> None:
    a, b = _hd95_tables()
    table = compare_models(
        a, b, generator=np.random.default_rng(0), n_boot=300, name_a="neurovision", name_b="unet3d"
    )
    expected_columns = [
        "n",
        "n_missing",
        "mean_neurovision",
        "mean_unet3d",
        "mean_diff",
        "improvement",
        "improvement_lo",
        "improvement_hi",
        "ci_lo",
        "ci_hi",
        "higher_is_better",
        "p_wilcoxon",
        "p_holm",
        "reject_holm",
        "n_zero",
        "cohens_dz",
        "hedges_g",
        "rank_biserial",
        "magnitude",
        "verdict",
    ]
    assert list(table.columns) == expected_columns
    assert table.index.name == "metric"


def test_compare_models_raises_on_empty_intersection() -> None:
    a = pd.DataFrame({"case_id": ["c1", "c2"], "dice_ET": [0.8, 0.9]}).set_index("case_id")
    b = pd.DataFrame({"case_id": ["c3", "c4"], "dice_ET": [0.7, 0.6]}).set_index("case_id")
    with pytest.raises(ValueError):
        compare_models(a, b, generator=np.random.default_rng(0))


def test_compare_models_raises_on_missing_requested_metric() -> None:
    a, b = _hd95_tables()
    with pytest.raises(ValueError):
        compare_models(a, b, generator=np.random.default_rng(0), metrics=["dice_ET"])


# ---------------------------------------------------------------------------
# 12. format_comparison
# ---------------------------------------------------------------------------


def test_format_comparison_warns_on_inconclusive() -> None:
    a, _ = _hd95_tables()
    b = a.copy()
    table = compare_models(a, b, generator=np.random.default_rng(0), n_boot=300)
    report = format_comparison(table, name_a="model", name_b="baseline")
    assert "WARNING" in report


def test_format_comparison_no_warning_when_all_conclusive() -> None:
    a, b = _hd95_tables()
    table = compare_models(a, b, generator=np.random.default_rng(0), n_boot=500)
    report = format_comparison(table, name_a="model", name_b="baseline")
    assert "WARNING" not in report
    assert "All metrics are statistically and practically conclusive." in report


# ---------------------------------------------------------------------------
# 13. load_per_case
# ---------------------------------------------------------------------------


def test_load_per_case_round_trip(tmp_path) -> None:
    df = pd.DataFrame(
        {"case_id": ["case_001", "case_002"], "dice_ET": [0.8, 0.9], "hd95_WT": [3.2, 4.1]}
    )
    path = tmp_path / "per_case_metrics.csv"
    df.to_csv(path, index=False)

    loaded = load_per_case(path)
    assert loaded.index.name == "case_id"
    assert list(loaded.index) == ["case_001", "case_002"]
    np.testing.assert_allclose(loaded["dice_ET"].to_numpy(), [0.8, 0.9])


def test_load_per_case_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_per_case(tmp_path / "does_not_exist.csv")
