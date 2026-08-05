"""Tests for `neurovision.uncertainty.risk_coverage`.

CPU only, tiny hand-built arrays/tensors, whole file well under 5 seconds.
Every expected numeric value below is hand-computed in the test itself
(never read back from the implementation), except where the module's own
docstring documents a convention (e.g. the coverage grid starting at `1/N`
rather than `0`) that has to be verified by running the code once and then
pinned -- those cases are called out explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurovision.uncertainty.risk_coverage import (
    RiskCoverageCurve,
    bootstrap_curve_ci,
    case_uncertainty_scalars,
    oracle_curve,
    random_curve,
    referral_table,
    risk_coverage_curve,
    uncertainty_error_correlation,
)

# ---------------------------------------------------------------------------
# 1. Hand-built, perfectly-ranked curve
# ---------------------------------------------------------------------------


def test_risk_coverage_curve_hand_built_perfectly_ranked() -> None:
    score = [0.9, 0.8, 0.5, 0.2]
    uncertainty = [0.1, 0.2, 0.3, 0.4]  # ascending, exactly matches descending score

    curve = risk_coverage_curve(uncertainty, score)

    np.testing.assert_allclose(curve.coverage, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(curve.n_retained, [1, 2, 3, 4])
    np.testing.assert_allclose(curve.performance, [0.9, 0.85, 2.2 / 3.0, 0.6])
    np.testing.assert_allclose(curve.risk, 1.0 - curve.performance)
    assert curve.n_cases == 4
    assert curve.n_dropped == 0


# ---------------------------------------------------------------------------
# 2. Anti-correlated uncertainty: worst cases ranked most confident
# ---------------------------------------------------------------------------


def test_anti_correlated_uncertainty_gives_worst_score_at_low_coverage() -> None:
    # Low uncertainty (most confident) is paired with the WORST score here --
    # the opposite of test 1 -- so performance at the lowest coverage should
    # be the worst score in the set, pinning that the sort direction is not
    # silently flipped somewhere.
    uncertainty = [0.1, 0.2, 0.3, 0.4]
    score = [0.2, 0.5, 0.8, 0.9]

    curve = risk_coverage_curve(uncertainty, score)

    assert curve.performance[0] == pytest.approx(min(score))
    np.testing.assert_allclose(curve.performance, [0.2, 0.35, 0.5, 0.6])


# ---------------------------------------------------------------------------
# 3. oracle_curve invariants on random data
# ---------------------------------------------------------------------------


def test_oracle_curve_is_monotone_and_dominates_model_curve() -> None:
    rng = np.random.default_rng(42)
    n = 60
    uncertainty = rng.normal(size=n)
    score = rng.uniform(0.0, 1.0, size=n)  # uncorrelated with uncertainty -- worst case for model

    model = risk_coverage_curve(uncertainty, score)
    oracle = oracle_curve(score)

    # Oracle performance is non-increasing as more (lower-quality) cases are
    # forced in by increasing coverage.
    assert np.all(np.diff(oracle.performance) <= 1e-12)
    # Oracle performance is the best any referral policy can achieve at each
    # coverage level, so it must be >= the model's at every point.
    assert np.all(oracle.performance >= model.performance - 1e-12)


# ---------------------------------------------------------------------------
# 4. random_curve is flat and its AURC matches the documented formula
# ---------------------------------------------------------------------------


def test_random_curve_is_flat_at_mean_score() -> None:
    score = [0.9, 0.8, 0.5, 0.2]
    curve = random_curve(score)

    mean_score = float(np.mean(score))
    np.testing.assert_allclose(curve.performance, [mean_score] * 4)
    np.testing.assert_allclose(curve.risk, [1.0 - mean_score] * 4)

    # AURC convention: the coverage grid is k/N for k=1..N, i.e. it starts at
    # 1/N (not 0) since coverage 0 retains no cases and has no defined
    # performance. The trapezoidal rule over a CONSTANT integrand equals
    # value * (x_max - x_min) regardless of intermediate point spacing, so
    # for a flat risk curve this is exactly (1 - mean_score) * (1 - 1/N),
    # NOT (1 - mean_score) -- verified by running the implementation before
    # writing this assertion, and documented in random_curve's docstring.
    n = len(score)
    expected_aurc = (1.0 - mean_score) * (1.0 - 1.0 / n)
    assert curve.aurc == pytest.approx(expected_aurc)


# ---------------------------------------------------------------------------
# 5. Perfectly-ranked uncertainty matches the oracle's AURC
# ---------------------------------------------------------------------------


def test_perfectly_ranked_uncertainty_matches_oracle_aurc() -> None:
    score = [0.9, 0.8, 0.5, 0.2]
    uncertainty = [0.1, 0.2, 0.3, 0.4]  # same ranking the oracle would produce

    model = risk_coverage_curve(uncertainty, score)
    oracle = oracle_curve(score)

    assert model.aurc == pytest.approx(oracle.aurc)
    np.testing.assert_allclose(model.performance, oracle.performance)


# ---------------------------------------------------------------------------
# 6. Ties in uncertainty preserve input order (stable sort)
# ---------------------------------------------------------------------------


def test_ties_in_uncertainty_preserve_input_order() -> None:
    uncertainty = [0.5, 0.5]
    score = [0.9, 0.1]  # case 0 has the better score

    curve = risk_coverage_curve(uncertainty, score)

    # A stable sort keeps case 0 first (its input position), so it alone is
    # retained at the lowest coverage level.
    assert curve.performance[0] == pytest.approx(0.9)
    assert curve.performance[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 7. NaN handling
# ---------------------------------------------------------------------------


def test_nan_in_score_and_uncertainty_are_dropped() -> None:
    uncertainty = [0.1, np.nan, 0.3, 0.4]
    score = [0.9, 0.8, np.nan, 0.2]
    # Case 1 has a NaN uncertainty, case 2 has a NaN score -- both dropped,
    # leaving cases 0 and 3.

    curve = risk_coverage_curve(uncertainty, score)

    assert curve.n_dropped == 2
    assert curve.n_cases == 2
    np.testing.assert_allclose(curve.coverage, [0.5, 1.0])
    np.testing.assert_allclose(curve.performance, [0.9, 0.55])


# ---------------------------------------------------------------------------
# 8. Raises
# ---------------------------------------------------------------------------


def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        risk_coverage_curve([0.1, 0.2], [0.9, 0.8, 0.7])


def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        risk_coverage_curve([], [])


def test_raises_on_fewer_than_two_usable_cases() -> None:
    with pytest.raises(ValueError):
        risk_coverage_curve([0.1, np.nan], [0.9, 0.8])


# ---------------------------------------------------------------------------
# 9. higher_is_better=False (HD95-like score)
# ---------------------------------------------------------------------------


def test_higher_is_better_false_risk_is_the_score_itself() -> None:
    # HD95-like: lower is better. Uncertainty correlated with it (more
    # uncertain -> worse/larger HD95).
    score = [1.0, 2.0, 3.0, 4.0]
    uncertainty = [0.1, 0.2, 0.3, 0.4]

    curve = risk_coverage_curve(uncertainty, score, higher_is_better=False)

    # risk == performance (the retained mean score), not 1 - score.
    np.testing.assert_allclose(curve.risk, curve.performance)
    np.testing.assert_allclose(curve.performance, [1.0, 1.5, 2.0, 2.5])

    # Referral (retaining only the most confident/lowest-uncertainty cases)
    # REDUCES risk relative to retaining everything.
    assert curve.risk[0] < curve.risk[-1]


# ---------------------------------------------------------------------------
# 10. bootstrap_curve_ci
# ---------------------------------------------------------------------------


def test_bootstrap_curve_ci_reproducible_and_well_formed() -> None:
    uncertainty = [0.1, 0.4, 0.2, 0.5, 0.3, 0.7, 0.6, 0.9, 0.8, 1.0]
    score = [0.95, 0.6, 0.9, 0.5, 0.85, 0.3, 0.4, 0.2, 0.25, 0.1]

    df1 = bootstrap_curve_ci(uncertainty, score, n_boot=200, generator=np.random.default_rng(7))
    df2 = bootstrap_curve_ci(uncertainty, score, n_boot=200, generator=np.random.default_rng(7))

    np.testing.assert_allclose(df1.index.to_numpy(), df2.index.to_numpy())
    np.testing.assert_allclose(df1["lo"].to_numpy(), df2["lo"].to_numpy())
    np.testing.assert_allclose(df1["hi"].to_numpy(), df2["hi"].to_numpy())

    assert np.all(df1["lo"].to_numpy() <= df1["performance"].to_numpy() + 1e-12)
    assert np.all(df1["performance"].to_numpy() <= df1["hi"].to_numpy() + 1e-12)

    # Index is the coverage grid: k/N for k=1..N.
    n = len(score)
    np.testing.assert_allclose(df1.index.to_numpy(), np.arange(1, n + 1) / n)

    # At coverage 1.0, the band is NOT zero-width: each bootstrap replicate's
    # "full coverage" set is itself a with-replacement resample of all N
    # cases, not the original set, so its mean carries genuine bootstrap
    # sampling variance around the point estimate -- verified by running the
    # implementation before writing this assertion (a naive expectation that
    # resampling "all" cases reproduces the original mean exactly is wrong).
    width_at_full_coverage = df1.iloc[-1]["hi"] - df1.iloc[-1]["lo"]
    assert width_at_full_coverage > 0.0


def test_bootstrap_curve_ci_requires_explicit_generator() -> None:
    with pytest.raises(ValueError):
        bootstrap_curve_ci([0.1, 0.2, 0.3], [0.9, 0.8, 0.7], generator=None)


# ---------------------------------------------------------------------------
# 11. referral_table
# ---------------------------------------------------------------------------


def test_referral_table_snaps_coverage_and_matches_at_full_coverage() -> None:
    rng = np.random.default_rng(3)
    n = 10
    uncertainty = rng.normal(size=n)
    score = rng.uniform(0.0, 1.0, size=n)

    table = referral_table(uncertainty, score, coverage_points=(1.0, 0.9))

    row_90 = table.loc[0.9]
    assert row_90["n_retained"] == 9  # round(0.9 * 10)
    assert row_90["coverage"] == pytest.approx(0.9)
    assert row_90["gain_over_random"] == pytest.approx(row_90["model"] - row_90["random"])

    row_100 = table.loc[1.0]
    assert row_100["model"] == pytest.approx(row_100["oracle"])
    assert row_100["model"] == pytest.approx(row_100["random"])


def test_referral_table_raises_on_out_of_range_coverage_point() -> None:
    with pytest.raises(ValueError):
        referral_table([0.1, 0.2, 0.3], [0.9, 0.8, 0.7], coverage_points=(1.5,))


# ---------------------------------------------------------------------------
# 12. uncertainty_error_correlation
# ---------------------------------------------------------------------------


def test_uncertainty_error_correlation_spearman_vs_pearson() -> None:
    uncertainty = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    score = -(uncertainty**3)  # perfectly monotone decreasing, but nonlinear

    result = uncertainty_error_correlation(uncertainty, score)

    assert result["spearman"] == pytest.approx(-1.0)
    assert result["pearson"] != pytest.approx(-1.0)
    assert result["n"] == 5


# ---------------------------------------------------------------------------
# 13 & 14. case_uncertainty_scalars
# ---------------------------------------------------------------------------


def _build_mi_pe() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds a (3, 2, 2, 2) mutual_information/predictive_entropy/mask trio.

    Region 0 (ET): all voxels = 1.0 (mi), 2.0 (pe); mask selects 2 of 8
    voxels -> mean is still 1.0 / 2.0 (constant field).
    Region 1 (TC): all voxels = 2.0 (mi), 4.0 (pe); mask selects all 8.
    Region 2 (WT): all voxels = 0.5 (mi), 1.0 (pe); mask selects NONE.
    """
    mi = torch.zeros((3, 2, 2, 2))
    mi[0] = 1.0
    mi[1] = 2.0
    mi[2] = 0.5
    pe = mi * 2.0

    mask = torch.zeros((3, 2, 2, 2), dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    mask[0, 1, 1, 1] = True
    mask[1, :, :, :] = True
    # mask[2] left all False
    return mi, pe, mask


def test_case_uncertainty_scalars_hand_computed() -> None:
    mi, pe, mask = _build_mi_pe()

    scalars = case_uncertainty_scalars(mi, pe, mask=mask)

    assert scalars["unc_mi_ET"] == pytest.approx(1.0)
    assert scalars["unc_pe_ET"] == pytest.approx(2.0)
    assert scalars["unc_mi_TC"] == pytest.approx(2.0)
    assert scalars["unc_pe_TC"] == pytest.approx(4.0)
    assert np.isnan(scalars["unc_mi_WT"])
    assert np.isnan(scalars["unc_pe_WT"])

    # NaN-skipping mean over the three regions.
    assert scalars["unc_mi_mean"] == pytest.approx((1.0 + 2.0) / 2.0)
    assert scalars["unc_pe_mean"] == pytest.approx((2.0 + 4.0) / 2.0)


def test_case_uncertainty_scalars_wrong_channel_count_raises() -> None:
    mi = torch.zeros((2, 2, 2, 2))
    pe = torch.zeros((2, 2, 2, 2))
    with pytest.raises(ValueError):
        case_uncertainty_scalars(mi, pe)


def test_case_uncertainty_scalars_accepts_batch_axis() -> None:
    mi, pe, mask = _build_mi_pe()
    mi_batched = mi.unsqueeze(0)  # (1, C, D, H, W)
    pe_batched = pe.unsqueeze(0)
    mask_batched = mask.unsqueeze(0)

    scalars_unbatched = case_uncertainty_scalars(mi, pe, mask=mask)
    scalars_batched = case_uncertainty_scalars(mi_batched, pe_batched, mask=mask_batched)

    assert scalars_unbatched.keys() == scalars_batched.keys()
    for key, value in scalars_unbatched.items():
        if np.isnan(value):
            assert np.isnan(scalars_batched[key])
        else:
            assert scalars_batched[key] == pytest.approx(value)


# ---------------------------------------------------------------------------
# RiskCoverageCurve dataclass sanity
# ---------------------------------------------------------------------------


def test_risk_coverage_curve_dataclass_fields() -> None:
    curve = risk_coverage_curve([0.1, 0.2, 0.3], [0.9, 0.8, 0.7])
    assert isinstance(curve, RiskCoverageCurve)
    assert isinstance(curve.aurc, float)
    assert curve.coverage.shape == curve.performance.shape == curve.risk.shape
