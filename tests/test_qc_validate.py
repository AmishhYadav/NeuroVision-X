"""Tests for `neurovision.analysis.qc_validate` -- the Gate C statistics module.

All synthetic, CPU only, whole file well under a few seconds. `n_boot` is
kept small (200, except test 6 which needs more replicates to reliably show
the paired-vs-unpaired width difference and says so in its own docstring).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neurovision.analysis.detection import auroc
from neurovision.analysis.qc_validate import (
    CellEndpoints,
    cell_endpoints,
    endpoints_table,
    falsification_check,
    gate_c_verdict,
    mark_family,
    silent_failure_table,
)

REGIONS = ("ET", "TC", "WT")


# ---------------------------------------------------------------------------
# falsification_check
# ---------------------------------------------------------------------------


def _dice_frame(prefix: str, case_ids: list[str], values: dict[str, np.ndarray]) -> pd.DataFrame:
    """Builds a `case_id`-indexed frame with `<prefix>_<region>` columns."""
    data = {f"{prefix}_{region}": values[region] for region in REGIONS}
    return pd.DataFrame(data, index=pd.Index(case_ids, name="case_id"))


def test_falsification_check_passes_on_identical_frames() -> None:
    case_ids = [f"case_{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    values = {region: rng.uniform(0.0, 1.0, size=10) for region in REGIONS}

    table = _dice_frame("true_dice", case_ids, values)
    published = _dice_frame("dice", case_ids, values)

    result = falsification_check(table, published, REGIONS, tol=0.01)

    assert list(result["region"]) == list(REGIONS)
    for _, row in result.iterrows():
        assert row["median_abs_diff"] == pytest.approx(0.0)
        assert row["max_abs_diff"] == pytest.approx(0.0)
        assert row["n_over_tol"] == 0
        assert row["n"] == 10


def test_falsification_check_raises_when_median_exceeds_tol() -> None:
    case_ids = [f"case_{i}" for i in range(10)]
    rng = np.random.default_rng(1)
    values = {region: rng.uniform(0.0, 1.0, size=10) for region in REGIONS}

    table = _dice_frame("true_dice", case_ids, values)
    published_values = {region: arr.copy() for region, arr in values.items()}
    # Perturb TC by 0.05 in half the cases -- enough to push the median
    # absolute difference above tol=0.01.
    published_values["TC"][:5] += 0.05
    published = _dice_frame("dice", case_ids, published_values)

    with pytest.raises(ValueError, match="TC"):
        falsification_check(table, published, REGIONS, tol=0.01)


def test_falsification_check_raises_on_disjoint_case_ids() -> None:
    values = {region: np.linspace(0.0, 1.0, 5) for region in REGIONS}
    table = _dice_frame("true_dice", [f"a_{i}" for i in range(5)], values)
    published = _dice_frame("dice", [f"b_{i}" for i in range(5)], values)

    with pytest.raises(ValueError):
        falsification_check(table, published, REGIONS, tol=0.01)


# ---------------------------------------------------------------------------
# cell_endpoints
# ---------------------------------------------------------------------------


def test_cell_endpoints_perfect_qc_gives_auroc_one() -> None:
    n = 20
    true_dice = np.linspace(0.3, 1.0, n)
    qc_pred = true_dice.copy()  # perfect predictor
    rng = np.random.default_rng(2)
    entropy = rng.uniform(0.0, 1.0, size=n)

    result = cell_endpoints(
        true_dice,
        qc_pred,
        entropy,
        cohort="test",
        region="ET",
        bad_dice_threshold=0.7,
        n_boot=200,
        ci=0.95,
        seed=42,
    )

    assert result.auroc_qc == pytest.approx(1.0)
    assert result.n == n
    assert result.n_positive == int(np.sum(true_dice < 0.7))


def test_cell_endpoints_uninformative_qc_gives_auroc_near_half() -> None:
    n = 20
    true_dice = np.linspace(0.3, 1.0, n)
    qc_pred = np.full(n, 0.5)  # constant score
    rng = np.random.default_rng(3)
    entropy = rng.uniform(0.0, 1.0, size=n)

    # Verify what this project's own auroc() does with a constant score,
    # rather than assuming: scipy's rankdata gives every tied entry the
    # AVERAGE rank, so the Mann-Whitney U identity collapses to exactly 0.5
    # (not NaN) for a fully-constant score.
    positive = true_dice < 0.7
    direct = auroc(-qc_pred, positive)
    assert direct == pytest.approx(0.5)

    result = cell_endpoints(
        true_dice,
        qc_pred,
        entropy,
        cohort="test",
        region="ET",
        bad_dice_threshold=0.7,
        n_boot=200,
        ci=0.95,
        seed=42,
    )
    assert result.auroc_qc == pytest.approx(0.5)


def test_cell_endpoints_bootstrap_is_paired() -> None:
    """The interval-correctness test: paired CI on delta_auroc must be narrower than unpaired.

    Both `qc_pred` and `entropy` are built to track `true_dice` (with
    independent noise), so they are strongly correlated with each other
    across cases. A PAIRED bootstrap -- the same resampled case set feeding
    both AUROCs in every replicate -- preserves that shared sampling
    variation and yields a narrower interval on their difference than an
    UNPAIRED bootstrap that resamples the two sides independently. `n_boot`
    is raised to 3000 for this test specifically because the effect is real
    but not always visible at the smaller `n_boot=200` used elsewhere.
    """
    rng = np.random.default_rng(123)
    n = 60
    true_dice = np.sort(rng.uniform(0.3, 1.0, size=n))
    # Enough noise that each AUROC has real bootstrap variance (a
    # near-perfect signal saturates AUROC near 1.0 on almost every
    # resample, leaving too little variance for the pairing effect to
    # show up at all) but the two scores still both track true_dice, so
    # they stay strongly correlated with each other across cases.
    qc_pred = true_dice + rng.normal(0.0, 0.12, size=n)
    entropy = (1.0 - true_dice) + rng.normal(0.0, 0.12, size=n)

    n_boot = 4000
    seed = 7
    result = cell_endpoints(
        true_dice,
        qc_pred,
        entropy,
        cohort="test",
        region="ET",
        bad_dice_threshold=0.7,
        n_boot=n_boot,
        ci=0.95,
        seed=seed,
    )
    paired_width = result.delta_ci_hi - result.delta_ci_lo

    # Unpaired: independent index draws for the QC side and the entropy
    # side, computed with a generator seeded the same way cell_endpoints
    # seeds its own.
    generator = np.random.default_rng(seed)
    idx_qc = generator.integers(0, n, size=(n_boot, n))
    idx_ent = generator.integers(0, n, size=(n_boot, n))
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        true_qc_b = true_dice[idx_qc[b]]
        auroc_qc_b = auroc(-qc_pred[idx_qc[b]], true_qc_b < 0.7)

        true_ent_b = true_dice[idx_ent[b]]
        auroc_ent_b = auroc(entropy[idx_ent[b]], true_ent_b < 0.7)

        deltas[b] = auroc_qc_b - auroc_ent_b

    valid = deltas[np.isfinite(deltas)]
    unpaired_lo, unpaired_hi = np.percentile(valid, [2.5, 97.5])
    unpaired_width = unpaired_hi - unpaired_lo

    assert paired_width < unpaired_width


def test_cell_endpoints_zero_positives_returns_nan_not_raise() -> None:
    n = 15
    true_dice = np.linspace(0.8, 1.0, n)  # every case is "good"
    rng = np.random.default_rng(4)
    qc_pred = rng.uniform(0.7, 1.0, size=n)
    entropy = rng.uniform(0.0, 1.0, size=n)

    result = cell_endpoints(
        true_dice,
        qc_pred,
        entropy,
        cohort="ssa",
        region="WT",
        bad_dice_threshold=0.7,
        n_boot=200,
        ci=0.95,
        seed=42,
    )

    assert result.n == n
    assert result.n_positive == 0
    assert np.isnan(result.auroc_qc)
    assert np.isnan(result.auroc_ent)
    assert np.isnan(result.delta_auroc)
    assert np.isnan(result.p_bootstrap)


def test_cell_endpoints_all_nan_entropy_returns_nan_not_raise() -> None:
    n = 10
    true_dice = np.linspace(0.3, 1.0, n)
    qc_pred = np.linspace(0.3, 1.0, n)
    entropy = np.full(n, np.nan)  # every case's predicted foreground was empty

    result = cell_endpoints(
        true_dice,
        qc_pred,
        entropy,
        cohort="ped",
        region="WT",
        bad_dice_threshold=0.7,
        n_boot=200,
        ci=0.95,
        seed=42,
    )

    assert result.n == 0
    assert result.n_positive == 0
    assert np.isnan(result.auroc_qc)
    assert np.isnan(result.delta_auroc)
    assert np.isnan(result.delta_ci_lo)
    assert np.isnan(result.p_bootstrap)
    assert result.n_valid_replicates == 0


def test_cell_endpoints_p_value_is_never_exactly_zero() -> None:
    true_dice = np.concatenate([np.full(20, 0.95), np.full(20, 0.3)])
    # Perfect, extreme separation.
    qc_pred = np.concatenate([np.full(20, 0.95), np.full(20, 0.1)])
    entropy = np.concatenate([np.full(20, 0.05), np.full(20, 0.9)])

    n_boot = 200
    result = cell_endpoints(
        true_dice,
        qc_pred,
        entropy,
        cohort="test",
        region="ET",
        bad_dice_threshold=0.7,
        n_boot=n_boot,
        ci=0.95,
        seed=42,
    )

    assert result.n_valid_replicates > 0
    assert result.p_bootstrap >= 1.0 / result.n_valid_replicates
    assert result.p_bootstrap > 0.0


def test_cell_endpoints_bias_sign() -> None:
    n = 15
    true_dice = np.linspace(0.3, 1.0, n)
    rng = np.random.default_rng(5)
    entropy = rng.uniform(0.0, 1.0, size=n)

    over = cell_endpoints(
        true_dice,
        true_dice + 0.1,
        entropy,
        cohort="test",
        region="ET",
        bad_dice_threshold=0.7,
        n_boot=200,
        ci=0.95,
        seed=42,
    )
    assert over.bias == pytest.approx(0.1)

    under = cell_endpoints(
        true_dice,
        true_dice - 0.1,
        entropy,
        cohort="test",
        region="ET",
        bad_dice_threshold=0.7,
        n_boot=200,
        ci=0.95,
        seed=42,
    )
    assert under.bias == pytest.approx(-0.1)


def test_cell_endpoints_is_deterministic_under_seed() -> None:
    n = 25
    rng = np.random.default_rng(6)
    true_dice = rng.uniform(0.2, 1.0, size=n)
    qc_pred = true_dice + rng.normal(0.0, 0.05, size=n)
    entropy = rng.uniform(0.0, 1.0, size=n)

    kwargs = dict(
        true_dice=true_dice,
        qc_pred=qc_pred,
        entropy=entropy,
        cohort="test",
        region="ET",
        bad_dice_threshold=0.7,
        n_boot=200,
        ci=0.95,
    )

    result_a = cell_endpoints(seed=42, **kwargs)
    result_b = cell_endpoints(seed=42, **kwargs)
    assert result_a == result_b

    result_c = cell_endpoints(seed=99, **kwargs)
    assert (
        result_c.delta_ci_lo != result_a.delta_ci_lo or result_c.delta_ci_hi != result_a.delta_ci_hi
    )


# ---------------------------------------------------------------------------
# mark_family / gate_c_verdict / silent_failure_table
# ---------------------------------------------------------------------------


def _make_cell(cohort: str, region: str, n_positive: int, **overrides: object) -> CellEndpoints:
    """Builds a `CellEndpoints` with sensible defaults, overridable per test."""
    defaults: dict[str, object] = dict(
        cohort=cohort,
        region=region,
        n=100,
        n_positive=n_positive,
        auroc_qc=0.7,
        auroc_ent=0.6,
        delta_auroc=0.1,
        delta_ci_lo=0.02,
        delta_ci_hi=0.18,
        p_bootstrap=0.01,
        spearman_qc=0.5,
        spearman_ci_lo=0.3,
        spearman_ci_hi=0.7,
        spearman_ent=-0.3,
        mae=0.1,
        bias=0.0,
        n_valid_replicates=200,
    )
    defaults.update(overrides)
    return CellEndpoints(**defaults)  # type: ignore[arg-type]


# Positive counts from preregistration_qc.md's table.
_POSITIVE_COUNTS = {
    ("test", "ET"): 17,
    ("test", "TC"): 11,
    ("test", "WT"): 1,
    ("ssa", "ET"): 10,
    ("ssa", "TC"): 12,
    ("ssa", "WT"): 3,
    ("ped", "ET"): 45,
    ("ped", "TC"): 71,
    ("ped", "WT"): 11,
}


def _preregistration_table() -> pd.DataFrame:
    cells = [
        _make_cell(cohort, region, n_positive)
        for (cohort, region), n_positive in _POSITIVE_COUNTS.items()
    ]
    return endpoints_table(cells)


def test_mark_family_excludes_in_distribution_and_low_positive_cells() -> None:
    table = _preregistration_table()
    marked = mark_family(table, in_distribution_cohort="test", min_positives=5)

    expected_family = {
        ("ssa", "ET"),
        ("ssa", "TC"),
        ("ped", "ET"),
        ("ped", "TC"),
        ("ped", "WT"),
    }
    actual_family = {
        (row["cohort"], row["region"]) for _, row in marked.iterrows() if row["in_family"]
    }
    assert actual_family == expected_family
    assert len(marked) == 9


def test_mark_family_holm_corrects_only_the_family() -> None:
    table = _preregistration_table()
    marked = mark_family(table, in_distribution_cohort="test", min_positives=5)

    for _, row in marked.iterrows():
        if row["in_family"]:
            assert not np.isnan(row["p_holm"])
            assert row["p_holm"] >= row["p_bootstrap"]
        else:
            assert np.isnan(row["p_holm"])


# ---------------------------------------------------------------------------
# gate_c_verdict
# ---------------------------------------------------------------------------


def _family_table(**cell_overrides: object) -> pd.DataFrame:
    """One in_family row plus one non-family row, for gate_c_verdict tests."""
    firing_defaults: dict[str, object] = dict(
        cohort="ssa",
        region="ET",
        n_positive=10,
        delta_auroc=0.1,
        delta_ci_lo=0.02,
        delta_ci_hi=0.18,
        p_bootstrap=0.001,
    )
    firing_defaults.update(cell_overrides)
    cell = _make_cell(**firing_defaults)  # type: ignore[arg-type]
    table = endpoints_table([cell])
    return mark_family(table, in_distribution_cohort="test", min_positives=5)


@pytest.mark.parametrize(
    ("overrides", "expected_verdict"),
    [
        ({}, "POSITIVE"),
        ({"delta_auroc": -0.05, "delta_ci_lo": -0.1, "delta_ci_hi": 0.05}, "NEGATIVE"),
        ({"p_bootstrap": 0.9}, "NEGATIVE"),
        ({"delta_ci_lo": -0.01}, "NEGATIVE"),
    ],
)
def test_gate_c_verdict_positive_requires_all_three_conditions(
    overrides: dict[str, object], expected_verdict: str
) -> None:
    table = _family_table(**overrides)
    verdict = gate_c_verdict(table, alpha=0.05)
    assert verdict["verdict"] == expected_verdict


def test_gate_c_verdict_is_json_serialisable() -> None:
    table = _family_table()
    verdict = gate_c_verdict(table, alpha=0.05)
    serialised = json.dumps(verdict)
    assert "POSITIVE" in serialised


def test_gate_c_verdict_empty_family_is_negative() -> None:
    table = _preregistration_table()
    # min_positives so high that nothing qualifies -- an empty family.
    marked = mark_family(table, in_distribution_cohort="test", min_positives=1000)
    verdict = gate_c_verdict(marked, alpha=0.05)

    assert verdict["verdict"] == "NEGATIVE"
    assert verdict["family_size"] == 0
    assert verdict["firing_cells"] == []
    json.dumps(verdict)  # still serialisable with an empty family


# ---------------------------------------------------------------------------
# silent_failure_table
# ---------------------------------------------------------------------------


def test_silent_failure_table_deltas_and_direction_flag() -> None:
    cells = [
        _make_cell("test", "ET", n_positive=17, spearman_qc=0.8, bias=0.0),
        _make_cell("ped", "ET", n_positive=45, spearman_qc=0.5, bias=0.2),
        _make_cell("ssa", "ET", n_positive=10, spearman_qc=0.6, bias=-0.1),
    ]
    table = endpoints_table(cells)

    result = silent_failure_table(table, in_distribution_cohort="test")
    by_cohort = result.set_index("cohort")

    in_dist_row = by_cohort.loc["test"]
    assert np.isnan(in_dist_row["delta_spearman_vs_in_distribution"])
    assert np.isnan(in_dist_row["delta_bias_vs_in_distribution"])
    assert bool(in_dist_row["bias_more_positive_than_in_distribution"]) is False

    ped_row = by_cohort.loc["ped"]
    assert ped_row["delta_spearman_vs_in_distribution"] == pytest.approx(0.5 - 0.8)
    assert ped_row["delta_bias_vs_in_distribution"] == pytest.approx(0.2 - 0.0)
    assert bool(ped_row["bias_more_positive_than_in_distribution"]) is True

    ssa_row = by_cohort.loc["ssa"]
    assert ssa_row["delta_bias_vs_in_distribution"] == pytest.approx(-0.1 - 0.0)
    assert bool(ssa_row["bias_more_positive_than_in_distribution"]) is False
