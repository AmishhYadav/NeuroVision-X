"""Tests for scripts/gate2_localisation.py.

Loaded by path, the same way tests/test_detection_stats.py loads the Gate 1
driver -- scripts/ is not a package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neurovision.analysis.statistics import paired_bootstrap_ci

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gate2_localisation.py"
_spec = importlib.util.spec_from_file_location("gate2_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
gate2_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["gate2_script"] = gate2_script
_spec.loader.exec_module(gate2_script)

build_verdict = gate2_script.build_verdict
fit_both_arms = gate2_script.fit_both_arms
score_cohort = gate2_script.score_cohort
_paired_bootstrap = gate2_script._paired_bootstrap
collect_cohort_samples = gate2_script.collect_cohort_samples


# ---------------------------------------------------------------------------
# The bootstrap must agree with the project's canonical implementation
# ---------------------------------------------------------------------------


def test_paired_bootstrap_interval_matches_paired_bootstrap_ci() -> None:
    """This driver draws its own replicates so Holm has a p-value.

    That is only acceptable if the interval it produces is the same interval
    analysis.statistics.paired_bootstrap_ci would produce, which this pins.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.1, size=80)
    b = a - rng.normal(0.02, 0.05, size=80)

    point, lo, hi, p = _paired_bootstrap(
        a, b, generator=np.random.default_rng(7), n_boot=4000, ci=0.95
    )
    canonical = paired_bootstrap_ci(a, b, generator=np.random.default_rng(7), n_boot=4000, ci=0.95)

    assert point == pytest.approx(canonical.point, abs=1e-12)
    assert lo == pytest.approx(canonical.lo, abs=0.002)
    assert hi == pytest.approx(canonical.hi, abs=0.002)
    assert 0.0 <= p <= 1.0


def test_paired_bootstrap_is_paired_not_independent() -> None:
    """A constant per-case difference must give a very narrow interval.

    Resampling the two arms independently would measure the spread of the
    absolute values instead, and would widen this interval enormously. Same
    property tests/test_statistics.py pins for the canonical function.
    """
    a = np.linspace(0.2, 0.95, 60)
    b = a - 0.05
    _, lo, hi, _ = _paired_bootstrap(
        a, b, generator=np.random.default_rng(1), n_boot=2000, ci=0.95
    )
    assert hi - lo < 0.01
    assert lo > 0.0


# ---------------------------------------------------------------------------
# The decision rule, applied verbatim
# ---------------------------------------------------------------------------


def _family(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(cohort: str, endpoint: str, delta: float, contains_zero: bool) -> dict:
    return {"cohort": cohort, "endpoint": endpoint, "delta": delta, "contains_zero": contains_zero}


_THRESHOLDS = OmegaConf.create({"delta_auroc": 0.01, "delta_recall": 0.02})


def test_verdict_pass_requires_both_conjuncts_on_the_same_cohort() -> None:
    family = _family(
        [
            _row("ssa", "delta_auroc", 0.03, False),
            _row("ssa", "delta_recall", 0.05, False),
            _row("ped", "delta_auroc", 0.001, True),
            _row("ped", "delta_recall", 0.001, True),
        ]
    )
    verdict = build_verdict(family, _THRESHOLDS, ["ssa", "ped"])
    assert verdict["verdict"] == "pass"
    assert verdict["passed_cohorts"] == ["ssa"]


def test_verdict_is_not_a_pass_when_the_conjuncts_are_split_across_cohorts() -> None:
    """The rule says 'on at least one external cohort', not 'somewhere among them'."""
    family = _family(
        [
            _row("ssa", "delta_auroc", 0.03, False),
            _row("ssa", "delta_recall", 0.001, True),
            _row("ped", "delta_auroc", 0.001, True),
            _row("ped", "delta_recall", 0.05, False),
        ]
    )
    verdict = build_verdict(family, _THRESHOLDS, ["ssa", "ped"])
    assert verdict["verdict"] == "partial"
    assert verdict["passed_cohorts"] == []


def test_verdict_partial_when_significant_but_below_magnitude() -> None:
    family = _family(
        [
            _row("ssa", "delta_auroc", 0.004, False),
            _row("ssa", "delta_recall", 0.004, False),
        ]
    )
    verdict = build_verdict(family, _THRESHOLDS, ["ssa"])
    assert verdict["verdict"] == "partial"


def test_verdict_fail_when_no_external_ci_excludes_zero() -> None:
    family = _family(
        [
            _row("ssa", "delta_auroc", 0.5, True),
            _row("ssa", "delta_recall", 0.5, True),
        ]
    )
    verdict = build_verdict(family, _THRESHOLDS, ["ssa"])
    assert verdict["verdict"] == "fail"


def test_an_in_distribution_pass_cannot_carry_the_gate() -> None:
    """brats_test is not external; a strong result there must not read as a pass."""
    family = _family(
        [
            _row("brats_test", "delta_auroc", 0.20, False),
            _row("brats_test", "delta_recall", 0.20, False),
            _row("ssa", "delta_auroc", 0.0, True),
            _row("ssa", "delta_recall", 0.0, True),
        ]
    )
    verdict = build_verdict(family, _THRESHOLDS, ["ssa"])
    assert verdict["verdict"] == "fail"
    assert verdict["per_cohort"]["brats_test"]["meets_threshold"] is True


# ---------------------------------------------------------------------------
# Fit and score, end to end on synthetic samples
# ---------------------------------------------------------------------------


def _synthetic_samples(n_cases: int, seed: int, informative: bool):
    """Errors depend on disagreement (when informative), entropy is pure noise."""
    rng = np.random.default_rng(seed)
    samples = {}
    for i in range(n_cases):
        n = 500
        control = rng.uniform(size=n)
        score = rng.uniform(size=n)
        driver = score if informative else rng.uniform(size=n)
        positive = driver > 0.75
        samples[f"CASE_{i:03d}"] = {
            "score": score.astype(np.float32),
            "control": control.astype(np.float32),
            "positive": positive,
        }
    return samples


def test_fit_and_score_detects_an_informative_second_feature() -> None:
    fit = _synthetic_samples(20, seed=0, informative=True)
    baseline, combined = fit_both_arms(fit)
    table = score_cohort(_synthetic_samples(15, seed=1, informative=True), baseline, combined, 0.05)

    assert (table["delta_auroc"] > 0).mean() > 0.9
    assert table["auroc_both"].mean() > table["auroc_entropy"].mean()


def test_fit_and_score_finds_nothing_when_the_second_feature_is_noise() -> None:
    """The null case must come back null -- a method that always finds an effect is useless."""
    fit = _synthetic_samples(20, seed=2, informative=False)
    baseline, combined = fit_both_arms(fit)
    table = score_cohort(_synthetic_samples(15, seed=3, informative=False), baseline, combined, 0.05)

    assert abs(table["delta_auroc"].mean()) < 0.05


def test_score_cohort_reports_nan_for_a_case_with_no_error() -> None:
    samples = {
        "CLEAN": {
            "score": np.linspace(0, 1, 100, dtype=np.float32),
            "control": np.linspace(0, 1, 100, dtype=np.float32),
            "positive": np.zeros(100, dtype=bool),
        }
    }
    fit = _synthetic_samples(10, seed=4, informative=True)
    baseline, combined = fit_both_arms(fit)
    table = score_cohort(samples, baseline, combined, 0.05)
    assert np.isnan(table.loc["CLEAN", "auroc_both"])
    assert np.isnan(table.loc["CLEAN", "delta_recall"])
