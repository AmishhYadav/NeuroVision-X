"""Tests for neurovision.analysis.failure_detection (CPU, tiny synthetic arrays)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from neurovision.analysis.detection import case_entropy_scalars
from neurovision.analysis.failure_detection import (
    aurc,
    detection_family,
    ensemble_disagreement_score,
    ensemble_entropy_score,
    estimable,
    oracle_aurc,
    paired_delta_bootstrap,
    pairwise_mask_disagreement,
    random_aurc,
    risk_coverage,
)


def _m(*idx: int, n: int = 8) -> np.ndarray:
    a = np.zeros(n, dtype=bool)
    a[list(idx)] = True
    return a


def test_disagreement_identical_disjoint_empty() -> None:
    assert pairwise_mask_disagreement([_m(0, 1), _m(0, 1)]) == 0.0
    assert pairwise_mask_disagreement([_m(0, 1), _m(2, 3)]) == 1.0
    assert pairwise_mask_disagreement([_m(), _m()]) == 0.0
    assert pairwise_mask_disagreement([_m(), _m(1)]) == 1.0


def test_disagreement_three_members_hand_computed() -> None:
    a, b, c = _m(0, 1), _m(0, 1), _m(1, 2)
    # Dice(a,b)=1, Dice(a,c)=Dice(b,c)=2*1/4=0.5 -> mean 2/3 -> 1/3
    assert pairwise_mask_disagreement([a, b, c]) == pytest.approx(1 / 3)


def test_disagreement_errors() -> None:
    with pytest.raises(ValueError):
        pairwise_mask_disagreement([_m(0)])
    with pytest.raises(ValueError):
        pairwise_mask_disagreement([_m(0, n=4), _m(0, n=5)])


def test_ensemble_disagreement_takes_worse_region() -> None:
    masks = {"WT": [_m(0, 1), _m(0, 1)], "TC": [_m(0), _m(1)], "ET": [_m(0), _m(1)]}
    assert ensemble_disagreement_score(masks) == 1.0
    assert ensemble_disagreement_score(masks, regions=("WT",)) == 0.0
    with pytest.raises(ValueError):
        ensemble_disagreement_score({"WT": masks["WT"]})


def _probs(value: float, shape=(3, 2, 2, 2)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def test_ensemble_entropy_extremes_and_empty() -> None:
    # Normalised entropy: p=0.5 everywhere -> 1.0 (ln2 / ln2). Strict '>' means
    # p=0.5 is not foreground, so make foreground slightly above 0.5 instead.
    assert ensemble_entropy_score([_probs(0.5001)]) == pytest.approx(1.0, abs=1e-6)
    assert ensemble_entropy_score([_probs(1.0), _probs(1.0)]) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(ensemble_entropy_score([_probs(0.1)]))  # empty foreground


def test_ensemble_entropy_matches_case_entropy_scalars() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 2.0, size=(3, 4, 4, 4)).astype(np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    expected = case_entropy_scalars(logits)["ent_mean_fg_mean"]
    assert ensemble_entropy_score([probs]) == pytest.approx(expected, abs=1e-5)
    assert np.isfinite(logit(0.5))  # keeps scipy import honest


def test_risk_coverage_and_aurc_hand_example() -> None:
    score = [0.1, 0.2, 0.3, 0.4]
    unusable = [False, True, False, True]
    rc = risk_coverage(score, unusable)
    assert rc["coverage"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert rc["risk"].tolist() == pytest.approx([0.0, 0.5, 1 / 3, 0.5])
    assert aurc(score, unusable) == pytest.approx((0 + 0.5 + 1 / 3 + 0.5) / 4)
    # oracle: usable first -> risks 0, 0, 1/3, 1/2
    assert oracle_aurc(unusable) == pytest.approx((1 / 3 + 0.5) / 4)
    # worst: unusable first -> 1, 1, 2/3, 1/2
    worst = aurc([0.4, 0.3, 0.2, 0.1], unusable)
    assert oracle_aurc(unusable) <= aurc(score, unusable) <= worst
    assert random_aurc(unusable) == 0.5


def test_risk_coverage_ties_stable() -> None:
    rc = risk_coverage([1.0, 1.0, 1.0], [True, False, False])
    assert rc["risk"].tolist() == pytest.approx([1.0, 0.5, 1 / 3])


def test_estimable() -> None:
    assert estimable([True] * 5 + [False] * 5)
    assert not estimable([True] * 4 + [False] * 50)
    assert not estimable([True] * 50 + [False] * 4)


def _toy(n: int = 40):
    rng = np.random.default_rng(1)
    u = np.arange(n) < n // 2
    perfect = u.astype(float) + rng.normal(0, 0.01, n)
    noise = rng.normal(size=n)
    return perfect, noise, u


def test_bootstrap_identical_scores() -> None:
    _, noise, u = _toy()
    r = paired_delta_bootstrap(noise, noise, u, "auroc", n_boot=200)
    assert r["delta"] == 0.0 and r["p"] == 1.0
    assert r["lo"] <= 0.0 <= r["hi"]


def test_bootstrap_perfect_vs_random() -> None:
    perfect, noise, u = _toy()
    r = paired_delta_bootstrap(perfect, noise, u, "auroc", n_boot=300)
    assert r["delta"] > 0 and r["lo"] > 0
    assert r["n"] == 40 and r["n_pos"] == 20
    assert r["n_boot_valid"] == 300  # stratified -> both classes always present
    r2 = paired_delta_bootstrap(perfect, noise, u, "aurc", n_boot=100)
    assert r2["delta"] < 0  # lower AURC is better


def test_bootstrap_deterministic_and_rejects_nan() -> None:
    perfect, noise, u = _toy()
    r1 = paired_delta_bootstrap(perfect, noise, u, "auroc", n_boot=100, seed=7)
    r2 = paired_delta_bootstrap(perfect, noise, u, "auroc", n_boot=100, seed=7)
    assert r1 == r2
    bad = noise.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        paired_delta_bootstrap(perfect, bad, u, "auroc", n_boot=10)


def _rows() -> pd.DataFrame:
    base = dict(cohort="test", target="dice", estimable=True)
    return pd.DataFrame(
        [
            dict(base, signal="A", delta=0.2, lo=0.1, hi=0.3, p=0.001),
            dict(base, signal="B", delta=-0.2, lo=-0.3, hi=-0.1, p=0.002),
            dict(base, signal="C", delta=0.1, lo=-0.1, hi=0.3, p=0.4),
            dict(base, signal="D", delta=0.5, lo=0.4, hi=0.6, p=0.0, estimable=False),
        ]
    )


def test_detection_family_holm_and_verdicts() -> None:
    out = detection_family(_rows())
    # Holm over 3 estimable rows only: 0.001*3, 0.002*2, 0.4*1
    assert out["p_holm"].iloc[:3].tolist() == pytest.approx([0.003, 0.004, 0.4])
    assert np.isnan(out["p_holm"].iloc[3])
    assert out["verdict"].tolist() == ["better", "worse", "inconclusive", "not_estimable"]


def test_detection_family_lower_is_better() -> None:
    out = detection_family(_rows(), higher_is_better=False)
    assert out["verdict"].tolist() == ["worse", "better", "inconclusive", "not_estimable"]
