"""Tests for `neurovision.analysis.detection`.

CPU only, tiny synthetic arrays built in the test, whole file well under a
few seconds. Numeric expectations are hand-computed or derived from known
statistical identities (Mann-Whitney U for AUROC, the partial-correlation
formula) rather than re-deriving the implementation.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neurovision.analysis.detection import (
    REGION_NAMES,
    auroc,
    case_entropy_scalars,
    partial_spearman,
    partial_spearman_ci,
    residualised_auroc,
    spearman,
)
from neurovision.analysis.statistics import BootstrapResult

# ---------------------------------------------------------------------------
# 1-4. case_entropy_scalars
# ---------------------------------------------------------------------------


def test_case_entropy_scalars_all_zero_logits_gives_max_entropy() -> None:
    logits = np.zeros((3, 4, 4, 4), dtype=np.float32)
    row = case_entropy_scalars(logits)
    for region in REGION_NAMES:
        assert row[f"ent_mean_{region}"] == pytest.approx(1.0, abs=1e-5)
        assert row[f"ent_max_{region}"] == pytest.approx(1.0, abs=1e-5)


def test_case_entropy_scalars_saturated_fp16_is_finite_not_nan() -> None:
    # Regression guard for the fp16 eps-clamp bug: a probability-space
    # implementation with p.clamp(1e-6, 1 - 1e-6) produces NaN here because
    # fp16's epsilon (~9.8e-4) rounds 1.0 - 1e-6 to exactly 1.0.
    logits = np.full((3, 4, 4, 4), 50.0, dtype=np.float16)
    row = case_entropy_scalars(logits)
    values = np.array(list(row.values()), dtype=np.float64)
    assert np.isfinite(values).all()
    # And genuinely near-zero entropy, not just "some finite number".
    for region in REGION_NAMES:
        assert row[f"ent_mean_{region}"] < 1e-6


def test_case_entropy_scalars_empty_foreground_is_nan_but_mean_is_finite() -> None:
    # Three regions, one per behaviour: max entropy (p=0.5), saturated
    # positive (foreground present), saturated negative (foreground empty).
    logits = np.zeros((3, 2, 2, 2), dtype=np.float32)
    logits[0] = 0.0  # ET: p=0.5 everywhere, foreground present (0.5 > 0.5 is False actually)
    logits[1] = 50.0  # TC: saturated positive, foreground present
    logits[2] = -50.0  # WT: saturated negative, foreground empty
    row = case_entropy_scalars(logits)

    # WT's predicted foreground is empty -> *_fg_* is NaN, *_mean_* is finite.
    assert np.isnan(row["ent_mean_fg_WT"])
    assert np.isfinite(row["ent_mean_WT"])

    # TC's predicted foreground is non-empty -> *_fg_* is finite.
    assert np.isfinite(row["ent_mean_fg_TC"])

    # The cross-region mean skips the NaN region rather than propagating it.
    assert np.isfinite(row["ent_mean_fg_mean"])


def test_case_entropy_scalars_signature_has_no_label_parameter() -> None:
    sig = inspect.signature(case_entropy_scalars)
    forbidden = ("label", "target", "gt", "truth")
    for name in sig.parameters:
        lowered = name.lower()
        assert not any(
            f in lowered for f in forbidden
        ), f"case_entropy_scalars must take no label argument, structurally; found {name!r}."


# ---------------------------------------------------------------------------
# 5-6. partial_spearman
# ---------------------------------------------------------------------------


def test_partial_spearman_removes_confound_only_correlation() -> None:
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = z + rng.normal(size=n)

    raw = spearman(x, y)
    partial = partial_spearman(x, y, z)

    assert abs(raw) > 0.3  # clearly non-zero: correlated purely through z
    assert abs(partial) < 0.1  # controlling for z removes essentially all of it


def test_partial_spearman_recovers_correlation_independent_of_confound() -> None:
    rng = np.random.default_rng(1)
    n = 500
    z = rng.normal(size=n)
    w = rng.normal(size=n)  # the real shared signal, unrelated to z's role
    x = w + 0.1 * rng.normal(size=n) + z
    y = w + 0.1 * rng.normal(size=n) + z

    partial = partial_spearman(x, y, z)
    assert partial > 0.7


def test_partial_spearman_nan_on_degenerate_denominator() -> None:
    # x is a deterministic function of z (r_xz = 1.0) -> denominator is 0.
    n = 30
    z = np.arange(n, dtype=np.float64)
    x = z.copy()
    y = np.random.default_rng(2).normal(size=n)
    result = partial_spearman(x, y, z)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# 7-8. auroc
# ---------------------------------------------------------------------------


def test_auroc_perfect_and_inverted_separation() -> None:
    score = np.array([1.0, 2.0, 3.0, 4.0])
    positive = np.array([False, False, True, True])
    assert auroc(score, positive) == pytest.approx(1.0)
    assert auroc(score[::-1], positive) == pytest.approx(0.0)


def test_auroc_random_score_near_half() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    score = rng.normal(size=n)
    positive = rng.random(n) > 0.5  # independent of score
    assert auroc(score, positive) == pytest.approx(0.5, abs=0.05)


def test_auroc_empty_class_is_nan() -> None:
    score = np.array([1.0, 2.0, 3.0])
    positive = np.array([False, False, False])
    assert np.isnan(auroc(score, positive))


def test_auroc_ties_constant_score_is_exactly_half() -> None:
    score = np.zeros(10)
    positive = np.array([True] * 5 + [False] * 5)
    assert auroc(score, positive) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 9. residualised_auroc
# ---------------------------------------------------------------------------


def test_residualised_auroc_catches_pure_reproduction_of_control() -> None:
    rng = np.random.default_rng(4)
    n = 400
    control = rng.normal(size=n)
    score = control**3 + 5.0  # a monotone transform of control -- same rank order
    noise = rng.normal(size=n) * 0.3
    positive = (control + noise) > 0.0

    result = residualised_auroc(score, control, positive)

    assert result["auroc_score"] > 0.8
    assert result["auroc_control"] > 0.8
    # Nothing survives once control's contribution is removed: score's rank
    # is IDENTICAL to control's rank (monotone transform), so the residual
    # is identically zero and the residual AUROC collapses to the
    # constant-score value of exactly 0.5.
    assert result["auroc_residual"] == pytest.approx(0.5, abs=1e-9)


def test_residualised_auroc_score_keeps_own_information() -> None:
    # score carries information about `positive` that control does NOT.
    rng = np.random.default_rng(5)
    n = 400
    control = rng.normal(size=n)  # unrelated to positive
    positive = rng.random(n) > 0.5
    score = np.where(positive, rng.normal(loc=3.0, size=n), rng.normal(loc=-3.0, size=n))

    result = residualised_auroc(score, control, positive)

    assert result["auroc_control"] == pytest.approx(0.5, abs=0.1)
    assert result["auroc_score"] > 0.9
    # Residualising against an uninformative control barely changes score's
    # own detection power.
    assert result["auroc_residual"] > 0.85


# ---------------------------------------------------------------------------
# 10. partial_spearman_ci
# ---------------------------------------------------------------------------


def test_partial_spearman_ci_contains_zero_for_independent_signal() -> None:
    rng = np.random.default_rng(6)
    n = 300
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = z + rng.normal(size=n)

    result = partial_spearman_ci(x, y, z, generator=np.random.default_rng(7), n_boot=1000)

    assert isinstance(result, BootstrapResult)
    assert result.contains_zero
    assert result.n == n
    assert result.n_boot == 1000


def test_partial_spearman_ci_excludes_zero_for_genuine_partial_correlation() -> None:
    rng = np.random.default_rng(8)
    n = 300
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    x = w + 0.1 * rng.normal(size=n) + z
    y = w + 0.1 * rng.normal(size=n) + z

    result = partial_spearman_ci(x, y, z, generator=np.random.default_rng(9), n_boot=1000)

    assert isinstance(result, BootstrapResult)
    assert not result.contains_zero
    assert result.lo > 0.0


def test_region_names_matches_the_canonical_ordering() -> None:
    """`detection.REGION_NAMES` must equal `data.transforms.REGION_NAMES`.

    `detection.py` re-declares this tuple rather than importing it, so the
    module stays numpy/scipy-only (`data.transforms` pulls in torch and
    MONAI) -- the same reasoning `figures.py` and `tables.py` use. The cost
    of a re-declaration is that the copies can drift, and this project has
    already been bitten by exactly that: `figures.REGION_ORDER` is
    ``("WT", "TC", "ET")`` while `data.transforms.REGION_NAMES` is
    ``("ET", "TC", "WT")``, so index 0 means a different region depending on
    which module you are in, and either mapping produces a perfectly
    plausible output with every label silently wrong.

    This test is what makes the re-declaration safe. It lives in the test
    suite, not in `detection.py`, precisely because a test may import torch
    where the module under test may not.
    """
    from neurovision.analysis.detection import REGION_NAMES as DETECTION_REGIONS
    from neurovision.data.transforms import REGION_NAMES as CANONICAL_REGIONS

    assert tuple(DETECTION_REGIONS) == tuple(CANONICAL_REGIONS), (
        "detection.REGION_NAMES has drifted from data.transforms.REGION_NAMES: "
        f"{tuple(DETECTION_REGIONS)} != {tuple(CANONICAL_REGIONS)}. Every "
        "`ent_*_<REGION>` column would be attributed to the wrong region."
    )
