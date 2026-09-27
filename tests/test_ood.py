"""Tests for `neurovision.analysis.ood`.

Pre-registration P1.4 (`docs/research_docs/preregistrations/preregistration_ood.md`).
Everything here is synthetic -- tiny hand-built arrays and `pandas.DataFrame`s, never
real BraTS data -- and every test runs in well under a second on CPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurovision.analysis.ood import (
    FEATURE_NAMES,
    MODALITIES,
    OODModel,
    extract_features,
    fit_ood_model,
    flag,
    ood_scores,
    quantile_cuts,
)
from neurovision.inference.gatekeeper import Decision, judge_ood_score


def _sphere_volume(
    shape: tuple[int, int, int], rng: np.random.Generator, radius: int = 6
) -> np.ndarray:
    """One float16 (D, H, W) volume: a z-scored "brain" sphere on a zero background."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(
        np.arange(d) - d // 2, np.arange(h) - h // 2, np.arange(w) - w // 2, indexing="ij"
    )
    sphere = (zz**2 + yy**2 + xx**2) <= radius**2
    volume = np.zeros(shape, dtype=np.float32)
    # z-scored: brain voxels mean ~0, SD ~1 (matches the preprocessing pipeline).
    volume[sphere] = rng.standard_normal(int(sphere.sum())).astype(np.float32)
    return volume.astype(np.float16), sphere


def _synthetic_image(rng: np.random.Generator, shape: tuple[int, int, int] = (20, 20, 20)):
    """A synthetic (4, D, H, W) float16 volume with one z-scored sphere per channel."""
    volumes = []
    for _ in MODALITIES:
        volume, _sphere = _sphere_volume(shape, rng)
        volumes.append(volume)
    return np.stack(volumes, axis=0)


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


def test_feature_names_46_unique():
    assert len(FEATURE_NAMES) == 46
    assert len(set(FEATURE_NAMES)) == 46


def test_extract_features_returns_exactly_feature_names():
    rng = np.random.default_rng(0)
    image = _synthetic_image(rng)
    features = extract_features(image)
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())


def test_extract_features_float16_no_overflow():
    # The float16-sum-overflow trap named in the spec: a naive std/skew/kurtosis
    # computed directly on float16 data can return inf. With ~a few thousand
    # nonzero voxels of standard-normal data this is well within float16 range
    # already, so the real regression guard is the assertion itself, run against
    # a cast-to-float32 implementation.
    rng = np.random.default_rng(1)
    image = _synthetic_image(rng, shape=(24, 24, 24))
    assert image.dtype == np.float16
    features = extract_features(image)
    values = np.array(list(features.values()), dtype=np.float64)
    assert np.all(np.isfinite(values))


def test_extract_features_wrong_shape_raises():
    rng = np.random.default_rng(2)
    bad = rng.standard_normal((3, 10, 10, 10)).astype(np.float16)
    with pytest.raises(ValueError):
        extract_features(bad)


def test_extract_features_too_few_nonzero_voxels_raises():
    image = np.zeros((4, 20, 20, 20), dtype=np.float16)
    # Give every modality a handful of nonzero voxels, well under the minimum.
    image[:, 0, 0, :5] = 1.0
    with pytest.raises(ValueError):
        extract_features(image)


def test_percentiles_monotone_per_modality():
    rng = np.random.default_rng(3)
    image = _synthetic_image(rng)
    features = extract_features(image)
    for modality in MODALITIES:
        values = [features[f"{modality}_p{p:02d}"] for p in (1, 5, 25, 50, 75, 95, 99)]
        assert values == sorted(values)


def test_correlation_of_identical_channels_is_one():
    rng = np.random.default_rng(4)
    volume, _ = _sphere_volume((20, 20, 20), rng)
    image = np.stack([volume, volume, volume, volume], axis=0)
    features = extract_features(image)
    for key, value in features.items():
        if key.startswith("corr_"):
            assert value == pytest.approx(1.0, abs=1e-6)


def test_log_brain_voxels_matches_hand_count():
    shape = (20, 20, 20)
    n_voxels = 20 * 20 * 20
    rng = np.random.default_rng(15)
    flat = np.zeros((4, n_voxels), dtype=np.float16)
    values = rng.standard_normal((4, n_voxels)).astype(np.float16)
    # A 100-voxel "core" nonzero in all four modalities, plus a 100-voxel
    # modality-specific extension each -- with modality 3's extension placed to
    # overlap modality 0's extension exactly. Every modality then has
    # 100 + 100 == 200 nonzero voxels (over `_MIN_NONZERO_VOXELS`), and the
    # ANY-modality union is a known, hand-computable count: 0-399 == 400 voxels,
    # not 800 -- core (0-99) + extra0 (100-199) + extra1 (200-299) +
    # extra2 (300-399), with extra3 == extra0 contributing nothing new.
    core = slice(0, 100)
    extra0, extra1, extra2 = slice(100, 200), slice(200, 300), slice(300, 400)
    extra3 = extra0  # exact overlap with modality 0's extension
    flat[0, core] = values[0, core]
    flat[0, extra0] = values[0, extra0]
    flat[1, core] = values[1, core]
    flat[1, extra1] = values[1, extra1]
    flat[2, core] = values[2, core]
    flat[2, extra2] = values[2, extra2]
    flat[3, core] = values[3, core]
    flat[3, extra3] = values[3, extra3]
    image = flat.reshape((4,) + shape)
    features = extract_features(image)
    assert features["log_brain_voxels"] == pytest.approx(np.log(400.0))


def test_extents_match_shape():
    rng = np.random.default_rng(5)
    image = _synthetic_image(rng, shape=(18, 22, 16))
    features = extract_features(image)
    assert features["extent_d"] == 18.0
    assert features["extent_h"] == 22.0
    assert features["extent_w"] == 16.0


# ---------------------------------------------------------------------------
# fit_ood_model / ood_scores
# ---------------------------------------------------------------------------


def _gaussian_features(n: int, n_features: int, rng: np.random.Generator) -> pd.DataFrame:
    """`n` rows drawn i.i.d. standard normal, columns named `FEATURE_NAMES[:n_features]`."""
    columns = FEATURE_NAMES[:n_features]
    data = rng.standard_normal((n, n_features))
    return pd.DataFrame(data, columns=columns)


def test_fit_and_score_mean_low_far_high():
    rng = np.random.default_rng(6)
    train = _gaussian_features(500, len(FEATURE_NAMES), rng)
    model = fit_ood_model(train, shrinkage=0.1)

    at_mean = pd.DataFrame([model.center], columns=FEATURE_NAMES)
    far_point = model.center + 10.0 * model.scale
    at_far = pd.DataFrame([far_point], columns=FEATURE_NAMES)

    score_mean = ood_scores(model, at_mean)[0]
    score_far = ood_scores(model, at_far)[0]
    assert score_mean == pytest.approx(0.0, abs=1e-6)
    assert score_far > score_mean
    assert score_far > 5.0


def test_score_invariant_to_constant_shift_in_both_fit_and_score():
    rng = np.random.default_rng(7)
    train = _gaussian_features(200, len(FEATURE_NAMES), rng)
    query = _gaussian_features(5, len(FEATURE_NAMES), rng)

    model_a = fit_ood_model(train)
    scores_a = ood_scores(model_a, query)

    shifted_train = train.copy()
    shifted_query = query.copy()
    column = FEATURE_NAMES[0]
    shifted_train[column] = shifted_train[column] + 100.0
    shifted_query[column] = shifted_query[column] + 100.0

    model_b = fit_ood_model(shifted_train)
    scores_b = ood_scores(model_b, shifted_query)

    np.testing.assert_allclose(scores_a, scores_b, atol=1e-8)


def test_scores_are_non_negative():
    rng = np.random.default_rng(8)
    train = _gaussian_features(100, len(FEATURE_NAMES), rng)
    model = fit_ood_model(train)
    query = _gaussian_features(20, len(FEATURE_NAMES), rng)
    scores = ood_scores(model, query)
    assert np.all(scores >= 0.0)


def test_fit_rejects_nan():
    rng = np.random.default_rng(9)
    train = _gaussian_features(50, len(FEATURE_NAMES), rng)
    train.loc[0, FEATURE_NAMES[0]] = np.nan
    with pytest.raises(ValueError, match=FEATURE_NAMES[0]):
        fit_ood_model(train)


def test_fit_rejects_too_few_rows():
    rng = np.random.default_rng(10)
    train = _gaussian_features(5, len(FEATURE_NAMES), rng)
    with pytest.raises(ValueError):
        fit_ood_model(train)


def test_fit_rejects_zero_sd_column():
    rng = np.random.default_rng(11)
    train = _gaussian_features(50, len(FEATURE_NAMES), rng)
    train[FEATURE_NAMES[0]] = 1.0  # constant -> zero SD
    with pytest.raises(ValueError):
        fit_ood_model(train)


def test_shrinkage_makes_rank_deficient_case_invertible():
    rng = np.random.default_rng(12)
    n_features = len(FEATURE_NAMES)
    # n < F: a plain sample covariance here would be exactly singular.
    train = _gaussian_features(20, n_features, rng)
    model = fit_ood_model(train, shrinkage=0.1)
    assert np.all(np.isfinite(model.precision))
    query = _gaussian_features(3, n_features, rng)
    scores = ood_scores(model, query)
    assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# OODModel.to_dict / from_dict
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_roundtrip_scores():
    rng = np.random.default_rng(13)
    train = _gaussian_features(100, len(FEATURE_NAMES), rng)
    model = fit_ood_model(train)
    query = _gaussian_features(10, len(FEATURE_NAMES), rng)
    scores_before = ood_scores(model, query)

    payload = model.to_dict()
    # JSON-safe: only lists/floats/ints, nothing numpy.
    assert isinstance(payload["center"], list)
    assert isinstance(payload["precision"], list)

    restored = OODModel.from_dict(payload)
    scores_after = ood_scores(restored, query)
    np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)


# ---------------------------------------------------------------------------
# quantile_cuts / flag
# ---------------------------------------------------------------------------


def test_quantile_cuts_ordering():
    rng = np.random.default_rng(14)
    val_scores = rng.exponential(size=500)
    caution_cut, refuse_cut = quantile_cuts(val_scores, 0.10, 0.02)
    assert caution_cut <= refuse_cut


@pytest.mark.parametrize(
    "caution_quantile,refuse_quantile",
    [
        (0.10, 0.10),  # equal, not strictly less
        (0.02, 0.10),  # inverted
        (0.0, 0.02),  # out of (0, 1)
        (0.10, 1.0),  # out of (0, 1)
        (-0.1, 0.02),
    ],
)
def test_quantile_cuts_rejects_bad_quantiles(caution_quantile, refuse_quantile):
    val_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        quantile_cuts(val_scores, caution_quantile, refuse_quantile)


def test_flag_boundaries_match_judge_ood_score_equality_rule():
    caution_cut, refuse_cut = 1.0, 2.0
    scores = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    flags = flag(scores, caution_cut, refuse_cut)

    for score, flagged in zip(scores, flags):
        verdict = judge_ood_score(float(score), (caution_cut, refuse_cut), enabled=True)
        assert flagged == verdict.decision.value

    # Explicit boundary check: exactly at a cut does not cross it.
    assert flags[1] == Decision.PROCEED.value  # score == caution_cut
    assert flags[3] == Decision.PROCEED_WITH_CAUTION.value  # score == refuse_cut
    assert flags[4] == Decision.REFUSE.value  # score > refuse_cut


def test_flag_returns_valid_decision_strings():
    scores = np.array([0.0, 10.0, 100.0])
    flags = flag(scores, 1.0, 2.0)
    valid = {Decision.PROCEED.value, Decision.PROCEED_WITH_CAUTION.value, Decision.REFUSE.value}
    assert all(f in valid for f in flags)
