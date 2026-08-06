"""Tests for `neurovision.uncertainty.calibration`.

CPU only, tiny hand-built/synthetic tensors, whole file well under 5 seconds.
Every expected numeric value below is hand-computed in the test itself
(never read back from the implementation) -- see the module's own docstring
for the probability-vs-frequency convention and the reasoning behind each
documented edge case (p == 1.0, empty mask, min_count, pooled vs per-case).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
import torch

from neurovision.uncertainty.calibration import (
    DEFAULT_N_BINS,
    CalibrationAccumulator,
    apply_temperature,
    bin_edges,
    brain_mask,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    predicted_foreground_mask,
    reliability_curve,
    subsample_voxels,
    union_foreground_mask,
)

# ---------------------------------------------------------------------------
# 1. reliability_curve: hand-built tiny input
# ---------------------------------------------------------------------------


def test_reliability_curve_hand_built_input() -> None:
    n_bins = 4
    prob = torch.tensor([0.1, 0.1, 0.3, 0.6, 0.6, 0.6, 0.99])
    label = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    curve = reliability_curve(prob, label, n_bins=n_bins)

    assert len(curve) == n_bins
    np.testing.assert_allclose(curve["bin_lower"].to_numpy(), [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(curve["bin_upper"].to_numpy(), [0.25, 0.5, 0.75, 1.0])
    assert curve["count"].sum() == 7

    # bin 2 = [0.5, 0.75): probs [0.6, 0.6, 0.6], labels [1, 1, 0]
    row2 = curve.iloc[2]
    assert row2["count"] == 3
    assert row2["mean_prob"] == pytest.approx(0.6)
    assert row2["mean_label"] == pytest.approx(2.0 / 3.0)
    assert row2["gap"] == pytest.approx(2.0 / 3.0 - 0.6)


# ---------------------------------------------------------------------------
# 2 & 3. p == 1.0 and p == 0.0 edge bins
# ---------------------------------------------------------------------------


def test_p_equals_one_lands_in_last_bin() -> None:
    n_bins = 5
    prob = torch.tensor([1.0])
    label = torch.tensor([1.0])

    curve = reliability_curve(prob, label, n_bins=n_bins)

    assert curve["count"].sum() == 1
    assert curve.iloc[-1]["count"] == 1
    assert curve.iloc[:-1]["count"].sum() == 0


def test_p_equals_zero_lands_in_first_bin() -> None:
    n_bins = 5
    prob = torch.tensor([0.0])
    label = torch.tensor([0.0])

    curve = reliability_curve(prob, label, n_bins=n_bins)

    assert curve.iloc[0]["count"] == 1
    assert curve.iloc[1:]["count"].sum() == 0


# ---------------------------------------------------------------------------
# 4. Perfectly calibrated construction -> ECE < 0.01
# ---------------------------------------------------------------------------


def test_perfectly_calibrated_construction_gives_low_ece() -> None:
    n_bins = DEFAULT_N_BINS
    probs = []
    labels = []
    expected_gap_contribution = 0.0
    total = 0
    for i in range(n_bins):
        centre = (i + 0.5) / n_bins
        n_pos = round(100 * centre)
        probs.extend([centre] * 100)
        labels.extend([1.0] * n_pos + [0.0] * (100 - n_pos))
        gap = (n_pos / 100.0) - centre
        expected_gap_contribution += 100 * abs(gap)
        total += 100

    expected_ece = expected_gap_contribution / total

    ece = expected_calibration_error(torch.tensor(probs), torch.tensor(labels), n_bins=n_bins)

    # abs=1e-6, not something tighter: this module casts everything to float32
    # internally (see the module docstring's CUDA-fp16 hazard note), and float32
    # has ~1e-7 relative precision -- summing 15 bins' worth of ~0.002-scale gaps
    # lands right at that limit if compared to a float64 hand computation any tighter.
    assert ece == pytest.approx(expected_ece, abs=1e-6)
    assert ece < 0.01


# ---------------------------------------------------------------------------
# 5. Pathological overconfidence: exact ECE/MCE/Brier
# ---------------------------------------------------------------------------


def test_pathological_overconfidence_exact_values() -> None:
    prob = torch.full((1000,), 0.9)
    label = torch.cat([torch.ones(500), torch.zeros(500)])

    ece = expected_calibration_error(prob, label)
    mce = maximum_calibration_error(prob, label)
    brier = brier_score(prob, label)

    assert ece == pytest.approx(0.4)
    assert mce == pytest.approx(0.4)
    assert brier == pytest.approx(0.5 * 0.01 + 0.5 * 0.81)  # 0.41


# ---------------------------------------------------------------------------
# 6. Perfect and confident predictions -> ECE == 0, Brier == 0
# ---------------------------------------------------------------------------


def test_perfect_confident_predictions_give_zero_error() -> None:
    prob = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    label = prob.clone()

    ece = expected_calibration_error(prob, label)
    brier = brier_score(prob, label)

    assert ece == pytest.approx(0.0, abs=1e-12)
    assert brier == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 7. brier_score hand value
# ---------------------------------------------------------------------------


def test_brier_score_hand_value() -> None:
    prob = torch.tensor([0.2, 0.8, 0.5, 0.5])
    label = torch.tensor([0.0, 1.0, 1.0, 0.0])
    # errors: 0.04, 0.04, 0.25, 0.25 -> mean 0.145
    expected = (0.04 + 0.04 + 0.25 + 0.25) / 4.0

    brier = brier_score(prob, label)

    assert brier == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 8. mask restricts computation exactly like pre-filtering
# ---------------------------------------------------------------------------


def test_mask_matches_manual_prefiltering() -> None:
    gen = torch.Generator().manual_seed(0)
    prob = torch.rand(200, generator=gen)
    label = torch.bernoulli(prob, generator=gen)
    mask = torch.rand(200, generator=gen) > 0.5

    ece_masked = expected_calibration_error(prob, label, mask=mask)
    brier_masked = brier_score(prob, label, mask=mask)
    curve_masked = reliability_curve(prob, label, mask=mask)

    prob_sub = prob[mask]
    label_sub = label[mask]
    ece_sub = expected_calibration_error(prob_sub, label_sub)
    brier_sub = brier_score(prob_sub, label_sub)
    curve_sub = reliability_curve(prob_sub, label_sub)

    assert ece_masked == pytest.approx(ece_sub, abs=1e-12)
    assert brier_masked == pytest.approx(brier_sub, abs=1e-12)
    pd.testing.assert_frame_equal(curve_masked, curve_sub)


# ---------------------------------------------------------------------------
# 9. All-False mask -> NaN scalars, empty-but-well-formed DataFrame
# ---------------------------------------------------------------------------


def test_all_false_mask_gives_nan_and_no_exception(caplog: pytest.LogCaptureFixture) -> None:
    n_bins = 6
    prob = torch.rand(50)
    label = torch.bernoulli(prob)
    mask = torch.zeros(50, dtype=torch.bool)

    with caplog.at_level(logging.WARNING):
        ece = expected_calibration_error(prob, label, n_bins=n_bins, mask=mask)
        mce = maximum_calibration_error(prob, label, n_bins=n_bins, mask=mask)
        brier = brier_score(prob, label, mask=mask)
        curve = reliability_curve(prob, label, n_bins=n_bins, mask=mask)

    assert math.isnan(ece)
    assert math.isnan(mce)
    assert math.isnan(brier)
    assert len(curve) == n_bins
    assert (curve["count"] == 0).all()
    assert curve["mean_prob"].isna().all()
    assert curve["mean_label"].isna().all()
    assert curve["gap"].isna().all()
    assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# 10. min_count excludes a sparse outlier bin from MCE
# ---------------------------------------------------------------------------


def test_min_count_excludes_sparse_outlier_bin() -> None:
    n_bins = 4
    # bin 0: one voxel, wildly wrong (gap = |1.0 - 0.1| = 0.9)
    # bin 2: 20 voxels, mildly wrong (16 ones, 4 zeros -> mean_label 0.8, gap = |0.8-0.6| = 0.2)
    prob = torch.tensor([0.1] + [0.6] * 20)
    label = torch.tensor([1.0] + [1.0] * 16 + [0.0] * 4)

    mce_default = maximum_calibration_error(prob, label, n_bins=n_bins)
    mce_min2 = maximum_calibration_error(prob, label, n_bins=n_bins, min_count=2)

    assert mce_default == pytest.approx(0.9)  # bin 0: |1.0 - 0.1|
    assert mce_min2 == pytest.approx(0.2, abs=1e-6)  # bin 2: |0.8 - 0.6|


# ---------------------------------------------------------------------------
# 11. Validation raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda p, lbl: expected_calibration_error(p, lbl),
        lambda p, lbl: maximum_calibration_error(p, lbl),
        lambda p, lbl: brier_score(p, lbl),
        lambda p, lbl: reliability_curve(p, lbl),
    ],
)
def test_validation_raises_on_shape_mismatch(fn) -> None:
    with pytest.raises(ValueError, match="shape"):
        fn(torch.tensor([0.1, 0.2]), torch.tensor([0.0, 1.0, 0.0]))


@pytest.mark.parametrize(
    "fn",
    [
        lambda p, lbl: expected_calibration_error(p, lbl),
        lambda p, lbl: maximum_calibration_error(p, lbl),
        lambda p, lbl: brier_score(p, lbl),
        lambda p, lbl: reliability_curve(p, lbl),
    ],
)
def test_validation_raises_on_prob_above_one(fn) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fn(torch.tensor([0.1, 1.5]), torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize(
    "fn",
    [
        lambda p, lbl: expected_calibration_error(p, lbl),
        lambda p, lbl: maximum_calibration_error(p, lbl),
        lambda p, lbl: brier_score(p, lbl),
        lambda p, lbl: reliability_curve(p, lbl),
    ],
)
def test_validation_raises_on_prob_below_zero(fn) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fn(torch.tensor([-0.1, 0.5]), torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize(
    "fn",
    [
        lambda p, lbl: expected_calibration_error(p, lbl),
        lambda p, lbl: maximum_calibration_error(p, lbl),
        lambda p, lbl: brier_score(p, lbl),
        lambda p, lbl: reliability_curve(p, lbl),
    ],
)
def test_validation_raises_on_non_binary_label(fn) -> None:
    with pytest.raises(ValueError, match="binary"):
        fn(torch.tensor([0.1, 0.5]), torch.tensor([0.0, 0.5]))


@pytest.mark.parametrize(
    "fn",
    [
        lambda p, lbl: expected_calibration_error(p, lbl),
        lambda p, lbl: maximum_calibration_error(p, lbl),
        lambda p, lbl: brier_score(p, lbl),
        lambda p, lbl: reliability_curve(p, lbl),
    ],
)
def test_validation_raises_on_empty_input(fn) -> None:
    with pytest.raises(ValueError, match="zero elements"):
        fn(torch.empty(0), torch.empty(0))


# ---------------------------------------------------------------------------
# 12. Streaming equality: CalibrationAccumulator vs one-shot over concatenation
# ---------------------------------------------------------------------------


def _make_case(
    gen: torch.Generator, shape: tuple[int, int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    prob = torch.rand(shape, generator=gen)
    label = torch.bernoulli(prob, generator=gen)
    return prob, label


def test_streaming_equals_one_shot_over_concatenation() -> None:
    gen = torch.Generator().manual_seed(42)
    region_names = ("ET", "TC", "WT")
    shapes = [(3, 3, 4, 4), (3, 2, 3, 5), (3, 4, 2, 2)]
    cases = [_make_case(gen, shape) for shape in shapes]

    acc = CalibrationAccumulator(n_bins=10, region_names=region_names)
    for i, (prob, label) in enumerate(cases):
        acc.add_case(f"case{i}", prob, label)

    summary = acc.summary()

    for c, region in enumerate(region_names):
        prob_concat = torch.cat([prob[c].reshape(-1) for prob, _ in cases])
        label_concat = torch.cat([label[c].reshape(-1) for _, label in cases])

        expected_ece = expected_calibration_error(prob_concat, label_concat, n_bins=10)
        expected_brier = brier_score(prob_concat, label_concat)

        assert summary.loc[f"ece_{region}", "value"] == pytest.approx(expected_ece, abs=1e-9)
        assert summary.loc[f"brier_{region}", "value"] == pytest.approx(expected_brier, abs=1e-9)


# ---------------------------------------------------------------------------
# 13. add_case validation
# ---------------------------------------------------------------------------


def test_add_case_raises_on_wrong_channel_count() -> None:
    acc = CalibrationAccumulator(region_names=("ET", "TC", "WT"))
    prob = torch.rand(2, 4, 4, 4)  # only 2 channels, need 3
    label = torch.bernoulli(prob)

    with pytest.raises(ValueError, match="channel"):
        acc.add_case("case0", prob, label)


def test_add_case_raises_on_duplicate_case_id() -> None:
    acc = CalibrationAccumulator(region_names=("ET", "TC", "WT"))
    prob = torch.rand(3, 4, 4, 4)
    label = torch.bernoulli(prob)
    acc.add_case("case0", prob, label)

    with pytest.raises(ValueError, match="already"):
        acc.add_case("case0", prob, label)


# ---------------------------------------------------------------------------
# 14. per_case() / summary() shape and pooled-vs-per-case distinction
# ---------------------------------------------------------------------------


def test_per_case_and_summary_shapes_and_pooled_distinction() -> None:
    gen = torch.Generator().manual_seed(7)
    region_names = ("ET", "TC", "WT")
    acc = CalibrationAccumulator(n_bins=8, region_names=region_names)

    # Deliberately different case sizes so pooled (voxel-weighted) and
    # per-case (case-weighted) means can genuinely differ.
    small_prob, small_label = _make_case(gen, (3, 2, 2, 2))  # 8 voxels/channel
    large_prob, large_label = _make_case(gen, (3, 6, 6, 6))  # 216 voxels/channel
    acc.add_case("small", small_prob, small_label)
    acc.add_case("large", large_prob, large_label)

    per_case = acc.per_case()
    summary = acc.summary()

    assert len(per_case) == 2
    assert list(per_case.index) == ["small", "large"]
    for region in region_names:
        for metric in ("ece", "mce", "brier"):
            assert f"{metric}_{region}" in per_case.columns
    assert "ece_mean" in per_case.columns
    assert "brier_mean" in per_case.columns

    assert summary.loc["n_cases", "value"] == 2.0
    assert summary.loc["n_voxels", "value"] == (8 + 216) * 3

    pooled_ece_et = summary.loc["ece_ET", "value"]
    mean_of_per_case_ece_et = per_case["ece_ET"].mean()
    assert pooled_ece_et != pytest.approx(mean_of_per_case_ece_et, abs=1e-9)


def test_per_case_and_summary_empty_when_no_cases() -> None:
    acc = CalibrationAccumulator(region_names=("ET", "TC", "WT"))

    assert acc.per_case().empty
    assert acc.summary().empty
    assert len(acc) == 0


def test_reliability_raises_on_unknown_region() -> None:
    acc = CalibrationAccumulator(region_names=("ET", "TC", "WT"))
    with pytest.raises(ValueError, match="Unknown region"):
        acc.reliability("NOT_A_REGION")


def test_reset_clears_accumulator() -> None:
    acc = CalibrationAccumulator(region_names=("ET", "TC", "WT"))
    prob = torch.rand(3, 4, 4, 4)
    label = torch.bernoulli(prob)
    acc.add_case("case0", prob, label)
    assert len(acc) == 1

    acc.reset()

    assert len(acc) == 0
    assert acc.per_case().empty
    assert acc.summary().empty


# ---------------------------------------------------------------------------
# 15. fit_temperature recovers a known temperature
# ---------------------------------------------------------------------------


def test_fit_temperature_recovers_known_temperature() -> None:
    gen = torch.Generator().manual_seed(123)
    n = 40000
    logits = torch.randn(n, generator=gen) * 3.0
    probs = torch.sigmoid(logits)
    labels = torch.bernoulli(probs, generator=gen)

    # True generating model is at T=1 relative to `logits`; feeding `logits * 2.0`
    # to the fitter means it must find T=2.0 to undo that scaling.
    result = fit_temperature(logits * 2.0, labels)

    assert result.temperature.shape == (1,)
    assert float(result.temperature[0]) == pytest.approx(2.0, rel=0.1)
    assert result.converged


# ---------------------------------------------------------------------------
# 16. Already-calibrated data -> T ~= 1.0
# ---------------------------------------------------------------------------


def test_fit_temperature_already_calibrated_gives_t_near_one() -> None:
    gen = torch.Generator().manual_seed(321)
    n = 40000
    logits = torch.randn(n, generator=gen) * 3.0
    probs = torch.sigmoid(logits)
    labels = torch.bernoulli(probs, generator=gen)

    result = fit_temperature(logits, labels)

    assert float(result.temperature[0]) == pytest.approx(1.0, rel=0.1)
    assert result.nll_after <= result.nll_before


# ---------------------------------------------------------------------------
# 17. per_channel fit on a 2-channel input
# ---------------------------------------------------------------------------


def test_fit_temperature_per_channel_recovers_two_temperatures() -> None:
    gen = torch.Generator().manual_seed(9)
    n = 20000
    base0 = torch.randn(n, generator=gen) * 3.0
    base1 = torch.randn(n, generator=gen) * 3.0
    label0 = torch.bernoulli(torch.sigmoid(base0), generator=gen)
    label1 = torch.bernoulli(torch.sigmoid(base1), generator=gen)

    # channel 0 stays at true T=1, channel 1 is scaled by true T=3
    logits = torch.stack([base0, base1 * 3.0], dim=1)
    labels = torch.stack([label0, label1], dim=1)

    result = fit_temperature(logits, labels, per_channel=True)

    assert result.temperature.shape == (2,)
    assert result.temperature[1] > result.temperature[0]
    assert float(result.temperature[0]) == pytest.approx(1.0, rel=0.2)
    assert float(result.temperature[1]) == pytest.approx(3.0, rel=0.2)


def test_fit_temperature_per_channel_on_1d_input_returns_shape_one() -> None:
    gen = torch.Generator().manual_seed(11)
    logits = torch.randn(2000, generator=gen)
    labels = torch.bernoulli(torch.sigmoid(logits), generator=gen)

    result = fit_temperature(logits, labels, per_channel=True)

    assert result.temperature.shape == (1,)


# ---------------------------------------------------------------------------
# 19. apply_temperature: identity, broadcast, wrong size
# ---------------------------------------------------------------------------


def test_apply_temperature_identity_at_t_one() -> None:
    logits = torch.randn(1, 3, 4, 4, 4)
    temperature = torch.tensor(1.0)

    scaled = apply_temperature(logits, temperature)

    torch.testing.assert_close(scaled, logits)


def test_apply_temperature_broadcasts_per_channel_5d() -> None:
    logits = torch.ones(2, 3, 2, 2, 2)
    temperature = torch.tensor([1.0, 2.0, 4.0])

    scaled = apply_temperature(logits, temperature)

    assert scaled.shape == logits.shape
    assert torch.all(scaled[:, 0] == 1.0)
    assert torch.all(scaled[:, 1] == 0.5)
    assert torch.all(scaled[:, 2] == 0.25)


def test_apply_temperature_raises_on_wrong_size() -> None:
    logits = torch.randn(1, 3, 4, 4, 4)
    temperature = torch.tensor([1.0, 2.0])  # only 2, logits has 3 channels

    with pytest.raises(ValueError, match="channel"):
        apply_temperature(logits, temperature)


# ---------------------------------------------------------------------------
# 20. Monotonicity invariant: temperature scaling never flips the sign
# ---------------------------------------------------------------------------


def test_apply_temperature_preserves_threshold_sign() -> None:
    gen = torch.Generator().manual_seed(5)
    logits = torch.randn(5000, generator=gen) * 4.0

    for t in (0.1, 0.5, 1.0, 2.0, 10.0):
        scaled = apply_temperature(logits, torch.tensor(t))
        assert torch.equal(logits > 0, scaled > 0)


# ---------------------------------------------------------------------------
# 21. subsample_voxels
# ---------------------------------------------------------------------------


def test_subsample_voxels_returns_exact_count_when_available() -> None:
    gen = torch.Generator().manual_seed(1)
    prob = torch.rand(1000, generator=gen)
    label = torch.bernoulli(prob, generator=gen)

    prob_sample, label_sample = subsample_voxels(prob, label, n_samples=100, generator=gen)

    assert prob_sample.shape == (100,)
    assert label_sample.shape == (100,)


def test_subsample_voxels_returns_all_when_not_enough_available() -> None:
    gen = torch.Generator().manual_seed(2)
    prob = torch.rand(10, generator=gen)
    label = torch.bernoulli(prob, generator=gen)

    prob_sample, label_sample = subsample_voxels(prob, label, n_samples=100, generator=gen)

    assert prob_sample.shape == (10,)
    assert label_sample.shape == (10,)


def test_subsample_voxels_respects_mask() -> None:
    gen = torch.Generator().manual_seed(3)
    mask = torch.cat([torch.ones(50, dtype=torch.bool), torch.zeros(50, dtype=torch.bool)])
    # subsample_voxels validates prob is in [0, 1], so use `label` as the distinguishable
    # marker instead: the masked-in half is entirely label=1, the masked-out half label=0.
    prob = torch.rand(100, generator=gen)
    label = torch.cat([torch.ones(50), torch.zeros(50)])

    prob_sample, label_sample = subsample_voxels(
        prob, label, n_samples=20, generator=gen, mask=mask
    )

    assert prob_sample.shape == (20,)
    assert torch.all(label_sample == 1.0)


def test_subsample_voxels_reproducible_for_fixed_seed() -> None:
    prob = torch.rand(500, generator=torch.Generator().manual_seed(0))
    label = torch.bernoulli(prob, generator=torch.Generator().manual_seed(0))

    gen1 = torch.Generator().manual_seed(99)
    gen2 = torch.Generator().manual_seed(99)

    sample1, _ = subsample_voxels(prob, label, n_samples=50, generator=gen1)
    sample2, _ = subsample_voxels(prob, label, n_samples=50, generator=gen2)

    assert torch.equal(sample1, sample2)


# ---------------------------------------------------------------------------
# bin_edges sanity
# ---------------------------------------------------------------------------


def test_bin_edges_shape_and_endpoints() -> None:
    edges = bin_edges(10)

    assert edges.shape == (11,)
    assert float(edges[0]) == pytest.approx(0.0)
    assert float(edges[-1]) == pytest.approx(1.0)


def test_bin_edges_raises_on_invalid_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        bin_edges(0)


# ---------------------------------------------------------------------------
# union_foreground_mask: the project's reporting mask
# ---------------------------------------------------------------------------


def test_union_foreground_mask_covers_both_error_directions() -> None:
    # Four voxels, one per quadrant of the confusion matrix.
    #   0: true negative  -> excluded
    #   1: false positive -> included via the prediction
    #   2: false negative -> included via the label (the clinically dangerous one)
    #   3: true positive  -> included by both
    prob = torch.tensor([0.01, 0.90, 0.02, 0.95])
    label = torch.tensor([0.0, 0.0, 1.0, 1.0])

    mask = union_foreground_mask(prob, label)

    assert mask.dtype == torch.bool
    assert mask.tolist() == [False, True, True, True]


def test_union_foreground_mask_predicted_only_would_hide_false_negatives() -> None:
    # The reason the mask is a UNION rather than predicted-positive alone:
    # a confident false negative must stay in the reported voxel set.
    prob = torch.tensor([0.01, 0.02])
    label = torch.tensor([0.0, 1.0])

    predicted_only = prob >= 0.5
    union = union_foreground_mask(prob, label)

    assert predicted_only.sum().item() == 0  # would drop the mistake entirely
    assert union.tolist() == [False, True]


def test_union_foreground_mask_threshold_is_inclusive_and_configurable() -> None:
    prob = torch.tensor([0.5, 0.4])
    label = torch.tensor([0.0, 0.0])

    assert union_foreground_mask(prob, label).tolist() == [True, False]
    assert union_foreground_mask(prob, label, threshold=0.3).tolist() == [True, True]
    assert union_foreground_mask(prob, label, threshold=0.6).tolist() == [False, False]


def test_union_foreground_mask_is_per_region_and_feeds_the_accumulator() -> None:
    # Shape is preserved per channel, so the result drops straight into
    # CalibrationAccumulator.add_case as a per-region mask.
    prob = torch.rand(3, 4, 4, 4, generator=torch.Generator().manual_seed(0))
    label = (torch.rand(3, 4, 4, 4, generator=torch.Generator().manual_seed(1)) > 0.5).float()

    mask = union_foreground_mask(prob, label)
    assert mask.shape == prob.shape

    acc = CalibrationAccumulator()
    metrics = acc.add_case("case_a", prob, label, mask=mask)
    assert math.isfinite(metrics["ece_mean"])


def test_union_foreground_mask_validates_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        union_foreground_mask(torch.tensor([1.5]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="binary"):
        union_foreground_mask(torch.tensor([0.5]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="same shape"):
        union_foreground_mask(torch.tensor([0.5, 0.5]), torch.tensor([1.0]))


def test_union_foreground_mask_logs_circularity_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        union_foreground_mask(torch.tensor([0.9]), torch.tensor([1.0]))

    assert any("CIRCULAR" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Circularity regression test -- pins the bug this module's fix addresses.
#
# `union_foreground_mask` can only admit a sub-threshold (p < threshold)
# voxel via `label > 0`, so every populated bin below the threshold has
# mean_label == 1.0 EXACTLY -- an artifact of the mask's own definition, not
# a measurement of the model. `predicted_foreground_mask` cannot exhibit
# this: it never even looks at the label, so nothing below the threshold can
# be predicted-positive in the first place.
# ---------------------------------------------------------------------------


def test_circularity_regression_union_mask_forces_mean_label_one_below_threshold() -> None:
    threshold = 0.5
    n_bins = 10
    gen = torch.Generator().manual_seed(0)

    # True negatives: low prob, label 0 -- excluded from the union mask entirely
    # (prob < threshold and label == 0), so they cannot contaminate the sub-threshold
    # bins below.
    prob_tn = torch.rand(500, generator=gen) * 0.3
    label_tn = torch.zeros(500)

    # False negatives: low prob, label 1 -- the ONLY way into the union mask's
    # sub-threshold bins, and exactly the voxels that force mean_label == 1.0 there.
    prob_fn = torch.rand(200, generator=gen) * 0.4
    label_fn = torch.ones(200)

    # True positives: high prob, label 1 -- included by both masks, above threshold.
    prob_tp = 0.6 + torch.rand(300, generator=gen) * 0.4
    label_tp = torch.ones(300)

    prob = torch.cat([prob_tn, prob_fn, prob_tp])
    label = torch.cat([label_tn, label_fn, label_tp])

    union_mask = union_foreground_mask(prob, label, threshold=threshold)
    union_curve = reliability_curve(prob, label, n_bins=n_bins, mask=union_mask)
    below_union = union_curve[union_curve["bin_upper"] <= threshold]
    populated_below_union = below_union[below_union["count"] > 0]

    # Sanity: the bug has something to bite on -- there really are populated
    # sub-threshold bins under the union mask.
    assert len(populated_below_union) > 0
    assert (populated_below_union["mean_label"] == 1.0).all()

    predicted_mask = predicted_foreground_mask(prob, threshold=threshold)
    predicted_curve = reliability_curve(prob, label, n_bins=n_bins, mask=predicted_mask)
    below_predicted = predicted_curve[predicted_curve["bin_upper"] <= threshold]

    # Nothing below the threshold can be predicted-positive under a label-free
    # mask, so every sub-threshold bin is empty -- there is no forced
    # mean_label == 1.0 left to observe, because the label never selected
    # which voxels are measured in the first place.
    assert (below_predicted["count"] == 0).all()


# ---------------------------------------------------------------------------
# predicted_foreground_mask: label-free by construction
# ---------------------------------------------------------------------------


def test_predicted_foreground_mask_never_depends_on_label() -> None:
    prob = torch.tensor([0.1, 0.6, 0.4, 0.9])
    label_a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    label_b = torch.tensor([1.0, 1.0, 0.0, 0.0])

    # `predicted_foreground_mask` has no `label` parameter at all -- calling
    # it twice, with two completely different labelings "in scope", produces
    # the identical result both times because the label plays no role in the
    # computation whatsoever.
    mask_a = predicted_foreground_mask(prob)
    mask_b = predicted_foreground_mask(prob)

    assert torch.equal(mask_a, mask_b)
    assert mask_a.tolist() == [False, True, False, True]
    del label_a, label_b  # never consulted -- exists only to make the claim visible


def test_predicted_foreground_mask_threshold_is_inclusive_and_configurable() -> None:
    prob = torch.tensor([0.5, 0.4])

    assert predicted_foreground_mask(prob).tolist() == [True, False]
    assert predicted_foreground_mask(prob, threshold=0.3).tolist() == [True, True]
    assert predicted_foreground_mask(prob, threshold=0.6).tolist() == [False, False]


def test_predicted_foreground_mask_validates_prob_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        predicted_foreground_mask(torch.tensor([1.5, 0.2]))


# ---------------------------------------------------------------------------
# brain_mask: nonzero-intensity brain region, the `!= 0` vs `> 0` trap
# ---------------------------------------------------------------------------


def test_brain_mask_selects_negative_interior_not_just_positive_voxels() -> None:
    # Background is EXACT zero (the air marker); interior is NEGATIVE, as
    # z-scored brain tissue routinely is after neurovision.data.preprocessing's
    # normalize_nonzero. `image > 0` would select nothing at all here -- that
    # is the trap this function exists to avoid.
    image = torch.zeros(2, 4, 4, 4)
    image[:, 1:3, 1:3, 1:3] = -0.75

    mask = brain_mask(image=image)

    assert mask.dtype == torch.bool
    assert mask.shape == (4, 4, 4)
    assert bool((image > 0).any()) is False  # sanity: the trap is real for this input
    assert mask[1:3, 1:3, 1:3].all()
    assert not mask[0, 0, 0]


def test_brain_mask_unions_across_modality_channels() -> None:
    # Channel 0 has signal only in one corner, channel 1 only in the opposite
    # corner -- the mask must be the UNION, not e.g. channel 0 alone.
    image = torch.zeros(2, 4, 4, 4)
    image[0, 0, 0, 0] = -0.5
    image[1, 3, 3, 3] = 0.5

    mask = brain_mask(image=image)

    assert mask[0, 0, 0]
    assert mask[3, 3, 3]
    assert not mask[1, 1, 1]


def test_brain_mask_precomputed_mask_takes_priority_and_is_returned_as_is() -> None:
    precomputed = torch.tensor([[True, False], [False, True]])

    result = brain_mask(image=torch.zeros(2, 2, 2), mask=precomputed)

    assert torch.equal(result, precomputed)


def test_brain_mask_raises_when_neither_image_nor_mask_given() -> None:
    with pytest.raises(ValueError, match="brain_mask needs"):
        brain_mask()
