"""Tests for neurovision.metrics.segmentation.

All tensors are tiny synthetic volumes (nothing bigger than ~(1, 3, 16, 16,
16)) so the whole file runs on CPU in about a second, matching the BraTS
region-overlap layout: channel order (ET, TC, WT), binary float masks.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from neurovision.metrics.segmentation import (
    REGION_NAMES,
    MetricAggregator,
    binarize,
    classes_to_regions,
    compute_case_metrics,
    dice_score,
    hd95,
    iou_score,
)

SHAPE = (1, 3, 8, 8, 8)


def _mask(shape: tuple[int, ...], coords: list[tuple[int, ...]]) -> torch.Tensor:
    """Builds a binary float mask with 1.0 at the given (b, c, d, h, w) coords."""
    m = torch.zeros(shape, dtype=torch.float32)
    for coord in coords:
        m[coord] = 1.0
    return m


# ---------------------------------------------------------------------------
# 1. Identical masks
# ---------------------------------------------------------------------------


def test_identical_masks_perfect_scores() -> None:
    torch.manual_seed(0)
    mask = (torch.rand(SHAPE) > 0.5).to(torch.float32)
    # Guarantee every channel has at least one foreground voxel so this is a
    # genuine non-degenerate case, not an accidental all-empty channel.
    mask[:, :, 0, 0, 0] = 1.0

    dice = dice_score(mask, mask)
    iou = iou_score(mask, mask)
    hd = hd95(mask, mask)

    assert dice.shape == (1, 3)
    assert iou.shape == (1, 3)
    assert hd.shape == (1, 3)
    assert torch.allclose(dice, torch.ones(1, 3))
    assert torch.allclose(iou, torch.ones(1, 3))
    assert torch.allclose(hd, torch.zeros(1, 3))


# ---------------------------------------------------------------------------
# 2. Disjoint non-empty masks
# ---------------------------------------------------------------------------


def test_disjoint_masks_zero_dice_iou() -> None:
    pred = _mask(SHAPE, [(0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 2, 0, 0, 0)])
    target = _mask(SHAPE, [(0, 0, 7, 7, 7), (0, 1, 7, 7, 7), (0, 2, 7, 7, 7)])

    dice = dice_score(pred, target)
    iou = iou_score(pred, target)

    assert torch.allclose(dice, torch.zeros(1, 3))
    assert torch.allclose(iou, torch.zeros(1, 3))


# ---------------------------------------------------------------------------
# 3. Half overlap: exact Dice 0.5, IoU 1/3
# ---------------------------------------------------------------------------


def test_half_overlap_exact_dice_and_iou() -> None:
    # pred covers voxels 0..3 (4 voxels), target covers voxels 2..5 (4
    # voxels) along one axis; they share voxels 2..3 (2 voxels = N/2 of each).
    # Dice = 2*|intersection| / (|pred| + |target|) = 2*2 / (4+4) = 0.5
    # IoU  = |intersection| / |union|                = 2 / (4+4-2) = 1/3
    pred = torch.zeros(1, 1, 8, 8, 8)
    pred[0, 0, 0, 0, 0:4] = 1.0
    target = torch.zeros(1, 1, 8, 8, 8)
    target[0, 0, 0, 0, 2:6] = 1.0

    dice = dice_score(pred, target)
    iou = iou_score(pred, target)

    assert dice[0, 0].item() == pytest.approx(0.5)
    assert iou[0, 0].item() == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# 4. Empty ground truth + empty prediction
# ---------------------------------------------------------------------------


def test_empty_gt_empty_pred() -> None:
    pred = torch.zeros(1, 1, 8, 8, 8)
    target = torch.zeros(1, 1, 8, 8, 8)

    dice_default = dice_score(pred, target, ignore_empty=False)
    dice_ignore = dice_score(pred, target, ignore_empty=True)
    hd = hd95(pred, target)

    assert dice_default[0, 0].item() == pytest.approx(1.0)
    assert math.isnan(dice_ignore[0, 0].item())
    assert hd[0, 0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Empty ground truth + non-empty prediction
# ---------------------------------------------------------------------------


def test_empty_gt_nonempty_pred() -> None:
    pred = torch.zeros(1, 1, 8, 8, 8)
    pred[0, 0, 0, 0, 0] = 1.0
    target = torch.zeros(1, 1, 8, 8, 8)

    dice = dice_score(pred, target, ignore_empty=False)
    hd = hd95(pred, target)

    assert dice[0, 0].item() == pytest.approx(0.0)
    assert math.isnan(hd[0, 0].item())


# ---------------------------------------------------------------------------
# 6. Non-empty ground truth + empty prediction
# ---------------------------------------------------------------------------


def test_nonempty_gt_empty_pred() -> None:
    pred = torch.zeros(1, 1, 8, 8, 8)
    target = torch.zeros(1, 1, 8, 8, 8)
    target[0, 0, 0, 0, 0] = 1.0

    dice = dice_score(pred, target, ignore_empty=False)
    hd = hd95(pred, target)

    assert dice[0, 0].item() == pytest.approx(0.0)
    assert math.isnan(hd[0, 0].item())


# ---------------------------------------------------------------------------
# 7. HD95 known geometry
# ---------------------------------------------------------------------------


def test_hd95_known_geometry() -> None:
    pred = torch.zeros(1, 1, 10, 10, 10)
    pred[0, 0, 5, 5, 5] = 1.0
    target = torch.zeros(1, 1, 10, 10, 10)
    target[0, 0, 5, 5, 8] = 1.0  # 3 voxels away on the last axis

    hd = hd95(pred, target)

    assert hd[0, 0].item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 8. classes_to_regions nesting
# ---------------------------------------------------------------------------


def test_classes_to_regions_nesting() -> None:
    label = torch.zeros(6, 6, 6, dtype=torch.int64)
    label[0, 0, 0] = 1  # necrotic core -> TC, WT only
    label[1, 1, 1] = 2  # edema -> WT only
    label[2, 2, 2] = 3  # enhancing tumor -> ET, TC, WT

    regions = classes_to_regions(label)

    assert regions.shape == (1, 3, 6, 6, 6)
    assert regions.dtype == torch.float32

    et, tc, wt = regions[0, 0], regions[0, 1], regions[0, 2]
    assert et.sum().item() == 1  # only the class-3 voxel
    assert tc.sum().item() == 2  # class 1 and class 3 voxels
    assert wt.sum().item() == 3  # class 1, 2, and 3 voxels

    # Nesting: every ET voxel is also TC and WT.
    assert torch.all(tc[et.bool()] == 1.0)
    assert torch.all(wt[et.bool()] == 1.0)
    # Every TC voxel is also WT.
    assert torch.all(wt[tc.bool()] == 1.0)


def test_classes_to_regions_raw_label_4_not_enhancing() -> None:
    # Guards against re-introducing MONAI's raw-label transform semantics:
    # this project's labels are already remapped to {0,1,2,3}, so a 4 must
    # NOT be treated as enhancing tumor.
    label = torch.zeros(4, 4, 4, dtype=torch.int64)
    label[0, 0, 0] = 4

    regions = classes_to_regions(label)

    assert regions.sum().item() == 0  # value 4 matches no region at all


# ---------------------------------------------------------------------------
# 10. binarize
# ---------------------------------------------------------------------------


def test_binarize_round_trip() -> None:
    logits = torch.tensor([-10.0, -0.01, 0.0, 0.01, 10.0])
    mask = binarize(logits, threshold=0.5)

    # sigmoid(-10) << 0.5, sigmoid(-0.01) < 0.5, sigmoid(0.0) == 0.5 (>=
    # threshold so it counts as positive), sigmoid(0.01) > 0.5, sigmoid(10) >> 0.5
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    assert torch.equal(mask, expected)
    assert mask.dtype == torch.float32


# ---------------------------------------------------------------------------
# 11. compute_case_metrics
# ---------------------------------------------------------------------------


def test_compute_case_metrics_keys_and_gt_empty() -> None:
    pred = torch.zeros(3, 8, 8, 8)
    target = torch.zeros(3, 8, 8, 8)
    # TC and WT have foreground, ET does not.
    target[1, 0, 0, 0] = 1.0
    target[2, 0, 0, 0] = 1.0
    pred[1, 0, 0, 0] = 1.0
    pred[2, 0, 0, 0] = 1.0

    metrics = compute_case_metrics(pred, target)

    expected_keys = set()
    for name in REGION_NAMES:
        expected_keys.update({f"dice_{name}", f"iou_{name}", f"hd95_{name}", f"gt_empty_{name}"})
    expected_keys.update({"dice_mean", "iou_mean", "hd95_mean"})
    assert set(metrics.keys()) == expected_keys

    assert metrics["gt_empty_ET"] == 1.0
    assert metrics["gt_empty_TC"] == 0.0
    assert metrics["gt_empty_WT"] == 0.0
    assert metrics["dice_ET"] == pytest.approx(1.0)  # empty/empty -> 1.0
    assert metrics["dice_TC"] == pytest.approx(1.0)  # identical single-voxel masks
    assert metrics["dice_WT"] == pytest.approx(1.0)


def test_compute_case_metrics_batch_of_1_channel_first() -> None:
    pred = torch.zeros(1, 3, 8, 8, 8)
    target = torch.zeros(1, 3, 8, 8, 8)
    metrics = compute_case_metrics(pred, target)
    assert metrics["dice_mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 12. MetricAggregator
# ---------------------------------------------------------------------------


def test_metric_aggregator_summary_and_per_case() -> None:
    agg = MetricAggregator(region_names=("A", "B"))
    agg.update("case1", {"dice_A": 1.0, "dice_B": 0.5})
    agg.update("case2", {"dice_A": 0.5, "dice_B": float("nan")})
    agg.update("case3", {"dice_A": 0.0, "dice_B": 0.5})

    per_case = agg.per_case()
    assert per_case.shape == (3, 2)
    assert list(per_case.index) == ["case1", "case2", "case3"]
    assert per_case.index.name == "case_id"

    summary = agg.summary()
    assert isinstance(summary, pd.DataFrame)

    # dice_A: [1.0, 0.5, 0.0] -> mean 0.5, count 3, n_missing 0
    assert summary.loc["dice_A", "mean"] == pytest.approx(0.5)
    assert summary.loc["dice_A", "median"] == pytest.approx(0.5)
    assert summary.loc["dice_A", "std"] == pytest.approx(0.5)  # sample std, ddof=1
    assert summary.loc["dice_A", "count"] == 3
    assert summary.loc["dice_A", "n_missing"] == 0

    # dice_B: [0.5, nan, 0.5] -> mean 0.5, count 2, n_missing 1
    assert summary.loc["dice_B", "mean"] == pytest.approx(0.5)
    assert summary.loc["dice_B", "median"] == pytest.approx(0.5)
    assert summary.loc["dice_B", "std"] == pytest.approx(0.0)
    assert summary.loc["dice_B", "count"] == 2
    assert summary.loc["dice_B", "n_missing"] == 1

    assert len(agg) == 3


def test_metric_aggregator_duplicate_case_id_raises() -> None:
    agg = MetricAggregator()
    agg.update("case1", {"dice_ET": 1.0})
    with pytest.raises(ValueError):
        agg.update("case1", {"dice_ET": 0.5})


def test_metric_aggregator_empty_returns_empty_dataframes() -> None:
    agg = MetricAggregator()
    assert agg.per_case().empty
    assert agg.summary().empty
    assert len(agg) == 0


def test_metric_aggregator_add_case() -> None:
    agg = MetricAggregator()
    pred = torch.zeros(3, 8, 8, 8)
    target = torch.zeros(3, 8, 8, 8)
    metrics = agg.add_case("case1", pred, target)
    assert metrics["dice_mean"] == pytest.approx(1.0)
    assert len(agg) == 1
    assert agg.per_case().loc["case1", "dice_mean"] == pytest.approx(1.0)


def test_metric_aggregator_reset() -> None:
    agg = MetricAggregator()
    agg.update("case1", {"dice_ET": 1.0})
    agg.reset()
    assert len(agg) == 0
    assert agg.per_case().empty


# ---------------------------------------------------------------------------
# 13. Shape validation
# ---------------------------------------------------------------------------


def test_dice_score_shape_mismatch_raises() -> None:
    pred = torch.zeros(1, 3, 8, 8, 8)
    target = torch.zeros(1, 3, 8, 8, 4)
    with pytest.raises(ValueError):
        dice_score(pred, target)


def test_dice_score_wrong_ndim_raises() -> None:
    pred = torch.zeros(3, 8, 8, 8)
    target = torch.zeros(3, 8, 8, 8)
    with pytest.raises(ValueError):
        dice_score(pred, target)


def test_iou_score_wrong_ndim_raises() -> None:
    pred = torch.zeros(3, 8, 8, 8)
    target = torch.zeros(3, 8, 8, 8)
    with pytest.raises(ValueError):
        iou_score(pred, target)


def test_hd95_shape_mismatch_raises() -> None:
    pred = torch.zeros(1, 3, 8, 8, 8)
    target = torch.zeros(1, 3, 8, 8, 4)
    with pytest.raises(ValueError):
        hd95(pred, target)


def test_classes_to_regions_multi_channel_raises() -> None:
    label = torch.zeros(1, 2, 6, 6, 6, dtype=torch.int64)
    with pytest.raises(ValueError):
        classes_to_regions(label)


def test_classes_to_regions_wrong_ndim_raises() -> None:
    label = torch.zeros(6, 6, dtype=torch.int64)
    with pytest.raises(ValueError):
        classes_to_regions(label)


def test_compute_case_metrics_batch_too_large_raises() -> None:
    pred = torch.zeros(2, 3, 8, 8, 8)
    target = torch.zeros(2, 3, 8, 8, 8)
    with pytest.raises(ValueError):
        compute_case_metrics(pred, target)


def test_compute_case_metrics_region_name_count_mismatch_raises() -> None:
    """Too few names must raise, not silently drop a region from the keys.

    With only two names for three channels the WT keys would go missing while
    dice_mean still averaged over all three -- a headline number that does not
    match the per-region numbers reported beside it.
    """
    pred = torch.zeros(1, 3, 8, 8, 8)
    target = torch.zeros(1, 3, 8, 8, 8)
    with pytest.raises(ValueError):
        compute_case_metrics(pred, target, region_names=("ET", "TC"))
