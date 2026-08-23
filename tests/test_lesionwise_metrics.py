"""Tests for neurovision.metrics.lesionwise.

Synthetic arrays only (nothing bigger than 40 voxels a side), so the whole file
runs on CPU in well under a second. This file is meant to run in the SEPARATE
`.venv-analysis` virtualenv (`.venv-analysis/bin/python -m pytest
tests/test_lesionwise_metrics.py`), since `panoptica` is not installed in the
project's main training `.venv`. `pytest.importorskip` below makes it skip
cleanly, rather than error, if run in the wrong environment.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

panoptica = pytest.importorskip("panoptica")

from neurovision.metrics.lesionwise import (  # noqa: E402
    LESIONWISE_METRIC_PREFIXES,
    lesionwise_case_metrics,
)
from neurovision.metrics.segmentation import REGION_NAMES, dice_score  # noqa: E402

_ET = REGION_NAMES.index("ET")


def _empty_case(shape: tuple[int, int, int] = (16, 16, 16)) -> tuple[torch.Tensor, torch.Tensor]:
    """A fresh all-zero (C, D, H, W) pred/target pair, one channel per region."""
    pred = torch.zeros((len(REGION_NAMES), *shape), dtype=torch.float32)
    target = torch.zeros((len(REGION_NAMES), *shape), dtype=torch.float32)
    return pred, target


# ---------------------------------------------------------------------------
# 1-3. The basic edge cases every region must resolve to a definite number.
# ---------------------------------------------------------------------------


def test_perfect_prediction_scores_one() -> None:
    pred, target = _empty_case((16, 16, 16))
    for idx in range(len(REGION_NAMES)):
        # Two well-separated lesions, each above the default min_lesion_voxels.
        target[idx, 1:5, 1:5, 1:5] = 1.0  # 4^3 = 64 voxels
        target[idx, 9:13, 9:13, 9:13] = 1.0  # 4^3 = 64 voxels
        pred[idx] = target[idx]

    metrics = lesionwise_case_metrics(pred, target)

    for region in REGION_NAMES:
        assert metrics[f"lwdice_{region}"] == pytest.approx(1.0)
        assert metrics[f"lwnsd_{region}"] == pytest.approx(1.0)
        assert metrics[f"lwf1_{region}"] == pytest.approx(1.0)
        assert metrics[f"lwtp_{region}"] == 2.0
        assert metrics[f"lwfp_{region}"] == 0.0
        assert metrics[f"lwfn_{region}"] == 0.0


def test_both_empty_region_scores_one() -> None:
    pred, target = _empty_case((10, 10, 10))
    metrics = lesionwise_case_metrics(pred, target)

    for region in REGION_NAMES:
        assert metrics[f"lwdice_{region}"] == pytest.approx(1.0)
        assert metrics[f"lwnsd_{region}"] == pytest.approx(1.0)
        assert metrics[f"lwf1_{region}"] == pytest.approx(1.0)
        assert metrics[f"lwtp_{region}"] == 0.0
        assert metrics[f"lwfp_{region}"] == 0.0
        assert metrics[f"lwfn_{region}"] == 0.0


def test_empty_prediction_scores_zero() -> None:
    pred, target = _empty_case((16, 16, 16))
    target[_ET, 2:8, 2:8, 2:8] = 1.0  # 6^3 = 216 voxels, well above the default filter

    metrics = lesionwise_case_metrics(pred, target)

    assert metrics["lwdice_ET"] == pytest.approx(0.0)
    assert metrics["lwnsd_ET"] == pytest.approx(0.0)
    assert metrics["lwf1_ET"] == pytest.approx(0.0)
    assert metrics["lwtp_ET"] == 0.0
    assert metrics["lwfp_ET"] == 0.0
    assert metrics["lwfn_ET"] == 1.0


def test_empty_reference_scores_zero() -> None:
    pred, target = _empty_case((16, 16, 16))
    pred[_ET, 2:8, 2:8, 2:8] = 1.0  # 6^3 = 216 voxels

    metrics = lesionwise_case_metrics(pred, target)

    assert metrics["lwdice_ET"] == pytest.approx(0.0)
    assert metrics["lwnsd_ET"] == pytest.approx(0.0)
    assert metrics["lwf1_ET"] == pytest.approx(0.0)
    assert metrics["lwtp_ET"] == 0.0
    assert metrics["lwfp_ET"] == 1.0
    assert metrics["lwfn_ET"] == 0.0


# ---------------------------------------------------------------------------
# 4. The definition, verified against a hand computation.
# ---------------------------------------------------------------------------


def test_hand_computed_one_matched_one_missed_one_spurious() -> None:
    pred, target = _empty_case((30, 30, 30))

    # Ref A / Pred A: same 5x5x5 cube, prediction shifted by 1 voxel along the
    # last axis -> IoU = 100 / 150 = 0.667 > the 0.5 default threshold, matched.
    target[_ET, 2:7, 2:7, 2:7] = 1.0  # 125 voxels
    pred[_ET, 2:7, 2:7, 3:8] = 1.0  # 125 voxels, overlap = 5*5*4 = 100 voxels

    # Ref B: nothing predicted there -> a false negative.
    target[_ET, 15:20, 15:20, 15:20] = 1.0  # 125 voxels

    # Pred C: nothing in the reference there -> a false positive.
    pred[_ET, 22:27, 22:27, 22:27] = 1.0  # 125 voxels

    metrics = lesionwise_case_metrics(pred, target)

    assert metrics["lwtp_ET"] == 1.0
    assert metrics["lwfp_ET"] == 1.0
    assert metrics["lwfn_ET"] == 1.0
    assert metrics["lwf1_ET"] == pytest.approx(0.5)  # 2*1 / (2*1 + 1 + 1)

    overlap = 5 * 5 * 4
    a = b = 5 * 5 * 5
    matched_dice = 2 * overlap / (a + b)  # hand-computed Dice of the matched pair
    assert metrics["lwdice_ET"] == pytest.approx(matched_dice / 3)  # tp / (tp+fp+fn) = 1/3


# ---------------------------------------------------------------------------
# 5. The entire reason this module exists.
# ---------------------------------------------------------------------------


def test_missed_lesion_costs_lesionwise_far_more_than_voxelwise() -> None:
    pred, target = _empty_case((40, 40, 40))

    # A big lesion, predicted perfectly.
    target[_ET, 2:22, 2:22, 2:22] = 1.0  # 20^3 = 8000 voxels
    pred[_ET, 2:22, 2:22, 2:22] = 1.0

    # A small satellite lesion the model never predicts at all.
    target[_ET, 30:35, 30:35, 30:34] = 1.0  # 5*5*4 = 100 voxels

    metrics = lesionwise_case_metrics(pred, target)

    voxelwise_dice = float(dice_score(pred.unsqueeze(0), target.unsqueeze(0))[0, _ET])
    assert voxelwise_dice > 0.95  # missing 100/8100 voxels barely moves voxel Dice

    # tp=1 (the big lesion), fn=1 (the missed satellite) -> sq_dsc(=1.0) * 1/2
    assert metrics["lwdice_ET"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# 6. The small-lesion filter.
# ---------------------------------------------------------------------------


def test_min_lesion_voxels_drops_subthreshold_lesions() -> None:
    pred, target = _empty_case((16, 16, 16))
    target[_ET, 2:4, 2:4, 2:4] = 1.0  # 2^3 = 8 voxels, missed entirely by pred

    metrics_50 = lesionwise_case_metrics(pred, target, min_lesion_voxels=50)
    assert metrics_50["lwfn_ET"] == 0.0  # 8 < 50 -> dropped before matching

    metrics_5 = lesionwise_case_metrics(pred, target, min_lesion_voxels=5)
    assert metrics_5["lwfn_ET"] == 1.0  # 8 >= 5 -> kept, and missed


# ---------------------------------------------------------------------------
# 7-8. Spacing and the NSD tolerance.
# ---------------------------------------------------------------------------


def test_nsd_tolerance_is_applied_not_panoptica_default() -> None:
    pred, target = _empty_case((16, 16, 16))
    target[_ET, 4:9, 4:9, 4:9] = 1.0  # 5^3 = 125 voxels
    pred[_ET, 4:9, 4:9, 5:10] = 1.0  # same cube, shifted 1 voxel; IoU = 0.667, matched

    metrics_loose = lesionwise_case_metrics(
        pred, target, spacing=(1.0, 1.0, 1.0), nsd_tolerance_mm=1.0
    )
    metrics_tight = lesionwise_case_metrics(
        pred, target, spacing=(1.0, 1.0, 1.0), nsd_tolerance_mm=0.1
    )

    # tp=1, fp=fn=0 here, so lwnsd == sq_nsd exactly -- this isolates the
    # threshold's effect from the tp/(tp+fp+fn) detection penalty.
    assert metrics_loose["lwnsd_ET"] > metrics_tight["lwnsd_ET"]
    assert metrics_loose["lwnsd_ET"] > 0.9


def test_anisotropic_spacing_changes_nsd() -> None:
    # A 1-voxel shift along axis 0 saturates NSD at 1.0 under isotropic spacing
    # (the shift is within the default 1mm tolerance on every axis), so this
    # compares isotropic against an anisotropic spacing that stretches exactly
    # the shifted axis -- otherwise two isotropic spacings would both give 1.0
    # and prove nothing.
    pred, target = _empty_case((30, 30, 30))
    target[_ET, 5:15, 5:15, 5:15] = 1.0
    pred[_ET, 6:16, 5:15, 5:15] = 1.0  # shifted by 1 voxel along axis 0

    metrics_iso = lesionwise_case_metrics(pred, target, spacing=(1.0, 1.0, 1.0))
    metrics_aniso = lesionwise_case_metrics(pred, target, spacing=(3.0, 1.0, 1.0))

    assert metrics_iso["lwnsd_ET"] == pytest.approx(1.0)
    assert metrics_aniso["lwnsd_ET"] == pytest.approx(0.7213114754098361)
    assert metrics_iso["lwnsd_ET"] != pytest.approx(metrics_aniso["lwnsd_ET"])


def test_disjoint_lesions_score_zero_without_raising() -> None:
    """A tp==0 case from two NON-empty, non-overlapping masks must not raise.

    This is a distinct edge case from the three empty-mask cases above: both
    masks have real lesions in them, connected-component labelling and IoU
    matching run for real, and matching simply finds no pair above threshold.
    """
    pred, target = _empty_case((30, 30, 30))
    target[_ET, 2:7, 2:7, 2:7] = 1.0  # 125 voxels
    pred[_ET, 20:25, 20:25, 20:25] = 1.0  # 125 voxels, IoU = 0.0 with the above

    metrics = lesionwise_case_metrics(pred, target)  # must not raise

    assert metrics["lwdice_ET"] == 0.0
    assert metrics["lwnsd_ET"] == 0.0
    assert metrics["lwf1_ET"] == 0.0
    assert metrics["lwtp_ET"] == 0.0
    assert metrics["lwfp_ET"] == 1.0
    assert metrics["lwfn_ET"] == 1.0


# ---------------------------------------------------------------------------
# 9. Shape and region-name validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pred, target, region_names",
    [
        pytest.param(torch.zeros(8, 8, 8), torch.zeros(3, 8, 8, 8), REGION_NAMES, id="ndim3"),
        pytest.param(
            torch.zeros(2, 3, 8, 8, 8),
            torch.zeros(2, 3, 8, 8, 8),
            REGION_NAMES,
            id="batch_of_2",
        ),
        pytest.param(
            torch.zeros(3, 8, 8, 8),
            torch.zeros(3, 9, 9, 9),
            REGION_NAMES,
            id="shape_mismatch",
        ),
        pytest.param(
            torch.zeros(3, 8, 8, 8),
            torch.zeros(3, 8, 8, 8),
            ("ET", "TC"),
            id="region_names_wrong_length",
        ),
    ],
)
def test_shape_and_region_name_validation(pred, target, region_names) -> None:
    with pytest.raises(ValueError):
        lesionwise_case_metrics(pred, target, region_names=region_names)


# ---------------------------------------------------------------------------
# 10. 4-D vs 5-D input.
# ---------------------------------------------------------------------------


def test_accepts_five_dim_and_four_dim_input_identically() -> None:
    pred, target = _empty_case((10, 10, 10))
    target[_ET, 2:7, 2:7, 2:7] = 1.0
    pred[_ET, 2:7, 2:7, 2:7] = 1.0

    metrics_4d = lesionwise_case_metrics(pred, target)
    metrics_5d = lesionwise_case_metrics(pred.unsqueeze(0), target.unsqueeze(0))

    assert metrics_4d == metrics_5d


# ---------------------------------------------------------------------------
# 11. The exact column set.
# ---------------------------------------------------------------------------


def test_returns_exact_column_set() -> None:
    pred, target = _empty_case((10, 10, 10))
    target[_ET, 2:7, 2:7, 2:7] = 1.0
    pred[_ET, 2:7, 2:7, 3:8] = 1.0

    metrics = lesionwise_case_metrics(pred, target)

    expected_keys = {
        f"{prefix}_{region}" for prefix in LESIONWISE_METRIC_PREFIXES for region in REGION_NAMES
    } | {f"{prefix}_mean" for prefix in ("lwdice", "lwnsd", "lwf1")}

    assert set(metrics.keys()) == expected_keys
    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} is not a float: {value!r}"
        assert not math.isnan(value), f"{key} is NaN"


# ---------------------------------------------------------------------------
# 12. Speed.
# ---------------------------------------------------------------------------


def test_runs_on_a_32_cubed_case_under_a_second() -> None:
    pred, target = _empty_case((32, 32, 32))
    for idx in range(len(REGION_NAMES)):
        target[idx, 4:12, 4:12, 4:12] = 1.0  # matched, shifted
        pred[idx, 4:12, 4:12, 5:13] = 1.0
        target[idx, 20:24, 20:24, 20:24] = 1.0  # a missed satellite

    start = time.perf_counter()
    lesionwise_case_metrics(pred, target)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
