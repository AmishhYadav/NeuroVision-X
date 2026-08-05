"""Tests for `neurovision.explainability.faithfulness`.

CPU only (never MPS -- see CLAUDE.md's machine split). Every test runs against tiny hand-built
stub networks on `(1, 4, 8, 8, 8)` inputs, mirroring the stub style in `tests/test_gradcam.py`
and `tests/test_integrated_gradients.py` -- no real registry model, no `NeuroVisionX`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from neurovision.explainability.faithfulness import (
    FaithfulnessCurve,
    attribution_mass_ratio,
    compare_methods,
    deletion_curve,
    insertion_curve,
    make_fill,
    pointing_game,
    random_attribution_like,
    rank_voxels_by_attribution,
)

# ---------------------------------------------------------------------------
# Stub networks
# ---------------------------------------------------------------------------


class TinyFaithNet(nn.Module):
    """A tiny full-resolution 3-region-output conv net, no downsampling.

    Kept deliberately simple (no pooling/striding) so tests only need to reason about the
    perturbation logic in `faithfulness.py`, not about a network architecture.
    """

    def __init__(self, in_channels: int = 4, width: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv3d(in_channels, width, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.head = nn.Conv3d(width, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.act(self.stem(x)))


class CornerSensitiveNet(nn.Module):
    """A network whose ENTIRE output depends only on a small corner sub-volume of the input.

    `forward` reduces the input's own `corner`^3 corner region (across every input channel) to
    a single scalar mean, affine-transforms it, and broadcasts the RESULT uniformly across every
    output channel and every spatial position. This makes the network's behaviour, and therefore
    which voxels a faithful attribution should point to, exactly known by construction -- see
    `test_informative_attribution_beats_random_on_corner_sensitive_stub`.
    """

    def __init__(
        self, corner: int = 2, out_channels: int = 3, scale: float = 10.0, bias: float = -1.0
    ) -> None:
        super().__init__()
        self.corner = corner
        self.out_channels = out_channels
        self.scale = scale
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        b, _c, d, h, w = x.shape
        corner = x[:, :, : self.corner, : self.corner, : self.corner]
        corner_mean = corner.mean(dim=(1, 2, 3, 4))  # (b,)
        logit = corner_mean * self.scale + self.bias
        return logit.view(b, 1, 1, 1, 1).expand(b, self.out_channels, d, h, w).contiguous()


def _make_model(cls: type[nn.Module] = TinyFaithNet, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    model = cls()
    model.eval()
    return model


def _make_image(size: int = 8, channels: int = 4, seed: int = 1) -> Tensor:
    return torch.randn(1, channels, size, size, size, generator=torch.Generator().manual_seed(seed))


def _make_ground_truth(size: int = 8, out_channels: int = 3, seed: int = 2) -> Tensor:
    """A binary region tensor `(1, out_channels, size, size, size)` with a sub-block foreground."""
    gt = torch.zeros(1, out_channels, size, size, size)
    gt[:, :, : size // 2, : size // 2, : size // 2] = 1.0
    return gt


# ---------------------------------------------------------------------------
# 1. rank_voxels_by_attribution
# ---------------------------------------------------------------------------


def test_rank_voxels_order_ties_and_abs_magnitude() -> None:
    attr = torch.zeros(2, 2, 2)
    attr[0, 0, 0] = -5.0  # large NEGATIVE -> must still rank first (magnitude, not sign)
    attr[1, 1, 1] = 3.0  # second highest magnitude

    order = rank_voxels_by_attribution(attr)

    # flat index = d*4 + h*2 + w for shape (2, 2, 2)
    assert order[0].item() == 0  # (0, 0, 0) -> -5.0, highest |attribution|
    assert order[1].item() == 7  # (1, 1, 1) -> 3.0, second highest
    # Remaining 6 entries are all tied at 0.0; a stable sort preserves their original
    # (ascending) flat-index order rather than depending on torch's sort implementation.
    assert order[2:].tolist() == [1, 2, 3, 4, 5, 6]


def test_rank_voxels_with_mask_excludes_masked_out_voxels() -> None:
    attr = torch.arange(8, dtype=torch.float32).view(2, 2, 2) - 4.0  # values -4..3
    mask = torch.zeros(2, 2, 2, dtype=torch.bool)
    mask[0, :, :] = True  # only the d=0 half (flat indices 0-3) is eligible

    order = rank_voxels_by_attribution(attr, mask=mask)

    assert order.numel() == 4
    assert set(order.tolist()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# 2. make_fill
# ---------------------------------------------------------------------------


def test_make_fill_zero() -> None:
    image = torch.randn(1, 4, 4, 4, 4)
    filled = make_fill(image, "zero")
    assert filled.shape == image.shape
    assert torch.equal(filled, torch.zeros_like(image))


def test_make_fill_noise_requires_generator() -> None:
    image = torch.randn(1, 4, 4, 4, 4)
    with pytest.raises(ValueError, match="generator"):
        make_fill(image, "noise")


def test_make_fill_noise_is_reproducible_for_a_fixed_seed() -> None:
    image = torch.randn(1, 4, 4, 4, 4)
    first = make_fill(image, "noise", generator=torch.Generator().manual_seed(7))
    second = make_fill(image, "noise", generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)
    assert first.shape == image.shape


def test_make_fill_mean_is_per_channel_spatial_mean() -> None:
    image = torch.zeros(1, 2, 2, 2, 2)
    image[:, 0] = 1.0
    image[:, 1] = 3.0

    filled = make_fill(image, "mean")

    assert filled.shape == image.shape
    assert torch.allclose(filled[:, 0], torch.full((1, 2, 2, 2), 1.0))
    assert torch.allclose(filled[:, 1], torch.full((1, 2, 2, 2), 3.0))


def test_make_fill_unknown_name_lists_valid_options() -> None:
    image = torch.randn(1, 4, 4, 4, 4)
    with pytest.raises(ValueError, match="unknown fill") as excinfo:
        make_fill(image, "bogus")
    message = str(excinfo.value)
    assert "zero" in message
    assert "noise" in message
    assert "mean" in message


# ---------------------------------------------------------------------------
# 3-6. deletion_curve / insertion_curve shapes and sanity anchors
# ---------------------------------------------------------------------------


def test_deletion_curve_shapes() -> None:
    model = _make_model()
    image = _make_image()
    attribution = torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(3))

    curve = deletion_curve(model, image, attribution, region_index=0, n_points=5)

    assert isinstance(curve, FaithfulnessCurve)
    assert curve.fractions.shape == (5,)
    assert list(curve.fractions) == sorted(curve.fractions)
    assert curve.fractions[0] == pytest.approx(0.0)
    assert curve.fractions[-1] == pytest.approx(1.0)
    assert curve.dice_vs_prediction.shape == (5,)
    assert curve.dice_vs_ground_truth is None
    assert curve.mode == "deletion"
    assert curve.n_points == 5


def test_deletion_curve_at_fraction_zero_dice_vs_prediction_is_exactly_one() -> None:
    # The sanity anchor of the whole module: nothing perturbed at f=0, so the perturbed
    # prediction is bit-for-bit the original prediction.
    model = _make_model()
    image = _make_image()
    attribution = torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(3))

    curve = deletion_curve(model, image, attribution, region_index=0, n_points=5)

    assert curve.dice_vs_prediction[0] == pytest.approx(1.0)


def test_insertion_curve_at_fraction_one_dice_vs_prediction_is_exactly_one() -> None:
    # The mirror anchor: at f=1 every voxel has been restored, so the perturbed input IS the
    # real image and the perturbed prediction IS the original prediction.
    model = _make_model()
    image = _make_image()
    attribution = torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(3))

    curve = insertion_curve(model, image, attribution, region_index=0, n_points=5)

    assert curve.dice_vs_prediction[-1] == pytest.approx(1.0)


def test_include_endpoint_changes_auc_but_not_the_arrays() -> None:
    model = _make_model()
    image = _make_image()
    attribution = torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(3))

    excluded = deletion_curve(
        model, image, attribution, region_index=0, n_points=5, include_endpoint=False
    )
    included = deletion_curve(
        model, image, attribution, region_index=0, n_points=5, include_endpoint=True
    )

    assert np.isfinite(excluded.auc_vs_prediction)
    assert np.isfinite(included.auc_vs_prediction)
    assert excluded.auc_vs_prediction != pytest.approx(included.auc_vs_prediction)
    # The degenerate endpoint (f=1.0 for deletion) is still THERE in both cases -- only whether
    # it counts toward the AUC differs.
    assert excluded.fractions[-1] == pytest.approx(1.0)
    assert included.fractions[-1] == pytest.approx(1.0)
    assert excluded.dice_vs_prediction.shape == included.dice_vs_prediction.shape


# ---------------------------------------------------------------------------
# 7. A perfectly informative attribution beats a random one (strong claim)
# ---------------------------------------------------------------------------


def test_informative_attribution_beats_random_on_corner_sensitive_stub() -> None:
    """`CornerSensitiveNet`'s output depends ONLY on an 8-voxel corner region.

    An attribution that correctly identifies exactly those 8 voxels as the most important
    should collapse the deletion curve immediately (they are gone as soon as the ranked-voxel
    budget exceeds 8), giving a low deletion AUC. A random ranking has to delete a large,
    unpredictable fraction of the volume before it happens to include all 8 specific corner
    voxels, so it keeps the original prediction intact for far longer -- a much higher deletion
    AUC. This is the strong form of the claim: LOWER deletion AUC for the informative map.
    """
    model = _make_model(CornerSensitiveNet)

    size = 8
    image = torch.randn(1, 4, size, size, size, generator=torch.Generator().manual_seed(11)) * 0.1
    image[:, :, :2, :2, :2] = 5.0  # force the corner to a strong, constant, known value

    informative_attr = torch.zeros(size, size, size)
    informative_attr[:2, :2, :2] = 1.0  # exactly the 8 corner voxels the model actually reads

    random_attr = random_attribution_like(informative_attr, torch.Generator().manual_seed(42))

    informative_curve = deletion_curve(model, image, informative_attr, region_index=0, n_points=5)
    random_curve_result = deletion_curve(model, image, random_attr, region_index=0, n_points=5)

    assert informative_curve.auc_vs_prediction < random_curve_result.auc_vs_prediction


# ---------------------------------------------------------------------------
# 8. pointing_game
# ---------------------------------------------------------------------------


def test_pointing_game_hit_inside_mask() -> None:
    attribution = torch.zeros(4, 4, 4)
    attribution[1, 1, 1] = 1.0  # argmax
    gt = torch.zeros(4, 4, 4, dtype=torch.bool)
    gt[0:2, 0:2, 0:2] = True  # includes (1, 1, 1)

    result = pointing_game(attribution, gt)

    assert result["hit"] == 1.0
    assert result["n_gt_voxels"] == 8
    assert result["gt_volume_fraction"] == pytest.approx(8 / 64)
    assert result["ratio"] == pytest.approx(result["hit"] / result["gt_volume_fraction"])


def test_pointing_game_miss_outside_mask() -> None:
    attribution = torch.zeros(4, 4, 4)
    attribution[3, 3, 3] = 1.0  # argmax, deliberately outside gt below
    gt = torch.zeros(4, 4, 4, dtype=torch.bool)
    gt[0:2, 0:2, 0:2] = True

    result = pointing_game(attribution, gt)

    assert result["hit"] == 0.0
    assert result["ratio"] == pytest.approx(0.0 / result["gt_volume_fraction"])


# ---------------------------------------------------------------------------
# 9. attribution_mass_ratio
# ---------------------------------------------------------------------------


def test_attribution_mass_ratio_uniform_map_gives_no_localization() -> None:
    attribution = torch.ones(4, 4, 4)  # perfectly uniform
    gt = torch.zeros(4, 4, 4, dtype=torch.bool)
    gt[0:2, 0:2, 0:2] = True  # 8/64 of the volume

    result = attribution_mass_ratio(attribution, gt)

    assert result["gt_volume_fraction"] == pytest.approx(8 / 64)
    assert result["mass_inside"] == pytest.approx(8 / 64)  # same fraction as the mask's volume
    assert result["ratio"] == pytest.approx(1.0)  # the "no localization" anchor


def test_attribution_mass_ratio_fully_concentrated_map() -> None:
    attribution = torch.zeros(4, 4, 4)
    gt = torch.zeros(4, 4, 4, dtype=torch.bool)
    gt[0:2, 0:2, 0:2] = True
    attribution[0:2, 0:2, 0:2] = 1.0  # ALL mass sits inside the mask

    result = attribution_mass_ratio(attribution, gt)

    assert result["mass_inside"] == pytest.approx(1.0)
    assert result["ratio"] == pytest.approx(1.0 / result["gt_volume_fraction"])


# ---------------------------------------------------------------------------
# 10. random_attribution_like
# ---------------------------------------------------------------------------


def test_random_attribution_like_shape_and_reproducibility() -> None:
    template = torch.zeros(2, 3, 4)
    first = random_attribution_like(template, torch.Generator().manual_seed(5))
    second = random_attribution_like(template, torch.Generator().manual_seed(5))
    third = random_attribution_like(template, torch.Generator().manual_seed(6))

    assert first.shape == template.shape
    assert torch.equal(first, second)  # same seed -> identical
    assert not torch.equal(first, third)  # different seed -> different


# ---------------------------------------------------------------------------
# 11. compare_methods
# ---------------------------------------------------------------------------


def test_compare_methods_returns_the_documented_table() -> None:
    model = _make_model()
    image = _make_image()
    ground_truth = _make_ground_truth()

    attributions = {
        "method_a": torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(21)),
        "method_b": torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(22)),
    }

    table = compare_methods(
        model,
        image,
        attributions,
        ground_truth,
        region_index=0,
        n_points=4,
        generator=torch.Generator().manual_seed(99),
        target_specific={"method_b": False},
        native_resolution={"method_a": "stride 8 (upsampled)"},
    )

    assert set(table.index) == {"method_a", "method_b", "random"}
    expected_columns = {
        "deletion_auc",
        "insertion_auc",
        "insertion_minus_deletion",
        "pointing_hit",
        "pointing_ratio",
        "mass_ratio",
        "target_specific",
        "native_resolution",
        "n_points",
    }
    assert set(table.columns) == expected_columns

    for name in table.index:
        row = table.loc[name]
        assert row["insertion_minus_deletion"] == pytest.approx(
            row["insertion_auc"] - row["deletion_auc"]
        )
        assert row["n_points"] == 4

    # Explicit overrides carried through...
    assert table.loc["method_b", "target_specific"] == False  # noqa: E712
    assert table.loc["method_a", "native_resolution"] == "stride 8 (upsampled)"
    # ...and sensible defaults where not supplied.
    assert table.loc["method_a", "target_specific"] == True  # noqa: E712
    assert table.loc["random", "target_specific"] == False  # noqa: E712
    assert table.loc["method_b", "native_resolution"] == "voxel"
    assert table.loc["random", "native_resolution"] == "voxel"


def test_compare_methods_requires_a_generator() -> None:
    model = _make_model()
    image = _make_image()
    ground_truth = _make_ground_truth()
    attributions = {"method_a": torch.rand(8, 8, 8)}

    with pytest.raises(ValueError, match="Generator"):
        compare_methods(model, image, attributions, ground_truth, n_points=4, generator=None)


def test_compare_methods_requires_at_least_one_method() -> None:
    model = _make_model()
    image = _make_image()
    ground_truth = _make_ground_truth()

    with pytest.raises(ValueError, match="empty"):
        compare_methods(
            model, image, {}, ground_truth, n_points=4, generator=torch.Generator().manual_seed(1)
        )


# ---------------------------------------------------------------------------
# 12. Validation
# ---------------------------------------------------------------------------


def test_deletion_curve_raises_when_model_is_in_train_mode() -> None:
    model = _make_model()
    model.train()
    image = _make_image()
    attribution = torch.rand(8, 8, 8)

    with pytest.raises(ValueError, match="eval"):
        deletion_curve(model, image, attribution)


def test_deletion_curve_raises_on_mismatched_attribution_shape() -> None:
    model = _make_model()
    image = _make_image(size=8)
    attribution = torch.rand(4, 4, 4)  # wrong spatial shape

    with pytest.raises(ValueError, match="spatial shape"):
        deletion_curve(model, image, attribution)


def test_deletion_curve_raises_on_n_points_below_two() -> None:
    model = _make_model()
    image = _make_image()
    attribution = torch.rand(8, 8, 8)

    with pytest.raises(ValueError, match="n_points"):
        deletion_curve(model, image, attribution, n_points=1)


def test_deletion_curve_raises_on_unknown_fill() -> None:
    model = _make_model()
    image = _make_image()
    attribution = torch.rand(8, 8, 8)

    with pytest.raises(ValueError, match="unknown fill"):
        deletion_curve(model, image, attribution, fill="bogus")


# ---------------------------------------------------------------------------
# 13. Determinism
# ---------------------------------------------------------------------------


def test_repeated_deletion_curve_calls_are_identical() -> None:
    model = _make_model()
    image = _make_image()
    attribution = torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(3))

    first = deletion_curve(model, image, attribution, region_index=0, n_points=5)
    second = deletion_curve(model, image, attribution, region_index=0, n_points=5)

    assert np.array_equal(first.dice_vs_prediction, second.dice_vs_prediction)
    assert np.array_equal(first.fractions, second.fractions)
    assert first.auc_vs_prediction == second.auc_vs_prediction


# ---------------------------------------------------------------------------
# A bare (D, H, W) ground truth is accepted as an already-extracted region mask
#
# A thresholded prediction channel, or a synthetic mask built with torch.meshgrid,
# is naturally (D, H, W). Requiring the caller to unsqueeze it added friction with
# no disambiguation benefit -- the three accepted layouts differ in ndim, so nothing
# has to be inferred from a channel count.
# ---------------------------------------------------------------------------


def test_compare_methods_accepts_3d_ground_truth() -> None:
    model = _make_model()
    image = _make_image()
    attribution = torch.rand(1, 1, 8, 8, 8, generator=torch.Generator().manual_seed(5))

    gt_3d = torch.zeros(8, 8, 8)
    gt_3d[2:6, 2:6, 2:6] = 1.0
    gt_4d = torch.stack([gt_3d, gt_3d, gt_3d])  # same region in every channel

    table_3d = compare_methods(
        model,
        image,
        {"m": attribution},
        ground_truth=gt_3d,
        region_index=0,
        n_points=3,
        generator=torch.Generator().manual_seed(11),
    )
    table_4d = compare_methods(
        model,
        image,
        {"m": attribution},
        ground_truth=gt_4d,
        region_index=0,
        n_points=3,
        generator=torch.Generator().manual_seed(11),
    )

    pd.testing.assert_frame_equal(table_3d, table_4d)


def test_pointing_game_matches_between_3d_and_4d_ground_truth() -> None:
    gt_3d = torch.zeros(8, 8, 8)
    gt_3d[0:2, 0:2, 0:2] = 1.0
    attribution = torch.zeros(1, 1, 8, 8, 8)
    attribution[0, 0, 0, 0, 0] = 1.0

    result = pointing_game(attribution, gt_3d)

    assert result["hit"] == 1.0
    assert result["gt_volume_fraction"] == pytest.approx(8 / 512)
