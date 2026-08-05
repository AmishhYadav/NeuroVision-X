"""Tests for `neurovision.explainability.gradcam`.

CPU only (never MPS -- see CLAUDE.md's machine split). Almost every test runs
against a tiny hand-built stub network so the file stays in the low seconds;
exactly ONE test builds a real registry model, and it is deliberately the only
slow thing here.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import Tensor, nn

from neurovision.explainability.gradcam import (
    GradCAMOutput,
    available_layers,
    center_patch_on_mask,
    grad_cam,
    resolve_layer,
)

# ---------------------------------------------------------------------------
# Stub networks
# ---------------------------------------------------------------------------


class TinySegNet(nn.Module):
    """A 3-level conv net emitting 3 logit channels, deliberately tiny.

    `stem` keeps full resolution, `mid` halves it, `head` maps back to 3
    region channels. `mid` is the interesting Grad-CAM target: it has its own
    spatial resolution, so a CAM taken there genuinely needs upsampling.
    """

    def __init__(self, in_channels: int = 4, width: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv3d(in_channels, width, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.mid = nn.Conv3d(width, width, kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.head = nn.Conv3d(width, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.stem(x))
        x = self.act(self.mid(x))
        return self.head(self.up(x))


class AllNegativeNet(TinySegNet):
    """Same as `TinySegNet` but with a large negative head bias.

    Every sigmoid output sits far below 0.5, so `grad_cam`'s default
    predicted-positive mask comes back empty and the whole-volume fallback
    path is exercised.
    """

    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.head.bias.fill_(-50.0)


class RaisingNet(TinySegNet):
    """Raises inside `forward`, AFTER the hooked `mid` layer has already run.

    This is what makes the hook-cleanup-on-exception test meaningful: the
    hook has fired and captured an activation by the time the exception is
    raised, so a `finally`-less implementation really would leave it attached.
    """

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.stem(x))
        x = self.act(self.mid(x))  # hook fires here
        raise RuntimeError("sentinel failure after the hooked layer ran")


def _make_model(cls: type[nn.Module] = TinySegNet) -> nn.Module:
    torch.manual_seed(0)
    model = cls()
    model.eval()
    return model


def _make_image(size: int = 16, channels: int = 4) -> Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(1, channels, size, size, size, generator=generator)


# ---------------------------------------------------------------------------
# 1-4. Shapes, normalization, relu, upsample
# ---------------------------------------------------------------------------


def test_shapes_match_input_and_target_layer() -> None:
    model = _make_model()
    image = _make_image(16)

    out = grad_cam(model, image, target_layer="mid", region_index=0)

    assert isinstance(out, GradCAMOutput)
    assert out.cam.shape == (1, 1, 16, 16, 16)  # upsampled to the input
    assert out.raw_cam.shape == (1, 1, 8, 8, 8)  # `mid` is stride 2
    assert out.channel_weights.shape == (6,)  # one alpha per `mid` output channel
    assert out.target_layer == "mid"
    assert out.region_index == 0


def test_normalize_gives_unit_range() -> None:
    model = _make_model()
    image = _make_image(16)

    out = grad_cam(model, image, target_layer="mid", normalize=True)

    assert float(out.cam.min()) >= 0.0
    assert float(out.cam.max()) == pytest.approx(1.0)


def test_relu_flag_changes_the_map_and_only_the_signed_one_has_negatives() -> None:
    model = _make_model()
    image = _make_image(16)

    relued = grad_cam(model, image, target_layer="mid", relu=True, normalize=False)
    signed = grad_cam(model, image, target_layer="mid", relu=False, normalize=False)

    assert float(relued.raw_cam.min()) >= 0.0
    # The signed map must genuinely contain evidence AGAINST the region, or this
    # test proves nothing about the two paths differing.
    assert float(signed.raw_cam.min()) < 0.0
    assert not torch.equal(relued.raw_cam, signed.raw_cam)


def test_upsample_false_keeps_target_layer_resolution() -> None:
    model = _make_model()
    image = _make_image(16)

    out = grad_cam(model, image, target_layer="mid", upsample=False)

    assert out.cam.shape == (1, 1, 8, 8, 8)
    assert out.cam.shape == out.raw_cam.shape


# ---------------------------------------------------------------------------
# 5. No parameter-gradient pollution (autograd.grad, not .backward)
# ---------------------------------------------------------------------------


def test_does_not_pollute_parameter_grads() -> None:
    model = _make_model()
    image = _make_image(16)
    assert all(p.grad is None for p in model.parameters())

    grad_cam(model, image, target_layer="mid")

    # .backward() would have populated every one of these. Stale grads surviving
    # into a later training step is silent and would corrupt the first optimizer
    # step after an explainability run.
    assert all(p.grad is None for p in model.parameters())


# ---------------------------------------------------------------------------
# 6-7. Hook cleanup, on both the happy path and the exception path
# ---------------------------------------------------------------------------


def test_hook_removed_on_success() -> None:
    model = _make_model()
    image = _make_image(16)

    grad_cam(model, image, target_layer="mid")

    assert len(resolve_layer(model, "mid")._forward_hooks) == 0


def test_hook_removed_when_forward_raises() -> None:
    model = _make_model(RaisingNet)
    image = _make_image(16)

    with pytest.raises(RuntimeError, match="sentinel failure"):
        grad_cam(model, image, target_layer="mid")

    # A hook left attached here would silently alter every later forward pass
    # of this model object, with nothing anywhere reporting it.
    assert len(resolve_layer(model, "mid")._forward_hooks) == 0


# ---------------------------------------------------------------------------
# 8. Works inside an enclosing no_grad context (the expected caller situation)
# ---------------------------------------------------------------------------


def test_works_under_enclosing_no_grad() -> None:
    model = _make_model()
    image = _make_image(16)

    with torch.no_grad():
        out = grad_cam(model, image, target_layer="mid")

    assert torch.isfinite(out.cam).all()


# ---------------------------------------------------------------------------
# 9-10. Validation
# ---------------------------------------------------------------------------


def test_raises_when_model_is_in_train_mode() -> None:
    model = _make_model()
    model.train()
    image = _make_image(16)

    with pytest.raises(ValueError, match="eval"):
        grad_cam(model, image, target_layer="mid")


def test_raises_on_out_of_range_region_index() -> None:
    model = _make_model()
    image = _make_image(16)

    with pytest.raises(ValueError, match="region_index"):
        grad_cam(model, image, target_layer="mid", region_index=7)


def test_unknown_layer_error_names_a_real_candidate() -> None:
    model = _make_model()

    with pytest.raises(ValueError, match="No submodule named") as excinfo:
        resolve_layer(model, "mid.nope")

    message = str(excinfo.value)
    assert "mid" in message
    assert "available_layers" in message


def test_grad_cam_propagates_unknown_layer_error() -> None:
    model = _make_model()
    image = _make_image(16)

    with pytest.raises(ValueError, match="No submodule named"):
        grad_cam(model, image, target_layer="does_not_exist")


def test_accepts_unbatched_input() -> None:
    model = _make_model()
    image = _make_image(16)[0]  # (C, D, H, W)

    out = grad_cam(model, image, target_layer="mid")

    assert out.cam.shape == (1, 1, 16, 16, 16)


# ---------------------------------------------------------------------------
# 11-12. Target mask behaviour
# ---------------------------------------------------------------------------


def test_explicit_target_mask_changes_the_result_and_is_counted() -> None:
    model = _make_model()
    image = _make_image(16)

    default_out = grad_cam(model, image, target_layer="mid")

    mask = torch.zeros(16, 16, 16, dtype=torch.bool)
    mask[:4, :4, :4] = True
    masked_out = grad_cam(model, image, target_layer="mid", target_mask=mask)

    assert masked_out.n_target_voxels == 64
    assert not torch.equal(default_out.cam, masked_out.cam)


def test_empty_predicted_mask_falls_back_to_whole_volume(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _make_model(AllNegativeNet)
    image = _make_image(16)

    with caplog.at_level(logging.WARNING):
        out = grad_cam(model, image, target_layer="mid")

    assert out.n_target_voxels == 16 * 16 * 16
    assert "falling back to the whole spatial extent" in caplog.text


# ---------------------------------------------------------------------------
# 13. center_patch_on_mask
# ---------------------------------------------------------------------------


def test_center_patch_clamps_inside_the_volume() -> None:
    image = torch.randn(1, 4, 20, 20, 20)
    mask = torch.zeros(20, 20, 20, dtype=torch.bool)
    mask[0, 0, 0] = True  # corner: an unclamped patch would start at a negative index

    patch, slices = center_patch_on_mask(image, mask, (8, 8, 8))

    assert patch.shape == (1, 4, 8, 8, 8)
    assert all(s.start == 0 and s.stop == 8 for s in slices)
    assert torch.equal(patch, image[:, :, slices[0], slices[1], slices[2]])


def test_center_patch_centres_on_the_mask() -> None:
    image = torch.randn(1, 4, 20, 20, 20)
    mask = torch.zeros(20, 20, 20, dtype=torch.bool)
    mask[10, 10, 10] = True

    _, slices = center_patch_on_mask(image, mask, (8, 8, 8))

    assert [s.start for s in slices] == [6, 6, 6]  # 10 - 8 // 2


def test_center_patch_empty_mask_falls_back_to_centre(
    caplog: pytest.LogCaptureFixture,
) -> None:
    image = torch.randn(1, 4, 20, 20, 20)
    mask = torch.zeros(20, 20, 20, dtype=torch.bool)

    with caplog.at_level(logging.WARNING):
        patch, slices = center_patch_on_mask(image, mask, (8, 8, 8))

    assert patch.shape == (1, 4, 8, 8, 8)
    assert [s.start for s in slices] == [6, 6, 6]  # 20 // 2 - 8 // 2
    assert "entirely empty" in caplog.text


def test_center_patch_raises_when_patch_exceeds_volume() -> None:
    image = torch.randn(1, 4, 20, 8, 20)
    mask = torch.ones(20, 8, 20, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"axis H"):
        center_patch_on_mask(image, mask, (8, 16, 8))


def test_center_patch_keeps_the_input_layout() -> None:
    image = torch.randn(4, 20, 20, 20)  # unbatched
    mask = torch.ones(20, 20, 20, dtype=torch.bool)

    patch, _ = center_patch_on_mask(image, mask, (8, 8, 8))

    assert patch.shape == (4, 8, 8, 8)


def test_center_patch_output_feeds_grad_cam() -> None:
    # The documented end-to-end workflow: crop around the prediction, explain the crop.
    model = _make_model()
    image = torch.randn(1, 4, 20, 20, 20, generator=torch.Generator().manual_seed(2))
    mask = torch.zeros(20, 20, 20, dtype=torch.bool)
    mask[8:12, 8:12, 8:12] = True

    patch, _ = center_patch_on_mask(image, mask, (16, 16, 16))
    out = grad_cam(model, patch, target_layer="mid")

    assert out.cam.shape == (1, 1, 16, 16, 16)


# ---------------------------------------------------------------------------
# 14. available_layers
# ---------------------------------------------------------------------------


def test_available_layers_all_resolve() -> None:
    model = _make_model()

    names = available_layers(model)

    assert names  # non-empty
    for name in names:
        assert isinstance(resolve_layer(model, name), nn.Module)


def test_available_layers_excludes_parameterless_activations() -> None:
    model = _make_model()

    names = available_layers(model)

    assert "act" not in names  # a bare ReLU carries no feature-map information
    assert "mid" in names


def test_available_layers_max_depth_filters() -> None:
    model = _make_model()

    shallow = available_layers(model, max_depth=1)

    assert all("." not in name for name in shallow)


# ---------------------------------------------------------------------------
# 15. Determinism
# ---------------------------------------------------------------------------


def test_repeated_calls_are_bitwise_identical() -> None:
    model = _make_model()
    image = _make_image(16)

    first = grad_cam(model, image, target_layer="mid")
    second = grad_cam(model, image, target_layer="mid")

    assert torch.equal(first.cam, second.cam)
    assert first.target_score == second.target_score


# ---------------------------------------------------------------------------
# 16. One integration test against a real registry model
# ---------------------------------------------------------------------------


def test_integration_with_real_unet3d() -> None:
    """The only test here that builds a real model; deliberately the slowest.

    `NeuroVisionX` is NOT used: its Swin branch requires >= 64 voxels on every
    axis (InstanceNorm3d raises at a 1^3 bottleneck below that), and a
    forward+backward at 64^3 takes ~5.5s on CPU -- far over this suite's budget.
    """
    from omegaconf import OmegaConf

    from neurovision.models import baseline  # noqa: F401  (registers "unet3d")
    from neurovision.models.registry import build_model

    # Mirrors tests/test_models.py::_unet3d_cfg, but at a fraction of the
    # production widths -- this test only needs a REAL registry-built model,
    # not a realistically sized one.
    cfg = OmegaConf.create(
        {
            "data": {"in_channels": 4, "num_classes": 3},
            "model": {
                "name": "unet3d",
                "in_channels": 4,
                "out_channels": 3,
                "channels": [8, 16, 32],
                "strides": [2, 2],
                "num_res_units": 1,
                "norm": "instance",
                "activation": "leakyrelu",
                "dropout": 0.0,
                "deep_supervision": False,
            },
        }
    )
    model = build_model(cfg)
    model.eval()

    names = available_layers(model)
    assert names
    image = torch.randn(1, 4, 32, 32, 32, generator=torch.Generator().manual_seed(3))

    out = grad_cam(model, image, target_layer=names[-1], region_index=0)

    assert out.cam.shape == (1, 1, 32, 32, 32)
    assert torch.isfinite(out.cam).all()
