"""Tests for `neurovision.inference.sliding_window`.

Uses a tiny 1x1x1-conv stub network, never a real UNet/SwinUNETR, so the
whole file runs on CPU in well under a second. The config is composed
through the REAL Hydra config tree (`hydra.compose` against `configs/`), not
hand-built, so these tests also prove `configs/inference/default.yaml`
actually composes and its `${data.patch_size}` interpolation resolves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from torch import Tensor, nn

from neurovision.inference.sliding_window import build_inferer, sliding_window_predict

CPU = torch.device("cpu")
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


class _TinyNet(nn.Module):
    """4 -> 3 channel 1x1x1 conv. Shape-preserving, instant on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 3, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def _compose_config(tmp_path: Path, overrides: list[str] | None = None) -> Any:
    """Composes the real Hydra config, with the mandatory `data.root_dir` set.

    `data.root_dir` is `???` in `configs/data/brats.yaml`, so it must be
    supplied for the config to compose at all, even though this test never
    reads real data from it.
    """
    all_overrides = [f"data.root_dir={tmp_path}", "device=cpu"] + (overrides or [])
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        return hydra.compose(config_name="config", overrides=all_overrides)


def test_output_shape_matches_input(tmp_path: Path) -> None:
    """roi_size=32 on a 40^3 volume still returns the 40^3 input shape."""
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[32,32,32]"])
    model = _TinyNet()
    image = torch.randn(1, 4, 40, 40, 40)

    result = sliding_window_predict(model, image, cfg, CPU)

    assert result.shape == (1, 3, 40, 40, 40)


def test_volume_smaller_than_roi_on_one_axis(tmp_path: Path) -> None:
    """A 24-voxel axis under a 32^3 roi must still return the input's shape.

    MONAI pads internally to fit the window, then crops the output back to
    the original size -- this checks that crop actually happens rather than
    leaking the padded shape out.
    """
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[32,32,32]"])
    model = _TinyNet()
    image = torch.randn(1, 4, 24, 40, 40)

    result = sliding_window_predict(model, image, cfg, CPU)

    assert result.shape == (1, 3, 24, 40, 40)


def test_output_dtype_is_float32(tmp_path: Path) -> None:
    """Output is float32 regardless of AMP, since AMP is off on CPU anyway."""
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[32,32,32]"])
    model = _TinyNet()
    image = torch.randn(1, 4, 32, 32, 32)

    result = sliding_window_predict(model, image, cfg, CPU)

    assert result.dtype == torch.float32


def test_model_is_set_to_eval(tmp_path: Path) -> None:
    """The function must call `model.eval()` itself, not assume it."""
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[32,32,32]"])
    model = _TinyNet()
    model.train()
    assert model.training is True

    sliding_window_predict(model, torch.randn(1, 4, 32, 32, 32), cfg, CPU)

    assert model.training is False


def test_output_does_not_require_grad(tmp_path: Path) -> None:
    """Inference runs under `torch.no_grad()`; the result must not track grad."""
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[32,32,32]"])
    model = _TinyNet()
    image = torch.randn(1, 4, 32, 32, 32)

    result = sliding_window_predict(model, image, cfg, CPU)

    assert result.requires_grad is False


def test_window_blending_preserves_interior_values(tmp_path: Path) -> None:
    """A deterministic identity-like stub must reproduce exact values.

    Output channel k is wired to equal input channel k exactly (1x1x1 conv,
    no bias). Since that prediction is spatially uniform, overlapping
    windows blend to the same value everywhere, so if the sliding-window
    machinery is stitching windows correctly the output must equal the
    input channels exactly. Uses mode="constant" to keep the expected value
    exact (no Gaussian center-weighting to reason about).
    """
    cfg = _compose_config(
        tmp_path,
        [
            "inference.sliding_window.roi_size=[16,16,16]",
            "inference.sliding_window.mode=constant",
        ],
    )
    model = _TinyNet()
    with torch.no_grad():
        model.conv.weight.zero_()
        model.conv.bias.zero_()
        for k in range(3):
            model.conv.weight[k, k, 0, 0, 0] = 1.0

    image = torch.randn(1, 4, 24, 24, 24)
    expected = image[:, :3, :, :, :]

    result = sliding_window_predict(model, image, cfg, CPU)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_build_inferer_reads_config(tmp_path: Path) -> None:
    """`build_inferer` must actually read the config, not hardcode defaults."""
    cfg = _compose_config(tmp_path, ["inference.sliding_window.overlap=0.25"])

    inferer = build_inferer(cfg)

    assert inferer.overlap == 0.25
