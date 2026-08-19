"""Tests for `neurovision.inference.tta`.

CPU only, tiny synthetic volumes, whole file well under a second. Stub
networks are small hand-built `nn.Module`s (never the full `NeuroVisionX`),
mirroring `tests/test_mc_dropout.py`'s style.

Test 3 (`test_identity_model_round_trips_to_sigmoid_of_input`) is the single
most important test in this file: it is the one that would fail if the
un-flip used a different axis set than the forward flip (e.g. an off-by-one
that flipped the channel axis instead of a spatial one). Every other test
checks internal consistency of the returned numbers, which would still hold
even under a wrong-but-self-consistent axis mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from torch import Tensor, nn

from neurovision.inference.tta import FLIP_AXES_8, TTAOutput, flip_combinations, tta_predict

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Config helper (mirrors tests/test_mc_dropout.py / tests/test_sliding_window.py)
# ---------------------------------------------------------------------------

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def _compose_config(tmp_path: Path, overrides: list[str] | None = None) -> Any:
    """Composes the real Hydra config, with the mandatory `data.root_dir` set."""
    all_overrides = [f"data.root_dir={tmp_path}", "device=cpu"] + (overrides or [])
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        return hydra.compose(config_name="config", overrides=all_overrides)


# ---------------------------------------------------------------------------
# Stub networks
# ---------------------------------------------------------------------------


class _IdentityNet(nn.Module):
    """Returns its input completely unchanged. Trivially flip-equivariant."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class _PointwiseNet(nn.Module):
    """A single 1x1x1 conv. Flip-equivariant: it acts per-voxel, so flipping
    the input then unflipping the output is exactly the same as not flipping
    at all -- there is no spatial context for a flip to disturb."""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _RampNet(nn.Module):
    """A 1x1x1 conv PLUS a fixed spatial ramp added along axis D (dim 2).

    Deliberately NOT flip-equivariant: the ramp depends on absolute position
    along D, so flipping the input before this runs (and unflipping after)
    gives a different result than not flipping at all -- proving the 8
    augmented passes genuinely differ rather than being silently collapsed.
    """

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        d = x.shape[2]
        ramp = torch.linspace(-3.0, 3.0, steps=d, device=x.device).view(1, 1, d, 1, 1)
        return self.conv(x) + ramp


# ---------------------------------------------------------------------------
# 1. flip_combinations: 8 unique tuples for 3 axes, identity first
# ---------------------------------------------------------------------------


def test_flip_combinations_three_axes_gives_8_unique_identity_first() -> None:
    combos = flip_combinations((0, 1, 2))

    assert len(combos) == 8
    assert len(set(combos)) == 8
    assert combos[0] == ()


def test_flip_axes_8_constant_matches_flip_combinations() -> None:
    assert FLIP_AXES_8 == flip_combinations((0, 1, 2))
    assert len(FLIP_AXES_8) == 8


# ---------------------------------------------------------------------------
# 2. flip_combinations with 2 axes gives 4
# ---------------------------------------------------------------------------


def test_flip_combinations_two_axes_gives_4() -> None:
    combos = flip_combinations((0, 1))

    assert len(combos) == 4
    assert combos[0] == ()
    assert set(combos) == {(), (0,), (1,), (0, 1)}


# ---------------------------------------------------------------------------
# 3. THE CRITICAL TEST: identity model round-trips to sigmoid(image)
# ---------------------------------------------------------------------------


def test_identity_model_round_trips_to_sigmoid_of_input(tmp_path: Path) -> None:
    """If the un-flip used a different axis set than the forward flip (e.g.
    an off-by-one flipping the channel axis instead of a spatial one), this
    test would fail -- an identity model's output would no longer match
    sigmoid(image) after being scrambled and un-scrambled inconsistently.
    """
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _IdentityNet()
    image = torch.randn(1, 4, 12, 12, 12)

    result = tta_predict(model, image, cfg, device=CPU)

    expected = torch.sigmoid(image.squeeze(0))
    torch.testing.assert_close(result.mean_prob, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. Equivariant model: all 8 passes agree, std_prob ~ 0
# ---------------------------------------------------------------------------


def test_equivariant_model_gives_near_zero_std(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(1, 4, 12, 12, 12)

    result = tta_predict(model, image, cfg, device=CPU)

    # Not bitwise-zero: gaussian-blended overlapping sliding windows sum in a
    # different floating-point order depending on which edge of the volume a
    # window starts from, so a genuinely equivariant model still picks up a
    # few ULPs of blending noise per flip. 1e-3 is far above that noise floor
    # and far below what a real (non-equivariant) disagreement would produce.
    assert result.std_prob.max().item() < 1e-3


# ---------------------------------------------------------------------------
# 5. Non-equivariant model: std_prob > 0 somewhere
# ---------------------------------------------------------------------------


def test_non_equivariant_model_gives_positive_std_somewhere(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _RampNet(channels=4)
    image = torch.randn(1, 4, 12, 12, 12)

    result = tta_predict(model, image, cfg, device=CPU)

    assert result.std_prob.max().item() > 0.0


# ---------------------------------------------------------------------------
# 6. N=1 (identity flip only): std_prob exactly zero, mean_prob == single pass
# ---------------------------------------------------------------------------


def test_single_identity_flip_gives_exact_zero_std_and_matches_single_pass(
    tmp_path: Path,
) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _RampNet(channels=4)
    image = torch.randn(1, 4, 12, 12, 12)

    result = tta_predict(model, image, cfg, flips=[()], device=CPU)

    zeros = torch.zeros_like(result.std_prob)
    assert torch.equal(result.std_prob, zeros)

    from neurovision.inference.sliding_window import sliding_window_predict

    single_pass_logits = sliding_window_predict(model, image, cfg, CPU)
    expected = torch.sigmoid(single_pass_logits.squeeze(0))
    torch.testing.assert_close(result.mean_prob, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. Output shapes match input spatial shape + model's output channel count
# ---------------------------------------------------------------------------


def test_output_shapes_match_input_spatial_shape(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(1, 4, 12, 16, 20)

    result = tta_predict(model, image, cfg, device=CPU)

    assert result.mean_prob.shape == (4, 12, 16, 20)
    assert result.std_prob.shape == (4, 12, 16, 20)


def test_unbatched_input_also_works(tmp_path: Path) -> None:
    """A (C, D, H, W) input (no batch dim) is accepted, same as batch-of-1."""
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(4, 12, 12, 12)

    result = tta_predict(model, image, cfg, device=CPU)

    assert result.mean_prob.shape == (4, 12, 12, 12)


# ---------------------------------------------------------------------------
# 8. num_augmentations and flips echoed back correctly
# ---------------------------------------------------------------------------


def test_num_augmentations_and_flips_echoed(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(1, 4, 12, 12, 12)
    requested_flips = [(), (0,), (1, 2)]

    result = tta_predict(model, image, cfg, flips=requested_flips, device=CPU)

    assert result.num_augmentations == 3
    assert result.flips == tuple(requested_flips)


# ---------------------------------------------------------------------------
# 9. progress callback called exactly N times, ending at (N, N)
# ---------------------------------------------------------------------------


def test_progress_callback_called_exactly_n_times(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(1, 4, 12, 12, 12)

    calls: list[tuple[int, int]] = []
    tta_predict(
        model, image, cfg, device=CPU, progress=lambda done, total: calls.append((done, total))
    )

    assert len(calls) == 8  # default FLIP_AXES_8
    assert calls[-1] == (8, 8)
    assert calls == [(i, 8) for i in range(1, 9)]


# ---------------------------------------------------------------------------
# 10. mean_prob is a valid probability everywhere
# ---------------------------------------------------------------------------


def test_mean_prob_in_unit_interval(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _RampNet(channels=4)
    image = torch.randn(1, 4, 12, 12, 12)

    result = tta_predict(model, image, cfg, device=CPU)

    assert torch.all(result.mean_prob >= 0.0)
    assert torch.all(result.mean_prob <= 1.0)


# ---------------------------------------------------------------------------
# 11. Memory discipline: accumulator style, not an N-deep stack
# ---------------------------------------------------------------------------


def test_no_n_deep_stack_is_retained_structurally() -> None:
    """`tta.py` must accumulate into two running sums (mirroring
    `mc_dropout.py`), never build a Python list of N full-size prediction
    tensors. A structural check on the source is a light but direct proxy:
    an accumulator implementation has no reason to call `list.append` on a
    tensor anywhere.
    """
    import neurovision.inference.tta as tta_module

    source = Path(tta_module.__file__).read_text()
    assert "append(" not in source


def test_runs_with_full_8_way_tta_on_a_larger_volume(tmp_path: Path) -> None:
    """N=8 on a volume larger than a single sliding-window ROI -- proves the
    accumulator path works end to end at the project's default augmentation
    count without retaining per-pass tensors (a stack here would be a
    visibly wasteful ~8x this volume's own size, which the accumulator
    design avoids).
    """
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(1, 4, 24, 24, 24)

    result = tta_predict(model, image, cfg, device=CPU)

    assert result.num_augmentations == 8
    assert result.mean_prob.shape == (4, 24, 24, 24)


# ---------------------------------------------------------------------------
# Extra: empty flips list raises
# ---------------------------------------------------------------------------


def test_empty_flips_raises(tmp_path: Path) -> None:
    import pytest

    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(1, 4, 12, 12, 12)

    with pytest.raises(ValueError, match="flips"):
        tta_predict(model, image, cfg, flips=[], device=CPU)


def test_batch_size_greater_than_one_raises(tmp_path: Path) -> None:
    import pytest

    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(2, 4, 12, 12, 12)

    with pytest.raises(ValueError, match="batch size"):
        tta_predict(model, image, cfg, device=CPU)


# ---------------------------------------------------------------------------
# TTAOutput is a plain dataclass (sanity, mirrors MCDropoutOutput's contract)
# ---------------------------------------------------------------------------


def test_tta_output_fields(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _PointwiseNet(channels=4)
    image = torch.randn(1, 4, 12, 12, 12)

    result = tta_predict(model, image, cfg, device=CPU)

    assert isinstance(result, TTAOutput)
    assert result.num_augmentations == len(result.flips)
