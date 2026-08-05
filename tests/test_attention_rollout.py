"""Tests for neurovision.explainability.attention_rollout.

CPU only. The model under test is a small REAL `monai.networks.nets.swin_unetr.
SwinTransformer` (embed_dim=6, window_size=2, depths=(2,2,2,2), 4 heads-worth of stages) on a
16^3 input -- not a stub, because the whole module is an inversion of MONAI's own pad/roll/
partition semantics, and a stub that faked those semantics would test nothing. Measured
end-to-end forward pass: ~4ms; the full file runs in well under a second.
"""

from __future__ import annotations

import logging

import pytest
import torch
from monai.networks.nets.swin_unetr import SwinTransformer, window_reverse

from neurovision.explainability.attention_rollout import (
    _padded_index_grid,
    _window_index_mapping,
    attention_rollout,
    attention_to_voxel_map,
    available_blocks,
    capture_attention,
    combine_stage_maps,
)


def _make_model() -> SwinTransformer:
    """A tiny real SwinTransformer: 4 stages, window_size=2, depths=(2,2,2,2)."""
    torch.manual_seed(0)
    model = SwinTransformer(
        in_chans=2,
        embed_dim=6,
        window_size=(2, 2, 2),
        patch_size=(2, 2, 2),
        depths=(2, 2, 2, 2),
        num_heads=(2, 2, 2, 2),
        use_checkpoint=False,
    )
    model.eval()
    return model


def _make_image() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(1, 2, 16, 16, 16, generator=generator)


# Stage 1's token grid is 8^3 (window_size=2 divides it evenly, so shift stays nonzero) -- the
# stage used for most tests below since both blocks keep a genuine shift.
_STAGE1_BLOCK0 = "layers1.0.blocks.0"
_STAGE1_BLOCK1 = "layers1.0.blocks.1"


# ---------------------------------------------------------------------------
# capture_attention
# ---------------------------------------------------------------------------


def test_capture_attention_one_entry_per_window_attention_head_axis_gone() -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    expected_names = available_blocks(model)
    assert sorted(capture.stage_names) == sorted(expected_names)
    assert set(capture.attention.keys()) == set(expected_names)

    # Stage 1: 8^3 token grid, window_size=2 -> nW = 4*4*4 = 64, N = 2*2*2 = 8. Head axis gone.
    assert tuple(capture.attention[_STAGE1_BLOCK0].shape) == (64, 8, 8)
    assert tuple(capture.attention[_STAGE1_BLOCK1].shape) == (64, 8, 8)


def test_captured_attention_rows_sum_to_one() -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    for name, attn in capture.attention.items():
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), name


def test_hooks_removed_after_successful_call() -> None:
    model = _make_model()
    image = _make_image()
    capture_attention(model, image)

    for _, module in model.named_modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_hooks_removed_after_exception() -> None:
    model = _make_model()
    # A batch of 0 channels forces model(image) to raise deep inside patch_embed, exercising
    # the finally block's hook cleanup on the error path.
    bad_image = torch.randn(1, 0, 16, 16, 16)

    with pytest.raises(Exception):
        capture_attention(model, bad_image)

    for _, module in model.named_modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_raises_when_model_in_training_mode() -> None:
    model = _make_model()
    model.train()
    image = _make_image()

    with pytest.raises(ValueError, match="eval mode"):
        capture_attention(model, image)


def test_raises_on_model_with_no_window_attention() -> None:
    model = torch.nn.Sequential(torch.nn.Conv3d(2, 4, 3, padding=1))
    model.eval()
    image = _make_image()

    with pytest.raises(ValueError, match="WindowAttention"):
        capture_attention(model, image)


# ---------------------------------------------------------------------------
# the index-grid inverse mapping -- the most important test in this file
# ---------------------------------------------------------------------------


def test_index_grid_round_trip_is_exact_for_shifted_block() -> None:
    """window_reverse + roll(+shift) + crop recovers the original index grid EXACTLY."""
    token_grid = (8, 8, 8)
    window_size = (2, 2, 2)
    shift_size = (1, 1, 1)

    original_idx, padded_shape = _padded_index_grid(token_grid, window_size)
    idx_windows, padded_shape_2 = _window_index_mapping(token_grid, window_size, shift_size)
    assert padded_shape == padded_shape_2

    dp, hp, wp = padded_shape
    recovered = window_reverse(idx_windows.unsqueeze(-1), window_size, [1, dp, hp, wp])
    recovered = torch.roll(recovered, shifts=shift_size, dims=(1, 2, 3))
    recovered = recovered[:, : token_grid[0], : token_grid[1], : token_grid[2], :]
    original_cropped = original_idx[:, : token_grid[0], : token_grid[1], : token_grid[2], :]

    assert torch.equal(recovered, original_cropped)


def test_index_grid_round_trip_is_exact_for_unshifted_block() -> None:
    token_grid = (8, 8, 8)
    window_size = (2, 2, 2)
    shift_size = (0, 0, 0)

    original_idx, padded_shape = _padded_index_grid(token_grid, window_size)
    idx_windows, _ = _window_index_mapping(token_grid, window_size, shift_size)

    dp, hp, wp = padded_shape
    recovered = window_reverse(idx_windows.unsqueeze(-1), window_size, [1, dp, hp, wp])
    recovered = torch.roll(recovered, shifts=shift_size, dims=(1, 2, 3))
    recovered = recovered[:, : token_grid[0], : token_grid[1], : token_grid[2], :]

    assert torch.equal(recovered, original_idx)


def test_index_grid_round_trip_is_exact_with_real_padding() -> None:
    """A token grid that does not divide evenly by the window size, so padding is non-trivial."""
    token_grid = (5, 5, 5)
    window_size = (2, 2, 2)
    shift_size = (1, 1, 1)

    original_idx, padded_shape = _padded_index_grid(token_grid, window_size)
    idx_windows, _ = _window_index_mapping(token_grid, window_size, shift_size)

    dp, hp, wp = padded_shape
    recovered = window_reverse(idx_windows.unsqueeze(-1), window_size, [1, dp, hp, wp])
    recovered = torch.roll(recovered, shifts=shift_size, dims=(1, 2, 3))
    recovered = recovered[:, : token_grid[0], : token_grid[1], : token_grid[2], :]
    original_cropped = original_idx[:, : token_grid[0], : token_grid[1], : token_grid[2], :]

    assert torch.equal(recovered, original_cropped)


def test_shifted_and_unshifted_blocks_map_differently() -> None:
    """If the roll were dropped, blocks.0 and blocks.1's mappings would be identical."""
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    map_block0 = attention_to_voxel_map(capture, _STAGE1_BLOCK0, upsample=False, normalize=False)
    map_block1 = attention_to_voxel_map(capture, _STAGE1_BLOCK1, upsample=False, normalize=False)

    assert map_block0.shape == map_block1.shape
    assert not torch.equal(map_block0, map_block1)


# ---------------------------------------------------------------------------
# attention_to_voxel_map
# ---------------------------------------------------------------------------


def test_attention_to_voxel_map_shapes() -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    upsampled = attention_to_voxel_map(capture, _STAGE1_BLOCK0, upsample=True)
    assert tuple(upsampled.shape) == (1, 1) + capture.input_shape

    at_token_resolution = attention_to_voxel_map(capture, _STAGE1_BLOCK0, upsample=False)
    assert tuple(at_token_resolution.shape) == (1, 1) + capture.token_grids[_STAGE1_BLOCK0]


def test_normalize_true_gives_unit_range() -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    out = attention_to_voxel_map(capture, _STAGE1_BLOCK0, normalize=True)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0 + 1e-6
    assert out.max().item() > 0.0  # real attention on random input is not perfectly constant


def test_normalize_constant_map_returns_zeros_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    # Force a constant attention matrix: uniform 1/N everywhere.
    name = _STAGE1_BLOCK0
    n = capture.attention[name].shape[-1]
    nw = capture.attention[name].shape[0]
    capture.attention[name] = torch.full((nw, n, n), 1.0 / n)

    with caplog.at_level(logging.WARNING):
        out = attention_to_voxel_map(capture, name, normalize=True)

    assert torch.equal(out, torch.zeros_like(out))
    assert any("constant" in record.message for record in caplog.records)


def test_uniform_attention_gives_spatially_uniform_voxel_map() -> None:
    """Positive control: the scatter is not shuffling positions."""
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    name = _STAGE1_BLOCK0
    n = capture.attention[name].shape[-1]
    nw = capture.attention[name].shape[0]
    value = 1.0 / n
    capture.attention[name] = torch.full((nw, n, n), value)

    out = attention_to_voxel_map(capture, name, upsample=False, normalize=False)
    assert torch.allclose(out, torch.full_like(out, value))


# ---------------------------------------------------------------------------
# attention_rollout
# ---------------------------------------------------------------------------


def test_rollout_raises_when_tokens_exceed_max_tokens() -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    # Stage 1's token grid is 8^3 = 512 tokens.
    with pytest.raises(ValueError, match="512") as exc_info:
        attention_rollout(capture, "layers1", max_tokens=10)
    assert "10" in str(exc_info.value)


def test_rollout_succeeds_on_small_stage_and_is_finite() -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    # Stage 3's token grid is 2^3 = 8 tokens -- well under the default max_tokens.
    out = attention_rollout(capture, "layers3")
    assert tuple(out.shape) == (1, 1) + capture.input_shape
    assert torch.isfinite(out).all()


def test_rollout_residual_weight_one_gives_uniform_map() -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    out = attention_rollout(capture, "layers3", residual_weight=1.0)
    assert out.std().item() < 1e-6


def test_rollout_unknown_stage_raises() -> None:
    model = _make_model()
    image = _make_image()
    capture = capture_attention(model, image)

    with pytest.raises(ValueError, match="no blocks found"):
        attention_rollout(capture, "not_a_real_stage")


# ---------------------------------------------------------------------------
# combine_stage_maps
# ---------------------------------------------------------------------------


def test_combine_stage_maps_mean_and_product() -> None:
    a = torch.full((1, 1, 2, 2, 2), 0.2)
    b = torch.full((1, 1, 2, 2, 2), 0.8)

    mean_out = combine_stage_maps([a, b], method="mean")
    assert torch.allclose(mean_out, torch.full_like(mean_out, 0.5))

    product_out = combine_stage_maps([a, b], method="product")
    assert torch.allclose(product_out, torch.full_like(product_out, 0.16))


def test_combine_stage_maps_unknown_method_raises() -> None:
    a = torch.zeros(1, 1, 2, 2, 2)
    with pytest.raises(ValueError, match="unknown method"):
        combine_stage_maps([a], method="max")


def test_combine_stage_maps_shape_mismatch_raises() -> None:
    a = torch.zeros(1, 1, 2, 2, 2)
    b = torch.zeros(1, 1, 3, 3, 3)
    with pytest.raises(ValueError, match="shape"):
        combine_stage_maps([a, b], method="mean")


# ---------------------------------------------------------------------------
# available_blocks
# ---------------------------------------------------------------------------


def test_available_blocks_are_real_submodules_in_forward_order() -> None:
    model = _make_model()
    names = available_blocks(model)

    assert len(names) > 0
    for name in names:
        module = model.get_submodule(name)
        assert hasattr(module, "attn")

    # Forward order: stage 1's blocks appear before stage 2's, which appear before stage 3's.
    stage1_positions = [i for i, n in enumerate(names) if n.startswith("layers1.")]
    stage2_positions = [i for i, n in enumerate(names) if n.startswith("layers2.")]
    assert max(stage1_positions) < min(stage2_positions)
