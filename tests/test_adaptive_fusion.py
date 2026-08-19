"""Tests for neurovision.models.fusion.adaptive_fusion.

CPU only, small tensors, whole file runs in well under 10 seconds. Configs
are built with OmegaConf.create, following the same `_cfg`-style helper
pattern as tests/test_cnn_encoder.py and tests/test_swin_encoder.py.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from neurovision.models.fusion.adaptive_fusion import (
    AdaptiveGatedFusion,
    AddFusion,
    BranchAmbiguity,
    ConcatFusion,
    GateGenerator,
    _window_partition,
    _window_reverse,
)
from neurovision.models.fusion.registry import available_fusions, build_fusion

CNN_CHANNELS = 32
SWIN_CHANNELS = 48
NUM_REGIONS = 3


def _cfg(**overrides: object) -> object:
    """Builds a full composed config mirroring cfg.model.fusion."""
    base = {
        "model": {
            "out_channels": NUM_REGIONS,
            "fusion": {
                "name": "adaptive_gated",
                "num_heads": 4,
                "window_size": 4,
                "full_attention_max_tokens": 512,
                "gate_channels": "scalar",
                "gate_reduction": 4,
                "num_groups": 8,
                "attn_dropout": 0.0,
                "proj_dropout": 0.0,
                "layer_scale_init": 1e-4,
                "use_checkpoint": False,
            },
        },
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.model.fusion, key, value, merge=True)
    return cfg


# ---------------------------------------------------------------------------
# shape tests, one per variant
# ---------------------------------------------------------------------------


def test_adaptive_gated_fusion_output_shape() -> None:
    block = AdaptiveGatedFusion(CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4)
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)
    out = block(cnn_feat, swin_feat)
    assert out.shape == (1, CNN_CHANNELS, 8, 8, 8)


def test_concat_fusion_output_shape() -> None:
    block = ConcatFusion(CNN_CHANNELS, SWIN_CHANNELS)
    cnn_feat = torch.randn(2, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(2, SWIN_CHANNELS, 6, 6, 6)
    out = block(cnn_feat, swin_feat)
    assert out.shape == (2, CNN_CHANNELS, 6, 6, 6)


def test_add_fusion_output_shape() -> None:
    block = AddFusion(CNN_CHANNELS, SWIN_CHANNELS)
    cnn_feat = torch.randn(2, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(2, SWIN_CHANNELS, 6, 6, 6)
    out = block(cnn_feat, swin_feat)
    assert out.shape == (2, CNN_CHANNELS, 6, 6, 6)


# ---------------------------------------------------------------------------
# return_gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate_channels,expected_gate_channels",
    [("scalar", 1), ("channel", CNN_CHANNELS)],
)
def test_adaptive_gated_fusion_return_gate(gate_channels: str, expected_gate_channels: int) -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, gate_channels=gate_channels
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)
    out = block(cnn_feat, swin_feat, return_gate=True)

    assert isinstance(out, tuple) and len(out) == 2
    fused, gate = out
    assert fused.shape == (1, CNN_CHANNELS, 8, 8, 8)
    assert gate is not None
    assert gate.shape == (1, expected_gate_channels, 8, 8, 8)
    # gate is a sigmoid output -- must be strictly inside (0, 1).
    assert torch.all(gate > 0.0)
    assert torch.all(gate < 1.0)


def test_concat_fusion_return_gate_is_none() -> None:
    block = ConcatFusion(CNN_CHANNELS, SWIN_CHANNELS)
    cnn_feat = torch.randn(1, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 6, 6, 6)
    out = block(cnn_feat, swin_feat, return_gate=True)

    assert isinstance(out, tuple) and len(out) == 2
    fused, gate = out
    assert isinstance(fused, torch.Tensor)
    assert gate is None


def test_add_fusion_return_gate_is_none() -> None:
    block = AddFusion(CNN_CHANNELS, SWIN_CHANNELS)
    cnn_feat = torch.randn(1, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 6, 6, 6)
    out = block(cnn_feat, swin_feat, return_gate=True)

    assert isinstance(out, tuple) and len(out) == 2
    fused, gate = out
    assert isinstance(fused, torch.Tensor)
    assert gate is None


# ---------------------------------------------------------------------------
# full vs windowed attention paths
# ---------------------------------------------------------------------------


def test_full_attention_path_runs() -> None:
    # 4*4*4 = 64 tokens <= full_attention_max_tokens=512: single global window.
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, full_attention_max_tokens=512
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 4, 4, 4)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 4, 4, 4)
    out = block(cnn_feat, swin_feat)
    assert out.shape == (1, CNN_CHANNELS, 4, 4, 4)


def test_windowed_attention_path_runs() -> None:
    # 12*12*12 = 1728 tokens > full_attention_max_tokens=512: windowed.
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, full_attention_max_tokens=512
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 12, 12, 12)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 12, 12, 12)
    out = block(cnn_feat, swin_feat)
    assert out.shape == (1, CNN_CHANNELS, 12, 12, 12)


def test_windowed_attention_handles_nondivisible_anisotropic_shape() -> None:
    # (13, 15, 17): none of these are multiples of window_size=4, and they
    # differ from each other -- exactly the case the key-padding mask exists
    # for. 13*15*17 = 3315 tokens > 512, so this takes the windowed path.
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, full_attention_max_tokens=512
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 13, 15, 17)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 13, 15, 17)
    out = block(cnn_feat, swin_feat)
    assert out.shape == (1, CNN_CHANNELS, 13, 15, 17)
    # Not just the shape: a fully-masked query row would make softmax produce
    # NaN, which would propagate silently into the loss. The module argues no
    # window can be entirely padding -- this is what pins that argument.
    assert torch.isfinite(out).all()


def test_full_and_single_window_paths_are_numerically_equivalent() -> None:
    # Same module instance, two different forward calls: one forced onto the
    # full-attention path (threshold set very high), one forced onto the
    # windowed path but with window_size >= every spatial dim so there is
    # still exactly one window and no padding. Padding-free single-window
    # attention over the whole map IS full attention over the whole map, so
    # the two must agree exactly.
    #
    # Scope, deliberately narrow: BOTH calls end up with one window and no
    # padding, so this only pins that the dispatch variable itself does not
    # change behaviour when degenerate. It is NOT a check on the partition
    # bookkeeping -- see the windowing-bookkeeping section below for that.
    torch.manual_seed(0)
    block = AdaptiveGatedFusion(CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4)
    block.eval()  # no dropout, no stochastic behaviour

    cnn_feat = torch.randn(1, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 6, 6, 6)

    block.cross_attn.full_attention_max_tokens = 10_000  # forces the full-attention path
    with torch.no_grad():
        out_full = block(cnn_feat, swin_feat)

    block.cross_attn.full_attention_max_tokens = 0  # forces the windowed path
    block.cross_attn.window_size = 6  # >= every spatial dim: exactly one window, no padding
    with torch.no_grad():
        out_windowed = block(cnn_feat, swin_feat)

    assert torch.allclose(out_full, out_windowed, atol=1e-6)


# ---------------------------------------------------------------------------
# windowing bookkeeping
#
# The equivalence test above forces the two dispatch branches but ends up with
# ONE window in both, so it cannot catch a transposition inside
# _window_partition: a wrong permute order would group spatially scattered
# voxels into a "window", _window_reverse would still undo it cleanly, every
# shape assertion would still pass, and attention would silently be computed
# over nonsense. The tests below are what actually pin that.
# ---------------------------------------------------------------------------


def test_window_partition_groups_spatially_contiguous_blocks() -> None:
    # Each token's value is its own flat coordinate index, so a window's
    # contents can be decoded back into coordinates and checked for
    # contiguity. Non-cubic window and anisotropic grid on purpose: a permute
    # bug that happens to be symmetric under a cubic window shows up here.
    D, H, W = 4, 6, 6
    window = (2, 3, 2)
    coords = torch.arange(D * H * W, dtype=torch.float32).view(1, D, H, W, 1)

    windows = _window_partition(coords, window)

    n_windows = (D // window[0]) * (H // window[1]) * (W // window[2])
    assert windows.shape == (n_windows, window[0] * window[1] * window[2], 1)

    for i in range(n_windows):
        flat = windows[i, :, 0].long().tolist()
        decoded = [(j // (H * W), (j // W) % H, j % W) for j in flat]
        for axis, extent in enumerate(window):
            values = sorted({c[axis] for c in decoded})
            # Every axis of a window must span exactly `extent` consecutive
            # coordinates -- that is what "spatially contiguous block" means.
            assert len(values) == extent
            assert values == list(range(values[0], values[0] + extent))


def test_window_partition_reverse_is_an_exact_roundtrip() -> None:
    D, H, W = 4, 6, 6
    window = (2, 3, 2)
    x = torch.randn(3, D, H, W, 5)

    restored = _window_reverse(_window_partition(x, window), window, 3, D, H, W)

    assert torch.equal(restored, x)


def test_window_partition_row_order_is_batch_major() -> None:
    # The key-padding mask is built once on a batch-invariant grid and tiled
    # with .repeat(B, 1, 1), which is only correct if partition rows are
    # batch-major (all of batch element 0's windows, then all of element 1's).
    # If the ordering were window-major, the mask would line up with the wrong
    # windows and only the padded border would be wrong -- invisible in any
    # shape test.
    D = H = W = 4
    window = (2, 2, 2)
    marked = torch.stack(
        [torch.zeros(D, H, W, 1), torch.ones(D, H, W, 1)]
    )  # batch element b filled with the value b

    windows = _window_partition(marked, window)

    n_windows = windows.shape[0] // 2
    assert (windows[:n_windows] == 0).all()
    assert (windows[n_windows:] == 1).all()


def test_windowed_padded_path_has_no_cross_batch_leakage() -> None:
    # Runs the padded, multi-window path with B=2 and checks each batch
    # element gets the same answer it would get alone. This is the end-to-end
    # form of the mask-tiling check above: any misalignment between the tiled
    # mask and the partitioned windows shows up as a per-element difference.
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, full_attention_max_tokens=0
    )
    block.eval()
    cnn_a, cnn_b = torch.randn(1, CNN_CHANNELS, 7, 5, 6), torch.randn(1, CNN_CHANNELS, 7, 5, 6)
    swin_a, swin_b = torch.randn(1, SWIN_CHANNELS, 7, 5, 6), torch.randn(1, SWIN_CHANNELS, 7, 5, 6)

    with torch.no_grad():
        batched = block(torch.cat([cnn_a, cnn_b]), torch.cat([swin_a, swin_b]))
        alone_a = block(cnn_a, swin_a)
        alone_b = block(cnn_b, swin_b)

    assert torch.allclose(batched[0:1], alone_a, atol=1e-6)
    assert torch.allclose(batched[1:2], alone_b, atol=1e-6)


def test_full_attention_matches_hand_rolled_reference() -> None:
    # Independent oracle: recompute the full-attention path by hand with an
    # explicit softmax(q @ k.T / sqrt(d)) @ v over the flattened token set,
    # using the module's own weights. Every other test in this file compares
    # the module against itself, so this is the only check that the attention
    # math (head split, scaling, which stream supplies q vs k/v) is right at
    # all, rather than merely self-consistent.
    torch.manual_seed(0)
    D = H = W = 4  # 64 tokens, under the default threshold -> full attention
    block = AdaptiveGatedFusion(CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=2)
    block.eval()
    attn = block.cross_attn

    cnn_feat = torch.randn(1, CNN_CHANNELS, D, H, W)
    swin_proj = torch.randn(1, CNN_CHANNELS, D, H, W)

    with torch.no_grad():
        actual = attn(cnn_feat, swin_proj)

        # Reference. Tokens in (B, N, C) layout, N = D*H*W in the same
        # row-major order _window_partition produces for a single window.
        q_tokens = attn.q_norm(cnn_feat.permute(0, 2, 3, 4, 1)).reshape(1, D * H * W, CNN_CHANNELS)
        kv_tokens = attn.kv_norm(swin_proj.permute(0, 2, 3, 4, 1)).reshape(
            1, D * H * W, CNN_CHANNELS
        )
        heads, head_dim = attn.num_heads, attn.head_dim

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(1, D * H * W, heads, head_dim).transpose(1, 2)

        q, k, v = (
            split(attn.q_proj(q_tokens)),
            split(attn.k_proj(kv_tokens)),
            split(attn.v_proj(kv_tokens)),
        )
        scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
        out = torch.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(1, D * H * W, CNN_CHANNELS)
        expected = attn.out_proj(out).view(1, D, H, W, CNN_CHANNELS).permute(0, 4, 1, 2, 3)

    assert torch.allclose(actual, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# layer_scale_init == 0 -> exact identity
# ---------------------------------------------------------------------------


def test_zero_layer_scale_init_gives_exact_identity() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, layer_scale_init=0.0
    )
    block.eval()
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)

    with torch.no_grad():
        out = block(cnn_feat, swin_feat)

    assert torch.equal(out, cnn_feat)


# ---------------------------------------------------------------------------
# gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_through_swin_branch() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, layer_scale_init=1e-2
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8, requires_grad=True)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8, requires_grad=True)

    out = block(cnn_feat, swin_feat)
    out.sum().backward()

    swin_proj_grad = block.swin_proj_conv.weight.grad
    q_proj_grad = block.cross_attn.q_proj.weight.grad

    assert swin_proj_grad is not None
    assert torch.any(swin_proj_grad != 0.0)
    assert q_proj_grad is not None
    assert torch.any(q_proj_grad != 0.0)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_spatial_shape_mismatch_raises() -> None:
    block = AdaptiveGatedFusion(CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4)
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 6, 6, 6)
    with pytest.raises(ValueError, match="misaligned"):
        block(cnn_feat, swin_feat)


def test_channel_count_mismatch_raises() -> None:
    block = AdaptiveGatedFusion(CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4)
    cnn_feat = torch.randn(1, CNN_CHANNELS + 1, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)
    with pytest.raises(ValueError, match="channels"):
        block(cnn_feat, swin_feat)


def test_cnn_channels_not_divisible_by_num_heads_raises() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        AdaptiveGatedFusion(30, SWIN_CHANNELS, num_heads=4, window_size=4)


def test_concat_fusion_channels_not_divisible_by_num_groups_raises() -> None:
    with pytest.raises(ValueError, match="num_groups"):
        ConcatFusion(cnn_channels=12, swin_channels=SWIN_CHANNELS, num_groups=8)


def test_bad_gate_channels_string_raises() -> None:
    with pytest.raises(ValueError, match="gate_channels"):
        AdaptiveGatedFusion(CNN_CHANNELS, SWIN_CHANNELS, gate_channels="bogus")


# ---------------------------------------------------------------------------
# gradient checkpointing
# ---------------------------------------------------------------------------


def test_use_checkpoint_matches_shape_and_backprops_in_train_mode() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_checkpoint=True
    )
    block.train()
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8, requires_grad=True)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8, requires_grad=True)

    out = block(cnn_feat, swin_feat)
    assert out.shape == (1, CNN_CHANNELS, 8, 8, 8)

    out.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} has no grad"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_available_fusions_contains_exactly_the_three_variants() -> None:
    assert set(available_fusions()) == {"adaptive_gated", "concat", "add"}


def test_build_fusion_returns_adaptive_gated() -> None:
    block = build_fusion(_cfg(name="adaptive_gated"), CNN_CHANNELS, SWIN_CHANNELS, level=0)
    assert isinstance(block, AdaptiveGatedFusion)


def test_build_fusion_returns_concat() -> None:
    block = build_fusion(_cfg(name="concat"), CNN_CHANNELS, SWIN_CHANNELS, level=0)
    assert isinstance(block, ConcatFusion)


def test_build_fusion_returns_add() -> None:
    block = build_fusion(_cfg(name="add"), CNN_CHANNELS, SWIN_CHANNELS, level=0)
    assert isinstance(block, AddFusion)


def test_build_fusion_unknown_name_raises_and_names_available() -> None:
    with pytest.raises(ValueError, match="adaptive_gated"):
        build_fusion(_cfg(name="bogus"), CNN_CHANNELS, SWIN_CHANNELS, level=0)


# ---------------------------------------------------------------------------
# ambiguity-conditioned gate
# ---------------------------------------------------------------------------


def test_ambiguity_on_forward_and_forward_with_branch_logits_shapes() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)

    out = block(cnn_feat, swin_feat)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, CNN_CHANNELS, 8, 8, 8)

    fused, branch_logits = block.forward_with_branch_logits(cnn_feat, swin_feat)
    assert fused.shape == (1, CNN_CHANNELS, 8, 8, 8)
    assert branch_logits is not None
    l_c, l_s = branch_logits
    assert l_c.shape == (1, NUM_REGIONS, 8, 8, 8)
    assert l_s.shape == (1, NUM_REGIONS, 8, 8, 8)


def test_ambiguity_off_forward_with_branch_logits_returns_none_and_fewer_params() -> None:
    block_on = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    block_off = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=False
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)

    fused, branch_logits = block_off.forward_with_branch_logits(cnn_feat, swin_feat)
    assert fused.shape == (1, CNN_CHANNELS, 8, 8, 8)
    assert branch_logits is None

    n_params_on = sum(p.numel() for p in block_on.parameters())
    n_params_off = sum(p.numel() for p in block_off.parameters())
    # Proves the ablation actually removes parameters rather than building an
    # unused BranchAmbiguity module.
    assert n_params_off < n_params_on


def test_gate_generator_conv_in_channels_reflect_ambiguity_switch() -> None:
    block_on = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    block_off = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=False
    )
    assert block_on.gate_generator.conv_in.in_channels == 2 * CNN_CHANNELS + 3 * NUM_REGIONS
    assert block_off.gate_generator.conv_in.in_channels == 2 * CNN_CHANNELS
    assert block_off.ambiguity is None


def test_branch_ambiguity_output_ranges_are_bounded() -> None:
    ambiguity_module = BranchAmbiguity(CNN_CHANNELS, NUM_REGIONS)
    cnn_feat = torch.randn(1, CNN_CHANNELS, 4, 4, 4)
    swin_proj = torch.randn(1, CNN_CHANNELS, 4, 4, 4)

    ambiguity, l_c, l_s = ambiguity_module(cnn_feat, swin_proj)

    assert ambiguity.shape == (1, 3 * NUM_REGIONS, 4, 4, 4)
    assert torch.all(ambiguity >= 0.0)
    assert torch.all(ambiguity <= 1.0)
    assert l_c.shape == (1, NUM_REGIONS, 4, 4, 4)
    assert l_s.shape == (1, NUM_REGIONS, 4, 4, 4)


def test_branch_ambiguity_saturated_logits_stay_finite() -> None:
    # Drives both branch logit convs to saturation: without the eps clamp
    # before the entropy logs, this produces NaN (log(0) = -inf, 0 * -inf =
    # NaN) that would silently propagate into the gate and every downstream
    # feature. Confirmed to fail without the clamp during development.
    ambiguity_module = BranchAmbiguity(CNN_CHANNELS, NUM_REGIONS)
    with torch.no_grad():
        ambiguity_module.cnn_logits.weight.fill_(1e4)
        ambiguity_module.swin_logits.weight.fill_(1e4)

    cnn_feat = torch.randn(1, CNN_CHANNELS, 4, 4, 4)
    swin_proj = torch.randn(1, CNN_CHANNELS, 4, 4, 4)

    ambiguity, l_c, l_s = ambiguity_module(cnn_feat, swin_proj)

    assert torch.isfinite(ambiguity).all()
    assert torch.isfinite(l_c).all()
    assert torch.isfinite(l_s).all()

    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    with torch.no_grad():
        block.ambiguity.cnn_logits.weight.fill_(1e4)
        block.ambiguity.swin_logits.weight.fill_(1e4)
        out = block(torch.randn(1, CNN_CHANNELS, 8, 8, 8), torch.randn(1, SWIN_CHANNELS, 8, 8, 8))
    assert torch.isfinite(out).all()


def test_branch_ambiguity_zeroed_weights_give_exact_disagreement_zero_entropy_one() -> None:
    # Pins the normalization constant (ln 2): with both logit conv WEIGHTS
    # zeroed (biases are already zero-initialized), every branch predicts
    # p=0.5 everywhere regardless of input, so disagreement is exactly 0 and
    # Bernoulli entropy at p=0.5 is exactly its maximum -- 1.0 after the
    # ln(2) normalization. A wrong normalization constant would show up here
    # and nowhere else.
    ambiguity_module = BranchAmbiguity(CNN_CHANNELS, NUM_REGIONS)
    with torch.no_grad():
        ambiguity_module.cnn_logits.weight.zero_()
        ambiguity_module.swin_logits.weight.zero_()

    cnn_feat = torch.randn(1, CNN_CHANNELS, 4, 4, 4)
    swin_proj = torch.randn(1, CNN_CHANNELS, 4, 4, 4)

    ambiguity, l_c, l_s = ambiguity_module(cnn_feat, swin_proj)
    disagreement, h_c, h_s = ambiguity.split(NUM_REGIONS, dim=1)

    assert torch.equal(disagreement, torch.zeros_like(disagreement))
    assert torch.allclose(h_c, torch.ones_like(h_c))
    assert torch.allclose(h_s, torch.ones_like(h_s))


def test_gradient_reaches_both_branch_logit_convs_through_gate_alone() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)

    fused = block(cnn_feat, swin_feat)
    fused.sum().backward()

    cnn_logits_grad = block.ambiguity.cnn_logits.weight.grad
    swin_logits_grad = block.ambiguity.swin_logits.weight.grad
    assert cnn_logits_grad is not None
    assert torch.any(cnn_logits_grad != 0.0)
    assert swin_logits_grad is not None
    assert torch.any(swin_logits_grad != 0.0)


def test_gradient_survives_checkpointing_for_branch_logits() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS,
        SWIN_CHANNELS,
        num_heads=4,
        window_size=4,
        use_ambiguity=True,
        use_checkpoint=True,
    )
    block.train()
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8, requires_grad=True)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8, requires_grad=True)

    fused, branch_logits = block.forward_with_branch_logits(cnn_feat, swin_feat)
    assert branch_logits is not None
    l_c, l_s = branch_logits

    loss = fused.sum() + l_c.sum() + l_s.sum()
    loss.backward()

    cnn_logits_grad = block.ambiguity.cnn_logits.weight.grad
    swin_logits_grad = block.ambiguity.swin_logits.weight.grad
    assert cnn_logits_grad is not None
    assert torch.any(cnn_logits_grad != 0.0)
    assert swin_logits_grad is not None
    assert torch.any(swin_logits_grad != 0.0)


def test_zero_layer_scale_init_is_exact_identity_with_ambiguity_on() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS,
        SWIN_CHANNELS,
        num_heads=4,
        window_size=4,
        layer_scale_init=0.0,
        use_ambiguity=True,
    )
    block.eval()
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)

    with torch.no_grad():
        out = block(cnn_feat, swin_feat)

    assert torch.equal(out, cnn_feat)


def test_concat_and_add_fusion_forward_with_branch_logits_returns_none() -> None:
    concat_block = ConcatFusion(CNN_CHANNELS, SWIN_CHANNELS)
    add_block = AddFusion(CNN_CHANNELS, SWIN_CHANNELS)
    cnn_feat = torch.randn(1, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 6, 6, 6)

    fused, branch_logits = concat_block.forward_with_branch_logits(cnn_feat, swin_feat)
    assert isinstance(fused, torch.Tensor)
    assert branch_logits is None

    fused, branch_logits = add_block.forward_with_branch_logits(cnn_feat, swin_feat)
    assert isinstance(fused, torch.Tensor)
    assert branch_logits is None


def test_build_fusion_defaults_use_ambiguity_true_when_key_absent() -> None:
    # An older config composed before use_ambiguity existed must still build,
    # and must build the full (ambiguity-conditioned) gate -- not silently
    # fall back to the content-only ablation.
    cfg = _cfg(name="adaptive_gated")
    assert "use_ambiguity" not in cfg.model.fusion
    block = build_fusion(cfg, CNN_CHANNELS, SWIN_CHANNELS, level=0)
    assert isinstance(block, AdaptiveGatedFusion)
    assert block.use_ambiguity is True
    assert block.ambiguity is not None


def test_build_fusion_respects_use_ambiguity_false() -> None:
    cfg = _cfg(name="adaptive_gated", use_ambiguity=False)
    block = build_fusion(cfg, CNN_CHANNELS, SWIN_CHANNELS, level=0)
    assert isinstance(block, AdaptiveGatedFusion)
    assert block.use_ambiguity is False
    assert block.ambiguity is None


def test_ambiguity_probes_send_no_gradient_into_branch_features() -> None:
    """Pins the `.detach()` in `BranchAmbiguity.forward`.

    The branch-supervision loss term (`neurovision.losses.multitask.MultiTaskLoss`) trains
    these probes against the region labels. If gradient from that term reached `cnn_feat` /
    `swin_feat` (the encoder outputs), the branch-supervision objective would push both
    encoders toward agreeing with each other -- collapsing the very disagreement signal the
    gate is supposed to read (see `BranchAmbiguity`'s module-level comment and
    `docs/research/contribution.md`). Verified by temporarily removing the two `.detach()`
    calls in `BranchAmbiguity.forward` and confirming this test then fails; restored
    afterwards.
    """
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8, requires_grad=True)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8, requires_grad=True)

    fused, branch_logits = block.forward_with_branch_logits(cnn_feat, swin_feat)
    assert branch_logits is not None
    l_c, l_s = branch_logits

    # Backward on the branch logits ALONE, not `fused` -- isolates the probes' own gradient
    # path from the gate/attention path, which legitimately does reach cnn_feat/swin_feat.
    (l_c.sum() + l_s.sum()).backward()

    assert cnn_feat.grad is None or torch.equal(cnn_feat.grad, torch.zeros_like(cnn_feat.grad))
    assert swin_feat.grad is None or torch.equal(swin_feat.grad, torch.zeros_like(swin_feat.grad))


def test_ambiguity_probe_weights_still_train_despite_detached_inputs() -> None:
    """The other half of the contract test 7 pins: the probe LEARNS even though the thing it
    measures does not move because of it.
    """
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8, requires_grad=True)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8, requires_grad=True)

    fused, branch_logits = block.forward_with_branch_logits(cnn_feat, swin_feat)
    assert branch_logits is not None
    l_c, l_s = branch_logits

    (l_c.sum() + l_s.sum()).backward()

    cnn_logits_grad = block.ambiguity.cnn_logits.weight.grad
    swin_logits_grad = block.ambiguity.swin_logits.weight.grad
    assert cnn_logits_grad is not None
    assert torch.any(cnn_logits_grad != 0.0)
    assert swin_logits_grad is not None
    assert torch.any(swin_logits_grad != 0.0)


def test_gate_generator_raises_on_extra_channels_ambiguity_mismatch() -> None:
    gate_off = GateGenerator(
        CNN_CHANNELS, gate_channels="scalar", gate_reduction=4, num_groups=8, extra_channels=0
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 4, 4, 4)
    swin_proj = torch.randn(1, CNN_CHANNELS, 4, 4, 4)
    bogus_ambiguity = torch.randn(1, 9, 4, 4, 4)

    with pytest.raises(ValueError, match="extra_channels"):
        gate_off(cnn_feat, swin_proj, ambiguity=bogus_ambiguity)

    gate_on = GateGenerator(
        CNN_CHANNELS, gate_channels="scalar", gate_reduction=4, num_groups=8, extra_channels=9
    )
    with pytest.raises(ValueError, match="extra_channels"):
        gate_on(cnn_feat, swin_proj, ambiguity=None)


def test_branch_ambiguity_entropy_is_finite_under_fp16_saturation() -> None:
    """A saturated probe must give finite entropy in fp16, not NaN.

    This is a regression guard for the bug that destroyed 10.5 GPU-hours of
    `neurovision` session 1 (2026-08-09). The entropy was computed from
    probabilities as `-(p*log(p) + (1-p)*log(1-p))`, guarded by
    `p.clamp(eps, 1 - eps)` with `eps = 1e-6`.

    That guard is exactly correct in fp32 and a COMPLETE NO-OP in fp16:
    fp16's epsilon is ~9.8e-4, so `1.0 - 1e-6` rounds to exactly 1.0 and the
    upper clamp clamps nothing. Under AMP the probes run in fp16, so any
    probability above ~0.9995 became exactly 1.0, `(1 - p)` became 0,
    `log(0)` was -inf, and `0 * -inf` was NaN. The NaN flowed through the gate
    into every fused feature and into the loss, and nothing raised -- training
    simply continued on NaN, with `best.pt` frozen at the last pre-divergence
    epoch.

    Two properties are asserted, and BOTH are needed. Finiteness alone would
    pass for an implementation that silently returns zeros; correctness alone
    (checked elsewhere in fp32) is what the old code already satisfied.

    The dtype is the entire point of the test -- run it in fp32 only and it
    passes against the broken implementation.
    """
    torch.manual_seed(0)
    module = BranchAmbiguity(channels=16, num_regions=3)
    # The state a confident probe reaches after ~10 epochs of real training:
    # a logit large enough that sigmoid saturates to exactly 1.0 in fp16.
    with torch.no_grad():
        module.cnn_logits.bias.fill_(30.0)
        module.swin_logits.bias.fill_(-30.0)

    for dtype in (torch.float32, torch.float16):
        mod = module.half() if dtype is torch.float16 else module.float()
        cnn_feat = torch.randn(1, 16, 4, 4, 4, dtype=dtype)
        swin_proj = torch.randn(1, 16, 4, 4, 4, dtype=dtype)
        ambiguity, _, _ = mod(cnn_feat, swin_proj)

        assert torch.isfinite(ambiguity).all(), (
            f"non-finite ambiguity in {dtype}: a saturated probe must give entropy 0, "
            "not NaN. See this test's docstring."
        )
        # Channels [0:num_regions] are disagreement, the rest are the two
        # entropies. A certain probe has zero entropy.
        entropy = ambiguity[:, 3:]
        assert entropy.max().item() < 1e-3, "a saturated probe should have ~zero entropy"


def test_branch_ambiguity_entropy_is_one_at_maximum_uncertainty() -> None:
    """p = 0.5 must give entropy exactly 1.0, since it is normalized by ln 2.

    Pairs with the fp16 finiteness test above: that one would also pass for an
    implementation that returned zeros everywhere, so the actual value has to
    be pinned somewhere too.
    """
    module = BranchAmbiguity(channels=16, num_regions=3).float()
    # Zero weights and biases -> logits 0 -> p = 0.5 for both branches.
    with torch.no_grad():
        for conv in (module.cnn_logits, module.swin_logits):
            conv.weight.zero_()
            conv.bias.zero_()

    ambiguity, _, _ = module(torch.randn(1, 16, 2, 2, 2), torch.randn(1, 16, 2, 2, 2))

    assert ambiguity[:, 3:].mean().item() == pytest.approx(1.0, abs=1e-5)
    # Both branches predict 0.5, so they agree exactly.
    assert ambiguity[:, :3].abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# forward_with_ambiguity -- inference-time exposure of the ambiguity signal
# ---------------------------------------------------------------------------


def test_forward_with_ambiguity_shape_matches_fused_spatial_dims() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)

    fused, ambiguity = block.forward_with_ambiguity(cnn_feat, swin_feat)

    assert fused.shape == (1, CNN_CHANNELS, 8, 8, 8)
    assert ambiguity is not None
    assert ambiguity.shape == (1, 3 * NUM_REGIONS, 8, 8, 8)


def test_forward_with_ambiguity_values_are_bounded_zero_one() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    cnn_feat = torch.randn(2, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(2, SWIN_CHANNELS, 6, 6, 6)

    _fused, ambiguity = block.forward_with_ambiguity(cnn_feat, swin_feat)

    assert ambiguity is not None
    assert torch.all(ambiguity >= 0.0)
    assert torch.all(ambiguity <= 1.0)


def test_concat_and_add_fusion_forward_with_ambiguity_returns_none() -> None:
    # Neither variant overrides forward_with_ambiguity -- the base class default must
    # make this work for free.
    concat_block = ConcatFusion(CNN_CHANNELS, SWIN_CHANNELS)
    add_block = AddFusion(CNN_CHANNELS, SWIN_CHANNELS)
    cnn_feat = torch.randn(1, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 6, 6, 6)

    fused, ambiguity = concat_block.forward_with_ambiguity(cnn_feat, swin_feat)
    assert fused.shape == (1, CNN_CHANNELS, 6, 6, 6)
    assert ambiguity is None
    assert "forward_with_ambiguity" not in type(concat_block).__dict__

    fused, ambiguity = add_block.forward_with_ambiguity(cnn_feat, swin_feat)
    assert fused.shape == (1, CNN_CHANNELS, 6, 6, 6)
    assert ambiguity is None
    assert "forward_with_ambiguity" not in type(add_block).__dict__


def test_adaptive_gated_fusion_use_ambiguity_false_forward_with_ambiguity_returns_none() -> None:
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=False
    )
    cnn_feat = torch.randn(1, CNN_CHANNELS, 6, 6, 6)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 6, 6, 6)

    fused, ambiguity = block.forward_with_ambiguity(cnn_feat, swin_feat)

    assert fused.shape == (1, CNN_CHANNELS, 6, 6, 6)
    assert ambiguity is None


def test_forward_with_ambiguity_fused_output_matches_plain_forward_exactly() -> None:
    # The test that catches a divergent second code path: forward_with_ambiguity must
    # compute the SAME fused tensor forward() does, not an approximation of it.
    torch.manual_seed(0)
    block = AdaptiveGatedFusion(
        CNN_CHANNELS, SWIN_CHANNELS, num_heads=4, window_size=4, use_ambiguity=True
    )
    block.eval()
    cnn_feat = torch.randn(1, CNN_CHANNELS, 8, 8, 8)
    swin_feat = torch.randn(1, SWIN_CHANNELS, 8, 8, 8)

    with torch.no_grad():
        expected = block(cnn_feat, swin_feat)
        fused, _ambiguity = block.forward_with_ambiguity(cnn_feat, swin_feat)

    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(fused, expected)
