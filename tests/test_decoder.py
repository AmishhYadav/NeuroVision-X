"""Tests for neurovision.models.decoder.unet_decoder: UNetDecoder, AttentionGate, _match_spatial.

CPU only, small tensors, whole file runs in well under 10 seconds. Skip widths
[8, 16, 32, 64] with num_groups=8 and a 16^3 top level keep the suite fast, following the
same small-tensor pattern as tests/test_cnn_encoder.py and tests/test_adaptive_fusion.py.
"""

from __future__ import annotations

import pytest
import torch

from neurovision.models.decoder.unet_decoder import AttentionGate, UNetDecoder, _match_spatial

SKIP_CHANNELS = [8, 16, 32, 64]
NUM_GROUPS = 8
TOP = 16  # spatial size of skips[0] (full resolution)


def _make_skips(
    channels: list[int] = SKIP_CHANNELS, top: int = TOP, batch: int = 1
) -> list[torch.Tensor]:
    """Builds a fine-to-coarse skip pyramid with the standard ceil(size/2) convention."""
    skips = []
    size = top
    for c in channels:
        skips.append(torch.randn(batch, c, size, size, size))
        size = -(-size // 2)  # ceil division, matching CNNEncoder/SwinEncoder
    return skips


# ---------------------------------------------------------------------------
# shape tests
# ---------------------------------------------------------------------------


def test_output_shapes_match_expected_pyramid() -> None:
    decoder = UNetDecoder(skip_channels=SKIP_CHANNELS, num_groups=NUM_GROUPS)
    skips = _make_skips()
    out = decoder(skips)

    assert len(out) == len(SKIP_CHANNELS) - 1
    # TOP=16 is a power of 2, so ceil(size/2) == floor(size/2) at every level -- the plain
    # TOP // 2**i formula below is exact, not an approximation of the pyramid-building one.
    for i, level in enumerate(out):
        assert level.shape == (1, SKIP_CHANNELS[i], TOP // (2**i), TOP // (2**i), TOP // (2**i))


def test_out_channels_and_num_stages_match_forward_output() -> None:
    decoder = UNetDecoder(skip_channels=SKIP_CHANNELS, num_groups=NUM_GROUPS)
    skips = _make_skips()
    out = decoder(skips)

    assert decoder.num_stages == len(out)
    assert len(decoder.out_channels) == len(out)
    for expected_c, level in zip(decoder.out_channels, out, strict=True):
        assert level.shape[1] == expected_c


def test_custom_decoder_channels_are_honoured() -> None:
    custom = [4, 24, 48]  # deliberately different from skip_channels[:-1] = [8, 16, 32]
    decoder = UNetDecoder(
        skip_channels=SKIP_CHANNELS, decoder_channels=custom, num_groups=4, blocks_per_stage=1
    )
    skips = _make_skips()
    out = decoder(skips)

    assert decoder.out_channels == custom
    for expected_c, level in zip(custom, out, strict=True):
        assert level.shape[1] == expected_c


@pytest.mark.parametrize("upsample", ["deconv", "interp"])
def test_both_upsample_modes_produce_identical_shapes(upsample: str) -> None:
    decoder = UNetDecoder(skip_channels=SKIP_CHANNELS, num_groups=NUM_GROUPS, upsample=upsample)
    skips = _make_skips()
    out = decoder(skips)

    assert len(out) == len(SKIP_CHANNELS) - 1
    for i, level in enumerate(out):
        assert level.shape == (1, SKIP_CHANNELS[i], TOP // (2**i), TOP // (2**i), TOP // (2**i))


# ---------------------------------------------------------------------------
# attention gates
# ---------------------------------------------------------------------------


def test_attention_gates_produce_same_shapes_and_backprop_to_psi() -> None:
    decoder = UNetDecoder(
        skip_channels=SKIP_CHANNELS, num_groups=NUM_GROUPS, use_attention_gates=True
    )
    skips = _make_skips()
    out = decoder(skips)

    for i, level in enumerate(out):
        assert level.shape == (1, SKIP_CHANNELS[i], TOP // (2**i), TOP // (2**i), TOP // (2**i))

    loss = sum(level.mean() for level in out)
    loss.backward()

    for gate in decoder.attention_gates:
        assert gate.psi.weight.grad is not None
        assert torch.any(gate.psi.weight.grad != 0.0)


# ---------------------------------------------------------------------------
# odd / anisotropic sizes -- pins _match_spatial end to end
# ---------------------------------------------------------------------------


def test_odd_anisotropic_sizes_match_skip_shapes_exactly() -> None:
    # Top level (13, 15, 17), then ceil(n/2) at each subsequent level:
    # (7, 8, 9), (4, 4, 5), (2, 2, 3).
    shapes = [(13, 15, 17), (7, 8, 9), (4, 4, 5), (2, 2, 3)]
    channels = SKIP_CHANNELS
    skips = [
        torch.randn(1, c, *shape, dtype=torch.float32)
        for c, shape in zip(channels, shapes, strict=True)
    ]

    decoder = UNetDecoder(skip_channels=channels, num_groups=NUM_GROUPS)
    out = decoder(skips)

    assert len(out) == len(shapes) - 1
    for i, level in enumerate(out):
        assert level.shape[2:] == shapes[i]
        assert level.shape[1] == channels[i]


# ---------------------------------------------------------------------------
# _match_spatial unit tests
# ---------------------------------------------------------------------------


def test_match_spatial_crops_from_the_end_not_centre() -> None:
    # arange-valued tensor: content check, not just shape. A size-3 axis cropped to 2 from
    # the END keeps [0, 1], not the centre [0, 1] vs [1, 2] -- distinguishing crop-from-end
    # from a centre crop needs an axis where the two disagree, so use size 3 -> 2 on one
    # axis while the others already match.
    x = torch.arange(1 * 1 * 3 * 2 * 2, dtype=torch.float32).view(1, 1, 3, 2, 2)
    out = _match_spatial(x, (2, 2, 2))

    assert out.shape == (1, 1, 2, 2, 2)
    # Content check: keeping index 0 and 1 along D (crop from the end, dropping index 2)
    # means out equals x[..., :2, :, :] exactly.
    assert torch.equal(out, x[:, :, :2, :, :])


def test_match_spatial_pads_when_smaller() -> None:
    x = torch.ones(1, 1, 2, 2, 2)
    out = _match_spatial(x, (3, 2, 2))

    assert out.shape == (1, 1, 3, 2, 2)
    assert torch.equal(out[:, :, :2, :, :], x)
    assert torch.equal(out[:, :, 2, :, :], torch.zeros(1, 1, 2, 2))


def test_match_spatial_raises_when_mismatch_exceeds_one_voxel() -> None:
    x = torch.ones(1, 1, 5, 5, 5)
    with pytest.raises(ValueError, match="1-voxel tolerance"):
        _match_spatial(x, (2, 5, 5))


# ---------------------------------------------------------------------------
# gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_from_finest_output_to_deepest_stage() -> None:
    decoder = UNetDecoder(skip_channels=SKIP_CHANNELS, num_groups=NUM_GROUPS)
    skips = _make_skips()
    out = decoder(skips)

    out[0].sum().backward()

    # The deepest stage's parameters (index num_stages - 1, nearest the bottleneck) must
    # get real gradient from the FINEST output -- this is the proof the coarse-to-fine
    # chain is actually connected, not merely that each stage is locally differentiable.
    deepest_conv = decoder.up_convs[decoder.num_stages - 1]
    assert deepest_conv.weight.grad is not None
    assert torch.any(deepest_conv.weight.grad != 0.0)


# ---------------------------------------------------------------------------
# gradient checkpointing
# ---------------------------------------------------------------------------


def test_use_checkpoint_in_train_mode_matches_shapes_and_backprops() -> None:
    decoder = UNetDecoder(skip_channels=SKIP_CHANNELS, num_groups=NUM_GROUPS, use_checkpoint=True)
    decoder.train()
    skips = [s.clone().requires_grad_(True) for s in _make_skips()]
    out = decoder(skips)

    for i, level in enumerate(out):
        assert level.shape == (1, SKIP_CHANNELS[i], TOP // (2**i), TOP // (2**i), TOP // (2**i))

    loss = sum(level.mean() for level in out)
    loss.backward()

    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"{name} has no grad"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_too_few_skip_channels_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 entries"):
        UNetDecoder(skip_channels=[8])


def test_decoder_channels_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="decoder_channels has"):
        UNetDecoder(skip_channels=SKIP_CHANNELS, decoder_channels=[8, 16])  # needs 3 entries


def test_decoder_channels_not_divisible_by_num_groups_raises() -> None:
    with pytest.raises(ValueError, match="num_groups"):
        UNetDecoder(skip_channels=SKIP_CHANNELS, decoder_channels=[6, 16, 32], num_groups=8)


def test_skip_channels_not_divisible_by_num_groups_raises() -> None:
    with pytest.raises(ValueError, match="num_groups"):
        UNetDecoder(skip_channels=[8, 16, 30], num_groups=8)


def test_bad_upsample_string_raises() -> None:
    with pytest.raises(ValueError, match="upsample"):
        UNetDecoder(skip_channels=SKIP_CHANNELS, upsample="bogus")


def test_blocks_per_stage_less_than_one_raises() -> None:
    with pytest.raises(ValueError, match="blocks_per_stage"):
        UNetDecoder(skip_channels=SKIP_CHANNELS, blocks_per_stage=0)


def test_forward_skip_count_mismatch_raises() -> None:
    decoder = UNetDecoder(skip_channels=SKIP_CHANNELS, num_groups=NUM_GROUPS)
    skips = _make_skips()[:-1]  # one too few
    with pytest.raises(ValueError, match="forward\\(\\) got"):
        decoder(skips)


def test_forward_skip_channel_count_mismatch_raises() -> None:
    decoder = UNetDecoder(skip_channels=SKIP_CHANNELS, num_groups=NUM_GROUPS)
    skips = _make_skips()
    skips[1] = torch.randn(1, SKIP_CHANNELS[1] + 1, *skips[1].shape[2:])
    with pytest.raises(ValueError, match=r"skips\[1\]"):
        decoder(skips)


# ---------------------------------------------------------------------------
# AttentionGate, standalone
# ---------------------------------------------------------------------------


def test_attention_gate_output_shape_and_bounded_scaling() -> None:
    gate = AttentionGate(skip_channels=16, gate_channels=16)
    skip = torch.randn(2, 16, 4, 4, 4)
    gating_signal = torch.randn(2, 16, 4, 4, 4)

    out = gate(skip, gating_signal)

    assert out.shape == skip.shape
    # attn in (0, 1) via sigmoid, so |out| <= |skip| elementwise.
    assert torch.all(out.abs() <= skip.abs() + 1e-6)
