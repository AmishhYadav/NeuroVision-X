"""Tests for neurovision.models.neurovision: NeuroVisionX and build_neurovision.

CPU only, whole file runs in well under 20 seconds. The tiny network used throughout has
cnn.channels=[8, 16, 24, 32] (4 levels) and swin.num_levels=3, feature_size=12 -- one more
CNN level than Swin levels, satisfying the stride-offset contract (see
neurovision.models.encoders.swin's module docstring), and giving a decoder with
num_stages=3, which is what lets deep_supervision_levels=3 (the production default) actually
be exercised here. fusion.num_heads=4 and num_groups=8 divide every channel width used
(16, 24, 32). Configs for the builder-level tests are built with OmegaConf.create, following
the same `_cfg`-style helper pattern as tests/test_swin_encoder.py and
tests/test_adaptive_fusion.py; the direct-construction tests build submodules by hand, which
is the whole point of NeuroVisionX's dependency-injected constructor.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from neurovision.losses.segmentation import DeepSupervisionLoss, DiceBCELoss
from neurovision.models import (
    neurovision as neurovision_module,  # noqa: F401  (registers "neurovision")
)
from neurovision.models.decoder.unet_decoder import UNetDecoder
from neurovision.models.encoders.cnn import CNNEncoder
from neurovision.models.encoders.swin import SwinEncoder
from neurovision.models.fusion.adaptive_fusion import AdaptiveGatedFusion, AddFusion, ConcatFusion
from neurovision.models.heads.multitask import MultiTaskHead, MultiTaskOutput
from neurovision.models.neurovision import NeuroVisionX, build_neurovision
from neurovision.models.registry import available_models, build_model

CNN_CHANNELS = [8, 16, 24, 32]
CNN_BLOCKS = [1, 1, 1, 1]
SWIN_FEATURE_SIZE = 12
SWIN_NUM_LEVELS = 3
NUM_GROUPS = 8
FUSION_HEADS = 4


# ---------------------------------------------------------------------------
# direct-construction helpers (dependency injection, no Hydra config)
# ---------------------------------------------------------------------------


def _build_cnn() -> CNNEncoder:
    # zero_init_residual=False here, deliberately, even though the production config uses
    # True: with it on, EVERY residual block's norm2.weight starts at exactly 0, which zeros
    # the gradient reaching everything upstream of norm2 (conv1, norm1, conv2) on the very
    # first optimizer step -- documented, expected behaviour (see cnn.py's
    # `_init_weights` docstring and tests/test_cnn_encoder.py), but it would make a plain
    # "does gradient reach a CNN conv weight" check here fail for a reason that has nothing
    # to do with this file. Off, so the gradient-flow tests below check what they say they
    # check.
    return CNNEncoder(
        in_channels=4,
        channels=CNN_CHANNELS,
        blocks_per_stage=CNN_BLOCKS,
        num_groups=NUM_GROUPS,
        dropout=0.0,
        use_checkpoint=False,
        zero_init_residual=False,
    )


def _build_swin() -> SwinEncoder:
    return SwinEncoder(
        in_channels=4,
        feature_size=SWIN_FEATURE_SIZE,
        depths=(1, 1, 1, 1),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        patch_size=2,
        num_levels=SWIN_NUM_LEVELS,
        use_checkpoint=False,
        normalize=True,
    )


def _build_fusion_blocks(
    cnn: CNNEncoder, swin: SwinEncoder, name: str = "adaptive_gated"
) -> nn.ModuleList:
    blocks = []
    for i in range(swin.num_levels):
        cnn_ch = cnn.out_channels[i + 1]
        swin_ch = swin.out_channels[i]
        if name == "adaptive_gated":
            blocks.append(
                AdaptiveGatedFusion(
                    cnn_ch, swin_ch, num_heads=FUSION_HEADS, window_size=4, num_groups=NUM_GROUPS
                )
            )
        elif name == "concat":
            blocks.append(ConcatFusion(cnn_ch, swin_ch, num_groups=NUM_GROUPS))
        elif name == "add":
            blocks.append(AddFusion(cnn_ch, swin_ch))
        else:
            raise ValueError(f"unknown test fusion name {name!r}")
    return nn.ModuleList(blocks)


def _build_decoder(cnn: CNNEncoder) -> UNetDecoder:
    return UNetDecoder(
        skip_channels=cnn.out_channels,
        decoder_channels=None,
        blocks_per_stage=1,
        num_groups=NUM_GROUPS,
        dropout=0.0,
        upsample="deconv",
        use_attention_gates=False,
        use_checkpoint=False,
    )


def _build_model(
    deep_supervision_levels: int = 3, fusion_name: str = "adaptive_gated"
) -> NeuroVisionX:
    cnn = _build_cnn()
    swin = _build_swin()
    fusion_blocks = _build_fusion_blocks(cnn, swin, fusion_name)
    decoder = _build_decoder(cnn)
    return NeuroVisionX(
        cnn_encoder=cnn,
        swin_encoder=swin,
        fusion_blocks=fusion_blocks,
        decoder=decoder,
        out_channels=3,
        deep_supervision_levels=deep_supervision_levels,
        head_dropout=0.0,
    )


class _FakeSubmodule(nn.Module):
    """Bare nn.Module carrying only the attributes NeuroVisionX's construction-time
    validation reads, so the invalid-combination tests do not need to construct real (and
    therefore always internally-consistent) encoders/decoders to trigger each check."""

    def __init__(self, **attrs: object) -> None:
        super().__init__()
        for key, value in attrs.items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# Hydra-config helper, for the builder-level tests
# ---------------------------------------------------------------------------


def _full_cfg(
    deep_supervision_levels: int = 3,
    ds_loss_enabled: bool | None = None,
    fusion_name: str = "adaptive_gated",
) -> object:
    """Builds a full composed config mirroring configs/model/neurovision.yaml, at the same
    tiny widths used by the direct-construction helpers above."""
    base = {
        "data": {"in_channels": 4, "num_classes": 3},
        "model": {
            "name": "neurovision",
            "in_channels": 4,
            "out_channels": 3,
            "encoder": {
                "cnn": {
                    "channels": CNN_CHANNELS,
                    "blocks_per_stage": CNN_BLOCKS,
                    "num_groups": NUM_GROUPS,
                    "dropout": 0.0,
                    "use_checkpoint": False,
                    "zero_init_residual": True,
                },
                "swin": {
                    "feature_size": SWIN_FEATURE_SIZE,
                    "depths": [1, 1, 1, 1],
                    "num_heads": [3, 6, 12, 24],
                    "window_size": 7,
                    "patch_size": 2,
                    "num_levels": SWIN_NUM_LEVELS,
                    "drop_rate": 0.0,
                    "attn_drop_rate": 0.0,
                    "dropout_path_rate": 0.0,
                    "use_checkpoint": False,
                    "normalize": True,
                },
            },
            "fusion": {
                "name": fusion_name,
                "num_heads": FUSION_HEADS,
                "window_size": 4,
                "full_attention_max_tokens": 512,
                "gate_channels": "scalar",
                "gate_reduction": 4,
                "num_groups": NUM_GROUPS,
                "attn_dropout": 0.0,
                "proj_dropout": 0.0,
                "layer_scale_init": 1.0e-4,
                "use_checkpoint": False,
            },
            "decoder": {
                "channels": None,
                "blocks_per_stage": 1,
                "num_groups": NUM_GROUPS,
                "dropout": 0.0,
                "upsample": "deconv",
                "use_attention_gates": False,
                "use_checkpoint": False,
            },
            "head": {"dropout": 0.0},
            "deep_supervision_levels": deep_supervision_levels,
        },
    }
    if ds_loss_enabled is not None:
        base["training"] = {"loss": {"deep_supervision": {"enabled": ds_loss_enabled}}}
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# 1. eval-mode shape: single Tensor
# ---------------------------------------------------------------------------


def test_eval_mode_returns_single_tensor_matching_input_spatial_shape() -> None:
    model = _build_model(deep_supervision_levels=3)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, Tensor)
    assert out.shape == (1, 3, 32, 32, 32)


# ---------------------------------------------------------------------------
# 2. train mode, deep_supervision_levels=3: list, highest resolution first
# ---------------------------------------------------------------------------


def test_train_mode_deep_supervision_returns_ordered_list() -> None:
    model = _build_model(deep_supervision_levels=3)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert isinstance(out, list)
    assert len(out) == 3
    for i, level in enumerate(out):
        expected_size = 32 // (2**i)
        assert level.shape == (1, 3, expected_size, expected_size, expected_size)


# ---------------------------------------------------------------------------
# 3. train mode, deep_supervision_levels=1: single Tensor, not a length-1 list
# ---------------------------------------------------------------------------


def test_train_mode_single_head_returns_tensor_not_list() -> None:
    model = _build_model(deep_supervision_levels=1)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert isinstance(out, Tensor)
    assert not isinstance(out, list)
    assert out.shape == (1, 3, 32, 32, 32)


# ---------------------------------------------------------------------------
# 4. loss integration: pins the output-ordering contract against the loss
# ---------------------------------------------------------------------------


def test_deep_supervision_loss_integration_is_finite_and_backprops() -> None:
    model = _build_model(deep_supervision_levels=3)
    model.train()
    loss_fn = DeepSupervisionLoss(DiceBCELoss())

    x = torch.randn(1, 4, 32, 32, 32)
    target = (torch.rand(1, 3, 32, 32, 32) > 0.5).float()

    preds = model(x)
    loss = loss_fn(preds, target)

    assert torch.isfinite(loss)
    loss.backward()

    conv1_weight = model.cnn_encoder.stages[0][0].conv1.weight
    assert conv1_weight.grad is not None
    assert torch.any(conv1_weight.grad != 0.0)


# ---------------------------------------------------------------------------
# 5. odd / anisotropic input: exact round trip through ceil-downsample + _match_spatial
# ---------------------------------------------------------------------------


def test_odd_anisotropic_input_output_shape_matches_exactly() -> None:
    model = _build_model(deep_supervision_levels=1)
    model.eval()
    x = torch.randn(1, 4, 36, 40, 28)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 36, 40, 28)


# ---------------------------------------------------------------------------
# 6. forward_with_gates
# ---------------------------------------------------------------------------


def test_forward_with_gates_adaptive_returns_bounded_gates() -> None:
    model = _build_model(fusion_name="adaptive_gated")
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, gates = model.forward_with_gates(x)

    assert isinstance(logits, Tensor)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert len(gates) == len(model.fusion_blocks) == SWIN_NUM_LEVELS
    for gate in gates:
        assert gate is not None
        assert torch.all(gate > 0.0)
        assert torch.all(gate < 1.0)


def test_forward_with_gates_concat_returns_none_gates() -> None:
    model = _build_model(fusion_name="concat")
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, gates = model.forward_with_gates(x)

    assert logits.shape == (1, 3, 32, 32, 32)
    assert len(gates) == SWIN_NUM_LEVELS
    assert all(gate is None for gate in gates)


def test_forward_with_gates_returns_single_tensor_even_in_training_mode() -> None:
    # forward_with_gates is the explainability path, not a training path: it always returns
    # a single logits tensor regardless of self.training.
    model = _build_model(deep_supervision_levels=3, fusion_name="adaptive_gated")
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    logits, gates = model.forward_with_gates(x)

    assert isinstance(logits, Tensor)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert len(gates) == SWIN_NUM_LEVELS


# ---------------------------------------------------------------------------
# 6b. forward_with_ambiguity
# ---------------------------------------------------------------------------


def test_forward_with_ambiguity_returns_tensor_logits_in_both_modes() -> None:
    model = _build_model(fusion_name="adaptive_gated")
    x = torch.randn(1, 4, 32, 32, 32)

    model.train()
    logits, ambiguity_maps = model.forward_with_ambiguity(x)
    assert isinstance(logits, Tensor)
    assert not isinstance(logits, list)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert len(ambiguity_maps) == len(model.fusion_blocks) == SWIN_NUM_LEVELS

    model.eval()
    with torch.no_grad():
        logits, ambiguity_maps = model.forward_with_ambiguity(x)
    assert isinstance(logits, Tensor)
    assert not isinstance(logits, list)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert len(ambiguity_maps) == SWIN_NUM_LEVELS


def test_forward_with_ambiguity_concat_returns_none_ambiguity() -> None:
    model = _build_model(fusion_name="concat")
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, ambiguity_maps = model.forward_with_ambiguity(x)

    assert logits.shape == (1, 3, 32, 32, 32)
    assert len(ambiguity_maps) == SWIN_NUM_LEVELS
    assert all(a is None for a in ambiguity_maps)


def test_cnn_only_forward_with_ambiguity_returns_empty_list() -> None:
    cnn = _build_cnn()
    decoder = _build_decoder(cnn)
    model = NeuroVisionX(
        cnn_encoder=cnn,
        swin_encoder=None,
        fusion_blocks=nn.ModuleList(),
        decoder=decoder,
        out_channels=3,
        deep_supervision_levels=1,
        head_dropout=0.0,
    )
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, ambiguity_maps = model.forward_with_ambiguity(x)

    assert logits.shape == (1, 3, 32, 32, 32)
    assert ambiguity_maps == []


# ---------------------------------------------------------------------------
# 7. gradient flow through both encoders and the decoder
# ---------------------------------------------------------------------------


def test_gradients_flow_through_both_encoders_and_decoder() -> None:
    model = _build_model(deep_supervision_levels=1)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)
    out.mean().backward()

    cnn_weight = model.cnn_encoder.stages[0][0].conv1.weight
    swin_weight = model.swin_encoder.swin.patch_embed.proj.weight
    decoder_weight = model.decoder.up_convs[0].weight

    for name, weight in [("cnn", cnn_weight), ("swin", swin_weight), ("decoder", decoder_weight)]:
        assert weight.grad is not None, f"{name} weight has no grad"
        assert torch.any(weight.grad != 0.0), f"{name} weight grad is all zero"


# ---------------------------------------------------------------------------
# 8. construction-time ValueErrors
# ---------------------------------------------------------------------------


def test_stride_offset_mismatch_raises() -> None:
    cnn = _FakeSubmodule(num_levels=3, out_channels=[8, 16, 24], strides=[1, 2, 4])
    swin = _FakeSubmodule(num_levels=3, out_channels=[12, 24, 48], strides=[2, 4, 8])

    with pytest.raises(ValueError, match="one more level"):
        NeuroVisionX(cnn, swin, nn.ModuleList(), _FakeSubmodule(), out_channels=3)


def test_strides_not_aligned_raises() -> None:
    cnn = _FakeSubmodule(num_levels=4, out_channels=[8, 16, 24, 32], strides=[1, 2, 4, 8])
    swin = _FakeSubmodule(num_levels=3, out_channels=[12, 24, 48], strides=[2, 4, 999])
    fusion_blocks = nn.ModuleList([nn.Identity() for _ in range(3)])

    with pytest.raises(ValueError, match="aligned"):
        NeuroVisionX(cnn, swin, fusion_blocks, _FakeSubmodule(), out_channels=3)


def test_fusion_blocks_count_mismatch_raises() -> None:
    cnn = _FakeSubmodule(num_levels=4, out_channels=[8, 16, 24, 32], strides=[1, 2, 4, 8])
    swin = _FakeSubmodule(num_levels=3, out_channels=[12, 24, 48], strides=[2, 4, 8])
    fusion_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])  # only 2, need 3

    with pytest.raises(ValueError, match="fusion_blocks"):
        NeuroVisionX(cnn, swin, fusion_blocks, _FakeSubmodule(), out_channels=3)


def test_decoder_skip_channels_mismatch_raises() -> None:
    cnn = _FakeSubmodule(num_levels=4, out_channels=[8, 16, 24, 32], strides=[1, 2, 4, 8])
    swin = _FakeSubmodule(num_levels=3, out_channels=[12, 24, 48], strides=[2, 4, 8])
    fusion_blocks = nn.ModuleList([nn.Identity() for _ in range(3)])
    decoder = _FakeSubmodule(skip_channels=[8, 16, 24, 999], out_channels=[8, 16, 24], num_stages=3)

    with pytest.raises(ValueError, match="different pyramid"):
        NeuroVisionX(cnn, swin, fusion_blocks, decoder, out_channels=3)


def test_deep_supervision_levels_out_of_range_raises() -> None:
    cnn = _FakeSubmodule(num_levels=4, out_channels=[8, 16, 24, 32], strides=[1, 2, 4, 8])
    swin = _FakeSubmodule(num_levels=3, out_channels=[12, 24, 48], strides=[2, 4, 8])
    fusion_blocks = nn.ModuleList([nn.Identity() for _ in range(3)])
    decoder = _FakeSubmodule(skip_channels=[8, 16, 24, 32], out_channels=[8, 16, 24], num_stages=3)

    with pytest.raises(ValueError, match="deep_supervision_levels"):
        NeuroVisionX(cnn, swin, fusion_blocks, decoder, out_channels=3, deep_supervision_levels=4)


# ---------------------------------------------------------------------------
# 9. builder-level deep-supervision/loss-config mismatch guard
# ---------------------------------------------------------------------------


def test_builder_deep_supervision_loss_mismatch_raises() -> None:
    cfg = _full_cfg(deep_supervision_levels=3, ds_loss_enabled=False)
    with pytest.raises(ValueError, match="deep_supervision_levels"):
        build_neurovision(cfg)


def test_builder_deep_supervision_loss_agreement_builds_fine() -> None:
    cfg = _full_cfg(deep_supervision_levels=3, ds_loss_enabled=True)
    model = build_neurovision(cfg)
    assert isinstance(model, NeuroVisionX)


def test_builder_with_no_training_group_still_builds() -> None:
    cfg = _full_cfg(deep_supervision_levels=3, ds_loss_enabled=None)
    assert not hasattr(cfg, "training") or "training" not in cfg
    model = build_neurovision(cfg)
    assert isinstance(model, NeuroVisionX)


# ---------------------------------------------------------------------------
# 10. registry
# ---------------------------------------------------------------------------


def test_neurovision_is_registered_and_build_model_returns_correct_type() -> None:
    assert "neurovision" in available_models()

    cfg = _full_cfg()
    model = build_model(cfg)

    assert isinstance(model, NeuroVisionX)


# ---------------------------------------------------------------------------
# 11. all three fusion variants assemble and run through the builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fusion_name", ["adaptive_gated", "concat", "add"])
def test_all_fusion_variants_build_and_run(fusion_name: str) -> None:
    cfg = _full_cfg(fusion_name=fusion_name)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 32, 32, 32)


# ---------------------------------------------------------------------------
# 12. cnn-only ablation: swin_encoder=None (configs/experiment/ablation_cnn_only.yaml)
# ---------------------------------------------------------------------------


def _build_cnn_only_model(deep_supervision_levels: int = 3) -> NeuroVisionX:
    cnn = _build_cnn()
    decoder = _build_decoder(cnn)
    return NeuroVisionX(
        cnn_encoder=cnn,
        swin_encoder=None,
        fusion_blocks=nn.ModuleList(),
        decoder=decoder,
        out_channels=3,
        deep_supervision_levels=deep_supervision_levels,
        head_dropout=0.0,
    )


def test_cnn_only_forward_returns_correct_shape() -> None:
    model = _build_cnn_only_model(deep_supervision_levels=1)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, Tensor)
    assert out.shape == (1, 3, 32, 32, 32)


def test_cnn_only_return_type_switch_survives() -> None:
    # Pins that the train-mode-list / eval-mode-tensor switch is unaffected by removing the
    # Swin branch.
    model = _build_cnn_only_model(deep_supervision_levels=3)

    model.train()
    x = torch.randn(1, 4, 32, 32, 32)
    out_train = model(x)
    assert isinstance(out_train, list)
    assert len(out_train) == 3

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert isinstance(out_eval, Tensor)
    assert not isinstance(out_eval, list)


def test_swin_none_with_nonempty_fusion_blocks_raises() -> None:
    cnn = _build_cnn()
    decoder = _build_decoder(cnn)
    fusion_blocks = nn.ModuleList([nn.Identity()])

    with pytest.raises(ValueError, match="nothing to fuse"):
        NeuroVisionX(
            cnn_encoder=cnn,
            swin_encoder=None,
            fusion_blocks=fusion_blocks,
            decoder=decoder,
            out_channels=3,
        )


def test_cnn_only_forward_with_gates_returns_empty_gate_list() -> None:
    model = _build_cnn_only_model(deep_supervision_levels=1)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, gates = model.forward_with_gates(x)

    assert logits.shape == (1, 3, 32, 32, 32)
    assert gates == []


def test_builder_swin_disabled_omits_swin_and_fusion_params() -> None:
    cfg_full = _full_cfg(ds_loss_enabled=True)
    cfg_cnn_only = _full_cfg(ds_loss_enabled=True)
    cfg_cnn_only.model.encoder.swin.enabled = False

    model_full = build_neurovision(cfg_full)
    model_cnn_only = build_neurovision(cfg_cnn_only)

    state_keys = model_cnn_only.state_dict().keys()
    assert not any(k.startswith("swin_encoder.") for k in state_keys)
    assert not any(k.startswith("fusion_blocks.") for k in state_keys)

    n_full = sum(p.numel() for p in model_full.parameters())
    n_cnn_only = sum(p.numel() for p in model_cnn_only.parameters())
    assert n_cnn_only < n_full


# ---------------------------------------------------------------------------
# 13. auxiliary heads (confidence / boundary) -- return-type switch and builder guards
# ---------------------------------------------------------------------------


def _build_model_with_heads(
    deep_supervision_levels: int = 3,
    confidence: bool = False,
    boundary: bool = False,
) -> NeuroVisionX:
    cnn = _build_cnn()
    swin = _build_swin()
    fusion_blocks = _build_fusion_blocks(cnn, swin)
    decoder = _build_decoder(cnn)
    heads = MultiTaskHead(
        decoder_channels=decoder.out_channels,
        out_channels=3,
        deep_supervision_levels=deep_supervision_levels,
        confidence=confidence,
        boundary=boundary,
        confidence_num_groups=NUM_GROUPS,
        boundary_num_groups=NUM_GROUPS,
    )
    return NeuroVisionX(
        cnn_encoder=cnn,
        swin_encoder=swin,
        fusion_blocks=fusion_blocks,
        decoder=decoder,
        out_channels=3,
        deep_supervision_levels=deep_supervision_levels,
        head_dropout=0.0,
        heads=heads,
    )


def test_eval_mode_returns_tensor_even_with_both_aux_heads_and_deep_supervision() -> None:
    # THE CRITICAL TEST: this is what keeps sliding-window inference working.
    model = _build_model_with_heads(deep_supervision_levels=3, confidence=True, boundary=True)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, Tensor)
    assert out.shape == (1, 3, 32, 32, 32)


def test_forward_return_type_switch_unperturbed_by_forward_with_ambiguity() -> None:
    # Regression guard for the addition of forward_with_ambiguity: NeuroVisionX.forward
    # must still return a plain Tensor in eval mode with both auxiliary heads enabled,
    # not a MultiTaskOutput or a list -- forward_with_ambiguity walks its own separate
    # encode/fuse/decode path and must not have touched forward's own switch.
    model = _build_model_with_heads(deep_supervision_levels=3, confidence=True, boundary=True)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, Tensor)
    assert not isinstance(out, list)
    assert not isinstance(out, MultiTaskOutput)
    assert out.shape == (1, 3, 32, 32, 32)


def test_train_mode_with_aux_head_returns_multitask_output() -> None:
    model = _build_model_with_heads(deep_supervision_levels=3, confidence=True, boundary=False)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert isinstance(out, MultiTaskOutput)
    assert len(out.seg) == 3
    assert out.confidence is not None
    assert out.boundary is None


def test_train_mode_no_aux_head_deep_supervision_returns_list_unchanged() -> None:
    model = _build_model_with_heads(deep_supervision_levels=3, confidence=False, boundary=False)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert isinstance(out, list)
    assert len(out) == 3


def test_train_mode_no_aux_head_single_level_returns_tensor_unchanged() -> None:
    model = _build_model_with_heads(deep_supervision_levels=1, confidence=False, boundary=False)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert isinstance(out, Tensor)
    assert not isinstance(out, list)


def test_forward_multitask_returns_multitask_output_in_both_modes() -> None:
    model = _build_model_with_heads(deep_supervision_levels=2, confidence=True, boundary=True)
    x = torch.randn(1, 4, 32, 32, 32)

    model.train()
    out_train = model.forward_multitask(x)
    assert isinstance(out_train, MultiTaskOutput)

    model.eval()
    with torch.no_grad():
        out_eval = model.forward_multitask(x)
    assert isinstance(out_eval, MultiTaskOutput)
    assert out_eval.confidence is not None
    assert out_eval.boundary is not None


def test_forward_with_gates_still_works_with_aux_heads_enabled() -> None:
    model = _build_model_with_heads(deep_supervision_levels=3, confidence=True, boundary=True)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, gates = model.forward_with_gates(x)

    assert isinstance(logits, Tensor)
    assert isinstance(gates, list)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert len(gates) == SWIN_NUM_LEVELS


def test_gradients_flow_to_both_auxiliary_heads() -> None:
    model = _build_model_with_heads(deep_supervision_levels=1, confidence=True, boundary=True)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)
    assert isinstance(out, MultiTaskOutput)
    total = out.seg[0].sum() + out.confidence.sum() + out.boundary.sum()
    total.backward()

    conf_weight = model.heads.confidence.conv1.weight
    bnd_weight = model.heads.boundary.conv1.weight

    for name, weight in [("confidence", conf_weight), ("boundary", bnd_weight)]:
        assert weight.grad is not None, f"{name} head weight has no grad"
        assert torch.any(weight.grad != 0.0), f"{name} head weight grad is all zero"


def _full_cfg_with_aux(
    confidence_enabled: bool,
    boundary_enabled: bool,
    loss_name: str,
    deep_supervision_levels: int = 3,
) -> object:
    """Same shape as `_full_cfg`, extended with `model.head.confidence` /
    `model.head.boundary` blocks and a training.loss.name so the aux-head/loss-name
    builder guard can be exercised."""
    cfg = _full_cfg(deep_supervision_levels=deep_supervision_levels, ds_loss_enabled=True)
    cfg.model.head.confidence = {
        "enabled": confidence_enabled,
        "hidden_channels": None,
        "num_groups": NUM_GROUPS,
        "dropout": 0.0,
    }
    cfg.model.head.boundary = {
        "enabled": boundary_enabled,
        "hidden_channels": None,
        "num_groups": NUM_GROUPS,
        "dropout": 0.0,
    }
    cfg.training.loss.name = loss_name
    return cfg


def test_builder_raises_when_aux_head_enabled_but_loss_is_not_multitask() -> None:
    cfg = _full_cfg_with_aux(confidence_enabled=True, boundary_enabled=False, loss_name="dice_ce")

    with pytest.raises(ValueError, match="multitask"):
        build_neurovision(cfg)


def test_builder_raises_when_loss_is_multitask_but_no_aux_head_enabled() -> None:
    cfg = _full_cfg_with_aux(
        confidence_enabled=False, boundary_enabled=False, loss_name="multitask"
    )

    with pytest.raises(ValueError, match="multitask"):
        build_neurovision(cfg)


def test_builder_aux_head_and_multitask_loss_agree_builds_fine() -> None:
    cfg = _full_cfg_with_aux(confidence_enabled=True, boundary_enabled=True, loss_name="multitask")

    model = build_neurovision(cfg)

    assert isinstance(model, NeuroVisionX)
    assert model.heads.has_auxiliary is True
    assert model.heads.confidence is not None
    assert model.heads.boundary is not None


def test_builder_swin_enabled_defaults_true_when_key_absent() -> None:
    # _full_cfg's swin block has no "enabled" key at all -- guards model-only test configs
    # (and any config predating this ablation switch) still building the full dual-encoder
    # model.
    cfg = _full_cfg(ds_loss_enabled=True)
    assert "enabled" not in cfg.model.encoder.swin

    model = build_neurovision(cfg)

    assert model.use_swin is True
    assert model.swin_encoder is not None
    assert len(model.fusion_blocks) == SWIN_NUM_LEVELS


# ---------------------------------------------------------------------------
# 14. branch-logits supervision (BranchAmbiguity plumbing to MultiTaskOutput)
# ---------------------------------------------------------------------------


def _build_model_with_branch_supervision(
    deep_supervision_levels: int = 3,
    fusion_name: str = "adaptive_gated",
    supervise_branch_logits: bool = True,
) -> NeuroVisionX:
    cnn = _build_cnn()
    swin = _build_swin()
    fusion_blocks = _build_fusion_blocks(cnn, swin, fusion_name)
    decoder = _build_decoder(cnn)
    heads = MultiTaskHead(
        decoder_channels=decoder.out_channels,
        out_channels=3,
        deep_supervision_levels=deep_supervision_levels,
        confidence=True,
        boundary=False,
        confidence_num_groups=NUM_GROUPS,
    )
    return NeuroVisionX(
        cnn_encoder=cnn,
        swin_encoder=swin,
        fusion_blocks=fusion_blocks,
        decoder=decoder,
        out_channels=3,
        deep_supervision_levels=deep_supervision_levels,
        head_dropout=0.0,
        heads=heads,
        supervise_branch_logits=supervise_branch_logits,
    )


def test_branch_logits_collected_in_train_mode_with_aux_heads() -> None:
    model = _build_model_with_branch_supervision(deep_supervision_levels=3)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert isinstance(out, MultiTaskOutput)
    assert out.branch_logits is not None
    assert len(out.branch_logits) == SWIN_NUM_LEVELS
    for level, pair in enumerate(out.branch_logits):
        assert isinstance(pair, tuple)
        l_c, l_s = pair
        # Fusion level i fuses CNN level i+1 / Swin level i, both at stride 2**(i+1).
        expected_size = 32 // (2 ** (level + 1))
        assert l_c.shape == (1, 3, expected_size, expected_size, expected_size)
        assert l_s.shape == (1, 3, expected_size, expected_size, expected_size)


def test_branch_supervision_eval_mode_returns_plain_tensor() -> None:
    # THE PINNED INVARIANT: eval mode must always return a Tensor, unconditionally, even with
    # branch supervision on -- sliding-window inference and MONAI's inferer assume it.
    model = _build_model_with_branch_supervision(deep_supervision_levels=3)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, Tensor)
    assert out.shape == (1, 3, 32, 32, 32)


def test_branch_logits_none_when_supervision_disabled() -> None:
    model = _build_model_with_branch_supervision(
        deep_supervision_levels=3, supervise_branch_logits=False
    )
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert isinstance(out, MultiTaskOutput)
    assert out.branch_logits is None


def test_branch_logits_none_not_list_of_nones_for_fusion_with_no_ambiguity() -> None:
    # ConcatFusion has no ambiguity mechanism: forward_with_branch_logits falls back to
    # FusionBlock's default, which always reports None. The collected result must be a bare
    # None, not [None, None, None] -- a partially/never-populated list would let a loss
    # silently supervise nothing while still looking like a populated field.
    model = _build_model_with_branch_supervision(
        deep_supervision_levels=3, fusion_name="concat", supervise_branch_logits=True
    )
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)

    assert isinstance(out, MultiTaskOutput)
    assert out.branch_logits is None


def test_gradients_reach_branch_logit_convs_at_every_fused_level() -> None:
    # THE test that proves the plumbing is real: a detached or dropped level would pass every
    # shape assertion above and fail only here.
    model = _build_model_with_branch_supervision(deep_supervision_levels=1)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    out = model(x)
    assert out.branch_logits is not None

    total = sum(l_c.sum() + l_s.sum() for l_c, l_s in out.branch_logits)
    total.backward()

    for level, block in enumerate(model.fusion_blocks):
        cnn_grad = block.ambiguity.cnn_logits.weight.grad
        swin_grad = block.ambiguity.swin_logits.weight.grad
        assert cnn_grad is not None, f"level {level}: cnn_logits conv has no grad"
        assert swin_grad is not None, f"level {level}: swin_logits conv has no grad"
        assert torch.any(cnn_grad != 0.0), f"level {level}: cnn_logits conv grad is all zero"
        assert torch.any(swin_grad != 0.0), f"level {level}: swin_logits conv grad is all zero"


def test_forward_with_gates_unaffected_by_branch_supervision() -> None:
    # forward_with_gates keeps its existing (logits, gate_maps) contract untouched regardless
    # of supervise_branch_logits -- scripts/extract_gates.py depends on it.
    model = _build_model_with_branch_supervision(deep_supervision_levels=3)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, gates = model.forward_with_gates(x)

    assert isinstance(logits, Tensor)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert len(gates) == SWIN_NUM_LEVELS
    for gate in gates:
        assert gate is not None
        assert torch.all(gate > 0.0)
        assert torch.all(gate < 1.0)


def _full_cfg_with_branch(
    branch_enabled: bool,
    use_ambiguity: bool = True,
    confidence_enabled: bool = True,
    boundary_enabled: bool = False,
    loss_name: str = "multitask",
) -> object:
    """Same shape as `_full_cfg_with_aux`, extended with `training.loss.multitask.branch` and
    `model.fusion.use_ambiguity` so the branch-supervision builder guards can be exercised."""
    cfg = _full_cfg_with_aux(
        confidence_enabled=confidence_enabled,
        boundary_enabled=boundary_enabled,
        loss_name=loss_name,
    )
    cfg.model.fusion.use_ambiguity = use_ambiguity
    cfg.training.loss.multitask = {"branch": {"enabled": branch_enabled}}
    return cfg


def test_builder_raises_when_branch_enabled_but_use_ambiguity_false() -> None:
    cfg = _full_cfg_with_branch(
        branch_enabled=True, use_ambiguity=False, confidence_enabled=True, loss_name="multitask"
    )

    with pytest.raises(ValueError, match="use_ambiguity"):
        build_neurovision(cfg)


def test_builder_raises_when_branch_enabled_but_no_aux_head() -> None:
    cfg = _full_cfg_with_branch(
        branch_enabled=True,
        use_ambiguity=True,
        confidence_enabled=False,
        boundary_enabled=False,
        loss_name="multitask",
    )

    with pytest.raises(ValueError, match="auxiliary head"):
        build_neurovision(cfg)


def test_builder_raises_when_branch_enabled_but_loss_not_multitask() -> None:
    cfg = _full_cfg_with_branch(
        branch_enabled=True,
        use_ambiguity=True,
        confidence_enabled=True,
        loss_name="dice_ce",
    )

    with pytest.raises(ValueError, match="No other loss knows what to do"):
        build_neurovision(cfg)


def test_builder_branch_supervision_agreement_builds_fine() -> None:
    cfg = _full_cfg_with_branch(
        branch_enabled=True, use_ambiguity=True, confidence_enabled=True, loss_name="multitask"
    )

    model = build_neurovision(cfg)

    assert isinstance(model, NeuroVisionX)
    assert model.supervise_branch_logits is True


def test_builder_branch_disabled_by_default_does_not_supervise() -> None:
    # training.loss.multitask.branch.enabled defaults to False when the key is absent
    # entirely, matching configs/training/default.yaml.
    cfg = _full_cfg_with_aux(confidence_enabled=True, boundary_enabled=False, loss_name="multitask")

    model = build_neurovision(cfg)

    assert model.supervise_branch_logits is False


# ---------------------------------------------------------------------------
# 7. forward_with_auxiliary
# ---------------------------------------------------------------------------


def test_forward_with_auxiliary_shapes_both_heads_enabled() -> None:
    model = _build_model_with_heads(deep_supervision_levels=3, confidence=True, boundary=True)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, confidence_logits, boundary_logits = model.forward_with_auxiliary(x)

    assert isinstance(logits, Tensor)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert confidence_logits is not None
    assert confidence_logits.shape == (1, 3, 32, 32, 32)
    assert boundary_logits is not None
    assert boundary_logits.shape == (1, 3, 32, 32, 32)


def test_forward_with_auxiliary_returns_none_for_disabled_heads() -> None:
    model = _build_model_with_heads(deep_supervision_levels=3, confidence=False, boundary=False)
    model.eval()
    x = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        logits, confidence_logits, boundary_logits = model.forward_with_auxiliary(x)

    assert isinstance(logits, Tensor)
    assert confidence_logits is None
    assert boundary_logits is None


def test_forward_with_auxiliary_segmentation_output_is_plain_tensor_in_train_mode() -> None:
    # This is the case that would silently break if forward_with_auxiliary were routed
    # through forward: with both auxiliary heads enabled and the model in train() mode,
    # forward() itself would return a MultiTaskOutput -- forward_with_auxiliary must still
    # hand back a plain Tensor for the segmentation logits regardless of self.training,
    # exactly like forward_with_gates / forward_with_ambiguity already do.
    model = _build_model_with_heads(deep_supervision_levels=3, confidence=True, boundary=True)
    model.train()
    x = torch.randn(1, 4, 32, 32, 32)

    logits, confidence_logits, boundary_logits = model.forward_with_auxiliary(x)

    assert isinstance(logits, Tensor)
    assert not isinstance(logits, list)
    assert not isinstance(logits, MultiTaskOutput)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert confidence_logits is not None
    assert boundary_logits is not None
