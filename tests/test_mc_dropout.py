"""Tests for `neurovision.inference.mc_dropout`.

CPU only, tiny synthetic tensors, whole file well under 60 seconds. Most
tests use a small hand-built stub network (never the full `NeuroVisionX`) so
they run in a fraction of a second; test 12 is the one exception, using a
real (tiny-width) `NeuroVisionX` built through the actual Hydra config tree
to guard the three-way return-type hazard end to end.

Test 11 (`test_n5_stochastic_model_gives_positive_mutual_information`) is the
single most important test in this file: it is the only one that proves
dropout was genuinely active across the N passes rather than
`sliding_window_predict`'s internal `model.eval()` silently switching it back
off. Every other test here only checks internal arithmetic consistency of
the returned numbers, which would still hold even on N identical
deterministic passes.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import hydra
import pytest
import torch
from torch import Tensor, nn

from neurovision.inference.mc_dropout import (
    count_active_dropout,
    dropout_enabled,
    logits_from_mean_prob,
    mc_dropout_predict,
)

CPU = torch.device("cpu")
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Stub networks
# ---------------------------------------------------------------------------


class _NoDropoutNet(nn.Module):
    """4 -> 3 channel 1x1x1 conv. No dropout submodule at all."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 3, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _InertDropoutNet(nn.Module):
    """Has a real Dropout3d submodule, but at p=0 -- inert regardless of mode."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(4, 8, kernel_size=1)
        self.dropout = nn.Dropout3d(p=0.0)
        self.conv2 = nn.Conv3d(8, 3, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.dropout(self.conv1(x)))


class _StochasticNet(nn.Module):
    """4 -> 8 -> Dropout3d(p) -> 3 channel network. Genuinely stochastic at p > 0."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(4, 8, kernel_size=1)
        self.dropout = nn.Dropout3d(p=p)
        self.conv2 = nn.Conv3d(8, 3, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.dropout(self.conv1(x)))


# ---------------------------------------------------------------------------
# Config helper (mirrors tests/test_sliding_window.py)
# ---------------------------------------------------------------------------


def _compose_config(tmp_path: Path, overrides: list[str] | None = None) -> Any:
    """Composes the real Hydra config, with the mandatory `data.root_dir` set."""
    all_overrides = [f"data.root_dir={tmp_path}", "device=cpu"] + (overrides or [])
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        return hydra.compose(config_name="config", overrides=all_overrides)


def _full_neurovision_cfg(tmp_path: Path, overrides: list[str] | None = None) -> Any:
    """Composes a tiny, full `+experiment=neurovision` config for test 12.

    Swin disabled (`model.encoder.swin.enabled=false`) to keep the branch
    count -- and therefore the forward pass -- small and fast; a 4-level CNN
    (channels [8, 16, 24, 32]) gives a decoder with exactly 3 stages, which
    matches the production `deep_supervision_levels=3` this experiment file
    sets, and `model.head.dropout=0.1` gives the segmentation heads an
    active dropout module for `mc_dropout_predict` to exploit.
    """
    base_overrides = [
        "+experiment=neurovision",
        "model.encoder.cnn.channels=[8,16,24,32]",
        "model.encoder.cnn.blocks_per_stage=[1,1,1,1]",
        "model.encoder.swin.enabled=false",
        "model.head.dropout=0.1",
        "data.patch_size=[32,32,32]",
    ]
    return _compose_config(tmp_path, base_overrides + (overrides or []))


# ---------------------------------------------------------------------------
# 1. count_active_dropout
# ---------------------------------------------------------------------------


def test_count_active_dropout_counts_correctly() -> None:
    assert count_active_dropout(_NoDropoutNet()) == 0
    assert count_active_dropout(_InertDropoutNet()) == 0  # p=0.0 -- present but inert
    assert count_active_dropout(_StochasticNet(p=0.3)) == 1


# ---------------------------------------------------------------------------
# 2. dropout_enabled: activates dropout, leaves model.training and other
#    submodules untouched
# ---------------------------------------------------------------------------


def test_dropout_enabled_activates_only_dropout_submodules() -> None:
    model = _StochasticNet(p=0.5)
    model.eval()
    assert model.training is False

    with dropout_enabled(model) as n_activated:
        assert n_activated == 1
        assert model.dropout.training is True
        assert model.conv1.training is False
        assert model.conv2.training is False
        # model.training itself is never touched by dropout_enabled.
        assert model.training is False

    assert model.dropout.training is False


# ---------------------------------------------------------------------------
# 3. dropout_enabled restores EXACT prior flags
# ---------------------------------------------------------------------------


def test_dropout_enabled_restores_exact_prior_flags() -> None:
    model = _StochasticNet(p=0.5)
    model.train()  # everything, including dropout, starts in train mode
    model.dropout.eval()  # ...except we manually put the dropout submodule in eval
    assert model.dropout.training is False
    assert model.training is True

    with dropout_enabled(model):
        assert model.dropout.training is True

    # Restored to its own PRIOR flag (eval), not to model.training's flag (train).
    assert model.dropout.training is False
    # model.training was never touched by dropout_enabled, so it is unchanged.
    assert model.training is True


# ---------------------------------------------------------------------------
# 4. dropout_enabled restores flags even when the body raises
# ---------------------------------------------------------------------------


def test_dropout_enabled_restores_flags_on_exception() -> None:
    model = _StochasticNet(p=0.5)
    model.eval()

    with pytest.raises(RuntimeError, match="boom"):
        with dropout_enabled(model):
            assert model.dropout.training is True
            raise RuntimeError("boom")

    assert model.dropout.training is False


# ---------------------------------------------------------------------------
# 5. require_dropout guard
# ---------------------------------------------------------------------------


def test_mc_dropout_predict_raises_without_active_dropout(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _NoDropoutNet()
    image = torch.randn(1, 4, 8, 8, 8)

    with pytest.raises(ValueError, match="dropout"):
        mc_dropout_predict(model, image, cfg, CPU, num_samples=2)


def test_mc_dropout_predict_does_not_raise_when_require_dropout_false(tmp_path: Path) -> None:
    cfg = _compose_config(
        tmp_path,
        ["inference.sliding_window.roi_size=[8,8,8]", "inference.mc_dropout.require_dropout=false"],
    )
    model = _NoDropoutNet()
    image = torch.randn(1, 4, 8, 8, 8)

    result = mc_dropout_predict(model, image, cfg, CPU, num_samples=2)

    assert result.num_samples == 2
    assert result.mean_prob.shape == (1, 3, 8, 8, 8)


# ---------------------------------------------------------------------------
# 6. num_samples < 1 guard
# ---------------------------------------------------------------------------


def test_mc_dropout_predict_raises_on_zero_samples(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _StochasticNet(p=0.5)
    image = torch.randn(1, 4, 8, 8, 8)

    with pytest.raises(ValueError, match="num_samples"):
        mc_dropout_predict(model, image, cfg, CPU, num_samples=0)


# ---------------------------------------------------------------------------
# 7. output shapes
# ---------------------------------------------------------------------------


def test_output_shapes_and_num_samples_echoed(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _StochasticNet(p=0.4)
    image = torch.randn(1, 4, 16, 16, 16)

    result = mc_dropout_predict(model, image, cfg, CPU, num_samples=3)

    expected_shape = (1, 3, 16, 16, 16)
    assert result.mean_prob.shape == expected_shape
    assert result.predictive_entropy.shape == expected_shape
    assert result.expected_entropy.shape == expected_shape
    assert result.mutual_information.shape == expected_shape
    assert result.num_samples == 3


# ---------------------------------------------------------------------------
# 8. decomposition identity: predictive = expected + mutual information
# ---------------------------------------------------------------------------


def test_entropy_decomposition_identity_holds(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _StochasticNet(p=0.4)
    image = torch.randn(1, 4, 16, 16, 16)

    result = mc_dropout_predict(model, image, cfg, CPU, num_samples=4)

    reconstructed = result.expected_entropy + result.mutual_information
    torch.testing.assert_close(result.predictive_entropy, reconstructed, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 9. value ranges
# ---------------------------------------------------------------------------


def test_value_ranges_are_bounded(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _StochasticNet(p=0.4)
    image = torch.randn(1, 4, 16, 16, 16)

    result = mc_dropout_predict(model, image, cfg, CPU, num_samples=4)

    assert torch.all(result.mean_prob >= 0.0)
    assert torch.all(result.mean_prob <= 1.0)
    entropy_maps = (result.predictive_entropy, result.expected_entropy, result.mutual_information)
    for entropy_map in entropy_maps:
        assert torch.all(entropy_map >= 0.0)
        assert torch.all(entropy_map <= _LN2 + 1e-5)
    assert torch.all(result.mutual_information >= 0.0)


# ---------------------------------------------------------------------------
# 10. analytic anchor: N=1 gives MI exactly 0
# ---------------------------------------------------------------------------


def test_n1_gives_zero_mutual_information(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _StochasticNet(p=0.4)
    image = torch.randn(1, 4, 16, 16, 16)

    result = mc_dropout_predict(model, image, cfg, CPU, num_samples=1)

    zeros = torch.zeros_like(result.mutual_information)
    assert torch.allclose(result.mutual_information, zeros, atol=1e-6)


# ---------------------------------------------------------------------------
# 11. THE CRITICAL TEST: proves dropout was genuinely active across passes
# ---------------------------------------------------------------------------


def test_n5_stochastic_model_gives_positive_mutual_information(tmp_path: Path) -> None:
    """If sliding_window_predict's internal model.eval() were switching dropout
    back off (i.e. if `set_eval=False` were missing or broken), every pass
    would be numerically identical and mutual_information.max() would be
    exactly 0.0 -- this test would fail. Every other test in this file would
    still pass even with that bug present.
    """
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _StochasticNet(p=0.5)
    image = torch.randn(1, 4, 16, 16, 16)

    result = mc_dropout_predict(model, image, cfg, CPU, num_samples=5)

    assert result.mutual_information.max().item() > 0.0


# ---------------------------------------------------------------------------
# 12. full NeuroVisionX regression guard for the three-way return type
# ---------------------------------------------------------------------------


def test_full_neurovisionx_with_aux_heads_and_deep_supervision(tmp_path: Path) -> None:
    """Builds a tiny but real NeuroVisionX (both auxiliary heads enabled,
    deep_supervision_levels=3, Swin branch disabled for speed) and runs it
    through mc_dropout_predict end to end. This is the regression guard for
    the three-way NeuroVisionX.forward return type: if MC-dropout ever tripped
    the model into returning a MultiTaskOutput or a list where the sliding-
    window inferer needs a plain Tensor, this test would fail with a type
    error rather than silently producing a wrong-shaped map.
    """
    from neurovision.models import baseline  # noqa: F401  (registers unet3d/swinunetr)
    from neurovision.models import neurovision as neurovision_module  # noqa: F401
    from neurovision.models.registry import build_model

    cfg = _full_neurovision_cfg(tmp_path, ["inference.mc_dropout.num_samples=2"])
    model = build_model(cfg)
    model.eval()

    x = torch.randn(1, 4, 32, 32, 32)
    result = mc_dropout_predict(model, x, cfg, CPU, num_samples=2)

    assert result.num_samples == 2
    for tensor in (
        result.mean_prob,
        result.predictive_entropy,
        result.expected_entropy,
        result.mutual_information,
    ):
        assert tensor.shape == (1, 3, 32, 32, 32)
        assert torch.isfinite(tensor).all()


# ---------------------------------------------------------------------------
# 13. seeding: reproducibility and difference across seeds
# ---------------------------------------------------------------------------


def test_same_seed_gives_bitwise_identical_mean_prob(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _StochasticNet(p=0.5)
    image = torch.randn(1, 4, 16, 16, 16)

    result1 = mc_dropout_predict(model, image, cfg, CPU, num_samples=4, seed=123)
    result2 = mc_dropout_predict(model, image, cfg, CPU, num_samples=4, seed=123)

    assert torch.equal(result1.mean_prob, result2.mean_prob)


def test_different_seeds_give_different_mean_prob(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _StochasticNet(p=0.5)
    image = torch.randn(1, 4, 16, 16, 16)

    result1 = mc_dropout_predict(model, image, cfg, CPU, num_samples=4, seed=123)
    result2 = mc_dropout_predict(model, image, cfg, CPU, num_samples=4, seed=456)

    assert not torch.equal(result1.mean_prob, result2.mean_prob)


# ---------------------------------------------------------------------------
# 14. seeding does not perturb the caller's global RNG stream
# ---------------------------------------------------------------------------


def test_seeded_call_does_not_perturb_global_rng_state(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[16,16,16]"])
    model = _StochasticNet(p=0.5)
    image = torch.randn(1, 4, 16, 16, 16)

    state_before = torch.get_rng_state()
    mc_dropout_predict(model, image, cfg, CPU, num_samples=3, seed=999)
    state_after = torch.get_rng_state()

    assert torch.equal(state_before, state_after)


# ---------------------------------------------------------------------------
# 15. logits_from_mean_prob round trip
# ---------------------------------------------------------------------------


def test_logits_from_mean_prob_round_trips() -> None:
    p = torch.linspace(0.01, 0.99, steps=50)

    recovered = torch.sigmoid(logits_from_mean_prob(p))

    torch.testing.assert_close(recovered, p, atol=1e-4, rtol=1e-4)


def test_logits_from_mean_prob_finite_at_boundaries() -> None:
    edge = torch.tensor([0.0, 1.0])

    edge_logits = logits_from_mean_prob(edge)

    assert torch.isfinite(edge_logits).all()


# ---------------------------------------------------------------------------
# 16. logits_from_mean_prob preserves discretization
# ---------------------------------------------------------------------------


def test_logits_from_mean_prob_preserves_threshold() -> None:
    torch.manual_seed(0)
    p = torch.rand(1000)

    logits = logits_from_mean_prob(p)

    assert torch.equal(p >= 0.5, logits >= 0.0)


# ---------------------------------------------------------------------------
# 17. output_device: "cpu" yields CPU tensors in every output map
# ---------------------------------------------------------------------------


def test_output_device_cpu_yields_cpu_tensors(tmp_path: Path) -> None:
    cfg = _compose_config(
        tmp_path,
        [
            "inference.sliding_window.roi_size=[16,16,16]",
            'inference.sliding_window.output_device="cpu"',
        ],
    )
    model = _StochasticNet(p=0.4)
    image = torch.randn(1, 4, 16, 16, 16)

    result = mc_dropout_predict(model, image, cfg, CPU, num_samples=3)

    for tensor in (
        result.mean_prob,
        result.predictive_entropy,
        result.expected_entropy,
        result.mutual_information,
    ):
        assert tensor.device.type == "cpu"


class _DeadBranchDropoutNet(nn.Module):
    """Active Dropout3d whose branch is scaled by exactly zero.

    A miniature of the real mechanism found in `CNNEncoder`: with
    `zero_init_residual=True`, `norm2.weight` is zeroed in every
    `ResidualBlock`, so the residual branch outputs exactly 0 and annihilates
    the `Dropout3d` applied before `conv2`. Ten active dropout modules, ten
    numerically identical passes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(4, 8, kernel_size=1)
        self.dropout = nn.Dropout3d(p=0.5)
        self.conv2 = nn.Conv3d(8, 3, kernel_size=1)
        self.skip = nn.Conv3d(4, 3, kernel_size=1)
        # Stands in for the zeroed norm2.weight.
        self.scale = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.skip(x) + self.scale * self.conv2(self.dropout(self.conv1(x)))


def test_warns_when_all_passes_were_deterministic_despite_active_dropout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Active dropout modules are NOT proof the N passes actually differed.

    `require_dropout` only counts modules, so it cannot catch a model whose
    dropout perturbation is annihilated downstream. Without this warning the
    caller gets a plausible-looking, entirely meaningless uncertainty map.
    """
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _DeadBranchDropoutNet().eval()

    with caplog.at_level(logging.WARNING):
        out = mc_dropout_predict(model, torch.randn(1, 4, 8, 8, 8), cfg, CPU, num_samples=4)

    assert out.mutual_information.max().item() == 0.0
    assert any(
        "deterministic" in record.message for record in caplog.records
    ), "mc_dropout_predict must warn when every pass was identical despite active dropout"


def test_no_deterministic_warning_at_num_samples_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """At N=1 mutual information is 0 by construction -- not a misconfiguration."""
    cfg = _compose_config(tmp_path, ["inference.sliding_window.roi_size=[8,8,8]"])
    model = _StochasticNet(p=0.5).eval()

    with caplog.at_level(logging.WARNING):
        mc_dropout_predict(model, torch.randn(1, 4, 8, 8, 8), cfg, CPU, num_samples=1)

    assert not any("deterministic" in record.message for record in caplog.records)
