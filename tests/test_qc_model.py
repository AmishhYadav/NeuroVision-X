"""Tests for neurovision.models.qc: the SegQC segmentation QC regressor.

CPU only, tiny tensors, whole file runs in well under a second. Configs are
built with `OmegaConf.create` mirroring `configs/model/segqc.yaml`, following
the same `_cfg`-style helper pattern as `tests/test_models.py`, except for
the last test, which composes the real `configs/` tree the same way
`tests/test_conformal_script.py::test_conformal_config_block_is_reachable_at_the_composed_path`
does -- that is the one that would catch a config key read from the wrong
composed path even though every hand-built-fixture test above it passed.
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from neurovision.models import qc  # noqa: F401  (registers "segqc")
from neurovision.models.qc import SegQC, build_segqc, predicted_dice
from neurovision.models.registry import build_model

# Real configs/ directory, resolved relative to this file -- never a
# hardcoded absolute path. Same pattern as tests/test_conformal_script.py's
# _CONFIG_DIR.
_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _segqc_cfg(**overrides: object) -> object:
    """Builds a full composed config mirroring configs/model/segqc.yaml."""
    base = {
        "model": {
            "name": "segqc",
            "in_channels": 3,
            "widths": [16, 32, 64, 128],
            "num_groups": 8,
            "dropout": 0.1,
        },
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.model, key, value, merge=True)
    return cfg


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_forward_shape_on_tiny_volume() -> None:
    model = SegQC()
    model.eval()
    x = torch.randn(1, 3, 32, 32, 32)

    start = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    elapsed = time.perf_counter() - start

    assert out.shape == (1,)
    assert elapsed < 1.0, f"forward took {elapsed:.3f}s, must be under 1s"


def test_handles_variable_input_sizes() -> None:
    # Proves the global-average-pool head, not a flatten-then-linear head:
    # a flatten head bakes in one spatial size at construction (via the
    # first Linear layer's in_features) and would raise a shape mismatch
    # the second call here, since 40x24x36 pools to a different flattened
    # length than 32x32x32 would.
    model = SegQC()
    model.eval()

    x_a = torch.randn(1, 3, 32, 32, 32)
    x_b = torch.randn(1, 3, 40, 24, 36)

    with torch.no_grad():
        out_a = model(x_a)
        out_b = model(x_b)

    assert out_a.shape == (1,)
    assert out_b.shape == (1,)


def test_batch_dimension_is_respected() -> None:
    model = SegQC()
    model.eval()
    x = torch.randn(4, 3, 24, 24, 24)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (4,)
    # Not all four outputs collapsed to the same value -- would happen if,
    # e.g., the batch dimension were accidentally pooled away too.
    assert out.unique().numel() > 1


def test_predicted_dice_is_in_unit_interval() -> None:
    logits = torch.tensor([-50.0, -1.0, 0.0, 1.0, 50.0, -1e6, 1e6])
    dice = predicted_dice(logits)

    assert torch.isfinite(dice).all()
    assert (dice >= 0.0).all()
    assert (dice <= 1.0).all()


def test_forward_returns_logits_not_probabilities() -> None:
    # Pins the logit-not-probability contract decisively: build several
    # randomly initialised models (seeded, reproducible), then FORCE each
    # one's final linear layer to a large scale so its raw output is
    # unambiguously outside [0, 1] -- if sigmoid were secretly baked into
    # forward (the bug this test exists to catch), the output would stay
    # inside [0, 1] no matter how the weights are scaled, since sigmoid
    # saturates but never leaves that range.
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 16, 16, 16, generator=generator)

    for seed in range(3):
        model = SegQC()
        gen = torch.Generator().manual_seed(seed)
        for param in model.parameters():
            param.data = torch.randn(param.shape, generator=gen) * 0.1
        # Force the final Linear layer's weight/bias to a large scale so the
        # output is decisively outside [0, 1] regardless of the (seeded)
        # random init drawn above.
        final_linear = model.head[-1]
        assert isinstance(final_linear, torch.nn.Linear)
        final_linear.weight.data *= 1000.0
        final_linear.bias.data += 1000.0

        model.eval()
        with torch.no_grad():
            out = model(x)

        assert torch.any((out < 0.0) | (out > 1.0)), (
            f"seed={seed}: forward output {out.tolist()} stayed inside [0, 1] -- "
            "looks like a sigmoid is baked into forward()"
        )


def test_registry_lookup_builds_the_model() -> None:
    cfg = _segqc_cfg()
    model = build_model(cfg)
    assert isinstance(model, SegQC)


def test_config_composes_and_selects_segqc() -> None:
    overrides = ["model=segqc", "data.root_dir=/unused/for/this/test"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    assert cfg.model.name == "segqc"
    assert cfg.model.in_channels == 3


def test_num_groups_mismatch_raises() -> None:
    # 30 is not evenly divisible by 8 -- both numbers must be named in the
    # error, not just "GroupNorm failed".
    with pytest.raises(ValueError) as exc_info:
        SegQC(widths=(16, 30), num_groups=8)
    message = str(exc_info.value)
    assert "30" in message
    assert "8" in message


def test_parameter_count_is_small() -> None:
    cfg = _segqc_cfg()
    model = build_segqc(cfg)
    assert _n_params(model) < 5e6
