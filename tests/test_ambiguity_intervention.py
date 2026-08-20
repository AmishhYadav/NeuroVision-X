"""Tests for scripts/ambiguity_intervention.py.

Follows the exact pattern of `tests/test_extract_ambiguity.py`: the script lives under
`scripts/`, not `src/`, so it is loaded via `importlib.util.spec_from_file_location` rather
than a normal package import, and `build_model` is monkeypatched to hand back a small, REAL
`NeuroVisionX` (built at tiny widths, with `adaptive_gated` fusion so it actually has an
ambiguity-conditioned gate to intervene on) rather than a stub -- the whole point of this
script is exercising `AdaptiveGatedFusion.ambiguity_transform` through `forward_with_gates`,
which a stub cannot stand in for.

No case here uses real BraTS data: synthetic `.npy` + `meta.json` trees are written under
`tmp_path`, mirroring `scripts/preprocess.py`'s output shape. CPU only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from neurovision.models import baseline  # noqa: F401 -- registers "unet3d"
from neurovision.models.decoder.unet_decoder import UNetDecoder
from neurovision.models.encoders.cnn import CNNEncoder
from neurovision.models.encoders.swin import SwinEncoder
from neurovision.models.fusion.adaptive_fusion import AdaptiveGatedFusion
from neurovision.models.heads.multitask import MultiTaskHead
from neurovision.models.neurovision import NeuroVisionX
from neurovision.training.checkpoint import save_checkpoint
from neurovision.utils.io import write_json, write_yaml

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ambiguity_intervention.py"
_spec = importlib.util.spec_from_file_location("ambiguity_intervention_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ambiguity_intervention_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["ambiguity_intervention_script"] = ambiguity_intervention_script
_spec.loader.exec_module(ambiguity_intervention_script)

run_intervention = ambiguity_intervention_script.run_intervention
_CONDITIONS = ambiguity_intervention_script._CONDITIONS

# Small, but real: a real Swin branch + windowed cross-attention needs enough spatial extent
# to have something to window over. Matches tests/test_extract_ambiguity.py's AUX_* recipe.
CNN_CHANNELS = [8, 16, 24, 32]
CNN_BLOCKS = [1, 1, 1, 1]
SWIN_FEATURE_SIZE = 12
SWIN_NUM_LEVELS = 3
NUM_GROUPS = 8
FUSION_HEADS = 4
NUM_REGIONS = 3
INPUT_SHAPE: tuple[int, int, int] = (32, 32, 32)


def _build_tiny_neurovision() -> NeuroVisionX:
    """A small, real NeuroVisionX with `adaptive_gated` fusion (use_ambiguity=True)."""
    cnn = CNNEncoder(
        in_channels=4,
        channels=CNN_CHANNELS,
        blocks_per_stage=CNN_BLOCKS,
        num_groups=NUM_GROUPS,
        dropout=0.0,
        use_checkpoint=False,
        zero_init_residual=False,
    )
    swin = SwinEncoder(
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
    blocks = [
        AdaptiveGatedFusion(
            cnn.out_channels[i + 1],
            swin.out_channels[i],
            num_heads=FUSION_HEADS,
            window_size=4,
            num_groups=NUM_GROUPS,
            num_regions=NUM_REGIONS,
            use_ambiguity=True,
        )
        for i in range(swin.num_levels)
    ]
    decoder = UNetDecoder(
        skip_channels=cnn.out_channels,
        decoder_channels=None,
        blocks_per_stage=1,
        num_groups=NUM_GROUPS,
        dropout=0.0,
        upsample="deconv",
        use_attention_gates=False,
        use_checkpoint=False,
    )
    heads = MultiTaskHead(
        decoder_channels=decoder.out_channels,
        out_channels=NUM_REGIONS,
        deep_supervision_levels=1,
        confidence=False,
        boundary=False,
        confidence_num_groups=NUM_GROUPS,
        boundary_num_groups=NUM_GROUPS,
    )
    return NeuroVisionX(
        cnn_encoder=cnn,
        swin_encoder=swin,
        fusion_blocks=nn.ModuleList(blocks),
        decoder=decoder,
        out_channels=NUM_REGIONS,
        deep_supervision_levels=1,
        head_dropout=0.0,
        heads=heads,
    )


def _write_synthetic_case(prep_dir: Path, case_id: str, seed: int) -> None:
    """Writes one synthetic preprocessed case: image.npy, label.npy, meta.json."""
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    image = rng.standard_normal((4, *INPUT_SHAPE)).astype(np.float16)
    np.save(case_dir / "image.npy", image)

    d, h, w = INPUT_SHAPE
    label = np.zeros(INPUT_SHAPE, dtype=np.uint8)
    label[d // 2 - 4 : d // 2 + 4, h // 2 - 4 : h // 2 + 4, w // 2 - 4 : w // 2 + 4] = 2
    label[d // 2 - 3 : d // 2 + 3, h // 2 - 3 : h // 2 + 3, w // 2 - 3 : w // 2 + 3] = 1
    label[d // 2 - 2 : d // 2 + 2, h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = 3
    np.save(case_dir / "label.npy", label)

    write_json(
        {
            "case_id": case_id,
            "has_label": True,
            "bbox": [[0, d], [0, h], [0, w]],
            "original_shape": list(INPUT_SHAPE),
            "spacing": [1.0, 1.0, 1.0],
        },
        case_dir / "meta.json",
    )


def _save_stub_checkpoint(checkpoint_dir: Path, model: nn.Module) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    fake_trained_cfg = OmegaConf.create({"model": {"name": "neurovision"}})
    save_checkpoint(
        checkpoint_dir, model, optimizer, epoch=0, global_step=0, cfg=fake_trained_cfg, is_best=True
    )


def _compose_cfg(
    tmp_path: Path,
    prep_dir: Path,
    splits_path: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    num_cases: str = "null",
):
    """Composes the REAL Hydra config, tiny-sized -- mirrors test_extract_ambiguity.py."""
    overrides = [
        f"data.root_dir={tmp_path}",
        f"data.preprocessing.out_dir={prep_dir}",
        f"data.splits.path={splits_path}",
        f"data.patch_size=[{INPUT_SHAPE[0]},{INPUT_SHAPE[1]},{INPUT_SHAPE[2]}]",
        "data.num_workers=0",
        "data.dataset_type=dataset",
        f"training.checkpoint.dir={checkpoint_dir}",
        f"explainability.ambiguity.out_dir={out_dir}",
        "explainability.ambiguity.split=test",
        f"explainability.ambiguity.num_cases={num_cases}",
        "inference.sliding_window.overlap=0.0",
        "wandb.mode=disabled",
        "device=cpu",
        "seed=42",
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def _setup(tmp_path: Path, n_cases: int = 2):
    """Shared fixture: N synthetic cases, a matching split file, and a tiny checkpoint."""
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = [f"case_{i:03d}" for i in range(n_cases)]
    for i, case_id in enumerate(case_ids):
        _write_synthetic_case(prep_dir, case_id, seed=i)
    write_yaml({"train": [], "val": [], "test": case_ids}, splits_path)

    model = _build_tiny_neurovision()
    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, model)

    return prep_dir, splits_path, checkpoint_dir, case_ids


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_cases: int = 2) -> pd.DataFrame:
    prep_dir, splits_path, checkpoint_dir, case_ids = _setup(tmp_path, n_cases=n_cases)
    monkeypatch.setattr(
        ambiguity_intervention_script, "build_model", lambda cfg: _build_tiny_neurovision()
    )
    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(
        tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, num_cases=str(n_cases)
    )
    per_case_df = run_intervention(cfg)
    # The script appends "_intervention" to out_dir, so it can never overwrite an
    # extract_ambiguity.py run's output at the same base path.
    return per_case_df, Path(f"{out_dir}_intervention"), case_ids


# ---------------------------------------------------------------------------
# 1. All four conditions run against a tiny stub config
# ---------------------------------------------------------------------------


def test_all_four_conditions_run_and_produce_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    per_case_df, out_dir, case_ids = _run(tmp_path, monkeypatch, n_cases=2)

    assert set(per_case_df["condition"].unique()) == set(_CONDITIONS)
    assert len(per_case_df) == len(case_ids) * len(_CONDITIONS)
    for case_id in case_ids:
        assert case_id in set(per_case_df["case_id"])

    for region in ("ET", "TC", "WT"):
        assert f"dice_{region}" in per_case_df.columns
    # At least one fusion level's gate divergence columns must be present.
    assert any(c.startswith("gate_absdiff_level") for c in per_case_df.columns)
    assert any(c.startswith("gate_spearman_level") for c in per_case_df.columns)


# ---------------------------------------------------------------------------
# 2. zero produces strictly larger gate divergence than baseline's (exactly 0)
# ---------------------------------------------------------------------------


def test_zero_condition_gate_divergence_exceeds_baselines_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    per_case_df, out_dir, case_ids = _run(tmp_path, monkeypatch, n_cases=2)

    absdiff_cols = [c for c in per_case_df.columns if c.startswith("gate_absdiff_level")]
    assert absdiff_cols, "expected at least one gate_absdiff_level* column"

    baseline_rows = per_case_df[per_case_df["condition"] == "baseline"]
    zero_rows = per_case_df[per_case_df["condition"] == "zero"]

    for col in absdiff_cols:
        # baseline compared against itself: exactly 0 by construction.
        assert (baseline_rows[col] == 0.0).all()
        # zero actively changes the gate's input -- at least one level, on at least one case,
        # must diverge from baseline more than baseline diverges from itself.
    assert (zero_rows[absdiff_cols].to_numpy() > 0.0).any()


# ---------------------------------------------------------------------------
# 3. Per-case CSV is rewritten incrementally
# ---------------------------------------------------------------------------


def test_per_case_csv_is_rewritten_incrementally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prep_dir, splits_path, checkpoint_dir, case_ids = _setup(tmp_path, n_cases=2)
    monkeypatch.setattr(
        ambiguity_intervention_script, "build_model", lambda cfg: _build_tiny_neurovision()
    )
    base_out_dir = tmp_path / "ambiguity_out"
    out_dir = Path(f"{base_out_dir}_intervention")
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, base_out_dir, num_cases="2")

    written_sizes: list[int] = []
    original_to_csv = pd.DataFrame.to_csv

    def _spy_to_csv(self, path_or_buf=None, *args, **kwargs):
        result = original_to_csv(self, path_or_buf, *args, **kwargs)
        per_case_path = out_dir / "intervention_per_case.csv"
        is_per_case_write = path_or_buf is not None and str(path_or_buf) == str(per_case_path)
        if is_per_case_write and per_case_path.is_file():
            written_sizes.append(len(pd.read_csv(per_case_path)))
        return result

    monkeypatch.setattr(pd.DataFrame, "to_csv", _spy_to_csv)
    run_intervention(cfg)

    # One write per case (each write carries that case's rows plus every prior case's), so row
    # counts must be non-decreasing and the file must exist after the FIRST case already --
    # i.e. strictly before the run finished, not only once at the very end.
    assert len(written_sizes) >= len(case_ids)
    assert written_sizes == sorted(written_sizes)
    assert written_sizes[0] == len(_CONDITIONS)  # first case: one row per condition
    assert written_sizes[-1] == len(case_ids) * len(_CONDITIONS)

    final_df = pd.read_csv(out_dir / "intervention_per_case.csv")
    assert len(final_df) == len(case_ids) * len(_CONDITIONS)


# ---------------------------------------------------------------------------
# 4. Summary's paired difference against baseline is exactly 0 for baseline row
# ---------------------------------------------------------------------------


def test_summary_baseline_row_diff_vs_baseline_is_exactly_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _per_case_df, out_dir, _case_ids = _run(tmp_path, monkeypatch, n_cases=2)

    summary_df = pd.read_csv(out_dir / "intervention_summary.csv", index_col="condition")
    assert "baseline" in summary_df.index

    for region in ("ET", "TC", "WT", "mean"):
        col = f"dice_{region}_diff_vs_baseline"
        assert col in summary_df.columns
        assert summary_df.loc["baseline", col] == pytest.approx(0.0, abs=1e-12)
        assert summary_df.loc["baseline", f"dice_{region}_diff_lo"] == pytest.approx(0.0, abs=1e-12)
        assert summary_df.loc["baseline", f"dice_{region}_diff_hi"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# extra coverage: config guards
# ---------------------------------------------------------------------------


def test_run_intervention_raises_for_model_with_no_forward_with_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class _NoGatesModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(4, NUM_REGIONS, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    prep_dir, splits_path, checkpoint_dir, _case_ids = _setup(tmp_path, n_cases=1)
    monkeypatch.setattr(ambiguity_intervention_script, "build_model", lambda cfg: _NoGatesModel())
    _save_stub_checkpoint(checkpoint_dir, _NoGatesModel())

    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, num_cases="1")

    with pytest.raises(TypeError, match="forward_with_gates"):
        run_intervention(cfg)


def test_run_intervention_raises_when_no_fusion_block_has_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def _build_no_ambiguity_model() -> NeuroVisionX:
        model = _build_tiny_neurovision()
        for block in model.fusion_blocks:
            block.ambiguity = None  # simulate a content-only-ablation checkpoint
        return model

    prep_dir, splits_path, checkpoint_dir, _case_ids = _setup(tmp_path, n_cases=1)
    monkeypatch.setattr(
        ambiguity_intervention_script, "build_model", lambda cfg: _build_no_ambiguity_model()
    )
    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, num_cases="1")

    with pytest.raises(ValueError, match="use_ambiguity=True"):
        run_intervention(cfg)
