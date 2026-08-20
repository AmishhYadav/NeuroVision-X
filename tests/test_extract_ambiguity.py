"""Tests for scripts/extract_ambiguity.py.

Follows the exact pattern of `tests/test_extract_gates.py`: the script lives
under `scripts/`, not `src/`, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import, and `build_model` is monkeypatched to hand back a small stub model
(never a full `NeuroVisionX`, which is far too slow for a sub-second test
suite) whose `forward_with_ambiguity` behaviour each test controls directly.

No case here uses real BraTS data: synthetic `.npy` + `meta.json` trees are
written under `tmp_path`, mirroring `scripts/preprocess.py`'s output shape.
CPU only, and the whole file runs in a few seconds.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from neurovision.models import baseline  # noqa: F401 -- registers "unet3d"
from neurovision.models.decoder.unet_decoder import UNetDecoder
from neurovision.models.encoders.cnn import CNNEncoder
from neurovision.models.encoders.swin import SwinEncoder
from neurovision.models.fusion.adaptive_fusion import AdaptiveGatedFusion, ConcatFusion
from neurovision.models.heads.multitask import MultiTaskHead
from neurovision.models.neurovision import NeuroVisionX
from neurovision.training.checkpoint import save_checkpoint
from neurovision.utils.io import write_json, write_yaml

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_ambiguity.py"
_spec = importlib.util.spec_from_file_location("extract_ambiguity_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
extract_ambiguity_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["extract_ambiguity_script"] = extract_ambiguity_script
_spec.loader.exec_module(extract_ambiguity_script)

_AmbiguityAtLevel = extract_ambiguity_script._AmbiguityAtLevel
summarize_case_ambiguity = extract_ambiguity_script.summarize_case_ambiguity
run_extraction = extract_ambiguity_script.run_extraction

VOLUME_SHAPE: tuple[int, int, int] = (16, 16, 16)
ROI_SIZE: tuple[int, int, int] = (8, 8, 8)
NUM_REGIONS = 3


# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------


class _FixedAmbiguityModel(nn.Module):
    """Stub whose `forward_with_ambiguity` returns a caller-supplied maps list."""

    def __init__(
        self, maps_factory: Callable[[Tensor], list[Tensor | None]], num_regions: int = NUM_REGIONS
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, num_regions, kernel_size=3, padding=1)
        self._maps_factory = maps_factory

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def forward_with_ambiguity(self, x: Tensor) -> tuple[Tensor, list[Tensor | None]]:
        return self.conv(x), self._maps_factory(x)


class _StubAmbiguityModel(nn.Module):
    """A small model exposing one real ambiguity level (stride 2), like production level 0."""

    def __init__(self, num_regions: int = NUM_REGIONS) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, num_regions, kernel_size=3, padding=1)
        self.num_regions = num_regions

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def forward_with_ambiguity(self, x: Tensor) -> tuple[Tensor, list[Tensor | None]]:
        logits = self.conv(x)
        b, _, d, h, w = x.shape
        ambiguity = torch.rand(b, 3 * self.num_regions, d // 2, h // 2, w // 2)
        return logits, [ambiguity]


# ---------------------------------------------------------------------------
# 1-2. _AmbiguityAtLevel
# ---------------------------------------------------------------------------


def test_ambiguity_at_level_matches_input_spatial_shape_and_channel_count():
    def maps_factory(x: Tensor) -> list[Tensor | None]:
        b, _, d, h, w = x.shape
        return [torch.rand(b, 3 * NUM_REGIONS, d // 2, h // 2, w // 2)]

    model = _FixedAmbiguityModel(maps_factory)
    wrapper = _AmbiguityAtLevel(model, level=0)

    x = torch.randn(2, 4, *ROI_SIZE)
    out = wrapper(x)

    assert out.shape == (2, 3 * NUM_REGIONS, *ROI_SIZE)


def test_ambiguity_at_level_raises_on_out_of_range_level():
    model = _FixedAmbiguityModel(lambda x: [torch.rand(x.shape[0], 3 * NUM_REGIONS, 4, 4, 4)])
    wrapper = _AmbiguityAtLevel(model, level=5)

    with pytest.raises(ValueError, match="out of range"):
        wrapper(torch.randn(1, 4, *ROI_SIZE))


def test_ambiguity_at_level_raises_when_map_is_none():
    model = _FixedAmbiguityModel(lambda x: [None, torch.rand(x.shape[0], 3 * NUM_REGIONS, 4, 4, 4)])
    wrapper = _AmbiguityAtLevel(model, level=0)

    with pytest.raises(ValueError, match="no ambiguity map"):
        wrapper(torch.randn(1, 4, *ROI_SIZE))


# ---------------------------------------------------------------------------
# 3. summarize_case_ambiguity -- exact column set, NaN on empty foreground
# ---------------------------------------------------------------------------


def _expected_columns(region_names: tuple[str, ...] = ("ET", "TC", "WT")) -> set[str]:
    columns = set()
    for region in region_names:
        columns.add(f"amb_dis_mean_{region}")
        columns.add(f"amb_dis_max_{region}")
        columns.add(f"amb_dis_mean_fg_{region}")
        columns.add(f"amb_hcnn_mean_fg_{region}")
        columns.add(f"amb_hswin_mean_fg_{region}")
    columns.add("amb_dis_mean_fg_mean")
    return columns


def test_summarize_case_ambiguity_returns_exact_documented_columns():
    shape = (NUM_REGIONS, 6, 6, 6)
    disagreement = torch.rand(*shape)
    entropy_cnn = torch.rand(*shape)
    entropy_swin = torch.rand(*shape)
    regions = torch.zeros(*shape)
    regions[:, 2:4, 2:4, 2:4] = 1.0  # nonempty foreground for every region

    row = summarize_case_ambiguity(disagreement, entropy_cnn, entropy_swin, regions)

    assert set(row.keys()) == _expected_columns()
    assert all(np.isfinite(v) for v in row.values())


def test_summarize_case_ambiguity_fg_columns_are_nan_when_region_empty():
    shape = (NUM_REGIONS, 6, 6, 6)
    disagreement = torch.rand(*shape)
    entropy_cnn = torch.rand(*shape)
    entropy_swin = torch.rand(*shape)
    regions = torch.zeros(*shape)
    regions[1, 2:4, 2:4, 2:4] = 1.0  # only TC (index 1) has foreground

    row = summarize_case_ambiguity(disagreement, entropy_cnn, entropy_swin, regions)

    # ET (index 0) and WT (index 2) have empty predicted foreground.
    for region in ("ET", "WT"):
        assert np.isnan(row[f"amb_dis_mean_fg_{region}"])
        assert np.isnan(row[f"amb_hcnn_mean_fg_{region}"])
        assert np.isnan(row[f"amb_hswin_mean_fg_{region}"])
    # TC has foreground, so its fg columns must be real numbers.
    assert np.isfinite(row["amb_dis_mean_fg_TC"])
    assert np.isfinite(row["amb_hcnn_mean_fg_TC"])
    assert np.isfinite(row["amb_hswin_mean_fg_TC"])
    # The whole-volume (non-fg) columns are unaffected by an empty region.
    assert np.isfinite(row["amb_dis_mean_ET"])
    assert np.isfinite(row["amb_dis_max_ET"])
    # amb_dis_mean_fg_mean NaN-skips: only TC contributed, so it equals TC's value.
    assert row["amb_dis_mean_fg_mean"] == pytest.approx(row["amb_dis_mean_fg_TC"])


def test_summarize_case_ambiguity_all_empty_gives_nan_mean():
    shape = (NUM_REGIONS, 6, 6, 6)
    disagreement = torch.rand(*shape)
    entropy_cnn = torch.rand(*shape)
    entropy_swin = torch.rand(*shape)
    regions = torch.zeros(*shape)  # every region empty

    row = summarize_case_ambiguity(disagreement, entropy_cnn, entropy_swin, regions)

    assert np.isnan(row["amb_dis_mean_fg_mean"])
    for region in ("ET", "TC", "WT"):
        assert np.isnan(row[f"amb_dis_mean_fg_{region}"])


# ---------------------------------------------------------------------------
# 4. summarize_case_ambiguity never reads a label
# ---------------------------------------------------------------------------


def test_summarize_case_ambiguity_has_no_label_parameter_and_succeeds_without_one():
    params = inspect.signature(summarize_case_ambiguity).parameters
    assert "label" not in params

    # No label was ever constructed, loaded, or passed anywhere above --
    # this call succeeds using only maps derived from the model's own
    # prediction.
    shape = (NUM_REGIONS, 4, 4, 4)
    row = summarize_case_ambiguity(
        torch.rand(*shape), torch.rand(*shape), torch.rand(*shape), torch.ones(*shape)
    )
    assert isinstance(row, dict)
    assert len(row) == len(_expected_columns())


# ---------------------------------------------------------------------------
# 5. End-to-end smoke test
# ---------------------------------------------------------------------------


def _write_synthetic_case(prep_dir: Path, case_id: str, seed: int) -> None:
    """Writes one synthetic preprocessed case: image.npy, label.npy, meta.json.

    Matches exactly what `scripts/preprocess.py` writes -- see
    `scripts/smoke_test.py`'s `_write_synthetic_case` for the same recipe.
    """
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    image = rng.standard_normal((4, *VOLUME_SHAPE)).astype(np.float16)
    np.save(case_dir / "image.npy", image)

    d, h, w = VOLUME_SHAPE
    label = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    label[d // 2 - 3 : d // 2 + 3, h // 2 - 3 : h // 2 + 3, w // 2 - 3 : w // 2 + 3] = 2
    label[d // 2 - 2 : d // 2 + 2, h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = 1
    label[d // 2 - 1 : d // 2 + 1, h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = 3
    np.save(case_dir / "label.npy", label)

    write_json(
        {
            "case_id": case_id,
            "has_label": True,
            "bbox": [[0, d], [0, h], [0, w]],
            "original_shape": list(VOLUME_SHAPE),
            "spacing": [1.0, 1.0, 1.0],
        },
        case_dir / "meta.json",
    )


def _compose_cfg(
    tmp_path: Path,
    prep_dir: Path,
    splits_path: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    logits_dir: Path | None = None,
    include_auxiliary: bool = False,
    include_gates: bool = False,
):
    """Composes the REAL Hydra config, tiny-sized -- mirrors scripts/smoke_test.py.

    `include_auxiliary` and `include_gates` both default to False here,
    deliberately overriding `configs/explainability/default.yaml`'s own
    `true` defaults -- most stub models in this file have neither a `heads`
    attribute nor real fusion blocks, so leaving the shipped defaults in
    place would make every pre-existing test in this file raise at
    `_AmbiguityAtLevel` construction. Tests that actually exercise either
    flag pass it explicitly.
    """
    overrides = [
        f"data.root_dir={tmp_path}",
        f"data.preprocessing.out_dir={prep_dir}",
        f"data.splits.path={splits_path}",
        f"data.patch_size=[{ROI_SIZE[0]},{ROI_SIZE[1]},{ROI_SIZE[2]}]",
        "data.num_workers=0",
        "data.dataset_type=dataset",
        f"training.checkpoint.dir={checkpoint_dir}",
        f"explainability.ambiguity.out_dir={out_dir}",
        "explainability.ambiguity.split=test",
        "explainability.ambiguity.num_cases=null",
        "explainability.ambiguity.level=0",
        "explainability.ambiguity.save_maps=true",
        "explainability.ambiguity.save_image=false",
        f"explainability.ambiguity.include_auxiliary={str(include_auxiliary).lower()}",
        f"explainability.ambiguity.include_gates={str(include_gates).lower()}",
        "wandb.mode=disabled",
        "device=cpu",
        "seed=42",
    ]
    if logits_dir is not None:
        overrides.append(f"explainability.ambiguity.logits_dir={logits_dir}")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def _write_case_logits(logits_dir: Path, case_id: str, seed: int) -> None:
    """Writes one case's precomputed logits, matching evaluate.py's save_logits shape."""
    logits_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((NUM_REGIONS, *VOLUME_SHAPE)).astype(np.float16)
    np.save(logits_dir / f"{case_id}.npy", logits)


def _save_stub_checkpoint(checkpoint_dir: Path, model: nn.Module) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    fake_trained_cfg = OmegaConf.create({"model": {"name": "stub"}})
    save_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        epoch=0,
        global_step=0,
        cfg=fake_trained_cfg,
        is_best=True,
    )


def test_run_extraction_writes_summary_and_manifest_indexed_by_case_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001"]
    for i, case_id in enumerate(case_ids):
        _write_synthetic_case(prep_dir, case_id, seed=i)
    write_yaml({"train": [], "val": [], "test": case_ids}, splits_path)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubAmbiguityModel())
    monkeypatch.setattr(extract_ambiguity_script, "build_model", lambda cfg: _StubAmbiguityModel())

    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir)

    summary_df = run_extraction(cfg)

    assert len(summary_df) == len(case_ids)
    assert summary_df.index.name == "case_id"
    assert set(summary_df.index) == set(case_ids)
    assert set(summary_df.columns) == _expected_columns() | {"level", "n_windows"}

    summary_csv = out_dir / "ambiguity_summary.csv"
    manifest_csv = out_dir / "ambiguity_manifest.csv"
    assert summary_csv.is_file()
    assert manifest_csv.is_file()

    summary_on_disk = pd.read_csv(summary_csv, index_col="case_id")
    assert len(summary_on_disk) == len(case_ids)

    manifest_df = pd.read_csv(manifest_csv, index_col="case_id")
    assert len(manifest_df) == len(case_ids)
    assert manifest_df.index.name == "case_id"
    assert (manifest_df["level"] == 0).all()
    assert (manifest_df["maps_saved"]).all()
    assert (summary_on_disk["n_windows"] > 0).all()
    # No logits_dir override was passed -- the plain-model pass ran, so
    # provenance is recorded as "model", not a directory.
    assert (manifest_df["logits_source"] == "model").all()

    for case_id in case_ids:
        assert (out_dir / f"{case_id}.npz").is_file()
        with np.load(out_dir / f"{case_id}.npz") as data:
            assert set(data.keys()) == {"disagreement", "entropy_cnn", "entropy_swin", "logits"}
            assert data["disagreement"].shape == (NUM_REGIONS, *VOLUME_SHAPE)
            assert data["disagreement"].dtype == np.float16
            assert data["logits"].shape == (NUM_REGIONS, *VOLUME_SHAPE)

    assert (out_dir / "ambiguity_config.yaml").is_file()


def test_run_extraction_raises_for_model_with_no_forward_with_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_synthetic_case(prep_dir, "case_000", seed=0)
    write_yaml({"train": [], "val": [], "test": ["case_000"]}, splits_path)

    class _NoAmbiguityModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(4, NUM_REGIONS, kernel_size=3, padding=1)

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(x)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _NoAmbiguityModel())
    monkeypatch.setattr(extract_ambiguity_script, "build_model", lambda cfg: _NoAmbiguityModel())

    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir)

    with pytest.raises(TypeError, match="forward_with_ambiguity"):
        run_extraction(cfg)

    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 6. explainability.ambiguity.logits_dir -- reusing precomputed eval logits
# ---------------------------------------------------------------------------


def test_run_extraction_reuses_logits_dir_and_produces_same_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001"]
    for i, case_id in enumerate(case_ids):
        _write_synthetic_case(prep_dir, case_id, seed=i)
    write_yaml({"train": [], "val": [], "test": case_ids}, splits_path)

    logits_dir = tmp_path / "precomputed_logits"
    for i, case_id in enumerate(case_ids):
        _write_case_logits(logits_dir, case_id, seed=100 + i)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubAmbiguityModel())
    monkeypatch.setattr(extract_ambiguity_script, "build_model", lambda cfg: _StubAmbiguityModel())

    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(
        tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, logits_dir=logits_dir
    )

    summary_df = run_extraction(cfg)

    assert len(summary_df) == len(case_ids)
    assert set(summary_df.columns) == _expected_columns() | {"level", "n_windows"}

    manifest_df = pd.read_csv(out_dir / "ambiguity_manifest.csv", index_col="case_id")
    assert (manifest_df["logits_source"] == str(logits_dir.resolve())).all()

    for case_id in case_ids:
        with np.load(out_dir / f"{case_id}.npz") as data:
            assert data["logits"].shape == (NUM_REGIONS, *VOLUME_SHAPE)


def test_run_extraction_raises_when_logits_dir_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_synthetic_case(prep_dir, "case_000", seed=0)
    write_yaml({"train": [], "val": [], "test": ["case_000"]}, splits_path)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubAmbiguityModel())
    monkeypatch.setattr(extract_ambiguity_script, "build_model", lambda cfg: _StubAmbiguityModel())

    out_dir = tmp_path / "ambiguity_out"
    missing_logits_dir = tmp_path / "does_not_exist"
    cfg = _compose_cfg(
        tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, logits_dir=missing_logits_dir
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_extraction(cfg)

    assert not out_dir.exists()


def test_run_extraction_raises_when_logits_dir_missing_a_selected_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001"]
    for i, case_id in enumerate(case_ids):
        _write_synthetic_case(prep_dir, case_id, seed=i)
    write_yaml({"train": [], "val": [], "test": case_ids}, splits_path)

    logits_dir = tmp_path / "precomputed_logits"
    # Only write logits for the FIRST case -- case_001 is missing.
    _write_case_logits(logits_dir, "case_000", seed=100)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubAmbiguityModel())
    monkeypatch.setattr(extract_ambiguity_script, "build_model", lambda cfg: _StubAmbiguityModel())

    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(
        tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, logits_dir=logits_dir
    )

    with pytest.raises(FileNotFoundError, match="1 of 2"):
        run_extraction(cfg)

    assert not out_dir.exists()


def test_run_extraction_raises_on_logits_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_synthetic_case(prep_dir, "case_000", seed=0)
    write_yaml({"train": [], "val": [], "test": ["case_000"]}, splits_path)

    # Deliberately the wrong spatial shape: VOLUME_SHAPE is (16, 16, 16),
    # this logits array is (NUM_REGIONS, 4, 4, 4) -- as if produced by a
    # different preprocessing run or cohort.
    logits_dir = tmp_path / "precomputed_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    wrong_shape_logits = rng.standard_normal((NUM_REGIONS, 4, 4, 4)).astype(np.float16)
    np.save(logits_dir / "case_000.npy", wrong_shape_logits)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubAmbiguityModel())
    monkeypatch.setattr(extract_ambiguity_script, "build_model", lambda cfg: _StubAmbiguityModel())

    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(
        tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, logits_dir=logits_dir
    )

    with pytest.raises(ValueError, match="case_000"):
        run_extraction(cfg)


# ---------------------------------------------------------------------------
# 7. include_auxiliary / include_gates -- channel_groups, slicing round-trips, raises,
#    and the new summary columns.
#
# These need a REAL (tiny) NeuroVisionX, not a duck-typed stub: include_auxiliary=True /
# include_gates=True reach into model.cnn_encoder / .swin_encoder / .use_swin /
# .fusion_blocks / .decoder / .heads directly (see _AmbiguityAtLevel.forward's inlined
# pyramid walk), which none of the stubs above implement. Same tiny-model recipe as
# tests/test_neurovision.py: cnn.channels=[8, 16, 24, 32] (one more level than
# swin.num_levels=3, satisfying the stride-offset contract), feature_size=12.
# ---------------------------------------------------------------------------

AUX_CNN_CHANNELS = [8, 16, 24, 32]
AUX_CNN_BLOCKS = [1, 1, 1, 1]
AUX_SWIN_FEATURE_SIZE = 12
AUX_SWIN_NUM_LEVELS = 3
AUX_NUM_GROUPS = 8
AUX_FUSION_HEADS = 4
AUX_INPUT_SHAPE: tuple[int, int, int] = (32, 32, 32)


def _build_tiny_neurovision(
    confidence: bool = False,
    boundary: bool = False,
    fusion_name: str = "adaptive_gated",
) -> NeuroVisionX:
    """Builds a small, real NeuroVisionX -- see this section's header comment."""
    cnn = CNNEncoder(
        in_channels=4,
        channels=AUX_CNN_CHANNELS,
        blocks_per_stage=AUX_CNN_BLOCKS,
        num_groups=AUX_NUM_GROUPS,
        dropout=0.0,
        use_checkpoint=False,
        zero_init_residual=False,
    )
    swin = SwinEncoder(
        in_channels=4,
        feature_size=AUX_SWIN_FEATURE_SIZE,
        depths=(1, 1, 1, 1),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        patch_size=2,
        num_levels=AUX_SWIN_NUM_LEVELS,
        use_checkpoint=False,
        normalize=True,
    )

    blocks = []
    for i in range(swin.num_levels):
        cnn_ch = cnn.out_channels[i + 1]
        swin_ch = swin.out_channels[i]
        if fusion_name == "adaptive_gated":
            blocks.append(
                AdaptiveGatedFusion(
                    cnn_ch,
                    swin_ch,
                    num_heads=AUX_FUSION_HEADS,
                    window_size=4,
                    num_groups=AUX_NUM_GROUPS,
                    num_regions=NUM_REGIONS,
                )
            )
        elif fusion_name == "concat":
            blocks.append(ConcatFusion(cnn_ch, swin_ch, num_groups=AUX_NUM_GROUPS))
        else:
            raise ValueError(f"unknown test fusion name {fusion_name!r}")
    fusion_blocks = nn.ModuleList(blocks)

    decoder = UNetDecoder(
        skip_channels=cnn.out_channels,
        decoder_channels=None,
        blocks_per_stage=1,
        num_groups=AUX_NUM_GROUPS,
        dropout=0.0,
        upsample="deconv",
        use_attention_gates=False,
        use_checkpoint=False,
    )
    heads = MultiTaskHead(
        decoder_channels=decoder.out_channels,
        out_channels=NUM_REGIONS,
        deep_supervision_levels=1,
        confidence=confidence,
        boundary=boundary,
        confidence_num_groups=AUX_NUM_GROUPS,
        boundary_num_groups=AUX_NUM_GROUPS,
    )
    return NeuroVisionX(
        cnn_encoder=cnn,
        swin_encoder=swin,
        fusion_blocks=fusion_blocks,
        decoder=decoder,
        out_channels=NUM_REGIONS,
        deep_supervision_levels=1,
        head_dropout=0.0,
        heads=heads,
    )


def test_channel_groups_matches_emitted_channel_count_both_flags() -> None:
    model = _build_tiny_neurovision(confidence=True, boundary=True)
    model.eval()
    wrapper = _AmbiguityAtLevel(
        model, level=0, include_auxiliary=True, include_gates=True, num_regions=NUM_REGIONS
    )

    x = torch.randn(1, 4, *AUX_INPUT_SHAPE)
    with torch.no_grad():
        out = wrapper(x)

    expected_channels = sum(size for _, size in wrapper.channel_groups)
    assert out.shape == (1, expected_channels, *AUX_INPUT_SHAPE)
    assert [name for name, _ in wrapper.channel_groups] == [
        "ambiguity",
        "gate",
        "confidence",
        "boundary",
    ]
    assert dict(wrapper.channel_groups) == {
        "ambiguity": 3 * NUM_REGIONS,
        "gate": len(model.fusion_blocks),
        "confidence": NUM_REGIONS,
        "boundary": NUM_REGIONS,
    }


def test_confidence_and_boundary_slices_round_trip_against_forward_with_auxiliary() -> None:
    model = _build_tiny_neurovision(confidence=True, boundary=True)
    model.eval()
    wrapper = _AmbiguityAtLevel(model, level=0, include_auxiliary=True, num_regions=NUM_REGIONS)

    x = torch.randn(1, 4, *AUX_INPUT_SHAPE)
    with torch.no_grad():
        out = wrapper(x)
        _logits, expected_confidence, expected_boundary = model.forward_with_auxiliary(x)

    groups = extract_ambiguity_script._split_channel_groups(out[0], wrapper.channel_groups)
    assert torch.allclose(groups["confidence"], expected_confidence[0], atol=1e-5)
    assert torch.allclose(groups["boundary"], expected_boundary[0], atol=1e-5)


def test_gate_slice_round_trips_against_forward_with_gates() -> None:
    # forward_with_gates does NOT upsample its gate maps (they stay at each fusion block's
    # native stride); _AmbiguityAtLevel upsamples every level to full input resolution so
    # MONAI's sliding-window inferer can stitch a single tensor -- so the expected values are
    # upsampled here with the SAME trilinear interpolation before comparing.
    model = _build_tiny_neurovision(confidence=False, boundary=False)
    model.eval()
    wrapper = _AmbiguityAtLevel(model, level=0, include_gates=True, num_regions=NUM_REGIONS)

    x = torch.randn(1, 4, *AUX_INPUT_SHAPE)
    with torch.no_grad():
        out = wrapper(x)
        _logits, expected_gates = model.forward_with_gates(x)

    groups = extract_ambiguity_script._split_channel_groups(out[0], wrapper.channel_groups)
    gate_group = groups["gate"]
    assert gate_group.shape[0] == len(expected_gates) == len(model.fusion_blocks)
    for i, expected_gate in enumerate(expected_gates):
        assert expected_gate is not None
        if tuple(expected_gate.shape[2:]) != AUX_INPUT_SHAPE:
            expected_gate = torch.nn.functional.interpolate(
                expected_gate, size=AUX_INPUT_SHAPE, mode="trilinear", align_corners=False
            )
        assert torch.allclose(gate_group[i], expected_gate[0, 0], atol=1e-5)


def test_include_auxiliary_raises_without_confidence_head() -> None:
    model = _build_tiny_neurovision(confidence=False, boundary=True)

    with pytest.raises(ValueError, match="explainability.ambiguity.include_auxiliary"):
        _AmbiguityAtLevel(model, level=0, include_auxiliary=True, num_regions=NUM_REGIONS)


def test_include_auxiliary_raises_without_boundary_head() -> None:
    model = _build_tiny_neurovision(confidence=True, boundary=False)

    with pytest.raises(ValueError, match="no boundary head"):
        _AmbiguityAtLevel(model, level=0, include_auxiliary=True, num_regions=NUM_REGIONS)


def test_include_gates_raises_when_model_has_no_fusion_blocks() -> None:
    # A stub with no fusion_blocks attribute at all -- the same shape of model every other
    # test in this file uses.
    model = _StubAmbiguityModel()

    with pytest.raises(ValueError, match="explainability.ambiguity.include_gates"):
        _AmbiguityAtLevel(model, level=0, include_gates=True, num_regions=NUM_REGIONS)


def test_include_gates_raises_when_a_fusion_block_reports_no_gate() -> None:
    # ConcatFusion has a real (non-empty) fusion_blocks list, so construction succeeds --
    # the raise only fires once forward() actually observes each block's return_gate=True
    # call giving back None.
    model = _build_tiny_neurovision(fusion_name="concat")
    model.eval()
    wrapper = _AmbiguityAtLevel(model, level=0, include_gates=True, num_regions=NUM_REGIONS)

    x = torch.randn(1, 4, *AUX_INPUT_SHAPE)
    with pytest.raises(ValueError, match="explainability.ambiguity.include_gates"):
        with torch.no_grad():
            wrapper(x)


def test_summarize_case_ambiguity_gains_conf_and_gate_columns() -> None:
    shape = (NUM_REGIONS, 6, 6, 6)
    disagreement = torch.rand(*shape)
    entropy_cnn = torch.rand(*shape)
    entropy_swin = torch.rand(*shape)
    regions = torch.zeros(*shape)
    regions[:, 2:4, 2:4, 2:4] = 1.0  # nonempty foreground for every region, including WT
    confidence_logits = torch.randn(*shape)
    gate = torch.rand(4, 6, 6, 6)  # 4 fusion levels

    row = summarize_case_ambiguity(
        disagreement,
        entropy_cnn,
        entropy_swin,
        regions,
        confidence_logits=confidence_logits,
        gate=gate,
    )

    expected_conf_columns = {"conf_mean_fg_mean"}
    for region in ("ET", "TC", "WT"):
        expected_conf_columns.add(f"conf_mean_{region}")
        expected_conf_columns.add(f"conf_mean_fg_{region}")
    expected_gate_columns = set()
    for level_idx in range(4):
        expected_gate_columns.add(f"gate_mean_{level_idx}")
        expected_gate_columns.add(f"gate_mean_fg_{level_idx}")

    assert expected_conf_columns <= set(row.keys())
    assert expected_gate_columns <= set(row.keys())
    assert all(np.isfinite(row[c]) for c in expected_conf_columns | expected_gate_columns)
    # No cross-level aggregate for the gate columns -- see summarize_case_ambiguity's
    # docstring for why (the levels have opposite polarity).
    assert not any(c.startswith("gate_mean_fg_mean") for c in row)


def test_summarize_case_ambiguity_gate_fg_columns_nan_when_wt_foreground_empty() -> None:
    shape = (NUM_REGIONS, 6, 6, 6)
    disagreement = torch.rand(*shape)
    entropy_cnn = torch.rand(*shape)
    entropy_swin = torch.rand(*shape)
    regions = torch.zeros(*shape)  # every region, including WT, empty
    gate = torch.rand(2, 6, 6, 6)

    row = summarize_case_ambiguity(disagreement, entropy_cnn, entropy_swin, regions, gate=gate)

    for level_idx in range(2):
        assert np.isnan(row[f"gate_mean_fg_{level_idx}"])
        # The whole-volume mean is unaffected by an empty foreground mask.
        assert np.isfinite(row[f"gate_mean_{level_idx}"])


def test_summarize_case_ambiguity_no_confidence_no_gate_matches_original_columns() -> None:
    # include_auxiliary=False / include_gates=False byte-for-byte backward compatibility:
    # calling with neither confidence_logits nor gate must return EXACTLY the original
    # column set, not a superset with NaN-filled new columns.
    shape = (NUM_REGIONS, 6, 6, 6)
    disagreement = torch.rand(*shape)
    entropy_cnn = torch.rand(*shape)
    entropy_swin = torch.rand(*shape)
    regions = torch.ones(*shape)

    row = summarize_case_ambiguity(disagreement, entropy_cnn, entropy_swin, regions)

    assert set(row.keys()) == _expected_columns()


def _build_tiny_checkpoint_and_cfg(
    tmp_path: Path, confidence: bool, boundary: bool, fusion_name: str = "adaptive_gated"
):
    """Shared setup for the include_auxiliary=True / include_gates=True end-to-end test:
    writes one synthetic case at AUX_INPUT_SHAPE (a real NeuroVisionX needs enough spatial
    extent for its Swin branch and window attention -- 16^3, used elsewhere in this file, is
    too small), a matching split file, and a checkpoint for a tiny real NeuroVisionX."""
    prep_dir = tmp_path / "prep_aux"
    splits_path = tmp_path / "splits_aux.yaml"
    case_id = "case_aux_000"

    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    image = rng.standard_normal((4, *AUX_INPUT_SHAPE)).astype(np.float16)
    np.save(case_dir / "image.npy", image)

    d, h, w = AUX_INPUT_SHAPE
    label = np.zeros(AUX_INPUT_SHAPE, dtype=np.uint8)
    label[d // 2 - 4 : d // 2 + 4, h // 2 - 4 : h // 2 + 4, w // 2 - 4 : w // 2 + 4] = 2
    label[d // 2 - 3 : d // 2 + 3, h // 2 - 3 : h // 2 + 3, w // 2 - 3 : w // 2 + 3] = 1
    label[d // 2 - 2 : d // 2 + 2, h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = 3
    np.save(case_dir / "label.npy", label)

    write_json(
        {
            "case_id": case_id,
            "has_label": True,
            "bbox": [[0, d], [0, h], [0, w]],
            "original_shape": list(AUX_INPUT_SHAPE),
            "spacing": [1.0, 1.0, 1.0],
        },
        case_dir / "meta.json",
    )
    write_yaml({"train": [], "val": [], "test": [case_id]}, splits_path)

    model = _build_tiny_neurovision(
        confidence=confidence, boundary=boundary, fusion_name=fusion_name
    )
    checkpoint_dir = tmp_path / "checkpoints_aux"
    _save_stub_checkpoint(checkpoint_dir, model)

    return prep_dir, splits_path, checkpoint_dir, case_id, model


def test_run_extraction_end_to_end_with_auxiliary_and_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir, splits_path, checkpoint_dir, case_id, model = _build_tiny_checkpoint_and_cfg(
        tmp_path, confidence=True, boundary=True
    )
    monkeypatch.setattr(
        extract_ambiguity_script,
        "build_model",
        lambda cfg: _build_tiny_neurovision(confidence=True, boundary=True),
    )

    out_dir = tmp_path / "ambiguity_out_aux"
    overrides_extra_patch_size = f"[{AUX_INPUT_SHAPE[0]},{AUX_INPUT_SHAPE[1]},{AUX_INPUT_SHAPE[2]}]"
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"data.root_dir={tmp_path}",
                f"data.preprocessing.out_dir={prep_dir}",
                f"data.splits.path={splits_path}",
                f"data.patch_size={overrides_extra_patch_size}",
                "data.num_workers=0",
                "data.dataset_type=dataset",
                f"training.checkpoint.dir={checkpoint_dir}",
                f"explainability.ambiguity.out_dir={out_dir}",
                "explainability.ambiguity.split=test",
                "explainability.ambiguity.num_cases=null",
                "explainability.ambiguity.level=0",
                "explainability.ambiguity.save_maps=true",
                "explainability.ambiguity.save_image=false",
                "explainability.ambiguity.include_auxiliary=true",
                "explainability.ambiguity.include_gates=true",
                "wandb.mode=disabled",
                "device=cpu",
                "seed=42",
            ],
        )

    summary_df = run_extraction(cfg)

    assert len(summary_df) == 1
    for region in ("ET", "TC", "WT"):
        assert f"conf_mean_{region}" in summary_df.columns
    for level_idx in range(len(model.fusion_blocks)):
        assert f"gate_mean_{level_idx}" in summary_df.columns

    manifest_df = pd.read_csv(out_dir / "ambiguity_manifest.csv", index_col="case_id")
    assert bool(manifest_df.loc[case_id, "include_auxiliary"])
    assert bool(manifest_df.loc[case_id, "include_gates"])

    with np.load(out_dir / f"{case_id}.npz") as data:
        assert {
            "disagreement",
            "entropy_cnn",
            "entropy_swin",
            "logits",
            "gate",
            "confidence",
            "boundary",
        } <= set(data.keys())
        assert data["gate"].shape == (len(model.fusion_blocks), *AUX_INPUT_SHAPE)
        assert data["confidence"].shape == (NUM_REGIONS, *AUX_INPUT_SHAPE)
        assert data["boundary"].shape == (NUM_REGIONS, *AUX_INPUT_SHAPE)


def test_run_extraction_include_auxiliary_false_include_gates_false_matches_original_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Byte-for-byte backward compatibility at the run_extraction level, using the ORIGINAL
    # small stub models this file has used throughout -- confirms the new flags default off
    # produce exactly the pre-existing artifact shape.
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_synthetic_case(prep_dir, "case_000", seed=0)
    write_yaml({"train": [], "val": [], "test": ["case_000"]}, splits_path)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubAmbiguityModel())
    monkeypatch.setattr(extract_ambiguity_script, "build_model", lambda cfg: _StubAmbiguityModel())

    out_dir = tmp_path / "ambiguity_out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir,
        include_auxiliary=False,
        include_gates=False,
    )

    summary_df = run_extraction(cfg)

    assert set(summary_df.columns) == _expected_columns() | {"level", "n_windows"}
    with np.load(out_dir / "case_000.npz") as data:
        assert set(data.keys()) == {"disagreement", "entropy_cnn", "entropy_swin", "logits"}
