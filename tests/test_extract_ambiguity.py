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
):
    """Composes the REAL Hydra config, tiny-sized -- mirrors scripts/smoke_test.py."""
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
