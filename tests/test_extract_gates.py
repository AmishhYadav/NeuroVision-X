"""Tests for scripts/extract_gates.py.

Follows the exact pattern of tests/test_evaluate_script.py: the script lives
under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import, and `build_model` is monkeypatched to hand back a small stub model
(never a full `NeuroVisionX`, which is far too slow for a sub-second test
suite) whose `forward_with_gates` behaviour each test controls directly.

No case here uses real BraTS data: synthetic `.npy` + `meta.json` trees are
written under `tmp_path`, mirroring `scripts/preprocess.py`'s output shape.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from neurovision.models import baseline  # noqa: F401 -- registers "unet3d"
from neurovision.training.checkpoint import save_checkpoint
from neurovision.utils.io import write_json, write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_gates.py"
_spec = importlib.util.spec_from_file_location("extract_gates_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
extract_gates_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["extract_gates_script"] = extract_gates_script
_spec.loader.exec_module(extract_gates_script)

select_cases = extract_gates_script.select_cases
tumor_centroid = extract_gates_script.tumor_centroid
crop_patch = extract_gates_script.crop_patch
extract_case_gates = extract_gates_script.extract_case_gates
run_extraction = extract_gates_script.run_extraction

CROPPED_SHAPE: tuple[int, int, int] = (24, 24, 24)
PATCH_SIZE: tuple[int, int, int] = (16, 16, 16)


# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------


class _StubGateModel(nn.Module):
    """A tiny model with `forward_with_gates`, at two fusion levels (strides 2, 4)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 3, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def forward_with_gates(self, x: Tensor) -> tuple[Tensor, list[Tensor | None]]:
        logits = self.conv(x)
        b, _, d, h, w = x.shape
        gates = [
            torch.rand(b, 1, d // 2, h // 2, w // 2),
            torch.rand(b, 1, d // 4, h // 4, w // 4),
        ]
        return logits, gates


class _FixedGatesModel(nn.Module):
    """Stub whose `forward_with_gates` returns a caller-supplied gates list."""

    def __init__(self, gates_factory: Callable[[Tensor], list[Tensor | None]]) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 3, kernel_size=3, padding=1)
        self._gates_factory = gates_factory

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def forward_with_gates(self, x: Tensor) -> tuple[Tensor, list[Tensor | None]]:
        return self.conv(x), self._gates_factory(x)


class _NoGatesModel(nn.Module):
    """A model with no `forward_with_gates` -- e.g. a plain unet3d checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 3, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _build_synthetic_label(shape: tuple[int, int, int]) -> np.ndarray:
    """Nested ET-subset-of-TC-subset-of-WT spheres, same recipe as smoke_test.py."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[dist < min_edge * 0.4] = 2  # ED shell -> completes WT
    label[dist < min_edge * 0.25] = 1  # NCR/NET core -> completes TC
    label[dist < min_edge * 0.12] = 3  # ET, innermost
    return label


def _write_case(prep_dir: Path, case_id: str, seed: int, has_label: bool = True) -> None:
    """Writes one synthetic preprocessed case: image.npy, label.npy, meta.json."""
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    image = rng.standard_normal((4, *CROPPED_SHAPE)).astype(np.float16)
    np.save(case_dir / "image.npy", image)

    # A real label.npy is always written, even for has_label=False cases,
    # because build_val_transforms' LoadImaged has no allow_missing_keys=True
    # and a case with no label.npy cannot pass through the shared pipeline at
    # all. NOTE this is NOT what scripts/preprocess.py produces for a genuinely
    # unlabeled case -- it writes no label.npy there -- so a has_label=False
    # fixture here exercises the meta-flag branch only, never the real
    # unlabeled-data path (which _validate_center_on now rejects up front).
    # Same convention as tests/test_evaluate_script.py's fixture.
    label = _build_synthetic_label(CROPPED_SHAPE)
    np.save(case_dir / "label.npy", label)

    write_json({"case_id": case_id, "has_label": has_label}, case_dir / "meta.json")


def _write_splits(path: Path, train: list[str], val: list[str], test: list[str]) -> None:
    write_yaml({"train": train, "val": val, "test": test}, path)


def _make_cfg(
    tmp_path: Path,
    prep_dir: Path,
    splits_path: Path,
    checkpoint_dir: Path,
    out_dir: Path | None = None,
    **gates_overrides: object,
) -> OmegaConf:
    """Builds a small explainability config mirroring config.yaml + explainability/default.yaml."""
    if out_dir is None:
        out_dir = tmp_path / "gates_out"

    gates = {
        "split": "test",
        "checkpoint": None,
        "out_dir": str(out_dir),
        "num_cases": None,
        "case_ids": None,
        "patch_size": list(PATCH_SIZE),
        "center_on": "label",
        "save_image": True,
    }
    gates.update(gates_overrides)

    base = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "splits": {"path": str(splits_path)},
            "preprocessing": {"out_dir": str(prep_dir)},
            "num_workers": 0,
            "patch_size": list(PATCH_SIZE),
        },
        "model": {"name": "stub"},
        "training": {"checkpoint": {"dir": str(checkpoint_dir)}},
        "explainability": {"gates": gates},
    }
    return OmegaConf.create(base)


def _save_stub_checkpoint(checkpoint_dir: Path, model: nn.Module) -> None:
    """Checkpoints a model instance, without training."""
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


# ---------------------------------------------------------------------------
# 1-2. crop_patch
# ---------------------------------------------------------------------------


def test_crop_patch_centred_crop_has_correct_values():
    volume = torch.arange(8 * 8 * 8).reshape(1, 8, 8, 8).float()
    patch, origin = crop_patch(volume, center=(4, 4, 4), patch_size=(4, 4, 4))

    assert origin == (2, 2, 2)
    assert torch.equal(patch, volume[:, 2:6, 2:6, 2:6])


def test_crop_patch_edge_clamped_shifts_window_instead_of_shrinking():
    volume = torch.arange(8 * 8 * 8).reshape(1, 8, 8, 8).float()

    # Centre at the volume's lower corner: the naive window would run
    # negative, so it must be shifted (not shrunk) to start at 0.
    patch_low, origin_low = crop_patch(volume, center=(0, 0, 0), patch_size=(4, 4, 4))
    assert origin_low == (0, 0, 0)
    assert patch_low.shape == (1, 4, 4, 4)
    assert torch.equal(patch_low, volume[:, 0:4, 0:4, 0:4])

    # Centre at the volume's upper corner: the window must shift down to
    # end exactly at the volume boundary, not run off the edge.
    patch_high, origin_high = crop_patch(volume, center=(7, 7, 7), patch_size=(4, 4, 4))
    assert origin_high == (4, 4, 4)
    assert patch_high.shape == (1, 4, 4, 4)
    assert torch.equal(patch_high, volume[:, 4:8, 4:8, 4:8])


def test_crop_patch_pads_when_volume_axis_shorter_than_patch_size():
    # Depth axis (3) is shorter than the requested patch depth (4).
    volume = torch.arange(3 * 8 * 8).reshape(1, 3, 8, 8).float()
    patch, origin = crop_patch(volume, center=(1, 4, 4), patch_size=(4, 4, 4))

    assert origin == (0, 2, 2)
    assert patch.shape == (1, 4, 4, 4)
    # The real (unpadded) depth slices come first, values preserved exactly.
    assert torch.equal(patch[:, :3], volume[:, 0:3, 2:6, 2:6])
    # The padded slice is exactly zero.
    assert torch.equal(patch[:, 3], torch.zeros(1, 4, 4))


def test_crop_patch_origin_indexes_back_into_the_source_volume():
    volume = torch.arange(10 * 10 * 10).reshape(1, 10, 10, 10).float()
    patch, (d0, h0, w0) = crop_patch(volume, center=(6, 2, 8), patch_size=(4, 4, 4))

    pd, ph, pw = 4, 4, 4
    assert torch.equal(patch, volume[:, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw])


# ---------------------------------------------------------------------------
# 3-4. tumor_centroid
# ---------------------------------------------------------------------------


def test_tumor_centroid_hand_placed_cube():
    label = torch.zeros(3, 10, 10, 10)
    label[2, 2:5, 3:6, 4:7] = 1.0  # WT channel, a cube spanning [2,4]x[3,5]x[4,6]

    centroid = tumor_centroid(label, region_index=2, case_id="case_A")

    assert centroid == (3, 4, 5)


def test_tumor_centroid_empty_channel_falls_back_to_geometric_centre(caplog):
    label = torch.zeros(3, 8, 6, 4)  # WT channel entirely empty

    with caplog.at_level(logging.WARNING):
        centroid = tumor_centroid(label, region_index=2, case_id="case_empty")

    assert centroid == (4, 3, 2)
    assert any("case_empty" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


# ---------------------------------------------------------------------------
# 5. End-to-end .npz contents
# ---------------------------------------------------------------------------


def test_run_extraction_writes_one_npz_per_case_with_expected_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001", "case_002"]
    for i, case_id in enumerate(case_ids):
        _write_case(prep_dir, case_id, seed=i, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubGateModel())
    monkeypatch.setattr(extract_gates_script, "build_model", lambda cfg: _StubGateModel())

    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    manifest_df = run_extraction(cfg)

    out_dir = Path(cfg.explainability.gates.out_dir)
    assert len(manifest_df) == len(case_ids)

    for case_id in case_ids:
        npz_path = out_dir / f"{case_id}.npz"
        assert npz_path.is_file()
        with np.load(npz_path) as data:
            keys = set(data.keys())
            assert keys == {"gate_level_0", "gate_level_1", "label", "image", "logits"}

            assert data["gate_level_0"].dtype == np.float16
            assert data["gate_level_0"].shape == (1, 8, 8, 8)
            assert data["gate_level_1"].dtype == np.float16
            assert data["gate_level_1"].shape == (1, 4, 4, 4)

            assert data["label"].dtype == np.uint8
            assert data["label"].shape == (3, *PATCH_SIZE)

            assert data["image"].dtype == np.float16
            assert data["image"].shape == (4, *PATCH_SIZE)

            assert data["logits"].dtype == np.float16
            assert data["logits"].shape == (3, *PATCH_SIZE)


# ---------------------------------------------------------------------------
# 6-7. None / empty gates lists
# ---------------------------------------------------------------------------


def test_run_extraction_skips_none_gate_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000"]
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    def gates_factory(x: Tensor) -> list[Tensor | None]:
        b, _, d, h, w = x.shape
        real_gate = torch.rand(b, 1, d // 2, h // 2, w // 2)
        return [real_gate, None]

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _FixedGatesModel(gates_factory))
    monkeypatch.setattr(
        extract_gates_script, "build_model", lambda cfg: _FixedGatesModel(gates_factory)
    )

    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    manifest_df = run_extraction(cfg)

    out_dir = Path(cfg.explainability.gates.out_dir)
    with np.load(out_dir / "case_000.npz") as data:
        keys = set(data.keys())
        assert "gate_level_0" in keys
        assert "gate_level_1" not in keys

    row = manifest_df.loc["case_000"]
    assert row["n_levels"] == 2
    assert row["n_gate_levels"] == 1
    assert row["gate_levels"] == "0"


def test_run_extraction_handles_empty_gates_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000"]
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _FixedGatesModel(lambda x: []))
    monkeypatch.setattr(
        extract_gates_script, "build_model", lambda cfg: _FixedGatesModel(lambda x: [])
    )

    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    manifest_df = run_extraction(cfg)

    out_dir = Path(cfg.explainability.gates.out_dir)
    with np.load(out_dir / "case_000.npz") as data:
        keys = set(data.keys())
        assert "logits" in keys
        assert not any(k.startswith("gate_level_") for k in keys)

    row = manifest_df.loc["case_000"]
    assert row["n_levels"] == 0
    assert row["n_gate_levels"] == 0
    assert row["gate_levels"] == ""


# ---------------------------------------------------------------------------
# 8. No forward_with_gates
# ---------------------------------------------------------------------------


def test_run_extraction_raises_before_output_dir_when_model_has_no_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000"]
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _NoGatesModel())
    monkeypatch.setattr(extract_gates_script, "build_model", lambda cfg: _NoGatesModel())

    out_dir = tmp_path / "gates_out"
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir=out_dir)

    with pytest.raises(TypeError, match="forward_with_gates"):
        run_extraction(cfg)

    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 9. case_ids not in split
# ---------------------------------------------------------------------------


def test_select_cases_raises_naming_ids_not_in_split(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=["case_000"])

    cfg = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        tmp_path / "checkpoints",
        case_ids=["case_000", "case_999"],
    )

    with pytest.raises(ValueError, match="case_999"):
        select_cases(cfg)


# ---------------------------------------------------------------------------
# 10. center_on="label" on an unlabeled case
# ---------------------------------------------------------------------------


def test_run_extraction_raises_when_centering_on_label_for_unlabeled_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_case(prep_dir, "unlabeled_case", seed=0, has_label=False)
    _write_splits(splits_path, train=[], val=[], test=["unlabeled_case"])

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubGateModel())
    monkeypatch.setattr(extract_gates_script, "build_model", lambda cfg: _StubGateModel())

    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, center_on="label")

    with pytest.raises(ValueError, match="unlabeled_case"):
        run_extraction(cfg)


# ---------------------------------------------------------------------------
# 11. gates_manifest.csv
# ---------------------------------------------------------------------------


def test_gates_manifest_has_one_row_per_case_with_all_documented_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001"]
    for i, case_id in enumerate(case_ids):
        _write_case(prep_dir, case_id, seed=i, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubGateModel())
    monkeypatch.setattr(extract_gates_script, "build_model", lambda cfg: _StubGateModel())

    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    run_extraction(cfg)

    out_dir = Path(cfg.explainability.gates.out_dir)
    manifest_path = out_dir / "gates_manifest.csv"
    assert manifest_path.is_file()

    df = pd.read_csv(manifest_path, index_col="case_id")
    assert len(df) == len(case_ids)
    assert set(df.index) == set(case_ids)

    expected_columns = {
        "center_d",
        "center_h",
        "center_w",
        "origin_d",
        "origin_h",
        "origin_w",
        "patch_d",
        "patch_h",
        "patch_w",
        "n_levels",
        "n_gate_levels",
        "gate_levels",
        "has_label",
        "center_on",
        "wt_empty",
    }
    assert expected_columns.issubset(set(df.columns))
    # `wt_empty` means two different things depending on this, so a reader of
    # the manifest alone must be able to tell which one it got.
    assert set(df["center_on"]) == {"label"}


def test_run_extraction_raises_naming_cases_with_no_label_file_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A case with no label.npy cannot pass build_val_transforms' LoadImaged.

    scripts/preprocess.py writes no label.npy for a genuinely unlabeled case,
    and the shared val pipeline is not built with allow_missing_keys=True, so
    such a case dies several frames deep inside MONAI without naming itself.
    Both center_on modes must reject it up front instead.
    """
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_case(prep_dir, "no_label_file", seed=0, has_label=False)
    (prep_dir / "no_label_file" / "label.npy").unlink()
    _write_splits(splits_path, train=[], val=[], test=["no_label_file"])

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubGateModel())
    monkeypatch.setattr(extract_gates_script, "build_model", lambda cfg: _StubGateModel())

    # center_on="prediction" is the mode that used to claim it supported this.
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, center_on="prediction")
    with pytest.raises(ValueError, match="no_label_file"):
        run_extraction(cfg)


# ---------------------------------------------------------------------------
# Bonus: center_on="prediction" runs without a ground-truth label.
# ---------------------------------------------------------------------------


def test_run_extraction_center_on_prediction_works_without_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_case(prep_dir, "unlabeled_case", seed=0, has_label=False)
    _write_splits(splits_path, train=[], val=[], test=["unlabeled_case"])

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubGateModel())
    monkeypatch.setattr(extract_gates_script, "build_model", lambda cfg: _StubGateModel())

    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, center_on="prediction")
    manifest_df = run_extraction(cfg)

    out_dir = Path(cfg.explainability.gates.out_dir)
    with np.load(out_dir / "unlabeled_case.npz") as data:
        assert "label" not in data
        assert "image" in data
        assert "logits" in data

    assert manifest_df.loc["unlabeled_case", "has_label"] == False  # noqa: E712
