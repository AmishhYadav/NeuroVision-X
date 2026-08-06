"""Tests for scripts/explain.py.

Follows the exact pattern of tests/test_extract_gates.py: the script lives
under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import, and `build_model` is monkeypatched to hand back a tiny stub model
(never a full `NeuroVisionX`, which is far too slow for a sub-second test
suite) with a nameable middle module usable as a Grad-CAM `target_layer`.

No case here uses real BraTS data: synthetic `.npy` + `meta.json` trees are
written under `tmp_path`, mirroring `scripts/preprocess.py`'s output shape.
Integrated Gradients/Grad-CAM/faithfulness configs use tiny values
(`n_steps`, `n_points`) so the whole file stays fast on a stub 2-conv model.
"""

from __future__ import annotations

import importlib.util
import sys
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

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "explain.py"
_spec = importlib.util.spec_from_file_location("explain_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
explain_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["explain_script"] = explain_script
_spec.loader.exec_module(explain_script)

select_cases = explain_script.select_cases
run_explanation = explain_script.run_explanation

CROPPED_SHAPE: tuple[int, int, int] = (24, 24, 24)
PATCH_SIZE: tuple[int, int, int] = (16, 16, 16)
OUT_CHANNELS = 3


# ---------------------------------------------------------------------------
# Stub model -- 4 in / 3 out, with a nameable middle Conv3d for Grad-CAM.
# ---------------------------------------------------------------------------


class _StubExplainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
        self.mid = nn.Conv3d(8, 8, kernel_size=3, padding=1)  # Grad-CAM target_layer
        self.conv2 = nn.Conv3d(8, OUT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.mid(x))
        return self.conv2(x)


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

    # A real label.npy is always written, even for has_label=False cases --
    # build_val_transforms' LoadImaged has no allow_missing_keys=True, so a
    # case with no label.npy at all cannot pass through the shared pipeline
    # (that specific case is exercised separately below by unlinking it).
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
    **attribution_overrides: object,
) -> OmegaConf:
    """Builds a small explainability config mirroring config.yaml + explainability/default.yaml."""
    if out_dir is None:
        out_dir = tmp_path / "attribution_out"

    attribution = {
        "split": "test",
        "checkpoint": None,
        "out_dir": str(out_dir),
        "num_cases": None,
        "case_ids": None,
        "patch_size": list(PATCH_SIZE),
        "regions": [0, 2],
        "save_image": True,
        "integrated_gradients": {
            "enabled": True,
            "n_steps": 2,
            "internal_batch_size": 1,
            "delta_tolerance": 0.05,
            "noise_scale": 1.0,
        },
        "grad_cam": {
            "enabled": True,
            "target_layer": "mid",
            "relu": True,
        },
        "attention": {
            "enabled": False,
            "stage": "layers3",
            "residual_weight": 0.5,
            "max_tokens": 4096,
        },
        "faithfulness": {
            "enabled": True,
            "n_points": 3,
            "fill": "zero",
        },
        "seed": 0,
    }
    for key, value in attribution_overrides.items():
        if isinstance(value, dict) and isinstance(attribution.get(key), dict):
            attribution[key] = {**attribution[key], **value}
        else:
            attribution[key] = value

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
        "model": {"name": "stub", "out_channels": OUT_CHANNELS},
        "training": {"checkpoint": {"dir": str(checkpoint_dir)}},
        "explainability": {"attribution": attribution},
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


def _setup_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_ids: list[str],
    has_label: bool = True,
    write_label_file: bool = True,
    **attribution_overrides: object,
) -> OmegaConf:
    """Common fixture: writes cases + splits + checkpoint, monkeypatches build_model."""
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    for i, case_id in enumerate(case_ids):
        _write_case(prep_dir, case_id, seed=i, has_label=has_label)
        if not write_label_file:
            (prep_dir / case_id / "label.npy").unlink()
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _StubExplainModel())
    monkeypatch.setattr(explain_script, "build_model", lambda cfg: _StubExplainModel())

    return _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, **attribution_overrides)


# ---------------------------------------------------------------------------
# 1. End-to-end
# ---------------------------------------------------------------------------


def test_run_explanation_writes_npz_and_all_csv_yaml_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000", "case_001"]
    cfg = _setup_case(tmp_path, monkeypatch, case_ids)

    run_explanation(cfg)

    out_dir = Path(cfg.explainability.attribution.out_dir)
    for case_id in case_ids:
        npz_path = out_dir / f"{case_id}.npz"
        assert npz_path.is_file()
        with np.load(npz_path) as data:
            keys = set(data.keys())
            expected = {
                "label",
                "image",
                "ig_region_0",
                "ig_abs_region_0",
                "cam_region_0",
                "ig_region_2",
                "ig_abs_region_2",
                "cam_region_2",
            }
            assert keys == expected

            assert data["ig_region_0"].dtype == np.float16
            assert data["ig_region_0"].shape == (4, *PATCH_SIZE)  # (C, D, H, W)
            assert data["ig_abs_region_0"].dtype == np.float16
            assert data["ig_abs_region_0"].shape == PATCH_SIZE  # (D, H, W)
            assert data["cam_region_0"].dtype == np.float16
            assert data["cam_region_0"].shape == PATCH_SIZE

            assert data["label"].dtype == np.uint8
            assert data["label"].shape == (3, *PATCH_SIZE)
            assert data["image"].dtype == np.float16
            assert data["image"].shape == (4, *PATCH_SIZE)

    assert (out_dir / "attribution_manifest.csv").is_file()
    assert (out_dir / "modality_attribution.csv").is_file()
    assert (out_dir / "faithfulness.csv").is_file()
    assert (out_dir / "explain_config.yaml").is_file()


# ---------------------------------------------------------------------------
# 2. modality_attribution.csv shape and normalization
# ---------------------------------------------------------------------------


def test_modality_attribution_csv_has_one_row_per_case_region_and_sums_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000", "case_001"]
    cfg = _setup_case(tmp_path, monkeypatch, case_ids)

    run_explanation(cfg)

    out_dir = Path(cfg.explainability.attribution.out_dir)
    df = pd.read_csv(out_dir / "modality_attribution.csv")

    assert len(df) == len(case_ids) * len(cfg.explainability.attribution.regions)
    assert set(zip(df["case_id"], df["region"])) == {(c, r) for c in case_ids for r in [0, 2]}

    modality_cols = ["attr_T1", "attr_T1CE", "attr_T2", "attr_FLAIR"]
    assert set(modality_cols).issubset(df.columns)
    sums = df[modality_cols].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. grad_cam.target_layer null raises before output dir exists
# ---------------------------------------------------------------------------


def test_grad_cam_target_layer_null_raises_before_output_dir_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out_dir = tmp_path / "attribution_out"
    cfg = _setup_case(
        tmp_path,
        monkeypatch,
        ["case_000"],
        out_dir=out_dir,
        grad_cam={"target_layer": None},
    )

    with pytest.raises(ValueError, match="Available layers"):
        run_explanation(cfg)

    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 4. Unknown target_layer raises, naming it
# ---------------------------------------------------------------------------


def test_unknown_grad_cam_target_layer_raises_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out_dir = tmp_path / "attribution_out"
    cfg = _setup_case(
        tmp_path,
        monkeypatch,
        ["case_000"],
        out_dir=out_dir,
        grad_cam={"target_layer": "does_not_exist"},
    )

    with pytest.raises(ValueError, match="does_not_exist"):
        run_explanation(cfg)

    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 5. Region index out of range raises, naming it
# ---------------------------------------------------------------------------


def test_region_index_out_of_range_raises_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out_dir = tmp_path / "attribution_out"
    cfg = _setup_case(tmp_path, monkeypatch, ["case_000"], out_dir=out_dir, regions=[0, 5])

    with pytest.raises(ValueError, match="5"):
        run_explanation(cfg)

    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 6. No label.npy / has_label False raise, naming the case
# ---------------------------------------------------------------------------


def test_case_with_no_label_file_raises_naming_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_case(
        tmp_path, monkeypatch, ["no_label_file"], has_label=False, write_label_file=False
    )

    with pytest.raises(ValueError, match="no_label_file"):
        run_explanation(cfg)


def test_case_with_has_label_false_raises_naming_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_case(tmp_path, monkeypatch, ["unlabeled_case"], has_label=False)

    with pytest.raises(ValueError, match="unlabeled_case"):
        run_explanation(cfg)


# ---------------------------------------------------------------------------
# 7. attention enabled on a model with no attention modules
# ---------------------------------------------------------------------------


def test_attention_enabled_with_no_attention_modules_raises_naming_no_swin_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out_dir = tmp_path / "attribution_out"
    cfg = _setup_case(
        tmp_path,
        monkeypatch,
        ["case_000"],
        out_dir=out_dir,
        attention={"enabled": True},
    )

    with pytest.raises(ValueError, match="Swin"):
        run_explanation(cfg)

    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 8. Model stays in eval mode throughout
# ---------------------------------------------------------------------------


def test_model_stays_in_eval_mode_throughout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case_ids = ["case_000"]
    cfg = _setup_case(tmp_path, monkeypatch, case_ids)

    built_models: list[nn.Module] = []

    def _build_and_track(cfg: object) -> nn.Module:
        model = _StubExplainModel()
        built_models.append(model)
        return model

    monkeypatch.setattr(explain_script, "build_model", _build_and_track)

    run_explanation(cfg)

    assert len(built_models) == 1
    assert built_models[0].training is False


# ---------------------------------------------------------------------------
# 9. faithfulness.csv contains a "random" row
# ---------------------------------------------------------------------------


def test_faithfulness_csv_contains_random_null_baseline_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000"]
    cfg = _setup_case(tmp_path, monkeypatch, case_ids)

    run_explanation(cfg)

    out_dir = Path(cfg.explainability.attribution.out_dir)
    df = pd.read_csv(out_dir / "faithfulness.csv")

    n_expected = len(case_ids) * len(cfg.explainability.attribution.regions) * 3  # ig+cam+random
    assert len(df) == n_expected
    random_rows = df[df["method"] == "random"]
    assert len(random_rows) == len(case_ids) * len(cfg.explainability.attribution.regions)
    assert bool((random_rows["target_specific"] == False).all())  # noqa: E712


# ---------------------------------------------------------------------------
# 10. Disabling a sub-analysis omits its arrays and rows without erroring
# ---------------------------------------------------------------------------


def test_disabling_integrated_gradients_omits_its_arrays_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000"]
    cfg = _setup_case(tmp_path, monkeypatch, case_ids, integrated_gradients={"enabled": False})

    run_explanation(cfg)

    out_dir = Path(cfg.explainability.attribution.out_dir)
    with np.load(out_dir / "case_000.npz") as data:
        keys = set(data.keys())
        assert not any(k.startswith("ig_") for k in keys)
        assert any(k.startswith("cam_") for k in keys)

    modality_df = pd.read_csv(out_dir / "modality_attribution.csv")
    assert modality_df.empty

    # grad_cam still ran, so faithfulness still has rows (grad_cam + random).
    faithfulness_df = pd.read_csv(out_dir / "faithfulness.csv")
    assert not faithfulness_df.empty
    assert "integrated_gradients" not in set(faithfulness_df["method"])
    assert "grad_cam" in set(faithfulness_df["method"])

    manifest_df = pd.read_csv(out_dir / "attribution_manifest.csv", index_col="case_id")
    assert bool(manifest_df.loc["case_000", "integrated_gradients_ran"]) is False
    assert bool(manifest_df.loc["case_000", "grad_cam_ran"]) is True
