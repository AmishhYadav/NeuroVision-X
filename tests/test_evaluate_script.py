"""Tests for scripts/evaluate.py's pure-ish, testable helper functions.

`main()` is a Hydra entry point and is awkward (and unnecessary) to unit test
directly, so these tests import the plain helpers underneath it --
`build_eval_dataloader`, `resolve_checkpoint`, `load_eval_model`,
`evaluate_case`, `run_evaluation` -- and exercise them against tiny synthetic
`.npy` + `meta.json` trees under `tmp_path`, never real BraTS data.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import,
following the exact same pattern as tests/test_train_script.py and
scripts/smoke_test.py.

No case here trains anything: checkpoints are produced by calling
`neurovision.training.checkpoint.save_checkpoint` directly on a freshly built
model.
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
import yaml
from omegaconf import OmegaConf
from torch import nn

from neurovision.metrics.lesionwise import LESIONWISE_METRIC_PREFIXES
from neurovision.models import baseline  # noqa: F401 -- registers "unet3d"
from neurovision.models.registry import build_model
from neurovision.training.checkpoint import save_checkpoint
from neurovision.utils.io import write_json, write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("evaluate_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
evaluate_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_script"] = evaluate_script
_spec.loader.exec_module(evaluate_script)

build_eval_dataloader = evaluate_script.build_eval_dataloader
resolve_checkpoint = evaluate_script.resolve_checkpoint
load_eval_model = evaluate_script.load_eval_model
run_evaluation = evaluate_script.run_evaluation
evaluate_case = evaluate_script.evaluate_case
resolve_lesionwise = evaluate_script.resolve_lesionwise

# panoptica lives only in .venv-analysis (see requirements-analysis.txt), so
# any test that actually RUNS lesion-wise scoring must skip cleanly in the
# training .venv rather than fail. Tests that only exercise resolve_lesionwise
# (no panoptica import) or the fail-fast ImportError path are NOT gated by
# this -- they must run (and pass) in both venvs.
_PANOPTICA_MISSING = importlib.util.find_spec("panoptica") is None

# 32^3 cropped inside a 40^3 original volume -- small enough to run in
# milliseconds on CPU, with room for a [4, 36) bbox on every axis.
CROPPED_SHAPE: tuple[int, int, int] = (32, 32, 32)
ORIGINAL_SHAPE: tuple[int, int, int] = (40, 40, 40)
BBOX: list[list[int]] = [[4, 36], [4, 36], [4, 36]]
SPACING: list[float] = [1.0, 1.0, 1.0]


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
    label[dist < min_edge * 0.35] = 2  # ED shell -> completes WT
    label[dist < min_edge * 0.22] = 1  # NCR/NET core -> completes TC
    label[dist < min_edge * 0.10] = 3  # ET, innermost
    return label


def _write_case(
    prep_dir: Path,
    case_id: str,
    seed: int,
    has_label: bool = True,
    write_label_file: bool | None = None,
) -> None:
    """Writes one synthetic preprocessed case: image.npy, (usually) label.npy, meta.json.

    Args:
        prep_dir: Root preprocessed directory.
        case_id: Case identifier; output goes to `<prep_dir>/<case_id>/`.
        seed: RNG seed for the image, so cases differ but are reproducible.
        has_label: Value written into `meta.json["has_label"]`.
        write_label_file: Whether to actually write `label.npy` to disk.
            Defaults to `has_label` (matching real preprocessing output),
            but a "has_label=False" test case still needs SOME label.npy on
            disk for `build_val_transforms`' `LoadImaged(keys=["image",
            "label"])` to succeed (it has no `allow_missing_keys=True`) --
            see the docstring note in run_evaluation's test below. Passing
            `True` explicitly writes a (unused-for-metrics) placeholder
            label file even when `has_label=False`.
    """
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    image = rng.standard_normal((4, *CROPPED_SHAPE)).astype(np.float16)
    np.save(case_dir / "image.npy", image)

    if write_label_file is None:
        write_label_file = has_label
    if write_label_file:
        label = _build_synthetic_label(CROPPED_SHAPE)
        np.save(case_dir / "label.npy", label)

    write_json(
        {
            "case_id": case_id,
            "original_shape": list(ORIGINAL_SHAPE),
            "cropped_shape": list(CROPPED_SHAPE),
            "bbox": BBOX,
            "affine": np.eye(4).tolist(),
            "spacing": SPACING,
            "has_label": has_label,
            "label_voxel_counts": None,
        },
        case_dir / "meta.json",
    )


def _write_splits(path: Path, train: list[str], val: list[str], test: list[str]) -> None:
    write_yaml({"train": train, "val": val, "test": test}, path)


def _make_cfg(
    tmp_path: Path,
    prep_dir: Path,
    splits_path: Path,
    checkpoint_dir: Path,
    mc_dropout_overrides: dict | None = None,
    **evaluation_overrides: object,
) -> OmegaConf:
    """Builds a small evaluation config mirroring config.yaml + inference/default.yaml.

    Every value is kept tiny (32^3 volumes, a 2-level U-Net, roi_size=16, 0
    dataloader workers) so a whole evaluation run stays under a second on CPU.

    `mc_dropout_overrides` merges into `inference.mc_dropout` (default off,
    matching `configs/inference/default.yaml`), separately from
    `**evaluation_overrides` which merges into `inference.evaluation`.
    """
    out_dir = evaluation_overrides.pop("out_dir", tmp_path / "eval_out")

    evaluation = {
        "split": "test",
        "checkpoint": None,
        "out_dir": str(out_dir),
        "save_predictions": True,
        "save_probabilities": False,
        "save_logits": False,
        "strict_arch_check": True,
        "boundary_bands": [[0.0, 2.0], [2.0, 5.0], [5.0, 10.0], [10.0, float("inf")]],
        # Off by default, matching configs/inference/default.yaml -- so every
        # existing test's meaning is unchanged unless a caller passes
        # lesionwise=... as one of the **evaluation_overrides below (which
        # replaces this whole sub-dict; resolve_lesionwise falls back to its
        # own per-key defaults for anything the override omits).
        "lesionwise": {
            "enabled": False,
            "min_lesion_voxels": 50,
            "matching_threshold": 0.5,
            "nsd_tolerance_mm": 1.0,
            "connectivity": 26,
        },
    }
    evaluation.update(evaluation_overrides)

    mc_dropout = {
        "enabled": False,
        "num_samples": 3,
        "seed": 0,
        "require_dropout": True,
        "predictions_from": "deterministic",
        "save_fields": ["mutual_information"],
    }
    if mc_dropout_overrides:
        mc_dropout.update(mc_dropout_overrides)

    base = {
        "seed": 0,
        "device": "cpu",
        "data": {
            "in_channels": 4,
            "num_classes": 3,
            "regions": ["ET", "TC", "WT"],
            "splits": {"path": str(splits_path)},
            "preprocessing": {"out_dir": str(prep_dir)},
            "num_workers": 0,
        },
        "model": {
            "name": "unet3d",
            "in_channels": 4,
            "out_channels": 3,
            "channels": [4, 8],
            "strides": [2],
            "num_res_units": 1,
            "norm": "instance",
            "activation": "leakyrelu",
            "dropout": 0.0,
            "deep_supervision": False,
        },
        "training": {
            "checkpoint": {"dir": str(checkpoint_dir)},
        },
        "inference": {
            "sliding_window": {
                "roi_size": [16, 16, 16],
                "sw_batch_size": 1,
                "overlap": 0.25,
                "mode": "constant",
                "sigma_scale": 0.125,
                "output_device": None,
                "padding_mode": "constant",
            },
            "postprocess": {
                "threshold": 0.5,
                "enforce_nesting": True,
                "min_component_size": 0,
                "connectivity": 1,
                "keep_largest_only": False,
                "et_min_volume": 0,
            },
            "mc_dropout": mc_dropout,
            "evaluation": evaluation,
        },
    }
    return OmegaConf.create(base)


def _save_model_checkpoint(checkpoint_dir: Path, cfg: OmegaConf, is_best: bool = True) -> None:
    """Builds a real (tiny) unet3d model and checkpoints it, without training."""
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        epoch=0,
        global_step=0,
        best_metric=0.5,
        best_metric_name="val/dice_mean",
        best_metric_mode="max",
        cfg=cfg,
        is_best=is_best,
    )


class _StochasticStubModel(nn.Module):
    """Tiny model with one active `Dropout3d`, for MC-dropout tests.

    `unet3d` under the tests' tiny config has `dropout: 0.0` (an
    `nn.Identity`, per this project's convention -- see
    `mc_dropout_predict`'s `require_dropout` guard), so it cannot exercise
    MC-dropout's stochastic-passes path at all. This stub keeps the same
    `(B, 4, D, H, W) -> (B, 3, D, H, W)` contract `sliding_window_predict`
    needs (`Conv3d` with `padding=1` preserves spatial shape) but adds a
    `Dropout3d(p=0.5)` so repeated forward passes under `dropout_enabled`
    genuinely differ.

    Channel 0 (ET) is pushed hard negative on every pass, so its
    deterministic (eval-mode) prediction is reliably empty after
    thresholding -- this is what lets `mi_mean_fg_ET` exercise its NaN path
    deterministically rather than depending on random init.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, 3, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv(x))
        out = out.clone()
        out[:, 0] = out[:, 0] - 20.0  # ET channel: always strongly negative
        return out


def _save_stub_checkpoint(checkpoint_dir: Path) -> None:
    """Checkpoints a freshly built `_StochasticStubModel`, without training."""
    model = _StochasticStubModel()
    optimizer = torch.optim.Adam(model.parameters())
    fake_trained_cfg = OmegaConf.create({"model": {"name": "stub"}})
    save_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        epoch=0,
        global_step=0,
        best_metric=0.5,
        best_metric_name="val/dice_mean",
        best_metric_mode="max",
        cfg=fake_trained_cfg,
        is_best=True,
    )


# ---------------------------------------------------------------------------
# 1. resolve_checkpoint
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_falls_back_to_best(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(tmp_path, tmp_path / "prep", tmp_path / "splits.yaml", checkpoint_dir)
    _save_model_checkpoint(checkpoint_dir, cfg)

    resolved = resolve_checkpoint(cfg)

    assert resolved == checkpoint_dir / "best.pt"
    assert resolved.is_file()


def test_resolve_checkpoint_raises_and_lists_directory_contents(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "last.pt").write_bytes(b"not a real checkpoint")
    cfg = _make_cfg(tmp_path, tmp_path / "prep", tmp_path / "splits.yaml", checkpoint_dir)

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_checkpoint(cfg)

    message = str(excinfo.value)
    assert "best.pt" in message
    assert "last.pt" in message  # existing directory contents are listed


# ---------------------------------------------------------------------------
# 2. build_eval_dataloader: bad split name / empty split
# ---------------------------------------------------------------------------


def test_build_eval_dataloader_raises_on_unknown_split(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, train=["c0"], val=["c1"], test=["c2"])
    for case_id in ("c0", "c1", "c2"):
        _write_case(prep_dir, case_id, seed=hash(case_id) % 1000)
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, tmp_path / "checkpoints")

    with pytest.raises(ValueError, match="Unknown split"):
        build_eval_dataloader(cfg, "not_a_real_split")


def test_build_eval_dataloader_raises_on_empty_split(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, train=["c0"], val=["c1"], test=[])
    for case_id in ("c0", "c1"):
        _write_case(prep_dir, case_id, seed=1)
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, tmp_path / "checkpoints")

    with pytest.raises(ValueError, match="empty|0 cases"):
        build_eval_dataloader(cfg, "test")


# ---------------------------------------------------------------------------
# 3. build_eval_dataloader: case_id order matches the split file
# ---------------------------------------------------------------------------


def test_build_eval_dataloader_case_id_order_matches_split_file(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    # Deliberately out-of-alphabetical order, to catch any accidental sort.
    test_order = ["case_B", "case_A", "case_C"]
    _write_splits(splits_path, train=[], val=[], test=test_order)
    for i, case_id in enumerate(test_order):
        _write_case(prep_dir, case_id, seed=i)
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, tmp_path / "checkpoints")

    loader, case_ids = build_eval_dataloader(cfg, "test")

    assert case_ids == test_order
    assert len(loader.dataset) == len(test_order)


# ---------------------------------------------------------------------------
# 4. load_eval_model: architecture check
# ---------------------------------------------------------------------------


def test_load_eval_model_raises_on_arch_mismatch(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(tmp_path, tmp_path / "prep", tmp_path / "splits.yaml", checkpoint_dir)

    # The checkpoint's OWN stored config claims a different model name than
    # what is about to be evaluated under -- built directly, not by training,
    # since only the stored config's model.name matters for this check.
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters())
    fake_trained_cfg = OmegaConf.create({"model": {"name": "swinunetr"}})
    save_checkpoint(checkpoint_dir, model, optimizer, epoch=0, global_step=0, cfg=fake_trained_cfg)
    checkpoint_path = checkpoint_dir / "last.pt"

    device = torch.device("cpu")
    with pytest.raises(ValueError, match="swinunetr"):
        load_eval_model(cfg, checkpoint_path, device)


def test_load_eval_model_skips_check_when_strict_arch_check_false(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(
        tmp_path,
        tmp_path / "prep",
        tmp_path / "splits.yaml",
        checkpoint_dir,
        strict_arch_check=False,
    )

    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters())
    fake_trained_cfg = OmegaConf.create({"model": {"name": "swinunetr"}})
    save_checkpoint(checkpoint_dir, model, optimizer, epoch=0, global_step=0, cfg=fake_trained_cfg)
    checkpoint_path = checkpoint_dir / "last.pt"

    device = torch.device("cpu")
    loaded_model, resume_state = load_eval_model(cfg, checkpoint_path, device)

    assert loaded_model is not None
    assert resume_state.config.model.name == "swinunetr"


# ---------------------------------------------------------------------------
# 5-9. run_evaluation end-to-end
# ---------------------------------------------------------------------------


def test_run_evaluation_writes_per_case_and_summary_csv(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001"]
    for i, case_id in enumerate(case_ids):
        _write_case(prep_dir, case_id, seed=i, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    _save_model_checkpoint(checkpoint_dir, cfg)

    per_case_df = run_evaluation(cfg)

    out_dir = Path(cfg.inference.evaluation.out_dir)
    assert (out_dir / "per_case_metrics.csv").is_file()
    assert (out_dir / "summary.csv").is_file()

    assert isinstance(per_case_df, pd.DataFrame)
    assert len(per_case_df) == 2
    assert set(case_ids) == set(per_case_df.index)
    for column in ("dice_ET", "dice_TC", "dice_WT", "dice_mean"):
        assert column in per_case_df.columns

    on_disk_df = pd.read_csv(out_dir / "per_case_metrics.csv", index_col="case_id")
    assert len(on_disk_df) == 2


def test_run_evaluation_boundary_bands_add_columns_and_null_omits_them(tmp_path: Path):
    """Boundary stratification is ADDITIVE -- it must not move an existing metric.

    An already-published results row (docs/experiments.md's baseline_unet3d)
    stays valid only if turning this on changes nothing but the column set.
    """
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=["case_000"])

    checkpoint_dir = tmp_path / "checkpoints"
    cfg_on = _make_cfg(
        tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir=tmp_path / "eval_on"
    )
    _save_model_checkpoint(checkpoint_dir, cfg_on)
    df_on = run_evaluation(cfg_on)

    cfg_off = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir=tmp_path / "eval_off",
        boundary_bands=None,
    )
    df_off = run_evaluation(cfg_off)

    # The band labels come from metrics.boundary.band_label, not from a
    # string re-derived here -- a drift in that format must fail this test.
    for region in ("ET", "TC", "WT"):
        for prefix in ("berr", "bfnr", "bfpr", "bn"):
            assert f"{prefix}_{region}_0-2" in df_on.columns
            assert f"{prefix}_{region}_10-inf" in df_on.columns
    assert not [c for c in df_off.columns if c.startswith("berr_")]

    # Every pre-existing metric is bit-identical between the two runs.
    shared = [c for c in df_off.columns if c in df_on.columns]
    assert shared, "the off-run produced no columns to compare"
    pd.testing.assert_frame_equal(df_on[shared], df_off[shared])

    # berr == bfnr + bfpr, on real evaluation output rather than a synthetic
    # array -- the identity that makes the over/under-segmentation split
    # trustworthy.
    for region in ("ET", "TC", "WT"):
        for label in ("0-2", "2-5", "5-10", "10-inf"):
            err = df_on[f"berr_{region}_{label}"].iloc[0]
            fnr = df_on[f"bfnr_{region}_{label}"].iloc[0]
            fpr = df_on[f"bfpr_{region}_{label}"].iloc[0]
            if pd.isna(err):
                assert pd.isna(fnr) and pd.isna(fpr)
                assert df_on[f"bn_{region}_{label}"].iloc[0] == 0.0
            else:
                assert err == pytest.approx(fnr + fpr)


def test_run_evaluation_prediction_shape_is_original_not_cropped(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001"]
    for i, case_id in enumerate(case_ids):
        _write_case(prep_dir, case_id, seed=i, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    _save_model_checkpoint(checkpoint_dir, cfg)

    run_evaluation(cfg)

    out_dir = Path(cfg.inference.evaluation.out_dir)
    for case_id in case_ids:
        pred_path = out_dir / "predictions" / f"{case_id}.npy"
        assert pred_path.is_file()
        pred = np.load(pred_path)
        assert pred.dtype == np.uint8
        assert set(np.unique(pred).tolist()).issubset({0, 1, 2, 3})
        # The load-bearing assertion: original shape, NOT the cropped
        # (32, 32, 32) shape the model actually predicted at.
        assert pred.shape == ORIGINAL_SHAPE


def test_run_evaluation_save_probabilities_flag(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000"]
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"

    # save_probabilities=False (default): no probabilities/ directory at all.
    cfg_off = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir=tmp_path / "eval_off",
        save_probabilities=False,
    )
    _save_model_checkpoint(checkpoint_dir, cfg_off)
    run_evaluation(cfg_off)
    assert not (Path(cfg_off.inference.evaluation.out_dir) / "probabilities").exists()

    # save_probabilities=True: float16 files in CROPPED shape.
    cfg_on = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir=tmp_path / "eval_on",
        save_probabilities=True,
    )
    run_evaluation(cfg_on)
    prob_path = Path(cfg_on.inference.evaluation.out_dir) / "probabilities" / "case_000.npy"
    assert prob_path.is_file()
    probs = np.load(prob_path)
    assert probs.dtype == np.float16
    assert probs.shape == (3, *CROPPED_SHAPE)


def test_run_evaluation_save_logits_flag(tmp_path: Path):
    """save_logits writes raw logits, and they are NOT probabilities.

    Temperature scaling needs logits: fp16 saturates any probability above
    ~0.99976 to exactly 1.0, whose logit is +inf, so the most-confident
    voxels -- the ones that drive miscalibration -- could not be fit from
    saved probabilities. This test pins that logits/ holds pre-sigmoid
    values by asserting they leave [0, 1], which probabilities never do.
    """
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000"]
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"

    cfg_off = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir=tmp_path / "eval_off",
        save_logits=False,
    )
    _save_model_checkpoint(checkpoint_dir, cfg_off)
    run_evaluation(cfg_off)
    assert not (Path(cfg_off.inference.evaluation.out_dir) / "logits").exists()

    cfg_on = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir=tmp_path / "eval_on",
        save_logits=True,
        save_probabilities=True,
    )
    run_evaluation(cfg_on)
    out_dir = Path(cfg_on.inference.evaluation.out_dir)

    logits_path = out_dir / "logits" / "case_000.npy"
    assert logits_path.is_file()
    assert np.load(logits_path).dtype == np.float16
    logits = np.load(logits_path).astype(np.float32)
    assert logits.shape == (3, *CROPPED_SHAPE)

    # The decisive assertion: sigmoid(logits) must reproduce the saved
    # probabilities. Equality of shape alone would also hold if the writer
    # had accidentally saved probabilities into logits/.
    probs = np.load(out_dir / "probabilities" / "case_000.npy").astype(np.float32)
    np.testing.assert_allclose(1.0 / (1.0 + np.exp(-logits)), probs, atol=2e-3)


def test_run_evaluation_unlabeled_case_skipped_for_metrics_but_predicted(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"

    _write_case(prep_dir, "labeled_case", seed=0, has_label=True)
    # has_label=False in meta.json, but a placeholder label.npy is still
    # written on disk -- see _write_case's docstring: build_val_transforms'
    # LoadImaged has no allow_missing_keys=True, so a case with no label.npy
    # at all cannot currently pass through the shared val transform
    # pipeline. Whether a case counts toward metrics is decided purely by
    # meta["has_label"], independent of whether label.npy happens to exist.
    _write_case(prep_dir, "unlabeled_case", seed=1, has_label=False, write_label_file=True)
    case_ids = ["labeled_case", "unlabeled_case"]
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    _save_model_checkpoint(checkpoint_dir, cfg)

    per_case_df = run_evaluation(cfg)

    assert len(per_case_df) == 1
    assert list(per_case_df.index) == ["labeled_case"]

    out_dir = Path(cfg.inference.evaluation.out_dir)
    assert (out_dir / "predictions" / "labeled_case.npy").is_file()
    assert (out_dir / "predictions" / "unlabeled_case.npy").is_file()


def test_run_evaluation_writes_eval_config_yaml(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000"]
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    _save_model_checkpoint(checkpoint_dir, cfg)

    run_evaluation(cfg)

    out_dir = Path(cfg.inference.evaluation.out_dir)
    eval_config_path = out_dir / "eval_config.yaml"
    assert eval_config_path.is_file()

    with eval_config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["model"]["name"] == "unet3d"


# ---------------------------------------------------------------------------
# 10-18. MC-dropout wiring
# ---------------------------------------------------------------------------


def _setup_mc_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_ids: list[str],
    mc_dropout_overrides: dict | None = None,
    **evaluation_overrides: object,
) -> OmegaConf:
    """Shared setup for the MC-dropout `run_evaluation` tests below.

    Writes `len(case_ids)` synthetic labeled cases, checkpoints a fresh
    `_StochasticStubModel`, and monkeypatches `evaluate_script.build_model`
    to return a NEW `_StochasticStubModel` instance (its random init does
    not matter -- `load_eval_model` immediately overwrites it with the
    checkpointed weights via `load_state_dict`). `strict_arch_check=False`
    because the checkpoint's stored `model.name` ("stub") does not match
    the config's `model.name` ("unet3d") -- irrelevant here since the model
    actually used is the monkeypatched stub either way.
    """
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    for i, case_id in enumerate(case_ids):
        _write_case(prep_dir, case_id, seed=i, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir)

    cfg = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        mc_dropout_overrides=mc_dropout_overrides,
        strict_arch_check=False,
        **evaluation_overrides,
    )
    monkeypatch.setattr(evaluate_script, "build_model", lambda cfg: _StochasticStubModel())
    return cfg


def test_mc_dropout_disabled_output_matches_baseline_file_set(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001"]
    for i, case_id in enumerate(case_ids):
        _write_case(prep_dir, case_id, seed=i, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir)
    _save_model_checkpoint(checkpoint_dir, cfg)

    run_evaluation(cfg)

    out_dir = Path(cfg.inference.evaluation.out_dir)
    entries = sorted(p.name for p in out_dir.iterdir())
    expected = ["eval_config.yaml", "per_case_metrics.csv", "predictions", "summary.csv"]
    assert entries == sorted(expected)
    assert not (out_dir / "uncertainty").exists()
    assert not (out_dir / "uncertainty_summary.csv").exists()


def test_mc_dropout_enabled_writes_uncertainty_npy_per_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000", "case_001"]
    cfg = _setup_mc_run(tmp_path, monkeypatch, case_ids, mc_dropout_overrides={"enabled": True})

    run_evaluation(cfg)

    out_dir = Path(cfg.inference.evaluation.out_dir)
    for case_id in case_ids:
        arr_path = out_dir / "uncertainty" / f"{case_id}.npy"
        assert arr_path.is_file()
        arr = np.load(arr_path)
        assert arr.dtype == np.float16
        assert arr.shape == (3, *CROPPED_SHAPE)


def test_mc_dropout_uncertainty_summary_csv_has_expected_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000", "case_001"]
    cfg = _setup_mc_run(tmp_path, monkeypatch, case_ids, mc_dropout_overrides={"enabled": True})

    run_evaluation(cfg)

    out_dir = Path(cfg.inference.evaluation.out_dir)
    csv_path = out_dir / "uncertainty_summary.csv"
    assert csv_path.is_file()

    df = pd.read_csv(csv_path, index_col="case_id")
    assert len(df) == len(case_ids)
    assert set(df.index) == set(case_ids)

    expected_columns = {"num_samples"}
    for region in ("ET", "TC", "WT"):
        expected_columns |= {
            f"mi_mean_{region}",
            f"mi_max_{region}",
            f"mi_mean_fg_{region}",
            f"entropy_mean_{region}",
        }
    assert expected_columns.issubset(set(df.columns))


def test_mc_dropout_mi_mean_fg_is_nan_for_empty_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # _StochasticStubModel always pushes the ET channel strongly negative,
    # so its deterministic prediction is reliably empty for ET.
    case_ids = ["case_000"]
    cfg = _setup_mc_run(tmp_path, monkeypatch, case_ids, mc_dropout_overrides={"enabled": True})

    run_evaluation(cfg)

    out_dir = Path(cfg.inference.evaluation.out_dir)
    df = pd.read_csv(out_dir / "uncertainty_summary.csv", index_col="case_id")
    assert pd.isna(df.loc["case_000", "mi_mean_fg_ET"])


def test_mc_dropout_save_fields_writes_multiple_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000"]
    cfg = _setup_mc_run(
        tmp_path,
        monkeypatch,
        case_ids,
        mc_dropout_overrides={
            "enabled": True,
            "save_fields": ["mutual_information", "predictive_entropy"],
        },
    )

    run_evaluation(cfg)

    out_dir = Path(cfg.inference.evaluation.out_dir)
    assert (out_dir / "uncertainty" / "case_000.npy").is_file()
    assert (out_dir / "entropy_total" / "case_000.npy").is_file()


def test_mc_dropout_unknown_save_field_raises_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000"]
    cfg = _setup_mc_run(
        tmp_path,
        monkeypatch,
        case_ids,
        mc_dropout_overrides={"enabled": True, "save_fields": ["not_a_real_field"]},
    )

    out_dir = Path(cfg.inference.evaluation.out_dir)
    with pytest.raises(ValueError, match="not_a_real_field"):
        run_evaluation(cfg)

    assert not out_dir.exists()


def test_mc_dropout_bogus_predictions_from_raises_naming_valid_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case_ids = ["case_000"]
    cfg = _setup_mc_run(
        tmp_path,
        monkeypatch,
        case_ids,
        mc_dropout_overrides={"enabled": True, "predictions_from": "bogus"},
    )

    out_dir = Path(cfg.inference.evaluation.out_dir)
    with pytest.raises(ValueError, match="deterministic"):
        run_evaluation(cfg)
    with pytest.raises(ValueError, match="mc_mean"):
        run_evaluation(cfg)

    assert not out_dir.exists()


def test_evaluate_case_mc_mean_changes_postprocess_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`predictions_from="mc_mean"` must feed a DIFFERENT tensor into
    `postprocess_logits` than the deterministic pass did -- proving the
    branch is actually live, not re-testing postprocess_logits itself."""
    calls: list[torch.Tensor] = []
    original_postprocess_logits = evaluate_script.postprocess_logits

    def _spy(logits: torch.Tensor, cfg: OmegaConf) -> torch.Tensor:
        calls.append(logits)
        return original_postprocess_logits(logits, cfg)

    monkeypatch.setattr(evaluate_script, "postprocess_logits", _spy)

    cfg = _make_cfg(
        tmp_path,
        tmp_path / "prep",
        tmp_path / "splits.yaml",
        tmp_path / "checkpoints",
        mc_dropout_overrides={"enabled": True, "predictions_from": "mc_mean"},
    )
    model = _StochasticStubModel()
    model.eval()
    batch = {"image": torch.randn(1, 4, *CROPPED_SHAPE)}
    device = torch.device("cpu")

    evaluate_case(model, batch, cfg, device)

    assert len(calls) == 2
    assert not torch.equal(calls[0], calls[1])


def test_mc_dropout_enabled_deterministic_predictions_leave_dice_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Turning MC-dropout on (with the default predictions_from="deterministic")
    must not move a single reported Dice value -- the regression that protects
    an already-published baseline row."""
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001"]
    for i, case_id in enumerate(case_ids):
        _write_case(prep_dir, case_id, seed=i, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    # ONE checkpoint, reused for both runs below: re-instantiating
    # _StochasticStubModel per run would give each run different random
    # conv weights, confounding "did MC-dropout move Dice" with "did the
    # model change".
    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir)
    monkeypatch.setattr(evaluate_script, "build_model", lambda cfg: _StochasticStubModel())

    cfg_off = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        mc_dropout_overrides={"enabled": False},
        strict_arch_check=False,
        out_dir=tmp_path / "eval_off",
    )
    per_case_off = run_evaluation(cfg_off)

    cfg_on = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        mc_dropout_overrides={"enabled": True, "predictions_from": "deterministic"},
        strict_arch_check=False,
        out_dir=tmp_path / "eval_on",
    )
    per_case_on = run_evaluation(cfg_on)

    dice_columns = ["dice_ET", "dice_TC", "dice_WT", "dice_mean"]
    pd.testing.assert_frame_equal(
        per_case_off[dice_columns].sort_index(), per_case_on[dice_columns].sort_index()
    )


# ---------------------------------------------------------------------------
# 19-24. Lesion-wise metrics wiring (cfg.inference.evaluation.lesionwise)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _PANOPTICA_MISSING,
    reason="panoptica is not installed in this venv (see requirements-analysis.txt); "
    "run from .venv-analysis to exercise lesion-wise scoring",
)
def test_run_evaluation_lesionwise_adds_columns_and_disabled_omits_them(tmp_path: Path):
    """Lesion-wise scoring is ADDITIVE -- it must not move an existing metric.

    Same reasoning as the boundary-bands additivity test above: an
    already-published results row stays valid only if turning this on
    changes nothing but the column set.
    """
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=["case_000"])

    checkpoint_dir = tmp_path / "checkpoints"
    cfg_on = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir=tmp_path / "eval_on",
        lesionwise={"enabled": True, "min_lesion_voxels": 0},
    )
    _save_model_checkpoint(checkpoint_dir, cfg_on)
    df_on = run_evaluation(cfg_on)

    cfg_off = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir=tmp_path / "eval_off",
        lesionwise={"enabled": False},
    )
    df_off = run_evaluation(cfg_off)

    for region in ("ET", "TC", "WT"):
        for prefix in LESIONWISE_METRIC_PREFIXES:
            assert f"{prefix}_{region}" in df_on.columns
            assert f"{prefix}_{region}" not in df_off.columns
    for prefix in ("lwdice", "lwnsd", "lwf1"):
        assert f"{prefix}_mean" in df_on.columns
        assert f"{prefix}_mean" not in df_off.columns

    # Every pre-existing metric is bit-identical between the two runs -- the
    # whole point of "additive only".
    shared = [c for c in df_off.columns if c in df_on.columns]
    assert shared, "the off-run produced no columns to compare"
    pd.testing.assert_frame_equal(df_on[shared], df_off[shared])


def test_resolve_lesionwise_absent_key_returns_none():
    """Backward compatibility: a config composed before this key existed
    (an older saved eval_config.yaml, a minimal test config) must still run,
    with lesion-wise scoring simply off. Runs in EVERY venv, unlike the
    tests above -- resolve_lesionwise never imports panoptica."""
    eval_cfg = OmegaConf.create({"split": "test"})
    assert resolve_lesionwise(eval_cfg) is None


def test_resolve_lesionwise_disabled_returns_none():
    eval_cfg = OmegaConf.create({"lesionwise": {"enabled": False, "min_lesion_voxels": 10}})
    assert resolve_lesionwise(eval_cfg) is None

    # The whole block set to null is equally "off", same as boundary_bands.
    eval_cfg_null = OmegaConf.create({"lesionwise": None})
    assert resolve_lesionwise(eval_cfg_null) is None


def test_resolve_lesionwise_returns_settings_when_enabled():
    eval_cfg = OmegaConf.create(
        {
            "lesionwise": {
                "enabled": True,
                "min_lesion_voxels": 25,
                "matching_threshold": 0.3,
                "nsd_tolerance_mm": 2.0,
                "connectivity": 6,
            }
        }
    )

    settings = resolve_lesionwise(eval_cfg)

    assert settings == {
        "min_lesion_voxels": 25,
        "matching_threshold": 0.3,
        "nsd_tolerance_mm": 2.0,
        "connectivity": 6,
    }
    assert isinstance(settings["min_lesion_voxels"], int)
    assert isinstance(settings["matching_threshold"], float)
    assert isinstance(settings["nsd_tolerance_mm"], float)
    assert isinstance(settings["connectivity"], int)


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_lesion_voxels": -1},
        {"matching_threshold": 0.0},
        {"matching_threshold": 1.5},
        {"nsd_tolerance_mm": 0.0},
        {"nsd_tolerance_mm": -1.0},
        {"connectivity": 10},
    ],
    ids=[
        "negative_min_lesion_voxels",
        "matching_threshold_zero",
        "matching_threshold_above_one",
        "nsd_tolerance_zero",
        "nsd_tolerance_negative",
        "connectivity_not_6_18_26",
    ],
)
def test_resolve_lesionwise_raises_on_invalid_settings(overrides: dict[str, object]):
    settings = {
        "enabled": True,
        "min_lesion_voxels": 50,
        "matching_threshold": 0.5,
        "nsd_tolerance_mm": 1.0,
        "connectivity": 26,
    }
    settings.update(overrides)
    eval_cfg = OmegaConf.create({"lesionwise": settings})

    with pytest.raises(ValueError):
        resolve_lesionwise(eval_cfg)


def test_run_evaluation_lesionwise_raises_before_inference_when_panoptica_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A missing panoptica must fail BEFORE any case is scored -- not after a
    sliding-window pass over the whole split. Not skipped: this is the test
    that protects a 25-minute real run from discovering the wrong venv only
    at the very end."""
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000"]
    _write_case(prep_dir, "case_000", seed=0, has_label=True)
    _write_splits(splits_path, train=[], val=[], test=case_ids)

    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        checkpoint_dir,
        out_dir=tmp_path / "eval_out",
        lesionwise={"enabled": True},
    )
    _save_model_checkpoint(checkpoint_dir, cfg)

    # Simulates "panoptica is not importable" regardless of what is actually
    # installed in the venv running this test -- monkeypatches the exact
    # entry point run_evaluation calls to check importability, reusing the
    # same message _load_panoptica itself raises.
    def _raise_missing() -> None:
        raise ImportError(
            "lesionwise_case_metrics() requires the 'panoptica' package, which is "
            "deliberately NOT installed in the project's main .venv. It lives in the "
            "separate '.venv-analysis' virtualenv described in requirements-analysis.txt."
        )

    monkeypatch.setattr(evaluate_script, "_load_panoptica", _raise_missing)

    with pytest.raises(ImportError, match="requirements-analysis.txt"):
        run_evaluation(cfg)

    # The decisive assertion: no per-case CSV was written, which is what
    # proves this failed before scoring the first case rather than after.
    out_dir = Path(cfg.inference.evaluation.out_dir)
    assert not (out_dir / "per_case_metrics.csv").exists()
