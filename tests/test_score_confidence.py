"""Tests for scripts/score_confidence.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import -- the exact
same pattern as tests/test_extract_ambiguity.py / tests/test_conformal_script.py.

No real BraTS data anywhere here: synthetic tiny volumes, CPU only, whole file runs in a few
seconds.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from neurovision.training.checkpoint import save_checkpoint
from neurovision.utils.io import write_json, write_yaml

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_confidence.py"
_spec = importlib.util.spec_from_file_location("score_confidence_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
score_confidence_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["score_confidence_script"] = score_confidence_script
_spec.loader.exec_module(score_confidence_script)

_ConfidenceWrapper = score_confidence_script._ConfidenceWrapper
label_free_sample_mask = score_confidence_script.label_free_sample_mask
draw_voxel_indices = score_confidence_script.draw_voxel_indices
bernoulli_entropy_nats = score_confidence_script.bernoulli_entropy_nats
score_case = score_confidence_script.score_case
run_scoring = score_confidence_script.run_scoring

VOLUME_SHAPE: tuple[int, int, int] = (12, 12, 12)
NUM_REGIONS = 3


# ---------------------------------------------------------------------------
# 1. Polarity -- the most important test in this file.
# ---------------------------------------------------------------------------


def test_confidence_polarity_is_probability_of_correct() -> None:
    """A confidence head that is PERFECTLY right (high logits exactly where the segmentation
    is correct) must give auroc_confidence near 1.0, not near 0.0.

    Construction: every voxel is predicted positive (seg_logits all large-positive), so
    `error = pred != target` is simply `target == 0`. The confidence head is built to be
    high-confidence (large positive logit) exactly where target == 1 (i.e. where the
    prediction is correct) and low-confidence (large negative logit) where target == 0 (i.e.
    where the prediction is an error). If the polarity in score_case were backwards -- scoring
    sigmoid(confidence_logits) directly as an "error score" instead of
    1 - sigmoid(confidence_logits) -- this would give an AUROC near 0.0 instead.
    """
    shape = (1, *VOLUME_SHAPE)
    # A small per-voxel jitter around 6.0 -- every voxel is still (overwhelmingly) predicted
    # positive, but the entropy control is not perfectly constant, which would otherwise make
    # residualised_auroc's rank-regression degenerate (all-identical x values).
    seg_logits = 6.0 + 0.01 * torch.arange(np.prod(VOLUME_SHAPE)).reshape(shape).float()

    target = torch.zeros(shape)
    target[0, :6] = 1.0  # half correct (target=1), half error (target=0)

    confidence_logits = torch.where(target > 0.5, torch.tensor(6.0), torch.tensor(-6.0))

    row = score_case(
        seg_logits,
        confidence_logits,
        target,
        spacing=(1.0, 1.0, 1.0),
        threshold=0.5,
        dilation_mm=100.0,  # generous enough that the whole volume is sampled
        max_voxels=2000,
        generator=np.random.default_rng(0),
        region_names=("R",),
    )

    assert row["skip_reason_R"] == ""
    assert row["auroc_confidence_R"] == pytest.approx(1.0, abs=1e-6)


def test_confidence_polarity_backwards_would_give_near_zero_auroc() -> None:
    """Sanity check on the test above: scoring the RAW (un-flipped) confidence probability as
    an error score on the exact same data gives an AUROC near 0.0 -- confirming the fixture
    actually discriminates the two polarities, not just a coincidence that both give ~0.5."""
    from neurovision.analysis.detection import auroc

    shape = (1, *VOLUME_SHAPE)
    seg_logits = torch.full(shape, 6.0)
    target = torch.zeros(shape)
    target[0, :6] = 1.0
    confidence_logits = torch.where(target > 0.5, torch.tensor(6.0), torch.tensor(-6.0))

    error = (torch.sigmoid(seg_logits) > 0.5) != (target > 0.5)
    wrong_polarity_score = torch.sigmoid(confidence_logits)  # NOT flipped -- the bug

    backwards_auroc = auroc(wrong_polarity_score.reshape(-1).numpy(), error.reshape(-1).numpy())
    assert backwards_auroc == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Entropy: finite under fp16 saturation
# ---------------------------------------------------------------------------


def test_entropy_is_finite_under_fp16_saturation() -> None:
    logits = torch.tensor([-40.0, 40.0, 0.0, -1000.0, 1000.0], dtype=torch.float16)
    entropy = bernoulli_entropy_nats(logits)
    assert torch.isfinite(entropy).all()
    assert (entropy >= 0.0).all()


# ---------------------------------------------------------------------------
# 3. label_free_sample_mask / draw_voxel_indices never touch a label
# ---------------------------------------------------------------------------


def test_sample_mask_never_uses_the_label() -> None:
    pred_region = np.zeros(VOLUME_SHAPE, dtype=bool)
    pred_region[4:8, 4:8, 4:8] = True
    spacing = (1.0, 1.0, 1.0)

    label_a = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    label_b = np.ones(VOLUME_SHAPE, dtype=np.uint8)

    # label_free_sample_mask and draw_voxel_indices have no `label` parameter at all --
    # calling them with two completely different labelings "in scope" produces the identical
    # result both times because the label plays no role in the computation whatsoever.
    mask_a = label_free_sample_mask(pred_region, spacing, dilation_mm=2.0)
    mask_b = label_free_sample_mask(pred_region, spacing, dilation_mm=2.0)
    assert np.array_equal(mask_a, mask_b)

    drawn_a = draw_voxel_indices(mask_a, max_voxels=20, generator=np.random.default_rng(0))
    drawn_b = draw_voxel_indices(mask_b, max_voxels=20, generator=np.random.default_rng(0))
    assert np.array_equal(drawn_a, drawn_b)

    del label_a, label_b  # never consulted -- exists only to make the claim visible


def test_sample_mask_all_foreground_and_all_empty_edge_cases() -> None:
    spacing = (1.0, 1.0, 1.0)

    all_foreground = np.ones(VOLUME_SHAPE, dtype=bool)
    assert label_free_sample_mask(all_foreground, spacing, dilation_mm=1.0).all()

    all_empty = np.zeros(VOLUME_SHAPE, dtype=bool)
    assert not label_free_sample_mask(all_empty, spacing, dilation_mm=1.0).any()


def test_draw_voxel_indices_caps_at_max_voxels_and_is_reproducible() -> None:
    mask = np.ones(VOLUME_SHAPE, dtype=bool)
    generator_a = np.random.default_rng(7)
    generator_b = np.random.default_rng(7)

    drawn_a = draw_voxel_indices(mask, max_voxels=50, generator=generator_a)
    drawn_b = draw_voxel_indices(mask, max_voxels=50, generator=generator_b)

    assert drawn_a.size == 50
    assert np.array_equal(drawn_a, drawn_b)
    assert len(set(drawn_a.tolist())) == 50  # without replacement


# ---------------------------------------------------------------------------
# 4. Single-class sample -> skipped, never scored as 0.5
# ---------------------------------------------------------------------------


def test_region_with_single_class_sample_is_skipped_not_scored() -> None:
    shape = (1, *VOLUME_SHAPE)
    seg_logits = torch.full(shape, 6.0)  # every voxel predicted positive
    target = torch.ones(shape)  # every voxel actually positive -> always correct, never error
    confidence_logits = torch.zeros(shape)

    row = score_case(
        seg_logits,
        confidence_logits,
        target,
        spacing=(1.0, 1.0, 1.0),
        threshold=0.5,
        dilation_mm=100.0,
        max_voxels=2000,
        generator=np.random.default_rng(0),
        region_names=("R",),
    )

    assert row["skip_reason_R"] == "single_class"
    assert row["n_voxels_R"] > 0  # voxels WERE drawn -- it is the class balance that skips it
    assert np.isnan(row["auroc_confidence_R"])
    assert np.isnan(row["auroc_entropy_R"])
    assert np.isnan(row["auroc_confidence_residual_R"])
    assert row["auroc_confidence_R"] != 0.5  # never silently substituted


def test_region_with_empty_predicted_foreground_is_skipped_as_empty_mask() -> None:
    shape = (1, *VOLUME_SHAPE)
    seg_logits = torch.full(shape, -6.0)  # every voxel predicted NEGATIVE -> empty foreground
    target = torch.zeros(shape)
    confidence_logits = torch.zeros(shape)

    row = score_case(
        seg_logits,
        confidence_logits,
        target,
        spacing=(1.0, 1.0, 1.0),
        threshold=0.5,
        dilation_mm=5.0,
        max_voxels=2000,
        generator=np.random.default_rng(0),
        region_names=("R",),
    )

    assert row["skip_reason_R"] == "empty_mask"
    assert row["n_voxels_R"] == 0
    assert np.isnan(row["auroc_confidence_R"])


# ---------------------------------------------------------------------------
# 5. _ConfidenceWrapper raises when the model has no confidence head
# ---------------------------------------------------------------------------


class _NoHeadsModel(nn.Module):
    """A stub with no `heads` attribute at all -- like baseline_unet3d."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, NUM_REGIONS, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _NoneConfidenceModel(nn.Module):
    """A stub with a real `heads` attribute, but `heads.confidence is None`."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(4, NUM_REGIONS, kernel_size=3, padding=1)
        self.heads = SimpleNamespace(confidence=None)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def test_raises_when_model_has_no_confidence_head() -> None:
    with pytest.raises(ValueError, match="model.heads.confidence"):
        _ConfidenceWrapper(_NoHeadsModel())


def test_raises_when_model_heads_confidence_is_none() -> None:
    with pytest.raises(ValueError, match="model.heads.confidence"):
        _ConfidenceWrapper(_NoneConfidenceModel())


# ---------------------------------------------------------------------------
# 6. Config block reachable at cfg.analysis.confidence
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_at_the_composed_path() -> None:
    """The REAL project config, composed through Hydra, must expose the confidence block at
    `cfg.analysis.confidence` -- the exact path scripts/score_confidence.py reads.

    Mirrors tests/test_conformal_script.py's
    test_conformal_config_block_is_reachable_at_the_composed_path: a hand-built OmegaConf
    fixture in every other test in this file could put "confidence" at the wrong nesting level
    and every one of those tests would still pass. Composing the real configs/ tree here is
    what closes that gap.
    """
    overrides = ["data.root_dir=/unused/for/this/test"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    assert "analysis" in cfg
    assert "confidence" in cfg.analysis

    conf_cfg = cfg.analysis.confidence
    expected_keys = {
        "checkpoint",
        "split",
        "out_dir",
        "max_voxels_per_case",
        "dilation_mm",
        "regions",
    }
    assert expected_keys <= set(conf_cfg.keys())
    assert conf_cfg.checkpoint is None
    assert conf_cfg.split == "test"
    assert conf_cfg.max_voxels_per_case == 20000
    assert conf_cfg.dilation_mm == 5.0
    assert list(conf_cfg.regions) == ["ET", "TC", "WT"]


# ---------------------------------------------------------------------------
# 7. End-to-end smoke test
# ---------------------------------------------------------------------------


class _FixedConfidenceModel(nn.Module):
    """A tiny, deterministic stand-in for NeuroVisionX: `forward_with_auxiliary` returns
    FIXED (input-independent) segmentation and confidence logit patterns, so the smoke test
    controls exactly which voxels are "correct" vs "error" via the ground-truth label alone,
    with no dependence on random conv weights."""

    def __init__(self, shape: tuple[int, int, int] = VOLUME_SHAPE, num_regions: int = NUM_REGIONS):
        super().__init__()
        # A real (if unused-in-output) parameter, so this is checkpoint-able and
        # torch.optim.Adam(model.parameters()) does not raise on an empty parameter list.
        self._dummy_param = nn.Parameter(torch.zeros(1))
        self.heads = SimpleNamespace(confidence=self._dummy_param)  # non-None marker only

        d, h, w = shape
        # Inner cube predicted positive, everything else predicted negative.
        pred_inner = np.zeros(shape, dtype=bool)
        pred_inner[3 : d - 3, 3 : h - 3, 3 : w - 3] = True
        seg_pattern = np.where(pred_inner, 6.0, -6.0).astype(np.float32)
        conf_pattern = np.zeros(shape, dtype=np.float32)  # sigmoid(0) = 0.5, deliberately
        # uninformative here -- polarity itself is exercised by
        # test_confidence_polarity_is_probability_of_correct, not this smoke test.
        self.register_buffer("seg_pattern", torch.from_numpy(np.stack([seg_pattern] * num_regions)))
        self.register_buffer(
            "conf_pattern", torch.from_numpy(np.stack([conf_pattern] * num_regions))
        )

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        return self.seg_pattern.unsqueeze(0).expand(b, -1, -1, -1, -1).clone()

    def forward_with_auxiliary(self, x: Tensor) -> tuple[Tensor, Tensor, None]:
        b = x.shape[0]
        seg = self.seg_pattern.unsqueeze(0).expand(b, -1, -1, -1, -1).clone()
        conf = self.conf_pattern.unsqueeze(0).expand(b, -1, -1, -1, -1).clone()
        return seg, conf, None


def _write_synthetic_case(prep_dir: Path, case_id: str, shape: tuple[int, int, int]) -> None:
    """Writes one synthetic preprocessed case: image.npy, label.npy, meta.json.

    The label is built so that, WITHIN the fixed predicted-foreground inner cube
    `_FixedConfidenceModel` always predicts, half the voxels (by z) are correct and half are
    errors -- guaranteeing every region has a mixed-class sample (never skipped) regardless of
    dilation_mm.
    """
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    image = rng.standard_normal((4, *shape)).astype(np.float16)
    np.save(case_dir / "image.npy", image)

    d, h, w = shape
    # BraTS class label {0, 1, 2, 3}. Half of the volume (by z) is enhancing tumor (class 3,
    # -> ET/TC/WT all True there); the other half is background (class 0, -> all False).
    label = np.zeros(shape, dtype=np.uint8)
    label[: d // 2] = 3
    np.save(case_dir / "label.npy", label)

    write_json(
        {
            "case_id": case_id,
            "has_label": True,
            "bbox": [[0, d], [0, h], [0, w]],
            "original_shape": list(shape),
            "spacing": [1.0, 1.0, 1.0],
        },
        case_dir / "meta.json",
    )


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


def _compose_cfg(
    tmp_path: Path,
    prep_dir: Path,
    splits_path: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    shape: tuple[int, int, int],
):
    overrides = [
        f"data.root_dir={tmp_path}",
        f"data.preprocessing.out_dir={prep_dir}",
        f"data.splits.path={splits_path}",
        f"data.patch_size=[{shape[0]},{shape[1]},{shape[2]}]",
        "data.num_workers=0",
        "data.dataset_type=dataset",
        f"training.checkpoint.dir={checkpoint_dir}",
        f"inference.sliding_window.roi_size=[{shape[0]},{shape[1]},{shape[2]}]",
        "inference.sliding_window.sw_batch_size=1",
        f"analysis.confidence.out_dir={out_dir}",
        "analysis.confidence.split=test",
        "analysis.confidence.max_voxels_per_case=2000",
        "analysis.confidence.dilation_mm=100.0",
        "wandb.mode=disabled",
        "device=cpu",
        "seed=42",
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def test_end_to_end_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    case_ids = ["case_000", "case_001", "case_002"]
    for case_id in case_ids:
        _write_synthetic_case(prep_dir, case_id, VOLUME_SHAPE)
    write_yaml({"train": [], "val": [], "test": case_ids}, splits_path)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _FixedConfidenceModel())
    monkeypatch.setattr(score_confidence_script, "build_model", lambda cfg: _FixedConfidenceModel())

    out_dir = tmp_path / "confidence_out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, VOLUME_SHAPE)

    per_case_df = run_scoring(cfg)

    assert len(per_case_df) == len(case_ids)
    assert per_case_df.index.name == "case_id"
    assert set(per_case_df.index) == set(case_ids)

    expected_columns = set()
    for region in ("ET", "TC", "WT"):
        expected_columns.add(f"auroc_confidence_{region}")
        expected_columns.add(f"auroc_entropy_{region}")
        expected_columns.add(f"auroc_confidence_residual_{region}")
        expected_columns.add(f"n_voxels_{region}")
        expected_columns.add(f"skip_reason_{region}")
    assert set(per_case_df.columns) == expected_columns

    per_case_csv = out_dir / "per_case_confidence.csv"
    summary_csv = out_dir / "summary.csv"
    comparison_csv = out_dir / "confidence_vs_entropy.csv"
    assert per_case_csv.is_file()
    assert summary_csv.is_file()
    assert comparison_csv.is_file()

    per_case_on_disk = pd.read_csv(per_case_csv, index_col="case_id")
    assert len(per_case_on_disk) == len(case_ids)

    summary_on_disk = pd.read_csv(summary_csv, index_col=0)
    assert {"mean", "std", "median", "count", "n_missing"} <= set(summary_on_disk.columns)
    for region in ("ET", "TC", "WT"):
        assert f"auroc_confidence_{region}" in summary_on_disk.index

    comparison_on_disk = pd.read_csv(comparison_csv, index_col="metric")
    assert set(comparison_on_disk.index) == {"auroc_ET", "auroc_TC", "auroc_WT"}
    assert "p_holm" in comparison_on_disk.columns
    assert "verdict" in comparison_on_disk.columns

    # No skips expected: the label was built so every region's fixed predicted-foreground
    # inner cube has a mixed correct/error sample.
    for region in ("ET", "TC", "WT"):
        assert (per_case_df[f"skip_reason_{region}"] == "").all()


def test_end_to_end_raises_before_out_dir_created_when_no_confidence_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prep_dir = tmp_path / "prep"
    splits_path = tmp_path / "splits.yaml"
    _write_synthetic_case(prep_dir, "case_000", VOLUME_SHAPE)
    write_yaml({"train": [], "val": [], "test": ["case_000"]}, splits_path)

    checkpoint_dir = tmp_path / "checkpoints"
    _save_stub_checkpoint(checkpoint_dir, _NoHeadsModel())
    monkeypatch.setattr(score_confidence_script, "build_model", lambda cfg: _NoHeadsModel())

    out_dir = tmp_path / "confidence_out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, checkpoint_dir, out_dir, VOLUME_SHAPE)

    with pytest.raises(ValueError, match="model.heads.confidence"):
        run_scoring(cfg)

    assert not out_dir.exists()
