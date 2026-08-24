"""Tests for `neurovision.analysis.qc_inference`.

Everything here is synthetic, tiny, and CPU-only -- no real BraTS data, no
GPU. `test_matches_train_qc_script_exactly` is the equivalence guard: this
module is a behaviour-preserving extraction from `scripts/train_qc.py`, so
its functions must return BIT-FOR-BIT identical results to the script's
private originals on the same input. That script is loaded via
`importlib.util.spec_from_file_location`, the same pattern
`tests/test_train_qc.py` already uses (it lives under `scripts/`, which is
not an importable package).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from neurovision.analysis import qc_inference

# ---------------------------------------------------------------------------
# Load scripts/train_qc.py the way tests/test_train_qc.py does, so this
# file's equivalence test can reach the script's private originals.
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_qc.py"
_spec = importlib.util.spec_from_file_location("train_qc_script_for_qc_inference", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
train_qc_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["train_qc_script_for_qc_inference"] = train_qc_script
_spec.loader.exec_module(train_qc_script)


SHAPE = (16, 16, 16)


# ---------------------------------------------------------------------------
# Synthetic data helpers -- same recipe used by tests/test_train_qc.py
# ---------------------------------------------------------------------------


def _build_label(shape: tuple[int, int, int] = SHAPE) -> np.ndarray:
    """A fixed nested ET-subset-of-TC-subset-of-WT sphere label."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.int64)
    label[dist < min_edge * 0.45] = 2
    label[dist < min_edge * 0.30] = 1
    label[dist < min_edge * 0.15] = 3
    return label


def _region_indicator(label: np.ndarray) -> np.ndarray:
    """(3, D, H, W) float32 array, channel order (ET, TC, WT)."""
    et = label == 3
    tc = et | (label == 1)
    wt = tc | (label == 2)
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


def _good_logits(label: np.ndarray, seed: int) -> np.ndarray:
    """Confident, mostly-correct logits: strongly positive inside each region, negative
    outside, plus a little noise."""
    rng = np.random.default_rng(seed)
    region = _region_indicator(label)
    logits = region * 12.0 - 6.0 + rng.normal(scale=0.3, size=region.shape)
    return logits.astype(np.float32)


def _write_case(
    prep_dir: Path,
    eval_dir: Path,
    case_id: str,
    label: np.ndarray,
    logits: np.ndarray,
) -> None:
    """Writes `eval_dir/logits/<case_id>.npy` and `prep_dir/<case_id>/{image.npy,label.npy}`,
    matching the layout `neurovision.data.preprocessing` actually writes (image.npy is
    `(4, D, H, W)` fp16, label.npy is uint8)."""
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / f"{case_id}.npy", logits.astype(np.float16))

    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", label.astype(np.uint8))

    rng = np.random.default_rng(abs(hash(case_id)) % (2**32))
    image = rng.normal(size=(4, *label.shape)).astype(np.float16)
    np.save(case_dir / "image.npy", image)


def _make_cfg(modality_index: int = 1, min_component_size: int = 0):
    """Minimal config exposing exactly the two nesting paths `load_case_arrays` reads:
    `cfg.inference.postprocess` and `cfg.analysis.qc.modality_index`."""
    return OmegaConf.create(
        {
            "inference": {
                "postprocess": {
                    "threshold": 0.5,
                    "enforce_nesting": True,
                    "min_component_size": min_component_size,
                    "connectivity": 1,
                    "keep_largest_only": False,
                    "et_min_volume": 0,
                }
            },
            "analysis": {"qc": {"modality_index": modality_index}},
        }
    )


# ---------------------------------------------------------------------------
# 1. entropy_from_logits matches a hand-computed value
# ---------------------------------------------------------------------------


def test_entropy_from_logits_matches_hand_computed() -> None:
    logits = torch.tensor([0.0, 1.0, -2.0, 3.5])
    entropy = qc_inference.entropy_from_logits(logits)

    p = torch.sigmoid(logits)
    expected = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    assert torch.allclose(entropy, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Finite entropy under fp16 saturation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_entropy_is_finite_under_fp16_saturation(dtype: torch.dtype) -> None:
    logits = torch.tensor([30.0, -30.0, 0.0], dtype=dtype)
    entropy = qc_inference.entropy_from_logits(logits.float())

    assert torch.isfinite(entropy).all()
    assert entropy[0].item() == pytest.approx(0.0, abs=1e-3)
    assert entropy[1].item() == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 3. resize_packed shape/dtype
# ---------------------------------------------------------------------------


def test_resize_packed_shapes_and_dtype() -> None:
    packed = torch.rand(3, 20, 24, 22)
    resized = qc_inference.resize_packed(packed, (8, 8, 8))

    assert resized.shape == (3, 8, 8, 8)
    assert resized.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. NEAREST for the mask channel, TRILINEAR for the others
# ---------------------------------------------------------------------------


def test_resize_packed_keeps_mask_channel_binary() -> None:
    torch.manual_seed(0)
    image = torch.rand(20, 24, 22)  # continuous
    mask = (torch.rand(20, 24, 22) > 0.5).float()  # binary
    entropy = torch.rand(20, 24, 22)  # continuous

    packed = torch.stack([image, mask, entropy], dim=0)
    resized = qc_inference.resize_packed(packed, (8, 8, 8))

    mask_values = set(torch.unique(resized[1]).tolist())
    assert mask_values <= {0.0, 1.0}

    # At least one continuous channel must actually take an intermediate
    # value -- otherwise this test cannot tell trilinear from nearest.
    image_intermediate = ((resized[0] > 1e-6) & (resized[0] < 1 - 1e-6)).any()
    entropy_intermediate = ((resized[2] > 1e-6) & (resized[2] < 1 - 1e-6)).any()
    assert bool(image_intermediate) or bool(entropy_intermediate)


# ---------------------------------------------------------------------------
# 5. pack_sample channel order
# ---------------------------------------------------------------------------


def test_pack_sample_channel_order() -> None:
    shape = (6, 6, 6)
    image = np.full(shape, 1.0, dtype=np.float32)
    entropy_full = np.full((3, *shape), 3.0, dtype=np.float32)

    mask = np.zeros((3, *shape), dtype=np.uint8)
    mask[1] = 1  # region_channel=1 (TC) is all-foreground

    arrays = qc_inference.CaseArrays(
        pred_mask=mask,
        label=np.zeros(shape, dtype=np.int64),
        image_modality=image,
        entropy=entropy_full,
    )

    sample = qc_inference.pack_sample(arrays, mask, region_channel=1, target_shape=(4, 4, 4))

    assert sample.shape == (3, 4, 4, 4)
    assert torch.allclose(sample[0], torch.full((4, 4, 4), 1.0))
    assert torch.allclose(sample[1], torch.full((4, 4, 4), 1.0))
    assert torch.allclose(sample[2], torch.full((4, 4, 4), 3.0))


# ---------------------------------------------------------------------------
# 6. pack_sample handles an all-background mask
# ---------------------------------------------------------------------------


def test_pack_sample_all_background_mask() -> None:
    shape = (6, 6, 6)
    arrays = qc_inference.CaseArrays(
        pred_mask=np.zeros((3, *shape), dtype=np.uint8),
        label=np.zeros(shape, dtype=np.int64),
        image_modality=np.zeros(shape, dtype=np.float32),
        entropy=np.zeros((3, *shape), dtype=np.float32),
    )
    mask = np.zeros((3, *shape), dtype=np.uint8)

    sample = qc_inference.pack_sample(arrays, mask, region_channel=0, target_shape=(4, 4, 4))

    assert sample.shape == (3, 4, 4, 4)
    assert torch.equal(sample[1], torch.zeros(4, 4, 4))


# ---------------------------------------------------------------------------
# 7. load_case_arrays roundtrip
# ---------------------------------------------------------------------------


def test_load_case_arrays_roundtrip(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    case_id = "case_000"
    label = _build_label()
    logits = _good_logits(label, seed=1)
    _write_case(prep_dir, eval_dir, case_id, label, logits)

    cfg = _make_cfg(modality_index=1)
    arrays = qc_inference.load_case_arrays(cfg, eval_dir, prep_dir, case_id)

    assert arrays.pred_mask.shape == (3, *SHAPE)
    assert arrays.pred_mask.dtype == np.uint8
    assert arrays.label.shape == SHAPE
    assert arrays.image_modality.shape == SHAPE
    assert arrays.image_modality.dtype == np.float32
    assert arrays.entropy.shape == (3, *SHAPE)
    assert arrays.entropy.dtype == np.float32


# ---------------------------------------------------------------------------
# 8. Missing-file errors
# ---------------------------------------------------------------------------


def test_load_case_arrays_missing_logits_raises(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    case_dir = prep_dir / "case_000"
    case_dir.mkdir(parents=True)
    np.save(case_dir / "label.npy", np.zeros(SHAPE, dtype=np.uint8))
    np.save(case_dir / "image.npy", np.zeros((4, *SHAPE), dtype=np.float16))
    (eval_dir / "logits").mkdir(parents=True)

    cfg = _make_cfg()
    with pytest.raises(FileNotFoundError, match="logits"):
        qc_inference.load_case_arrays(cfg, eval_dir, prep_dir, "case_000")


def test_load_case_arrays_missing_case_dir_raises(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    label = _build_label()
    logits = _good_logits(label, seed=2)
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True)
    np.save(logits_dir / "case_000.npy", logits.astype(np.float16))
    prep_dir.mkdir(parents=True)

    cfg = _make_cfg()
    with pytest.raises(FileNotFoundError):
        qc_inference.load_case_arrays(cfg, eval_dir, prep_dir, "case_000")


# ---------------------------------------------------------------------------
# 9. Equivalence guard -- must match scripts/train_qc.py's private originals
# exactly (torch.equal / np.array_equal, not allclose).
# ---------------------------------------------------------------------------


def test_matches_train_qc_script_exactly(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    case_id = "case_000"
    label = _build_label()
    logits = _good_logits(label, seed=3)
    _write_case(prep_dir, eval_dir, case_id, label, logits)

    cfg = _make_cfg(modality_index=2)

    # entropy_from_logits
    test_logits = torch.tensor([30.0, -30.0, 0.0, 1.5, -4.2])
    lib_entropy = qc_inference.entropy_from_logits(test_logits)
    script_entropy = train_qc_script.entropy_from_logits(test_logits)
    assert torch.equal(lib_entropy, script_entropy)

    # resize_packed / _resize_packed
    packed = torch.rand(3, 20, 24, 22)
    lib_resized = qc_inference.resize_packed(packed, (8, 8, 8))
    script_resized = train_qc_script._resize_packed(packed, (8, 8, 8))
    assert torch.equal(lib_resized, script_resized)

    # load_case_arrays / _load_case_arrays
    lib_arrays = qc_inference.load_case_arrays(cfg, eval_dir, prep_dir, case_id)
    script_arrays = train_qc_script._load_case_arrays(cfg, eval_dir, prep_dir, case_id)

    assert np.array_equal(lib_arrays.pred_mask, script_arrays.pred_mask)
    assert np.array_equal(lib_arrays.label, script_arrays.label)
    assert np.array_equal(lib_arrays.image_modality, script_arrays.image_modality)
    assert np.array_equal(lib_arrays.entropy, script_arrays.entropy)
    assert lib_arrays.pred_mask.dtype == script_arrays.pred_mask.dtype
    assert lib_arrays.entropy.dtype == script_arrays.entropy.dtype
