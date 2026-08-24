"""Tests for scripts/train_qc.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import -- the same pattern as tests/test_calibrate_script.py and
tests/test_conformal_script.py.

Everything here is synthetic, tiny (<=20^3 volumes), and CPU-only. Case ids
are tag-prefixed so a train/held-out pair can safely share one prep_dir.
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
import torch.nn.functional as F

from neurovision.data.qc_pairs import DegradationSpec, generate_pairs
from neurovision.data.transforms import REGION_NAMES

# Real configs/ directory, resolved relative to this file -- so the
# "reachable at the composed path" test composes the PROJECT's actual
# config, not a hand-built stand-in. Same pattern as
# tests/test_conformal_script.py's _CONFIG_DIR.
_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_qc.py"
_spec = importlib.util.spec_from_file_location("train_qc_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
train_qc_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["train_qc_script"] = train_qc_script
_spec.loader.exec_module(train_qc_script)

QCPairsDataset = train_qc_script.QCPairsDataset
CaseGroupedSampler = train_qc_script.CaseGroupedSampler
resolve_dirs = train_qc_script.resolve_dirs
split_case_ids = train_qc_script.split_case_ids
entropy_from_logits = train_qc_script.entropy_from_logits
train_one_epoch = train_qc_script.train_one_epoch
run_training = train_qc_script.run_training
_load_case_arrays = train_qc_script._load_case_arrays
_resize_packed = train_qc_script._resize_packed
_shared_case_ids = train_qc_script._shared_case_ids

SHAPE = (16, 16, 16)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _build_label(shape: tuple[int, int, int] = SHAPE) -> np.ndarray:
    """A fixed nested ET-subset-of-TC-subset-of-WT sphere label, same recipe used elsewhere
    in this project's tests (test_calibrate_script.py, test_conformal_script.py)."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.int64)
    label[dist < min_edge * 0.45] = 2  # ED shell -> completes WT
    label[dist < min_edge * 0.30] = 1  # NCR/NET core -> completes TC
    label[dist < min_edge * 0.15] = 3  # ET, innermost
    return label


def _region_indicator(label: np.ndarray) -> np.ndarray:
    """(3, D, H, W) float32 array, channel order (ET, TC, WT) -- matches REGION_NAMES."""
    et = label == 3
    tc = et | (label == 1)
    wt = tc | (label == 2)
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


def _good_logits(label: np.ndarray, seed: int) -> np.ndarray:
    """Confident, mostly-correct logits: strongly positive inside each region, negative
    outside, plus a little noise. sigmoid(+/-6) is ~0.9975 / ~0.0025."""
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
    image: np.ndarray | None = None,
) -> None:
    """Writes `eval_dir/logits/<case_id>.npy` and `prep_dir/<case_id>/{image.npy,label.npy}`."""
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / f"{case_id}.npy", logits.astype(np.float16))

    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", label.astype(np.uint8))

    if image is None:
        rng = np.random.default_rng(abs(hash(case_id)) % (2**32))
        image = rng.normal(size=(4, *label.shape)).astype(np.float32)
    np.save(case_dir / "image.npy", image.astype(np.float16))


def _write_split(
    tmp_path: Path,
    prep_dir: Path,
    tag: str,
    n_cases: int,
    *,
    shape: tuple[int, int, int] = SHAPE,
    seed_offset: int = 0,
) -> Path:
    """Writes a whole split (several cases) sharing one tag prefix. Returns the eval_dir."""
    eval_dir = tmp_path / f"eval_{tag}"
    for i in range(n_cases):
        case_id = f"{tag}_{i:03d}"
        label = _build_label(shape)
        logits = _good_logits(label, seed_offset + i)
        _write_case(prep_dir, eval_dir, case_id, label, logits)
    return eval_dir


def _make_cfg(
    train_eval_dir: Path | None,
    train_prep_dir: Path,
    heldout_eval_dir: Path | None,
    heldout_prep_dir: Path,
    out_dir: Path,
    *,
    target_shape: tuple[int, int, int] = (8, 8, 8),
    regions: tuple[str, ...] = ("ET", "TC", "WT"),
    modality_index: int = 1,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 1.0e-3,
    num_workers: int = 0,
    max_cases: int | None = None,
    min_component_size: int = 0,
    seed: int = 0,
    val_frac: float = 0.2,
):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "seed": seed,
            "device": "cpu",
            "analysis": {
                "qc": {
                    "train_eval_dir": str(train_eval_dir) if train_eval_dir is not None else None,
                    "train_prep_dir": str(train_prep_dir),
                    "heldout_eval_dir": (
                        str(heldout_eval_dir) if heldout_eval_dir is not None else None
                    ),
                    "heldout_prep_dir": str(heldout_prep_dir),
                    "val_frac": val_frac,
                    "out_dir": str(out_dir),
                    "modality_index": modality_index,
                    "target_shape": list(target_shape),
                    "regions": list(regions),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_workers": num_workers,
                    "max_cases": max_cases,
                }
            },
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
            "model": {
                "name": "segqc",
                "in_channels": 3,
                "widths": [4, 8],
                "num_groups": 2,
                "dropout": 0.0,
            },
        }
    )


# ---------------------------------------------------------------------------
# 1. Length
# ---------------------------------------------------------------------------


def test_dataset_length_matches_cases_times_specs_times_regions(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "train", 3)
    cfg = _make_cfg(eval_dir, prep_dir, eval_dir, prep_dir, tmp_path / "out")

    specs = [DegradationSpec("identity", 0.0), DegradationSpec("erode", 1.0)]
    dataset = QCPairsDataset(cfg, eval_dir, prep_dir, specs=specs)

    assert len(dataset) == 3 * len(specs) * 3  # 3 cases x 2 specs x 3 regions


# ---------------------------------------------------------------------------
# 2. Item shape and range
# ---------------------------------------------------------------------------


def test_dataset_item_shape_and_range(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "train", 1)
    cfg = _make_cfg(
        eval_dir, prep_dir, eval_dir, prep_dir, tmp_path / "out", target_shape=(8, 8, 8)
    )
    dataset = QCPairsDataset(cfg, eval_dir, prep_dir, specs=[DegradationSpec("identity", 0.0)])

    sample, target = dataset[0]

    assert sample.shape == (3, 8, 8, 8)
    assert sample.dtype == torch.float32
    assert target.dtype == torch.float32
    assert target.ndim == 0
    assert 0.0 <= float(target.item()) <= 1.0


# ---------------------------------------------------------------------------
# 3. Nearest for the mask, trilinear for image/entropy
# ---------------------------------------------------------------------------


def test_mask_channel_stays_binary_after_resize(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "train", 1, shape=(20, 20, 20))
    cfg = _make_cfg(
        eval_dir, prep_dir, eval_dir, prep_dir, tmp_path / "out", target_shape=(8, 8, 8)
    )
    dataset = QCPairsDataset(cfg, eval_dir, prep_dir, specs=[DegradationSpec("identity", 0.0)])

    sample, _ = dataset[0]
    image_channel, mask_channel, entropy_channel = sample[0], sample[1], sample[2]

    mask_values = torch.unique(mask_channel)
    assert set(mask_values.tolist()) <= {0.0, 1.0}

    # Continuous channels must NOT collapse to a two-value set the way a
    # (bug-)nearest-resized mask would.
    assert torch.unique(image_channel).numel() > 2
    assert torch.unique(entropy_channel).numel() > 2


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_same_index_gives_same_sample(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "train", 2)
    cfg = _make_cfg(eval_dir, prep_dir, eval_dir, prep_dir, tmp_path / "out")
    # speckle/shift/drop_component are stochastic -- a real test of the
    # per-index generator, unlike "identity" which is deterministic by
    # construction regardless of seeding.
    dataset = QCPairsDataset(cfg, eval_dir, prep_dir, specs=[DegradationSpec("speckle", 0.5)])

    index = len(dataset) // 2
    sample_a, target_a = dataset[index]
    sample_b, target_b = dataset[index]

    assert torch.equal(sample_a, sample_b)
    assert torch.equal(target_a, target_b)


# ---------------------------------------------------------------------------
# 5. fp16-saturation entropy
# ---------------------------------------------------------------------------


def test_entropy_is_finite_under_fp16_saturation() -> None:
    # fp16's own range comfortably holds +/-40 (max ~65504), which is exactly
    # the kind of saturated, highly-confident logit a trained segmentation
    # net produces -- the value that turned a clamp-based entropy into NaN
    # (see entropy_from_logits' docstring).
    logits = torch.tensor([40.0, -40.0, 0.0, 12.3, -0.001], dtype=torch.float32)
    entropy = entropy_from_logits(logits)

    assert torch.isfinite(entropy).all()
    # A saturated logit's entropy should be (numerically) zero -- a certain
    # prediction carries no uncertainty.
    assert entropy[0].item() == pytest.approx(0.0, abs=1e-6)
    assert entropy[1].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Target Dice computed at full resolution
# ---------------------------------------------------------------------------


def test_target_dice_is_computed_at_full_resolution(tmp_path: Path) -> None:
    shape = (20, 20, 20)
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    case_id = "case_000"
    label = _build_label(shape)
    logits = _good_logits(label, seed=1)
    _write_case(prep_dir, eval_dir, case_id, label, logits)

    target_shape = (8, 8, 8)
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        eval_dir,
        prep_dir,
        tmp_path / "out",
        target_shape=target_shape,
        regions=("WT",),
    )
    spec = DegradationSpec("erode", 3.0)
    dataset = QCPairsDataset(cfg, eval_dir, prep_dir, specs=[spec])

    index = 0  # only spec, only region (WT)
    sample, target = dataset[index]

    # Recompute the SAME pair independently, the same way the dataset does
    # internally, to get the true full-resolution Dice.
    arrays = _load_case_arrays(cfg, eval_dir, prep_dir, case_id)
    generator = np.random.default_rng([int(cfg.seed), index])
    pairs = generate_pairs(
        arrays.pred_mask, arrays.label, generator=generator, specs=[spec], per_region=True
    )
    region_channel = REGION_NAMES.index("WT")
    expected_pair = next(p for p in pairs if p.region_index == region_channel)
    expected_dice = expected_pair.dice[region_channel]

    assert float(target.item()) == pytest.approx(expected_dice, abs=1e-6)

    # And prove the full-res number is not simply reproducible by resizing
    # first: recomputing Dice from the ALREADY-DOWNSAMPLED mask/label gives a
    # measurably different answer, so this test could not have passed by
    # accident if the implementation resized before scoring.
    label_wt = torch.from_numpy(_region_indicator(label)[region_channel])
    label_wt_resized = F.interpolate(label_wt[None, None], size=target_shape, mode="nearest")[0, 0]
    resized_mask = sample[1]
    intersection = (resized_mask * label_wt_resized).sum()
    denom = resized_mask.sum() + label_wt_resized.sum()
    naive_low_res_dice = float((2.0 * intersection / denom).item()) if denom > 0 else 1.0

    assert naive_low_res_dice != pytest.approx(expected_dice, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. Held-out separation guard
# ---------------------------------------------------------------------------


def test_heldout_dir_is_never_used_for_training(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    train_eval_dir = _write_split(tmp_path, prep_dir, "train", 2)
    heldout_eval_dir = _write_split(tmp_path, prep_dir, "held", 2, seed_offset=100)

    cfg = _make_cfg(train_eval_dir, prep_dir, heldout_eval_dir, prep_dir, tmp_path / "out")
    dataset = QCPairsDataset(
        cfg, train_eval_dir, prep_dir, specs=[DegradationSpec("identity", 0.0)]
    )
    assert all(case_id.startswith("train_") for case_id in dataset._case_ids)
    assert not any(case_id.startswith("held_") for case_id in dataset._case_ids)

    # Pointing both configs at the SAME directory must raise.
    cfg_same = _make_cfg(train_eval_dir, prep_dir, train_eval_dir, prep_dir, tmp_path / "out2")
    with pytest.raises(ValueError, match="same"):
        resolve_dirs(cfg_same)

    # The "a/./b" form must be caught too -- Path.resolve() normalizes it to
    # the identical real directory.
    dotted = train_eval_dir.parent / "." / train_eval_dir.name
    cfg_dotted = _make_cfg(train_eval_dir, prep_dir, dotted, prep_dir, tmp_path / "out3")
    with pytest.raises(ValueError, match="same"):
        resolve_dirs(cfg_dotted)


# ---------------------------------------------------------------------------
# 8. Config reachable at the composed path
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_at_the_composed_path() -> None:
    """The REAL project config, composed through Hydra, must expose the QC
    block at `cfg.analysis.qc` -- the exact path `scripts/train_qc.py` reads
    (`resolve_dirs`, `QCPairsDataset.__init__`, `run_training`).

    Same regression shape as
    tests/test_conformal_script.py::test_conformal_config_block_is_reachable_at_the_composed_path:
    a hand-built OmegaConf fixture that puts "qc" at the wrong nesting level
    would pass every other test in this file while the real composed config
    never produces that shape.
    """
    overrides = ["data.root_dir=/unused/for/this/test"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    assert "analysis" in cfg
    assert "qc" in cfg.analysis
    assert "qc" not in cfg  # NOT cfg.qc

    qc_cfg = cfg.analysis.qc
    expected_keys = {
        "train_eval_dir",
        "train_prep_dir",
        "heldout_eval_dir",
        "heldout_prep_dir",
        "val_frac",
        "out_dir",
        "modality_index",
        "target_shape",
        "regions",
        "epochs",
        "batch_size",
        "lr",
        "num_workers",
        "max_cases",
    }
    assert expected_keys <= set(qc_cfg.keys())


# ---------------------------------------------------------------------------
# 9. One real training step
# ---------------------------------------------------------------------------


def test_one_training_step_runs_and_loss_is_finite(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from neurovision.models.qc import build_segqc

    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "train", 2, shape=(12, 12, 12))
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        eval_dir,
        prep_dir,
        tmp_path / "out",
        target_shape=(6, 6, 6),
        regions=("WT",),
        batch_size=2,
    )
    dataset = QCPairsDataset(cfg, eval_dir, prep_dir, specs=[DegradationSpec("identity", 0.0)])
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = build_segqc(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
    before = {name: param.detach().clone() for name, param in model.state_dict().items()}

    mean_loss, global_step = train_one_epoch(
        model, loader, optimizer, torch.device("cpu"), global_step=0
    )

    assert math_isfinite(mean_loss)
    assert global_step == len(loader)

    after = model.state_dict()
    changed = any(not torch.equal(before[name], after[name]) for name in before)
    assert changed


def math_isfinite(value: float) -> bool:
    import math

    return math.isfinite(value)


# ---------------------------------------------------------------------------
# 10. QCPairsDataset's fast path (generate_one_pair) must match the
# generate_pairs path exactly, on both the packed array and the target.
# ---------------------------------------------------------------------------


def test_dataset_fast_path_matches_generate_pairs_path_exactly(tmp_path: Path) -> None:
    """`__getitem__` uses `generate_one_pair` for speed (skips scoring 3 of the
    4 pairs `generate_pairs` would compute). This proves that optimisation
    changed nothing observable: rebuilding the same sample by hand through
    `generate_pairs` -- the ORIGINAL, unoptimised path -- with a generator in
    the same starting state must give a BIT-FOR-BIT identical packed array
    and target. Uses a STOCHASTIC spec (shift) and a region with earlier
    regions in the per-region loop (WT, region_index=2) specifically because
    those are exactly the conditions under which a naive "skip the earlier
    calls" optimisation would have desynced the generator and silently
    diverged.
    """
    shape = (16, 16, 16)
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    case_id = "case_000"
    label = _build_label(shape)
    logits = _good_logits(label, seed=7)
    _write_case(prep_dir, eval_dir, case_id, label, logits)

    target_shape = (6, 6, 6)
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        eval_dir,
        prep_dir,
        tmp_path / "out",
        target_shape=target_shape,
        regions=("WT",),
    )
    spec = DegradationSpec("shift", 3.0)
    dataset = QCPairsDataset(cfg, eval_dir, prep_dir, specs=[spec])

    index = 0  # only spec, only region (WT, region_index=2 -- 2 earlier regions)
    fast_sample, fast_target = dataset[index]

    # The ORIGINAL, unoptimised path: generate_pairs with a generator in the
    # SAME starting state (fresh, same seed derivation).
    arrays = _load_case_arrays(cfg, eval_dir, prep_dir, case_id)
    region_channel = REGION_NAMES.index("WT")
    slow_generator = np.random.default_rng([int(cfg.seed), index])
    slow_pairs = generate_pairs(
        arrays.pred_mask, arrays.label, generator=slow_generator, specs=[spec], per_region=True
    )
    slow_pair = next(p for p in slow_pairs if p.region_index == region_channel)

    image_channel = torch.from_numpy(arrays.image_modality)
    mask_channel = torch.from_numpy(slow_pair.mask[region_channel].astype(np.float32))
    entropy_channel = torch.from_numpy(arrays.entropy[region_channel])
    packed = torch.stack([image_channel, mask_channel, entropy_channel], dim=0)
    slow_sample = _resize_packed(packed, target_shape)
    slow_target = torch.tensor(float(slow_pair.dice[region_channel]), dtype=torch.float32)

    assert torch.equal(fast_sample, slow_sample)
    assert torch.equal(fast_target, slow_target)


# ---------------------------------------------------------------------------
# 11. CaseGroupedSampler
# ---------------------------------------------------------------------------


def test_sampler_keeps_each_cases_pairs_contiguous() -> None:
    sampler = CaseGroupedSampler(num_cases=6, per_case=4, seed=0)
    sampler.set_epoch(0)

    indices = list(iter(sampler))
    assert len(indices) == len(sampler) == 24

    # Every run of 4 consecutive indices must belong to exactly one case
    # (floor(index / per_case) constant within the run) -- and the full
    # index set must be exactly {0, ..., 23}, so no index is skipped or
    # repeated across the shuffled case order.
    assert sorted(indices) == list(range(24))
    for chunk_start in range(0, 24, 4):
        chunk = indices[chunk_start : chunk_start + 4]
        case_positions = {i // 4 for i in chunk}
        assert len(case_positions) == 1


def test_sampler_permutes_case_order_across_epochs() -> None:
    sampler = CaseGroupedSampler(num_cases=8, per_case=3, seed=0)

    sampler.set_epoch(0)
    order_epoch_0 = [i // 3 for i in iter(sampler)][::3]

    sampler.set_epoch(1)
    order_epoch_1 = [i // 3 for i in iter(sampler)][::3]

    assert order_epoch_0 != order_epoch_1


def test_sampler_same_epoch_gives_same_order() -> None:
    sampler_a = CaseGroupedSampler(num_cases=8, per_case=3, seed=0)
    sampler_a.set_epoch(3)
    sampler_b = CaseGroupedSampler(num_cases=8, per_case=3, seed=0)
    sampler_b.set_epoch(3)

    assert list(iter(sampler_a)) == list(iter(sampler_b))


def test_sampler_epoch_order_is_a_pure_function_of_epoch_not_call_history() -> None:
    """This is the resume-safety guarantee: jumping straight to `set_epoch(2)`
    (as a resumed run does) must give the SAME order as a fresh run that
    called `set_epoch(0)`, then `set_epoch(1)`, then `set_epoch(2)` in
    sequence -- proving the order depends only on the epoch NUMBER, never on
    how many times `set_epoch`/`__iter__` were called before it."""
    fresh_run = CaseGroupedSampler(num_cases=5, per_case=2, seed=7)
    for epoch in range(3):
        fresh_run.set_epoch(epoch)
        list(iter(fresh_run))  # simulate actually training that epoch
    fresh_run_epoch_2_order = list(iter(fresh_run))

    resumed_run = CaseGroupedSampler(num_cases=5, per_case=2, seed=7)
    resumed_run.set_epoch(2)  # jumps straight there, as a resumed run does
    resumed_run_epoch_2_order = list(iter(resumed_run))

    assert fresh_run_epoch_2_order == resumed_run_epoch_2_order


def test_dataloader_rejects_shuffle_true_with_sampler() -> None:
    """Documents the mutual exclusivity run_training relies on: shuffle=False
    is REQUIRED whenever a sampler is passed."""
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.zeros(6, 1))
    sampler = CaseGroupedSampler(num_cases=3, per_case=2, seed=0)
    sampler.set_epoch(0)

    with pytest.raises(ValueError):
        DataLoader(dataset, batch_size=2, shuffle=True, sampler=sampler)


# ---------------------------------------------------------------------------
# 12. split_case_ids -- the case-disjoint fit/select split model selection
# now reads from, instead of a second directory.
# ---------------------------------------------------------------------------


def test_split_case_ids_is_deterministic_and_disjoint() -> None:
    case_ids = [f"case_{i:03d}" for i in range(10)]

    fit_a, select_a = split_case_ids(case_ids, val_frac=0.3, seed=42)
    fit_b, select_b = split_case_ids(case_ids, val_frac=0.3, seed=42)

    assert fit_a == fit_b
    assert select_a == select_b

    assert set(fit_a).isdisjoint(select_a)
    assert set(fit_a) | set(select_a) == set(case_ids)
    assert len(fit_a) > 0
    assert len(select_a) > 0

    # Both sides must be sorted -- see split_case_ids' docstring.
    assert fit_a == sorted(fit_a)
    assert select_a == sorted(select_a)


def test_split_case_ids_differs_by_seed() -> None:
    # 30 ids makes an accidental collision between two independent
    # permutations of the selection side implausible.
    case_ids = [f"case_{i:03d}" for i in range(30)]

    _, select_seed_0 = split_case_ids(case_ids, val_frac=0.3, seed=0)
    _, select_seed_1 = split_case_ids(case_ids, val_frac=0.3, seed=1)

    assert select_seed_0 != select_seed_1


@pytest.mark.parametrize("bad_val_frac", [0.0, 1.0, -0.1])
def test_split_case_ids_rejects_bad_val_frac(bad_val_frac: float) -> None:
    case_ids = [f"case_{i:03d}" for i in range(5)]
    with pytest.raises(ValueError):
        split_case_ids(case_ids, val_frac=bad_val_frac, seed=0)


def test_split_case_ids_rejects_single_case() -> None:
    with pytest.raises(ValueError):
        split_case_ids(["only_case"], val_frac=0.2, seed=0)


# ---------------------------------------------------------------------------
# 13. QCPairsDataset's explicit case_ids path
# ---------------------------------------------------------------------------


def test_qc_pairs_dataset_honours_explicit_case_ids(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "train", 5)
    cfg = _make_cfg(eval_dir, prep_dir, eval_dir, prep_dir, tmp_path / "out")

    # A strict subset of the 5 written cases -- max_cases is left at its
    # default (None) in cfg, proving it is NOT applied on this path either.
    chosen_ids = ["train_001", "train_003"]
    specs = [DegradationSpec("identity", 0.0), DegradationSpec("erode", 1.0)]
    dataset = QCPairsDataset(cfg, eval_dir, prep_dir, specs=specs, case_ids=chosen_ids)

    assert dataset.num_cases == len(chosen_ids)
    assert len(dataset) == len(chosen_ids) * len(specs) * 3  # 3 regions (ET, TC, WT)


def test_qc_pairs_dataset_rejects_unknown_case_id(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "train", 3)
    cfg = _make_cfg(eval_dir, prep_dir, eval_dir, prep_dir, tmp_path / "out")

    with pytest.raises(ValueError, match="train_999"):
        QCPairsDataset(cfg, eval_dir, prep_dir, case_ids=["train_000", "train_999"])


# ---------------------------------------------------------------------------
# 14. resolve_dirs -- heldout_eval_dir is now optional
# ---------------------------------------------------------------------------


def test_resolve_dirs_allows_null_heldout(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "train", 2)
    cfg = _make_cfg(eval_dir, prep_dir, None, prep_dir, tmp_path / "out")

    train_eval_dir, train_prep_dir, heldout_eval_dir, heldout_prep_dir = resolve_dirs(cfg)

    assert train_eval_dir == eval_dir
    assert heldout_eval_dir is None


# ---------------------------------------------------------------------------
# 15. run_training -- model selection reads the SELECT split, never the
# optional heldout_eval_dir. This is the point of the whole change.
# ---------------------------------------------------------------------------


def _write_split_with_fixed_image_rng(
    tmp_path: Path, prep_dir: Path, tag: str, n_cases: int, *, seed_offset: int = 0
) -> Path:
    """Like `_write_split`, but seeds the synthetic image from an explicit int,
    never from `hash(case_id)` (`_write_case`'s own default).

    Python randomizes `str.__hash__` per PROCESS by default (a security
    feature, `PYTHONHASHSEED`), so `_write_case`'s default `image` -- seeded
    from `hash(case_id)` -- silently differs across separate test-process
    launches even though every OTHER piece of this project's randomness is
    seeded explicitly. That is invisible to every other test in this file
    (none of them compares exact epoch-to-epoch metric orderings), but
    `test_run_training_selects_on_the_select_split_not_the_heldout_dir`
    below reads exactly that ordering, so it needs bit-identical synthetic
    data on every run, not just within one process.
    """
    eval_dir = tmp_path / f"eval_{tag}"
    for i in range(n_cases):
        case_id = f"{tag}_{i:03d}"
        label = _build_label(SHAPE)
        logits = _good_logits(label, seed_offset + i)
        image = (
            np.random.default_rng(seed_offset + i + 9000)
            .normal(size=(4, *SHAPE))
            .astype(np.float32)
        )
        _write_case(prep_dir, eval_dir, case_id, label, logits, image=image)
    return eval_dir


def test_run_training_selects_on_the_select_split_not_the_heldout_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neurovision.utils.seed import set_seed

    # A short specs list -- monkeypatched onto the exact name
    # QCPairsDataset's default resolves at call time (train_qc_script's
    # imported DEFAULT_SPECS), since run_training has no specs override of
    # its own and always builds its datasets with specs=None.
    short_specs = [
        DegradationSpec("erode", 2.0),
        DegradationSpec("dilate", 2.0),
        DegradationSpec("speckle", 0.4),
    ]
    monkeypatch.setattr(train_qc_script, "DEFAULT_SPECS", short_specs)

    prep_dir = tmp_path / "prep"
    # 6 cases in train_eval_dir: split_case_ids(val_frac=0.34, ...) gives 4
    # fit / 2 select regardless of seed (round(6 * 0.34) == 2). 4 cases in
    # heldout_eval_dir, a genuinely different split. This exact (cases,
    # cfg.seed=22, specs) combination was found by search -- then verified
    # stable across 6+ fresh process launches, including with
    # PYTHONHASHSEED randomized -- to make the select-side and heldout-side
    # argmax epochs land on DIFFERENT epochs with a comfortable margin
    # (>0.1 Spearman between the best epoch and the runner-up on each side),
    # so this assertion cannot flip on ordinary floating-point jitter. See
    # `test_run_training_without_heldout_dir_writes_no_heldout_columns`,
    # which reuses the ordinary hash-seeded `_write_split` -- that test
    # never compares WHICH epoch won, only that the right columns exist.
    train_eval_dir = _write_split_with_fixed_image_rng(tmp_path, prep_dir, "train", 6)
    heldout_eval_dir = _write_split_with_fixed_image_rng(
        tmp_path, prep_dir, "held", 4, seed_offset=500
    )

    cfg = _make_cfg(
        train_eval_dir,
        prep_dir,
        heldout_eval_dir,
        prep_dir,
        tmp_path / "out",
        regions=("WT",),
        epochs=4,
        batch_size=4,
        lr=1.0e-2,
        val_frac=0.34,
        seed=22,
    )

    # Matches main()'s own convention: seed everything (including the
    # model's random initialisation, which is NOT reached by any of this
    # project's seeded-generator plumbing) before calling run_training.
    set_seed(22)
    result = run_training(cfg)

    history = pd.read_csv(result["history_csv"])
    assert {"select_mae", "select_spearman", "heldout_mae", "heldout_spearman"} <= set(
        history.columns
    )

    best_select_epoch = int(history.loc[history["select_spearman"].idxmax(), "epoch"])
    best_heldout_epoch = int(history.loc[history["heldout_spearman"].idxmax(), "epoch"])
    # The fixture is deliberately constructed so these two differ -- proving
    # is_best really reads the select column and not the heldout one.
    assert best_select_epoch != best_heldout_epoch

    best_path = result["checkpoint_dir"] / "best.pt"
    assert best_path.is_file()
    payload = torch.load(best_path, weights_only=True)
    assert payload["epoch"] == best_select_epoch
    assert payload["best_metric"] == pytest.approx(history["select_spearman"].max())


def test_run_training_without_heldout_dir_writes_no_heldout_columns(tmp_path: Path) -> None:
    from neurovision.utils.seed import set_seed

    prep_dir = tmp_path / "prep"
    train_eval_dir = _write_split(tmp_path, prep_dir, "train", 4, shape=(12, 12, 12))
    cfg = _make_cfg(
        train_eval_dir,
        prep_dir,
        None,
        prep_dir,
        tmp_path / "out",
        regions=("WT",),
        target_shape=(6, 6, 6),
        epochs=2,
        batch_size=2,
        val_frac=0.5,
    )

    set_seed(0)
    result = run_training(cfg)

    history = pd.read_csv(result["history_csv"])
    assert {"epoch", "train_loss", "select_mae", "select_spearman"} <= set(history.columns)
    assert not any(col.startswith("heldout_") for col in history.columns)

    best_path = result["checkpoint_dir"] / "best.pt"
    assert best_path.is_file()
