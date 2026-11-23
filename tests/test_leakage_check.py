"""Tests for scripts/leakage_check.py (synthetic volumes only)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_SPEC = importlib.util.spec_from_file_location(
    "leakage_check",
    Path(__file__).resolve().parent.parent / "scripts" / "leakage_check.py",
)
lc = importlib.util.module_from_spec(_SPEC)
sys.modules["leakage_check"] = lc
_SPEC.loader.exec_module(lc)

SHAPE = (16, 16, 8)
GRID = (4, 4, 2)
FULL_BBOX = [[0, 16], [0, 16], [0, 8]]


def write_case(root, cid, image, label, bbox=FULL_BBOX):
    d = root / cid
    d.mkdir(parents=True)
    np.save(d / "image.npy", image.astype(np.float16))
    np.save(d / "label.npy", label.astype(np.uint8))
    meta = {
        "original_shape": list(SHAPE),
        "bbox": bbox,
        "cropped_shape": [hi - lo for lo, hi in bbox],
    }
    (d / "meta.json").write_text(json.dumps(meta))


def tumour(box):
    lab = np.zeros(SHAPE, dtype=np.uint8)
    lab[box] = 1
    return lab


def write_splits(path, train, val, test):
    path.write_text(yaml.safe_dump({"train": train, "val": val, "test": test}))
    return path


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    prep = tmp_path / "prep"
    base = rng.normal(size=(4, *SHAPE)).astype(np.float32)
    lab = tumour((slice(4, 9), slice(4, 9), slice(2, 5)))
    write_case(prep, "a", base, lab)  # train
    write_case(prep, "a_copy", base, lab)  # test, byte-identical
    noisy = base + 0.01 * rng.normal(size=base.shape).astype(np.float32)
    write_case(prep, "a_noisy", noisy, lab)  # val, near-duplicate
    other = rng.normal(size=(4, *SHAPE)).astype(np.float32)
    write_case(prep, "other", other, tumour((slice(9, 14), slice(9, 14), slice(4, 7))))
    # Same content as "a" but stored as a smaller crop at a different bbox.
    bbox = [[2, 14], [3, 15], [1, 7]]
    framed = np.zeros((4, *SHAPE), dtype=np.float32)
    framed[:, 2:14, 3:15, 1:7] = base[:, 2:14, 3:15, 1:7]
    write_case(
        prep,
        "a_cropped",
        base[:, 2:14, 3:15, 1:7],
        lab[2:14, 3:15, 1:7],
        bbox,
    )
    splits = write_splits(
        tmp_path / "splits.yaml",
        ["a", "other"],
        ["a_noisy", "a_cropped"],
        ["a_copy"],
    )
    return prep, splits, tmp_path / "out"


def run(world, **kw):
    prep, splits, out = world
    return lc.run(prep, splits, out, grid=GRID, top_k=20, **kw), out


def test_exact_duplicate_found_and_flagged(world):
    summary, out = run(world)
    dups = pd.read_csv(out / "exact_duplicates.csv")
    img = dups[dups.hash_kind == "image"]
    assert len(img) == 1
    assert set(img.iloc[0].case_ids.split(";")) == {"a", "a_copy"}
    assert bool(img.iloc[0].cross_split)
    flagged = pd.read_csv(out / "flagged.csv")
    pairs = {frozenset((r.case_a, r.case_b)) for r in flagged.itertuples()}
    assert frozenset(("a", "a_copy")) in pairs
    assert summary["n_flagged"] == len(flagged)


def test_near_duplicate_ranks_above_unrelated(world):
    _, out = run(world)
    top = pd.read_csv(out / "pairs_top.csv")
    cross = top[top.kind == "cross"].reset_index(drop=True)

    def score(x, y):
        m = cross[
            ((cross.case_a == x) & (cross.case_b == y))
            | ((cross.case_a == y) & (cross.case_b == x))
        ]
        return float(m.image_score.iloc[0])

    assert score("a_noisy", "a_copy") > 0.99
    assert score("a_noisy", "a_copy") > score("other", "a_noisy")
    assert cross.image_score.is_monotonic_decreasing
    assert cross.image_score.iloc[0] > 0.99


def test_bbox_reconstruction_matches_uncropped_twin(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.normal(size=(4, *SHAPE)).astype(np.float32)
    bbox = [[2, 14], [2, 14], [0, 8]]
    full = np.zeros_like(base)
    full[:, 2:14, 2:14, :] = base[:, 2:14, 2:14, :]
    lab = np.zeros(SHAPE, dtype=np.uint8)
    write_case(tmp_path, "full", full, lab)
    write_case(tmp_path, "crop", base[:, 2:14, 2:14, :], lab[2:14, 2:14, :], bbox)
    v_full, _, _ = lc.case_descriptors(tmp_path / "full", GRID)
    v_crop, _, _ = lc.case_descriptors(tmp_path / "crop", GRID)
    assert float(v_full @ v_crop) == pytest.approx(1.0, abs=1e-3)


def test_empty_label_gives_zero_vector_no_nan(tmp_path):
    rng = np.random.default_rng(2)
    prep = tmp_path / "prep"
    img = rng.normal(size=(4, *SHAPE))
    write_case(prep, "e1", img, np.zeros(SHAPE))
    write_case(prep, "e2", img, np.zeros(SHAPE))
    splits = write_splits(tmp_path / "s.yaml", ["e1"], ["e2"], [])
    _, tv, empty = lc.case_descriptors(prep / "e1", GRID)
    assert empty and not tv.any()
    out = tmp_path / "out"
    summary = lc.run(prep, splits, out, grid=GRID)
    assert set(summary["empty_label_cases"]) == {"e1", "e2"}
    top = pd.read_csv(out / "pairs_top.csv")
    assert not top[["image_score", "tumour_score"]].isna().any().any()
    text = (out / "summary.json").read_text()
    assert "NaN" not in text


def test_missing_case_raises(world, tmp_path):
    prep, _, out = world
    splits = write_splits(tmp_path / "bad.yaml", ["a", "ghost"], ["a_noisy"], ["a_copy"])
    with pytest.raises(FileNotFoundError, match="ghost"):
        lc.run(prep, splits, out, grid=GRID)


def test_split_overlap_reported(world, tmp_path):
    prep, _, out = world
    splits = write_splits(tmp_path / "ov.yaml", ["a", "other"], ["a"], ["a_copy"])
    summary = lc.run(prep, splits, out, grid=GRID)
    assert summary["split_overlap"] == {"a": ["train", "val"]}
    assert json.loads((out / "summary.json").read_text())["split_overlap"]
