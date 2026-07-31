"""Tests for neurovision.data.dataset.

All cases are tiny synthetic .npy arrays built inline under pytest's
`tmp_path` -- never real BraTS data -- so the whole suite runs on CPU well
under a second. See CLAUDE.md for the project's testing rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from monai.data import CacheDataset, Dataset, PersistentDataset
from monai.transforms import Lambdad

from neurovision.data.dataset import (
    build_data_dicts,
    build_dataset,
    load_splits,
    make_splits,
)
from neurovision.utils.io import read_yaml

_IMAGE_SHAPE = (4, 8, 8, 8)
_LABEL_SHAPE = (8, 8, 8)


def _make_case(prep_dir: Path, case_id: str, with_label: bool = True) -> None:
    """Write a synthetic preprocessed case: image.npy (+ label.npy)."""
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros(_IMAGE_SHAPE, dtype=np.float16)
    np.save(case_dir / "image.npy", image)
    if with_label:
        label = np.zeros(_LABEL_SHAPE, dtype=np.uint8)
        np.save(case_dir / "label.npy", label)


# --- build_data_dicts -------------------------------------------------


def test_build_data_dicts_returns_expected_keys_and_string_values(tmp_path: Path):
    _make_case(tmp_path, "case_001")
    dicts = build_data_dicts(["case_001"], tmp_path)
    assert len(dicts) == 1
    entry = dicts[0]
    assert entry["case_id"] == "case_001"
    assert entry["image"] == str(tmp_path / "case_001" / "image.npy")
    assert entry["label"] == str(tmp_path / "case_001" / "label.npy")
    assert isinstance(entry["image"], str)
    assert isinstance(entry["label"], str)
    assert not isinstance(entry["image"], Path)


def test_build_data_dicts_omits_label_when_missing(tmp_path: Path):
    _make_case(tmp_path, "case_002", with_label=False)
    dicts = build_data_dicts(["case_002"], tmp_path)
    assert "label" not in dicts[0]
    assert "image" in dicts[0]


def test_build_data_dicts_raises_filenotfound_naming_case(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc_info:
        build_data_dicts(["missing_case"], tmp_path)
    assert "missing_case" in str(exc_info.value)


# --- make_splits --------------------------------------------------------


def _ids(n: int) -> list[str]:
    return [f"case_{i:03d}" for i in range(n)]


def test_make_splits_gives_70_15_15_and_every_case_once(tmp_path: Path):
    case_ids = _ids(100)
    out_path = tmp_path / "splits.yaml"
    splits = make_splits(case_ids, out_path, fractions=(0.7, 0.15, 0.15), seed=42)

    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15

    union = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert union == set(case_ids)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(case_ids)


def test_make_splits_is_deterministic_same_seed(tmp_path: Path):
    case_ids = _ids(50)
    splits_a = make_splits(case_ids, tmp_path / "a.yaml", seed=7)
    splits_b = make_splits(case_ids, tmp_path / "b.yaml", seed=7)
    assert splits_a == splits_b


def test_make_splits_different_seed_gives_different_split(tmp_path: Path):
    case_ids = _ids(50)
    splits_a = make_splits(case_ids, tmp_path / "a.yaml", seed=1)
    splits_b = make_splits(case_ids, tmp_path / "b.yaml", seed=2)
    assert splits_a != splits_b


def test_make_splits_sorts_input_order_does_not_matter(tmp_path: Path):
    case_ids = _ids(50)
    shuffled_ids = list(reversed(case_ids))
    splits_a = make_splits(case_ids, tmp_path / "a.yaml", seed=42)
    splits_b = make_splits(shuffled_ids, tmp_path / "b.yaml", seed=42)
    assert splits_a == splits_b


def test_make_splits_reuses_existing_file(tmp_path: Path):
    case_ids = _ids(30)
    out_path = tmp_path / "splits.yaml"
    first = make_splits(case_ids, out_path, seed=1)
    second = make_splits(case_ids, out_path, seed=999)
    assert second == first


def test_make_splits_raises_value_error_on_mismatched_case_set(tmp_path: Path):
    case_ids = _ids(30)
    out_path = tmp_path / "splits.yaml"
    make_splits(case_ids, out_path, seed=1)

    new_case_ids = _ids(35)  # a superset -> different set
    with pytest.raises(ValueError) as exc_info:
        make_splits(new_case_ids, out_path, seed=1)
    assert "overwrite" in str(exc_info.value)


def test_make_splits_overwrite_true_regenerates(tmp_path: Path):
    case_ids = _ids(30)
    out_path = tmp_path / "splits.yaml"
    first = make_splits(case_ids, out_path, seed=1)
    second = make_splits(case_ids, out_path, seed=999, overwrite=True)
    assert second != first


def test_make_splits_raises_on_empty_ids(tmp_path: Path):
    with pytest.raises(ValueError):
        make_splits([], tmp_path / "splits.yaml")


def test_make_splits_raises_on_duplicate_ids(tmp_path: Path):
    with pytest.raises(ValueError):
        make_splits(["a", "b", "a"], tmp_path / "splits.yaml")


def test_make_splits_raises_on_fractions_not_summing_to_one(tmp_path: Path):
    with pytest.raises(ValueError):
        make_splits(_ids(10), tmp_path / "splits.yaml", fractions=(0.5, 0.4, 0.4))


def test_make_splits_raises_on_wrong_length_fractions(tmp_path: Path):
    with pytest.raises(ValueError):
        make_splits(_ids(10), tmp_path / "splits.yaml", fractions=(0.5, 0.5))


def test_make_splits_writes_meta_block(tmp_path: Path):
    case_ids = _ids(20)
    out_path = tmp_path / "splits.yaml"
    make_splits(case_ids, out_path, fractions=(0.6, 0.2, 0.2), seed=123)

    raw = read_yaml(out_path)
    assert raw["meta"]["seed"] == 123
    assert raw["meta"]["fractions"] == [0.6, 0.2, 0.2]
    assert raw["meta"]["n_cases"] == 20


# --- load_splits --------------------------------------------------------


def test_load_splits_round_trips(tmp_path: Path):
    case_ids = _ids(20)
    out_path = tmp_path / "splits.yaml"
    written = make_splits(case_ids, out_path, seed=5)
    loaded = load_splits(out_path)
    assert loaded == written


def test_load_splits_raises_filenotfound(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path / "nonexistent.yaml")


def test_load_splits_raises_value_error_on_missing_key(tmp_path: Path):
    from neurovision.utils.io import write_yaml

    out_path = tmp_path / "bad.yaml"
    write_yaml({"train": ["a"], "val": ["b"]}, out_path)  # missing "test"
    with pytest.raises(ValueError):
        load_splits(out_path)


# --- build_dataset --------------------------------------------------------


def _trivial_transform():
    return Lambdad(keys=["case_id"], func=lambda x: x)


def test_build_dataset_plain(tmp_path: Path):
    _make_case(tmp_path, "case_001")
    dicts = build_data_dicts(["case_001"], tmp_path)
    ds = build_dataset(dicts, _trivial_transform(), dataset_type="dataset")
    assert isinstance(ds, Dataset)
    assert not isinstance(ds, CacheDataset)


def test_build_dataset_cache(tmp_path: Path):
    _make_case(tmp_path, "case_001")
    dicts = build_data_dicts(["case_001"], tmp_path)
    ds = build_dataset(dicts, _trivial_transform(), dataset_type="cache", cache_rate=1.0)
    assert isinstance(ds, CacheDataset)


def test_build_dataset_persistent(tmp_path: Path):
    _make_case(tmp_path, "case_001")
    dicts = build_data_dicts(["case_001"], tmp_path)
    cache_dir = tmp_path / "pcache"
    ds = build_dataset(dicts, _trivial_transform(), dataset_type="persistent", cache_dir=cache_dir)
    assert isinstance(ds, PersistentDataset)
    assert cache_dir.is_dir()


def test_build_dataset_unknown_type_raises(tmp_path: Path):
    _make_case(tmp_path, "case_001")
    dicts = build_data_dicts(["case_001"], tmp_path)
    with pytest.raises(ValueError):
        build_dataset(dicts, _trivial_transform(), dataset_type="bogus")


def test_build_dataset_persistent_requires_cache_dir(tmp_path: Path):
    _make_case(tmp_path, "case_001")
    dicts = build_data_dicts(["case_001"], tmp_path)
    with pytest.raises(ValueError):
        build_dataset(dicts, _trivial_transform(), dataset_type="persistent", cache_dir=None)
