"""Tests for scripts/make_splits.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import, following the same pattern as `tests/test_train_script.py` and
`scripts/smoke_test.py`. Composing the real Hydra config exercises the real
`configs/` directory (including the `data.splits.overwrite` key added
alongside this script), the same way `scripts/smoke_test.py::_compose_config`
does.

Everything here lives under `tmp_path` -- no test reads, writes, or deletes
anything under the real `data/` directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import hydra
import numpy as np
import pytest

from neurovision.data.dataset import load_splits

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_splits.py"
_spec = importlib.util.spec_from_file_location("make_splits_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
make_splits_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["make_splits_script"] = make_splits_script
_spec.loader.exec_module(make_splits_script)

discover_case_ids = make_splits_script.discover_case_ids
run_make_splits = make_splits_script.run_make_splits

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _write_valid_case(prep_dir: Path, case_id: str) -> None:
    """Writes a minimal valid preprocessed case: image.npy + meta.json."""
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "image.npy", np.zeros((2, 2, 2), dtype=np.float16))
    (case_dir / "meta.json").write_text("{}")


def _compose_config(root_dir: Path, prep_dir: Path, splits_path: Path, overwrite: bool = False):
    """Composes the real Hydra config with make_splits-relevant overrides.

    Mirrors scripts/smoke_test.py::_compose_config -- uses Hydra's
    programmatic API since this test picks its own fixed overrides rather
    than taking them from a CLI.
    """
    overrides = [
        f"data.root_dir={root_dir}",
        f"data.preprocessing.out_dir={prep_dir}",
        f"data.splits.path={splits_path}",
        f"data.splits.overwrite={'true' if overwrite else 'false'}",
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


# ---------------------------------------------------------------------------
# 1. discover_case_ids returns exactly the valid case ids, sorted
# ---------------------------------------------------------------------------


def test_discover_case_ids_returns_sorted_valid_cases(tmp_path: Path):
    prep_dir = tmp_path / "preprocessed"
    for case_id in ("case_002", "case_000", "case_001"):
        _write_valid_case(prep_dir, case_id)

    result = discover_case_ids(prep_dir)

    assert result == ["case_000", "case_001", "case_002"]


# ---------------------------------------------------------------------------
# 2. discover_case_ids skips directories missing either required file
# ---------------------------------------------------------------------------


def test_discover_case_ids_skips_incomplete_directories(tmp_path: Path, caplog):
    prep_dir = tmp_path / "preprocessed"
    _write_valid_case(prep_dir, "good_case")

    # Has image.npy but no meta.json.
    no_meta_dir = prep_dir / "no_meta_case"
    no_meta_dir.mkdir(parents=True)
    np.save(no_meta_dir / "image.npy", np.zeros((2, 2, 2), dtype=np.float16))

    # Has meta.json but no image.npy.
    no_image_dir = prep_dir / "no_image_case"
    no_image_dir.mkdir(parents=True)
    (no_image_dir / "meta.json").write_text("{}")

    import logging

    with caplog.at_level(logging.WARNING):
        result = discover_case_ids(prep_dir)

    assert result == ["good_case"]
    warning_text = " ".join(record.message for record in caplog.records)
    assert "no_meta_case" in warning_text
    assert "no_image_case" in warning_text


# ---------------------------------------------------------------------------
# 3. discover_case_ids raises on missing / empty prep_dir
# ---------------------------------------------------------------------------


def test_discover_case_ids_raises_file_not_found_on_missing_dir(tmp_path: Path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_case_ids(missing_dir)


def test_discover_case_ids_raises_value_error_on_empty_dir(tmp_path: Path):
    empty_dir = tmp_path / "preprocessed"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        discover_case_ids(empty_dir)


# ---------------------------------------------------------------------------
# 4. run_make_splits: no overlap, no loss
# ---------------------------------------------------------------------------


def test_run_make_splits_writes_file_with_no_overlap_and_no_loss(tmp_path: Path):
    prep_dir = tmp_path / "preprocessed"
    case_ids = [f"case_{i:03d}" for i in range(20)]
    for case_id in case_ids:
        _write_valid_case(prep_dir, case_id)

    splits_path = tmp_path / "splits.yaml"
    cfg = _compose_config(tmp_path, prep_dir, splits_path)

    splits = run_make_splits(cfg)

    assert splits_path.is_file()

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert train_set | val_set | test_set == set(case_ids)
    assert train_set & val_set == set()
    assert train_set & test_set == set()
    assert val_set & test_set == set()


# ---------------------------------------------------------------------------
# 5. Re-running with overwrite=false reuses the SAME split (freeze holds)
# ---------------------------------------------------------------------------


def test_run_make_splits_reuses_existing_split_when_not_overwriting(tmp_path: Path):
    prep_dir = tmp_path / "preprocessed"
    case_ids = [f"case_{i:03d}" for i in range(20)]
    for case_id in case_ids:
        _write_valid_case(prep_dir, case_id)

    splits_path = tmp_path / "splits.yaml"
    cfg = _compose_config(tmp_path, prep_dir, splits_path, overwrite=False)

    first_splits = run_make_splits(cfg)
    second_splits = run_make_splits(cfg)

    assert first_splits == second_splits


# ---------------------------------------------------------------------------
# 6. Split sizes match configured fractions within rounding
# ---------------------------------------------------------------------------


def test_run_make_splits_sizes_match_fractions(tmp_path: Path):
    prep_dir = tmp_path / "preprocessed"
    case_ids = [f"case_{i:03d}" for i in range(20)]
    for case_id in case_ids:
        _write_valid_case(prep_dir, case_id)

    splits_path = tmp_path / "splits.yaml"
    cfg = _compose_config(tmp_path, prep_dir, splits_path)

    splits = run_make_splits(cfg)

    # 20 cases at 0.7 / 0.15 / 0.15 -> 14 / 3 / 3.
    assert len(splits["train"]) == 14
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 3


# ---------------------------------------------------------------------------
# 7. Determinism: same seed, different output paths -> identical splits
# ---------------------------------------------------------------------------


def test_run_make_splits_is_deterministic_across_output_paths(tmp_path: Path):
    prep_dir = tmp_path / "preprocessed"
    case_ids = [f"case_{i:03d}" for i in range(20)]
    for case_id in case_ids:
        _write_valid_case(prep_dir, case_id)

    splits_path_a = tmp_path / "splits_a.yaml"
    splits_path_b = tmp_path / "splits_b.yaml"

    cfg_a = _compose_config(tmp_path, prep_dir, splits_path_a)
    cfg_b = _compose_config(tmp_path, prep_dir, splits_path_b)

    splits_a = run_make_splits(cfg_a)
    splits_b = run_make_splits(cfg_b)

    assert splits_a == splits_b

    # Also confirm what's on disk agrees with what was returned.
    assert load_splits(splits_path_a) == splits_a
    assert load_splits(splits_path_b) == splits_b
