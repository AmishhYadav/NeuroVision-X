"""Tests for scripts/preprocess.py's pure helper functions.

`main()` is a Hydra entry point and is awkward (and unnecessary) to unit
test directly, so these tests import only the plain helpers -- `summarize`,
`directory_size_bytes`, `format_size` -- and exercise them on hand-built data
and tiny `tmp_path` trees. No multiprocessing, no real BraTS data, no Hydra
invocation.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import.
The path is built relative to this test file (`Path(__file__).resolve()...`),
never hardcoded, so this works regardless of where the repo is checked out.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"
_spec = importlib.util.spec_from_file_location("preprocess_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
preprocess_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["preprocess_script"] = preprocess_script
_spec.loader.exec_module(preprocess_script)

directory_size_bytes = preprocess_script.directory_size_bytes
format_size = preprocess_script.format_size
summarize = preprocess_script.summarize


# --- directory_size_bytes ---------------------------------------------------


def test_directory_size_bytes_sums_known_file_sizes(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "b.bin").write_bytes(b"y" * 250)
    assert directory_size_bytes(tmp_path) == 350


def test_directory_size_bytes_includes_nested_subdirectories(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    nested = tmp_path / "case_001"
    nested.mkdir()
    (nested / "image.npy").write_bytes(b"z" * 500)
    deeper = nested / "sub"
    deeper.mkdir()
    (deeper / "extra.npy").write_bytes(b"w" * 25)
    assert directory_size_bytes(tmp_path) == 100 + 500 + 25


def test_directory_size_bytes_empty_directory_returns_zero(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert directory_size_bytes(empty_dir) == 0


def test_directory_size_bytes_missing_directory_returns_zero(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    assert directory_size_bytes(missing) == 0


# --- format_size -------------------------------------------------------------


def test_format_size_bytes():
    assert format_size(500) == "500.00 B"


def test_format_size_megabytes():
    assert format_size(5 * 1024 * 1024) == "5.00 MB"


def test_format_size_gigabytes():
    assert format_size(int(2.5 * 1024**3)) == "2.50 GB"


# --- summarize ----------------------------------------------------------------


def _success(case_id: str, skipped: bool, cropped_shape: tuple[int, int, int], **classes):
    counts = {"n_class_0": 0, "n_class_1": 0, "n_class_2": 0, "n_class_3": 0}
    counts.update(classes)
    return {
        "case_id": case_id,
        "original_shape": (240, 240, 155),
        "cropped_shape": cropped_shape,
        "spacing": (1.0, 1.0, 1.0),
        "image_bytes": 1000,
        "label_bytes": 100,
        "skipped": skipped,
        **counts,
    }


def test_summarize_counts_processed_skipped_failed(tmp_path: Path):
    summaries = [
        _success("case_001", skipped=False, cropped_shape=(100, 100, 100)),
        _success("case_002", skipped=True, cropped_shape=(100, 100, 100)),
        {"case_id": "case_003", "error": "corrupt header"},
    ]
    stats = summarize(summaries, tmp_path)
    assert stats["n_found"] == 3
    assert stats["n_processed"] == 1
    assert stats["n_skipped"] == 1
    assert stats["n_failed"] == 1
    assert stats["failures"] == [{"case_id": "case_003", "error": "corrupt header"}]


def test_summarize_sums_per_class_voxel_counts(tmp_path: Path):
    summaries = [
        _success(
            "case_001",
            skipped=False,
            cropped_shape=(100, 100, 100),
            n_class_1=10,
            n_class_2=20,
            n_class_3=5,
        ),
        _success(
            "case_002",
            skipped=False,
            cropped_shape=(100, 100, 100),
            n_class_1=1,
            n_class_2=2,
            n_class_3=3,
        ),
    ]
    stats = summarize(summaries, tmp_path)
    assert stats["voxel_totals"] == {"n_class_1": 11, "n_class_2": 22, "n_class_3": 8}


def test_summarize_mean_and_median_cropped_shape(tmp_path: Path):
    summaries = [
        _success("case_001", skipped=False, cropped_shape=(100, 120, 140)),
        _success("case_002", skipped=False, cropped_shape=(200, 140, 160)),
    ]
    stats = summarize(summaries, tmp_path)
    assert stats["mean_cropped_shape"] == (150.0, 130.0, 150.0)
    assert stats["median_cropped_shape"] == (150.0, 130.0, 150.0)


def test_summarize_empty_list_does_not_raise(tmp_path: Path):
    stats = summarize([], tmp_path)
    assert stats["n_found"] == 0
    assert stats["n_processed"] == 0
    assert stats["n_skipped"] == 0
    assert stats["n_failed"] == 0
    assert stats["mean_cropped_shape"] is None
    assert stats["median_cropped_shape"] is None
    assert stats["voxel_totals"] == {"n_class_1": 0, "n_class_2": 0, "n_class_3": 0}
