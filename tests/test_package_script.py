"""Tests for scripts/package_for_kaggle.py.

`main()` wraps CLI/argparse plumbing around `run()`, which is what these
tests exercise directly -- following the same pattern as
tests/test_preprocess_script.py and tests/test_train_script.py: no
subprocess, no real BraTS data, everything under `tmp_path`.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import. The path is built relative to this test file, never hardcoded, so
this works regardless of where the repo is checked out.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

from neurovision.utils.io import write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "package_for_kaggle.py"
_spec = importlib.util.spec_from_file_location("package_for_kaggle_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
package_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["package_for_kaggle_script"] = package_script
_spec.loader.exec_module(package_script)

build_arg_parser = package_script.build_arg_parser
run = package_script.run


def _make_prep_tree(root: Path, case_ids: list[str]) -> None:
    """Build a tiny synthetic preprocessed cache: one dir per case, two files each."""
    for case_id in case_ids:
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        (case_dir / "image.npy").write_bytes(b"x" * 100)
        (case_dir / "label.npy").write_bytes(b"y" * 20)


def _make_splits_file(path: Path, case_ids: list[str]) -> None:
    n = len(case_ids)
    n_train = max(1, n - 1)
    write_yaml(
        {
            "train": case_ids[:n_train],
            "val": case_ids[n_train:],
            "test": [],
            "meta": {"seed": 42, "fractions": [0.7, 0.15, 0.15], "n_cases": n},
        },
        path,
    )


def _parse(argv: list[str]):
    return build_arg_parser().parse_args(argv)


# --- happy path ---------------------------------------------------------------


def test_happy_path_builds_expected_structure_with_matching_contents(tmp_path: Path):
    case_ids = ["case_001", "case_002", "case_003"]
    prep_dir = tmp_path / "prep"
    _make_prep_tree(prep_dir, case_ids)
    splits_path = tmp_path / "splits.yaml"
    _make_splits_file(splits_path, case_ids)
    out_dir = tmp_path / "out"

    args = _parse(
        ["--prep-dir", str(prep_dir), "--splits", str(splits_path), "--out", str(out_dir)]
    )
    run(args)

    assert (out_dir / "preprocessed").is_dir()
    assert (out_dir / "splits.yaml").is_file()
    for case_id in case_ids:
        case_out = out_dir / "preprocessed" / case_id
        assert case_out.is_dir()
        src_case = prep_dir / case_id
        assert (case_out / "image.npy").read_bytes() == (src_case / "image.npy").read_bytes()
        assert (case_out / "label.npy").read_bytes() == (src_case / "label.npy").read_bytes()
    assert (out_dir / "splits.yaml").read_bytes() == splits_path.read_bytes()


# --- dataset-metadata.json -----------------------------------------------------


def test_dataset_metadata_json_written_with_slug_and_title(tmp_path: Path):
    case_ids = ["case_001"]
    prep_dir = tmp_path / "prep"
    _make_prep_tree(prep_dir, case_ids)
    splits_path = tmp_path / "splits.yaml"
    _make_splits_file(splits_path, case_ids)
    out_dir = tmp_path / "out"

    args = _parse(
        [
            "--prep-dir",
            str(prep_dir),
            "--splits",
            str(splits_path),
            "--out",
            str(out_dir),
            "--slug",
            "myuser/neurovision-brats-prep",
        ]
    )
    run(args)

    meta_path = out_dir / "dataset-metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())
    assert meta["id"] == "myuser/neurovision-brats-prep"
    assert meta["title"] == "neurovision-brats-prep"
    assert meta["licenses"] == [{"name": "CC0-1.0"}]


def test_dataset_metadata_json_absent_without_slug(tmp_path: Path):
    case_ids = ["case_001"]
    prep_dir = tmp_path / "prep"
    _make_prep_tree(prep_dir, case_ids)
    splits_path = tmp_path / "splits.yaml"
    _make_splits_file(splits_path, case_ids)
    out_dir = tmp_path / "out"

    args = _parse(
        ["--prep-dir", str(prep_dir), "--splits", str(splits_path), "--out", str(out_dir)]
    )
    run(args)

    assert not (out_dir / "dataset-metadata.json").exists()


# --- missing case ---------------------------------------------------------------


def test_missing_case_directory_raises_and_names_the_missing_id(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    _make_prep_tree(prep_dir, ["case_001"])
    splits_path = tmp_path / "splits.yaml"
    _make_splits_file(splits_path, ["case_001", "case_002_missing"])
    out_dir = tmp_path / "out"

    args = _parse(
        ["--prep-dir", str(prep_dir), "--splits", str(splits_path), "--out", str(out_dir)]
    )
    with pytest.raises(ValueError, match="case_002_missing"):
        run(args)


# --- junk exclusion ---------------------------------------------------------------


def test_junk_files_excluded_from_output(tmp_path: Path):
    case_ids = ["case_001"]
    prep_dir = tmp_path / "prep"
    _make_prep_tree(prep_dir, case_ids)
    (prep_dir / ".DS_Store").write_bytes(b"junk")
    (prep_dir / "case_001" / "._image.npy").write_bytes(b"junk")
    splits_path = tmp_path / "splits.yaml"
    _make_splits_file(splits_path, case_ids)
    out_dir = tmp_path / "out"

    args = _parse(
        ["--prep-dir", str(prep_dir), "--splits", str(splits_path), "--out", str(out_dir)]
    )
    run(args)

    found_names = {p.name for p in (out_dir / "preprocessed").rglob("*")}
    assert ".DS_Store" not in found_names
    assert "._image.npy" not in found_names


# --- dry run ---------------------------------------------------------------


def test_dry_run_writes_nothing_but_reports_nonzero_size(tmp_path: Path):
    case_ids = ["case_001", "case_002"]
    prep_dir = tmp_path / "prep"
    _make_prep_tree(prep_dir, case_ids)
    splits_path = tmp_path / "splits.yaml"
    _make_splits_file(splits_path, case_ids)
    out_dir = tmp_path / "out"

    args = _parse(
        [
            "--prep-dir",
            str(prep_dir),
            "--splits",
            str(splits_path),
            "--out",
            str(out_dir),
            "--dry-run",
        ]
    )
    report = run(args)

    assert not out_dir.exists()
    assert report["total_bytes"] > 0


# --- size warning ---------------------------------------------------------------


def test_size_warning_logged_when_over_tiny_limit(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    case_ids = ["case_001"]
    prep_dir = tmp_path / "prep"
    _make_prep_tree(prep_dir, case_ids)
    splits_path = tmp_path / "splits.yaml"
    _make_splits_file(splits_path, case_ids)
    out_dir = tmp_path / "out"

    args = _parse(
        [
            "--prep-dir",
            str(prep_dir),
            "--splits",
            str(splits_path),
            "--out",
            str(out_dir),
            "--limit-gb",
            "1e-9",
        ]
    )
    with caplog.at_level(logging.WARNING, logger=package_script.logger.name):
        run(args)

    assert any("EXCEEDS" in record.message for record in caplog.records)


# --- metadata.csv absent ---------------------------------------------------------------


def test_metadata_csv_absent_warns_but_succeeds(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    case_ids = ["case_001"]
    prep_dir = tmp_path / "prep"
    _make_prep_tree(prep_dir, case_ids)
    splits_path = tmp_path / "splits.yaml"
    _make_splits_file(splits_path, case_ids)
    out_dir = tmp_path / "out"

    args = _parse(
        ["--prep-dir", str(prep_dir), "--splits", str(splits_path), "--out", str(out_dir)]
    )
    with caplog.at_level(logging.WARNING, logger=package_script.logger.name):
        report = run(args)

    assert report is not None
    assert (out_dir / "preprocessed").is_dir()
    assert not (out_dir / "metadata.csv").exists()
    assert any("No metadata.csv" in record.message for record in caplog.records)
