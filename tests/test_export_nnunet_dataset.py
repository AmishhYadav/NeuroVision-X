"""Tests for scripts/export_nnunet_dataset.py.

`main()` wraps CLI/argparse plumbing around `run()`, which is what these tests exercise
directly -- following `tests/test_package_script.py`'s pattern: no subprocess, no real
BraTS data, everything under `tmp_path`.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import (same as
`tests/test_package_script.py`). The path is built relative to this test file, never
hardcoded, so this works regardless of where the repo is checked out.

This module (and hence this whole test file) needs `nnunetv2`, which lives only in
`.venv-analysis` (see `requirements-analysis.txt`) -- never in the main `.venv`.
Following `tests/test_clinical_resample.py`'s exact `pytest.importorskip("nnunetv2")`
idiom, applied at the top of EVERY test function body (not module scope), so this file
still collects -- and skips cleanly, one test at a time -- when the main `.venv` runs the
suite. `export_nnunet_dataset.build_splits_final` only imports `nnunetv2` lazily inside
itself, but every test here reaches that code path (directly or via `run()`), so gating
per-function rather than per-module is the more honest signal about what actually failed.

Deviation from the spec's literal fixture size, noted here rather than silently: nnU-Net's
own `generate_crossval_split` (which `build_splits_final` calls unmodified, per this
module's design) always requests `n_splits=5`, and scikit-learn's `KFold` raises
`ValueError` if `n_splits` exceeds the sample count. Two train cases -- as the spec's test
1 describes -- cannot produce a 5-fold split with nnU-Net's own algorithm at all, so the
happy-path fixture below uses 5 train cases (the minimum n_splits=5 admits), not 2. Every
other assertion in that test matches the spec as written.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import pytest

from neurovision.utils.io import read_json, write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_nnunet_dataset.py"
_spec = importlib.util.spec_from_file_location("export_nnunet_dataset_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
export_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["export_nnunet_dataset_script"] = export_script
_spec.loader.exec_module(export_script)

build_arg_parser = export_script.build_arg_parser
run = export_script.run

_SHAPE = (4, 4, 4)
_AFFINE = np.eye(4)


def _write_raw_case(
    raw_root: Path,
    case_id: str,
    t1_value: float = 1.0,
    t1ce_value: float = 2.0,
    t2_value: float = 3.0,
    flair_value: float = 4.0,
    seg_data: np.ndarray | None = None,
    subdir: str = "",
) -> Path:
    """Writes one synthetic raw BraTS case: 4 tiny image volumes (+ optional seg).

    Each modality gets a distinct constant value so a test can tell, after export, which
    source file actually landed in which output channel.

    Args:
        raw_root: Root raw directory.
        case_id: BraTS-style case id.
        t1_value, t1ce_value, t2_value, flair_value: Constant fill value per modality.
        seg_data: If given, written as `<case_id>_seg.nii` (int16, values in {0,1,2,4}).
        subdir: Extra nesting under raw_root/case_id (e.g. "wrap1/wrap2"), to exercise
            depth-independent discovery.

    Returns:
        The case's directory.
    """
    case_dir = raw_root / subdir / case_id if subdir else raw_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for suffix, value in (
        ("_t1.nii", t1_value),
        ("_t1ce.nii", t1ce_value),
        ("_t2.nii", t2_value),
        ("_flair.nii", flair_value),
    ):
        data = np.full(_SHAPE, value, dtype=np.float32)
        nib.save(nib.Nifti1Image(data, _AFFINE), str(case_dir / f"{case_id}{suffix}"))
    if seg_data is not None:
        nib.save(
            nib.Nifti1Image(seg_data.astype(np.int16), _AFFINE),
            str(case_dir / f"{case_id}_seg.nii"),
        )
    return case_dir


def _default_seg() -> np.ndarray:
    """A tiny seg volume spot-checking all three nonzero BraTS labels at known voxels."""
    seg = np.zeros(_SHAPE, dtype=np.int16)
    seg[0, 0, 0] = 4  # enhancing tumor -> nnU-Net class 3
    seg[1, 1, 1] = 2  # edema -> nnU-Net class 1
    seg[2, 2, 2] = 1  # necrotic core -> nnU-Net class 2
    return seg


def _parse(argv: list[str]):
    return build_arg_parser().parse_args(argv)


# --- happy path ---------------------------------------------------------------


def test_happy_path_builds_expected_nnunet_layout(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")

    # 5 train cases, not the spec's literal 2 -- see module docstring for why
    # (nnU-Net's own generate_crossval_split needs n_splits=5 <= n_samples).
    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    test_ids = ["BraTS2021_09000"]

    raw_dir = tmp_path / "raw"
    for cid in train_ids:
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())
    # BraTS2021's public release ships _seg for every case, including ones our
    # split holds out as test -- included here to prove it is never copied out.
    _write_raw_case(raw_dir, test_ids[0], seg_data=_default_seg())

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": [], "test": test_ids}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset999_HappyPath"
    args = _parse(["--raw-dir", str(raw_dir), "--splits", str(splits_path), "--out", str(out_dir)])
    run(args)

    images_tr = sorted((out_dir / "imagesTr").iterdir())
    assert len(images_tr) == len(train_ids) * 4
    for cid in train_ids:
        for i in range(4):
            assert (out_dir / "imagesTr" / f"{cid}_{i:04d}.nii.gz").is_file()

    labels_tr = sorted((out_dir / "labelsTr").iterdir())
    assert len(labels_tr) == len(train_ids)
    spot_check_label = nib.load(str(out_dir / "labelsTr" / f"{train_ids[0]}.nii.gz"))
    spot_check_data = np.asarray(spot_check_label.dataobj)
    assert spot_check_data[0, 0, 0] == 3  # raw 4 -> 3
    assert spot_check_data[1, 1, 1] == 1  # raw 2 -> 1
    assert spot_check_data[2, 2, 2] == 2  # raw 1 -> 2
    assert spot_check_data[3, 3, 3] == 0  # raw 0 -> 0

    images_ts = sorted((out_dir / "imagesTs").iterdir())
    assert len(images_ts) == len(test_ids) * 4
    for i in range(4):
        assert (out_dir / "imagesTs" / f"{test_ids[0]}_{i:04d}.nii.gz").is_file()
    assert not (out_dir / "labelsTs").exists()

    dataset_json = read_json(out_dir / "dataset.json")
    assert dataset_json == {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "Flair"},
        "labels": {
            "background": 0,
            "whole tumor": [1, 2, 3],
            "tumor core": [2, 3],
            "enhancing tumor": [3],
        },
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
        "regions_class_order": [1, 2, 3],
        "name": "Dataset999_HappyPath",
    }

    splits_final = read_json(out_dir / "splits_final.json")
    assert len(splits_final) == 5
    for fold in splits_final:
        train_set = set(fold["train"])
        val_set = set(fold["val"])
        assert train_set | val_set == set(train_ids)
        assert train_set & val_set == set()


# --- val exclusion ---------------------------------------------------------------


def test_val_split_cases_never_appear_in_output(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")

    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    val_ids = ["BraTS2021_50000", "BraTS2021_50001"]
    test_ids = ["BraTS2021_09000"]

    raw_dir = tmp_path / "raw"
    for cid in train_ids:
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())
    for cid in val_ids:
        # Val cases exist as perfectly valid raw cases -- the point is that
        # the split file, not the raw dir, decides what gets exported.
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())
    _write_raw_case(raw_dir, test_ids[0], seg_data=_default_seg())

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": val_ids, "test": test_ids}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset998_ValExclusion"
    args = _parse(["--raw-dir", str(raw_dir), "--splits", str(splits_path), "--out", str(out_dir)])
    run(args)

    all_output_names = "\n".join(str(p) for p in out_dir.rglob("*") if p.is_file())
    for val_id in val_ids:
        assert val_id not in all_output_names

    splits_final = read_json(out_dir / "splits_final.json")
    for fold in splits_final:
        assert not (set(fold["train"]) & set(val_ids))
        assert not (set(fold["val"]) & set(val_ids))


# --- missing case ---------------------------------------------------------------


def test_missing_case_raises_and_names_the_case_and_file(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")

    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    raw_dir = tmp_path / "raw"
    for cid in train_ids:
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())

    # Delete t1ce for one case -- t1 (used for directory discovery) still exists.
    broken_case = train_ids[0]
    (raw_dir / broken_case / f"{broken_case}_t1ce.nii").unlink()

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": [], "test": []}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset997_Missing"
    args = _parse(["--raw-dir", str(raw_dir), "--splits", str(splits_path), "--out", str(out_dir)])
    with pytest.raises(FileNotFoundError) as excinfo:
        run(args)

    message = str(excinfo.value)
    assert broken_case in message
    assert "t1ce" in message


# --- unexpected label value ---------------------------------------------------------------


def test_unexpected_label_value_raises(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")

    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    raw_dir = tmp_path / "raw"
    for cid in train_ids[:-1]:
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())

    bad_case = train_ids[-1]
    bad_seg = np.zeros(_SHAPE, dtype=np.int16)
    bad_seg[0, 0, 0] = 3  # not in {0, 1, 2, 4}
    _write_raw_case(raw_dir, bad_case, seg_data=bad_seg)

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": [], "test": []}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset996_BadLabel"
    args = _parse(["--raw-dir", str(raw_dir), "--splits", str(splits_path), "--out", str(out_dir)])
    with pytest.raises(RuntimeError) as excinfo:
        run(args)

    message = str(excinfo.value)
    assert bad_case in message
    assert "3" in message


# --- dry run ---------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")

    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    test_ids = ["BraTS2021_09000"]
    raw_dir = tmp_path / "raw"
    for cid in train_ids:
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())
    _write_raw_case(raw_dir, test_ids[0])  # no seg needed for a test case

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": [], "test": test_ids}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset995_DryRun"
    args = _parse(
        [
            "--raw-dir",
            str(raw_dir),
            "--splits",
            str(splits_path),
            "--out",
            str(out_dir),
            "--dry-run",
        ]
    )
    report = run(args)

    assert not out_dir.exists()
    assert report == {"train": len(train_ids), "test": len(test_ids), "dry_run": True}


# --- --out naming convention ---------------------------------------------------------------


def test_out_basename_must_match_dataset_pattern(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": [], "val": [], "test": []}, splits_path)

    out_dir = tmp_path / "outputs" / "not_a_dataset_folder"
    args = _parse(
        [
            "--raw-dir",
            str(tmp_path / "raw"),
            "--splits",
            str(splits_path),
            "--out",
            str(out_dir),
        ]
    )
    with pytest.raises(ValueError, match="not_a_dataset_folder"):
        run(args)

    assert not out_dir.exists()


# --- existing non-empty --out ---------------------------------------------------------------


def test_existing_nonempty_out_requires_force(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")

    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    raw_dir = tmp_path / "raw"
    for cid in train_ids:
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": [], "test": []}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset994_Force"
    out_dir.mkdir(parents=True)
    (out_dir / "unrelated_leftover_file.txt").write_text("stale data from a previous run\n")

    args = _parse(["--raw-dir", str(raw_dir), "--splits", str(splits_path), "--out", str(out_dir)])
    with pytest.raises(FileExistsError):
        run(args)

    args_forced = _parse(
        [
            "--raw-dir",
            str(raw_dir),
            "--splits",
            str(splits_path),
            "--out",
            str(out_dir),
            "--force",
        ]
    )
    run(args_forced)

    assert (out_dir / "dataset.json").is_file()
    assert not (out_dir / "unrelated_leftover_file.txt").exists()


# --- --force must not destroy --out before raw files are confirmed ---------------------------


def test_force_does_not_delete_out_before_raw_files_are_confirmed(tmp_path: Path) -> None:
    """A --force re-run against an incomplete --raw-dir must not touch --out at all.

    Regression test: discovery (`discover_case_files` for every case) must run BEFORE
    the --out non-empty/--force/rmtree step, so a missing raw case aborts with --out's
    PREVIOUS contents (a possibly large, expensive-to-rebuild prior export) still intact
    -- not deleted and then left with nothing, which is what happened when the rmtree
    ran first.
    """
    pytest.importorskip("nnunetv2")

    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    raw_dir = tmp_path / "raw"
    for cid in train_ids:
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())

    # Break one required file AFTER writing every case -- t1 (used for
    # directory discovery) still exists, only t1ce is missing.
    broken_case = train_ids[0]
    (raw_dir / broken_case / f"{broken_case}_t1ce.nii").unlink()

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": [], "test": []}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset991_ForceSafety"
    out_dir.mkdir(parents=True)
    marker = out_dir / "marker_from_previous_export.txt"
    marker.write_text("a previous, valid export lives here\n")

    args = _parse(
        [
            "--raw-dir",
            str(raw_dir),
            "--splits",
            str(splits_path),
            "--out",
            str(out_dir),
            "--force",
        ]
    )
    with pytest.raises(FileNotFoundError):
        run(args)

    # --out must be untouched: the rmtree must never have run.
    assert marker.is_file()
    assert marker.read_text() == "a previous, valid export lives here\n"


# --- t1 / t1ce anchoring ---------------------------------------------------------------


def test_t1_glob_does_not_match_t1ce(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")

    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    raw_dir = tmp_path / "raw"
    # Distinguishable values: t1=111.0, t1ce=222.0 -- if the t1 glob ever
    # accidentally matched the t1ce file (substring collision), the _0000
    # channel below would come back holding 222.0 instead of 111.0.
    for cid in train_ids:
        _write_raw_case(raw_dir, cid, t1_value=111.0, t1ce_value=222.0, seg_data=_default_seg())

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": [], "test": []}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset993_T1Anchor"
    args = _parse(["--raw-dir", str(raw_dir), "--splits", str(splits_path), "--out", str(out_dir)])
    run(args)

    target_case = train_ids[0]
    t1_channel = np.asarray(
        nib.load(str(out_dir / "imagesTr" / f"{target_case}_0000.nii.gz")).dataobj
    )
    t1ce_channel = np.asarray(
        nib.load(str(out_dir / "imagesTr" / f"{target_case}_0001.nii.gz")).dataobj
    )
    assert np.all(t1_channel == 111.0)
    assert np.all(t1ce_channel == 222.0)


# --- splits_final.json matches nnU-Net's own algorithm ---------------------------------------


def test_splits_final_matches_nnunetv2_reference(tmp_path: Path) -> None:
    pytest.importorskip("nnunetv2")
    from nnunetv2.utilities.crossval_split import generate_crossval_split

    train_ids = [f"BraTS2021_{i:05d}" for i in range(1, 6)]
    raw_dir = tmp_path / "raw"
    for cid in train_ids:
        _write_raw_case(raw_dir, cid, seg_data=_default_seg())

    splits_path = tmp_path / "splits.yaml"
    write_yaml({"train": train_ids, "val": [], "test": []}, splits_path)

    out_dir = tmp_path / "nnunet_raw" / "Dataset992_ReferenceMatch"
    args = _parse(["--raw-dir", str(raw_dir), "--splits", str(splits_path), "--out", str(out_dir)])
    run(args)

    written = read_json(out_dir / "splits_final.json")

    expected_raw = generate_crossval_split(sorted(train_ids), seed=12345, n_splits=5)
    expected = [
        {"train": [str(c) for c in fold["train"]], "val": [str(c) for c in fold["val"]]}
        for fold in expected_raw
    ]

    assert written == expected
