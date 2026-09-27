"""Tests for scripts/nnunet_predict_session.py.

No real nnU-Net and no real BraTS: `subprocess.run` is monkeypatched with a
fake that writes tiny synthetic `.nii` label volumes, and every "trained
model folder" / "imagesTs" is built from tiny `torch.save` checkpoint dicts
and 4x4x4 nibabel NIfTIs under `tmp_path`. The whole file runs on CPU in well
under a second.

The script lives under scripts/, not src/, so it is loaded via
`tests/script_loader.py`'s `importlib.util.spec_from_file_location` pattern,
same as every other scripts/ test file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from tests.script_loader import load_script

nnunet_predict_session = load_script("nnunet_predict_session")

_TRAINER_DIR = "nnUNetTrainer__nnUNetPlans__3d_fullres"
_FOLD_DIR = "fold_all"
_DATASET_FOLDER = "Dataset901_NeuroVisionXBraTS21"
_CASE_IDS = ["BraTS2021_00001", "BraTS2021_00002"]
_CHANNELS = ("0000", "0001", "0002", "0003")


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _write_checkpoint(
    path: Path,
    epoch: int = 1000,
    trainer_name: str = "nnUNetTrainer",
    finite: bool = True,
) -> None:
    checkpoint = {
        "current_epoch": epoch,
        "trainer_name": trainer_name,
        "init_args": {"plans": {}, "configuration": "3d_fullres", "fold": "all"},
        "logging": {"train_losses": [0.5, 0.4] if finite else [0.1, float("nan")]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _make_model_folder(
    root: Path,
    name: str = _DATASET_FOLDER,
    ending: str = ".nii",
    epoch: int = 1000,
    checkpoint_name: str = "checkpoint_final.pth",
) -> Path:
    """Builds a fake nnU-Net results tree: <root>/<name>/<trainer>/fold_all/<checkpoint>."""
    model_folder = root / name / _TRAINER_DIR
    fold_dir = model_folder / _FOLD_DIR
    fold_dir.mkdir(parents=True)
    _write_checkpoint(fold_dir / checkpoint_name, epoch=epoch)
    (model_folder / "dataset.json").write_text(json.dumps({"file_ending": ending}))
    (model_folder / "plans.json").write_text("{}")
    return model_folder


def _make_nifti(path: Path, value: int = 0) -> None:
    arr = np.full((4, 4, 4), value, dtype=np.uint8)
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))


def _make_images_ts(root: Path, case_ids: list[str] = _CASE_IDS, ending: str = ".nii") -> Path:
    images_ts = root / "imagesTs"
    images_ts.mkdir(parents=True)
    for cid in case_ids:
        for ch in _CHANNELS:
            _make_nifti(images_ts / f"{cid}_{ch}{ending}")
    return images_ts


def _make_splits(path: Path, test_ids: list[str] = _CASE_IDS) -> None:
    path.write_text(yaml.safe_dump({"train": [], "val": [], "test": list(test_ids)}))


def _make_fake_predict_run(case_ids: list[str], ending: str = ".nii"):
    """Returns `(fake_run, calls)`: a subprocess.run stand-in that writes fake predictions."""
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], check: bool = True, **kwargs: object
    ) -> subprocess.CompletedProcess:
        calls.append(list(cmd))
        out_dir = Path(cmd[cmd.index("-o") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        for cid in case_ids:
            arr = np.zeros((4, 4, 4), dtype=np.uint8)
            arr[0, 0, 0] = 1  # label 1 = ED, a valid nnU-Net label for this dataset.
            nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(out_dir / f"{cid}{ending}"))
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run, calls


def _base_argv(
    model_root: Path,
    images_root: Path,
    splits_path: Path,
    out_root: Path,
    summary_path: Path,
    expect_epoch: int = 1000,
    device: str = "cpu",
    extra: list[str] | None = None,
) -> list[str]:
    argv = [
        "--model-search-root",
        str(model_root),
        "--images-search-root",
        str(images_root),
        "--out-root",
        str(out_root),
        "--splits",
        str(splits_path),
        "--expect-epoch",
        str(expect_epoch),
        "--device",
        device,
        "--summary-json",
        str(summary_path),
    ]
    if extra:
        argv += extra
    return argv


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_both_arms_primary_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    out_root = tmp_path / "out"
    splits_path = tmp_path / "splits.yaml"
    summary_path = tmp_path / "summary.json"

    _make_model_folder(model_root)
    _make_images_ts(images_root)
    _make_splits(splits_path)

    fake_run, calls = _make_fake_predict_run(_CASE_IDS)
    monkeypatch.setattr(nnunet_predict_session.subprocess, "run", fake_run)

    argv = _base_argv(model_root, images_root, splits_path, out_root, summary_path)
    rc = nnunet_predict_session.main(argv)

    assert rc == 0
    assert len(calls) == 2
    # Primary (tta_on, mirroring on) runs first, without --disable_tta.
    assert "--disable_tta" not in calls[0]
    # Secondary (tta_off) runs second, with --disable_tta.
    assert "--disable_tta" in calls[1]

    for arm in ("tta_on", "tta_off"):
        arm_dir = out_root / arm
        for cid in _CASE_IDS:
            gz_path = arm_dir / f"{cid}.nii.gz"
            nii_path = arm_dir / f"{cid}.nii"
            assert gz_path.is_file()
            assert not nii_path.exists()
            arr = np.asarray(nib.load(str(gz_path)).dataobj)
            assert arr[0, 0, 0] == 1

    summary = json.loads(summary_path.read_text())
    assert summary["health"] == "OK"
    assert summary["checkpoint_epoch"] == 1000
    assert summary["dry_run"] is False
    assert set(summary["arms"]) == {"tta_on", "tta_off"}
    assert summary["arms"]["tta_on"]["n_cases"] == 2
    assert summary["arms"]["tta_off"]["n_cases"] == 2


# ---------------------------------------------------------------------------
# Preflight failures
# ---------------------------------------------------------------------------


def test_wrong_epoch_raises(tmp_path: Path) -> None:
    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    splits_path = tmp_path / "splits.yaml"

    _make_model_folder(model_root, epoch=999)
    _make_images_ts(images_root)
    _make_splits(splits_path)

    args = nnunet_predict_session.parse_args(
        _base_argv(model_root, images_root, splits_path, tmp_path / "out", tmp_path / "s.json")
    )
    with pytest.raises(RuntimeError, match="current_epoch"):
        nnunet_predict_session.run(args)


def test_checkpoint_best_only_raises(tmp_path: Path) -> None:
    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    splits_path = tmp_path / "splits.yaml"

    _make_model_folder(model_root, checkpoint_name="checkpoint_best.pth")
    _make_images_ts(images_root)
    _make_splits(splits_path)

    args = nnunet_predict_session.parse_args(
        _base_argv(model_root, images_root, splits_path, tmp_path / "out", tmp_path / "s.json")
    )
    with pytest.raises(FileNotFoundError, match="checkpoint_final.pth"):
        nnunet_predict_session.run(args)


def test_two_model_folders_raises(tmp_path: Path) -> None:
    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    splits_path = tmp_path / "splits.yaml"

    _make_model_folder(model_root, name="Dataset901_NeuroVisionXBraTS21")
    _make_model_folder(model_root, name="Dataset901_SomeOtherRun")
    _make_images_ts(images_root)
    _make_splits(splits_path)

    args = nnunet_predict_session.parse_args(
        _base_argv(model_root, images_root, splits_path, tmp_path / "out", tmp_path / "s.json")
    )
    with pytest.raises(FileNotFoundError):
        nnunet_predict_session.run(args)


def test_missing_case_in_images_ts_raises(tmp_path: Path) -> None:
    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    splits_path = tmp_path / "splits.yaml"

    _make_model_folder(model_root)
    _make_images_ts(images_root, case_ids=_CASE_IDS[:1])  # second case missing
    _make_splits(splits_path, test_ids=_CASE_IDS)

    args = nnunet_predict_session.parse_args(
        _base_argv(model_root, images_root, splits_path, tmp_path / "out", tmp_path / "s.json")
    )
    with pytest.raises(ValueError, match=_CASE_IDS[1]):
        nnunet_predict_session.run(args)


def test_extra_case_in_images_ts_raises(tmp_path: Path) -> None:
    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    splits_path = tmp_path / "splits.yaml"
    extra_id = "BraTS2021_99999"

    _make_model_folder(model_root)
    _make_images_ts(images_root, case_ids=_CASE_IDS + [extra_id])
    _make_splits(splits_path, test_ids=_CASE_IDS)

    args = nnunet_predict_session.parse_args(
        _base_argv(model_root, images_root, splits_path, tmp_path / "out", tmp_path / "s.json")
    )
    with pytest.raises(ValueError, match=extra_id):
        nnunet_predict_session.run(args)


def test_file_ending_mismatch_raises(tmp_path: Path) -> None:
    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    splits_path = tmp_path / "splits.yaml"

    _make_model_folder(model_root, ending=".nii")
    _make_images_ts(images_root, ending=".nii.gz")  # mismatched with the model's ".nii"
    _make_splits(splits_path)

    args = nnunet_predict_session.parse_args(
        _base_argv(model_root, images_root, splits_path, tmp_path / "out", tmp_path / "s.json")
    )
    with pytest.raises(ValueError, match="file_ending"):
        nnunet_predict_session.run(args)


def test_dry_run_runs_no_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    splits_path = tmp_path / "splits.yaml"
    summary_path = tmp_path / "summary.json"

    _make_model_folder(model_root)
    _make_images_ts(images_root)
    _make_splits(splits_path)

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called in --dry-run")

    monkeypatch.setattr(nnunet_predict_session.subprocess, "run", fail_run)

    argv = _base_argv(
        model_root, images_root, splits_path, tmp_path / "out", summary_path, extra=["--dry-run"]
    )
    rc = nnunet_predict_session.main(argv)

    assert rc == 0
    summary = json.loads(summary_path.read_text())
    assert summary["dry_run"] is True
    assert summary["health"] == "OK"


def test_require_cuda_on_cpu_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nnunet_predict_session.torch.cuda, "is_available", lambda: False)

    model_root = tmp_path / "model_input"
    images_root = tmp_path / "images_input"
    splits_path = tmp_path / "splits.yaml"

    _make_model_folder(model_root)
    _make_images_ts(images_root)
    _make_splits(splits_path)

    argv = _base_argv(
        model_root,
        images_root,
        splits_path,
        tmp_path / "out",
        tmp_path / "s.json",
        extra=["--require-cuda"],
    )
    args = nnunet_predict_session.parse_args(argv)
    with pytest.raises(RuntimeError, match="require-cuda"):
        nnunet_predict_session.run(args)


# ---------------------------------------------------------------------------
# Post-prediction sanity checks (called directly -- no subprocess needed to
# exercise these; each targets exactly one check inside
# normalize_arm_output/check_arm_completeness/check_first_case_labels/
# find_images_ts_dir).
# ---------------------------------------------------------------------------


def test_check_arm_completeness_missing_case_raises(tmp_path: Path) -> None:
    # Only the first case's .nii.gz was written -- as if a fake nnU-Net run
    # silently dropped the second case.
    out_dir = tmp_path / "arm"
    out_dir.mkdir()
    (out_dir / f"{_CASE_IDS[0]}.nii.gz").write_bytes(b"not a real nifti, contents irrelevant")

    with pytest.raises(ValueError, match=_CASE_IDS[1]):
        nnunet_predict_session.check_arm_completeness(out_dir, _CASE_IDS)


def test_check_first_case_labels_label4_raises(tmp_path: Path) -> None:
    # nnU-Net's own convention for this dataset is {0, 1, 2, 3} (1=ED, 2=NCR,
    # 3=ET); label 4 is BraTS's raw ET label, which must never appear in an
    # nnU-Net prediction that has already gone through the project's remap.
    out_dir = tmp_path / "arm"
    out_dir.mkdir()
    case_id = _CASE_IDS[0]
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[0, 0, 0] = 4
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(out_dir / f"{case_id}.nii.gz"))

    with pytest.raises(ValueError, match=r"\[4\]"):
        nnunet_predict_session.check_first_case_labels(out_dir, [case_id])


def test_normalize_arm_output_gzip_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The production code's only non-trivial step in the gzip round trip is
    # the byte copy (`shutil.copyfileobj`) -- monkeypatching THAT to write a
    # different array's bytes into the .nii.gz (instead of a faithful copy of
    # the .nii it was handed) is the cleanest seam to make the round-trip
    # verification actually see a mismatch, without touching production code.
    out_dir = tmp_path / "arm"
    out_dir.mkdir()
    case_id = _CASE_IDS[0]
    nii_path = out_dir / f"{case_id}.nii"
    _make_nifti(nii_path, value=1)

    other_path = tmp_path / "other.nii"
    _make_nifti(other_path, value=9)
    other_bytes = other_path.read_bytes()

    def fake_copyfileobj(f_in: object, f_out: object, *args: object, **kwargs: object) -> None:
        f_out.write(other_bytes)  # a DIFFERENT array's bytes, not a copy of f_in.

    monkeypatch.setattr(nnunet_predict_session.shutil, "copyfileobj", fake_copyfileobj)

    with pytest.raises(AssertionError, match="Gzip round trip"):
        nnunet_predict_session.normalize_arm_output(out_dir, ".nii", [case_id])


def test_find_images_ts_dir_two_candidates_raises(tmp_path: Path) -> None:
    root = tmp_path / "images_input"
    (root / "run_a" / "imagesTs").mkdir(parents=True)
    (root / "run_b" / "imagesTs").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="found 2"):
        nnunet_predict_session.find_images_ts_dir(root)
