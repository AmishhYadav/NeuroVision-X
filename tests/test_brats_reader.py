"""Tests for neurovision.data.brats: BraTS directory scanning and indexing.

All tests build synthetic case directories under pytest's `tmp_path` — never
real BraTS data — and touch empty placeholder files, since this module only
resolves paths and never opens a volume. See CLAUDE.md for the project-wide
testing rules this suite follows.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neurovision.data.brats import BratsCase, scan_brats_root, write_case_index

# --- helpers ---


_DEFAULT_ROLES = ("t1", "t1ce", "t2", "flair", "seg")


def _make_case_2020(root: Path, case_id: str, roles: tuple[str, ...] = _DEFAULT_ROLES) -> Path:
    """Create a 2020/2021-style case directory with the given roles present."""
    suffixes = {"t1": "_t1", "t1ce": "_t1ce", "t2": "_t2", "flair": "_flair", "seg": "_seg"}
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for role in roles:
        (case_dir / f"{case_id}{suffixes[role]}.nii.gz").touch()
    return case_dir


def _make_case_2023(root: Path, case_id: str, roles: tuple[str, ...] = _DEFAULT_ROLES) -> Path:
    """Create a 2023+-style case directory with the given roles present."""
    suffixes = {"t1": "-t1n", "t1ce": "-t1c", "t2": "-t2w", "flair": "-t2f", "seg": "-seg"}
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for role in roles:
        (case_dir / f"{case_id}{suffixes[role]}.nii.gz").touch()
    return case_dir


# --- scanning: basic conventions ---


def test_scans_2020_style_root(tmp_path: Path):
    _make_case_2020(tmp_path, "BraTS20_Training_001")
    _make_case_2020(tmp_path, "BraTS20_Training_002")
    cases = scan_brats_root(tmp_path)
    assert len(cases) == 2
    assert [c.case_id for c in cases] == ["BraTS20_Training_001", "BraTS20_Training_002"]


def test_scans_2020_style_root_sorted_order(tmp_path: Path):
    # Create out of alphabetical order to prove the scan sorts, not just
    # happens to preserve creation order.
    _make_case_2020(tmp_path, "BraTS20_Training_010")
    _make_case_2020(tmp_path, "BraTS20_Training_002")
    _make_case_2020(tmp_path, "BraTS20_Training_001")
    cases = scan_brats_root(tmp_path)
    assert [c.case_id for c in cases] == [
        "BraTS20_Training_001",
        "BraTS20_Training_002",
        "BraTS20_Training_010",
    ]


def test_scans_2023_style_root_and_maps_modalities(tmp_path: Path):
    _make_case_2023(tmp_path, "BraTS-GLI-00000-000")
    cases = scan_brats_root(tmp_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "BraTS-GLI-00000-000"
    assert case.t1.name == "BraTS-GLI-00000-000-t1n.nii.gz"
    assert case.t1ce.name == "BraTS-GLI-00000-000-t1c.nii.gz"
    assert case.t2.name == "BraTS-GLI-00000-000-t2w.nii.gz"
    assert case.flair.name == "BraTS-GLI-00000-000-t2f.nii.gz"
    assert case.seg is not None
    assert case.seg.name == "BraTS-GLI-00000-000-seg.nii.gz"


def test_t1_and_t1ce_do_not_collide_regression(tmp_path: Path):
    """Regression test: naive substring/glob matching would confuse t1/t1ce."""
    _make_case_2020(tmp_path, "BraTS20_Training_003")
    cases = scan_brats_root(tmp_path)
    case = cases[0]
    assert case.t1.name.endswith("_t1.nii.gz")
    assert case.t1ce.name.endswith("_t1ce.nii.gz")
    assert case.t1 != case.t1ce
    assert not case.t1.name.endswith("_t1ce.nii.gz")


def test_mixed_root_resolves_both_conventions(tmp_path: Path):
    _make_case_2020(tmp_path, "BraTS20_Training_001")
    _make_case_2023(tmp_path, "BraTS-GLI-00000-000")
    cases = scan_brats_root(tmp_path)
    assert len(cases) == 2
    ids = {c.case_id for c in cases}
    assert ids == {"BraTS20_Training_001", "BraTS-GLI-00000-000"}


def test_modality_paths_order(tmp_path: Path):
    case_dir = _make_case_2020(tmp_path, "BraTS20_Training_001")
    cases = scan_brats_root(tmp_path)
    case = cases[0]
    paths = case.modality_paths
    assert len(paths) == 4
    assert paths == [case.t1, case.t1ce, case.t2, case.flair]
    assert paths[0].parent == case_dir


# --- errors: missing files ---


def test_missing_single_role_raises_with_case_id_and_role(tmp_path: Path):
    _make_case_2020(tmp_path, "BraTS20_Training_005", roles=("t1", "t2", "flair", "seg"))
    with pytest.raises(ValueError) as exc_info:
        scan_brats_root(tmp_path)
    message = str(exc_info.value)
    assert "BraTS20_Training_005" in message
    assert "t1ce" in message


def test_multiple_incomplete_cases_all_reported(tmp_path: Path):
    _make_case_2020(tmp_path, "BraTS20_Training_005", roles=("t1", "t2", "flair"))
    _make_case_2020(tmp_path, "BraTS20_Training_012", roles=("t1", "t1ce", "t2", "seg"))
    with pytest.raises(ValueError) as exc_info:
        scan_brats_root(tmp_path)
    message = str(exc_info.value)
    assert "BraTS20_Training_005" in message
    assert "BraTS20_Training_012" in message
    assert "flair" in message


def test_require_seg_false_allows_missing_seg(tmp_path: Path):
    _make_case_2020(tmp_path, "BraTS20_Training_001", roles=("t1", "t1ce", "t2", "flair"))
    cases = scan_brats_root(tmp_path, require_seg=False)
    assert len(cases) == 1
    assert cases[0].seg is None


def test_require_seg_true_raises_when_seg_missing(tmp_path: Path):
    _make_case_2020(tmp_path, "BraTS20_Training_001", roles=("t1", "t1ce", "t2", "flair"))
    with pytest.raises(ValueError) as exc_info:
        scan_brats_root(tmp_path, require_seg=True)
    assert "seg" in str(exc_info.value)


def test_require_seg_false_still_requires_all_four_modalities(tmp_path: Path):
    _make_case_2020(tmp_path, "BraTS20_Training_001", roles=("t1", "t1ce", "t2"))
    with pytest.raises(ValueError) as exc_info:
        scan_brats_root(tmp_path, require_seg=False)
    assert "flair" in str(exc_info.value)


# --- edge cases ---


def test_nonexistent_root_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        scan_brats_root("/this/path/does/not/exist/anywhere")


def test_empty_root_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError):
        scan_brats_root(tmp_path)


def test_hidden_and_underscore_dirs_are_skipped(tmp_path: Path):
    _make_case_2020(tmp_path, "BraTS20_Training_001")
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / "_private_dir").mkdir()
    cases = scan_brats_root(tmp_path)
    assert len(cases) == 1
    assert cases[0].case_id == "BraTS20_Training_001"


def test_plain_nii_fallback_when_no_gz(tmp_path: Path):
    case_id = "BraTS20_Training_001"
    case_dir = tmp_path / case_id
    case_dir.mkdir()
    for suffix in ("_t1", "_t1ce", "_t2", "_flair", "_seg"):
        (case_dir / f"{case_id}{suffix}.nii").touch()
    cases = scan_brats_root(tmp_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.t1.name == f"{case_id}_t1.nii"
    assert case.seg is not None
    assert case.seg.name == f"{case_id}_seg.nii"


def test_gz_preferred_over_plain_nii_when_both_present(tmp_path: Path):
    case_id = "BraTS20_Training_001"
    case_dir = _make_case_2020(tmp_path, case_id)
    (case_dir / f"{case_id}_t1.nii").touch()
    cases = scan_brats_root(tmp_path)
    assert cases[0].t1.name == f"{case_id}_t1.nii.gz"


# --- write_case_index ---


def test_write_case_index_writes_header_and_rows(tmp_path: Path):
    case_dir = _make_case_2020(tmp_path, "BraTS20_Training_001")
    cases = scan_brats_root(tmp_path)
    out_path = tmp_path / "index" / "cases.csv"
    result = write_case_index(cases, out_path)
    assert result == out_path
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["case_id", "t1", "t1ce", "t2", "flair", "seg"]
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "BraTS20_Training_001"
    assert rows[0]["t1"] == str(case_dir / "BraTS20_Training_001_t1.nii.gz")


def test_write_case_index_none_seg_is_empty_field(tmp_path: Path):
    case = BratsCase(
        case_id="X",
        t1=tmp_path / "X_t1.nii.gz",
        t1ce=tmp_path / "X_t1ce.nii.gz",
        t2=tmp_path / "X_t2.nii.gz",
        flair=tmp_path / "X_flair.nii.gz",
        seg=None,
    )
    out_path = tmp_path / "cases.csv"
    write_case_index([case], out_path)
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["seg"] == ""


def test_write_case_index_creates_missing_parent_dir(tmp_path: Path):
    case = BratsCase(
        case_id="X",
        t1=tmp_path / "X_t1.nii.gz",
        t1ce=tmp_path / "X_t1ce.nii.gz",
        t2=tmp_path / "X_t2.nii.gz",
        flair=tmp_path / "X_flair.nii.gz",
        seg=None,
    )
    out_path = tmp_path / "a" / "b" / "c" / "cases.csv"
    assert not out_path.parent.exists()
    write_case_index([case], out_path)
    assert out_path.is_file()
