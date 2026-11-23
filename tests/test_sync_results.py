"""Tests for scripts/sync_results.py. Synthetic files in tmp_path only."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_results.py"
_spec = importlib.util.spec_from_file_location("sync_results", _SCRIPT)
sr = importlib.util.module_from_spec(_spec)
sys.modules["sync_results"] = sr
_spec.loader.exec_module(sr)

BIG = 10**6
EXT = [".csv", ".json"]


def _write(root: Path, rel: str, text: str = "a,b\n1,2\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _select(src, ext=EXT, max_bytes=BIG, names=(), globs=()):
    return sr.select_files(src, list(ext), max_bytes, list(names), list(globs))


@pytest.fixture
def dirs(tmp_path):
    src = tmp_path / "outputs"
    src.mkdir()
    return src, tmp_path / "results"


def test_small_csv_copied_identically(dirs):
    src, dst = dirs
    _write(src, "run/summary.csv", "x\n1\n")
    sel = _select(src)
    report = sr.sync(src, dst, sel, dry_run=False)
    assert report.copied == 1
    assert (dst / "run/summary.csv").read_text() == "x\n1\n"


def test_oversize_skipped(dirs):
    src, _ = dirs
    _write(src, "big.csv", "x" * 100)
    _write(src, "small.csv", "x")
    sel = _select(src, max_bytes=10)
    assert sel.selected == [Path("small.csv")]
    assert sel.skipped_oversize == [(Path("big.csv"), 100)]


def test_deny_list_overrides_allowed_extensions(dirs):
    src, dst = dirs
    _write(src, "a.npy")
    _write(src, "vol.nii.gz")
    _write(src, "ok.csv")
    sel = _select(src, ext=[".npy", ".gz", ".csv"])
    assert sel.selected == [Path("ok.csv")]
    assert Path("vol.nii.gz") in sel.skipped_denied
    sr.sync(src, dst, sel, dry_run=False)
    assert not (dst / "a.npy").exists()
    assert not (dst / "vol.nii.gz").exists()


def test_denied_with_unallowed_suffix_is_silent(dirs):
    src, _ = dirs
    _write(src, "slice.dcm")
    sel = _select(src)
    assert sel.skipped_denied == []


def test_exclude_dir_names_exact_component(dirs):
    src, _ = dirs
    _write(src, "a/logits/x.csv")
    _write(src, "a/logits_summary.csv")
    _write(src, "eval_test/summary.csv")
    sel = _select(src, names=["logits", "test"])
    assert sel.selected == [Path("a/logits_summary.csv"), Path("eval_test/summary.csv")]


def test_exclude_glob(dirs):
    src, _ = dirs
    _write(src, "report_gt/reports/c1.json", "{}")
    _write(src, "report_gt/summary.csv")
    sel = _select(src, globs=["report_*/reports/*"])
    assert sel.selected == [Path("report_gt/summary.csv")]


def test_symlinked_file_skipped(dirs):
    src, _ = dirs
    real = _write(src, "real.csv")
    (src / "link.csv").symlink_to(real)
    assert _select(src).selected == [Path("real.csv")]


def test_idempotent_and_manifest_stable(dirs):
    src, dst = dirs
    _write(src, "a/one.csv", "1")
    _write(src, "b.json", "{}")
    sel = _select(src)
    first = sr.sync(src, dst, sel, dry_run=False)
    m1 = sr.write_manifest(dst, "MANIFEST.sha256").read_bytes()
    second = sr.sync(src, dst, sel, dry_run=False)
    m2 = sr.write_manifest(dst, "MANIFEST.sha256").read_bytes()
    assert second.copied == 0
    assert second.unchanged == first.copied == 2
    assert m1 == m2


def test_modified_source_recopied(dirs):
    src, dst = dirs
    f = _write(src, "a.csv", "old")
    sr.sync(src, dst, _select(src), dry_run=False)
    f.write_text("new")
    report = sr.sync(src, dst, _select(src), dry_run=False)
    assert report.copied == 1
    assert (dst / "a.csv").read_text() == "new"


def test_dry_run_writes_nothing(dirs):
    src, dst = dirs
    _write(src, "a.csv")
    report = sr.sync(src, dst, _select(src), dry_run=True)
    assert report.copied == 1
    assert not dst.exists()


def test_manifest_format(dirs):
    src, dst = dirs
    _write(src, "z.csv", "1")
    _write(src, "a/b.csv", "2")
    sr.sync(src, dst, _select(src), dry_run=False)
    manifest = sr.write_manifest(dst, "MANIFEST.sha256")
    text = manifest.read_text()
    assert text.endswith("\n")
    lines = text.splitlines()
    assert all(re.fullmatch(r"[0-9a-f]{64}  \S+", ln) for ln in lines)
    paths = [ln.split("  ", 1)[1] for ln in lines]
    assert paths == ["a/b.csv", "z.csv"]
