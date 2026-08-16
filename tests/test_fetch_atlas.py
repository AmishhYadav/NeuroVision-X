"""Tests for scripts/fetch_atlas.py.

No test in this module ever reaches the real network: every fixture archive
is a tiny local zip served through a `file://` URL, and an autouse fixture
below rewires `urllib.request.urlopen` to raise if anything ever asks it for
a non-`file://` URL. That guard is itself asserted directly in
`test_no_real_network_calls_are_made`.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import pytest
import scripts.fetch_atlas as fetch_atlas_mod
from omegaconf import OmegaConf

ArchiveSpec = fetch_atlas_mod.ArchiveSpec
REQUIRED_MEMBERS = fetch_atlas_mod.REQUIRED_MEMBERS


# --------------------------------------------------------------------------
# Network guard -- applies to every test in this module.
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch):
    """Rewires urlopen to allow only `file://` URLs for the whole module.

    Anything reaching for a real (http/https) URL raises `AssertionError`
    immediately, instead of the test hanging or actually hitting a server.
    """
    real_urlopen = urllib.request.urlopen

    def guarded_urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else url.full_url
        if not target.startswith("file://"):
            raise AssertionError(f"Test attempted a real network request to {target!r}")
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)
    yield


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


def _make_archive_zip(path: Path, members: dict[str, bytes]) -> None:
    """Writes a small zip file at `path` with the given member contents."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_members() -> dict[str, dict[str, bytes]]:
    """A complete, valid set of fixture members: one dict per archive name.

    Together these cover exactly `REQUIRED_MEMBERS`, nested under `sri24/`
    -- mirroring the real archives, which each extract into one shared
    top-level `sri24/` directory (see `configs/anatomy/sri24.yaml`).
    """
    return {
        "labels": {
            "sri24/tzo116plus.nii": b"tzo-label-volume",
            "sri24/SRI24-tzo116plus.txt": b"1 Precentral_L 255 0 0 0\n",
        },
        "tissue": {
            "sri24/tissues.nii": b"tissue-hard-segmentation",
            "sri24/pbmap_GM.nii": b"gm-probability-map",
            "sri24/pbmap_WM.nii": b"wm-probability-map",
            "sri24/pbmap_CSF.nii": b"csf-probability-map",
        },
        "anatomy": {
            "sri24/spgr.nii": b"t1-anatomy-volume",
            "sri24/LICENSE": b"CC-BY-SA 4.0",
        },
    }


def _build_archives(tmp_path: Path, members_by_archive: dict[str, dict[str, bytes]]) -> dict:
    """Writes one zip per archive under tmp_path/fixtures.

    Returns:
        `{archive_name: {"url": ..., "sha256": ..., "bytes": ...}}`, ready
        to drop straight into a `cfg.anatomy.archives`-shaped mapping.
    """
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    info = {}
    for name, members in members_by_archive.items():
        zip_path = fixtures_dir / f"{name}.zip"
        _make_archive_zip(zip_path, members)
        info[name] = {
            "url": zip_path.resolve().as_uri(),
            "sha256": _sha256_of_file(zip_path),
            "bytes": zip_path.stat().st_size,
        }
    return info


def _build_cfg(tmp_path: Path, archive_info: dict, *, atlas_dir: Path | None = None):
    """Builds an OmegaConf tree shaped like `cfg.anatomy` in the real config."""
    if atlas_dir is None:
        atlas_dir = tmp_path / "atlas"
    return OmegaConf.create(
        {
            "dir": str(atlas_dir),
            "subdir": "sri24",
            "version": "2.0",
            "source": "NITRC group_id=214 (test fixture)",
            "licence": "CC-BY-SA",
            "archives": {
                name: {"url": spec["url"], "sha256": spec["sha256"], "bytes": spec["bytes"]}
                for name, spec in archive_info.items()
            },
        }
    )


def _spec_for(archive_info: dict, name: str) -> ArchiveSpec:
    spec = archive_info[name]
    return ArchiveSpec(name=name, url=spec["url"], sha256=spec["sha256"], size_bytes=spec["bytes"])


# --------------------------------------------------------------------------
# 1. Happy path
# --------------------------------------------------------------------------


def test_fetch_atlas_happy_path(tmp_path: Path) -> None:
    archives = _build_archives(tmp_path, _default_members())
    cfg = _build_cfg(tmp_path, archives)

    atlas_dir = fetch_atlas_mod.fetch_atlas(cfg)

    assert atlas_dir == Path(cfg.dir) / "sri24"
    for member in REQUIRED_MEMBERS:
        assert (atlas_dir / member).is_file(), f"missing {member}"


# --------------------------------------------------------------------------
# 2. Checksum mismatch
# --------------------------------------------------------------------------


def test_checksum_mismatch_raises_and_extracts_nothing(tmp_path: Path) -> None:
    archives = _build_archives(tmp_path, _default_members())
    archives["labels"]["sha256"] = "0" * 64  # deliberately wrong
    cfg = _build_cfg(tmp_path, archives)

    with pytest.raises(ValueError, match="labels"):
        fetch_atlas_mod.fetch_atlas(cfg)

    atlas_dir = Path(cfg.dir) / cfg.subdir
    assert not atlas_dir.exists()


# --------------------------------------------------------------------------
# 3. Size mismatch, checked before the hash
# --------------------------------------------------------------------------


def test_verify_size_mismatch_raises_with_correct_hash(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"hello world")
    correct_sha256 = hashlib.sha256(b"hello world").hexdigest()
    # Hash is right; size is wrong. If size were not checked first (or at
    # all), this would pass verification -- it does not.
    spec = ArchiveSpec(name="labels", url="file:///unused", sha256=correct_sha256, size_bytes=999)

    with pytest.raises(ValueError, match="size mismatch"):
        fetch_atlas_mod.verify(path, spec)


# --------------------------------------------------------------------------
# 4. Idempotent
# --------------------------------------------------------------------------


def test_fetch_archive_is_cached_on_second_call(tmp_path: Path) -> None:
    archives = _build_archives(tmp_path, {"labels": _default_members()["labels"]})
    spec = _spec_for(archives, "labels")
    cache_dir = tmp_path / "cache"

    first = fetch_atlas_mod.fetch_archive(spec, cache_dir)
    assert first.action == "downloaded"

    second = fetch_atlas_mod.fetch_archive(spec, cache_dir)
    assert second.action == "cached"


def test_second_fetch_atlas_call_makes_no_download_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archives = _build_archives(tmp_path, _default_members())
    cfg = _build_cfg(tmp_path, archives)

    fetch_atlas_mod.fetch_atlas(cfg)

    call_urls: list[str] = []
    real_download = fetch_atlas_mod.download

    def counting_download(url, dest, **kwargs):
        call_urls.append(url)
        return real_download(url, dest, **kwargs)

    monkeypatch.setattr(fetch_atlas_mod, "download", counting_download)

    fetch_atlas_mod.fetch_atlas(cfg)

    assert call_urls == []


# --------------------------------------------------------------------------
# 5. force=True re-downloads
# --------------------------------------------------------------------------


def test_force_redownloads_even_with_valid_cache(tmp_path: Path) -> None:
    archives = _build_archives(tmp_path, {"labels": _default_members()["labels"]})
    spec = _spec_for(archives, "labels")
    cache_dir = tmp_path / "cache"

    fetch_atlas_mod.fetch_archive(spec, cache_dir)
    result = fetch_atlas_mod.fetch_archive(spec, cache_dir, force=True)

    assert result.action == "downloaded"


# --------------------------------------------------------------------------
# 6. Corrupt cache is re-downloaded once, then succeeds
# --------------------------------------------------------------------------


def test_corrupt_cache_is_redownloaded_once(tmp_path: Path) -> None:
    archives = _build_archives(tmp_path, {"labels": _default_members()["labels"]})
    spec = _spec_for(archives, "labels")
    cache_dir = tmp_path / "cache"

    fetch_atlas_mod.fetch_archive(spec, cache_dir)  # populate a good cache
    cached_path = cache_dir / "labels.zip"
    cached_path.write_bytes(b"corrupted-garbage-not-a-valid-zip")

    result = fetch_atlas_mod.fetch_archive(spec, cache_dir)

    assert result.action == "downloaded"
    fetch_atlas_mod.verify(cached_path, spec)  # does not raise


# --------------------------------------------------------------------------
# 7. Zip traversal is blocked
# --------------------------------------------------------------------------


def test_safe_extract_blocks_relative_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil_relative.zip"
    _make_archive_zip(zip_path, {"../evil.txt": b"pwned"})
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    with pytest.raises(ValueError, match="traversal"):
        fetch_atlas_mod.safe_extract(zip_path, dest_dir)

    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_blocks_absolute_member_name(tmp_path: Path) -> None:
    # An absolute path *inside* tmp_path -- still absolute (which is what is
    # under test), but never touches anywhere outside the sandboxed tmp_path
    # even if the guard under test had a bug.
    outside_target = tmp_path / "outside_evil.txt"
    zip_path = tmp_path / "evil_absolute.zip"
    _make_archive_zip(zip_path, {str(outside_target): b"pwned"})
    dest_dir = tmp_path / "dest2"
    dest_dir.mkdir()

    with pytest.raises(ValueError, match="absolute"):
        fetch_atlas_mod.safe_extract(zip_path, dest_dir)

    assert not outside_target.exists()


def test_safe_extract_blocks_symlink_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil_symlink.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        info = zipfile.ZipInfo("sneaky_link")
        info.create_system = 3  # Unix, so external_attr holds a file mode
        info.external_attr = 0o120777 << 16  # S_IFLNK
        zf.writestr(info, "/etc/passwd")
    dest_dir = tmp_path / "dest3"
    dest_dir.mkdir()

    with pytest.raises(ValueError, match="symlink"):
        fetch_atlas_mod.safe_extract(zip_path, dest_dir)

    assert list(dest_dir.iterdir()) == []


# --------------------------------------------------------------------------
# 8. Missing required member
# --------------------------------------------------------------------------


def test_fetch_atlas_raises_on_missing_required_member(tmp_path: Path) -> None:
    members = _default_members()
    del members["labels"]["sri24/tzo116plus.nii"]
    archives = _build_archives(tmp_path, members)
    cfg = _build_cfg(tmp_path, archives)

    with pytest.raises(FileNotFoundError, match="tzo116plus.nii"):
        fetch_atlas_mod.fetch_atlas(cfg)


# --------------------------------------------------------------------------
# 9. PROVENANCE.json
# --------------------------------------------------------------------------


def test_provenance_file_is_written(tmp_path: Path) -> None:
    archives = _build_archives(tmp_path, _default_members())
    cfg = _build_cfg(tmp_path, archives)

    fetch_atlas_mod.fetch_atlas(cfg)

    provenance_path = Path(cfg.dir) / "PROVENANCE.json"
    assert provenance_path.is_file()

    data = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert data["version"] == "2.0"
    assert data["source"]
    assert data["licence"] == "CC-BY-SA"
    assert "fetched_at_utc" in data and data["fetched_at_utc"]

    for name, spec in archives.items():
        assert data["archives"][name]["sha256"] == spec["sha256"]
        assert data["archives"][name]["action"] == "downloaded"


# --------------------------------------------------------------------------
# 10. Atomic download
# --------------------------------------------------------------------------


def test_atomic_download_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "out" / "archive.zip"

    class _BrokenResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, n: int = -1) -> bytes:
            raise OSError("simulated failure partway through the download")

    def fake_urlopen(url, *args, **kwargs):
        return _BrokenResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(OSError):
        fetch_atlas_mod.download("file:///does/not/matter", dest)

    assert not dest.exists()
    # dest.parent is created by download() itself; it must be left empty,
    # including of the hidden tempfile.mkstemp() temp file.
    assert os.listdir(dest.parent) == []


# --------------------------------------------------------------------------
# 11. No real network anywhere in this module
# --------------------------------------------------------------------------


def test_no_real_network_calls_are_made(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_urlopen = urllib.request.urlopen

    def spy_urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else url.full_url
        calls.append(target)
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", spy_urlopen)

    archives = _build_archives(tmp_path, _default_members())
    cfg = _build_cfg(tmp_path, archives)
    fetch_atlas_mod.fetch_atlas(cfg)

    assert calls, "expected fetch_atlas to call urlopen at least once"
    assert all(url.startswith("file://") for url in calls)
