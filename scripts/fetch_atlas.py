"""Fetches the SRI24 atlas from NITRC into a configured directory.

The SRI24 atlas backs the interpretable pipeline (Phase 0 onward, see
`docs/research/interpretable_pipeline_plan.md` and
`docs/research/phase0_atlas_findings.md`), but it is licensed **CC-BY-SA** --
so unlike BraTS-derived artifacts, it must never be vendored into the repo.
This script downloads it at setup time instead, verifies every archive
against a pinned SHA-256 (a silently different atlas would produce plausible
reports about the wrong anatomy, which is unrecoverable downstream), extracts
it, and records provenance so a report can be traced back to the exact
parcellation that produced it.

Example usage:

    python scripts/fetch_atlas.py
    python scripts/fetch_atlas.py +force=true

This is a CLI entry point, so `main` prints a short human-facing summary to
stdout on purpose -- the no-bare-print rule applies to library code, not to
scripts whose whole job is producing terminal output. Everything else in this
module logs through `logging`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from neurovision.utils.io import ensure_dir, format_size
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and
# on any machine -- no absolute paths. Copied from scripts/preprocess.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Every file the rest of the interpretable pipeline needs to already be on
# disk before it can run. Checked once, after all three archives are
# extracted, since no single archive holds all of them.
REQUIRED_MEMBERS: tuple[str, ...] = (
    "tzo116plus.nii",
    "SRI24-tzo116plus.txt",
    "tissues.nii",
    "pbmap_GM.nii",
    "pbmap_WM.nii",
    "pbmap_CSF.nii",
    "spgr.nii",
    "LICENSE",
)

PROVENANCE_FILENAME = "PROVENANCE.json"

_LICENCE_WARNING = (
    "SRI24 is licensed CC-BY-SA. Do not commit or redistribute it without "
    "share-alike attribution."
)


@dataclass(frozen=True)
class ArchiveSpec:
    """One downloadable SRI24 archive, as declared in `configs/anatomy/sri24.yaml`.

    Attributes:
        name: Archive key (`"labels"`, `"tissue"`, or `"anatomy"`).
        url: Download URL.
        sha256: Expected SHA-256 hex digest of the downloaded `.zip`.
        size_bytes: Expected size of the downloaded `.zip`, in bytes.
    """

    name: str
    url: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class FetchResult:
    """Outcome of fetching one archive.

    Attributes:
        name: Archive key, matching the `ArchiveSpec` it came from.
        action: What actually happened this call. `"downloaded"` when the
            archive was freshly pulled over the network; `"cached"` when an
            already-downloaded copy existed and passed verification, so no
            network request was made. `"reused"` is reserved by this API for
            a future caller that also skips re-extraction of an
            already-extracted archive -- `fetch_archive` itself never
            returns it, since caching only concerns the downloaded `.zip`.
        path: Path to the verified `.zip` on disk (inside the cache dir).
        sha256: The archive's verified SHA-256 hex digest (equal to the
            spec's expected value -- verification would have raised
            otherwise).
    """

    name: str
    action: str
    path: Path
    sha256: str


def archive_specs(cfg: DictConfig) -> list[ArchiveSpec]:
    """Builds the list of `ArchiveSpec` declared in an anatomy config.

    Args:
        cfg: The `anatomy` config sub-tree (e.g. `hydra_cfg.anatomy`), which
            carries an `archives` mapping of name -> `{url, sha256, bytes}`.

    Returns:
        One `ArchiveSpec` per entry in `cfg.archives`, in config order.
    """
    return [
        ArchiveSpec(name=name, url=spec.url, sha256=spec.sha256, size_bytes=int(spec.bytes))
        for name, spec in cfg.archives.items()
    ]


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Computes the SHA-256 hex digest of a file, streamed in chunks.

    Args:
        path: File to hash.
        chunk_size: Bytes read per iteration. Streaming (rather than reading
            the whole file into memory) matters here -- these archives run
            into the tens of megabytes.

    Returns:
        The hex digest string.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, dest: Path, *, timeout: float = 300.0) -> None:
    """Downloads `url` to `dest`, atomically.

    Streams to a temp file in `dest`'s own directory, then swaps it into
    place with `os.replace` -- the same pattern
    `neurovision.training.checkpoint._atomic_torch_save` uses, and for the
    same reason: `os.replace` is only atomic within one filesystem, and a
    process killed mid-download must never leave a partial file that has
    already replaced (or masquerades as) a good one. The temp file is
    removed on any failure.

    Args:
        url: Source URL. Also accepts `file://` URLs, which is how tests in
            this project exercise this function without touching a network.
        dest: Final destination path.
        timeout: Seconds to wait on the connection, forwarded to
            `urllib.request.urlopen`.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            with open(tmp_path, "wb") as out:
                shutil.copyfileobj(response, out)
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info("Downloaded %s -> %s", url, dest)


def verify(path: Path, spec: ArchiveSpec) -> None:
    """Verifies a downloaded archive against its expected size and checksum.

    Size is checked first because it is nearly free, before paying for a
    full-file SHA-256 pass over a file that is already known to be wrong.

    Args:
        path: Path to the downloaded archive.
        spec: The `ArchiveSpec` it is supposed to match.

    Raises:
        ValueError: If the size or checksum does not match. A mismatch here
            is a hard failure, never a warning -- a silently different atlas
            produces plausible-looking reports about the wrong anatomy.
    """
    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        raise ValueError(
            f"Archive '{spec.name}' size mismatch: expected {spec.size_bytes} bytes, "
            f"got {actual_size} bytes ({path})."
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"Archive '{spec.name}' checksum mismatch: expected sha256={spec.sha256}, "
            f"got sha256={actual_sha256} ({path})."
        )


def _reject_unsafe_member(name: str, info: zipfile.ZipInfo) -> None:
    """Raises if a zip member name/mode is unsafe to extract.

    Args:
        name: The member's `filename` from the zip's central directory.
        info: The member's full `ZipInfo`, used to detect symlinks.

    Raises:
        ValueError: If the member name is absolute, or the member is a
            symlink.
    """
    if os.path.isabs(name):
        raise ValueError(f"Zip member '{name}' has an absolute path; refusing to extract.")

    # The top 16 bits of external_attr hold the Unix file mode when the
    # archive was created on a Unix system (info.create_system == 3). A
    # symlink member could point anywhere on the filesystem the extraction
    # process can reach, entirely outside the member-name traversal check
    # below.
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode):
        raise ValueError(f"Zip member '{name}' is a symlink; refusing to extract.")


def safe_extract(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extracts a zip archive, guarding against path traversal and symlinks.

    Every member's target path is resolved and checked to be inside
    `dest_dir` *before* anything is written, and absolute member names or
    symlink members are rejected outright. This runs on files fetched over
    the network, so the guard is not optional -- a crafted `../../` member
    name could otherwise overwrite an arbitrary file on the machine running
    this script.

    Args:
        zip_path: Path to the `.zip` file to extract.
        dest_dir: Directory to extract into. Created if it does not exist.

    Returns:
        Paths to every extracted file (directories are created but not
        included in the returned list).

    Raises:
        ValueError: If any member would extract outside `dest_dir`, has an
            absolute name, or is a symlink. Raised before any member of
            *this* zip is written to disk.
    """
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        # Two passes: validate every member first, write only afterwards.
        # This is what makes a traversal attempt fail before touching disk
        # at all, rather than after some earlier, legitimate member of the
        # same archive was already extracted.
        planned: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in zf.infolist():
            name = info.filename
            _reject_unsafe_member(name, info)

            target = (dest_dir / name).resolve()
            try:
                target.relative_to(dest_dir)
            except ValueError as exc:
                raise ValueError(
                    f"Zip member '{name}' escapes destination directory {dest_dir} "
                    f"(path traversal in {zip_path})."
                ) from exc
            planned.append((info, target))

        extracted: list[Path] = []
        for info, target in planned:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(target)

    logger.info("Extracted %d file(s) from %s into %s", len(extracted), zip_path, dest_dir)
    return extracted


def fetch_archive(spec: ArchiveSpec, cache_dir: Path, *, force: bool = False) -> FetchResult:
    """Fetches one archive into `cache_dir`, downloading only when necessary.

    If a cached copy already exists and passes `verify`, it is reused with
    no network request -- this is what makes the script safely re-runnable.
    If a cached copy exists but fails verification, it is treated as
    corrupt: a warning is logged and it is re-downloaded once. `force=True`
    always re-downloads regardless of what is cached.

    Args:
        spec: The archive to fetch.
        cache_dir: Directory the downloaded `.zip` is cached under.
        force: If True, re-download even if a valid cached copy exists.

    Returns:
        A `FetchResult` describing what happened and where the verified
        `.zip` now lives.

    Raises:
        ValueError: If the freshly downloaded archive still fails
            verification (a corrupt or unexpected upstream file).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{spec.name}.zip"

    if not force and dest.is_file():
        try:
            verify(dest, spec)
        except ValueError:
            logger.warning(
                "Cached archive '%s' at %s failed verification; re-downloading.",
                spec.name,
                dest,
            )
        else:
            logger.info("Archive '%s' already cached and verified at %s", spec.name, dest)
            return FetchResult(name=spec.name, action="cached", path=dest, sha256=spec.sha256)

    download(spec.url, dest)
    verify(dest, spec)
    return FetchResult(name=spec.name, action="downloaded", path=dest, sha256=spec.sha256)


def write_provenance(cfg: DictConfig, dest: Path, results: Sequence[FetchResult]) -> Path:
    """Writes `PROVENANCE.json` recording exactly what atlas is on disk.

    A generated report must be traceable to the exact parcellation that
    produced it, months later -- a terminal log nobody kept does not do
    that.

    Args:
        cfg: The `anatomy` config sub-tree (see `archive_specs`).
        dest: Directory to write `PROVENANCE.json` into -- the parent of the
            extracted `sri24/` directory, i.e. `cfg.dir`.
        results: One `FetchResult` per archive that was fetched.

    Returns:
        Path to the written `PROVENANCE.json`.
    """
    dest = ensure_dir(dest)
    results_by_name = {result.name: result for result in results}

    archives_info = {}
    for spec in archive_specs(cfg):
        result = results_by_name.get(spec.name)
        archives_info[spec.name] = {
            "url": spec.url,
            "sha256": spec.sha256,
            "action": result.action if result is not None else None,
        }

    provenance = {
        "version": cfg.version,
        "source": cfg.source,
        "licence": cfg.licence,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "archives": archives_info,
    }

    provenance_path = dest / PROVENANCE_FILENAME
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote atlas provenance to %s", provenance_path)
    return provenance_path


def fetch_atlas(cfg: DictConfig, *, force: bool = False) -> Path:
    """Fetches, verifies, and extracts the full SRI24 atlas.

    Args:
        cfg: The `anatomy` config sub-tree (`hydra_cfg.anatomy`), carrying
            `dir`, `subdir`, `version`, `source`, `licence`, and `archives`.
        force: If True, re-download every archive even if a valid cached
            copy exists.

    Returns:
        Path to the extracted atlas directory (`cfg.dir/cfg.subdir`).

    Raises:
        ValueError: If any archive fails checksum/size verification. No
            archive is extracted until every archive has been fetched and
            verified, so a failure here leaves nothing extracted.
        FileNotFoundError: If, after extraction, any file the rest of the
            pipeline needs (see `REQUIRED_MEMBERS`) is still missing.
    """
    root = Path(cfg.dir)
    cache_dir = root / "_archives"
    extract_root = root

    specs = archive_specs(cfg)

    # Fetch (and verify) every archive before extracting any of them, so a
    # verification failure on archive 3 of 3 can never leave archives 1 and
    # 2 partially extracted on disk.
    results = [fetch_archive(spec, cache_dir, force=force) for spec in specs]

    for result in results:
        safe_extract(result.path, extract_root)

    atlas_dir = extract_root / cfg.subdir
    missing = [name for name in REQUIRED_MEMBERS if not (atlas_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"SRI24 atlas at {atlas_dir} is missing required file(s): {', '.join(missing)}. "
            "The archives may be incomplete, or the upstream extraction layout has changed."
        )

    write_provenance(cfg, extract_root, results)

    logger.warning(_LICENCE_WARNING)

    return atlas_dir


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Fetches the SRI24 atlas described by `cfg.anatomy`.

    Args:
        cfg: The full config Hydra composed from `configs/` plus any CLI
            overrides. Reads `cfg.anatomy` (see `configs/anatomy/sri24.yaml`)
            and the optional `+force=true` override.
    """
    setup_logging(level="INFO")

    atlas_cfg = cfg.anatomy
    force = bool(cfg.get("force", False))

    dest = fetch_atlas(atlas_cfg, force=force)

    provenance_path = dest.parent / PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    specs = archive_specs(atlas_cfg)
    total_bytes = sum(spec.size_bytes for spec in specs)

    print("=" * 70)
    print("SRI24 atlas fetch summary")
    print("=" * 70)
    print(f"Destination: {dest}")
    print(f"Version:     {atlas_cfg.version}")
    print()
    print("Archives:")
    for spec in specs:
        action = provenance["archives"].get(spec.name, {}).get("action", "?")
        print(f"  {spec.name:10s} {action:12s} ({format_size(spec.size_bytes)})")
    print()
    print(f"Total archive size: {format_size(total_bytes)}")
    print()
    print(_LICENCE_WARNING)
    print("=" * 70)


if __name__ == "__main__":
    main()
