"""Small, boring I/O helpers used across the project.

Every function here takes a path as an argument — never a hardcoded location —
so the same code works unchanged on the Mac (dev) and on Kaggle (training).
Reads use UTF-8 explicitly because the platform default encoding differs
between macOS and the Linux images Kaggle runs, and we don't want that to be
a silent source of file-reading bugs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Size past which an upload is worth a second look, in GB. Deliberately well
# under Kaggle's actual per-dataset ceiling (200 GB for both private and
# public datasets as of 2026-08) -- crossing this means "this will take a long
# time to upload and is worth confirming", NOT "this will be rejected".
#
# Do not confuse the dataset ceiling with the ~20 GB /kaggle/working OUTPUT
# quota, which is a genuinely hard limit and the one that constrains how many
# checkpoints a session can keep (see training.checkpoint.keep_last_n). An
# earlier version of this constant conflated the two and would have pushed us
# into needlessly subsetting BraTS 2021.
#
# Shared by scripts/preprocess.py and scripts/package_for_kaggle.py so the two
# scripts can never silently disagree about what "too big" means.
KAGGLE_DATASET_WARN_GB = 60.0


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and any missing parents) if it does not exist.

    Args:
        path: Directory to create. Treated as a directory, not a file path.

    Returns:
        The same path as a `Path`, so callers can chain, e.g.
        `out = ensure_dir(cfg.output_dir) / "run.json"`.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    """Read a JSON file.

    Args:
        path: Path to the `.json` file.

    Returns:
        The parsed JSON content (typically a `dict` or `list`).

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    logger.debug("Read JSON from %s", path)
    return obj


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Write an object to a JSON file, creating parent directories as needed.

    Args:
        obj: Object to serialize. Must be JSON-serializable, or serializable
            via `str()` (see note below).
        path: Destination `.json` file path.
        indent: Indentation level passed to `json.dump`.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        # default=str lets non-JSON-native values (Path, numpy scalars,
        # torch.device, etc.) fall back to their string form instead of
        # raising a TypeError — useful for dumping metrics/config dicts that
        # were assembled from mixed sources.
        json.dump(obj, f, indent=indent, default=str)
    logger.debug("Wrote JSON to %s", path)


def read_yaml(path: str | Path) -> Any:
    """Read a YAML file.

    Args:
        path: Path to the `.yaml`/`.yml` file.

    Returns:
        The parsed YAML content, or `None` if the file is empty.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        # safe_load only builds plain Python types (dict, list, str, etc.).
        # yaml.load can be made to construct arbitrary Python objects from
        # crafted YAML tags, and config files aren't guaranteed to be ours.
        obj = yaml.safe_load(f)
    logger.debug("Read YAML from %s", path)
    return obj


def write_yaml(obj: Any, path: str | Path) -> None:
    """Write an object to a YAML file, creating parent directories as needed.

    Args:
        obj: Object to serialize (typically a `dict`).
        path: Destination `.yaml`/`.yml` file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        # sort_keys=False preserves the order the caller built the dict in,
        # which matters for config readability (grouped, not alphabetized).
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False)
    logger.debug("Wrote YAML to %s", path)


def directory_size_bytes(path: str | Path) -> int:
    """Sum the size of every file under a directory, recursively.

    Args:
        path: Directory to measure. Need not exist.

    Returns:
        Total size in bytes of all files found under `path` (including
        nested subdirectories), or 0 if `path` does not exist or is empty.
    """
    root = Path(path)
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def format_size(num_bytes: int) -> str:
    """Render a byte count as a human-readable size string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        A string like `"512.00 B"`, `"3.40 MB"`, or `"12.34 GB"`, scaling by
        1024 through B, KB, MB, GB, TB, PB (stopping at PB regardless of
        magnitude).
    """
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"
