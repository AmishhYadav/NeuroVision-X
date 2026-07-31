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
