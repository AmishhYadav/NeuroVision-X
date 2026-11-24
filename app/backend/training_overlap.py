"""Flags clinical jobs whose DICOM study is a patient the model trained on (W6.2).

`knowledge/known_training_overlap.yaml` lists DICOM SeriesInstanceUIDs of studies
known to match BraTS training cases. A PROCEED on such a study shows the pipeline
runs on scanner DICOM; it is no evidence of accuracy on unseen data, so the API
reports the match and the UI can say so.

PyYAML is not pinned directly; it arrives with hydra-core (via omegaconf).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_overlap_registry(path: Path) -> list[dict[str, Any]]:
    """Reads the overlap registry YAML.

    Args:
        path: Path to the YAML file (top-level `entries:` list).

    Returns:
        The list of entry dicts. Empty if the file does not exist (a warning is logged).

    Raises:
        ValueError: If `entries` is missing / not a list, or an entry has no
            `series_instance_uids` list.
    """
    path = Path(path)
    if not path.is_file():
        logger.warning("Training-overlap registry not found at %s; no jobs will be flagged", path)
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError(f"{path}: expected a top-level 'entries' list")
    for i, entry in enumerate(data["entries"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("series_instance_uids"), list):
            raise ValueError(f"{path}: entry {i} is missing a 'series_instance_uids' list")
    return data["entries"]


@lru_cache(maxsize=8)
def _cached_registry(path_str: str) -> tuple[dict[str, Any], ...]:
    """Loads the registry once per path (tuple so the cached value is not mutated)."""
    return tuple(load_overlap_registry(Path(path_str)))


def get_overlap_registry(path: Path) -> list[dict[str, Any]]:
    """Returns the registry at `path`, cached per path for the life of the process."""
    return list(_cached_registry(str(path)))


def match_training_overlap(
    ingest_result: dict[str, Any] | None, registry: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Checks whether a job's assigned series belong to a known training patient.

    Only series the ingest step actually ASSIGNED to a modality role count; a
    rejected series is not part of what the model saw.

    Args:
        ingest_result: A job's `ingest_result` dict (may be None).
        registry: Entries from `load_overlap_registry`.

    Returns:
        For the first matching entry, a dict with `source_id`, `brats_id`,
        `split`, `status`, `evidence` and `matched_series` (sorted UIDs);
        None if nothing matches.
    """
    if not ingest_result:
        return None
    assignments = ingest_result.get("assignments")
    if not assignments:
        return None
    assigned = {
        uid
        for uid, info in assignments.items()
        if isinstance(info, dict) and info.get("outcome") == "assigned"
    }
    for entry in registry:
        matched = assigned.intersection(entry["series_instance_uids"])
        if matched:
            return {
                "source_id": entry.get("source_id"),
                "brats_id": entry.get("brats_id"),
                "split": entry.get("split"),
                "status": entry.get("status"),
                "evidence": entry.get("evidence"),
                "matched_series": sorted(matched),
            }
    return None
