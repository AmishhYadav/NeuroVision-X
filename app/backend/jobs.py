"""Shared on-disk root for clinical jobs.

Resolved from the `NVX_JOB_DIR` environment variable, repo-relative by
default. The NIfTI upload job machinery that used to live in this module
(`create_job`, `run_job`, `start_job`, `get_job`, `list_jobs`, `delete_job`,
and their validation helpers) was removed on 2026-09-18 because nothing in
the product called it any more -- the DICOM clinical path in
`clinical_jobs.py` replaced it. `job_root` is the one symbol other modules
still import.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import REPO_ROOT, Settings


def job_root(settings: Settings) -> Path:
    """Directory holding all upload jobs' raw / preprocessed / cache files.

    Resolved from the `NVX_JOB_DIR` environment variable, repo-relative by
    default -- the same pattern `config._path_env` uses. That helper is not
    imported here (this module must not modify `config.py`, and importing a
    private helper from it would be the same coupling in a different
    disguise), so the three lines are reimplemented instead.

    Args:
        settings: Resolved backend settings. Accepted for a consistent
            signature across this module's functions, but not itself
            consulted -- job storage is deliberately independent of
            `settings.prep_dir` / `settings.cache_dir`.

    Returns:
        The resolved job root directory. Created if it did not already
        exist.
    """
    raw = os.environ.get("NVX_JOB_DIR", "outputs/demo_jobs")
    p = Path(raw).expanduser()
    root = p if p.is_absolute() else (REPO_ROOT / p)
    root.mkdir(parents=True, exist_ok=True)
    return root
