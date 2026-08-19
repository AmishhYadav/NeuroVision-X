"""Upload -> preprocess -> segment -> report, as a pollable background job.

This is the wiring between two things that already exist: offline BraTS
preprocessing (`neurovision.data.preprocessing.preprocess_case`) and live
CPU inference (`app.backend.inference.segment_case`). Nothing here trains or
evaluates anything; it lets a demo user upload four raw MRI volumes, kicks
off preprocessing + segmentation for that one case, and exposes the result's
progress through a small in-memory job store the API layer can poll.

Two things worth stating up front, because they are easy to get wrong for an
upload endpoint:

1. **Client-supplied filenames never touch a path.** `create_job` takes a
   mapping of a FIXED role name (`"t1"` / `"t1ce"` / `"t2"` / `"flair"`) to
   raw bytes -- no filename parameter exists anywhere in this module. A
   payload is always written to `<job_dir>/raw/<role>.nii.gz`, where `<role>`
   is one of those four literal strings. A crafted "filename" like
   `"../../etc/passwd"` therefore has nothing to act on: there is no
   filename-shaped input for it to be.
2. **Job state is a module-level dict, in-process only.** It does not
   survive a server restart, and it is not shared across multiple worker
   processes. This is deliberate for a single-process demo -- see
   `CLAUDE.md`'s "the demo is a viewer, not a measuring instrument" framing
   for the same reasoning applied elsewhere in `app/`. A real multi-worker
   deployment would need a shared job store (a database, Redis, ...); this
   one does not have that requirement.

Following `inference.py`'s lazy-import discipline: nothing in this module
imports torch at module scope. `neurovision.data.preprocessing` is safe to
import eagerly (it is itself dependency-light -- only numpy and nibabel, no
torch, no monai). `app.backend.inference.segment_case` is the one call in
this module that reaches torch (via `inference._load_model`), so it is
imported lazily inside `run_job`, exactly where it is used.
"""

from __future__ import annotations

import dataclasses
import gzip
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

from neurovision.data.brats import BratsCase
from neurovision.data.preprocessing import preprocess_case

from . import inference
from .config import REPO_ROOT, Settings

logger = logging.getLogger(__name__)

# The four modality roles a job requires, in the fixed order everything in
# this module iterates them -- also the order `BratsCase.modality_paths`
# expects (t1, t1ce, t2, flair).
_MODALITY_ROLES = ("t1", "t1ce", "t2", "flair")
_REQUIRED_ROLES = frozenset(_MODALITY_ROLES)

# A 240x240x155 float32 NIfTI is ~35 MB uncompressed, so 400 MB is generous
# headroom for a single large volume. This exists purely so an unbounded
# upload cannot fill the disk -- it is NOT settings-dependent, unlike every
# path in this module, because it is a resource-abuse limit rather than a
# location.
MAX_UPLOAD_BYTES = 400 * 1024 * 1024

# Affine entries in a NIfTI header are float32 in the file even though
# nibabel reads them back as float64 -- comparing two affines that came from
# independently-written files with exact equality would fail on harmless
# round-off. 1e-3 mm is far tighter than any real registration difference
# and far looser than float32 rounding noise.
_AFFINE_ATOL = 1e-3

JobState = Literal["queued", "running", "done", "failed"]


@dataclass
class Job:
    """State of one upload-to-report job.

    Attributes:
        job_id: Server-generated identifier (`uuid.uuid4().hex`). Also used
            as the case id passed to `preprocess_case` / `segment_case`.
        state: Current lifecycle state.
        stage: Human-readable label for what is happening right now (e.g.
            `"preprocessing"`, `"segmenting"`, `"segmenting: running
            inference"`, `"done"`). Free text -- callers should not match on
            it beyond display.
        progress: Fraction complete, `0.0`..`1.0`, monotonically
            non-decreasing over the job's lifetime.
        case_id: The preprocessed-case identifier this job produces (equal
            to `job_id`; kept as its own field so downstream code reads it
            the same way it reads any other case id, without knowing it
            happens to equal `job_id`).
        error: Failure message, set only when `state == "failed"`.
        created_at: `time.time()` when the job was created.
        updated_at: `time.time()` of the most recent state change.
    """

    job_id: str
    state: JobState
    stage: str
    progress: float
    case_id: str
    error: str | None
    created_at: float
    updated_at: float


# In-process job store. See the module docstring: this does not survive a
# restart and is not shared across processes, on purpose. `_LOCK` guards
# both dicts together -- they are always updated as a pair (see
# `create_job` / `delete_job`), so one lock is simpler than two that would
# always be taken together anyway.
_LOCK = threading.Lock()
_JOBS: dict[str, Job] = {}
# job_id -> label_convention, forwarded to preprocess_case by run_job. Not a
# Job field: it is an input to the job, not part of its observable state,
# and adding it to Job would mean every API consumer has to know a field it
# never needs to read.
_LABEL_CONVENTIONS: dict[str, str] = {}


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
            `settings.prep_dir` / `settings.cache_dir` (see `run_job`'s
            docstring for why).

    Returns:
        The resolved job root directory. Created if it did not already
        exist.
    """
    raw = os.environ.get("NVX_JOB_DIR", "outputs/demo_jobs")
    p = Path(raw).expanduser()
    root = p if p.is_absolute() else (REPO_ROOT / p)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_gzip(data: bytes) -> bytes:
    """Returns `data` gzip-compressed, unless it already is.

    Every stored upload is named `<role>.nii.gz` regardless of whether the
    client sent compressed or uncompressed bytes, so the stored file's
    content must actually match that extension -- `nib.load` picks its
    codec from the extension, and handing it a `.nii.gz` file that is not
    really gzip would fail later, inside `preprocess_case`, far from
    whichever upload caused it.
    """
    if data[:2] == b"\x1f\x8b":  # gzip magic bytes
        return data
    return gzip.compress(data)


def _validate_roles(uploads: Mapping[str, bytes]) -> None:
    """Raises unless `uploads` has exactly the four required modality roles."""
    given = set(uploads.keys())
    missing = sorted(_REQUIRED_ROLES - given)
    if missing:
        raise ValueError(f"create_job: missing required upload role(s): {', '.join(missing)}")

    extra = sorted(given - _REQUIRED_ROLES)
    if extra:
        raise ValueError(
            f"create_job: unexpected upload role(s): {', '.join(extra)}; "
            f"expected exactly {sorted(_REQUIRED_ROLES)}"
        )


def _validate_payload_size(role: str, data: bytes) -> None:
    """Raises if a payload is empty or exceeds `MAX_UPLOAD_BYTES`."""
    if not data:
        raise ValueError(f"create_job: upload for {role!r} is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"create_job: upload for {role!r} is {len(data)} bytes, "
            f"over the {MAX_UPLOAD_BYTES}-byte limit"
        )


def _load_and_validate_nifti(role: str, data: bytes) -> tuple[tuple[int, ...], np.ndarray]:
    """Parses one upload as a NIfTI volume and returns its shape and affine.

    Writes `data` to a throwaway temp file rather than nibabel's in-memory
    `Nifti1Image.from_bytes` (which only accepts an UNCOMPRESSED byte
    string): a real upload may or may not be gzip-compressed, and `nib.load`
    picks the right codec from the file EXTENSION -- exactly the detection
    this needs, and not worth reimplementing by hand.

    Args:
        role: Which of the four modality roles this payload claims to be.
            Used only to name the role in a raised error message.
        data: Raw file bytes.

    Returns:
        `(shape, affine)`: `shape` a 3-tuple of ints, `affine` a 4x4
        float64 array.

    Raises:
        ValueError: If `data` does not parse as a readable NIfTI volume, or
            parses to something other than a 3D volume.
    """
    suffix = ".nii.gz" if data[:2] == b"\x1f\x8b" else ".nii"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            img = nib.load(tmp.name)
            shape = tuple(int(s) for s in img.shape)
            affine = np.asarray(img.affine, dtype=np.float64)
            # nib.load is lazy about the data block -- force a real read now
            # so truncated/corrupt content is caught HERE, naming the role,
            # instead of surfacing later inside preprocess_case with no
            # indication of which of the four uploads was the problem.
            np.asarray(img.dataobj)
    except Exception as exc:
        raise ValueError(f"upload for {role!r} is not a valid NIfTI volume: {exc}") from exc

    if len(shape) != 3:
        raise ValueError(f"upload for {role!r} must be a 3D volume, got shape {shape!r}")
    return shape, affine


def _validate_consistent_geometry(
    geometry: dict[str, tuple[tuple[int, ...], np.ndarray]],
) -> None:
    """Raises if any upload disagrees with the reference (`t1`) on shape or affine.

    Args:
        geometry: Role -> `(shape, affine)`, one entry per uploaded
            modality, in `_MODALITY_ROLES` order.

    Raises:
        ValueError: Naming the reference role and the first disagreeing
            role, on either a shape or an affine mismatch (affines compared
            within `_AFFINE_ATOL`). Proceeding on mismatched geometry would
            produce a plausible-looking but spatially wrong segmentation, so
            this is a hard failure rather than a warning.
    """
    roles = list(geometry)
    ref_role = roles[0]
    ref_shape, ref_affine = geometry[ref_role]
    for role in roles[1:]:
        shape, affine = geometry[role]
        if shape != ref_shape:
            raise ValueError(
                f"create_job: shape mismatch between {ref_role!r} {ref_shape} "
                f"and {role!r} {shape}"
            )
        if not np.allclose(affine, ref_affine, atol=_AFFINE_ATOL):
            raise ValueError(f"create_job: affine mismatch between {ref_role!r} and {role!r}")


def create_job(
    settings: Settings,
    uploads: Mapping[str, bytes],
    *,
    label_convention: str = "brats2021",
) -> Job:
    """Validates four uploaded MRI volumes, stores them, and queues a job.

    Args:
        settings: Resolved backend settings, used only to locate
            `job_root(settings)`.
        uploads: Mapping from modality role (exactly `"t1"`, `"t1ce"`,
            `"t2"`, `"flair"`) to raw file bytes. There is no filename
            parameter anywhere in this function's signature -- see the
            module docstring for why that is the point, not an omission.
        label_convention: Forwarded to `preprocess_case` when the job later
            runs. There is no ground-truth segmentation for an uploaded
            case today, so this presently has no observable effect; kept as
            a parameter so a future caller that DOES have a label does not
            need this function's signature to change.

    Returns:
        A new `Job` in state `"queued"`.

    Raises:
        ValueError: On any of -- missing or unexpected upload roles; an
            empty or over-`MAX_UPLOAD_BYTES` payload; a payload that does
            not parse as a NIfTI volume; a non-3D volume; or a shape/affine
            mismatch between two of the four uploads. Every message names
            the offending role(s), and nothing is written to disk unless
            all four uploads pass every check.
    """
    _validate_roles(uploads)

    geometry: dict[str, tuple[tuple[int, ...], np.ndarray]] = {}
    for role in _MODALITY_ROLES:
        data = uploads[role]
        _validate_payload_size(role, data)
        geometry[role] = _load_and_validate_nifti(role, data)
    _validate_consistent_geometry(geometry)

    job_id = uuid.uuid4().hex
    raw_dir = job_root(settings) / job_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for role in _MODALITY_ROLES:
        # Fixed, role-derived filename ONLY -- never anything derived from
        # the caller's input beyond the role key itself.
        (raw_dir / f"{role}.nii.gz").write_bytes(_ensure_gzip(uploads[role]))

    now = time.time()
    job = Job(
        job_id=job_id,
        state="queued",
        stage="queued",
        progress=0.0,
        case_id=job_id,
        error=None,
        created_at=now,
        updated_at=now,
    )
    with _LOCK:
        _JOBS[job_id] = job
        _LABEL_CONVENTIONS[job_id] = label_convention
    logger.info("Queued job %s", job_id)
    return job


def get_job(job_id: str) -> Job | None:
    """Current state of one job, or `None` if unknown.

    Returns the SAME `Job` instance held in the in-process store, not a
    copy -- a caller polling this while `run_job` executes (on this thread
    or another) sees live `stage` / `progress` updates for free.

    Args:
        job_id: A job id, as returned by `create_job`.

    Returns:
        The `Job`, or `None` if `job_id` is not known (never seen, or
        already deleted).
    """
    with _LOCK:
        return _JOBS.get(job_id)


def list_jobs() -> list[Job]:
    """All known jobs, newest first (by `created_at`)."""
    with _LOCK:
        return sorted(_JOBS.values(), key=lambda job: job.created_at, reverse=True)


def _update_job(job: Job, **fields: object) -> None:
    """Mutates `job`'s fields in place and bumps `updated_at`, under the lock."""
    with _LOCK:
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def run_job(settings: Settings, job_id: str) -> Job:
    """Runs one queued job SYNCHRONOUSLY: preprocess the upload, then segment it.

    Safe to call directly (e.g. from `start_job`'s worker thread, or
    directly from a request handler for a quick synchronous demo). Never
    raises for a job-level failure: any exception from preprocessing or
    segmentation is caught, recorded on the `Job` itself (`state="failed"`,
    `error=str(exc)`), logged at ERROR with a traceback, and the job is
    returned -- a failed job must stay reportable through `get_job`, not
    crash whatever is running it.

    Args:
        settings: Resolved backend settings. `settings.checkpoint` /
            `settings.experiment` drive the segmentation step.
            `settings.prep_dir` and `settings.cache_dir` are NOT used
            directly: this job gets its own, namespaced under
            `job_root(settings)/job_id/`, so an uploaded case can never
            collide with (or pollute) the app's shared preprocessed-case or
            live-inference-cache directories.
        job_id: A job created by `create_job`.

    Returns:
        The same `Job`, now in a terminal state (`"done"` or `"failed"`).

    Raises:
        KeyError: If `job_id` names no known job -- this is a programming
            error by the caller, not a job-level failure, so it is not
            swallowed the way an in-job exception is.
    """
    job = get_job(job_id)
    if job is None:
        raise KeyError(f"run_job: unknown job_id {job_id!r}")

    job_prep_dir = job_root(settings) / job_id / "prep"

    try:
        _update_job(job, state="running", stage="preprocessing", progress=0.05)

        raw_dir = job_root(settings) / job_id / "raw"
        case = BratsCase(
            case_id=job.case_id,
            t1=raw_dir / "t1.nii.gz",
            t1ce=raw_dir / "t1ce.nii.gz",
            t2=raw_dir / "t2.nii.gz",
            flair=raw_dir / "flair.nii.gz",
            # No ground truth for an uploaded case. preprocess_case already
            # treats `case.seg` as optional (see its `has_label` branch), so
            # this needs no special-casing beyond passing None through.
            seg=None,
        )
        preprocess_case(
            case,
            job_prep_dir,
            label_convention=_LABEL_CONVENTIONS.get(job_id, "brats2021"),
            target_axcodes=("L", "P", "S"),
        )

        _update_job(job, stage="segmenting", progress=0.3)

        if not inference.checkpoint_available(settings):
            reason = inference.inference_status(settings)["reason"]
            raise RuntimeError(reason or "no checkpoint configured for live inference")

        # Lazy: this is the one call in this module that reaches torch (via
        # inference._load_model), so it must not happen at module import
        # time -- see the module docstring.
        from .inference import segment_case

        job_settings = dataclasses.replace(
            settings, prep_dir=job_prep_dir, cache_dir=job_root(settings) / job_id / "cache"
        )

        def _forward_progress(stage: str, fraction: float) -> None:
            # segment_case reports its OWN progress on roughly 0.1..1.0;
            # remap into this job's 0.3..0.95 slice so the two stages read
            # as one continuously advancing bar instead of jumping backwards
            # the moment segmentation starts.
            _update_job(
                job, stage=f"segmenting: {stage}", progress=min(0.3 + fraction * 0.65, 0.95)
            )

        segment_case(job_settings, job.case_id, progress=_forward_progress)

        _update_job(job, state="done", stage="done", progress=1.0)
        return job
    except Exception as exc:  # noqa: BLE001 - a failed job must stay reportable, never crash
        logger.error("Job %s failed at stage %r", job_id, job.stage, exc_info=True)
        _update_job(job, state="failed", error=str(exc))
        return job


def start_job(settings: Settings, job_id: str) -> None:
    """Runs a queued job on a background daemon thread.

    Fire-and-forget: progress and the terminal result are read back through
    `get_job`, never through this function (it has no return value). The
    thread is a daemon so it can never keep the process alive on its own --
    consistent with the module docstring's "in-process only" caveat.

    Args:
        settings: Resolved backend settings, forwarded to `run_job`.
        job_id: A job created by `create_job`.
    """
    thread = threading.Thread(
        target=run_job, args=(settings, job_id), daemon=True, name=f"nvx-job-{job_id[:8]}"
    )
    thread.start()


def job_case_dir(settings: Settings, job_id: str) -> Path:
    """Preprocessed case directory a job writes to (finished or not).

    A pure path computation, not a check that `image.npy` already exists --
    safe to call before, during, or after `run_job`.

    Args:
        settings: Resolved backend settings, used to locate `job_root`.
        job_id: A job id. If it names a job this process still remembers,
            the job's own `case_id` is used; otherwise `job_id` itself is
            used (they are equal for every job `create_job` has produced),
            so this still resolves sensibly after a restart wiped the
            in-memory store but left files on disk.

    Returns:
        `job_root(settings)/job_id/prep/<case_id>`.
    """
    job = get_job(job_id)
    case_id = job.case_id if job is not None else job_id
    return job_root(settings) / job_id / "prep" / case_id


def delete_job(settings: Settings, job_id: str) -> bool:
    """Removes a job and everything it wrote to disk.

    Args:
        settings: Resolved backend settings, used to locate `job_root`.
        job_id: A job id.

    Returns:
        `True` if the job existed (in the in-memory store, on disk, or
        both) and was removed; `False` if there was nothing to remove.

    Raises:
        ValueError: If `job_id` resolves to a directory outside
            `job_root(settings)` -- e.g. via a crafted id containing `".."`.
            This is the same "never trust an id used as a path component"
            discipline `create_job` applies to upload roles, checked here
            because `job_id`, unlike a role name, is not restricted to a
            fixed set of literal strings.
    """
    root = job_root(settings).resolve()
    job_dir = (root / job_id).resolve()
    if job_dir != root and root not in job_dir.parents:
        raise ValueError(
            f"delete_job: resolved job directory {job_dir} falls outside "
            f"job_root {root}; refusing to delete"
        )

    with _LOCK:
        existed_in_store = _JOBS.pop(job_id, None) is not None
        _LABEL_CONVENTIONS.pop(job_id, None)

    existed_on_disk = job_dir.is_dir()
    if existed_on_disk:
        shutil.rmtree(job_dir)

    return existed_in_store or existed_on_disk
