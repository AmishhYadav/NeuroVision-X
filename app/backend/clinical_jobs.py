"""Real-DICOM-study -> refusal-gated segmentation, as a pollable background job.

`app/backend/jobs.py` wires the DEMO path: four already-registered,
already-skull-stripped BraTS-style NIfTI files in, preprocess, segment. This
module wires the REAL path: a raw hospital DICOM study (a zip of an
arbitrarily-nested folder of `.dcm` files) in, through every Milestone 4
Phase E gate that has to run before a model is allowed to see it, out to a
PROCEED / PROCEED_WITH_CAUTION / REFUSE decision. The pipeline, in order:

    E1 ingest (`neurovision.data.dicom_ingest.ingest_study`)
      -> E3 input QC on the raw ingested volumes
      -> E2 clinical preprocessing (co-registration, atlas registration,
         skull-stripping; `neurovision.data.clinical_preprocess`)
      -> E3 input QC again, now with a brain mask available
      -> the EXISTING research preprocessing path
         (`neurovision.data.preprocessing.preprocess_case`, unmodified)
      -> the EXISTING segmentation path (`app.backend.inference.segment_case`,
         unmodified, called with `save_logits=True`)
      -> two label-free safety signals computed HERE (`_live_predicted_dice`,
         `_live_conformal_band_width`)
      -> E5 the gatekeeper (`neurovision.inference.gatekeeper.run_gatekeeper`)

E6 (DICOM-SEG export, `_export_dicom_seg`) now runs after the gatekeeper, as
a SUPPLEMENTARY artifact -- never a requirement for the clinical decision the
job has already made by the time it runs. The chain: uncrop the cropped
research-frame prediction back to the full atlas-space grid E2 produced
(`neurovision.inference.postprocess.uncrop_to_original`); resample that
ATLAS-SPACE CLASS MAP (not yet split into regions) back into the center
modality's native (pre-E2) geometry through E2's own saved inverse transform
(`neurovision.data.clinical_resample.resample_mask_to_source`); only THEN
split the resampled, native-space class map into the three nested ET/TC/WT
region channels `neurovision.reporting.dicom_seg.write_dicom_seg` wants;
find that modality's own raw DICOM headers under this job's `raw_dicom/`
directory; and write the SEG object. A geometry mismatch there (`write_dicom_seg`'s
own named refusal) or any other failure in this chain is caught, logged, and
turned into "no SEG object for this job" -- it can never turn an otherwise-good
segmentation into a `"failed"` job, mirroring exactly the failure-isolation
philosophy the Grad-CAM block right before it already established.

Before any of that chain runs, `_validate_dicom_seg_cfg` checks the STATIC,
config-derived parts of `cfg.clinical.dicom_seg` (`segmentation_type`,
`series_description` length) exactly once per job, from a call site OUTSIDE
`_export_dicom_seg`'s own narrow `except ValueError` around
`write_dicom_seg`'s call. Those two checks are properties of the deployed
CONFIG, not of any one case -- wrong for every job, forever, if wrong at
all -- so they must never blend into the same routine per-job
geometry-refusal WARNING `_export_dicom_seg` logs for a genuine per-case
outcome. A validation failure still reaches `run_clinical_job`'s own generic
"DICOM-SEG export failed unexpectedly" catch (the same "unexpected exception
in the export chain" path any other unexpected failure in this chain already
takes), logged at ERROR with a full traceback -- distinguishable from, never
conflated with, a genuine refusal, and still never `"failed"`: this remains a
supplementary artifact.

**`"refused"` is a distinct, successful terminal state, never `"failed"`.**
A `state="refused"` job means one of the label-free gates (E3's input QC, or
E5's gatekeeper) looked at this exact study and correctly said "no" for a
named, structured reason -- the pipeline working exactly as designed. A
`state="failed"` job means something unexpected blew up (a bad checkpoint
path, a bug, a missing dependency). Conflating the two would hide the one
signal this whole module exists to produce: a study can be refused without
anything having gone wrong. Every `state="refused"` job still carries a
non-`None` `error` naming which check refused and why, exactly like a
`"failed"` job's `error` -- the DIFFERENCE is `state`, never the presence of
`error`.

**Which model runs a clinical job is fixed, independent of the generic
backend `Settings`.** `app.backend.config.Settings` (used by the demo's
`/upload` + `/cases` routes) defaults to `experiment="baseline_unet3d"` --
correct for browsing precomputed comparison cases, but the QC checkpoint
(`outputs/neurovision/qc/best.pt`) and this project's calibrated
`gatekeeper.enabled_signals` were measured ONLY against the `neurovision`
checkpoint (see `configs/clinical/default.yaml`'s `gatekeeper.enabled_signals`
comment and the master plan's Gate C result). So every public function here
that needs `settings` only for `jobs.job_root(settings)` path resolution
reads `settings.checkpoint` / `settings.experiment` from NOWHERE -- the
segmentation step always builds its OWN dedicated `Settings` via
`_clinical_segmentation_settings`, pinned to `experiment="neurovision"` and a
checkpoint resolved from the (new) `NVX_CLINICAL_CHECKPOINT` environment
variable, never from whatever the generic backend `Settings` happens to name.

**Lazy imports, and why.** Following `jobs.py`'s and `inference.py`'s own
discipline: `app.backend.inference.segment_case` is imported inside
`run_clinical_job`, at the point it is called, for the same reason
`jobs.run_job` does -- it is the one call that reaches torch via
`inference._load_model`, and a module that imports torch eagerly at import
time is a module that can fail to import on a machine with no GPU stack
configured, for no reason a caller who only wants `create_clinical_job` would
expect. Hydra composition (`_compose_clinical_cfg`, `_load_qc_model_and_cfg`)
likewise imports `hydra` lazily, inside the function body, and is guarded by
the SAME `app.backend.inference._HYDRA_LOCK` every other Hydra composition in
this backend uses -- `hydra.compose` mutates a process-global singleton, so
two threads composing at once can corrupt each other's state.

**In-process job store.** Same shape as `jobs.py`: a module-level dict behind
one lock, not shared across processes, does not survive a restart. See
`jobs.py`'s own module docstring for the reasoning; it applies unchanged
here. This module's jobs live in a DIFFERENT in-memory dict (`_CLINICAL_JOBS`,
keyed by a freshly generated `uuid.uuid4().hex`, so collision with a `jobs.py`
job id is not a real possibility) but under the SAME `jobs.job_root(settings)`
parent directory, each in its own `<job_id>/` subdirectory with different
internal layout (`raw_dicom/`, `ingest/`, `clinical_prep/`, `prep/`, `cache/`)
than a `jobs.py` job's (`raw/`, `prep/`, `cache/`).
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import threading
import time
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np
import torch
from scipy.special import expit

from neurovision.analysis.qc_inference import CaseArrays, entropy_from_logits, pack_sample
from neurovision.data.brats import BratsCase
from neurovision.data.clinical_preprocess import PreprocessResult, preprocess_clinical_study
from neurovision.data.dicom_ingest import ROLES, IngestResult, ingest_study
from neurovision.data.preprocessing import preprocess_case
from neurovision.data.transforms import REGION_NAMES
from neurovision.inference.gatekeeper import Decision, GateSignals, run_gatekeeper
from neurovision.inference.input_qc import InputQCReport, Severity, load_volume_infos, run_input_qc
from neurovision.inference.postprocess import postprocess_logits, uncrop_to_original
from neurovision.models.qc import build_segqc
from neurovision.models.qc import predicted_dice as qc_predicted_dice
from neurovision.training.checkpoint import load_checkpoint
from neurovision.utils.device import get_device

from . import inference, jobs, volumes
from .config import REPO_ROOT, Settings, _path_env

logger = logging.getLogger(__name__)

# The one deployed model a clinical job always runs, regardless of what
# experiment the generic app.backend.config.Settings the rest of the backend
# is configured to show. See the module docstring's "which model" section.
_CLINICAL_EXPERIMENT = "neurovision"

# The reserved role key `neurovision.inference.input_qc.load_volume_infos`
# treats specially (see that module's `_BRAIN_MASK_KEY` and its docstring).
# Kept as this module's own literal, rather than importing the private name,
# because it is part of that function's PUBLIC docstring contract ("a path
# keyed `"brain_mask"`"), not an implementation detail.
_BRAIN_MASK_ROLE_KEY = "brain_mask"

# Must match neurovision.anatomy.localize's internal `_UNLABELLED_NAME`
# exactly -- it is not exported, since it is an implementation detail of that
# module's table, but `_generate_report`'s own min_frac filter has to name it
# too when deciding which row `min_frac` filtering may never drop. Same
# constant, same reasoning, as `scripts/localize.py`'s own
# `_UNLABELLED_STRUCTURE_NAME`.
_UNLABELLED_STRUCTURE_NAME = "unlabelled"

# A real multi-series brain MRI study (localisers, scouts, T1/T1CE/T2/FLAIR,
# sometimes DWI/perfusion series a real hospital PACS exports alongside them)
# can run to several hundred MB compressed; 2 GiB is generous headroom over
# that so a genuine study is never rejected, while still bounding how much an
# unbounded upload can make this process buffer in memory at once -- the same
# resource-abuse-limit reasoning as `jobs.py`'s `MAX_UPLOAD_BYTES`, just sized
# for a whole DICOM study instead of one NIfTI volume.
MAX_STUDY_ZIP_BYTES = 2 * 1024**3

JobState = Literal["queued", "running", "done", "refused", "failed"]


@dataclass
class ClinicalJob:
    """State of one clinical (raw DICOM study) job.

    See the module docstring for why `"refused"` is a distinct state from
    `"failed"`, and for the fixed pipeline order that populates the fields
    below, roughly left-to-right through the pipeline.

    Attributes:
        job_id: Server-generated identifier (`uuid.uuid4().hex`).
        state: Current lifecycle state.
        stage: Human-readable label for what is happening right now. Free
            text -- callers should not match on it beyond display.
        progress: Fraction complete, `0.0`..`1.0`, monotonically
            non-decreasing over the job's lifetime.
        case_id: The preprocessed-case identifier this job produces (equal
            to `job_id`).
        error: Set whenever `state` is `"refused"` or `"failed"`, naming the
            reason (which check refused, or what broke). `None` otherwise.
        ingest_result: `_ingest_result_to_dict` of E1's `IngestResult`, or
            `None` before E1 has run. Written even when E1 leads straight to
            a refusal -- it is the audit trail.
        input_qc_pre: E3's `InputQCReport.to_dict()` from the check run on
            the raw ingested volumes, before E2. `None` before that check
            has run.
        input_qc_post: E3's `InputQCReport.to_dict()` from the check run
            after E2, with the brain mask (when E2 produced one). `None`
            before that check has run.
        preprocess_warnings: E2's `PreprocessResult.warnings`, or `None`
            before E2 has run.
        gatekeeper_decision: E5's `GateDecision.to_dict()`, or `None` before
            the gatekeeper has run.
        created_at: `time.time()` when the job was created.
        updated_at: `time.time()` of the most recent state change.
    """

    job_id: str
    state: JobState
    stage: str
    progress: float
    case_id: str
    error: str | None
    ingest_result: dict[str, Any] | None
    input_qc_pre: dict[str, Any] | None
    input_qc_post: dict[str, Any] | None
    preprocess_warnings: tuple[str, ...] | None
    gatekeeper_decision: dict[str, Any] | None
    created_at: float
    updated_at: float


# In-process job store -- see the module docstring for why this is a
# separate dict from `jobs._JOBS`, under the same lock-guards-both-together
# pattern `jobs.py` uses.
_LOCK = threading.Lock()
_CLINICAL_JOBS: dict[str, ClinicalJob] = {}


def _conformal_fit_path() -> Path:
    """Path to the frozen conformal `fit.json` for the one deployed model.

    Hardcoded to `outputs/conformal/neurovision/fit.json` rather than
    derived from any `Settings`/`cfg` value -- this artifact belongs to the
    ONE deployed `neurovision` checkpoint, the same convention
    `configs/analysis/default.yaml`'s `qc_validate.checkpoint` already uses
    for the same reason (see that file's comment on why it is not
    `${output_dir}`-interpolated). A function, not a bare module-level
    constant, so a test can monkeypatch it to point at a small synthetic
    `fit.json` instead of the real repo file.

    Returns:
        `REPO_ROOT / "outputs/conformal/neurovision/fit.json"`.
    """
    return REPO_ROOT / "outputs" / "conformal" / _CLINICAL_EXPERIMENT / "fit.json"


# ---------------------------------------------------------------------------
# create_clinical_job / get / list / delete / start
# ---------------------------------------------------------------------------


def create_clinical_job(settings: Settings, dicom_zip: bytes) -> ClinicalJob:
    """Validates and extracts a raw DICOM study .zip, then queues a job.

    Every zip member is checked for path traversal ("zip slip") BEFORE any
    member is extracted: a member whose path would resolve outside this
    job's own `raw_dicom/` directory (e.g. `"../../etc/passwd"`, or an
    absolute path) causes the WHOLE archive to be rejected, with nothing
    extracted -- the same "never trust a caller-controlled path component"
    discipline `jobs.py`'s module docstring states for upload role names,
    applied here to a zip member's internal path instead.

    Args:
        settings: Resolved backend settings, used only to locate
            `jobs.job_root(settings)`.
        dicom_zip: Raw bytes of a `.zip` archive containing one DICOM
            study's files, in an arbitrarily nested directory layout.

    Returns:
        A new `ClinicalJob` in state `"queued"`.

    Raises:
        ValueError: If `dicom_zip` is empty; is larger than
            `MAX_STUDY_ZIP_BYTES`; is not a valid zip archive; or contains a
            member whose path would extract outside this job's raw DICOM
            directory (naming the offending member).
    """
    if not dicom_zip:
        raise ValueError("create_clinical_job: dicom_zip is empty")
    if len(dicom_zip) > MAX_STUDY_ZIP_BYTES:
        raise ValueError(
            f"create_clinical_job: dicom_zip is {len(dicom_zip)} bytes, over the "
            f"{MAX_STUDY_ZIP_BYTES}-byte limit"
        )

    try:
        archive = zipfile.ZipFile(io.BytesIO(dicom_zip))
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"create_clinical_job: dicom_zip is not a valid zip archive: {exc}"
        ) from exc

    job_id = uuid.uuid4().hex
    raw_dicom_dir = jobs.job_root(settings) / job_id / "raw_dicom"
    root_resolved = raw_dicom_dir.resolve()

    with archive:
        members = [m for m in archive.infolist() if not m.is_dir()]

        # Validate EVERY member before extracting ANY of them: a zip-slip
        # guard that rejects the archive after having already extracted a
        # few "safe" members first would still have written attacker-chosen
        # content to disk.
        for member in members:
            destination = (raw_dicom_dir / member.filename).resolve()
            escapes = destination != root_resolved and root_resolved not in destination.parents
            if escapes:
                raise ValueError(
                    f"create_clinical_job: zip member {member.filename!r} would extract "
                    "outside this job's raw DICOM directory; refusing the whole archive "
                    "and extracting nothing."
                )

        raw_dicom_dir.mkdir(parents=True, exist_ok=True)
        for member in members:
            archive.extract(member, path=raw_dicom_dir)

    now = time.time()
    job = ClinicalJob(
        job_id=job_id,
        state="queued",
        stage="queued",
        progress=0.0,
        case_id=job_id,
        error=None,
        ingest_result=None,
        input_qc_pre=None,
        input_qc_post=None,
        preprocess_warnings=None,
        gatekeeper_decision=None,
        created_at=now,
        updated_at=now,
    )
    with _LOCK:
        _CLINICAL_JOBS[job_id] = job
    logger.info("Queued clinical job %s (%d DICOM file(s))", job_id, len(members))
    return job


def get_clinical_job(job_id: str) -> ClinicalJob | None:
    """Current state of one clinical job, or `None` if unknown.

    Returns the SAME `ClinicalJob` instance held in the in-process store,
    not a copy -- a caller polling this while `run_clinical_job` executes
    sees live `stage` / `progress` updates for free.

    Args:
        job_id: A job id, as returned by `create_clinical_job`.

    Returns:
        The `ClinicalJob`, or `None` if `job_id` is not known.
    """
    with _LOCK:
        return _CLINICAL_JOBS.get(job_id)


def list_clinical_jobs() -> list[ClinicalJob]:
    """All known clinical jobs, newest first (by `created_at`)."""
    with _LOCK:
        return sorted(_CLINICAL_JOBS.values(), key=lambda job: job.created_at, reverse=True)


def _update_clinical_job(job: ClinicalJob, **fields: object) -> None:
    """Mutates `job`'s fields in place and bumps `updated_at`, under the lock."""
    with _LOCK:
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def delete_clinical_job(settings: Settings, job_id: str) -> bool:
    """Removes a clinical job and everything it wrote to disk.

    Args:
        settings: Resolved backend settings, used to locate `jobs.job_root`.
        job_id: A job id.

    Returns:
        `True` if the job existed (in the in-memory store, on disk, or
        both) and was removed; `False` if there was nothing to remove.

    Raises:
        ValueError: If `job_id` resolves to a directory outside
            `jobs.job_root(settings)` -- the same "never trust an id used as
            a path component" discipline `jobs.delete_job` applies.
    """
    root = jobs.job_root(settings).resolve()
    job_dir = (root / job_id).resolve()
    if job_dir != root and root not in job_dir.parents:
        raise ValueError(
            f"delete_clinical_job: resolved job directory {job_dir} falls outside "
            f"job_root {root}; refusing to delete"
        )

    with _LOCK:
        existed_in_store = _CLINICAL_JOBS.pop(job_id, None) is not None

    existed_on_disk = job_dir.is_dir()
    if existed_on_disk:
        shutil.rmtree(job_dir)

    return existed_in_store or existed_on_disk


def start_clinical_job(settings: Settings, job_id: str) -> None:
    """Runs a queued clinical job on a background daemon thread.

    Fire-and-forget, same pattern as `jobs.start_job`: progress and the
    terminal result are read back through `get_clinical_job`, never through
    this function's return value (it has none).

    Args:
        settings: Resolved backend settings, forwarded to `run_clinical_job`.
        job_id: A job created by `create_clinical_job`.
    """
    thread = threading.Thread(
        target=run_clinical_job,
        args=(settings, job_id),
        daemon=True,
        name=f"nvx-clinical-job-{job_id[:8]}",
    )
    thread.start()


# ---------------------------------------------------------------------------
# Config composition
# ---------------------------------------------------------------------------


def _compose_clinical_cfg() -> Any:
    """Composes the shared, CPU-only config used by E1, E2, E3 and E5.

    Unlike `inference._compose_cfg`, this never builds a segmentation model
    (E1-E3 read only `cfg.clinical.*`; E5 reads only
    `cfg.clinical.gatekeeper`), so it needs no `+experiment=...` override --
    just `device=cpu`, so nothing downstream can disagree about the device.
    `cfg.clinical.*` and `cfg.analysis.*` are both reachable off the result:
    `configs/config.yaml`'s `defaults:` list already includes `clinical:
    default` and `analysis: default`.

    Guarded by `inference._HYDRA_LOCK` -- the SAME lock every other Hydra
    composition in this backend uses, since `hydra.compose` mutates a
    process-global singleton and two threads composing at once can corrupt
    each other's state.

    Returns:
        The composed `DictConfig`.
    """
    import hydra

    config_dir = str(REPO_ROOT / "configs")
    with inference._HYDRA_LOCK:
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name="config", overrides=["device=cpu"])
    return cfg


def _load_qc_model_and_cfg(qc_checkpoint: Path) -> tuple[Any, Any]:
    """Builds `SegQC`, composes its own `model=segqc` config, and loads its checkpoint.

    Degrades gracefully rather than crashing the whole job: a fresh clone of
    this repo may not have `outputs/neurovision/qc/best.pt` restored yet, and
    a missing QC checkpoint should mean "predicted_dice is unavailable for
    this case", never a hard failure of the whole clinical pipeline.

    Args:
        qc_checkpoint: Path to the trained `SegQC` checkpoint (this
            project's convention: `outputs/neurovision/qc/best.pt`).

    Returns:
        `(model, cfg)`: `model` in eval mode on CPU, and the composed config
        it was built from (`_live_predicted_dice` needs
        `cfg.inference.postprocess` off it). `(None, None)` if
        `qc_checkpoint` does not exist on disk.
    """
    if not qc_checkpoint.is_file():
        logger.warning(
            "_load_qc_model_and_cfg: no SegQC checkpoint at %s; predicted_dice will be "
            "unavailable for this clinical job.",
            qc_checkpoint,
        )
        return None, None

    import hydra

    config_dir = str(REPO_ROOT / "configs")
    with inference._HYDRA_LOCK:
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name="config", overrides=["model=segqc", "device=cpu"])

    device = get_device(cfg)  # resolves to cpu, since cfg.device == "cpu" above
    model = build_segqc(cfg).to(device)
    load_checkpoint(
        qc_checkpoint, model, optimizer=None, map_location=str(device), restore_rng=False
    )
    model.eval()
    logger.info("_load_qc_model_and_cfg: loaded SegQC checkpoint %s", qc_checkpoint)
    return model, cfg


def clinical_segmentation_settings(job_prep_dir: Path, job_cache_dir: Path) -> Settings:
    """Builds a dedicated `Settings` for the segmentation step of a clinical job.

    A clinical job ALWAYS segments with the `neurovision` checkpoint,
    independent of whatever experiment the generic `app.backend.config
    .Settings` the rest of the backend is configured to show (see the
    module docstring). So this function never reads a caller-supplied
    `Settings` for `checkpoint` / `experiment` at all: `checkpoint` is
    resolved from the (new) `NVX_CLINICAL_CHECKPOINT` environment variable,
    repo-relative-by-default via `_path_env` (the same resolution
    `app.backend.config` already uses for every other path-shaped setting),
    and `experiment` is the literal `"neurovision"`.

    `eval_dir` / `max_cases` / `report_dir` are never read by
    `app.backend.inference.segment_case`, so they are set to harmless,
    unused placeholders here rather than left as a required-positional
    crash. `demo_overlap` DOES affect `segment_case` (it drives the
    sliding-window overlap `inference._compose_cfg` composes with), so it is
    resolved from the same `NVX_DEMO_OVERLAP` environment variable
    `app.backend.config.get_settings()` itself reads, rather than a bare
    hardcoded number.

    Args:
        job_prep_dir: This job's own preprocessed-case directory
            (`jobs.job_root(settings)/job_id/prep`).
        job_cache_dir: This job's own live-inference cache directory
            (`jobs.job_root(settings)/job_id/cache`).

    Returns:
        A `Settings` ready to pass to `inference.segment_case`.
    """
    checkpoint = _path_env("NVX_CLINICAL_CHECKPOINT", "outputs/neurovision/checkpoints/best.pt")
    demo_overlap = float(os.environ.get("NVX_DEMO_OVERLAP", "0.25"))
    return Settings(
        prep_dir=job_prep_dir,
        eval_dir=job_cache_dir,  # unused by segment_case; harmless placeholder
        checkpoint=checkpoint,
        experiment=_CLINICAL_EXPERIMENT,
        cache_dir=job_cache_dir,
        max_cases=1,  # unused by segment_case
        demo_overlap=demo_overlap,
        report_dir=job_cache_dir,  # unused by segment_case; harmless placeholder
    )


# ---------------------------------------------------------------------------
# The two derived, label-free safety signals
# ---------------------------------------------------------------------------


def _live_predicted_dice(
    cfg: Any,
    qc_model: Any,
    logits: np.ndarray,
    image_modality: np.ndarray,
    regions: Sequence[str],
) -> dict[str, float]:
    """Runs the QC model on one case's saved logits, one region at a time.

    Args:
        cfg: The `model=segqc` composed config `qc_model` was built from
            (read for `cfg.inference.postprocess` and
            `cfg.analysis.qc.target_shape`).
        qc_model: The loaded `SegQC`, in eval mode.
        logits: `(3, D, H, W)` raw model logits, channel order `(ET, TC,
            WT)` -- exactly what `inference.cached_logits_path` persists.
        image_modality: `(D, H, W)` one MRI modality's voxel values (picked
            by `cfg.analysis.qc.modality_index` from the case's
            preprocessed `image.npy`).
        regions: Region names to score, e.g. `("WT", "TC")`.

    Returns:
        region -> the QC model's predicted Dice in `[0, 1]`, one entry per
        entry of `regions`.
    """
    pred_mask = (
        postprocess_logits(torch.from_numpy(logits).unsqueeze(0), cfg)[0].numpy().astype(np.uint8)
    )  # (3, D, H, W)
    entropy = entropy_from_logits(torch.from_numpy(logits)).numpy()  # (3, D, H, W), nats

    # `label` is a documented, deliberate dummy: `pack_sample` never reads
    # `.label` (verified by reading its body before writing this code), and
    # there is no ground truth for a live clinical case -- a zero array is a
    # structural placeholder, never a claim about this case.
    arrays = CaseArrays(
        pred_mask=pred_mask,
        label=np.zeros(pred_mask.shape[1:], dtype=np.int64),
        image_modality=image_modality,
        entropy=entropy,
    )

    target_shape = tuple(int(v) for v in cfg.analysis.qc.target_shape)
    result: dict[str, float] = {}
    for region in regions:
        region_channel = REGION_NAMES.index(region)
        packed = pack_sample(arrays, arrays.pred_mask, region_channel, target_shape)
        packed = packed.unsqueeze(0)  # (1, 3, D', H', W'): pack_sample has no batch dim
        with torch.no_grad():
            dice = qc_predicted_dice(qc_model(packed))
        result[region] = float(dice.item())
    return result


def _live_conformal_band_width(
    logits: np.ndarray,
    region_channel: int,
    fitted_threshold: float,
    reference_threshold: float = 0.5,
) -> float:
    """Ratio of the conformal (fitted-threshold) mask size to the reference-threshold mask size.

    Fully label-free: unlike `neurovision.uncertainty.conformal.CaseLossCurve
    .mask_inflation` (which needs ground truth to build its false-negative
    curve across many calibration cases), this is just two voxel counts of
    the SAME predicted probability map at two thresholds -- exactly the
    signal `judge_conformal_band` in `neurovision.inference.gatekeeper`
    needs, computed with no label at all.

    Args:
        logits: `(3, D, H, W)` raw model logits, channel order `(ET, TC,
            WT)`.
        region_channel: Which of the 3 region channels to read (0=ET, 1=TC,
            2=WT, per `neurovision.data.transforms.REGION_NAMES`).
        fitted_threshold: The region's fitted conformal threshold (from
            `_load_conformal_fitted_thresholds`).
        reference_threshold: The ordinary segmentation threshold the fitted
            threshold is compared against. Default `0.5`, this project's
            standard operating point.

    Returns:
        `fitted_voxels / ref_voxels`, or `float("nan")` if the
        reference-threshold mask has zero voxels (an empty reference mask
        makes the ratio undefined, not zero -- mirroring
        `CaseLossCurve.mask_inflation`'s own 0/0 handling).
        `neurovision.inference.gatekeeper`'s own `_is_bad_number` already
        treats NaN as "unmeasured" and REFUSEs an enabled signal on it, so
        this is not special-cased any further here.
    """
    prob = expit(logits[region_channel].astype(np.float64))
    ref_voxels = int((prob > reference_threshold).sum())
    fitted_voxels = int((prob > fitted_threshold).sum())
    if ref_voxels == 0:
        return float("nan")
    return fitted_voxels / ref_voxels


def clinical_conformal_band_mask(
    logits: np.ndarray,
    region_channel: int,
    fitted_threshold: float,
    *,
    reference_threshold: float = 0.5,
) -> np.ndarray:
    """Per-voxel conformal band, for visualization: which voxels are in which mask.

    Pure and label-free, like `_live_conformal_band_width` (which this does
    NOT replace -- that function stays the gatekeeper's scalar signal; this
    one is for drawing an overlay, at the SAME two thresholds on the SAME
    two masks).

    Args:
        logits: `(3, D, H, W)` raw model logits, channel order (ET, TC, WT).
        region_channel: Which region channel to read (0=ET, 1=TC, 2=WT).
        fitted_threshold: The region's fitted conformal threshold.
        reference_threshold: The ordinary segmentation threshold, default 0.5.

    Returns:
        `(D, H, W)` uint8 array:
          - 0   = outside the conservative (fitted-threshold) mask entirely
          - 128 = inside the conservative mask but outside the reference
                  (0.5) mask -- the "uncertain band": voxels only the
                  distribution-free guarantee covers, not the point estimate
          - 255 = inside the reference (0.5) mask (and therefore, by the
                  masks' nesting, inside the conservative mask too, since a
                  lower/more permissive threshold can only add voxels, never
                  remove them -- asserted below rather than assumed, since a
                  mis-fitted threshold on the wrong side of 0.5 would
                  silently produce a nonsensical band otherwise)

    Raises:
        ValueError: If `fitted_threshold` is on the wrong side of
            `reference_threshold` for this data -- i.e. some voxel is inside
            the reference (0.5) mask but NOT inside the conservative
            (fitted-threshold) mask. This should never happen for a validly
            fitted conformal threshold under this project's
            false-negative-rate loss (see the module docstring's conformal
            section), so it is treated as a data problem worth surfacing
            loudly rather than a buffer that would look plausible while
            encoding a violated invariant.
    """
    prob = expit(logits[region_channel].astype(np.float64))
    reference_mask = prob > reference_threshold
    conservative_mask = prob > fitted_threshold

    # Voxels the reference mask claims but the "conservative" mask does not
    # -- if any exist, fitted_threshold is not actually more permissive than
    # reference_threshold, and the nesting this function's Returns section
    # promises does not hold.
    violation = reference_mask & ~conservative_mask
    if violation.any():
        region_name = REGION_NAMES[region_channel]
        raise ValueError(
            f"clinical_conformal_band_mask: invariant violated for region {region_name!r} "
            f"(region_channel={region_channel}) -- fitted_threshold={fitted_threshold} "
            f"produced a conservative mask SMALLER than the reference mask at "
            f"reference_threshold={reference_threshold} ({int(violation.sum())} voxel(s) "
            "affected). A validly-fitted conformal threshold must sit on the permissive side "
            "of the reference threshold; refusing to return a band buffer that would encode "
            "this violated invariant."
        )

    band = np.zeros(prob.shape, dtype=np.uint8)
    band[conservative_mask] = 128
    band[reference_mask] = 255  # overwrites 128 wherever both masks agree
    return band


def _load_conformal_fitted_thresholds(regions: Sequence[str], alpha: float) -> dict[str, float]:
    """Loads the fitted conformal threshold for each region, at one alpha, from `fit.json`.

    Degrades gracefully when the file itself is absent (same philosophy as
    `_load_qc_model_and_cfg`): a caller with an empty dict back means
    `conformal_band` is simply not measured for this job, not a crash.

    Args:
        regions: Region names to look up, e.g. `("WT", "TC")`.
        alpha: `cfg.clinical.gatekeeper.conformal_alpha`, e.g. `0.10`. Keys
            are matched with a PLAIN f-string (`f"{alpha}"`), which for
            `alpha=0.1` gives `"0.1"` -- matching the key `"WT__alpha_0.1"`
            `scripts/conformal.py`'s `_fit_payload` actually writes. A
            fixed-decimals format (`f"{alpha:.2f}"` -> `"0.10"`) would
            silently miss every entry.

    Returns:
        region -> fitted threshold. `{}` if `_conformal_fit_path()` does not
        exist on disk.

    Raises:
        ValueError: If a region's key is absent from the file, or if that
            entry's `"threshold"` is `None` (an infeasible fit cannot back a
            live signal).
    """
    path = _conformal_fit_path()
    if not path.is_file():
        logger.warning(
            "_load_conformal_fitted_thresholds: no fit.json at %s; conformal_band will be "
            "unavailable for this clinical job.",
            path,
        )
        return {}

    payload = json.loads(path.read_text())
    result: dict[str, float] = {}
    for region in regions:
        key = f"{region}__alpha_{alpha}"
        if key not in payload:
            raise ValueError(f"_load_conformal_fitted_thresholds: missing key {key!r} in {path}.")
        threshold = payload[key]["threshold"]
        if threshold is None:
            raise ValueError(
                f"_load_conformal_fitted_thresholds: {path}'s entry for {key!r} has "
                "threshold=null (an infeasible fit); it cannot back a live signal."
            )
        result[region] = float(threshold)
    return result


# ---------------------------------------------------------------------------
# _ingest_result_to_dict
# ---------------------------------------------------------------------------


def _ingest_result_to_dict(result: IngestResult) -> dict[str, Any]:
    """Turns E1's `IngestResult` into a plain, `json.dumps`-safe dict.

    `IngestResult` has no built-in `.to_dict()`. Mirrors the shape
    `dicom_ingest.ingest_study`'s own `_build_manifest` writes to
    `ingest_manifest.json` on disk, rebuilt here from the already-in-memory
    result so a `ClinicalJob`'s own state carries the same audit trail
    without re-reading that file.

    Args:
        result: An `IngestResult` from `neurovision.data.dicom_ingest
            .ingest_study`.

    Returns:
        A JSON-serialisable dict: `paths` (role -> str path), `assignments`
        (series_uid -> `{role, score, reasons, outcome}`), `missing_roles`,
        `rejected` (`[{series_uid, reason}, ...]`), `warnings`.
    """
    return {
        "paths": {role: str(path) for role, path in result.paths.items()},
        "assignments": {
            uid: {
                "role": assignment.role,
                "score": assignment.score,
                "reasons": list(assignment.reasons),
                "outcome": assignment.outcome.value,
            }
            for uid, assignment in result.assignments.items()
        },
        "missing_roles": list(result.missing_roles),
        "rejected": [{"series_uid": uid, "reason": reason} for uid, reason in result.rejected],
        "warnings": list(result.warnings),
    }


# ---------------------------------------------------------------------------
# _export_dicom_seg (E6): the mask, as a DICOM Segmentation object.
# ---------------------------------------------------------------------------


def _class_map_to_regions(class_map: np.ndarray) -> np.ndarray:
    """Expands a `{0,1,2,3}` class map into `(3, D, H, W)` binary (ET, TC, WT) regions.

    The inverse of `neurovision.reporting.dicom_seg.classes_from_regions` --
    `write_dicom_seg` needs region channels IN, not a class map. Same nesting
    definition as `neurovision.data.transforms.ConvertToRegionsd._convert`
    and `app.backend.volumes.region_voxel_counts` (the canonical source of
    truth for these three lines): ET is class 3 alone, TC is necrotic core
    (1) or enhancing (3), WT is any nonzero class. Written directly rather
    than reusing `ConvertToRegionsd` itself, which is a MONAI dict-transform
    built for a keyed, tensor-based data-loading pipeline and is not a clean
    fit for a bare numpy array outside one.

    Args:
        class_map: `(D, H, W)` integer array, values in `{0, 1, 2, 3}`.

    Returns:
        `(3, D, H, W)` `uint8` array, channel order `(ET, TC, WT)`, values
        in `{0, 1}`.
    """
    et = class_map == 3
    tc = et | (class_map == 1)
    wt = tc | (class_map == 2)
    return np.stack([et, tc, wt], axis=0).astype(np.uint8)


def _series_uid_for_role(ingest_result: IngestResult, role: str) -> str | None:
    """Finds which series_uid E1 assigned to `role`, or `None` if none was.

    Args:
        ingest_result: E1's result.
        role: One of `neurovision.data.dicom_ingest.ROLES`.

    Returns:
        The series_uid whose `RoleAssignment.role == role`, or `None` if no
        series in `ingest_result.assignments` was assigned that role.
    """
    for uid, assignment in ingest_result.assignments.items():
        if assignment.role == role:
            return uid
    return None


def _collect_source_datasets(raw_dicom_dir: Path, series_uid: str) -> list[Any]:
    """Reads every DICOM file under `raw_dicom_dir` belonging to one series, header-only.

    Mirrors `neurovision.data.dicom_ingest.convert_series`'s own series-matching
    scan exactly (that function is the reference for "the real, correct way to
    collect one series' files" under this job's `raw_dicom/` directory): walk
    every file recursively, read its header only (`stop_before_pixels=True`,
    matching `dicom_seg.read_source_geometry`'s own stated "header only -- pixel
    data need not be loaded" contract), silently skip anything that is not
    readable DICOM (a DICOMDIR index, a stray README -- the same tolerance
    `read_series_headers`/`convert_series` already have for a real study
    folder), and keep the ones whose `SeriesInstanceUID` matches. This does NOT
    re-run role assignment; the series_uid to look for is already decided (see
    `_series_uid_for_role`).

    Args:
        raw_dicom_dir: This job's own extracted DICOM study directory
            (`<job_dir>/raw_dicom`).
        series_uid: The target series' `SeriesInstanceUID`.

    Returns:
        One `pydicom.Dataset` per matching file, in `raw_dicom_dir.rglob("*")`
        order (sorted, for a deterministic result). Possibly empty.
    """
    import pydicom

    datasets: list[Any] = []
    for path in sorted(Path(raw_dicom_dir).rglob("*")):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:  # noqa: BLE001 - a study folder may hold non-DICOM files
            continue
        if str(getattr(dataset, "SeriesInstanceUID", "")) == series_uid:
            datasets.append(dataset)
    return datasets


def _validate_dicom_seg_cfg(cfg: Any) -> None:
    """Validates the STATIC, config-derived parts of `cfg.clinical.dicom_seg`.

    Deliberately separate from `write_dicom_seg`'s own `ValueError`s (raised
    inside `neurovision.reporting.dicom_seg.write_dicom_seg`, per its
    `Raises` docstring): that function's checks mix genuine PER-CASE outcomes
    (a geometry mismatch against one job's own source series; one job's
    all-empty predicted class map) with checks that are actually properties
    of the CONFIG alone (`segmentation_type` not `"BINARY"`; an over-length
    `series_description`) -- wrong here means wrong for every job, forever,
    not just this one. `_export_dicom_seg`'s own `except ValueError` around
    `write_dicom_seg`'s call treats every one of those identically (a routine
    WARNING, no SEG object for this job), which is correct for the per-case
    ones but would quietly bury a config bug behind that same unremarkable
    per-job log line. This function exists so the caller (`run_clinical_job`)
    can check the config-derived pieces ONCE, from a call site that is NOT
    inside that narrow `except ValueError`, so a config bug instead surfaces
    through the caller's generic "something unexpected happened in this
    chain" handling -- logged distinguishably (see that call site) rather
    than silently.

    Does not check `regions`' shape/values (also named in
    `write_dicom_seg`'s `Raises` section): that one is driven entirely by
    `_class_map_to_regions`'s own output, which is well-formed by
    construction, never by anything in `cfg` -- there is no static config
    value to validate for it.

    Args:
        cfg: The composed config; reads `cfg.clinical.dicom_seg`.

    Raises:
        ValueError: If `cfg.clinical.dicom_seg.segmentation_type` is not
            `"BINARY"`, or if `cfg.clinical.dicom_seg.series_description` is
            longer than 64 characters -- DICOM's Long String (LO) value
            representation, which is what `SeriesDescription` uses, caps a
            value at 64 characters (the same limit
            `write_dicom_seg` itself enforces immediately before writing;
            checking it here as well means a misconfigured description is
            caught before any of this job's uncrop/resample work runs, not
            only after `write_dicom_seg` gets around to it).
    """
    dicom_seg_cfg = cfg.clinical.dicom_seg

    if dicom_seg_cfg.segmentation_type != "BINARY":
        raise ValueError(
            "_validate_dicom_seg_cfg: cfg.clinical.dicom_seg.segmentation_type must be "
            f"'BINARY', got {dicom_seg_cfg.segmentation_type!r}. This is a STATIC "
            "configuration problem -- it would misconfigure DICOM-SEG export for every "
            "clinical job, not just this one -- so it is checked here, separately from "
            "write_dicom_seg's own per-job ValueError, precisely so it cannot be mistaken for "
            "a routine per-case geometry refusal."
        )

    series_description = str(dicom_seg_cfg.series_description)
    if len(series_description) > 64:
        raise ValueError(
            "_validate_dicom_seg_cfg: cfg.clinical.dicom_seg.series_description is "
            f"{len(series_description)} characters, over DICOM's Long String (LO) "
            "SeriesDescription limit of 64. This is a STATIC configuration problem, checked "
            "here separately from write_dicom_seg's own per-job ValueError for the same "
            "reason as the segmentation_type check above."
        )


def _export_dicom_seg(
    job: ClinicalJob,
    preprocess_result: PreprocessResult,
    ingest_result: IngestResult,
    clinical_settings: Settings,
    job_dir: Path,
    cfg: Any,
) -> Path | None:
    """Exports E6's DICOM-SEG object for one finished clinical job, or gives up cleanly.

    The chain, in the order actually executed: (1) uncrop the cropped
    research-frame prediction back to the FULL, uncropped atlas-space grid E2
    produced (`neurovision.inference.postprocess.uncrop_to_original`, using
    `<job_prep_dir>/<case_id>/meta.json`'s `bbox`/`original_shape` -- read via
    `app.backend.volumes.read_meta`, the same helper the research/demo path
    already uses for this exact file); (2) resample THAT ATLAS-SPACE CLASS MAP
    -- not yet split into regions -- into the center-role modality's native
    (pre-E2) geometry, through E2's own saved inverse transform
    (`neurovision.data.clinical_resample.resample_mask_to_source`); this order
    matters and cannot be reversed, because `resample_mask_to_source`'s own
    docstring states its `mask` argument is a class map, not region channels;
    (3) only THEN split the resampled, now-native-space class map into the
    three nested ET/TC/WT region channels `write_dicom_seg` actually wants
    (`_class_map_to_regions`); (4) find the center-role series' own raw DICOM
    headers under this job's `raw_dicom/` directory (`_series_uid_for_role` +
    `_collect_source_datasets`); (5) write the SEG object
    (`neurovision.reporting.dicom_seg.write_dicom_seg`).

    `preprocess_result.plan.center_role` is used throughout -- as the resample
    target AND as the DICOM-SEG reference series -- rather than any separately
    chosen "which modality to attach the SEG to" policy, matching how this
    project already treats the center modality as the pipeline's anchor.

    This is a SUPPLEMENTARY artifact for PACS/OHIF interoperability, never a
    requirement for the clinical decision the job has already made by the time
    this runs (see `run_clinical_job`'s own Grad-CAM block, computed just
    before this one, for the same philosophy). Three "cannot export, but
    nothing is wrong" cases are absorbed HERE, returning `None` rather than
    raising: no series was assigned the center role at all; no raw DICOM files
    under `raw_dicom/` actually match that series_uid; and `write_dicom_seg`'s
    own named refusal (`ValueError` -- e.g. a residual geometry mismatch after
    a correct resample, which should be rare and worth logging distinctly from
    a bug, but is handled identically). Any OTHER exception in this chain (a
    missing E2 transformations directory, a bad resample, a corrupt atlas
    NIfTI, ...) is NOT caught here -- it propagates, so `run_clinical_job`'s
    own outer try/except around this call (mirroring the Grad-CAM loop's
    per-region try/except) is the second, generic layer of the same
    failure-isolation contract.

    Args:
        job: The clinical job being exported. `job.case_id` names the cached
            prediction and the output filename.
        preprocess_result: E2's result -- `.outputs[plan.center_role]` is read
            for the atlas affine; `.plan.center_role` is the modality the SEG
            is anchored to throughout; `.transformations_dir` backs the
            resample.
        ingest_result: E1's result -- `.assignments` is searched for the
            series_uid E1 gave the center role; `.paths[center_role]` is the
            target native-geometry NIfTI `resample_mask_to_source` resamples
            onto.
        clinical_settings: This job's segmentation `Settings` (from
            `clinical_segmentation_settings`) -- `.prep_dir` is where
            `meta.json` for `job.case_id` lives (read via
            `app.backend.volumes.read_meta`), and it locates the cached
            prediction via `inference.cached_prediction_path`.
        job_dir: This job's own root directory (`<job root>/<job_id>`).
            `raw_dicom/` (E1's extracted study) and `dicom_seg/`/
            `dicom_seg_work/` (this function's own output and resample
            working directory) all live under here.
        cfg: The composed config, passed straight through to `write_dicom_seg`
            (reads `cfg.clinical.dicom_seg`).

    Returns:
        Path to the written `.dcm` file, or `None` if no SEG object could be
        produced for one of the reasons named above (logged at the point it
        happened).
    """
    from neurovision.data.clinical_resample import resample_mask_to_source
    from neurovision.reporting.dicom_seg import write_dicom_seg

    center_role = preprocess_result.plan.center_role

    # --- 1. Uncrop the research-frame (cropped) prediction back to the full,
    # uncropped atlas-space grid E2 produced. ---------------------------------
    pred_path = inference.cached_prediction_path(clinical_settings, job.case_id)
    cropped_class_map = np.load(pred_path)  # (D, H, W), cropped research frame
    meta = volumes.read_meta(job.case_id, clinical_settings)
    atlas_class_map = uncrop_to_original(cropped_class_map, meta.bbox, meta.original_shape)

    # --- 2. Resample the atlas-space CLASS MAP -- resample_mask_to_source's
    # own docstring is explicit that `mask` is a class map, not region
    # channels, so this must happen BEFORE the region split, not after. ------
    atlas_affine = nib.load(str(preprocess_result.outputs[center_role])).affine
    resample_dir = job_dir / "dicom_seg_work"
    resampled_path = resample_mask_to_source(
        atlas_class_map,
        atlas_affine,
        preprocess_result.transformations_dir,
        center_role,
        ingest_result.paths[center_role],
        out_dir=resample_dir,
    )
    native_class_map = np.asarray(nib.load(str(resampled_path)).dataobj).astype(np.uint8)

    # --- 3. NOW split the resampled, native-space class map into the three
    # nested region channels write_dicom_seg wants. --------------------------
    regions = _class_map_to_regions(native_class_map)

    # --- 4. Find the center-role series' own raw DICOM headers. -------------
    series_uid = _series_uid_for_role(ingest_result, center_role)
    if series_uid is None:
        logger.warning(
            "_export_dicom_seg: no ingest assignment names role %r as a series_uid for job "
            "%s; skipping DICOM-SEG export.",
            center_role,
            job.job_id,
        )
        return None

    raw_dicom_dir = job_dir / "raw_dicom"
    source_datasets = _collect_source_datasets(raw_dicom_dir, series_uid)
    if not source_datasets:
        logger.warning(
            "_export_dicom_seg: no DICOM file(s) under %s match series_uid %r for job %s; "
            "skipping DICOM-SEG export.",
            raw_dicom_dir,
            series_uid,
            job.job_id,
        )
        return None

    # --- 5. Write, or accept write_dicom_seg's own named refusal. -----------
    out_path = job_dir / "dicom_seg" / f"{job.case_id}.dcm"
    try:
        write_dicom_seg(cfg, regions, source_datasets, out_path)
    except ValueError as exc:
        # write_dicom_seg's own refusal (a geometry mismatch, an all-empty
        # class map, ...) -- see its docstring's Raises section. A correct
        # resample should make this rare; it is worth logging distinctly from
        # a bug, but the outcome (no SEG artifact, job still reaches "done")
        # is identical either way.
        logger.warning("_export_dicom_seg: write_dicom_seg refused for job %s: %s", job.job_id, exc)
        return None

    logger.info("_export_dicom_seg: wrote DICOM-SEG for job %s to %s", job.job_id, out_path)
    return out_path


# ---------------------------------------------------------------------------
# _generate_report: the Phase 4 structured anatomical report, computed live.
# ---------------------------------------------------------------------------


def _generate_report(
    job: ClinicalJob,
    clinical_settings: Settings,
    job_dir: Path,
    cfg: Any,
) -> Path | None:
    """Builds and caches one clinical job's structured anatomical report (Phase 4).

    `scripts/report.py` (the offline batch driver) only JOINS a `burden.csv`
    and an `anatomy.csv` that `scripts/burden.py` / `scripts/localize.py`
    already wrote for MANY cases -- a clinical job has exactly one case and
    nothing written for it yet, so there is no batch CSV to join. This
    function instead calls the single-case functions those two batch
    drivers call underneath (`localize_case`, `summarize_case`,
    `burden_profile`, `involvement_profile`) directly, on THIS job's own
    already-cached prediction and its own `meta.json`, and feeds their
    outputs straight into `build_report` -- the same assembly
    `scripts/report.py::report_one` does, minus the CSV round trip. One
    consequence of skipping that round trip: `report_one` pops
    `INVOLVEMENT_FIELDS` back OUT of an `anatomy_summary` row only because
    `scripts/localize.py`'s batch driver had already merged them IN for the
    CSV to carry as one row. `involvement_profile` is called here directly
    and its result is never merged into `summarize_case`'s own output in the
    first place, so there is nothing to split back out.

    Two pieces of `localize_one`'s own post-processing are replicated here,
    NOT skipped: (1) the `cfg.analysis.localize.min_frac` filter, dropping
    non-`"unlabelled"` rows whose `frac_of_structure` AND `frac_of_tumour`
    are both below threshold, before the table reaches `summarize_case` /
    `build_report`; and (2) overwriting `summarize_case`'s
    `distance_to_eloquent_mm` (which can only ever be `0.0` on overlap or
    `NaN` otherwise, since a tidy table carries no voxel coordinates) with
    the real geometric distance from `distance_to_eloquent`, and
    `near_eloquent` recomputed from it -- otherwise a lesion sitting well
    within the configured near-eloquent threshold, but not overlapping an
    eloquent structure, would render as `distance_mm: null, near_eloquent:
    no` instead of the true, computable answer.

    Same failure-isolation PHILOSOPHY as `_export_dicom_seg` -- a report is
    a SUPPLEMENTARY artifact, never a requirement for the clinical decision
    already made by the time this runs (only reached after the gatekeeper's
    PROCEED/CAUTION) -- but a DIFFERENT SHAPE: `_export_dicom_seg` absorbs a
    few named, expected "cannot export, but nothing is wrong" cases itself
    and returns `None` for them. Every input this function touches (this
    job's own just-written `meta.json` and cached prediction; the atlas and
    knowledge files `scripts/fetch_atlas.py` and this repo's `knowledge/`
    directory are expected to provide) is either genuinely present or a real
    configuration problem worth surfacing loudly, so this function does not
    catch anything itself -- it raises freely, and `run_clinical_job`'s own
    try/except around this call (mirroring its DICOM-SEG try/except) is the
    ONLY layer that turns a failure here into "no report for this job",
    never `"failed"`.

    The atlas, knowledge base, and involvement groups are reloaded fresh on
    EVERY call, never cached across jobs at module scope. This matches the
    established convention elsewhere in this backend
    (`_load_qc_model_and_cfg` reloads the QC checkpoint every job; every
    Hydra composition here recomposes every call) for the same reason: a
    corrected `knowledge/eloquence_map.yaml` or a re-fetched atlas must take
    effect on the very next job, not after a process restart. The cost is a
    few small NIfTI files and two YAML files (single-digit MB, well under
    the segmentation step this function runs after), which this project's
    own convention already judges cheap enough to pay per request rather
    than risk a stale in-memory copy.

    Args:
        job: The clinical job to report on. `job.case_id` names the cached
            prediction and `meta.json`, and becomes the report's own
            `case_id`.
        clinical_settings: This job's segmentation `Settings` (from
            `clinical_segmentation_settings`) -- `.prep_dir` is where
            `meta.json` for `job.case_id` lives, and
            `inference.cached_prediction_path` locates the cached class map.
        job_dir: This job's own root directory (`<job root>/<job_id>`). The
            report is cached at `<job_dir>/report/<case_id>.json`, mirroring
            `_export_dicom_seg`'s own `<job_dir>/dicom_seg/<case_id>.dcm`
            convention.
        cfg: The composed config (`_compose_clinical_cfg`'s return value).
            Reads `cfg.anatomy`, `cfg.analysis.localize`,
            `cfg.analysis.burden`, and `cfg.analysis.report`.

    Returns:
        Path to the written `<case_id>.json`. This function never actually
        returns `None` itself -- the `| None` in the signature matches the
        same "supplementary, may not exist" contract `_export_dicom_seg`'s
        return type states; here it is `run_clinical_job`'s wrapping
        try/except that turns a raised exception into "no report for this
        job", not an internal `None` return.

    Raises:
        FileNotFoundError: If this job's `meta.json`, its cached prediction,
            or a required atlas/knowledge file is missing.
        ValueError: See `build_report`, `localize_case`, `load_knowledge`,
            or `load_involvement_groups`.
    """
    import math

    from neurovision.anatomy.atlas import load_atlas
    from neurovision.anatomy.burden import CaseGeometry, burden_profile, region_mask
    from neurovision.anatomy.involvement import (
        involvement_profile,
        load_involvement_groups,
        load_involvement_notes,
    )
    from neurovision.anatomy.localize import (
        atlas_for_case,
        distance_to_eloquent,
        eloquent_union_mask,
        load_classification,
        load_knowledge,
        localize_case,
        summarize_case,
    )
    from neurovision.reporting.report import Provenance, build_report, write_report
    from neurovision.utils.io import read_json, read_yaml

    # --- This job's own artifacts first, so an incomplete pipeline fails
    # fast, before anything as comparatively expensive as an atlas load. ----
    meta_path = clinical_settings.prep_dir / job.case_id / "meta.json"
    meta = read_json(meta_path)  # a raw dict, not a CaseMeta -- see this
    # function's docstring: localize_case/atlas_for_case/CaseGeometry.from_meta
    # all index it like a plain Mapping (meta["bbox"], meta["affine"], ...),
    # which app.backend.volumes.CaseMeta does not expose.

    pred_path = inference.cached_prediction_path(clinical_settings, job.case_id)
    classes = np.load(pred_path)  # (D, H, W) uint8, cropped research frame, {0,1,2,3}

    localize_cfg = cfg.analysis.localize
    burden_cfg = cfg.analysis.burden
    report_cfg = cfg.analysis.report

    # --- Atlas + knowledge base, reloaded fresh every job -- see docstring. #
    atlas = load_atlas(cfg.anatomy)
    knowledge = load_knowledge(localize_cfg.eloquence_map, localize_cfg.lobe_map, atlas)

    # `KnowledgeBase` (above) carries no `version` field of its own --
    # `Classification` does, and the lobe map's version lives in that YAML
    # file directly. Read exactly like scripts/report.py::load_inputs does,
    # rather than adding a version field to KnowledgeBase for one caller.
    classification = load_classification(localize_cfg.eloquence_map)
    lobe_doc = read_yaml(localize_cfg.lobe_map)
    knowledge_versions = {
        "eloquence_map": classification.version,
        "aal_lobes": int(lobe_doc["version"]),
    }

    # Unlike scripts/localize.py::resolve_involvement, this reads
    # cfg.analysis.localize.involvement directly rather than defensively via
    # `.get(...)`: that defensiveness exists there for an OLD, recorded
    # localize_config.yaml snapshot that might predate the key. cfg here is
    # always freshly composed by _compose_clinical_cfg, so the committed
    # configs/analysis/default.yaml's involvement block is always present.
    involvement_cfg = localize_cfg.involvement
    groups = None
    involvement_caveats: tuple[str, ...] = ()
    if bool(involvement_cfg.enabled):
        groups = load_involvement_groups(involvement_cfg.groups_map, atlas)
        involvement_caveats = load_involvement_notes(involvement_cfg.groups_map)

    # --- The four single-case functions scripts/burden.py's profile_case and
    # scripts/localize.py's localize_one each call, run directly for this one
    # case. ------------------------------------------------------------------
    regions = [str(r) for r in localize_cfg.regions]
    anatomy_table = localize_case(
        classes, atlas, meta, cropped=True, regions=regions, knowledge=knowledge
    )

    # Bug fix: this function used to pass localize_case's raw table straight
    # through, with no filtering at all. scripts/localize.py::localize_one
    # drops any non-"unlabelled" row where BOTH frac_of_structure and
    # frac_of_tumour are below cfg.analysis.localize.min_frac, BEFORE the
    # table reaches summarize_case/build_report -- this keeps boundary-noise,
    # single-voxel partial-volume overlaps out of the report's
    # eloquence.involved list and out of n_structures_involved. Replicated
    # here exactly (same three-part drop_mask localize_one builds; the
    # "unlabelled" row is never dropped by it).
    min_frac = float(localize_cfg.min_frac)
    drop_mask = (
        (anatomy_table["frac_of_structure"] < min_frac)
        & (anatomy_table["frac_of_tumour"] < min_frac)
        & (anatomy_table["structure"] != _UNLABELLED_STRUCTURE_NAME)
    )
    anatomy_table = anatomy_table.loc[~drop_mask].reset_index(drop=True)

    anatomy_summary = summarize_case(anatomy_table, knowledge)

    # Bug fix: summarize_case can only report 0.0 (the table already shows
    # overlap with an eloquent structure) or NaN (it doesn't) for
    # distance_to_eloquent_mm -- a tidy structure table carries no voxel
    # coordinates, so it has no geometric distance computation of its own by
    # design (see its own docstring). Left as-is, build_report's
    # `_near_eloquent` treats that NaN as "unmeasured" and reports
    # near_eloquent=False even when the true, computable distance is inside
    # the configured near-eloquent threshold -- a confident, well-formed "no"
    # where the right answer is "yes". Overwrite both fields with the real
    # geometric measurement here, replicating
    # scripts/localize.py::localize_one's own override exactly: the WT
    # (whole-tumour) mask against the knowledge base's eloquent-structure
    # union, cropped to this case's bbox the same way `atlas_for_case` crops
    # the parcellation, then compared against the same near-eloquent
    # threshold.
    wt_mask = region_mask(classes, "WT")
    spacing = tuple(float(s) for s in meta["spacing"])
    eloquent_mask = eloquent_union_mask(atlas, knowledge)
    bbox = tuple(tuple(int(v) for v in pair) for pair in meta["bbox"])
    cropped_eloquent_mask = eloquent_mask[tuple(slice(start, end) for start, end in bbox)]
    distance_mm = distance_to_eloquent(wt_mask, cropped_eloquent_mask, spacing=spacing)
    near_eloquent_mm = float(knowledge.near_eloquent_mm)
    # False (never NaN-propagated) when the distance is NaN: "near an
    # eloquent structure" cannot be true of an undefined distance -- same
    # rule localize_one applies.
    near_eloquent = bool(not math.isnan(distance_mm) and distance_mm <= near_eloquent_mm)
    anatomy_summary["distance_to_eloquent_mm"] = distance_mm
    anatomy_summary["near_eloquent"] = near_eloquent

    geom = CaseGeometry.from_meta(meta, cropped=True, midline_index=burden_cfg.midline_index)
    burden = burden_profile(
        classes,
        geom,
        min_volume_mm3=float(burden_cfg.min_volume_mm3),
        connectivity=int(burden_cfg.connectivity),
    )

    involvement = None
    if groups is not None:
        # WT (never ET/TC), reusing the `wt_mask` already computed above for
        # the eloquence-distance fix: this layer answers "what does the
        # lesion touch overall", the same scope Phase 3b's own driver uses it
        # for -- see scripts/localize.py::localize_one's comment on this
        # exact choice.
        parcellation, tissue = atlas_for_case(atlas, meta, cropped=True)
        involvement = involvement_profile(
            wt_mask,
            parcellation,
            tissue,
            atlas,
            groups,
            geom,
            min_overlap_mm3=float(involvement_cfg.min_overlap_mm3),
            lobe=knowledge.lobe,
        )

    coverage_line = knowledge.coverage_line(len(knowledge.eloquence))

    provenance = Provenance(
        atlas_name=atlas.name,
        atlas_version=atlas.version,
        atlas_source=atlas.source,
        atlas_licence=str(cfg.anatomy.licence),
        knowledge_versions=knowledge_versions,
        # A live clinical job's mask is always the deployed model's own
        # prediction -- there is no ground truth to report against here.
        segmentation_source="prediction",
        # No <eval_dir> exists for a live job (see this module's docstring,
        # "which model" section) -- the job id is this artifact's own
        # sufficient provenance back to exactly which run produced it.
        segmentation_dir=f"live clinical pipeline, job {job.job_id}",
        # No cheap, correct way to attach a git SHA to a live server request
        # the way scripts/report.py::git_revision does ONCE per batch run --
        # left unset (Provenance.code_revision is optional) rather than
        # guessed.
        code_revision=None,
        generated_utc=datetime.now(UTC).isoformat(),
    )

    report = build_report(
        job.case_id,
        burden,
        anatomy_table,
        anatomy_summary,
        provenance,
        evidence=knowledge.evidence,
        citation=knowledge.citation,
        classification_name=knowledge.classification_name,
        coverage_line=coverage_line,
        coverage_gaps=knowledge.coverage_gaps,
        near_eloquent_mm=knowledge.near_eloquent_mm,
        top_n=int(report_cfg.top_n),
        involvement=involvement,
        involvement_caveats=involvement_caveats,
    )

    out_dir = job_dir / "report"
    written = write_report(report, out_dir, markdown=False)
    return written["json"]


# ---------------------------------------------------------------------------
# run_clinical_job
# ---------------------------------------------------------------------------


def run_clinical_job(settings: Settings, job_id: str) -> ClinicalJob:
    """Runs one queued clinical job SYNCHRONOUSLY, end to end.

    Never raises for a job-level outcome: any exception is caught,
    `state` is set to `"failed"`, `error=str(exc)`, and it is logged at
    ERROR with a traceback -- same contract as `jobs.run_job`. A REFUSAL
    from E3 or E5, in contrast, is not an exception at all: it is a normal
    return with `state="refused"` (see the module docstring for why the two
    must never be conflated).

    Args:
        settings: Resolved backend settings. Used only to locate
            `jobs.job_root(settings)` -- see the module docstring for why
            `settings.checkpoint` / `settings.experiment` are never read
            here.
        job_id: A job created by `create_clinical_job`.

    Returns:
        The same `ClinicalJob`, now in a terminal state (`"done"`,
        `"refused"` or `"failed"`).

    Raises:
        KeyError: If `job_id` names no known job -- a programming error by
            the caller, not a job-level outcome, so it is not swallowed the
            way an in-job exception is.
    """
    job = get_clinical_job(job_id)
    if job is None:
        raise KeyError(f"run_clinical_job: unknown job_id {job_id!r}")

    job_dir = jobs.job_root(settings) / job_id
    raw_dicom_dir = job_dir / "raw_dicom"

    try:
        _update_clinical_job(job, state="running", stage="ingest", progress=0.05)
        cfg = _compose_clinical_cfg()

        # --- E1: DICOM ingest --------------------------------------------
        ingest_out_dir = job_dir / "ingest"
        ingest_result = ingest_study(cfg, raw_dicom_dir, ingest_out_dir)
        _update_clinical_job(
            job, ingest_result=_ingest_result_to_dict(ingest_result), progress=0.15
        )

        if not ingest_result.paths:
            _update_clinical_job(
                job,
                state="refused",
                stage="ingest",
                error=(
                    "No DICOM series could be assigned to any required modality role "
                    f"(missing: {', '.join(ingest_result.missing_roles)})."
                ),
            )
            return job

        # --- E3, pre-E2 ----------------------------------------------------
        _update_clinical_job(job, stage="input_qc (pre-preprocessing)")
        volumes, _brain_mask = load_volume_infos(ingest_result.paths)
        report_pre: InputQCReport = run_input_qc(cfg, volumes, brain_mask=None)
        _update_clinical_job(job, input_qc_pre=report_pre.to_dict(), progress=0.2)

        if report_pre.verdict is Severity.REFUSE:
            reasons = "; ".join(f.message for f in report_pre.refusals())
            _update_clinical_job(
                job,
                state="refused",
                stage="input_qc (pre-preprocessing)",
                error=f"Input QC refused before clinical preprocessing: {reasons}",
            )
            return job

        # --- E2: clinical preprocessing --------------------------------------
        _update_clinical_job(job, stage="clinical_preprocessing")
        clinical_prep_dir = job_dir / "clinical_prep"
        preprocess_result = preprocess_clinical_study(
            cfg, ingest_result.paths, out_dir=clinical_prep_dir
        )
        _update_clinical_job(job, preprocess_warnings=preprocess_result.warnings, progress=0.5)

        # --- E3, post-E2 -----------------------------------------------------
        _update_clinical_job(job, stage="input_qc (post-preprocessing)")
        paths_for_qc: dict[str, Path] = dict(preprocess_result.outputs)
        if preprocess_result.brain_mask.is_file():
            paths_for_qc[_BRAIN_MASK_ROLE_KEY] = preprocess_result.brain_mask
        volumes2, brain_mask_arr = load_volume_infos(paths_for_qc)
        report_post: InputQCReport = run_input_qc(cfg, volumes2, brain_mask=brain_mask_arr)
        _update_clinical_job(job, input_qc_post=report_post.to_dict(), progress=0.55)

        if report_post.verdict is Severity.REFUSE:
            reasons = "; ".join(f.message for f in report_post.refusals())
            _update_clinical_job(
                job,
                state="refused",
                stage="input_qc (post-preprocessing)",
                error=f"Input QC refused after clinical preprocessing: {reasons}",
            )
            return job

        # --- Research preprocessing (existing, unmodified path) --------------
        _update_clinical_job(job, stage="research_preprocessing")
        missing_roles = [role for role in ROLES if role not in preprocess_result.outputs]
        if missing_roles:
            # E1/E3 should already have refused for this -- checked
            # defensively anyway, so a gap in an earlier gate surfaces as a
            # structured "refused", never a KeyError misreported as "failed".
            _update_clinical_job(
                job,
                state="refused",
                stage="research_preprocessing",
                error=(
                    "Clinical preprocessing did not produce all required modalities; "
                    f"missing: {', '.join(missing_roles)}."
                ),
            )
            return job

        case = BratsCase(
            case_id=job.case_id,
            t1=preprocess_result.outputs["t1"],
            t1ce=preprocess_result.outputs["t1ce"],
            t2=preprocess_result.outputs["t2"],
            flair=preprocess_result.outputs["flair"],
            seg=None,
        )
        job_prep_dir = job_dir / "prep"
        preprocess_case(
            case, job_prep_dir, label_convention="brats2021", target_axcodes=("L", "P", "S")
        )
        _update_clinical_job(job, progress=0.6)

        # --- Segmentation (existing, unmodified path; ALWAYS neurovision) ----
        _update_clinical_job(job, stage="segmenting")
        job_cache_dir = job_dir / "cache"
        clinical_settings = clinical_segmentation_settings(job_prep_dir, job_cache_dir)

        if not inference.checkpoint_available(clinical_settings):
            # A missing DEPLOYED checkpoint is a server-configuration
            # problem, not a per-case refusal -- it must surface as "failed",
            # since it says nothing about whether THIS study is safe to
            # segment.
            raise RuntimeError(
                f"no checkpoint at {clinical_settings.checkpoint}; set NVX_CLINICAL_CHECKPOINT "
                "to the deployed neurovision checkpoint's .pt file."
            )

        # Lazy: see the module docstring's "lazy imports" note.
        from .inference import segment_case

        def _forward_progress(stage: str, fraction: float) -> None:
            _update_clinical_job(
                job, stage=f"segmenting: {stage}", progress=min(0.6 + fraction * 0.25, 0.85)
            )

        segment_case(clinical_settings, job.case_id, save_logits=True, progress=_forward_progress)

        # --- Signals + gatekeeper --------------------------------------------
        _update_clinical_job(job, stage="gatekeeper", progress=0.85)

        qc_checkpoint = Path(str(cfg.analysis.qc_validate.checkpoint))
        if not qc_checkpoint.is_absolute():
            qc_checkpoint = REPO_ROOT / qc_checkpoint
        qc_model, qc_cfg = _load_qc_model_and_cfg(qc_checkpoint)

        logits_path = inference.cached_logits_path(clinical_settings, job.case_id)
        logits = np.load(logits_path).astype(np.float32)  # (3, D, H, W)

        image = np.load(job_prep_dir / job.case_id / "image.npy").astype(np.float32)
        image_modality = image[int(cfg.analysis.qc.modality_index)]  # (D, H, W)

        regions = [str(r) for r in cfg.clinical.gatekeeper.regions]

        predicted_dice_map: dict[str, float] | None = None
        if qc_model is not None:
            predicted_dice_map = _live_predicted_dice(
                qc_cfg, qc_model, logits, image_modality, regions
            )

        alpha = float(cfg.clinical.gatekeeper.conformal_alpha)
        fitted = _load_conformal_fitted_thresholds(regions, alpha)
        conformal_band_map: dict[str, float] | None = None
        if fitted:
            missing_fitted = [r for r in regions if r not in fitted]
            if missing_fitted:
                logger.warning(
                    "run_clinical_job: no fitted conformal threshold for region(s) %s; "
                    "conformal_band will only be reported for %s.",
                    missing_fitted,
                    sorted(fitted),
                )
            conformal_band_map = {
                region: _live_conformal_band_width(
                    logits, REGION_NAMES.index(region), fitted[region]
                )
                for region in regions
                if region in fitted
            }

        signals = GateSignals(
            input_qc=report_post,
            predicted_dice=predicted_dice_map,
            conformal_band=conformal_band_map,
            ood_score=None,
        )
        decision = run_gatekeeper(cfg, signals)
        _update_clinical_job(job, gatekeeper_decision=decision.to_dict())

        if decision.decision is Decision.REFUSE:
            reasons = "; ".join(f"{v.signal}: {v.message}" for v in decision.refusals())
            _update_clinical_job(
                job, state="refused", stage="gatekeeper", error=f"Gatekeeper refused: {reasons}"
            )
            return job

        # --- Explainability (supplementary; failures here must never fail the job) --
        # Only reached once the gatekeeper has already said PROCEED/CAUTION -- a
        # refused segmentation is never explained (see this module's docstring: it
        # has already been rejected). Grad-CAM is a supplementary feature, not core
        # to producing a valid segmentation, so each region gets its OWN try/except:
        # a WT failure must not also skip TC, and neither must ever turn an
        # otherwise-good job into "failed". `inference.explain_case` already handles
        # the normal, expected "no predicted foreground for this region" case
        # gracefully (see `center_patch_on_mask`'s own empty-mask fallback), so a
        # raise here means something more unusual happened -- worth logging loudly,
        # not worth losing the whole job over.
        _update_clinical_job(job, stage="explaining", progress=0.9)
        for region in ("WT", "TC"):
            try:
                inference.explain_case(clinical_settings, job.case_id, region)
            except Exception:  # noqa: BLE001 - explainability must never fail the job
                logger.error(
                    "run_clinical_job: Grad-CAM failed for job %s, region %s",
                    job_id,
                    region,
                    exc_info=True,
                )

        # --- DICOM-SEG export (E6; supplementary, must never fail the job) --
        # `_validate_dicom_seg_cfg` checks the STATIC, config-derived parts of
        # cfg.clinical.dicom_seg (segmentation_type, series_description
        # length) ONCE, here -- deliberately OUTSIDE `_export_dicom_seg`'s own
        # narrow `except ValueError` around write_dicom_seg's call, so a
        # config bug (wrong for EVERY job, forever) cannot blend into that
        # per-job geometry-refusal WARNING. `_export_dicom_seg` itself already
        # absorbs write_dicom_seg's own named refusal and a couple of
        # "nothing to export" edge cases internally (returning None); this
        # try/except is the second, generic layer that catches anything else
        # in the chain -- a config bug raised by the validation call just
        # below, a missing E2 transformations directory, a bad resample, ...
        # -- same failure-isolation pattern as the Grad-CAM loop just above,
        # applied to a single call instead of a per-region loop. Either way
        # the job still reaches "done": a config bug is made DISTINGUISHABLE
        # (this ERROR log with a traceback, vs. the routine WARNING
        # `_export_dicom_seg` logs for a genuine per-case refusal), not
        # FATAL -- this remains a supplementary artifact.
        _update_clinical_job(job, stage="exporting_dicom_seg", progress=0.95)
        try:
            _validate_dicom_seg_cfg(cfg)
            dicom_seg_path = _export_dicom_seg(
                job, preprocess_result, ingest_result, clinical_settings, job_dir, cfg
            )
            if dicom_seg_path is None:
                logger.info(
                    "run_clinical_job: no DICOM-SEG object produced for job %s (see the "
                    "preceding log line for why).",
                    job_id,
                )
        except Exception:  # noqa: BLE001 - DICOM-SEG export must never fail the job
            logger.error(
                "run_clinical_job: DICOM-SEG export failed unexpectedly for job %s",
                job_id,
                exc_info=True,
            )

        # --- Structured anatomical report (Phase 4; supplementary, must
        # never fail the job) -- same failure-isolation philosophy as
        # Grad-CAM and DICOM-SEG export just above. Unlike DICOM-SEG's own
        # export function, `_generate_report` absorbs nothing internally
        # (see its own docstring for why); this try/except is the ONLY
        # layer that keeps a report-generation failure from turning an
        # otherwise-good job into "failed".
        _update_clinical_job(job, stage="generating_report", progress=0.98)
        try:
            report_path = _generate_report(job, clinical_settings, job_dir, cfg)
            logger.info(
                "run_clinical_job: wrote structured report for job %s to %s",
                job_id,
                report_path,
            )
        except Exception:  # noqa: BLE001 - report generation must never fail the job
            logger.error(
                "run_clinical_job: report generation failed unexpectedly for job %s",
                job_id,
                exc_info=True,
            )

        _update_clinical_job(job, state="done", stage="done", progress=1.0)
        return job
    except Exception as exc:  # noqa: BLE001 - a failed job must stay reportable, never crash
        logger.error("Clinical job %s failed at stage %r", job_id, job.stage, exc_info=True)
        _update_clinical_job(job, state="failed", error=str(exc))
        return job
