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

E6 (DICOM-SEG export) is deliberately NOT wired in here. After E2 the
predicted mask lives in atlas (SRI24) space, not in the geometry of the
source DICOM series it came from, and `neurovision.reporting.dicom_seg`'s
writer REFUSES whenever a mask's geometry does not match its reference
series (see `configs/clinical/default.yaml`'s `dicom_seg:` block comment) --
which, without a "resample the mask back through E2's saved inverse
transform" step that has not been built yet, is every real case. Wiring E6
in today would mean it refuses every time; that follow-up is a separate,
not-yet-scoped task, so no function in this module ever imports
`neurovision.reporting.dicom_seg`.

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
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from scipy.special import expit

from neurovision.analysis.qc_inference import CaseArrays, entropy_from_logits, pack_sample
from neurovision.data.brats import BratsCase
from neurovision.data.clinical_preprocess import preprocess_clinical_study
from neurovision.data.dicom_ingest import ROLES, IngestResult, ingest_study
from neurovision.data.preprocessing import preprocess_case
from neurovision.data.transforms import REGION_NAMES
from neurovision.inference.gatekeeper import Decision, GateSignals, run_gatekeeper
from neurovision.inference.input_qc import InputQCReport, Severity, load_volume_infos, run_input_qc
from neurovision.inference.postprocess import postprocess_logits
from neurovision.models.qc import build_segqc
from neurovision.models.qc import predicted_dice as qc_predicted_dice
from neurovision.training.checkpoint import load_checkpoint
from neurovision.utils.device import get_device

from . import inference, jobs
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

        _update_clinical_job(job, state="done", stage="done", progress=1.0)
        return job
    except Exception as exc:  # noqa: BLE001 - a failed job must stay reportable, never crash
        logger.error("Clinical job %s failed at stage %r", job_id, job.stage, exc_info=True)
        _update_clinical_job(job, state="failed", error=str(exc))
        return job
