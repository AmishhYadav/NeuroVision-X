"""FastAPI HTTP layer for the NeuroVision-X demo.

This module only reads what `volumes.py` and `config.py` already know how to
read; it does no numpy math of its own beyond reshaping bytes back into
arrays and averaging over slices for the profile ribbon. In particular it
never imports torch -- the demo serves PRECOMPUTED evaluation artifacts
(`scripts/evaluate.py` output), so a viewer session costs no GPU time and
needs no checkpoint to be present.

Route handlers are thin on purpose: filesystem/geometry errors are translated
to HTTP responses once, in `_register_exception_handlers`, rather than by a
try/except copied into every route.

The `/clinical/*` routes below are the one exception to "never imports
torch": they import `app.backend.clinical_jobs`, which imports `torch`
eagerly at its own module scope (it needs it for `postprocess_logits` and
the QC model). So importing `clinical_jobs` here means this whole API module
now requires torch to be importable, same as every other module in this
backend that already does (CPU-only, per CLAUDE.md constraint 3 -- never
GPU-only).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import re
import zipfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import clinical_jobs, inference, jobs
from .config import REPO_ROOT, Settings, get_settings
from .volumes import (
    MODALITIES,
    REGION_NAMES,
    CaseMeta,
    case_metrics,
    list_cases,
    load_clinical_uncertainty,
    load_mask,
    load_modality,
    load_uncertainty,
    read_meta,
    region_voxel_counts,
)

logger = logging.getLogger(__name__)

# The two Vite dev-server origins are always allowed; NVX_CORS_ORIGINS adds
# more (comma-separated) for e.g. a deployed frontend on its own domain.
_DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

_MASK_SOURCES = ("prediction", "label")


def _round4(value: float | None) -> float | None:
    """Rounds a metric to 4 decimals, passing `None` through unchanged."""
    return None if value is None else round(value, 4)


def _binary_response(data: bytes, shape: tuple[int, int, int]) -> Response:
    """Wraps raw volume bytes with the headers the frontend needs to reshape them.

    The body carries no dtype or shape information of its own -- it is a flat
    `(D, H, W)` uint8 buffer -- so both travel as headers on every binary
    response rather than being re-derived (or guessed) on the client.
    """
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "X-Volume-Shape": ",".join(str(s) for s in shape),
            "X-Volume-Dtype": "uint8",
            "Cache-Control": "private, max-age=3600",
        },
    )


router = APIRouter(prefix="/api")


@lru_cache(maxsize=1)
def _atlas_bundle() -> tuple[Any, np.ndarray, list[dict[str, Any]]]:
    """Loads and caches the SRI24 atlas, its structure-index volume, and its structure table.

    Composes the SAME CPU-only clinical config `_generate_report` uses
    (`clinical_jobs._compose_clinical_cfg`) and loads the atlas and knowledge
    base the same way it does (`load_atlas(cfg.anatomy)`,
    `load_knowledge(cfg.analysis.localize.eloquence_map,
    cfg.analysis.localize.lobe_map, atlas)`) -- so the structures this route
    serves to the twin can never silently drift from the ones a clinical
    report names.

    `lru_cache(maxsize=1)`: `structure_index_volume` collapses the full
    240x240x155 raw parcellation into one `uint8` volume (~9 MB) exactly
    once per process, rather than on every request -- it is pure, read-only
    array/table arithmetic derived entirely from the atlas on disk, so
    caching it costs nothing a request could observe.

    Imports of `neurovision.anatomy.*` are local to this function rather
    than at module scope, matching this file's own convention (see the
    module docstring): unlike `clinical_jobs`, which imports torch eagerly,
    `api.py` itself imports nothing from the deep-learning stack at import
    time.

    Returns:
        `(atlas, index_volume, table)` -- the loaded `Atlas`, its
        `(D, H, W)` `uint8` structure-index volume in ORIGINAL (uncropped)
        geometry, and its `structure_table` rows.
    """
    from neurovision.anatomy.atlas import load_atlas
    from neurovision.anatomy.atlas_export import structure_index_volume, structure_table
    from neurovision.anatomy.localize import load_knowledge

    cfg = clinical_jobs._compose_clinical_cfg()
    atlas = load_atlas(cfg.anatomy)
    knowledge = load_knowledge(
        cfg.analysis.localize.eloquence_map, cfg.analysis.localize.lobe_map, atlas
    )
    index_volume = structure_index_volume(atlas)
    table = structure_table(atlas, knowledge)
    return atlas, index_volume, table


@router.get("/atlas/structures")
def get_atlas_structures() -> dict[str, Any]:
    """Returns the atlas's structure table -- name, laterality, lobe, eloquence, per index.

    The `"index"` field of each row is the 1-based value that structure
    carries in the `uint8` volume served by `/clinical/jobs/{job_id}/atlas`
    and `/cases/{case_id}/atlas`, so the twin can label a shell it picks off
    that volume without a second lookup call.
    """
    atlas, _, table = _atlas_bundle()
    return {
        "atlas": atlas.name,
        "version": atlas.version,
        "n_structures": len(table),
        "structures": table,
    }


def _atlas_volume_response(meta: CaseMeta) -> Response:
    """Crops the cached atlas structure-index volume to one case's own bbox and wraps it.

    The cached volume from `_atlas_bundle` is in ORIGINAL (uncropped)
    geometry -- the same frame `atlas.parcellation` and a saved
    `scripts/evaluate.py` prediction are in -- while every volume this demo
    and clinical pipeline actually SERVES (`/volume`, `/mask`,
    `/uncertainty`, ...) is that case's own bounding-box crop, per its own
    `meta.json`. Cropping the atlas with a DIFFERENT bbox than the one that
    produced the served volumes would not fail loudly: it would silently
    shift every structure by the crop offset and still produce a plausible,
    entirely wrong picture -- the same trap `atlas_for_case` guards against
    in the batch report path, applied here to the live viewer.

    Args:
        meta: The case's own `CaseMeta`, carrying its `bbox` and `shape`.

    Returns:
        A `_binary_response` wrapping the cropped `uint8` volume, with
        `X-Uncertainty-Kind: atlas-structure-index`.

    Raises:
        ValueError: The crop produced a shape other than `meta.shape` --
            `meta.bbox` does not match the atlas volume's own geometry.
    """
    _, index_volume, _ = _atlas_bundle()
    (d0, d1), (h0, h1), (w0, w1) = meta.bbox
    cropped = index_volume[d0:d1, h0:h1, w0:w1]
    if cropped.shape != meta.shape:
        raise ValueError(
            f"atlas crop produced {cropped.shape}, expected {meta.shape} -- meta.bbox does not "
            "match the atlas structure-index volume's own geometry"
        )
    data = np.ascontiguousarray(cropped, dtype=np.uint8).tobytes()
    response = _binary_response(data, meta.shape)
    response.headers["X-Uncertainty-Kind"] = "atlas-structure-index"
    return response


def _report_json_path(case_id: str, settings: Settings) -> Path:
    """`<report_dir>/<case_id>.json` -- the one file every report route reads first."""
    return settings.report_dir / f"{case_id}.json"


def _has_report(case_id: str, settings: Settings) -> bool:
    """Cheap existence check, used by `/cases`, `/cases/{case_id}` and `/health`."""
    return _report_json_path(case_id, settings).exists()


def _has_any_reports(settings: Settings) -> bool:
    """Whether `report_dir` exists and holds at least one `*.json` report."""
    return settings.report_dir.exists() and any(settings.report_dir.glob("*.json"))


def _expected_segmentation_dir(settings: Settings, segmentation_source: str | None) -> Path:
    """Which directory a `"prediction"` report's provenance must name.

    Always `predictions_dir` -- the directory the demo's mask overlay is
    actually read from. A `"label"` report never reaches this function; see
    `_reject_ground_truth_report` for why it is refused outright rather than
    checked against `prep_dir`.
    """
    return settings.predictions_dir.resolve()


def _reject_ground_truth_report(path: Path, settings: Settings) -> None:
    """Refuses a ground-truth-derived report, because the viewer displays a prediction.

    Checking a `"label"` report against `prep_dir` proves only that the
    report is internally consistent with where its mask came from -- and a
    ground-truth report IS consistent with `prep_dir`, so that check passes
    and the report is served. But every overlay this demo draws comes from
    `predictions_dir`, so the panel would then describe the ground-truth mask
    while the picture shows the model's. The structure list, the eloquent
    involvement and every burden number would be right about a mask that is
    not on screen, and nothing would fail.

    That is the same failure the provenance guard exists to prevent, one
    level up: it is not enough for a report to be self-consistent, it has to
    describe the segmentation being displayed. Comparing ground truth against
    a prediction is a real and wanted capability -- it is the whole of Phase
    5 -- but it is a deliberate two-column feature, not something an
    `NVX_REPORT_DIR` typo should turn on silently.
    """
    raise ValueError(
        f"report {path} was generated from ground-truth labels "
        "(provenance.segmentation_source='label'), but this viewer displays the prediction "
        f"from {settings.predictions_dir.resolve()}. Serving it would describe a mask that is "
        "not the one on screen. Point NVX_REPORT_DIR at a report directory generated from that "
        "same prediction directory."
    )


def _load_verified_report(case_id: str, settings: Settings) -> dict[str, Any]:
    """Reads one case's report JSON and checks it describes the segmentation being served.

    See the module docstring and CLAUDE.md's "Three eval directories differ
    only by suffix" note: `outputs/report_gt`, `outputs/report_baseline` and
    `outputs/report_neurovision` are sibling directories that each hold a
    complete, plausible report for every case, so a misconfigured
    `NVX_REPORT_DIR` would silently show a reader a structure list the
    picture on screen does not support.

    Raises:
        FileNotFoundError: No report file for this case -> 404 via the
            registered handler.
        json.JSONDecodeError: The file is not valid JSON -> 500.
        ValueError: The report's provenance names a different segmentation
            directory than the one this server displays -> 500.
    """
    path = _report_json_path(case_id, settings)
    if not path.exists():
        raise FileNotFoundError(
            f"no report at {path}; scripts/report.py has not been run for case {case_id!r}"
        )
    try:
        report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # Re-raised with the path folded into the message -- json.loads has
        # no way to know which file it was given, and a 500 with no path is
        # not enough to find the broken file later.
        raise json.JSONDecodeError(f"{path}: {exc.msg}", exc.doc, exc.pos) from exc

    provenance = report.get("provenance", {})
    source = provenance.get("segmentation_source")
    if source == "label":
        _reject_ground_truth_report(path, settings)

    raw_dir = provenance.get("segmentation_dir")
    report_seg_dir = Path(str(raw_dir)).resolve() if raw_dir else None
    expected = _expected_segmentation_dir(settings, source)

    if report_seg_dir != expected:
        raise ValueError(
            f"report {path} describes segmentation_source={source!r}, "
            f"segmentation_dir={report_seg_dir}, but this server is configured to display "
            f"segmentation from {expected}. This report describes a different segmentation "
            "than the masks currently being displayed."
        )
    return report


@router.get("/health")
def get_health() -> dict[str, Any]:
    """Reports what the server can currently see, without ever raising.

    This is the endpoint the frontend polls to explain a broken setup to the
    user (missing eval dir, no checkpoint, ...), so it must degrade to a
    plausible-looking response rather than a 500 when a path is absent.
    """
    settings = get_settings()
    try:
        case_count = len(list_cases(settings))
    except Exception:  # noqa: BLE001 - health must never raise, whatever breaks
        logger.exception("case_count lookup failed during health check")
        case_count = 0
    return {
        "status": "ok",
        "experiment": settings.experiment,
        "eval_dir": str(settings.eval_dir),
        "prep_dir": str(settings.prep_dir),
        "checkpoint_present": settings.checkpoint.exists(),
        "case_count": case_count,
        "has_metrics": settings.metrics_csv.exists(),
        "report_dir": str(settings.report_dir),
        "has_reports": _has_any_reports(settings),
    }


@router.get("/cases")
def get_cases() -> dict[str, Any]:
    """Lists cases in `list_cases()`'s own order (ranked by descending Dice)."""
    settings = get_settings()
    cases: list[dict[str, Any]] = []
    for case_id in list_cases(settings):
        try:
            meta = read_meta(case_id, settings)
        except Exception as exc:  # noqa: BLE001 - one bad case must not break the list
            logger.warning("skipping case %s in list: %s", case_id, exc)
            continue
        metrics = case_metrics(case_id)
        if metrics is None:
            dice_mean, dice = None, None
        else:
            dice_mean = _round4(metrics["dice_mean"])
            dice = {region: _round4(v) for region, v in metrics["dice"].items()}
        cases.append(
            {
                "case_id": case_id,
                "dice_mean": dice_mean,
                "dice": dice,
                "has_label": meta.has_label,
                "has_logits": meta.has_logits,
                "has_report": _has_report(case_id, settings),
            }
        )
    return {"cases": cases}


@router.get("/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    """Returns geometry, reported metrics and per-region volumes for one case."""
    settings = get_settings()
    meta = read_meta(case_id, settings)  # FileNotFoundError -> 404, via the handler below
    metrics = case_metrics(case_id)

    pred = np.frombuffer(load_mask(case_id, "prediction", settings), dtype=np.uint8).reshape(
        meta.shape
    )
    regions: dict[str, Any] = {
        "prediction": region_voxel_counts(pred, meta.spacing),
        "label": None,
    }
    if meta.has_label:
        label = np.frombuffer(load_mask(case_id, "label", settings), dtype=np.uint8).reshape(
            meta.shape
        )
        regions["label"] = region_voxel_counts(label, meta.spacing)

    return {
        "meta": meta.to_json(),
        "metrics": metrics,
        "regions": regions,
        "has_report": _has_report(case_id, settings),
    }


@router.get("/cases/{case_id}/volume/{modality}")
def get_case_volume(case_id: str, modality: str) -> Response:
    """Returns one MRI modality as a raw uint8 `(D, H, W)` buffer."""
    if modality not in MODALITIES:
        raise HTTPException(
            status_code=404,
            detail=f"unknown modality {modality!r}; expected one of {list(MODALITIES)}",
        )
    settings = get_settings()
    meta = read_meta(case_id, settings)
    data = load_modality(case_id, modality, settings)
    return _binary_response(data, meta.shape)


@router.get("/cases/{case_id}/mask/{source}")
def get_case_mask(case_id: str, source: str) -> Response:
    """Returns a `{0,1,2,3}` class map (prediction or ground truth) as raw uint8 bytes."""
    if source not in _MASK_SOURCES:
        raise HTTPException(
            status_code=404,
            detail=f"unknown mask source {source!r}; expected one of {list(_MASK_SOURCES)}",
        )
    settings = get_settings()
    meta = read_meta(case_id, settings)
    if source == "label" and not meta.has_label:
        raise HTTPException(status_code=404, detail=f"case {case_id!r} has no label.npy")
    if source == "prediction" and not meta.has_prediction:
        raise HTTPException(status_code=404, detail=f"case {case_id!r} has no saved prediction")
    data = load_mask(case_id, source, settings)  # type: ignore[arg-type]
    return _binary_response(data, meta.shape)


@router.get("/cases/{case_id}/uncertainty")
def get_case_uncertainty(case_id: str) -> Response:
    """Returns per-voxel predictive entropy as raw uint8 bytes, scaled to [0, 255]."""
    settings = get_settings()
    meta = read_meta(case_id, settings)
    try:
        data = load_uncertainty(case_id, settings)
    except FileNotFoundError:
        # Spelled out here rather than left to the generic handler: the
        # frontend needs the exact remediation, and load_uncertainty's own
        # message is written for a log line, not a user-facing error.
        raise HTTPException(
            status_code=404,
            detail=(
                f"no saved logits for {case_id}; re-run scripts/evaluate.py with "
                "inference.evaluation.save_logits=true"
            ),
        ) from None
    response = _binary_response(data, meta.shape)
    # This is a single deterministic pass's entropy, NOT an MC-dropout
    # epistemic estimate. The frontend reads this header to label the layer
    # correctly rather than assuming -- see the CLAUDE.md note on the two
    # uncertainty sources never being interchanged.
    response.headers["X-Uncertainty-Kind"] = "predictive-entropy-single-pass"
    return response


@router.get("/cases/{case_id}/atlas")
def get_case_atlas(case_id: str) -> Response:
    """Returns this demo case's atlas structure-index volume, cropped to its own bbox.

    Same 404-on-unknown-case behaviour as `/cases/{case_id}/volume/{modality}`
    (`read_meta` raises `FileNotFoundError`, handled globally): there is
    nothing case-specific about the atlas itself, only about the crop that
    aligns it to this case's own geometry -- see `_atlas_volume_response`.
    """
    settings = get_settings()
    meta = read_meta(case_id, settings)
    return _atlas_volume_response(meta)


@lru_cache(maxsize=32)
def _compute_profile(case_id: str) -> dict[str, Any]:
    """Builds the per-slice tumour/error/entropy ribbon for one case.

    Cached because it reads the ~20 MB logits file, and the frontend requests
    this once per case switch. Cleared implicitly whenever the process
    restarts; tests that reuse a process across cases must clear it by hand
    (`_compute_profile.cache_clear()`), same as `volumes._metrics_table`.
    """
    settings = get_settings()
    meta = read_meta(case_id, settings)

    pred = np.frombuffer(load_mask(case_id, "prediction", settings), dtype=np.uint8).reshape(
        meta.shape
    )
    tumor = pred > 0

    error = None
    if meta.has_label:
        label = np.frombuffer(load_mask(case_id, "label", settings), dtype=np.uint8).reshape(
            meta.shape
        )
        error = tumor != (label > 0)

    entropy = None
    if meta.has_logits:
        # Read straight from the same bytes `/uncertainty` serves, rather
        # than recomputing from logits, so the ribbon and the overlay can
        # never disagree about what "entropy" means for this case.
        raw = np.frombuffer(load_uncertainty(case_id, settings), dtype=np.uint8).reshape(meta.shape)
        entropy = raw.astype(np.float32) / 255.0

    planes: dict[str, Any] = {}
    for name, axis, reduce_axes in (
        ("sagittal", 0, (1, 2)),
        ("coronal", 1, (0, 2)),
        ("axial", 2, (0, 1)),
    ):
        planes[name] = {
            "n": meta.shape[axis],
            "tumor": np.round(tumor.mean(axis=reduce_axes), 5).tolist(),
            "error": None if error is None else np.round(error.mean(axis=reduce_axes), 5).tolist(),
            "entropy": (
                None if entropy is None else np.round(entropy.mean(axis=reduce_axes), 5).tolist()
            ),
        }
    return {"case_id": case_id, "planes": planes}


@router.get("/cases/{case_id}/profile")
def get_case_profile(case_id: str) -> dict[str, Any]:
    """Returns per-slice tumour fraction, disagreement rate and mean entropy."""
    return _compute_profile(case_id)


@router.get("/report/{case_id}")
def get_report(case_id: str) -> dict[str, Any]:
    """Returns one case's Phase 4 structured report, unchanged, as JSON.

    The report is read from `<settings.report_dir>/<case_id>.json` (written
    by `scripts/report.py`) and returned exactly as written -- the
    disclaimer, `not_claimed` block and provenance are required fields of
    the artifact, not something this route is allowed to drop. See
    `_load_verified_report` for the provenance guard that must pass first.
    """
    settings = get_settings()
    return _load_verified_report(case_id, settings)


@router.get("/report/{case_id}/markdown")
def get_report_markdown(case_id: str) -> Response:
    """Returns the rendered `<case_id>.md` report, after the same provenance guard as JSON.

    The sibling JSON is read and verified FIRST -- a markdown file with no
    JSON next to it (or one that fails the provenance guard) is unverifiable
    text and is refused rather than served.
    """
    settings = get_settings()
    _load_verified_report(case_id, settings)  # raises 404/500 exactly as the JSON route does

    md_path = settings.report_dir / f"{case_id}.md"
    if not md_path.exists():
        raise FileNotFoundError(
            f"no markdown report at {md_path}; scripts/report.py has not been run with "
            f"markdown=true for case {case_id!r}"
        )
    return Response(
        content=md_path.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


def _clinical_job_to_json(job: clinical_jobs.ClinicalJob) -> dict[str, Any]:
    """Serialises one `ClinicalJob` dataclass to a plain JSON-able dict.

    Uses `dataclasses.asdict` -- every field (`ingest_result`, `input_qc_pre`,
    `input_qc_post`, `gatekeeper_decision`, ...) is already a plain,
    JSON-serialisable dict or `None`, so no further translation is needed
    here.
    """
    return dataclasses.asdict(job)


@router.post("/clinical/upload", status_code=202)
async def post_clinical_upload(dicom_zip: UploadFile = File(...)) -> dict[str, Any]:
    """Accepts a raw DICOM study `.zip`, validates it, and queues a clinical job.

    A `ValueError` from `clinical_jobs.create_clinical_job` (empty upload,
    oversized upload, not a valid zip, or a zip-slip-attempting member)
    surfaces as 400 with the underlying message -- unlike the generic 500
    `_register_exception_handlers` gives every other `ValueError` in this
    file, because a bad upload is the caller's fault, not a server-side data
    inconsistency. Same reasoning `post_upload` already applies to a bad
    NIfTI upload.
    """
    settings = get_settings()
    data = await dicom_zip.read()
    try:
        job = clinical_jobs.create_clinical_job(settings, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    clinical_jobs.start_clinical_job(settings, job.job_id)
    return _clinical_job_to_json(job)


@router.get("/clinical/jobs")
def get_clinical_jobs() -> dict[str, Any]:
    """Lists every known clinical job, newest first."""
    return {"jobs": [_clinical_job_to_json(job) for job in clinical_jobs.list_clinical_jobs()]}


@router.get("/clinical/jobs/{job_id}")
def get_clinical_job_detail(job_id: str) -> dict[str, Any]:
    """Returns one clinical job's full state, or 404 if `job_id` is unknown.

    Carries `ingest_result`, `input_qc_pre`, `input_qc_post`,
    `gatekeeper_decision`, `state` and `error` exactly as
    `clinical_jobs.ClinicalJob` stores them -- everything a refusal-banner UI
    needs to explain a `"refused"` outcome, or a caution/success UI needs for
    a `"done"` one. A `"refused"` job is served the same way as any other
    (200, full payload): see `clinical_jobs.py`'s module docstring for why
    refusal is a distinct, successful terminal state, never an error.
    """
    job = clinical_jobs.get_clinical_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown clinical job {job_id!r}")
    return _clinical_job_to_json(job)


@router.delete("/clinical/jobs/{job_id}")
def delete_clinical_job_route(job_id: str) -> dict[str, Any]:
    """Deletes a clinical job and everything it wrote to disk, or 404 if it did not exist."""
    settings = get_settings()
    if not clinical_jobs.delete_clinical_job(settings, job_id):
        raise HTTPException(status_code=404, detail=f"unknown clinical job {job_id!r}")
    return {"job_id": job_id, "deleted": True}


def _require_done_clinical_job(job_id: str) -> clinical_jobs.ClinicalJob:
    """Returns a known clinical job in state `"done"`, or raises the matching HTTP error.

    A `"refused"` job is a legitimate terminal state (see
    `clinical_jobs.py`'s module docstring), not an error -- so the 409
    message below just states the actual state, worded neutrally, rather
    than implying refusal is a bug.
    """
    job = clinical_jobs.get_clinical_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown clinical job {job_id!r}")
    if job.state != "done":
        raise HTTPException(
            status_code=409,
            detail=f"clinical job {job_id!r} is not done yet (state={job.state!r})",
        )
    return job


def _clinical_job_settings(settings: Settings, job_id: str) -> Settings:
    """`Settings` pointing at a clinical job's own prep/cache directories AND its pinned model.

    A clinical job's segmentation always ran against the pinned
    `neurovision` experiment/checkpoint, never whatever the generic backend
    `Settings` happens to name -- see `clinical_jobs.py`'s module docstring,
    "Which model runs a clinical job is fixed" section. So this goes through
    `clinical_jobs.clinical_segmentation_settings`, the same function
    `run_clinical_job` itself uses to build the `Settings` it segments with.
    """
    job_prep_dir = jobs.job_root(settings) / job_id / "prep"
    job_cache_dir = jobs.job_root(settings) / job_id / "cache"
    return clinical_jobs.clinical_segmentation_settings(job_prep_dir, job_cache_dir)


@router.get("/clinical/jobs/{job_id}/volume/{modality}")
def get_clinical_job_volume(job_id: str, modality: str) -> Response:
    """Returns one clinical job's preprocessed modality volume, exactly like `/cases/.../volume`."""
    if modality not in MODALITIES:
        raise HTTPException(
            status_code=404,
            detail=f"unknown modality {modality!r}; expected one of {list(MODALITIES)}",
        )
    job = _require_done_clinical_job(job_id)
    job_settings = _clinical_job_settings(get_settings(), job_id)
    meta = read_meta(job.case_id, job_settings)
    data = load_modality(job.case_id, modality, job_settings)
    return _binary_response(data, meta.shape)


@router.get("/clinical/jobs/{job_id}/mask/prediction")
def get_clinical_job_mask(job_id: str) -> Response:
    """Returns the clinical job's segmentation as a `{0,1,2,3}` class map, raw uint8 bytes.

    Does NOT go through `volumes.load_mask`: `inference.segment_case` writes
    its prediction in the job's own CROPPED geometry, not the original
    DICOM/BraTS geometry `load_mask` re-crops with `meta.bbox`. The
    prediction is cached in the job's own cropped geometry already, so it is
    re-encoded directly here instead.
    """
    job = _require_done_clinical_job(job_id)
    job_settings = _clinical_job_settings(get_settings(), job_id)
    meta = read_meta(job.case_id, job_settings)
    pred_path = inference.cached_prediction_path(job_settings, job.case_id)
    if not pred_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"clinical job {job_id!r} is done but has no cached segmentation at "
            f"{pred_path}",
        )
    arr = np.load(pred_path, mmap_mode="r")
    data = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    return _binary_response(data, meta.shape)


@router.get("/clinical/jobs/{job_id}/geometry")
def get_clinical_job_geometry(job_id: str) -> dict[str, Any]:
    """Returns the clinical job's geometry -- `shape`, `spacing`, `bbox` and `planes`.

    This exists because the 3D twin (the mesh/volume-in-mL view) needs
    `shape`, `spacing` and `bbox` to build its geometry and convert voxel
    counts to millilitres, and no clinical binary route carries that:
    `_binary_response` (used by `/volume`, `/mask`, `/uncertainty`,
    `/conformal-band`, `/gradcam`) only ever puts `shape` on the wire, via
    the `X-Volume-Shape` header, and never `spacing` or `bbox` at all. This
    is the one clinical route that serves geometry, and T3/T6 in
    `docs/research/tool_completion_plan.md` both reuse it rather than each
    growing their own copy of this lookup.

    The response is exactly `CaseMeta.to_json()` -- the same dict shape
    `GET /api/cases/{case_id}` nests under its own `"meta"` key -- returned
    here at the top level instead, so a client that already knows how to
    read `cases/{id}`'s `meta` block can read this response unchanged. Built
    from the job's OWN cached case (via `_clinical_job_settings`, same as
    every other `/clinical/jobs/{job_id}/...` route), so `case_id` in the
    body is the job id. `has_label` is always `False` (a clinical job never
    has a ground-truth label). `has_prediction` and `has_logits` are always
    `False` too, even for a job that DOES have a cached prediction/logits --
    `CaseMeta` checks `Settings.predictions_dir` / `logits_dir`
    (`eval_dir/"predictions"`, `eval_dir/"logits"`), and
    `clinical_segmentation_settings` points `eval_dir` at a harmless,
    unused placeholder (see that function's docstring), never at where
    `inference.cached_prediction_path` / `cached_logits_path` actually write.
    That is honest, not a bug this route should paper over: use
    `/mask/prediction` and `/uncertainty` returning 200 (rather than this
    flag) to tell whether this job has a cached prediction or logits.
    """
    job = _require_done_clinical_job(job_id)
    job_settings = _clinical_job_settings(get_settings(), job_id)
    try:
        meta = read_meta(job.case_id, job_settings)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"clinical job {job_id!r} is done but has no cached meta.json ({exc})",
        ) from None
    return meta.to_json()


@router.get("/clinical/jobs/{job_id}/atlas")
def get_clinical_job_atlas(job_id: str) -> Response:
    """Returns this clinical job's atlas structure-index volume, cropped to its own bbox.

    Reads `meta.bbox`/`meta.shape` from the job's own cached `meta.json`
    (`_clinical_job_settings`, same as `/geometry`) and crops the cached
    atlas volume with it -- see `_atlas_volume_response`'s docstring for why
    that crop must use THIS job's own bbox: after E2 the study is
    co-registered onto the SRI24 grid and the volumes this route's siblings
    serve are that grid cropped by the job's bbox, so the atlas has to be
    cropped identically or every structure shifts by the crop offset, a
    plausible, wrong picture.
    """
    job = _require_done_clinical_job(job_id)
    job_settings = _clinical_job_settings(get_settings(), job_id)
    meta = read_meta(job.case_id, job_settings)
    return _atlas_volume_response(meta)


@router.get("/clinical/jobs/{job_id}/uncertainty")
def get_clinical_job_uncertainty(job_id: str) -> Response:
    """Returns the clinical job's per-voxel predictive entropy, like `/cases/.../uncertainty`.

    Reads from the job's OWN namespaced logits cache
    (`inference.cached_logits_path`, via `volumes.load_clinical_uncertainty`)
    rather than the demo path's `settings.logits_dir` -- see that function's
    docstring for why the two must not be conflated. A missing logits file
    raises `FileNotFoundError`, which propagates to the global handler
    registered in `_register_exception_handlers` (404), same as every other
    "nothing to serve" case in this module.
    """
    job = _require_done_clinical_job(job_id)
    job_settings = _clinical_job_settings(get_settings(), job_id)
    meta = read_meta(job.case_id, job_settings)
    data = load_clinical_uncertainty(job.case_id, job_settings)
    response = _binary_response(data, meta.shape)
    # Same label the demo's /cases/.../uncertainty route uses -- a single
    # deterministic pass's entropy, never an MC-dropout epistemic estimate.
    # The frontend's PREDICTIVE_ENTROPY_SINGLE_PASS constant matches this
    # exact string.
    response.headers["X-Uncertainty-Kind"] = "predictive-entropy-single-pass"
    return response


@router.get("/clinical/jobs/{job_id}/conformal-band/{region}")
def get_clinical_job_conformal_band(job_id: str, region: str) -> Response:
    """Returns the clinical job's per-voxel conformal band for one region.

    `region` must be one of `cfg.clinical.gatekeeper.regions` (currently
    `[WT, TC]`) -- an unrecognised region, a region with no cached logits, or
    a region with no fitted conformal threshold are all 404s, since each
    means there is nothing to serve for THIS job, not a server-side fault.
    A `ValueError` from `clinical_jobs.clinical_conformal_band_mask` (the
    "not thresholds of the same array" invariant -- see that function's
    docstring) propagates to the global `ValueError` handler as a 500: a
    real data inconsistency this route cannot resolve by re-asking. Unlike
    before F5 (2026-09-18), a fitted threshold on the RESTRICTIVE side of the
    reference threshold (as the deployed `conformal_alpha=0.10` fit actually
    is for both WT and TC -- see `outputs/conformal/neurovision/fit.json`) is
    no longer treated as that invariant violation; it is a normal, expected
    side, distinguished for the caller via `X-Conformal-Side`.

    Four extra headers describe the two thresholds the returned band was
    built from, since the byte encoding alone (see
    `clinical_jobs.clinical_conformal_band_mask`'s Returns section) cannot
    say which side of `reference_threshold` `fitted_threshold` fell on:
    `X-Conformal-Threshold` (the fitted threshold, 4 decimal places),
    `X-Conformal-Reference` (always `"0.5"`, this project's standard
    operating point), `X-Conformal-Side` (`"permissive"` or `"restrictive"`,
    from `clinical_jobs.conformal_band_side`), and `X-Conformal-Alpha` (the
    alpha the fitted threshold was calibrated at).
    """
    job = _require_done_clinical_job(job_id)
    cfg = clinical_jobs._compose_clinical_cfg()
    valid_regions = [str(r) for r in cfg.clinical.gatekeeper.regions]
    if region not in valid_regions:
        raise HTTPException(
            status_code=404,
            detail=f"unknown region {region!r}; expected one of {valid_regions}",
        )

    job_settings = _clinical_job_settings(get_settings(), job_id)
    meta = read_meta(job.case_id, job_settings)

    logits_path = inference.cached_logits_path(job_settings, job.case_id)
    if not logits_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"clinical job {job_id!r} is done but has no cached logits at {logits_path}",
        )
    logits = np.load(logits_path).astype(np.float32)

    alpha = float(cfg.clinical.gatekeeper.conformal_alpha)
    # Display route: a missing/infeasible fit must fall through to the 404 branch
    # below, never raise a 500 -- so strict=False.
    fitted = clinical_jobs._load_conformal_fitted_thresholds([region], alpha, strict=False)
    if region not in fitted:
        raise HTTPException(
            status_code=404,
            detail=f"no fitted conformal threshold for region {region!r} (alpha={alpha})",
        )

    fitted_threshold = fitted[region]
    reference_threshold = 0.5
    data = clinical_jobs.clinical_conformal_band_mask(
        logits,
        REGION_NAMES.index(region),
        fitted_threshold,
        reference_threshold=reference_threshold,
    )
    response = _binary_response(np.ascontiguousarray(data, dtype=np.uint8).tobytes(), meta.shape)
    response.headers["X-Uncertainty-Kind"] = "conformal-band"
    response.headers["X-Conformal-Threshold"] = f"{fitted_threshold:.4f}"
    response.headers["X-Conformal-Reference"] = f"{reference_threshold}"
    response.headers["X-Conformal-Side"] = clinical_jobs.conformal_band_side(
        fitted_threshold, reference_threshold
    )
    response.headers["X-Conformal-Alpha"] = f"{alpha}"
    return response


_GRADCAM_REGIONS = ("WT", "TC")


@router.get("/clinical/jobs/{job_id}/gradcam/{region}")
def get_clinical_job_gradcam(job_id: str, region: str) -> Response:
    """Returns the clinical job's Seg-Grad-CAM heatmap for one region.

    `region` must be one of `"WT"`, `"TC"` -- the same two regions
    `inference.explain_case` computes (see that function's docstring for why `"ET"`
    is out of scope). An unrecognised region, or a region with no cached heatmap
    (e.g. that region's Grad-CAM failed during the job's run -- see
    `run_clinical_job`'s failure-isolation handling -- or the job predates this
    feature), is a 404: there is nothing to serve for THIS job, not a server-side
    fault.
    """
    job = _require_done_clinical_job(job_id)
    if region not in _GRADCAM_REGIONS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown region {region!r}; expected one of {list(_GRADCAM_REGIONS)}",
        )

    job_settings = _clinical_job_settings(get_settings(), job_id)
    meta = read_meta(job.case_id, job_settings)

    gradcam_path = inference.cached_gradcam_path(job_settings, job.case_id, region)
    if not gradcam_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"clinical job {job_id!r} has no cached Grad-CAM for region {region!r} at "
                f"{gradcam_path}"
            ),
        )
    data = np.load(gradcam_path)
    response = _binary_response(np.ascontiguousarray(data, dtype=np.uint8).tobytes(), meta.shape)
    response.headers["X-Uncertainty-Kind"] = "gradcam"
    return response


def _clinical_dicom_seg_path(settings: Settings, job: clinical_jobs.ClinicalJob) -> Path:
    """Where `clinical_jobs._export_dicom_seg` would have cached this job's DICOM-SEG object.

    Mirrors that function's own `out_path` convention exactly (see its
    docstring): `<job_dir>/dicom_seg/<case_id>.dcm`, where `job_dir` is
    `jobs.job_root(settings) / job.job_id`. Not itself derived from
    `_export_dicom_seg`, since that function returns a path only on success
    -- this route needs to name the expected path even when nothing was ever
    written there, to report exactly that as a 404.
    """
    return jobs.job_root(settings) / job.job_id / "dicom_seg" / f"{job.case_id}.dcm"


@router.get("/clinical/jobs/{job_id}/dicom-seg")
def get_clinical_job_dicom_seg(job_id: str) -> FileResponse:
    """Returns the clinical job's DICOM-SEG object as a binary `.dcm` download.

    Unlike every other binary route in this module, this is a FILE download --
    a complete, self-describing DICOM object with its own header -- not a flat
    shape-plus-raw-bytes volume buffer, so it does not go through
    `_binary_response`. `fastapi.responses.FileResponse` serves the cached
    file directly, streamed from disk, with `Content-Type: application/dicom`
    (the registered DICOM media type) and a `Content-Disposition` filename so
    a browser download keeps the `.dcm` extension.

    404 if the job is not done yet, or if it has no cached SEG object -- e.g.
    `clinical_jobs._export_dicom_seg` refused or failed for this job (see that
    function's own failure-isolation contract) or the job predates this
    feature; either way, there is nothing to serve for THIS job, not a
    server-side fault.
    """
    job = _require_done_clinical_job(job_id)
    dicom_seg_path = _clinical_dicom_seg_path(get_settings(), job)
    if not dicom_seg_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(f"clinical job {job_id!r} has no cached DICOM-SEG object at {dicom_seg_path}"),
        )
    return FileResponse(
        path=dicom_seg_path,
        media_type="application/dicom",
        filename=dicom_seg_path.name,
    )


def _clinical_report_path(settings: Settings, job: clinical_jobs.ClinicalJob) -> Path:
    """Where `clinical_jobs._generate_report` would have cached this job's structured report.

    Mirrors `_clinical_dicom_seg_path`'s own convention exactly (see its
    docstring): `<job_dir>/report/<case_id>.json`. Not itself derived from
    `_generate_report`, since that function returns a path only on success --
    this route needs to name the expected path even when nothing was ever
    written there, to report exactly that as a 404.
    """
    return jobs.job_root(settings) / job.job_id / "report" / f"{job.case_id}.json"


def _load_merged_clinical_report(job_id: str) -> dict[str, Any]:
    """Loads a clinical job's cached report JSON, merged with any entered pathology (T5.5).

    Mirrors the demo's `/report/{case_id}` route: the cached JSON file is
    read and parsed, then returned as a plain dict, rather than streamed
    back as a raw file -- unlike the DICOM-SEG route, which serves an opaque
    binary `.dcm` via `FileResponse`. Unlike `/report/{case_id}`, this does
    NOT run `_load_verified_report`'s provenance cross-check: that guard
    exists because a demo-viewer `NVX_REPORT_DIR` could point at a report
    generated from a DIFFERENT segmentation than the one `predictions_dir`
    is currently serving. A clinical job's report has no equivalent
    configuration seam -- its path is derived entirely from `job_id`, and
    `clinical_jobs._generate_report` always computes it from THIS job's own
    cached prediction -- so there is nothing to cross-check.

    404 if the job is unknown or not done yet is wrong -- see
    `_require_done_clinical_job`: unknown is 404, not-done is 409, matching
    every other clinical job route. A separate 404 covers a done job with no
    cached report (e.g. report generation failed for this job -- see
    `run_clinical_job`'s failure-isolation handling -- or the job predates
    this feature): either way there is nothing to serve for THIS job, not a
    server-side fault.

    T5.5 read-time merge: if `<job_dir>/pathology.json` exists AND the
    cached report has a `"molecular"` block, that block is replaced with
    `merge_pathology(report["molecular"], entered, knowledge)` before the
    result is returned -- the cached JSON FILE on disk is never rewritten,
    only the dict this call returns. This is why `PUT .../pathology` never
    has to touch the report file: every correction made there is visible
    here on the very next read (JSON or Markdown, both call this same
    helper). A report with no `"molecular"` key at all (a job that predates
    T5.4) is returned untouched -- there is nothing to merge into. A corrupt
    or failed-validation `pathology.json` (see `_load_pathology_or_none`) is
    logged at WARNING and the report is returned unmerged: a bad side file
    must never turn a working report read into a 500.
    """
    job = _require_done_clinical_job(job_id)
    settings = get_settings()
    report_path = _clinical_report_path(settings, job)
    if not report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"clinical job {job_id!r} has no cached report at {report_path}",
        )
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))

    pathology_path = _clinical_pathology_path(settings, job)
    if pathology_path.is_file() and report.get("molecular") is not None:
        from neurovision.reporting.molecular import merge_pathology

        knowledge = _molecular_knowledge()
        entered = _load_pathology_or_none(pathology_path, knowledge)
        if entered is not None:
            report["molecular"] = merge_pathology(report["molecular"], entered, knowledge)
    return report


@router.get("/clinical/jobs/{job_id}/report")
def get_clinical_job_report(job_id: str) -> dict[str, Any]:
    """Returns the clinical job's structured anatomical report (Phase 4), as JSON.

    See `_load_merged_clinical_report` for the lookup, 404/409 rules and the
    T5.5 pathology merge -- this route just returns its result as-is.
    """
    return _load_merged_clinical_report(job_id)


@router.get("/clinical/jobs/{job_id}/report/markdown")
def get_clinical_job_report_markdown(job_id: str) -> Response:
    """Returns the clinical job's report rendered as Markdown, from the same merged dict.

    Rendered on the fly, via `render_markdown`, from the SAME dict
    `_load_merged_clinical_report` gives the JSON route -- so any pathology
    entered through `PUT .../pathology` shows up here immediately too, and,
    like the JSON route, no `.md` file is ever written or cached under the
    job directory: this route's only output is the HTTP response body.
    Error codes (404 unknown job / no cached report, 409 not done) are
    exactly the JSON route's, since they come from the same helper.
    """
    from neurovision.reporting.report import render_markdown

    report = _load_merged_clinical_report(job_id)
    return Response(
        content=render_markdown(report),
        media_type="text/markdown; charset=utf-8",
    )


# --- T6.3: POST /clinical/jobs/{job_id}/export -------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024  # 20 MB -- a clinical bundle is emailed, not streamed.
_UNSAFE_SNAPSHOT_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _sanitise_snapshot_name(filename: str | None, index: int, taken: set[str]) -> str:
    """Turns one uploaded snapshot's filename into a safe, unique `.png` zip entry name.

    `Path(...).name` first, so a path-like filename (e.g. `"../../evil.png"`,
    the zip-slip shape CLAUDE.md's trap list warns about elsewhere in this
    project) contributes no directory component at all -- only its final
    segment is ever considered. The extension is then forced to `.png`
    regardless of what was uploaded (a bundle only ever holds PNGs, by the
    time this function runs the signature check already passed), and every
    character outside `[A-Za-z0-9._-]` in what remains is replaced with `_`.
    If the result collides with a name already placed in this same zip
    (`taken`), `-2`, `-3`, ... is inserted before `.png` until it does not.

    Args:
        filename: The upload's client-supplied filename, or `None`.
        index: This upload's position in the request, used to name it when
            `filename` is empty.
        taken: Names already used in this bundle; mutated to record the
            name this call returns.

    Returns:
        A zip-safe entry name, e.g. `"evil_name.png"`, with no directory
        component and no repeat within `taken`.
    """
    raw_name = Path(filename or f"snapshot-{index}").name
    stem = Path(raw_name).stem or f"snapshot-{index}"
    safe_stem = _UNSAFE_SNAPSHOT_CHARS.sub("_", stem)
    candidate = f"{safe_stem}.png"
    suffix = 2
    while candidate in taken:
        candidate = f"{safe_stem}-{suffix}.png"
        suffix += 1
    taken.add(candidate)
    return candidate


def _manifest_line(name: str, content: bytes) -> str:
    """One `MANIFEST.txt` line: `<name>  <size bytes>  sha256=<hex>`, for one bundled entry."""
    return f"{name}  {len(content)} bytes  sha256={hashlib.sha256(content).hexdigest()}"


@router.post("/clinical/jobs/{job_id}/export")
async def export_clinical_job(
    job_id: str, snapshots: list[UploadFile] = File(default=[])
) -> Response:
    """Bundles a clinical job's report, DICOM-SEG object and viewer snapshots into one ZIP.

    The whole archive is built in an in-memory `io.BytesIO` and returned as
    this single response's body -- nothing is written under the job
    directory. Unlike every route above that serves a file already cached
    on disk (`get_clinical_job_dicom_seg`, `get_clinical_job_report`, ...),
    this route's only output is bytes on the wire, so calling it twice, or
    never, never changes what is on disk for this job.

    The archive holds:
        - `report.json`: `_load_merged_clinical_report(job_id)` (the T5.5
          pathology-merged dict `GET .../report` also returns), re-encoded
          with `json.dumps(..., indent=2)`. That helper's own 404 (unknown
          job, not done, or done with no cached report) is left to
          propagate -- a bundle with no report is not a bundle, so there is
          nothing useful to zip.
        - `report.md`: `render_markdown` on that SAME dict, so the Markdown
          in the zip can never disagree with the JSON sitting next to it.
        - `dicom-seg.dcm`: this job's cached DICOM-SEG bytes, if
          `_clinical_dicom_seg_path` names an existing file. Omitted, not
          failed, otherwise -- a missing SEG object (export refused or
          failed for this case) must not block the rest of the bundle;
          `MANIFEST.txt` records the omission instead.
        - `job.json`: the bytes of `clinical_jobs._job_json_path` if that
          file survived on disk, else a fresh `json.dumps` of the exact dict
          `GET /clinical/jobs/{job_id}` returns -- so the bundle always
          carries some record of the job's state even for an
          in-memory-only job.
        - `snapshots/<safe name>.png`: one entry per uploaded file (see
          `_sanitise_snapshot_name`). Each is rejected with 400 if over
          `_MAX_SNAPSHOT_BYTES` or if its first 8 bytes are not the PNG
          signature -- a bad upload must not silently ship a non-image
          inside a clinical bundle.
        - `MANIFEST.txt`: a first line naming the job, case and generation
          time, then one `_manifest_line` per entry above (or an `ABSENT`
          line for a missing DICOM-SEG) -- so a reader of the zip, offline,
          with no access to the live job, can tell which artifacts were
          present and verify every byte against its recorded sha256.

    Args:
        job_id: The clinical job to export.
        snapshots: Zero or more viewer-snapshot PNGs, uploaded as
            `multipart/form-data` under the repeated field name
            `"snapshots"`.

    Returns:
        A `Response` with `media_type="application/zip"` and a
        `Content-Disposition` naming the download `neurovision-<job_id>.zip`.
    """
    settings = get_settings()
    job = _require_done_clinical_job(job_id)

    from neurovision.reporting.report import render_markdown

    report = _load_merged_clinical_report(job_id)
    report_json_bytes = json.dumps(report, indent=2).encode("utf-8")
    report_md_bytes = render_markdown(report).encode("utf-8")

    dicom_seg_path = _clinical_dicom_seg_path(settings, job)
    dicom_seg_bytes = dicom_seg_path.read_bytes() if dicom_seg_path.is_file() else None

    job_json_path = clinical_jobs._job_json_path(settings, job_id)
    if job_json_path.is_file():
        job_json_bytes = job_json_path.read_bytes()
    else:
        job_json_bytes = json.dumps(_clinical_job_to_json(job), indent=2).encode("utf-8")

    snapshot_entries: list[tuple[str, bytes]] = []
    taken_names: set[str] = set()
    for index, upload in enumerate(snapshots):
        content = await upload.read()
        if len(content) > _MAX_SNAPSHOT_BYTES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"snapshot {upload.filename!r} is {len(content)} bytes, over the "
                    f"{_MAX_SNAPSHOT_BYTES} byte limit"
                ),
            )
        if content[:8] != _PNG_SIGNATURE:
            raise HTTPException(
                status_code=400,
                detail=f"snapshot {upload.filename!r} is not a valid PNG (bad signature)",
            )
        name = _sanitise_snapshot_name(upload.filename, index, taken_names)
        snapshot_entries.append((name, content))

    manifest_lines = [
        f"NeuroVision-X export  job={job_id}  case={job.case_id}  "
        f"generated_utc={datetime.now(UTC).isoformat()}"
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", report_json_bytes)
        manifest_lines.append(_manifest_line("report.json", report_json_bytes))

        archive.writestr("report.md", report_md_bytes)
        manifest_lines.append(_manifest_line("report.md", report_md_bytes))

        if dicom_seg_bytes is not None:
            archive.writestr("dicom-seg.dcm", dicom_seg_bytes)
            manifest_lines.append(_manifest_line("dicom-seg.dcm", dicom_seg_bytes))
        else:
            manifest_lines.append("dicom-seg.dcm  ABSENT (no cached DICOM-SEG for this job)")

        archive.writestr("job.json", job_json_bytes)
        manifest_lines.append(_manifest_line("job.json", job_json_bytes))

        for name, content in snapshot_entries:
            entry_name = f"snapshots/{name}"
            archive.writestr(entry_name, content)
            manifest_lines.append(_manifest_line(entry_name, content))

        archive.writestr("MANIFEST.txt", "\n".join(manifest_lines) + "\n")

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="neurovision-{job_id}.zip"'},
    )


@lru_cache(maxsize=1)
def _molecular_knowledge() -> Any:
    """Loads and caches `knowledge/molecular_markers.yaml` (T5.5).

    Composes the SAME CPU-only clinical config `clinical_jobs._generate_report`
    composes (`clinical_jobs._compose_clinical_cfg`) and reads
    `cfg.analysis.report.molecular_markers` -- the same key a live clinical
    report's "Confirmed pathology" block is built from -- so the vocabulary
    this route validates an entered value against can never silently drift
    from the one a report was assembled with. `cfg.analysis.report
    .molecular_markers` is a repo-relative path string
    (`"knowledge/molecular_markers.yaml"`), resolved against `REPO_ROOT`
    exactly like `config._path_env` resolves every other repo-relative path
    in this backend -- never a bare relative path left to the process's
    current working directory.

    `lru_cache(maxsize=1)`: `load_molecular_knowledge` parses and validates
    one small YAML file -- pure, read-only, the same file for every request
    in this process -- so caching it costs nothing a request could observe.
    Same reasoning `_atlas_bundle` already uses for the atlas.

    Import of `neurovision.reporting.molecular` is local to this function,
    matching this file's own convention for `neurovision.*` imports (see the
    module docstring and `_atlas_bundle`).

    Returns:
        The loaded `neurovision.reporting.molecular.MolecularKnowledge`.
    """
    from neurovision.reporting.molecular import load_molecular_knowledge

    cfg = clinical_jobs._compose_clinical_cfg()
    path = REPO_ROOT / str(cfg.analysis.report.molecular_markers)
    return load_molecular_knowledge(path)


def _clinical_pathology_path(settings: Settings, job: clinical_jobs.ClinicalJob) -> Path:
    """Where a clinical job's user-entered pathology values are persisted.

    `<job_dir>/pathology.json` -- a sibling of `_clinical_report_path`'s
    `<job_dir>/report/<case_id>.json` and `_clinical_dicom_seg_path`'s
    `<job_dir>/dicom_seg/<case_id>.dcm`, but not namespaced under its own
    subdirectory: there is exactly one pathology file per job, written
    interactively after the job finishes (by `PUT .../pathology`), never
    produced by the pipeline run itself.
    """
    return jobs.job_root(settings) / job.job_id / "pathology.json"


def _load_pathology_or_none(path: Path, knowledge: Any) -> dict[str, str] | None:
    """Reads and validates `<job_dir>/pathology.json`, or `None` if it is corrupt.

    Two distinct fault kinds are folded into the same `None` outcome, both
    logged at WARNING rather than raised: the file is not valid JSON (or not
    a JSON object), or it parses but names an unknown marker / an
    out-of-vocabulary value (`ValueError` from `validate_entered` -- e.g. a
    marker's `allowed_values` changed in a newer
    `knowledge/molecular_markers.yaml` than the one that wrote this file).
    Either way this is a corrupt SIDE file, never the report or the job
    itself, so a caller must never 500 a read because of it -- see
    `get_clinical_job_report`'s docstring for where this matters most.

    Args:
        path: `<job_dir>/pathology.json`. Caller checks existence first.
        knowledge: The loaded `MolecularKnowledge` to validate against.

    Returns:
        The validated `{marker_or_"histology": value}` dict (possibly empty,
        if the file legitimately contains `{}`), or `None` if the file could
        not be read as valid, in-vocabulary pathology.
    """
    from neurovision.reporting.molecular import validate_entered

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("ignoring corrupt pathology file %s: not valid JSON (%s)", path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("ignoring corrupt pathology file %s: not a JSON object", path)
        return None
    try:
        return validate_entered(knowledge, raw)
    except ValueError as exc:
        logger.warning("ignoring corrupt pathology file %s: %s", path, exc)
        return None


def _write_clinical_pathology(path: Path, pathology: dict[str, str]) -> None:
    """Writes `<job_dir>/pathology.json` atomically: a temp file, then `os.replace`.

    Same pattern `clinical_jobs._persist_clinical_job` uses for `job.json` --
    a reader (`GET .../pathology`, or `GET .../report`'s merge step) must
    never observe a half-written file. Unlike that function this one is
    allowed to raise: writing pathology IS the point of `PUT .../pathology`,
    so an I/O failure here is a real 500, not something to swallow.

    Args:
        path: `<job_dir>/pathology.json`.
        pathology: The full, already-validated `{marker_or_"histology":
            value}` mapping to write (the merged result, not just the newly
            entered keys).
    """
    tmp_path = path.parent / f"{path.name}.tmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(pathology, indent=2))
    os.replace(tmp_path, path)


@router.put("/clinical/jobs/{job_id}/pathology")
def put_clinical_job_pathology(job_id: str, pathology: dict[str, str]) -> dict[str, Any]:
    """Records entered pathology for a clinical job (T5.5) -- the one mutating job route.

    The body is a partial `{marker_or_"histology": value}` object -- keys not
    sent are left at whatever they already were, so a caller can enter IDH
    today and MGMT next week without re-sending IDH. Every key/value is
    checked with `neurovision.reporting.molecular.validate_entered` against
    the same knowledge base `_molecular_knowledge` caches; an unknown key or
    an out-of-vocabulary value is a 400 with `validate_entered`'s own
    message, which already names the allowed values.

    The merged result is written to `<job_dir>/pathology.json`
    (`_write_clinical_pathology`, atomic). The cached report JSON at
    `_clinical_report_path` is NEVER rewritten here -- `GET .../report`
    merges the two live, on every read (see that route's docstring), so a
    correction made here is visible on the very next report fetch without
    this route needing to know anything about report structure.

    Args:
        job_id: The clinical job to record pathology for.
        pathology: The partial entered-values mapping, as JSON.

    Returns:
        `{"job_id", "pathology": <merged dict>, "cns5": <cns5_lookup
        result>}` -- `cns5` is computed from the FULL merged set (not just
        the newly entered keys), same as `merge_pathology` does internally.

    Raises:
        HTTPException: 404 unknown job, 409 not done
            (`_require_done_clinical_job`), 400 if `pathology` fails
            `validate_entered`.
    """
    from neurovision.reporting.molecular import cns5_lookup, validate_entered

    job = _require_done_clinical_job(job_id)
    knowledge = _molecular_knowledge()
    try:
        checked = validate_entered(knowledge, pathology)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = get_settings()
    path = _clinical_pathology_path(settings, job)
    # No file yet (this job's first PUT) reads as "nothing entered" -- same
    # as a corrupt existing file, which also degrades to "nothing entered"
    # (logged at WARNING inside the helper) rather than blocking this write.
    # The caller is actively correcting the record right now, and refusing
    # that because of an unrelated earlier corruption would be the more
    # harmful failure mode.
    existing = _load_pathology_or_none(path, knowledge) if path.is_file() else {}
    merged = {**(existing or {}), **checked}
    _write_clinical_pathology(path, merged)

    return {"job_id": job_id, "pathology": merged, "cns5": cns5_lookup(knowledge, merged)}


@router.get("/clinical/jobs/{job_id}/pathology")
def get_clinical_job_pathology(job_id: str) -> dict[str, Any]:
    """Returns a clinical job's currently entered pathology values.

    Used by the pathology panel to reload its own state (e.g. after a page
    refresh). A missing or corrupt `pathology.json` both read as "nothing
    entered yet" (`{}`) -- see `_load_pathology_or_none` -- rather than 404
    or 500, since "no pathology entered" is a normal state for a job, not an
    error.

    Returns:
        `{"job_id", "pathology": <dict>}`.

    Raises:
        HTTPException: 404 unknown job, 409 not done
            (`_require_done_clinical_job`).
    """
    job = _require_done_clinical_job(job_id)
    settings = get_settings()
    path = _clinical_pathology_path(settings, job)
    pathology: dict[str, str] = {}
    if path.is_file():
        knowledge = _molecular_knowledge()
        pathology = _load_pathology_or_none(path, knowledge) or {}
    return {"job_id": job_id, "pathology": pathology}


def _register_exception_handlers(app: FastAPI) -> None:
    """Maps `volumes.py`'s exceptions onto HTTP responses, once for every route.

    `FileNotFoundError` and `KeyError` mean "the thing you asked for isn't
    there" -> 404. `ValueError` from `_crop_to_meta` means the saved
    prediction and `meta.json` came from different preprocessing runs, and
    `ValueError` from `_load_verified_report` means a report's provenance
    names a different segmentation than the one being displayed -- both are
    server-side data inconsistencies the caller did nothing wrong to
    trigger, so they are a 500 with the message intact rather than a 404.
    `json.JSONDecodeError` (a malformed report file) is registered
    separately, ahead of `ValueError`, because it IS a `ValueError` subclass
    and would otherwise be caught by that handler first with a less specific
    message.
    """

    @app.exception_handler(FileNotFoundError)
    async def _handle_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def _handle_bad_key(request: Request, exc: KeyError) -> JSONResponse:
        # KeyError's own __str__ re-wraps its message in repr() quoting
        # (str(KeyError("x")) == "'x'"), which would double-quote a message
        # that is already a plain sentence. Read the original arg instead.
        message = exc.args[0] if exc.args else str(exc)
        return JSONResponse(status_code=404, content={"detail": str(message)})

    @app.exception_handler(json.JSONDecodeError)
    async def _handle_bad_json(request: Request, exc: json.JSONDecodeError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": f"malformed report JSON: {exc}"})

    @app.exception_handler(ValueError)
    async def _handle_bad_geometry(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})


def create_app() -> FastAPI:
    """Builds the demo's FastAPI app.

    Reads no configuration itself beyond `NVX_CORS_ORIGINS`; every data path
    is resolved lazily inside route handlers via `get_settings()`, so this
    function does no filesystem I/O of its own except the one existence
    check that decides whether to mount the built frontend.
    """
    app = FastAPI(title="NeuroVision-X Demo API")

    # 1. CORS -- must be added before routes are registered so it wraps them.
    origins = list(_DEFAULT_ORIGINS)
    extra = os.environ.get("NVX_CORS_ORIGINS", "")
    origins.extend(origin.strip() for origin in extra.split(",") if origin.strip())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # POST/DELETE cover the upload/job routes added alongside the
        # read-only GET routes above; PUT covers T5.5's one mutating route,
        # PUT /clinical/jobs/{job_id}/pathology -- a cross-origin browser
        # client would otherwise fail CORS preflight on these methods.
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        # Without this, a cross-origin fetch() in the Vite dev server can see
        # the response body but not these headers -- and the frontend cannot
        # reshape a flat buffer without knowing its shape. X-Uncertainty-Kind
        # belongs here for the same reason: the frontend labels the layer from
        # it, and a hidden header would make it fall back to a generic label
        # for a quantity that must never be mislabelled as epistemic.
        expose_headers=["X-Volume-Shape", "X-Volume-Dtype", "X-Uncertainty-Kind"],
    )
    _register_exception_handlers(app)

    # 2. Routes, under /api.
    app.include_router(router)

    # 2b. Reload any clinical job whose job.json survived a prior process's
    # restart (see clinical_jobs.py's module docstring, "Persistence").
    # Wrapped in try/except: a bad job directory on disk must never stop the
    # app from starting.
    try:
        n_rehydrated = clinical_jobs.rehydrate_clinical_jobs(get_settings())
        if n_rehydrated:
            logger.info("create_app: rehydrated %d clinical job(s) from disk", n_rehydrated)
    except Exception:  # noqa: BLE001 - a bad job dir must never fail app creation
        logger.error("create_app: failed to rehydrate clinical jobs", exc_info=True)

    # 3. Static frontend, LAST -- mounted at "/" it would otherwise shadow
    # /api if registered first. Optional: the API must work standalone with
    # no frontend built, e.g. while iterating on it from the docs or curl.
    frontend_dist = REPO_ROOT / "app" / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    else:
        logger.info("no built frontend at %s; serving the API only", frontend_dist)

    return app
