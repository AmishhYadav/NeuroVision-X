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
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import REPO_ROOT, get_settings
from .volumes import (
    MODALITIES,
    case_metrics,
    list_cases,
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

    return {"meta": meta.to_json(), "metrics": metrics, "regions": regions}


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


def _register_exception_handlers(app: FastAPI) -> None:
    """Maps `volumes.py`'s exceptions onto HTTP responses, once for every route.

    `FileNotFoundError` and `KeyError` mean "the thing you asked for isn't
    there" -> 404. `ValueError` from `_crop_to_meta` means the saved
    prediction and `meta.json` came from different preprocessing runs -- a
    server-side data inconsistency the caller did nothing wrong to trigger,
    so it is a 500 with the message intact rather than a 404.
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
        allow_methods=["GET"],
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

    # 3. Static frontend, LAST -- mounted at "/" it would otherwise shadow
    # /api if registered first. Optional: the API must work standalone with
    # no frontend built, e.g. while iterating on it from the docs or curl.
    frontend_dist = REPO_ROOT / "app" / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    else:
        logger.info("no built frontend at %s; serving the API only", frontend_dist)

    return app
