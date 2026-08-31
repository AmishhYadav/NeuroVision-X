"""Live inference for the demo backend: segment ONE preprocessed case on CPU.

The rest of the demo (`api.py`, `volumes.py`) serves PRECOMPUTED
`scripts/evaluate.py` output and never imports torch, so a viewer session
costs no GPU time and needs no checkpoint. This module is the one place that
breaks that rule on purpose -- it is what lets a user segment a case that was
never run through `scripts/evaluate.py`, using the checkpoint named by
`NVX_CHECKPOINT`.

Two properties this module must hold, because a machine running the demo may
have neither a checkpoint nor torch usefully installed:

1. Importing this module must never fail because a checkpoint is missing.
   `checkpoint_available` / `inference_status` never raise.
2. Importing this module must not import torch or Hydra at module scope.
   Both are imported lazily, inside the functions that actually need them,
   so `import app.backend.inference` alone costs nothing and cannot fail on
   a machine with no GPU stack.

Everything here runs on CPU (`neurovision.utils.device.get_device("cpu")`,
never `.cuda()` or an implicit "auto" resolution) -- see CLAUDE.md's "Mac is
a correctness harness, not a compute device" rule. A prediction is written in
CROPPED geometry, the same frame `prep_dir`'s `image.npy`/`label.npy` are in
and the frame `app/backend/volumes.py` serves from -- it is deliberately NOT
uncropped back to original BraTS geometry (see `segment_case`'s docstring).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Settings

logger = logging.getLogger(__name__)

# --- Grad-CAM target layer -----------------------------------------------
#
# Empirically chosen, not guessed from reading the architecture source -- see
# `explain_case`'s docstring for why that matters. `available_layers()` on a
# real `neurovision` (`NeuroVisionX`) checkpoint returns 302 candidate names.
# The ones shaped like a "middle decoder stage" (see
# `neurovision.explainability.gradcam.grad_cam`'s own docstring on layer
# choice) are the 4 top-level `UNetDecoder` stages,
# `decoder.conv_blocks.{0,1,2,3}`, fine to coarse: stage 0 runs at full
# resolution (one 1x1x1 conv away from the segmentation logits -- a CAM there
# is nearly the mask itself), stage 3 sits right next to the bottleneck (very
# blurry). `decoder.conv_blocks.2` is the middle-of-range pick, VERIFIED end
# to end with a real `grad_cam()` forward+backward call against a real
# `neurovision` model at 64^3 (this checkpoint's own trained
# `data.patch_size`): it ran in ~3.3s, produced a finite `(1, 1, 64, 64, 64)`
# cam correctly upsampled from its native `(1, 1, 16, 16, 16)` resolution
# (stride 4, i.e. halfway between stage 0's stride 1 and the stride-16
# bottleneck), with a nonzero target score.
#
# This constant is tied to the `NeuroVisionX` architecture specifically --
# `explain_case` is only ever called against the pinned `neurovision`
# checkpoint in this codebase (see `app/backend/clinical_jobs.py`'s module
# docstring), so that is not a real limitation in practice, but calling it
# against a different experiment (e.g. `baseline_unet3d`, a plain MONAI
# `UNet` with no `decoder.conv_blocks` submodule at all) will raise a clear
# `ValueError` from `resolve_layer` rather than silently doing something
# meaningless.
_GRADCAM_TARGET_LAYER = "decoder.conv_blocks.2"

# `explain_case` is scoped to these two regions only -- the same two the
# clinical gatekeeper and the conformal-band route already operate on
# (`cfg.clinical.gatekeeper.regions`), not a third ("ET") nobody downstream
# reads yet. A deliberate scoping choice, not an oversight -- see
# `cached_gradcam_path`'s docstring.
_GRADCAM_REGIONS = ("WT", "TC")

# --- Locking -----------------------------------------------------------
#
# Two different hazards, two different locks:
#
# 1. `_HYDRA_LOCK` guards `hydra.initialize_config_dir` / `hydra.compose`.
#    Hydra composition mutates a process-global singleton (`GlobalHydra`),
#    so two threads composing a config at the same time can corrupt each
#    other's state. This lock's critical section is just the compose call,
#    so it does not block one case's model build/inference while another
#    case's config is being composed.
# 2. `_CASE_LOCKS` (one `threading.Lock` per `(experiment, case_id)`, built
#    lazily under `_CASE_LOCKS_GUARD`) serialises the build+infer+write for
#    ONE case, so two concurrent requests for the same case cannot both
#    decide the cache is empty and race to write it. Different case ids (or
#    the same case id under a different experiment) get different locks and
#    run fully in parallel -- this is what the web server needs, since one
#    slow case must not block every other request.
_HYDRA_LOCK = threading.Lock()
_CASE_LOCKS_GUARD = threading.Lock()
_CASE_LOCKS: dict[tuple[str, str], threading.Lock] = {}


def _lock_for(experiment: str, case_id: str) -> threading.Lock:
    """Returns the (lazily created) lock guarding one `(experiment, case_id)` pair."""
    key = (experiment, case_id)
    with _CASE_LOCKS_GUARD:
        lock = _CASE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CASE_LOCKS[key] = lock
        return lock


def checkpoint_available(settings: Settings) -> bool:
    """True if a usable checkpoint file exists for live inference.

    Args:
        settings: Resolved backend settings.

    Returns:
        Whether `settings.checkpoint` exists and is a regular file. Never
        raises -- an unreadable path (permission error, a symlink loop, ...)
        also reads as "not available" rather than propagating an OS error to
        a caller that only wants a yes/no answer.
    """
    try:
        return settings.checkpoint.is_file()
    except OSError:
        logger.warning("could not stat checkpoint path %s", settings.checkpoint, exc_info=True)
        return False


def inference_status(settings: Settings) -> dict[str, Any]:
    """Human-readable description of whether live inference can run, and why not.

    Args:
        settings: Resolved backend settings.

    Returns:
        `{"available": bool, "checkpoint": str, "experiment": str,
        "reason": str | None}`. `reason` is `None` when `available` is True,
        and a specific, actionable sentence naming the environment variable
        to set otherwise. Never raises.
    """
    available = checkpoint_available(settings)
    reason = None
    if not available:
        reason = (
            f"no checkpoint at {settings.checkpoint}; set NVX_CHECKPOINT to point at a "
            "trained checkpoint's .pt file to enable live inference"
        )
    return {
        "available": available,
        "checkpoint": str(settings.checkpoint),
        "experiment": settings.experiment,
        "reason": reason,
    }


def cached_prediction_path(settings: Settings, case_id: str) -> Path:
    """Where a live-inference prediction for `case_id` is cached.

    Namespaced by `settings.experiment`: two different checkpoints must
    never share a cache entry, or switching `NVX_EXPERIMENT` (to point at a
    different trained model) would silently keep serving the OLD model's
    segmentation for a case that was already cached under the old name.

    Args:
        settings: Resolved backend settings.
        case_id: Case identifier.

    Returns:
        `<cache_dir>/<experiment>/<case_id>.npy`. The directory is NOT
        created here -- only `segment_case` writes to it.
    """
    return settings.cache_dir / settings.experiment / f"{case_id}.npy"


def cached_logits_path(settings: Settings, case_id: str) -> Path:
    """Where raw (pre-postprocessing) logits for a live-inference run of `case_id` are cached.

    Namespaced by `settings.experiment` for the same reason
    `cached_prediction_path` is: two different checkpoints must never share a
    cache entry, or switching `NVX_EXPERIMENT` (to point at a different
    trained model) would silently mix logits from the OLD model into a
    directory a caller now expects to hold the NEW one's.

    This mirrors the on-disk convention `scripts/evaluate.py` already uses
    for every saved eval run (`<eval_dir>/logits/<case_id>.npy`, fp16), so a
    downstream consumer that reads logits from an eval directory can read
    from this cache with the same code path.

    Args:
        settings: Resolved backend settings.
        case_id: Case identifier.

    Returns:
        `<cache_dir>/<experiment>/logits/<case_id>.npy`. The directory is NOT
        created here -- only `segment_case` (when called with
        `save_logits=True`) writes to it.
    """
    return settings.cache_dir / settings.experiment / "logits" / f"{case_id}.npy"


def cached_gradcam_path(settings: Settings, case_id: str, region: str) -> Path:
    """Where a Seg-Grad-CAM heatmap for one region of a live-inference run of `case_id` is cached.

    Namespaced by `settings.experiment` for the same reason `cached_logits_path` is:
    two different checkpoints must never share a cache entry, or switching
    `NVX_EXPERIMENT` (to point at a different trained model) would silently mix one
    model's explanation into a directory a caller now expects to hold another's.

    Scoped to `region in {"WT", "TC"}` only -- deliberately not `"ET"`. These are the
    same two regions the clinical gatekeeper's signals and the conformal-band route
    (`app.backend.clinical_jobs.clinical_conformal_band_mask`,
    `cfg.clinical.gatekeeper.regions`) already operate on, so this cache follows that
    existing scope rather than introducing a third region nobody downstream reads yet.
    This function does not itself validate `region` (see `explain_case`, which does,
    for the function that actually computes something) -- it only builds a path.

    Args:
        settings: Resolved backend settings.
        case_id: Case identifier.
        region: One of `"WT"`, `"TC"`.

    Returns:
        `<cache_dir>/<experiment>/gradcam/<region>/<case_id>.npy`. The directory is
        NOT created here -- only `explain_case` writes to it.
    """
    return settings.cache_dir / settings.experiment / "gradcam" / region / f"{case_id}.npy"


def _report(progress: Callable[[str, float], None] | None, stage: str, fraction: float) -> None:
    """Calls `progress(stage, fraction)` if a callback was given."""
    if progress is not None:
        progress(stage, fraction)


def _compose_cfg(settings: Settings) -> Any:
    """Composes the real Hydra config for `settings.experiment`, CPU-only.

    Mirrors how `scripts/evaluate.py` composes its config (same `configs/`
    directory, same `hydra.initialize_config_dir` / `hydra.compose`
    programmatic API `scripts/smoke_test.py` also uses), but does it lazily
    and against paths taken entirely from `settings` -- never a hardcoded
    path.

    `+experiment={settings.experiment}` selects the exact architecture and
    hyperparameters the checkpoint was trained under (e.g. `baseline_unet3d`,
    `neurovision`); `data.root_dir` / `data.preprocessing.out_dir` are
    pointed at `settings.prep_dir` so composition succeeds without a real
    BraTS root even though `data.root_dir` is Hydra-mandatory; `device=cpu`
    and the sliding-window overlap come from `settings` so the composed
    config matches what this module actually runs.

    Args:
        settings: Resolved backend settings.

    Returns:
        The composed `DictConfig`.
    """
    import hydra

    config_dir = str(REPO_ROOT / "configs")
    overrides = [
        f"+experiment={settings.experiment}",
        f"data.root_dir={settings.prep_dir}",
        f"data.preprocessing.out_dir={settings.prep_dir}",
        "device=cpu",
        f"inference.sliding_window.overlap={settings.demo_overlap}",
    ]
    with _HYDRA_LOCK:
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def _load_model(settings: Settings) -> tuple[Any, Any, Any]:
    """Composes the config, builds the model, and loads the checkpoint's weights.

    Args:
        settings: Resolved backend settings.

    Returns:
        `(model, cfg, device)`: the model in eval mode on the CPU device, the
        composed config (needed downstream by `sliding_window_predict` /
        `postprocess_logits`), and the resolved `torch.device`.
    """
    # Importing the models package (via build_model's own module) runs the
    # @register_model decorators for both "unet3d"/"swinunetr" (baseline.py)
    # and "neurovision" (neurovision.py) -- see
    # src/neurovision/models/__init__.py, which imports both submodules for
    # exactly this side effect. No separate import is needed here.
    from neurovision.models.registry import build_model
    from neurovision.training.checkpoint import load_checkpoint
    from neurovision.utils.device import get_device

    cfg = _compose_cfg(settings)
    device = get_device("cpu")  # never "auto", never .cuda() -- the demo is CPU-only

    model = build_model(cfg)
    model = model.to(device)
    resume_state = load_checkpoint(
        settings.checkpoint, model, map_location=str(device), restore_rng=False
    )
    model.eval()
    logger.info(
        "Loaded checkpoint %s for live inference: epoch=%d",
        settings.checkpoint,
        resume_state.start_epoch - 1,
    )
    return model, cfg, device


def _predict_logits(model: Any, cfg: Any, device: Any, image_path: Path) -> Any:
    """Runs sliding-window inference and returns raw logits (no sigmoid, no threshold).

    Args:
        model: A model already built and loaded, in eval mode, on `device`.
        cfg: The composed config `model` was built from.
        device: The resolved `torch.device` (CPU).
        image_path: Path to the case's preprocessed `image.npy`,
            `(C, D, H, W)` float16, channel order `(t1, t1ce, t2, flair)`.

    Returns:
        Raw logits, shape `(1, 3, D, H, W)`, channel order `(ET, TC, WT)`.
    """
    import numpy as np
    import torch

    from neurovision.inference.sliding_window import sliding_window_predict

    image = np.asarray(np.load(image_path), dtype=np.float32)  # (C, D, H, W)
    image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, C, D, H, W)

    with torch.no_grad():
        return sliding_window_predict(model, image_tensor, cfg, device)  # (1, 3, D, H, W)


def _postprocess_to_classes(logits: Any, cfg: Any) -> Any:
    """Thresholds, nests and collapses raw logits into a `{0,1,2,3}` class map.

    Args:
        logits: Raw logits, shape `(1, 3, D, H, W)`, from `_predict_logits`.
        cfg: The composed config, read for `cfg.inference.postprocess`.

    Returns:
        A `numpy.ndarray[uint8]` of shape `(D, H, W)`, values in `{0, 1, 2,
        3}`, in the SAME cropped geometry `logits` was computed in -- see
        `segment_case`'s docstring for why this is never uncropped.
    """
    from neurovision.inference.postprocess import postprocess_logits, regions_to_classes

    regions = postprocess_logits(logits, cfg)  # (1, 3, D, H, W), nested ET<=TC<=WT
    classes = regions_to_classes(regions)  # (1, D, H, W), values in {0,1,2,3}
    return classes[0].cpu().numpy().astype("uint8")  # (D, H, W)


def _atomic_np_save(array: Any, destination: Path) -> None:
    """Writes `array` to `destination` atomically, mirroring `training/checkpoint.py`.

    Temp file in the SAME directory as `destination`, then `os.replace` --
    atomic on macOS and Linux as long as both are on one filesystem, which is
    why the temp file is not placed under `/tmp`. A killed write can
    therefore never leave a half-written cache entry that has already
    replaced a good one; the destination either has the old content or the
    new content, never a truncated one.

    Args:
        array: A numpy array to save with `np.save`.
        destination: Final `.npy` path.
    """
    import numpy as np

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".npy.tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        # np.save appends ".npy" to a PATH argument that lacks one, which
        # would silently write to a different filename than tmp_path (whose
        # suffix is ".npy.tmp", not ".npy") and break the os.replace below.
        # Passing an open file OBJECT instead makes numpy write exactly to
        # that file, with no filename manipulation.
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
        os.replace(tmp_path, destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def segment_case(
    settings: Settings,
    case_id: str,
    *,
    force: bool = False,
    save_logits: bool = False,
    progress: Callable[[str, float], None] | None = None,
) -> Path:
    """Segments one PREPROCESSED case with the configured checkpoint, caching the result.

    Geometry: the returned prediction is in CROPPED geometry -- the same
    frame `<prep_dir>/<case_id>/image.npy` and `label.npy` are in, and the
    frame `app/backend/volumes.py` serves everything else from. This is
    deliberately NOT uncropped back to original BraTS geometry (unlike
    `scripts/evaluate.py`'s saved `predictions/`, which uncrop for BraTS
    submission validity): uncropping here would just have to be undone again
    by the exact re-crop `volumes.py` already applies to `evaluate.py`'s
    output, and doing it twice is two chances to get the bbox offset wrong
    while still looking plausible -- the geometry trap CLAUDE.md documents.

    Args:
        settings: Resolved backend settings.
        case_id: Case identifier; must have a `<prep_dir>/<case_id>/image.npy`.
        force: If True, recompute and overwrite even if a cached prediction
            already exists.
        save_logits: If True, also persist the raw (pre-postprocessing)
            logits to `cached_logits_path(settings, case_id)`, fp16, shape
            `(3, D, H, W)` -- the same on-disk convention
            `scripts/evaluate.py` uses for its saved `logits/`. Default
            False, so every existing caller that never asks for logits is
            byte-for-byte unaffected.

            A cache is only a hit if it can satisfy THIS call: if the
            prediction is already cached but `save_logits=True` and no
            logits file exists yet (e.g. an earlier call ran with
            `save_logits=False`), that is treated exactly like `force=True`
            for this call only -- inference is re-run and BOTH files are
            rewritten. This is deliberate: silently returning the old
            prediction would leave the caller believing logits were saved
            when they never were.
        progress: Optional callback invoked with `(stage_name, fraction)` at
            `("loading model", 0.1)`, `("running inference", 0.3)`,
            `("post-processing", 0.85)`, `("done", 1.0)`. Not called at all
            on a cache hit -- nothing is done in that case, so there is no
            stage to report.

    Returns:
        Path to the cached `.npy` prediction (uint8, values in `{0,1,2,3}`,
        shape `(D, H, W)` matching the case's cropped image shape). Callers
        that also passed `save_logits=True` and want the logits path call
        `cached_logits_path(settings, case_id)` themselves afterward -- the
        same pattern `cached_prediction_path` already establishes.

    Raises:
        FileNotFoundError: If `settings.checkpoint` does not exist, or if
            `<prep_dir>/<case_id>/image.npy` does not exist.
    """
    out_path = cached_prediction_path(settings, case_id)
    logits_path = cached_logits_path(settings, case_id)

    # A prediction cache hit only counts if it satisfies THIS call: a caller
    # asking for logits that were never saved must not be handed a stale
    # prediction with no logits to go with it (see the `save_logits` Args
    # note above) -- so that case is folded into the same "must recompute"
    # condition as force=True.
    logits_missing = save_logits and not logits_path.is_file()

    # Fast path: no torch, no Hydra, no lock -- just filesystem checks. Most
    # calls in a running demo hit this, since a case is only ever segmented
    # once per (experiment, case_id).
    if out_path.is_file() and not force and not logits_missing:
        logger.info("Using cached live-inference prediction for %s at %s", case_id, out_path)
        return out_path

    image_path = settings.prep_dir / case_id / "image.npy"
    if not image_path.is_file():
        raise FileNotFoundError(
            f"no preprocessed case {case_id!r} at {image_path} (expected under "
            f"settings.prep_dir={settings.prep_dir})"
        )

    if not checkpoint_available(settings):
        raise FileNotFoundError(
            f"no checkpoint at {settings.checkpoint}; set NVX_CHECKPOINT to a trained "
            "checkpoint's .pt file to enable live inference"
        )

    lock = _lock_for(settings.experiment, case_id)
    with lock:
        # Re-check under the lock: another thread may have finished
        # segmenting this exact case (and, if requested, its logits) while
        # this one was waiting to acquire it, in which case there is nothing
        # left to do (unless force=True or the logits are still missing).
        logits_missing = save_logits and not logits_path.is_file()
        if out_path.is_file() and not force and not logits_missing:
            logger.info(
                "Using cached live-inference prediction for %s at %s (written while waiting "
                "for the lock)",
                case_id,
                out_path,
            )
            return out_path

        _report(progress, "loading model", 0.1)
        model, cfg, device = _load_model(settings)

        _report(progress, "running inference", 0.3)
        logits = _predict_logits(model, cfg, device, image_path)

        _report(progress, "post-processing", 0.85)
        classes = _postprocess_to_classes(logits, cfg)
        _atomic_np_save(classes, out_path)
        logger.info("Wrote live-inference prediction for %s to %s", case_id, out_path)

        if save_logits:
            import numpy as np

            # Drop the leading batch dim of 1 (logits is `(1, 3, D, H, W)`)
            # and cast to fp16 to match the project-wide convention for
            # saved logits on disk -- see cached_logits_path's docstring.
            logits_np = logits.squeeze(0).cpu().numpy().astype(np.float16)  # (3, D, H, W)
            _atomic_np_save(logits_np, logits_path)
            logger.info("Wrote live-inference logits for %s to %s", case_id, logits_path)

        _report(progress, "done", 1.0)
        return out_path


def explain_case(
    settings: Settings,
    case_id: str,
    region: str,
    *,
    force: bool = False,
) -> Path:
    """Computes (or returns the cached) Seg-Grad-CAM heatmap for one region of one case.

    A sibling of `segment_case`, not a modification of it: this function is only
    ever useful AFTER `segment_case` has already produced a cached prediction for
    `case_id` (see the `Raises` section) -- it explains an existing segmentation, it
    does not produce one. Every call that is not a cache hit loads a FRESH model via
    `_load_model` (the same helper `segment_case` uses), exactly like `segment_case`
    does; there is no model-caching layer anywhere in this backend today (see
    `_load_model`'s own body), so this is a second full model load per job, on
    purpose, following the existing convention rather than introducing a new one.

    Geometry: like `segment_case`, the returned heatmap is in CROPPED geometry -- the
    same frame `<prep_dir>/<case_id>/image.npy` and the cached prediction are in (see
    `segment_case`'s own docstring for why that frame is never uncropped). The actual
    Seg-Grad-CAM computation only ever runs on a small PATCH cropped around the
    region's predicted footprint (see `neurovision.explainability.gradcam`'s
    top-of-file docstring, hazard 4, for why a whole-volume backward pass does not
    fit this project's memory budget); the result is then written into an all-zero,
    FULL-image-shaped buffer at that patch's own location (the `slices`
    `center_patch_on_mask` returns), never returned in patch-local coordinates alone.
    Voxels outside the patch are left at 0, meaning "no evidence computed here" --
    the same convention this project's other per-voxel buffers (predictive entropy,
    the conformal band) already use for "nothing to show" there.

    Args:
        settings: Resolved backend settings.
        case_id: Case identifier; must have both a `<prep_dir>/<case_id>/image.npy`
            and an already-cached prediction (i.e. `segment_case` must have run for
            this exact `(settings.experiment, case_id)` first).
        region: One of `"WT"`, `"TC"` -- see `cached_gradcam_path`'s docstring for
            why `"ET"` is deliberately out of scope here, not an oversight.
        force: If True, recompute and overwrite even if a cached heatmap already
            exists.

    Returns:
        Path to the cached `.npy` heatmap: `uint8`, values in `[0, 255]`, shape
        matching the case's cropped image shape (i.e. `image.npy`'s spatial shape).

    Raises:
        ValueError: If `region` is not `"WT"` or `"TC"`.
        FileNotFoundError: If `<prep_dir>/<case_id>/image.npy` does not exist, or if
            no prediction has been cached yet for this `(settings.experiment,
            case_id)` (call `segment_case` first).
    """
    if region not in _GRADCAM_REGIONS:
        raise ValueError(
            f"region must be one of {_GRADCAM_REGIONS!r}, got {region!r}. 'ET' is "
            "deliberately out of scope for live Grad-CAM -- see cached_gradcam_path's "
            "docstring."
        )

    out_path = cached_gradcam_path(settings, case_id, region)

    # Fast path, mirroring segment_case's own cache-hit short-circuit: no torch, no
    # Hydra, no model load -- just a filesystem check.
    if out_path.is_file() and not force:
        logger.info("Using cached Seg-Grad-CAM for %s/%s at %s", case_id, region, out_path)
        return out_path

    image_path = settings.prep_dir / case_id / "image.npy"
    if not image_path.is_file():
        raise FileNotFoundError(
            f"no preprocessed case {case_id!r} at {image_path} (expected under "
            f"settings.prep_dir={settings.prep_dir})"
        )

    pred_path = cached_prediction_path(settings, case_id)
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"no cached prediction for case {case_id!r} at {pred_path}; call "
            "segment_case(settings, case_id) before explain_case for the same "
            "(settings.experiment, case_id)."
        )

    # Same (experiment, case_id) lock segment_case uses -- so two concurrent
    # requests for the same case (e.g. one segmenting, one explaining, or two
    # explain_case calls for different regions) cannot both decide the cache is
    # empty and race to write it, and, worse, both pay the cost of a full model
    # load. See the module docstring's "Locking" section.
    lock = _lock_for(settings.experiment, case_id)
    with lock:
        # Re-check under the lock, mirroring segment_case: another thread may
        # have finished computing this exact heatmap while this one was
        # waiting to acquire the lock, in which case there is nothing left to
        # do (unless force=True).
        if out_path.is_file() and not force:
            logger.info(
                "Using cached Seg-Grad-CAM for %s/%s at %s (written while waiting " "for the lock)",
                case_id,
                region,
                out_path,
            )
            return out_path

        import numpy as np
        import torch

        from neurovision.data.transforms import REGION_NAMES
        from neurovision.explainability.gradcam import center_patch_on_mask, grad_cam

        image = np.asarray(np.load(image_path), dtype=np.float32)  # (C, D, H, W)
        prediction = np.asarray(np.load(pred_path))  # (D, H, W), values in {0, 1, 2, 3}

        # Same region definitions app/backend/volumes.py's region_voxel_counts uses --
        # that function is the source of truth, but exposes no reusable single-region
        # helper, so these are duplicated here as plain boolean expressions rather than
        # imported. ET (mask == 3) is not one of the two cases below -- see
        # cached_gradcam_path's docstring for why it is out of scope.
        if region == "TC":
            region_mask = (prediction == 1) | (prediction == 3)
        else:  # region == "WT" -- already validated above to be "WT" or "TC"
            region_mask = prediction > 0

        model, cfg, device = _load_model(settings)
        if model.training:
            # _load_model already leaves the model in eval mode, so this never actually
            # runs -- but grad_cam RAISES if model.training is True rather than fixing
            # it silently (see its own top-of-file docstring, hazard 1), so this makes
            # that requirement visible here too instead of relying on an implementation
            # detail of a function this one happens to call.
            model.eval()

        # The checkpoint's own trained patch size (cfg.data.patch_size), never an
        # arbitrary new number -- the same config _load_model already composed.
        patch_size = tuple(int(p) for p in cfg.data.patch_size)
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, C, D, H, W)
        mask_tensor = torch.from_numpy(region_mask)  # (D, H, W)
        patch, slices = center_patch_on_mask(image_tensor, mask_tensor, patch_size)

        region_index = REGION_NAMES.index(region)
        # target_mask=None (the default): grad_cam builds its OWN predicted-positive
        # mask from a fresh forward pass on the PATCH. region_mask above was only used
        # to decide WHERE to crop -- it must not also be passed in as target_mask.
        result = grad_cam(
            model,
            patch,
            target_layer=_GRADCAM_TARGET_LAYER,
            region_index=region_index,
            target_mask=None,
        )

        # Place the patch back into full-image geometry: an all-zero buffer at
        # image.npy's own spatial shape, with the CAM's values written in at exactly
        # the slices center_patch_on_mask cropped from. result.cam (not .raw_cam) is
        # already upsampled to the patch's own spatial shape by grad_cam's default
        # upsample=True, so it matches `slices`'s extents exactly.
        full = np.zeros(image.shape[1:], dtype=np.float32)  # (D, H, W)
        cam_patch = result.cam[0, 0].cpu().numpy()  # (patch_D, patch_H, patch_W)
        full[slices] = cam_patch

        # cam is already [0, 1]-normalized by grad_cam's default normalize=True -- the
        # same uint8 scaling this project's other normalized per-voxel maps already use
        # (see entropy_from_logits's own caller).
        cam_uint8 = (full * 255).astype(np.uint8)

        _atomic_np_save(cam_uint8, out_path)
        logger.info("Wrote Seg-Grad-CAM for %s/%s to %s", case_id, region, out_path)
        return out_path
