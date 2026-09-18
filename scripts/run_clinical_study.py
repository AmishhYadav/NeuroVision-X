"""Hydra entry point that runs the live clinical pipeline on ONE raw DICOM study.

T0.2 of `docs/research/tool_completion_plan.md`: "makes T0 a command, not a
story". Before this script existed, exercising the real clinical pipeline
(`app.backend.clinical_jobs`) end to end on a real DICOM study directory --
E1 ingest through E6 DICOM-SEG export and the Phase 4 report -- meant driving
it by hand from a scratch script; this is that driver made reproducible. It
replaces the ad-hoc scratch driver that produced
`outputs/clinical_jobs/a37fcaad8b324fc59cc076ee56807a11/t0_summary.json`.

What it does: zips `cfg.clinical.study_dir` in memory, creates a clinical job
under `cfg.clinical.out_dir` (the same job store `app/backend/api.py`'s
`/api/clinical/*` routes use), runs it SYNCHRONOUSLY to completion while a
background thread polls `job.stage` once a second, and writes
`<out_dir>/<job_id>/summary.json` -- the job's own state plus a stage
timeline, wall-clock time, and which supplementary artifacts (report,
DICOM-SEG, Grad-CAM, logits) actually landed on disk.

Example usage:

    .venv-clinical/bin/python scripts/run_clinical_study.py \\
        +clinical.study_dir=data/fixtures/dicom/UPENN-GBM-00001 \\
        +clinical.out_dir=outputs/clinical_jobs

`+clinical.study_dir` / `+clinical.out_dir` use Hydra's `+` prefix because
`configs/clinical/default.yaml` does not define either key -- they name a
single study run, not a property of the pipeline's own configuration.

This must run under `.venv-clinical` (pydicom / highdicom / HD-BET live only
there -- see `configs/clinical/default.yaml`'s header comment and
`requirements-clinical.txt`) for a REAL study. The test for this module runs
in the main `.venv`, with the pipeline itself faked (see
`tests/test_run_clinical_study_script.py`), so it needs none of that.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import math
import os
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

if TYPE_CHECKING:
    from app.backend.config import Settings

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/rebuild_predictions.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# scripts/run_clinical_study.py -> scripts -> repo root. Inserted onto
# sys.path below so `app.backend` imports work no matter what the caller's
# cwd is -- `app/` is a real package (has __init__.py) but is not installed
# and is not under src/, so it is not on sys.path merely by virtue of
# pyproject.toml's `pythonpath = ["src"]` pytest setting.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# How often the background thread polls job.stage, in seconds. One second
# is fine granularity for a pipeline whose stages each take seconds to
# minutes on CPU -- see clinical_jobs.py's own stage list.
_POLL_INTERVAL_S = 1.0


def zip_directory(study_dir: Path) -> bytes:
    """Zips every file under `study_dir`, recursively, in memory.

    Pure and side-effect-free (no disk writes) so it can be exercised on its
    own -- the same "make it a standalone, testable helper" pattern
    `rebuild_predictions.py` follows for its own building blocks.

    Args:
        study_dir: A directory of `.dcm` files, arbitrarily nested.

    Returns:
        Raw bytes of a `.zip` archive (DEFLATE-compressed). Every regular
        file under `study_dir` is included, with an arcname relative to
        `study_dir` itself (so the archive's top level is the study's
        contents, not `study_dir`'s own name) -- exactly the shape
        `app.backend.clinical_jobs.create_clinical_job` expects.

    Raises:
        FileNotFoundError: If `study_dir` does not exist or is not a
            directory.
        ValueError: If `study_dir` contains no regular files. A zip archive
            with zero members is still a syntactically valid zip --
            `create_clinical_job` would accept it, extract nothing, and only
            fail much later at E1 ingest with "no DICOM series" -- so an
            empty study is refused here instead, at the source, with a
            message that actually names the problem.
    """
    study_dir = Path(study_dir)
    if not study_dir.is_dir():
        raise FileNotFoundError(f"zip_directory: {study_dir} is not a directory")

    buffer = io.BytesIO()
    n_files = 0
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(study_dir.rglob("*")):
            if not path.is_file():
                continue
            # .as_posix(): the zip format's own arcname convention is
            # forward slashes regardless of platform, and this project runs
            # on both macOS and Linux (CLAUDE.md constraint 2).
            arcname = path.relative_to(study_dir).as_posix()
            archive.write(path, arcname=arcname)
            n_files += 1

    if n_files == 0:
        raise ValueError(f"zip_directory: {study_dir} contains no files")
    return buffer.getvalue()


def scan_artifacts(job_dir: Path, case_id: str) -> dict[str, bool]:
    """Checks which of a finished clinical job's supplementary artifacts landed on disk.

    Every path below matches the exact layout `app.backend.clinical_jobs`
    and `app.backend.inference` actually write (verified against a real run,
    `outputs/clinical_jobs/a37fcaad8b324fc59cc076ee56807a11/`), not a guess:
    Grad-CAM is one file per region under its own subdirectory
    (`inference.cached_gradcam_path`), never a filename that merely contains
    the region name.

    Args:
        job_dir: One clinical job's own root directory (`<out_dir>/<job_id>`).
        case_id: The job's case id (equal to its job id).

    Returns:
        `{"report", "dicom_seg", "gradcam_WT", "gradcam_TC", "logits",
        "job_json"}` -> whether that artifact exists. `"logits"` is `True`
        only if the logits directory exists AND is non-empty (an empty
        directory is not a saved logits cache).
    """
    job_dir = Path(job_dir)
    cache_root = job_dir / "cache" / "neurovision"
    logits_dir = cache_root / "logits"
    return {
        "report": (job_dir / "report" / f"{case_id}.json").is_file(),
        "dicom_seg": (job_dir / "dicom_seg" / f"{case_id}.dcm").is_file(),
        "gradcam_WT": (cache_root / "gradcam" / "WT" / f"{case_id}.npy").is_file(),
        "gradcam_TC": (cache_root / "gradcam" / "TC" / f"{case_id}.npy").is_file(),
        "logits": logits_dir.is_dir() and any(logits_dir.iterdir()),
        "job_json": (job_dir / "job.json").is_file(),
    }


def _json_sanitize(value: Any) -> Any:
    """Recursively replaces non-finite floats and numpy scalars with JSON-safe values.

    `neurovision.inference.gatekeeper.GateDecision.to_dict()` and
    `neurovision.inference.input_qc.InputQCReport.to_dict()` do not strip
    NaN from an unmeasured or undefined signal (e.g. `predicted_dice` for a
    case with no enhancing tumour) -- so `ClinicalJob.gatekeeper_decision` /
    `.input_qc_pre` / `.input_qc_post` can carry a literal `float("nan")`
    all the way into this script's summary. Plain `json.dumps` writes NaN as
    the bare (non-standard, JSON-illegal) token `NaN`, which most JSON
    parsers other than Python's own reject -- this function turns every one
    into an explicit `null` instead, applied once, recursively, before
    `run_study` calls `json.dumps(..., allow_nan=False)`. `allow_nan=False`
    is kept as a second, independent safety net: if this function ever
    misses a case, `json.dumps` raises loudly rather than writing invalid
    JSON silently.

    Args:
        value: Any JSON-dumpable-ish structure -- typically
            `dataclasses.asdict(job)` -- possibly containing NaN/Inf floats
            or numpy scalar types.

    Returns:
        The same structure with every non-finite `float` (or `np.floating`)
        replaced by `None`, and every numpy scalar converted to its plain
        Python equivalent.
    """
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.generic):
        # np.generic covers every numpy scalar type (np.float32, np.int64,
        # np.bool_, ...); .item() converts to the matching plain Python
        # type, then this recurses once more so a NaN-valued np.float32
        # still gets caught by the float branch above.
        return _json_sanitize(value.item())
    if isinstance(value, dict):
        return {key: _json_sanitize(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    return value


def run_study(study_dir: Path, out_dir: Path, *, settings: Settings | None = None) -> Path:
    """Runs one raw DICOM study through the clinical pipeline, synchronously, to completion.

    Args:
        study_dir: A directory of `.dcm` files, arbitrarily nested.
        out_dir: Where clinical jobs live -- `NVX_JOB_DIR` is set to this
            (as an absolute path) before anything in `app.backend` is
            imported or `get_settings()` is called, since
            `app.backend.jobs.job_root` reads `NVX_JOB_DIR` directly from
            the environment (not from `Settings`) and `get_settings` is
            `functools.lru_cache`d -- see `app/backend/config.py` and
            `app/backend/jobs.py::job_root`.
        settings: A pre-built `Settings` to use instead of resolving one
            from the environment via `get_settings()`. `None` (the default)
            builds one the normal way; a caller (this module's own tests)
            that wants to bypass `get_settings()`'s process-wide cache
            entirely can pass its own instead.

    Returns:
        Path to the written `<out_dir>/<job_id>/summary.json`.
    """
    study_dir = Path(study_dir).resolve()
    out_dir = Path(out_dir).resolve()

    # Must happen before app.backend.config / app.backend.clinical_jobs are
    # imported and before get_settings() is ever called in this process --
    # see this function's own docstring and the module docstring's step 1.
    os.environ["NVX_JOB_DIR"] = str(out_dir)

    # dcm2niix ships inside .venv-clinical/bin (requirements-clinical.txt),
    # but neurovision.data.dicom_ingest.resolve_dcm2niix only ever searches
    # PATH -- it has no notion of "whatever venv this interpreter came
    # from". Invoking this script by its absolute interpreter path (e.g.
    # `.venv-clinical/bin/python scripts/run_clinical_study.py`, exactly the
    # module docstring's own example usage) without first `source
    # .venv-clinical/bin/activate` leaves PATH pointing at the SYSTEM
    # python's bin directories, so dcm2niix silently fails to resolve at E1
    # ingest.
    #
    # `sys.executable` is NOT the right source for that directory: inside a
    # venv it is typically a SYMLINK to the base interpreter (e.g. a
    # Homebrew/uv python), and `Path(sys.executable).resolve()` follows that
    # symlink straight OUT of the venv, landing in the base interpreter's
    # own bin dir -- never where dcm2niix actually lives. `sys.prefix`, by
    # contrast, IS the venv root by definition (a venv always sets
    # `sys.prefix` to itself, symlinks notwithstanding), so deriving the bin
    # dir from `sys.prefix` is correct regardless of how `sys.executable`
    # happens to be wired.
    venv_bin_dir = str(Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin"))
    existing_path = os.environ.get("PATH", "")
    path_entries = existing_path.split(os.pathsep) if existing_path else []
    if venv_bin_dir not in path_entries:
        os.environ["PATH"] = (
            os.pathsep.join([venv_bin_dir, existing_path]) if existing_path else venv_bin_dir
        )

    from app.backend import clinical_jobs
    from app.backend.config import get_settings

    if settings is None:
        settings = get_settings()

    payload = zip_directory(study_dir)
    job = clinical_jobs.create_clinical_job(settings, payload)
    logger.info("run_clinical_study: created job %s for study_dir=%s", job.job_id, study_dir)

    # A daemon thread records every stage change while run_clinical_job (in
    # the main thread, below) does the actual work -- the same
    # "poll a live job object from another thread" pattern the /clinical
    # frontend itself uses against the HTTP API, just in-process here.
    stage_times: list[dict[str, Any]] = []
    start = time.monotonic()
    stop_polling = threading.Event()

    def _poll_stage() -> None:
        last_stage: str | None = None

        def _record_if_changed() -> None:
            # A closure over last_stage (via nonlocal) rather than a return
            # value: both call sites below (the loop body and the final,
            # unconditional call after it) need to share the same
            # "have we already recorded this stage" state.
            nonlocal last_stage
            current = clinical_jobs.get_clinical_job(job.job_id)
            if current is None or current.stage == last_stage:
                return
            last_stage = current.stage
            elapsed = round(time.monotonic() - start, 1)
            stage_times.append({"stage": current.stage, "t": elapsed})
            logger.info("run_clinical_study: stage=%s t=%.1fs", current.stage, elapsed)

        while not stop_polling.is_set():
            _record_if_changed()
            current = clinical_jobs.get_clinical_job(job.job_id)
            if current is not None and current.state not in ("queued", "running"):
                break
            stop_polling.wait(_POLL_INTERVAL_S)

        # One last, UNCONDITIONAL read after the loop, regardless of which of
        # the two conditions above ended it. Without this, a stage set
        # between the loop's last read and the job actually finishing (e.g.
        # run_clinical_job setting stage="done" and returning almost
        # immediately afterwards, or run_study's own `finally` firing
        # stop_polling while this thread was mid-`wait`) is silently
        # dropped -- the run's true final stage would never appear in
        # stage_times at all.
        _record_if_changed()

    poll_thread = threading.Thread(target=_poll_stage, daemon=True, name="nvx-clinical-study-poll")
    poll_thread.start()

    try:
        # run_clinical_job already turns any internal failure into
        # state="failed" on the job itself and never raises for a job-level
        # outcome (see its own docstring) -- this try/except exists only for
        # the residual case of a programming error (e.g. an unknown job_id)
        # so the summary is still written with whatever is known, rather
        # than the run being lost with no artifact at all.
        clinical_jobs.run_clinical_job(settings, job.job_id)
    except Exception:  # noqa: BLE001 - see comment above: record, do not lose, do not re-raise
        logger.error(
            "run_clinical_study: run_clinical_job raised unexpectedly for job %s",
            job.job_id,
            exc_info=True,
        )
    finally:
        stop_polling.set()
        poll_thread.join(timeout=5.0)

    wall_s = round(time.monotonic() - start, 1)
    final_job = clinical_jobs.get_clinical_job(job.job_id)
    if final_job is None:
        raise RuntimeError(f"run_clinical_study: job {job.job_id} vanished from the job store")

    job_dir = clinical_jobs.jobs.job_root(settings) / job.job_id
    artifacts = scan_artifacts(job_dir, final_job.case_id)

    summary = dataclasses.asdict(final_job)
    summary["stage_times"] = stage_times
    summary["wall_s"] = wall_s
    summary["artifacts"] = artifacts

    summary_path = job_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # _json_sanitize first (turns any buried NaN/Inf into None), then
    # allow_nan=False as a second, independent check -- see
    # _json_sanitize's own docstring for why both exist.
    summary_path.write_text(
        json.dumps(_json_sanitize(summary), indent=2, default=str, allow_nan=False)
    )

    decision = None
    if final_job.gatekeeper_decision is not None:
        decision = final_job.gatekeeper_decision.get("decision")
    logger.info(
        "run_clinical_study: job %s finished state=%s gatekeeper_decision=%s summary=%s",
        job.job_id,
        final_job.state,
        decision,
        summary_path,
    )
    return summary_path


def release_hydra() -> None:
    """Clears Hydra's process-global singleton so `run_study` can compose its own config.

    `@hydra.main` (below) initialises `GlobalHydra` once, for the whole
    process, so it can compose `cfg`. But `run_study` -> `run_clinical_job`
    -> `_compose_clinical_cfg` composes its OWN config per job, independently,
    via `hydra.initialize_config_dir` (guarded by
    `app.backend.inference._HYDRA_LOCK`, since it also mutates that same
    singleton) -- and `initialize_config_dir` refuses to run at all while a
    `GlobalHydra` instance is already live (`ValueError: GlobalHydra is
    already initialized`). This script needs nothing more from Hydra once it
    has read `cfg.clinical.study_dir` / `cfg.clinical.out_dir` out of `cfg`,
    so `main` releases the singleton right after reading them and before
    calling `run_study` -- freeing the backend's own composition to run.
    """
    GlobalHydra.instance().clear()


def exit_code_for_state(state: str) -> int:
    """Maps a finished clinical job's terminal `state` to a process exit code.

    Args:
        state: A `ClinicalJob.state` value read back out of the written
            `summary.json` -- typically `"done"`, `"refused"` or
            `"failed"`.

    Returns:
        `0` for `"done"` or `"refused"` -- both are successful pipeline
        outcomes; see `app.backend.clinical_jobs`'s own module docstring for
        why a refusal is never a failure. `1` for `"failed"`, or for any
        other, unexpected value.
    """
    return 0 if state in ("done", "refused") else 1


@hydra.main(config_path=_CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Reads `cfg.clinical.study_dir` / `cfg.clinical.out_dir` and runs `run_study`.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides
            (`clinical.study_dir` / `clinical.out_dir` are supplied on the
            CLI with Hydra's `+` prefix -- see the module docstring).

    Raises:
        ValueError: If either `cfg.clinical.study_dir` or
            `cfg.clinical.out_dir` is not set.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)

    study_dir = OmegaConf.select(cfg, "clinical.study_dir")
    out_dir = OmegaConf.select(cfg, "clinical.out_dir")
    if study_dir is None or out_dir is None:
        raise ValueError(
            "run_clinical_study: both clinical.study_dir and clinical.out_dir must be set on "
            "the command line, e.g. '+clinical.study_dir=data/fixtures/dicom/UPENN-GBM-00001 "
            "+clinical.out_dir=outputs/clinical_jobs'."
        )

    # Nothing below this line needs Hydra's own composed cfg any further --
    # see release_hydra's docstring for why this has to happen before
    # run_study, not inside it.
    release_hydra()

    summary_path = run_study(Path(str(study_dir)), Path(str(out_dir)))
    state = json.loads(summary_path.read_text())["state"]
    sys.exit(exit_code_for_state(state))


if __name__ == "__main__":
    main()
