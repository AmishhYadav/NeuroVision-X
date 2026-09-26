"""Hydra entry point for P1.2, the real-DICOM validation of the clinical path.

Binding design: `docs/research/real_dicom_validation_protocol.md` -- read it FIRST.
`neurovision.analysis.real_dicom_scoring` is the pure scoring half (no file I/O, no
pipeline, no Hydra); this script is the other half: it locates each RSNA study's zip,
runs it through the REAL clinical pipeline exactly as `scripts/run_clinical_study.py`
does (same `run_study` driver, deployed `neurovision` checkpoint, deployed gate), scores
the result against the matching BraTS 2021 ground truth, and aggregates the protocol's
endpoints -- completion, usable rate end to end, front-end cost -- EXCLUDING the pilot
case from every statistic.

**Not external validation.** Every case here is a BraTS 2021 TEST-split patient (same
distribution the deployed model trained toward), re-run through real DICOM instead of
the already-preprocessed research NIfTI. This isolates what the clinical front end
(ingest, co-registration, skull-stripping) costs on its own -- see the protocol's "What
this is not" section.

## Why the clinical-path call is a monkeypatched seam, not a Hydra override string

`create_clinical_job`/`run_clinical_job` (`app/backend/clinical_jobs.py`) compose their
OWN Hydra config internally, per job, via the private `_compose_clinical_cfg()` -- with
no parameter this driver could set per study. The one thing this protocol needs that
function to do differently per study is set `cfg.clinical.ingest.role_overrides`
(`series_uid -> role`): RSNA study folders are named `FLAIR` / `T1w` / `T1wCE` / `T2w`,
and the ingest rule table does not recognise the token `"T1wCE"`
(`neurovision.data.dicom_ingest`'s own module docstring), so the RSNA folder name is
the only role signal used here -- stated openly, exactly as the protocol requires, never
folded into the shared `configs/clinical/default.yaml` (which would make the
automatically-measured role assignment everywhere else in the clinical pipeline
meaningless -- see that file's own comment on `role_overrides`).

A plain Hydra CLI override string (`+clinical.ingest.role_overrides.<uid>=<role>`)
cannot express this safely: a DICOM `SeriesInstanceUID` is dot-separated
(`"1.2.840.113619...`), and Hydra's override grammar treats every dot as a nested-key
separator, so the UID would be split into a deep, wrong dict shape instead of one flat
key. `run_clinical_path` instead temporarily replaces `_compose_clinical_cfg` with a
wrapper that calls the ORIGINAL function (so every other setting is composed exactly as
`run_clinical_study.py` composes it) and then mutates the returned `DictConfig`'s
`clinical.ingest.role_overrides` in place, restoring the original function in a
`finally` no matter how the run ends.

## Resumability and disk

Measured at ~6 min/study on CPU (protocol's own estimate) x 40 studies is hours, and a
Mac session can be interrupted -- so every case's result is written ATOMICALLY (temp
file + `os.replace`, see `_write_result_atomic`) to its own
`<work_dir>/cases/<case_id>/result.json` the moment it is scored. A case whose
`result.json` already exists AND parses is skipped on the next run (`process_case`); one
that fails to parse (e.g. corrupted, or -- before this atomic-write discipline existed --
left half-written by a killed process) is treated as absent and re-run, logged at
WARNING, never a crash. Each clinical
job's large, volume-sized artifacts (raw DICOM, ingest NIfTIs, registered volumes,
cached predictions/logits/Grad-CAM -- CLAUDE.md's trap 10, "volume-sized artifacts are
caches, not results") are deleted right after scoring, keeping only `job.json` /
`summary.json` / `report/` / `dicom_seg/`, to bound disk over 40 studies.

## A "done" pipeline outcome and a successful SCORING are two different facts

The gatekeeper's decision (accepted or not) and this module's ability to score the
result against ground truth afterwards can fail independently -- a missing
`data/preprocessed/brats/<case_id>` directory, or a genuine geometry mismatch between
our SRI24 registration and BraTS's own, says nothing about whether the clinical
pipeline itself behaved correctly. So `_run_case` records the pipeline's own outcome
(`status`, `accepted`, `gate_decision`) BEFORE attempting to score, in a separate
`try`/`except` from the scoring step itself -- a scoring failure never overwrites
`status="done"`/`accepted=True`, it only leaves `scored=False`. `summarise()` then
excludes every `scored=False` "done" row from `usable_rate`/`front_end_cost` entirely
(both numerator and denominator, not merely labelled "not usable" -- that would
silently inflate the reported silent-failure rate), reporting how many with a separate
`n_scoring_failed` count. And because a GENUINE geometry mismatch (as opposed to a
missing ground-truth file) would hit every one of the 40 sample cases identically,
`_check_pilot_geometry` inspects the pilot's own row (the pilot always runs first) and
aborts the whole run before repeating that failure 40 times -- see `_score_all_cases`.

## CPU only, every path from config

Nothing here trains or touches a GPU. Every input and output location comes from
`cfg.analysis.real_dicom_validation` (see `configs/analysis/default.yaml`). Running a
REAL study needs `.venv-clinical` (pydicom / dcm2niix / HD-BET live only there -- same
requirement as `scripts/run_clinical_study.py`, see its own module docstring); this
module's tests run in the main `.venv`, with the clinical-path call faked.

Example usage (a real run):

    .venv-clinical/bin/python scripts/validate_real_dicom.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.error_budget import bootstrap_rate_ci
from neurovision.analysis.real_dicom_scoring import (
    lateralisation_check,
    roundtrip_self_test,
    score_case,
    to_reference_grid,
    uncrop_to_full,
)
from neurovision.analysis.statistics import load_per_case, paired_bootstrap_ci, wilcoxon_signed_rank
from neurovision.data.dicom_ingest import read_series_headers
from neurovision.utils.io import ensure_dir, read_json, read_yaml, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on any
# machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# The header this protocol requires on every run -- see the module docstring's "Not
# external validation" note and the spec's "Log header lines say ..." requirement.
_HEADER_NOTE = (
    "test-split BraTS 2021 patients via RSNA DICOM -- front-end isolation, "
    "NOT external validation."
)

# A clinical job's own volume-sized subdirectories -- deleted after each case is
# scored (see the module docstring's "Resumability and disk" section). Never
# "cache"/"prep" alone: `raw_dicom`/`ingest`/`clinical_prep` are equally large
# intermediates this protocol has no further use for once a case is scored.
_LARGE_JOB_SUBDIRS: tuple[str, ...] = ("raw_dicom", "ingest", "clinical_prep", "prep", "cache")


# ---------------------------------------------------------------------------
# Config + case list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealDicomValidationConfig:
    """Resolved `cfg.analysis.real_dicom_validation`.

    Attributes:
        cases_file: The frozen `configs/data/rsna_validation_cases.yaml`.
        zips_dir: Directory of per-study zips, named `<rsna_id>.zip`.
        work_dir: Root output directory for this run.
        research_per_case: The research-path `per_case_metrics.csv` to compare
            the front-end cost against.
        gt_root: Root of the preprocessed BraTS tree holding `<case_id>/label.npy`
            / `meta.json`.
        usable_threshold: Note 47's usable-segmentation bar (WT and TC Dice).
        n_boot: Bootstrap replicate count for every CI in this module.
        seed: Seed for every `np.random.default_rng` this module constructs.
        include_pilot: Whether to also run the pilot case through the full
            clinical pipeline (still always excluded from every statistic).
        role_by_folder: RSNA folder name -> this project's role name, e.g.
            `{"FLAIR": "flair", "T1w": "t1", "T1wCE": "t1ce", "T2w": "t2"}`.
    """

    cases_file: Path
    zips_dir: Path
    work_dir: Path
    research_per_case: Path
    gt_root: Path
    usable_threshold: float
    n_boot: int
    seed: int
    include_pilot: bool
    role_by_folder: dict[str, str]


def _resolve_config(cfg: DictConfig) -> RealDicomValidationConfig:
    """Reads `cfg.analysis.real_dicom_validation` into a `RealDicomValidationConfig`."""
    rd_cfg = cfg.analysis.real_dicom_validation
    return RealDicomValidationConfig(
        cases_file=Path(str(rd_cfg.cases_file)),
        zips_dir=Path(str(rd_cfg.zips_dir)),
        work_dir=Path(str(rd_cfg.work_dir)),
        research_per_case=Path(str(rd_cfg.research_per_case)),
        gt_root=Path(str(rd_cfg.gt_root)),
        usable_threshold=float(rd_cfg.usable_threshold),
        n_boot=int(rd_cfg.n_boot),
        seed=int(rd_cfg.seed),
        include_pilot=bool(rd_cfg.include_pilot),
        role_by_folder={str(k): str(v) for k, v in dict(rd_cfg.role_by_folder).items()},
    )


@dataclass(frozen=True)
class CaseSpec:
    """One case from `configs/data/rsna_validation_cases.yaml`.

    Attributes:
        case_id: BraTS 2021 case id, e.g. `"BraTS2021_00032"`.
        rsna_id: The matching RSNA-MICCAI 5-digit study id, e.g. `"00032"` --
            also the zip's stem, `<rsna_id>.zip`.
        pilot: `True` for the one training-case pilot (`docs/research/
            real_dicom_validation_protocol.md`'s "Pilot" section) -- never
            counted in any reported statistic, only run to check the geometry
            and the scorer.
    """

    case_id: str
    rsna_id: str
    pilot: bool


def load_case_specs(cases_file: Path) -> list[CaseSpec]:
    """Loads the pilot and the 40 sample cases from the frozen cases file.

    Args:
        cases_file: `configs/data/rsna_validation_cases.yaml`, or an equivalent
            file with `pilot: {case_id, rsna_id}` and `cases: [{case_id,
            rsna_id}, ...]` keys.

    Returns:
        `[pilot_spec, *sample_specs]`, pilot first. `rsna_id` is coerced to a
        zero-padded 5-digit string regardless of how YAML parsed it (some
        entries are quoted strings, some are bare numbers that YAML's int
        resolver happens to leave as strings too -- `.zfill(5)` makes every
        entry's type irrelevant).
    """
    payload = read_yaml(cases_file)
    pilot_entry = payload["pilot"]
    pilot = CaseSpec(
        case_id=str(pilot_entry["case_id"]),
        rsna_id=str(pilot_entry["rsna_id"]).zfill(5),
        pilot=True,
    )
    cases = [
        CaseSpec(case_id=str(entry["case_id"]), rsna_id=str(entry["rsna_id"]).zfill(5), pilot=False)
        for entry in payload["cases"]
    ]
    return [pilot, *cases]


# ---------------------------------------------------------------------------
# Step 1: self-test, before any case is scored
# ---------------------------------------------------------------------------


def run_self_tests(gt_root: Path, pilot: CaseSpec, first_sample: CaseSpec) -> None:
    """Runs `roundtrip_self_test` on the pilot's and the first sample case's ground truth.

    Per the protocol's "Self-test first" rule: a ground-truth round trip must score
    Dice exactly 1.0 on every region before any real study is scored. Deliberately
    does not catch `roundtrip_self_test`'s `RuntimeError` -- letting it propagate is
    what "abort if it fails" means: nothing below this call in `run_validation` may
    ever run.

    Args:
        gt_root: Root of the preprocessed BraTS tree (`<case_id>/label.npy` /
            `meta.json`).
        pilot: The pilot case spec.
        first_sample: The first (index 0) sample case spec.

    Raises:
        RuntimeError: A round trip did not score Dice 1.0 on every region --
            propagated from `roundtrip_self_test`, unmodified.
    """
    for spec in (pilot, first_sample):
        case_dir = gt_root / spec.case_id
        gt_cropped = np.load(case_dir / "label.npy")
        meta = read_json(case_dir / "meta.json")
        roundtrip_self_test(gt_cropped, meta)
        logger.info("validate_real_dicom: round-trip self-test OK for %s.", spec.case_id)


# ---------------------------------------------------------------------------
# Role overrides from RSNA folder names
# ---------------------------------------------------------------------------


def build_role_overrides(rsna_study_dir: Path, role_by_folder: Mapping[str, str]) -> dict[str, str]:
    """Reads one DICOM series' UID per RSNA role folder, mapping series_uid -> our role.

    The RSNA-MICCAI layout puts exactly one series under each of `FLAIR/`, `T1w/`,
    `T1wCE/`, `T2w/`. The RSNA folder NAME is the only role signal used here -- see the
    module docstring's "Why the clinical-path call is a monkeypatched seam" section
    for why this substitutes for automatic role assignment only for this protocol's
    RSNA-named folders, and adds no new token to the ingest's own vocabulary.

    Args:
        rsna_study_dir: One study's own directory (holding `FLAIR/`, `T1w/`,
            `T1wCE/`, `T2w/` subfolders).
        role_by_folder: RSNA folder name -> role name, e.g. `{"FLAIR": "flair", ...}`.

    Returns:
        `series_uid -> role`, one entry per folder in `role_by_folder`.

    Raises:
        FileNotFoundError: A configured folder does not exist under `rsna_study_dir`.
        ValueError: A folder does not contain exactly one DICOM series.
    """
    overrides: dict[str, str] = {}
    for folder_name, role in role_by_folder.items():
        folder = rsna_study_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"build_role_overrides: no {folder_name!r} folder at {folder}.")
        headers = read_series_headers(folder)
        if len(headers) != 1:
            raise ValueError(
                f"build_role_overrides: expected exactly one DICOM series under {folder} "
                f"(RSNA-MICCAI layout: one series per role folder), found {len(headers)}."
            )
        overrides[headers[0].series_uid] = role
    return overrides


def run_clinical_path(
    rsna_study_dir: Path, jobs_out_dir: Path, role_overrides: Mapping[str, str]
) -> Path:
    """Runs the real clinical pipeline on one study directory, with `role_overrides` applied.

    Reuses `scripts.run_clinical_study.run_study` unmodified -- the same synchronous
    driver `/clinical` and T0.2 already use. See the module docstring for why
    `role_overrides` is applied by temporarily replacing `app.backend.clinical_jobs
    ._compose_clinical_cfg`, restored in a `finally`, rather than by a Hydra override
    string or a change to `configs/clinical/default.yaml`.

    Args:
        rsna_study_dir: One study's own DICOM folder, zipped in memory by `run_study`
            itself (`run_study` -> `zip_directory`), exactly as for any other study.
        jobs_out_dir: Where the clinical job store lives for this run
            (`cfg.analysis.real_dicom_validation.work_dir/jobs`).
        role_overrides: `series_uid -> role`, from `build_role_overrides`.

    Returns:
        Path to the written `<jobs_out_dir>/<job_id>/summary.json`.
    """
    # Lazy: these two modules reach app.backend / torch, exactly the "lazy imports"
    # discipline scripts/run_clinical_study.py's own module docstring states, so a
    # test that only exercises this module's pure aggregation functions never pays
    # for it.
    import scripts.run_clinical_study as run_clinical_study_script
    from app.backend import clinical_jobs as clinical_jobs_module

    original_compose = clinical_jobs_module._compose_clinical_cfg

    def _compose_with_role_overrides() -> Any:
        cfg = original_compose()
        OmegaConf.set_struct(cfg, False)
        cfg.clinical.ingest.role_overrides = dict(role_overrides)
        OmegaConf.set_struct(cfg, True)
        return cfg

    clinical_jobs_module._compose_clinical_cfg = _compose_with_role_overrides
    try:
        return run_clinical_study_script.run_study(rsna_study_dir, jobs_out_dir)
    finally:
        clinical_jobs_module._compose_clinical_cfg = original_compose


def run_clinical_path_for_case(
    rsna_study_dir: Path, jobs_out_dir: Path, role_by_folder: Mapping[str, str]
) -> Path:
    """`build_role_overrides` then `run_clinical_path` -- the one seam `_run_case` calls.

    Kept as its own function (rather than inlined in `_run_case`) so a test can
    monkeypatch exactly this call with a fake that returns a synthetic job directory's
    `summary.json`, bypassing real DICOM/Hydra/torch entirely while still exercising
    `_run_case`'s own uncrop/scoring logic on whatever that fake wrote to disk.

    Args:
        rsna_study_dir: One study's own DICOM folder.
        jobs_out_dir: Where the clinical job store lives for this run.
        role_by_folder: RSNA folder name -> role name.

    Returns:
        Path to the written `<jobs_out_dir>/<job_id>/summary.json`.
    """
    role_overrides = build_role_overrides(rsna_study_dir, role_by_folder)
    logger.info(
        "validate_real_dicom: role_overrides for %s (RSNA folder names only, no ingest "
        "vocabulary token added): %s",
        rsna_study_dir,
        role_overrides,
    )
    return run_clinical_path(rsna_study_dir, jobs_out_dir, role_overrides)


def _unzip_study(zip_path: Path, extract_root: Path, rsna_id: str) -> Path:
    """Extracts `<rsna_id>.zip` into `extract_root`, returning its `<rsna_id>/` study dir.

    Args:
        zip_path: The study's zip file.
        extract_root: Directory to extract into (created if missing).
        rsna_id: The RSNA study id -- also the zip's expected top-level folder name
            (RSNA-MICCAI layout: `<rsna_id>/{FLAIR,T1w,T1wCE,T2w}/*.dcm`).

    Returns:
        `extract_root / rsna_id`.

    Raises:
        FileNotFoundError: The zip does not contain a top-level `<rsna_id>/` folder.
    """
    ensure_dir(extract_root)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)
    study_dir = extract_root / rsna_id
    if not study_dir.is_dir():
        raise FileNotFoundError(
            f"_unzip_study: expected {zip_path} to contain a top-level {rsna_id!r} folder "
            f"(the RSNA-MICCAI layout this protocol assumes); found no {study_dir}."
        )
    return study_dir


def _cleanup_job_dir(job_dir: Path) -> None:
    """Deletes a clinical job's volume-sized subdirectories, best-effort.

    Keeps `job.json`, `summary.json`, `report/`, `dicom_seg/` -- everything small
    enough to be worth inspecting after the fact. See the module docstring's
    "Resumability and disk" section.
    """
    for name in _LARGE_JOB_SUBDIRS:
        shutil.rmtree(job_dir / name, ignore_errors=True)


# ---------------------------------------------------------------------------
# Step 2: one case, resumable
# ---------------------------------------------------------------------------


def _blank_row(spec: CaseSpec) -> dict[str, Any]:
    """The default `result.json` row for one case, before anything has run."""
    return {
        "case_id": spec.case_id,
        "rsna_id": spec.rsna_id,
        "pilot": spec.pilot,
        "status": None,
        "refusing_stage": None,
        "refusing_reason": None,
        "gate_decision": None,
        "accepted": False,
        "runtime_s": None,
        "dice_ET": float("nan"),
        "dice_TC": float("nan"),
        "dice_WT": float("nan"),
        "hd95_ET": float("nan"),
        "hd95_TC": float("nan"),
        "hd95_WT": float("nan"),
        "lateralisation": None,
        "mirrored": False,
        # True only once score_case AND lateralisation_check have BOTH returned
        # without raising -- see the "done, but scoring failed" carve-out in
        # _run_case and summarise() below. A "done" row can have scored=False;
        # its usability is UNKNOWN, not "not usable".
        "scored": False,
        # Set only when `to_reference_grid` itself raised a ValueError (a real
        # geometry mismatch, not a flip/permute) -- see _check_pilot_geometry.
        "geometry_mismatch": None,
        "error": None,
    }


def _run_case(
    spec: CaseSpec,
    ccfg: RealDicomValidationConfig,
    run_clinical_path_fn: Callable[
        [Path, Path, Mapping[str, str]], Path
    ] = run_clinical_path_for_case,
) -> dict[str, Any]:
    """Runs one case end to end: unzip -> clinical path -> uncrop -> score -> lateralisation.

    Never raises for a case-level outcome: any exception is caught, logged with a full
    traceback, and recorded in the returned row's `error` field (with `status="failed"`
    if no status was recorded yet) -- one bad case must never abort the whole 40-case
    run. Always deletes the unzipped DICOM and the job's volume-sized intermediates
    before returning (`finally`), whether the case succeeded or not.

    Args:
        spec: The case to run.
        ccfg: The resolved validation config.
        run_clinical_path_fn: `(rsna_study_dir, jobs_out_dir, role_by_folder) ->
            summary_path` -- defaults to `run_clinical_path_for_case` (the real
            clinical pipeline). A test replaces this with a fake that writes a
            synthetic job directory and returns its `summary.json` path, so this
            function's own uncrop/scoring logic still runs against real files.

    Returns:
        One row, shaped like `_blank_row`'s keys, `dice_*`/`hd95_*` filled in and
        `lateralisation`/`mirrored` set only when `status == "done"`.
    """
    start = time.monotonic()
    extract_root = ccfg.work_dir / "_extract" / spec.case_id
    jobs_out_dir = ccfg.work_dir / "jobs"
    row = _blank_row(spec)
    job_dir: Path | None = None

    try:
        zip_path = ccfg.zips_dir / f"{spec.rsna_id}.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(f"_run_case: no RSNA study zip at {zip_path}.")
        rsna_study_dir = _unzip_study(zip_path, extract_root, spec.rsna_id)

        summary_path = run_clinical_path_fn(rsna_study_dir, jobs_out_dir, ccfg.role_by_folder)
        job_dir = summary_path.parent
        summary = read_json(summary_path)

        row["status"] = summary.get("state")
        row["runtime_s"] = summary.get("wall_s")
        if row["status"] in ("refused", "failed"):
            row["refusing_stage"] = summary.get("stage")
            row["refusing_reason"] = summary.get("error")
        gate = summary.get("gatekeeper_decision")
        row["gate_decision"] = gate.get("decision") if gate else None
        # "accepted" = PROCEED or PROCEED_WITH_CAUTION (matches
        # neurovision.analysis.error_budget.ACCEPTED_DECISIONS): run_clinical_job
        # only ever reaches state="done" past the gatekeeper's own REFUSE check
        # (app/backend/clinical_jobs.py::run_clinical_job), so state=="done" IS
        # "accepted" exactly, with no need to re-inspect gate_decision here.
        row["accepted"] = row["status"] == "done"

        if row["status"] == "done":
            # A SEPARATE try/except from the outer one: the pipeline's own
            # outcome (status="done", accepted=True, gate_decision) is already
            # recorded above and must stay truthful even if SCORING itself then
            # fails (missing ground truth, a genuine geometry mismatch, a shape
            # mismatch, ...) -- it is the gate that decided to accept this case,
            # not our ability to score it afterwards. `scored` (default False,
            # set True only at the very end of this block) is what distinguishes
            # "done and scored" from "done but unscored" downstream, in
            # summarise()'s usable-rate/front-end-cost carve-out.
            try:
                # Lazy: reused ONLY for their path-resolution contract
                # (clinical_segmentation_settings / cached_prediction_path), never
                # re-derived here -- see app/backend/clinical_jobs.py and
                # app/backend/inference.py.
                from app.backend import clinical_jobs, inference

                job_case_id = str(summary["case_id"])
                clinical_settings = clinical_jobs.clinical_segmentation_settings(
                    job_dir / "prep", job_dir / "cache"
                )
                pred_path = inference.cached_prediction_path(clinical_settings, job_case_id)
                pred_cropped = np.load(pred_path)
                job_meta = read_json(clinical_settings.prep_dir / job_case_id / "meta.json")
                pred_full = uncrop_to_full(pred_cropped, job_meta)

                gt_dir = ccfg.gt_root / spec.case_id
                gt_cropped = np.load(gt_dir / "label.npy")
                gt_meta = read_json(gt_dir / "meta.json")
                gt_full = uncrop_to_full(gt_cropped, gt_meta)

                # The GT meta's affine is the reference -- the protocol's own "Grid"
                # rule. Only the clinical prediction is mapped onto it.
                reference_affine = np.asarray(gt_meta["affine"], dtype=np.float64)
                pred_affine = np.asarray(job_meta["affine"], dtype=np.float64)
                try:
                    pred_on_ref = to_reference_grid(pred_full, pred_affine, reference_affine)
                except ValueError as exc:
                    # Recorded separately from a generic scoring failure: a REAL
                    # geometry mismatch here (not a flip/permute) would hit every
                    # sample case identically (our SRI24 registration vs. BraTS's
                    # own), so `_check_pilot_geometry` inspects this field on the
                    # pilot's row (run first) to abort the whole 40-case run
                    # before it repeats the same failure 40 times.
                    row["geometry_mismatch"] = str(exc)
                    raise

                spacing = tuple(float(s) for s in gt_meta["spacing"])
                row.update(score_case(pred_on_ref, gt_full, spacing))

                lateral = lateralisation_check(pred_on_ref, gt_full, reference_affine)
                row["lateralisation"] = lateral
                row["mirrored"] = lateral == "mirrored"
                if row["mirrored"]:
                    # ERROR, not WARNING: per the protocol, "mirrored" is never
                    # silently dropped -- it must be visible in the log, not just
                    # the per_case.csv column a reader might not open.
                    logger.error(
                        "validate_real_dicom: case %s scored 'mirrored' by "
                        "lateralisation_check -- flagged in the row, kept in every "
                        "table.",
                        spec.case_id,
                    )
                row["scored"] = True
            except Exception as exc:  # noqa: BLE001 - a done case's failure to SCORE
                # must never overwrite its (already-recorded, truthful) pipeline
                # outcome -- see this block's own opening comment.
                logger.error(
                    "validate_real_dicom: case %s reached 'done' but scoring failed "
                    "(excluded from usable-rate stats and front-end cost, never "
                    "counted as a silent failure)",
                    spec.case_id,
                    exc_info=True,
                )
                row["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - one case's failure must never abort the run
        logger.error("validate_real_dicom: case %s errored", spec.case_id, exc_info=True)
        if row["status"] is None:
            row["status"] = "failed"
        row["error"] = str(exc)
    finally:
        if row["runtime_s"] is None:
            row["runtime_s"] = round(time.monotonic() - start, 1)
        shutil.rmtree(extract_root, ignore_errors=True)
        if job_dir is not None:
            _cleanup_job_dir(job_dir)

    return row


def result_path(work_dir: Path, case_id: str) -> Path:
    """Where one case's `result.json` lives, and the file `process_case` resumes from."""
    return work_dir / "cases" / case_id / "result.json"


def _write_result_atomic(row: dict[str, Any], result_file: Path) -> None:
    """Writes one case's `result.json` atomically (temp file, then `os.replace`).

    Mirrors `app.backend.clinical_jobs._persist_clinical_job`'s own tmp-then-replace
    pattern. CLAUDE.md's hard constraint 1 requires every long-running script here to
    "survive a session kill at any point" -- a Mac process killed mid-write must never
    leave a half-written `result.json` that a LATER, resumed run would then read as a
    (corrupt) completed case. Written to `<result_file>.tmp` first; `os.replace` is
    atomic on both macOS and Linux, so a reader only ever sees the old file or the new
    one, never a partial one.

    Args:
        row: The case's row (JSON-serialisable).
        result_file: Destination, e.g. `result_path(work_dir, case_id)`.
    """
    ensure_dir(result_file.parent)
    tmp_path = result_file.parent / f"{result_file.name}.tmp"
    tmp_path.write_text(json.dumps(row, indent=2, default=str))
    os.replace(tmp_path, result_file)


def process_case(
    spec: CaseSpec,
    ccfg: RealDicomValidationConfig,
    run_case_fn: Callable[[CaseSpec, RealDicomValidationConfig], dict[str, Any]] = _run_case,
) -> dict[str, Any]:
    """Runs one case, or loads its already-written `result.json` (resumable).

    An existing `result.json` that fails to parse (e.g. left half-written by a killed
    process, before this function wrote atomically -- or simply corrupted) is treated
    as ABSENT, not as a crash: logged at WARNING and the case is re-run, exactly as if
    `result.json` had never existed.

    Args:
        spec: The case to run.
        ccfg: The resolved validation config.
        run_case_fn: `(spec, ccfg) -> row` -- defaults to `_run_case`. A test
            replaces this to assert it is never called when `result.json` already
            exists and parses.

    Returns:
        The case's row (freshly computed, or loaded from disk).
    """
    result_file = result_path(ccfg.work_dir, spec.case_id)
    if result_file.is_file():
        try:
            row = read_json(result_file)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "validate_real_dicom: case %s has an unparseable result.json at %s (%s); "
                "treating it as absent and re-running.",
                spec.case_id,
                result_file,
                exc,
            )
        else:
            logger.info(
                "validate_real_dicom: case %s already scored (%s); skipping.",
                spec.case_id,
                result_file,
            )
            return row

    row = run_case_fn(spec, ccfg)
    _write_result_atomic(row, result_file)
    return row


# ---------------------------------------------------------------------------
# Step 3: aggregation -- pilot excluded from every statistic
# ---------------------------------------------------------------------------


def label_usable(per_case: pd.DataFrame, *, threshold: float) -> pd.Series:
    """Note 47's usable-segmentation label: `dice_WT >= threshold AND dice_TC >= threshold`.

    Args:
        per_case: A frame with `dice_WT` and `dice_TC` columns.
        threshold: The Dice bar, e.g. `0.7`.

    Returns:
        A boolean `Series`, aligned to `per_case.index`. NaN in either column
        compares `False` under `>=` on both sides, so an unscored case is
        correctly labelled not-usable without any extra handling.
    """
    wt = pd.to_numeric(per_case["dice_WT"], errors="coerce")
    tc = pd.to_numeric(per_case["dice_TC"], errors="coerce")
    return (wt >= threshold) & (tc >= threshold)


def completion_counts(scored: pd.DataFrame) -> dict[str, Any]:
    """Completion fractions: by status, and by `(status, refusing_stage)` for the rest.

    Args:
        scored: Non-pilot rows, with `status` and `refusing_stage` columns.

    Returns:
        `{"by_status": {status: n}, "by_status_and_stage": [{"status", "refusing_stage",
        "n"}, ...]}`. A refused or failed study is counted here exactly like any other
        row -- nothing is excluded (the protocol's "nothing is excluded after
        download" rule).
    """
    by_status = {str(k): int(v) for k, v in scored["status"].value_counts().items()}

    non_done = scored[scored["status"].isin(["refused", "failed"])]
    stage_counts = (
        non_done.groupby(["status", "refusing_stage"], dropna=False).size().reset_index(name="n")
    )
    by_stage = [
        {
            "status": str(r["status"]),
            "refusing_stage": (None if pd.isna(r["refusing_stage"]) else str(r["refusing_stage"])),
            "n": int(r["n"]),
        }
        for _, r in stage_counts.iterrows()
    ]
    return {"by_status": by_status, "by_status_and_stage": by_stage}


def usable_rate_stats(
    scored: pd.DataFrame, *, n_boot: int, seed: int, ci: float = 0.95
) -> dict[str, Any]:
    """The protocol's usable-rate-end-to-end endpoint, with percentile bootstrap CIs.

    Args:
        scored: Non-pilot rows, with boolean `accepted` and `usable` columns.
        n_boot: Bootstrap replicate count.
        seed: Seed for the one `np.random.default_rng` shared by all three CIs here
            (consumed sequentially, deterministic given `seed`).
        ci: Confidence level, e.g. `0.95`.

    Returns:
        `n`, `n_accepted`, and `(point, lo, hi)` for `p_accepted_and_usable`,
        `p_usable_given_accepted` (over the `accepted` subset only -- `nan` if
        nothing was accepted, matching `bootstrap_rate_ci`'s own empty-input
        behaviour) and `p_silent_failure` (accepted AND unusable -- the number that
        matters).
    """
    generator = np.random.default_rng(seed)
    accepted = scored["accepted"].astype(bool).to_numpy()
    usable = scored["usable"].astype(bool).to_numpy()

    accepted_and_usable = (accepted & usable).astype(float)
    accepted_and_unusable = (accepted & ~usable).astype(float)  # silent failure
    usable_given_accepted = usable[accepted].astype(float)

    p_au, au_lo, au_hi = bootstrap_rate_ci(
        accepted_and_usable, n_boot=n_boot, ci=ci, generator=generator
    )
    p_sf, sf_lo, sf_hi = bootstrap_rate_ci(
        accepted_and_unusable, n_boot=n_boot, ci=ci, generator=generator
    )
    p_uga, uga_lo, uga_hi = bootstrap_rate_ci(
        usable_given_accepted, n_boot=n_boot, ci=ci, generator=generator
    )

    return {
        "n": int(len(scored)),
        "n_accepted": int(accepted.sum()),
        "p_accepted_and_usable": p_au,
        "p_accepted_and_usable_lo": au_lo,
        "p_accepted_and_usable_hi": au_hi,
        "p_usable_given_accepted": p_uga,
        "p_usable_given_accepted_lo": uga_lo,
        "p_usable_given_accepted_hi": uga_hi,
        "p_silent_failure": p_sf,
        "p_silent_failure_lo": sf_lo,
        "p_silent_failure_hi": sf_hi,
    }


def front_end_cost(
    done: pd.DataFrame,
    research: pd.DataFrame,
    *,
    regions: Sequence[str] = ("ET", "TC", "WT"),
    n_boot: int,
    seed: int,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Per-region front-end cost: paired `clinical - research` Dice, over completed studies.

    Args:
        done: Non-pilot rows with `status == "done"`, holding `case_id` and
            `dice_<R>` columns (the CLINICAL-path Dice, from `score_case`).
        research: The research-path `per_case_metrics.csv`, with `case_id` and
            `dice_<R>` columns.
        regions: Regions to report, in this order.
        n_boot: Bootstrap replicate count.
        seed: Seed for the one `np.random.default_rng` shared across regions and
            statistics (consumed sequentially).
        ci: Confidence level.

    Returns:
        One row per region: `n` (paired case count), `mean_diff`/`median_diff`
        (+`_lo`/`_hi`, `clinical - research`, descriptive, not Holm-corrected --
        the protocol's own "reported once" rule) and `wilcoxon_statistic`/
        `wilcoxon_pvalue`. A region with fewer than 2 paired cases is skipped
        (logged), since no CI or test is meaningful on that few.
    """
    dice_cols = [f"dice_{r}" for r in regions]
    merged = done[["case_id", *dice_cols]].merge(
        research[["case_id", *dice_cols]],
        on="case_id",
        how="inner",
        suffixes=("_clinical", "_research"),
    )

    dropped = sorted(set(done["case_id"]) - set(merged["case_id"]))
    if dropped:
        logger.warning(
            "front_end_cost: %d done case(s) have no matching row in the research-path "
            "per_case_metrics.csv and were dropped from the paired comparison: %s",
            len(dropped),
            dropped,
        )

    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for region in regions:
        clinical = merged[f"dice_{region}_clinical"].to_numpy(dtype=float)
        research_vals = merged[f"dice_{region}_research"].to_numpy(dtype=float)
        if len(clinical) < 2:
            logger.warning(
                "front_end_cost: fewer than 2 paired case(s) for region %s; skipping.", region
            )
            continue

        mean_boot = paired_bootstrap_ci(
            clinical, research_vals, n_boot=n_boot, ci=ci, generator=generator, statistic="mean"
        )
        median_boot = paired_bootstrap_ci(
            clinical, research_vals, n_boot=n_boot, ci=ci, generator=generator, statistic="median"
        )
        wil = wilcoxon_signed_rank(clinical, research_vals)

        rows.append(
            {
                "region": region,
                "n": int(len(clinical)),
                "mean_diff": mean_boot.point,
                "mean_diff_lo": mean_boot.lo,
                "mean_diff_hi": mean_boot.hi,
                "median_diff": median_boot.point,
                "median_diff_lo": median_boot.lo,
                "median_diff_hi": median_boot.hi,
                "wilcoxon_statistic": wil.statistic,
                "wilcoxon_pvalue": wil.pvalue,
            }
        )
    return pd.DataFrame(rows)


def summarise(per_case: pd.DataFrame, ccfg: RealDicomValidationConfig) -> dict[str, Any]:
    """The protocol's endpoints, computed over every row EXCEPT the pilot.

    A "done" row that reached the clinical pipeline but could not be SCORED
    (`scored=False` -- missing ground truth, a geometry mismatch, any exception
    inside `_run_case`'s scoring block) has an UNKNOWN outcome, not a known-unusable
    one: it is excluded entirely (both numerator and denominator) from
    `usable_rate`/`front_end_cost`, never counted as a silent failure just because its
    Dice happens to be NaN. `completion` still counts it under `status="done"` -- the
    pipeline itself really did complete for that case.

    Args:
        per_case: Every case's row, including the pilot (`pilot` column).
        ccfg: The resolved validation config.

    Returns:
        `{"note", "n", "usable_threshold", "n_boot", "seed", "n_scoring_failed",
        "completion", "usable_rate", "front_end_cost"}`. `front_end_cost` is `[]` if
        `ccfg.research_per_case` does not exist, or no case is both `"done"` and
        `scored`.

    Raises:
        ValueError: `per_case` has no non-pilot row.
    """
    non_pilot = per_case[~per_case["pilot"].astype(bool)].copy()
    if non_pilot.empty:
        raise ValueError("summarise: per_case has no non-pilot row to summarise.")
    non_pilot["usable"] = label_usable(non_pilot, threshold=ccfg.usable_threshold)

    scoring_failed_mask = (non_pilot["status"] == "done") & (~non_pilot["scored"].astype(bool))
    n_scoring_failed = int(scoring_failed_mask.sum())
    if n_scoring_failed:
        logger.warning(
            "validate_real_dicom: %d 'done' case(s) reached the clinical pipeline but could "
            "not be scored -- excluded from usable-rate stats and front-end cost (never "
            "counted as a silent failure); see each case's own ERROR log line for its "
            "exception.",
            n_scoring_failed,
        )
    usable_rate_input = non_pilot[~scoring_failed_mask]

    summary: dict[str, Any] = {
        "note": _HEADER_NOTE,
        "n": int(len(non_pilot)),
        "usable_threshold": ccfg.usable_threshold,
        "n_boot": ccfg.n_boot,
        "seed": ccfg.seed,
        "n_scoring_failed": n_scoring_failed,
        "completion": completion_counts(non_pilot),
        "usable_rate": usable_rate_stats(usable_rate_input, n_boot=ccfg.n_boot, seed=ccfg.seed),
    }

    done_scored = non_pilot[(non_pilot["status"] == "done") & (non_pilot["scored"].astype(bool))]
    if ccfg.research_per_case.is_file() and not done_scored.empty:
        research = load_per_case(ccfg.research_per_case).reset_index()
        cost_table = front_end_cost(
            done_scored, research, regions=("ET", "TC", "WT"), n_boot=ccfg.n_boot, seed=ccfg.seed
        )
        summary["front_end_cost"] = cost_table.to_dict(orient="records")
    else:
        logger.warning(
            "validate_real_dicom: skipping front-end cost (research_per_case=%s exists=%s, "
            "%d done+scored case(s)).",
            ccfg.research_per_case,
            ccfg.research_per_case.is_file(),
            len(done_scored),
        )
        summary["front_end_cost"] = []

    return summary


# ---------------------------------------------------------------------------
# Geometry preflight -- abort before the 40-case run, not after
# ---------------------------------------------------------------------------


def _check_pilot_geometry(spec: CaseSpec, row: Mapping[str, Any]) -> None:
    """Aborts the whole run if the PILOT's clinical prediction failed `to_reference_grid`.

    A genuine geometry mismatch there (our SRI24 registration disagreeing with BraTS's
    own by more than a flip/permute) is systematic, not case-specific -- it would hit
    every one of the 40 sample cases identically. The pilot runs first (`run_validation`
    always orders it first when `include_pilot`), so checking it alone catches this
    before ~4 h is spent repeating the same failure on the sample cases.

    Args:
        spec: The case `row` belongs to. A no-op unless `spec.pilot`.
        row: The case's row, from `_run_case` / `process_case`.

    Raises:
        RuntimeError: `spec.pilot` is `True` and `row["geometry_mismatch"]` is set.
    """
    if not spec.pilot:
        return
    mismatch = row.get("geometry_mismatch")
    if mismatch:
        raise RuntimeError(
            f"validate_real_dicom: pilot case {spec.case_id} failed to_reference_grid with "
            f"a geometry mismatch: {mismatch}. This would hit every sample case identically "
            "(a scoring-grid mismatch between our SRI24 registration and BraTS's own) -- "
            "aborting before scoring the sample cases. Fix the registration/reference-grid "
            "mismatch, then re-run (already-scored sample cases, if any, resume from their "
            "own result.json)."
        )


def _score_all_cases(
    ordered_specs: Sequence[CaseSpec],
    ccfg: RealDicomValidationConfig,
    run_case_fn: Callable[[CaseSpec, RealDicomValidationConfig], dict[str, Any]] = _run_case,
) -> list[dict[str, Any]]:
    """`process_case` over every spec in order, aborting early on a pilot geometry mismatch.

    Args:
        ordered_specs: Cases to run. When the pilot is included, `run_validation`
            always places it first -- this function itself does not reorder anything.
        ccfg: The resolved validation config.
        run_case_fn: Forwarded to `process_case`; see its own docstring.

    Returns:
        One row per spec in `ordered_specs`, in order -- UNLESS a pilot geometry
        mismatch aborts the run first, in which case only the rows computed so far
        are lost (nothing is returned) but each one's `result.json` (already written
        by `process_case`) survives on disk for the next, resumed run.

    Raises:
        RuntimeError: The pilot's row has a `geometry_mismatch` -- see
            `_check_pilot_geometry`.
    """
    rows: list[dict[str, Any]] = []
    for spec in ordered_specs:
        row = process_case(spec, ccfg, run_case_fn=run_case_fn)
        rows.append(row)
        _check_pilot_geometry(spec, row)
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_validation(cfg: DictConfig) -> dict[str, Path]:
    """Runs the full P1.2 protocol: self-test -> every case (resumable) -> aggregate.

    Args:
        cfg: The full composed Hydra config. Reads `cfg.analysis.real_dicom_validation`.

    Returns:
        `{"per_case_csv": Path, "summary_json": Path}`.

    Raises:
        RuntimeError: The self-test (`run_self_tests`) failed -- nothing is scored --
            or the pilot's own clinical prediction failed `to_reference_grid` with a
            genuine geometry mismatch (`_check_pilot_geometry`) -- nothing beyond the
            pilot is scored.
        ValueError: `cases_file` has zero sample cases.
    """
    ccfg = _resolve_config(cfg)
    ensure_dir(ccfg.work_dir)
    logger.info("validate_real_dicom: %s", _HEADER_NOTE)

    specs = load_case_specs(ccfg.cases_file)
    pilot_spec = next(s for s in specs if s.pilot)
    sample_specs = [s for s in specs if not s.pilot]
    if not sample_specs:
        raise ValueError(f"validate_real_dicom: {ccfg.cases_file} has no sample cases.")

    run_self_tests(ccfg.gt_root, pilot_spec, sample_specs[0])
    logger.info("validate_real_dicom: self-tests passed; scoring %d case(s).", len(sample_specs))

    if not ccfg.include_pilot:
        logger.info(
            "validate_real_dicom: include_pilot=False -- skipping the pilot geometry "
            "preflight (no clinical prediction is produced for the pilot this run)."
        )
    ordered_specs = ([pilot_spec] if ccfg.include_pilot else []) + sample_specs
    rows = _score_all_cases(ordered_specs, ccfg)

    per_case = pd.DataFrame(rows)
    per_case_path = ccfg.work_dir / "per_case.csv"
    per_case.to_csv(per_case_path, index=False)
    logger.info("validate_real_dicom: wrote %s (%d row(s)).", per_case_path, len(per_case))

    summary = summarise(per_case, ccfg)
    summary_path = ccfg.work_dir / "summary.json"
    write_json(summary, summary_path)
    logger.info("validate_real_dicom: wrote %s.", summary_path)

    return {"per_case_csv": per_case_path, "summary_json": summary_path}


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the P1.2 real-DICOM validation protocol, per the composed config."""
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_validation(cfg)


if __name__ == "__main__":
    main()
