"""Hydra entry point for W3.4, real-DICOM v2 (carried reference + overridden thick-slice refusals).

Binding design: `docs/research_docs/protocols/real_dicom_validation_protocol.md`, section
"Amendment 1" and its erratum. Read that first. This is the DRIVER (files, the real clinical
pipeline, registration); the statistics live in `neurovision.analysis.real_dicom_v2_endpoints`
and the carried-reference maths in `neurovision.analysis.carried_reference`. The frozen P1.2
driver `scripts/validate_real_dicom.py` is imported from, never edited.

**NOT external validation.** Every study is a BraTS 2021 test-split patient re-run through its
original DICOM. This measures what the clinical front end costs and whether the thick-slice
refusal was right, nothing about other cohorts.

## What a run does, per study (90 studies, one at a time)

1. DEPLOYED run: the real clinical path with deployed settings (role overrides from the RSNA
   folder names, exactly as P1.2). The outcome (refused / accepted) is what every end-to-end
   endpoint uses.
2. If that run was refused at INPUT QC for spacing or anisotropy ("thick-slice refusal"), a
   SECOND, OVERRIDE run of the same study with only `spacing_max_mm` and
   `anisotropy_refuse_ratio` raised so they cannot fire. It answers only "was refusing right?".
3. Each run that produced a mask is scored two ways: against BraTS's own atlas-space label
   (as P1.2), and against the label CARRIED into our atlas space (ANTs rigid registration of
   BraTS's T1CE onto our T1CE). Carry QC (brain-mask Dice) screens registration failures only;
   it is never the left-right check (repo trap 3, `lateralisation_check` stays as in P1.2).
4. Bridge: for the 40 P1.2 studies with a stored mask, the re-run mask must reproduce the stored
   one (WT Dice >= `bridge_min_dice`).

## Files kept per run (repo trap 10: volume-sized artifacts are caches)

A clinical job directory is ~700 MB. After each run only these survive, in
`<out_dir>/cases/<case_id>/<deployed|override>/` (about 8-15 MB):

- `atlas_space_mask.nii.gz`  our predicted label, on the atlas grid
- `t1ce.nii.gz`              skull-stripped atlas-space T1CE (`<job>/clinical_prep/t1ce.nii.gz`)
- `brain_mask.nii.gz`        HD-BET brain mask (`<job>/clinical_prep/brain_mask.nii.gz`)
- `job.json`, `summary.json` job state (carries the structured input-QC report)
- `result.json`              this driver's scored row (written atomically; the resume marker)

## Thick-slice detection is structural, not textual

`job.json["input_qc_pre"]["findings"]` is a list of `{"check", "severity", ...}`. A run is a
thick-slice refusal iff the job state is `refused` and a finding whose `check` is EXACTLY
`"spacing_range"` or `"anisotropy"` has severity `"refuse"`. No substring match on prose.

## Resumability

Every run writes its own `result.json` atomically (temp file + `os.replace`). A parsed
`result.json` is skipped on the next invocation; an unparseable one is re-run. One study at a
time (repo rule: one heavy local job).

## Stages

`analysis.real_dicom_v2.stage` is `run`, `summarise` or `all` (default). Real run, in
`.venv-clinical` (ants, HD-BET and dcm2niix live only there; this module still IMPORTS in the
main `.venv`, which is how its tests run):

    .venv-clinical/bin/python scripts/real_dicom_v2.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Same trick as scripts/validate_real_dicom.py: `python scripts/real_dicom_v2.py` puts
# scripts/ (not the repo root) on sys.path, and `import scripts.validate_real_dicom` needs
# the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.validate_real_dicom as p12  # noqa: E402 - needs the sys.path line above

from neurovision.analysis.carried_reference import (  # noqa: E402
    CarryResult,
    brain_overlap_dice,
    brats_reference_on_grid,
    carry_label,
    score_against_carried,
)
from neurovision.analysis.real_dicom_scoring import (  # noqa: E402
    lateralisation_check,
    score_case,
    to_reference_grid,
    uncrop_to_full,
)
from neurovision.analysis.real_dicom_v2_endpoints import (  # noqa: E402
    REQUIRED_COLUMNS,
    bridge_check,
    e1_front_end_cost,
    e2_refusal_check,
    e3_table,
    e4_caution_variant,
)
from neurovision.inference.input_qc import Severity  # noqa: E402
from neurovision.utils.io import ensure_dir, read_json, read_yaml, write_json  # noqa: E402
from neurovision.utils.logging import setup_logging  # noqa: E402
from neurovision.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(_REPO_ROOT / "configs")

_HEADER_NOTE = (
    "test-split BraTS 2021 patients via RSNA DICOM -- carried-reference front-end cost and "
    "thick-slice refusal check, NOT external validation."
)

DEPLOYED = "deployed"
OVERRIDE = "override"

# The two input-QC check ids that mean "thick slices". Compared with `==` on the structured
# `check` field -- never `in` on prose (a short token matches a longer word, repo trap).
THICK_SLICE_CHECKS = frozenset({"spacing_range", "anisotropy"})

# The ONLY two clinical settings the override run may change (Amendment 1, (b)).
_OVERRIDABLE_QC_KEYS = frozenset({"spacing_max_mm", "anisotropy_refuse_ratio"})

# Everything a run keeps on disk (see the module docstring). `result.json` is added by the
# resume logic. Anything else in a job directory is deleted.
KEPT_MASK = "atlas_space_mask.nii.gz"
KEPT_T1CE = "t1ce.nii.gz"
KEPT_BRAIN = "brain_mask.nii.gz"
KEPT_JOB_FILES = ("job.json", "summary.json")
KEPT_FILES = (KEPT_MASK, KEPT_T1CE, KEPT_BRAIN, *KEPT_JOB_FILES)

_REGIONS = ("ET", "TC", "WT")

# A deployed run whose state/stage says a mask was produced: finished, or refused by the
# gatekeeper AFTER segmentation. Compared with `==` (other stage names look similar).
_GATEKEEPER_STAGE = "gatekeeper"

# How many cases the mirrored-mask self-test may try before giving up (a midline tumour is
# symmetric, so one case can be unable to show the flip).
_MIRROR_SELF_TEST_TRIES = 5


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _resolve_path(value: Any) -> Path:
    """A config path as a `Path`; relative paths are resolved against the repo root."""
    path = Path(str(value))
    return path if path.is_absolute() else _REPO_ROOT / path


@dataclass(frozen=True)
class V2Config:
    """Resolved `cfg.analysis.real_dicom_v2` (every path absolute).

    Attributes:
        cases_files: The two frozen case lists (40 P1.2 + 50 new).
        zips_dir: Directory of `<rsna_id>.zip` study zips.
        out_dir: Output root.
        gt_root: Preprocessed BraTS tree (`<case_id>/{image.npy,label.npy,meta.json}`).
        research_per_case: Research-path `per_case_metrics.csv`.
        role_by_folder: RSNA folder name -> role.
        usable_threshold: WT and TC Dice bar for "usable".
        carry_qc_threshold: Carry-QC brain-mask Dice bar.
        e1_margin: E1 equivalence margin.
        bridge_min_dice: Re-run vs stored P1.2 mask WT Dice bar.
        n_boot: Bootstrap replicates.
        seed: Seed for bootstraps.
        carry_seed: ANTs rigid registration seed.
        override: The two input-QC settings raised for the override run.
        p12_job_map: P1.2 `case_id -> job_id` map.
        p12_jobs_dir: P1.2 job store holding the stored masks.
        expected_n_cases: Registered study count (90).
        stage: `run`, `summarise` or `all`.
    """

    cases_files: tuple[Path, ...]
    zips_dir: Path
    out_dir: Path
    gt_root: Path
    research_per_case: Path
    role_by_folder: dict[str, str]
    usable_threshold: float
    carry_qc_threshold: float
    e1_margin: float
    bridge_min_dice: float
    n_boot: int
    seed: int
    carry_seed: int
    override: dict[str, float]
    p12_job_map: Path
    p12_jobs_dir: Path
    expected_n_cases: int
    stage: str


def resolve_config(cfg: DictConfig) -> V2Config:
    """Reads `cfg.analysis.real_dicom_v2` into a `V2Config`.

    Raises:
        ValueError: `stage` is not one of run / summarise / all, or `override` names a key
            other than the two registered ones.
    """
    block = cfg.analysis.real_dicom_v2
    stage = str(block.get("stage", "all"))
    if stage not in ("run", "summarise", "all"):
        raise ValueError(f"real_dicom_v2: stage must be run | summarise | all, got {stage!r}.")
    override = {str(k): float(v) for k, v in dict(block.override).items()}
    if set(override) != set(_OVERRIDABLE_QC_KEYS):
        raise ValueError(
            f"real_dicom_v2: override must set exactly {sorted(_OVERRIDABLE_QC_KEYS)}, "
            f"got {sorted(override)}."
        )
    return V2Config(
        cases_files=tuple(_resolve_path(p) for p in block.cases_files),
        zips_dir=_resolve_path(block.zips_dir),
        out_dir=_resolve_path(block.out_dir),
        gt_root=_resolve_path(block.gt_root),
        research_per_case=_resolve_path(block.research_per_case),
        role_by_folder={str(k): str(v) for k, v in dict(block.role_by_folder).items()},
        usable_threshold=float(block.usable_threshold),
        carry_qc_threshold=float(block.carry_qc_threshold),
        e1_margin=float(block.e1_margin),
        bridge_min_dice=float(block.bridge_min_dice),
        n_boot=int(block.n_boot),
        seed=int(block.seed),
        carry_seed=int(block.carry_seed),
        override=override,
        p12_job_map=_resolve_path(block.p12_job_map),
        p12_jobs_dir=_resolve_path(block.p12_jobs_dir),
        # 90 is the registered n (Amendment 1, (c)); a smaller number only in tests.
        expected_n_cases=int(block.get("expected_n_cases", 90)),
        stage=stage,
    )


def load_specs(cases_files: Sequence[Path], expected_n: int) -> list[p12.CaseSpec]:
    """Loads every case from the case files; the pilot is never run.

    Args:
        cases_files: YAML files with a `cases: [{case_id, rsna_id}, ...]` key. A `pilot:`
            key, if present, is ignored.
        expected_n: The registered number of studies.

    Returns:
        Specs in file order (`pilot=False` on all).

    Raises:
        ValueError: A case id appears twice, or the total is not `expected_n`.
    """
    specs: list[p12.CaseSpec] = []
    for file in cases_files:
        payload = read_yaml(file)
        for entry in payload["cases"]:
            specs.append(
                p12.CaseSpec(
                    case_id=str(entry["case_id"]),
                    rsna_id=str(entry["rsna_id"]).zfill(5),  # YAML may parse '00009' as 9
                    pilot=False,
                )
            )
    ids = [s.case_id for s in specs]
    duplicates = sorted({c for c in ids if ids.count(c) > 1})
    if duplicates:
        raise ValueError(f"real_dicom_v2: duplicate case id(s) across the case files: {duplicates}")
    if len(specs) != expected_n:
        raise ValueError(f"real_dicom_v2: expected {expected_n} studies, found {len(specs)}.")
    return specs


def check_zips_present(specs: Sequence[p12.CaseSpec], zips_dir: Path) -> None:
    """Aborts BEFORE any case runs if any study zip is missing (the run must cover all).

    Raises:
        FileNotFoundError: Names every missing zip.
    """
    missing = [str(zips_dir / f"{s.rsna_id}.zip") for s in specs]
    missing = [p for p in missing if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"real_dicom_v2: {len(missing)} study zip(s) missing, refusing to start a partial "
            f"run: {missing}"
        )


# ---------------------------------------------------------------------------
# Thick-slice detection and the config override seam
# ---------------------------------------------------------------------------


def is_thick_slice_refusal(summary: Mapping[str, Any]) -> bool:
    """True iff the job was refused at input QC by a spacing or anisotropy finding.

    Reads the STRUCTURED input-QC report (`input_qc_pre`, the report from
    `InputQCReport.to_dict()`): a finding with `check` exactly `spacing_range` or `anisotropy`
    and severity `refuse`. Prose in `error` is never searched.

    Args:
        summary: A job's `summary.json` / `job.json` as a dict.
    """
    if summary.get("state") != "refused":
        return False
    report = summary.get("input_qc_pre")
    if not report:
        return False
    refuse = Severity.REFUSE.value
    return any(
        finding.get("check") in THICK_SLICE_CHECKS and finding.get("severity") == refuse
        for finding in report.get("findings", [])
    )


def apply_composed_overrides(
    cfg: Any, role_overrides: Mapping[str, str], qc_override: Mapping[str, float] | None
) -> None:
    """Mutates a composed clinical config in place: role overrides, plus optional QC override.

    The role overrides are set on every run (as P1.2). `qc_override` is set only for the
    override run and may name only `spacing_max_mm` and `anisotropy_refuse_ratio`; both must
    already exist under `cfg.clinical.input_qc` (so a typo cannot silently add a new key).

    Raises:
        ValueError: `qc_override` names a key outside the two allowed, or a missing one.
    """
    OmegaConf.set_struct(cfg, False)
    cfg.clinical.ingest.role_overrides = dict(role_overrides)
    if qc_override:
        unknown = set(qc_override) - _OVERRIDABLE_QC_KEYS
        if unknown:
            raise ValueError(f"apply_composed_overrides: not an overridable key: {sorted(unknown)}")
        for key, value in qc_override.items():
            if key not in cfg.clinical.input_qc:
                raise ValueError(f"apply_composed_overrides: clinical.input_qc has no {key!r}.")
            cfg.clinical.input_qc[key] = float(value)
    OmegaConf.set_struct(cfg, True)


# Signature of the clinical-path seam: (study_dir, jobs_dir, role_by_folder, qc_override)
# -> summary.json path. Tests replace it with a fake that writes a tiny job directory.
RunClinicalFn = Callable[[Path, Path, Mapping[str, str], "Mapping[str, float] | None"], Path]
CarryFn = Callable[..., CarryResult]


def run_clinical_path_v2(
    rsna_study_dir: Path,
    jobs_out_dir: Path,
    role_by_folder: Mapping[str, str],
    qc_override: Mapping[str, float] | None,
) -> Path:
    """Runs the REAL clinical pipeline on one study, optionally with the two QC settings raised.

    Same mechanism as `validate_real_dicom.run_clinical_path`: the job composes its own Hydra
    config via `app.backend.clinical_jobs._compose_clinical_cfg`, so that function is
    temporarily wrapped (restored in `finally`) to apply `apply_composed_overrides` to what it
    returns. Everything not named there is the deployed `configs/clinical/default.yaml`.

    Returns:
        Path to the written `<jobs_out_dir>/<job_id>/summary.json`.
    """
    role_overrides = p12.build_role_overrides(rsna_study_dir, role_by_folder)
    # Lazy: these reach torch / app.backend (same discipline as run_clinical_study.py).
    import scripts.run_clinical_study as run_clinical_study_script
    from app.backend import clinical_jobs as clinical_jobs_module

    original_compose = clinical_jobs_module._compose_clinical_cfg

    def _compose_with_overrides() -> Any:
        cfg = original_compose()
        apply_composed_overrides(cfg, role_overrides, qc_override)
        return cfg

    clinical_jobs_module._compose_clinical_cfg = _compose_with_overrides
    try:
        return run_clinical_study_script.run_study(rsna_study_dir, jobs_out_dir)
    finally:
        clinical_jobs_module._compose_clinical_cfg = original_compose


# ---------------------------------------------------------------------------
# Keeping only the listed files
# ---------------------------------------------------------------------------


def _atlas_mask_from_cache(
    job_dir: Path, summary: Mapping[str, Any], t1ce: Path, dest: Path
) -> None:
    """Rebuilds the atlas-space mask from the job's cached prediction.

    Needed for studies the GATEKEEPER refused: segmentation ran but the DICOM-SEG export (which
    writes `dicom_seg_work/atlas_space_mask.nii.gz`) only runs for accepted studies. Same steps
    as `_export_dicom_seg` step 1 and P1.2's `_run_case`: uncrop the cached prediction with the
    job's own meta, give it the atlas T1CE's affine.
    """
    from app.backend import clinical_jobs, inference  # lazy: torch / app.backend

    case_id = str(summary["case_id"])
    settings = clinical_jobs.clinical_segmentation_settings(job_dir / "prep", job_dir / "cache")
    pred = np.load(inference.cached_prediction_path(settings, case_id))
    meta = read_json(settings.prep_dir / case_id / "meta.json")
    full = uncrop_to_full(pred, meta)
    nib.save(nib.Nifti1Image(full.astype(np.uint8), nib.load(str(t1ce)).affine), str(dest))


def keep_artifacts(
    job_dir: Path, summary: Mapping[str, Any], kept_dir: Path, *, has_mask: bool
) -> list[str]:
    """Copies the files worth keeping out of a job directory into `kept_dir`.

    Always: `job.json` and `summary.json` (small; `job.json` holds the structured QC report).
    If `has_mask`: also the atlas-space mask, the skull-stripped atlas-space T1CE and the
    brain mask. The caller deletes `job_dir` afterwards.

    Where things are (the pipeline's own code):
    - T1CE: `<job>/clinical_prep/t1ce.nii.gz` (`clinical_preprocess.build_plan`:
      `outputs = {role: out_dir / f"{role}.nii.gz"}`, `out_dir = job_dir / "clinical_prep"`,
      `app/backend/clinical_jobs.py::run_clinical_job`). It is the HD-BET skull-stripped
      output, the very file the research preprocessing then crops and z-scores for the model.
    - brain mask: `<job>/clinical_prep/brain_mask.nii.gz` (`build_plan`, `brain_mask_path`).
    - mask: `<job>/dicom_seg_work/atlas_space_mask.nii.gz` if the DICOM-SEG export ran, else
      rebuilt from the cached prediction (`_atlas_mask_from_cache`).

    Returns:
        Names of the files now in `kept_dir`.

    Raises:
        FileNotFoundError: `has_mask` but the T1CE or brain mask is missing.
    """
    ensure_dir(kept_dir)
    for name in KEPT_JOB_FILES:
        if (job_dir / name).is_file():
            shutil.copy2(job_dir / name, kept_dir / name)
    if has_mask:
        prep = job_dir / "clinical_prep"
        for src_name, dst_name in (("t1ce.nii.gz", KEPT_T1CE), ("brain_mask.nii.gz", KEPT_BRAIN)):
            if not (prep / src_name).is_file():
                raise FileNotFoundError(f"keep_artifacts: no {src_name} under {prep}.")
            shutil.copy2(prep / src_name, kept_dir / dst_name)
        stored = job_dir / "dicom_seg_work" / "atlas_space_mask.nii.gz"
        if stored.is_file():
            shutil.copy2(stored, kept_dir / KEPT_MASK)
        else:
            _atlas_mask_from_cache(job_dir, summary, kept_dir / KEPT_T1CE, kept_dir / KEPT_MASK)
    return sorted(p.name for p in kept_dir.iterdir())


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


# ---------------------------------------------------------------------------
# Scoring one kept run
# ---------------------------------------------------------------------------


def _load_nifti_on_reference(path: Path, reference_affine: np.ndarray, dtype: Any) -> np.ndarray:
    """Loads a NIfTI and flips/permutes it onto the BraTS reference grid (never resamples)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj).astype(dtype)
    return to_reference_grid(data, img.affine, reference_affine)


def _stored_p12_mask_path(case_id: str, job_map: Mapping[str, str], jobs_dir: Path) -> Path | None:
    """The stored P1.2 atlas-space mask for a case, or None when P1.2 stored none."""
    job_id = job_map.get(case_id)
    if job_id is None:
        return None
    path = jobs_dir / job_id / "dicom_seg_work" / "atlas_space_mask.nii.gz"
    return path if path.is_file() else None


def score_kept_run(
    spec: p12.CaseSpec,
    kept_dir: Path,
    vcfg: V2Config,
    carry_fn: CarryFn,
    *,
    stored_mask: Path | None,
) -> dict[str, Any]:
    """Scores one run from its kept files: atlas-space, carried-reference and (optional) bridge.

    Two independent try-blocks: a carry failure (e.g. registration) leaves the atlas-space
    numbers in place, and the reverse. Failures are recorded in `error` / `carry_error`, never
    raised.

    Args:
        spec: The study.
        kept_dir: Directory holding the kept mask / T1CE / brain mask.
        vcfg: Resolved config.
        carry_fn: `carry_label`-compatible callable (tests pass a fake).
        stored_mask: Stored P1.2 mask for the bridge, or None.

    Returns:
        Row fields: `dice_*`, `hd95_*`, `lateralisation`, `mirrored`, `scored`,
        `carried_dice_*`, `carry_qc_dice`, `translation_mm`, `rotation_deg`, `carried_scored`,
        `bridge_wt_dice`, `geometry_mismatch`, `error`, `carry_error`.
    """
    nan = float("nan")
    out: dict[str, Any] = {
        **{f"dice_{r}": nan for r in _REGIONS},
        **{f"hd95_{r}": nan for r in _REGIONS},
        **{f"carried_dice_{r}": nan for r in _REGIONS},
        "lateralisation": None,
        "mirrored": False,
        "scored": False,
        "carry_qc_dice": nan,
        "translation_mm": None,
        "rotation_deg": nan,
        "carried_scored": False,
        "bridge_wt_dice": nan,
        "geometry_mismatch": None,
        "error": None,
        "carry_error": None,
    }
    gt_dir = vcfg.gt_root / spec.case_id
    gt_meta = read_json(gt_dir / "meta.json")
    reference_affine = np.asarray(gt_meta["affine"], dtype=np.float64)
    spacing = tuple(float(s) for s in gt_meta["spacing"])

    # --- atlas-space score (as P1.2) -------------------------------------------------------
    try:
        try:
            mask_ref = _load_nifti_on_reference(kept_dir / KEPT_MASK, reference_affine, np.uint8)
        except ValueError as exc:
            out["geometry_mismatch"] = str(exc)  # a real grid mismatch, not a flip
            raise
        gt_full = uncrop_to_full(np.load(gt_dir / "label.npy"), gt_meta)
        out.update(score_case(mask_ref, gt_full, spacing))
        lateral = lateralisation_check(mask_ref, gt_full, reference_affine)
        out["lateralisation"] = lateral
        out["mirrored"] = lateral == "mirrored"
        if out["mirrored"]:
            logger.error("real_dicom_v2: %s scored 'mirrored' -- flagged, kept.", spec.case_id)
        out["scored"] = True
    except Exception as exc:  # noqa: BLE001 - one study's scoring failure must not stop the run
        logger.error(
            "real_dicom_v2: atlas-space scoring failed for %s", spec.case_id, exc_info=True
        )
        out["error"] = str(exc)
        return out  # carried scoring and bridge both need mask_ref

    # --- bridge: re-run mask vs the stored P1.2 mask (WT Dice) -----------------------------
    if stored_mask is not None:
        try:
            stored_ref = _load_nifti_on_reference(stored_mask, reference_affine, np.uint8)
            out["bridge_wt_dice"] = score_against_carried(mask_ref, stored_ref)["dice_WT"]
        except Exception:  # noqa: BLE001 - leaves NaN, which bridge_check counts as below
            logger.error("real_dicom_v2: bridge failed for %s", spec.case_id, exc_info=True)

    # --- carried-reference score -----------------------------------------------------------
    try:
        ours_t1ce = _load_nifti_on_reference(kept_dir / KEPT_T1CE, reference_affine, np.float32)
        ours_brain = _load_nifti_on_reference(kept_dir / KEPT_BRAIN, reference_affine, np.uint8) > 0
        moving_t1ce, moving_label = brats_reference_on_grid(gt_dir)
        carry = carry_fn(
            moving_t1ce, moving_label, ours_t1ce, spacing=spacing, seed=vcfg.carry_seed
        )
        out["carry_qc_dice"] = brain_overlap_dice(carry.carried_brain_mask, ours_brain)
        out["translation_mm"] = [float(t) for t in carry.translation_mm]
        out["rotation_deg"] = float(carry.rotation_deg)
        out.update(
            {
                f"carried_{k}": v
                for k, v in score_against_carried(mask_ref, carry.carried_label).items()
            }
        )
        out["carried_scored"] = True
    except Exception as exc:  # noqa: BLE001
        logger.error("real_dicom_v2: carried scoring failed for %s", spec.case_id, exc_info=True)
        out["carry_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# One run (deployed or override), resumable
# ---------------------------------------------------------------------------


def run_dir(vcfg: V2Config, case_id: str, variant: str) -> Path:
    """`<out_dir>/cases/<case_id>/<variant>`: the kept files and `result.json` live here."""
    return vcfg.out_dir / "cases" / case_id / variant


def _run_variant(
    spec: p12.CaseSpec,
    variant: str,
    vcfg: V2Config,
    run_fn: RunClinicalFn,
    carry_fn: CarryFn,
    job_map: Mapping[str, str],
) -> dict[str, Any]:
    """Runs one study once (deployed, or with the QC override), keeps files, scores.

    Never raises for a case-level outcome: any exception is recorded in `error` (with
    `status="failed"` if no status was seen yet). The job directory and the unzipped DICOM
    are always deleted before returning.
    """
    start = time.monotonic()
    kept_dir = run_dir(vcfg, spec.case_id, variant)
    extract_root = vcfg.out_dir / "_extract" / f"{spec.case_id}__{variant}"
    jobs_dir = vcfg.out_dir / "_jobs"
    nan = float("nan")
    row: dict[str, Any] = {
        "case_id": spec.case_id,
        "rsna_id": spec.rsna_id,
        "variant": variant,
        "job_id": None,
        "status": None,
        "refusing_stage": None,
        "refusing_reason": None,
        "gate_decision": None,
        "accepted": False,
        "thick_slice_refusal": False,
        "has_mask": False,
        "runtime_s": None,
        "kept_files": [],
        "kept_mb": nan,
        "bridge_stored_mask": False,
        "error": None,
    }
    job_dir: Path | None = None
    try:
        zip_path = vcfg.zips_dir / f"{spec.rsna_id}.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(f"no RSNA study zip at {zip_path}.")
        study_dir = p12._unzip_study(zip_path, extract_root, spec.rsna_id)
        qc_override = vcfg.override if variant == OVERRIDE else None
        summary_path = run_fn(study_dir, jobs_dir, vcfg.role_by_folder, qc_override)
        job_dir = summary_path.parent
        summary = read_json(summary_path)

        row["job_id"] = summary.get("job_id")
        row["status"] = summary.get("state")
        row["runtime_s"] = summary.get("wall_s")
        if row["status"] in ("refused", "failed"):
            row["refusing_stage"] = summary.get("stage")
            row["refusing_reason"] = summary.get("error")
        gate = summary.get("gatekeeper_decision")
        row["gate_decision"] = gate.get("decision") if gate else None
        row["accepted"] = row["status"] == "done"  # done only ever follows a non-REFUSE gate
        row["thick_slice_refusal"] = is_thick_slice_refusal(summary)
        row["has_mask"] = row["status"] == "done" or (
            row["status"] == "refused" and row["refusing_stage"] == _GATEKEEPER_STAGE
        )

        # Bridge applies to the deployed run only, and only where P1.2 stored a mask.
        stored = None
        if variant == DEPLOYED:
            stored = _stored_p12_mask_path(spec.case_id, job_map, vcfg.p12_jobs_dir)
            row["bridge_stored_mask"] = stored is not None

        # The pipeline's own outcome is recorded above; a keep/score failure below must not
        # overwrite it (same separation as P1.2's `_run_case`).
        try:
            row["kept_files"] = keep_artifacts(job_dir, summary, kept_dir, has_mask=row["has_mask"])
            if row["has_mask"]:
                row.update(score_kept_run(spec, kept_dir, vcfg, carry_fn, stored_mask=stored))
        except Exception as exc:  # noqa: BLE001
            logger.error("real_dicom_v2: keep/score failed for %s", spec.case_id, exc_info=True)
            row["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - one study must never abort the whole run
        logger.error("real_dicom_v2: %s (%s) errored", spec.case_id, variant, exc_info=True)
        if row["status"] is None:
            row["status"] = "failed"
        row["error"] = str(exc)
    finally:
        if row["runtime_s"] is None:
            row["runtime_s"] = round(time.monotonic() - start, 1)
        shutil.rmtree(extract_root, ignore_errors=True)
        if job_dir is not None:
            shutil.rmtree(job_dir, ignore_errors=True)  # ~700 MB; the kept files were copied out
        if kept_dir.is_dir():
            row["kept_mb"] = round(_dir_size_mb(kept_dir), 2)
            logger.info(
                "real_dicom_v2: %s (%s) kept %s = %.1f MB",
                spec.case_id,
                variant,
                row["kept_files"],
                row["kept_mb"],
            )
    return row


def process_variant(
    spec: p12.CaseSpec,
    variant: str,
    vcfg: V2Config,
    run_fn: RunClinicalFn,
    carry_fn: CarryFn,
    job_map: Mapping[str, str],
) -> tuple[dict[str, Any], bool]:
    """Runs one variant, or loads its parsed `result.json` (resume).

    An unparseable `result.json` is treated as absent: warned about, the run directory is wiped
    (no stale kept files) and the study re-run.

    Returns:
        `(row, ran)`: `ran` is False when the row came from disk.
    """
    result_file = run_dir(vcfg, spec.case_id, variant) / "result.json"
    if result_file.is_file():
        try:
            return read_json(result_file), False
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("real_dicom_v2: unparseable %s (%s); re-running.", result_file, exc)
    shutil.rmtree(run_dir(vcfg, spec.case_id, variant), ignore_errors=True)
    row = _run_variant(spec, variant, vcfg, run_fn, carry_fn, job_map)
    p12._write_result_atomic(_jsonable(row), result_file)
    return row, True


# ---------------------------------------------------------------------------
# Self-tests and the run stage
# ---------------------------------------------------------------------------


def mirrored_self_test(gt_root: Path, specs: Sequence[p12.CaseSpec]) -> None:
    """A left-right flipped ground truth must be flagged "mirrored" by `lateralisation_check`.

    Proves the P1.2 lateralisation check can still fail before it is trusted. Tries the first
    few cases because a midline tumour looks the same flipped.

    Raises:
        RuntimeError: None of the tried cases was flagged.
    """
    for spec in list(specs)[:_MIRROR_SELF_TEST_TRIES]:
        case_dir = gt_root / spec.case_id
        meta = read_json(case_dir / "meta.json")
        gt = uncrop_to_full(np.load(case_dir / "label.npy"), meta)
        affine = np.asarray(meta["affine"], dtype=np.float64)
        lr_axis = next(i for i, c in enumerate(nib.aff2axcodes(affine)) if c in ("L", "R"))
        flipped = np.ascontiguousarray(np.flip(gt, axis=lr_axis))
        if lateralisation_check(flipped, gt, affine) == "mirrored":
            logger.info("real_dicom_v2: mirrored-mask self-test OK on %s.", spec.case_id)
            return
    raise RuntimeError(
        "real_dicom_v2: mirrored-mask self-test failed -- lateralisation_check did not flag a "
        f"flipped ground truth on the first {_MIRROR_SELF_TEST_TRIES} cases."
    )


def run_stage(
    vcfg: V2Config,
    *,
    run_fn: RunClinicalFn = run_clinical_path_v2,
    carry_fn: CarryFn = carry_label,
) -> list[p12.CaseSpec]:
    """Runs every study (resumable, one at a time): deployed run, plus override if thick-slice.

    Order of business: load and de-duplicate the case lists, check ALL zips exist, run the
    P1.2 round-trip self-test and the mirrored-mask self-test, then the studies.

    Args:
        vcfg: Resolved config.
        run_fn: Clinical-path seam (tests pass a fake).
        carry_fn: Carry seam (tests pass a fake).

    Returns:
        The specs, in run order.
    """
    specs = load_specs(vcfg.cases_files, vcfg.expected_n_cases)
    check_zips_present(specs, vcfg.zips_dir)
    if not vcfg.p12_job_map.is_file():
        raise FileNotFoundError(f"real_dicom_v2: P1.2 job map missing at {vcfg.p12_job_map}.")
    job_map = {str(k): str(v) for k, v in read_json(vcfg.p12_job_map).items()}

    # The P1.2 round trip needs two specs; with a single study use it twice.
    p12.run_self_tests(vcfg.gt_root, specs[0], specs[min(1, len(specs) - 1)])
    mirrored_self_test(vcfg.gt_root, specs)
    ensure_dir(vcfg.out_dir)
    logger.info("real_dicom_v2: %s Running %d studies.", _HEADER_NOTE, len(specs))

    seconds_spent = 0.0
    n_ran = 0
    for i, spec in enumerate(specs):
        t0 = time.monotonic()
        deployed, ran = process_variant(spec, DEPLOYED, vcfg, run_fn, carry_fn, job_map)
        if deployed.get("thick_slice_refusal"):
            _, ran_override = process_variant(spec, OVERRIDE, vcfg, run_fn, carry_fn, job_map)
            ran = ran or ran_override
        if ran:
            seconds_spent += time.monotonic() - t0
            n_ran += 1
        remaining = len(specs) - (i + 1)
        eta_min = (seconds_spent / n_ran) * remaining / 60 if n_ran else float("nan")
        logger.info(
            "real_dicom_v2: [%d/%d] %s status=%s thick=%s%s ETA ~%.0f min",
            i + 1,
            len(specs),
            spec.case_id,
            deployed.get("status"),
            deployed.get("thick_slice_refusal"),
            "" if ran else " (resumed)",
            eta_min,
        )
    return specs


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Recursively turns NaN into None and numpy scalars into Python ones (strict JSON)."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def build_per_case_frame(specs: Sequence[p12.CaseSpec], vcfg: V2Config) -> pd.DataFrame:
    """One row per study in the `REQUIRED_COLUMNS` schema (plus extra diagnostic columns).

    Raises:
        FileNotFoundError: A study has no deployed `result.json` (run stage incomplete), or an
            override result is missing for a thick-slice refusal.
    """
    research = pd.read_csv(vcfg.research_per_case).set_index("case_id")
    nan = float("nan")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        dep_file = run_dir(vcfg, spec.case_id, DEPLOYED) / "result.json"
        if not dep_file.is_file():
            raise FileNotFoundError(f"real_dicom_v2: no deployed result for {spec.case_id}.")
        dep = read_json(dep_file)
        ov: dict[str, Any] = {}
        if dep.get("thick_slice_refusal"):
            ov_file = run_dir(vcfg, spec.case_id, OVERRIDE) / "result.json"
            if not ov_file.is_file():
                raise FileNotFoundError(f"real_dicom_v2: no override result for {spec.case_id}.")
            ov = read_json(ov_file)

        def num(d: Mapping[str, Any], key: str) -> float:
            value = d.get(key)
            return nan if value is None else float(value)

        row: dict[str, Any] = {
            "case_id": spec.case_id,
            "status": dep.get("status"),
            "refusing_stage": dep.get("refusing_stage"),
            "thick_slice_refusal": bool(dep.get("thick_slice_refusal")),
            "accepted": bool(dep.get("accepted")),
            "gate_decision": dep.get("gate_decision"),
            "carry_qc_dice": num(dep, "carry_qc_dice"),
            "override_status": ov.get("status"),
            "override_accepted": bool(ov["accepted"]) if ov else None,
            "override_gate_decision": ov.get("gate_decision"),
            "override_carry_qc_dice": num(ov, "carry_qc_dice"),
            # extras
            "rsna_id": spec.rsna_id,
            "refusing_reason": dep.get("refusing_reason"),
            "runtime_s": dep.get("runtime_s"),
            "lateralisation": dep.get("lateralisation"),
            "mirrored": bool(dep.get("mirrored")),
            "translation_mm": dep.get("translation_mm"),
            "rotation_deg": num(dep, "rotation_deg"),
            "bridge_stored_mask": bool(dep.get("bridge_stored_mask")),
            "bridge_wt_dice": num(dep, "bridge_wt_dice"),
            "error": dep.get("error"),
            "override_error": ov.get("error"),
        }
        for r in _REGIONS:
            row[f"atlas_dice_{r}"] = num(dep, f"dice_{r}")
            row[f"carried_dice_{r}"] = num(dep, f"carried_dice_{r}")
            row[f"override_atlas_dice_{r}"] = num(ov, f"dice_{r}")
            row[f"override_carried_dice_{r}"] = num(ov, f"carried_dice_{r}")
            in_research = spec.case_id in research.index and f"dice_{r}" in research.columns
            row[f"research_dice_{r}"] = (
                float(research.loc[spec.case_id, f"dice_{r}"]) if in_research else nan
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:  # a programming error here would otherwise surface deep inside an endpoint
        raise RuntimeError(f"build_per_case_frame: missing schema columns {missing}")
    return df


def _md_table(df: pd.DataFrame) -> str:
    """A GitHub-markdown table without needing `tabulate`."""
    if df.empty:
        return "(empty)\n"

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.4f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |"]
    lines.append("|" + "---|" * len(df.columns))
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    """Current commit SHA, or None if git cannot say (provenance nicety, never fatal)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def summarise_stage(
    specs: Sequence[p12.CaseSpec], vcfg: V2Config, cfg: DictConfig
) -> dict[str, Path]:
    """Builds `per_case.csv`, runs E1-E4 and the bridge, writes every output file.

    E4 is a post-hoc variant that the registered rule reports ONLY when E2 reads
    `REFUSES USABLE STUDIES`; otherwise `e4.json` records that it is not reported and
    `summary.md` says so, with no E4 numbers.

    Returns:
        Output name -> path.
    """
    out = ensure_dir(vcfg.out_dir)
    df = build_per_case_frame(specs, vcfg)
    paths = {"per_case": out / "per_case.csv"}
    df.to_csv(paths["per_case"], index=False)

    common = {
        "usable_threshold": vcfg.usable_threshold,
        "carry_qc_threshold": vcfg.carry_qc_threshold,
    }
    e1 = e1_front_end_cost(
        df,
        margin=vcfg.e1_margin,
        carry_qc_threshold=vcfg.carry_qc_threshold,
        n_boot=vcfg.n_boot,
        seed=vcfg.seed,
    )
    e2 = e2_refusal_check(df, **common)
    e3 = e3_table(df, **common, n_boot=vcfg.n_boot, seed=vcfg.seed)
    report_e4 = e2["verdict"] == "REFUSES USABLE STUDIES"
    if report_e4:
        e4 = e4_caution_variant(df, **common, n_boot=vcfg.n_boot, seed=vcfg.seed)
    else:
        e4 = {
            "endpoint": "E4",
            "reported": False,
            "reason": "Registered rule: E4 is reported only if E2 reads REFUSES USABLE STUDIES.",
            "e2_verdict": e2["verdict"],
        }

    # Bridge population: studies where P1.2 stored a mask. A missing re-run mask (NaN) counts
    # as below the bar -- reproduction failed.
    bridged = df[df["bridge_stored_mask"]].set_index("case_id")["bridge_wt_dice"]
    bridge = bridge_check(bridged, min_dice=vcfg.bridge_min_dice)

    write_json(_jsonable(e1), out / "e1.json")
    write_json(_jsonable(e2), out / "e2.json")
    e3.to_csv(out / "e3_table.csv", index=False)
    write_json(_jsonable(e4), out / "e4.json")
    write_json(_jsonable(bridge), out / "bridge.json")

    counts = _counts(df, vcfg)
    paths["summary_md"] = out / "summary.md"
    paths["summary_md"].write_text(_summary_markdown(e1, e2, e3, e4, bridge, counts, report_e4))
    write_json(
        _jsonable(
            {
                "note": _HEADER_NOTE,
                "git_sha": _git_sha(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "config": OmegaConf.to_container(cfg.analysis.real_dicom_v2, resolve=True),
                "counts": counts,
                "input_sha256": {
                    **{f"cases_file:{p.name}": _sha256(p) for p in vcfg.cases_files},
                    f"research_per_case:{vcfg.research_per_case.name}": _sha256(
                        vcfg.research_per_case
                    ),
                },
            }
        ),
        out / "run_meta.json",
    )
    logger.info("real_dicom_v2: wrote outputs under %s (E2 = %s).", out, e2["verdict"])
    return paths


def _counts(df: pd.DataFrame, vcfg: V2Config) -> dict[str, int]:
    """Study counts every table must be read against."""
    has_mask = df["status"].eq("done") | (
        df["status"].eq("refused") & df["refusing_stage"].eq(_GATEKEEPER_STAGE)
    )
    qc = pd.to_numeric(df["carry_qc_dice"], errors="coerce")
    return {
        "n_studies": int(len(df)),
        "n_done": int(df["status"].eq("done").sum()),
        "n_refused": int(df["status"].eq("refused").sum()),
        "n_failed": int(df["status"].eq("failed").sum()),
        "n_thick_slice_refusals": int(df["thick_slice_refusal"].sum()),
        "n_override_runs": int(df["override_status"].notna().sum()),
        "n_deployed_with_mask": int(has_mask.sum()),
        "n_carry_qc_failed": int((has_mask & ~(qc >= vcfg.carry_qc_threshold)).sum()),
        "n_mirrored": int(df["mirrored"].sum()),
        "n_bridge_stored_masks": int(df["bridge_stored_mask"].sum()),
    }


def _summary_markdown(
    e1: Mapping[str, Any],
    e2: Mapping[str, Any],
    e3: pd.DataFrame,
    e4: Mapping[str, Any],
    bridge: Mapping[str, Any],
    counts: Mapping[str, int],
    report_e4: bool,
) -> str:
    """The human-readable summary. Every section carries the NOT external validation label."""
    label = "NOT external validation"
    lines = [
        "# Real-DICOM v2 (W3.4) -- summary",
        "",
        f"**{label}.** {_HEADER_NOTE}",
        "",
        "## Counts",
        "",
        *(f"- {k}: {v}" for k, v in counts.items()),
        "",
        f"## E1 -- front-end cost after removing registration disagreement ({label})",
        "",
        f"**Verdict: {e1['verdict']}** (margin {e1['margin']}, n = {e1['n_population']}, "
        f"carry-QC excluded {e1['n_excluded_carry_qc']}).",
        "",
        _md_table(pd.DataFrame(e1["per_region"])),
        f"## E2 -- was the thick-slice refusal right? ({label})",
        "",
        f"**Verdict: {e2['verdict']}**. Usable on the carried reference: "
        f"{e2['carried']['k']}/{e2['carried']['n']} = {e2['carried']['fraction']:.3f} "
        f"(95% CI {e2['carried']['ci_lo']:.3f}-{e2['carried']['ci_hi']:.3f}); "
        f"{e2['n_thick']} thick-slice refusals, {e2['n_excluded']} excluded (carry QC / unscored).",
        "",
        f"## E3 -- Table XV v2 ({label})",
        "",
        _md_table(e3),
        f"## E4 -- CAUTION instead of REFUSE for thick slices ({label})",
        "",
    ]
    if report_e4:
        lines += [
            f"**{e4['label']}** (not the deployed rule). {e4['n_thick_slice']} thick-slice "
            f"studies, {e4['n_thick_slice_accepted']} accepted when overridden.",
            "",
            "```json",
            json.dumps(_jsonable(e4), indent=2),
            "```",
            "",
        ]
    else:
        lines += [
            "Not reported: by the registered rule E4 is reported only if E2 reads "
            "REFUSES USABLE STUDIES.",
            "",
        ]
    lines += [
        "## Bridge to P1.2",
        "",
        f"{bridge['n']} stored masks compared, bar WT Dice >= {bridge['min_dice']}: "
        f"all pass = {bridge['all_pass']}; below the bar: {bridge['below_min_dice']}.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_v2(
    cfg: DictConfig,
    *,
    run_fn: RunClinicalFn = run_clinical_path_v2,
    carry_fn: CarryFn = carry_label,
) -> dict[str, Path]:
    """Runs the requested stage(s) of the W3.4 protocol.

    Args:
        cfg: Full composed Hydra config; reads `cfg.analysis.real_dicom_v2`.
        run_fn: Clinical-path seam.
        carry_fn: Carry seam.

    Returns:
        Output name -> path (empty for stage `run`).
    """
    vcfg = resolve_config(cfg)
    specs: list[p12.CaseSpec] = []
    if vcfg.stage in ("run", "all"):
        specs = run_stage(vcfg, run_fn=run_fn, carry_fn=carry_fn)
    if vcfg.stage in ("summarise", "all"):
        specs = specs or load_specs(vcfg.cases_files, vcfg.expected_n_cases)
        return summarise_stage(specs, vcfg, cfg)
    return {}


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs W3.4, real-DICOM v2. Stage via `+analysis.real_dicom_v2.stage=run|summarise|all`."""
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    # Each clinical job composes its own Hydra config, which fails while @hydra.main's
    # singleton is live (see validate_real_dicom.release_hydra's docstring).
    p12.release_hydra()
    run_v2(cfg)


if __name__ == "__main__":
    main()
