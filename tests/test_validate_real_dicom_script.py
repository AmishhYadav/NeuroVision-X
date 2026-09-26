"""Tests for `scripts/validate_real_dicom.py`, the P1.2 real-DICOM validation driver.

Everything here is synthetic, small, and CPU-only -- no real DICOM, no pydicom
(`build_role_overrides`'s `read_series_headers` call is monkeypatched wherever it would
otherwise run), no Hydra composition of the real clinical pipeline, no `.venv-clinical`.
The `_run_case`-level tests monkeypatch `run_clinical_path_fn` -- the one seam that would
otherwise reach the real clinical pipeline -- with a fake that writes a synthetic job
directory (`prep/<job_id>/{image.npy,meta.json}`, `cache/neurovision/<job_id>.npy`,
`summary.json`) and returns its `summary.json` path, so `_run_case`'s own uncrop/
reference-grid/scoring/lateralisation logic still runs for real, against real (tiny)
arrays.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scripts.validate_real_dicom as validate_real_dicom

from neurovision.utils.io import write_json

CaseSpec = validate_real_dicom.CaseSpec
RealDicomValidationConfig = validate_real_dicom.RealDicomValidationConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _lps_affine() -> list[list[float]]:
    """BraTS/SRI24's own affine convention -- axis 0 is the L/R axis (verified against
    `nibabel.aff2axcodes`), which is what `lateralisation_check` needs to find."""
    return np.diag([-1.0, -1.0, 1.0, 1.0]).tolist()


def _write_case_dir(root: Path, case_id: str, label: np.ndarray) -> None:
    """Writes a minimal `<root>/<case_id>/{label.npy,meta.json}`, no real crop."""
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", label.astype(np.int64))
    shape = list(label.shape)
    meta = {
        "case_id": case_id,
        "original_shape": shape,
        "cropped_shape": shape,
        "bbox": [[0, s] for s in shape],
        "affine": _lps_affine(),
        "spacing": [1.0, 1.0, 1.0],
    }
    write_json(meta, case_dir / "meta.json")


def _write_synthetic_job(
    jobs_out_dir: Path,
    job_id: str,
    *,
    pred_cropped: np.ndarray,
    state: str = "done",
    stage: str = "done",
    error: str | None = None,
    gate_decision: str | None = "proceed",
    wall_s: float = 1.2,
) -> Path:
    """Writes a synthetic clinical job directory and returns its `summary.json` path.

    Matches the REAL layout `_run_case` reads: `clinical_jobs.clinical_segmentation_settings`
    + `inference.cached_prediction_path` resolve the prediction to
    `<job_dir>/cache/neurovision/<job_id>.npy`, and its `meta.json` to
    `<job_dir>/prep/<job_id>/meta.json`.
    """
    job_dir = jobs_out_dir / job_id
    prep_case_dir = job_dir / "prep" / job_id
    prep_case_dir.mkdir(parents=True, exist_ok=True)
    shape = list(pred_cropped.shape)
    meta = {
        "case_id": job_id,
        "original_shape": shape,
        "cropped_shape": shape,
        "bbox": [[0, s] for s in shape],
        "affine": _lps_affine(),
        "spacing": [1.0, 1.0, 1.0],
    }
    write_json(meta, prep_case_dir / "meta.json")

    cache_dir = job_dir / "cache" / "neurovision"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{job_id}.npy", pred_cropped.astype(np.uint8))

    summary = {
        "job_id": job_id,
        "case_id": job_id,
        "state": state,
        "stage": stage,
        "error": error,
        "gatekeeper_decision": (
            {"decision": gate_decision, "verdicts": []} if gate_decision else None
        ),
        "wall_s": wall_s,
    }
    summary_path = job_dir / "summary.json"
    write_json(summary, summary_path)
    return summary_path


def _make_zip(zips_dir: Path, rsna_id: str) -> None:
    """A trivial RSNA-layout zip -- content never matters, since role assignment happens
    inside the faked `run_clinical_path_fn`, not inside `_run_case` itself."""
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zips_dir / f"{rsna_id}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for folder in ("FLAIR", "T1w", "T1wCE", "T2w"):
            archive.writestr(f"{rsna_id}/{folder}/IM001.dcm", b"placeholder")


def _make_ccfg(tmp_path: Path, **overrides: object) -> RealDicomValidationConfig:
    defaults: dict[str, object] = {
        "cases_file": tmp_path / "cases.yaml",
        "zips_dir": tmp_path / "zips",
        "work_dir": tmp_path / "work",
        "research_per_case": tmp_path / "research.csv",
        "gt_root": tmp_path / "gt",
        "usable_threshold": 0.7,
        "n_boot": 200,
        "seed": 0,
        "include_pilot": True,
        "role_by_folder": {"FLAIR": "flair", "T1w": "t1", "T1wCE": "t1ce", "T2w": "t2"},
    }
    defaults.update(overrides)
    return RealDicomValidationConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def test_process_case_resumes_when_result_exists(tmp_path: Path) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_TEST", rsna_id="00001", pilot=False)
    result_file = validate_real_dicom.result_path(ccfg.work_dir, spec.case_id)
    result_file.parent.mkdir(parents=True)
    write_json({"case_id": spec.case_id, "status": "done", "pilot": False}, result_file)

    calls: list[CaseSpec] = []

    def _fake_run_case(spec_: CaseSpec, ccfg_: RealDicomValidationConfig) -> dict[str, object]:
        calls.append(spec_)
        raise AssertionError("run_case_fn must not be called when result.json already exists")

    row = validate_real_dicom.process_case(spec, ccfg, run_case_fn=_fake_run_case)

    assert row["status"] == "done"
    assert calls == []


def test_process_case_runs_and_writes_result_when_absent(tmp_path: Path) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_TEST", rsna_id="00001", pilot=False)

    def _fake_run_case(spec_: CaseSpec, ccfg_: RealDicomValidationConfig) -> dict[str, object]:
        return {"case_id": spec_.case_id, "status": "done", "pilot": False}

    row = validate_real_dicom.process_case(spec, ccfg, run_case_fn=_fake_run_case)

    assert row["status"] == "done"
    result_file = validate_real_dicom.result_path(ccfg.work_dir, spec.case_id)
    assert result_file.is_file()


# ---------------------------------------------------------------------------
# role_overrides built from RSNA folder names
# ---------------------------------------------------------------------------


def test_build_role_overrides_from_folder_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_dir = tmp_path / "study"
    role_by_folder = {"FLAIR": "flair", "T1w": "t1", "T1wCE": "t1ce", "T2w": "t2"}
    for folder in role_by_folder:
        (study_dir / folder).mkdir(parents=True)

    def _fake_read_series_headers(folder: Path) -> list[SimpleNamespace]:
        return [SimpleNamespace(series_uid=f"uid-{Path(folder).name}")]

    monkeypatch.setattr(validate_real_dicom, "read_series_headers", _fake_read_series_headers)

    overrides = validate_real_dicom.build_role_overrides(study_dir, role_by_folder)

    assert overrides == {
        "uid-FLAIR": "flair",
        "uid-T1w": "t1",
        "uid-T1wCE": "t1ce",
        "uid-T2w": "t2",
    }


def test_build_role_overrides_missing_folder_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    # Only FLAIR present; T1w etc. missing entirely.
    (study_dir / "FLAIR").mkdir()

    def _fake_read_series_headers(folder: Path) -> list[SimpleNamespace]:
        return [SimpleNamespace(series_uid="uid")]

    monkeypatch.setattr(validate_real_dicom, "read_series_headers", _fake_read_series_headers)

    with pytest.raises(FileNotFoundError):
        validate_real_dicom.build_role_overrides(
            study_dir, {"FLAIR": "flair", "T1w": "t1", "T1wCE": "t1ce", "T2w": "t2"}
        )


# ---------------------------------------------------------------------------
# Self-test abort path
# ---------------------------------------------------------------------------


def test_run_self_tests_aborts_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gt_root = tmp_path / "gt"
    pilot = CaseSpec(case_id="PILOT", rsna_id="00000", pilot=True)
    first = CaseSpec(case_id="CASE1", rsna_id="00001", pilot=False)
    for spec in (pilot, first):
        _write_case_dir(gt_root, spec.case_id, np.zeros((2, 2, 2), dtype=np.int64))

    calls: list[str] = []

    def _fake_roundtrip_self_test(gt_cropped: np.ndarray, meta: dict) -> None:
        calls.append(meta["case_id"])
        raise RuntimeError("round trip broken")

    monkeypatch.setattr(validate_real_dicom, "roundtrip_self_test", _fake_roundtrip_self_test)

    with pytest.raises(RuntimeError, match="round trip broken"):
        validate_real_dicom.run_self_tests(gt_root, pilot, first)

    # Aborted on the FIRST case checked (the pilot) -- the second is never reached.
    assert calls == ["PILOT"]


# ---------------------------------------------------------------------------
# Completion counts + usable-rate arithmetic on hand-made rows
# ---------------------------------------------------------------------------


def test_completion_counts_keeps_refused_and_failed_in_denominator() -> None:
    scored = pd.DataFrame(
        [
            {
                "case_id": "a",
                "status": "done",
                "refusing_stage": None,
                "accepted": True,
                "dice_WT": 0.9,
                "dice_TC": 0.9,
            },
            {
                "case_id": "b",
                "status": "refused",
                "refusing_stage": "input_qc (pre-preprocessing)",
                "accepted": False,
                "dice_WT": float("nan"),
                "dice_TC": float("nan"),
            },
            {
                "case_id": "c",
                "status": "failed",
                "refusing_stage": None,
                "accepted": False,
                "dice_WT": float("nan"),
                "dice_TC": float("nan"),
            },
        ]
    )

    counts = validate_real_dicom.completion_counts(scored)

    assert counts["by_status"] == {"done": 1, "refused": 1, "failed": 1}
    refused_row = next(r for r in counts["by_status_and_stage"] if r["status"] == "refused")
    assert refused_row["refusing_stage"] == "input_qc (pre-preprocessing)"
    assert refused_row["n"] == 1


def test_usable_rate_stats_arithmetic_matches_hand_computation() -> None:
    # 4 correct_accept, 1 silent_failure (accepted, unusable), 1 over_refusal
    # (not accepted, usable), 1 correct_refusal -- n=7, matching the G5 taxonomy's
    # four cells (docs/research/error_budget_protocol.md).
    rows = (
        [{"accepted": True, "usable": True}] * 4
        + [{"accepted": True, "usable": False}] * 1
        + [{"accepted": False, "usable": True}] * 1
        + [{"accepted": False, "usable": False}] * 1
    )
    scored = pd.DataFrame(rows)

    stats = validate_real_dicom.usable_rate_stats(scored, n_boot=500, seed=0)

    assert stats["n"] == 7
    assert stats["n_accepted"] == 5
    assert stats["p_accepted_and_usable"] == pytest.approx(4 / 7)
    assert stats["p_silent_failure"] == pytest.approx(1 / 7)
    assert stats["p_usable_given_accepted"] == pytest.approx(4 / 5)
    # CIs bracket the point estimate.
    assert stats["p_accepted_and_usable_lo"] <= stats["p_accepted_and_usable"]
    assert stats["p_accepted_and_usable"] <= stats["p_accepted_and_usable_hi"]


def test_label_usable_requires_both_regions_and_nan_is_not_usable() -> None:
    per_case = pd.DataFrame(
        {
            "dice_WT": [0.9, 0.9, float("nan")],
            "dice_TC": [0.9, 0.5, 0.9],
        }
    )

    usable = validate_real_dicom.label_usable(per_case, threshold=0.7)

    assert list(usable) == [True, False, False]


# ---------------------------------------------------------------------------
# pilot excluded from every statistic
# ---------------------------------------------------------------------------


def test_summarise_excludes_pilot(tmp_path: Path) -> None:
    per_case = pd.DataFrame(
        [
            {
                "case_id": "PILOT",
                "pilot": True,
                "status": "failed",
                "refusing_stage": None,
                "accepted": False,
                "scored": False,
                "dice_ET": float("nan"),
                "dice_TC": float("nan"),
                "dice_WT": float("nan"),
            },
            {
                "case_id": "a",
                "pilot": False,
                "status": "done",
                "refusing_stage": None,
                "accepted": True,
                "scored": True,
                "dice_ET": 0.9,
                "dice_TC": 0.9,
                "dice_WT": 0.9,
            },
            {
                "case_id": "b",
                "pilot": False,
                "status": "refused",
                "refusing_stage": "gatekeeper",
                "accepted": False,
                "scored": False,
                "dice_ET": float("nan"),
                "dice_TC": float("nan"),
                "dice_WT": float("nan"),
            },
        ]
    )
    ccfg = _make_ccfg(tmp_path, research_per_case=tmp_path / "does_not_exist.csv")

    summary = validate_real_dicom.summarise(per_case, ccfg)

    # n excludes the pilot: 2 sample cases, not 3.
    assert summary["n"] == 2
    assert summary["completion"]["by_status"] == {"done": 1, "refused": 1}
    assert "PILOT" not in {r.get("case_id") for r in summary.get("front_end_cost", [])}
    assert summary["front_end_cost"] == []  # no research_per_case on disk


# ---------------------------------------------------------------------------
# front-end cost: pairing on matching case_ids only
# ---------------------------------------------------------------------------


def test_front_end_cost_pairs_matching_case_ids_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    done = pd.DataFrame(
        [
            {"case_id": "a", "dice_ET": 0.80, "dice_TC": 0.85, "dice_WT": 0.90},
            {"case_id": "b", "dice_ET": 0.70, "dice_TC": 0.75, "dice_WT": 0.80},
            {"case_id": "missing_in_research", "dice_ET": 0.5, "dice_TC": 0.5, "dice_WT": 0.5},
        ]
    )
    research = pd.DataFrame(
        [
            {"case_id": "a", "dice_ET": 0.85, "dice_TC": 0.90, "dice_WT": 0.95},
            {"case_id": "b", "dice_ET": 0.75, "dice_TC": 0.80, "dice_WT": 0.85},
            {"case_id": "only_in_research", "dice_ET": 0.6, "dice_TC": 0.6, "dice_WT": 0.6},
        ]
    )

    with caplog.at_level(logging.WARNING):
        table = validate_real_dicom.front_end_cost(done, research, n_boot=200, seed=0)

    assert set(table["region"]) == {"ET", "TC", "WT"}
    row_et = table[table["region"] == "ET"].iloc[0]
    assert row_et["n"] == 2  # "missing_in_research" and "only_in_research" both dropped
    assert row_et["mean_diff"] == pytest.approx(-0.05)  # clinical - research, both cases
    assert "missing_in_research" in caplog.text


# ---------------------------------------------------------------------------
# _run_case: the clinical-path call faked, real scoring logic exercised
# ---------------------------------------------------------------------------


def _fake_run_clinical_path_factory(pred_cropped: np.ndarray, job_id: str = "job123"):
    def _fake(rsna_study_dir: Path, jobs_out_dir: Path, role_by_folder: dict) -> Path:
        assert rsna_study_dir.is_dir()
        return _write_synthetic_job(jobs_out_dir, job_id, pred_cropped=pred_cropped)

    return _fake


def test_run_case_done_scores_matching_prediction(tmp_path: Path) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_TEST2", rsna_id="00002", pilot=False)
    _make_zip(ccfg.zips_dir, spec.rsna_id)

    gt = np.zeros((4, 4, 4), dtype=np.int64)
    gt[0, 1, 1] = 3  # a single ET voxel near one L/R edge
    _write_case_dir(ccfg.gt_root, spec.case_id, gt)

    fake_run = _fake_run_clinical_path_factory(gt.astype(np.uint8))

    row = validate_real_dicom._run_case(spec, ccfg, run_clinical_path_fn=fake_run)

    assert row["status"] == "done"
    assert row["accepted"] is True
    assert row["gate_decision"] == "proceed"
    assert row["dice_ET"] == pytest.approx(1.0)
    assert row["dice_TC"] == pytest.approx(1.0)
    assert row["dice_WT"] == pytest.approx(1.0)
    assert row["lateralisation"] == "ok"
    assert row["mirrored"] is False

    # Large intermediates cleaned up; the unzipped DICOM is gone too.
    job_dir = ccfg.work_dir / "jobs" / "job123"
    assert not (job_dir / "cache").exists()
    assert not (job_dir / "prep").exists()
    assert (job_dir / "summary.json").is_file()
    assert not (ccfg.work_dir / "_extract" / spec.case_id).exists()


def test_run_case_flags_mirrored_prediction(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_TEST3", rsna_id="00003", pilot=False)
    _make_zip(ccfg.zips_dir, spec.rsna_id)

    gt = np.zeros((4, 4, 4), dtype=np.int64)
    gt[0, 1, 1] = 3
    _write_case_dir(ccfg.gt_root, spec.case_id, gt)

    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    pred[3, 1, 1] = 3  # gt mirrored along axis 0, the L/R axis for this affine
    fake_run = _fake_run_clinical_path_factory(pred, job_id="job_mirrored")

    with caplog.at_level(logging.ERROR):
        row = validate_real_dicom._run_case(spec, ccfg, run_clinical_path_fn=fake_run)

    assert row["status"] == "done"
    assert row["dice_WT"] == pytest.approx(0.0)
    assert row["lateralisation"] == "mirrored"
    assert row["mirrored"] is True
    assert "mirrored" in caplog.text


def test_run_case_refused_study_is_recorded_and_not_scored(tmp_path: Path) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_TEST4", rsna_id="00004", pilot=False)
    _make_zip(ccfg.zips_dir, spec.rsna_id)

    def _fake_run(rsna_study_dir: Path, jobs_out_dir: Path, role_by_folder: dict) -> Path:
        job_id = "job_refused"
        job_dir = jobs_out_dir / job_id
        job_dir.mkdir(parents=True)
        summary = {
            "job_id": job_id,
            "case_id": job_id,
            "state": "refused",
            "stage": "input_qc (pre-preprocessing)",
            "error": "Input QC refused before clinical preprocessing: thick slices.",
            "gatekeeper_decision": None,
            "wall_s": 3.4,
        }
        summary_path = job_dir / "summary.json"
        write_json(summary, summary_path)
        return summary_path

    row = validate_real_dicom._run_case(spec, ccfg, run_clinical_path_fn=_fake_run)

    assert row["status"] == "refused"
    assert row["accepted"] is False
    assert row["refusing_stage"] == "input_qc (pre-preprocessing)"
    assert row["gate_decision"] is None
    assert np.isnan(row["dice_WT"])
    assert row["error"] is None  # a refusal is a successful outcome, not a case-level error


def test_run_case_missing_zip_is_recorded_as_failed(tmp_path: Path) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_TEST5", rsna_id="00005", pilot=False)
    # No zip written at all.

    def _unreachable(rsna_study_dir: Path, jobs_out_dir: Path, role_by_folder: dict) -> Path:
        raise AssertionError("run_clinical_path_fn must not be reached with no zip present")

    row = validate_real_dicom._run_case(spec, ccfg, run_clinical_path_fn=_unreachable)

    assert row["status"] == "failed"
    assert row["accepted"] is False
    assert row["error"] is not None


# ---------------------------------------------------------------------------
# Fix 1: a "done" case whose SCORING raises is never a silent failure
# ---------------------------------------------------------------------------


def test_run_case_done_scoring_failure_keeps_pipeline_outcome_but_scored_false(
    tmp_path: Path,
) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_NO_GT", rsna_id="00006", pilot=False)
    _make_zip(ccfg.zips_dir, spec.rsna_id)
    # Deliberately do NOT write a ground-truth case dir under ccfg.gt_root --
    # np.load on the missing label.npy raises inside _run_case's scoring block.

    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    fake_run = _fake_run_clinical_path_factory(pred, job_id="job_no_gt")

    row = validate_real_dicom._run_case(spec, ccfg, run_clinical_path_fn=fake_run)

    # The pipeline's own outcome is untouched by the scoring failure.
    assert row["status"] == "done"
    assert row["accepted"] is True
    assert row["gate_decision"] == "proceed"
    # But it was never actually scored.
    assert row["scored"] is False
    assert row["error"] is not None
    assert np.isnan(row["dice_WT"])
    assert np.isnan(row["dice_TC"])


def test_summarise_excludes_scoring_failed_from_usable_rate_and_reports_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # "b" is accepted (gate said PROCEED) but was never scored -- under the OLD
    # (pre-fix) logic its NaN Dice would compare False under label_usable and get
    # counted as accepted-and-unusable, i.e. a fabricated silent failure.
    per_case = pd.DataFrame(
        [
            {
                "case_id": "a",
                "pilot": False,
                "status": "done",
                "refusing_stage": None,
                "accepted": True,
                "scored": True,
                "dice_ET": 0.9,
                "dice_TC": 0.9,
                "dice_WT": 0.9,
            },
            {
                "case_id": "b",
                "pilot": False,
                "status": "done",
                "refusing_stage": None,
                "accepted": True,
                "scored": False,
                "dice_ET": float("nan"),
                "dice_TC": float("nan"),
                "dice_WT": float("nan"),
            },
        ]
    )
    ccfg = _make_ccfg(tmp_path, research_per_case=tmp_path / "does_not_exist.csv")

    with caplog.at_level(logging.WARNING):
        summary = validate_real_dicom.summarise(per_case, ccfg)

    assert summary["n_scoring_failed"] == 1
    # "b" excluded from BOTH the numerator and the denominator -- n=1, not 2.
    assert summary["usable_rate"]["n"] == 1
    assert summary["usable_rate"]["n_accepted"] == 1
    assert summary["usable_rate"]["p_accepted_and_usable"] == pytest.approx(1.0)
    assert summary["usable_rate"]["p_silent_failure"] == pytest.approx(0.0)
    assert "could not be scored" in caplog.text
    # completion still counts "b" under status="done" -- the pipeline itself
    # genuinely completed for that case.
    assert summary["completion"]["by_status"]["done"] == 2


# ---------------------------------------------------------------------------
# Fix 2: a pilot geometry mismatch aborts the whole run before the sample cases
# ---------------------------------------------------------------------------


def test_check_pilot_geometry_raises_on_mismatch() -> None:
    pilot_spec = CaseSpec(case_id="PILOT", rsna_id="00000", pilot=True)
    row = {"geometry_mismatch": "reoriented affine does not match reference_affine"}

    with pytest.raises(RuntimeError, match="geometry mismatch"):
        validate_real_dicom._check_pilot_geometry(pilot_spec, row)


def test_check_pilot_geometry_noop_when_no_mismatch() -> None:
    pilot_spec = CaseSpec(case_id="PILOT", rsna_id="00000", pilot=True)
    validate_real_dicom._check_pilot_geometry(pilot_spec, {"geometry_mismatch": None})


def test_check_pilot_geometry_noop_for_non_pilot_even_with_mismatch() -> None:
    spec = CaseSpec(case_id="X", rsna_id="00001", pilot=False)
    validate_real_dicom._check_pilot_geometry(spec, {"geometry_mismatch": "boom"})


def test_score_all_cases_aborts_before_sample_cases_on_pilot_geometry_mismatch(
    tmp_path: Path,
) -> None:
    ccfg = _make_ccfg(tmp_path)
    pilot_spec = CaseSpec(case_id="PILOT", rsna_id="00000", pilot=True)
    sample_spec = CaseSpec(case_id="SAMPLE", rsna_id="00001", pilot=False)

    calls: list[str] = []

    def _fake_run_case(spec_: CaseSpec, ccfg_: RealDicomValidationConfig) -> dict:
        calls.append(spec_.case_id)
        if spec_.pilot:
            row = validate_real_dicom._blank_row(spec_)
            row["status"] = "done"
            row["accepted"] = True
            row["geometry_mismatch"] = "reoriented affine does not match reference_affine"
            row["error"] = "to_reference_grid: ..."
            return row
        raise AssertionError("the sample case must never run after a pilot geometry abort")

    with pytest.raises(RuntimeError, match="geometry mismatch"):
        validate_real_dicom._score_all_cases(
            [pilot_spec, sample_spec], ccfg, run_case_fn=_fake_run_case
        )

    assert calls == ["PILOT"]  # the sample case was never reached


def test_score_all_cases_runs_every_case_when_pilot_geometry_ok(tmp_path: Path) -> None:
    ccfg = _make_ccfg(tmp_path)
    pilot_spec = CaseSpec(case_id="PILOT", rsna_id="00000", pilot=True)
    sample_spec = CaseSpec(case_id="SAMPLE", rsna_id="00001", pilot=False)

    def _fake_run_case(spec_: CaseSpec, ccfg_: RealDicomValidationConfig) -> dict:
        row = validate_real_dicom._blank_row(spec_)
        row["status"] = "done"
        row["accepted"] = True
        row["scored"] = True
        return row

    rows = validate_real_dicom._score_all_cases(
        [pilot_spec, sample_spec], ccfg, run_case_fn=_fake_run_case
    )

    assert [r["case_id"] for r in rows] == ["PILOT", "SAMPLE"]


# ---------------------------------------------------------------------------
# Fix 3: result.json writes are atomic; an unparseable result.json is "absent"
# ---------------------------------------------------------------------------


def test_process_case_writes_result_atomically_no_tmp_leftover(tmp_path: Path) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_ATOMIC", rsna_id="00007", pilot=False)

    def _fake_run_case(spec_: CaseSpec, ccfg_: RealDicomValidationConfig) -> dict:
        return {"case_id": spec_.case_id, "status": "done", "pilot": False}

    validate_real_dicom.process_case(spec, ccfg, run_case_fn=_fake_run_case)

    result_file = validate_real_dicom.result_path(ccfg.work_dir, spec.case_id)
    tmp_file = result_file.parent / f"{result_file.name}.tmp"
    assert result_file.is_file()
    assert not tmp_file.exists()
    assert json.loads(result_file.read_text())["status"] == "done"


def test_process_case_treats_unparseable_result_as_absent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ccfg = _make_ccfg(tmp_path)
    spec = CaseSpec(case_id="BraTS2021_CORRUPT", rsna_id="00008", pilot=False)
    result_file = validate_real_dicom.result_path(ccfg.work_dir, spec.case_id)
    result_file.parent.mkdir(parents=True)
    result_file.write_text("{not valid json")  # simulates a half-written / corrupt file

    calls: list[str] = []

    def _fake_run_case(spec_: CaseSpec, ccfg_: RealDicomValidationConfig) -> dict:
        calls.append(spec_.case_id)
        return {"case_id": spec_.case_id, "status": "done", "pilot": False}

    with caplog.at_level(logging.WARNING):
        row = validate_real_dicom.process_case(spec, ccfg, run_case_fn=_fake_run_case)

    assert calls == [spec.case_id]  # re-run, NOT skipped
    assert row["status"] == "done"
    assert "unparseable" in caplog.text
    assert result_file.is_file()
    assert not (result_file.parent / f"{result_file.name}.tmp").exists()
    assert json.loads(result_file.read_text())["status"] == "done"
