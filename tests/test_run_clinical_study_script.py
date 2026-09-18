"""Tests for `scripts/run_clinical_study.py`, the T0.2 CLI driver.

Everything here is synthetic, small, and CPU-only -- no real DICOM files, no
ANTs/HD-BET, no trained checkpoint. Reuses `tests/test_app_clinical_jobs.py`'s
`_wire_full_pipeline_to_gatekeeper` to fake every heavy stage of the clinical
pipeline up to (and including) the gatekeeper decision, exactly the way that
module's own "reaches done" tests do.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import hydra
import pytest
import scripts.run_clinical_study as run_clinical_study_script
from app.backend import clinical_jobs, inference
from hydra.core.global_hydra import GlobalHydra

from neurovision.inference.gatekeeper import Decision
from tests.test_app_clinical_jobs import _settings, _wire_full_pipeline_to_gatekeeper

# --- zip_directory -------------------------------------------------------


def test_zip_directory_lists_exact_relative_names(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    (study_dir / "SERIES1").mkdir(parents=True)
    (study_dir / "SERIES1" / "IM001.dcm").write_bytes(b"one")
    (study_dir / "SERIES1" / "IM002.dcm").write_bytes(b"two")
    (study_dir / "SERIES2").mkdir(parents=True)
    (study_dir / "SERIES2" / "IM001.dcm").write_bytes(b"three")

    payload = run_clinical_study_script.zip_directory(study_dir)

    with zipfile.ZipFile(zipfile.io.BytesIO(payload)) as archive:
        names = sorted(archive.namelist())

    assert names == [
        "SERIES1/IM001.dcm",
        "SERIES1/IM002.dcm",
        "SERIES2/IM001.dcm",
    ]


def test_zip_directory_missing_study_dir_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_clinical_study_script.zip_directory(tmp_path / "does_not_exist")


def test_zip_directory_no_files_raises_value_error(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_study"
    # A subdirectory with no files in it at all -- still zero regular files
    # to archive, which is exactly the case this guards against.
    (empty_dir / "EMPTY_SERIES").mkdir(parents=True)

    with pytest.raises(ValueError, match="no files"):
        run_clinical_study_script.zip_directory(empty_dir)


# --- scan_artifacts --------------------------------------------------------


def test_scan_artifacts_reports_exactly_which_artifacts_exist(tmp_path: Path) -> None:
    job_dir = tmp_path / "job123"
    case_id = "job123"

    (job_dir / "report").mkdir(parents=True)
    (job_dir / "report" / f"{case_id}.json").write_text("{}")

    (job_dir / "cache" / "neurovision" / "gradcam" / "WT").mkdir(parents=True)
    (job_dir / "cache" / "neurovision" / "gradcam" / "WT" / f"{case_id}.npy").write_bytes(b"x")
    # TC gradcam deliberately absent.

    (job_dir / "cache" / "neurovision" / "logits").mkdir(parents=True)
    (job_dir / "cache" / "neurovision" / "logits" / f"{case_id}.npy").write_bytes(b"x")

    (job_dir / "job.json").write_text("{}")
    # dicom_seg deliberately absent entirely.

    result = run_clinical_study_script.scan_artifacts(job_dir, case_id)

    assert result == {
        "report": True,
        "dicom_seg": False,
        "gradcam_WT": True,
        "gradcam_TC": False,
        "logits": True,
        "job_json": True,
    }


def test_scan_artifacts_empty_logits_dir_is_false(tmp_path: Path) -> None:
    job_dir = tmp_path / "job123"
    case_id = "job123"
    (job_dir / "cache" / "neurovision" / "logits").mkdir(parents=True)

    result = run_clinical_study_script.scan_artifacts(job_dir, case_id)

    assert result["logits"] is False


# --- run_study: end to end, pipeline faked ---------------------------------


def _make_study_dir(tmp_path: Path) -> Path:
    """A tiny placeholder "DICOM study" directory -- content never matters here,
    since every test using this fakes the pipeline stages that would read it."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "IM001.dcm").write_bytes(b"placeholder dicom bytes")
    (study_dir / "sub" / "IM002.dcm").parent.mkdir(parents=True)
    (study_dir / "sub" / "IM002.dcm").write_bytes(b"placeholder dicom bytes")
    return study_dir


def test_run_study_end_to_end_reaches_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study_dir = _make_study_dir(tmp_path)
    out_dir = tmp_path / "clinical_out"
    settings = _settings(tmp_path)

    # Set via monkeypatch (not a bare os.environ[...] = ...) so pytest
    # restores the pre-test value on teardown -- run_study itself also sets
    # this env var unconditionally, but monkeypatch already recorded the
    # PRE-test value the moment setenv below ran, so that later, identical
    # overwrite does not defeat the restore.
    monkeypatch.setenv("NVX_JOB_DIR", str(out_dir))

    _wire_full_pipeline_to_gatekeeper(monkeypatch, tmp_path)

    def _fake_explain_case(job_settings, case_id, region, **kwargs):
        return Path("/fake/gradcam.npy")

    monkeypatch.setattr(inference, "explain_case", _fake_explain_case)

    summary_path = run_clinical_study_script.run_study(study_dir, out_dir, settings=settings)

    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())

    assert summary["state"] == "done"
    assert summary["stage_times"]
    assert summary["artifacts"]["job_json"] is True
    assert summary["wall_s"] >= 0


def test_run_study_sanitizes_nan_in_gatekeeper_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gatekeeper_decision` can carry a real NaN (e.g. an unmeasured signal) --
    the written summary.json must still be strict, parseable JSON with that
    value turned into `null`, not a raised exception or a bare `NaN` token."""
    study_dir = _make_study_dir(tmp_path)
    out_dir = tmp_path / "clinical_out"
    settings = _settings(tmp_path)
    monkeypatch.setenv("NVX_JOB_DIR", str(out_dir))

    _wire_full_pipeline_to_gatekeeper(monkeypatch, tmp_path)
    monkeypatch.setattr(inference, "explain_case", lambda *a, **k: Path("/fake/gradcam.npy"))

    class _NaNDecision:
        """A minimal stand-in for GateDecision whose to_dict() carries a NaN,
        exactly the shape gatekeeper._jsonable does not strip today."""

        decision = Decision.PROCEED

        def to_dict(self):
            return {"decision": "PROCEED", "some_signal": float("nan")}

    monkeypatch.setattr(clinical_jobs, "run_gatekeeper", lambda cfg, signals: _NaNDecision())

    summary_path = run_clinical_study_script.run_study(study_dir, out_dir, settings=settings)

    # json.loads succeeding at all already proves no bare NaN token leaked --
    # Python's own json.loads accepts NaN by default, so the real assertion
    # is the value itself, not merely that parsing did not raise.
    summary = json.loads(summary_path.read_text())
    assert summary["gatekeeper_decision"]["some_signal"] is None
    assert "NaN" not in summary_path.read_text()


def test_poll_stage_captures_final_stage_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage set right before run_clinical_job returns must still be recorded."""
    study_dir = _make_study_dir(tmp_path)
    out_dir = tmp_path / "clinical_out"
    settings = _settings(tmp_path)
    monkeypatch.setenv("NVX_JOB_DIR", str(out_dir))

    _wire_full_pipeline_to_gatekeeper(monkeypatch, tmp_path)
    monkeypatch.setattr(inference, "explain_case", lambda *a, **k: Path("/fake/gradcam.npy"))

    sentinel_stage = "sentinel_final_stage"
    original_run_clinical_job = clinical_jobs.run_clinical_job

    def _wrapped_run_clinical_job(settings_arg, job_id):
        result = original_run_clinical_job(settings_arg, job_id)
        # Mutates the job's stage AFTER the real run_clinical_job has
        # already returned -- run_study's poll thread must still pick this
        # up in its one final, unconditional read after the loop ends.
        clinical_jobs._update_clinical_job(settings_arg, result, stage=sentinel_stage)
        return result

    monkeypatch.setattr(clinical_jobs, "run_clinical_job", _wrapped_run_clinical_job)

    summary_path = run_clinical_study_script.run_study(study_dir, out_dir, settings=settings)
    summary = json.loads(summary_path.read_text())

    assert summary["stage_times"][-1]["stage"] == sentinel_stage


def test_run_study_records_summary_when_run_clinical_job_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """run_clinical_job raising unexpectedly must not lose the run: summary.json
    is still written (with whatever state the job store holds), and the
    traceback is logged at ERROR rather than swallowed silently."""
    study_dir = _make_study_dir(tmp_path)
    out_dir = tmp_path / "clinical_out"
    settings = _settings(tmp_path)
    monkeypatch.setenv("NVX_JOB_DIR", str(out_dir))

    def _raising_run_clinical_job(settings_arg, job_id):
        raise RuntimeError("sentinel: unexpected failure before any stage ran")

    monkeypatch.setattr(clinical_jobs, "run_clinical_job", _raising_run_clinical_job)

    with caplog.at_level(logging.ERROR):
        summary_path = run_clinical_study_script.run_study(study_dir, out_dir, settings=settings)

    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    # run_clinical_job never got a chance to update the job at all, so the
    # job store still holds its pre-run state.
    assert summary["state"] == "queued"
    assert "run_clinical_job raised unexpectedly" in caplog.text


def test_run_study_prepends_interpreter_bin_dir_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_dir = _make_study_dir(tmp_path)
    out_dir = tmp_path / "clinical_out"
    settings = _settings(tmp_path)
    monkeypatch.setenv("NVX_JOB_DIR", str(out_dir))
    monkeypatch.setenv("PATH", "/definitely/not/the/interpreters/bin")

    _wire_full_pipeline_to_gatekeeper(monkeypatch, tmp_path)
    monkeypatch.setattr(inference, "explain_case", lambda *a, **k: Path("/fake/gradcam.npy"))

    run_clinical_study_script.run_study(study_dir, out_dir, settings=settings)

    interpreter_bin_dir = str(Path(sys.executable).resolve().parent)
    assert os.environ["PATH"].startswith(interpreter_bin_dir)


# --- exit_code_for_state ----------------------------------------------------


def test_exit_code_for_state_done_is_zero() -> None:
    assert run_clinical_study_script.exit_code_for_state("done") == 0


def test_exit_code_for_state_refused_is_zero() -> None:
    assert run_clinical_study_script.exit_code_for_state("refused") == 0


def test_exit_code_for_state_failed_is_one() -> None:
    assert run_clinical_study_script.exit_code_for_state("failed") == 1


# --- release_hydra: proves the GlobalHydra-already-initialized failure -----


def test_release_hydra_clears_an_already_initialized_global_hydra() -> None:
    """Reproduces the real failure this helper exists to fix.

    `@hydra.main` leaves `GlobalHydra` initialized for the rest of the
    process. `hydra.initialize_config_dir` used WITHOUT the `with` form
    mirrors that exactly -- the context-manager form clears itself on exit,
    which would never reach the failure condition
    `_compose_clinical_cfg`'s own `hydra.initialize_config_dir` call hits
    (`ValueError: GlobalHydra is already initialized`) once `main` has
    already composed its own `cfg`.
    """
    assert not GlobalHydra.instance().is_initialized()
    try:
        hydra.initialize_config_dir(
            version_base="1.3", config_dir=run_clinical_study_script._CONFIG_DIR
        )
        assert GlobalHydra.instance().is_initialized()

        run_clinical_study_script.release_hydra()

        assert not GlobalHydra.instance().is_initialized()
    finally:
        # Never leak this into another test regardless of how the assertions
        # above went.
        GlobalHydra.instance().clear()
