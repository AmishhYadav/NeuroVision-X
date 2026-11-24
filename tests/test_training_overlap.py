"""Tests for `app.backend.training_overlap` and its `/api/clinical/jobs` wiring (W6.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from app.backend import api, clinical_jobs, config, training_overlap
from fastapi.testclient import TestClient

from tests.test_app_clinical_api import _valid_study_zip

REAL_REGISTRY = config.REPO_ROOT / "knowledge" / "known_training_overlap.yaml"
LISTED_UID = "1.3.6.1.4.1.14519.5.2.1.74959210626421264963078150067043317963"


def _entry(source_id: str, uids: list[str]) -> dict:
    return {
        "source_id": source_id,
        "brats_id": f"BraTS_{source_id}",
        "split": "train",
        "status": "confirmed",
        "evidence": "test",
        "series_instance_uids": uids,
    }


def _ingest(outcome: str = "assigned", uid: str = "uid-a") -> dict:
    return {"assignments": {uid: {"role": "t1", "outcome": outcome}}}


def test_match_assigned_uid() -> None:
    result = training_overlap.match_training_overlap(_ingest(), [_entry("A", ["uid-a", "uid-z"])])
    assert result is not None
    assert result["source_id"] == "A"
    assert result["matched_series"] == ["uid-a"]


def test_match_ignores_rejected_series() -> None:
    assert (
        training_overlap.match_training_overlap(_ingest("rejected"), [_entry("A", ["uid-a"])])
        is None
    )


@pytest.mark.parametrize("ingest", [None, {}, {"assignments": {}}, {"paths": {}}])
def test_match_no_assignments(ingest: dict | None) -> None:
    assert training_overlap.match_training_overlap(ingest, [_entry("A", ["uid-a"])]) is None


def test_match_second_entry_wins() -> None:
    registry = [_entry("A", ["other"]), _entry("B", ["uid-a"])]
    result = training_overlap.match_training_overlap(_ingest(), registry)
    assert result is not None
    assert result["source_id"] == "B"
    assert result["brats_id"] == "BraTS_B"


def test_load_ok(tmp_path: Path) -> None:
    path = tmp_path / "reg.yaml"
    path.write_text(yaml.safe_dump({"entries": [_entry("A", ["u"])]}))
    assert training_overlap.load_overlap_registry(path)[0]["source_id"] == "A"


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert training_overlap.load_overlap_registry(tmp_path / "nope.yaml") == []


def test_load_malformed_raises(tmp_path: Path) -> None:
    no_entries = tmp_path / "a.yaml"
    no_entries.write_text("foo: 1\n")
    with pytest.raises(ValueError, match="entries"):
        training_overlap.load_overlap_registry(no_entries)
    no_uids = tmp_path / "b.yaml"
    no_uids.write_text(yaml.safe_dump({"entries": [{"source_id": "A"}]}))
    with pytest.raises(ValueError, match="series_instance_uids"):
        training_overlap.load_overlap_registry(no_uids)


def test_real_registry_loads() -> None:
    entries = training_overlap.load_overlap_registry(REAL_REGISTRY)
    assert len(entries) == 2
    by_id = {e["source_id"]: e for e in entries}
    assert by_id["UPENN-GBM-00002"]["status"] == "confirmed"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("NVX_PREP_DIR", str(tmp_path / "prep"))
    monkeypatch.setenv("NVX_EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("NVX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NVX_JOB_DIR", str(tmp_path / "jobs"))
    (tmp_path / "prep").mkdir()
    (tmp_path / "eval").mkdir()
    clinical_jobs._CLINICAL_JOBS.clear()
    config.get_settings.cache_clear()
    yield TestClient(api.create_app())
    clinical_jobs._CLINICAL_JOBS.clear()
    config.get_settings.cache_clear()


def _job_with_uid(uid: str) -> str:
    job = clinical_jobs.create_clinical_job(config.get_settings(), _valid_study_zip())
    job.ingest_result = {"assignments": {uid: {"role": "t1", "outcome": "assigned"}}}
    return job.job_id


def test_api_flags_overlapping_job(client: TestClient) -> None:
    job_id = _job_with_uid(LISTED_UID)
    body = client.get(f"/api/clinical/jobs/{job_id}").json()
    assert body["training_overlap"]["brats_id"] == "BraTS2021_01202"
    assert body["training_overlap"]["matched_series"] == [LISTED_UID]


def test_api_unflagged_job_is_null(client: TestClient) -> None:
    job_id = _job_with_uid("9.9.9.not.listed")
    body = client.get(f"/api/clinical/jobs/{job_id}").json()
    assert body["training_overlap"] is None
