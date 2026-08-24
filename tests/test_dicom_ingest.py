"""Tests for neurovision.data.dicom_ingest.

The pure-logic tests (rule table, `assign_roles`) use only the plain
`SeriesHeader` dataclass and run in the project's main `.venv` -- no
`pydicom` involved. The I/O-layer tests are guarded with
`pytest.importorskip("pydicom")`, following the idiom already used in
`tests/test_lesionwise_metrics.py`, so they skip cleanly in `.venv` and only
run in `.venv-clinical`.

Everything here is synthetic: no real patient data, no real BraTS data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from neurovision.data.dicom_ingest import (
    AMBIGUITY_MARGIN,
    ROLES,
    IngestResult,
    RoleAssignment,
    SeriesHeader,
    SeriesOutcome,
    assign_roles,
    classify_series,
    ingest_study,
    normalise_tokens,
    resolve_dcm2niix,
)


def _header(
    series_uid: str = "1.2.3",
    series_number: int | None = 1,
    series_description: str = "",
    protocol_name: str = "",
    sequence_name: str = "",
    scanning_sequence: tuple[str, ...] = (),
    sequence_variant: tuple[str, ...] = (),
    scan_options: tuple[str, ...] = (),
    image_type: tuple[str, ...] = ("ORIGINAL", "PRIMARY"),
    echo_time: float | None = None,
    repetition_time: float | None = None,
    inversion_time: float | None = None,
    contrast_agent: str = "",
    n_instances: int = 20,
) -> SeriesHeader:
    """Build a `SeriesHeader` with sensible defaults, overriding what a test cares about."""
    return SeriesHeader(
        series_uid=series_uid,
        series_number=series_number,
        series_description=series_description,
        protocol_name=protocol_name,
        sequence_name=sequence_name,
        scanning_sequence=scanning_sequence,
        sequence_variant=sequence_variant,
        scan_options=scan_options,
        image_type=image_type,
        echo_time=echo_time,
        repetition_time=repetition_time,
        inversion_time=inversion_time,
        contrast_agent=contrast_agent,
        n_instances=n_instances,
    )


# ---------------------------------------------------------------------------
# 1. normalise_tokens
# ---------------------------------------------------------------------------


def test_normalise_tokens_splits_on_non_alphanumerics() -> None:
    tokens = normalise_tokens("AX T1 POST +C")
    assert "t1" in tokens
    assert "c" in tokens
    assert "post" in tokens

    # "T1CE" has no separator inside it -- it must survive as ONE token,
    # never decompose into "t1" and "ce" (the substring trap CLAUDE.md warns
    # about: a naive rule would read "t1" out of "t1ce" via `in`).
    t1ce_tokens = normalise_tokens("T1CE")
    assert t1ce_tokens == ("t1ce",)
    assert "t1" not in t1ce_tokens


# ---------------------------------------------------------------------------
# 2. T1CE is never classified as plain T1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "AX T1 POST GD",
        "T1 +C",
        "T1CE",
        "T1 C+",
        "t1_ce_ax",
        "MPRAGE POST",
    ],
)
def test_t1ce_is_not_classified_as_t1(description: str) -> None:
    header = _header(series_description=description)
    assignment = classify_series(header)
    assert assignment.role == "t1ce", (description, assignment)


# ---------------------------------------------------------------------------
# 3. "T2 FLAIR" is FLAIR, never T2.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description",
    ["AX T2 FLAIR", "T2-FLAIR", "FLAIR", "dark fluid"],
)
def test_t2_flair_is_flair_not_t2(description: str) -> None:
    header = _header(series_description=description)
    assignment = classify_series(header)
    assert assignment.role == "flair", (description, assignment)


# ---------------------------------------------------------------------------
# 4. Plain T1 and T2 classify correctly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "expected_role"),
    [
        ("AX T1", "t1"),
        ("MPRAGE", "t1"),
        ("AX T2", "t2"),
        ("T2 TSE", "t2"),
    ],
)
def test_plain_t1_and_t2_classify_correctly(description: str, expected_role: str) -> None:
    header = _header(series_description=description)
    assignment = classify_series(header)
    assert assignment.role == expected_role, (description, assignment)


# ---------------------------------------------------------------------------
# 5. contrast_agent field alone promotes T1 -> T1CE.
# ---------------------------------------------------------------------------


def test_contrast_agent_field_alone_promotes_t1_to_t1ce() -> None:
    plain = _header(series_description="AX T1", contrast_agent="")
    contrast = _header(series_description="AX T1", contrast_agent="Gadovist")

    assert classify_series(plain).role == "t1"
    assert classify_series(contrast).role == "t1ce"


# ---------------------------------------------------------------------------
# 6. Geometry-only classification -- empty description, TE/TR/TI alone.
# ---------------------------------------------------------------------------


def test_geometry_only_classification() -> None:
    t1_geometry = _header(echo_time=8.0, repetition_time=500.0)
    t2_geometry = _header(echo_time=100.0, repetition_time=3000.0)
    flair_geometry = _header(inversion_time=1800.0, scanning_sequence=("IR",))

    assert classify_series(t1_geometry).role == "t1"
    assert classify_series(t2_geometry).role == "t2"
    assert classify_series(flair_geometry).role == "flair"


# ---------------------------------------------------------------------------
# 7. Localizers and derived series are rejected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        _header(image_type=("ORIGINAL", "PRIMARY", "LOCALIZER")),
        _header(image_type=("DERIVED", "SECONDARY")),
        _header(series_description="AX LOCALIZER"),
        _header(series_description="Scout"),
        _header(series_description="Calibration scan"),
    ],
)
def test_localizer_and_derived_are_rejected(header: SeriesHeader) -> None:
    assignment = classify_series(header)
    assert assignment.role is None


# ---------------------------------------------------------------------------
# 8. Diffusion and perfusion series are rejected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description",
    ["AX DWI", "ADC MAP", "DTI TRACE", "PERFUSION DSC", "ASL", "SWI", "MRA", "TOF", "BOLD fMRI"],
)
def test_diffusion_and_perfusion_are_rejected(description: str) -> None:
    header = _header(series_description=description)
    assignment = classify_series(header)
    assert assignment.role is None, (description, assignment)


# ---------------------------------------------------------------------------
# 9. Too few instances is rejected (assign_roles level -- it needs the
#    min_instances threshold, which classify_series alone does not know).
# ---------------------------------------------------------------------------


def test_too_few_instances_is_rejected() -> None:
    header = _header(series_description="AX T1", n_instances=2)
    role_headers, assignments, rejected, _warnings = assign_roles([header], min_instances=8)

    assert role_headers == {}
    assert rejected == ((header.series_uid, "too few instances (2 < 8)"),)
    assert assignments[header.series_uid].role is None


# ---------------------------------------------------------------------------
# 10. assign_roles picks one series per role.
# ---------------------------------------------------------------------------


def test_assign_roles_picks_one_series_per_role() -> None:
    headers = [
        _header(series_uid="uid-t1", series_description="AX T1"),
        _header(series_uid="uid-t1ce", series_description="AX T1 POST GD"),
        _header(series_uid="uid-t2", series_description="AX T2"),
        _header(series_uid="uid-flair", series_description="AX T2 FLAIR"),
    ]

    role_headers, assignments, rejected, warnings = assign_roles(headers, min_instances=8)

    assert set(role_headers) == set(ROLES)
    assert role_headers["t1"].series_uid == "uid-t1"
    assert role_headers["t1ce"].series_uid == "uid-t1ce"
    assert role_headers["t2"].series_uid == "uid-t2"
    assert role_headers["flair"].series_uid == "uid-flair"
    assert rejected == ()
    assert len(assignments) == 4


# ---------------------------------------------------------------------------
# 11. The tie-break is deterministic and independent of input order.
# ---------------------------------------------------------------------------


def test_assign_roles_tie_break_is_deterministic() -> None:
    # Identical descriptions -> identical scores -> identical n_instances,
    # so the documented tie-break falls through to "lowest series_number".
    low_number = _header(
        series_uid="uid-b", series_description="AX T1", series_number=2, n_instances=20
    )
    high_number = _header(
        series_uid="uid-a", series_description="AX T1", series_number=9, n_instances=20
    )

    forward = assign_roles([low_number, high_number], min_instances=8)
    reversed_order = assign_roles([high_number, low_number], min_instances=8)

    assert forward[0]["t1"].series_uid == "uid-b"
    # Unchanged when the input list is reversed -- this is what actually
    # proves the tie-break does not depend on input order.
    assert reversed_order[0]["t1"].series_uid == "uid-b"


# ---------------------------------------------------------------------------
# 12. Missing roles are reported, not raised.
# ---------------------------------------------------------------------------


def test_assign_roles_reports_missing_roles_without_raising() -> None:
    headers = [
        _header(series_uid="uid-t1", series_description="AX T1"),
        _header(series_uid="uid-t2", series_description="AX T2"),
    ]

    role_headers, _assignments, _rejected, _warnings = assign_roles(headers, min_instances=8)

    assert set(role_headers) == {"t1", "t2"}
    missing_roles = tuple(role for role in ROLES if role not in role_headers)
    assert missing_roles == ("t1ce", "flair")


# ---------------------------------------------------------------------------
# 13. Empty input does not raise.
# ---------------------------------------------------------------------------


def test_assign_roles_empty_input_does_not_raise() -> None:
    role_headers, assignments, rejected, warnings = assign_roles([], min_instances=8)

    assert role_headers == {}
    assert assignments == {}
    assert rejected == ()
    assert warnings != ()


# ---------------------------------------------------------------------------
# 14. Overrides take precedence and warn when they displace a winner.
# ---------------------------------------------------------------------------


def test_overrides_take_precedence_and_warn() -> None:
    headers = [
        _header(series_uid="uid-t1-auto", series_description="AX T1"),
        _header(series_uid="uid-t1-manual", series_description="AX LOCALIZER"),  # rejected
    ]

    role_headers, _assignments, _rejected, warnings = assign_roles(
        headers, min_instances=8, overrides={"uid-t1-manual": "t1"}
    )

    assert role_headers["t1"].series_uid == "uid-t1-manual"
    assert any("displaced" in w and "uid-t1-auto" in w for w in warnings)


# ---------------------------------------------------------------------------
# 15. Overrides reject an unknown uid or an unknown role.
# ---------------------------------------------------------------------------


def test_overrides_reject_unknown_uid_and_unknown_role() -> None:
    headers = [_header(series_uid="uid-t1", series_description="AX T1")]

    with pytest.raises(ValueError, match="unknown series_uid"):
        assign_roles(headers, min_instances=8, overrides={"does-not-exist": "t1"})

    with pytest.raises(ValueError, match="unknown role"):
        assign_roles(headers, min_instances=8, overrides={"uid-t1": "not_a_role"})


# ---------------------------------------------------------------------------
# 16. An ambiguous series is reported, not guessed.
# ---------------------------------------------------------------------------


def test_ambiguous_series_is_reported_not_guessed() -> None:
    # A contrived description carrying both a T1 token and a T2 token gives
    # two roles the exact same non-zero score -- margin 0.0, well under
    # AMBIGUITY_MARGIN, so classify_series must refuse to guess.
    header = _header(series_description="T1 T2")
    assignment = classify_series(header)

    assert assignment.role is None
    assert assignment.outcome is SeriesOutcome.AMBIGUOUS
    assert "t1" in assignment.reasons[0] and "t2" in assignment.reasons[0]
    assert assignment.score > 0.0


def test_ambiguity_margin_is_a_positive_constant() -> None:
    # Sanity check on the constant itself, since several tests above rely
    # on it being strictly between the "support" weight and the "token"
    # weight.
    assert AMBIGUITY_MARGIN > 0.0


# ---------------------------------------------------------------------------
# 16b. `outcome is ASSIGNED` iff `role is not None` -- the invariant the
#      SeriesOutcome enum exists to preserve, across every path that can
#      produce a None role plus the one path that assigns a role.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("description", "image_type"),
    [
        # Rejected outright: a localiser image type.
        ("AX LOCALIZER", ("ORIGINAL", "PRIMARY", "LOCALIZER")),
        # Ambiguous: T1 and T2 tokens score identically.
        ("T1 T2", ("ORIGINAL", "PRIMARY")),
        # No evidence at all: nothing in the rule table matches.
        ("", ("ORIGINAL", "PRIMARY")),
        # Cleanly assigned: an unambiguous T1 description.
        ("AX T1", ("ORIGINAL", "PRIMARY")),
    ],
)
def test_outcome_is_assigned_iff_role_is_not_none(
    description: str, image_type: tuple[str, ...]
) -> None:
    header = _header(series_description=description, image_type=image_type)
    assignment = classify_series(header)

    assert (assignment.outcome is SeriesOutcome.ASSIGNED) == (assignment.role is not None)


# ---------------------------------------------------------------------------
# 16c. The manifest records the outcome for every series, not just the role.
#      Mirrors test_ingest_study_writes_a_manifest below, but exercises the
#      manifest-building helper directly with synthetic RoleAssignment
#      values, so it runs even without pydicom or a real dcm2niix binary.
# ---------------------------------------------------------------------------


def test_manifest_records_the_outcome(tmp_path: Path) -> None:
    from neurovision.data.dicom_ingest import _build_manifest

    assignments = {
        "uid-assigned": RoleAssignment(
            role="t1",
            score=3.0,
            reasons=("description token(s) ['t1'] indicate T1-weighted",),
            outcome=SeriesOutcome.ASSIGNED,
        ),
        "uid-rejected": RoleAssignment(
            role=None,
            score=0.0,
            reasons=("image_type contains ['LOCALIZER']",),
            outcome=SeriesOutcome.REJECTED,
        ),
        "uid-ambiguous": RoleAssignment(
            role=None,
            score=3.0,
            reasons=("t1 vs t2; margin 0.00 < 1.5",),
            outcome=SeriesOutcome.AMBIGUOUS,
        ),
        "uid-no-evidence": RoleAssignment(
            role=None,
            score=0.0,
            reasons=("no rule matched any of ROLES",),
            outcome=SeriesOutcome.NO_EVIDENCE,
        ),
    }

    manifest = _build_manifest(
        study_dir=tmp_path / "study",
        out_dir=tmp_path / "out",
        paths={"t1": tmp_path / "out" / "t1.nii.gz"},
        missing_roles=("t1ce", "t2", "flair"),
        assignments=assignments,
        rejected=(("uid-rejected", "image_type contains ['LOCALIZER']"),),
        warnings=(),
    )

    assert manifest["series"]["uid-assigned"]["outcome"] == "assigned"
    assert manifest["series"]["uid-rejected"]["outcome"] == "rejected"
    assert manifest["series"]["uid-ambiguous"]["outcome"] == "ambiguous"
    assert manifest["series"]["uid-no-evidence"]["outcome"] == "no_evidence"
    # The outcome is serialised as its plain string value, not as `repr`
    # (which would leak "SeriesOutcome.REJECTED" into the audit trail).
    for entry in manifest["series"].values():
        assert isinstance(entry["outcome"], str)
        assert "SeriesOutcome" not in entry["outcome"]


# ---------------------------------------------------------------------------
# Round-trip sanity on the dataclasses (cheap, but catches an API drift).
# ---------------------------------------------------------------------------


def test_role_assignment_and_ingest_result_are_frozen_dataclasses() -> None:
    assignment = RoleAssignment(
        role="t1", score=3.0, reasons=("x",), outcome=SeriesOutcome.ASSIGNED
    )
    with pytest.raises(AttributeError):
        assignment.role = "t2"  # type: ignore[misc]

    result = IngestResult(
        paths={},
        assignments={},
        missing_roles=ROLES,
        rejected=(),
        warnings=(),
    )
    with pytest.raises(AttributeError):
        result.missing_roles = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# I/O layer -- each test below starts with its own `pytest.importorskip`, so
# ONLY these tests skip in `.venv` (no pydicom) while every pure-logic test
# above still runs. A module-level `importorskip` would skip the whole file,
# which is exactly the "guarded test that skips everywhere" CLAUDE.md warns
# against -- it would also (wrongly) skip every test above this point.
# ---------------------------------------------------------------------------


# A fixed StudyInstanceUID shared by every synthetic file this module
# writes -- realistic enough (one study, several series) without needing
# every test to plumb a study UID through.
_SYNTHETIC_STUDY_UID = "1.2.840.99999.1"


def _write_synthetic_dicom(
    path: Path,
    *,
    series_uid: str,
    series_number: int,
    series_description: str,
    echo_time: float | None = None,
    repetition_time: float | None = None,
    instance_number: int = 1,
) -> None:
    """Write a minimal DICOM file that `pydicom` AND `dcm2niix` both accept.

    `dcm2niix` (unlike a bare `pydicom.dcmread`) refuses a file that has no
    pixel data or image geometry -- "No valid DICOM images were found" --
    so this needs a real, if tiny, image: a 4x4 uint16 slice, MR modality,
    and enough geometry (position/orientation/spacing) for dcm2niix to
    stack multiple instances into one volume.
    """
    import numpy as np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID = _SYNTHETIC_STUDY_UID
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.SeriesDescription = series_description
    ds.Modality = "MR"
    if echo_time is not None:
        ds.EchoTime = echo_time
    if repetition_time is not None:
        ds.RepetitionTime = repetition_time

    # Minimal image geometry: a 4x4 single-slice image, one per instance,
    # stacked along Z by InstanceNumber.
    ds.Rows = 4
    ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pixels = np.full((4, 4), fill_value=instance_number, dtype=np.uint16)
    ds.PixelData = pixels.tobytes()

    # Transfer syntax already comes from file_meta above; setting the two
    # legacy is_little_endian/is_implicit_VR attributes as well is
    # deprecated in recent pydicom and unneeded.
    ds.save_as(str(path), enforce_file_format=True)


# ---------------------------------------------------------------------------
# 17. read_series_headers groups instances by SeriesInstanceUID.
# ---------------------------------------------------------------------------


def test_read_series_headers_groups_by_series_uid(tmp_path: Path) -> None:
    pytest.importorskip("pydicom")
    from neurovision.data.dicom_ingest import read_series_headers

    t1_uid = "1.2.3.1"
    t2_uid = "1.2.3.2"

    for i in range(3):
        _write_synthetic_dicom(
            tmp_path / f"t1_{i:03d}.dcm",
            series_uid=t1_uid,
            series_number=1,
            series_description="AX T1",
            echo_time=8.0,
            repetition_time=500.0,
            instance_number=i + 1,
        )
    for i in range(5):
        _write_synthetic_dicom(
            tmp_path / f"t2_{i:03d}.dcm",
            series_uid=t2_uid,
            series_number=2,
            series_description="AX T2",
            echo_time=100.0,
            repetition_time=3000.0,
            instance_number=i + 1,
        )

    headers = read_series_headers(tmp_path)
    headers_by_uid = {h.series_uid: h for h in headers}

    assert set(headers_by_uid) == {t1_uid, t2_uid}
    assert headers_by_uid[t1_uid].n_instances == 3
    assert headers_by_uid[t2_uid].n_instances == 5
    assert headers_by_uid[t1_uid].series_description == "AX T1"
    assert headers_by_uid[t1_uid].echo_time == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 18. resolve_dcm2niix raises an actionable error when the binary is absent.
#     Does not need pydicom, but lives here since it is the I/O layer.
# ---------------------------------------------------------------------------


def test_resolve_dcm2niix_raises_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="requirements-clinical.txt"):
        resolve_dcm2niix(None)


# ---------------------------------------------------------------------------
# 19. ingest_study writes a manifest. Needs the real dcm2niix binary, so it
#     is additionally skipped when that binary is not resolvable on PATH.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("dcm2niix") is None, reason="dcm2niix binary not on PATH")
def test_ingest_study_writes_a_manifest(tmp_path: Path) -> None:
    pytest.importorskip("pydicom")
    from types import SimpleNamespace

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    out_dir = tmp_path / "out"

    t1_uid = "1.2.3.10"
    for i in range(10):
        _write_synthetic_dicom(
            study_dir / f"t1_{i:03d}.dcm",
            series_uid=t1_uid,
            series_number=1,
            series_description="AX T1",
            echo_time=8.0,
            repetition_time=500.0,
            instance_number=i + 1,
        )

    # `ingest_study` only ever reads attributes off `cfg`, so a plain
    # SimpleNamespace stands in for a real Hydra `DictConfig` here without
    # needing omegaconf installed in `.venv-clinical`.
    ingest_cfg = SimpleNamespace(dcm2niix_path=None, min_instances=8, role_overrides={})
    cfg = SimpleNamespace(clinical=SimpleNamespace(ingest=ingest_cfg))

    result = ingest_study(cfg, study_dir, out_dir)

    assert "t1" in result.paths
    assert result.paths["t1"].is_file()
    manifest_path = out_dir / "ingest_manifest.json"
    assert manifest_path.is_file()

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["roles_written"]["t1"] == str(result.paths["t1"])
    assert t1_uid in manifest["series"]
    assert manifest["series"][t1_uid]["role"] == "t1"
    assert manifest["series"][t1_uid]["outcome"] == "assigned"
    assert set(manifest["missing_roles"]) == {"t1ce", "t2", "flair"}
