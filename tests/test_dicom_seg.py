"""Tests for neurovision.reporting.dicom_seg.

The pure-logic tests (segment definitions, the class-map adapter, the
geometry check) use only numpy/torch and the real composed Hydra config, and
run in the project's main `.venv`. The I/O-layer tests that actually build a
DICOM Segmentation object are guarded with `pytest.importorskip("highdicom")`,
following the idiom `tests/test_dicom_ingest.py` already uses for `pydicom`,
so they skip cleanly in `.venv` and only run in `.venv-clinical`.

Everything here is synthetic: no real patient data, no real BraTS data.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from neurovision.reporting.dicom_seg import (
    SEGMENT_DEFINITIONS,
    GeometryCheck,
    check_geometry_against_source,
    classes_from_regions,
    segment_masks,
    write_dicom_seg,
)
from neurovision.utils.io import read_yaml

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# `hydra` and `neurovision.inference.postprocess` (which imports `monai`) are
# both absent from `.venv-clinical` (see dicom_seg.py's lazy import of
# regions_to_classes) -- so neither is imported at this file's module scope.
# Only the two tests that actually need one of them import it locally, and
# each is guarded with `pytest.importorskip` so it skips cleanly instead of
# breaking collection of every OTHER test in this file when run there.


def _compose_config(tmp_path: Path, overrides: list[str] | None = None) -> Any:
    """Composes the real Hydra config, mirroring tests/test_tta.py's helper."""
    hydra = pytest.importorskip("hydra")
    all_overrides = [f"data.root_dir={tmp_path}", "device=cpu"] + (overrides or [])
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        return hydra.compose(config_name="config", overrides=all_overrides)


def _dicom_seg_cfg_namespace() -> Any:
    """A `cfg.clinical.dicom_seg`-shaped `SimpleNamespace`, read straight off

    `configs/clinical/default.yaml` -- the REAL values, including the real
    disclaimer string, but without needing hydra/omegaconf (not installed in
    `.venv-clinical`). Mirrors `tests/test_dicom_ingest.py`'s `ingest_cfg`
    `SimpleNamespace` fixture, which uses the same idiom for the same reason.
    `write_dicom_seg` only ever reads plain attributes off `cfg`, so this
    stands in for a composed `DictConfig` exactly as well as the real thing.
    """
    config_path = Path(__file__).resolve().parent.parent / "configs" / "clinical" / "default.yaml"
    dicom_seg_raw = read_yaml(config_path)["dicom_seg"]
    return SimpleNamespace(clinical=SimpleNamespace(dicom_seg=SimpleNamespace(**dicom_seg_raw)))


def _nested_regions(shape: tuple[int, int, int] = (4, 5, 5)) -> np.ndarray:
    """Builds a small, properly nested (ET subset TC subset WT) region array."""
    et = np.zeros(shape, dtype=np.float32)
    tc = np.zeros(shape, dtype=np.float32)
    wt = np.zeros(shape, dtype=np.float32)

    wt[1:3, 1:4, 1:4] = 1.0
    tc[1:3, 1:3, 1:3] = 1.0
    et[1:2, 1:2, 1:2] = 1.0

    return np.stack([et, tc, wt], axis=0)


# ---------------------------------------------------------------------------
# 1. classes_from_regions matches postprocess.regions_to_classes exactly.
# ---------------------------------------------------------------------------


def test_classes_from_regions_matches_postprocess() -> None:
    pytest.importorskip("monai")  # postprocess.py imports monai at its module top
    from neurovision.inference.postprocess import regions_to_classes

    regions = _nested_regions()

    adapter_result = classes_from_regions(regions)
    reference_result = regions_to_classes(torch.as_tensor(regions, dtype=torch.float32)).numpy()

    assert adapter_result.dtype == np.uint8
    np.testing.assert_array_equal(adapter_result, reference_result.astype(np.uint8))


# ---------------------------------------------------------------------------
# 2. classes_from_regions rejects bad input.
# ---------------------------------------------------------------------------


def test_classes_from_regions_rejects_wrong_channel_count() -> None:
    bad = np.zeros((2, 4, 5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(3, D, H, W\)"):
        classes_from_regions(bad)


def test_classes_from_regions_rejects_non_binary_values() -> None:
    regions = _nested_regions()
    regions[0, 0, 0, 0] = 0.7  # a soft (sigmoid) value, not thresholded yet
    with pytest.raises(ValueError, match="non-binary"):
        classes_from_regions(regions)


# ---------------------------------------------------------------------------
# 3. The decisive one: segment numbers are stable when a class is absent.
# ---------------------------------------------------------------------------


def test_segment_numbers_are_stable_when_a_class_is_absent() -> None:
    full_case = _nested_regions()  # NCR, edema and ET all present

    no_et = full_case.copy()
    no_et[0] = 0.0  # zero out the ET channel entirely -- 2.6% of BraTS cases

    full_masks = segment_masks(classes_from_regions(full_case))
    no_et_masks = segment_masks(classes_from_regions(no_et))

    ncr_label = "Necrotic and non-enhancing tumour core"
    edema_label = "Peritumoral edema"
    et_label = "Enhancing tumour"

    # SEGMENT_DEFINITIONS itself is a fixed module constant -- the numbers
    # never move, whatever is present in any given case.
    assert {d.label: d.number for d in SEGMENT_DEFINITIONS} == {
        ncr_label: 1,
        edema_label: 2,
        et_label: 3,
    }

    def present_dicom_numbers(masks: dict[int, np.ndarray]) -> dict[str, int]:
        # Mirrors write_dicom_seg's own "which segments actually get
        # written, at what contiguous DICOM position" logic -- reimplemented
        # here (rather than imported) so this test needs no highdicom.
        present = [d for d in SEGMENT_DEFINITIONS if masks[d.class_value].any()]
        return {d.label: i + 1 for i, d in enumerate(present)}

    full_numbers = present_dicom_numbers(full_masks)
    no_et_numbers = present_dicom_numbers(no_et_masks)

    assert et_label in full_numbers
    assert et_label not in no_et_numbers  # ET dropped, not zero-filled

    # The decisive assertion: dropping the trailing (ET) segment must not
    # renumber the segments before it. A number that shifted when a case
    # happens to lack enhancing tumour would make two studies' SEG objects
    # incomparable.
    assert full_numbers[ncr_label] == no_et_numbers[ncr_label] == 1
    assert full_numbers[edema_label] == no_et_numbers[edema_label] == 2


# ---------------------------------------------------------------------------
# 4. segment_masks partitions the class map.
# ---------------------------------------------------------------------------


def test_segment_masks_partition_the_class_map() -> None:
    class_map = classes_from_regions(_nested_regions())
    masks = segment_masks(class_map)

    union = np.zeros_like(class_map, dtype=bool)
    for value_a, mask_a in masks.items():
        union |= mask_a
        for value_b, mask_b in masks.items():
            if value_a == value_b:
                continue
            assert not np.any(mask_a & mask_b), (value_a, value_b)

    np.testing.assert_array_equal(union, class_map != 0)


# ---------------------------------------------------------------------------
# 5-7. check_geometry_against_source.
# ---------------------------------------------------------------------------


def test_geometry_check_reports_every_reason() -> None:
    result = check_geometry_against_source(
        mask_shape=(10, 240, 240),
        mask_spacing_mm=(1.0, 1.0, 1.0),
        source_shape=(12, 256, 256),
        source_spacing_mm=(1.2, 1.0, 1.0),
    )
    assert isinstance(result, GeometryCheck)
    assert result.ok is False
    assert len(result.reasons) == 2
    assert any("shape" in r for r in result.reasons)
    assert any("spacing" in r for r in result.reasons)


def test_geometry_check_spacing_tolerance() -> None:
    atol = 1e-3

    just_inside = check_geometry_against_source(
        mask_shape=(10, 10, 10),
        mask_spacing_mm=(1.0, 1.0, 1.0),
        source_shape=(10, 10, 10),
        source_spacing_mm=(1.0 + atol / 2, 1.0, 1.0),
        spacing_atol=atol,
    )
    assert just_inside.ok is True

    just_outside = check_geometry_against_source(
        mask_shape=(10, 10, 10),
        mask_spacing_mm=(1.0, 1.0, 1.0),
        source_shape=(10, 10, 10),
        source_spacing_mm=(1.0 + atol * 2, 1.0, 1.0),
        spacing_atol=atol,
    )
    assert just_outside.ok is False
    assert len(just_outside.reasons) == 1


def test_geometry_check_ok_on_a_match() -> None:
    result = check_geometry_against_source(
        mask_shape=(155, 240, 240),
        mask_spacing_mm=(1.0, 1.0, 1.0),
        source_shape=(155, 240, 240),
        source_spacing_mm=(1.0, 1.0, 1.0),
    )
    assert result.ok is True
    assert result.reasons == ()


# ---------------------------------------------------------------------------
# 8. cfg.clinical.dicom_seg is reachable at the composed path, cfg.dicom_seg
#    is not -- the exact trap CLAUDE.md and this spec both call out.
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_at_the_composed_path(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path)

    assert "dicom_seg" not in cfg  # top level: must NOT exist there
    dicom_seg_cfg = cfg.clinical.dicom_seg  # composed one level deeper

    for key in (
        "series_description",
        "series_number",
        "manufacturer",
        "manufacturer_model_name",
        "software_versions",
        "device_serial_number",
        "segmentation_type",
    ):
        assert key in dicom_seg_cfg, key

    assert dicom_seg_cfg.segmentation_type == "BINARY"
    assert "RESEARCH ONLY" in dicom_seg_cfg.series_description


# ---------------------------------------------------------------------------
# I/O layer -- guarded. Each test starts with its own `pytest.importorskip`,
# so only these skip in `.venv` (no highdicom) while every test above still
# runs -- a module-level importorskip would (wrongly) skip the whole file.
# ---------------------------------------------------------------------------


_ROWS = 6
_COLS = 6
_N_SLICES = 5


def _make_source_datasets(n_slices: int = _N_SLICES, spacing_mm: float = 1.0) -> list[Any]:
    """Builds a small, self-consistent synthetic DICOM source series.

    `n_slices` single-frame MR datasets, header-only (no real pixel data
    needed by this module -- `write_dicom_seg` never reads source pixels),
    with the Patient/Study/Frame-of-Reference/geometry attributes highdicom
    requires to build a Segmentation referencing them.
    """
    import numpy as _np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_reference_uid = generate_uid()

    datasets = []
    for i in range(n_slices):
        file_meta = FileMetaDataset()
        # The REAL MR Image Storage SOP Class UID, not generate_uid(). highdicom
        # 0.28.1 looks this value up in its own sop_class_iod_map while copying
        # Patient/Study attributes from the source series, and a syntactically
        # valid but non-standard UID raises KeyError there. With a random UID
        # every write_dicom_seg test failed in .venv-clinical -- i.e. the only
        # environment where they run at all -- so the round-trip was never
        # actually exercised anywhere.
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset("source", {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "MR"
        ds.PatientID = "SYNTHETIC"
        ds.PatientName = "Synthetic^Test"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.StudyDate = "20260101"
        ds.StudyTime = "000000"
        ds.AccessionNumber = ""
        ds.StudyID = ""
        ds.SeriesNumber = 1
        ds.InstanceNumber = i + 1
        ds.Rows = _ROWS
        ds.Columns = _COLS
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [spacing_mm, spacing_mm]
        ds.SliceThickness = spacing_mm
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * spacing_mm]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelData = _np.zeros((_ROWS, _COLS), dtype=_np.uint16).tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        datasets.append(ds)

    return datasets


def _regions_matching_source(n_slices: int = _N_SLICES, include_et: bool = True) -> np.ndarray:
    """Region array whose spatial shape exactly matches `_make_source_datasets`."""
    shape = (n_slices, _ROWS, _COLS)
    et = np.zeros(shape, dtype=np.float32)
    tc = np.zeros(shape, dtype=np.float32)
    wt = np.zeros(shape, dtype=np.float32)

    wt[1:4, 1:5, 1:5] = 1.0
    tc[1:4, 1:4, 1:4] = 1.0
    if include_et:
        et[1:3, 1:3, 1:3] = 1.0

    return np.stack([et, tc, wt], axis=0)


# ---------------------------------------------------------------------------
# 9. Round-trip: write, then read back with highdicom.
# ---------------------------------------------------------------------------


def test_write_dicom_seg_round_trips(tmp_path: Path) -> None:
    hd = pytest.importorskip("highdicom")

    cfg = _dicom_seg_cfg_namespace()
    sources = _make_source_datasets()
    regions = _regions_matching_source(include_et=True)
    out_path = tmp_path / "seg" / "case.dcm"

    written_path = write_dicom_seg(cfg, regions, sources, out_path)
    assert written_path == out_path
    assert out_path.is_file()

    seg = hd.seg.segread(str(out_path))

    assert len(seg.SegmentSequence) == 3
    labels = {s.SegmentLabel for s in seg.SegmentSequence}
    assert labels == {d.label for d in SEGMENT_DEFINITIONS}

    designators = {
        s.SegmentedPropertyTypeCodeSequence[0].CodingSchemeDesignator for s in seg.SegmentSequence
    }
    assert designators == {"SCT"}


# ---------------------------------------------------------------------------
# 10. Refuses on geometry mismatch, and writes NOTHING.
# ---------------------------------------------------------------------------


def test_write_dicom_seg_refuses_on_geometry_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("highdicom")

    cfg = _dicom_seg_cfg_namespace()
    sources = _make_source_datasets(n_slices=_N_SLICES)
    # One slice too few for the mask -- a plain shape mismatch.
    regions = _regions_matching_source(n_slices=_N_SLICES + 1, include_et=True)
    out_path = tmp_path / "seg" / "mismatch.dcm"

    with pytest.raises(ValueError, match="shape mismatch"):
        write_dicom_seg(cfg, regions, sources, out_path)

    assert not out_path.exists()
    assert not out_path.parent.exists() or not any(out_path.parent.iterdir())


# ---------------------------------------------------------------------------
# 11. The disclaimer is a required field, verbatim.
# ---------------------------------------------------------------------------


def test_series_description_carries_the_disclaimer(tmp_path: Path) -> None:
    hd = pytest.importorskip("highdicom")

    cfg = _dicom_seg_cfg_namespace()
    sources = _make_source_datasets()
    regions = _regions_matching_source(include_et=True)
    out_path = tmp_path / "seg" / "disclaimer.dcm"

    write_dicom_seg(cfg, regions, sources, out_path)
    seg = hd.seg.segread(str(out_path))

    assert str(cfg.clinical.dicom_seg.series_description) in str(seg.SeriesDescription)
    assert str(cfg.clinical.dicom_seg.disclaimer) in str(seg.ImageComments)


# ---------------------------------------------------------------------------
# 14. An over-long series_description raises rather than being silently
#     truncated, and nothing is written.
# ---------------------------------------------------------------------------


def test_over_long_series_description_raises_rather_than_truncating(tmp_path: Path) -> None:
    pytest.importorskip("highdicom")

    cfg = _dicom_seg_cfg_namespace()
    long_description = "x" * 80  # 80 > 64, the DICOM LO value representation's limit
    cfg.clinical.dicom_seg.series_description = long_description
    sources = _make_source_datasets()
    regions = _regions_matching_source(include_et=True)
    out_path = tmp_path / "seg" / "too_long.dcm"

    with pytest.raises(ValueError, match=r"80.*64"):
        write_dicom_seg(cfg, regions, sources, out_path)

    assert not out_path.exists()
    assert not out_path.parent.exists() or not any(out_path.parent.iterdir())


# ---------------------------------------------------------------------------
# 15. The real composed config's series_description is within the DICOM
#     limit -- the tripwire that catches someone lengthening it later.
# ---------------------------------------------------------------------------


def test_config_series_description_is_within_the_dicom_limit(tmp_path: Path) -> None:
    cfg = _compose_config(tmp_path)
    series_description = cfg.clinical.dicom_seg.series_description
    assert len(series_description) <= 64, (
        f"cfg.clinical.dicom_seg.series_description is {len(series_description)} characters, "
        "over the DICOM LO value representation's 64-character limit -- write_dicom_seg will "
        "raise at write time. Shorten it and move any additional text to "
        "cfg.clinical.dicom_seg.disclaimer."
    )


# ---------------------------------------------------------------------------
# 12. An all-empty class map raises.
# ---------------------------------------------------------------------------


def test_empty_class_map_raises(tmp_path: Path) -> None:
    pytest.importorskip("highdicom")

    cfg = _dicom_seg_cfg_namespace()
    sources = _make_source_datasets()
    empty_regions = np.zeros((3, _N_SLICES, _ROWS, _COLS), dtype=np.float32)
    out_path = tmp_path / "seg" / "empty.dcm"

    with pytest.raises(ValueError, match="no tumour voxels"):
        write_dicom_seg(cfg, empty_regions, sources, out_path)

    assert not out_path.exists()


# ---------------------------------------------------------------------------
# 13. A case with no enhancing tumour writes two segments; the omission is
#     logged rather than the segment being zero-filled.
# ---------------------------------------------------------------------------


def test_absent_segment_is_omitted_not_zero_filled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    hd = pytest.importorskip("highdicom")

    cfg = _dicom_seg_cfg_namespace()
    sources = _make_source_datasets()
    regions = _regions_matching_source(include_et=False)
    out_path = tmp_path / "seg" / "no_et.dcm"

    with caplog.at_level("INFO", logger="neurovision.reporting.dicom_seg"):
        write_dicom_seg(cfg, regions, sources, out_path)

    seg = hd.seg.segread(str(out_path))
    assert len(seg.SegmentSequence) == 2
    labels = {s.SegmentLabel for s in seg.SegmentSequence}
    assert labels == {"Necrotic and non-enhancing tumour core", "Peritumoral edema"}
    assert "Enhancing tumour" not in labels

    assert any(
        "omitting" in record.message and "Enhancing tumour" in record.message
        for record in caplog.records
    )
