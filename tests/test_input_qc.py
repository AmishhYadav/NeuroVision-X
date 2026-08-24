"""Tests for `neurovision.inference.input_qc`.

Milestone 4, Phase E, tasks E3 (input QC gate) and E4 (missing-sequence
refusal). Everything here is synthetic -- tiny numpy arrays and, where a
file must exist on disk, tiny nibabel-written NIfTIs -- never real patient
or BraTS data, and every test runs in well under a second on CPU.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from neurovision.data.dicom_ingest import ROLES
from neurovision.inference import input_qc
from neurovision.inference.input_qc import (
    CHECK_IDS,
    Finding,
    InputQCReport,
    Severity,
    VolumeInfo,
    check_brain_mask,
    check_finite_values,
    check_geometry_consistency,
    check_intensity_sanity,
    check_sequence_completeness,
    check_shape_against_expected,
    check_spacing,
    describe_volume,
    load_volume_infos,
    run_input_qc,
)

# Real configs/ directory, resolved relative to this file -- same pattern as
# tests/test_clinical_preprocess.py and tests/test_train_qc.py, so the
# "reachable at the composed path" test composes the PROJECT's actual
# config, not a hand-built stand-in.
_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

SHAPE = (8, 8, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eye_affine(spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """A diagonal affine with the given per-axis voxel spacing, mm."""
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    return affine


def _brain_like(shape: tuple[int, int, int] = SHAPE, value: float = 100.0) -> np.ndarray:
    """A volume with a nonzero 'brain' interior and a zero background border."""
    data = np.zeros(shape, dtype=np.float32)
    data[1:-1, 1:-1, 1:-1] = value
    # A little texture so percentiles/MAD are not all identical.
    rng = np.random.default_rng(0)
    interior = data[1:-1, 1:-1, 1:-1]
    data[1:-1, 1:-1, 1:-1] = interior + rng.uniform(-5.0, 5.0, size=interior.shape).astype(
        np.float32
    )
    return data


def _make_volume_info(
    role: str,
    *,
    shape: tuple[int, int, int] = SHAPE,
    affine: np.ndarray | None = None,
    data: np.ndarray | None = None,
    brain_mask: np.ndarray | None = None,
) -> VolumeInfo:
    if affine is None:
        affine = _eye_affine()
    if data is None:
        data = _brain_like(shape)
    return describe_volume(role, data, affine, brain_mask=brain_mask)


def _four_role_volumes(
    shape: tuple[int, int, int] = SHAPE,
    affine: np.ndarray | None = None,
    brain_mask: np.ndarray | None = None,
) -> dict[str, VolumeInfo]:
    affine = _eye_affine() if affine is None else affine
    return {
        role: _make_volume_info(role, shape=shape, affine=affine, brain_mask=brain_mask)
        for role in ROLES
    }


def _make_cfg(**overrides: object) -> SimpleNamespace:
    """A minimal fake config matching configs/clinical/default.yaml's `input_qc:` block.

    Plain `SimpleNamespace`, the same pattern used in
    tests/test_clinical_preprocess.py's `_make_cfg` -- only attribute access
    is required by this module's functions.
    """
    defaults = dict(
        required_roles=list(ROLES),
        affine_atol=1.0e-4,
        spacing_min_mm=0.3,
        spacing_max_mm=3.0,
        anisotropy_warn_ratio=2.0,
        anisotropy_refuse_ratio=6.0,
        expected_shape=list(SHAPE),
        brain_volume_min_ml=0.05,
        brain_volume_max_ml=5.0,
        min_nonzero_fraction_in_brain=0.5,
        skull_present_warn_fraction=0.45,
        min_dynamic_range=1.0e-6,
    )
    defaults.update(overrides)
    qc = SimpleNamespace(**defaults)
    return SimpleNamespace(clinical=SimpleNamespace(input_qc=qc))


def _brain_mask(shape: tuple[int, int, int] = SHAPE) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:-1, 1:-1, 1:-1] = 1
    return mask


def _finding_by_check(findings: tuple[Finding, ...], check: str) -> Finding:
    matches = [f for f in findings if f.check == check]
    assert len(matches) == 1, f"expected exactly one {check!r} finding, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# 1. Structural label-free guard
# ---------------------------------------------------------------------------


def test_no_function_in_this_module_takes_a_label() -> None:
    forbidden = {"label", "labels", "gt", "ground_truth", "y_true", "target"}
    checked = 0
    for name, obj in inspect.getmembers(input_qc, inspect.isfunction):
        if name.startswith("_") or obj.__module__ != input_qc.__name__:
            continue
        checked += 1
        params = set(inspect.signature(obj).parameters)
        offending = params & forbidden
        assert not offending, f"{name} takes forbidden parameter(s) {offending}"
    assert checked > 0


# ---------------------------------------------------------------------------
# 2-3. Sequence completeness
# ---------------------------------------------------------------------------


def test_missing_sequence_refuses_and_names_it() -> None:
    volumes = {r: _make_volume_info(r) for r in ROLES if r != "flair"}
    finding = check_sequence_completeness(volumes.keys(), ROLES)
    assert finding.severity is Severity.REFUSE
    assert "flair" in finding.message.lower()


def test_complete_study_passes() -> None:
    mask = _brain_mask()
    volumes = _four_role_volumes(brain_mask=mask)
    cfg = _make_cfg()
    report = run_input_qc(cfg, volumes, brain_mask=mask)
    assert report.verdict is Severity.OK
    assert all(f.severity is Severity.OK for f in report.findings)


# ---------------------------------------------------------------------------
# 4-5. Geometry consistency
# ---------------------------------------------------------------------------


def test_geometry_mismatch_refuses() -> None:
    volumes = _four_role_volumes()
    volumes["t2"] = _make_volume_info("t2", shape=(6, 6, 6))
    findings = check_geometry_consistency(volumes, affine_atol=1.0e-4)
    geometry = _finding_by_check(findings, "geometry_consistency")
    assert geometry.severity is Severity.REFUSE
    assert "t2" in geometry.message

    volumes2 = _four_role_volumes()
    shifted = _eye_affine()
    shifted[0, 3] = 5.0  # translate far past affine_atol
    volumes2["flair"] = _make_volume_info("flair", affine=shifted)
    findings2 = check_geometry_consistency(volumes2, affine_atol=1.0e-4)
    geometry2 = _finding_by_check(findings2, "geometry_consistency")
    assert geometry2.severity is Severity.REFUSE
    assert "flair" in geometry2.message


def test_single_modality_geometry_check_is_ok_not_an_error() -> None:
    volumes = {"t1": _make_volume_info("t1")}
    findings = check_geometry_consistency(volumes, affine_atol=1.0e-4)
    geometry = _finding_by_check(findings, "geometry_consistency")
    assert geometry.severity is Severity.OK


# ---------------------------------------------------------------------------
# 6. Finite values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nan_or_inf_refuses(bad_value: float) -> None:
    data = _brain_like()
    data[2, 2, 2] = bad_value
    volumes = {"t1": describe_volume("t1", data, _eye_affine())}
    (finding,) = check_finite_values(volumes)
    assert finding.severity is Severity.REFUSE
    assert "t1" in finding.message


# ---------------------------------------------------------------------------
# 7. Singular affine
# ---------------------------------------------------------------------------


def test_singular_affine_refuses() -> None:
    singular = _eye_affine()
    singular[2, :] = 0.0  # zero out a row -> determinant 0
    volumes = {"t1": _make_volume_info("t1", affine=singular)}
    findings = check_geometry_consistency(volumes, affine_atol=1.0e-4)
    affine_finding = _finding_by_check(findings, "affine_invertible")
    assert affine_finding.severity is Severity.REFUSE
    assert "t1" in affine_finding.message


# ---------------------------------------------------------------------------
# 8. Spacing and anisotropy
# ---------------------------------------------------------------------------


def test_spacing_out_of_range_refuses() -> None:
    cfg = _make_cfg()
    too_fine = _make_volume_info("t1", affine=_eye_affine((0.1, 0.1, 0.1)))
    finding, _ = check_spacing({"t1": too_fine}, cfg.clinical.input_qc)
    assert finding.severity is Severity.REFUSE
    assert "t1" in finding.message

    too_coarse = _make_volume_info("t1", affine=_eye_affine((5.0, 5.0, 5.0)))
    finding2, _ = check_spacing({"t1": too_coarse}, cfg.clinical.input_qc)
    assert finding2.severity is Severity.REFUSE


@pytest.mark.parametrize(
    ("spacing", "expected_severity"),
    [
        ((1.0, 1.0, 3.0), Severity.WARN),  # ratio 3.0: between warn (2.0) and refuse (6.0)
        ((1.0, 1.0, 8.0), Severity.REFUSE),  # ratio 8.0: at/above refuse (6.0)
    ],
)
def test_anisotropy_warns_then_refuses(
    spacing: tuple[float, float, float], expected_severity: Severity
) -> None:
    cfg = _make_cfg(spacing_min_mm=0.1, spacing_max_mm=10.0)
    vi = _make_volume_info("t1", affine=_eye_affine(spacing))
    _, aniso = check_spacing({"t1": vi}, cfg.clinical.input_qc)
    assert aniso.check == "anisotropy"
    assert aniso.severity is expected_severity


# ---------------------------------------------------------------------------
# 9. Expected shape
# ---------------------------------------------------------------------------


def test_unexpected_shape_warns_but_does_not_refuse() -> None:
    volumes = {"t1": _make_volume_info("t1", shape=(6, 6, 6))}
    (finding,) = check_shape_against_expected(volumes, SHAPE)
    assert finding.severity is Severity.WARN


# ---------------------------------------------------------------------------
# 10-11. Dynamic range
# ---------------------------------------------------------------------------


def test_constant_and_empty_volumes_refuse() -> None:
    cfg = _make_cfg()
    zero_volume = describe_volume("t1", np.zeros(SHAPE, dtype=np.float32), _eye_affine())
    (finding,) = check_intensity_sanity({"t1": zero_volume}, cfg.clinical.input_qc)
    assert finding.severity is Severity.REFUSE

    constant_data = np.full(SHAPE, 7.0, dtype=np.float32)
    constant_volume = describe_volume("t1", constant_data, _eye_affine())
    (finding2,) = check_intensity_sanity({"t1": constant_volume}, cfg.clinical.input_qc)
    assert finding2.severity is Severity.REFUSE


def test_dynamic_range_is_scale_invariant() -> None:
    cfg = _make_cfg()
    data = _brain_like()
    vi = describe_volume("t1", data, _eye_affine())
    vi_scaled = describe_volume("t1", data * 1000.0, _eye_affine())

    (finding,) = check_intensity_sanity({"t1": vi}, cfg.clinical.input_qc)
    (finding_scaled,) = check_intensity_sanity({"t1": vi_scaled}, cfg.clinical.input_qc)
    assert finding.severity == finding_scaled.severity is Severity.OK

    ratio = finding.detail["per_role"]["t1"]["ratio"]
    ratio_scaled = finding_scaled.detail["per_role"]["t1"]["ratio"]
    # float32 arithmetic (both `_brain_like` and the *1000 scaling stay
    # float32) accumulates enough rounding noise across ~500 voxels that an
    # exact-to-1e-6 comparison is too strict; 1e-3 relative is still far
    # tighter than any real severity-flipping difference would need.
    assert ratio == pytest.approx(ratio_scaled, rel=1e-3)


# ---------------------------------------------------------------------------
# 12-14. Brain mask
# ---------------------------------------------------------------------------


def test_brain_volume_bounds() -> None:
    cfg = _make_cfg(brain_volume_min_ml=0.05, brain_volume_max_ml=5.0)
    spacing = (1.0, 1.0, 1.0)

    too_small = np.zeros(SHAPE, dtype=np.uint8)
    too_small[0, 0, 0] = 1  # 1 voxel = 0.001 mL, well under min
    volumes_small = _four_role_volumes(brain_mask=too_small)
    findings_small = check_brain_mask(too_small, spacing, volumes_small, cfg.clinical.input_qc)
    volume_finding_small = _finding_by_check(findings_small, "brain_volume")
    assert volume_finding_small.severity is Severity.REFUSE

    too_large = np.ones(SHAPE, dtype=np.uint8)  # all 512 voxels, still tiny in mL at 1mm^3
    # Use a coarse spacing to push volume above the max bound.
    volumes_large = _four_role_volumes(brain_mask=too_large)
    findings_large = check_brain_mask(
        too_large, (10.0, 10.0, 10.0), volumes_large, cfg.clinical.input_qc
    )
    volume_finding_large = _finding_by_check(findings_large, "brain_volume")
    assert volume_finding_large.severity is Severity.REFUSE

    plausible = _brain_mask()  # 6^3 = 216 voxels
    volumes_ok = _four_role_volumes(brain_mask=plausible)
    findings_ok = check_brain_mask(plausible, (1.5, 1.5, 1.5), volumes_ok, cfg.clinical.input_qc)
    volume_finding_ok = _finding_by_check(findings_ok, "brain_volume")
    assert volume_finding_ok.severity is Severity.OK


def test_zero_mask_refuses_without_dividing_by_zero() -> None:
    cfg = _make_cfg()
    volumes = _four_role_volumes()
    zero_mask = np.zeros(SHAPE, dtype=np.uint8)
    findings = check_brain_mask(zero_mask, (1.0, 1.0, 1.0), volumes, cfg.clinical.input_qc)
    volume_finding = _finding_by_check(findings, "brain_volume")
    fraction_finding = _finding_by_check(findings, "nonzero_in_brain")
    assert volume_finding.severity is Severity.REFUSE
    assert fraction_finding.severity is Severity.OK  # visibly skipped, not a crash


def test_mask_shape_mismatch_refuses() -> None:
    cfg = _make_cfg()
    # volumes are built against a correctly-shaped mask (so n_nonzero_in_mask
    # is populated, and check_brain_mask does not raise); the mask actually
    # passed to run_input_qc below is a DIFFERENT, wrongly-shaped array --
    # exercising the geometry_consistency shape-mismatch path, not the
    # caller-error guard in check_brain_mask.
    volumes = _four_role_volumes(brain_mask=_brain_mask())
    mismatched_mask = np.ones((4, 4, 4), dtype=np.uint8)
    report = run_input_qc(cfg, volumes, brain_mask=mismatched_mask)
    geometry = _finding_by_check(report.findings, "geometry_consistency")
    assert geometry.severity is Severity.REFUSE
    assert "brain_mask" in geometry.message.lower() or "mask" in geometry.message.lower()
    assert report.verdict is Severity.REFUSE


def test_missing_mask_is_skipped_visibly() -> None:
    cfg = _make_cfg()
    volumes = _four_role_volumes()
    report = run_input_qc(cfg, volumes, brain_mask=None)
    volume_finding = _finding_by_check(report.findings, "brain_volume")
    fraction_finding = _finding_by_check(report.findings, "nonzero_in_brain")
    assert volume_finding.severity is Severity.OK
    assert "unavailable" in volume_finding.message.lower()
    assert fraction_finding.severity is Severity.OK
    assert "unavailable" in fraction_finding.message.lower()


def test_skull_present_warns() -> None:
    cfg = _make_cfg(skull_present_warn_fraction=0.1)
    data = np.ones(SHAPE, dtype=np.float32) * 50.0  # 100% nonzero -> above 0.1
    volumes = {"t1": describe_volume("t1", data, _eye_affine())}
    findings = check_brain_mask(None, (1.0, 1.0, 1.0), volumes, cfg.clinical.input_qc)
    skull_finding = _finding_by_check(findings, "skull_present")
    assert skull_finding.severity is Severity.WARN


# ---------------------------------------------------------------------------
# 15-16. nonzero_in_brain is a true voxel-wise intersection, not a whole-
# volume approximation
# ---------------------------------------------------------------------------


def test_nonzero_in_brain_is_a_true_intersection() -> None:
    """A modality nonzero ONLY outside the mask must REFUSE.

    Old (buggy) behaviour compared a modality's WHOLE-VOLUME nonzero count
    against the mask's voxel count, so a channel that is nonzero everywhere
    except inside the brain mask -- exactly backwards -- would still pass:
    total nonzero voxels here (296, the border shell) comfortably clears
    `min_nonzero_fraction_in_brain` (0.5) against the mask's 216 voxels
    (296 / 216 ~= 1.37 >= 0.5). The true intersection is zero, so the fixed
    check must REFUSE.
    """
    cfg = _make_cfg()  # default min_nonzero_fraction_in_brain=0.5
    mask = _brain_mask()  # interior 6x6x6 = 216 voxels, the "brain"
    mask_bool = mask.astype(bool)

    data = np.zeros(SHAPE, dtype=np.float32)
    data[~mask_bool] = 50.0  # nonzero ONLY on the border shell, outside the mask

    vi = describe_volume("t1", data, _eye_affine(), brain_mask=mask)
    # Sanity-check the construction: whole-volume nonzero count alone would
    # have passed the old approximation, while the true intersection is 0.
    assert vi.n_nonzero == int((~mask_bool).sum()) == 296
    assert vi.n_nonzero_in_mask == 0
    old_approx_fraction = vi.n_nonzero / int(mask_bool.sum())
    assert old_approx_fraction >= cfg.clinical.input_qc.min_nonzero_fraction_in_brain

    findings = check_brain_mask(mask, (1.0, 1.0, 1.0), {"t1": vi}, cfg.clinical.input_qc)
    fraction_finding = _finding_by_check(findings, "nonzero_in_brain")
    assert fraction_finding.severity is Severity.REFUSE
    assert fraction_finding.detail["in_brain_fractions"]["t1"] == 0.0


def test_describe_volume_rejects_a_mask_of_the_wrong_shape() -> None:
    data = _brain_like()  # shape (8, 8, 8)
    wrong_shaped_mask = np.ones((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\(4, 4, 4\).*\(8, 8, 8\)"):
        describe_volume("t1", data, _eye_affine(), brain_mask=wrong_shaped_mask)


def test_check_brain_mask_raises_when_infos_lack_the_mask() -> None:
    """No quiet fallback: a mask passed to check_brain_mask, but not to the
    describe_volume calls that built `volumes`, must raise -- not silently
    approximate with the whole-volume count.
    """
    cfg = _make_cfg()
    mask = _brain_mask()
    volumes = _four_role_volumes()  # built WITHOUT this mask
    with pytest.raises(ValueError, match="n_nonzero_in_mask"):
        check_brain_mask(mask, (1.0, 1.0, 1.0), volumes, cfg.clinical.input_qc)


# ---------------------------------------------------------------------------
# 17-18. Report composition
# ---------------------------------------------------------------------------


def test_verdict_is_the_worst_severity() -> None:
    ok = Finding("a", Severity.OK, "ok", {})
    warn = Finding("b", Severity.WARN, "warn", {})
    refuse = Finding("c", Severity.REFUSE, "refuse", {})

    mixed = InputQCReport(
        verdict=input_qc._worst_severity((ok.severity, warn.severity, refuse.severity)),
        findings=(ok, warn, refuse),
    )
    assert mixed.verdict is Severity.REFUSE

    # Alphabetically "ok" < "refuse" < "warn" -- if the ordering were ever
    # computed by string comparison instead of the explicit rank table, this
    # case (OK + WARN, no REFUSE present) would risk giving the wrong
    # answer by accident. Assert the correct one explicitly.
    ok_warn = input_qc._worst_severity((ok.severity, warn.severity))
    assert ok_warn is Severity.WARN


def test_report_to_dict_is_json_serialisable() -> None:
    cfg = _make_cfg()
    mask = _brain_mask()
    volumes = _four_role_volumes(brain_mask=mask)
    report = run_input_qc(cfg, volumes, brain_mask=mask)
    payload = report.to_dict()
    serialised = json.dumps(payload)
    assert "verdict" in json.loads(serialised)


# ---------------------------------------------------------------------------
# 19. Deterministic reference role
# ---------------------------------------------------------------------------


def test_geometry_reference_role_is_deterministic() -> None:
    v_a = _make_volume_info("t1")
    v_b = _make_volume_info("t1ce")
    v_c = _make_volume_info("flair")

    order1 = {"flair": v_c, "t1": v_a, "t1ce": v_b}
    order2 = {"t1ce": v_b, "t1": v_a, "flair": v_c}

    findings1 = check_geometry_consistency(order1, affine_atol=1.0e-4)
    findings2 = check_geometry_consistency(order2, affine_atol=1.0e-4)

    geo1 = _finding_by_check(findings1, "geometry_consistency")
    geo2 = _finding_by_check(findings2, "geometry_consistency")
    assert geo1.detail["reference_role"] == geo2.detail["reference_role"] == "t1"
    assert geo1 == geo2


# ---------------------------------------------------------------------------
# 20. Every declared check id appears
# ---------------------------------------------------------------------------


def test_every_check_emits_a_finding_even_when_it_passes() -> None:
    cfg = _make_cfg()
    mask = _brain_mask()
    volumes = _four_role_volumes(brain_mask=mask)
    report = run_input_qc(cfg, volumes, brain_mask=mask)
    ids_seen = {f.check for f in report.findings}
    assert ids_seen == CHECK_IDS
    assert all(f.severity is Severity.OK for f in report.findings)


# ---------------------------------------------------------------------------
# 21. Real composed config
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_at_the_composed_path() -> None:
    """The REAL project config, composed through Hydra, must expose the
    input QC block at `cfg.clinical.input_qc` -- the exact path
    `run_input_qc` reads.
    """
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config")

    assert "clinical" in cfg
    assert "input_qc" in cfg.clinical
    assert "input_qc" not in cfg  # NOT cfg.input_qc

    qc_cfg = cfg.clinical.input_qc
    expected_keys = {
        "required_roles",
        "affine_atol",
        "spacing_min_mm",
        "spacing_max_mm",
        "anisotropy_warn_ratio",
        "anisotropy_refuse_ratio",
        "expected_shape",
        "brain_volume_min_ml",
        "brain_volume_max_ml",
        "min_nonzero_fraction_in_brain",
        "skull_present_warn_fraction",
        "min_dynamic_range",
    }
    assert expected_keys <= set(qc_cfg.keys())

    # run_input_qc must actually work against the real composed config, not
    # just expose the right keys -- a driver whose ten unit tests pass
    # against a hand-built OmegaConf fixture while the real script dies on
    # its first line is exactly the trap CLAUDE.md records.
    mask = _brain_mask()
    volumes = _four_role_volumes(brain_mask=mask)
    report = run_input_qc(cfg, volumes, brain_mask=mask)
    assert isinstance(report, InputQCReport)


# ---------------------------------------------------------------------------
# 22. load_volume_infos error shape
# ---------------------------------------------------------------------------


def test_load_volume_infos_rejects_a_non_nifti_naming_the_role(tmp_path: Path) -> None:
    bad_path = tmp_path / "t1.nii.gz"
    bad_path.write_bytes(b"not a nifti file")
    with pytest.raises(ValueError, match="t1"):
        load_volume_infos({"t1": bad_path})


def test_load_volume_infos_reads_real_niftis_and_splits_out_the_mask(tmp_path: Path) -> None:
    affine = _eye_affine()
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(_brain_like(), affine), str(t1_path))
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(_brain_mask().astype(np.float32), affine), str(mask_path))

    volumes, mask = load_volume_infos({"t1": t1_path, "brain_mask": mask_path})
    assert set(volumes) == {"t1"}
    assert isinstance(volumes["t1"], VolumeInfo)
    assert mask is not None
    assert mask.shape == SHAPE
