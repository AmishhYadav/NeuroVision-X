"""Tests for `scripts/real_dicom_v2.py`, the W3.4 real-DICOM v2 driver.

Everything is synthetic and CPU-only: no real DICOM, no ANTs, no clinical pipeline. The two
seams that would reach them are replaced:

- `run_fn` (the clinical path) by `FakeClinical`, which writes a tiny fake job directory
  (summary/job json, NIfTIs, plus junk that must be cleaned up) and records how it was called.
- `carry_fn` (ANTs registration) by `_fake_carry`, which carries the label with the identity.

Statistics are not re-tested here (see `test_real_dicom_v2_endpoints.py`).
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import scripts.real_dicom_v2 as v2
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from neurovision.analysis.carried_reference import CarryResult
from neurovision.analysis.real_dicom_v2_endpoints import REQUIRED_COLUMNS
from neurovision.utils.io import write_json

SHAPE = (8, 8, 6)
AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])  # axis 0 is the left-right axis ("L")
OVERRIDE = {"spacing_max_mm": 1000.0, "anisotropy_refuse_ratio": 1000.0}


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _label() -> np.ndarray:
    """An asymmetric (left-right) tumour, so a mirrored copy is detectable."""
    label = np.zeros(SHAPE, dtype=np.int64)
    label[1:3, 2:5, 2:4] = 1
    label[1:2, 3:4, 3:4] = 3
    return label


def _write_gt(gt_root: Path, case_id: str, label: np.ndarray) -> None:
    """`<gt_root>/<case_id>/{image,label}.npy + meta.json`, uncropped (bbox = whole grid)."""
    case_dir = gt_root / case_id
    case_dir.mkdir(parents=True)
    np.save(case_dir / "label.npy", label)
    image = np.zeros((4, *SHAPE), dtype=np.float32)
    image[:, 1:7, 1:7, 1:5] = 1.0  # a "brain" block
    np.save(case_dir / "image.npy", image)
    write_json(
        {
            "case_id": case_id,
            "original_shape": list(SHAPE),
            "cropped_shape": list(SHAPE),
            "bbox": [[0, s] for s in SHAPE],
            "affine": AFFINE.tolist(),
            "spacing": [1.0, 1.0, 1.0],
        },
        case_dir / "meta.json",
    )


def _save_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, AFFINE), str(path))


def _qc_report(*checks: tuple[str, str]) -> dict[str, Any]:
    """A structured input-QC report, shaped like `InputQCReport.to_dict()`."""
    findings = [
        {"check": c, "severity": s, "message": f"{c} message", "detail": {}} for c, s in checks
    ]
    verdict = "refuse" if any(s == "refuse" for _, s in checks) else "ok"
    return {"verdict": verdict, "findings": findings}


class FakeClinical:
    """Stands in for `run_clinical_path_v2`. `kinds` maps rsna_id -> behaviour.

    Behaviours: "ok" (done), "thick" (input-QC refusal on spacing; done when overridden),
    "other_refusal" (input-QC refusal on geometry), "gate_refused" (gatekeeper refusal: a
    mask exists but no dicom_seg_work file).
    """

    def __init__(self, kinds: dict[str, str], cases: dict[str, str]) -> None:
        self.kinds = kinds
        self.cases = cases  # rsna_id -> case_id
        self.calls: list[tuple[str, dict[str, float] | None]] = []

    def __call__(
        self,
        study_dir: Path,
        jobs_dir: Path,
        role_by_folder: Any,
        qc_override: Any,
    ) -> Path:
        rsna = study_dir.name
        self.calls.append((rsna, None if qc_override is None else dict(qc_override)))
        kind = self.kinds[rsna]
        if kind == "thick" and qc_override is not None:
            kind = "ok"
        job_id = f"job{len(self.calls):03d}"
        job_dir = jobs_dir / job_id
        state, stage, error, gate, qc = "done", "done", None, "proceed", _qc_report()
        if kind == "thick":
            qc = _qc_report(("spacing_range", "refuse"), ("anisotropy", "refuse"))
            state, stage, gate = "refused", "input_qc (pre-preprocessing)", None
            error = "Input QC refused before clinical preprocessing: thick"
        elif kind == "other_refusal":
            qc = _qc_report(("geometry_consistency", "refuse"))
            state, stage, gate = "refused", "input_qc (pre-preprocessing)", None
            error = "Input QC refused before clinical preprocessing: spacing of slices"  # prose!
        elif kind == "gate_refused":
            state, stage, gate = "refused", "gatekeeper", "refuse"
            error = "Gatekeeper refused: x"

        record = {
            "job_id": job_id,
            "case_id": job_id,
            "state": state,
            "stage": stage,
            "error": error,
            "input_qc_pre": qc,
            "gatekeeper_decision": {"decision": gate} if gate else None,
        }
        job_dir.mkdir(parents=True)
        write_json(record, job_dir / "job.json")
        write_json({**record, "wall_s": 1.5}, job_dir / "summary.json")
        # Junk that a real job leaves behind and the driver must delete.
        (job_dir / "raw_dicom").mkdir()
        (job_dir / "raw_dicom" / "big.dcm").write_bytes(b"x" * 1000)
        _save_nifti(
            job_dir / "clinical_prep" / "intermediate" / "atlas__t1ce.nii.gz",
            np.ones(SHAPE, np.float32),
        )
        (job_dir / "prep").mkdir()
        (job_dir / "prep" / "image.npy").write_bytes(b"x")

        if kind in ("ok", "gate_refused"):
            label = np.load(Path(self.cases["_gt_root"]) / self.cases[rsna] / "label.npy")
            t1ce = np.zeros(SHAPE, np.float32)
            t1ce[1:7, 1:7, 1:5] = 1.0
            _save_nifti(job_dir / "clinical_prep" / "t1ce.nii.gz", t1ce)
            _save_nifti(
                job_dir / "clinical_prep" / "brain_mask.nii.gz", (t1ce > 0).astype(np.uint8)
            )
            if kind == "ok":
                _save_nifti(
                    job_dir / "dicom_seg_work" / "atlas_space_mask.nii.gz", label.astype(np.uint8)
                )
        return job_dir / "summary.json"


def _fake_carry(
    moving_image: np.ndarray,
    moving_label: np.ndarray,
    fixed_image: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    seed: int,
) -> CarryResult:
    """Identity 'registration': the label is carried unchanged."""
    return CarryResult(
        carried_label=moving_label.copy(),
        carried_brain_mask=moving_image != 0,
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=0.0,
        transform_params=[],
    )


def _setup(
    tmp_path: Path,
    kinds: dict[str, str],
    *,
    stored_for: tuple[str, ...] = (),
    extra: dict[str, Any] | None = None,
) -> tuple[Any, FakeClinical, Path]:
    """Writes GT, zips, case yaml, job map (+ stored P1.2 masks) and returns (cfg, fake, root)."""
    gt_root = tmp_path / "gt"
    zips = tmp_path / "zips"
    zips.mkdir(parents=True)
    cases = {"_gt_root": str(gt_root)}
    entries = []
    for i, rsna in enumerate(kinds):
        case_id = f"BraTS2021_{i:05d}"
        cases[rsna] = case_id
        _write_gt(gt_root, case_id, _label())
        with zipfile.ZipFile(zips / f"{rsna}.zip", "w") as zf:
            zf.writestr(f"{rsna}/FLAIR/IM1.dcm", b"x")
        entries.append({"case_id": case_id, "rsna_id": rsna})
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text(json.dumps({"cases": entries}))  # JSON is valid YAML

    job_map: dict[str, str] = {}
    for rsna in stored_for:
        job_map[cases[rsna]] = "p12job"
    if stored_for:
        _save_nifti(
            tmp_path / "p12jobs" / "p12job" / "dicom_seg_work" / "atlas_space_mask.nii.gz",
            _label().astype(np.uint8),
        )
    write_json(job_map, tmp_path / "job_map.json")
    research = pd.DataFrame(
        {
            "case_id": [cases[r] for r in kinds],
            **{f"dice_{r}": [0.9] * len(kinds) for r in ("ET", "TC", "WT")},
        }
    )
    research.to_csv(tmp_path / "research.csv", index=False)

    block = {
        "cases_files": [str(cases_file)],
        "zips_dir": str(zips),
        "out_dir": str(tmp_path / "out"),
        "gt_root": str(gt_root),
        "research_per_case": str(tmp_path / "research.csv"),
        "role_by_folder": {"FLAIR": "flair", "T1w": "t1", "T1wCE": "t1ce", "T2w": "t2"},
        "usable_threshold": 0.7,
        "carry_qc_threshold": 0.90,
        "e1_margin": 0.05,
        "bridge_min_dice": 0.99,
        "n_boot": 100,
        "seed": 42,
        "carry_seed": 42,
        "override": dict(OVERRIDE),
        "p12_job_map": str(tmp_path / "job_map.json"),
        "p12_jobs_dir": str(tmp_path / "p12jobs"),
        "expected_n_cases": len(kinds),
        **(extra or {}),
    }
    cfg = OmegaConf.create({"seed": 42, "analysis": {"real_dicom_v2": block}})
    return cfg, FakeClinical(kinds, cases), tmp_path / "out"


def _run(cfg: Any, fake: FakeClinical, **kw: Any) -> dict[str, Path]:
    return v2.run_v2(cfg, run_fn=fake, carry_fn=_fake_carry, **kw)


def _result(out: Path, case_id: str, variant: str) -> dict[str, Any]:
    return json.loads((out / "cases" / case_id / variant / "result.json").read_text())


# ---------------------------------------------------------------------------
# Thick-slice detection
# ---------------------------------------------------------------------------


def test_thick_slice_detected_from_structured_checks_only() -> None:
    thick = {"state": "refused", "input_qc_pre": _qc_report(("spacing_range", "refuse"))}
    aniso = {"state": "refused", "input_qc_pre": _qc_report(("anisotropy", "refuse"))}
    assert v2.is_thick_slice_refusal(thick)
    assert v2.is_thick_slice_refusal(aniso)
    # Prose mentions spacing, but the structured check is something else: NOT thick-slice.
    prose = {
        "state": "refused",
        "error": "refused: voxel spacing and anisotropy",
        "input_qc_pre": _qc_report(("geometry_consistency", "refuse")),
    }
    assert not v2.is_thick_slice_refusal(prose)
    # A WARN on anisotropy is not a refusal; a done job is never a refusal; no report -> False.
    assert not v2.is_thick_slice_refusal(
        {"state": "refused", "input_qc_pre": _qc_report(("anisotropy", "warn"))}
    )
    assert not v2.is_thick_slice_refusal({**thick, "state": "done"})
    assert not v2.is_thick_slice_refusal({"state": "refused", "input_qc_pre": None})


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, val in d.items():
            out.update(_flatten(val, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): d}


def test_composed_config_differs_only_in_the_two_qc_keys() -> None:
    base = {
        "clinical": {
            "ingest": {"role_overrides": {}},
            "input_qc": {
                "spacing_max_mm": 3.0,
                "anisotropy_refuse_ratio": 6.0,
                "spacing_min_mm": 0.3,
            },
        }
    }
    deployed, overridden = OmegaConf.create(base), OmegaConf.create(base)
    roles = {"1.2.3": "t1ce"}
    v2.apply_composed_overrides(deployed, roles, None)
    v2.apply_composed_overrides(overridden, roles, OVERRIDE)
    a = _flatten(OmegaConf.to_container(deployed))
    b = _flatten(OmegaConf.to_container(overridden))
    changed = {k for k in a if a[k] != b[k]}
    assert changed == {
        "clinical.input_qc.spacing_max_mm",
        "clinical.input_qc.anisotropy_refuse_ratio",
    }
    assert deployed.clinical.ingest.role_overrides == overridden.clinical.ingest.role_overrides


def test_override_rejects_unknown_or_missing_keys() -> None:
    cfg = OmegaConf.create({"clinical": {"ingest": {}, "input_qc": {"spacing_max_mm": 3.0}}})
    with pytest.raises(ValueError, match="not an overridable key"):
        v2.apply_composed_overrides(cfg, {}, {"spacing_min_mm": 0.0})
    with pytest.raises(ValueError, match="no 'anisotropy_refuse_ratio'"):
        v2.apply_composed_overrides(cfg, {}, {"anisotropy_refuse_ratio": 9.0})


def test_real_clinical_compose_seam_is_wrapped_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the REAL composed clinical config: only the two QC keys differ, compose is restored."""
    import scripts.run_clinical_study as run_clinical_study_script
    from app.backend import clinical_jobs

    GlobalHydra.instance().clear()
    original = clinical_jobs._compose_clinical_cfg
    seen: list[dict[str, Any]] = []

    def fake_run_study(study_dir: Path, out_dir: Path) -> Path:
        seen.append(_flatten(OmegaConf.to_container(clinical_jobs._compose_clinical_cfg())))
        return out_dir / "summary.json"

    monkeypatch.setattr(run_clinical_study_script, "run_study", fake_run_study)
    monkeypatch.setattr(v2.p12, "build_role_overrides", lambda d, r: {"1.2.3": "t1ce"})
    v2.run_clinical_path_v2(Path("s"), Path("o"), {}, None)
    v2.run_clinical_path_v2(Path("s"), Path("o"), {}, OVERRIDE)
    assert clinical_jobs._compose_clinical_cfg is original
    deployed, overridden = seen
    changed = {k for k in deployed if deployed[k] != overridden[k]}
    assert changed == {
        "clinical.input_qc.spacing_max_mm",
        "clinical.input_qc.anisotropy_refuse_ratio",
    }
    assert overridden["clinical.input_qc.spacing_max_mm"] == 1000.0
    # The role override is applied identically in both runs.
    assert (
        deployed["clinical.ingest.role_overrides.1.2.3"]
        == "t1ce"
        == overridden["clinical.ingest.role_overrides.1.2.3"]
    )


# ---------------------------------------------------------------------------
# The run stage
# ---------------------------------------------------------------------------


def test_override_run_only_for_thick_slice_refusals(tmp_path: Path) -> None:
    kinds = {"00001": "ok", "00002": "thick", "00003": "other_refusal", "00004": "gate_refused"}
    cfg, fake, out = _setup(tmp_path, kinds)
    _run(cfg, fake)

    by_rsna: dict[str, list[Any]] = {}
    for rsna, override in fake.calls:
        by_rsna.setdefault(rsna, []).append(override)
    assert by_rsna["00001"] == [None]
    assert by_rsna["00003"] == [None]  # prose says "spacing" but the structured check differs
    assert by_rsna["00004"] == [None]
    assert by_rsna["00002"] == [None, OVERRIDE]  # exactly one override run, only the 2 keys

    thick_id = fake.cases["00002"]
    dep, ov = _result(out, thick_id, "deployed"), _result(out, thick_id, "override")
    assert dep["thick_slice_refusal"] and dep["status"] == "refused" and not dep["has_mask"]
    assert ov["status"] == "done" and ov["accepted"] and ov["scored"]
    assert not (out / "cases" / fake.cases["00001"] / "override").exists()


def test_scoring_atlas_and_carried_and_gatekeeper_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kinds = {"00001": "ok", "00004": "gate_refused"}
    cfg, fake, out = _setup(tmp_path, kinds)

    def fake_mask_from_cache(job_dir: Path, summary: Any, t1ce: Path, dest: Path) -> None:
        nib.save(nib.Nifti1Image(_label().astype(np.uint8), AFFINE), str(dest))

    monkeypatch.setattr(v2, "_atlas_mask_from_cache", fake_mask_from_cache)
    _run(cfg, fake)

    ok = _result(out, fake.cases["00001"], "deployed")
    assert ok["accepted"] and ok["scored"] and ok["carried_scored"]
    assert ok["dice_WT"] == pytest.approx(1.0) and ok["carried_dice_WT"] == pytest.approx(1.0)
    assert ok["carry_qc_dice"] == pytest.approx(1.0)
    assert ok["lateralisation"] == "ok" and ok["mirrored"] is False
    assert ok["translation_mm"] == [0.0, 0.0, 0.0]

    gate = _result(out, fake.cases["00004"], "deployed")
    # Gatekeeper refusal: not accepted, but a mask exists and is scored.
    assert gate["status"] == "refused" and not gate["accepted"]
    assert gate["refusing_stage"] == "gatekeeper" and gate["has_mask"] and gate["scored"]
    assert gate["gate_decision"] == "refuse"


def test_cleanup_keeps_exactly_the_listed_files(tmp_path: Path) -> None:
    cfg, fake, out = _setup(tmp_path, {"00001": "ok", "00002": "thick"})
    _run(cfg, fake)

    ok_dir = out / "cases" / fake.cases["00001"] / "deployed"
    assert sorted(p.name for p in ok_dir.iterdir()) == sorted([*v2.KEPT_FILES, "result.json"])
    # A refusal before segmentation keeps only the two json files (+ result).
    refused_dir = out / "cases" / fake.cases["00002"] / "deployed"
    assert sorted(p.name for p in refused_dir.iterdir()) == sorted(
        [*v2.KEPT_JOB_FILES, "result.json"]
    )
    # The job store and the unzipped DICOM are gone entirely.
    assert not any((out / "_jobs").iterdir())
    assert not any((out / "_extract").iterdir())
    assert _result(out, fake.cases["00001"], "deployed")["kept_mb"] < 15


def test_resume_skips_parsed_results_and_reruns_corrupt_ones(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg, fake, out = _setup(tmp_path, {"00001": "ok", "00002": "ok"})
    _run(cfg, fake)
    assert len(fake.calls) == 2

    _run(cfg, fake)  # everything parsed -> nothing re-run
    assert len(fake.calls) == 2

    corrupt = out / "cases" / fake.cases["00002"] / "deployed" / "result.json"
    corrupt.write_text("{not json")
    with caplog.at_level(logging.WARNING):
        _run(cfg, fake)
    assert [c[0] for c in fake.calls] == ["00001", "00002", "00002"]
    assert "unparseable" in caplog.text
    assert json.loads(corrupt.read_text())["status"] == "done"


def test_missing_zip_aborts_before_any_case_runs(tmp_path: Path) -> None:
    cfg, fake, out = _setup(tmp_path, {"00001": "ok", "00002": "ok", "00003": "ok"})
    (tmp_path / "zips" / "00002.zip").unlink()
    (tmp_path / "zips" / "00003.zip").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        _run(cfg, fake)
    assert "00002.zip" in str(exc.value) and "00003.zip" in str(exc.value)  # lists all missing
    assert fake.calls == []
    assert not (out / "cases").exists()


def test_duplicate_case_id_raises(tmp_path: Path) -> None:
    f1, f2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    f1.write_text(json.dumps({"cases": [{"case_id": "BraTS2021_00001", "rsna_id": "00001"}]}))
    f2.write_text(json.dumps({"cases": [{"case_id": "BraTS2021_00001", "rsna_id": "00001"}]}))
    with pytest.raises(ValueError, match="duplicate"):
        v2.load_specs([f1, f2], expected_n=2)
    with pytest.raises(ValueError, match="expected 3"):
        v2.load_specs([f1], expected_n=3)


def test_pilot_in_case_file_is_never_run(tmp_path: Path) -> None:
    f = tmp_path / "a.yaml"
    f.write_text(
        json.dumps(
            {
                "pilot": {"case_id": "BraTS2021_00000", "rsna_id": "00000"},
                "cases": [{"case_id": "BraTS2021_00001", "rsna_id": 1}],
            }
        )
    )
    specs = v2.load_specs([f], expected_n=1)
    assert [(s.case_id, s.rsna_id) for s in specs] == [("BraTS2021_00001", "00001")]


def test_mirrored_self_test_fails_on_a_symmetric_ground_truth(tmp_path: Path) -> None:
    sym = np.zeros(SHAPE, dtype=np.int64)
    sym[3:5, 2:5, 2:4] = 1  # centred on the left-right axis: a flip changes nothing
    _write_gt(tmp_path / "gt", "BraTS2021_00001", sym)
    spec = v2.p12.CaseSpec("BraTS2021_00001", "00001", False)
    with pytest.raises(RuntimeError, match="mirrored-mask self-test failed"):
        v2.mirrored_self_test(tmp_path / "gt", [spec])


def test_bridge_value_and_nan_when_no_stored_mask(tmp_path: Path) -> None:
    kinds = {"00001": "ok", "00002": "ok"}
    cfg, fake, out = _setup(tmp_path, kinds, stored_for=("00001",))
    _run(cfg, fake)
    with_stored = _result(out, fake.cases["00001"], "deployed")
    without = _result(out, fake.cases["00002"], "deployed")
    assert with_stored["bridge_stored_mask"] is True
    assert with_stored["bridge_wt_dice"] == pytest.approx(1.0)
    assert without["bridge_stored_mask"] is False
    assert without["bridge_wt_dice"] is None  # NaN, stored as null

    df = v2.build_per_case_frame(
        v2.load_specs(v2.resolve_config(cfg).cases_files, 2), v2.resolve_config(cfg)
    ).set_index("case_id")
    assert df.loc[fake.cases["00001"], "bridge_wt_dice"] == pytest.approx(1.0)
    assert np.isnan(df.loc[fake.cases["00002"], "bridge_wt_dice"])


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------


def _fake_result(
    out: Path,
    case_id: str,
    *,
    thick: bool,
    dice: float,
    qc: float = 0.95,
    override_dice: float = 0.9,
) -> None:
    """Writes deployed (and, for thick, override) result.json files directly."""

    def run(variant: str, status: str, d: float, *, thick_flag: bool) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "variant": variant,
            "status": status,
            "refusing_stage": "input_qc (pre-preprocessing)" if status == "refused" else None,
            "refusing_reason": None,
            "gate_decision": None if status == "refused" else "proceed",
            "accepted": status == "done",
            "thick_slice_refusal": thick_flag,
            "has_mask": status == "done",
            "bridge_stored_mask": False,
            **{f"dice_{r}": (d if status == "done" else None) for r in ("ET", "TC", "WT")},
            **{f"carried_dice_{r}": (d if status == "done" else None) for r in ("ET", "TC", "WT")},
            "carry_qc_dice": qc if status == "done" else None,
            "mirrored": False,
        }

    base = out / "cases" / case_id
    (base / "deployed").mkdir(parents=True)
    write_json(
        run("deployed", "refused" if thick else "done", dice, thick_flag=thick),
        base / "deployed" / "result.json",
    )
    if thick:
        (base / "override").mkdir()
        write_json(
            run("override", "done", override_dice, thick_flag=True),
            base / "override" / "result.json",
        )


def _summarise_fixture(
    tmp_path: Path, *, override_dice: float, n_thick: int = 6
) -> tuple[Path, Any]:
    kinds = {f"{i:05d}": "ok" for i in range(1, n_thick + 5)}
    cfg, fake, out = _setup(tmp_path, kinds)
    for i, rsna in enumerate(kinds):
        case_id = fake.cases[rsna]
        thick = i < n_thick
        _fake_result(out, case_id, thick=thick, dice=0.80 + 0.01 * i, override_dice=override_dice)
    cfg.analysis.real_dicom_v2.stage = "summarise"
    return out, cfg


def test_summarise_writes_schema_frame_and_all_output_files(tmp_path: Path) -> None:
    out, cfg = _summarise_fixture(tmp_path, override_dice=0.9)
    paths = v2.run_v2(cfg)

    df = pd.read_csv(out / "per_case.csv")
    assert set(REQUIRED_COLUMNS) <= set(df.columns)
    assert len(df) == 10 and df["thick_slice_refusal"].sum() == 6
    for name in (
        "e1.json",
        "e2.json",
        "e3_table.csv",
        "e4.json",
        "bridge.json",
        "summary.md",
        "run_meta.json",
    ):
        assert (out / name).is_file(), name
    assert paths["summary_md"] == out / "summary.md"

    meta = json.loads((out / "run_meta.json").read_text())
    assert {"git_sha", "timestamp_utc", "config", "counts", "input_sha256"} <= set(meta)
    assert meta["counts"]["n_studies"] == 10 and meta["counts"]["n_thick_slice_refusals"] == 6
    assert any(k.startswith("research_per_case") and v for k, v in meta["input_sha256"].items())
    assert "NOT external validation" in (out / "summary.md").read_text()
    # NaN is written as null, so every json file is strict JSON.
    json.loads((out / "e1.json").read_text(), parse_constant=lambda c: pytest.fail(c))


def test_e4_reported_only_when_e2_refuses_usable_studies(tmp_path: Path) -> None:
    # 6/6 usable overridden studies: Clopper-Pearson lower bound 0.54 > 0.5.
    out, cfg = _summarise_fixture(tmp_path / "usable", override_dice=0.9)
    v2.run_v2(cfg)
    assert json.loads((out / "e2.json").read_text())["verdict"] == "REFUSES USABLE STUDIES"
    md = (out / "summary.md").read_text()
    assert "post-hoc variant" in md
    assert json.loads((out / "e4.json").read_text())["label"] == "post-hoc variant"

    # 0/6 usable: REFUSAL JUSTIFIED, so E4 must be absent from the summary and the json.
    out2, cfg2 = _summarise_fixture(tmp_path / "unusable", override_dice=0.2)
    v2.run_v2(cfg2)
    assert json.loads((out2 / "e2.json").read_text())["verdict"] == "REFUSAL JUSTIFIED"
    md2 = (out2 / "summary.md").read_text()
    assert "post-hoc variant" not in md2
    assert "Not reported" in md2
    e4 = json.loads((out2 / "e4.json").read_text())
    assert e4["reported"] is False and "p_accepted_and_usable" not in e4


def test_summarise_fails_clearly_when_a_study_has_no_result(tmp_path: Path) -> None:
    out, cfg = _summarise_fixture(tmp_path, override_dice=0.9)
    first = next((out / "cases").iterdir())
    (first / "deployed" / "result.json").unlink()
    with pytest.raises(FileNotFoundError, match="no deployed result"):
        v2.run_v2(cfg)


def test_stage_must_be_valid(tmp_path: Path) -> None:
    cfg, _, _ = _setup(tmp_path, {"00001": "ok"}, extra={"stage": "bogus"})
    with pytest.raises(ValueError, match="stage"):
        v2.resolve_config(cfg)
