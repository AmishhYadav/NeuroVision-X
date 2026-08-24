"""Tests for `neurovision.data.clinical_preprocess`.

Split the same way the module is split. Tests 1-12 exercise the planning
layer (`resolve_use_gpu`, `resolve_atlas_name`, `build_plan`) -- pure, no
heavy dependency, must run in `.venv`. Tests 13-14 exercise the execution
layer and are guarded with `pytest.importorskip("brainles_preprocessing")`,
following the idiom already used in `tests/test_dicom_ingest.py`: the
`importorskip` call lives inside each guarded test's body, not at module
scope, so every unguarded test above still runs in the main suite.

No real patient data, no real BraTS data, no real ANTs/HD-BET run -- both
need weight downloads and minutes of CPU, which is exactly what CLAUDE.md's
testing rules forbid. Tiny synthetic NIfTIs are written with `nibabel`
wherever a file must exist on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from neurovision.data.clinical_preprocess import (
    PreprocessPlan,
    build_plan,
    resolve_atlas_name,
    resolve_use_gpu,
)
from neurovision.data.dicom_ingest import ROLES

# Real configs/ directory, resolved relative to this file -- same pattern as
# tests/test_train_qc.py's _CONFIG_DIR, so the "reachable at the composed
# path" test composes the PROJECT's actual config, not a hand-built stand-in.
_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _write_tiny_nifti(path: Path) -> Path:
    """Write a trivial 4x4x4 NIfTI volume, for tests that only need a file to exist."""
    data = np.ones((4, 4, 4), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(path))
    return path


def _make_cfg(tmp_path: Path, **preprocess_overrides: object) -> SimpleNamespace:
    """Build a minimal fake config matching configs/clinical/default.yaml's shape.

    Plain `SimpleNamespace` rather than `OmegaConf.create`, deliberately:
    `omegaconf` (like `hydra`) lives only in `.venv`, and this fixture is
    used by tests that must also collect and run under `.venv-clinical`
    (tests 13-14). `resolve_use_gpu` / `build_plan` only ever do attribute
    access on `cfg`, which `SimpleNamespace` supports identically.

    Only the keys `resolve_use_gpu` / `build_plan` actually read, so a stray
    real key being renamed upstream would NOT be caught here -- that is
    exactly what `test_config_block_is_reachable_at_the_composed_path`
    exists to catch, against the real composed config.
    """
    preprocess = {
        "center_modality": "t1ce",
        "atlas": "BRATS_SRI24",
        "n4_bias_correction": False,
        "brain_extraction": True,
        "defacing": False,
        "use_gpu": None,
        "keep_intermediate": True,
        "out_dir": str(tmp_path / "clinical_preprocess"),
    }
    preprocess.update(preprocess_overrides)
    return SimpleNamespace(
        device="cpu",
        clinical=SimpleNamespace(preprocess=SimpleNamespace(**preprocess)),
    )


# ---------------------------------------------------------------------------
# 1-2. resolve_use_gpu
# ---------------------------------------------------------------------------


def test_resolve_use_gpu_none_derives_from_device(tmp_path: Path) -> None:
    cfg_cpu = _make_cfg(tmp_path, use_gpu=None)
    assert resolve_use_gpu(cfg_cpu) is False

    cfg_forced_true = _make_cfg(tmp_path, use_gpu=True)
    assert resolve_use_gpu(cfg_forced_true) is True

    cfg_forced_false = _make_cfg(tmp_path, use_gpu=False)
    assert resolve_use_gpu(cfg_forced_false) is False


def test_resolve_use_gpu_never_defaults_to_true(tmp_path: Path) -> None:
    """The whole point of this function.

    `AtlasCentricPreprocessor.__init__` (brainles_preprocessing==0.6.13)
    defaults `use_gpu=True` no matter what hardware is present. With
    `use_gpu=None` in config and `cfg.device="cpu"`, `resolve_use_gpu` must
    resolve to False -- never silently inherit that dependency default.
    """
    cfg = _make_cfg(tmp_path, use_gpu=None)
    assert cfg.device == "cpu"
    assert resolve_use_gpu(cfg) is False


# ---------------------------------------------------------------------------
# 3. resolve_atlas_name
# ---------------------------------------------------------------------------


def test_resolve_atlas_name_accepts_valid_and_rejects_invalid() -> None:
    assert resolve_atlas_name("BRATS_SRI24") == "BRATS_SRI24"

    with pytest.raises(ValueError, match="BRATS_SRI24") as excinfo:
        resolve_atlas_name("NOT_A_REAL_ATLAS")
    # The error message must list every valid name, not just the one that
    # happened to match the regex above.
    message = str(excinfo.value)
    for valid_name in ("BRATS_SRI24", "SRI24", "BRATS_MNI152", "MNI152"):
        assert valid_name in message


# ---------------------------------------------------------------------------
# 4-8, 11. build_plan
# ---------------------------------------------------------------------------


def _four_role_inputs(tmp_path: Path) -> dict[str, Path]:
    return {role: _write_tiny_nifti(tmp_path / f"{role}.nii.gz") for role in ROLES}


def test_build_plan_moving_roles_are_in_canonical_order(tmp_path: Path) -> None:
    files = _four_role_inputs(tmp_path)
    cfg = _make_cfg(tmp_path, center_modality="t1ce")

    scrambled = {
        "flair": files["flair"],
        "t1": files["t1"],
        "t2": files["t2"],
        "t1ce": files["t1ce"],
    }
    plan_a = build_plan(cfg, scrambled, out_dir=tmp_path / "out_a")
    assert plan_a.moving_roles == ("t1", "t2", "flair")

    # Rebuilt in the OPPOSITE dict order -- if moving_roles ever leaked dict
    # insertion order, only one of these two orderings would trip it.
    reversed_dict = {
        "t1ce": files["t1ce"],
        "t2": files["t2"],
        "t1": files["t1"],
        "flair": files["flair"],
    }
    plan_b = build_plan(cfg, reversed_dict, out_dir=tmp_path / "out_b")
    assert plan_b.moving_roles == ("t1", "t2", "flair")


def test_build_plan_rejects_missing_center_role(tmp_path: Path) -> None:
    inputs = {"t1": _write_tiny_nifti(tmp_path / "t1.nii.gz")}
    cfg = _make_cfg(tmp_path, center_modality="t1ce")

    with pytest.raises(ValueError, match="t1ce"):
        build_plan(cfg, inputs, out_dir=tmp_path / "out")


def test_build_plan_reports_all_missing_files_at_once(tmp_path: Path) -> None:
    inputs = {
        "t1ce": _write_tiny_nifti(tmp_path / "t1ce.nii.gz"),
        "t1": tmp_path / "does_not_exist_t1.nii.gz",
        "t2": tmp_path / "does_not_exist_t2.nii.gz",
    }
    cfg = _make_cfg(tmp_path, center_modality="t1ce")

    with pytest.raises(FileNotFoundError) as excinfo:
        build_plan(cfg, inputs, out_dir=tmp_path / "out")
    message = str(excinfo.value)
    assert "does_not_exist_t1.nii.gz" in message
    assert "does_not_exist_t2.nii.gz" in message


def test_build_plan_rejects_unknown_role_key(tmp_path: Path) -> None:
    inputs = {
        "t1ce": _write_tiny_nifti(tmp_path / "t1ce.nii.gz"),
        "dwi": _write_tiny_nifti(tmp_path / "dwi.nii.gz"),
    }
    cfg = _make_cfg(tmp_path, center_modality="t1ce")

    with pytest.raises(ValueError, match="dwi"):
        build_plan(cfg, inputs, out_dir=tmp_path / "out")


def test_build_plan_allows_three_modalities_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    inputs = {
        role: _write_tiny_nifti(tmp_path / f"{role}.nii.gz") for role in ("t1", "t1ce", "flair")
    }
    cfg = _make_cfg(tmp_path, center_modality="t1ce")

    with caplog.at_level(logging.WARNING):
        plan = build_plan(cfg, inputs, out_dir=tmp_path / "out")

    assert set(plan.inputs) == {"t1", "t1ce", "flair"}
    assert plan.moving_roles == ("t1", "flair")
    assert any("t2" in record.message for record in caplog.records)


def test_build_plan_creates_no_directories(tmp_path: Path) -> None:
    inputs = _four_role_inputs(tmp_path)
    cfg = _make_cfg(tmp_path, center_modality="t1ce")
    out_dir = tmp_path / "out_not_yet_created"

    build_plan(cfg, inputs, out_dir=out_dir)

    assert not out_dir.exists()


def test_build_plan_output_paths(tmp_path: Path) -> None:
    inputs = _four_role_inputs(tmp_path)
    cfg = _make_cfg(tmp_path, center_modality="t1ce")
    out_dir = tmp_path / "out"

    plan = build_plan(cfg, inputs, out_dir=out_dir)

    for role in ROLES:
        assert plan.outputs[role] == out_dir / f"{role}.nii.gz"
    assert plan.brain_mask_path == out_dir / "brain_mask.nii.gz"
    assert plan.log_file == out_dir / "preprocess.log"


def test_build_plan_intermediate_dirs_follow_keep_intermediate(tmp_path: Path) -> None:
    inputs = _four_role_inputs(tmp_path)

    cfg_kept = _make_cfg(tmp_path, center_modality="t1ce", keep_intermediate=True)
    plan_kept = build_plan(cfg_kept, inputs, out_dir=tmp_path / "out_kept")
    # brain_extraction on, n4/defacing off (config defaults), 4 modalities
    # supplied -> coregistration has something to do.
    assert set(plan_kept.intermediate_dirs) == {
        "coregistration",
        "atlas_registration",
        "atlas_correction",
        "brain_extraction",
        "transformations",
    }
    for stage, directory in plan_kept.intermediate_dirs.items():
        assert directory == plan_kept.out_dir / "intermediate" / stage

    cfg_not_kept = _make_cfg(tmp_path, center_modality="t1ce", keep_intermediate=False)
    plan_not_kept = build_plan(cfg_not_kept, inputs, out_dir=tmp_path / "out_not_kept")
    assert plan_not_kept.intermediate_dirs == {}


def test_build_plan_out_dir_with_existing_contents_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    inputs = _four_role_inputs(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "t1ce.nii.gz").write_bytes(b"stale")
    cfg = _make_cfg(tmp_path, center_modality="t1ce")

    with caplog.at_level(logging.WARNING):
        build_plan(cfg, inputs, out_dir=out_dir)

    assert any("overwritten" in record.message for record in caplog.records)


def test_build_plan_is_frozen_dataclass(tmp_path: Path) -> None:
    inputs = _four_role_inputs(tmp_path)
    cfg = _make_cfg(tmp_path, center_modality="t1ce")
    plan = build_plan(cfg, inputs, out_dir=tmp_path / "out")

    assert isinstance(plan, PreprocessPlan)
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        plan.center_role = "t1"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 12. Real composed config
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_at_the_composed_path() -> None:
    """The REAL project config, composed through Hydra, must expose the
    clinical preprocessing block at `cfg.clinical.preprocess` -- the exact
    path `build_plan` and `resolve_use_gpu` read.

    Same regression shape as
    tests/test_train_qc.py::test_config_block_is_reachable_at_the_composed_path:
    a hand-built OmegaConf fixture that puts "preprocess" at the wrong
    nesting level would pass every other test in this file while the real
    composed config never produces that shape.

    `hydra` lives only in `.venv` (the main training stack), not
    `.venv-clinical`, hence the `importorskip` -- unlike the module's own
    `pytest.importorskip("brainles_preprocessing")` guards below, this one
    is about test-environment availability, not about the code under test.
    """
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config")

    assert "clinical" in cfg
    assert "preprocess" in cfg.clinical
    assert "preprocess" not in cfg  # NOT cfg.preprocess

    preprocess_cfg = cfg.clinical.preprocess
    expected_keys = {
        "center_modality",
        "atlas",
        "n4_bias_correction",
        "brain_extraction",
        "defacing",
        "use_gpu",
        "keep_intermediate",
        "out_dir",
    }
    assert expected_keys <= set(preprocess_cfg.keys())


# ---------------------------------------------------------------------------
# 13-14. Execution layer -- guarded, needs .venv-clinical.
# ---------------------------------------------------------------------------


def test_atlas_member_names_match_the_dependency() -> None:
    """The upgrade tripwire: our hardcoded `_ATLAS_NAMES` must equal the
    installed brainles_preprocessing.constants.Atlas's own member names.
    """
    pytest.importorskip("brainles_preprocessing")
    from brainles_preprocessing.constants import Atlas

    from neurovision.data.clinical_preprocess import _ATLAS_NAMES

    assert _ATLAS_NAMES == tuple(member.name for member in Atlas)


def test_run_plan_raises_when_outputs_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.run()` is monkeypatched to a no-op -- no real ANTs, no real HD-BET,
    no weight download, no network call for the atlas fetch either. This
    only checks that `run_plan` refuses to trust brainles silently produced
    nothing.
    """
    pytest.importorskip("brainles_preprocessing")
    from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
    from brainles_preprocessing.preprocessor import atlas_centric_preprocessor as acp_module

    from neurovision.data.clinical_preprocess import build_plan, run_plan

    # AtlasCentricPreprocessor.__init__ fetches the real atlas file over the
    # network when given an Atlas enum member -- avoid that entirely.
    monkeypatch.setattr(acp_module, "fetch_atlases", lambda: tmp_path)
    monkeypatch.setattr(AtlasCentricPreprocessor, "run", lambda self, **kwargs: None)

    inputs = {
        "t1ce": _write_tiny_nifti(tmp_path / "t1ce.nii.gz"),
        "t1": _write_tiny_nifti(tmp_path / "t1.nii.gz"),
    }
    cfg = _make_cfg(tmp_path, center_modality="t1ce")
    out_dir = tmp_path / "out"
    plan = build_plan(cfg, inputs, out_dir=out_dir)

    with pytest.raises(RuntimeError) as excinfo:
        run_plan(plan)

    message = str(excinfo.value)
    assert str(plan.outputs["t1ce"]) in message
    assert str(plan.outputs["t1"]) in message
    assert str(plan.brain_mask_path) in message
