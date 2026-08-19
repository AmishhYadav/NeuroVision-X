"""Tests for `app.backend.inference`, live CPU inference for the demo backend.

Everything here is synthetic and small: a tiny random preprocessed case
(32^3) and a tiny checkpoint built from the exact same composed Hydra config
`inference.py` itself would compose for `settings.experiment` -- built via
the module's own `_compose_cfg` helper, so the checkpoint's architecture can
never accidentally drift from what `segment_case` will build to load it into.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from app.backend import inference as inf
from app.backend.config import REPO_ROOT, Settings

CASE_ID = "SYNTH_001"
CASE_SHAPE = (32, 32, 32)  # (D, H, W), tiny on purpose -- CPU, seconds not minutes
EXPERIMENT = "baseline_unet3d"  # a real experiment config already in configs/experiment/


def _settings(
    tmp_path: Path,
    *,
    checkpoint: Path | None = None,
    experiment: str = EXPERIMENT,
) -> Settings:
    """Builds a `Settings` pointed entirely at `tmp_path`, bypassing env vars / caching."""
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir(exist_ok=True)
    return Settings(
        prep_dir=prep_dir,
        eval_dir=tmp_path / "eval",
        checkpoint=checkpoint if checkpoint is not None else tmp_path / "no_such_checkpoint.pt",
        experiment=experiment,
        cache_dir=tmp_path / "cache",
        max_cases=10,
        demo_overlap=0.25,
        report_dir=tmp_path / "reports",
    )


def _write_case(prep_dir: Path, case_id: str, shape: tuple[int, int, int] = CASE_SHAPE) -> None:
    """Writes a minimal synthetic preprocessed case: just `image.npy`, 4 modalities."""
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    image = rng.normal(size=(4, *shape)).astype(np.float16)
    np.save(case_dir / "image.npy", image)


def _build_checkpoint(settings: Settings, checkpoint_path: Path) -> None:
    """Builds a model from the SAME composed config `segment_case` will use, and saves it.

    Using `inf._compose_cfg` (the module's own private helper) rather than
    hand-building a config is the point: it guarantees the saved
    `model_state_dict` matches the architecture `segment_case` will build to
    load it into, with zero chance of the two silently drifting apart.
    """
    from neurovision.models.registry import build_model
    from neurovision.training.checkpoint import save_checkpoint

    cfg = inf._compose_cfg(settings)
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_checkpoint(
        checkpoint_path.parent,
        model,
        optimizer,
        epoch=0,
        global_step=1,
        best_metric=0.5,
        best_metric_name="val/dice_mean",
        best_metric_mode="max",
    )
    # save_checkpoint always writes last.pt into out_dir; point the caller's
    # checkpoint path at it directly (out_dir == checkpoint_path.parent).
    last = checkpoint_path.parent / "last.pt"
    assert last.is_file()
    if last != checkpoint_path:
        last.replace(checkpoint_path)


@pytest.fixture
def ready_settings(tmp_path: Path) -> Settings:
    """A `Settings` with both a synthetic case and a matching checkpoint on disk."""
    settings = _settings(tmp_path, checkpoint=tmp_path / "ckpt" / "model.pt")
    _write_case(settings.prep_dir, CASE_ID)
    _build_checkpoint(settings, settings.checkpoint)
    return settings


# --- 1. Import hygiene ----------------------------------------------------


def test_importing_inference_does_not_import_torch() -> None:
    """`import app.backend.inference` alone must never pull in torch or Hydra."""
    script = (
        "import sys\n"
        "import app.backend.inference\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
        "assert 'hydra' not in sys.modules, sorted(m for m in sys.modules if 'hydra' in m)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


# --- 2/3. checkpoint_available / inference_status -------------------------


def test_checkpoint_available_false_for_missing_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)  # checkpoint deliberately does not exist
    assert inf.checkpoint_available(settings) is False


def test_checkpoint_available_true_for_real_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"not a real checkpoint, existence is all that matters here")
    settings = _settings(tmp_path, checkpoint=checkpoint)
    assert inf.checkpoint_available(settings) is True


def test_inference_status_reports_missing_checkpoint_with_actionable_reason(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    status = inf.inference_status(settings)
    assert status["available"] is False
    assert status["experiment"] == EXPERIMENT
    assert status["checkpoint"] == str(settings.checkpoint)
    assert status["reason"]  # non-empty
    assert "NVX_CHECKPOINT" in status["reason"]


def test_inference_status_never_raises_on_a_broken_settings(tmp_path: Path) -> None:
    # A checkpoint path that is a directory, not a file -- .is_file() must
    # calmly return False rather than this function raising.
    weird = tmp_path / "a_directory"
    weird.mkdir()
    settings = _settings(tmp_path, checkpoint=weird)
    status = inf.inference_status(settings)
    assert status["available"] is False
    assert status["reason"]


# --- 4. cached_prediction_path ---------------------------------------------


def test_cached_prediction_path_is_namespaced_by_experiment(tmp_path: Path) -> None:
    settings_a = _settings(tmp_path, experiment="baseline_unet3d")
    settings_b = _settings(tmp_path, experiment="neurovision")
    path_a = inf.cached_prediction_path(settings_a, CASE_ID)
    path_b = inf.cached_prediction_path(settings_b, CASE_ID)
    assert path_a != path_b
    assert "baseline_unet3d" in str(path_a)
    assert "neurovision" in str(path_b)
    assert path_a.name == path_b.name == f"{CASE_ID}.npy"
    # Never created eagerly.
    assert not path_a.parent.exists()


# --- 5-7, 10-11: segment_case, the real thing -------------------------------


def test_segment_case_produces_a_cropped_shape_uint8_class_map(ready_settings: Settings) -> None:
    out_path = inf.segment_case(ready_settings, CASE_ID)
    assert out_path == inf.cached_prediction_path(ready_settings, CASE_ID)
    assert out_path.is_file()

    result = np.load(out_path)
    assert result.shape == CASE_SHAPE
    assert result.dtype == np.uint8
    assert set(np.unique(result).tolist()) <= {0, 1, 2, 3}


def test_segment_case_second_call_is_cached_not_recomputed(ready_settings: Settings) -> None:
    out_path = inf.segment_case(ready_settings, CASE_ID)
    mtime_1 = out_path.stat().st_mtime_ns
    content_1 = np.load(out_path).copy()

    out_path_2 = inf.segment_case(ready_settings, CASE_ID)
    mtime_2 = out_path_2.stat().st_mtime_ns

    assert out_path_2 == out_path
    assert mtime_2 == mtime_1
    np.testing.assert_array_equal(np.load(out_path_2), content_1)


def test_segment_case_force_recomputes_and_rewrites(ready_settings: Settings) -> None:
    out_path = inf.segment_case(ready_settings, CASE_ID)
    mtime_1 = out_path.stat().st_mtime_ns

    out_path_2 = inf.segment_case(ready_settings, CASE_ID, force=True)
    mtime_2 = out_path_2.stat().st_mtime_ns

    assert out_path_2 == out_path
    assert mtime_2 >= mtime_1  # rewritten -- may tie on a coarse filesystem clock, never regress


def test_segment_case_missing_checkpoint_raises_with_checkpoint_path_in_message(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)  # no checkpoint written
    _write_case(settings.prep_dir, CASE_ID)
    with pytest.raises(FileNotFoundError, match=str(settings.checkpoint)):
        inf.segment_case(settings, CASE_ID)


def test_segment_case_missing_case_raises_with_case_id_in_message(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"placeholder -- the case check must fire before this is ever read")
    settings = _settings(tmp_path, checkpoint=checkpoint)
    # No image.npy written for this case at all.
    with pytest.raises(FileNotFoundError, match="NO_SUCH_CASE"):
        inf.segment_case(settings, "NO_SUCH_CASE")


def test_segment_case_progress_is_monotonic_and_ends_at_one(ready_settings: Settings) -> None:
    calls: list[tuple[str, float]] = []
    inf.segment_case(
        ready_settings,
        CASE_ID,
        force=True,
        progress=lambda stage, frac: calls.append((stage, frac)),
    )

    assert len(calls) >= 2
    fractions = [f for _, f in calls]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert calls[-1][0] == "done"


def test_segment_case_output_respects_region_nesting(ready_settings: Settings) -> None:
    """ET (class 3) must always be a subset of TC ({1,3}), which is a subset of WT (>0).

    Uses the same class-map convention as `app/backend/volumes.py`'s
    `region_voxel_counts`: TC = necrotic-or-enhancing, WT = any foreground.
    """
    out_path = inf.segment_case(ready_settings, CASE_ID)
    classes = np.load(out_path)

    et = classes == 3
    tc = (classes == 1) | (classes == 3)
    wt = classes > 0

    assert np.all(tc[et]), "an ET voxel exists that is not part of TC"
    assert np.all(wt[tc]), "a TC voxel exists that is not part of WT"
