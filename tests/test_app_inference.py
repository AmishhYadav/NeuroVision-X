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


# --- 12. save_logits (optional, default-off) --------------------------------


def test_segment_case_default_save_logits_false_writes_no_logits_file(
    ready_settings: Settings,
) -> None:
    """`save_logits` defaults to False: no logits file must appear at all."""
    out_path = inf.segment_case(ready_settings, CASE_ID)
    assert out_path.is_file()
    assert not inf.cached_logits_path(ready_settings, CASE_ID).is_file()


def test_segment_case_save_logits_true_on_fresh_case_writes_both_files(
    ready_settings: Settings,
) -> None:
    """A fresh call with `save_logits=True` must produce the prediction AND the logits."""
    out_path = inf.segment_case(ready_settings, CASE_ID, save_logits=True)
    logits_path = inf.cached_logits_path(ready_settings, CASE_ID)

    assert out_path.is_file()
    assert logits_path.is_file()

    logits = np.load(logits_path)
    assert logits.shape == (3, *CASE_SHAPE)
    assert logits.dtype == np.float16


def test_segment_case_save_logits_true_recomputes_when_prediction_cached_without_logits(
    ready_settings: Settings,
) -> None:
    """A prior save_logits=False call must not leave a stale prediction with no logits.

    Calling again with save_logits=True must NOT silently return the old
    prediction as-is -- it must recompute so that both the prediction and
    the logits file exist afterward.
    """
    first_out = inf.segment_case(ready_settings, CASE_ID, save_logits=False)
    logits_path = inf.cached_logits_path(ready_settings, CASE_ID)
    assert first_out.is_file()
    assert not logits_path.is_file()  # nothing saved logits yet

    second_out = inf.segment_case(ready_settings, CASE_ID, save_logits=True)

    assert second_out == first_out
    assert second_out.is_file()
    assert logits_path.is_file(), "save_logits=True must not silently no-op on a stale cache"

    logits = np.load(logits_path)
    assert logits.shape == (3, *CASE_SHAPE)
    assert logits.dtype == np.float16


def test_segment_case_save_logits_true_both_cached_skips_recomputation(
    ready_settings: Settings,
) -> None:
    """Once both the prediction and the logits are cached, a repeat call must not recompute."""
    out_path = inf.segment_case(ready_settings, CASE_ID, save_logits=True)
    logits_path = inf.cached_logits_path(ready_settings, CASE_ID)
    pred_mtime_1 = out_path.stat().st_mtime_ns
    logits_mtime_1 = logits_path.stat().st_mtime_ns
    pred_content_1 = np.load(out_path).copy()
    logits_content_1 = np.load(logits_path).copy()

    out_path_2 = inf.segment_case(ready_settings, CASE_ID, save_logits=True)

    assert out_path_2 == out_path
    assert out_path_2.stat().st_mtime_ns == pred_mtime_1
    assert logits_path.stat().st_mtime_ns == logits_mtime_1
    np.testing.assert_array_equal(np.load(out_path_2), pred_content_1)
    np.testing.assert_array_equal(np.load(logits_path), logits_content_1)


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


# --- 13+. cached_gradcam_path / explain_case --------------------------------
#
# explain_case's hardcoded _GRADCAM_TARGET_LAYER ("decoder.conv_blocks.2") only
# exists on the real NeuroVisionX architecture (the "neurovision" experiment),
# not on "baseline_unet3d"'s plain MONAI UNet -- so these tests build a real
# "neurovision" checkpoint instead of reusing EXPERIMENT/ready_settings above.
# That checkpoint (and the one real forward+backward grad_cam pass it costs) is
# the slow part of this file, so it is built ONCE per module and reused
# read-only by every test below, rather than once per test.

GRADCAM_EXPERIMENT = "neurovision"
GRADCAM_CASE_ID = "SYNTH_GRADCAM_001"
# Strictly larger than the checkpoint's own trained 64^3 patch size on every
# axis, so placing the Grad-CAM patch back into full geometry leaves a real
# "outside the patch" shell to assert is exactly 0 -- a volume exactly equal
# to the patch size would trivially pass that check with no shell left at all.
GRADCAM_CASE_SHAPE = (80, 80, 80)


@pytest.fixture(scope="module")
def gradcam_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Builds one real 'neurovision' checkpoint, shared read-only by every test below."""
    ckpt_dir = tmp_path_factory.mktemp("gradcam_ckpt")
    checkpoint = ckpt_dir / "model.pt"
    settings = _settings(ckpt_dir, checkpoint=checkpoint, experiment=GRADCAM_EXPERIMENT)
    _build_checkpoint(settings, checkpoint)
    return checkpoint


@pytest.fixture
def gradcam_settings(tmp_path: Path, gradcam_checkpoint: Path) -> Settings:
    """A `Settings` using the shared real 'neurovision' checkpoint, with its own case."""
    settings = _settings(tmp_path, checkpoint=gradcam_checkpoint, experiment=GRADCAM_EXPERIMENT)
    _write_case(settings.prep_dir, GRADCAM_CASE_ID, GRADCAM_CASE_SHAPE)
    return settings


def _write_gradcam_prediction(settings: Settings) -> None:
    """Writes a cached class-map prediction with a small enhancing-tumor blob near a corner.

    The blob (class 3, enhancing -- counts toward both TC and WT) sits at
    indices [15:25) on every axis, centroid 19.5 -> rounds to 20. A 64-wide
    patch centred at 20 would start at 20 - 32 = -12 on every axis, clamped by
    `center_patch_on_mask` to `max(0, min(-12, 80 - 64)) = 0` -- so the patch
    deterministically occupies exactly `[0:64)` on every axis, independent of
    any rounding subtlety, letting the tests below hand-verify the "outside
    the patch" region directly.
    """
    prediction = np.zeros(GRADCAM_CASE_SHAPE, dtype=np.uint8)
    prediction[15:25, 15:25, 15:25] = 3
    out_path = inf.cached_prediction_path(settings, GRADCAM_CASE_ID)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, prediction)


def test_cached_gradcam_path_is_namespaced_by_experiment_and_region(tmp_path: Path) -> None:
    settings_a = _settings(tmp_path, experiment="baseline_unet3d")
    settings_b = _settings(tmp_path, experiment="neurovision")
    path_wt = inf.cached_gradcam_path(settings_a, CASE_ID, "WT")
    path_tc = inf.cached_gradcam_path(settings_a, CASE_ID, "TC")
    path_other_experiment = inf.cached_gradcam_path(settings_b, CASE_ID, "WT")

    assert path_wt != path_tc
    assert path_wt != path_other_experiment
    assert "baseline_unet3d" in str(path_wt)
    assert "neurovision" in str(path_other_experiment)
    assert path_wt.name == f"{CASE_ID}.npy"
    # Never created eagerly.
    assert not path_wt.parent.exists()


def test_explain_case_invalid_region_raises_value_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)  # no checkpoint/case needed -- must raise before either is read
    with pytest.raises(ValueError, match="WT"):
        inf.explain_case(settings, CASE_ID, "ET")


def test_explain_case_missing_prediction_raises_file_not_found(gradcam_settings: Settings) -> None:
    # image.npy exists (from the gradcam_settings fixture), but segment_case was
    # never called for this case -- no cached prediction exists yet.
    with pytest.raises(FileNotFoundError, match=GRADCAM_CASE_ID):
        inf.explain_case(gradcam_settings, GRADCAM_CASE_ID, "WT")


def test_explain_case_produces_full_shape_output_zero_outside_patch(
    gradcam_settings: Settings,
) -> None:
    _write_gradcam_prediction(gradcam_settings)

    out_path = inf.explain_case(gradcam_settings, GRADCAM_CASE_ID, "TC")

    assert out_path == inf.cached_gradcam_path(gradcam_settings, GRADCAM_CASE_ID, "TC")
    assert out_path.is_file()

    cam = np.load(out_path)
    assert cam.shape == GRADCAM_CASE_SHAPE  # the full image shape, not the 64^3 patch
    assert cam.dtype == np.uint8

    # See _write_gradcam_prediction's docstring: the patch deterministically
    # occupies exactly [0:64) on every axis, so index >= 64 on ANY axis is
    # strictly outside it and must carry no evidence (0 = "not computed here").
    assert np.all(cam[64:, :, :] == 0)
    assert np.all(cam[:, 64:, :] == 0)
    assert np.all(cam[:, :, 64:] == 0)

    # And INSIDE the patch (the same [0:64) region on every axis), there must be
    # a real, non-degenerate heatmap -- a regression that always hit grad_cam's
    # all-zero fallback path would still leave the outside-patch checks above
    # green, so this guards against that silently-blank case.
    assert cam[:64, :64, :64].max() > 0


def test_explain_case_second_call_is_cached_not_recomputed(
    gradcam_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_gradcam_prediction(gradcam_settings)

    out_path = inf.explain_case(gradcam_settings, GRADCAM_CASE_ID, "WT")
    content_1 = np.load(out_path).copy()

    # If the second call tried to reload the model, this raises -- proving the
    # cache hit short-circuits before ever reaching _load_model.
    def _raise_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("explain_case must not reload the model on a cache hit")

    monkeypatch.setattr(inf, "_load_model", _raise_if_called)

    out_path_2 = inf.explain_case(gradcam_settings, GRADCAM_CASE_ID, "WT")

    assert out_path_2 == out_path
    np.testing.assert_array_equal(np.load(out_path_2), content_1)
