"""Tests for scripts/nnunet_session.py.

Every function tested here is either pure (should_stop, check_identity), pure
filesystem manipulation (link_preprocessed, find_preprocessed_dataset,
restore_results, find_prev_fold_dir), a small torch.load reader
(read_checkpoint_meta), or exercised against a FAKE trainer object
(run_budgeted_training) -- none of it needs `nnunetv2` (which lives only in
`.venv-analysis`, see `requirements-analysis.txt`) or a CUDA device, so this
whole file runs in the main `.venv`, on CPU, in well under a second.

The script lives under scripts/, not src/, so it is loaded via
`tests/script_loader.py`'s `importlib.util.spec_from_file_location` pattern,
same as every other scripts/ test file.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from tests.script_loader import load_script

nnunet_session = load_script("nnunet_session")

should_stop = nnunet_session.should_stop
check_identity = nnunet_session.check_identity
link_preprocessed = nnunet_session.link_preprocessed
find_preprocessed_dataset = nnunet_session.find_preprocessed_dataset
restore_results = nnunet_session.restore_results
find_prev_fold_dir = nnunet_session.find_prev_fold_dir
read_checkpoint_meta = nnunet_session.read_checkpoint_meta
run_budgeted_training = nnunet_session.run_budgeted_training

_DATASET_FOLDER = "Dataset901_NeuroVisionXBraTS21"
_CONFIG_DIR = "nnUNetPlans_3d_fullres"
_TRAINER_DIR = "nnUNetTrainer__nnUNetPlans__3d_fullres"
_FOLD_NAME = "fold_all"


# ---------------------------------------------------------------------------
# should_stop
# ---------------------------------------------------------------------------


def test_should_stop_no_durations_uses_first_epoch_estimate() -> None:
    # predicted = 1.5 * 10 = 15.
    assert should_stop(90.0, [], budget_s=100.0, first_epoch_estimate_s=10.0) is True
    assert should_stop(50.0, [], budget_s=100.0, first_epoch_estimate_s=10.0) is False


def test_should_stop_uses_max_of_last_5_durations() -> None:
    # 6 entries: the oldest (1.0) must be dropped, leaving max([2,100,3,4,5]) = 100.
    durations = [1.0, 2.0, 100.0, 3.0, 4.0, 5.0]
    predicted = 1.5 * 100.0
    assert should_stop(0.0, durations, budget_s=predicted - 0.1, first_epoch_estimate_s=1.0) is True
    assert (
        should_stop(0.0, durations, budget_s=predicted + 0.1, first_epoch_estimate_s=1.0) is False
    )


def test_should_stop_exact_budget_boundary_does_not_stop() -> None:
    # elapsed + predicted == budget exactly -> strict ">" means "do not stop".
    elapsed, estimate, factor = 90.0, 10.0, 1.5
    budget = elapsed + factor * estimate
    assert should_stop(elapsed, [], budget_s=budget, first_epoch_estimate_s=estimate) is False
    assert should_stop(elapsed + 1e-6, [], budget_s=budget, first_epoch_estimate_s=estimate) is True


# ---------------------------------------------------------------------------
# check_identity
# ---------------------------------------------------------------------------


def _make_meta(**overrides: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "current_epoch": 5,
        "trainer_name": "nnUNetTrainer",
        "plans": {"a": 1, "b": [1, 2, 3]},
        "configuration": "3d_fullres",
        "fold": "all",
        "train_loss_finite": True,
    }
    meta.update(overrides)
    return meta


_PASSING_KWARGS = dict(
    expect_start_epoch=5,
    expect_trainer="nnUNetTrainer",
    plans_json={"a": 1, "b": [1, 2, 3]},
    configuration="3d_fullres",
    fold="all",
)


def test_check_identity_fresh_run_with_no_checkpoint_passes() -> None:
    check_identity(None, **{**_PASSING_KWARGS, "expect_start_epoch": 0})


def test_check_identity_tuple_vs_list_plans_are_equal_after_round_trip() -> None:
    meta = _make_meta(plans={"a": 1, "b": (1, 2, 3)})
    check_identity(meta, **_PASSING_KWARGS)


def test_check_identity_passing_case() -> None:
    check_identity(_make_meta(), **_PASSING_KWARGS)


def test_check_identity_raises_expect_zero_but_checkpoint_exists() -> None:
    with pytest.raises(RuntimeError, match="expect_start_epoch is 0"):
        check_identity(_make_meta(), **{**_PASSING_KWARGS, "expect_start_epoch": 0})


def test_check_identity_raises_expect_nonzero_but_no_checkpoint() -> None:
    with pytest.raises(RuntimeError, match="no checkpoint was found"):
        check_identity(None, **_PASSING_KWARGS)


def test_check_identity_raises_current_epoch_mismatch() -> None:
    with pytest.raises(RuntimeError, match="current_epoch"):
        check_identity(_make_meta(current_epoch=4), **_PASSING_KWARGS)


def test_check_identity_raises_trainer_name_mismatch() -> None:
    with pytest.raises(RuntimeError, match="trainer_name"):
        check_identity(_make_meta(trainer_name="OtherTrainer"), **_PASSING_KWARGS)


def test_check_identity_raises_plans_mismatch() -> None:
    with pytest.raises(RuntimeError, match="plans"):
        check_identity(_make_meta(plans={"a": 999}), **_PASSING_KWARGS)


def test_check_identity_raises_configuration_mismatch() -> None:
    with pytest.raises(RuntimeError, match="configuration"):
        check_identity(_make_meta(configuration="2d"), **_PASSING_KWARGS)


def test_check_identity_raises_fold_mismatch() -> None:
    with pytest.raises(RuntimeError, match="fold"):
        check_identity(_make_meta(fold=0), **_PASSING_KWARGS)


def test_check_identity_raises_nonfinite_train_loss() -> None:
    with pytest.raises(RuntimeError, match="train_losses"):
        check_identity(_make_meta(train_loss_finite=False), **_PASSING_KWARGS)


# ---------------------------------------------------------------------------
# link_preprocessed / find_preprocessed_dataset
# ---------------------------------------------------------------------------


def _make_src_dataset(root: Path, name: str = _DATASET_FOLDER) -> Path:
    """Builds a synthetic read-only-style preprocessed dataset dir under `root`."""
    dataset_dir = root / name
    config_dir = dataset_dir / _CONFIG_DIR
    config_dir.mkdir(parents=True)
    (dataset_dir / "nnUNetPlans.json").write_text('{"plans": true}')
    (dataset_dir / "dataset.json").write_text('{"dataset": true}')
    (dataset_dir / "dataset_fingerprint.json").write_text('{"fp": true}')
    (config_dir / "case_001.b2nd").write_bytes(b"volume-bytes")
    (config_dir / "case_001.pkl").write_bytes(b"pickle-bytes")
    (config_dir / "case_001_seg.b2nd").write_bytes(b"seg-bytes")
    return dataset_dir


def test_link_preprocessed_builds_real_dir_with_per_file_symlinks(tmp_path: Path) -> None:
    src = _make_src_dataset(tmp_path / "src")
    dst_root = tmp_path / "dst"

    dataset_dir = link_preprocessed(src, dst_root, _DATASET_FOLDER)

    assert dataset_dir == dst_root / _DATASET_FOLDER
    for name in ("nnUNetPlans.json", "dataset.json", "dataset_fingerprint.json"):
        copied = dataset_dir / name
        assert copied.is_file()
        assert not copied.is_symlink()
        assert copied.read_text() == (src / name).read_text()

    config_dst = dataset_dir / _CONFIG_DIR
    assert config_dst.is_dir()
    assert not config_dst.is_symlink()  # a real directory, never a whole-dir symlink
    for name in ("case_001.b2nd", "case_001.pkl", "case_001_seg.b2nd"):
        linked = config_dst / name
        assert linked.is_symlink()
        assert linked.resolve() == (src / _CONFIG_DIR / name).resolve()


def test_link_preprocessed_is_idempotent(tmp_path: Path) -> None:
    src = _make_src_dataset(tmp_path / "src")
    dst_root = tmp_path / "dst"

    link_preprocessed(src, dst_root, _DATASET_FOLDER)
    dataset_dir = link_preprocessed(src, dst_root, _DATASET_FOLDER)  # must not raise

    linked = dataset_dir / _CONFIG_DIR / "case_001.b2nd"
    assert linked.is_symlink()
    assert linked.resolve() == (src / _CONFIG_DIR / "case_001.b2nd").resolve()


def test_link_preprocessed_raises_on_missing_plans_json(tmp_path: Path) -> None:
    src = tmp_path / "src" / _DATASET_FOLDER
    (src / _CONFIG_DIR).mkdir(parents=True)
    (src / "dataset.json").write_text("{}")
    (src / "dataset_fingerprint.json").write_text("{}")
    # nnUNetPlans.json deliberately missing.

    with pytest.raises(FileNotFoundError, match="nnUNetPlans.json"):
        link_preprocessed(src, tmp_path / "dst", _DATASET_FOLDER)


def test_find_preprocessed_dataset_one_candidate(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    src = _make_src_dataset(mount)
    found = find_preprocessed_dataset(mount, _DATASET_FOLDER)
    assert found == src.resolve()


def test_find_preprocessed_dataset_zero_candidates(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    with pytest.raises(FileNotFoundError, match="found 0"):
        find_preprocessed_dataset(mount, _DATASET_FOLDER)


def test_find_preprocessed_dataset_two_candidates(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    _make_src_dataset(mount / "run_a")
    _make_src_dataset(mount / "run_b")
    with pytest.raises(FileNotFoundError, match="found 2"):
        find_preprocessed_dataset(mount, _DATASET_FOLDER)


def test_link_preprocessed_copies_are_writable_even_from_readonly_source(tmp_path: Path) -> None:
    # A Kaggle input mount is read-only: chmod the source's top-level JSONs
    # read-only before copying, exactly like a real mount would present them.
    src = _make_src_dataset(tmp_path / "src")
    for name in ("nnUNetPlans.json", "dataset.json", "dataset_fingerprint.json"):
        (src / name).chmod(0o444)

    dataset_dir = link_preprocessed(src, tmp_path / "dst", _DATASET_FOLDER)

    for name in ("nnUNetPlans.json", "dataset.json", "dataset_fingerprint.json"):
        dst_file = dataset_dir / name
        assert os.access(dst_file, os.W_OK)
        # This is what nnU-Net's own on_train_start does on session 2: write
        # straight over the restored file. It must not raise PermissionError.
        dst_file.write_text('{"overwritten": true}')
        assert dst_file.read_text() == '{"overwritten": true}'


def test_link_preprocessed_and_find_with_non_default_plans_name(tmp_path: Path) -> None:
    # A non-default --plans (e.g. a ResEnc preset) must not be silently
    # matched against the hardcoded "nnUNetPlans" name.
    plans_identifier = "nnUNetResEncUNetMPlans"
    configuration_dir = f"{plans_identifier}_3d_fullres"
    mount = tmp_path / "mount"
    dataset_dir = mount / _DATASET_FOLDER
    config_dir = dataset_dir / configuration_dir
    config_dir.mkdir(parents=True)
    (dataset_dir / f"{plans_identifier}.json").write_text('{"plans": true}')
    (dataset_dir / "dataset.json").write_text('{"dataset": true}')
    (dataset_dir / "dataset_fingerprint.json").write_text('{"fp": true}')
    (config_dir / "case_001.b2nd").write_bytes(b"volume-bytes")

    found = find_preprocessed_dataset(mount, _DATASET_FOLDER, plans_identifier=plans_identifier)
    assert found == dataset_dir.resolve()

    # The DEFAULT plans_identifier must NOT find this dataset -- it has no
    # nnUNetPlans.json, only the custom name's.
    with pytest.raises(FileNotFoundError, match="found 0"):
        find_preprocessed_dataset(mount, _DATASET_FOLDER)

    linked_dir = link_preprocessed(
        found,
        tmp_path / "dst",
        _DATASET_FOLDER,
        configuration_dir=configuration_dir,
        plans_identifier=plans_identifier,
    )
    assert (linked_dir / f"{plans_identifier}.json").is_file()
    assert (linked_dir / configuration_dir / "case_001.b2nd").is_symlink()


# ---------------------------------------------------------------------------
# restore_results / find_prev_fold_dir
# ---------------------------------------------------------------------------


def test_restore_results_copies_fold_files_and_trainer_level_jsons(tmp_path: Path) -> None:
    prev_trainer_dir = tmp_path / "prev" / _TRAINER_DIR
    prev_fold_dir = prev_trainer_dir / _FOLD_NAME
    prev_fold_dir.mkdir(parents=True)
    (prev_fold_dir / "checkpoint_latest.pth").write_bytes(b"ckpt-bytes")
    (prev_fold_dir / "training_log_2026_01_01.txt").write_text("epoch 3 done")
    (prev_fold_dir / "progress.png").write_bytes(b"png-bytes")
    (prev_trainer_dir / "plans.json").write_text('{"plans": 1}')
    (prev_trainer_dir / "dataset.json").write_text('{"dataset": 1}')
    (prev_trainer_dir / "dataset_fingerprint.json").write_text('{"fp": 1}')

    dst_fold_dir = tmp_path / "dst" / _TRAINER_DIR / _FOLD_NAME
    restore_results(prev_fold_dir, dst_fold_dir)

    assert (dst_fold_dir / "checkpoint_latest.pth").read_bytes() == b"ckpt-bytes"
    assert (dst_fold_dir / "training_log_2026_01_01.txt").read_text() == "epoch 3 done"
    assert (dst_fold_dir / "progress.png").is_file()
    dst_trainer_dir = dst_fold_dir.parent
    assert (dst_trainer_dir / "plans.json").read_text() == '{"plans": 1}'
    assert (dst_trainer_dir / "dataset.json").is_file()
    assert (dst_trainer_dir / "dataset_fingerprint.json").is_file()


def test_restore_results_copies_are_writable_even_from_readonly_source(tmp_path: Path) -> None:
    # Exactly the failure the coordinator's CPU rehearsal hit: session 1's
    # results, read back from a read-only Kaggle input mount, must not carry
    # their read-only mode into session 2's writable results tree -- nnU-Net's
    # own on_train_start (plans.json) and save_checkpoint (checkpoint_latest.pth)
    # both write straight over these files.
    prev_trainer_dir = tmp_path / "prev" / _TRAINER_DIR
    prev_fold_dir = prev_trainer_dir / _FOLD_NAME
    prev_fold_dir.mkdir(parents=True)
    (prev_fold_dir / "checkpoint_latest.pth").write_bytes(b"ckpt-bytes")
    (prev_trainer_dir / "plans.json").write_text('{"plans": 1}')
    (prev_trainer_dir / "dataset.json").write_text('{"dataset": 1}')
    (prev_trainer_dir / "dataset_fingerprint.json").write_text('{"fp": 1}')
    (prev_fold_dir / "checkpoint_latest.pth").chmod(0o444)
    (prev_trainer_dir / "plans.json").chmod(0o444)

    dst_fold_dir = tmp_path / "dst" / _TRAINER_DIR / _FOLD_NAME
    restore_results(prev_fold_dir, dst_fold_dir)

    ckpt_dst = dst_fold_dir / "checkpoint_latest.pth"
    plans_dst = dst_fold_dir.parent / "plans.json"
    for dst_file in (ckpt_dst, plans_dst):
        assert os.access(dst_file, os.W_OK)
    ckpt_dst.write_bytes(b"overwritten")
    assert ckpt_dst.read_bytes() == b"overwritten"


def test_restore_results_none_is_a_no_op(tmp_path: Path) -> None:
    dst_fold_dir = tmp_path / "dst" / _TRAINER_DIR / _FOLD_NAME
    restore_results(None, dst_fold_dir)
    assert not dst_fold_dir.exists()


def test_find_prev_fold_dir_none(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    assert find_prev_fold_dir(mount, _TRAINER_DIR, _FOLD_NAME) is None


def test_find_prev_fold_dir_one(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    fold_dir = mount / "run1" / _TRAINER_DIR / _FOLD_NAME
    fold_dir.mkdir(parents=True)
    assert find_prev_fold_dir(mount, _TRAINER_DIR, _FOLD_NAME) == fold_dir.resolve()


def test_find_prev_fold_dir_two_raises(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    (mount / "run1" / _TRAINER_DIR / _FOLD_NAME).mkdir(parents=True)
    (mount / "run2" / _TRAINER_DIR / _FOLD_NAME).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        find_prev_fold_dir(mount, _TRAINER_DIR, _FOLD_NAME)


# ---------------------------------------------------------------------------
# read_checkpoint_meta
# ---------------------------------------------------------------------------


def test_read_checkpoint_meta(tmp_path: Path) -> None:
    checkpoint = {
        "current_epoch": 7,
        "trainer_name": "nnUNetTrainer",
        "init_args": {"plans": {"a": 1}, "configuration": "3d_fullres", "fold": "all"},
        "logging": {"train_losses": [0.5, 0.4, 0.3]},
    }
    path = tmp_path / "checkpoint_latest.pth"
    torch.save(checkpoint, path)

    meta = read_checkpoint_meta(path)

    assert meta == {
        "current_epoch": 7,
        "trainer_name": "nnUNetTrainer",
        "plans": {"a": 1},
        "configuration": "3d_fullres",
        "fold": "all",
        "train_loss_finite": True,
    }


def test_read_checkpoint_meta_detects_nonfinite_train_loss(tmp_path: Path) -> None:
    checkpoint = {
        "current_epoch": 1,
        "trainer_name": "nnUNetTrainer",
        "init_args": {"plans": {}, "configuration": "3d_fullres", "fold": "all"},
        "logging": {"train_losses": [0.1, math.nan]},
    }
    path = tmp_path / "checkpoint_latest.pth"
    torch.save(checkpoint, path)

    meta = read_checkpoint_meta(path)

    assert meta["train_loss_finite"] is False


# ---------------------------------------------------------------------------
# run_budgeted_training, against a FAKE trainer (no nnunetv2, no torch grads)
# ---------------------------------------------------------------------------


class _FakeLogger:
    """Minimal stand-in for nnU-Net's own `MetaLogger`/`LocalLogger`."""

    def __init__(self) -> None:
        self.data: dict[str, list] = {
            "train_losses": [],
            "dice_per_class_or_region": [],
            "ema_fg_dice": [],
        }

    def log(self, key: str, value: object, step: int) -> None:
        self.data.setdefault(key, []).append(value)

    def get_value(self, key: str, step: int | None) -> object:
        values = self.data.get(key, [])
        return values if step is None else values[step]


class _FakeTrainer:
    """A plain-Python stand-in for `nnUNetTrainer`, recording every hook call.

    Reproduces just enough of the real trainer's interface for
    `run_budgeted_training` to drive it: the hook methods it calls, in the
    order it calls them, plus the handful of attributes it reads
    (`current_epoch`, `num_epochs`, the two iteration counts, the two
    dataloaders, `device`, `logger`, `save_every`).
    """

    def __init__(
        self,
        num_epochs: int = 5,
        current_epoch: int = 0,
        num_iterations_per_epoch: int = 2,
        num_val_iterations_per_epoch: int = 2,
    ) -> None:
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_val_iterations_per_epoch = num_val_iterations_per_epoch
        self.dataloader_train = itertools.repeat("train_batch")
        self.dataloader_val = itertools.repeat("val_batch")
        self.device = torch.device("cpu")
        self.logger = _FakeLogger()
        self.save_every = 50  # upstream's real default; run_budgeted_training must override it
        self.calls: list[str] = []
        self.on_train_end_calls = 0

    def on_train_start(self) -> None:
        self.calls.append("on_train_start")

    def on_epoch_start(self) -> None:
        self.calls.append("on_epoch_start")

    def on_train_epoch_start(self) -> None:
        self.calls.append("on_train_epoch_start")

    def train_step(self, batch: object) -> dict[str, float]:
        self.calls.append("train_step")
        return {"loss": 0.1}

    def on_train_epoch_end(self, outputs: list[dict[str, float]]) -> None:
        self.calls.append("on_train_epoch_end")
        mean_loss = sum(o["loss"] for o in outputs) / len(outputs)
        self.logger.log("train_losses", mean_loss, self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.calls.append("on_validation_epoch_start")

    def validation_step(self, batch: object) -> dict[str, float]:
        self.calls.append("validation_step")
        return {"loss": 0.2}

    def on_validation_epoch_end(self, outputs: list[dict[str, float]]) -> None:
        self.calls.append("on_validation_epoch_end")
        self.logger.log("dice_per_class_or_region", [0.8, 0.7, 0.9], self.current_epoch)
        self.logger.log("ema_fg_dice", 0.75, self.current_epoch)

    def on_epoch_end(self) -> None:
        self.calls.append("on_epoch_end")
        self.current_epoch += 1

    def on_train_end(self) -> None:
        self.calls.append("on_train_end")
        self.on_train_end_calls += 1


class _CountingClock:
    """A deterministic fake wall clock: each call returns `t`, then `t += step`."""

    def __init__(self, step: float = 1.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.t
        self.t += self.step
        return value


def _install_fake_batchgenerators(monkeypatch: pytest.MonkeyPatch) -> type:
    """Registers fake `batchgenerators` augmenter modules/classes in `sys.modules`.

    `batchgenerators` (an nnunetv2 dependency) is not installed in the main
    `.venv`, so this fakes just enough of its package structure for
    `from batchgenerators.dataloading.multi_threaded_augmenter import
    MultiThreadedAugmenter` (and the `nondet_...` sibling) to succeed, letting
    the test exercise the real isinstance-gated shutdown branch in
    `nnunet_session._shutdown_dataloaders` rather than only its ImportError
    fallback.

    Returns:
        A fake augmenter class with a `_finish` method the test can assert on.
    """

    class _FakeAugmenter:
        def __init__(self) -> None:
            self._it = itertools.repeat("batch")
            self.finish_called = False

        def __next__(self) -> str:
            return next(self._it)

        def _finish(self) -> None:
            self.finish_called = True

    for mod_name, attr_name in (
        ("batchgenerators", None),
        ("batchgenerators.dataloading", None),
        ("batchgenerators.dataloading.multi_threaded_augmenter", "MultiThreadedAugmenter"),
        (
            "batchgenerators.dataloading.nondet_multi_threaded_augmenter",
            "NonDetMultiThreadedAugmenter",
        ),
    ):
        fake_module = types.ModuleType(mod_name)
        if attr_name is not None:
            setattr(fake_module, attr_name, _FakeAugmenter)
        monkeypatch.setitem(sys.modules, mod_name, fake_module)

    return _FakeAugmenter


def test_run_budgeted_training_large_budget_runs_all_epochs_and_finishes() -> None:
    trainer = _FakeTrainer(num_epochs=5, current_epoch=0)
    result = run_budgeted_training(
        trainer, budget_s=1e9, first_epoch_estimate_s=1.0, clock=_CountingClock(step=1.0)
    )

    assert result["finished"] is True
    assert result["epochs_run"] == 5
    assert result["start_epoch"] == 0
    assert result["end_epoch"] == 5
    assert trainer.on_train_end_calls == 1


def test_run_budgeted_training_stops_at_exactly_2_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_augmenter_cls = _install_fake_batchgenerators(monkeypatch)
    trainer = _FakeTrainer(num_epochs=5, current_epoch=0)
    train_augmenter = fake_augmenter_cls()
    trainer.dataloader_train = train_augmenter

    # Worked out from _CountingClock(step=1.0), first_epoch_estimate_s=1.0,
    # safety_factor=1.5 (default): the budget check before epoch 2 sees
    # elapsed=7.0, predicted=1.5, and 7.0 + 1.5 = 8.5 > 8.0 -- so exactly 2
    # epochs (0 and 1) run before the 3rd is refused.
    result = run_budgeted_training(
        trainer, budget_s=8.0, first_epoch_estimate_s=1.0, clock=_CountingClock(step=1.0)
    )

    assert result["epochs_run"] == 2
    assert result["finished"] is False
    assert result["start_epoch"] == 0
    assert result["end_epoch"] == 2
    assert trainer.on_train_end_calls == 0
    assert train_augmenter.finish_called is True  # dataloader shutdown was actually reached


def test_run_budgeted_training_resumes_from_nonzero_current_epoch() -> None:
    trainer = _FakeTrainer(num_epochs=5, current_epoch=3)
    result = run_budgeted_training(
        trainer, budget_s=1e9, first_epoch_estimate_s=1.0, clock=_CountingClock(step=1.0)
    )

    assert result["start_epoch"] == 3
    assert result["end_epoch"] == 5
    assert result["epochs_run"] == 2
    assert result["finished"] is True
    assert trainer.on_train_end_calls == 1


def test_run_budgeted_training_hook_call_order_matches_upstream() -> None:
    trainer = _FakeTrainer(
        num_epochs=1, current_epoch=0, num_iterations_per_epoch=2, num_val_iterations_per_epoch=2
    )
    run_budgeted_training(
        trainer, budget_s=1e9, first_epoch_estimate_s=1.0, clock=_CountingClock(step=1.0)
    )

    # nnUNetTrainer.run_training's own order (nnunetv2 v2.8.1,
    # training/nnUNetTrainer/nnUNetTrainer.py, lines 1417-1438).
    assert trainer.calls == [
        "on_train_start",
        "on_epoch_start",
        "on_train_epoch_start",
        "train_step",
        "train_step",
        "on_train_epoch_end",
        "on_validation_epoch_start",
        "validation_step",
        "validation_step",
        "on_validation_epoch_end",
        "on_epoch_end",
        "on_train_end",
    ]


def test_run_budgeted_training_sets_save_every_to_one() -> None:
    trainer = _FakeTrainer(num_epochs=1, current_epoch=0)
    assert trainer.save_every == 50  # upstream's real default, before the call
    run_budgeted_training(
        trainer, budget_s=1e9, first_epoch_estimate_s=1.0, clock=_CountingClock(step=1.0)
    )
    assert trainer.save_every == 1


def test_run_budgeted_training_elapsed_before_s_can_exhaust_budget_immediately() -> None:
    # Regression for the review finding: the budget clock must start at
    # process start (main()'s CLI parsing, CUDA check, dataset linking,
    # trainer construction -- all real wall-clock time), not at the moment
    # run_budgeted_training happens to be entered. elapsed_before_s=9.0
    # against budget_s=10.0 leaves no room even for the FIRST epoch.
    trainer = _FakeTrainer(num_epochs=5, current_epoch=0)

    result = run_budgeted_training(
        trainer,
        budget_s=10.0,
        first_epoch_estimate_s=1.0,
        clock=_CountingClock(step=1.0),
        elapsed_before_s=9.0,
    )

    assert result["epochs_run"] == 0
    assert result["finished"] is False
    assert result["start_epoch"] == 0
    assert result["end_epoch"] == 0
    assert trainer.on_train_end_calls == 0
    assert trainer.calls == ["on_train_start"]  # the loop body never ran
