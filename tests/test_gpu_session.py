"""Tests for scripts/gpu_session.py.

Every check here runs on tiny synthetic data, on CPU, in well under a second
each -- except the one full end-to-end run, which is the whole point of that
test and is reported separately (see this module's own timing note at the
bottom). `scripts/` is not an importable package, so the script under test
is loaded via `tests.script_loader.load_script`, the same pattern every other
`tests/test_*_script.py` file uses.

Nothing here touches a GPU, real BraTS data, or the real repo's `outputs/`
directory -- every path is built under pytest's `tmp_path`.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from neurovision.training.checkpoint import save_checkpoint
from neurovision.utils.io import write_yaml
from tests.script_loader import load_script

gpu_session_script = load_script("gpu_session")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _minimal_data_root(tmp_path: Path) -> Path:
    """A data root with the right shape but no real case files.

    Enough for every test that stops before `build_dataloaders` actually
    reads a case (preflight checks, --dry-run, ckpt-src validation) -- those
    only need `resolve_data_root` and Hydra composition to succeed.
    """
    root = tmp_path / "data_root"
    (root / "preprocessed").mkdir(parents=True)
    write_yaml({"train": [], "val": [], "test": []}, root / "splits.yaml")
    return root


def _build_and_save_checkpoint(
    out_dir: Path,
    epoch: int = 3,
    wandb_run_id: str | None = "run-abc",
    corrupt_model: bool = False,
    corrupt_optimizer: bool = False,
    also_best: bool = True,
) -> Path:
    """Writes a real, loadable last.pt (+ best.pt) under `out_dir`.

    One optimizer step is run first so Adam's `state` dict is actually
    populated (exp_avg / exp_avg_sq) -- a freshly constructed optimizer has
    an empty state, which would make the "NaN in optimizer state" tests
    vacuous.
    """
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()

    if corrupt_model:
        with torch.no_grad():
            next(model.parameters())[0, 0] = float("nan")
    if corrupt_optimizer:
        state = next(iter(optimizer.state.values()))
        state["exp_avg"][0] = float("inf")

    save_checkpoint(
        out_dir,
        model,
        optimizer,
        epoch=epoch,
        global_step=epoch * 10,
        wandb_run_id=wandb_run_id,
        is_best=also_best,
    )
    return out_dir


def _init_git_repo(path: Path) -> str:
    """Initializes a throwaway git repo with one commit, returns its SHA."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# 1. Data-root discovery (mirrors kaggle_train.ipynb cell 9)
# ---------------------------------------------------------------------------


def test_discover_data_root_finds_dir_within_depth(tmp_path: Path):
    target = tmp_path / "a" / "b"  # depth 2, within the default max_depth of 3
    (target / "preprocessed").mkdir(parents=True)
    write_yaml({"train": [], "val": [], "test": []}, target / "splits.yaml")

    found = gpu_session_script.discover_data_root(tmp_path)

    assert found == target


def test_discover_data_root_fails_beyond_depth(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c" / "d"  # depth 4, beyond the default max_depth of 3
    (target / "preprocessed").mkdir(parents=True)
    write_yaml({"train": [], "val": [], "test": []}, target / "splits.yaml")

    with pytest.raises(FileNotFoundError, match="Expected exactly one"):
        gpu_session_script.discover_data_root(tmp_path)


def test_discover_data_root_fails_on_zero_hits(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Expected exactly one"):
        gpu_session_script.discover_data_root(tmp_path)


def test_resolve_data_root_with_explicit_data_root(tmp_path: Path):
    root = _minimal_data_root(tmp_path)
    assert gpu_session_script.resolve_data_root(str(root), None) == root


def test_resolve_data_root_with_data_root_missing_layout_raises(tmp_path: Path):
    root = tmp_path / "data_root"
    root.mkdir()  # no preprocessed/, no splits.yaml
    with pytest.raises(FileNotFoundError, match="does not hold both"):
        gpu_session_script.resolve_data_root(str(root), None)


def test_resolve_data_root_uses_search_root(tmp_path: Path):
    target = tmp_path / "mount" / "dataset"
    (target / "preprocessed").mkdir(parents=True)
    write_yaml({"train": [], "val": [], "test": []}, target / "splits.yaml")

    assert gpu_session_script.resolve_data_root(None, str(tmp_path)) == target


# ---------------------------------------------------------------------------
# 2. Checkpoint carry-forward + identity checks (notebook cell 11)
# ---------------------------------------------------------------------------


def test_copy_checkpoints_copies_both_last_and_best(tmp_path: Path):
    src = tmp_path / "prior_session"
    _build_and_save_checkpoint(src, epoch=10, also_best=True)
    dst_dir = tmp_path / "checkpoints"

    last_dst, best_dst = gpu_session_script.copy_checkpoints(src, dst_dir)

    assert last_dst == dst_dir / "last.pt"
    assert last_dst.is_file()
    assert best_dst == dst_dir / "best.pt"
    assert best_dst is not None and best_dst.is_file()


def test_copy_checkpoints_missing_last_raises(tmp_path: Path):
    src = tmp_path / "empty_src"
    src.mkdir()
    with pytest.raises(FileNotFoundError, match="last.pt"):
        gpu_session_script.copy_checkpoints(src, tmp_path / "checkpoints")


def test_copy_checkpoints_refuses_to_roll_back(tmp_path: Path):
    src = tmp_path / "prior_session"
    _build_and_save_checkpoint(src, epoch=3)
    dst_dir = tmp_path / "checkpoints"
    _build_and_save_checkpoint(dst_dir, epoch=10, wandb_run_id="already-ahead")

    with pytest.raises(RuntimeError, match="roll progress back"):
        gpu_session_script.copy_checkpoints(src, dst_dir)

    # Destination must be untouched -- still epoch 10, not overwritten by src's epoch 3.
    payload = torch.load(dst_dir / "last.pt", map_location="cpu", weights_only=True)
    assert payload["epoch"] == 10
    assert payload["wandb_run_id"] == "already-ahead"


def test_copy_checkpoints_skips_when_epochs_equal(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    src = tmp_path / "prior_session"
    _build_and_save_checkpoint(src, epoch=7, wandb_run_id="src-run")
    dst_dir = tmp_path / "checkpoints"
    _build_and_save_checkpoint(dst_dir, epoch=7, wandb_run_id="dst-run")

    with caplog.at_level(logging.INFO):
        last_dst, _ = gpu_session_script.copy_checkpoints(src, dst_dir)

    # The copy was skipped -- the destination's OWN payload must survive.
    payload = torch.load(last_dst, map_location="cpu", weights_only=True)
    assert payload["wandb_run_id"] == "dst-run"
    assert any("skipping the copy" in record.message for record in caplog.records)


def test_main_ckpt_expect_epoch_mismatch_exits_nonzero(tmp_path: Path):
    data_root = _minimal_data_root(tmp_path)
    ckpt_src = tmp_path / "prior_session"
    _build_and_save_checkpoint(ckpt_src, epoch=5, wandb_run_id="abc")

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--ckpt-src",
        str(ckpt_src),
        "--expect-epoch",
        "999",
        "--wandb-mode",
        "disabled",
        "--dry-run",
        "--override",
        f"output_dir={tmp_path / 'outputs'}",
    ]

    assert gpu_session_script.main(argv) == 1


def test_main_ckpt_expect_wandb_id_mismatch_exits_nonzero(tmp_path: Path):
    data_root = _minimal_data_root(tmp_path)
    ckpt_src = tmp_path / "prior_session"
    _build_and_save_checkpoint(ckpt_src, epoch=5, wandb_run_id="the-real-run")

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--ckpt-src",
        str(ckpt_src),
        "--expect-wandb-id",
        "some-other-run",
        "--wandb-mode",
        "disabled",
        "--dry-run",
        "--override",
        f"output_dir={tmp_path / 'outputs'}",
    ]

    assert gpu_session_script.main(argv) == 1


def test_main_ckpt_matching_expectations_dry_run_succeeds(tmp_path: Path):
    data_root = _minimal_data_root(tmp_path)
    ckpt_src = tmp_path / "prior_session"
    _build_and_save_checkpoint(ckpt_src, epoch=5, wandb_run_id="the-real-run")

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--ckpt-src",
        str(ckpt_src),
        "--expect-epoch",
        "5",
        "--expect-wandb-id",
        "the-real-run",
        "--wandb-mode",
        "disabled",
        "--dry-run",
        "--override",
        f"output_dir={tmp_path / 'outputs'}",
    ]

    assert gpu_session_script.main(argv) == 0


# ---------------------------------------------------------------------------
# 2b. In-place resume (college cluster: no --ckpt-src, last.pt already sits
#     in the run's own checkpoint dir because the disk is persistent)
# ---------------------------------------------------------------------------


def test_run_preflight_verifies_in_place_checkpoint_without_ckpt_src(tmp_path: Path):
    data_root = _minimal_data_root(tmp_path)
    output_dir = tmp_path / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    _build_and_save_checkpoint(checkpoint_dir, epoch=12, wandb_run_id="cluster-run")

    args = gpu_session_script.parse_args(
        [
            "--experiment",
            "overfit2",
            "--data-root",
            str(data_root),
            "--wandb-mode",
            "disabled",
            "--expect-epoch",
            "12",
            "--expect-wandb-id",
            "cluster-run",
            "--override",
            f"output_dir={output_dir}",
        ]
    )

    cfg = gpu_session_script.run_preflight(args)  # must not raise

    assert Path(cfg.training.checkpoint.dir) == checkpoint_dir


def test_run_preflight_in_place_epoch_mismatch_raises(tmp_path: Path):
    data_root = _minimal_data_root(tmp_path)
    output_dir = tmp_path / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    _build_and_save_checkpoint(checkpoint_dir, epoch=12)

    args = gpu_session_script.parse_args(
        [
            "--experiment",
            "overfit2",
            "--data-root",
            str(data_root),
            "--wandb-mode",
            "disabled",
            "--expect-epoch",
            "999",
            "--override",
            f"output_dir={output_dir}",
        ]
    )

    with pytest.raises(RuntimeError, match="epoch"):
        gpu_session_script.run_preflight(args)


def test_run_preflight_expect_epoch_without_any_checkpoint_fails(tmp_path: Path):
    data_root = _minimal_data_root(tmp_path)
    output_dir = tmp_path / "outputs"  # no checkpoints/ dir at all

    args = gpu_session_script.parse_args(
        [
            "--experiment",
            "overfit2",
            "--data-root",
            str(data_root),
            "--wandb-mode",
            "disabled",
            "--expect-epoch",
            "5",
            "--override",
            f"output_dir={output_dir}",
        ]
    )

    with pytest.raises(FileNotFoundError, match="nothing to check"):
        gpu_session_script.run_preflight(args)


def test_main_expect_epoch_without_any_checkpoint_exits_nonzero(tmp_path: Path):
    data_root = _minimal_data_root(tmp_path)
    output_dir = tmp_path / "outputs"

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--wandb-mode",
        "disabled",
        "--expect-epoch",
        "5",
        "--dry-run",
        "--override",
        f"output_dir={output_dir}",
    ]

    assert gpu_session_script.main(argv) == 1


# ---------------------------------------------------------------------------
# 3. NaN/Inf detection in model and optimizer state
# ---------------------------------------------------------------------------


def test_find_nonfinite_model_tensors_detects_and_names():
    state_dict = {
        "encoder.weight": torch.tensor([1.0, 2.0]),
        "decoder.bias": torch.tensor([float("nan"), 0.0]),
    }
    assert gpu_session_script.find_nonfinite_model_tensors(state_dict) == ["decoder.bias"]


def test_find_nonfinite_optimizer_tensors_detects_and_names():
    optimizer_state_dict = {
        "state": {
            0: {
                "exp_avg": torch.tensor([0.1, 0.2]),
                "exp_avg_sq": torch.tensor([float("inf"), 0.0]),
            }
        },
        "param_groups": [],
    }
    bad = gpu_session_script.find_nonfinite_optimizer_tensors(optimizer_state_dict)
    assert bad == ["optimizer.state[0].exp_avg_sq"]


def test_verify_checkpoint_finite_raises_on_corrupt_model(tmp_path: Path):
    out_dir = tmp_path / "ckpt"
    _build_and_save_checkpoint(out_dir, corrupt_model=True)
    payload = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=True)

    with pytest.raises(RuntimeError, match="non-finite"):
        gpu_session_script.verify_checkpoint_finite(payload)


def test_verify_checkpoint_finite_raises_on_corrupt_optimizer(tmp_path: Path):
    out_dir = tmp_path / "ckpt"
    _build_and_save_checkpoint(out_dir, corrupt_optimizer=True)
    payload = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=True)

    with pytest.raises(RuntimeError, match="non-finite"):
        gpu_session_script.verify_checkpoint_finite(payload)


def test_verify_checkpoint_finite_passes_on_healthy_checkpoint(tmp_path: Path):
    out_dir = tmp_path / "ckpt"
    _build_and_save_checkpoint(out_dir)
    payload = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=True)

    gpu_session_script.verify_checkpoint_finite(payload)  # must not raise


# ---------------------------------------------------------------------------
# 4. --expect-sha
# ---------------------------------------------------------------------------


def test_check_git_sha_mismatch_raises(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    with pytest.raises(RuntimeError, match="HEAD is at"):
        gpu_session_script.check_git_sha(repo, expect_sha="0" * 40)


def test_check_git_sha_match_returns_sha(tmp_path: Path):
    repo = tmp_path / "repo"
    sha = _init_git_repo(repo)

    assert gpu_session_script.check_git_sha(repo, expect_sha=sha) == sha


def test_check_git_sha_no_expectation_never_raises(tmp_path: Path):
    repo = tmp_path / "repo"
    sha = _init_git_repo(repo)

    assert gpu_session_script.check_git_sha(repo, expect_sha=None) == sha


# ---------------------------------------------------------------------------
# 5. --require-cuda on a CPU box
# ---------------------------------------------------------------------------


def test_run_cuda_smoke_check_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gpu_session_script.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="require-cuda"):
        gpu_session_script.run_cuda_smoke_check()


def test_main_require_cuda_fails_cleanly_on_cpu_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(gpu_session_script.torch.cuda, "is_available", lambda: False)
    data_root = _minimal_data_root(tmp_path)

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--require-cuda",
        "--wandb-mode",
        "disabled",
        "--dry-run",
    ]

    assert gpu_session_script.main(argv) == 1


# ---------------------------------------------------------------------------
# 6. --dry-run runs preflight only
# ---------------------------------------------------------------------------


def test_dry_run_succeeds_without_ckpt_src(tmp_path: Path):
    data_root = _minimal_data_root(tmp_path)

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--wandb-mode",
        "disabled",
        "--dry-run",
    ]

    assert gpu_session_script.main(argv) == 0


def test_dry_run_never_calls_load_run_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = _minimal_data_root(tmp_path)

    def _must_not_be_called() -> None:
        raise AssertionError("load_run_training must not be called during --dry-run")

    monkeypatch.setattr(gpu_session_script, "load_run_training", _must_not_be_called)

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--wandb-mode",
        "disabled",
        "--dry-run",
    ]

    assert gpu_session_script.main(argv) == 0


# ---------------------------------------------------------------------------
# 7. NVX_HEALTH line format, and the metrics-side NaN check (CLAUDE.md trap #2:
#    AMP's GradScaler skips a step with inf/NaN grads, so WEIGHTS stay finite
#    while the LOSS is NaN for the whole run -- the checkpoint scan alone
#    would print OK on exactly that failure).
# ---------------------------------------------------------------------------


def test_print_health_line_ok(capsys: pytest.CaptureFixture[str]):
    line = gpu_session_script.print_health_line(7, 5, 0.1234, 2.5, [])

    assert (
        line == "NVX_HEALTH: OK | epoch=7 | best_epoch=5 | loss=0.1234 | "
        "grad_norm_median=2.5 | nonfinite=[]"
    )
    assert line in capsys.readouterr().out


def test_print_health_line_nan(capsys: pytest.CaptureFixture[str]):
    line = gpu_session_script.print_health_line(3, None, float("nan"), None, ["decoder.weight"])

    assert (
        line == "NVX_HEALTH: NAN | epoch=3 | best_epoch=None | loss=nan | "
        "grad_norm_median=None | nonfinite=['decoder.weight']"
    )
    assert line in capsys.readouterr().out


def test_find_nonfinite_metrics_detects_nan_loss():
    metrics = {"train/loss_epoch": float("nan"), "val/dice_mean": 0.9, "epoch": 3}
    assert gpu_session_script.find_nonfinite_metrics(metrics) == ["train/loss_epoch"]


def test_find_nonfinite_metrics_all_finite_is_empty():
    metrics = {"train/loss_epoch": 0.5, "val/dice_mean": 0.9}
    assert gpu_session_script.find_nonfinite_metrics(metrics) == []


def test_read_best_epoch_absent_is_none(tmp_path: Path):
    assert gpu_session_script.read_best_epoch(tmp_path / "no_such_dir") is None


def test_read_best_epoch_reads_saved_epoch(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    _build_and_save_checkpoint(checkpoint_dir, epoch=9, also_best=True)

    assert gpu_session_script.read_best_epoch(checkpoint_dir) == 9


def test_main_nan_loss_with_finite_checkpoint_is_nan_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The exact CLAUDE.md trap #2 shape: a perfectly finite checkpoint, a NaN loss.

    `run_training` is monkeypatched (not run for real) so this isolates the
    health-check logic itself; `test_end_to_end_overfit2_one_epoch_prints_health_ok`
    below covers the real pipeline.
    """
    data_root = _minimal_data_root(tmp_path)
    output_dir = tmp_path / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    _build_and_save_checkpoint(checkpoint_dir, epoch=1, also_best=True)  # finite

    def _fake_run_training(cfg) -> dict[str, float]:
        return {"train/loss_epoch": float("nan"), "train/grad_norm_median": 0.5}

    monkeypatch.setattr(gpu_session_script, "load_run_training", lambda: _fake_run_training)

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--wandb-mode",
        "disabled",
        "--override",
        f"output_dir={output_dir}",
    ]

    code = gpu_session_script.main(argv)
    captured = capsys.readouterr()

    assert code == 1
    assert "NVX_HEALTH: NAN" in captured.out
    assert "train/loss_epoch" in captured.out


# ---------------------------------------------------------------------------
# 8. One real, full end-to-end CPU run (overfit2, 1 epoch)
# ---------------------------------------------------------------------------


def test_end_to_end_overfit2_one_epoch_prints_health_ok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The real pipeline, real Hydra config, real Trainer -- nothing stubbed.

    Two tiny synthetic cases sized exactly to the (overridden) patch size, so
    RandCropByPosNegLabeld and sliding-window validation each need only one
    crop/window. `+experiment=overfit2` trains and validates on the same 2
    cases (`data.overfit_n: 2`), so a single epoch already produces a
    val/dice_mean and a best.pt.
    """
    data_root = tmp_path / "data_root"
    prep_dir = data_root / "preprocessed"
    case_ids = ("case_000", "case_001")
    for i, case_id in enumerate(case_ids):
        case_dir = prep_dir / case_id
        case_dir.mkdir(parents=True)
        rng = np.random.default_rng(i)
        image = rng.standard_normal((4, 32, 32, 32)).astype(np.float16)
        label = np.zeros((32, 32, 32), dtype=np.uint8)
        label[8:16, 8:16, 8:16] = 1  # NCR/NET -> in TC, WT
        label[18:24, 18:24, 18:24] = 3  # ET -> in ET, TC, WT
        np.save(case_dir / "image.npy", image)
        np.save(case_dir / "label.npy", label)
    write_yaml({"train": list(case_ids), "val": [], "test": []}, data_root / "splits.yaml")

    argv = [
        "--experiment",
        "overfit2",
        "--data-root",
        str(data_root),
        "--wandb-mode",
        "disabled",
        "--override",
        f"output_dir={tmp_path / 'outputs'}",
        "--override",
        "device=cpu",
        "--override",
        "data.patch_size=[32,32,32]",
        "--override",
        "training.epochs=1",
        "--override",
        "training.val_interval=1",
    ]

    code = gpu_session_script.main(argv)
    captured = capsys.readouterr()

    assert code == 0
    assert "NVX_HEALTH: OK" in captured.out
