"""Tests for the shell wrappers in scripts/cluster/.

No tmux, sbatch, GPU or data is needed: every script has an NVX_PRINT_ONLY=1
mode that prints the command it would run, and that is what is checked here.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLUSTER_DIR = Path(__file__).resolve().parent.parent / "scripts" / "cluster"
SCRIPTS = ["run_tmux.sh", "run.sbatch", "nnunet.sh"]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def _run(script: str, args: list[str], **env_vars: str) -> subprocess.CompletedProcess[str]:
    """Runs a cluster script with a clean NVX_/SLURM_ environment plus `env_vars`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NVX_", "SLURM_"))}
    env["PYTHON"] = sys.executable  # an absolute, always-present interpreter
    env.update(env_vars)
    return subprocess.run(
        ["bash", str(CLUSTER_DIR / script), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )


def _printed_command(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("NVX_CMD: ")]
    assert len(lines) == 1, result.stdout
    return lines[0]


SBATCH_ENV = {
    "NVX_EXPERIMENT": "neurovision",
    "NVX_DATA_ROOT": "/data/brats",
    "NVX_MAX_HOURS": "10",
    "NVX_PRINT_ONLY": "1",
}
NNUNET_ENV = {
    "NVX_NNUNET_RAW": "/n/raw",
    "NVX_NNUNET_PREPROCESSED": "/n/pre",
    "NVX_NNUNET_RESULTS": "/n/res",
    "NVX_BUDGET_HOURS": "10.5",
    "NVX_EXPECT_START_EPOCH": "413",
    "NVX_SUMMARY_JSON": "/n/summary.json",
    "NVX_PRINT_ONLY": "1",
}


@pytest.mark.parametrize("script", SCRIPTS)
def test_bash_syntax(script: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(CLUSTER_DIR / script)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_run_tmux_requires_log_dir() -> None:
    result = _run("run_tmux.sh", ["job", "--", "--experiment", "x"])
    assert result.returncode != 0
    assert "NVX_LOG_DIR" in result.stderr


def test_run_tmux_requires_double_dash_and_args() -> None:
    result = _run("run_tmux.sh", ["job"], NVX_LOG_DIR="/logs")
    assert result.returncode != 0
    assert "expected '<session-name> -- <gpu_session.py args...>'" in result.stderr


def test_run_tmux_print_only() -> None:
    result = _run(
        "run_tmux.sh",
        ["job", "--", "--experiment", "neurovision", "--data-root", "/d"],
        NVX_LOG_DIR="/logs",
        NVX_PRINT_ONLY="1",
        NVX_EXPECT_SHA="abc123",
    )
    cmd = _printed_command(result)
    assert "scripts/gpu_session.py" in cmd
    assert "--experiment neurovision --data-root /d" in cmd
    assert "--expect-sha abc123" in cmd
    assert "NVX_LOG: /logs/job_" in result.stdout


def test_run_tmux_omits_sha_when_unset() -> None:
    result = _run(
        "run_tmux.sh",
        ["job", "--", "--experiment", "x"],
        NVX_LOG_DIR="/logs",
        NVX_PRINT_ONLY="1",
    )
    assert "--expect-sha" not in _printed_command(result)


def test_run_tmux_rejects_bad_session_name() -> None:
    result = _run("run_tmux.sh", ["a.b", "--", "--experiment", "x"], NVX_LOG_DIR="/logs")
    assert result.returncode != 0
    assert "session name" in result.stderr


@pytest.mark.parametrize("missing", ["NVX_EXPERIMENT", "NVX_MAX_HOURS", "NVX_DATA_ROOT"])
def test_sbatch_missing_required_var(missing: str) -> None:
    env = {k: v for k, v in SBATCH_ENV.items() if k != missing}
    result = _run("run.sbatch", [], **env)
    assert result.returncode != 0
    # Dropping NVX_DATA_ROOT is reported as "set NVX_DATA_ROOT (or NVX_SEARCH_ROOT)".
    assert missing in result.stderr


def test_sbatch_print_only_minimal() -> None:
    cmd = _printed_command(_run("run.sbatch", [], **SBATCH_ENV))
    assert "scripts/gpu_session.py" in cmd
    assert "--experiment neurovision" in cmd
    assert "--data-root /data/brats" in cmd
    assert "--require-cuda" in cmd
    assert "--override training.max_hours=10" in cmd
    for omitted in ("--ckpt-src", "--expect-epoch", "--expect-sha", "--wandb-mode"):
        assert omitted not in cmd


def test_sbatch_print_only_all_optionals() -> None:
    cmd = _printed_command(
        _run(
            "run.sbatch",
            [],
            NVX_CKPT_SRC="/ckpt",
            NVX_EXPECT_EPOCH="35",
            NVX_EXPECT_SHA="abc123",
            NVX_WANDB_MODE="offline",
            **SBATCH_ENV,
        )
    )
    assert "--ckpt-src /ckpt" in cmd
    assert "--expect-epoch 35" in cmd
    assert "--expect-sha abc123" in cmd
    assert "--wandb-mode offline" in cmd


def test_sbatch_rejects_non_numeric_max_hours() -> None:
    result = _run("run.sbatch", [], **{**SBATCH_ENV, "NVX_MAX_HOURS": "soon"})
    assert result.returncode != 0
    assert "NVX_MAX_HOURS" in result.stderr


def test_sbatch_wall_limit_guard() -> None:
    one_hour_left = str(int(time.time()) + 3600)
    too_long = _run(
        "run.sbatch", [], **{**SBATCH_ENV, "NVX_MAX_HOURS": "0.8"}, SLURM_JOB_END_TIME=one_hour_left
    )
    assert too_long.returncode != 0
    assert "wall" in too_long.stderr
    fine = _run(
        "run.sbatch", [], **{**SBATCH_ENV, "NVX_MAX_HOURS": "0.4"}, SLURM_JOB_END_TIME=one_hour_left
    )
    assert "training.max_hours=0.4" in _printed_command(fine)


@pytest.mark.parametrize(
    "missing",
    [k for k in NNUNET_ENV if k != "NVX_PRINT_ONLY"],
)
def test_nnunet_missing_required_var(missing: str) -> None:
    result = _run("nnunet.sh", [], **{k: v for k, v in NNUNET_ENV.items() if k != missing})
    assert result.returncode != 0
    assert missing in result.stderr


def test_nnunet_print_only() -> None:
    cmd = _printed_command(_run("nnunet.sh", [], **NNUNET_ENV))
    for expected in (
        "scripts/nnunet_session.py",
        "PYTHONPATH=",
        "--raw-root /n/raw",
        "--preprocessed-root /n/pre",
        "--results-root /n/res",
        "--budget-hours 10.5",
        "--expect-start-epoch 413",
        "--device cuda",
        "--require-cuda",
        "--summary-json /n/summary.json",
    ):
        assert expected in cmd
    assert "--search-root" not in cmd
    assert "--prev-results-search-root" not in cmd


def test_nnunet_print_only_optionals() -> None:
    cmd = _printed_command(
        _run(
            "nnunet.sh",
            [],
            NVX_NNUNET_SEARCH_ROOT="/in",
            NVX_NNUNET_PREV_RESULTS_ROOT="/prev",
            **NNUNET_ENV,
        )
    )
    assert "--search-root /in" in cmd
    assert "--prev-results-search-root /prev" in cmd


# ---------------------------------------------------------------------------
# Guards and stubs
# ---------------------------------------------------------------------------


def _stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _with_path(directory: Path) -> str:
    return f"{directory}{os.pathsep}{os.environ['PATH']}"


def test_run_tmux_rejects_empty_session_name() -> None:
    result = _run("run_tmux.sh", ["", "--", "--experiment", "x"], NVX_LOG_DIR="/logs")
    assert result.returncode != 0


def test_run_tmux_detects_expect_sha_equals_form() -> None:
    result = _run(
        "run_tmux.sh",
        ["job", "--", "--expect-sha=abc"],
        NVX_LOG_DIR="/logs",
        NVX_PRINT_ONLY="1",
        NVX_EXPECT_SHA="abc",
    )
    assert _printed_command(result).count("expect-sha") == 1


def test_sbatch_outside_checkout(tmp_path: Path) -> None:
    result = _run("run.sbatch", [], **SBATCH_ENV, SLURM_SUBMIT_DIR=str(tmp_path))
    assert result.returncode == 2
    assert "submit from inside the repo checkout" in result.stderr


@pytest.mark.parametrize(
    "script, env, args",
    [
        ("run_tmux.sh", {"NVX_LOG_DIR": "/logs"}, ["job", "--", "--experiment", "x"]),
        ("run.sbatch", SBATCH_ENV, []),
        ("nnunet.sh", NNUNET_ENV, []),
    ],
)
def test_expect_sha_mismatch_exits_3(script: str, env: dict, args: list) -> None:
    # Print-only is OFF here: the SHA check runs before anything is launched.
    env = {**env, "NVX_PRINT_ONLY": "0", "NVX_EXPECT_SHA": "0" * 40}
    result = _run(script, args, **env)
    assert result.returncode == 3
    assert "NVX_EXPECT_SHA" in result.stderr


def test_nnunet_wall_limit_guard() -> None:
    one_hour_left = str(int(time.time()) + 3600)
    bad = _run(
        "nnunet.sh",
        [],
        **{**NNUNET_ENV, "NVX_BUDGET_HOURS": "0.8"},
        SLURM_JOB_END_TIME=one_hour_left,
    )
    assert bad.returncode == 3
    assert "NVX_BUDGET_HOURS" in bad.stderr
    ok = _run(
        "nnunet.sh",
        [],
        **{**NNUNET_ENV, "NVX_BUDGET_HOURS": "0.4"},
        SLURM_JOB_END_TIME=one_hour_left,
    )
    assert "--budget-hours 0.4" in _printed_command(ok)


def test_nnunet_warns_when_wall_limit_unknown() -> None:
    result = _run("nnunet.sh", [], **NNUNET_ENV)
    assert result.returncode == 0
    assert "could not read the SLURM wall limit" in result.stderr


# (squeue %L output, hours that fit, hours that do not)
@pytest.mark.parametrize(
    "left, fits, too_long",
    [
        ("1-02:03:04", "20", "26"),  # 26.05 h left
        ("02:03:04", "1.5", "1.6"),  # 2.05 h left
        ("03:04", None, "0.1"),  # 0.05 h left: nothing fits
    ],
)
def test_squeue_parsing(tmp_path: Path, left: str, fits: str | None, too_long: str) -> None:
    _stub(tmp_path, "squeue", f'echo "{left}"\n')
    env = {"PATH": _with_path(tmp_path), "SLURM_JOB_ID": "42"}
    bad = _run("run.sbatch", [], **{**SBATCH_ENV, "NVX_MAX_HOURS": too_long}, **env)
    assert bad.returncode == 3
    if fits is not None:
        ok = _run("run.sbatch", [], **{**SBATCH_ENV, "NVX_MAX_HOURS": fits}, **env)
        assert f"training.max_hours={fits}" in _printed_command(ok)


def test_squeue_unlimited_warns(tmp_path: Path) -> None:
    _stub(tmp_path, "squeue", "echo UNLIMITED\n")
    result = _run("run.sbatch", [], **SBATCH_ENV, PATH=_with_path(tmp_path), SLURM_JOB_ID="42")
    assert result.returncode == 0
    assert "could not read the SLURM wall limit" in result.stderr


def test_run_tmux_existing_session_refused(tmp_path: Path) -> None:
    _stub(tmp_path, "tmux", "exit 0\n")  # has-session succeeds => name is taken
    result = _run(
        "run_tmux.sh",
        ["job", "--", "--experiment", "x"],
        NVX_LOG_DIR=str(tmp_path / "logs"),
        PATH=_with_path(tmp_path),
    )
    assert result.returncode == 5
    assert "already exists" in result.stderr


def test_run_tmux_print_only_lists_env_names_not_values(tmp_path: Path) -> None:
    result = _run(
        "run_tmux.sh",
        ["job", "--", "--experiment", "x"],
        NVX_LOG_DIR=str(tmp_path / "logs"),
        NVX_PRINT_ONLY="1",
        WANDB_API_KEY="s3cret-value",
    )
    assert result.returncode == 0
    assert "NVX_ENVFILE: " in result.stdout and "/.job.env" in result.stdout
    vars_line = next(ln for ln in result.stdout.splitlines() if ln.startswith("NVX_ENVFILE_VARS"))
    assert "WANDB_API_KEY" in vars_line and "PYTHONPATH" in vars_line
    assert "s3cret-value" not in result.stdout
    assert not (tmp_path / "logs").exists()


def test_run_tmux_full_path_with_stub_tmux(tmp_path: Path) -> None:
    """Stub tmux runs the inner command in a CLEAN environment, like an old tmux server."""
    stubs, logs = tmp_path / "stubs", tmp_path / "logs"
    stubs.mkdir()
    _stub(
        stubs,
        "tmux",
        'printf "%s\\n" "$@" >> "$NVX_LOG_DIR/tmux_argv"\n'
        'case "$1" in\n'
        "  has-session) exit 1 ;;\n"
        '  new-session) ls -l "$NVX_LOG_DIR"/.job.env > "$NVX_LOG_DIR/envfile_ls"\n'
        '    env -i PATH="$PATH" bash -c "${@: -1}" || true ;;\n'
        "esac\n",
    )
    _stub(
        stubs,
        "fakepython",
        'for a in "$@"; do echo "ARG:$a"; done\necho "KEY:$WANDB_API_KEY"\n'
        'echo "EXP:$NVX_EXPERIMENT"\nexit 7\n',
    )
    logs.mkdir()
    result = _run(
        "run_tmux.sh",
        ["job", "--", "--override", "x=two words"],
        NVX_LOG_DIR=str(logs),
        PATH=_with_path(stubs),
        PYTHON="fakepython",
        WANDB_API_KEY="s3cret-value",
        NVX_EXPERIMENT="neurovision",
    )
    assert result.returncode == 0, result.stderr
    log_text = next(logs.glob("job_*.log")).read_text()
    assert "NVX_EXIT_CODE: 7" in log_text  # python's code survives the tee pipe
    assert "ARG:x=two words" in log_text  # spaced argument intact
    assert "KEY:s3cret-value" in log_text  # env file delivered the secret
    assert "EXP:neurovision" in log_text
    assert "s3cret-value" not in (logs / "tmux_argv").read_text()  # never in tmux argv
    assert (logs / "envfile_ls").read_text().startswith("-rw-------")  # mode 600
    assert not (logs / ".job.env").exists()  # deleted before python ran
