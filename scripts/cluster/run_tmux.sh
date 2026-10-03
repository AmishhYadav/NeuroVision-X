#!/usr/bin/env bash
# Run scripts/gpu_session.py inside a detached tmux session on an SSH box.
#
# Usage:
#   export NVX_LOG_DIR=/some/persistent/dir        # required: where logs go
#   scripts/cluster/run_tmux.sh <session-name> -- <gpu_session.py args...>
#
# Example:
#   scripts/cluster/run_tmux.sh nvx-main -- --experiment neurovision \
#       --data-root "$DATA" --require-cuda --wandb-mode online \
#       --override training.max_hours=10
#
# Why each guard exists:
#   * NVX_LOG_DIR is required: a log that lands in a temp dir is a log you lose
#     (the capacity-control run's GPU hours are "approximate" because its log
#     was never kept).
#   * Refuses a session name that already exists: starting a second job on top
#     of a running one would fight over the same checkpoint directory.
#   * tmux keeps the job alive if your SSH connection drops. Training itself
#     resumes from last.pt, so a killed box is recoverable, but a dropped
#     laptop connection should not even cost you an epoch.
#   * NVX_EXPECT_SHA (optional): if set, `git rev-parse HEAD` must equal it,
#     and --expect-sha is added to the python command. Checking a branch name
#     proves nothing about which commit is really checked out.
#
#   * A tmux server that is already running starts new sessions in ITS OWN
#     environment, not yours, so your venv, WANDB_API_KEY and NVX_* variables
#     would be lost. So: (1) the interpreter is resolved to an absolute path,
#     and (2) WANDB_API_KEY (if set), PYTHONPATH and every set NVX_* variable
#     are written to an env file $NVX_LOG_DIR/.<name>.env created with mode 600
#     (umask 077). The tmux command sources it and deletes it BEFORE python
#     starts. The secret is deliberately never put in the tmux command line,
#     where `ps` would show it to other users.
#
# This script does NOT clone. The caller clones the repo and then runs
# `git checkout <sha>` as a separate step (never `git clone --depth 1`).
#
# Debugging: NVX_PRINT_ONLY=1 prints the command, the env file path and the
# NAMES (never values) of the variables it would carry, then exits without
# touching tmux or the filesystem.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
# Absolute path, because a pre-existing tmux server will not have our venv on PATH.
PYTHON_REQUESTED="$PYTHON"
PYTHON="$(command -v "$PYTHON_REQUESTED")" || {
    echo "error: interpreter '$PYTHON_REQUESTED' not found (set PYTHON=...)." >&2
    exit 2
}

usage() {
    echo "usage: NVX_LOG_DIR=<dir> $0 <session-name> -- <gpu_session.py args...>" >&2
}

if [ -z "${NVX_LOG_DIR:-}" ]; then
    echo "error: NVX_LOG_DIR is not set. Export it to a persistent directory for logs." >&2
    usage
    exit 2
fi
if [ "$#" -lt 3 ] || [ "$2" != "--" ] || [ -z "$1" ]; then
    echo "error: expected '<session-name> -- <gpu_session.py args...>' (at least one arg)." >&2
    usage
    exit 2
fi

NAME="$1"
shift 2
# tmux treats '.' and ':' in names as separators, so keep names plain.
case "$NAME" in
    "" | *[!A-Za-z0-9_-]*)
        echo "error: session name '$NAME' may only contain letters, digits, '_' and '-'." >&2
        exit 2
        ;;
esac

ARGS=("$@")

# Optional SHA pin: verify HEAD ourselves, and pass it on unless already given.
if [ -n "${NVX_EXPECT_SHA:-}" ]; then
    have_sha_flag=0
    for a in "${ARGS[@]}"; do
        case "$a" in --expect-sha | --expect-sha=*) have_sha_flag=1 ;; esac
    done
    if [ "$have_sha_flag" -eq 0 ]; then
        ARGS+=(--expect-sha "$NVX_EXPECT_SHA")
    fi
    if [ "${NVX_PRINT_ONLY:-0}" != "1" ]; then
        head_sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
        if [ "$head_sha" != "$NVX_EXPECT_SHA" ]; then
            echo "error: HEAD is $head_sha but NVX_EXPECT_SHA is $NVX_EXPECT_SHA." >&2
            echo "Clone, then 'git checkout <sha>' as a separate step." >&2
            exit 3
        fi
    fi
fi

CMD=("$PYTHON" "$REPO_ROOT/scripts/gpu_session.py" "${ARGS[@]}")
LOG="$NVX_LOG_DIR/${NAME}_$(date +%Y%m%d_%H%M%S).log"
ENVFILE="$NVX_LOG_DIR/.${NAME}.env"

# Variables the job needs from THIS shell: PYTHONPATH (so neurovision imports),
# the W&B secret if present, and every NVX_* variable.
ENV_NAMES=(PYTHONPATH)
if [ -n "${WANDB_API_KEY:-}" ]; then ENV_NAMES+=(WANDB_API_KEY); fi
for v in $(compgen -v NVX_ || true); do ENV_NAMES+=("$v"); done
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "${NVX_PRINT_ONLY:-0}" = "1" ]; then
    printf 'NVX_CMD: '
    printf '%q ' "${CMD[@]}"
    printf '\nNVX_LOG: %s\nNVX_ENVFILE: %s\nNVX_ENVFILE_VARS: %s\n' "$LOG" "$ENVFILE" "${ENV_NAMES[*]}"
    exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "error: tmux is not installed on this machine." >&2
    exit 4
fi
# '=' makes tmux match the name exactly instead of as a prefix.
if tmux has-session -t "=$NAME" 2>/dev/null; then
    echo "error: tmux session '$NAME' already exists. Pick another name or kill it." >&2
    exit 5
fi

mkdir -p "$NVX_LOG_DIR"
# Env file: mode 600 from the first byte (umask 077), values shell-quoted.
(
    umask 077
    : >"$ENVFILE"
    for v in "${ENV_NAMES[@]}"; do
        printf 'export %s=%q\n' "$v" "${!v}" >>"$ENVFILE"
    done
)
CMD_STR="$(printf '%q ' "${CMD[@]}")"
ROOT_Q="$(printf '%q' "$REPO_ROOT")"
LOG_Q="$(printf '%q' "$LOG")"
ENV_Q="$(printf '%q' "$ENVFILE")"
# Order: load + delete the env file first (so it never lingers, even if cd
# fails), then cd (a failure still writes NVX_EXIT_CODE), then run with
# pipefail so the exit code is python's, not tee's. The code goes into the log
# because the tmux window disappears when the job ends.
INNER="set -o pipefail; . $ENV_Q; rm -f $ENV_Q; "
INNER="${INNER}cd $ROOT_Q || { echo \"NVX_EXIT_CODE: 1\" | tee -a $LOG_Q; exit 1; }; "
INNER="${INNER}$CMD_STR 2>&1 | tee -a $LOG_Q; rc=\${PIPESTATUS[0]}; echo \"NVX_EXIT_CODE: \$rc\" | tee -a $LOG_Q; exit \$rc"
tmux new-session -d -s "$NAME" "bash -c $(printf '%q' "$INNER")"

echo "started tmux session '$NAME'"
echo "  attach:   tmux attach -t $NAME     (detach again with Ctrl-b d)"
echo "  tail log: tail -f $LOG"
echo "  success shows a line 'NVX_HEALTH: OK' in the log; 'NVX_EXIT_CODE: 0' ends it."
