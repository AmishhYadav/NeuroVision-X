#!/usr/bin/env bash
# SLURM batch script: run scripts/gpu_session.py on one GPU.
#
# Submit FROM INSIDE the repo checkout (SLURM copies this script elsewhere, so
# the repo root is taken from the submit directory in that case):
#   cd <repo> && git checkout <sha>
#   sbatch --export=ALL,NVX_EXPERIMENT=neurovision,NVX_DATA_ROOT=/path/to/data,\
# NVX_MAX_HOURS=10,NVX_EXPECT_SHA=<sha>,NVX_WANDB_MODE=online \
#       scripts/cluster/run.sbatch
#
# Environment variables (all run arguments come from here):
#   required  NVX_EXPERIMENT     -> --experiment
#   required  NVX_DATA_ROOT      -> --data-root   (or NVX_SEARCH_ROOT -> --search-root)
#   required  NVX_MAX_HOURS      -> --override training.max_hours=...
#   optional  NVX_CKPT_SRC       -> --ckpt-src
#   optional  NVX_EXPECT_EPOCH   -> --expect-epoch
#   optional  NVX_EXPECT_WANDB_ID-> --expect-wandb-id
#   optional  NVX_EXPECT_SHA     -> checked against `git rev-parse HEAD` AND --expect-sha
#   optional  NVX_WANDB_MODE     -> --wandb-mode (online/offline/disabled)
# Unset optional variables are simply left off the command line.
#
# WHY NVX_MAX_HOURS IS MANDATORY: the trainer has no SIGTERM handler. If SLURM
# kills the job at the wall limit mid-epoch, that epoch is lost and nothing is
# saved on the way out. training.max_hours makes the trainer stop BETWEEN epochs
# on its own. The script refuses to start unless NVX_MAX_HOURS is at least 0.5 h
# below the wall limit (read from $SLURM_JOB_END_TIME, else `squeue`, by
# scripts/cluster/_walllimit.sh; if neither works it warns and continues, so
# check --time yourself). The 0.5 h
# margin covers preflight checks, the last epoch overshooting its estimate, and
# the final checkpoint scan.
#
# RESUME: a preempted or finished-slice job is resumed by RESUBMITTING with
#   NVX_CKPT_SRC=<persistent checkpoint dir that holds last.pt>
#   NVX_EXPECT_EPOCH=<the epoch stored in that last.pt>
# The preflight then refuses a checkpoint from the wrong run or one with NaNs.
# Keep checkpoints on a PERSISTENT filesystem, never node-local /tmp.
#
# This script does NOT clone; clone and `git checkout <sha>` yourself, as
# separate steps (never `git clone --depth 1`).
# NVX_PRINT_ONLY=1 prints the command and exits without running it.
#
# --- SLURM resources. Partition/account are NOT known yet: fill in below. ---
#SBATCH --job-name=nvx-train
#SBATCH --gres=gpu:1
# Log goes to the directory you submit from; %x = job name, %j = job id.
#SBATCH --output=%x_%j.log
# TODO(partition): #SBATCH --partition=<PARTITION>
# TODO(account):   #SBATCH --account=<ACCOUNT>
# TODO(wall):      #SBATCH --time=HH:MM:SS     (then NVX_MAX_HOURS <= this - 0.5 h)
set -euo pipefail

PYTHON="${PYTHON:-python}"

# Under sbatch, BASH_SOURCE points at a spooled copy, so prefer the submit dir.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel 2>/dev/null)" || {
        echo "error: submit from inside the repo checkout (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR is not in a git repo)." >&2
        exit 2
    }
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [ ! -f "$REPO_ROOT/scripts/gpu_session.py" ]; then
    echo "error: $REPO_ROOT does not look like the repo (scripts/gpu_session.py missing)." >&2
    echo "Submit with sbatch from inside the repo checkout." >&2
    exit 2
fi

# --- required variables ---
missing=0
for v in NVX_EXPERIMENT NVX_MAX_HOURS; do
    if [ -z "${!v:-}" ]; then
        echo "error: required environment variable $v is not set." >&2
        missing=1
    fi
done
if [ -z "${NVX_DATA_ROOT:-}" ] && [ -z "${NVX_SEARCH_ROOT:-}" ]; then
    echo "error: set NVX_DATA_ROOT (or NVX_SEARCH_ROOT)." >&2
    missing=1
fi
if [ -n "${NVX_DATA_ROOT:-}" ] && [ -n "${NVX_SEARCH_ROOT:-}" ]; then
    echo "error: set only one of NVX_DATA_ROOT / NVX_SEARCH_ROOT." >&2
    missing=1
fi
if [ "$missing" -ne 0 ]; then exit 2; fi

if ! awk -v v="$NVX_MAX_HOURS" 'BEGIN { exit !(v ~ /^[0-9]+(\.[0-9]+)?$/ && v + 0 > 0) }'; then
    echo "error: NVX_MAX_HOURS='$NVX_MAX_HOURS' must be a positive number (hours)." >&2
    exit 2
fi

# --- wall-limit guard (shared helper; warns if the limit cannot be read) ---
# shellcheck source=scripts/cluster/_walllimit.sh
. "$REPO_ROOT/scripts/cluster/_walllimit.sh"
nvx_check_wall_limit "$NVX_MAX_HOURS" NVX_MAX_HOURS || exit 3

# --- optional SHA pin ---
if [ -n "${NVX_EXPECT_SHA:-}" ] && [ "${NVX_PRINT_ONLY:-0}" != "1" ]; then
    head_sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [ "$head_sha" != "$NVX_EXPECT_SHA" ]; then
        echo "error: HEAD is $head_sha but NVX_EXPECT_SHA is $NVX_EXPECT_SHA." >&2
        exit 3
    fi
fi

# --- build the command (--require-cuda always: this job was given a GPU) ---
CMD=(env "PYTHONPATH=$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$REPO_ROOT/scripts/gpu_session.py"
    --experiment "$NVX_EXPERIMENT" --require-cuda)
if [ -n "${NVX_DATA_ROOT:-}" ]; then CMD+=(--data-root "$NVX_DATA_ROOT"); fi
if [ -n "${NVX_SEARCH_ROOT:-}" ]; then CMD+=(--search-root "$NVX_SEARCH_ROOT"); fi
if [ -n "${NVX_CKPT_SRC:-}" ]; then CMD+=(--ckpt-src "$NVX_CKPT_SRC"); fi
if [ -n "${NVX_EXPECT_EPOCH:-}" ]; then CMD+=(--expect-epoch "$NVX_EXPECT_EPOCH"); fi
if [ -n "${NVX_EXPECT_WANDB_ID:-}" ]; then CMD+=(--expect-wandb-id "$NVX_EXPECT_WANDB_ID"); fi
if [ -n "${NVX_EXPECT_SHA:-}" ]; then CMD+=(--expect-sha "$NVX_EXPECT_SHA"); fi
if [ -n "${NVX_WANDB_MODE:-}" ]; then CMD+=(--wandb-mode "$NVX_WANDB_MODE"); fi
CMD+=(--override "training.max_hours=$NVX_MAX_HOURS")

if [ "${NVX_PRINT_ONLY:-0}" = "1" ]; then
    printf 'NVX_CMD: '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "$REPO_ROOT"
"${CMD[@]}"
