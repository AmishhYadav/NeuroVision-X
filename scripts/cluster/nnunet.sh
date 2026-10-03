#!/usr/bin/env bash
# Run one budgeted slice of nnU-Net (Gate A) via scripts/nnunet_session.py on a
# cluster / SSH box. Mirrors the Kaggle notebook's invocation
# (outputs/kaggle_kernels/gatea-s4/gatea_session.ipynb).
#
# Usage (directly, inside tmux, or called from an sbatch job):
#   export NVX_NNUNET_RAW=...           # -> --raw-root
#   export NVX_NNUNET_PREPROCESSED=...  # -> --preprocessed-root
#   export NVX_NNUNET_RESULTS=...       # -> --results-root (keep on PERSISTENT disk)
#   export NVX_BUDGET_HOURS=10.5        # -> --budget-hours (whole-session allowance)
#   export NVX_EXPECT_START_EPOCH=0     # -> --expect-start-epoch (0 = fresh run;
#                                       #    otherwise the epoch the last slice ended at)
#   export NVX_SUMMARY_JSON=...         # -> --summary-json
#   scripts/cluster/nnunet.sh
# Optional:
#   NVX_NNUNET_SEARCH_ROOT        -> --search-root (finds the read-only preprocessed dataset)
#   NVX_NNUNET_PREV_RESULTS_ROOT  -> --prev-results-search-root (restore a prior slice's
#                                    results; not needed if NVX_NNUNET_RESULTS already
#                                    holds them on persistent disk)
#   NVX_EXPECT_SHA                -> verified against `git rev-parse HEAD` (nnunet_session.py
#                                    has no --expect-sha flag, so it is checked here only)
#
# Why the guards: the session stops itself BETWEEN epochs when NVX_BUDGET_HOURS
# is used up, so set it below the scheduler's wall limit. Inside a SLURM job
# this script refuses a budget less than 0.5 h under the wall limit (and warns
# if the limit cannot be read). --expect-start-epoch makes a checkpoint from
# the wrong slice fail fast instead of wasting hours. --require-cuda stops a
# silent CPU run.
#
# This script does NOT clone; clone and `git checkout <sha>` as separate steps
# (never `git clone --depth 1`). nnunetv2 must be installed in the interpreter
# given by $PYTHON (the project keeps it in .venv-analysis).
# NVX_PRINT_ONLY=1 prints the command and exits without running it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"

missing=0
for v in NVX_NNUNET_RAW NVX_NNUNET_PREPROCESSED NVX_NNUNET_RESULTS \
    NVX_BUDGET_HOURS NVX_EXPECT_START_EPOCH NVX_SUMMARY_JSON; do
    if [ -z "${!v:-}" ]; then
        echo "error: required environment variable $v is not set." >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then exit 2; fi

if ! awk -v v="$NVX_BUDGET_HOURS" 'BEGIN { exit !(v ~ /^[0-9]+(\.[0-9]+)?$/ && v + 0 > 0) }'; then
    echo "error: NVX_BUDGET_HOURS='$NVX_BUDGET_HOURS' must be a positive number (hours)." >&2
    exit 2
fi
case "$NVX_EXPECT_START_EPOCH" in
    *[!0-9]*)
        echo "error: NVX_EXPECT_START_EPOCH='$NVX_EXPECT_START_EPOCH' must be a non-negative integer." >&2
        exit 2
        ;;
esac

# Wall-limit guard (shared helper; warns if the limit cannot be read, e.g. when
# not running inside a SLURM job).
# shellcheck source=scripts/cluster/_walllimit.sh
. "$REPO_ROOT/scripts/cluster/_walllimit.sh"
nvx_check_wall_limit "$NVX_BUDGET_HOURS" NVX_BUDGET_HOURS || exit 3

if [ -n "${NVX_EXPECT_SHA:-}" ] && [ "${NVX_PRINT_ONLY:-0}" != "1" ]; then
    head_sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [ "$head_sha" != "$NVX_EXPECT_SHA" ]; then
        echo "error: HEAD is $head_sha but NVX_EXPECT_SHA is $NVX_EXPECT_SHA." >&2
        exit 3
    fi
fi

CMD=(env "PYTHONPATH=$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$REPO_ROOT/scripts/nnunet_session.py"
    --raw-root "$NVX_NNUNET_RAW"
    --preprocessed-root "$NVX_NNUNET_PREPROCESSED"
    --results-root "$NVX_NNUNET_RESULTS"
    --budget-hours "$NVX_BUDGET_HOURS"
    --expect-start-epoch "$NVX_EXPECT_START_EPOCH"
    --device cuda --require-cuda
    --summary-json "$NVX_SUMMARY_JSON")
if [ -n "${NVX_NNUNET_SEARCH_ROOT:-}" ]; then CMD+=(--search-root "$NVX_NNUNET_SEARCH_ROOT"); fi
if [ -n "${NVX_NNUNET_PREV_RESULTS_ROOT:-}" ]; then
    CMD+=(--prev-results-search-root "$NVX_NNUNET_PREV_RESULTS_ROOT")
fi

if [ "${NVX_PRINT_ONLY:-0}" = "1" ]; then
    printf 'NVX_CMD: '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

# The Kaggle notebook also creates the (unused-by-training) raw dir.
mkdir -p "$NVX_NNUNET_RAW"
cd "$REPO_ROOT"
"${CMD[@]}"
