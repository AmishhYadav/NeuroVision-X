#!/usr/bin/env bash
#
# reproduce.sh — the exact command sequence from raw BraTS to every number in
# the paper.
#
# This is the executable half of docs/reproducibility.md. That document says
# WHICH versions, seeds, hardware and runtimes; this script says WHAT to run,
# in what order, and refuses to let a step run before the step it depends on
# has produced its output.
#
# Two kinds of step live here, and the difference is not cosmetic:
#
#   LOCAL   — runs here, on CPU, when you invoke it. preprocess, splits, tests,
#             packaging, figures.
#   MANUAL  — cannot run here. Training and evaluation happen in a Kaggle
#             notebook session on a T4; there is no headless CLI for a Kaggle
#             GPU session that this script could shell out to. These steps
#             PRINT the exact cell-1 contents to paste and then stop, rather
#             than pretending to have done something.
#
# Every path comes from an environment variable with a repo-relative default
# (project hard constraint 2: no hardcoded paths). Override any of them:
#
#   BRATS_RAW=/data/brats ./scripts/reproduce.sh preprocess
#
# Usage:
#   ./scripts/reproduce.sh                 # list the steps and their state
#   ./scripts/reproduce.sh <step> [...]    # run one or more steps, in order
#   ./scripts/reproduce.sh all             # every LOCAL step, stopping at train
#   ./scripts/reproduce.sh --help
#
# Steps, in dependency order:
#   env         install the Python environment                        LOCAL
#   fetch       download and untar raw BraTS 2021                     LOCAL
#   preprocess  raw NIfTI  ->  .npy cache + metadata.csv              LOCAL
#   splits      freeze train/val/test into configs/data/splits.yaml   LOCAL
#   verify      pytest + smoke test + ruff — the pre-Kaggle gate      LOCAL
#   package     assemble the Kaggle upload folder                     LOCAL
#   upload      push that folder to Kaggle as a dataset               LOCAL
#   train       one training run                                      MANUAL
#   evaluate    score a checkpoint on the frozen test split           MANUAL
#   pull        bring Kaggle results back into outputs/               LOCAL
#   atlas       download SRI24 and run the Phase 0 alignment gate     LOCAL
#   pipeline    localisation -> burden -> report, for one segmentation LOCAL
#   phase5      report agreement + population anatomy                 LOCAL
#   figures     regenerate every paper figure and table               LOCAL
#
# The last three are the interpretable pipeline. They consume saved
# predictions, run entirely on CPU, and cost zero GPU hours.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths. Derived from THIS FILE's location, never from the caller's cwd, so the
# script behaves identically whether invoked as ./scripts/reproduce.sh or from
# somewhere else entirely.
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"

BRATS_RAW="${BRATS_RAW:-data/raw/BraTS2021_Training_Data}"
PREP_DIR="${PREP_DIR:-data/preprocessed/brats}"
SPLITS="${SPLITS:-configs/data/splits.yaml}"
UPLOAD_DIR="${UPLOAD_DIR:-outputs/kaggle_upload}"

# Two different accounts. GitHub is AmishhYadav, Kaggle is amishyadav123 —
# the slug and the repo URL do not share a username.
KAGGLE_USER="${KAGGLE_USER:-amishyadav123}"
DATA_SLUG="${DATA_SLUG:-${KAGGLE_USER}/neurovision-brats-prep}"
RAW_SLUG="${RAW_SLUG:-dschettler8845/brats-2021-task1}"
REPO_URL="${REPO_URL:-https://github.com/AmishhYadav/NeuroVision-X.git}"

# The run to reproduce. Override on the command line for the other rows:
#   EXPERIMENT=baseline_swinunetr ./scripts/reproduce.sh train
EXPERIMENT="${EXPERIMENT:-baseline_unet3d}"
SPLIT="${SPLIT:-test}"

# The interpretable pipeline. PIPELINE_SOURCE is "label" (ground truth) or
# "prediction"; with "prediction" you must also give PIPELINE_EVAL_DIR. The
# burden and localisation runs for one segmentation MUST agree on both, or
# scripts/report.py refuses to join them -- see its module docstring.
ATLAS_DIR="${ATLAS_DIR:-data/atlas}"
PIPELINE_SOURCE="${PIPELINE_SOURCE:-label}"
PIPELINE_EVAL_DIR="${PIPELINE_EVAL_DIR:-}"
PIPELINE_TAG="${PIPELINE_TAG:-gt}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step_banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

require_file() { [ -e "$1" ] || die "$2"; }

# ---------------------------------------------------------------------------
# 0. env — the Python environment
# ---------------------------------------------------------------------------
# requirements.txt is a DEV lockfile for Python 3.11 (pyproject pins
# >=3.11,<3.12). Kaggle runs 3.12 with its own ABI-linked scientific stack and
# installs a SUBSET of this file — see the `# kaggle-exclude:` line in
# requirements.txt, which the Kaggle notebooks parse at run time. Do not
# install this file wholesale on Kaggle.
step_env() {
  step_banner "env (LOCAL)"
  if [ ! -x "$PY" ]; then
    bold "Creating .venv on Python 3.11"
    echo "  uv venv --python 3.11 .venv    # or: python3.11 -m venv .venv"
    echo "  .venv/bin/pip install -r requirements.txt"
    echo "  .venv/bin/pip install -e ."
    die "No interpreter at $PY. Run the three commands above, then re-run this step."
  fi
  "$PY" -m pip install -q -r requirements.txt
  "$PY" -m pip install -q -e .
  "$PY" - <<'EOF'
import importlib.metadata as md
import platform
import sys

print(f"python {sys.version.split()[0]}  {platform.platform()}")
for pkg in ("torch", "monai", "numpy", "hydra-core", "wandb"):
    print(f"  {pkg} {md.version(pkg)}")
EOF
  echo "Compare against the pinned table in docs/reproducibility.md."
}

# ---------------------------------------------------------------------------
# 1. fetch — raw BraTS 2021
# ---------------------------------------------------------------------------
# BraTS 2021 is mirrored on Kaggle as a single 13.4 GB tar, so no Synapse
# registration is needed. It uses the 2020-style _t1/_t1ce/_t2/_flair/_seg
# suffixes that scan_brats_root already handles.
step_fetch() {
  step_banner "fetch (LOCAL, ~13.4 GB download)"
  if [ -d "$BRATS_RAW" ]; then
    echo "Raw tree already present at $BRATS_RAW — skipping download."
    return 0
  fi
  command -v kaggle >/dev/null || die "kaggle CLI not found. pip install kaggle, then place kaggle.json (see docs/kaggle_workflow.md §1)."
  mkdir -p "$(dirname "$BRATS_RAW")"
  kaggle datasets download -d "$RAW_SLUG" -p "$(dirname "$BRATS_RAW")" --unzip
  tar -xf "$(dirname "$BRATS_RAW")/BraTS2021_Training_Data.tar" -C "$(dirname "$BRATS_RAW")"
  echo "Raw tree at $BRATS_RAW"
}

# ---------------------------------------------------------------------------
# 2. preprocess — the only step that touches raw NIfTI
# ---------------------------------------------------------------------------
# Per case: nonzero z-score per modality, crop to the nonzero bbox of the RAW
# image (not the normalized one), remap labels {0,1,2,4} -> {0,1,2,3}, write
# float16 image.npy + uint8 label.npy + meta.json. Training never reads a NIfTI
# again. Resumable: a case whose output exists is skipped.
#
# meta.json's bbox + original_shape are load-bearing — evaluate.py uncrops
# predictions back into original BraTS geometry with them.
step_preprocess() {
  step_banner "preprocess (LOCAL, ~5 min at num_workers=8 on an M4)"
  require_file "$BRATS_RAW" "Raw BraTS tree not found at $BRATS_RAW. Run: ./scripts/reproduce.sh fetch"
  "$PY" scripts/preprocess.py \
      "data.root_dir=$BRATS_RAW" \
      "data.preprocessing.out_dir=$PREP_DIR"
  echo "Expect: 1251/1251 cases, 0 failed, ~34 GB, metadata.csv written."
}

# ---------------------------------------------------------------------------
# 3. splits — frozen once, never reshuffled
# ---------------------------------------------------------------------------
# Seed 42, fractions 70/15/15, computed over the cases that ACTUALLY exist on
# disk (sorted first, so filesystem iteration order cannot leak in). The
# expected result is 875/187/189 over 1251 cases.
#
# make_splits raises rather than overwriting: regenerating changes which cases
# are in val/test and silently invalidates every number already measured. If
# you genuinely mean it, add data.splits.overwrite=true and understand that
# every result in docs/experiments.md must then be recomputed.
step_splits() {
  step_banner "splits (LOCAL, seconds)"
  require_file "$PREP_DIR" "No preprocessed cache at $PREP_DIR. Run: ./scripts/reproduce.sh preprocess"
  if [ -f "$SPLITS" ]; then
    echo "$SPLITS already exists — frozen, leaving it alone."
    "$PY" - "$SPLITS" <<'EOF'
import sys

import yaml

s = yaml.safe_load(open(sys.argv[1]))
sizes = {k: len(v) for k, v in s.items() if isinstance(v, list)}
print("  ", sizes, s.get("meta"))
EOF
    return 0
  fi
  "$PY" scripts/make_splits.py \
      "data.root_dir=$BRATS_RAW" \
      "data.preprocessing.out_dir=$PREP_DIR" \
      "data.splits.path=$SPLITS"
}

# ---------------------------------------------------------------------------
# 4. verify — the pre-Kaggle gate
# ---------------------------------------------------------------------------
# Run this before every Kaggle session. A rationed GPU hour spent discovering a
# wiring bug is the failure this step exists to prevent.
#
# Note what it does NOT catch: three CUDA-only faults have shipped past a green
# CPU suite in this project (a metrics device mismatch, CuPy HD95, RNG restore
# on resume). The Mac is a correctness harness for logic, not for device
# placement.
step_verify() {
  step_banner "verify (LOCAL, ~15 s total)"
  bold "pytest — expect 819 passed in ~10 s"
  "$PY" -m pytest
  bold "smoke test — real pipeline on 2 synthetic cases, expect SMOKE TEST PASSED in ~4 s"
  "$PY" scripts/smoke_test.py
  bold "ruff"
  "$PY" -m ruff check src scripts tests
}

# ---------------------------------------------------------------------------
# 5. package — build the Kaggle upload folder
# ---------------------------------------------------------------------------
# Hardlinks by default (the cache is 34 GB; copying it doubles disk use for
# nothing when both paths are on one volume) and excludes macOS/Python junk,
# which matters because packaging runs on a Mac and training does not.
#
# --dry-run first, always: it validates that every case id in the split has a
# directory on disk BEFORE you spend 40 minutes uploading.
step_package() {
  step_banner "package (LOCAL, minutes)"
  require_file "$SPLITS" "No split file at $SPLITS. Run: ./scripts/reproduce.sh splits"
  "$PY" scripts/package_for_kaggle.py \
      --prep-dir "$PREP_DIR" --splits "$SPLITS" \
      --out "$UPLOAD_DIR" --slug "$DATA_SLUG" --dry-run
  bold "Dry run above. Re-running for real:"
  "$PY" scripts/package_for_kaggle.py \
      --prep-dir "$PREP_DIR" --splits "$SPLITS" \
      --out "$UPLOAD_DIR" --slug "$DATA_SLUG"
}

# ---------------------------------------------------------------------------
# 6. upload — push the dataset to Kaggle
# ---------------------------------------------------------------------------
# --dir-mode zip matters: the cache is thousands of small .npy files, and
# uploading them individually over the API is far slower than one zip that
# Kaggle unpacks server-side. Kaggle's per-dataset limit is 200 GB; the ~20 GB
# figure that constrains keep_last_n is the /kaggle/working OUTPUT quota, which
# is a different limit entirely.
step_upload() {
  step_banner "upload (LOCAL, ~1 h for 11.3 GB zipped)"
  require_file "$UPLOAD_DIR" "No upload folder at $UPLOAD_DIR. Run: ./scripts/reproduce.sh package"
  command -v kaggle >/dev/null || die "kaggle CLI not found — see docs/kaggle_workflow.md §1."
  echo "First upload:"
  echo "  kaggle datasets create  -p $UPLOAD_DIR --dir-mode zip"
  echo "Later refresh (after re-preprocessing):"
  echo "  kaggle datasets version -p $UPLOAD_DIR -m 'regenerated with N cases' --dir-mode zip"
  echo
  read -r -p "Run 'kaggle datasets create' now? [y/N] " reply
  case "$reply" in
    [yY]) kaggle datasets create -p "$UPLOAD_DIR" --dir-mode zip ;;
    *) echo "Skipped. Run one of the two commands above by hand." ;;
  esac
}

# ---------------------------------------------------------------------------
# 7. train — MANUAL, Kaggle T4
# ---------------------------------------------------------------------------
step_train() {
  step_banner "train (MANUAL — Kaggle notebook, T4)"
  cat <<EOF
Open notebooks/kaggle_train.ipynb on Kaggle. Cell 1 is the ONLY cell you edit.

  REPO_URL   = "$REPO_URL"
  GIT_REF    = "main"          # pin to a COMMIT SHA for a reproducible run --
                               # see the "known gaps" section of
                               # docs/reproducibility.md, the run of record did
                               # not record which commit it trained
  DATA_SLUG  = "$DATA_SLUG"
  CKPT_SLUG  = None            # None = fresh; else the previous session's
                               # notebook output, to resume
  EXPERIMENT = "$EXPERIMENT"
  USE_WANDB  = True
  OVERRIDES  = ["model=unet3d", "data.dataset_type=dataset", "data.num_workers=2"]

kernel-metadata.json must request machine_shape: NvidiaTeslaT4. The P100 is
sm_60 and the stock image's torch targets sm_70+, so a P100 session reports
CUDA available and then fails every kernel launch. Cell 5 executes a real matmul
to catch exactly that.

The notebook composes the real Hydra config and calls scripts/train.py's
run_training in-process. The equivalent CLI, if you ever train somewhere with a
shell, is:

  python scripts/train.py +experiment=$EXPERIMENT \\
      data.root_dir=<mounted-dataset> \\
      data.preprocessing.out_dir=<mounted-dataset>/preprocessed \\
      data.splits.path=<mounted-dataset>/splits.yaml \\
      training.checkpoint.dir=/kaggle/working/checkpoints \\
      data.num_workers=2

RESUME IS THE SAME COMMAND. scripts/train.py auto-finds last.pt in the
checkpoint dir. Attach the previous session's output as CKPT_SLUG, re-run, and
check the log's first line: it says "FRESH:" or "RESUME: ... from epoch N".
max_hours: 11.0 makes the trainer stop cleanly under Kaggle's 12 h kill.

Repeat until epochs_done == epochs_planned, then record the row in
docs/experiments.md (GPU hours SUMMED across every session — one run is one
row, however many sessions it took).
EOF
}

# ---------------------------------------------------------------------------
# 8. evaluate — MANUAL, Kaggle T4
# ---------------------------------------------------------------------------
# Every reported number comes from here, on the TEST split, at
# inference.sliding_window.overlap 0.5. The val/dice_mean in W&B is a
# monitoring signal at overlap 0.25 and is NOT comparable — it must never be
# pasted into a results table.
step_evaluate() {
  step_banner "evaluate (MANUAL — Kaggle notebook, ~15 min for 189 cases)"
  cat <<EOF
Open notebooks/kaggle_evaluate.ipynb. Attach two data sources: the preprocessed
dataset, and a dataset containing best.pt. Cell 1:

  REPO_URL   = "$REPO_URL"
  GIT_REF    = "main"
  SPLIT      = "$SPLIT"
  EXPERIMENT = "$EXPERIMENT"
  CKPT_NAME  = "best.pt"
  OVERRIDES  = ["model=unet3d", "data.num_workers=2"]

Equivalent CLI:

  python scripts/evaluate.py \\
      data.root_dir=<mounted-dataset> \\
      data.preprocessing.out_dir=<mounted-dataset>/preprocessed \\
      data.splits.path=<mounted-dataset>/splits.yaml \\
      experiment_name=$EXPERIMENT \\
      inference.evaluation.split=$SPLIT \\
      inference.evaluation.checkpoint=<path-to>/best.pt \\
      inference.evaluation.out_dir=/kaggle/working/eval_$SPLIT \\
      wandb.mode=disabled

Writes per_case_metrics.csv, summary.csv, predictions/<case>.npy (uint8, in
ORIGINAL 240x240x155 geometry) and eval_config.yaml. Metrics are computed in
CROPPED space -- uncropping both sides adds identical background to numerator
and denominator, so Dice and HD95 are numerically identical and only slower.
HD95 is in MILLIMETRES here because evaluate.py passes spacing from meta.json.

Do not turn on inference.postprocess.et_min_volume to lift ET Dice. On BraTS
2021 only 2.6% of cases have no enhancing tumor, so it buys almost nothing and
it launders exactly the overconfidence the calibration claim exists to expose.
If a number was ever produced with it on, say so in docs/experiments.md.
EOF
}

# ---------------------------------------------------------------------------
# 9. pull — bring Kaggle output back
# ---------------------------------------------------------------------------
step_pull() {
  step_banner "pull (LOCAL)"
  cat <<EOF
Download the evaluation notebook's output (Kaggle notebook page -> Output ->
Download, or the CLI below) and unpack it so that

  outputs/eval_${SPLIT}/{per_case_metrics.csv,summary.csv,predictions/,eval_config.yaml}

exists. notebooks/09_paper_figures.ipynb reads exactly those paths.

  kaggle kernels output <kaggle-user>/<notebook-slug> -p outputs/

If W&B ran in offline mode, the training curves are inside the training
notebook's output too, and are uploaded afterwards with:

  wandb sync outputs/wandb_offline/offline-run-*
EOF
  for d in outputs/eval_*; do
    [ -d "$d" ] && echo "  present: $d"
  done
  true
}

# ---------------------------------------------------------------------------
# 10. atlas — SRI24 + the Phase 0 alignment gate
# ---------------------------------------------------------------------------
# SRI24 is CC-BY-SA and is downloaded, never committed. The gate is not
# decoration: the atlas is anterior-posterior mirrored relative to BraTS voxel
# indexing, and brain-mask Dice CANNOT see a left-right flip -- it scores
# HIGHER on a mirrored atlas. Laterality is the check that can, and it must
# pass before any localisation number means anything.
step_atlas() {
  step_banner "atlas (LOCAL, CPU)"
  "$PY" scripts/fetch_atlas.py anatomy.dir="$ATLAS_DIR"
  "$PY" scripts/validate_atlas.py anatomy.dir="$ATLAS_DIR"
  echo
  echo "The lobe-distribution check is ADVISORY and never gates: its natural"
  echo "summary statistic prefers the WRONG orientation (rank correlation"
  echo "+0.975 mirrored vs +0.872 correct). Read the brain-mask Dice and the"
  echo "laterality pair count; those are the gate."
}

# ---------------------------------------------------------------------------
# 11. pipeline — localisation, burden and report for ONE segmentation
# ---------------------------------------------------------------------------
# Phases 1, 2, 3a, 3b and 4. Everything here reads saved arrays and writes
# CSVs; no model runs. Set PIPELINE_SOURCE/PIPELINE_EVAL_DIR/PIPELINE_TAG to
# choose which segmentation is profiled, and run it once per segmentation you
# want to compare in phase5.
step_pipeline() {
  step_banner "pipeline (LOCAL, CPU) — source=$PIPELINE_SOURCE tag=$PIPELINE_TAG"
  require_file "$ATLAS_DIR" "Atlas missing. Run: ./scripts/reproduce.sh atlas"

  # Under `set -u`, "${arr[@]}" on an EMPTY array is an unbound-variable error
  # in bash 3.2 -- which is what macOS ships. The `${arr[@]+"${arr[@]}"}` form
  # expands to nothing when the array is empty and is safe on 3.2 and on 5.
  local extra=()
  if [ "$PIPELINE_SOURCE" = "prediction" ]; then
    [ -n "$PIPELINE_EVAL_DIR" ] || die \
      "PIPELINE_SOURCE=prediction needs PIPELINE_EVAL_DIR=<dir with predictions/>"
    require_file "$PIPELINE_EVAL_DIR/predictions" "No predictions/ in $PIPELINE_EVAL_DIR"
    extra=("analysis.localize.eval_dir=$PIPELINE_EVAL_DIR")
  fi

  "$PY" scripts/localize.py \
      analysis.localize.source="$PIPELINE_SOURCE" \
      analysis.localize.split="$SPLIT" \
      ${extra[@]+"${extra[@]}"} \
      anatomy.dir="$ATLAS_DIR" \
      output_dir="outputs/localize_$PIPELINE_TAG"

  local burden_extra=()
  if [ "$PIPELINE_SOURCE" = "prediction" ]; then
    burden_extra=("analysis.burden.eval_dir=$PIPELINE_EVAL_DIR")
  fi
  "$PY" scripts/burden.py \
      analysis.burden.source="$PIPELINE_SOURCE" \
      analysis.burden.split="$SPLIT" \
      ${burden_extra[@]+"${burden_extra[@]}"} \
      output_dir="outputs/burden_$PIPELINE_TAG"

  "$PY" scripts/report.py \
      analysis.report.burden_dir="outputs/burden_$PIPELINE_TAG" \
      analysis.report.localize_dir="outputs/localize_$PIPELINE_TAG" \
      output_dir="outputs/report_$PIPELINE_TAG"

  echo
  echo "Wrote outputs/{localize,burden,report}_$PIPELINE_TAG."
  echo "report.py refuses a burden_dir and localize_dir that disagree on source,"
  echo "split or resolved source directory: those directories differ only by"
  echo "suffix, the join on case_id succeeds either way, and mixing them yields"
  echo "a ground-truth burden profile beside a prediction-derived structure list"
  echo "with nothing failing."
}

# ---------------------------------------------------------------------------
# 12. phase5 — report agreement and population anatomy
# ---------------------------------------------------------------------------
# Needs `pipeline` to have been run for ground truth AND for every model being
# compared. Patch size is a controlled variable: the three 64^3 runs may share
# a Holm family, the superseded 96^3 run may not (it differs in epochs too).
step_phase5() {
  step_banner "phase5 (LOCAL, CPU)"
  require_file "outputs/report_gt/reports" \
      "No ground-truth reports. Run: PIPELINE_TAG=gt ./scripts/reproduce.sh pipeline"

  local pairs=()
  for tag in neurovision baseline capacity_control; do
    if [ -d "outputs/report_$tag/reports" ]; then
      pairs+=("+analysis.report_agreement.pred_dirs.$tag=outputs/report_$tag/reports")
    else
      echo "  [skip] outputs/report_$tag/reports absent — not scored"
    fi
  done
  [ "${#pairs[@]}" -gt 0 ] || die "No model report directories found; run pipeline per model first."

  "$PY" scripts/report_agreement.py \
      analysis.report_agreement.gt_dir=outputs/report_gt/reports \
      ${pairs[@]+"${pairs[@]}"} \
      output_dir=outputs/report_agreement

  local pop=()
  for split_tag in gt gt_train gt_val; do
    if [ -d "outputs/localize_$split_tag" ]; then pop+=("outputs/localize_$split_tag"); fi
  done
  [ "${#pop[@]}" -gt 0 ] || die "No outputs/localize_gt* directories to build a cohort from."
  "$PY" scripts/population_stats.py \
      "+analysis.population.localize_dirs=[$(IFS=,; echo "${pop[*]}")]" \
      output_dir=outputs/population_gt

  echo
  echo "READ THE DEGENERATE-FIELD WARNING. Measured over all 1251 cases,"
  echo "frac_near_eloquent is 1.0: every glioma in this dataset is within 10 mm"
  echo "of a Sawaya-listed structure. That is saturation, not agreement, and it"
  echo "must never be reported as a 100% success rate."
}

# ---------------------------------------------------------------------------
# 13. figures — every paper figure and table, from files, on CPU
# ---------------------------------------------------------------------------
# The notebook recomputes NO metric. It loads what evaluate.py wrote, formats,
# and saves. Every artifact it cannot produce is recorded with a reason and
# printed in §11's audit; §10 lists the figures that have no producer at all
# yet. Edit its cell 1 (the RUNS manifest) to point at the eval directories
# that exist.
#
# nbconvert is NOT in requirements.txt — the stack is fixed and this script is
# not a reason to add a dependency. If it happens to be installed, the notebook
# is executed headlessly; otherwise this step prints what to run by hand.
step_figures() {
  step_banner "figures (LOCAL, CPU)"
  require_file "notebooks/09_paper_figures.ipynb" "Missing notebooks/09_paper_figures.ipynb"
  if ! "$PY" -m jupyter nbconvert --version >/dev/null 2>&1; then
    cat <<EOF
nbconvert is not installed in this environment, and it is deliberately not in
requirements.txt. Run the notebook by hand instead:

  jupyter lab notebooks/09_paper_figures.ipynb     # edit cell 1, Run All

Or install it just for this (it is a dev tool, not part of the pinned stack):

  $PY -m pip install nbconvert ipykernel
  ./scripts/reproduce.sh figures
EOF
    return 0
  fi
  "$PY" -m jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=1800 \
      --output "$REPO_ROOT/outputs/09_paper_figures.executed.ipynb" \
      notebooks/09_paper_figures.ipynb
  echo "Wrote outputs/paper/figures/*.{pdf,png} and outputs/paper/tables/*.{md,tex}"
  echo "READ THE §11 AUDIT in the executed notebook: it names every artifact that"
  echo "was skipped and why. A figure absent from the paper because a file was"
  echo "missing is a bug you want to find there, not in review."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
mark() { if [ -e "$1" ]; then printf '  [done]    %-11s %s\n' "$2" "$1"; else printf '  [missing] %-11s %s\n' "$2" "$1"; fi; }

status() {
  bold "NeuroVision-X — reproduce.sh"
  echo "repo:       $REPO_ROOT"
  echo "python:     $PY"
  echo "experiment: $EXPERIMENT   split: $SPLIT"
  echo
  bold "State of the pipeline"
  mark "$PY"            env
  mark "$BRATS_RAW"     fetch
  mark "$PREP_DIR"      preprocess
  mark "$SPLITS"        splits
  mark "$UPLOAD_DIR"    package
  mark "outputs/eval_${SPLIT}/summary.csv" evaluate
  mark "$ATLAS_DIR"     atlas
  mark "outputs/report_gt/reports" pipeline
  mark "outputs/report_agreement/agreement_summary.csv" phase5
  mark "outputs/paper/tables" figures
  echo
  echo "Run a step:  ./scripts/reproduce.sh <step>"
  echo "Steps:       env fetch preprocess splits verify package upload train evaluate pull"
  echo "             atlas pipeline phase5 figures"
  echo "             all   (every LOCAL step up to and including package)"
  echo "Full detail: docs/reproducibility.md"
}

usage() { sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

main() {
  if [ "$#" -eq 0 ]; then status; exit 0; fi
  for arg in "$@"; do
    case "$arg" in
      -h|--help)  usage ;;
      env)        step_env ;;
      fetch)      step_fetch ;;
      preprocess) step_preprocess ;;
      splits)     step_splits ;;
      verify)     step_verify ;;
      package)    step_package ;;
      upload)     step_upload ;;
      train)      step_train ;;
      evaluate)   step_evaluate ;;
      pull)       step_pull ;;
      atlas)      step_atlas ;;
      pipeline)   step_pipeline ;;
      phase5)     step_phase5 ;;
      figures)    step_figures ;;
      all)
        step_env; step_fetch; step_preprocess; step_splits; step_verify; step_package
        step_banner "STOP — the next step is training, which is MANUAL"
        echo "Run: ./scripts/reproduce.sh upload train"
        ;;
      *) die "Unknown step '$arg'. Run with --help." ;;
    esac
  done
}

main "$@"
