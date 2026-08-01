# Kaggle Workflow

How to get NeuroVision-X training runs onto Kaggle's free GPU tier, keep them
alive across the 12-hour session limit, and resume them correctly. Written
for the project author; assumes no prior Kaggle experience.

## Contents

1. [One-time setup](#1-one-time-setup)
2. [Preparing and uploading the preprocessed dataset](#2-preparing-and-uploading-the-preprocessed-dataset)
3. [The notebook](#3-the-notebook)
4. [Running a long training session](#4-running-a-long-training-session)
5. [Resuming across sessions — the core loop](#5-resuming-across-sessions--the-core-loop)
6. [Troubleshooting](#6-troubleshooting)
7. [Quick reference](#7-quick-reference)

---

## 1. One-time setup

Install the Kaggle CLI:

```bash
pip install kaggle
```

Get an API token: Kaggle account settings → **API** → "Create New Token" (the
exact label may change; look for the button that downloads a `kaggle.json`
credentials file). This downloads `kaggle.json` to your `Downloads` folder.

Place it where the CLI expects it and lock down its permissions — it is a
bearer credential, equivalent to a password:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Verify the CLI is authenticated:

```bash
kaggle datasets list -m
```

If this lists (or empty-lists without an auth error) your own datasets,
credentials are working.

**Phone-verify the account.** Settings → Phone Verification (or wherever
Kaggle currently files this — the intent is "prove you're not a bot account"
so Kaggle unlocks paid resources). Without phone verification you get
**neither GPU accelerators nor internet access** inside notebooks. Since
`scripts/train.py` needs `pip install` for anything not preinstalled in the
Kaggle image, and since a real training run needs a GPU, phone verification
is required before anything else in this document works. A notebook that
needs internet also needs it explicitly turned on in that notebook's
settings panel (see [Section 3](#3-the-notebook)) — phone verification only
makes the toggle available, it doesn't turn it on for you.

---

## 2. Preparing and uploading the preprocessed dataset

`scripts/package_for_kaggle.py` bundles the preprocessed `.npy` cases
(written by `scripts/preprocess.py` to `data.preprocessing.out_dir`), the
`metadata.csv` beside them, and the frozen split file into one upload-shaped
folder — so the Kaggle dataset mirrors exactly what local training reads.

```bash
python scripts/package_for_kaggle.py \
    --prep-dir data/preprocessed/brats \
    --splits configs/data/splits.yaml \
    --out outputs/kaggle_upload \
    --slug <your-username>/neurovision-brats-prep
```

Flags:

| Flag | Purpose |
|---|---|
| `--prep-dir` | Preprocessed cache root (what `data.preprocessing.out_dir` points at). Required. |
| `--splits` | Path to the split YAML. Required. |
| `--out` | Upload folder to build. Required. |
| `--slug` | Kaggle dataset slug, `username/dataset-name`. Writes `dataset-metadata.json`, which `kaggle datasets create` requires. |
| `--title` | Dataset title. Defaults to the slug's last segment. |
| `--limit-gb` | Warn above this size. Defaults to `KAGGLE_DATASET_WARN_GB` (20). |
| `--copy` | Force real copies instead of hardlinks. |
| `--dry-run` | Measure and report; write nothing. |
| `--force` | Overwrite a non-empty `--out`. |

It hardlinks by default (the cache is tens of GB; copying it doubles disk use
and wall time for nothing when both paths are on one volume) and falls back
to copying across filesystems. It excludes `.DS_Store`, `._*`, `*.pyc` and
`__pycache__`, which matters because packaging runs on macOS.

Run it with `--dry-run` first — it validates that **every case id in the split
has a directory on disk** and reports the size, before you spend anything on
an upload. Discovering a missing case after a 40-minute upload and a queued
GPU session is the failure this check exists to prevent.

The upload flow:

**First upload:**

```bash
kaggle datasets create -p <out-folder> --dir-mode zip
```

**Later refreshes**, after re-running preprocessing (e.g. more cases):

```bash
kaggle datasets version -p <out-folder> -m "regenerated with N cases" --dir-mode zip
```

`--dir-mode zip` matters: the preprocessed cache is thousands of small
`.npy` files (one `image.npy` + `label.npy` per case, per the layout
`preprocess.py` writes). Uploading them individually over the Kaggle API is
far slower than zipping the folder once and letting Kaggle unpack it
server-side.

Kaggle datasets are **private by default** when created this way, and that
default should not be changed for this project — BraTS data-use terms do
not permit public redistribution of the imaging data, and a private Kaggle
dataset keeps it visible only to your own account (and anyone you
explicitly add as a collaborator).

---

## 3. The notebook

NeuroVision-X notebooks are **thin driver notebooks**: the notebook attaches
the code and the data, then shells out to `scripts/train.py`. It contains no
training logic of its own — every piece of actual logic (the trainer, the
data pipeline, the model, checkpointing) lives in `src/neurovision/` and
`scripts/`, exactly as it does when run locally. This is what makes "runs
unmodified on Kaggle" (a hard constraint in `CLAUDE.md`) meaningful: if
training logic lived in notebook cells, it would drift from what runs and
gets tested on the Mac. See `notebooks/` in the repository layout for where
these driver notebooks live (currently only the analysis notebook
`01_verify_preprocessing.ipynb` exists there; the Kaggle driver notebook
itself does not exist yet — write it when you set up your first Kaggle
session, and link it here).

**Attach the dataset.** In the notebook editor: "Add Data" (Kaggle's label
for this action may change; the intent is "attach a dataset to this
notebook's `/kaggle/input/`") → search for your private dataset from
[Section 2](#2-preparing-and-uploading-the-preprocessed-dataset) → attach.
It mounts read-only at `/kaggle/input/<dataset-slug>/`.

**A minimal notebook cell:**

```bash
!pip install -q -r requirements.txt --no-deps  # or the torch-excluding variant
!python scripts/train.py \
    data.root_dir=/kaggle/input/<slug> \
    data.preprocessing.out_dir=/kaggle/input/<slug>/preprocessed \
    data.splits.path=/kaggle/input/<slug>/splits.yaml \
    training.checkpoint.dir=/kaggle/working/checkpoints
```

What each override does:

- `data.root_dir` / `data.preprocessing.out_dir` / `data.splits.path` —
  point the data pipeline at the attached, read-only dataset mount instead
  of a local Mac path. `CLAUDE.md`'s "no hardcoded paths" rule is why this
  works at all: every path is a Hydra override, never baked into code.
- `training.checkpoint.dir=/kaggle/working/checkpoints` — this **must**
  point somewhere under `/kaggle/working`. `/kaggle/input` is read-only, so
  `neurovision.training.checkpoint.save_checkpoint` would fail outright if
  told to write there. `/kaggle/working` is also the **only** writable path
  whose contents survive into the notebook version's saved output (capped
  at roughly 20 GB — verify current values, Kaggle changes these:
  https://www.kaggle.com/docs/notebooks, https://www.kaggle.com/docs/datasets).
  A checkpoint written anywhere else is lost the moment the session ends.

Regarding `requirements.txt`: install it, but **not** `torch`/`torchvision`.
The Kaggle GPU image ships a CUDA-matched PyTorch build already; installing
the pinned `torch==2.13.0` from `requirements.txt` would pull a generic wheel
over it and silently lose GPU support. Either edit the install line to skip
those two packages. Derive the Kaggle install list from `requirements.txt`
rather than maintaining a second pinned file — a duplicate will drift from
the original, and the failure mode of that drift is losing the GPU:

```bash
!grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/requirements-kaggle.txt
!pip install -q -r /tmp/requirements-kaggle.txt
```

Confirm it worked before training starts:

```bash
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` means the CUDA build survived. `False` means pip replaced it and the
run would fall back to CPU — except `get_device` raises instead, which is
deliberate (a silent CPU fallback would burn the whole 12-hour session at
CPU speed).

Before any of this: **enable GPU and internet in the notebook's settings
panel** (usually a side panel labelled "Notebook options" or similar — the
label may move, the two toggles you need are for an accelerator/GPU and for
internet access). Both require phone verification (Section 1).

**Run the smoke test first.** Before committing any GPU hours, run
`scripts/smoke_test.py` as the very first cell of the very first session on
a new setup:

```bash
!python scripts/smoke_test.py
```

It builds two tiny synthetic cases, runs the real dataset/model/loss/trainer
pipeline end to end on them for 2 epochs, and asserts checkpoints were
written and are loadable. It exits 0 ("SMOKE TEST PASSED") or 1 ("SMOKE TEST
FAILED") and takes seconds, not hours — it exists specifically to catch a
broken path override or a missing dependency before it costs rationed GPU
time.

---

## 4. Running a long training session

> **An interactive notebook session is not suitable for training.** It dies
> when your browser tab disconnects, and it has a short idle timeout even
> while connected. A multi-hour run started by clicking "Run" cell-by-cell in
> the live editor will not survive you closing the laptop lid.

Long runs must instead use **"Save Version" → "Save & Run All (Commit)"**
(exact wording may shift with Kaggle's UI — look for the option that commits
the notebook and runs it headless, top to bottom, independent of your
browser). This executes every cell in order, unattended, up to the 12-hour
session limit (verify current values — Kaggle changes these:
https://www.kaggle.com/docs/notebooks), and saves whatever is in
`/kaggle/working` at the end (or at the time limit) as that version's output.
This is the mechanism the entire resume workflow in Section 5 depends on:
there is no output to attach next session unless the run was committed this
way.

`configs/training/default.yaml` sets `training.max_hours: 11.0`, one hour
below Kaggle's 12-hour cap. `Trainer.train()` checks, before starting each
new epoch, whether elapsed time plus the running mean epoch duration would
exceed this budget, and if so stops **before** that epoch starts rather than
being killed mid-epoch by Kaggle. This trades a little wall-clock time (the
run stops slightly early) for a guarantee: the last epoch's checkpoint is
always a clean, complete save, never a truncated one from a mid-epoch kill.
Do not raise `max_hours` past a value comfortably under 12 — comfortably
means leaving room for whatever the slowest single epoch has taken so far,
not just the average.

The run's W&B run ID is stored inside the checkpoint payload (`wandb_run_id`
in `save_checkpoint`'s payload, per `checkpoint.py`). When a later session
resumes from that checkpoint, `scripts/train.py`'s `init_wandb` reads it back
out and passes `id=..., resume="allow"` to `wandb.init`, so a resumed session
continues the same W&B run's history instead of starting a new, disconnected
one. You don't need to do anything for this to work — it falls out of
resuming from the right checkpoint file.

---

## 5. Resuming across sessions — the core loop

This is the part that makes a 12-hour session limit survivable for a
multi-day training run. There are two ways to carry a checkpoint from one
committed session into the next.

### Mechanism A (recommended): attach the previous notebook's output as input

After a committed ("Save & Run All") session finishes, its `/kaggle/working`
becomes that specific version's saved output. In your next session, use "Add
Data" again, but this time search your own notebook's past output (Kaggle
labels this something like "Your Work" or "Notebook Output" — the label may
change; the intent is "attach a previous version of one of my own notebooks
as a data source"). It mounts at
`/kaggle/input/<previous-notebook-slug>/`.

Then, in the new session's first cell, copy the checkpoint into the writable
directory before training:

```bash
!mkdir -p /kaggle/working/checkpoints
!cp /kaggle/input/<previous-notebook-slug>/checkpoints/last.pt /kaggle/working/checkpoints/
!python scripts/train.py \
    data.root_dir=/kaggle/input/<dataset-slug> \
    data.preprocessing.out_dir=/kaggle/input/<dataset-slug>/preprocessed \
    data.splits.path=/kaggle/input/<dataset-slug>/splits.yaml \
    training.checkpoint.dir=/kaggle/working/checkpoints
```

**Why the copy step is required, and not optional:** `/kaggle/input/...` is
read-only. `neurovision.training.checkpoint.save_checkpoint` needs to write
new `last.pt`/`best.pt` files as training progresses, and it can only do
that under `/kaggle/working`. If `training.checkpoint.dir` pointed straight
at the read-only input mount, the first save would fail. So the previous
session's `last.pt` has to be copied into the fresh, writable
`/kaggle/working/checkpoints` first — this is the single most common mistake
in this workflow, and it produces no error, just a silent fresh start (see
[Section 6](#6-troubleshooting)).

No command-line flag changes between a fresh run and a resume:
`python scripts/train.py` is the same invocation either way.
`select_resume_checkpoint` (in `scripts/train.py`) calls
`find_resume_checkpoint(cfg.training.checkpoint.dir)`, which looks for
`last.pt` in that directory and, if present, resumes from it automatically;
if absent, training starts fresh from epoch 0.

### Mechanism B: version a checkpoints dataset

Instead of chaining notebook outputs, push checkpoints to a dedicated Kaggle
dataset between sessions:

```bash
kaggle datasets version -p <checkpoints-folder> -m "epoch 42" --dir-mode zip
```

and attach that dataset (as input, read-only) in the next session, copying
`last.pt` out of it the same way as Mechanism A. This is more explicit — the
checkpoint history lives in its own dataset, independent of notebook
version history, and it's easier to point a totally different notebook at
the same checkpoint chain — but it costs a manual download/upload round trip
between sessions instead of Kaggle chaining outputs for you automatically.

### The per-session loop, concretely

1. Attach the previous session's output (Mechanism A) or checkpoints dataset
   (Mechanism B) as an input to the new notebook.
2. Copy `last.pt` from the read-only input mount into
   `/kaggle/working/checkpoints/`.
3. "Save Version" → "Save & Run All (Commit)".
4. Wait for the run to finish or hit `max_hours`.
5. Repeat from step 1 for the next session, until training completes.

### Useful CLI commands

```bash
kaggle kernels push -p <notebook-folder>
kaggle kernels status <username>/<notebook-slug>
kaggle kernels output <username>/<notebook-slug> -p ./kaggle_output
```

`kaggle kernels push` submits a notebook version from local files —
described by `kernel-metadata.json` in that folder, which `kaggle kernels
init -p <notebook-folder>` generates a template for. The fields that matter
for this workflow: `enable_gpu` (must be `true`), `enable_internet` (must be
`true` if `pip install` runs in the notebook), `dataset_sources` (the input
datasets to attach — your preprocessed data, and, per the resume loop above,
the previous checkpoints output or dataset), and `kernel_sources` (attaching
another kernel's output as input, the Mechanism-A path, when doing this via
the CLI instead of the notebook UI).

---

## 6. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Session killed with no final checkpoint | `training.max_hours` set too high relative to the 12 h cap, or `training.checkpoint.dir` points outside `/kaggle/working` (nothing was actually saved to the persisted output). |
| Training runs on CPU despite GPU enabled | `pip install -r requirements.txt` reinstalled `torch`, overwriting Kaggle's CUDA-matched build with a CPU wheel. Exclude `torch`/`torchvision` from the install, per the caveat at the top of `requirements.txt`. |
| `RuntimeError: Device 'cuda' was requested but torch.cuda.is_available() is False` | Either the GPU accelerator isn't enabled in the notebook's settings panel, or the same `torch` reinstall problem above (`get_device` in `neurovision.utils.device` raises rather than silently falling back to CPU — see the "Decisions worth remembering" note in `CLAUDE.md`). |
| Out of disk in `/kaggle/working` | Too many periodic checkpoints retained. Lower `training.checkpoint.keep_last_n` (default 2 in `configs/training/default.yaml`); `last.pt` and `best.pt` are never pruned, but `epoch_NNNN.pt` snapshots are, down to the newest `keep_last_n`. Checkpoints are large — roughly 155 MB for unet3d, roughly 754 MB for SwinUNETR-B — so a handful of stale snapshots can consume a meaningful fraction of the ~20 GB `/kaggle/working` cap. |
| `pip install` fails | Internet is not enabled in the notebook's settings panel, or the account is not phone-verified (Section 1) — without phone verification, the internet toggle isn't available at all. |
| Resume silently starts from epoch 0 | `last.pt` was never copied into the writable `training.checkpoint.dir` — it's still sitting in the read-only `/kaggle/input/...` mount, so `find_resume_checkpoint` finds nothing there and `scripts/train.py` takes the fresh-start branch. Check the very first line of the training log: it prints `FRESH: starting a new training run from epoch 0` or `RESUME: continuing training from epoch N (checkpoint: ...)` — this is the direct way to confirm which branch actually ran, rather than inferring it from Dice curves later. |

---

## 7. Quick reference

```bash
# One-time
pip install kaggle
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list -m

# Upload preprocessed data (see Section 2)
kaggle datasets create -p <out-folder> --dir-mode zip          # first time
kaggle datasets version -p <out-folder> -m "..." --dir-mode zip  # refresh

# Smoke-test the pipeline before spending GPU hours
python scripts/smoke_test.py

# Fresh run (in a notebook cell, after attaching the dataset)
python scripts/train.py \
    data.root_dir=/kaggle/input/<slug> \
    data.preprocessing.out_dir=/kaggle/input/<slug>/preprocessed \
    data.splits.path=/kaggle/input/<slug>/splits.yaml \
    training.checkpoint.dir=/kaggle/working/checkpoints

# Resume (same command; last.pt must be copied into the writable dir first)
mkdir -p /kaggle/working/checkpoints
cp /kaggle/input/<previous-notebook-slug>/checkpoints/last.pt /kaggle/working/checkpoints/
python scripts/train.py ... training.checkpoint.dir=/kaggle/working/checkpoints

# Kernel management
kaggle kernels push -p <notebook-folder>
kaggle kernels status <username>/<notebook-slug>
kaggle kernels output <username>/<notebook-slug> -p ./kaggle_output
```

**Every long run:** commit via "Save Version" → "Save & Run All (Commit)".
Never rely on an interactive session for anything longer than a few minutes.
