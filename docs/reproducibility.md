# Reproducibility

Everything needed to re-derive a NeuroVision-X number: seeds, package versions,
hardware, and measured runtimes.

`scripts/reproduce.sh` is the executable companion to this document — it holds
the command sequence, this file holds the environment those commands ran in.
Start with `./scripts/reproduce.sh` (no arguments) to see which stages have
already produced output on this machine.

Everything below marked **measured** was read off a real artifact in this repo
(`outputs/wandb_offline/`, `outputs/eval_test/`, file mtimes, a live `pytest`
run) on 2026-08-06. Everything marked *estimate* is arithmetic, and says so.

---

## 1. What is currently reproducible

| Artifact | State |
|---|---|
| Preprocessed BraTS 2021 cache (1251 cases) | **Reproducible.** Deterministic transform, no RNG involved |
| Frozen split 875 / 187 / 189 | **Reproducible.** Seed 42, and the file is checked in |
| `baseline_unet3d` training run | **Re-runnable, not bit-exact.** See §4 and §8 |
| `baseline_unet3d` test metrics (`outputs/eval_test/`) | **Reproducible from the checkpoint.** Inference is deterministic |
| `baseline_swinunetr`, `neurovision`, the 6-row ablation grid | **Not run yet.** No GPU hours spent on them |
| ECE / calibration, MC-dropout uncertainty, fusion gate maps, explainability panels | **No producer wired into a script yet.** `notebooks/09_paper_figures.ipynb` §10 is the authoritative list |

One run of record exists. Everything else in this document describes the
procedure that run followed, so the next run can follow it identically.

---

## 2. Data provenance

| | |
|---|---|
| Dataset | BraTS 2021 Task 1 training set |
| Source | Kaggle `dschettler8845/brats-2021-task1` (13.4 GB, one `BraTS2021_Training_Data.tar`) — no Synapse registration needed |
| Naming | 2020-style `_t1` / `_t1ce` / `_t2` / `_flair` / `_seg` suffixes, matched **exactly**, never by substring (`"_t1" in name` is also true of `_t1ce.nii.gz`) |
| Raw size on disk | 25 GB unpacked — **measured** |
| Cases scanned | 1251 |
| Cases preprocessed | 1251, 0 failed — **measured** (`data/preprocessed/brats/metadata.csv`, 1251 rows) |
| Preprocessed size | 34 GB, ~28 MB/case (float16 image + uint8 label) — **measured** |
| Upload to Kaggle | `amishyadav123/neurovision-brats-prep`, private, 11.3 GB zipped — **measured** from the upload log |

Preprocessing per case: nonzero z-score per modality, crop to the nonzero bbox
of the **raw** image (not the normalized one — a constant-foreground channel has
`std == 0` and would vanish from the union bbox), label remap `{0,1,2,4}` →
`{0,1,2,3}`, then `image.npy` (float16) + `label.npy` (uint8) + `meta.json`.

`meta.json`'s `bbox` and `original_shape` are load-bearing: `evaluate.py`
uncrops predictions back into original 240×240×155 BraTS geometry with them, and
`uncrop_to_original` raises if the two disagree rather than writing a plausible
but spatially shifted volume.

**Label statistics, measured on all 1251 cases** (`metadata.csv`):

| | Count | Fraction |
|---|---|---|
| Cases with zero enhancing tumor (`n_class_3 == 0`) | 33 | 2.6% |
| Cases with zero necrotic core (`n_class_1 == 0`) | 43 | 3.4% |
| Cases with an entirely empty label | 0 | 0% |

This matters because metrics use `ignore_empty=False` (the BraTS convention): a
region absent from the ground truth scores Dice 1.0 if the prediction is also
empty. The widely quoted "~35% of BraTS cases have no ET" does **not** hold for
this training set, so the convention moves headline ET Dice by well under a
point here. It must still be stated in the paper — the choice is invisible
otherwise — but it is not load-bearing. Re-measure if the dataset changes.

---

## 3. Splits

```yaml
# configs/data/splits.yaml — checked in, frozen
train: 875
val:   187
test:  189
meta:  {seed: 42, fractions: [0.7, 0.15, 0.15], n_cases: 1251}
```

The split seed (`data.splits.seed`) is **separate from the run seed** (`cfg.seed`)
and does not change when you sweep seeds. `make_splits` raises rather than
overwriting an existing file; regenerating takes an explicit
`data.splits.overwrite=true` and invalidates every number already measured
against the old split.

Case ids are sorted before shuffling, so the split does not depend on filesystem
iteration order.

---

## 4. Seeds and determinism

**Seed of record: 42.** Set in `configs/config.yaml`, applied by
`neurovision.utils.seed.set_seed(cfg.seed)` at the top of both
`scripts/train.py` and `scripts/evaluate.py`, before any dataset or model is
constructed.

`set_seed` seeds, in order: Python `random`, NumPy's legacy global RNG,
`torch.manual_seed`, `torch.cuda.manual_seed_all` (skipped with no CUDA
device), and MONAI's `set_determinism` — which is the only thing that reaches
the RNG state MONAI's random transforms carry internally.

### What seeding guarantees

Reproducible data order, augmentation choices, weight initialization, and
crop positions.

### What it does not guarantee

**Bit-exactness on a GPU.** Several 3D CUDA convolution and pooling backward
kernels have no deterministic implementation at all. `set_determinism` sets
`cudnn.deterministic = True` and `cudnn.benchmark = False`; this project
restores `benchmark = True`, because disabling autotuning costs throughput on
fixed 96³ patches without buying a guarantee that was never available.
`torch.use_deterministic_algorithms(True)` is deliberately *not* called — it
would raise on those kernels rather than make them deterministic.

Consequence for the paper: report **mean ± std across seeds**, never a single
run's number as if it were exact. Two runs of the same config at seed 42 will
differ. Treat a gap smaller than roughly a Dice point as unresolved until it
has been re-run at 3 seeds.

### Resume preserves RNG state

Checkpoints carry Python, NumPy, torch and CUDA RNG states plus the epoch,
global step and W&B run id, so a run split across Kaggle sessions is one
continuous trajectory rather than a fresh RNG at each restart. Resume is
**epoch-granular**: dataloader position inside an epoch is not recoverable with
`num_workers > 0`, so a kill mid-epoch loses that epoch (bounded by
`save_every_n_epochs: 1`).

`start_epoch = saved_epoch + 1` — the saved epoch already completed, and
resuming *at* it would retrain it and desynchronize the epoch-indexed LR
schedule.

### Known caveat: the augmentation stream depends on `data.num_workers`

`scripts/train.py` uses `torch.utils.data.DataLoader` (not `monai.data.DataLoader`)
with no `worker_init_fn`. Torch reseeds `random`, `torch` and `np.random`
per worker, but MONAI's random transforms draw from their **own** private
`np.random.RandomState`, which those reseeds do not touch — each worker
process inherits an identical copy of it.

So the augmentation applied to a given case is a function of how many workers
there are. This is deterministic and therefore reproducible, but `num_workers`
is a variable, not a performance knob: **keep `data.num_workers=2` when
reproducing the run of record.** (`monai.data.DataLoader` exists precisely to
fix this, via a `worker_init_fn` that reseeds each transform. Switching to it
would change results and so has not been done mid-experiment.)

### Other RNG surfaces

| Surface | Seeding |
|---|---|
| MC-dropout (`inference.mc_dropout.seed`) | `null` — leaves the global torch RNG alone. An int seeds it before the N passes and restores the prior state afterwards, so a per-case loop does not depend on how many cases ran before it |
| Integrated Gradients noise baseline | Takes an explicit `generator` argument |
| Bootstrap CIs in `analysis/statistics.py` | Explicit `rng` argument; the paper's default is stated wherever a CI is reported |

---

## 5. Software versions

Two environments, deliberately different. `requirements.txt` is a **dev
lockfile for Python 3.11**; Kaggle runs Python 3.12 with a curated,
mutually ABI-compatible scientific stack, and installing our pins over it
breaks things that do not look like install failures (our `torch` pin replaces
the CUDA build with a CPU wheel; our `numpy` pin breaks Kaggle's
ABI-linked `scipy`). The Kaggle notebooks therefore install
`requirements.txt` **minus** the packages named on its own
`# kaggle-exclude:` line — one source of truth, no second pinned file to drift.

### Local (macOS dev + all CPU work) — measured

| Package | Version | | Package | Version |
|---|---|---|---|---|
| python | 3.11.15 | | hydra-core | 1.3.4 |
| torch | 2.13.0 | | omegaconf | 2.3.1 |
| monai | 1.6.0 | | wandb | 0.28.1 |
| numpy | 2.4.6 | | einops | 0.8.2 |
| scipy | 1.17.1 | | captum | 0.9.0 |
| pandas | 3.0.5 | | matplotlib | 3.11.1 |
| nibabel | 5.4.2 | | tqdm | 4.70.0 |
| SimpleITK | 2.5.6 | | pytest / ruff / black | 9.1.1 / 0.16.1 / 26.5.1 |
| scikit-image | 0.26.0 | | | |

### Kaggle (the run of record) — measured

Read from `outputs/wandb_offline/offline-run-20260802_064314-nz5y7li7/files/requirements.txt`,
i.e. the environment the training process actually saw.

| Package | Version | Note |
|---|---|---|
| python | 3.12.13 | Kaggle image; `pyproject.toml` pins `<3.12`, so the notebooks use `sys.path` rather than `pip install -e` |
| torch | 2.10.0+cu128 | **Kaggle's build.** Never overwritten — a CPU wheel silently loses the GPU |
| torchvision | 0.25.0+cu128 | Kaggle's |
| numpy | 2.0.2 | Kaggle's. Overwriting it breaks Kaggle's `scipy.ndimage` at import |
| scipy | 1.16.3 | Kaggle's |
| pandas | 2.3.3 | Kaggle's |
| scikit-image | 0.25.2 | Kaggle's |
| matplotlib | 3.10.0 | Kaggle's |
| monai | 1.6.0 | **Installed from our pins** — identical to local |
| nibabel | 5.4.2 | Installed from our pins |
| SimpleITK | 2.5.6 | Installed from our pins |
| hydra-core | 1.3.4 | Installed from our pins |
| einops | 0.8.2 | Installed from our pins. A hard requirement — `SwinUNETR.forward` calls `einops.rearrange`, so a missing einops fails at *run* time, not import time |
| wandb | 0.28.1 | Installed from our pins |

The version skew between the two columns is real and is why the CPU suite
cannot be treated as a proxy for the GPU environment (see §8).

Two dependency traps worth naming, because both were hit and both are dormant
rather than fixed:

- **`scikit-image`** is deprecating `remove_small_objects(min_size=...)` in
  favour of `max_size`, with flipped semantics (the new one removes objects
  smaller than *or equal to*). `inference.postprocess.min_component_size: 50`
  currently drops 49-voxel components and keeps 50-voxel ones; after that
  migration the same number would drop 50 too, shifting every reported HD95.
  Check before bumping the pin.
- **CuPy** — MONAI's `compute_hausdorff_distance` uses a CuPy distance
  transform when its inputs are on CUDA, and on the Kaggle T4 image that JIT
  fails outright (`CompileException: Thrust requires at least C++17`). Both
  `Trainer.validate` and `scripts/evaluate.py` therefore move tensors to CPU
  before scoring. Do not "optimize" that away.

---

## 6. Hardware

### Local — MacBook Pro M4

macOS 26.5.1, arm64. Runs all code, all preprocessing, all tests, all figure
and table generation, and the paper. **Use `device="cpu"`, never MPS** — MPS
support for 3D convolutions is incomplete and fails silently or obscurely. The
Mac is a correctness harness, not a compute device.

### Kaggle — the run of record, measured

Read from `wandb-metadata.json` for both sessions of run `nz5y7li7`:

| | |
|---|---|
| GPU | **Tesla T4** (sm_75), 16 GB. 2 present on the machine; the code uses one |
| CPU | 2 physical / 4 logical |
| RAM | 33.66 GB — **measured**, and notably *not* the ~13 GB this project's notes assumed. `data.num_workers=2` and `dataset_type: dataset` were both chosen under the smaller figure and are therefore conservative, not tuned |
| OS | Linux 6.12.90, glibc 2.35 |
| Session limit | 12 h hard kill; `training.max_hours: 11.0` stops cleanly below it |
| Output quota | ~20 GB on `/kaggle/working` — this is what constrains `keep_last_n: 2`, and is a *different* limit from the 200 GB per-dataset upload ceiling |

**The P100 is not usable.** Measured 2026-08-01 on a real session: the stock
image's PyTorch targets sm_70+ and the P100 is sm_60, giving
`Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the
current PyTorch installation`. Request `machine_shape: NvidiaTeslaT4` in
`kernel-metadata.json`. `torch.cuda.is_available()` returns `True` on that
broken P100 — the only honest check is executing a kernel, which
`notebooks/kaggle_train.ipynb` cell 5 does.

---

## 7. Measured runtimes

### Local (M4, CPU)

| Step | Wall time | How measured |
|---|---|---|
| Preprocess 1251 cases, `num_workers=8` | **~5 min** | Span of `image.npy` mtimes: 17:14:18 → 17:19:15, 2026-08-01 |
| `pytest` (819 tests) | **9.8 s** | Live run, 2026-08-06 |
| `python scripts/smoke_test.py` | **3.7 s** | Live run, 2026-08-06 |
| Package for Kaggle | minutes | Hardlinks, so it does not copy 34 GB |
| Upload 11.3 GB zipped | ~1 h | Upload log; bandwidth-bound |
| Paper figures + tables | CPU-only, minutes | `notebooks/09_paper_figures.ipynb`; `figures.py` never imports torch |

`pytest` from the repo root works with no `PYTHONPATH` (`pythonpath = ["src"]`).
Note `pyproject.toml` already sets `addopts = "-q"`, so an extra `-q` stacks to
`-qq` and silently drops the "N passed" line — run plain `pytest`.

### Kaggle T4 — `baseline_unet3d`, run `nz5y7li7`, measured

One run, two sessions, resumed via `last.pt`:

| | Session 1 | Session 2 | Total |
|---|---|---|---|
| Started (UTC) | 2026-08-01 16:30:31 | 2026-08-02 06:43:14 | |
| Wall clock | 39,357.9 s = **10.93 h** | 19,950.8 s = **5.54 h** | **16.47 GPU-h** |
| Global step at end | 114,625 | 175,000 | |
| Epochs | 1 – 131 | 132 – 200 | **200 / 200** |
| Mean s/epoch (incl. validation) | 300 s | 289 s | ~295 s |
| Last logged `train/epoch_time_seconds` | 239 s | 343 s | |
| Peak VRAM | 1.167 GB | 1.166 GB | |

875 optimizer steps per epoch = 875 train cases × `samples_per_volume: 4` ÷
`batch_size: 1`, flattened by `list_data_collate` into 4 patches of 96³ per
step. `175000 / 875 = 200` exactly, which is what pins the epoch count.

**Peak VRAM of 1.17 GB against a 16 GB card is the headline number for
planning.** The 3D U-Net leaves enormous headroom; SwinUNETR-B (62.19M params,
~754 MB checkpoints) and NeuroVision-X (34.88M) will not. Projected activation
memory for NeuroVision-X at 96³, 4 patches/step is ~4.5–5.5 GB unchecked (an
*estimate*, from saved-tensor byte measurements on CPU — not from a GPU run).

Evaluation: ~1.3 s/case, so a 189-case split is roughly 15 minutes on a T4 —
this figure comes from `notebooks/kaggle_evaluate.ipynb`'s own header, not from
a timing instrument in the code.

### Cost estimates for the runs not yet done (*estimates, not measurements*)

| Run | Est. GPU-h | Basis |
|---|---|---|
| `baseline_swinunetr` | ~30 | ~3× the U-Net per epoch (62M params + gradient checkpointing) |
| `neurovision` | between the two | 34.88M params, plus fusion and two auxiliary heads |
| 6-row ablation grid @ 40 epochs | see `python scripts/run_ablation_grid.py` | That script exists to price the grid; re-run it with a measured `--sec-per-step` once one exists |

Free-tier Kaggle is ~30 GPU-hours per week.

---

## 8. The run of record — and where it diverges from the checked-in config

**Read this before comparing anything to it.**

| | |
|---|---|
| `experiment_name` | `baseline_unet3d` |
| Model | `unet3d` — MONAI `UNet`, 12.87M params, channels `[32,64,128,256,320]`, `num_res_units: 2`, instance norm, dropout 0.1 |
| Seed | 42 |
| W&B | project `neurovision-x`, run id `nz5y7li7`, mode `offline` (synced afterwards). Entity `amishyadav126-svkm-s-narsee-monjee-institute-of-manageme` |
| Evaluated | `scripts/evaluate.py`, **test** split, 189 cases, `best.pt`, sliding-window overlap 0.5, gaussian blending |

The run did **not** use `+experiment=baseline_unet3d`. It composed the root
config plus CLI overrides, and three values differ from what that experiment
file now specifies:

| Key | Run of record | `configs/experiment/baseline_unet3d.yaml` |
|---|---|---|
| `training.epochs` | **200** | 100 |
| `training.val_interval` | **2** | 5 |
| `training.scheduler.warmup_epochs` | **10** | 5 |
| `training.sliding_window.overlap` (validation) | **0.5** | 0.25 |

`_baseline_common.yaml` was written on 2026-08-02, after this run started on
2026-08-01. So `+experiment=baseline_unet3d` today produces a **different,
shorter run** than the one in `outputs/eval_test/`.

Two honest options, and they must not be mixed:

1. Re-run `baseline_unet3d` under the experiment file, at 100 epochs, and
   discard the numbers below. This is the right choice if the baseline table
   is to be internally consistent, since `baseline_swinunetr` and
   `neurovision` will both run 100 epochs from `_baseline_common`.
2. Keep the 200-epoch number and state the schedule difference wherever it
   appears. A 200-epoch baseline against a 100-epoch proposed model is a
   comparison that favours the baseline, so it is at least not
   self-serving — but it is still not a controlled comparison.

Option 1 is the recommendation. The whole point of `_baseline_common.yaml` is
that any Dice/HD95/ECE gap is attributable to the architecture and nothing else.

### Test-split results as they stand — measured, `outputs/eval_test/summary.csv`

| Metric | ET | TC | WT | mean |
|---|---|---|---|---|
| Dice | 0.8587 | 0.9157 | 0.9354 | 0.9033 |
| IoU | 0.7893 | 0.8659 | 0.8837 | 0.8463 |
| HD95 (mm) | 4.018 | 5.209 | 5.356 | 5.009 |

n = 189 for every row except **HD95 ET, where n = 183**: 6 cases are NaN
because exactly one side of the region was empty, which `hd95()` deliberately
does not collapse into a number. Both-empty is overridden to 0.0 (correctly
predicting "no tumor here" is not a boundary error and must not vanish from a
NaN-skipping mean). The old BraTS 373.13 mm penalty is deliberately not used —
it turns HD95 into a high-variance Dice proxy and would undercut the
boundary-accuracy claim.

`gt_empty_ET` on the test split is 0.0265 (5 of 189 cases), so those five score
ET Dice as empty-vs-empty rather than by real overlap.

HD95 is in **millimetres** here because `evaluate.py` passes `spacing` from
`meta.json`. The trainer's validation HD95 is in **voxels** and is a monitoring
signal only — it must never enter a results table. Same for `val/dice_mean`,
which the run of record logged at overlap 0.5 but which `_baseline_common`
computes at 0.25; only `scripts/evaluate.py` output is reportable.

### Known reproducibility gaps in this run

1. **No commit SHA was recorded.** `notebooks/kaggle_train.ipynb` clones
   `GIT_REF = "main"`, and W&B captured no git metadata, so the exact source
   revision that produced `nz5y7li7` is not recoverable — only that it was
   `main` between 2026-08-01 16:30 and 2026-08-02 12:15 UTC (commits `3144175`
   through `9665c70`). **Fix for every future run: set `GIT_REF` to a commit
   SHA, not a branch name.** `scripts/reproduce.sh train` prints this reminder.
2. Neither the training nor the evaluation notebook writes its own wall-clock
   timing; the numbers in §7 were reconstructed from W&B's `_runtime`.
3. The `epochs=200` / `val_interval=2` overrides were typed into the notebook's
   `OVERRIDES` list for that session and are not preserved anywhere in the
   repository — they were recovered for this document from the W&B config
   snapshot. An experiment file, or a checkpoint-config diff printed at
   startup, would have made this unnecessary.

---

## 9. Conventions that change the numbers

These are choices, not defaults, and each one is invisible in a results table
unless stated:

| Convention | Setting | Effect if changed |
|---|---|---|
| Empty-region scoring | `ignore_empty=False` (BraTS) | Region absent from GT scores Dice 1.0 when the prediction is also empty. Affects 2.6% of cases (ET) |
| Discretization | sigmoid + per-channel threshold 0.5. **No argmax anywhere** | The three regions are nested and overlapping; argmax is wrong twice over |
| Component filtering | `min_component_size: 50`, `connectivity: 1`, `keep_largest_only: false` | Speckle removal costs HD95 far more than Dice |
| Nesting | `enforce_nesting: true`, applied **last**, unioning inner into outer | Intersecting instead would delete confident ET voxels because the weaker TC channel missed them |
| ET volume trick | `et_min_volume: 0` — **off** | Turning it on buys headline ET Dice by zeroing small ET predictions. It launders the exact overconfidence the calibration claim exists to expose. If ever used, say so |
| Evaluation geometry | Metrics in **cropped** space, predictions saved **uncropped** | Numerically identical either way; uncropping matters only for the saved artifact being valid BraTS geometry |
| Sliding window (reporting) | roi 96³, overlap **0.5**, gaussian blending | Training-time validation uses a cheaper overlap and is not comparable |

Statistical reporting (`src/neurovision/analysis/statistics.py`): every "A beats
B" claim goes through `compare_models`, which pairs on `case_id`, resamples
**case indices** into the difference array (not the two score arrays
independently), reports a paired bootstrap CI *and* a Wilcoxon signed-rank test,
and applies Holm–Bonferroni across the whole table in one family. The family
must be fixed **before** looking at p-values. `verdict` is conservative:
`inconclusive` if *either* the CI contains 0 or the adjusted p exceeds alpha.

---

## 10. Verifying a fresh checkout

```bash
git clone https://github.com/AmishhYadav/NeuroVision-X.git
cd NeuroVision-X
uv venv --python 3.11 .venv && .venv/bin/pip install -r requirements.txt -e .

./scripts/reproduce.sh verify     # 819 tests ~10 s, smoke test ~4 s, ruff
./scripts/reproduce.sh            # what has and has not been produced here
```

`scripts/smoke_test.py` runs the real pipeline — real `Dataset`, real MONAI
transforms, real registry-built model and loss, real `Trainer`, real
checkpointing, real sliding-window validation — against two synthetic
preprocessed cases, in about 4 seconds. Run it before every Kaggle session.

**What a green CPU suite does not prove.** Three CUDA-only faults have shipped
past it: a metrics device mismatch in `evaluate.py`, the CuPy HD95 failure in
`Trainer.validate`, and an RNG-restore `TypeError` on GPU resume (found at
epoch 130 of a real run, after 296 local tests passed). Any code touching
`.numpy()`, a device transfer, or a MONAI metric needs to be read with "what if
this tensor is on CUDA?" in mind, because no local test will ask.
