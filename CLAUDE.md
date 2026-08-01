# CLAUDE.md — NeuroVision-X

Project instructions for Claude Code. Read this before doing anything in this repo.

---

## What this project is

**NeuroVision-X** — 3D brain tumor segmentation on BraTS multi-modal MRI (T1, T1CE, T2, FLAIR).

Architecture: dual encoder (3D CNN + Swin Transformer) → adaptive gated cross-attention fusion → U-Net decoder → three heads (segmentation, confidence, boundary). Plus MC-dropout uncertainty, calibration analysis, and explainability.

**The research claim is reliability, not raw accuracy.** The headline result we are aiming for is "competitive Dice with substantially better calibration and boundary accuracy," not "+0.4 Dice over SwinUNETR." Design and evaluation decisions should serve that claim.

The author is new to deep learning. Explain non-obvious choices in comments and in the chat response. Prefer clear code over clever code.

---

## Orchestration model — read this first

**You (the main session) run on Opus. You are the architect, not the typist.**

Your job is to think, decide, specify, review, and teach. Implementation is delegated to Sonnet subagents. This is deliberate: Opus reasoning is reserved for design decisions and correctness judgment, while routine code generation goes to a faster model.

### What you do yourself (do NOT delegate)

- Architectural and research decisions, and any trade-off analysis
- Reading and interpreting results, metrics, loss curves, failure modes
- Writing the **spec** that a subagent implements
- Reviewing what a subagent returns, and **explaining it back to me** — I am learning, so this is not optional
- Anything requiring judgment about the research claim, statistics, or the paper
- Small edits: a one-line fix is faster done directly than delegated

### What you delegate to `py-implementer` (Sonnet)

Any new module, any substantial refactor, any file of more than ~40 lines. Write a precise spec first — file path, public API with type signatures, expected tensor shapes, edge cases, and the tests that must pass. A vague delegation produces vague code and you will have to redo it.

### What you delegate to `test-runner` (Sonnet)

Running pytest, the smoke test, or lint. Test output is verbose and belongs in a subagent's context, not yours. Ask for a summary of failures, not the full log.

### What you delegate to `code-reviewer` (Sonnet)

Read-only review of a module after implementation, before I see it. Catches shape bugs, device leaks, hardcoded paths, and missing config plumbing.

### What you delegate to `docs-writer` (Sonnet)

Docstrings, MkDocs pages, README sections, and appending to `docs/experiments.md`.

### The standard loop

```
1. Opus  — decide the design, state the trade-off, get my agreement if it matters
2. Opus  — write a precise implementation spec
3. Sonnet (py-implementer) — implement module + tests
4. Sonnet (test-runner)    — run the suite, report failures only
5. Sonnet (code-reviewer)  — review against the spec and the constraints below
6. Opus  — read the result, judge it, explain it to me in plain terms, iterate
```

Run steps 3–5 in the background where possible and keep talking to me while they work. If a subagent returns something that violates a constraint below, fix the spec and re-delegate rather than patching the output yourself.

**Never delegate more than one module per subagent invocation.** I need to read and understand each piece.

---

## Hard constraints — do not violate these

These apply to you and to every subagent. Subagents load this file, so they are bound by it too — but restate the relevant ones in each spec, because a subagent starts with no memory of our conversation.

1. **All training code must run unmodified on a Kaggle notebook: 1× P100 (16 GB VRAM) or 2× T4, ~13 GB RAM, 12-hour session limit.**
2. **No hardcoded paths, ever.** Every path comes from config. The same code runs on macOS (dev) and Linux/Kaggle (training).
3. **No CUDA-only assumptions.** Device is resolved once, from config, via a `get_device()` utility. Code must run on CPU for tests.
4. **Every training script must support full resume** — model, optimizer, scheduler, AMP scaler, epoch, global step, RNG states, and W&B run ID. Sessions get killed at 12 hours; resume is the foundation, not a feature.
5. **Mixed precision (AMP) on by default** for CUDA, off for CPU/MPS.
6. **Default patch size is 96×96×96.** Do not raise it without being asked.
7. **No new dependency without asking first.** The stack below is fixed.
8. **Every model component ships with a CPU shape test** that runs in under a second.

---

## Machine split

| Machine | Role |
|---|---|
| MacBook Pro M4 (local) | All code, all preprocessing, all tests, inference, explainability, figures, docs, paper |
| Kaggle GPU | Gradient descent only — baseline training, fusion training, ablations, final runs |

If something can run on a CPU, it does not belong in a GPU session. Kaggle hours are rationed (~30/week, free tier).

**Use `device="cpu"` for all local tests, not MPS.** MPS support for 3D convolutions is incomplete and fails silently or obscurely. The Mac is a correctness harness, not a compute device.

---

## Stack — fixed

- Python 3.11
- PyTorch (latest stable compatible with MONAI)
- **MONAI** — transforms, `CacheDataset`, SwinUNETR, DiceCELoss, sliding-window inference, metrics. Prefer MONAI over hand-rolling.
- Nibabel, SimpleITK — NIfTI I/O and resampling
- NumPy, Pandas
- **Hydra** — all configuration
- **Weights & Biases** — the only experiment tracker. No MLflow, no TensorBoard.
- Captum — Integrated Gradients
- pytest, Ruff (with `I` rules for import sorting), Black
- scipy / scikit-image for volumetric ops

**Not used:** Optuna, Docker (until release), Git LFS, mypy, isort, OpenCV, PyTorch Lightning, MONAI bundles (until release).

---

## Repository layout

```
neurovision-x/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── .claude/
│   ├── settings.json         # subagent model policy
│   └── agents/               # py-implementer, test-runner, code-reviewer, docs-writer
├── configs/                  # Hydra configs
│   ├── config.yaml           # root / defaults
│   ├── data/
│   ├── model/
│   ├── training/
│   └── experiment/           # one file per named experiment & ablation
├── src/neurovision/
│   ├── data/                 # readers, preprocessing, dataset, transforms
│   ├── models/
│   │   ├── encoders/         # cnn.py, swin.py
│   │   ├── fusion/           # the novel module
│   │   ├── decoder/
│   │   ├── heads/
│   │   └── registry.py
│   ├── losses/
│   ├── metrics/
│   ├── training/             # trainer, checkpoint, scheduler
│   ├── inference/            # sliding window, MC dropout, postprocess
│   ├── uncertainty/          # calibration, ECE, reliability
│   ├── explainability/
│   ├── visualization/
│   └── utils/                # seed, device, logging, io
├── scripts/                  # CLI entry points (preprocess.py, train.py, evaluate.py)
├── notebooks/                # analysis + the thin Kaggle driver notebook
├── tests/
├── docs/
└── outputs/                  # gitignored
```

---

## Coding conventions

- Type hints on all public function signatures. Docstrings on all public functions (short, Google style).
- Config objects passed in; never read global state inside a module.
- Registry pattern for models and losses so experiments are selected by string in config.
- Logging via the `logging` module, never bare `print` in library code.
- Tensor shapes documented in docstrings as `(B, C, D, H, W)`.
- Randomness only through the seeded generator from `utils/seed.py`.

---

## Testing rules

- **Every model component gets a shape test** on tiny random tensors (e.g. `(1, 4, 32, 32, 32)`), running on CPU in under a second.
- Losses and metrics get tests against hand-computed values (perfect prediction → Dice 1.0; disjoint → Dice 0.0).
- Data pipeline tests use synthetic volumes generated in the test, never real BraTS data.
- The full suite must run on the Mac, on CPU, in under ~60 seconds. If a test needs a GPU, it does not belong in the suite.

---

## How to work with me

- **One module at a time.** Do not generate large multi-file implementations in a single turn, even via subagents.
- After each module: state what it does, how to run its test, and anything I should verify by eye.
- **Explain the delegated code back to me.** A subagent wrote it; I still have to understand it. Walk me through anything non-obvious in a few sentences.
- If a design decision has a real trade-off (memory vs. accuracy, MONAI vs. custom), say so and give a recommendation rather than silently picking.
- If something I ask for conflicts with the constraints above, say so instead of doing it.
- Before writing code that touches memory (models, batch sizes, caching), sanity-check it against the 16 GB VRAM budget.

---

## Current status

> Update this section as the project moves. Claude Code reads it to know where we are.

- **Phase:** 1 — data pipeline
- **Milestone:** 1 (baselines)
- **Working:**
  - Repo skeleton, `.gitignore`, `pyproject.toml` (Ruff E/F/I/UP, line length 100, Black), `requirements.txt` (exact pins), README
  - Local env: `.venv` on Python 3.11.15, full stack installed (torch 2.13.0, MONAI 1.6.0). Package installed editable (`uv pip install -e .`) so `scripts/` run without `PYTHONPATH`
  - `src/neurovision/utils/` — `device.py` (`get_device`, `amp_enabled`), `seed.py` (`set_seed`), `logging.py` (`setup_logging`), `io.py` (json/yaml/`ensure_dir`)
  - Hydra config system — `configs/config.yaml` root + `data/brats.yaml`, `model/unet3d.yaml`, `training/default.yaml`. `scripts/show_config.py` prints the composed config
  - `src/neurovision/data/brats.py` — `scan_brats_root` handles both BraTS 2020 (`_t1`) and 2023+ (`-t1n`) naming, `write_case_index`
  - `src/neurovision/data/preprocessing.py` — nonzero z-score, nonzero-bbox crop, label remap, float16 image / uint8 label `.npy` + per-case `meta.json`
  - `scripts/preprocess.py` — Hydra-driven, multiprocessing + tqdm, resumable (skips processed cases), writes `metadata.csv`, prints total output size with a Kaggle-limit warning
  - `src/neurovision/visualization/qc.py` — `plot_case_slices` (4 modalities × 3 planes, colour label overlay), `plot_intensity_histograms` (before/after normalization)
  - `notebooks/01_verify_preprocessing.ipynb` — visual QC on 3 cases. Reads BOTH the raw tree (for "before" histograms) and the preprocessed output. Set `BRATS_RAW`, optionally `NEUROVISION_PREP_DIR`
  - `src/neurovision/data/dataset.py` — `build_data_dicts`, `make_splits` (frozen seeded 70/15/15), `load_splits`, `build_dataset` (Dataset / CacheDataset / PersistentDataset by config string)
  - `src/neurovision/data/transforms.py` — `ConvertToRegionsd` (labels → ET/TC/WT), `build_train_transforms`, `build_val_transforms`
  - `src/neurovision/losses/` — `registry.py` (`register_loss`, `build_loss`, `available_losses`), `segmentation.py` (`DiceBCELoss`, `DeepSupervisionLoss`, `dice_ce` builder)
  - `src/neurovision/metrics/` — `segmentation.py`: `classes_to_regions`, `binarize`, `dice_score`, `iou_score`, `hd95` (all MONAI functional under the hood), `compute_case_metrics` (one case → flat dict), `MetricAggregator` (`add_case` / `update` / `per_case()` / `summary()` DataFrames)
  - `src/neurovision/models/` — `registry.py` (`register_model`, `build_model`, `available_models`), `baseline.py` (`unet3d` → MONAI `UNet` 12.87M params, `swinunetr` → MONAI `SwinUNETR-B` 62.19M params). `configs/model/swinunetr.yaml` added alongside `unet3d.yaml`
  - `src/neurovision/training/checkpoint.py` — `save_checkpoint` (atomic, writes `last.pt` + optional `best.pt` + pruned `epoch_NNNN.pt`), `load_checkpoint` → `ResumeState`, `find_resume_checkpoint`. Persists model / optimizer / scheduler / AMP scaler / epoch / global step / best metric + name + mode / python+numpy+torch+CUDA RNG / W&B run ID / resolved config
  - `src/neurovision/training/trainer.py` — `Trainer` (AMP autocast + GradScaler, gradient accumulation with end-of-epoch flush, gradient clipping, linear-warmup→cosine `LambdaLR` stepped per epoch, sliding-window validation, per-epoch checkpointing, best-tracking on `cfg.training.checkpoint.monitor`, predictive `max_hours` stop, tqdm, optional W&B)
  - `scripts/train.py` — Hydra entry point. `build_dataloaders` / `init_wandb` / `select_resume_checkpoint` / `run_training` / `main`. Auto-resumes from `last.pt` in the checkpoint dir; `python scripts/train.py` is the command for BOTH starting and resuming
  - `scripts/smoke_test.py` — end-to-end CPU gate. Generates 2 synthetic preprocessed cases (nested ET⊂TC⊂WT labels), composes the REAL Hydra config programmatically, runs `run_training`, asserts `last.pt`/`best.pt` exist and load under `weights_only=True`, `start_epoch == 2`, and `val/dice_mean` is finite. **~4s, exit 0/1.** Run it before every Kaggle session
  - `tests/test_resume.py` — 7 tests, one per silent resume failure: exact weights, epoch off-by-one, Adam moment buffers, scheduler-curve continuity, `best_metric` carry-over, no in-process dependency, RNG continuity
  - `notebooks/kaggle_train.ipynb` — thin Kaggle driver. Clones the repo, installs `requirements.txt` minus torch/torchvision, reads `WANDB_API_KEY` from Kaggle Secrets, validates the attached dataset and any resume checkpoint, composes the real Hydra config, calls `run_training`. Only cell 1 is edited per session
  - `src/neurovision/inference/sliding_window.py` — `build_inferer`, `sliding_window_predict` (raw logits, no discretization). Driven by `cfg.inference.sliding_window`, separate from `cfg.training.sliding_window`
  - `src/neurovision/inference/postprocess.py` — `enforce_nesting`, `remove_small_components`, `keep_largest_component`, `regions_to_classes`, `uncrop_to_original`, `postprocess_logits`
  - `configs/inference/default.yaml` — `sliding_window` / `postprocess` / `evaluation` blocks, wired into `configs/config.yaml`'s defaults after `training`
  - `scripts/evaluate.py` — Hydra entry point. `build_eval_dataloader` / `resolve_checkpoint` / `load_eval_model` / `evaluate_case` / `run_evaluation` / `main`. Writes `per_case_metrics.csv` (rewritten every case), `summary.csv`, `predictions/<case>.npy` (uint8, uncropped), optional `probabilities/<case>.npy` (float16, cropped), `eval_config.yaml`
  - **288 tests, CPU, ~6s.** Bare `pytest` works from repo root (`pythonpath = ["src"]`)
- **Next:** set `training.batch_size=1` (see effective-batch note above), then the Kaggle baseline run
- **Not done yet:** repo is **not under git** — no `git init` has been run

### Decisions worth remembering

- `get_device` raises on an explicit `"cuda"` request when CUDA is absent rather than falling back to CPU — a silent fallback would burn a 12-hour Kaggle session at CPU speed. `"auto"` never resolves to MPS.
- `set_seed` restores `cudnn.benchmark = True` after MONAI's `set_determinism` turns it off. 3D conv backward kernels are nondeterministic regardless, so the project reports mean ± std across seeds; disabling autotuning cost throughput on fixed 96³ patches without buying a guarantee we never claimed. Pass `cudnn_benchmark=False` for variable-shape work such as sliding-window inference.
- On Kaggle, install `requirements.txt` **without** `torch`/`torchvision` — the image ships a CUDA-matched build, and pip would replace it with a wheel that loses the GPU.
- BraTS modality suffixes are matched **exactly**, never by substring or glob: `"_t1" in name` is also true for `_t1ce.nii.gz`, which would silently make `t1` and `t1ce` the same file. Note also that 2023+ names FLAIR as `-t2f`, which a similarity-based mapping would wrongly pair with `t2`.
- The preprocessing crop bbox is computed from the **raw** image, not the normalized one. A channel with constant foreground has `std == 0`, so `normalize_nonzero` zeros it out and its support would vanish from the union bbox — cropping away most of the brain.
- `meta.json`'s `bbox` + `original_shape` are load-bearing: predictions must be un-cropped back into original geometry to be a valid BraTS submission.
- **Dataset size:** ~38 MB/case preprocessed (float16 image + uint8 label). The full 1251-case BraTS 2021 is roughly 48 GB, which is **fine** — Kaggle allows 200 GB per dataset, private or public (checked 2026-08). Do not confuse that ceiling with the ~20 GB `/kaggle/working` OUTPUT quota, which *is* hard and is what constrains `keep_last_n`. An earlier note here conflated the two and would have pushed the project into needlessly subsetting BraTS 2021 — and since splits freeze over whatever case set exists when they are written, subsetting first and growing later means regenerating splits and invalidating anything already reported. Preprocess the full set, freeze splits over the full set, upload once. `KAGGLE_DATASET_WARN_GB` (60) is a "long upload, confirm you meant it" nudge, not a limit.
- **Raw data source:** BraTS 2021 is on Kaggle as `dschettler8845/brats-2021-task1` (13.4 GB, single `BraTS2021_Training_Data.tar`), so the `kaggle` CLI fetches it directly — no Synapse registration. It uses the 2020-style `_t1`/`_t1ce`/`_t2`/`_flair`/`_seg` suffixes that `scan_brats_root` already handles, not the 2023 `-t1n` ones.
- **Two different account names:** GitHub is `AmishhYadav`, Kaggle is `amishyadav123`. Dataset slugs and repo URLs do not share a username.
- **Never use MONAI's `ConvertToMultiChannelBasedOnBratsClassesd`.** It tests for enhancing tumor at raw label `4`, but preprocessing already remapped labels to `{0,1,2,3}`. It would produce an all-zero ET channel silently — a model that never learns enhancing tumor, with no error anywhere. Use `neurovision.data.transforms.ConvertToRegionsd`.
- **Transform order:** crop before converting to regions. `RandCropByPosNegLabeld` needs the single-channel integer label to find foreground; a 3-channel binary mask breaks its sampling.
- **Kaggle uses `dataset_type: dataset` (no caching).** Offline preprocessing already removed the expensive work, so the cacheable prefix is just an `.npy` memmap read. `CacheDataset` costs ~120 MB RAM per case (float32 after region expansion) against ~13 GB total; `PersistentDataset` writes a float32 cache ~3x larger than the float16 source. Use `cache` locally on a small subset when iterating on transforms.
- **Never use `monai.losses.DiceCELoss` for the region targets.** Its `forward` picks the CE term by channel count — `self.ce(...) if input.shape[1] != 1 else self.bce(...)` — so with 3 region channels it applies softmax cross-entropy even when constructed with `sigmoid=True`. Softmax pushes ET/TC/WT to be mutually exclusive, but they are nested. Use `neurovision.losses.segmentation.DiceBCELoss` (`DiceLoss(sigmoid=True)` + `BCEWithLogitsLoss`), which scores exactly 0 on a perfect prediction.
- **`smooth_nr` must stay non-zero** (1e-5, equal to `smooth_dr`). With `smooth_nr=0`, a correctly predicted *empty* region scores Dice 0 — measured loss 1.000, the worst possible — for the right answer. Many BraTS cases have no enhancing tumor, so this would push the model to hallucinate it.
- **The real training batch is `batch_size × samples_per_volume`, not `batch_size`.** `RandCropByPosNegLabeld` returns a LIST of `samples_per_volume` crops per case (even at 1), so `scripts/train.py` must pass `collate_fn=monai.data.list_data_collate` — plain `default_collate` produces a list of per-position dicts and `batch["image"]` raises `TypeError` on the first step. `list_data_collate` flattens the list, so `batch_size: 2` with `samples_per_volume: 4` sends **8** patches of 96³ per step. Verified: `(8, 4, ...)`. That is ~4× the activation memory the `batch_size: 2 # 16 GB VRAM` comment assumed and OOMs SwinUNETR-B on a P100. Standard MONAI BraTS recipe is `batch_size: 1` with `samples_per_volume: 4`.
- **Gradient accumulation flushes at epoch end.** When batch count is not a multiple of `grad_accum_steps`, the trailing window never triggers a step — and without an explicit flush its gradients are neither applied nor zeroed, so they survive into the next epoch and are summed into its first step. Silent: nothing errors, the gradients are just wrong at every epoch boundary.
- **`scaler.unscale_` must run before `clip_grad_norm_`.** Gradients stay multiplied by the AMP scale factor (starts at 65536) until `unscale_`. Clipping first compares a ~65536× inflated norm against `grad_clip_norm: 1.0`, so every step gets clipped to nearly nothing and the model silently stops learning.
- **Resume order in `scripts/train.py` is checkpoint-then-W&B, deliberately inverted.** `wandb.init(id=..., resume="allow")` needs the run id, which only exists after the checkpoint is read. Initialising W&B first creates a fresh run and orphans the one the checkpoint belongs to. Hence `Trainer(..., wandb_run=None)` then `trainer.wandb_run = run`.
- **Checkpoints are written atomically** — temp file in the *same* directory, then `os.replace`. A SIGKILL partway through a 754 MB `torch.save` would otherwise leave a truncated `last.pt` that has already replaced the good one, losing the whole run. Same directory matters: `os.replace` is only atomic within one filesystem. Not fsync'd — the threat model is process SIGKILL at the 12-hour limit, which does not lose page cache; fsync would only add protection against kernel panic, at 1-3s per epoch.
- **Checkpoints must stay `weights_only=True`-loadable.** torch 2.13 defaults `torch.load` to `weights_only=True`, and a payload holding an OmegaConf `DictConfig` or numpy's RNG ndarray fails with `UnpicklingError: Weights only load failed`. So the config is stored as a resolved YAML *string* and numpy's RNG state as a dict with the array as a tensor. `weights_only=False` would work but makes loading any checkpoint arbitrary code execution. `tests/test_checkpoint.py` has a direct `torch.load(..., weights_only=True)` regression guard — if it fails, someone added an unsafe object to the payload.
- **`start_epoch = saved_epoch + 1`.** The saved epoch already completed; resuming *at* it retrains that epoch and desynchronizes the epoch-indexed LR schedule (including `warmup_epochs`), silently training something the config no longer describes.
- **Resume is epoch-granular, not step-granular.** Dataloader position within an epoch is not recoverable with `num_workers > 0`, so a kill mid-epoch loses that epoch. `save_every_n_epochs: 1` caps the loss at one epoch — keep epochs short enough that this is acceptable.
- **Checkpoint sizes: unet3d ~155 MB, SwinUNETR-B ~754 MB** (weights + Adam's two fp32 moment buffers; AMP does not shrink these). Periodic snapshots every 10 epochs over 300 epochs would be ~22.6 GB, past Kaggle's 20 GB `/kaggle/working` quota — hence `keep_last_n`, which globs only `epoch_*.pt` and sorts numerically so `last.pt`/`best.pt` can never be pruned.
- **`best_metric_mode` is stored in the checkpoint.** Dice and IoU are maximized; loss, HD95 and ECE are minimized, and the calibration claim rests on the minimized ones. A `best_metric` fallback of `-inf` under a minimized metric means nothing ever compares as better and `best.pt` silently stops updating for the rest of the run, so the fallback sentinel follows the mode.
- **Model `out_channels` is 3, never 4.** 4 is the raw *class* count `{0,1,2,3}` and also the modality count (`in_channels`), so it is easy to write by accident. The heads are 3 sigmoid channels over the nested ET/TC/WT regions; a 4-channel head makes `BCEWithLogitsLoss` raise against the 3-channel target on the first step. Both model configs interpolate `out_channels: ${data.num_classes}`.
- **SwinUNETR requires inputs ≥ 64 voxels on every axis.** It downsamples 32×, so a 32³ input reaches the bottleneck at 1³ and `InstanceNorm3d` raises `ValueError: Expected more than 1 spatial element when training`. `.eval()` does not help — `InstanceNorm3d` has `track_running_stats=False` and computes per-instance statistics in eval too. The 96³ training patch is fine (3³ bottleneck); this only constrains shape tests, which is why `tests/test_models.py` uses 32³ for the U-Net but 64³ for Swin, at `feature_size=12` (production 48 takes 1.84s forward on CPU, over the suite's budget).
- **`einops` is a hard requirement for SwinUNETR**, not an optional extra — `SwinUNETR.forward` calls `einops.rearrange`, so a missing einops fails at *run* time with `OptionalImportError`, not at import. Pinned in `requirements.txt`; the Kaggle image already ships it.
- **`use_checkpoint: true` by default for SwinUNETR.** At 96³, batch 2, AMP: params+Adam+grads ≈ 1.0 GB, activations ≈ 11-13 GB without checkpointing (~13 GB total, no headroom on a 16 GB P100) vs ≈ 4-6 GB with it. Costs ~20-30% step time. Turn it off only at `batch_size: 1`.
- **Metrics use `ignore_empty=False` (the BraTS convention):** a region absent from the ground truth scores Dice 1.0 if the prediction is also empty, 0.0 if it predicted anything. ~35% of BraTS cases have no enhancing tumor, so this choice moves headline ET Dice by several points and papers rarely state which convention they used. `compute_case_metrics` therefore also emits a `gt_empty_<region>` flag per case, so the ignore-empty mean can be recomputed from the saved per-case table without re-running inference — and `summary().loc["gt_empty_ET", "mean"]` gives the fraction of the eval set where ET Dice is empty-vs-empty scoring rather than real overlap.
- **HD95 empty-case handling is ours, not MONAI's.** `compute_hausdorff_distance` returns NaN for all three degenerate cases (both empty, either side empty) and cannot distinguish them. `hd95()` overrides both-empty to **0.0** — correctly predicting "no tumor here" is not a boundary error and must not vanish from a NaN-skipping mean — and leaves one-sided-empty as NaN, counted in `summary()`'s `n_missing` column. Deliberately not the old BraTS 373.13mm penalty, which turns HD95 into a high-variance Dice proxy and would undercut the boundary-accuracy claim.
- **Splits are frozen once written** (`configs/data/splits.yaml`). Adding cases later raises rather than reshuffling val/test under an already-reported result; regenerating takes an explicit `overwrite=True`.
- **There is no argmax anywhere in inference.** The heads are 3 *overlapping* sigmoid regions, so argmax is wrong twice: the channels are not mutually exclusive (a voxel is legitimately 1 in all three), and argmax forces every background voxel into some region instead of allowing "none of the above". Discretization is sigmoid + per-channel threshold; `postprocess.regions_to_classes` is the inverse of `metrics.classes_to_regions` and is what produces the `{0,1,2,3}` map. Its assignment order is outer-to-inner (`WT`→2, then `TC`→1, then `ET`→3) so the inner region overwrites — reversed, WT paints class 2 over every ET voxel and enhancing tumor silently vanishes from the output.
- **`enforce_nesting` unions inner into outer, never intersects.** Intersection would delete a confidently-predicted ET voxel because the lower-confidence TC channel missed it there — discarding the model's strongest signal to satisfy its weakest. It runs LAST in `postprocess_logits` because raw thresholded output was never nested to begin with (three independent heads), and `keep_largest_component` can additionally pick a different largest component per channel. Note `remove_small_components` alone does *not* break valid nesting: containment is monotonic, so a surviving ET component sits inside a TC component that is at least as large.
- **`MONAI KeepLargestConnectedComponent()` must be given `applied_labels` explicitly.** Its default auto-detection calls `get_unique_labels(img, is_onehot=True, discard=0)`, which unconditionally discards channel *index* 0 — correct for a label map where value 0 is background, wrong here where channel 0 is ET. Left at the default, a case where ET is the only non-empty channel is silently skipped entirely. Same family as the `ConvertToMultiChannelBasedOnBratsClassesd` trap.
- **Evaluation metrics are computed in CROPPED space, predictions are saved UNCROPPED.** The ground-truth `label.npy` is already cropped, and uncropping both sides adds identical background to numerator and denominator — Dice and HD95 come out numerically identical, just slower. Uncropping matters only for the saved artifact, which has to be valid BraTS geometry. `uncrop_to_original` raises when the bbox extent disagrees with the array's spatial shape: that means prediction and `meta.json` came from different preprocessing runs, and placing it anyway yields a spatially shifted submission that looks entirely plausible.
- **`evaluate.py` moves both tensors to CPU before `add_case`.** `regions` is on `device`, `batch["label"]` comes straight off the DataLoader on CPU. Passing them as-is raises a device mismatch inside MONAI's `compute_dice` — but only on CUDA. On a CPU box they agree by accident and the whole suite passes, so no test can catch this class of bug; it is pinned by comment. CPU rather than `device` also keeps HD95's intermediate buffers over a ~240³ volume off the 16 GB budget.
- **HD95 needs `spacing` passed from `meta.json`**, or it is reported in voxels, not millimetres. The boundary-accuracy half of the research claim is a millimetre claim.
- **`inference.sliding_window.output_device` is MONAI's `device` kwarg, not its `sw_device`.** MONAI has both: `device` is where the stitched full-volume output accumulates, `sw_device` is where each cropped window runs through the model. The config field is deliberately *not* named `sw_device` because the two are trivially confusable. Setting it to `"cpu"` means `sliding_window_predict` returns a CPU tensor, on purpose.
- **`et_min_volume` exists and is off.** Zeroing a small ET prediction entirely scores Dice 1.0 on the ~35% of cases with no enhancing tumor under `ignore_empty=False`, buying headline ET Dice without the model being any more accurate. It launders exactly the overconfidence the calibration claim is supposed to expose. Turn it on only to reproduce someone else's number, and say so if you do.
- **`skimage`'s `remove_small_objects` is deprecating `min_size` in favour of `max_size`, with flipped semantics** — the new parameter removes objects smaller than *or equal to* its value, the old one only smaller. `min_component_size: 50` currently drops 49-voxel components and keeps 50-voxel ones; after that migration the same number would drop 50 too, shifting every reported HD95. Deps are pinned, so this is dormant — check it before bumping scikit-image.
