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

1. **All training code must run unmodified on a Kaggle notebook: T4 (16 GB VRAM), ~13 GB RAM, 12-hour session limit.** **The P100 is no longer usable** — measured 2026-08-01 on a real Kaggle session: the stock image's PyTorch targets sm_70+ and the P100 is sm_60, so `Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation`. Request `machine_shape: NvidiaTeslaT4` in `kernel-metadata.json`. VRAM budget is unchanged (T4 is also 16 GB), and AMP is *better* on T4 — it has tensor cores, the P100 did not.
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
  - `src/neurovision/models/encoders/cnn.py` — `CNNEncoder`, the CNN half of the dual encoder. `len(channels)` levels, level `i` at stride `2**i` (level 0 is a full-resolution stem), `forward` returns a fine-to-coarse `list[Tensor]`, exposes `out_channels` / `strides` / `num_levels` so the decoder and fusion module do not re-derive widths. GroupNorm only, `Dropout3d`, optional per-stage gradient checkpointing, Kaiming fan-out init matched to LeakyReLU, `zero_init_residual` flag. `build_cnn_encoder(cfg)` reads `cfg.model.encoder.cnn` — **no YAML for it exists yet**, and it requires a `zero_init_residual` key. Deliberately NOT in the model registry (that holds whole networks)
  - `src/neurovision/models/fusion/` — `registry.py` (`register_fusion`, `build_fusion(cfg, cnn_channels, swin_channels, level)`, `available_fusions`; a builder needs the per-level widths, so unlike the model/loss registries it takes more than `cfg`), `adaptive_fusion.py`: `FusionBlock` base (shared `forward(cnn_feat, swin_feat, return_gate=False)` contract, output always `cnn_channels`-wide), `AdaptiveGatedFusion` (the novel module: Swin→CNN width adapter, `GateGenerator`, `WindowedCrossAttention`, gated residual merge with `layer_scale`), plus ablation baselines `ConcatFusion` and `AddFusion`. Registered as `adaptive_gated` / `concat` / `add`. `build_*` reads `cfg.model.fusion` — **no YAML for it exists yet**
  - `src/neurovision/models/decoder/unet_decoder.py` — `UNetDecoder`, `AttentionGate`, `_match_spatial`. Takes skips fine-to-coarse, returns decoder FEATURES fine-to-coarse (length `len(skips) - 1`, bottleneck excluded — it is an input, not an output). Owns no heads. Exposes `out_channels` / `num_stages` / `skip_channels`. Reuses `ResidualBlock` from `encoders/cnn.py`. 12.93M params at skip widths `[32, 64, 128, 256, 320]`
  - `src/neurovision/models/heads/segmentation.py` — `SegmentationHead`, a `Dropout3d` + 1x1x1 conv emitting raw logits (no sigmoid: the loss and `inference/postprocess` each apply their own). Zero bias init. One instance per supervised resolution
  - `src/neurovision/models/neurovision.py` — `NeuroVisionX` + `@register_model("neurovision")`. Dependency-injected (submodules built by `build_neurovision(cfg)` and passed in, so tests can assemble mismatched pieces to prove the validation fires). Five construction-time checks: CNN must have exactly one more level than Swin, strides must align, one fusion block per Swin level, decoder skip widths must equal CNN widths, `deep_supervision_levels` in `1..decoder.num_stages`. `forward_with_gates(x)` returns `(logits, gate_maps)` for the explainability figure
  - `configs/model/neurovision.yaml` — the full model config, and the first YAML backing `cfg.model.encoder.cnn` / `.swin` / `cfg.model.fusion`, which had no config file until now. `configs/experiment/neurovision.yaml` — includes `_baseline_common` exactly like the two baseline experiments, so the only difference from them is the architecture
  - **Measured parameter count, from the production config: 34,877,167 total** — CNN encoder 18,854,208 / Swin encoder 2,043,330 / fusion blocks 1,049,284 / decoder 12,929,664 / heads 681. Sits between `unet3d` (12.87M) and `swinunetr` (62.19M). Note the CNN branch, not the transformer, is the parameter bulk: the Swin branch is only 2.0M because `num_levels=4` drops `layers4`, which is 75% of a full SwinViT-B backbone
  - **403 tests, CPU, ~7s.** Bare `pytest` works from repo root (`pythonpath = ["src"]`)
- **Data is ready:** BraTS 2021, 1251/1251 cases preprocessed (34.22 GB, 0 failed), splits frozen at 875/187/189 in `configs/data/splits.yaml`, visual QC passed (`outputs/qc/`), `scripts/smoke_test.py` green. `training.batch_size` now defaults to 1 (4 patches/step)
- **Next:** upload `outputs/kaggle_upload` to Kaggle (`amishyadav123/neurovision-brats-prep`, PRIVATE), push the repo, then the Kaggle baseline run via `notebooks/kaggle_train.ipynb`. The full model is now trainable end to end (`+experiment=neurovision`) but has never seen a GPU or real data — the baselines come first, since the whole claim is relative to them
- **Not built yet, from the architecture in "What this project is":** MC-dropout uncertainty, calibration/ECE analysis, explainability (Integrated Gradients), and the confidence + boundary heads. `SegmentationHead` is the only one of the three heads that exists; `forward_with_gates` is in place for the fusion gate-map figure but nothing consumes it yet
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
- **`torch.cuda.is_available()` is NOT a sufficient GPU check.** On a Kaggle P100 it returns `True` while every kernel launch fails, because the image's torch build does not target sm_60. The only honest test is executing something: `(torch.randn(64,64,device="cuda") @ torch.randn(64,64,device="cuda")).sum().item()`. `notebooks/kaggle_train.ipynb` does exactly this and reports the device name and `sm_XX`. A guard that only calls `is_available()` would have let an entire session start on an unusable GPU.
- **A Kaggle dataset does not reliably mount at `/kaggle/input/<slug>`.** Measured 2026-08-01: `/kaggle/input` contained only `['datasets']`, one level deeper than documented. The notebook therefore *discovers* the data root by globbing the first few levels for a directory holding both `preprocessed/` and `splits.yaml`, and raises listing what is actually mounted when the match is not unique. Never hardcode the mount path.
- **Kaggle accelerator is selectable from `kernel-metadata.json`** via `machine_shape` (`NvidiaTeslaT4` | `NvidiaTeslaP100`) — no UI step. But **Kaggle Secrets are per-notebook and cannot be set from metadata**: adding `WANDB_API_KEY` to the account is not enough, it must be attached to each notebook in Add-ons → Secrets. Hence `USE_WANDB` in the driver notebook, so a throwaway run does not die on a missing secret.
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
- **`zero_init_residual` makes `conv2` receive a gradient of exactly zero on the first optimizer step. That is expected, not a dead layer.** Zero-initializing `norm2.weight` in each `ResidualBlock` makes the residual branch output 0, so the block starts as an exact identity — a standard stabilizer for residual nets. But GroupNorm computes `weight * normalized + bias`, so the gradient reaching `conv2` is scaled by that same zero. `norm2.weight` itself gets a non-zero gradient, moves off zero after one step, and `conv2` trains from step 2 onward. A zeroed `conv2` row in a step-1 gradient histogram is therefore correct behaviour. `tests/test_cnn_encoder.py::test_zero_init_gives_conv2_zero_grad_on_first_step_only` pins both halves. The trick's benefit scales with depth and this encoder is shallow (~7 blocks), so it is a flag rather than a hardcoded default — ablate it if early training looks fine without it.
- **The two encoder branches agree on spatial shape at every level, including odd sizes — measured, not assumed.** `CNNEncoder` downsamples with padded stride-2 convolutions giving `ceil(D/2)`. MONAI's Swin downsamples by patch merging, which *looks* like it would truncate — but `PatchMergingV2.forward` pads first (`pad_input = (h % 2 == 1) or ...; x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))`), so it also produces `ceil(D/2)`. Verified equal across 96³, 64³, 100³, 90³, 33×35×37 and anisotropic shapes: CNN levels 1..4 match Swin outputs 0..3 exactly. (An earlier version of this entry claimed they diverge at odd sizes and that 96³ was therefore load-bearing for shape reasons — that was wrong, and would have discouraged patch-size ablations for no reason.) The alignment test in `tests/test_swin_encoder.py` is what keeps this true across MONAI upgrades; do not delete it.
- **The Swin branch has no stride-1 feature, and cannot be given one.** `patch_embed` downsamples 2× before any attention runs, so Swin's pyramid is strides 2/4/8/16/32 while the CNN's is 1/2/4/8/... Making Swin emit stride 1 would need `patch_size=1`, i.e. attention over 884,736 tokens at 96³ — an instant OOM, and it discards every pretrained checkpoint. This is not a limitation to work around: SwinUNETR itself never fuses a transformer feature at full resolution (its `encoder1` is a plain conv on the raw input). So the full-resolution CNN level 0 stays **CNN-only** and passes to the decoder unfused, and fusion happens at strides 2/4/8/16 where both branches exist.
- **`einops` is a hard requirement for SwinUNETR**, not an optional extra — `SwinUNETR.forward` calls `einops.rearrange`, so a missing einops fails at *run* time with `OptionalImportError`, not at import. Pinned in `requirements.txt`; the Kaggle image already ships it.
- **`use_checkpoint: true` by default for SwinUNETR.** At 96³, batch 2, AMP: params+Adam+grads ≈ 1.0 GB, activations ≈ 11-13 GB without checkpointing (~13 GB total, no headroom on a 16 GB P100) vs ≈ 4-6 GB with it. Costs ~20-30% step time. Turn it off only at `batch_size: 1`.
- **Fusion cross-attention is WINDOWED, and that is a memory decision, not a modelling preference.** At the finest fused level (stride 2) a 96³ patch is 48³ = 110,592 tokens. Measured against the 16 GB budget at B=4, fp16, 4 heads: full attention needs 110592² × 16 ≈ 2×10¹¹ score entries (impossible); pooled-KV / spatial-reduction attention (PVT-style, r=4) needs ≈ 6.1 GB; non-overlapping 4³ windows need ≈ 226 MB. Windowed wins by ~27× over the obvious alternative. The cost — cross-attention itself sees no global context — is acceptable *specifically here* because keys/values come from the Swin branch, which already did shifted-window global mixing across its own stages; cross-attention's job is aligning CNN geometry against context Swin already built, not building it. `WindowedCrossAttention` additionally uses `F.scaled_dot_product_attention`, whose backends never materialize even that 226 MB. Do not "improve" this to global attention without redoing these numbers.
- **`full_attention_max_tokens` (default 512) is a runtime rule, not a per-level hardcode.** A level whose feature map has at most that many voxels gets ONE window equal to the whole map — which *is* full global attention, so both regimes share one code path and cannot drift apart. At 96³ this fires only at stride 16 (6³ = 216 tokens, a 1.5 MB score matrix). Consequence: the attention pattern depends on input shape, which is why the block has no relative-position-bias table (a table sized to a fixed window would break the rule). Safe because inference is fixed-size sliding-window over 96³ patches.
- **Estimated fusion activation memory at 96³, B=4, fp16 (measured-by-hand, before any GPU run):** ~790 MB at stride 2, ~200 MB at stride 4, ~50 MB at stride 8, ~8 MB at stride 16 — **~1.05 GB total**, from ~14 feature-map-sized tensors kept per block for backward. `use_checkpoint: true` on the fusion blocks drops that to ~150 MB for ~30% step time on those blocks. `gate_channels: "channel"` adds ~113 MB at stride 2 alone, which is why `"scalar"` is the default (it is also the interpretable one — a per-voxel gate map is directly plottable for the paper).
- **The fusion merge is `cnn + layer_scale * gate * attn`, never `gate * cnn + (1 - gate) * swin`.** The CNN branch carries the full-resolution detail the boundary-accuracy claim rests on and must never be attenuated by a learned gate. Under this form the gate means exactly one thing — how much transformer context is admitted at this voxel — which is what makes the gate map a publishable figure rather than an uninterpretable blend coefficient. `layer_scale` is a `(1, C, 1, 1, 1)` parameter initialized to 1e-4 (same reasoning as `zero_init_residual` in `cnn.py`); at exactly 0.0 the block is an EXACT identity, so "fusion off" in the ablation is literally the CNN-only network, not an approximation of it.
- **`GateGenerator` zero-inits its output conv's BIAS but deliberately not its WEIGHT.** Bias-zero centres the gate on sigmoid(0) = 0.5 (no prior on either branch); the random weight leaves a spread (measured ~0.27-0.73 on random input, mean ~0.50). Zeroing the weight too would make the gate flat 0.5 everywhere — and since `layer_scale` already starts the block as a near-identity, that would remove the only source of spatial symmetry-breaking in the thing whose entire job is to vary spatially.
- **`_window_partition`'s row order is batch-major and the key-padding mask depends on it.** The mask is built once on a batch-invariant `(1, Dp, Hp, Wp, 1)` validity grid and tiled with `.repeat(B, 1, 1)`; if partition rows were window-major instead, the mask would align with the wrong windows and only the padded border would be wrong — invisible to every shape test. Note the whole class of bug here: a wrong permute inside `_window_partition` would group spatially scattered voxels into a "window", `_window_reverse` would still undo it cleanly, and every shape assertion would still pass while attention ran over nonsense. `tests/test_adaptive_fusion.py` pins this with a coordinate-encoding contiguity test, an exact round-trip, a batch-major ordering test, a B>1 no-cross-batch-leakage test on the padded path, and a hand-rolled `softmax(qk/√d)v` oracle — the last being the only test in the file that checks the attention math against something other than the module itself. Do not delete these.
- **`NeuroVisionX.forward` returns a `Tensor` OR a `list[Tensor]`, and the switch is `self.training and deep_supervision_levels > 1`.** Two independent things force this. Sliding-window inference (and MONAI's inferer) call `model(patch)` and expect a tensor, so eval must never return a list. `DeepSupervisionLoss.forward` takes a sequence ordered HIGHEST RESOLUTION FIRST and upsamples each entry to the target internally, so `[full_res, stride2, stride4]` is already the order it wants — do not reverse it. **MC-dropout hazard:** uncertainty estimation re-runs the network with dropout active, and doing that by calling `model.train()` flips the return type and breaks sliding-window inference. Enable the `Dropout3d` modules individually and leave the model in `eval()`.
- **`model.deep_supervision_levels` and `training.loss.deep_supervision.enabled` must agree, and `build_neurovision` raises if they do not.** A model emitting a list into a loss expecting one tensor is a crash; the reverse (one output, `DeepSupervisionLoss` wrapper) silently reduces to the plain loss and quietly trains something the config does not describe. The check is skipped when there is no `training` group at all, so model-only test configs still build. Consequence worth knowing: `model=neurovision` does NOT compose against the stock training config, which has deep supervision off for the single-output `unet3d` baseline. Use `+experiment=neurovision`, which turns it on — deliberately not flipped in `training/default.yaml`, so the baselines keep composing unchanged.
- **The decoder's `_match_spatial` crops from the END of each axis, never centre.** Both encoders halve with `ceil` and pad at the end, so `2 * ceil(n / 2)` overshoots the skip by at most one voxel and that voxel is the trailing pad. A centre crop would shift every decoder feature half a voxel against its skip and compound up the ladder — a segmentation systematically offset from the image, with plausible output and no shape test ever failing. It raises when the mismatch exceeds 1 voxel per axis, which means a skip list from a different network rather than a rounding effect. Pinned by a content test (arange values, checks the LEADING values survive) and end-to-end on an odd anisotropic pyramid.
- **Decoder attention gates default OFF, on purpose.** The novel contribution is the fusion module. Turning on a second attention mechanism by default makes it impossible to attribute a Dice gain to the fusion block rather than to the gates. `use_attention_gates` exists to be ablated, not to be on. (They also gate at the skip's resolution using the already-upsampled decoder feature, so nothing is resampled inside the gate — Oktay et al.'s original uses a coarser gating signal and resamples internally.)
- **Decoder upsampling is `ConvTranspose3d` by default with `upsample: interp` as an escape hatch.** Transposed convolution is learnable and is what nnU-Net and SwinUNETR use, but its overlapping kernel footprint can checkerboard — periodic texture along a boundary. This project's headline claim is boundary accuracy, so `interp` (trilinear + 1x1x1 conv, cannot checkerboard, costs more memory since it interpolates at the input width) exists specifically to rule that out if HD95 ever looks periodic.
- **Measured activation memory of the full model at 96³ (saved-tensor bytes via `torch.autograd.graph.saved_tensors_hooks`, B=1, fp32, on CPU — real numbers, not arithmetic).** Baseline 4.53 GB with only the Swin branch checkpointed (the config default). Each lever, measured individually: CNN encoder checkpointing −1.58 GB, decoder −1.25 GB, fusion −0.51 GB; CNN + decoder together 1.70 GB total; all three 1.19 GB total. Convert to AMP by roughly ×0.5-0.6 (not exactly half — norm layers stay fp32 under autocast) and multiply by the real batch. Add ~0.56 GB for fp32 weights + grads + Adam's two moments (AMP does not shrink these) and ~1 GB for CUDA context, cuDNN workspace and allocator fragmentation.
- **The batch that matters for this model is `batch_size × samples_per_volume`, and at 96³ that is what decides whether it fits.** With `batch_size: 1` (4 patches/step, the project default): ~4.5-5.5 GB of activations unless checkpointed, ~6-7 GB total — fits with headroom. With `batch_size: 2` (8 patches/step): ~9-11 GB of activations, ~11-13 GB total — fits ONLY with checkpointing on, and OOMs a 16 GB T4 without it. Turning on CNN + decoder checkpointing (~20-30% step time each) is the cheapest lever and buys back more than the fusion block's does.
- **Metrics use `ignore_empty=False` (the BraTS convention):** a region absent from the ground truth scores Dice 1.0 if the prediction is also empty, 0.0 if it predicted anything. **Measured on our actual data (BraTS 2021 training, 1251 cases): 33 cases, 2.6%, have zero enhancing tumor** — not the ~35% widely quoted for BraTS and previously asserted in this file. That figure does not hold for the 2021 training set, which is overwhelmingly high-grade glioma. So the convention moves headline ET Dice by well under a point here, rather than by several. It still must be stated in the paper — the choice is invisible otherwise — but it is not the load-bearing caveat this file used to claim. (43 cases, 3.4%, have zero necrotic core; no case has an entirely empty label.) Re-measure if the dataset ever changes: `metadata.csv` column `n_class_3`. `compute_case_metrics` therefore also emits a `gt_empty_<region>` flag per case, so the ignore-empty mean can be recomputed from the saved per-case table without re-running inference — and `summary().loc["gt_empty_ET", "mean"]` gives the fraction of the eval set where ET Dice is empty-vs-empty scoring rather than real overlap.
- **HD95 empty-case handling is ours, not MONAI's.** `compute_hausdorff_distance` returns NaN for all three degenerate cases (both empty, either side empty) and cannot distinguish them. `hd95()` overrides both-empty to **0.0** — correctly predicting "no tumor here" is not a boundary error and must not vanish from a NaN-skipping mean — and leaves one-sided-empty as NaN, counted in `summary()`'s `n_missing` column. Deliberately not the old BraTS 373.13mm penalty, which turns HD95 into a high-variance Dice proxy and would undercut the boundary-accuracy claim.
- **Splits are frozen once written** (`configs/data/splits.yaml`). Adding cases later raises rather than reshuffling val/test under an already-reported result; regenerating takes an explicit `overwrite=True`.
- **There is no argmax anywhere in inference.** The heads are 3 *overlapping* sigmoid regions, so argmax is wrong twice: the channels are not mutually exclusive (a voxel is legitimately 1 in all three), and argmax forces every background voxel into some region instead of allowing "none of the above". Discretization is sigmoid + per-channel threshold; `postprocess.regions_to_classes` is the inverse of `metrics.classes_to_regions` and is what produces the `{0,1,2,3}` map. Its assignment order is outer-to-inner (`WT`→2, then `TC`→1, then `ET`→3) so the inner region overwrites — reversed, WT paints class 2 over every ET voxel and enhancing tumor silently vanishes from the output.
- **`enforce_nesting` unions inner into outer, never intersects.** Intersection would delete a confidently-predicted ET voxel because the lower-confidence TC channel missed it there — discarding the model's strongest signal to satisfy its weakest. It runs LAST in `postprocess_logits` because raw thresholded output was never nested to begin with (three independent heads), and `keep_largest_component` can additionally pick a different largest component per channel. Note `remove_small_components` alone does *not* break valid nesting: containment is monotonic, so a surviving ET component sits inside a TC component that is at least as large.
- **`MONAI KeepLargestConnectedComponent()` must be given `applied_labels` explicitly.** Its default auto-detection calls `get_unique_labels(img, is_onehot=True, discard=0)`, which unconditionally discards channel *index* 0 — correct for a label map where value 0 is background, wrong here where channel 0 is ET. Left at the default, a case where ET is the only non-empty channel is silently skipped entirely. Same family as the `ConvertToMultiChannelBasedOnBratsClassesd` trap.
- **Evaluation metrics are computed in CROPPED space, predictions are saved UNCROPPED.** The ground-truth `label.npy` is already cropped, and uncropping both sides adds identical background to numerator and denominator — Dice and HD95 come out numerically identical, just slower. Uncropping matters only for the saved artifact, which has to be valid BraTS geometry. `uncrop_to_original` raises when the bbox extent disagrees with the array's spatial shape: that means prediction and `meta.json` came from different preprocessing runs, and placing it anyway yields a spatially shifted submission that looks entirely plausible.
- **`evaluate.py` moves both tensors to CPU before `add_case`.** `regions` is on `device`, `batch["label"]` comes straight off the DataLoader on CPU. Passing them as-is raises a device mismatch inside MONAI's `compute_dice` — but only on CUDA. On a CPU box they agree by accident and the whole suite passes, so no test can catch this class of bug; it is pinned by comment. CPU rather than `device` also keeps HD95's intermediate buffers over a ~240³ volume off the 16 GB budget.
- **HD95 needs `spacing` passed from `meta.json`**, or it is reported in voxels, not millimetres. The boundary-accuracy half of the research claim is a millimetre claim. Note `Trainer.validate` does NOT pass spacing — its HD95 is in voxels and is for monitoring only; the reported numbers come from `scripts/evaluate.py`, which does.
- **`map_location=str(device)` puts the RNG state tensors on CUDA too.** `load_checkpoint` maps the WHOLE payload, and the numpy RNG state is stored as a torch tensor (to keep `weights_only=True` loadable). So on a GPU resume, `saved["state"].numpy()` raises `TypeError: can't convert cuda:0 device type tensor to numpy`, and one line later `torch.set_rng_state` rejects a non-CPU ByteTensor. Both restore paths now `.cpu()` first. Measured on Kaggle 2026-08-02 resuming a real run at epoch 130 — after 296 CPU tests passed. `tests/test_checkpoint.py` has two regression tests using a stand-in whose `.numpy()` fails until `.cpu()` is called; they fail without the fix.
- **Pattern: three separate CUDA-only faults have now shipped past a green CPU suite** (metrics device mismatch in `evaluate.py`, CuPy HD95 in `Trainer.validate`, RNG restore here). The Mac is a correctness harness for *logic*, not for device placement. Any code that touches `.numpy()`, a device transfer, or a MONAI metric should be read with "what if this tensor is on CUDA?" in mind, because no local test will ask that question.
- **Never pass CUDA tensors into `compute_case_metrics` / `add_case`.** MONAI's `compute_hausdorff_distance` computes its distance transform via **CuPy** when the inputs are on CUDA, and on the Kaggle T4 image that CuPy JIT fails outright: `CompileException: Thrust requires at least C++17` (measured 2026-08-01). The CPU path uses scipy and has no such dependency. Both `Trainer.validate` and `scripts/evaluate.py` therefore `.cpu()` prediction and target before scoring — which also keeps HD95's intermediates over a ~240³ volume off the 16 GB VRAM budget. This bug is invisible on the Mac: on CPU the tensors are already on CPU, so all 296 tests pass either way.
- **`inference.sliding_window.output_device` is MONAI's `device` kwarg, not its `sw_device`.** MONAI has both: `device` is where the stitched full-volume output accumulates, `sw_device` is where each cropped window runs through the model. The config field is deliberately *not* named `sw_device` because the two are trivially confusable. Setting it to `"cpu"` means `sliding_window_predict` returns a CPU tensor, on purpose.
- **`et_min_volume` exists and is off.** Zeroing a small ET prediction entirely scores Dice 1.0 on cases with no ground-truth enhancing tumor under `ignore_empty=False`, buying headline ET Dice without the model being any more accurate. On BraTS 2021 that is only 2.6% of cases (see the measured note above), so the trick buys far less here than the folklore suggests — which makes turning it on even harder to justify. It launders exactly the overconfidence the calibration claim is supposed to expose. Turn it on only to reproduce someone else's number, and say so if you do.
- **`skimage`'s `remove_small_objects` is deprecating `min_size` in favour of `max_size`, with flipped semantics** — the new parameter removes objects smaller than *or equal to* its value, the old one only smaller. `min_component_size: 50` currently drops 49-voxel components and keeps 50-voxel ones; after that migration the same number would drop 50 too, shifting every reported HD95. Deps are pinned, so this is dormant — check it before bumping scikit-image.
