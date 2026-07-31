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
  - **145 tests, CPU, ~3.5s.** Bare `pytest` works from repo root (`pythonpath = ["src"]`)
- **Next:** losses (`DiceCELoss` wired through a registry, sigmoid for overlapping regions), metrics (per-region Dice, HD95), then the U-Net baseline model
- **Not done yet:** repo is **not under git** — no `git init` has been run

### Decisions worth remembering

- `get_device` raises on an explicit `"cuda"` request when CUDA is absent rather than falling back to CPU — a silent fallback would burn a 12-hour Kaggle session at CPU speed. `"auto"` never resolves to MPS.
- `set_seed` restores `cudnn.benchmark = True` after MONAI's `set_determinism` turns it off. 3D conv backward kernels are nondeterministic regardless, so the project reports mean ± std across seeds; disabling autotuning cost throughput on fixed 96³ patches without buying a guarantee we never claimed. Pass `cudnn_benchmark=False` for variable-shape work such as sliding-window inference.
- On Kaggle, install `requirements.txt` **without** `torch`/`torchvision` — the image ships a CUDA-matched build, and pip would replace it with a wheel that loses the GPU.
- BraTS modality suffixes are matched **exactly**, never by substring or glob: `"_t1" in name` is also true for `_t1ce.nii.gz`, which would silently make `t1` and `t1ce` the same file. Note also that 2023+ names FLAIR as `-t2f`, which a similarity-based mapping would wrongly pair with `t2`.
- The preprocessing crop bbox is computed from the **raw** image, not the normalized one. A channel with constant foreground has `std == 0`, so `normalize_nonzero` zeros it out and its support would vanish from the union bbox — cropping away most of the brain.
- `meta.json`'s `bbox` + `original_shape` are load-bearing: predictions must be un-cropped back into original geometry to be a valid BraTS submission.
- **Dataset size:** ~38 MB/case preprocessed (float16 image + uint8 label). The full 1251-case BraTS 2021 is roughly 48 GB, over Kaggle's per-dataset guideline. Plan on a subset via `data.preprocessing.limit=N`; the script prints actual size and warns past 20 GB.
- **Never use MONAI's `ConvertToMultiChannelBasedOnBratsClassesd`.** It tests for enhancing tumor at raw label `4`, but preprocessing already remapped labels to `{0,1,2,3}`. It would produce an all-zero ET channel silently — a model that never learns enhancing tumor, with no error anywhere. Use `neurovision.data.transforms.ConvertToRegionsd`.
- **Transform order:** crop before converting to regions. `RandCropByPosNegLabeld` needs the single-channel integer label to find foreground; a 3-channel binary mask breaks its sampling.
- **Kaggle uses `dataset_type: dataset` (no caching).** Offline preprocessing already removed the expensive work, so the cacheable prefix is just an `.npy` memmap read. `CacheDataset` costs ~120 MB RAM per case (float32 after region expansion) against ~13 GB total; `PersistentDataset` writes a float32 cache ~3x larger than the float16 source. Use `cache` locally on a small subset when iterating on transforms.
- **Splits are frozen once written** (`configs/data/splits.yaml`). Adding cases later raises rather than reshuffling val/test under an already-reported result; regenerating takes an explicit `overwrite=True`.
