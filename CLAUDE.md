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

- **Phase:** 0 — repository scaffolding
- **Milestone:** 1 (baselines)
- **Working:** nothing yet
- **Next:** subagent setup, then repo skeleton, config system, utils
