# CLAUDE.md — NeuroVision-X

Project instructions for Claude Code. Read this before doing anything in this repo.

**This file is deliberately short.** The long-form records live where they belong:

| File | What it holds |
|---|---|
| `docs/research/master_plan.md` | **THE ACTIVE PLAN.** Read it before planning any work. Starting cold? Its §4 *Execution order* is the queue, the dependency arrows and the working agreement |
| `docs/experiments.md` | Every run and every measured result, notes 1–44 |
| `docs/paper/claims_and_evidence.md` | The gate on what may be written. A claim not in that table does not go in the paper |
| `docs/lessons.md` | The traps, with evidence. Each one already cost GPU hours, a wrong number, or a silent bug |
| `docs/project_state.md` | The Milestone 1–3 build record, archived |
| `docs/gpu_session_checklist.md` | Rules for a GPU session, each written against a loss already suffered |
| `docs/reproducibility.md` | Which artifacts are caches, and the exact command to rebuild each |

---

## What this project is

**NeuroVision-X** — 3D brain tumour segmentation on BraTS multi-modal MRI (T1, T1CE, T2, FLAIR),
wrapped in a clinical-imaging pipeline.

Model: dual encoder (3D CNN + Swin Transformer) → adaptive gated cross-attention fusion → U-Net
decoder → three heads (segmentation, confidence, boundary). Plus MC-dropout uncertainty, calibration,
explainability, an atlas-based anatomical report, and a demo viewer with live upload.

**The thesis, as of Milestone 5 (rewritten 2026-09-26 to match the evidence):**

> A tumour segmentation model wrapped in a distribution-free error bound and a refusal gate: the
> bound holds in distribution, fails under distribution shift in proportion to the shift, and the
> gate cannot see cohort-level shift. We measure where it breaks, and what a new site needs in order
> to restore it.

**Not** "safe to deploy on data it was not trained on" — that was the Milestone 4 thesis and note 47
refuted it (paediatric silent-failure rate 49.5%; the gate is nearly inert on SSA).
**Not** "reliability, not raw accuracy" — that was the Milestone 1–3 framing and the data refuted it.
**Not** a SOTA architecture claim — the founding hypothesis returned a pre-registered null.

**What this is for:** a college course project (semester demo + report, late Nov 2026), then the
author's capstone (summer 2027), plus a journal paper (MELBA). Not a product, not deployed, not for
clinical use.

The author is new to deep learning. Explain non-obvious choices in comments and in the chat response.
Prefer clear code over clever code.

---

## Orchestration model

**You (the main session) run on Opus. You are the architect, not the typist.**

| Do yourself | Delegate |
|---|---|
| Architecture and research decisions, trade-off analysis | `py-implementer` (Sonnet) — any new module, any substantial refactor, any file over ~40 lines |
| Reading and interpreting results, metrics, failure modes | `test-runner` — pytest, smoke test, lint. Ask for failures only, not the full log |
| Writing the **spec** a subagent implements | `code-reviewer` — read-only review after implementation, before the user sees it |
| Reviewing what comes back, and **explaining it back to me** — I am learning, this is not optional | `docs-writer` — docstrings, MkDocs, README, appending to `docs/experiments.md` |
| Judgement about the research claim, statistics, the paper | |
| Config, requirements files, docs, plan and status updates | |

**Opus writes no production code. This is a hard rule, not a preference.** Every `.py` file in
`src/`, `scripts/`, `tests/` and `app/` — new or modified, one line or five hundred — is written by a
Sonnet subagent (`py-implementer`) from a spec Opus wrote. Opus reads, judges, and sends it back if
it is wrong; it does not pick up the keyboard. The exception is a genuine one-liner that a
round-trip would make slower to fix than to read, and even then the fix is described in the response
so it is visible. Opus *may* write directly: YAML config, `requirements*.txt`, Markdown, and the plan
and status boards.

Why: a spec that survives regeneration is worth more than a patch that does not, and the author has
to be able to read every module as a self-contained artifact with its own tests.

Standard loop: decide the design → write a precise spec → `py-implementer` → `test-runner` →
`code-reviewer` → read, judge, explain in plain terms, iterate. Run the middle three in the
background and keep talking to me while they work.

A subagent starts with no memory of our conversation, so **restate the relevant hard constraints in
every spec**. If one returns something that violates a constraint, fix the spec and re-delegate
rather than patching the output yourself. **Never delegate more than one module per invocation** — I
need to read and understand each piece.

---

## Hard constraints — do not violate these

1. **Training code runs unmodified on any single CUDA card with ≥16 GB VRAM, and survives a session
   kill at any point.** 16 GB is the *portability floor*, not the target — a config may opt into more
   (college GPU access since 2026-08-19) but must still be runnable at 16 GB, if necessary with
   gradient checkpointing on. The card is never assumed present at import time. For any Kaggle
   fallback: request `machine_shape: NvidiaTeslaT4`; **the P100 is unusable** (stock PyTorch targets
   sm_70+, P100 is sm_60).
2. **No hardcoded paths, ever.** Every path comes from config. The same code runs on macOS and Linux.
3. **No CUDA-only assumptions.** Device resolved once from config via `get_device()`. Code runs on CPU
   for tests.
4. **Every training script supports full resume** — model, optimizer, scheduler, AMP scaler, epoch,
   global step, RNG states, W&B run ID. A shared cluster preempts jobs; resume is the foundation.
5. **AMP on by default** for CUDA, off for CPU/MPS.
6. **Default patch size is 96³.** Do not raise it without being asked. (The trained comparison family
   is at 64³ — see `docs/experiments.md`.)
7. **No new dependency without asking first.** The stack below is fixed. Approved additions for
   Milestone 4 are isolated in `requirements-analysis.txt` and `requirements-clinical.txt`; the root
   `requirements.txt` stays unchanged.
8. **Every model component ships with a CPU shape test** on tiny tensors, running in under a second.

---

## Machine split

| Machine | Role |
|---|---|
| MacBook Pro M4 (local) | All code, preprocessing, tests, evaluation, explainability, figures, docs, paper |
| Dedicated GPU (college cluster) | Gradient descent only |
| Kaggle GPU (fallback) | Same role, rationed and session-capped; keep configs runnable there |

If it can run on a CPU, it does not belong in a GPU session. Deterministic evaluation measures **~1.4 min/case** on the M4 at `sw_batch_size=1` — all 189 test
cases in ~4.5 h (measured 2026-09-20, note 48; the old "~15 cases/min / 25 minutes" figure here was
wrong by two orders of magnitude and made a healthy run look stalled). Only checkpoints and logs cross the wire
back from the GPU box; never `logits/`, `predictions/` or `uncertainty/`, which are caches and are
cheaper to rebuild locally.

**Use `device="cpu"` for all local tests, never MPS.** MPS 3D convolution support is incomplete and
fails silently or obscurely. The Mac is a correctness harness, not a compute device.

**Only one heavy local job at a time.** Parallel shards exhausted application memory twice.

---

## Stack — fixed

Python 3.11 · PyTorch · **MONAI** (transforms, `CacheDataset`, SwinUNETR, sliding-window inference,
metrics — prefer MONAI over hand-rolling) · Nibabel, SimpleITK · NumPy, Pandas · **Hydra** for all
config · **Weights & Biases** as the only tracker · Captum · pytest, Ruff (with `I`), Black · scipy /
scikit-image.

**Not used:** Optuna, Docker (until release), Git LFS, mypy, isort, OpenCV, PyTorch Lightning, MONAI
bundles (until release), TensorBoard, MLflow.

---

## Repository layout

```
src/neurovision/    data/ models/{encoders,fusion,decoder,heads} losses/ metrics/
                    training/ inference/ uncertainty/ anatomy/ reporting/ analysis/
                    explainability/ visualization/ utils/
scripts/            CLI entry points, all Hydra-driven off configs/config.yaml
configs/            data/ model/ training/ inference/ calibration/ explainability/
                    anatomy/ analysis/ experiment/
app/                backend/ (FastAPI) + frontend/ (Vite + React + TS + Tailwind)
knowledge/          versioned knowledge base (eloquence map, lobes, involvement groups)
tests/ docs/ notebooks/ outputs/ (gitignored)
```

Three parallel registries — models, losses, fusion blocks — all `@register_*("name")` decorators over
a `build_*(cfg)` function, looked up by a string in config. A new one must be imported in its
package's `__init__.py` for the decorator to run.

---

## Coding conventions

- Type hints and short Google-style docstrings on all public functions.
- Config objects passed in; never read global state inside a module.
- Registry pattern for models and losses so experiments are selected by string in config.
- `logging` module, never bare `print` in library code.
- Tensor shapes documented as `(B, C, D, H, W)`.
- Randomness only through the seeded generator in `utils/seed.py`.

---

## Testing rules

- Every model component gets a shape test on tiny random tensors, CPU, under a second.
- Losses and metrics get tests against hand-computed values (perfect prediction → Dice 1.0).
- Data pipeline tests use synthetic volumes, never real BraTS data.
- The full suite runs on the Mac, on CPU, in well under a minute. A test needing a GPU does not
  belong in the suite.
- Run **plain `pytest`** — `pyproject.toml` already sets `addopts = "-q"`, so a second `-q` stacks to
  `-qq` and silently drops the pass count.
- **An analysis fix is not verified by its unit tests.** It is verified by re-running the real
  analysis and checking the output moved in the predicted direction. This project has already shipped
  a commit whose message, memory note and 1,000 green tests all claimed a circular-mask bug was fixed
  while every reported number stayed circular.

---

## How to work with me

- **One module at a time.** No large multi-file implementations in a single turn, even via subagents.
- After each module: what it does, how to run its test, what I should verify by eye.
- **Explain delegated code back to me.** A subagent wrote it; I still have to understand it.
- If a design decision has a real trade-off, say so and give a recommendation rather than silently
  picking.
- If something I ask for conflicts with a constraint above, say so instead of doing it.
- Sanity-check anything touching memory (models, batch sizes, caching) against the VRAM budget first.

---

## Current status — 2026-09-26

**Phase: Milestone 5. The live queue is `docs/research/master_plan.md` §4.3, the "Milestone 5" block
at its top** — read it before planning anything. Why it changed shape:
`docs/research/project_review_2026-09-24.md`. Everything else in `master_plan.md` §4.3 is history.

**Where the science stands** (authority: `docs/paper/claims_and_evidence.md`, read it before writing
anything). One clean positive: ET Dice **+0.0267** over a matched U-Net (p_holm 1.4e-21, n=189),
~79% architecture / ~21% capacity, now checked against a measured seed noise floor (D1, note 49:
+0.0021) — but **never yet tested against nnU-Net** (Gate A, P4). Conformal risk control holds in
distribution and breaks under shift in proportion to the shift (C13/C14). The pipeline returns a
usable mask for 85% / 78% / 24% of test / SSA / PED studies, with silent failure 4.2% / 18.3% /
49.5% (note 47). Gate C is MIXED (one cell of five). D0 heavy augmentation and D1 seed 43 are both
12/12 inconclusive. The founding hypothesis is a pre-registered null.

**What is built.** Everything through Milestone 3 (`docs/project_state.md`), plus lesion-wise
metrics, TTA (wired, unmeasured), conformal risk control, the QC model, the end-to-end error budget,
and the full clinical pipeline — DICOM zip in → ingest → input QC → co-registration / SRI24 /
HD-BET → input QC → segmentation (deployed `neurovision` seed 42) → QC-model + conformal signals →
PROCEED / CAUTION / REFUSE, at `/clinical`, with the 3D twin, atlas shells, report, entered-pathology
molecular panel, DICOM-SEG and zip export. A `"refused"` job is a successful outcome, never a failure.
2,230 tests passing, 35 skipped, ~60 s (verified 2026-09-24). Serving command:
`docs/reproducibility.md` §5.

**Open author actions:** eyeball the twin on job `9c2cc294` (P0.7). The Kaggle RSNA-MICCAI rules
are accepted (2026-09-26), so T0.4 is unblocked (P1.2).

**Data on disk.** `data/preprocessed/{brats,brats_ssa,brats_ped}` — `brats` is backed only by the live
Kaggle dataset `amishyadav123/neurovision-brats-prep`, so **do not delete it**. Raw data was deleted
2026-08-19; SHA-256 manifests are committed to `docs/data_manifests/` for SSA/PED only —
**raw BraTS 2021 has no manifest**, so a re-download is verified by re-preprocessing it byte-identical
to `data/preprocessed/brats`. **Five** checkpoints survive: `neurovision_seed43`
(`outputs/neurovision_seed43/checkpoints/best.pt`, epoch 79, D1's arm — a noise-floor measurement,
not a deployment candidate), plus four others
— `neurovision`, `baseline_unet3d`, `ablation_content_only_gate`
(`outputs/ablation_content_only_gate/checkpoints/checkpoints/best.pt`, epoch 79, val dice_mean
0.8933, verified loadable 2026-08-23), and `neurovision_heavy_aug`
(`outputs/neurovision_heavy_aug/checkpoints/best.pt`, epoch 79, val dice_mean 0.89375, D0's arm —
note 48, verdict NULL, so it is a measurement and never a deployment candidate). Only `capacity_control` is permanently lost, and it has
neither predictions, logits nor checkpoint — so **no capacity-control number can ever be re-scored
under a new metric.**

**Saved fp16 logits, the artifact every re-scoring plan runs on** (inventory taken 2026-08-23; note
that `neurovision`'s live directories are nested one level deeper than the others, so a
`outputs/*/logits` glob misses them):

| Run | val (187) | test (189) | SSA (60) | PED (99) |
|---|---|---|---|---|
| `neurovision` | `outputs/neurovision/eval_val/logits` | `outputs/neurovision/eval_test/logits` | `outputs/eval_ssa_neurovision/logits` | `outputs/eval_ped_neurovision/logits` |
| `baseline_unet3d` | `outputs/eval_val_baseline_unet3d/logits` | `outputs/eval_test_baseline_unet3d/logits` | `outputs/eval_ssa_baseline_unet3d/logits` | `outputs/eval_ped_baseline_unet3d/logits` |
| `neurovision_heavy_aug` | — | `outputs/eval_test_neurovision_heavy_aug/logits` | `outputs/eval_ssa_neurovision_heavy_aug/logits` | `outputs/eval_ped_neurovision_heavy_aug/logits` |
| `neurovision_seed43` | — | `outputs/eval_test_neurovision_seed43/logits` | `outputs/eval_ssa_neurovision_seed43/logits` | `outputs/eval_ped_neurovision_seed43/logits` |
| `ablation_content_only_gate` | — | — | — | — |
| `capacity_control` | — | — | — | — |

~31 GB in total (D0 and D1 each added ~6 GB); ~69 GiB free as of 2026-09-24,
was 119 GiB after the 2026-09-15 reclaim (`docs/reproducibility.md` §11 lists
exactly what went — every `ambiguity_*` and MC `uncertainty/` cache, `nnunet_raw`, `data/raw`). Both
complete rows mean lesion-wise re-scoring and the whole conformal phase need **zero inference**. `ablation_content_only_gate` has no saved volume artifact at all but
its checkpoint survives, so it costs one CPU pass per split to bring back — at the measured ~1.4 min/case that is ~4.5 h for
test, not the 15 minutes this file used to claim.

Two traps in that table. `outputs/eval_test` is **not** `neurovision` — it is the superseded
200-epoch/96³ U-Net (ET 0.8587), and it is the only directory holding `predictions/` rather than
`logits/`. And `outputs/neurovision/eval_test/summary.csv` is the real published row: ET 0.8708593.

---

## The ten traps that cost the most

One line each. **The evidence for every one is in `docs/lessons.md` — read that file before touching
the subsystem it names.** Do not delete an entry there because it looks obvious; each already fooled
someone.

1. **A calibration reporting mask must never be defined using the ground-truth label.** Ours was, and
   it manufactured 41–57% of the reported ECE behind a fully green test suite.
2. **An `eps` clamp sized for fp32 is a no-op in fp16.** `1.0 - 1e-6` rounds to exactly 1.0, giving
   `log(0)` → NaN. Cost: 10.5 GPU-hours of training on NaN with nothing raising.
3. **Brain-mask Dice scores *higher* on a left–right mirrored atlas** (0.9416 vs 0.9394), so the
   widest-margin check in the alignment gate is blind to the worst error it exists to catch.
4. **Boundary-error shares must be weighted by voxel count, not rate** — this corrected our own
   headline figure from 92% to 74%.
5. **`sliding_window_predict` calls `model.eval()`, which recurses**, so the obvious MC-dropout
   implementation is silently deterministic and returns a meaningless uncertainty map.
6. **`sw_batch_size: 4` is a memory multiplier on CPU** — 12.39 GiB peak vs 5.42 at 1, *and* 57%
   slower. On the Mac, whole-volume jobs run at `sw_batch_size=1`, `data.num_workers=0`.
7. **Never use MONAI's `ConvertToMultiChannelBasedOnBratsClassesd` or `DiceCELoss` here.** The first
   tests for raw label 4 after we have already remapped; the second applies softmax CE to nested
   region channels. Both fail silently.
8. **Never pass CUDA tensors into MONAI's HD95** — it routes through CuPy, which fails to compile on
   the Kaggle image. Three separate CUDA-only faults have now shipped past a green CPU suite.
9. **A probe must be built to reach the failure condition, not merely to run.** Two GPU-hour losses
   share this shape. And verifying `origin/main` proves nothing about a pinned SHA.
10. **Volume-sized artifacts are caches, not results** — but a cache is only regenerable while its
    checkpoint exists, and this project has already lost two checkpoints.

Honourable mention, because it recurs in a new disguise every time: **any short token you match
against a path or prose is a substring of some longer word there.** `"_t1" in name` is true for
`_t1ce`; `"val" in basename` is true for `eval_test`; an "absent phrase" assertion fails against a
document that quotes the phrase for a correct reason. Three instances so far.
