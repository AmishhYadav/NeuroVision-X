# Execution Plan — Milestone 3

**Written:** 2026-08-19 · **Horizon:** ~6 weeks · **Status:** active

**Relationship to other documents.** This supersedes the *sequencing and gates* of
`docs/research/improvement_plan.md` wherever the two disagree; that document's
measurements, cost table and fallback analysis remain valid and are cited here
rather than repeated. Read `docs/experiments.md` notes 11–30 for the numbers
behind every claim below. `docs/research/contribution.md` holds the P1–P5
ablation ladder definitions this plan schedules.

---

## Context — why this plan exists

The project is in a strange position. The engineering is finished and unusually
rigorous: 1,373 tests, a full training/evaluation/analysis/reporting stack, a
working demo, and a trained model that beats a **parameter-matched** capacity
control by **+0.0211 ET Dice (p_holm 7.3e-19)**. Almost no comparable project
runs that control.

But six hypotheses were tested properly and **five came back null or negative**:

| Claim | Outcome | Evidence |
|---|---|---|
| Beats a matched baseline on accuracy | **Holds** | ET +0.0267, p_holm 7.2e-22 |
| Better calibrated | Fails | `ece_mean` inconclusive; neurovision is *worse* uncalibrated (0.0446 vs 0.0395) |
| More accurate boundaries | Fails | all HD95 inconclusive; WT nominally worse |
| Better uncertainty | Fails | AURC gain 37.6% vs 40.6% — within noise |
| Accuracy → better report | Fails | 1 of 25 metrics conclusive |
| Holds up on new data | **Fails badly** | pooled n=159: `dice_TC` **−0.0333, p_holm 0.0132, worse**; PED alone −0.0595, p_holm 0.0002 |

The diagnosis is not bad luck. Five of the six were **downstream** claims — each
asked "does a small ET Dice gain also buy me X?" A gain of 0.02 Dice is a
handful of voxels at a tumour margin. It cannot move a whole-volume calibration
metric, cannot flip which anatomical structures a tumour overlaps, and cannot
survive a distribution shift larger than itself. Each null was predictable from
an effect-size argument that was never made before spending the compute.

**This plan does two things.** It converts the negatives from wounds into the
argument, by pivoting the contribution to something the negatives *motivate*: a
free failure-detection signal that only a dual-encoder model can produce. And it
fixes the process error — every phase below states its pass condition **before**
it runs, and the first two phases cost zero GPU hours so the central bet is
tested for two days of CPU time rather than 180 GPU-hours.

**Governing rule for the whole plan:** before any experiment, state the smallest
effect the test can detect and the plausible size of the real effect. If
plausible < detectable, do not run it.

---

## Disk and migration

### Disk

**Situation.** `/System/Volumes/Data`: **460 GiB total, 58 GiB free (87% full)**.
The repo is 144 GB by `du`, but ~34 GB of that is **hardlinked, not duplicated**
(`outputs/kaggle_upload/preprocessed/` shares inodes with
`data/preprocessed/brats/`), so true physical footprint is ~110 GB.

> **Trap to avoid:** deleting `outputs/kaggle_upload/` looks like it frees 34 GB.
> It frees **≈0 bytes** — the inodes are still referenced by
> `data/preprocessed/brats/`. Do not count it.

Deletion is organised in tiers by risk. Nothing irreplaceable is touched.

**Tier 1 — zero risk (~22 GB)**

| Target | Size | Why safe |
|---|---|---|
| `data/raw/BraTS2021_Training_Data.tar` | 12 GB | Byte-identical content to the extracted tree sitting next to it |
| every `predictions/` directory (8 dirs) | 9.6 GB | Exactly reconstructible from `logits/` via `postprocess_logits` with each run's own recorded config — already verified to reproduce published ET Dice to a delta of **0** |
| `__pycache__/` across the repo | 127 MB | Regenerates on any Python run |

**Tier 2 — low risk, re-downloadable (~42 GB)**

| Target | Size | Recovery path |
|---|---|---|
| `data/raw/BraTS2021_Training_Data/` | 12 GB | `kaggle datasets download dschettler8845/brats-2021-task1` (13.4 GB) |
| `data/external/` (PED 20 GB + SSA 10 GB) | 30 GB | Re-download from the BraTS 2023 challenge portal |

Preprocessing is finished, validated and QC'd for all three cohorts — the 38 GB
of preprocessed arrays is what training and evaluation actually read. Raw is
only needed to *re-*preprocess. **Before deleting, write checksum manifests** to
`docs/data_manifests/` and commit them, so a re-download can be verified
identical.

**Tier 3 — superseded artifacts whose checkpoints are already gone (~13 GB)**

| Target | Size | Note |
|---|---|---|
| `outputs/eval_test_capacity_control/{logits,predictions}` | 5.0 GB | Checkpoint recorded as `/tmp/capout/checkpoints/best.pt` — **gone**. Keep `per_case_metrics.csv`, `summary.csv`, `eval_config.yaml` (all < 1 MB); they are what notes 19–21 were computed from. The capacity control is retrained in Phase 3 anyway |
| `outputs/baseline_unet3d_e130/` volume dumps | 6.8 GB | Superseded 96³/200-epoch run; checkpoint recorded under `~/Downloads` — gone. Keep every CSV |
| `outputs/neurovision/eval_val_cpu_partial/` | 1.5 GB | Aborted partial run, superseded by the complete `eval_val` |

**Tier 4 — after Phase 2 completes (~19 GB)**

The remaining `logits/` for runs whose checkpoints still exist (`neurovision`,
`baseline_unet3d`) are regenerable on the Mac CPU in ~25 min/split. **Do not
delete these yet** — Phases 1 and 2 read them for the single-pass entropy
baseline. Delete once Phase 2's results are saved.

**Net effect: 58 GiB free → ~135 GiB after Tiers 1–3, → ~154 GiB after Tier 4.**

**The standing rule that prevents recurrence:** volume-sized artifacts
(`logits/`, `predictions/`, `uncertainty/`) are **caches, not results**. Results
are `per_case_metrics.csv`, `summary.csv`, `eval_config.yaml` and the comparison
CSVs — all under 1 MB and all committable. Regenerate caches on demand; never
treat them as archival.

### Migration to a GPU box

**The docs are already on GitHub.** `.gitignore` excludes only `/data/`,
`/outputs/`, `/checkpoints/`, `/wandb/`, `*.npy`, `*.nii(.gz)`, `.venv/`,
`node_modules/`. Everything else is tracked — **256 files, 8.5 MiB total**:
all of `docs/`, all of `configs/` including the frozen `splits.yaml` /
`splits_ssa.yaml` / `splits_ped.yaml`, plus `knowledge/`, `notebooks/`,
`tests/`, `src/`, `scripts/`, `app/`. A `git clone` gets 100% of it.

**What is not on GitHub and is needed on the GPU box:**

| Item | Size | How it gets there |
|---|---|---|
| `data/preprocessed/brats/` | 34 GB | Already published as the Kaggle dataset `amishyadav123/neurovision-brats-prep`. Pull with the `kaggle` CLI + API token, or `rsync` from the Mac |
| `data/preprocessed/{brats_ssa,brats_ped}/` | 4.2 GB | Publish as a second Kaggle dataset via `scripts/package_for_kaggle.py`, or `rsync`. Only needed for Phase 4.1 |
| Resume checkpoints | 405 MB / 147 MB | `rsync`/`scp`, only when resuming an interrupted run |

> **Consequence of the disk plan:** once raw data is deleted,
> `data/preprocessed/brats/` is the only local copy. It is *not* unbacked — the
> identical 34 GB is live as a Kaggle dataset. That published dataset is now
> load-bearing; do not delete it.

**What comes back FROM the GPU — and what must not.** Come back: **checkpoints
only**, plus the full session log, plus the W&B run. Must not: `logits/`,
`predictions/`, `uncertainty/`. Deterministic evaluation runs on the Mac CPU at
~15 cases/min — all 189 test cases in ~25 minutes — so a full evaluation is
cheaper to redo locally from the checkpoint than to transfer. This is already
the established pattern and is what keeps the Mac's disk from refilling.

**Four hard rules, each written against a loss already suffered:**

1. **Checkpoints go to a persistent path and are copied off before the session
   ends.** The capacity control's went to `/tmp/capout/` and is permanently
   gone; the 200-epoch baseline's went to `~/Downloads` and is gone. Both runs
   are unreproducible without retraining.
2. **`GIT_REF` is pinned to a commit SHA, never a branch**, and the resolved
   HEAD is printed into the run log. Verifying `origin/main` proves nothing
   about a pinned SHA.
3. **Fetch and keep the training log.** The capacity control's GPU hours are
   recorded as "~8, approximate" purely because its log was never retrieved.
4. **W&B online**, not offline, on a box with internet. Offline mode plus an
   ephemeral filesystem is how run metrics get lost.

**First action on new hardware, before any long run:** a 2-epoch timing probe
reporting measured step time and peak VRAM. Never schedule from an estimated
speedup. And the probe must reach the **failure condition**, not merely execute
— this project has lost GPU-hours twice to probes that ran without reaching the
state they needed to test.

---

## Patch size: why the controlled comparisons stay at 64³

The evidence for 96³ is real. The superseded 96³/200-epoch U-Net has *lower* ET
Dice than `neurovision` (0.8587 vs 0.8709) yet produces the **better** structured
report: multifocality agreement 78.3% vs 73.0%, ET-volume agreement 0.0500 vs
0.0533. All three models over-report multifocality (30.7–40.7%) against a true
rate of 22.8%, and the failure being measured is *one lesion fragmenting into
several* — a **context** failure, not a boundary one. More context per forward
pass is exactly what a bigger patch buys.

The cost is the problem. Measured on this architecture:

| Patch | Step time | Peak VRAM (T4) | Per-epoch | 80 epochs |
|---|---|---|---|---|
| 64³ | 1.12 s/it | 6.17 GiB, no checkpointing | 0.272 h | **21.8 h** |
| 96³ | 3.59 s/it | 13.59 GiB, *with* checkpointing | 0.875 h | **70 h** |

96³ is **3.2× the wall-clock**. And a 96³ model cannot be compared against a 64³
baseline — patch size would be a confound sitting directly on the architecture
claim. So "just train it at 96³" means retraining `neurovision`, `baseline_unet3d`
**and** `capacity_control` at 96³, plus the multi-seed and ablation work on top.
The essential GPU programme goes from ~180 T4-hours to roughly ~600, which does
not fit six weeks on any single card.

**Resolution, three parts:**

1. **Phase 2.4 runs a free inference-ROI sweep first.** Evaluate the *existing*
   64³ checkpoints at inference windows of 64³, 96³, 128³. This separates
   "trained on small patches" from "inferred through small windows". If the
   multifocality over-reporting is an inference-window effect it is fixable for
   **zero GPU hours**. Never been tested; costs a CPU afternoon.
2. **Controlled comparisons stay at 64³.** The existing three-model result is
   valid, internally matched and publication-ready. Nothing invalidates it.
3. **One matched 96³ pair** (`neurovision` + `baseline_unet3d`, same seed, same
   schedule) runs in Phase 4.2 as its own internally-matched experiment, ~80
   T4-hours. Its priority is set by Phase 2.4's outcome.

---

## The pivot — what the project is now about

The architecture computes something no single-encoder model can, and the project
currently throws it away.

Every voxel is read twice: once by the **CNN branch** (local texture), once by
the **Swin branch** (global context). Inside each fusion block,
`BranchAmbiguity` (`src/neurovision/models/fusion/adaptive_fusion.py:382-519`)
runs two 1×1×1 probe convolutions on **detached** branch features and computes:

```
disagreement = |sigmoid(l_cnn) − sigmoid(l_swin)|        # (B, 3, D, H, W), in [0,1]
h_cnn, h_swin = normalised Bernoulli entropies            # from logits via softplus
ambiguity = cat([disagreement, h_cnn, h_swin], dim=1)     # (B, 9, D, H, W)
```

`ambiguity` is handed to the gate at `adaptive_fusion.py:787-789` and then
**discarded**. It is a local variable; no public method returns it.

A U-Net has one encoder. It has no second opinion to disagree with. It cannot
produce this signal at any price.

**The claim:** disagreement between two ways of seeing the same image is highest
where the image is unfamiliar. On a routine adult glioma the branches agree. On
a pediatric case — where the model is measurably, catastrophically worse
(`dice_TC` −0.0595) — they should diverge, giving a warning **before** anyone
knows the right answer.

| Method | What it measures | Cost | Who can use it |
|---|---|---|---|
| Prediction confidence | closeness to threshold | 1 pass | any model |
| MC-dropout | sensitivity to weights | 10 passes | any model |
| Deep ensemble | disagreement between models | 5 models | any model, 5× training |
| **Branch disagreement** | disagreement between two **ways of seeing** | **free** | **dual-encoder only** |

---

## Phase 0 — Housekeeping (Days 1–2, zero GPU)

**Fixes:** the disk ceiling, and the artifact-loss pattern that has already
destroyed two checkpoints.

| # | Action | Component |
|---|---|---|
| 0.1 | Write and commit checksum manifests for `data/external/` and `data/raw/` before deleting anything | new `docs/data_manifests/` |
| 0.2 | Verify every small CSV in `eval_test_capacity_control/`, `baseline_unet3d_e130/`, `eval_test/` is present and readable, then execute Tier 1–3 deletions | `outputs/` |
| 0.3 | Publish `data/preprocessed/{brats_ssa,brats_ped}` as a Kaggle dataset | reuse `scripts/package_for_kaggle.py` |
| 0.4 | Encode the four migration rules as a checklist | new `docs/gpu_session_checklist.md` |
| 0.5 | Document the regenerable-cache list with the exact rebuild command for each | `docs/reproducibility.md` |

**Verification:** `df -h` shows ≥ 130 GiB free; `pytest` green (1,373 tests,
~25 s); `python scripts/smoke_test.py` exits 0.

---

## Phase 1 — Is the signal real? (Week 1, zero GPU) · **HARD GATE**

**Fixes:** the project's central unknown, for two days of CPU instead of 180
GPU-hours.

### 1.1 — Expose the ambiguity map (small, additive code change)

The map is unreachable today. `forward_with_gates` returns only the post-gate
sigmoid; `forward_with_branch_logits` returns raw branch logits but is **only
ever called when `self.training` is True** (`neurovision.py:359`), so it is dead
in eval mode.

Mirror the existing `forward_with_gates` idiom exactly:

| File | Change |
|---|---|
| `src/neurovision/models/fusion/adaptive_fusion.py` | Add `FusionBlock.forward_with_ambiguity(cnn_feat, swin_feat) -> tuple[Tensor, Tensor \| None]` returning `(fused, None)` by default — so `ConcatFusion`/`AddFusion` satisfy it for free, exactly as they do for `return_gate`. Override in `AdaptiveGatedFusion` to return the `(B, 3*num_regions, D, H, W)` ambiguity tensor |
| `src/neurovision/models/neurovision.py` | Add `NeuroVisionX.forward_with_ambiguity(x) -> tuple[Tensor, list[Tensor \| None]]`, built by walking the pyramid manually like `forward_with_gates` (`neurovision.py:395-435`) so it **always returns a single logits tensor regardless of `self.training`** |

**Constraints for the implementer:** do not touch `forward`'s three return types
(`Tensor` / `list[Tensor]` / `MultiTaskOutput`) — sliding-window inference
depends on eval mode returning a plain `Tensor`. Do not remove the `.detach()`
on the probe inputs; it is load-bearing twice over. CPU shape test on
`(1, 4, 32, 32, 32)` running under one second.

### 1.2 — `scripts/extract_ambiguity.py`

Mirror `scripts/extract_gates.py` (Hydra entry, `resolve_*_checkpoint`,
`load_*_model`, per-case `.npz` + manifest CSV + resolved-config dump).

**One design difference, and it matters:** gates uses a single tumour-centred
crop, which is fine for a picture but useless for a per-case score. Case-level
detection needs whole-volume coverage. Solution that reuses existing machinery:
a thin `nn.Module` wrapper whose `forward(x)` returns **one chosen level's**
ambiguity, trilinearly upsampled to input resolution. That returns exactly one
output, so MONAI's inferer can stitch it — feed it straight to `build_inferer` /
`sliding_window_predict`. No new stitching code.

Outputs: per-case `.npz` (voxel-wise disagreement + both entropies) and
`ambiguity_summary.csv` — **shaped exactly like `uncertainty_summary.csv`,
indexed by `case_id`**. This is deliberate: `build_risk_coverage`
(`scripts/calibrate.py:900-1015`) reads its column names from
`cfg.calibration.risk_coverage.uncertainty_column` and has **zero
MC-dropout-specific logic**, so pointing it at a disagreement column makes the
whole risk-coverage / referral-table / correlation stack work with no refactor.

> **Known landmine:** two divergent conventions already exist for reducing a
> voxel map to a per-case scalar —
> `uncertainty/risk_coverage.py:case_uncertainty_scalars` (union mask, `unc_*`
> columns) is fully tested but **called by nothing**, while
> `scripts/evaluate.py:640-669` uses its own inline reducer (predicted-foreground
> mask, `mi_*` columns). Use the `evaluate.py` convention for consistency with
> existing files and say so in the docstring. Do not introduce a third.

### 1.3 — The two tests, in order

**Test A — is it flat?** Extract on 10 cases spanning all three cohorts and look
at the maps. There is a real mechanical reason to fear flatness: the
branch-supervision loss trains both probes toward the *same* label, which may
have taught the branches to agree everywhere.

**Test B — does it beat what we already have?** *This is the test that actually
decides the project.*

The model already emits a confidence value per voxel. Low confidence — mostly at
the tumour edge — is an "unsure" map available for free. **So does the plain
U-Net.** If disagreement merely lights up where confidence is already low, it is
a duplicate of something the baseline can also produce, and the "only a dual
encoder can do this" claim dies regardless of how pretty the map is.

The bar is **incremental**, not absolute:

- **Voxel level.** AUROC for predicting per-voxel error, disagreement vs
  single-pass predictive entropy. Report both, plus the AUROC of
  entropy-**residualised** disagreement. Entropy is computable directly from the
  already-saved `logits/` — no model run needed for the baseline.
- **Case level.** Spearman(score, per-case Dice) for each, plus **partial**
  Spearman for disagreement controlling for entropy.

Both use `src/neurovision/analysis/statistics.py` (`paired_bootstrap_ci`,
`holm_bonferroni`) so every number arrives with a CI and a correction.

### Gate 1 — decision rule, fixed now

| Outcome | Rule | Consequence |
|---|---|---|
| **Pass** | Partial Spearman \|ρ\| ≥ 0.20 with a CI excluding zero, **and** residualised voxel AUROC ≥ 0.60 | Proceed to Phase 2. The pivot is real |
| **Partial** | Adds signal over entropy but weakly (CI excludes zero, magnitude below threshold) | Proceed, but reframe the claim as *efficiency* — equal detection at 1 pass instead of 10 — not superiority |
| **Fail** | Flat, or no incremental signal over entropy | **Stop the uncertainty line.** Fall back to §Fallback. Cost: two days |

**Write these thresholds into `docs/research/preregistration_ambiguity.md` and
commit before running the test.** Ten minutes of work; it is the difference
between "we fixed the criteria in advance" being a claim and being a fact with a
git timestamp.

---

## Phase 2 — The referral system (Week 2, zero GPU)

**Fixes:** turns a correlation into a deployable artifact, and produces the demo
centrepiece.

| # | Step | Component |
|---|---|---|
| 2.1 | Case-level failure detection: AUROC for "Dice < 0.7", per cohort — BraTS test (n=189), SSA (n=60), PED (n=99) | `analysis/statistics.py` |
| 2.2 | Risk-coverage under shift: point `calibration.risk_coverage.uncertainty_column` at the disagreement column, run on all three cohorts. `risk_coverage.csv` already saves the oracle ceiling and random null alongside — a curve hugging random is a real, reportable negative | `scripts/calibrate.py`, `configs/calibration/default.yaml` |
| 2.3 | Referral table: refer the least-trusted *k*% and measure the remainder. Compare against entropy, MC-dropout MI (both already computed), and random | `uncertainty/risk_coverage.py:referral_table` |
| 2.4 | **Inference-ROI sweep** — the free 96³ experiment. Re-evaluate existing 64³ checkpoints at ROI 64³ / 96³ / 128³, measure multifocality over-reporting at each | `scripts/evaluate.py` + `scripts/burden.py` |
| 2.5 | Demo overlay: new uncertainty layer labelled **"branch disagreement · representational"**, never "epistemic" or "MC-dropout"; plus a per-case trust banner | `app/frontend/`, `app/backend/api.py` |

**Statistical correction built in from the start.** "Disagreement matches
MC-dropout at 1/10 the cost" is an **equivalence** claim. Testing it as "no
significant difference" and getting a non-significant result proves nothing —
that is precisely the underpowered-null this project has hit six times.
Pre-specify a **TOST with a stated margin**: *disagreement AUROC is within 0.03
of MC-dropout's, two one-sided tests, α = 0.05*. Choose the margin before
looking at any number.

**Power reality check, also fixed now.** PED is n=99, SSA n=60. A 95% CI on an
AUROC at those sizes is roughly ±0.10. State targets as **CI bounds, never point
estimates**: *"AUROC CI lower bound excludes 0.70"*, not *"AUROC ≥ 0.80"*.

### Gate 2

Referral on disagreement beats entropy **and** MC-dropout MI on SSA/PED
specifically (CI on the paired difference excludes zero). Pass → failure
detection is the paper's headline. Fail → the claim narrows to "free and equal to
MC-dropout at 1/10 the cost", still publishable at UNSURE.

---

## Phase 3 — GPU: the two essential run sets (Weeks 2–4)

**Launch the moment hardware lands — independent of Gates 1 and 2.** Both run
sets are valuable whichever way the pivot goes, so they must not wait on it.

Costs in **T4-equivalent hours** (measured, `improvement_plan.md` §Phase C). On a
modern 24 GB+ card expect roughly 3–4× faster wall-clock.

### 3.1 — Multi-seed (C1, ~70 T4-h) · highest-value single item

Two additional seeds each for `neurovision`, `baseline_unet3d`,
`capacity_control_unet3d`, at 64³/80 epochs.

**Fixes the project's most exploitable weakness:** every number currently comes
from one training run, so no margin can be defended as larger than run-to-run
noise. A reviewer asks "does +0.0211 survive a different seed?" and today the
answer is unanswerable. After this, every number is a mean with a spread.

Two free by-products: it **restores the lost capacity-control checkpoint**, and
the three seeds form a **deep ensemble** — the strongest possible comparator for
Phase 2, at no extra training cost.

*Reporting honestly:* three seeds gives 2 degrees of freedom on the variance.
Report mean ± range descriptively and keep the paired per-case test **within**
each seed. Do not state a seed-level pass/fail threshold — the design cannot
support one.

### 3.2 — Ablation ladder (C2, ~110 T4-h) · the contribution test

Five parameter-matched rungs, configs already written in `configs/experiment/`:

`ablation_cnn_only` → `ablation_fusion_add` → `ablation_fusion_concat` →
`ablation_content_only_gate` → `neurovision`

**Fixes the single largest hole in the project.** `contribution.md` records
**P2 = NOT RUN** — "the load-bearing experiment, and its absence is the single
largest hole in the contribution." Right now "the ambiguity conditioning is what
carries the gain" is an assertion, not a result. A reviewer kills the paper on
that line.

`ablation_content_only_gate` is parameter-matched to `neurovision` within
**0.018%** (6,360 params), so any difference cannot be attributed to capacity. It
also **double-serves Phase 2**: it has no disagreement signal by construction,
making it the exact control for the uncertainty claim.

**Pre-run checklist (non-negotiable):** pin `GIT_REF` to a SHA and verify the
fp16-entropy fix is in the pinned tree; `grad_clip_norm: 5.0` identical across
every run in a comparison family (at 1.0, 66–70% of steps clipped, silently
changing the effective learning rate and confounding the ablation);
`scripts/smoke_test.py` green; fetch and keep the log.

---

## Phase 4 — GPU: closing the named gaps (Weeks 4–5)

Ranked by value, so the tail can be cut without damaging anything.

| # | Run set | Cost | Fixes |
|---|---|---|---|
| 4.1 | **Pooled multi-cohort** (BraTS+SSA+PED, 1 seed) | ~40 T4-h | The OOD failure directly. Also the strongest test of the pivot: *does the failure detector still work once the obvious fix has been applied?* A detector that survives that is far stronger than one that only works on an under-trained model |
| 4.2 | **Matched 96³ pair** (`neurovision` + `baseline_unet3d`, 1 seed) | ~80 T4-h | The patch-size question, done so it is interpretable. Priority set by Phase 2.4: if multifocality is an inference-window effect, demote; if a training-patch effect, promote |
| 4.3 | `baseline_swinunetr` @64³ | ~50 T4-h | No transformer baseline exists on our splits. Published SwinUNETR BraTS numbers are not a substitute — they are on the official validation set, ours is a random split of training |

**Write the answer to 4.1's success case in advance.** If pooled training closes
the shift gap, a reviewer asks why anyone needs a failure detector rather than
more diverse training data. The answer: no training set covers every deployment
shift, so detection and coverage are complements, not substitutes. Put that in
the paper rather than discovering it in review.

**Zero-GPU wins to run alongside** (`improvement_plan.md` §D, all committed but
never measured): flip TTA (`a800bfb`, expected +0.003–0.008 Dice); postprocess
re-ablation from saved logits; scoring the confidence head, which was trained at
weight 0.05 and has never been evaluated at all.

---

## Phase 5 — Write it (Weeks 5–6)

`notebooks/09_paper_figures.ipynb` regenerates every figure and table from result
files in one run and prints an audit of what was skipped and why. Point its
`RUNS` manifest at the new result directories; the figures follow.

**Paper structure — the negatives lead, they do not hide:**

1. A dual-encoder fusion architecture conditioned on inter-branch disagreement
   gives **+0.0211 ET Dice over a parameter-matched capacity control**, now with
   error bars across three seeds, decomposed 79% architecture / 21% capacity.
2. We test, **pre-registered**, whether that gain propagates to calibration,
   boundary accuracy, uncertainty-based referral, structured anatomical
   reporting, and out-of-distribution transfer. **It propagates to none of them**
   — and on external cohorts it **reverses** (`dice_TC` −0.0333, p_holm 0.0132).
3. We characterise why: gains of this magnitude are boundary-local, while every
   downstream quantity is volume-dominated, structure-dominated, or
   shift-dominated.
4. The architecture nonetheless yields a signal a single-encoder model cannot
   compute, and we show it detects its own out-of-distribution failures at 1/10
   the cost of MC-dropout.

Point 2 is the part almost nobody publishes, and the receipts exist. "Our model
is +0.4 Dice, therefore better for clinical use" is the standard unstated claim
in this literature; this project can show it is unsupported.

Plus three methodology findings already measured and worth a section of their
own: a calibration mask must never be defined using the ground-truth label (the
original inflated reported ECE by **41–57%**); boundary-error shares must be
weighted by voxel count, not rate (correcting this project's own figure from
**92% to 74%**); and the standard brain-mask-Dice check for a mirrored atlas
scores **higher** on the mirrored version (0.9416 vs 0.9394), making it blind to
the worst error it exists to catch.

**Venues:** MIDL short paper, MICCAI **UNSURE** workshop, MELBA — all of which
accept negative and replication results explicitly.

---

## Schedule

| Week | CPU track (always runs) | GPU track (starts when hardware lands) |
|---|---|---|
| 1 | Phase 0 housekeeping · Phase 1 · **Gate 1** | 2-epoch timing probe |
| 2 | Phase 2 · **Gate 2** | 3.1 multi-seed launches |
| 3 | Tier-4 deletions · figure regeneration | 3.1 completes → 3.2 ablation ladder |
| 4 | Analysis of 3.1/3.2 · zero-GPU wins | 3.2 completes → 4.1 pooled |
| 5 | Paper draft · all tables | 4.2 96³ pair (if warranted) → 4.3 if time |
| 6 | Paper finalisation · demo rehearsal | buffer for failed runs |

**Cut order if time runs short:** 4.3 first, then 4.2, then 4.1. Phases 1–3 are
the project; Phase 4 is completeness.

---

## Fallback — if Gate 1 fails

The project is still complete, and this is worth stating plainly so the gate can
be failed honestly rather than argued around:

- a controlled architecture result with a parameter-matched capacity control,
  **with error bars** from Phase 3.1
- a measured five-rung ablation ladder from Phase 3.2
- **six** rigorously obtained, pre-registered negative results, including an
  out-of-distribution sign reversal
- three methodology findings, each of which changes how a metric should be
  computed
- a working, demonstrable system

That is a genuine and honest paper for MELBA or UNSURE without a single further
GPU hour beyond Phase 3. Every phase above raises the ceiling; **none is
load-bearing for the floor.**

---

## Verification

| Level | Check |
|---|---|
| Unit | `pytest` from repo root — 1,373 tests, ~25 s, CPU. Note `pyproject.toml` already sets `addopts = "-q"`; adding another `-q` stacks to `-qq` and drops the pass count. Run plain `pytest` |
| Frontend | `cd app/frontend && npm test` (76 vitest) · `npm run test:e2e` (46 checks against a live backend, asserting on rendered pixels) before any demo |
| Integration | `python scripts/smoke_test.py` — end-to-end CPU gate, ~4 s, exit 0/1. Run before every GPU session |
| New code | Every new model method ships a CPU shape test on `(1, 4, 32, 32, 32)` running under one second |
| Analysis | **An analysis fix is not verified by its unit tests — it is verified by re-running the real analysis and checking the output moved in the predicted direction.** This project has already shipped a commit whose message, CLAUDE.md entry and 1,000 green tests all claimed a circular-mask bug was fixed while every reported number stayed circular, because the tests covered the new helpers and not the call sites |
| Disk | `df -h` after each deletion tier |
| Reproducibility | Every GPU run: pinned SHA in the log, checkpoint copied off the box, log retained, W&B run online |

---

## Immediate next actions

1. Phase 0.1–0.2 — manifests, then Tier 1–3 deletions (~77 GB freed)
2. Write and commit `docs/research/preregistration_ambiguity.md` with Gate 1's
   thresholds
3. Spec and delegate the `forward_with_ambiguity` model change
4. Spec and delegate `scripts/extract_ambiguity.py`
5. Run Test A on 10 cases and **look at the maps**

Steps 3–5 are the whole bet, and they cost two days of CPU time.
