# Project history — NeuroVision-X, from the first commit to today

**Written 2026-09-27.** Covers the first commit (2026-07-31) to commit `fb8b1b4` (2026-09-27):
396 commits, 56 numbered result notes, 11 pre-registrations, about 170 Kaggle T4 GPU-hours.

**What this file is for.** It is the project's memory in time order: what was built, every GPU
run, every result, every claim that died, and every major decision with the reason behind it. Use
it to get your bearings again, and to write the story arc of the paper (introduction, the
"how we got here" part of the discussion, limitations).

**What it is not.** It is not the authority on any number. Every number here is copied from
`experiments.md` (the note numbers are given) or `claims_and_evidence.md` (the C-numbers). If this
file and those two ever disagree, **those two win**, and this file should be fixed. The live queue
is `docs/research/master_plan.md` §4.3; where the project stands today is `status_2026-09-27.md`.

**Keep it current.** When a gate fires or a big decision is taken, add a line to §3 (the
timeline), §5 (the scoreboard) or §6 (the decision log). One line each, with the date and the note.

---

## Contents

1. [The whole story on one page](#1-the-whole-story-on-one-page)
2. [Timeline at a glance](#2-timeline-at-a-glance)
3. [The detailed chronology](#3-the-detailed-chronology)
4. [Every GPU run](#4-every-gpu-run)
5. [The hypothesis scoreboard — what we believed, what happened](#5-the-hypothesis-scoreboard)
6. [Major decisions log](#6-major-decisions-log)
7. [Incidents and traps, in the order they happened](#7-incidents-and-traps-in-the-order-they-happened)
8. [How the codebase and data grew](#8-how-the-codebase-and-data-grew)
9. [Where it stands on 2026-09-27, and what is open](#9-where-it-stands-on-2026-09-27)
10. [Using this history in the paper](#10-using-this-history-in-the-paper)

---

## 1. The whole story on one page

NeuroVision-X began on 2026-07-31 as a 3D brain-tumour segmentation project on BraTS 2021
multi-modal MRI (T1, T1CE, T2, FLAIR). The model is a **dual encoder** — a 3D CNN and a Swin
Transformer — joined by a **gated cross-attention fusion** block at four scales, a U-Net decoder,
and three heads (segmentation, confidence, boundary). It is a college course project (semester demo
and report, late November 2026), then a capstone (summer 2027), and a journal paper (MELBA).

The project has had **three identities**. Each one was replaced because the evidence said so.

| # | Identity | When | What happened to it |
|---|---|---|---|
| 1 | **"A disagreement-conditioned fusion gate makes a better, more reliable segmenter."** The gate reads how much the CNN and Swin branches disagree at each voxel. The promised benefit was better calibration and boundaries, not raw Dice | 2026-08-04 → 08-16 | The model **was more accurate** (ET Dice +0.0267, note 12), and ~79% of that is architecture, not width (note 19). But calibration, boundary accuracy and MC-dropout risk-coverage all came back **within noise** (notes 13, 16, 17). Accurate, not reliable. Later the P2 ablation showed the disagreement conditioning itself adds **nothing measurable** (note 38) — the founding hypothesis is a pre-registered null |
| 2 | **"An interpretable pipeline"** (scan → atlas-grounded structured report), then **"disagreement is a free failure detector"** | 2026-08-16 → 08-23 | The report is **stable** whatever the segmenter (1 of 25 metrics differ, note 23), so a better model does not make a better report. Disagreement says *where* a mask is wrong (Gate 1, note 35) but does not improve a working detector (Gate 2, note 37) and is worse than free entropy (note 39). The accuracy gain **does not transfer** to BraTS-Africa or paediatric data (notes 26, 30) |
| 3 | **"A segmentation model wrapped in a distribution-free error bound and a refusal gate — measured where it breaks"** | 2026-08-23 → now | The bound holds in distribution and **breaks under shift in proportion to the shift** (note 42). The refusal gate cannot see cohort-level shift (note 47). A new site can restore the bound with ~10–40 labelled cases, except paediatric tumour core (note 52). On real hospital DICOM the front end loses a lot (note 55). This is the current thesis |

**The one clean positive:** ET Dice **+0.0267** over a matched U-Net (p_holm 1.4e-21, n=189),
replicated at a second seed pair (+0.0247, note 54), ~13x the seed noise floor (note 49). It has
**not yet** been tested against nnU-Net (Gate A is training on Kaggle now).

**The current thesis** (from `CLAUDE.md`, rewritten 2026-09-26):

> A tumour segmentation model wrapped in a distribution-free error bound and a refusal gate: the
> bound holds in distribution, fails under distribution shift in proportion to the shift, and the
> gate cannot see cohort-level shift. We measure where it breaks, and what a new site needs in order
> to restore it.

---

## 2. Timeline at a glance

| Dates | Phase | Headline |
|---|---|---|
| 07-31 → 08-02 | **Milestone 1a — foundations** | Repo, Hydra config, preprocessing, resumable trainer, Kaggle driver. First baseline U-Net trained (200 epochs, 96³) |
| 08-02 → 08-06 | **Milestone 1b — the model and its claim** | Dual encoder, fusion gate, heads, MC-dropout, explainability, statistics, figures. Contribution claim written (08-04). Gate made to actually read disagreement (08-06) |
| 08-06 | **The calibration trap** | First calibration run found the reporting mask used the label: 41–57% of ECE was an artifact |
| 08-06 → 08-09 | **Budget and probes** | 60 GPU-h budget. Timing probe: model 10.7x the U-Net per epoch → re-plan to **64³ patches, 80 epochs** |
| 08-09 → 08-16 | **The NaN run, then the real run** | Session 1 trained on NaN for 10.5 GPU-h (fp16 entropy bug). Fixed, re-run healthy in 3 sessions, 23.1 GPU-h |
| 08-14 | **Demo app** | FastAPI + React slice viewer |
| 08-16 → 08-17 | **Milestone 1 results** | Accurate, not reliable. Capacity control: gain is ~79% architecture |
| 08-16 → 08-19 | **Milestone 2 — interpretable pipeline** | Atlas, localisation, burden, involvement, reports, report agreement. Phase 5 negative: better Dice ≠ better report |
| 08-19 | **External validation** | BraTS-Africa and BraTS-PED: the gain does not transfer; TC worse under shift. Temperature scaling fails under shift |
| 08-19 → 08-23 | **Milestone 3 — disagreement as failure detector** | Gate 1 PARTIAL, P1 answered (gate organised by anatomy), Gate 2 negative, P2 null, "1/10 the cost" refuted |
| 08-23 | **Milestone 4 master plan** | Pivot to a safety pipeline: conformal bound, QC model, refusal gate, clinical front end, error budget |
| 08-23 → 08-24 | **Track 1: A and B** | Lesion-wise metrics (voxel Dice overstated by 0.10–0.32), conformal risk control (holds in distribution, breaks under shift), confidence head scored |
| 08-24 → 08-27 | **Phase C and Phase E** | QC model (Gate C mixed positive, grows optimistic under shift). DICOM-in clinical pipeline E1–E7 at `/clinical` |
| 08-27 → 09-15 | **D0 and nnU-Net prep** | Heavy-augmentation run launched; nnU-Net export and timing probe. GPU track parked 09-15 |
| 09-18 → 09-19 | **Tool completion T0–T6, demo** | First real DICOM studies, 3D twin, atlas shells, molecular panel, export. Demo given 09-19 |
| 09-18 → 09-24 | **Error budget, D0, D1** | End-to-end usable 85/78/24%, silent failure 4/18/50%. D0 null. D1 seed noise measured |
| 09-24 | **Outside-view review** | Three logical breaks found; thesis must be rewritten |
| 09-26 → 09-27 | **Milestone 5** | Thesis rewritten. Local recalibration, real-DICOM validation, intended-use scoping, second U-Net seed, OOD score. Gate A on Kaggle |

---

## 3. The detailed chronology

Dates are 2026. "Note N" is `experiments.md` note N. Commit hashes are short SHAs.

### 3.1 Milestone 1a — foundations (07-31 → 08-02)

**07-31.** Repository initialised (`4ad0bc2`): `src/neurovision/` package layout, Hydra configs,
logging and I/O utilities. The hard constraints were set on day one and have held since: runs on
any single ≥16 GB CUDA card and survives a kill at any point; no hardcoded paths; no CUDA-only
assumptions; full resume; AMP on for CUDA; default patch 96³; no new dependency without asking;
every model component has a CPU shape test under one second.

**08-01.** A very dense day:
- Preprocessing: nonzero z-score, crop to brain bounding box, label remap, float16 image / uint8
  label `.npy` plus `meta.json`. All **1,251** BraTS 2021 training cases preprocessed (34.22 GB,
  0 failed). Splits frozen, seeded 70/15/15: **875 train / 187 val / 189 test** (`configs/data/splits.yaml`).
  All later numbers are on this random split of the *training* set, which matters later (§5, C12).
- The preprocessed data was uploaded as the Kaggle dataset `amishyadav123/neurovision-brats-prep`.
  After raw data was deleted on 08-19, this dataset became the only backup of `data/preprocessed/brats`.
- Model/loss registries; MONAI U-Net (12.87M) and SwinUNETR-B (62.19M) baselines.
- Trainer with AMP, gradient accumulation, clipping, warmup-cosine schedule, sliding-window
  validation, atomic checkpoints, full resume (model, optimizer, scheduler, scaler, epoch, step, RNG,
  W&B id) with 7 resume tests. `scripts/smoke_test.py` (a 4-second end-to-end CPU gate).
- `notebooks/kaggle_train.ipynb`: the thin Kaggle driver.
- Kaggle lessons learned the hard way: the **P100 is unusable** (stock PyTorch targets sm_70+);
  datasets mount one level deeper than documented; MONAI HD95 on CUDA tensors routes through CuPy and
  fails to compile on Kaggle, so HD95 moved to CPU (`9c39b1b`).
- **First training run:** `baseline_unet3d`, 200 epochs at 96³, two Kaggle sessions (10.93 h +
  5.54 h = 16.5 GPU-h). Test Dice ET 0.8587 / TC 0.9157 / WT 0.9354. **No commit SHA was recorded**
  (the notebook cloned `main`), so its exact source is unrecoverable. This row was later superseded
  (note 1) and is **not** the comparison baseline.

**08-02.** RNG restore on CUDA fixed (`96a1bba`: `.cpu()` before `.numpy()`). Shared
`_baseline_common.yaml` written so every arm inherits one schedule. Kaggle evaluation notebook.

### 3.2 Milestone 1b — the model and its claim (08-02 → 08-06)

**08-04.** `contribution.md` written — **the founding pre-registration**. The claim: prior fusion
gates derive their weight from branch content; ours reads an explicit local ambiguity signal (the
per-voxel disagreement between the CNN branch's and Swin branch's own region predictions, plus each
branch's entropy). Predictions P1–P5, including **P1** (the gate opens where local evidence is
ambiguous, e.g. the tumour margin) and **P2** (a content-only gate, rung 2 of an ablation ladder,
must lose; if it ties, report the null). The predicted benefit: **better calibration and boundary
accuracy, not a uniform Dice gain.** This prediction later turned out backwards.

**08-04 → 08-06.** The full model was built module by module: `CNNEncoder` (with
`zero_init_residual`), Swin encoder, `AdaptiveGatedFusion` (windowed cross-attention, gate
generator, merge `cnn + layer_scale * gate * attn`), U-Net decoder, segmentation head, confidence
and boundary heads, `MultiTaskLoss`, MC-dropout inference, integrated gradients / Grad-CAM,
faithfulness, statistics (`paired_bootstrap_ci`, Wilcoxon, Holm, `compare_models`), the figure and
table toolkit, boundary-stratified metrics, `scripts/calibrate.py`, `scripts/explain.py`,
`scripts/extract_gates.py`, and the paper-figure notebook.

**08-06 — the claim and the code disagreed.** Between 08-04 and 08-06, the document said the gate
reads disagreement, while `GateGenerator` only saw `cat([cnn_feat, swin_proj])` — a content-only
gate, i.e. the declared null of P2. `BranchAmbiguity` was added (`61bb949`, `acfcf1f`): two
1×1×1 probes read **detached** branch features, and the gate gets `[features, disagreement,
H(p_cnn), H(p_swin)]`. The P2 ablation (`use_ambiguity: false`) is parameter-matched to 0.018%.

### 3.3 The calibration trap (08-06)

The first real calibration run (baseline U-Net at epoch 130, 189 test cases) looked wrong: every
whole-tumour reliability bin below 0.47 had a mean label of exactly 1.000. Cause: the reporting
mask was `(p >= threshold) | (label > 0)` — it **used the ground truth to pick voxels**, so it
manufactured **41–57% of the reported ECE**. All 984 tests passed. The first "fix" (`63a437b`)
added label-free masks but never changed the call sites, so every number stayed circular behind a
green suite and a commit message saying it was fixed. After the real fix, the honest baseline ECE
was 0.0565 uncalibrated and **0.0158 after temperature scaling**. Two rules came from this and are
now in `CLAUDE.md`: a calibration mask never touches the label, and **an analysis fix is verified by
re-running the real analysis**, not by unit tests. Also decided: any calibration claim must beat a
*temperature-scaled* baseline.

### 3.4 Budget, probes and the re-plan (08-06 → 08-09)

**08-06 — the budget.** 60 GPU-h, two weeks, Kaggle free tier (~30 h/week), single seed. Ranking
decision: **the contribution ablation (P2) outranks baseline breadth.** So `baseline_swinunetr`
(~25 h), the 6-row architecture grid, P2 rung 1, and every extra seed were cut. The paper must
therefore say the transformer baseline is absent for compute reasons.

**08-08 — the timing probe fired the abort trigger.** Five probe versions (all under 0.15 GPU-h each):
- v1: CUDA OOM on the first step. A T4 really has **14.56 GiB**, not 16.
- v2: gradient checkpointing on; OOM moved to backward (allocator fragmentation).
- v3: completed. **3.6 s/step = 0.875 h/epoch** at 96³, i.e. ~91 h for 100 epochs, needed twice —
  more than 3x the whole budget. The model is **10.7x the U-Net per epoch**, at 93% of the card.
- v4: at 64³, no checkpointing: **1.12 s/step**, 6.17 GiB. The first estimate that landed on target.
- v5: gradient norms logged — at `grad_clip_norm: 1.0`, 66–70% of steps were clipped, i.e. the
  model trained at an effective learning rate the config did not describe, and the P2 arms would
  have clipped at different rates. Raised to **5.0**.

**The re-plan:** every arm at **64³ patches, 80 epochs**, `val_interval` 10, checkpointing off. A
per-module profile showed fusion is only **1.5%** of the forward pass (decoder 69%), so the novel
part was kept intact and the data schedule was cut instead. **GIT_REF is pinned to a commit SHA**
from now on (the 200-epoch run's lost provenance). The first pinned run died in the clone cell:
`git clone -b` does not accept a SHA (`4960604`).

**08-08/09 — the comparison baseline.** `baseline_unet3d` 80ep/64³, one session, **~3.2 GPU-h**,
pinned `6ee28a7`. Test Dice **ET 0.8442 / TC 0.9058 / WT 0.9276**. The 64³/80-epoch cut costs about
1 Dice point versus the 96³/200-epoch run, identically for every arm (note 7). Also corrected: the
often-quoted "92% of error lies within 2 mm" was computed from rates, not voxel counts; the real WT
figure is **~74%** (`2407466`).

### 3.5 The NaN run, then the real run (08-09 → 08-16)

**08-09 — 10.5 GPU-hours on NaN.** `neurovision` session 1 (pinned `92f404b`) ran perfectly by
every operational signal and stopped cleanly at epoch 38 — but the loss had been `nan` since about
epoch 10–19 and `best.pt` was frozen at epoch 9. Cause: `BranchAmbiguity` computed entropy from
probabilities with `p.clamp(1e-6, 1-1e-6)`. In fp16, `1 - 1e-6` rounds to exactly 1.0, so once a
probe saturated, `0 * log(0)` = NaN. Fix: entropy from **logits** via softplus (`9a21c1b`).
Why no probe caught it: v4/v5 ran 3 epochs on 50 cases and never saturated. A new
**saturation probe** (20 epochs at 10x learning rate, 0.35 GPU-h) was built to reach the failure
condition and proved the fix. Lesson: *a probe must be built to reach the failure condition, not
merely to run.* Note that the P2 ablation has no `BranchAmbiguity`, so had this shipped, P2 would
have compared a NaN run against a healthy one.

**08-09 → 08-16 — run 2, healthy.** Pinned `7caacfa`, three chained sessions: 10.34 h (epochs
0–35), 10.5 h (36–72), 2.3 h (73–79) = **23.1 GPU-h**, 80/80 epochs, W&B `cc2l5j1c`. `best.pt` at
**epoch 69** (val dice_mean 0.8938) lives in session 2's output. Session 3 was marked ERROR only
because its verification cell demanded a `best.pt` that legitimately did not exist (fixed
`8045f49`).

**08-14 — the demo app.** FastAPI backend + Vite/React/TS/Tailwind frontend, a canvas slice viewer
over precomputed outputs, separate dependencies from training. Five viewer lessons (head on its
side, entropy invisible under alpha-by-entropy, a 502 behind the dev proxy, `"val"` inside
`"eval"`).

### 3.6 Milestone 1 results — accurate, not reliable (08-16 → 08-17)

**08-16.** Evaluated on a T4 (11.9 s/case — on the Mac it measured 136 s/case, 21x the U-Net).
Paired over the same 189 test cases, Holm across six metrics (note 12):

| | neurovision | baseline_unet3d (80ep/64³) | Δ | verdict |
|---|---|---|---|---|
| Dice ET | 0.8709 | 0.8442 | **+0.0267** | **better**, p_holm 1.4e-21 |
| Dice TC | 0.9161 | 0.9058 | **+0.0103** | **better** |
| Dice WT | 0.9321 | 0.9276 | +0.0045 | inconclusive |
| HD95 (all three) | | | | inconclusive |

The same day, all three reliability routes were measured and **none** held:
- **Boundary accuracy** (note 13): fewer errors at every distance, in proportion. The share in the
  0–2 mm band is unchanged. "Fewer errors overall", not "better at the margin".
- **Calibration** (note 16): the model needs about the same temperature as the baseline; mean ECE
  is inconclusive both uncalibrated and scaled.
- **MC-dropout risk-coverage** (note 17, N=10, 4.2 h on a T4): normalised AURC 37.6% vs 40.6%,
  within noise. A plain U-Net's MC-dropout already ranks its own failures about as well.

**Consequence:** the model is more **accurate**, not more **reliable**. The reliability framing was
dropped.

**08-17 — the capacity control** (`capacity_control_unet3d`, a U-Net widened to 34.83M params vs
34.91M, ~8 GPU-h, one session, pinned `f363067`). Test ET 0.8497. Against its pre-registered
outcomes it lands on the "architecture" branch (note 19):

| Contribution to +0.0267 ET | ET Dice | share |
|---|---|---|
| Width alone (capacity − baseline) | +0.0055 | 20.6% |
| Architecture (neurovision − capacity) | **+0.0211**, p_holm 7.3e-19 | 79.4% |

**Loss:** the capacity control's checkpoint was written to Kaggle `/tmp/` and never retrieved. It
has no checkpoint, logits or predictions, so **it can never be re-scored** under a new metric.

### 3.7 Milestone 2 — the interpretable pipeline (08-16 → 08-19)

**08-16 — the pivot.** `interpretable_pipeline_plan.md`: stop claiming trust, build what the
accuracy result enables — a scan-to-report pipeline. Revised the same day after one question: *what
happens without a neuroanatomy reviewer?* None is available, so the knowledge base had to be built
only from published sources, VASARI equivalence could not be claimed without the primary source,
and midline shift was **declined outright** (no MRI ground truth exists).

Built in three days, all on CPU for zero GPU hours:
- **Phase 0 — atlas gate** (passed 08-16). SRI24 lives on the BraTS 240×240×155 grid, but it is
  **front–back mirrored** relative to BraTS indexing, and one of its files is additionally
  left–right mirrored. Found and fixed per file from each file's own affine. Found also: brain-mask
  Dice scores *higher* on a left–right mirrored atlas (0.9416 vs 0.9394), so it cannot catch the
  worst error (trap 3). Details: `phase0_atlas_findings.md`.
- **Phase 1** localisation (122 merged structures; an explicit `unlabelled` row, ~31% of a tumour,
  because AAL labels grey matter only). **Phase 2** Tier-C eloquence map (23 structures, Sawaya).
  **Phase 3a** burden profile (57 columns). **Phase 3b** involvement (ventricles, deep white matter,
  tissue fractions, epicentre; deliberately not labelled VASARI). **Phase 4** report library, driver,
  demo API route and panel, with disclaimer and `not_claimed` as required fields. **Phase 5**
  report agreement and population statistics.

**Results, 08-16 → 08-19:**
- **Note 18:** the ET gain does reach the report (ET volume error, empty-ET detection), but report
  agreement is **not monotonic in Dice** — the older 96³ U-Net made a better report on some fields.
  All models over-report multifocality (30.7–40.7% vs a true 22.8%). A labelling trap:
  `outputs/eval_test` is the 96³ U-Net, not `neurovision`.
- **Note 23 — Phase 5 NEGATIVE:** 25 report metrics, 189 paired cases: **1 of 25** conclusive.
  16 of 25 medians identical across three models. The report is dominated by *which structures the
  tumour touches*, which a few margin voxels rarely change. The layer is **stable** — good for
  deployment, fatal to "better model, better report".
- **Note 24:** the eloquence layer is **degenerate**: 100% of BraTS gliomas are "near eloquent",
  98.8% at 0.0 mm.
- **Note 25:** population anatomy over 1,251 cases. A laterality bug had shown "midline 34.8%"
  (true 0.27%) by folding the `unlabelled` placeholder into midline.

### 3.8 External validation (08-19)

- **Support for external BraTS-format cohorts** (`204908f`). BraTS-Africa (SSA, 60 cases) ships
  BraTS 2023 labels and RAS axes (front–back *and* left–right reversed vs BraTS 2021); handled from
  each file's affine, verified bitwise-identical on existing cases. **Nothing is ever fitted on
  SSA or PED** — all cases sit in `test`.
- **Note 26 — SSA:** ET gain gone (−0.0008, inconclusive); 0 of 8 comparisons survive Holm.
  `neurovision` degrades more than the baseline (direction only). Uncorrected, ET would have read
  "significantly worse" — Holm returned it to inconclusive.
- **HD95-under-shift pre-registered** (`1e2d7de`) before BraTS-PED (99 cases) was touched, because
  SSA had hinted at better boundaries under shift. **Result: H1 NOT CONFIRMED** (`1e603d0`).
- **Note 30 — PED + pooled:** `neurovision` is **significantly worse on tumour core** under shift
  (PED `dice_TC` −0.0595, p_holm 0.0002; pooled −0.0333, p_holm 0.0132). The SSA boundary pattern
  was noise — the second time a promising direction survived only until n grew.
- **Note 29:** temperature scaling does not transfer: on SSA ET it makes calibration **4x worse**.
- **Note 27–28:** the ET gain is chiefly recovered enhancing tumour at the margin plus fewer far
  false positives; the model costs **23.7x the FLOPs** of the U-Net for it.
- Also: live inference and upload jobs in the demo, flip TTA module, logits replay (`706a81d`) so
  metrics can be recomputed without the model.
- **Disk:** raw data deleted after SHA-256 manifests were committed (SSA/PED only; raw BraTS 2021
  has no manifest). Volume-sized outputs declared **caches**, not results.

### 3.9 Milestone 3 — disagreement as a free failure detector (08-19 → 08-23)

**08-19 — the reframe** (`improvement_plan.md`, then `execution_plan.md`). The claim: a dual
encoder carries two independent readings of every voxel; where they disagree is a per-voxel
uncertainty that a single-encoder model cannot produce, free, matching MC-dropout at 1/10 the cost,
and more useful under shift. The execution plan fixed the process error behind the earlier nulls:
**every phase states its pass condition and an effect-size check before it runs.** Gate 1 was
pre-registered (`preregistration_ambiguity.md`).

- **Note 31 (08-20) — Gate 1 Test A:** the disagreement map is not flat and not a re-encoding of
  entropy (voxel Spearman 0.08–0.35). The Swin probe is systematically less sure than the CNN probe.
- **Note 32 (08-20) — P1 answered, but not as predicted.** The fusion gate is strongly organised by
  anatomy (level 1: 0.98 inside the tumour → 0.33 outside), with **opposite polarity at adjacent
  scales**. It does **not** peak at the margin. Do not write "the gate opens at the boundary".
- **Note 33 (08-20):** forensics — small-component removal merges channels, and scikit-image's
  `min_size` is already inclusive. Impact negligible; **documented, not changed**, to keep
  comparability with the unrecoverable capacity control.
- **Note 34 (08-20), exploratory:** a gate read-out predicts case Dice on PED, where entropy fails.
- **08-22:** local extraction must run **serially**, one process at a time — parallel shards
  exhausted memory twice. `sw_batch_size: 4` is a memory multiplier on CPU.
- **P2 ablation trained on Kaggle** (08-22 → 08-23): 24.16 GPU-h over three sessions, pinned
  `7caacfa`, W&B `ddkbitjp`.
- **Note 35 (08-23) — Gate 1 PARTIAL:** disagreement localises error per voxel beyond entropy on
  all three cohorts (residualised AUROC 0.578 / 0.569 / 0.677), but **does not rank cases** outside
  distribution. Consequence: Gate 2 was **respecified around voxel-level localisation before it
  ran** (`preregistration_gate2.md`).
- **Note 36:** a bigger inference window (96³) does not fix multifocality and costs WT Dice.
- **Note 37 — Gate 2: PARTIAL by rule, negative in substance.** Adding disagreement to entropy
  makes the operating detector worse or unchanged everywhere. *Incremental information is not
  incremental utility.*
- **Note 38 — P2: THE NULL.** The content-only gate matches `neurovision` on every metric
  (ET +0.0022, CI −0.0067 to +0.0152) while beating the baseline by +0.0244. **~92% of the
  architecture gain is the gated fusion itself; the disagreement conditioning — the claimed
  novelty — adds nothing measurable.** Gated cross-attention fusion is already published.
- **Note 39 — "1/10 the cost" REFUTED.** MC-dropout maps for SSA/PED on a T4 (3.3 GPU-h). Paired
  TOST at a margin of 0.03 fixed in advance: disagreement is **worse** than MC on both cohorts;
  **free single-pass entropy is equivalent to 10-sample MC-dropout.**
- **Note 40:** BraTS 2026 Challenge 3 (generalisability) closed on 31 July 2026. Target 2027.

### 3.10 Milestone 4 — the master plan (08-23)

`master_plan.md` was written after the three results above. Its diagnosis:
1. **Every dead claim was a downstream claim** — can a +0.02 Dice gain buy calibration, reports,
   robustness? A gain that size is a handful of voxels; each null was predictable by an effect-size
   argument that was never made first.
2. **The project had never faced the bar the field applies** — nnU-Net, lesion-wise metrics.

The new goal: *a pipeline that produces a segmentation with a statistically guaranteed error
bound, knows when to refuse, ingests real clinical MRI, and is validated end to end, with the
compounded error budget published.* Phases A (housekeeping, strong baseline), B (conformal), C (QC
model), D (generalisation fixes), E (clinical front end), F (IDH, gated), G (error budget), H
(write-up). A full cut list (§6 below). `claims_and_evidence.md` created as the gate on what may be
written. `CLAUDE.md` condensed, with the long records moved to `lessons.md` and `project_state.md`.
Gate A (`preregistration_strong_baseline.md`) and the conformal study were pre-registered.

### 3.11 Track 1: lesion-wise metrics, conformal, confidence head (08-23 → 08-24)

- **A1:** separate `requirements-analysis.txt` and `requirements-clinical.txt`; root untouched.
- **A2/A3 — note 41, lesion-wise re-scoring** of every saved run (1,070 cases, zero GPU).
  **Voxel Dice overstated performance by 0.10–0.32**; `neurovision` WT reads 0.9321 voxel-wise,
  **0.7183 lesion-wise**. In distribution the gain is **larger** lesion-wise (ET +0.0508, TC
  +0.0371); the mechanism is **fewer spurious lesions**. It still does not transfer. PED tumour
  core is a floor (0.234 for both models, 64 of 99 exact ties). Exploratory.
- **B1/B2 — note 42, conformal risk control.** Threshold fitted on val, applied frozen. **In
  distribution the bound holds, 6/6 cells, both models.** Under shift, **7 of 12 cells violated for
  each model**, ordered by the size of the shift: SSA WT ~1.1x nominal, PED WT 1.4–1.9x, **PED TC
  3.5–11.5x**. The cost of the guarantee is small, and at looser α negative (C15).
- **A4 — note 43:** flip TTA wired; its measurement is ~34 h on the Mac, so it moved to the GPU.
- **A5 — note 44:** the confidence head, never scored before, is **beaten by free entropy on all
  three regions** (at chance on WT). Entropy is now undefeated against three mechanisms (C17).

### 3.12 Phase C and Phase E (08-24 → 08-27)

**Phase C — the QC model.** A 3D CNN (`SegQC`) trained to predict a mask's own Dice from image +
mask, on synthetically degraded masks (six degradation kinds, blind to the label). Pre-registered
before its first training run. A bug where model selection read the test split was caught and fixed
(`2a881ec`). **Gate C result (08-26):** POSITIVE by the rule on **one cell of five** (PED·TC,
ΔAUROC +0.1686 over entropy), but significantly **worse** than entropy on SSA·TC and PED·WT. And
under shift its error became **more optimistic in all six external cells** (C19) — it grows
confident exactly where the segmentation is worst.

**Phase E — the clinical front end** (08-24 → 08-26), in `.venv-clinical`:
E1 DICOM study → four named NIfTIs · E2 co-registration, SRI24 registration, HD-BET skull
stripping · E3/E4 twelve label-free input-QC checks with named refusal reasons · E5 the gatekeeper
(PROCEED / CAUTION / REFUSE), thresholds calibrated on val, not invented · E6 DICOM-SEG export ·
E7 UI: 3D digital-twin viewer, landing page, `/clinical` page with refusal banner and gatekeeper
panel. Wired end to end in `clinical_jobs.py`; a `"refused"` job is a success state, not a failure.
`requirements-clinical.txt` was missing `hydra-core` (fixed). E6, entropy, conformal band and
Grad-CAM wired into live jobs 08-31 (`54c03a6`).

**08-26 — Phase F (IDH) decided "not yet".**

### 3.13 D0, nnU-Net preparation, GPU parked (08-27 → 09-15)

- **08-27:** D0 heavy augmentation pre-registered. **08-31:** four new transforms (rotation, gamma,
  bias field, elastic), off by default. D0 sessions 1 and 2 ran on Kaggle.
- **09-01:** G1 — our frozen split exported to nnU-Net v2 layout (only the 875 train cases).
- **09-02:** nnU-Net timing probe ran on Kaggle; its log sat unread until 09-18 because the
  download broke.
- **09-15 — GPU track parked** (author): finish the tool first. "A complete clinical tool with a
  slightly weaker model is worth more than an incomplete tool with a marginally better one." Second
  disk reclaim (83 GB; `nnunet_raw`, every MC uncertainty cache, re-downloaded raw data).

### 3.14 Tool completion T0–T6 and the demo (09-18 → 09-19)

**09-18**, one very long day (`tool_completion_plan.md`, board in `tool_completion_log.md`):
- **T0 — first real DICOM through the pipeline** (note 45). The RSNA fixture returned 403 (rules
  not accepted), so TCIA UPENN-GBM studies were used. **UPENN-GBM-00002: `done` / PROCEED in
  361 s** (job `9c2cc294`). **UPENN-GBM-00001: refused reproducibly** on predicted Dice (3 mm FLAIR).
  Seven things the green suite could not catch (F1–F7), e.g. pre-registration QC refused every real
  study (fixed with a `pre_registration` stage, not a threshold change); HD-BET's defaults took
  2 h 10 min and 16 GB on CPU (now `fast`, no TTA); **the shipped 3D twin was geometrically wrong
  twice over** (wrong strides, left-handed = mirrored); DICOM-SEG would have mis-registered.
- T0.5 job persistence · T1 twin on the clinical screen · T2 uncertainty / Grad-CAM painted on the
  twin · T3 atlas structure shells · T4 shape descriptors (optional `geometry` block) · T5 molecular
  panel (user-entered pathology + CNS5 lookup, no imaging claim) · T6 export (markdown, snapshot,
  zip) · a plain-language report page.
- **Note 46 — nnU-Net cost:** 1.087 s/iteration on a T4, so the default 1000-epoch fold is
  **~75.5 GPU-h**, above Gate A's 60 GPU-h abort bound, and cutting epochs was forbidden by the
  pre-registration. **Gate A not launched.** The GPU track was **unparked** the same day with quota
  rules (a run must be verified to finish in one go; no relaunching to "rehydrate"; a run that
  breaks its abort bound is not launched).
- Simplification review; D1 pre-registered; Phase G error-budget protocol fixed before any number.

**09-19 — the semester demo was presented** and went fine. The same day, **note 47 — the
end-to-end error budget** (Phase G), with the deployed frozen gate and nothing refitted:

| cohort | P(accepted AND usable) | silent failure (accepted but unusable) |
|---|---|---|
| BraTS test (189) | **0.852** | **4.2%** |
| SSA (60) | **0.783** | **18.3%** |
| PED (99) | **0.242** | **49.5%** |

In distribution the gate refuses 4 usable studies per unusable one caught. On SSA it is nearly
inert (2 refusals in 60; 11 of 12 unusable masks passed). On PED its precision is 0.88 but recall
0.32, and **no operating point** makes PED deployable. Every `conformal_band` refusal in
distribution was an over-refusal. The gate cannot see cohort-level shift (C21), and gating does not
restore a broken guarantee (C22).

### 3.15 D0 and D1 resolve (09-20 → 09-24)

- **09-20 — note 48, D0 NULL.** `neurovision_heavy_aug`, 23.7 GPU-h. Primary pooled SSA+PED
  `dice_TC` +0.0117, CI [−0.0006, +0.0245] — a near miss, recorded as one. **12 of 12 comparisons
  inconclusive.** Recipe unchanged. Also found: evaluation on the Mac is ~1.4 min/case, not the
  "15 cases/min" `CLAUDE.md` claimed.
- **09-20 → 09-24 — note 49, D1:** `neurovision_seed43`, three Kaggle sessions, pinned `9c770ce`.
  **12 of 12 inconclusive** against seed 42. The seed noise floor on test ET is **+0.0021** — ~13x
  smaller than the headline, so C1 stands. D0's effect is the same size as seed noise, which
  confirms D0's null.

### 3.16 The outside-view review and Milestone 5 (09-24 → 09-27)

**09-24 — `project_review_2026-09-24.md`.** Strong as research, not viable as a clinical product.
Tool ~90% done, experiments ~65%, paper ~10%. **Three logical breaks:** the thesis in `CLAUDE.md`
still claimed "safe to deploy on data it was not trained on"; the one positive had never faced a
strong baseline; the main finding's novelty had not been checked against 2026 preprints. Best next
step: the Mondrian recalibration arm, which was pre-registered and never run.

**09-26 — Milestone 5.** Thesis rewritten to the current one; one live queue (`master_plan.md`
§4.3, phases P0–P5 with dates).
- **Note 50 (P0.3):** both demo DICOM studies are **BraTS 2021 training patients**
  (00002 = `BraTS2021_01202`; 00001 ≈ `BraTS2021_01034`), found by image correlation. The PROCEED
  demo shows the pipeline *runs*, not that it generalises.
- **Note 51 (P0.4), post-hoc:** `conformal_band` made **display-only**. Test usable 0.852 →
  0.915; PED silent failure 49 → 53 of 99 (accepted by the author because P1.3 scopes PED out).
- P0.5 honest UI wording: the guarantee is *on average, in distribution, not per patient*.
- P0.6 dead code removed. P2.1 `init_from` for fine-tuning. P2.2 frozen cross-fit splits.
  P2.4 `gpu_session.py` launcher.
- **Note 52 (P1.1), counterfactual:** a new site can restore the conformal bound with its own
  labelled cases — **~10–40 of them** depending on α — for WT on SSA and PED and TC on SSA.
  **Paediatric tumour core cannot be restored at any k.** All four registered predictions held.
- Pre-registrations: Gate A Amendment 1 (full 1000-epoch nnU-Net, chained on Kaggle, fold all,
  95 GPU-h bound, Auto3DSeg dropped), D3 cross-fitted fine-tune, multiseed Amendment 1 (a second
  U-Net seed), real-DICOM validation protocol with a frozen 40-case sample.
- **GPU moved to Kaggle now** rather than waiting for the college card; cap 30 GPU-h/week.
  `baseline_unet3d_seed43` trained (3 AMP-overflow steps skipped by GradScaler; run valid). Gate A
  preprocessing ran as a CPU kernel; Gate A session 1 launched (209 s/epoch → ~58 GPU-h total).

**09-27**, another long day:
- **Note 53 (P1.3):** `intended_use` gate signal from DICOM `PatientAge` (adults only). PED
  accepted 0.798 → **0.000 by construction** — reported as **scoping, never detection**. Missing age
  → CAUTION, a documented exception. **Enabled in the live gate** after P1.2.
- **Note 54:** **C1 replicated across two seed pairs** — +0.0247 at seed 43, seed-averaged +0.0257.
  Baseline seed noise: 2 of 12 cells resolve (test ET +0.0041).
- **Note 55 (P1.2) — real DICOM, same patients.** 40 BraTS test-split patients through the real
  clinical path from their original RSNA-MICCAI DICOM. **40% refused for thick slices**, 2 by the
  gate, 22 accepted; usable | accepted **0.36**, silent failure **0.35**. Median WT front-end cost
  **−0.23** — the prediction of ≤0.05 **failed ~5x**. Post-hoc: mostly a median **5.9 mm**
  disagreement between our SRI24 registration and BraTS's own (undoing translation alone: WT 0.83).
  Not external validation.
- **Note 56 (P1.4) — pre-registered negative.** An input-statistics OOD score flags **60% of SSA**
  vs 9.5% of test, but ranks failed masks barely above chance (AUROC 0.56–0.60). It removes 3 of 11
  SSA silent failures (4 required), so it **stays display-only**. It detects the shifted *cohort*,
  not the failed *study* — the honest use is a site-level alarm (post-hoc reading).
- P2.3 nnU-Net importer and scorer (label 1↔2 swap, geometry gate, ground-truth round trip). P2.5b
  fresh-clone rehearsal: 2,456 tests pass. P2.6 GPU request drafted (~60 GPU-h, submit ~Nov 5).
  P3.1 semester report Results/Discussion/Limitations drafted. P3.2 demo runbook. P3.3
  `related_work.md`: C14 no longer says "nobody has measured" (arXiv 2606.20115 did, on FeTS
  sites). Third disk reclaim (~13 GiB); the REFUSE demo job was deleted in it and regenerated as
  `67b67b1d`. Paper sources gathered under `docs/research_docs/`.
- **Gate A session 2 done** (epochs 188–372, 21.9 GPU-h cumulative); projected ~61 GPU-h total.

---

## 4. Every GPU run

All on Kaggle Tesla T4 (14.56 GiB usable). "GPU-h" is session wall-clock. Checkpoint column says
whether the weights still exist — a run without a checkpoint can never be re-scored.

| # | Run | Dates | Sessions / pinned SHA | GPU-h | Outcome | Checkpoint |
|---|---|---|---|---|---|---|
| 1 | `baseline_unet3d` 200ep/96³ | 08-01 → 08-02 | 2 · **no SHA** (`main`) | 16.5 | Test ET 0.8587. Superseded (not the matched schedule); its predictions kept in `outputs/eval_test` | lost |
| 2 | `probe_neurovision` v1–v5 | 08-08 | 5 short · — | ~0.4 | OOM ×2, then 0.875 h/epoch at 96³ → re-plan to 64³; grad clip 1.0 → 5.0 | — |
| 3 | `baseline_unet3d` attempt 1 | 08-08 | 1 · `6ee28a7` | ~0.02 | Died in clone cell (`git clone -b <sha>`) | — |
| 4 | **`baseline_unet3d` 80ep/64³** | 08-08/09 | 1 · `6ee28a7` | ~3.2 | **The comparison row.** Test ET 0.8442 / TC 0.9058 / WT 0.9276 | ✓ |
| 5 | `neurovision` session 1 (attempt 1) | 08-09 | 1 · `92f404b` | **10.5 wasted** | Trained on NaN from ~epoch 10 (fp16 entropy) | discarded |
| 6 | `probe_saturation` | 08-09 | 1 · `7c1d5a0` | 0.35 | Proved the logit-entropy fix at the failure condition | — |
| 7 | **`neurovision` run 2** | 08-09 → 08-16 | 3 · `7caacfa` | **23.1** | Test ET 0.8709 / TC 0.9161 / WT 0.9321. Best epoch 69. **The deployed model** | ✓ |
| 8 | `neurovision` eval test/val | 08-16 | 2 | ~1.2 | 11.9 s/case on T4 (136 s/case on the Mac) | — |
| 9 | MC-dropout test, `neurovision` | 08-16 | 1 | 4.2 | Risk-coverage within noise (note 17). Baseline's MC ran on the Mac | — |
| 10 | `capacity_control_unet3d` | 08-16/17 | 1 · `f363067` | ~8 | ET 0.8497 → gain is 79% architecture. Log not kept, so hours approximate | **lost** (Kaggle `/tmp/`) |
| 11 | **`ablation_content_only_gate`** (P2) | 08-22 → 08-23 | 3 · `7caacfa` | **24.16** | ET 0.8686 → **P2 null**. Evaluated on GPU to match `neurovision` | ✓ |
| 12 | MC-dropout maps SSA / PED | 08-23 | 2 | 3.3 | Entropy ≡ MC; disagreement worse (note 39) | — |
| 13 | nnU-Net v2 timing probe | 09-02 | 1 | ~0.5 | 1.087 s/it → 75.5 GPU-h/fold; Gate A not launched (note 46) | — |
| 14 | **`neurovision_heavy_aug`** (D0) | ~08-31 → 09-19 | 3 · `54c03a6` | **23.7** | 12/12 inconclusive → NULL (note 48) | ✓ |
| 15 | **`neurovision_seed43`** (D1) | 09-20 → 09-23 | 3 · `9c770ce` | ~23 | Seed noise floor +0.0021 (note 49) | ✓ |
| 16 | **`baseline_unet3d_seed43`** | 09-26 | 1 | ~5 | C1 replicated at seed 43 (note 54) | ✓ |
| 17 | Gate A nnU-Net prep | 09-26 | CPU kernel | 0 | Preprocessing for nnU-Net | — |
| 18 | **Gate A nnU-Net, sessions 1–2** | 09-26 → 09-27 | chained · `c7778c2` driver | ~21.9 so far | Epoch 372 of 1000, 209–222 s/epoch, healthy. Projected ~61 GPU-h | in progress |

**Total so far: roughly 170 GPU-h** (several rows are approximate). Of that, 10.5 h were lost to
the NaN bug and ~16.5 h went to a baseline that was superseded.

Still planned (Kaggle, ≤28 GPU-h/week): Gate A sessions 3–6 and prediction; D3 fine-tunes (SSA
cf0/cf1, PED cf0/cf1); flip-TTA measurement. Stretch, for the college card: D2 pooled training,
a re-run capacity control.

---

## 5. The hypothesis scoreboard

Everything the project believed at some point, in the order it was tested. ✓ supported ·
✗ refuted or null · ~ mixed. The C-number is the row in `claims_and_evidence.md`.

| When tested | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 08-16 | The dual encoder beats a matched U-Net on Dice | ✓ ET +0.0267, TC +0.0103; WT inconclusive | note 12, C1 |
| 08-16 | …with better boundary accuracy | ✗ fewer errors everywhere, not at the margin | note 13 |
| 08-16 | …with better calibration | ✗ within noise vs a temperature-scaled baseline | note 16 |
| 08-16 | …and better MC-dropout risk-coverage | ✗ within noise | note 17 |
| 08-17 | The gain is architecture, not width | ✓ 79% architecture / 21% capacity | note 19, C2 |
| 08-19 | A better segmentation gives a better structured report | ✗ 1 of 25 metrics | note 23, C6 |
| 08-19 | The eloquence field carries per-case information | ✗ degenerate (100% near eloquent) | note 24 |
| 08-19 | The Dice gain transfers to BraTS-Africa | ✗ ET −0.0008, inconclusive | note 26, C7 |
| 08-19 | HD95 improves under shift (pre-registered) | ✗ H1 not confirmed; TC significantly worse | note 30 |
| 08-19 | Temperature scaling survives shift | ✗ 4x worse on SSA ET | note 29 |
| 08-20 | The fusion gate opens at ambiguous zones / the margin (P1) | ~ gate strongly organised by anatomy, but not peaking at the margin | note 32 |
| 08-23 | Disagreement detects failure beyond entropy (Gate 1) | ~ PARTIAL: yes per voxel, no per case | note 35, C4 |
| 08-23 | Adding disagreement improves a working detector (Gate 2) | ✗ worse or unchanged | note 37, C5 |
| 08-23 | The disagreement conditioning produces the gain (P2 — founding hypothesis) | ✗ **pre-registered null** | note 38, C3 |
| 08-23 | Bigger inference window fixes multifocality | ✗ refuted, costs WT Dice | note 36 |
| 08-23 | Disagreement matches MC-dropout at 1/10 the cost | ✗ worse; **entropy ≡ MC** | note 39, C8, C9 |
| 08-24 | The gain survives lesion-wise scoring | ✓ larger (ET +0.0508), exploratory; ✗ still does not transfer | note 41, C10–C12 |
| 08-24 | A conformal bound on the miss rate holds in distribution | ✓ 6/6, both models | note 42, C13 |
| 08-24 | …and survives distribution shift | ✗ breaks, graded by shift size | note 42, C14 |
| 08-24 | The confidence head is useful uncertainty | ✗ beaten by entropy on all regions | note 44, C16 |
| 08-26 | A QC model beats entropy at flagging bad masks (Gate C) | ~ one cell of five; worse on two | C18 |
| 08-26 | The QC model stays reliable under shift | ✗ grows more optimistic in all six external cells | C19 |
| 09-19 | The refusal gate makes the pipeline safe under shift | ✗ silent failure 4.2 / 18.3 / 49.5% | note 47, C20–C22 |
| 09-20 | Heavy augmentation closes the shift gap (D0) | ✗ NULL, 12/12 inconclusive | note 48, C23 |
| 09-24 | The headline is bigger than seed noise (D1) | ✓ 13x the noise floor | note 49, C1 |
| 09-26 | A new site can restore the bound with local labels | ✓ ~10–40 cases, **except PED TC** (counterfactual) | note 52, C24 |
| 09-27 | The headline replicates at a second seed pair | ✓ +0.0247 | note 54, C1 |
| 09-27 | The clinical front end costs ≤0.05 WT Dice on real DICOM | ✗ −0.23, prediction failed ~5x | note 55, C25 |
| 09-27 | An input OOD score earns a place as a refusal signal | ✗ switch-on rule failed; sees cohort, not failed study | note 56, C26 |
| open | The gain survives nnU-Net (Gate A) | — training | prereg strong_baseline |
| open | Cross-fitted fine-tuning fixes PED tumour core (D3) | — pre-registered | prereg finetune |
| open | Flip TTA helps | — pre-registered | prereg tta |

**The pattern worth saying in the paper:** single-pass predictive entropy — free, in every model —
was never beaten by any uncertainty mechanism this project built (C17). And twice a promising
signal survived only until the sample grew (notes 23 and 30).

---

## 6. Major decisions log

Every decision that shaped the project, with the reason. "Author" means Amish decided; the rest
were agreed in session and recorded in the plan files. Decisions marked **binding** are still in
force.

### Setup and engineering rules

| Date | Decision | Why |
|---|---|---|
| 07-31 | **Binding.** Fixed stack: PyTorch, MONAI, Hydra, W&B, pytest/Ruff/Black. No Lightning, Optuna, TensorBoard, MLflow, Docker (until release) | One tool per job; prefer MONAI over hand-rolled code |
| 07-31 | **Binding.** Eight hard constraints (16 GB floor, resume, no hardcoded paths, no CUDA assumptions, AMP, 96³ default, no unasked dependencies, CPU shape tests) | A shared/preemptible GPU and a Mac for everything else |
| 08-01 | **Binding.** Mac = code, tests, evaluation, analysis; GPU = gradient descent only. Kaggle T4 only, never P100 | P100 unsupported by stock PyTorch; GPU hours are rationed |
| 08-01 | **Binding.** Preprocess once offline; splits frozen 875/187/189, seeded; adding cases raises | Reported numbers must never move under a reshuffle |
| 08-01 | **Binding.** Three overlapping sigmoid regions (ET ⊂ TC ⊂ WT), never softmax/argmax; own DiceBCE loss; never MONAI's `ConvertToMultiChannelBasedOnBratsClassesd` or `DiceCELoss` | Both MONAI helpers fail silently on this label layout |
| 08-06 | **Binding.** Statistics: paired bootstrap over case indices + Wilcoxon + Holm, family fixed before looking; `compare_models` verdict is conservative | Every "A beats B" must survive this |
| 08-06 | **Binding.** Calibration mask never uses the label; calibration claims only vs a temperature-scaled baseline | The circular-mask bug |
| standing | **Binding.** Opus is the architect; every `.py` file is written by a Sonnet `py-implementer` from a spec; one module at a time; explain it back to the author | The author is learning and must be able to read each module |
| 08-08 | **Binding.** Pin `GIT_REF` to a pushed SHA for every GPU run; commit and push every change | The 200-epoch baseline lost its provenance; Kaggle clones the repo at run time |
| 08-09 | **Binding.** A probe must reach the failure condition | 10.5 GPU-h lost to NaN |
| 08-19 | **Binding.** Nothing is ever fitted on SSA or PED; they sit entirely in `test` | Otherwise they stop being external |
| 08-19 | **Binding.** Volume-sized outputs are caches; save fp16 logits so everything can be re-scored without inference | Disk; and re-scoring (lesion-wise, conformal) became free |
| 08-22 | **Binding.** One heavy local job at a time; `sw_batch_size=1` on the Mac | Parallel shards exhausted memory twice |
| 08-23 | **Binding.** Pre-register every gate before the first number; nothing above a `## Result` line is ever edited; negatives are deliverables | The earlier nulls were predictable and not pre-checked |
| 08-23 | **Binding.** `claims_and_evidence.md` is the gate: a claim not in it does not go in the paper | Eight negatives make it easy to write around them |
| 08-23 | **Binding.** New dependencies isolated in `requirements-analysis.txt` / `requirements-clinical.txt`; root `requirements.txt` unchanged | Keep the Kaggle training environment clean |
| 08-23 | **Binding.** Additive only — a new feature must not move a published number (tested by frame equality) | Protects every published row |

### Research direction

| Date | Decision | Why |
|---|---|---|
| 08-04 | The contribution is a fusion gate conditioned on inter-branch disagreement; predictions P1–P5 written first | The only part not already published |
| 08-06 | 60 GPU-h, single seed; the P2 ablation outranks baseline breadth; SwinUNETR, grid, extra seeds cut | Without P2 the paper is "we built a fusion model" |
| 08-08 | All arms at 64³ patches, 80 epochs; keep the architecture intact | Probe: 96³ cost >3x the budget; fusion is only 1.5% of the forward pass |
| 08-16 | Drop the reliability framing; add a parameter-matched capacity control | Calibration, boundary and risk-coverage all null |
| 08-16 | Pivot to an interpretable scan-to-report pipeline; knowledge base from published sources only; no VASARI claim; midline shift declined | No neuroanatomy reviewer; no MRI midline ground truth |
| 08-19 | Reframe around disagreement as a free failure detector; every phase states pass condition and effect size first | Five of six hypotheses null; process error named |
| 08-20 | Post-processing channel merging: document, don't change | Effect negligible; changing it would break comparability with the unrecoverable capacity control |
| 08-23 | Respecify Gate 2 around voxel-level localisation **before** running it; decline case-level referral | Gate 1 showed case-level is null externally |
| 08-23 | **Milestone 4 pivot:** conformal bound + QC model + refusal gate + clinical front end + error budget. Target BraTS 2027 | The architecture claim is ~92% published territory; these have high prior of useful outcomes |
| 08-23 | **Cut list, not to be re-litigated without new evidence:** MGMT, survival, WHO grade, midline shift, eloquence verdict, remaining ablation rungs, full 96³ retrain, SwinUNETR (demoted), new fusion variants | Each fails an effect-size, data or validity argument (`master_plan.md` §3) |
| 08-23 | **Rejected: patient-facing framing.** Research / education / decision-support only, enforced in code and copy | A diagnostic claim is a regulated device |
| 08-24 | Flip-TTA measurement moves to the GPU | 8x inference is ~34 h on the Mac |
| 08-26 | Author: Phase F (IDH) not yet | Large download and training run; tool works without it |
| 09-15 | Author: **park the GPU track, finish the tool first** | A complete tool beats a marginally better model |
| 09-18 | Pre-E2 input QC uses a `pre_registration` stage, not a looser threshold; HD-BET `fast`, no TTA on CPU | Raw series legitimately differ in grid; HD-BET defaults unusable on 16 GB |
| 09-18 | Author: unpark GPU with quota rules; **Gate A not launched** on Kaggle under its own abort bound | 75.5 GPU-h > 60; cutting epochs forbidden |
| 09-20 | D0 null → training recipe unchanged; deployed model stays `neurovision` seed 42 | Pre-registered rule |
| 09-26 | **Milestone 5:** thesis rewritten to match the evidence; one live queue; deadlines fixed (GPU request ~11-05, semester ~11-24, MELBA ~Apr 2027, capstone summer 2027, IDH go/no-go 2027-01-15) | The outside review's three logical breaks |
| 09-26 | Author: `conformal_band` display-only; accept PED cost (49 → 53 silent failures) because intended-use scoping follows | 12 of 12 in-distribution band refusals were wrong |
| 09-26 | Author: intended use = adults only from `PatientAge`; missing age is CAUTION, a documented exception to "empty signal = REFUSE" | De-identification often strips age |
| 09-26 | Author: **run every Phase 4 GPU job on Kaggle now**, cap 30 GPU-h/week (plan ≤28). Gate A Amendment 1: full 1000-epoch nnU-Net chained across sessions, fold all (875 cases), 95 GPU-h bound, Auto3DSeg dropped | Deadlines need Gate A; the college card comes too late |
| 09-26 | Demo DICOM studies are training patients — say so in the demo and report | Note 50 |
| 09-27 | `intended_use` enabled in the live gate (after the P1.2 run, which was frozen against the old gate) | P1.2 protocol ordering |
| 09-27 | `ood_score` stays display-only | Pre-registered switch-on rule failed |
| 09-27 | C14 no longer claims "nobody has measured" | arXiv 2606.20115 measured it on FeTS sites |

---

## 7. Incidents and traps, in the order they happened

Each cost GPU hours, a wrong number, or a silent bug. The evidence is in `lessons.md`; the ten worst
are indexed in `CLAUDE.md`.

| Date | What went wrong | Cost | Rule now |
|---|---|---|---|
| 08-01 | P100 reports CUDA available but every kernel fails | a session | T4 only |
| 08-01 | MONAI HD95 on CUDA tensors → CuPy → fails on Kaggle | a session | CPU tensors into metrics (trap 8) |
| 08-01 | 200-epoch baseline cloned `main`, no SHA recorded | provenance of a published row | Pin `GIT_REF` |
| 08-02 | RNG state restored to CUDA broke `.numpy()` | resume bug | `.cpu()` first; third CUDA-only fault past a green CPU suite |
| 08-06 | Contribution doc and gate code disagreed for two days | would have trained the null as "ours" | Re-read P2 when the gate's input changes |
| 08-06 | Calibration mask used the label → 41–57% of ECE was artifact; first fix didn't reach call sites | a false result, nearly published | Trap 1; verify by re-running the analysis |
| 08-08 | T4 has 14.56 GiB, not 16; AMP factor ~1.0x not 0.55x | two OOM probes | Measure memory, don't assume |
| 08-08 | `grad_clip_norm: 1.0` clipped 66–70% of steps | would have confounded P2 | Set from measured gradient norms |
| 08-08 | `git clone -b <sha>` fails; `!git` failure doesn't stop a cell | a session | `subprocess.run(check=True)` |
| 08-09 | fp16 entropy clamp was a no-op → NaN training | **10.5 GPU-h** | Trap 2; logits-based entropy; probes reach failure |
| 08-09 | "92% of error within 2 mm" computed from rates | wrong headline | Trap 4; weight by voxel count → 74% |
| 08-14 | Demo viewer drew the head on its side; `"val"` matched `"eval"` | wrong display | Substring trap (honourable mention) |
| 08-16 | Session 3 marked ERROR for a legitimately absent `best.pt` | a scare | Guards must allow the legitimate uncommon case |
| 08-16 | `outputs/eval_test` is the 96³ U-Net, not `neurovision` | a wrong committed claim | Read `eval_config.yaml`, not the directory name |
| 08-17 | Capacity control checkpoint left in Kaggle `/tmp/` | **can never be re-scored** | Copy checkpoints off the box before a session ends (trap 10) |
| 08-16 | Brain-mask Dice prefers a mirrored atlas | alignment gate blind to worst error | Trap 3 |
| 08-19 | Laterality table: `unlabelled` folded into midline (34.8% vs 0.27%) | wrong population figure | Report unlabelled as its own class |
| 08-20 | `remove_small_components` merges channels; `min_size` inclusive | wrong documentation | Documented |
| 08-22 | `sw_batch_size: 4` = 12.39 GiB peak on CPU | local jobs impossible | Trap 6 |
| 08-23 | `git add -A` with two parallel subagents → a lying commit | wrong commit message | Never `git add -A` with parallel agents |
| 08-23 | Driver wrote to the baseline's directory (`output_dir` interpolation) | mis-filed results twice | Explicit `out_dir` keys |
| 08-24 | Hydra config group composes one level deeper; hand-built test config hid it | driver read the wrong key | Test with the real composed config |
| 08-24 | QC model selection read the test split | would have leaked test | Case-level val split for selection |
| 08-24 | Test fixture seeded from `hash()` differs per process | flaky ordering test | Seed from a literal |
| 09-02 | nnU-Net probe log unread for 16 days (download broke) | two weeks of planning blind | `kaggle kernels logs` fetches the log only |
| 09-18 | Shipped 3D twin was mirrored and had wrong strides; nobody had looked | a wrong visual in the demo path | Look at the real render |
| 09-18 | Script green on a fully faked pipeline failed twice on real data | two runs | Trap 9: a script is verified on the real fixture |
| 09-20 | "15 cases/min" evaluation speed in `CLAUDE.md` was 100x wrong | a healthy run looked stalled | Measured ~1.4 min/case |
| 09-27 | Clinical job peaks ~12 GB; many jobs in one process thrash the 16 GB Mac | P1.2 run restarts | Memory-capped watchdog, resume per case |
| 09-27 | Recalibration figure filter correct for one cohort deleted data in another | a wrong figure | Check filters per cohort |
| 09-27 | Trailing `git checkout -- .` in a compound command reverted uncommitted work | lost edits | No destructive git in compound commands |

---

## 8. How the codebase and data grew

### Tests over time

| Date | Tests passing | Milestone |
|---|---|---|
| 08-06 | 984 → ~1,000 | Model + analysis stack |
| 08-22 | 1,373 | Interpretable pipeline, Milestone 3 |
| 08-23 | 1,630 | Master plan written |
| 08-26 | 2,004 | Clinical pipeline wired |
| 09-18 | 2,058 | Tool completion |
| 09-24 | 2,230 (+35 skipped, ~60 s) | Review |
| 09-27 | 2,456 in a fresh clone | Fresh-clone rehearsal |

Frontend: vitest unit tests and a headless-Chrome E2E harness that asserts on rendered pixels.

### What exists now (by area)

- **Research path** (`.venv`): data, models (`unet3d`, `swinunetr`, `neurovision`, `segqc`), losses,
  training with full resume, sliding-window / MC-dropout / TTA inference, metrics (voxel,
  boundary-stratified, lesion-wise), calibration, conformal risk control, local recalibration,
  statistics, explainability (IG, Grad-CAM, gate maps), error budget, OOD score, nnU-Net import.
- **Interpretable layer:** SRI24 atlas, localisation, eloquence, burden, involvement, shape
  descriptors, reports (JSON + Markdown), report agreement, population statistics.
- **Clinical path** (`.venv-clinical`): DICOM ingest, co-registration / SRI24 / HD-BET, input QC,
  gatekeeper (input QC, predicted Dice, intended use live; conformal band and OOD display-only),
  DICOM-SEG export, molecular panel, job persistence, export bundle.
- **App:** FastAPI backend + React frontend — research viewer, 3D digital twin with atlas shells
  and uncertainty layers, `/clinical` upload page, report pages.
- **Documents:** 11 pre-registrations, 2 protocols, the claims table, 56 result notes, lessons,
  reproducibility, model card, related work, semester report draft, demo runbook, GPU request draft.

### Data

| Dataset | Cases | Role | Status |
|---|---|---|---|
| BraTS 2021 training set | 1,251 → 875/187/189 | Train / val / test | Preprocessed on disk; **only backup is the Kaggle dataset** — do not delete. Raw deleted 08-19, **no SHA manifest** |
| BraTS-Africa (SSA) 2023 | 60 | External test only | Preprocessed; raw deleted; SHA manifest committed |
| BraTS-PEDs 2023 | 99 | External test only | Same |
| TCIA UPENN-GBM-00001/2/3 | 3 DICOM studies | Clinical demo fixtures | Found to be BraTS training patients (note 50) |
| RSNA-MICCAI DICOM | 40 test-split patients (+1 pilot) | Real-DICOM validation (P1.2) | Frozen sample, seed 42 |
| UCSF-PDGM | — | Phase F (IDH), parked | Go/no-go 2027-01-15 |

Disk reclaims: 08-19 (raw data + caches, 58 → 133 GiB free), 09-15 (83 GB: `nnunet_raw`, MC
caches), 09-27 (~13 GiB). Five checkpoints survive: `neurovision`, `baseline_unet3d`,
`ablation_content_only_gate`, `neurovision_heavy_aug`, `neurovision_seed43` (plus
`baseline_unet3d_seed43` and the QC model). Two are lost for good: the 200-epoch baseline and the
capacity control.

---

## 9. Where it stands on 2026-09-27

Full detail and restart commands: `status_2026-09-27.md`. In short:

- **Proven:** C1 (+0.0267 ET, replicated, above seed noise), C2 (79% architecture), the conformal
  bound in distribution (C13), graded failure under shift (C14), end-to-end error budget (C20–C22),
  local recalibration recipe (C24), real-DICOM front-end cost (C25).
- **Running:** Gate A (nnU-Net) on Kaggle, epoch 372 of 1000. Until it reports, every C1 sentence
  ends with "not yet tested against nnU-Net".
- **Next on GPU:** Gate A sessions 3–6 and prediction, D3 fine-tunes, flip-TTA measurement.
- **Author-only:** eyeball the 3D twin on job `9c2cc294` (P0.7); write the report's Introduction
  and Methods (P3.1); submit the GPU request ~11-05 (P2.6); rehearse the demo and tag
  `semester-2026` ~11-24 (P3.2, P3.4).
- **Later:** IDH go/no-go 2027-01-15; MELBA draft February, submission ~April 2027; capstone
  summer 2027.

---

## 10. Using this history in the paper

- **The story arc.** A model that was built to be *more reliable* turned out to be *more accurate
  but not more reliable*; its claimed novelty turned out not to carry the gain; and the reliability
  question was rebuilt as a measurable property of a wrapper — which holds in distribution and
  breaks in proportion to shift. The negatives are the evidence for the final claim, not an
  embarrassment inside it.
- **Methods:** §3.1–3.4 (data, split, schedule, the 64³/80-epoch re-plan and why), §6 (statistics
  and pre-registration rules), `reproducibility.md`, the pre-registrations.
- **Results:** §5 is the ordered list of what was tested; always take numbers from the notes.
- **Discussion / limitations:** §3.6 (why reliability failed), §3.9 (information ≠ utility, the
  P2 null), §3.14 and §3.16 (the gate cannot see cohort shift; real DICOM loses a lot),
  `claims_and_evidence.md` §3.
- **Things that must travel with every quote:** single-seed for most models; no transformer
  baseline on our split; voxel Dice on a random split of the training set is not comparable to
  published BraTS numbers; lesion-wise results are exploratory; α=0.20 conformal thresholds are
  grid-censored; the conformal bound is an average over studies, never per patient; recalibration
  results are counterfactual; real-DICOM results are not external validation.
