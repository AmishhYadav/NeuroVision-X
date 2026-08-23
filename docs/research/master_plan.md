# Master Plan — Milestone 4

**Written:** 2026-08-23 · **Horizon:** ~6 months · **Status:** ACTIVE
**Compute:** 200+ GPU-h on a modern card (college cluster)
**Priority:** a working clinical-imaging tool first, publication alongside it

**Relationship to other documents.** This supersedes the *sequencing and gates* of
`execution_plan.md` and `improvement_plan.md`. Their measurements, cost tables, pre-registrations and
fallback analyses remain valid and are cited here rather than repeated. The authoritative record of
what was measured is `docs/experiments.md`; the gate on what may be written is
`docs/paper/claims_and_evidence.md`; the traps that cost real money are `docs/lessons.md`; the
Milestone 1–3 build record is `docs/project_state.md`.

> **Starting a session cold, with no context? Go straight to §4, *Execution order*.** It carries
> the reading order, the queue with its current state, the dependency arrows, and the working
> agreement for building each module. Sections 0–3 are the *why*; §4 is the *what next*.

---

## 0. Context — why this plan exists

The engineering is finished and rigorous: 1,630 tests, a full train/evaluate/analyse/report stack, a
working demo with live upload and CPU inference, two external cohorts, and a parameter-matched
capacity control that almost no comparable project runs. The science, tested honestly, mostly came
back null.

Nine pre-registered or matched comparisons have resolved. **One is positive**: ET Dice +0.0267 over a
matched U-Net, of which +0.0211 is architectural rather than capacity. **Eight are null or negative**,
including the project's founding hypothesis — the content-only gate ablation matched the full model
(+0.0022, CI −0.0067 to +0.0152, p_holm 0.17), so the disagreement conditioning contributes nothing
measurable; and note 39 showed branch disagreement is *worse* than free single-pass entropy as an
error localiser, while entropy itself is statistically equivalent to 10-sample MC-dropout.

Two structural problems sit behind that record. Both are fixable.

**1. Every dead claim was a downstream claim.** "Does a +0.02 Dice gain also buy calibration, boundary
accuracy, better reports, OOD robustness?" A gain of that size is a handful of voxels at a tumour
margin. It cannot move a volume-dominated metric, cannot flip which anatomical structure a tumour
overlaps, and cannot survive a distribution shift larger than itself. Each null was predictable from
an effect-size argument that was never made before the compute was spent.

**2. The project has never been tested against the bar the field actually applies.** The baseline is a
MONAI U-Net at 64³, 80 epochs, AdamW 1e-4. nnU-Net trains 1000 epochs with heavy augmentation,
five-fold ensembling and a larger patch. MICCAI 2024's *nnU-Net Revisited: A Call for Rigorous
Validation in 3D Medical Image Segmentation* benchmarked exactly this class of claim and found that
architectural gains over weak baselines evaporate against a properly configured CNN U-Net — and that
SwinUNETR, nnFormer and CoTr all fail to match CNNs. The metrics are legacy too: BraTS moved to
**lesion-wise Dice and normalised surface Dice in 2023** precisely because voxel-wise Dice favours
large lesions and hides missed small ones, and this project still reports voxel-wise Dice on a random
split of the 2021 *training* set, which is not comparable to any published BraTS number.

**What this plan does.** It stops defending an architecture claim that is already ~92% explained by
gated cross-attention fusion — published territory — and rebuilds the project around something the
negatives *motivate* and the literature says is achievable:

> **A brain-tumour analysis pipeline that produces a segmentation with a statistically guaranteed
> error bound, knows when to refuse, ingests real clinical MRI, and is validated end-to-end across
> four cohorts — with the compounded error budget published rather than assumed.**

Every phase below has a high prior of a positive, useful outcome rather than another null, because
each is either a theorem (conformal risk control), a well-replicated empirical result (fine-tuning
recovers most of the sub-Saharan generalisation gap; segmentation-QC networks reach 95–99% good/bad
separation), or deterministic engineering (DICOM ingest, skull-stripping, co-registration).

---

## 1. Evidence ledger — what is settled

Authority: `docs/paper/claims_and_evidence.md`.

**Proven, keep.** ET +0.0267 vs the matched baseline (p_holm 1.4e-21, n=189 paired). 79% architecture
/ 21% capacity, via a width-matched control (34.83M vs 34.91M params). The interpretable report layer
is *stable* with respect to segmentation quality — a negative for "better model, better report", but
useful for deployment framing. Single-pass predictive entropy is equivalent to 10-sample MC-dropout
for voxel-level error localisation (paired TOST at a margin of 0.03 AUROC, fixed in advance).

**Dead, never write again.** Better calibrated; better boundary accuracy; better uncertainty or
risk-coverage; "the disagreement-conditioned gate is what works"; better structured reports;
"disagreement equals MC-dropout at 1/10 the cost"; any claim on WT at a saturated ~0.93.

**Open and worth money.** Does the accuracy gain survive a *strong* baseline? Does pooled or
fine-tuned training close the out-of-distribution gap? Does a guaranteed-coverage layer hold under
real distribution shift?

---

## 2. Design principles — binding on every phase

1. **Effect-size gate before compute.** State the smallest effect the test can detect and the
   plausible size of the real effect. If plausible < detectable, the experiment does not run.
2. **Errors multiply in a cascade.** Five stages at 95% each is 77% end-to-end. Every stage emits a
   confidence and the pipeline has an explicit **REFUSE** state. The end-to-end error budget is a
   deliverable, not an afterthought.
3. **Downstream models train on *predicted* masks, never ground-truth masks.** Training on perfect
   masks and deploying on the model's own masks is train/serve skew and guarantees deployment
   failure. Mask-derived features are additionally unstable to small mask perturbations, so a
   downstream model should see image + mask rather than mask-derived numbers alone.
4. **Additive only.** New capability must not move an already-published number. Enforced the way
   `tests/test_evaluate_script.py` already enforces it for the boundary metrics: run evaluation with
   and without the feature, assert frame equality on the shared columns.
5. **An analysis fix is verified by re-running the real analysis**, not by its unit tests. This
   project has already shipped a commit whose message, memory note and 1,000 green tests all claimed
   a circular-mask bug was fixed while every reported number stayed circular.
6. **Pre-register before measuring.** One `docs/research/preregistration_*.md` per gate, committed
   with a git timestamp before the first number exists.
7. **Dependency isolation.** The training environment stays clean. New stacks live in
   `requirements-analysis.txt` (CPU/stats) and `requirements-clinical.txt` (registration,
   skull-stripping, DICOM), mirroring the existing `app/backend/requirements.txt` split.

---

## 3. Feature adjudication — everything considered, with verdicts

Cross-questioned on three axes: is the signal real, is it validatable with data we can obtain, and
does it survive an effect-size argument.

| Candidate | Verdict | Reason |
|---|---|---|
| Strong baseline: nnU-Net v2 + MONAI Auto3DSeg SegResNet, on our exact split | **ADD — highest priority** | The single largest credibility risk in the project. nnU-Net is ~12–24 h/fold on an A100 for BraTS-scale data. Auto3DSeg is already inside MONAI (our fixed stack, no new dependency) and took 1st place in BraTS-Africa 2023 |
| Lesion-wise Dice + NSD via `panoptica` | **ADD** | The official BraTS metric since 2023. `pip install panoptica`, Python 3.10+, pure CPU. Our measured multifocality over-reporting (30.7–41.3% against a true 22.8%) is exactly the failure voxel-wise Dice hides |
| Conformal risk control — a guaranteed bound on the mask's miss rate | **ADD — new centrepiece** | Distribution-free, calibration-only, CPU-only, ~5-line core algorithm (Angelopoulos et al., ICLR 2024). Model-agnostic. Converts "uncertainty" from a lost claim into a shipped product feature |
| Conformal coverage **under distribution shift** (SSA, PED) | **ADD — the novel science** | Conformal validity assumes exchangeability, which real shift violates. Nobody has measured this for 3D tumour segmentation. We already hold two external cohorts with ground truth, saved logits, and the calibration machinery. Zero GPU hours |
| Segmentation QC model — predict Dice with no ground truth | **ADD** | Literature reports 95–99% separation of good from bad segmentations. Trainable from our own 376 val+test cases plus synthetic mask degradation. This is the reliability claim rebuilt as a component instead of as a property of one architecture |
| Pooled / fine-tuned multi-cohort training (BraTS + SSA + PED) | **ADD** | Attacks the measured OOD failure directly (`dice_TC` −0.0333 pooled, p_holm 0.0132). Replicated repeatedly in the BraTS-Africa literature. ~40 GPU-h |
| Multi-seed: 3 seeds × 3 models | **ADD** | No number in this project currently has a noise floor. Free by-products: it restores the lost capacity-control checkpoint, and the seeds form a deep ensemble — the correct uncertainty comparator, since ensembles beat MC-dropout in the brain-tumour uncertainty literature |
| Real-MRI front-end: DICOM → NIfTI, co-registration, atlas registration, skull-strip, N4 | **ADD** | `brainles-preprocessing` (pip, ANTs default + HD-BET default, SRI24 atlas) plus `dcm2niix`. This is the difference between a demo and a tool. Today `app/backend/jobs.py` runs only `preprocess_case` — reorient, nonzero z-score, crop — so a skulled, unregistered clinical scan produces nonsense with nothing failing |
| DICOM-SEG export | **ADD (small)** | `highdicom` / `dcmqi`. Output loads in any PACS or OHIF viewer. Cheap, and a strong credibility signal |
| Refusal gate (input QC + OOD score + QC model + conformal band width) | **ADD** | The pipeline's safety valve, and the honest home for the failure-detection work |
| Flip TTA | **ADD (free)** | `src/neurovision/inference/tta.py` exists with `tta_predict`, and is **not wired into `scripts/evaluate.py`** — no config key, no call site. Expected +0.003–0.008 Dice for zero GPU hours |
| Confidence-head scoring | **ADD (free)** | Trained at weight 0.05 and never once scored. Either a fourth uncertainty comparator or an honest "the head learned nothing" |
| IDH molecular subtyping on UCSF-PDGM | **ADD — gated, Phase F** | 495 subjects, IDH status for all, NIfTI, already skull-stripped and ANTs co-registered at 1 mm isotropic, CC BY 4.0, DOI 10.7937/tcia.bdgf-8v37. Recent multicentre deep learning reports externally-validated AUC ~0.88–0.90. **Download only the four structural sequences plus segmentation** — the full v5 collection is 142 GB against ~133 GiB free |
| MGMT methylation | **CUT** | RSNA-MICCAI 2021 topped out near 0.62 AUC and follow-up work argues it is not recoverable from routine MRI. Building it means building a known null. `interpretable_pipeline_plan.md` already called it a poor bet |
| Survival prediction | **CUT** | Near-chance in BraTS for years, and no outcome data exists in any cohort we hold |
| WHO grade | **CUT as stated** | CNS5 grading is integrated — histology plus IDH, 1p/19q, ATRX, TERT, CDKN2A/B — and is not determinable from MRI. If ever built from the BraTS 2019 split it must be named **GBM vs non-GBM**, because "LGG" there merges grades I–III |
| Midline shift | **STAYS CUT** | Re-checked 2026-08-23: public midline-shift ground truth is essentially CT-only and institutional; no validated public MRI set exists. Deformable registration could estimate it, but an unvalidatable number inside a clinical-looking report is worse than no number |
| Eloquence verdict | **STAYS CUT** | Degenerate on this cohort — 100% "near eloquent", distance exactly 0.0 mm for 98.8%. The report keeps only the graded per-structure quantities |
| Remaining four ablation rungs | **CUT** | P2 already showed the mechanism they would dissect does not carry the gain. ~90 GPU-h for a finer decomposition of a null |
| Full 96³ retrain of all three models | **DEFER** | ~600 GPU-h for the matched programme. Phase 2.4 already refuted the inference-window hypothesis and found 96³ inference *worsens* `dice_WT`. Revisit only if lesion-wise metrics implicate training patch size |
| `baseline_swinunetr` on our splits | **DEMOTE** | Superseded in value by nnU-Net / Auto3DSeg, which is the bar reviewers actually apply. Keep as a stretch row |
| LLM report writing | **ADD, tightly constrained** | The language model sees **computed fields, never pixels**. Factual-error rates in AI-generated radiology reports run 8–22% in recent evaluations. Every sentence must be traceable to a field, and the deterministic template stays the fallback |
| Patient-facing framing | **REJECTED — flagged to the author** | Presenting a tumour segmentation to a patient as a finding is a diagnostic claim and a regulated device. Every cleared product in this space — NeuroQuant Brain Tumor, Neosoma HGG, Cercare Oncology Virtual Expert, VUNO Med-DeepBrain — is clinician/PACS-facing. This tool is positioned as **research / education / decision-support**, never diagnosis, and that framing is enforced in code and copy |
| BraTS 2026 challenge submission | **FLAG — unresolved, act in week 1** | Challenge 3 of the 2026 cluster is *"Generalizability of brain tumour sub-region segmentation across tumour entities"*, a near-perfect fit for this project's measured weakness. MICCAI 2026 runs 27 Sep – 1 Oct 2026 and today is 23 Aug 2026, so the window is likely closed; historical BraTS validation phases run July–August with MLCube/Docker submission. **The Synapse portal (syn74274097) is a JavaScript app and its dates are not publicly fetchable — check while logged in.** If closed, target BraTS 2027 and use the released data regardless |

---

## 4. Execution order — start here if you have no context

**This section exists for a session that opens the repo cold, months from now, with nothing in
memory.** It is the queue, the dependency arrows, and the working agreement. It is deliberately
*directional* rather than rigid: the ordering constraints below are real, but where two items have no
arrow between them, pick whichever suits the session you have.

### 4.1 Orientation — the first fifteen minutes

Read in this order, and stop when you have enough to act:

1. `CLAUDE.md` — constraints, conventions, the ten traps. Short by design.
2. This file, §0 to §3 — why the project changed shape, and what was cut.
3. `docs/paper/claims_and_evidence.md` — the gate on what may be asserted. **Nothing is written into
   the paper that is not in that table.**
4. §4.3 below — the queue. Find the first unfinished item.
5. Only then: the phase detail in §5 for whatever you are about to build.

`docs/lessons.md` is not read front to back. It is read *by subsystem*, immediately before touching
that subsystem, using the ten-trap index in `CLAUDE.md` to find the right entry.

### 4.2 How to tell what is already done

Do not trust this document's status claims over the filesystem — it will drift. Check, in order:

| Question | Command |
|---|---|
| What was done recently? | `git log --oneline -25` |
| Which experiments have results? | `ls outputs/` and the note titles in `docs/experiments.md` |
| Which gates have fired? | `cat outputs/*/*verdict.json`, and the `## Result` section of each `docs/research/preregistration_*.md` |
| Does the repo still build? | plain `pytest` (expect ~1,630 passing, ~35 s) and `python scripts/smoke_test.py` |

**Then update the status board in §4.3 and commit it.** A queue that nobody ticks off is worse than no
queue, because the next session trusts it.

### 4.3 The queue

`[ ]` not started · `[~]` in progress · `[x]` done. **Update these marks as you go.**

#### Track 1 — CPU. No hardware needed. Start here.

| # | Item | State | Done when |
|---|---|---|---|
| A1 | Dependency files: `requirements-analysis.txt`, `requirements-clinical.txt` | `[x]` | Both files exist, pinned; root `requirements.txt` untouched; `docs/reproducibility.md` says which env is for what |
| A2 | `src/neurovision/metrics/lesionwise.py` + additive wiring into `scripts/evaluate.py` | `[x]` | Lesion-wise columns appear in `per_case_metrics.csv` when enabled, and an additivity test proves no existing column moved |
| A3 | Re-score every existing run lesion-wise via `scripts/replay_logits.py` | `[x]` | A lesion-wise row exists for `neurovision`, `baseline_unet3d`, `capacity_control`, `ablation_content_only_gate`, on test and val |
| A4 | Wire flip TTA into `scripts/evaluate.py` | `[ ]` | `cfg.inference.tta` exists; measured on val then test; result recorded as its own note |
| A5 | Score the confidence head | `[ ]` | A number exists, and `docs/experiments.md` says whether the head learned anything |
| A6 | Resolve the BraTS 2026 Challenge-3 deadline from a logged-in Synapse session | `[x]` | Note 40 in `docs/experiments.md` carries a date instead of an uncertainty |
| B1 | `src/neurovision/uncertainty/conformal.py` + `scripts/conformal.py` | `[~]` | λ̂ fitted on val, applied frozen to test, realised risk ≤ α on test |
| B2 | Apply the frozen λ̂ to SSA and PED; weighted/Mondrian variant | `[ ]` | A table of nominal α vs realised risk per cohort, with CIs |

#### Track 2 — GPU. Starts the day the cluster is live. Runs in parallel with Track 1.

| # | Item | State | Done when |
|---|---|---|---|
| G0 | 2-epoch timing probe on the new hardware | `[ ]` | Measured step time and peak VRAM in a log, and the probe **reached the failure condition**, not merely executed |
| G1 | `scripts/export_nnunet_dataset.py` — our frozen split to nnU-Net layout | `[ ]` | `splits_final.json` contains exactly our 875 train cases; no val or test case appears in it |
| A7 | nnU-Net v2 `3d_fullres` single fold + Auto3DSeg SegResNet single fold | `[ ]` | Both scored through **our** `scripts/evaluate.py` metric path on the same 189 test cases |
| — | **GATE A** | `[ ]` | `docs/research/preregistration_strong_baseline.md` has a `## Result` section, and `claims_and_evidence.md` is updated to match |

#### Later — kept coarse on purpose, because Gate A may reshape them

| Phase | Item | Precondition |
|---|---|---|
| C | QC model: pair generator → `models/qc.py` → train → per-cohort validation → Gate C | A2 done (needs a metric to regress against) |
| D | D1 multi-seed → D2 pooled multi-cohort → D3 fine-tune-on-SSA | GPU free after A7. D1 also unblocks B3 |
| B3 | Deep-ensemble comparator, completing the uncertainty ladder | D1 seeds exist |
| E | E1 DICOM ingest → E2 preprocessing → E3 input QC → E4 missing-sequence refusal → E5 gatekeeper → E6 DICOM-SEG → E7 UI | E5 needs B1 and C; the rest is independent |
| F | IDH on UCSF-PDGM | Explicit go/no-go after Phase C. Costs a large download and a training run |
| G | End-to-end error budget | Everything above that will actually ship |
| H | Write-up and release | G |

### 4.4 The dependency arrows that actually bind

Everything else is free ordering.

```
A1 ──▶ A2 ──▶ A3
        │
        └────▶ (Gate A scoring)   ← lesion-wise ET Dice is a CO-PRIMARY endpoint,
                                     so A2 is on the critical path for the GPU gate
A2 ──▶ C1 (QC pair generator needs the metric it regresses against)

B1 ──▶ B2 ──▶ E5 (the gatekeeper reads the conformal band width)
C  ──▶ E5     (the gatekeeper reads the QC model's estimate)

G0 ──▶ G1 ──▶ A7 ──▶ GATE A
D1 ──▶ B3     (a deep ensemble needs the seeds)

A4, A5, A6 — independent of everything. Good filler for a short session.
```

**The one non-obvious arrow:** A2 blocks the *scoring* of Gate A, not the training. Launch A7 on the
GPU whenever hardware appears; just make sure A2 lands before you try to read the result, or you will
be tempted to call the gate on voxel Dice alone, which is precisely the shortcut the pre-registration
forbids.

### 4.5 Parallelism rules

- **One heavy local job at a time.** Parallel shards exhausted application memory twice. Sharding a
  job into independent case chunks is fine; running two whole-volume jobs is not.
- Track 1 and Track 2 are genuinely parallel — one is your Mac, the other is the cluster.
- Within Track 1, build one module at a time. `CLAUDE.md` requires it and the reason is that the
  author has to understand each piece.
- A GPU run and a CPU analysis of a *different* artifact can overlap freely.

### 4.6 When a gate fires — the mechanical checklist

Gates are the only points where the plan forks, and each fork is already written down, so this is
bookkeeping rather than judgement:

1. Write the `## Result` section of that gate's pre-registration. **Nothing above that line may be
   edited** — a prediction edited after the fact is not a prediction.
2. Add a numbered note to `docs/experiments.md` with the numbers, the CIs, and the caveats that must
   travel with them.
3. Update `docs/paper/claims_and_evidence.md` — move claims between the "supported" and "do not
   write" tables as the result requires.
4. Update the status board in §4.3 and the `Current status` block in `CLAUDE.md`.
5. Commit all of it together, so the result and its consequences share one timestamp.

If a gate comes back negative, **that is a completed deliverable, not a failure.** Eight of this
project's nine resolved comparisons were negative and the plan is built on top of them.

### 4.7 If you only have a short session

Ordered by how little context they need:

- **30 minutes:** A1, or A6, or update the status board against the filesystem.
- **Half a day:** A4 (TTA wiring), or A5 (confidence-head scoring), or A3 once A2 exists.
- **A full day or more:** A2, B1, C2, E2 — these are real builds and deserve the full loop.
- **Never start in a short session:** anything on the GPU. A probe that is not watched is a probe that
  silently wastes hours, and this project has lost 10.5 GPU-hours to exactly that.

### 4.8 Standing working agreement

Every module follows the same loop, from `CLAUDE.md`:

> decide the design → write a precise spec → `py-implementer` builds module + tests → `test-runner`
> runs the suite and reports failures only → `code-reviewer` checks it against the spec and the hard
> constraints → read it, judge it, **explain it back to the author in plain terms**, iterate.

A spec that produces good code contains, every time:

1. **File path** and whether it is new or a modification.
2. **Public API** with full type signatures.
3. **Tensor shapes** in and out, as `(B, C, D, H, W)`.
4. **Edge cases** that must be handled — empty region, all-background mask, NaN, a cohort with no
   ground truth.
5. **The tests that must pass**, named, including the CPU shape test under one second.
6. **The relevant hard constraints, restated.** A subagent starts with no memory of the conversation.
   Constraints 2 (no hardcoded paths), 3 (no CUDA assumptions) and 8 (CPU shape test) apply to almost
   everything.
7. **What must NOT change** — the additivity requirement, and any already-published number the module
   sits near.

If a subagent returns something that violates a constraint, fix the spec and re-delegate rather than
patching the output by hand. The patched version will not survive the next regeneration.

### 4.9 Two things that are the author's call, not the implementer's

- **When the GPU track starts.** Cluster access is outside this document.
- **Whether Phase F (IDH) is gated in.** Decide after Phase C ships. It costs a large dataset
  download and a training run, and the project functions completely without it.

Everything else — what gets built, in what order, with what endpoints and what decision rules — is
settled above. That was the point of writing it down before starting.

---

## 5. Phases

Two tracks run in parallel. The **CPU track** never blocks on hardware. The **GPU track** starts the
moment the cluster is available. Phase letters, not numbers, to avoid collision with the old plan.

---

### Phase A — Housekeeping and the honest bar
*Week 1 · CPU + first GPU block*

**Goal.** Establish what this project's numbers mean against the field's bar, before building anything
on top of them.

| # | Task | Files |
|---|---|---|
| A1 | Add `requirements-analysis.txt` (panoptica, statsmodels) and `requirements-clinical.txt` (brainles-preprocessing, antspyx, HD-BET, dcm2niix, highdicom). Root `requirements.txt` untouched | new files; document in `docs/reproducibility.md` |
| A2 | `src/neurovision/metrics/lesionwise.py` — wrap `panoptica` to emit lesion-wise Dice / NSD / F1 / TP / FP / FN per region per case, in the exact column shape `per_case_metrics.csv` already uses, so `analysis/statistics.compare_models`, `visualization/tables.py` and the forest plot all work unchanged | new module; wire into `scripts/evaluate.py` behind `cfg.inference.evaluation.lesionwise`, default **off**, strictly additive |
| A3 | Re-score every existing run lesion-wise from saved logits using `scripts/replay_logits.py`, which already exists and has already been used on four eval directories. No GPU, no re-inference | `outputs/replay/*` |
| A4 | Wire flip TTA: add `cfg.inference.tta`, call `tta_predict` in `scripts/evaluate.py`. Measure on val first, then test | `src/neurovision/inference/tta.py` (exists, unwired), `scripts/evaluate.py` |
| A5 | Score the confidence head for the first time | new `scripts/score_confidence.py` |
| A6 | Resolve the BraTS 2026 Challenge-3 deadline from a logged-in Synapse session; record the answer in `docs/experiments.md` | — |
| A7 | **GPU: nnU-Net v2, `3d_fullres`, single fold, on our exact frozen split** (`configs/data/splits.yaml` exported to nnU-Net's `splits_final.json`, so the comparison is paired on the same 189 test cases). Plus MONAI Auto3DSeg SegResNet, single fold, as a second modern comparator | new `scripts/export_nnunet_dataset.py` |

**Pre-registration, committed before A7 runs:** `docs/research/preregistration_strong_baseline.md`.

**Gate A — the credibility gate.**

| Outcome | Meaning | Consequence |
|---|---|---|
| `neurovision` beats nnU-Net, CI excluding zero | The architecture claim survives the real bar | C1/C2 become strong claims and open the paper |
| Inconclusive | Competitive with the field standard at roughly 1/12 the training schedule | Report as parity-at-lower-cost. A good result. Architecture stops being the headline |
| nnU-Net beats `neurovision` | The +0.0267 was substantially a weak-baseline artifact | **Retire the architecture claim.** The pipeline and guarantee work becomes the whole contribution. Learning this now, pre-registered, reads as rigour; learning it in review does not |

**Cost.** A1–A6 ≈ 4 CPU days. A7 ≈ 25–40 GPU-h.

**Verification.** `pytest` green. An additivity test proving lesion-wise columns move no existing
column. And a sanity check with a known direction: lesion-wise ET for `baseline_unet3d` must come out
*lower* than its voxel-wise 0.8442 — lesion-wise metrics penalise missed small lesions, so a higher
number means the wiring is wrong.

---

### Phase B — Guaranteed coverage, and where the guarantee breaks
*Weeks 2–4 · zero GPU*

**Goal.** The project's new scientific centrepiece and a real product feature. Every artifact it needs
is already on disk.

**B1 — Conformal risk control on the segmentation mask.**
New `src/neurovision/uncertainty/conformal.py`. Given a per-case loss that is monotone in a threshold
λ — use **per-case false-negative rate** on whole tumour and tumour core — calibrate λ̂ on the **val**
split (n=187) with the finite-sample correction so that the expected loss is bounded by α, then apply
λ̂ **frozen** to test. The output is a conservative mask whose expected miss rate is bounded by α, plus
the "uncertain band" lying between the point-estimate mask and the conservative one.

Reuse rather than rebuild: `uncertainty/calibration.py` for masking, subsampling and the accumulator
pattern; `analysis/statistics.py` for bootstrap CIs and Holm correction; and copy the structural
guard from `scripts/calibrate.py`, which *raises* when `fit_dir` and `apply_dir` resolve to the same
path — a fit-and-report-on-one-split convention documented only in a docstring is one CLI override
away from being violated silently.

Driver: `scripts/conformal.py`, config block under `configs/calibration/default.yaml`.

**B2 — Does the guarantee survive distribution shift?** Apply the val-calibrated λ̂, unchanged, to
**SSA (n=60)** and **PED (n=99)**. Measure realised risk against nominal α. Conformal validity assumes
exchangeability and these cohorts violate it by construction, so this is a genuine open question.
Then implement and compare a **weighted / Mondrian** variant that recalibrates per cohort on a
held-out slice.

**Pre-registration:** `docs/research/preregistration_conformal.md`, fixing α ∈ {0.05, 0.10, 0.20}, the
loss definition, the cohorts and the reporting rule before any number exists.

**Why this phase is structurally different from every previous one.** B1 *cannot* fail — it is a
theorem, and it ships as a feature whatever the numbers say. B2 is where the finding lives, and both
outcomes are publishable and useful: if coverage holds under shift, that is a strong safety result;
if it degrades, we have quantified by how much, which nobody has done for 3D tumour segmentation, and
it directly motivates the refusal gate in Phase E.

**B3 — Deep-ensemble comparator.** Scheduled here, run when the Phase D seeds land. Completes the
uncertainty ladder: entropy / MC-dropout / deep ensemble / conformal. Note 39 already established that
entropy ≡ MC-dropout; the ensemble arm is the one rung missing.

**Deliverable.** `outputs/conformal/`, a pre-registration with a result section, and a new demo layer
rendering the bounded mask.

---

### Phase C — The QC model: predicting your own error
*Weeks 3–5 · mostly CPU*

**Goal.** A second, independently trained model that reads image + predicted mask + entropy map and
outputs an estimated Dice, with no ground truth available.

| # | Task | Notes |
|---|---|---|
| C1 | Training-pair generator: from each case with ground truth, synthesise (image, degraded mask, true Dice) triples by eroding, dilating, dropping components and shifting. Thousands of pairs from 376 real cases | Labels come from `metrics/segmentation.py` |
| C2 | `src/neurovision/models/qc.py` — small 3D CNN regressor registered as `@register_model("segqc")`, 3-channel input. CPU shape test on `(1, 3, 32, 32, 32)` under one second | The registry pattern already exists and takes a `cfg` |
| C3 | Train on predictions **from the deployed model**, never on ground-truth masks (principle 3) | ~2–6 GPU-h, or CPU overnight |
| C4 | Validate: Spearman(predicted Dice, true Dice) and AUROC for "Dice < 0.7", reported **per cohort** — in-distribution, SSA, PED | Literature bar: ρ > 0.7 in distribution |
| C5 | Silent-failure test: does the QC model itself degrade under shift? Report honestly — this is a known open problem and measuring it is a contribution | — |

**Gate C.** The QC model must beat the free baseline — mean predicted-foreground entropy — on
case-level AUROC for "Dice < 0.7", with a CI excluding zero, on at least one external cohort. If it
does not, it still ships as a displayed number but makes no claim.

---

### Phase D — Fixing the generalisation failure
*Weeks 4–8 · GPU*

**Goal.** Stop reporting the OOD failure and start closing it. The phase with the clearest positive
prior in the plan — the sub-Saharan literature replicates this result.

| # | Run | Cost | Purpose |
|---|---|---|---|
| D1 | **Multi-seed**: 3 seeds × {`neurovision`, `baseline_unet3d`, `capacity_control_unet3d`} at 64³ | ~70 GPU-h | A noise floor for every number in the project; restores the lost capacity-control checkpoint; yields the deep ensemble for B3 |
| D2 | **Pooled multi-cohort**: BraTS + SSA + PED, one seed, with a held-out slice of each external cohort | ~40 GPU-h | Attacks `dice_TC` −0.0333 directly. Needs a new `configs/data/splits_pooled.yaml`, **frozen before training** like every other split file |
| D3 | **Fine-tune-on-SSA arm**: start from the BraTS checkpoint, fine-tune on the SSA training slice | ~8 GPU-h | The cheap, deployable answer; the literature says this recovers most of the gap |

Pre-register the endpoint: per-cohort Dice and lesion-wise Dice against the current BraTS-only model,
paired, Holm-corrected across the family.

**Write the answer to success in advance,** because it will be asked in review: if pooled training
closes the gap, why does anyone need a refusal gate? Because no training set covers every deployment
shift — detection and coverage are complements, not substitutes. That sentence belongs in the paper,
not in a rebuttal.

**Cut order if hours run short:** D1 from 3 seeds to 2, then D3. Never D2.

---

### Phase E — The clinical front-end: real MRI in
*Weeks 6–12 · CPU*

**Goal.** The difference between a demo and a tool. Today `app/backend/jobs.py` accepts four NIfTIs
and runs `preprocess_case` — reorient to LPS, per-channel nonzero z-score, crop to nonzero bbox. It
does not skull-strip, co-register, bias-correct, or accept DICOM, so a real hospital scan produces a
plausible-looking wrong answer with nothing failing.

| # | Component | Implementation |
|---|---|---|
| E1 | **DICOM ingest** — a study folder to four named NIfTIs | `dcm2niix`, with series assignment from DICOM metadata plus a rule table and a manual override in the UI. A learned sequence classifier (published at 98–99% on brain MRI) is a Phase-G stretch, not required for v1 |
| E2 | **Preprocessing** — co-registration → SRI24 atlas registration → HD-BET skull-strip → optional N4 | `brainles-preprocessing`'s `AtlasCentricPreprocessor`, pinned. New `src/neurovision/data/clinical_preprocess.py` wraps it so the research path is untouched |
| E3 | **Input QC gate** — geometry, spacing, sequence completeness, brain-mask sanity, intensity-range sanity, rejected with a specific and actionable message | Reuse the validation idiom already in `jobs.py::create_job`, where a `ValueError` surfaces as HTTP 400 rather than 500 |
| E4 | **Missing-sequence handling** | v1: detect and **refuse with a named reason**. Synthesis (the BraSyn line of work, built on our exact BraTS 2021 cases) is a Phase-G stretch |
| E5 | **Refusal gate** — input QC + case-level OOD score + QC-model predicted Dice + conformal band width, combined into PROCEED / PROCEED WITH CAUTION / REFUSE, with the reason surfaced | new `src/neurovision/inference/gatekeeper.py`; thresholds calibrated on val and frozen |
| E6 | **DICOM-SEG export** | `highdicom` / `dcmqi`, so output loads in OHIF or any PACS |
| E7 | **UI** — bounded mask, QC estimate, refusal banner, existing report panel | `app/frontend/`; the existing E2E harness asserts on rendered pixels, so extend it rather than replacing it |

**Framing, enforced in code and copy:** research / education / decision-support. Not diagnostic. The
report artifact already carries its disclaimer as a *required field* rather than as README text —
extend that same discipline to the viewer and to any exported DICOM object.

**Verification.** An end-to-end run on a real public DICOM brain-MRI study. Then a round-trip check:
take a BraTS case whose Dice is already published, push it through the *clinical* path, and confirm
the recomputed Dice lands within a stated tolerance. That is the discipline that caught the demo's
geometry re-crop (0.9851 recomputed against a published 0.9851) — a plausible-looking overlay proves
nothing; matching an already-published number does.

---

### Phase F — Downstream model: IDH status
*Weeks 10–16 · gated on Phases A–C being delivered*

The one downstream model with a signal that survives external validation.

| # | Task | Notes |
|---|---|---|
| F1 | Download UCSF-PDGM from TCIA — **only the four structural sequences plus the tumour segmentation**, not the 142 GB full collection (HARDI, ASL, SWI and the diffusion maps are not needed). `df -h` before and after | CC BY 4.0; cite DOI 10.7937/tcia.bdgf-8v37 |
| F2 | Preprocess through the same pipeline as BraTS; freeze `configs/data/splits_ucsf.yaml` | Source data is already skull-stripped and ANTs co-registered at 1 mm isotropic |
| F3 | IDH classifier: tumour-cropped 3D CNN over image + **predicted** mask | Literature external-validated AUC ~0.88–0.90 |
| F4 | **Control for age.** IDH-mutant patients are markedly younger, so a model can score well by learning age rather than imaging. Report AUC with and without age, plus an age-only baseline | This is the control that makes the result credible, and the analogue of the capacity control that made C2 credible |
| F5 | Abstention through conformal prediction — the classifier declines when the prediction set is not a singleton | Reuses Phase B machinery |

**Gate F.** Must beat the age-only baseline with a CI excluding zero. If it does not, report that —
it is a real and interesting negative — and do **not** ship the module.

---

### Phase G — End-to-end validation and the error budget
*Weeks 14–20*

The phase that makes this a contribution rather than a collection of parts. Almost nobody publishes
compounded end-to-end reliability for a clinical imaging pipeline; everyone benchmarks stages in
isolation.

| # | Task |
|---|---|
| G1 | Measure each stage's reliability independently on the same cases: ingest, preprocessing, segmentation, conformal coverage, QC estimate, refusal gate, report |
| G2 | Compute the **compounded end-to-end** success rate, and the rate once the refusal gate is allowed to abstain — the central number of the paper |
| G3 | Coverage/accuracy trade-off at the pipeline level: as the refusal rate rises, what happens to the accuracy of what is accepted? |
| G4 | Run the whole pipeline on all four cohorts — BraTS test, SSA, PED, and UCSF-PDGM as a genuinely unseen fourth distribution |
| G5 | Failure-mode taxonomy: every way the pipeline failed, with counts |
| G6 | Stretch: learned sequence classifier (E1), missing-modality synthesis (E4) |

**Deliverable.** One table stating, per cohort: the pipeline produced a usable bounded result for X%
of studies, correctly refused Y%, and of those it accepted, Z% met the guarantee.

---

### Phase H — Write-up and release
*Weeks 18–24*

`notebooks/09_paper_figures.ipynb` already regenerates every figure and table from result files in one
run and prints an audit of what it skipped and why — point its `RUNS` manifest at the new
directories and the figures follow.

**Paper structure — the negatives lead, the positives are earned.**

1. A pipeline with a distribution-free guarantee on its own error and an explicit refusal state.
2. That guarantee's validity measured under two real distribution shifts — the novel result.
3. A pre-registered study of what a small segmentation gain does *not* buy (nine comparisons), with
   Gate A settling how large the gain really is against a properly configured baseline.
4. Three methodology findings already measured: a calibration mask must never be defined using the
   ground-truth label (the original inflated reported ECE by 41–57%); boundary-error shares must be
   weighted by voxel count rather than rate (correcting this project's own figure from 92% to 74%);
   and the standard brain-mask-Dice check for a mirrored atlas scores *higher* on the mirrored version
   (0.9416 vs 0.9394), making it blind to the worst error it exists to catch.

**Venues.** MICCAI **UNSURE** workshop — the 2026 deadline was 8 July 2026, so target 2027; accepted
papers are invited to extend into a MELBA special issue. Also MIDL, and MELBA directly. Plus BraTS
2027 Challenge 3 (generalizability) if the 2026 window has closed.

**Release.** Model card, one `docs/` page per pipeline stage, MkDocs build, archived DOI.

---

## 6. Cut list — decided, so they are not re-litigated

MGMT methylation · survival prediction · WHO grade · midline shift · the eloquence verdict · the
remaining four ablation rungs · the full 96³ retrain · `baseline_swinunetr` (demoted) ·
patient-facing diagnostic framing · any new fusion-architecture variant.

Reasons are in §3. Re-opening any of these requires new evidence, not a new mood.

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| nnU-Net beats `neurovision` at Gate A | Moderate–high | Pre-registered in advance, so it reads as rigour rather than defeat. No other phase depends on the architecture claim surviving |
| Conformal coverage collapses under shift | Moderate | That *is* the finding, and it motivates the refusal gate. The weighted/Mondrian variant is the constructive answer |
| **Scope explosion — the top killer** | **High** | Phases A–C are the floor and are CPU-heavy. F and G are explicitly gated. The cut order is written down and is not renegotiated mid-phase |
| ANTs / HD-BET install pain on macOS ARM | Moderate | Isolated in `requirements-clinical.txt`; the research path never imports it. If ARM builds fail, the clinical path runs on the Linux GPU box |
| UCSF-PDGM disk blowout (142 GB full collection) | Moderate | Series-filtered download only; `df -h` before and after; the preprocessed BraTS dataset is load-bearing and must not be evicted |
| A new stage silently moves an old number | Moderate | Principle 4, plus the existing additive-columns test pattern |
| Cluster preemption | Moderate | Resume already works end to end (hard constraint 4); `GIT_REF` pinned to a SHA; checkpoints copied off the box before the session ends |

---

## 8. Dependency and data plan

**New Python dependencies**, approved 2026-08-23, all isolated from the training environment:

| File | Contents | Used by |
|---|---|---|
| `requirements-analysis.txt` | `panoptica`, `statsmodels` | Phases A, B — CPU analysis only |
| `requirements-clinical.txt` | `brainles-preprocessing`, `antspyx`, `HD-BET`, `dcm2niix`, `highdicom` | Phase E — the clinical path only |
| separate env on the GPU box | `nnunetv2` | Phase A7 only; never installed into the project environment |

Root `requirements.txt` is unchanged, so any Kaggle fallback session still installs exactly what it
installs today.

**New data.** UCSF-PDGM, series-filtered (Phase F). Existing `data/preprocessed/{brats, brats_ssa,
brats_ped}` stays and remains load-bearing — `brats` is backed only by the live Kaggle dataset
`amishyadav123/neurovision-brats-prep`. New splits (`splits_pooled.yaml`, `splits_ucsf.yaml`) are
frozen on creation like every existing split file.

---

## 9. Timeline

| Month | CPU track | GPU track |
|---|---|---|
| 1 | A1–A6, B1 | A7 strong baselines |
| 2 | B2 conformal under shift, C1–C2 | D1 multi-seed |
| 3 | C3–C5 QC model, B3 ensemble comparator | D2 pooled, D3 fine-tune |
| 4 | E1–E4 clinical front-end | F1–F3 IDH, if gated in |
| 5 | E5–E7 refusal gate, UI, DICOM-SEG | F4–F5, spare capacity |
| 6 | G end-to-end validation, H write-up | buffer for failed runs |

---

## 10. Verification — applies to every phase

| Level | Check |
|---|---|
| Unit | plain `pytest` from the repo root (1,630 collected). `pyproject.toml` already sets `addopts = "-q"`; a second `-q` stacks to `-qq` and silently drops the pass count |
| Integration | `python scripts/smoke_test.py` exits 0 before every GPU session |
| Frontend | `cd app/frontend && npm test`, then `npm run test:e2e` against a live backend before any demo |
| New model code | CPU shape test on `(1, C, 32, 32, 32)` running under one second |
| Analysis | re-run the real analysis and confirm the output moved in the predicted direction. Unit tests are not sufficient and have already given a false green on this exact class of bug |
| Additivity | run with and without the new feature; assert frame equality on the shared columns |
| GPU session | pinned SHA in the log, checkpoint copied off the box, log retained, W&B online |

---

## 11. What success looks like in six months

A tool that ingests a real DICOM brain-MRI study, refuses what it should refuse with a stated reason,
and otherwise returns a tumour segmentation carrying a distribution-free bound on its miss rate, an
estimated quality score, and a structured anatomical report — validated end to end on four cohorts,
two of them external, with the compounded error budget published rather than assumed.

Plus a paper whose central claim is a *measured safety property* rather than a Dice delta, and in
which nine negative results are the evidence for that claim rather than an embarrassment buried
inside it.
