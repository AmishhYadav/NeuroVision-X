# Improvement Plan — from "a model with nulls" to a contribution

Written 2026-08-19. Supersedes nothing; read alongside `docs/experiments.md`
notes 11–25 and `docs/research/contribution.md`.

---

## 0. The problem this plan solves

The project currently reads as: a dual-encoder model that wins +0.0267 ET Dice
over a parameter-matched control, plus five null results (calibration, boundary
accuracy, MC-dropout risk-coverage, report agreement, external transfer of the
Dice gain). That is a defensible *paper* and a weak *project*. Beating SwinUNETR
on BraTS is not reachable on 30 rationed GPU hours a week and should not be
attempted.

The differentiator is not accuracy. It is a **capability a single-encoder model
cannot have by construction**, plus the fact that this repo already measures
things almost nobody measures.

---

## 1. The reframe

**Current framing (weak):** "Dual encoder + gated fusion gives better Dice."
Reviewer response: marginal, and fusion gating is heavily published.

**New framing (the plan):**

> A dual-encoder segmenter carries two *independent readings* of every voxel —
> a local-texture reading (CNN) and a global-context reading (transformer).
> Where those readings disagree is a measurable, per-voxel signal that a
> single-encoder network cannot produce at all. We show that this
> **inter-branch disagreement** is a usable uncertainty estimate: it is free
> (one forward pass, no sampling), it detects segmentation failure better than
> MC-dropout at 10x the cost, and — unlike in-distribution calibration metrics —
> its value *grows* as the test distribution shifts away from training.

Three properties make this a real contribution rather than a relabelling:

1. **Structural exclusivity.** `BranchAmbiguity` produces `|p_cnn - p_swin|`
   plus both branches' Bernoulli entropies from *detached* probe convs. A
   U-Net, SwinUNETR, or any single-backbone model has no second opinion to
   compare against. The architecture is justified by what it can *measure*,
   not by what it scores.
2. **Different uncertainty family.** MC-dropout estimates *weight* uncertainty;
   predictive entropy estimates *outcome* uncertainty. Branch disagreement
   estimates **representational** uncertainty — "the two ways of looking at
   this image do not agree here". No sampling, no ensemble, no extra passes.
3. **It fills the exact hole the nulls left.** The reliability claim died
   because MC-dropout is a weak estimator competing against temperature
   scaling, which is one scalar fit in seconds. Disagreement is not competing
   on calibration; it competes on **failure detection under shift**, where
   temperature scaling has nothing to offer.

**Deliverable framing for the capstone:** not "a segmentation model", but
**a segmentation system that knows when it should not be trusted**, verified on
three cohorts of increasing distribution shift.

---

## 2. Why this is credible before we measure it

Not speculation — the supporting evidence is already on disk.

- `outputs/compare_ssa/neurovision_vs_baseline_ssa.csv` (n=60, BraTS-Africa):
  Dice is flat (ET −0.0008, WT −0.0016) but **HD95 favours `neurovision` in all
  three regions** — ET −2.22 mm, TC −2.05 mm, WT −1.51 mm, `dz` −0.16 to −0.18,
  `hd95_mean` p_wilcoxon 0.117. Three of three in the same direction with
  consistent effect size at n=60 is an underpowered signal, not an absent one.
- `outputs/eval_ped_baseline_unet3d/summary.csv` (n=99, BraTS-PED):
  ET Dice **0.573, std 0.356**. That is catastrophic, bimodal failure — exactly
  the regime a referral system exists for, and a cohort where "know when you
  don't know" has obvious clinical meaning.
- The trained checkpoint has working probes: `use_ambiguity: true`,
  `training.loss.multitask.branch.enabled: true`, `weight: 0.1` (verified in
  `outputs/neurovision/eval_test/eval_config.yaml`).

So the story has in-distribution nulls, an out-of-distribution signal, and a
mechanism that is already trained and costs nothing to read out.

---

## 3. Phases, with decision gates

Every phase states its cost, its gate, and what to do if the gate fails. **Do
not proceed past a failed gate by rationalising it** — that is what produced the
current situation.

### Phase A — Salvage and measure what already exists (zero GPU, ~1 week)

| Step | Work | Cost |
|---|---|---|
| A1 | Finish the interrupted PED eval for `neurovision` — currently **19/99 cases** in `outputs/eval_ped_neurovision/` (no `summary.csv`, no `eval_config.yaml`). Re-run to completion. | ~1 h CPU |
| A2 | Paired comparison on PED (n=99) via `analysis/statistics.compare_models`. Then **pooled** SSA+PED (n=159) on HD95 per region. | minutes |
| A3 | New script `scripts/extract_ambiguity.py` — sliding-window extraction of the branch-disagreement field over a whole volume, saved per case. | ~2 h build, ~1.5 h/split CPU |
| A4 | Score disagreement as a **voxel-level error detector**: AUROC / AUPRC of `disagreement` vs `pred != label`, against three comparators — predictive entropy (1 pass), MC-dropout mutual information (N=10), and a random null. | ~1 h |

**Gate A-1 (robustness):** does HD95 favour `neurovision` on the pooled n=159
with a Holm-corrected CI excluding zero?
- *Pass* → the robustness claim is real. It becomes contribution #2.
- *Fail* → drop the HD95 claim entirely. Phase A4 still stands alone.

**Gate A-2 (the core bet):** does disagreement beat MC-dropout mutual
information at error detection, at 1/10 the compute?
- *Pass* → this is the paper's headline. Continue to Phase B.
- *Partial* (comparable, not better) → still publishable as "equal quality at
  1/10 cost, single pass", which is a real efficiency claim. Continue.
- *Fail* (disagreement is uninformative) → see §5 fallback. Do not continue to
  Phase B.

**Known risk on Gate A-2:** the branch-supervision loss trains *both* probes
toward the same label, so the branches may have learned to agree everywhere and
the map may be near-degenerate. Check this first and cheaply: extract on ~10
cases, plot the disagreement histogram and its spatial distribution. If it is
flat/zero everywhere, stop before building the full pipeline. If it concentrates
at boundaries and on hard cases, proceed — that concentration *is* the result.

---

### Phase B — Build the referral system (zero GPU, ~1 week)

The deliverable that makes this a project and not a table.

| Step | Work |
|---|---|
| B1 | **Case-level failure detection.** Aggregate the per-voxel disagreement field into a per-case scalar (mean over predicted foreground, 95th percentile, and foreground-volume-weighted — pick by val, report all three). Score AUROC for "this case will have Dice < 0.7". |
| B2 | **Risk-coverage under shift.** Reuse the existing `scripts/calibrate.py` risk-coverage plumbing, but drive it with the disagreement score on all three cohorts. Report oracle ceiling and random null alongside, as that module already requires. |
| B3 | **Referral table.** "Refer the top X% most-uncertain cases to a human; remaining-case Dice rises from A to B." This is the clinically legible artifact and the demo's centrepiece. |
| B4 | **Demo integration.** New overlay in `app/frontend` for the disagreement field, plus a case-level trust banner. Reuse the `X-Uncertainty-Kind` header discipline — the layer must be labelled `branch disagreement · representational`, never as epistemic or MC-dropout uncertainty. |

**Gate B:** does referral on disagreement beat referral on entropy and on
MC-dropout MI, on SSA and PED specifically? The in-distribution result is
expected to be weak (it already is — that is note 17); the claim is that the
gap opens up under shift.

---

### Phase C — Spend the GPU hours (~25 h, one experiment)

Only one training run is affordable. It must serve the new story.

**Run: `ablation_content_only_gate`** (~23 h, 3 chained sessions, already
specced in `configs/experiment/ablation_content_only_gate.yaml`, parameter-matched
to 0.018%).

It now does **double duty**, which is why it is the right spend:

1. It is the pre-registered P2 rung that tests whether the ambiguity
   conditioning — rather than the mere existence of a gate — carries the
   architectural gain. Without it the contribution is argued from design, not
   measured.
2. Under the new framing it is the **direct control for the uncertainty claim**:
   `use_ambiguity: false` removes the disagreement signal from the gate, so this
   run answers "is the disagreement map useful *because* the model was trained
   to consume it, or would any dual-encoder produce it anyway?"

Pre-run checklist (from `docs/experiments.md` and CLAUDE.md):
- Pin `GIT_REF` to a **SHA**, not `main`. Clone the pinned tree and assert the
  fp16-entropy fix is in it before launching.
- `grad_clip_norm: 5.0` — must match the `neurovision` run (it does; verified in
  `eval_config.yaml`). Do not change it mid-comparison.
- Run `scripts/smoke_test.py` before the session.
- Fetch and keep the kernel log this time (the capacity control's hours are
  approximate because it was not fetched).

**If Phase A's gates both fail**, do not spend these hours here — spend them on
multi-cohort training (§5).

---

### Phase D — Cheap accuracy and robustness wins (zero GPU, opportunistic)

Run these in parallel with Phase C's GPU sessions. None need a GPU.

| Step | Work | Expected |
|---|---|---|
| D1 | **Flip TTA** — already committed (`a800bfb`), never measured. | +0.003–0.008 Dice, and usually better-calibrated than a single pass |
| D2 | **Postprocess re-ablation from saved logits.** Free: predictions already reproduce Dice exactly from logits. Target the 10 mm+ band — 1,047 false-positive voxels vs 5 false-negative, surviving `min_component_size: 50`. Add an **uncertainty-gated component filter** (drop a component whose mean disagreement exceeds a threshold). | Removes a distinct, addressable failure mode, and gives disagreement a second job |
| D3 | **Confidence-head evaluation.** The head was trained (weight 0.05) and its output was **never scored**. Calibration was measured on segmentation probabilities only. AUROC of confidence vs per-voxel error. | Either a fourth uncertainty comparator, or an honest "the head learned nothing" note |
| D4 | **Inference ROI sweep** on existing 64³ checkpoints — separates train-patch effect from inference-window effect on the multifocality over-reporting (30.7–40.7% vs true 22.8%). | Explains the Phase 5 negative mechanically |

---

## 4. What the paper claims when this is done

Ordered by strength. Every one is either already measured or gated above.

1. **Inter-branch disagreement is a free representational-uncertainty estimate**
   that matches or beats MC-dropout at failure detection for 1/10 the compute,
   and is structurally unavailable to single-encoder models. *(Gate A-2)*
2. **Its value grows with distribution shift** — near-useless in-distribution
   (consistent with our own nulls), useful on BraTS-Africa, most useful on
   BraTS-PED. *(Gate B)*
3. **Architecture, not capacity, and measured that way.** +0.0211 ET over a
   parameter-matched 34.83 M control (p_holm 7.3e-19); the gain decomposes
   20.6% capacity / 79.4% architecture. Very few fusion papers run this control.
   *(Done)*
4. **Boundary robustness under shift**, if Gate A-1 passes. *(Gated)*
5. **Reliability nulls, obtained rigorously** — architectural uncertainty
   mechanisms do not beat one scalar of temperature scaling in-distribution
   (baseline ECE 0.0565 → 0.0158). Reported as a finding, not buried.
6. **Methodology audit** — the calibration reporting mask must not be defined
   using the label (`union_foreground_mask` inflated ECE 41–57% and produced
   MCE ≈ 0.99 that was pure artifact); boundary-error shares must be
   voxel-weighted, not rate-normalised (92% → 74%); brain-mask Dice is
   structurally blind to an L–R mirrored atlas (0.9416 mirrored vs 0.9394
   correct).
7. **The interpretable report layer is invariant** to segmentation quality
   across ET Dice 0.85–0.87 — a deployment-stability result, stated as such.

Contributions 5–7 are what make this defensible rather than lucky. Do not drop
them to make the story cleaner.

**Venues that take exactly this shape:** MIDL (short paper), the MICCAI
**UNSURE** workshop (uncertainty for safe ML in medical imaging), and **MELBA**,
which explicitly accepts negative and replication results.

---

## 5. Fallbacks, by which gate failed

- **Gate A-2 fails (disagreement is degenerate).** The dual encoder gives no
  usable second opinion. Fall back to the **methodology paper**: contributions
  3, 5, 6, 7 above. Spend the GPU hours on **multi-cohort training** instead
  (pool BraTS-2023 GLI + SSA + PED) and claim generalisation directly rather
  than measuring it — "cross-cohort training closes the shift gap by X" is a
  real, if less novel, result.
- **Gate A-1 fails, A-2 passes.** Drop the HD95 robustness claim. The
  uncertainty story stands alone and is still the headline.
- **Both pass.** Proceed as written; this is the strong version.
- **Both fail.** Contributions 3, 5, 6, 7 still constitute a complete honest
  paper and a working demo. That is the floor, and the floor is already built.

---

## 6. Sequencing

```
Week 1   A1 A2 A3 A4          -> Gate A-1, Gate A-2
Week 2   B1 B2 B3 B4          -> Gate B          (parallel: D1 D3)
Week 3   Phase C session 1-2  (parallel: D2 D4)
Week 4   Phase C session 3, evaluate ablation on CPU
Week 5+  figures, tables, write-up
```

Phase C is the only item that consumes GPU hours. Everything else runs on the
Mac. That is deliberate — the budget is spent, and the plan is built so that a
failed gate costs days, not GPU sessions.

---

## 7. First action

Finish `outputs/eval_ped_neurovision/` (19/99 cases, interrupted). Everything in
Phase A depends on having both models scored on both shifted cohorts.
