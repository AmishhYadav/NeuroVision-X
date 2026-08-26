# Pre-registration — can a segmentation-QC model predict its own error, and does it survive shift?

**Written:** 2026-08-24, **before** `scripts/train_qc.py` has been run on real data even once, before
any QC checkpoint exists, before `scripts/validate_qc.py` exists, and before a single predicted-Dice
number has been computed for any case in any cohort. The git timestamp on this file is the evidence
for that ordering.

**Scope.** This registers **Phase C** of `docs/research/master_plan.md` — C3 (the training run), C4
(per-cohort validation, which is **Gate C**) and C5 (the silent-failure test).

---

## Background, and why this phase has a positive prior

Phase C is the third of the Milestone 4 phases chosen because the literature says a positive outcome
is achievable, rather than because a downstream consequence of a +0.0267 Dice gain sounded plausible.
Segmentation-QC networks — a second model that reads an image and a mask and regresses the mask's
own Dice with no ground truth available — are a replicated result, reported at 95–99% good/bad
separation across several anatomies. The master plan's stated literature bar is Spearman **ρ > 0.7 in
distribution**.

But the honest question is not whether a QC model works. It is whether it beats **the free
alternative**. This project has already learned that lesson at cost: note 39 showed branch
disagreement is *worse* than free single-pass predictive entropy as an error localiser, and note 44
showed the trained confidence head is beaten by that same free entropy on all three regions. A third
learned uncertainty signal that loses to `ent_mean_fg_R` would be the third instance of one pattern,
and it must be given the same chance to lose.

So the comparator is fixed here, before the QC model has produced a number: **mean predicted-foreground
entropy**, already computed and cached for all three evaluation cohorts in
`outputs/detection/entropy_cache_*.csv`, costing zero additional compute.

The second question — C5 — is one the QC literature mostly does not answer: **does the QC model
itself fail silently under distribution shift?** A QC model that reports "this mask is fine" on a
cohort where the segmentation is in fact broken is worse than no QC model, because it converts a
visible failure into an invisible one. We hold two shifted cohorts with ground truth. Measuring this
costs nothing and is a contribution whichever way it lands.

---

## What is trained, fixed now

| | |
|---|---|
| Architecture | `SegQC` (`src/neurovision/models/qc.py`), registered as `segqc`, 3-channel input |
| Input channels | `(t1ce, mask, entropy)`, in that order, whole volume resized to 64³ |
| Target | that region's Dice against the label, computed at **full resolution** before any resize |
| Training masks | degradations of the **deployed model's own predictions**, reconstructed from `outputs/neurovision/eval_val/logits`. Never ground-truth masks — master plan principle 3 |
| Training cases | the frozen **val** split, n = 187, minus the selection slice below |
| Selection slice | `analysis.qc.val_frac = 0.2` of those cases, **case-level and seeded**, disjoint from the fitting cases. 63 degradations of one case share a volume, so a sample-level split would put near-duplicates on both sides |
| Checkpoint used for every number below | `best.pt`, the epoch with the highest Spearman **on the selection slice** — a split that is neither test, nor SSA, nor PED |
| Regions | ET, TC, WT, one region per training sample |

**The test split is not used for model selection.** `analysis.qc.heldout_eval_dir` defaults to `null`
for exactly this reason, and stays `null` while this gate is open.

---

## The endpoints, fixed now

Every number below is computed on the **real deployed prediction** — the identity degradation, i.e.
the mask the model actually produces — not on a synthetic degradation. Degraded pairs are how the QC
model is *trained*; they are not what it is *scored* on, because at deployment there is exactly one
mask and it is the model's own.

### Primary endpoint (Gate C)

For each cohort × region cell:

- **AUROC_QC** — case-level AUROC for the event `true Dice < 0.7`, with score `−predicted_dice`
  (a lower predicted Dice should indicate a worse case).
- **AUROC_ent** — the same AUROC with score `ent_mean_fg_R`, the free baseline.
- **ΔAUROC = AUROC_QC − AUROC_ent.**

ΔAUROC is the primary endpoint. **Not** AUROC_QC on its own: an AUROC of 0.85 that the free baseline
also reaches is not a result.

### Secondary endpoints

- **Spearman(predicted Dice, true Dice)**, per cohort × region — against the literature bar ρ > 0.7,
  read on the in-distribution cohort.
- **MAE** = mean |predicted − true|.
- **Signed bias** = mean(predicted − true). Sign matters and is not a nuisance: a positive bias means
  the QC model *over-states* mask quality, which is the dangerous direction.

### C5, the silent-failure endpoints

- The change in Spearman and in signed bias from the in-distribution cohort to each shifted cohort.
- **Directional prediction, fixed now:** signed bias will be **more positive on PED than on the
  in-distribution test split** — the QC model will over-estimate Dice where the segmentation model is
  worst. Recorded so that the opposite result is a real surprise rather than a retrofitted narrative.

---

## Design, fixed now

| | |
|---|---|
| Segmentation model | `neurovision` only. The QC model is a property of a deployed model, and only one model is deployed |
| Cohorts | BraTS **test** n = 189 (in-distribution), BraTS-Africa **SSA** n = 60, BraTS-PEDs **PED** n = 99 |
| Inputs | saved fp16 logits, the preprocessed images, and the labels. No inference, no GPU |
| Threshold defining a bad case | Dice < 0.7, taken from the master plan and not tuned |
| Randomness | the bootstrap only, seeded at 42 |
| Bootstrap | 10,000 replicates, resampling **case indices** with replacement, percentile CI at 95% |
| Pairing | QC and entropy are scored on **the same resampled case indices in every replicate**. They are two scores for one set of cases, so an unpaired interval would inflate the variance of their difference |

### Which cells enter the gate family

A cohort × region cell with **fewer than 5 positive cases** (`true Dice < 0.7`) is reported but is
**excluded from the gate family**: an AUROC estimated from four positives carries an interval so wide
that including it only costs the family a multiplicity correction.

The positive counts are label-side facts about already-published `per_case_metrics.csv` files, so
they are known at the time of writing and are stated here rather than discovered later:

| Cohort | ET | TC | WT |
|---|---|---|---|
| test (n=189) | 17 | 11 | **1** |
| SSA (n=60) | 10 | 12 | **3** |
| PED (n=99) | 45 | 71 | 11 |

Applying the rule: **test·WT** and **SSA·WT** are excluded. The gate family is the five *external*
cells — SSA·ET, SSA·TC, PED·ET, PED·TC, PED·WT. The two in-distribution cells (test·ET, test·TC) are
reported and are the source of the ρ > 0.7 comparison, but Gate C is defined on external cohorts, so
they are not in the family.

### Multiplicity

Bootstrap two-sided p-values for `ΔAUROC ≠ 0`, Holm–Bonferroni corrected across the **five** family
cells. `analysis/statistics.py::holm_bonferroni` already implements this and is reused.

---

## The decision rule, fixed now

**Gate C fires POSITIVE** if, in at least one of the five family cells, ΔAUROC > 0 **and** its
Holm-corrected p < 0.05 **and** its 95% bootstrap CI excludes 0.

**Gate C fires NEGATIVE** otherwise. Per the master plan: on a negative, the QC estimate still ships
as a displayed number in the viewer, and **makes no claim** — it may not be described as detecting
failures, and no sentence asserting that the QC model adds anything over entropy may enter the paper.

There is no third outcome and no re-scoring on a different threshold, a different region set, or a
different score direction. If ΔAUROC comes back reliably *negative* — the QC model losing to free
entropy — that is a completed deliverable and the third replication of this project's most robust
empirical pattern, and it will be written up as such.

---

## The falsification check that runs before any endpoint

`scripts/validate_qc.py` must, for every cohort, recompute each case's identity-pair Dice — the true
Dice of the undegraded prediction, as the QC pipeline reconstructs it — and compare it against the
already-published `dice_R` in that cohort's `per_case_metrics.csv`.

**If the median absolute difference exceeds 0.01 for any region, the script raises and no endpoint is
reported.** The two numbers are the same quantity computed through two independent paths (the
evaluation pipeline, and the QC pair generator reading saved logits), so a disagreement means one of
the paths reconstructs a different mask than it claims to — a different threshold, a different
post-processing rule, or a geometry mismatch.

This is the discipline that caught the demo's geometry re-crop, where a plausible overlay proved
nothing and matching an already-published 0.9851 proved everything. A QC model regressed against a
Dice that is not the published Dice would produce a table of entirely real-looking numbers about a
mask nobody ever evaluated.

---

## What would make this pre-registration void

- Training the QC model on ground-truth masks rather than degraded predictions.
- Selecting the checkpoint on any of test, SSA or PED.
- Changing the 0.7 threshold, the ≥5-positive rule, the family definition, or the score direction
  after seeing any number.
- Reporting AUROC_QC without ΔAUROC beside it.

---

## Result

**Run:** `scripts/validate_qc.py model=segqc`, 2026-08-26. Falsification check passed for all three
cohorts (identity-pair Dice matched the published `dice_R` in every case; see
`outputs/qc_validation/falsification.csv`). Full table: `outputs/qc_validation/cells.csv`. Verdict
record: `outputs/qc_validation/gate_c_verdict.json`.

**Gate C fires POSITIVE.** Exactly one of the five family cells clears the pre-registered bar:

| Cohort | Region | ΔAUROC | 95% CI | p_holm |
|---|---|---|---|---|
| **PED** | **TC** | **+0.1686** | **[0.0686, 0.2747]** | **0.006** |
| SSA | ET | +0.0375 | [−0.1096, 0.1758] | 1.0 |
| SSA | TC | −0.1951 | [−0.3551, −0.0429] | 0.050 |
| PED | ET | +0.0290 | [−0.0959, 0.1556] | 1.0 |
| PED | WT | −0.1898 | [−0.3312, −0.0667] | 0.010 |

**This is a mixed result, not a clean win, and must be written up as one.** The QC model beats free
entropy on PED·TC by a wide, clearly significant margin. But in the same five-cell family, it *loses*
to free entropy — significantly, in the opposite direction — on SSA·TC and PED·WT. The decision rule
fixed in advance asks only whether *at least one* cell clears the bar, and one does, so the gate is
POSITIVE by that rule. The honest sentence for the paper is: **the QC model adds real, significant
value on tumour-core detection in the paediatric cohort, and is significantly worse than the free
baseline on two other cells in the same family** — not "the QC model works" and not "the QC model beats
entropy." In-distribution (test), no cell reaches the bar at all (not in the gate family regardless,
reported for the ρ > 0.7 comparison only) — spearman_qc is 0.45–0.57 there, below the literature bar,
though this project's own free-entropy baseline is already unusually strong in distribution, which is
the same pattern note 39 and note 44 established for the other two learned-uncertainty attempts.

**C5, the silent-failure test, is unambiguous and the more important number.** In **all six** external
cells (SSA and PED × ET/TC/WT), the QC model's bias shifts *more positive* than its in-distribution
bias (`bias_more_positive_than_in_distribution = True` everywhere, `outputs/qc_validation/silent_failure.csv`).
The largest shift is PED·TC: bias moves from −0.070 in distribution to +0.228 under shift, a
Δ of +0.298. **Under distribution shift, the QC model systematically becomes more optimistic about
mask quality, exactly when it should become more cautious.** This is the failure mode C5 was
registered to look for, and it is present everywhere it was measured. It does not reverse Gate C's
verdict (the pre-registered rule is silent on bias direction), but it is the dominant fact for the
refusal gate: `enabled_signals` may include `predicted_dice`, per Gate C firing POSITIVE, but any
document or UI surfacing the QC model's number to an external-cohort case must carry this bias warning
alongside it, and the gate's calibrated thresholds must not assume the QC model is honest about a case
that looks like SSA or PED.

**Consequence for `configs/clinical/default.yaml`'s `gatekeeper.enabled_signals`.** `predicted_dice` is
now enabled — Gate C fired POSITIVE, per the master plan's binding rule. The claim carried alongside it
in any report or paper text is exactly the two paragraphs above, never "the QC model detects failures"
unqualified.
