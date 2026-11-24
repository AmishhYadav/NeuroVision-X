# Pre-registration — case-level failure detection: which label-free signal finds the bad mask?

**Written:** 2026-11-24, before any ensemble-disagreement or ensemble-entropy number exists and before
any signal has been scored against the W2.5 volume endpoint. MasterPlan W2.3; analysis-budget family
3 of 6 (§B3 rule 4). **External registration:** Zenodo DOI [10.5281/zenodo.24491280](https://doi.org/10.5281/zenodo.24491280), deposited 2026-11-24: snapshot of tag `semester-2026` (commit `894a607`). The protocol as registered is the copy in that snapshot; only this line was added afterwards.

**What is already known, stated so it cannot be re-discovered as a result:** Gate C (note 52,
`preregistration_qc.md`) scored SegQC against foreground entropy on the Dice < 0.7 event, per region,
and was MIXED (QC better in one cell of five, worse in one). The OOD score (note 56) separates SSA
from BraTS as a cohort but does not rank failed masks. Those AUROCs exist; this protocol does not
re-test them as new findings. What is new: (i) the case-level target is the pipeline's own "usable"
definition (both WT and TC), not one region at a time; (ii) a second target, W2.5's volume endpoint;
(iii) two ensemble signals that need no training; (iv) area under the risk–coverage curve beside
AUROC; (v) one Holm family across everything.

---

## Why this run exists

The refusal gate is only as good as the signal it thresholds. The paper's §B2 item 6 must say which
label-free signal best separates usable from unusable masks, in distribution and under shift — and
whether the ranking survives the shift. Paper Table XII (gate detection) mixed endpoints from three
analyses run at different times; this replaces it with one table from one run.

## Design

**Cohorts and cases.** BraTS 2021 test (n = 189), BraTS-Africa SSA (n = 60), BraTS-PEDs PED (n = 99)
— every case of each, as in every other table. Nothing in this protocol is fitted on any of them.

**The audited model.** `neurovision` seed 42 (deployed). Its deployed mask (sigmoid → threshold 0.5 →
deployed post-processing, `configs/inference/default.yaml`) is the mask whose usability is
predicted. nnU-Net enters through W2.6 (entropy and SegQC transferred, stated as such), not here.

**Signals (all computed without the label; higher score = predicted worse):**

| ID | Signal | Score |
|---|---|---|
| ENT | mean foreground entropy of the deployed model (the free baseline) | `ent_fg` as in `analysis/detection.case_entropy_scalars` |
| QC | SegQC predicted Dice (deployed `outputs/neurovision/qc/best.pt`, trained on val only) | `−min(predicted_dice_WT, predicted_dice_TC)` |
| ENS-PD | pairwise mask disagreement across the three `neurovision` members (seed 42, seed 43, heavy-aug), each with deployed post-processing | `1 − min_{R ∈ {WT,TC}} mean_{3 pairs} Dice_R(member_i, member_j)`; a pair where both masks are empty counts as Dice 1 |
| ENS-ENT | entropy of the three-member mean sigmoid probability | exactly ENT's convention applied to the mean probability: binary entropy / ln 2 (range 0–1), averaged over each region's own foreground (mean probability > 0.5), then averaged over the regions whose foreground is non-empty |
| OOD | input-statistics OOD score (frozen model, `results/ood/model.json`) | `ood_score` as saved |

**Empty foreground.** ENT and ENS-ENT are undefined (NaN) when no region has a predicted voxel. Such a
case is assigned **+∞** (ranked most suspicious) for that signal — a study that reaches the gate with
no predicted tumour is one a reader must look at — and the number of such cases per cohort is
reported. No case is dropped from any cell.

The three members already have saved fp16 logits on all three cohorts (`CLAUDE.md` logits table);
no inference. Seed 43 and heavy-aug are measurement arms, never deployment candidates — their use
here is as a disagreement probe only.

**Targets (fixed, per case, from the label):**

- **T-Dice** (primary): unusable ⇔ `dice_WT < 0.7` or `dice_TC < 0.7` (`error_budget.label_usable`,
  the pipeline's registered definition).
- **T-Vol** (secondary target, same family): unusable ⇔ volume error > 25% on WT or on ET
  (`analysis/volume_endpoint.py`, W2.5; empty reference: any predicted voxel is a failure).

**Metrics per (signal, cohort, target):**

1. **AUROC** for the event "unusable", score as above.
2. **AURC**: area under the risk–coverage curve, cases accepted in increasing score order, risk =
   unusable fraction among accepted; reported with the oracle AURC and the random AURC (= base rate)
   beside it so the number has a scale.

**Estimability rule.** A (cohort, target) cell with fewer than 5 unusable or fewer than 5 usable
cases is reported as "not estimable" and leaves the family (it is not replaced).

## The family, fixed

**Primary contrast:** ΔAUROC = AUROC(signal) − AUROC(ENT), for signal ∈ {QC, ENS-PD, ENS-ENT, OOD}, on
each of 3 cohorts × 2 targets → at most **24** comparisons, **one Holm correction**, α = 0.05.
Paired case-resampling bootstrap (n_boot 10,000, seed 42, stratified by the target so every resample
keeps both classes), two-sided p from the bootstrap distribution of Δ (proportion of resampled Δ on
the far side of 0, doubled), 95% percentile CI.

Verdict per comparison: **BETTER** (Δ > 0, CI excludes 0, Holm-significant) · **WORSE** (mirror) ·
**INCONCLUSIVE** otherwise (resolution-limited, never "equal").

**Secondary, descriptive (not in the family):** ΔAURC with the same bootstrap CIs; AUROC of each
signal alone with CI; the signal ranking per cell; the ENS-PD and ENS-ENT correlation with ENT
(Spearman) so a reader can see whether the ensemble adds information or re-expresses entropy.

**Registered predictions (directional, not gating):**

1. In distribution (test, T-Dice), ENS-PD ≥ ENT: three members disagree where the mask is
   ambiguous, and disagreement captures shape failures entropy smooths over.
2. OOD is the worst case-level detector in distribution (it was built to see cohorts, note 56) and
   not better than ENT on SSA.
3. On PED, every signal's AUROC is lower than on test for T-Dice, because the PED failure is a
   label-definition mismatch the models are *confidently* wrong about (note 58).

## Power line (§B3 rule 6)

AUROC standard error ≈ √(AUC(1−AUC)/min(n₊,n₋)) as an upper-bound heuristic. Counts under T-Dice from
the deployed error budget (`results/error_budget/per_case_*.csv`, checked 2026-11-24): unusable
**12 of 189** on test, **12 of 60** on SSA, **72 of 99** on PED (27 usable). The minority class is 12,
12 and 27. With paired signals on the same cases the Δ variance is smaller than the sum; resolvable
|ΔAUROC| after Holm over 24 is about **0.15 on test and SSA, about 0.10 on PED**. Gate C's measured
QC − ENT differences were of order 0.05–0.10, so most test and SSA cells are expected to be
INCONCLUSIVE; PED is the best-powered cohort. This is stated now so an all-INCONCLUSIVE table is read
as "not resolved at these n", never as "the signals are equivalent". T-Vol counts are unknown until
W2.5 runs; the estimability rule applies to them unchanged. *(Filled in 2026-11-24, after W2.5 and
before any detection number, from `results/volume_endpoint/neurovision_*/per_case_volumes.csv`
— class counts only, no signal was scored: T-Vol unusable **37 of 189** on test, **28 of 60** on SSA,
**58 of 99** on PED; minority class 37, 28, 41. Every T-Vol cell is estimable and better powered than
its T-Dice twin on test and SSA (resolvable |ΔAUROC| ≈ 0.10 rather than 0.15).)* The run launches because the plausible
ensemble-vs-entropy gap on shape failures (the PED and SSA TC failures are large) can exceed 0.10–0.15,
and because the AURC table is needed by W2.4 regardless of the Holm verdicts.

## What would make this invalid

- Choosing the ensemble members, the score definitions, the masking convention, the 0.7 / 25% bars or
  the family after any ENS-PD, ENS-ENT or T-Vol AUROC has been computed.
- Fitting anything (a threshold, a combination of signals, a calibration) on test, SSA or PED.
- Using any `neurovision_ft_*`, pooled or nnU-Net logits as an ensemble member.
- Reporting an INCONCLUSIVE contrast as "no difference".

## Cost

Ensemble signals: three members' saved logits replayed with deployed post-processing on 348 cases
(≈ 1,050 replays, ~2–3 s each) — about 1 CPU-h, one job. Everything else is arithmetic over saved
per-case tables.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first ENS-PD,
ENS-ENT or T-Vol detection number exists.)*
