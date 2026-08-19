# Pre-registration — inter-branch disagreement as a failure detector

**Written:** 2026-08-19, **before** `NeuroVisionX.forward_with_ambiguity` existed and
before any ambiguity map had been extracted from a trained checkpoint. The git
timestamp on this file is the evidence for that ordering, which is the entire
point of writing it down first.

**Scope.** This registers Gate 1 of `docs/research/execution_plan.md` — the test
that decides whether the project's pivot is real. It costs zero GPU hours.

---

## Background and motivation

`neurovision` beats a parameter-matched capacity control by +0.0211 ET Dice
(p_holm 7.3e-19) in-distribution, but five downstream hypotheses have failed:
calibration, boundary accuracy, MC-dropout risk-coverage, report agreement, and
out-of-distribution transfer — the last conclusively *reversing* on external
cohorts (`dice_TC` −0.0333, p_holm 0.0132, n=159; PED alone −0.0595, p_holm 0.0002).

The architecture computes a quantity a single-encoder model cannot: per-voxel
disagreement between the CNN and Swin branches, from two detached probe
convolutions inside `BranchAmbiguity`
(`src/neurovision/models/fusion/adaptive_fusion.py`). Today it is consumed by the
fusion gate and discarded. The proposal is that it is a failure detector — highest
where the input is unfamiliar, and therefore able to flag out-of-distribution
cases *before* ground truth is known.

---

## Why the obvious test would be wrong

The naive test is "is the disagreement map non-empty, and does it correlate with
error?" That test can pass while the idea is worthless.

Every segmentation model already emits a per-voxel confidence, and its predictive
entropy is high in exactly the places a disagreement map is expected to be high —
along the tumour margin. **A plain U-Net produces that map for free.** If
disagreement merely reproduces entropy, it is a duplicate of something the
baseline can also compute, and the claim that motivates the whole pivot — *only a
dual-encoder model can produce this* — is false regardless of how strong the raw
correlation looks.

The endpoint is therefore **incremental over entropy**, never absolute.

---

## Hypotheses

**H1 (primary).** Inter-branch disagreement carries failure-detection signal that
is **not** present in the single-pass predictive entropy of the same model.

**H0.** Disagreement is either spatially flat, or is a redundant re-encoding of
predictive entropy, adding no information about where or when the model fails.

A specific mechanism could make H0 true and is acknowledged in advance: the
branch-supervision term (`training.loss.multitask.branch`, weight 0.1) trains both
probes toward the *same* region labels, which may have driven the branches to
agree everywhere. If that has happened, the objective destroyed its own input.

---

## Endpoints, fixed before measurement

**Data.** All three cohorts, evaluated separately and never pooled for the primary
endpoint: BraTS 2021 test (n=189, in-distribution), BraTS-SSA (n=60),
BraTS-PED (n=99). Model: `outputs/neurovision/checkpoints/best.pt`. Baseline
quantity: single-pass predictive entropy computed from the already-saved `logits/`,
so no additional model run defines the comparator.

**Primary endpoint A — case level.** Partial Spearman correlation between mean
disagreement and per-case Dice, **controlling for mean predictive entropy**.
Reported per cohort with a bootstrap CI from
`analysis.statistics.paired_bootstrap_ci`.

**Primary endpoint B — voxel level.** AUROC for predicting per-voxel error, from
disagreement **residualised on predictive entropy**. Reported alongside the raw
AUROC of each quantity separately, so the reader can see how much of the raw
number was already available from entropy.

**Multiplicity.** Holm–Bonferroni across the full family of primary endpoints
(2 endpoints × 3 cohorts = 6 tests), α = 0.05, fixed now. The family is not
re-drawn after seeing any p-value.

---

## Decision rule

| Outcome | Condition | Consequence |
|---|---|---|
| **Pass** | Partial Spearman \|ρ\| ≥ 0.20 with a CI excluding zero, **and** residualised voxel AUROC ≥ 0.60, on **at least one external cohort** (SSA or PED) | Proceed to Phase 2. The pivot is the paper's headline |
| **Partial** | CI excludes zero but magnitudes fall below those thresholds | Proceed, with the claim reframed as **efficiency** — equal detection at 1 forward pass instead of MC-dropout's 10 — not superiority |
| **Fail** | Map is spatially flat, **or** no endpoint clears both the CI and the threshold on any external cohort | **Stop the uncertainty line.** Fall back to the plan's §Fallback. Cost: two CPU days |

The external-cohort requirement is deliberate. A detector that works only
in-distribution does not address the failure that motivated it.

---

## Power, stated in advance

At n=60 (SSA) and n=99 (PED), a 95% CI on an AUROC is roughly ±0.10. This study is
therefore **not** powered to resolve small differences, and no claim will be made
from a point estimate. Every reported endpoint carries a CI, and every threshold
above is stated as a bound the CI must clear — not as a point estimate to beat.

---

## Pre-committed rules

1. **This file stays in the repository whatever the outcome**, exactly as
   `preregistration_external_hd95.md` did when H1 was not confirmed.
2. **The thresholds are not adjusted after seeing data.** If a threshold turns out
   to have been badly chosen, that is reported as a limitation, not corrected in
   place.
3. **No re-running on a subset.** Running the family, seeing one endpoint fail, and
   re-running on fewer cohorts or metrics destroys the error-rate guarantee.
4. **A downstream equivalence claim needs an equivalence test.** The Phase 2 claim
   "disagreement matches MC-dropout at 1/10 the cost" will be tested by **TOST with
   a margin of 0.03 AUROC**, two one-sided tests at α = 0.05 — registered here so
   the margin cannot be chosen after the fact. A non-significant difference is
   never reported as equivalence.
5. **The negative is publishable.** If H0 holds, the finding is that a
   dual-encoder's branch disagreement is redundant with single-pass entropy — a
   useful result for anyone else considering the same architecture, and it will be
   written up as one.

---

## Result

*To be filled in after the test runs. Do not edit anything above this line.*
