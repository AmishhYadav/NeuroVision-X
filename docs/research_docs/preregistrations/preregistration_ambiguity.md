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

## Addendum — where each analysis parameter was fixed, and when

**Written 2026-08-22, while the extraction was still running and before any
endpoint had been computed.** `outputs/detection/` did not exist at the time of
writing; the driver has never been run. That ordering is the point of this
section, and `git log` is the evidence for it.

The body above fixes the *endpoints, thresholds and multiplicity*. It does not
name several operational choices that could each move a number, so this section
records where each one was actually fixed. None of them is being chosen now.

| Parameter | Value | Fixed by | When |
|---|---|---|---|
| Fusion level the ambiguity map is read from | **0** (finest fused, stride 2) | `configs/explainability/default.yaml`, `explainability.ambiguity.level`, and the 65 cases already extracted at that level | 2026-08-20 |
| Case-level score | `amb_dis_mean_fg_mean` | `configs/analysis/default.yaml`, `analysis.detection.case.score_column` | 2026-08-20 (`05287fc`) |
| Case-level control | `ent_mean_fg_mean` | same block, `control_column` | 2026-08-20 (`05287fc`) |
| Case-level outcome | `dice_mean` | same block, `metric_column` | 2026-08-20 (`05287fc`) |
| Voxel sampling mask | `predicted_dilated`, 10.0 mm | same block, `voxel.mask` / `voxel.dilation_mm` | 2026-08-20 (`05287fc`) |
| Voxels sampled per case | 20,000 | same block, `voxel.max_voxels_per_case` | 2026-08-20 (`05287fc`) |
| Bootstrap | 10,000 replicates, 95% | same block, `bootstrap` | 2026-08-20 (`05287fc`) |

**The one thing edited today, stated plainly so it cannot be mistaken for
tuning:** `analysis.detection.cohorts[*].ambiguity_dirs` gained four shard
directories per cohort (`_w0` … `_w3`). That list is a set of **file paths**,
not an analysis parameter — it names *which cases are available*, and the
answer it changes is "all of them" rather than "the 65 in-distribution ones
that happened to be extracted first". The external cohorts had **no** extracted
ambiguity at all before today, so the pre-registered pass condition — which is
scored only on SSA or PED — was not computable in any form. No threshold, mask,
column, level or bootstrap setting was touched.

**Masking convention, for the record.** Both the disagreement scalar and the
entropy comparator use the **predicted-foreground** mask, and neither can see
the ground-truth label: `summarize_case_ambiguity` takes no label argument at
all, and the voxel sampler builds its mask from the prediction alone. This is
deliberate and is the direct consequence of this project's own
`union_foreground_mask` failure, where a label-derived reporting mask
manufactured 41–57% of a reported ECE behind 984 passing tests.

---

## Result

**Run 2026-08-23. Verdict: PARTIAL.** Computed by `scripts/detection_stats.py`
over all 348 cases (BraTS test 189, SSA 60, PED 99), from ambiguity maps
extracted by `scripts/extract_ambiguity_serial.py` and the already-saved
`logits/` of the `neurovision` evaluation runs. Artifacts:
`outputs/detection/detection_{case_level,voxel_level,family}.csv` and
`detection_verdict.json`. The family was run **once**, on the complete data,
after every cohort finished extracting — per pre-committed rule 3.

### Primary endpoint A — case level (partial Spearman, controlling for entropy)

| Cohort | n | ρ(disagreement) | ρ(entropy) | **ρ_partial** | 95% CI | p_holm |
|---|---|---|---|---|---|---|
| BraTS test | 189 | −0.472 | −0.659 | **−0.393** | (−0.512, −0.265) | 0.0006 |
| SSA | 59 | +0.006 | −0.782 | **−0.173** | (−0.432, +0.090) | 0.4132 |
| PED | 98 | +0.015 | −0.554 | **+0.000** | (−0.223, +0.211) | 0.9676 |

### Primary endpoint B — voxel level (residualised AUROC for per-voxel error)

| Cohort | region | AUROC(disagreement) | AUROC(entropy) | **residualised** | 95% CI | p_holm |
|---|---|---|---|---|---|---|
| BraTS test | ANY | 0.694 | 0.888 | **0.578** | (0.560, 0.596) | 0.0025 |
| SSA | ANY | 0.726 | 0.875 | **0.569** | (0.545, 0.590) | 0.0025 |
| PED | ANY | 0.777 | 0.785 | **0.677** | (0.655, 0.698) | 0.0025 |
| BraTS test | ET | 0.867 | 0.942 | **0.733** | (0.708, 0.758) | — |
| SSA | ET | 0.818 | 0.914 | **0.689** | (0.666, 0.714) | — |
| PED | ET | 0.870 | 0.889 | **0.784** | (0.754, 0.817) | — |

Per-region rows are secondary; `ANY` is the pre-registered endpoint. Holm was
applied to the 6-test family (2 endpoints × 3 cohorts), **4 rejected**.

### Why this is PARTIAL and not PASS

PASS required, **on one external cohort**, |ρ_partial| ≥ 0.20 with a CI
excluding zero **and** residualised voxel AUROC ≥ 0.60. No external cohort
satisfies both. PED clears the voxel threshold decisively (0.677, CI lower
bound 0.655) but its case-level partial correlation is **exactly nil**
(+0.0001). SSA clears neither threshold, though its voxel CI does exclude 0.5.

It is not FAIL either: the map is not flat (Test A, note 31), and on **both**
external cohorts the residualised voxel AUROC's CI excludes 0.5 at
p_holm = 0.0025.

### What the numbers actually say

**The signal is spatial, not case-ranking.** Disagreement tells you *where* a
prediction is wrong, beyond what entropy already says, and it keeps doing so
out of distribution. It does **not** tell you *which case* will be bad — not
externally. The case-level columns show why, and they are worth reading
directly: on SSA and PED the raw correlation between disagreement and Dice is
+0.006 and +0.015, i.e. nothing, while **entropy alone is already a strong
case-quality predictor there** (ρ = −0.782 and −0.554). There is no headroom
for disagreement to add case-level information on top of that, and it adds
none.

**ET is where the signal lives, on every cohort:** residualised AUROC 0.733
(test), 0.689 (SSA), 0.784 (PED) — the highest of any region everywhere, and on
PED entropy's own ET AUROC (0.889) is barely above disagreement's (0.870), so
most of what disagreement knows there is *not* redundant with entropy. This
matters for the project specifically: ET is the region carrying the accuracy
gain, and the region whose gain failed to transfer to SSA.

### Two cases dropped at case level, and the asymmetry is deliberate

`BraTS-SSA-00215-000` and `BraTS-PED-00051-000` have an **empty predicted WT
mask**, so the predicted-foreground-masked scalar is NaN and they leave the
case-level correlation (n = 59 and 98, not 60 and 99). At voxel level the same
two cases fall back to whole-volume sampling rather than being dropped, so
voxel `n_cases` is the full 60 and 99. Both behaviours are correct for their
endpoint, but they are not the same denominator and must not be reported as
one.

### Consequence, per the decision rule

Proceed to Phase 2 with the claim **reframed as the rule requires** — equal
detection at one forward pass instead of MC-dropout's ten, plus spatial error
localisation that survives distribution shift — and **never as superiority over
entropy at the case level**, which this data directly refutes on both external
cohorts.

One consequence for Phase 2's design, which the plan wrote before these
numbers existed: Gate 2 is specified as *referral on disagreement beats entropy
and MC-dropout MI on SSA/PED*. Referral is a **case-level** operation, and
case-level is precisely the endpoint that came back null externally. Running
Gate 2 as written is therefore predicted to fail, and it should be respecified
around the endpoint that did survive — voxel- and region-level error
localisation, ET first — before it is run, not after.
