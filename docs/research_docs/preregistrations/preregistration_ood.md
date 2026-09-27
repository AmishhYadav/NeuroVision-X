# Pre-registration — an input-statistics OOD score for the gate (P1.4)

**Written:** 2026-09-27, before any feature is extracted or any score exists. The git timestamp is
the evidence. `master_plan.md` Milestone 5, P1.4 (stretch; P1.1–P1.3 landed first).

---

## Why this exists

The deployed gate judges each mask by a QC model's predicted Dice. It is blind to cohort-level
shift: on SSA it refuses 2 of 60 studies while passing 11 of 12 unusable masks (C21). The QC model
grows *more* optimistic under shift (C19). A signal read from the **input** rather than the output
might see the shift the QC model misses. The gate has had an `ood_score` slot since Milestone 4, but
no scorer was ever built. This builds one, measures it, and decides in advance whether it is
switched on.

## The score, fixed now

**Features, per study, label-free, from the preprocessed 4-channel volume** (the same volume the
model sees; in the clinical path it exists after research preprocessing). Over the nonzero (brain)
voxels of each modality (t1, t1ce, t2, flair), in float32:

- percentiles 1, 5, 25, 50, 75, 95, 99 (7 × 4 = 28)
- skewness and excess kurtosis (2 × 4 = 8)
- Pearson correlation between each pair of modalities over the joint brain mask (6)
- log of the brain voxel count, and the three cropped extents in voxels (4)

46 features in all. Mean and standard deviation are **not** used, because nonzero z-scoring fixes
them at 0 and 1.

**Model.** Standardise each feature with the **train** split's mean and SD. Fit a Gaussian with
shrinkage covariance, Σ̂ = (1 − λ)·S + λ·diag(S), with **λ = 0.1 fixed now** (not tuned). The score is
the Mahalanobis distance to the train mean. Fitted on `train` (875); nothing is fitted on val, test,
SSA or PED.

**Thresholds.** The gate's own rule, unchanged. On the **val** split (187), the 90th percentile of
the score is the CAUTION cut (`caution_quantile` 0.10), and the 98th percentile is the REFUSE cut
(`refuse_quantile` 0.02). HIGH is bad. Frozen before test, SSA or PED is scored.

## Endpoints

1. **Flag rates per cohort** (CAUTION-or-worse, and REFUSE), with bootstrap CIs: test (control),
   SSA, PED.
2. **Case-level:** AUROC of the score for "unusable" (note 47: WT or TC Dice < 0.7) within each
   cohort. This is descriptive, with a bootstrap CI.
3. **Primary: error-budget re-score.** Run `scripts/error_budget.py` with
   `enabled_signals=[input_qc, predicted_dice, intended_use, ood_score]`, against the same run
   without `ood_score`. Report per cohort: accepted, silent failure, over-refusal, at bar 0.7.

## Predictions, fixed now

- (a) **Control:** test flag rates are near 10% / 2%, within their CIs. They are set on val, and
  test is the same distribution.
- (b) **PED:** flagged CAUTION-or-worse at ≥ 50%. Children's brains are smaller and differently
  contrasted, so brain volume alone should separate them. That is an age proxy as much as a scan
  signal, and it is reported as such. `intended_use` already refuses PED, so this matters only for
  a paediatric study with no age in its header.
- (c) **SSA:** flag rate above test's, with the CIs separated. This is the question the run exists
  for. It is uncertain: SSA shift may live in image quality, which the percentiles partly see, or
  in tumour appearance, which they do not.
- (d) **Case-level AUROC ≤ 0.65 in SSA and in PED.** An input-shift score finds unusual studies, not
  failed masks.

## Decision rule — switched on in the live gate or not, fixed now

`ood_score` joins `enabled_signals` **only if both** hold on the primary endpoint:

1. SSA silent failure falls by at least **4 of 11** studies; and
2. test P(accepted AND usable) falls by at most **0.05** (the over-refusal cost in distribution).

Otherwise it stays display-only, like `conformal_band`. That is reported as a negative result, not
tuned. λ, the features and the quantiles are **not** revisited after seeing a number.

## What would make this invalid

- Fitting or standardising on anything but `train`; setting thresholds on anything but `val`.
- Changing features, λ or quantiles after any test, SSA or PED score exists.
- Reporting the PED flag rate as detection of paediatric *failure*. It is detection of paediatric
  *input*.

## Cost

CPU only. It reads ~1,410 preprocessed volumes once (train, val, test, SSA, PED). No inference.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first number
exists.)*

**Run 2026-09-27**, driver `scripts/ood_score.py` (`d5150a6`), outputs `outputs/ood/`. 1,410 volumes
(train 875, val 187, test 189, SSA 60, PED 99), 0 skipped. Nothing above was changed. Cuts from val:
`caution_cut` 7.557, `refuse_cut` 9.935. Experiment note 56, claim C26.

**Endpoint 1 — flag rates** (95% bootstrap CI):

| Cohort | n | CAUTION-or-worse | REFUSE |
|---|---|---|---|
| test | 189 | 0.095 [0.058, 0.138] | 0.037 [0.011, 0.064] |
| SSA | 60 | **0.600** [0.483, 0.717] | 0.150 [0.067, 0.250] |
| PED | 99 | **0.505** [0.404, 0.606] | 0.101 [0.040, 0.162] |

**Endpoint 2 — AUROC for "unusable"** (descriptive): test 0.565 [0.367, 0.763] (12 unusable),
SSA 0.589 [0.422, 0.748] (12), PED 0.600 [0.461, 0.734] (72).

**Endpoint 3 (primary) — error budget at bar 0.7**, `[input_qc, predicted_dice, intended_use]` →
`+ ood_score` (`outputs/error_budget_ood_baseline` → `outputs/error_budget_ood`; the baseline
reproduces `outputs/error_budget_intended_use` exactly):

| Cohort | accepted | P(accepted AND usable) | silent failure | over-refusal |
|---|---|---|---|---|
| test | 0.958 → 0.926 | 0.915 → **0.884** (−0.032) | 8 → 8 | 4 → 10 |
| SSA | 0.967 → 0.817 | 0.783 → 0.683 | **11 → 8** (−3) | 1 → 7 |
| PED | 0 → 0 | 0 → 0 | 0 → 0 | unchanged (`intended_use` refuses all) |

Only REFUSE removes a study; CAUTION is still accepted. The score's REFUSE newly removed 6 test
studies (0 unusable) and 9 SSA studies (3 unusable, 6 usable).

**Predictions.** (a) control near 10% / 2% — **held** (REFUSE 3.7%, CI covers 2%). (b) PED ≥ 50% —
**held**, barely (0.505); detection of paediatric *input*, not failure. (c) SSA above test with CIs
separated — **held** (0.60 vs 0.095). (d) AUROC ≤ 0.65 in SSA and PED — **held** on the point
estimates (0.59, 0.60); the upper CI bounds reach 0.75.

**Decision rule.** (1) SSA silent failure falls by ≥ 4 of 11 — **FAILED** (3 of 11). (2) test
P(accepted AND usable) falls by ≤ 0.05 — held (0.032). Both are required, so **`ood_score` stays
display-only**, like `conformal_band`. Not tuned: λ, features and quantiles are unchanged.

**Reading.** The score sees the cohort-level shift the QC model misses (60% of SSA flagged vs 9.5%
of test), which is the thing C21 says the gate cannot see. It does not tell good SSA masks from bad
ones (AUROC 0.59), so as a per-study REFUSE it throws away two usable studies for each failure it
catches. It is useful as a **cohort-level alarm** (a site whose studies are flagged at 6× the
in-distribution rate is out of distribution), not as a per-study gate.

