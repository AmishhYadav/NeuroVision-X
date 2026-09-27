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
