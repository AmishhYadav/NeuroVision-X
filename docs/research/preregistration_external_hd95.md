# Pre-registration — HD95 under distribution shift

**Written 2026-08-19, BEFORE any BraTS-PED case was preprocessed, evaluated or
analysed.** Committed ahead of the test so the git timestamp is the evidence. If this
file's commit does not precede the commit carrying the BraTS-PED results, the test
below is exploratory and must be reported as such.

---

## Where the hypothesis came from

It was **generated** on BraTS-Africa (SSA, n = 60), which therefore **cannot confirm
it**. SSA is the discovery set. What follows is the confirmatory test on a second,
untouched external cohort.

Observation that generated it — `neurovision` vs `baseline_unet3d`:

| metric | in-domain BraTS test (n=189) | external SSA (n=60) |
|---|---|---|
| dice_ET | +0.0267 (conclusive) | −0.0008 (inconclusive) |
| hd95_ET | +0.67 mm (inconclusive) | +2.22 mm (inconclusive) |
| hd95_TC | +0.27 mm (inconclusive) | +2.05 mm (inconclusive) |
| hd95_WT | **−0.37 mm** (worse) | +1.51 mm (inconclusive) |
| hd95_mean | +0.21 mm (inconclusive) | +1.68 mm (inconclusive) |

Positive = `neurovision` better. In-domain the HD95 advantage is small and **mixed in
sign**; on SSA all four rows favour `neurovision`, at 3–5x the magnitude, and the WT
sign flips. Relative gain on ET roughly doubles (13.6% → 27.6%), so it is not merely
that HD95 is larger out-of-distribution and has more room to move.

Zero of eight SSA comparisons survived Holm. Nothing above is a result.

## Hypothesis

**H1.** Under distribution shift, `neurovision` achieves lower (better) HD95 than
`baseline_unet3d`, across regions — i.e. the architecture buys **boundary localisation
robustness under shift**, rather than overlap accuracy in-distribution.

Mechanistic reason it is plausible rather than fitted: the fusion gate is conditioned
on inter-branch disagreement. Disagreement between the CNN and Swin branches should be
largest exactly when the input is unfamiliar. An architecture that routes on that
signal should therefore help most off-distribution — which is where H1 says it helps.

**H0.** No consistent HD95 advantage under shift. The SSA pattern was noise.

## Test — fixed before seeing the data

- **Cohort:** ASNR-MICCAI BraTS2023-PED (pediatric high-grade glioma), all cases, every
  one assigned to `test`. Nothing fitted on it — no model, no temperature, no threshold.
- **Models:** `neurovision` (`outputs/neurovision/checkpoints/best.pt`) vs
  `baseline_unet3d` (`outputs/checkpoints/baseline_unet3d/best.pt`). The capacity
  control is excluded because its checkpoint was never retrieved from Kaggle.
- **Config:** identical to SSA and to the published in-domain runs — `roi_size
  [64,64,64]`, `overlap 0.5`. Any deviation invalidates the comparison.
- **Primary endpoint:** `hd95_ET`, `hd95_TC`, `hd95_WT` via
  `analysis.statistics.compare_models`, paired, **Holm-corrected across exactly those
  three rows and no others**. The Holm family is fixed here, now, at three.
- **Secondary, reported but not part of the family:** `dice_ET/TC/WT`, `hd95_mean`.
- **Decision rule:** H1 is supported only if at least one of the three primary rows has
  `verdict == "better"` after Holm AND all three point the same direction
  (`improvement > 0`). Any other outcome is reported as failure to confirm.

## Committed in advance

1. The Holm family is the three primary HD95 rows. It will not be re-scoped after
   seeing p-values — that destroys the error-rate guarantee.
2. SSA will **not** be pooled with PED to manufacture significance, and SSA will not be
   re-analysed to support H1.
3. If H1 fails, this file stays in the repository and the negative outcome is reported.
   No swapping to whichever metric happened to work.
4. Pediatric glioma is a different disease entity from adult glioma, not merely a
   different scanner population. A drop in absolute performance is **expected** and is
   not itself evidence about H1 — H1 is about the *difference between the two models*,
   which is why every comparison is paired within-case.
5. Known confounds carried over from SSA and to be reported alongside: sample size
   limits power; `ignore_empty=False` gives free Dice on empty-ET cases and the rate
   differs between cohorts; left–right orientation rests on the affine rather than on
   content; HD95 is NaN when exactly one side of a region is empty, so `n_missing` must
   be reported per row.

## Result

*To be filled in after the test. Left empty deliberately.*
