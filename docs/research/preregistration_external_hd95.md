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

## Result — H1 NOT CONFIRMED

Test run 2026-08-19 on BraTS-PED, n = 99, config identical to SSA and to the published
in-domain runs (`roi_size [64,64,64]`, `overlap 0.5`).

**Primary endpoint, Holm across exactly the three rows fixed above:**

| metric | n | n_missing | neurovision | baseline | improvement (mm) | CI | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|
| hd95_ET | 78 | 21 | 9.3291 | 10.2241 | +0.8950 | [-1.9192, 4.3155] | 0.7150 | inconclusive |
| hd95_TC | 88 | 11 | 18.1827 | 15.8909 | **-2.2918** | [-4.6511, 0.0938] | 0.0071 | inconclusive |
| hd95_WT | 98 | 1 | 12.5489 | 12.8619 | +0.3130 | [-2.8418, 3.3613] | 0.7150 | inconclusive |

Positive = `neurovision` better. **Zero rows reach `better` after Holm, and the three do
NOT point the same direction — TC reverses.** Both halves of the decision rule fail, so H1
is not confirmed on either count.

Note `hd95_TC` has `p_holm` = 0.0071 but its bootstrap CI straddles zero (upper bound
+0.0938). `verdict` resolves that disagreement conservatively, as it is designed to. The
direction of that row is `neurovision` **worse** by 2.29 mm, so treating the small p-value
as support for H1 would invert the finding.

**Secondary (reported, deliberately NOT in the primary Holm family):**

| metric | neurovision | baseline | improvement | CI | p_holm | verdict |
|---|---|---|---|---|---|---|
| dice_ET | 0.5634 | 0.5733 | -0.0099 | [-0.0554, 0.0385] | 1.0000 | inconclusive |
| dice_TC | 0.4394 | 0.4990 | **-0.0595** | [-0.0923, -0.0282] | **0.0001** | **worse** |
| dice_WT | 0.8490 | 0.8710 | -0.0220 | [-0.0463, -0.0037] | 1.0000 | inconclusive |
| hd95_mean | 13.5520 | 14.1541 | +0.6021 | [-1.5828, 2.9357] | 1.0000 | inconclusive |

**`neurovision` is CONCLUSIVELY WORSE than the baseline on pediatric tumour core** —
-0.0595 Dice, p_holm 0.0001, CI entirely below zero — and directionally worse on all three
Dice regions.

**Interpretation.** The SSA-generated pattern did not replicate. Across two independent
external cohorts the picture is now: on BraTS-Africa the architecture's advantage vanishes;
on BraTS-PED it reverses and the proposed model is measurably worse on TC. The consistent
HD95 direction observed on SSA (all four rows favouring `neurovision`) was noise, which is
exactly why it required a held-out confirmatory cohort rather than a re-analysis of the set
that generated it.

This closes the last open route to a positive claim for the architecture outside its
training distribution. Per commitment 3 above, this file stays in the repository and the
negative outcome is reported. Do not swap to whichever metric happened to work.

**Caveats that do not rescue it.** Pediatric high-grade glioma is a different disease
entity (necrosis-dominant: 3.77M NCR voxels vs 845k edema, inverted relative to adult
glioma), so both models drop hard in absolute terms — but H1 concerned the paired
difference within each case, which is what was tested. `n_missing` is high on ET (21 of 99)
because HD95 is NaN when exactly one side of a region is empty, and 11.1% of PED cases have
no ground-truth ET.
