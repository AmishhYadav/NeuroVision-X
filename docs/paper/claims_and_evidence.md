# Claims and evidence — what this paper can and cannot say

**Written 2026-08-23, after Gate 1, Gate 2 and P2 all resolved.** This is the
bridge between `docs/experiments.md` (what was measured) and the manuscript
(what gets asserted). Every row names the artifact behind it. **A claim that is
not in this table does not go in the paper.**

The rule this file exists to enforce: this project has produced eight negative
or null results, several of them against its own founding hypothesis. Writing
around them would be the single easiest way to turn a defensible paper into an
indefensible one, and the pre-registrations make the negatives discoverable by
any reviewer who reads the repository.

---

## 1. Claims the evidence supports

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | The proposed dual-encoder architecture improves enhancing-tumour segmentation over a matched U-Net baseline | ET Dice +0.0267, CI [0.0166, 0.0393], p_holm 1.4e-21, n=189 paired | **Strong.** Large effect, tiny p, matched training recipe |
| C2 | The gain is mostly architectural, not a capacity artifact | Width-matched control (34.83M vs 34.91M params): width alone buys +0.0055; architecture +0.0211 | **Strong.** The control most papers never run |
| C3 | Within the architecture, the gain comes from **gated cross-attention fusion**, not from conditioning the gate on inter-branch disagreement | Content-only gate: +0.0189 over the capacity control, and **inconclusive** against the full model (+0.0022, CI −0.0067 to +0.0152) | **Strong for the positive half, and it is a NULL for the mechanism.** Note 38 |
| C4 | Inter-branch disagreement carries per-voxel error information that predictive entropy does not | Residualised voxel AUROC 0.578 / 0.569 / 0.677 (test / SSA / PED), all CIs excluding 0.5, p_holm 0.0025 | **Moderate.** Pre-registered; survives distribution shift |
| C5 | That information does **not** make a better error detector | Entropy + disagreement vs entropy alone: worse in distribution on both endpoints, better only on PED AUROC | **Strong negative.** Note 37 |
| C6 | The interpretable reporting layer is stable with respect to segmentation quality | 189 paired cases, 25 report metrics, 1 conclusive difference; 16 of 25 identical medians across three models | **Strong negative**, and useful for deployment framing |
| C7 | The accuracy gain does not transfer out of distribution | SSA/PED: ET gain gone; TC significantly worse under shift | **Strong negative.** Pre-registered external validation |
| C8 | Single-pass predictive entropy is **equivalent to 10-sample MC-dropout** for voxel-level error localisation | Paired TOST @ 0.03 AUROC: SSA +0.0214 (p 0.024), PED −0.0129 (p 3.7e-05), both equivalent | **Moderate-strong.** Margin fixed in advance; a finding about the *baseline*, not about our model |
| C9 | Inter-branch disagreement is **worse** than both entropy and MC-dropout as a localiser | SSA −0.157, PED −0.056 against MC, intervals entirely outside the margin; beats MC in 30 of 159 cases | **Strong negative.** Note 39 |
| C10 | Under **lesion-wise** scoring — the metric BraTS has used since 2023 — the architecture gain in distribution is **larger** than voxel Dice showed: ET **+0.0508** (CI +0.0225 to +0.0798, p_holm 1.9e-15) and TC **+0.0371** (CI +0.0097 to +0.0653, p_holm 2.7e-10), n=189 paired | Note 41. Re-scored from saved logits; provenance verified to 1e-17 Dice against the committed per-case tables | **Moderate. EXPLORATORY — not pre-registered.** Cohen's dz 0.25 / 0.19, i.e. small. Holm applied within cohort, not across cohorts |
| C11 | The mechanism of that gain is **fewer spurious lesions**, not better recall | Test ET false-positive lesions 0.32/case vs 0.47 for the baseline (32% fewer); false negatives near-identical (0.28 vs 0.30). SSA ET 0.98 vs 1.38 | **Moderate, exploratory.** Note 41. Consistent with the multifocality over-reporting of notes 22–25 |
| C12 | Voxel Dice **overstates** this project's performance, most severely on WT | Lesion-wise minus voxel-wise Dice: −0.10 to −0.32 across all runs and splits; `neurovision` WT test reads 0.9321 voxel-wise and **0.7183** lesion-wise | **Strong, descriptive.** Note 41. Applies to both models equally, so it is a statement about the metric, not about either architecture |
| C13 | A **distribution-free bound on the mask's miss rate holds in distribution**: calibrating a threshold on val and applying it frozen to test gives realised risk at 0.64×–0.96× of nominal α, 6/6 cells, for BOTH models | Note 42; pre-registered in `preregistration_conformal.md` before any number existed | **Strong.** It is a theorem, and it holds for an arbitrarily bad model — which is the point, not a weakness |
| C14 | **That guarantee does NOT survive distribution shift, and it fails by an amount ordered by how far the shift is** — SSA WT ~1.1× nominal, PED WT 1.4–1.9×, PED TC **3.5×–11.5×** | Note 42. 7 of 12 cells violated for each model independently, in substantially the same cells | **Strong, and it is the novel result.** Nobody has measured conformal coverage under shift for 3D tumour segmentation. Graded failure, not uniform collapse |
| C15 | The cost of the guarantee is small, and at looser α it is **negative** | In distribution at α=0.05 the conservative mask grows 1.10× (WT) / 1.18× (TC); at α=0.10 and 0.20 inflation is **0.88×–0.96×**, i.e. smaller than the default 0.5-threshold mask | **Strong, descriptive.** The default operating point is already more conservative than a 10%-miss guarantee requires |

## 2. Claims the evidence does NOT support — do not write these

| Claim | Why it is dead |
|---|---|
| "Better calibrated" | Calibration is within noise against a **temperature-scaled** baseline. The bar is ECE 0.0158, not 0.0565 |
| "Better boundary accuracy" | Boundary-stratified error is within noise. P3 failed as stated |
| "Better uncertainty / risk-coverage" | MC-dropout risk-coverage within noise; Gate 2 negative |
| "The disagreement-conditioned gate is what works" | P2 null. This is the founding hypothesis and it is not supported |
| "Better structured reports" | 1 of 25 metrics. Phase 5 negative |
| "Flag voxels by entropy + disagreement" | Contradicted by Gate 2's own table |
| "Disagreement equals MC-dropout at 1/10 the cost" | **Measured 2026-08-23 and REFUTED.** Disagreement is not equivalent to MC on either external cohort and the intervals sit outside the margin on the wrong side. The equivalence that DOES hold belongs to free single-pass entropy (C8), not to this architecture. Note 39 |
| Any claim on **WT** | Every WT comparison is inconclusive — voxel-wise, and lesion-wise too (+0.0300, CI **−0.0021** to +0.0620). Note the stated *reason* changed on 2026-08-24: WT is **not** "saturated at ~0.93". Lesion-wise it is 0.7183, and the 0.93 was a measurement artifact of a union region whose voxel count is dominated by one large component. The rule survives; the justification for it does not |
| "The lesion-wise gain transfers out of distribution" | **Measured 2026-08-24 and it does not.** All six external comparisons (SSA and PED × ET/TC/WT) have CIs straddling zero. Every point estimate favours `neurovision`, which is exactly why this row exists: six positive-looking numbers with six CIs crossing zero is an underpowered null, not transfer. Note 41 |
| Any architecture claim on **pediatric tumour core** | Both models score lesion-wise Dice 0.234 with **64 of 99 cases exact ties**. They fail identically; there is no gradient to claim on |

## 3. Limitations that must appear in the paper

1. **Single seed.** No seed-to-seed noise floor, so no margin below ~0.005 Dice is claimable, and the P2 null is "no detectable difference at n=189", not equivalence. A TOST with a pre-set margin would be needed for the latter.
2. **No transformer baseline on our splits.** SwinUNETR was cut for budget. Published BraTS-2021 SwinUNETR numbers are on the official validation set and are **not** a substitute for a matched comparison on a random split of the training set.
3. **Gate 2's combiner is fitted by pooled likelihood while its endpoints are per-case rank metrics.** A rank-optimised combiner might do better; refitting after seeing the result would be post-hoc.
4. **`ignore_empty=False`** for empty regions (BraTS convention). Measured effect on this data: only 2.6% of cases have no ET, so it moves headline ET Dice by well under a point — but the convention must be stated.
5. **The eloquence layer is degenerate on this cohort** (100% "near eloquent"), so no per-case information; reporting agreement there would be true and misleading.
6. **One ablation rung of five.** The remaining four have no GPU hours behind them.
7. **The headline metric changed under us.** Every pre-2026-08-24 number in this project is voxel-wise Dice on a random split of the BraTS 2021 *training* set. That is not comparable to any published BraTS number, and lesion-wise re-scoring (note 41) shows it overstates performance by 0.10–0.32 Dice. Both conventions must be reported, and the lesion-wise one is the comparable one.
8. **The lesion-wise family is exploratory.** It was not pre-registered before it was computed. Only `preregistration_strong_baseline.md` names lesion-wise ET as a co-primary endpoint, and that gate has not yet run.
9. **Every conformal α=0.20 threshold is a boundary solution.** λ̂ = 0.95 is the largest value in the grid, so the reported figure is censored by the grid rather than measured at it — the true λ̂ is "≥ 0.95".
10. **The conformal calibration split is not clean.** Every checkpoint was selected on val by `val/dice_mean`, so λ̂ fitted there could in principle be optimistic. This was predicted in advance and did not materialise — in-distribution risk never exceeded α — but the caveat belongs in the paper rather than only in the pre-registration.

## 4. The honest framing

Not "our model is better." The paper is a **controlled study of what a
dual-encoder disagreement gate does and does not buy**: a real, attributable
accuracy gain in enhancing tumour; a mechanism that is measurably active
(P1/note 32) and measurably *not* responsible for the gain (P2/note 38); an
uncertainty signal with genuine incremental information (Gate 1) that
nonetheless fails to improve a working detector (Gate 2), is worse than free
entropy as a localiser (note 39), and whose removal costs no accuracy (P2); and
no transfer under distribution shift.

The methodological contribution is the discipline: pre-registration written
before the numbers, thresholds fixed in advance, a width-matched capacity
control, two external cohorts, and negatives reported rather than buried. That
is what makes this publishable, and it is why the negatives are assets in the
write-up rather than embarrassments to be minimised.

**Realistic venue:** an uncertainty/evaluation-focused workshop (e.g. UNSURE at
MICCAI) or a journal that accepts rigorous mixed-result studies. Not a
main-conference SOTA claim.
