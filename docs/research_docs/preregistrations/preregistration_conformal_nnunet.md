# Pre-registration — the conformal bound on nnU-Net (does the shift failure replicate?)

**Written:** 2026-10-13, after Gate A (note 61, C29) and before any nnU-Net probability map, threshold
or risk number exists. The git timestamp is the evidence for that ordering. Author decision
2026-10-13: keep `neurovision` deployed, and test whether the paper's central finding holds on the
field's standard model, instead of swapping models.

---

## Why this run exists

Gate A retired the architecture claim: nnU-Net segments better than `neurovision` (ET 0.8879 vs
0.8709). The paper's contribution is now the reliability layer: a conformal bound that holds in
distribution and **breaks under shift in proportion to the shift** (C13, C14). A reviewer's first
question will be: *is that a property of your weaker model, or of the method?* Until now the bound
was measured on `neurovision` and a matched U-Net, both trained on the same recipe. This run asks
the same question on nnU-Net, with the protocol unchanged.

Both outcomes are publishable and both are written down here, before the numbers:

- **It replicates**: the failure under shift is not a weak-model artefact. The standard model fails
  the same way, and the case for a refusal gate and local recalibration is stronger.
- **It does not replicate**: a stronger model buys robustness under shift, which is itself a
  finding, and the C14 wording must then say "for models trained on our recipe".

## Design

**The model.** The Gate A arm, unchanged: nnU-Net v2 `3d_fullres`, `fold all`,
`checkpoint_final.pth` (epoch 1000), Kaggle output `amishyadav123/neurovision-gatea-s7`. Inference
with nnU-Net's defaults (Gaussian sliding window, step 0.5, **mirroring on**, as Gate A's primary
arm) plus `--save_probabilities`. No retraining, no tuning.

**The protocol.** `preregistration_conformal.md` §§ "The loss", "Design", "Endpoints and decision
rules", unchanged: per-case false-negative rate of the conservative mask $\{p \ge \tau\}$; regions
**WT and TC primary**, ET secondary; empty ground truth excluded per region and counted;
α ∈ {0.05, 0.10, 0.20}; $\hat\tau$ fitted on BraTS **val** (n = 187) and applied **frozen** to test
(n = 189), BraTS-Africa (n = 60) and BraTS-PEDs (n = 99); 95% bootstrap CIs, 10,000 resamples,
seed 42; the same threshold grid; `scripts/conformal.py` unmodified. Nothing is fitted on SSA or
PED.

One difference, stated because it favours nnU-Net: its val split is **clean**. `neurovision`'s
checkpoint was selected on val; nnU-Net's `checkpoint_final` was chosen by the schedule, never by
val. So the "val is optimistically biased" threat of the original design does not apply here.

**Region probabilities.** nnU-Net was trained in region mode (`dataset.json` labels: whole tumour
[1,2,3], tumour core [2,3], enhancing tumour [3]), so its saved probabilities are already one
sigmoid per region. They are converted to our (ET, TC, WT) channel order and stored as fp16 logits,
the format every other arm's conformal analysis reads. Each case is checked: thresholding the
converted probabilities at 0.5 must reproduce nnU-Net's own exported label map.

**The input route, and the bridge check that guards it.** The raw SSA and PED scans were deleted
(2026-08-19; manifests in `docs/data_manifests/`), and are not on Kaggle. Every cohort's
*preprocessed* arrays are (`amishyadav123/neurovision-brats{,-ssa,-ped}-prep`). Our preprocessing
only reorients (flip/transpose, no resampling), z-scores each channel over its nonzero voxels, and
crops to the brain; nnU-Net z-scores over the nonzero mask itself. So nnU-Net is fed our
preprocessed arrays, for **all four** cohorts (val, test, SSA, PED), one route for everything.

Because that route differs from the raw files Gate A used, it must first be shown to give the same
model behaviour. **Bridge check, fixed now:** the test split predicted through this route and scored
by `replay_case` with default post-processing must match Gate A's raw-route scores
(`outputs/score_nnunet/gatea_tta_on/per_case_metrics.csv`) to within **0.003** on mean Dice for
each of ET, TC and WT. If it fails, the run is reported as **BRIDGE FAILED**: no conformal number
from this route is read, and the fallback (re-downloading the raw cohorts from Synapse, an author
action) is decided then.

## Endpoints and decision rules

**N1: in distribution (test).** For each region and α, realised mean FNR with 95% CI; the original
B1 rule (holds if point estimate ≤ α; violated if CI lower bound > α).

**N2: under shift (SSA, PED). This is the question.** The original B2 table, same rule per cell
(VIOLATED if CI lower bound > α; holds if CI upper bound ≤ α; else inconclusive). Verdict:

| Outcome | Rule | Reading |
|---|---|---|
| **REPLICATES** | PED · TC VIOLATED at all three α, **and** the shift ordering holds: for each primary region, mean realised/α over the three α is larger on PED than on SSA | The C14 failure is a property of the method under this shift, not of our recipe. C14 is quoted for both models |
| **PARTIAL** | Any VIOLATED cell under shift, but REPLICATES' rule not met | Reported cell by cell; C14 is restated as "on both models, coverage degrades under shift" only for the cells that agree |
| **DOES NOT REPLICATE** | No cell VIOLATED on either cohort | A stronger model holds the bound under these shifts. C14 is restricted to models trained on our recipe, and that restriction travels with every quote |

**Mandatory secondary: the cost.** As in the original: $\hat\tau$ itself, the inflation ratio
$|M(\hat\tau)| / |M(0.5)|$ and the uncertain-band volume, per cohort, region and α.

**Secondary, descriptive (no verdict):**

1. **Is PED · TC restorable?** $\hat R(\tau_{\min})$ on PED (C24's infeasibility check), and the
   local-recalibration curve over k, computed from the cached loss curves exactly as for C24.
2. **Segmentation under shift.** nnU-Net − `neurovision` paired Dice on SSA and PED, via
   `scripts/compare_family.py`: {`dice_ET`, `dice_TC`, `dice_WT`} × {SSA, PED}, one Holm family of
   6. Answers whether nnU-Net's in-distribution lead (C29) survives shift. Reported as measured;
   it gates nothing.

## What would make this invalid

- Any change to the loss, α grid, threshold grid, regions, calibration split or bootstrap.
- Reading any conformal or Dice number on SSA/PED before the bridge check has passed and been
  recorded.
- Fitting any threshold on SSA or PED and reporting it as that cohort's coverage (the local-
  recalibration curve is a counterfactual, labelled as such, as in C24).
- Using `checkpoint_best.pth`, or any nnU-Net run other than the Gate A model.

## Cost and abort condition

Prediction with probabilities on 535 cases at ~7–12 s/case on a T4: **~1.5–2 GPU-h**, one session,
week of 2026-10-17, alongside D2 (~20 GPU-h). **Abort** if the session needs more than **5 GPU-h**.
Logits are written to Kaggle's temporary disk and discarded; only loss curves, per-case CSVs and
the fitted results leave the box. They are regenerable from the s7 checkpoint by re-running the
notebook.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first number
exists.)*
