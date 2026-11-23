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

**Recorded 2026-11-23. Verdict: REPLICATES.** (Experiments note 62, claim C30.) Read strictly
against the rules above; nothing above the `## Result` line was edited.

**Run.** Kaggle `neurovision-gatea-conformal` v2, `GIT_REF 539534b`, 2.20 GPU-h (inside the 5 GPU-h
abort bound; v1 crashed in the bridge cell after 0.8 GPU-h on a CSV-format bug and was re-run
unchanged). Gate A's `checkpoint_final.pth` (epoch 1000), nnU-Net 2.8.1, mirroring on,
`--save_probabilities`; 187 val, 189 test, 60 SSA, 99 PED cases, 7.8–8.2 s/case. Region conversion
check: thresholding the converted probabilities at 0.5 reproduced nnU-Net's own label map in every
case (largest mismatching voxel fraction 6.4e-7, tolerance 1e-5).

**Bridge check: PASS**, recorded before any SSA/PED number was read. Test split through our
preprocessed route vs Gate A's raw route: ΔDice ET +0.0002, TC −0.0001, WT +0.00002 (bound 0.003).
`results/conformal/nnunet_gatea/bridge.json`.

**N1 — in distribution (test, n = 189).** Holds in 6/6 cells.

| Region | α | τ̂ (val) | realised mean FNR [95% CI] | rule |
|---|---|---|---|---|
| WT | 0.05 | 0.0158 | 0.0296 [0.0228, 0.0370] | holds |
| WT | 0.10 | 0.875 | 0.0749 [0.0639, 0.0863] | holds |
| WT | 0.20 | 0.95 | 0.0873 [0.0760, 0.0995] | holds |
| TC | 0.05 | 0.0032 | 0.0443 [0.0301, 0.0611] | holds |
| TC | 0.10 | 0.90 | 0.0944 [0.0759, 0.1154] | holds |
| TC | 0.20 | 0.95 | 0.1017 [0.0824, 0.1227] | holds |

**N2 — under shift.** VIOLATED = CI lower bound > α; holds = CI upper bound ≤ α; else inconclusive.

| Cohort · region | α 0.05 | α 0.10 | α 0.20 |
|---|---|---|---|
| SSA · WT (n = 60) | 0.0521 [0.0290, 0.0904] inconclusive | 0.1210 [0.0952, 0.1579] inconclusive | 0.1391 [0.1129, 0.1766] holds |
| SSA · TC | 0.0866 [0.0448, 0.1418] inconclusive | **0.1823 [0.1347, 0.2369] VIOLATED** | 0.1950 [0.1462, 0.2530] inconclusive |
| PED · WT (n = 99) | **0.0678 [0.0532, 0.0847] VIOLATED** | 0.1133 [0.0945, 0.1346] inconclusive | 0.1229 [0.1033, 0.1440] holds |
| PED · TC | **0.5549 [0.4899, 0.6195] VIOLATED** | **0.6240 [0.5647, 0.6832] VIOLATED** | **0.6302 [0.5717, 0.6878] VIOLATED** |

5 of 12 cells VIOLATED (`neurovision`: 7 of 12, note 42).

**Applying the verdict rule.** (i) PED · TC VIOLATED at all three α: yes. (ii) For each primary
region, mean realised/α over the three α larger on PED than on SSA: WT 1.035 vs 0.982, yes; TC 6.83
vs 1.51, yes. Both conditions met → **REPLICATES**: the failure under shift is not an artefact of
the weaker recipe; the field's standard model fails the same way, worst on paediatric tumour core.

**How the verdict is quoted under the 2026-11-15 thesis (MasterPlan §B1).** Condition (ii) was
written when the thesis was "in proportion to the shift". With three cohorts it cannot order shifts
by size, and the largest failure (PED · TC) is mostly a label-definition mismatch (note 58). The
verdict is reported as registered; condition (ii) is quoted only as "PED fails more than SSA on
both regions", never as a dose–response.

**Mandatory secondary: the cost.** Mask inflation |M(τ̂)|/|M(0.5)| on test: α 0.05 WT 1.11×, TC
1.15×; α 0.10 0.97× / 0.97×; α 0.20 0.94× / 0.95× (the guaranteed mask is *smaller* than the
default at the looser α, as for `neurovision`, C15). Under shift the same pattern, except PED · TC at
α 0.05: mean 1.68× (median 1.18×; 8 of 99 cases skipped for an empty default mask).
**Grid caveat:** at α 0.20 τ̂ = 0.95 is the top of the registered threshold grid for both regions
(val risk at 0.95: WT 0.105, TC 0.100, well under 0.20), so the α 0.20 cells measure the risk at the
grid ceiling, not at a binding threshold. MasterPlan W2.1(b) extends the grid; nothing here is
re-fitted.

**Secondary 1 — is PED · TC restorable?** No. R̂(τ_min = 1e-4) on PED · TC = **0.486**
(`neurovision`: 0.356, note 52): even a mask of every voxel with p ≥ 1e-4 misses half of the
paediatric tumour core. Local recalibration (`scripts/local_recalibration.py`, unchanged, 1000
splits per k, seed 42; counterfactual, never this cohort's coverage): PED · TC infeasible at every
k and α. Restorable elsewhere with the floors: SSA WT 30/11/5, SSA TC never/19/6, PED WT 27/11/5
labelled cases at α 0.05/0.10/0.20 (`neurovision`: SSA TC 39/13/5). SSA · TC at α 0.05 is now
infeasible for nnU-Net because its own floor (R̂(τ_min) = 0.0525) sits above α: sharper
probabilities leave less room at the permissive end. `results/local_recalibration_nnunet/`.

**Secondary 2 — segmentation under shift** (nnU-Net − `neurovision`, Holm family of 6, paired
bootstrap 10,000 + Wilcoxon, `results/compare_family/nnunet_conformal_shift_dice/family.csv`):

| Cohort | dice_ET | dice_TC | dice_WT |
|---|---|---|---|
| SSA (n = 60) | −0.0162 [−0.0573, +0.0191] inconclusive | +0.0245 [+0.0027, +0.0507] inconclusive (p_holm 0.20) | **+0.0238 [+0.0103, +0.0412] better** |
| PED (n = 99) | +0.0148 [−0.0346, +0.0616] inconclusive | **+0.0471 [+0.0124, +0.0843] better** | **+0.0423 [+0.0202, +0.0702] better** |

nnU-Net's in-distribution lead survives shift on WT (both cohorts) and PED TC; ET is not resolved
on either cohort.

**What this changes.** C14 is now quoted for both models: the bound breaks under shift for the
standard model too. Beside it, descriptive and not registered: nnU-Net's realised risk is lower
than `neurovision`'s in 9 of the 12 shifted cells (all 6 PED cells; higher in SSA WT α 0.10 and
SSA TC α 0.05/0.10), and it violates 5 cells to `neurovision`'s 7 — "a stronger model shrinks the
damage" where the label definition is unchanged, and fails at a higher floor where it is not.
