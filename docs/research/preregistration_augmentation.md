# Pre-registration — does heavier training augmentation close part of the SSA/PED generalisation gap?

**Written:** 2026-08-27, before `neurovision_heavy_aug` has been trained, and before any number from
this comparison exists. The git timestamp on this file is the evidence for that ordering.

**Scope.** This is a new item, not yet numbered in `docs/research/master_plan.md` §4 — proposed as
**D0**, run before or alongside D1 (multi-seed), since it is cheap relative to the rest of Phase D and
its answer changes what recipe D1/D2/D3 should even use. It does not replace D2 (pooled multi-cohort
training) or D3 (fine-tune-on-SSA) — those see real target-domain data directly and have stronger
literature backing for closing this specific gap. This experiment asks the cheaper, weaker question
first: does *simulating* scanner/anatomy variation help even without seeing the target domain at all.

---

## Background

The measured failure this targets: `neurovision` trained only on BraTS is **worse** than
`baseline_unet3d` on tumour core under distribution shift — pooled SSA+PED (n=159), `dice_TC`
**-0.0333, p_holm 0.0132, verdict "worse"** (note 30). PED alone is worse still: `dice_TC` -0.0595,
p_holm 0.0002 (n=99). This is the project's clearest unresolved negative and the one Phase D exists to
attack.

The current training recipe's augmentation (`configs/experiment/_baseline_common.yaml`, shared by
every trained arm so the architecture comparison stays controlled) is: independent flips on all three
axes (p=0.5 each), one 90° rotation (p=0.5), ±10% intensity scale, ±10% intensity shift, light Gaussian
noise (p=0.15, std=0.01). This covers geometric flips and mild brightness noise. It does not simulate
the kind of variation that plausibly separates BraTS from BraTS-Africa/BraTS-PED: different scanner
hardware, different bias-field characteristics, different contrast rendering, imperfect patient
positioning.

**Proposed addition**, on top of the existing set (additive, nothing removed):

| Transform | MONAI op | Proposed magnitude | Prob | Rationale |
|---|---|---|---|---|
| Small-angle rotation | `RandRotated` | ±15° per axis | 0.3 | Existing `rot90` only covers 90° steps; real positioning differences are continuous and small |
| Gamma / contrast | `RandAdjustContrastd` | γ ∈ [0.7, 1.5] | 0.3 | Simulates different scanner contrast rendering; nonlinear, unlike the existing linear ±10% scale/shift |
| Simulated bias field | `RandBiasFieldd` | coeff ∈ [0.0, 0.3] | 0.3 | MRI's characteristic smooth intensity gradient varies by coil/scanner; not simulated at all today |
| Elastic deformation | `Rand3DElasticd` | sigma ∈ [5, 8] vox, magnitude ∈ [50, 150] | 0.2 | Simulates anatomical/registration variation. Deliberately the smallest probability and most conservative magnitude of the four — it is the transform most likely to distort a small ET lesion boundary if too aggressive, which is exactly the region already weakest under shift |

All four use MONAI transforms already inside the fixed stack (`monai.transforms`) — no new dependency.
Magnitudes are fixed **now** and must not be tuned after seeing any validation number (see "What would
make this invalid" below).

---

## Why the obvious version of this would be wrong

1. **No noise floor exists yet.** D1 (multi-seed) has not run, so there is exactly one trained seed of
   `neurovision` to compare against. A single-seed-vs-single-seed comparison cannot separate "the
   augmentation helped" from "this seed happened to land differently." This is a real, stated
   limitation, not something to discover after the fact — the decision rule below is built to be
   conservative about it, and the result should be sanity-checked against D1's seed-to-seed spread
   once it exists.
2. **The comparator must be recipe-identical except for augmentation.** `neurovision`'s existing
   checkpoint was trained via `configs/experiment/neurovision.yaml` + `_baseline_common.yaml`: seed 42,
   64³ patches, 80 epochs, AdamW 1e-4, cosine schedule. `neurovision_heavy_aug` must inherit the exact
   same file, changing only the `data.augment` block — same seed, same patch size, same epoch budget.
   Changing anything else turns this from an augmentation ablation into an uncontrolled comparison.
3. **Elastic deformation can hurt exactly the metric this is meant to help.** ET is already the
   smallest, most volatile region under shift. An overly aggressive elastic warp could make small-lesion
   boundaries harder to learn, not easier. This is why its magnitude is the most conservative of the
   four and its effect is reported on ET specifically, not folded silently into a mean.

---

## Arms

| Arm | Definition |
|---|---|
| `neurovision` | existing checkpoint, unchanged — no retraining. Recipe read from `configs/experiment/neurovision.yaml` |
| `neurovision_heavy_aug` | identical config, seed 42, 64³, 80 epochs — only `data.augment` gains the four transforms above |

---

## Endpoints

**Primary:** paired `dice_TC`, pooled SSA+PED (the exact n=159 cohort and pairing note 30 used), via
`analysis.statistics.compare_models` — paired bootstrap CI over case indices, Wilcoxon signed-rank,
Holm correction across the full endpoint family fixed below (not re-selected after seeing results).

**Secondary, same family, Holm-corrected together:**
- `dice_ET`, `dice_WT` on pooled SSA+PED
- `dice_ET`, `dice_TC`, `dice_WT` on BraTS test (n=189) — the in-domain check; must not regress
- Lesion-wise ET/TC/WT on all of the above (co-primary status elsewhere in this project; reported here
  as secondary since the shift failure was measured voxel-wise in note 30)

**Power statement, fixed now.** At n=189 (BraTS test), the smallest detectable `dice` difference is
~0.005–0.008 (established in `preregistration_strong_baseline.md`). At n=159 (pooled SSA+PED) and
n=99 (PED alone), the detectable difference is larger — this comparison is **not** powered to detect a
small effect, and a null result here does not mean "augmentation doesn't help," only "not enough to
see at this sample size with one seed." That distinction must travel with the result.

---

## Decision rule — fixed before any number exists

| Outcome | Verdict | Consequence |
|---|---|---|
| `dice_TC` pooled shift better, CI excluding zero, Holm-significant; BraTS-test `dice_*` not significantly worse | **ADOPT** | Heavy augmentation folds into the shared recipe (`_baseline_common.yaml`) before D1/D2/D3 run, so the rest of Phase D inherits it |
| Inconclusive on the primary endpoint | **NULL, recipe unchanged** | Reported honestly as underpowered-or-absent (see power statement). Default augmentation stays as-is; not folded in without evidence, matching how every other undemonstrated change in this project is handled. D2/D3 proceed on the current recipe |
| `dice_TC` pooled shift better, but BraTS-test regresses, CI excluding zero | **TRADE-OFF, do not adopt as default** | Reported as a genuine accuracy-vs-robustness trade-off — a real, useful finding on its own, but not folded into the shared recipe used for the paper's headline comparisons |
| `dice_TC` pooled shift worse or unchanged | **REJECT** | Augmentation alone does not attack this gap; motivates prioritising D2 (pooled training) and D3 (fine-tune-on-SSA), which see real target-domain statistics rather than simulating them |

**No re-selecting the endpoint family after seeing results.** The primary is `dice_TC` pooled SSA+PED
because that is the exact metric and cohort note 30 measured as failing — this experiment exists to
answer that specific finding, not a convenient nearby one.

---

## What would make this experiment invalid

- Any magnitude or probability in the table above changed after seeing a validation curve.
- `neurovision_heavy_aug` trained at a different seed, patch size, or epoch count than the existing
  `neurovision` checkpoint.
- Metrics computed by anything other than `scripts/evaluate.py`'s existing path on the same case sets
  `outputs/compare_shift/` already used for note 30.
- Reporting only the regions/cohorts where the result looks favourable — the family above is fixed and
  reported in full regardless of outcome.

---

## Cost and abort condition

Baseline arm costs nothing further — `neurovision` is already trained (23.1 GPU-h already spent).
New cost is one `neurovision_heavy_aug` run, same recipe class as the original, so expected in the
same order as the original 23.1 GPU-h. Elastic deformation is more expensive per-sample on the CPU-side
data loader than the existing transforms; **time the first few epochs before committing to the full 80**
and abort/report if per-epoch wall-clock implies materially more than ~30 GPU-h for this one run.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first number exists.)*

**Completed 2026-09-20.** Verdict: **NULL — recipe unchanged.**

### What ran

`neurovision_heavy_aug`, three chained Kaggle T4 sessions (`neurovision-d0-s1`, `-s2`, `-s3`),
**23.7 GPU-h** (10.27 + 10.40 + 3.04), 80/80 epochs, `NVX_HEALTH: OK` at every session exit, peak
VRAM 6.17 GiB. Final checkpoint epoch 79, `val/dice_mean` 0.89375 against the seed-42 `neurovision`
checkpoint's 0.89380 — i.e. the two recipes are indistinguishable on the selection metric itself.
Checkpoint config verified against this file before evaluation: `experiment_name=neurovision_heavy_aug`,
seed 42, 64³, 80 epochs, only the `data.augment` block differing, exactly the four transforms above.

Evaluation: `scripts/evaluate.py` on BraTS test (189), SSA (60) and PED (99), Mac CPU,
`sw_batch_size=1`, `save_logits=true`; lesion-wise columns via `scripts/replay_logits.py` off those
saved logits, no second inference pass. The family was corrected once across all 12 comparisons by
`scripts/compare_family.py` (`outputs/compare_family/d0_heavy_aug/family.csv`), not per table.

### The family, in full, as declared

| Cohort | Metric | heavy_aug | neurovision | Δ | 95% CI | p_holm (m=12) | Verdict |
|---|---|---|---|---|---|---|---|
| **pooled SSA+PED (PRIMARY)** | **dice_TC** | 0.5814 | 0.5697 | **+0.0117** | [−0.0006, +0.0245] | 0.936 | **INCONCLUSIVE** |
| pooled SSA+PED | dice_ET | 0.6638 | 0.6445 | +0.0193 | [−0.0038, +0.0441] | 1 | inconclusive |
| pooled SSA+PED | dice_WT | 0.8712 | 0.8667 | +0.0045 | [−0.0023, +0.0130] | 1 | inconclusive |
| BraTS test | dice_ET | 0.8715 | 0.8709 | +0.0007 | [−0.0030, +0.0056] | 0.010 | inconclusive |
| BraTS test | dice_TC | 0.9155 | 0.9161 | −0.0006 | [−0.0073, +0.0053] | 0.781 | inconclusive |
| BraTS test | dice_WT | 0.9324 | 0.9321 | +0.0004 | [−0.0026, +0.0041] | 0.121 | inconclusive |
| BraTS test | lwdice_ET | 0.7630 | 0.7550 | +0.0080 | [−0.0143, +0.0319] | 0.065 | inconclusive |
| BraTS test | lwdice_TC | 0.8120 | 0.8150 | −0.0030 | [−0.0246, +0.0185] | 0.305 | inconclusive |
| BraTS test | lwdice_WT | 0.7077 | 0.7183 | −0.0106 | [−0.0359, +0.0143] | 0.183 | inconclusive |
| pooled SSA+PED | lwdice_ET | 0.5111 | 0.4957 | +0.0154 | [−0.0111, +0.0415] | 1 | inconclusive |
| pooled SSA+PED | lwdice_TC | 0.3508 | 0.3569 | −0.0061 | [−0.0367, +0.0249] | 1 | inconclusive |
| pooled SSA+PED | lwdice_WT | 0.6483 | 0.6420 | +0.0064 | [−0.0239, +0.0369] | 1 | inconclusive |

**12 of 12 inconclusive.** The primary endpoint's CI contains zero by 0.0006, which is a near miss and
is reported as such rather than rounded into a result.

### Consequence, per the decision rule

Row 2 of the table above fires: **NULL, recipe unchanged.** `_baseline_common.yaml`'s augmentation
stays as it is. D1 (`preregistration_multiseed.md`) therefore trains seed 43 on the **current**
augmentation block, which is what that file's recipe rule prescribes for any verdict other than ADOPT.
D2 and D3 proceed on the current recipe. Nothing here is folded into the deployed model; the clinical
checkpoint remains `neurovision` seed 42.

### One descriptive observation, explicitly not a claim

The pooled null is an average over two cohorts that moved in opposite directions:

| Cohort | Δ dice_ET | Δ dice_TC | Δ dice_WT |
|---|---|---|---|
| SSA (n=60) | −0.0032 | −0.0159 | +0.0027 |
| PED (n=99) | +0.0329 | +0.0284 | +0.0057 |

Heavier augmentation helps the paediatric cohort on every region and does not help — slightly hurts —
the sub-Saharan one. That is a plausible mechanism (simulated contrast/bias-field/anatomy variation
is closer to the paediatric shift than to the SSA one) and it is **not** something this experiment
may assert: the endpoint family is fixed and pooled, the split above was not pre-registered, and the
per-cohort n is smaller still. It is recorded because it should drive what D2/D3 test next, not
because it is a finding.

### What this does and does not license

- It does **not** license "augmentation closes the generalisation gap." It does not.
- It does **not** license "augmentation is useless under shift" either. The power statement fixed
  above applies: at n=159 with one seed, an effect of the size actually observed on the primary
  endpoint (+0.0117) is below what this comparison can resolve. D1 will put a number on how much of
  that is seed noise.
- It does confirm the in-distribution additivity the design principle requires: no BraTS-test number
  moved (largest |Δ| 0.0007 voxel-wise), so the augmentation is not trading in-domain accuracy for
  anything.
