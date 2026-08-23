# Pre-registration — does the architecture gain survive a properly configured baseline?

**Written:** 2026-08-23, **before** nnU-Net or MONAI Auto3DSeg has been installed, exported to, or
run in this project, and before any number from either exists. The git timestamp on this file is the
evidence for that ordering, which is the entire point of writing it first.

**Scope.** This registers **Gate A** of `docs/research/master_plan.md` — the credibility gate. It is
the most consequential single experiment remaining in the project, because it decides whether the
one surviving positive claim is real or is an artifact of a weak comparator.

---

## Background

`neurovision` beats `baseline_unet3d` by **+0.0267 ET Dice** (CI 0.0166–0.0393, p_holm 1.4e-21,
n=189 paired) and beats a width-matched capacity control by **+0.0211** (p_holm 7.3e-19), which
decomposes the gain as ~79% architecture / ~21% capacity. That is the project's only positive result,
and every other claim built on it has already failed.

But both comparators are **our own** U-Net, trained on **our own** recipe: 64³ patches, 80 epochs,
AdamW at 1e-4, `batch_size: 1` × `samples_per_volume: 4`, no ensembling, no test-time augmentation.
The field's reference implementation, nnU-Net, trains for 1000 epochs with SGD and a poly schedule,
aggressive augmentation, a larger patch, five-fold cross-validation and ensembling.

MICCAI 2024's *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*
benchmarked precisely this class of claim across CNN-, Transformer- and Mamba-based architectures and
found that reported gains repeatedly fail to hold up against a properly configured CNN U-Net, with
inadequate baselines named as the primary validation shortcoming. It is therefore an open, live and
entirely plausible possibility that **+0.0267 measures the weakness of our recipe rather than the
strength of our architecture.**

This experiment is run *because* that possibility is uncomfortable, not despite it. If it is true, the
project needs to know now — while it can be reported as rigour — rather than in review.

---

## Why the obvious version of this test would be wrong

Three ways to get a misleading answer, each pre-emptively closed:

1. **Different test cases.** nnU-Net generates its own five-fold split by default. Comparing its
   internal validation score against our test score compares two different case sets and is
   meaningless. nnU-Net is therefore forced onto **our frozen split** via `splits_final.json`, and
   scored on **the same 189 test cases**, so every comparison stays paired.
2. **Different metric conventions.** Our Dice uses `ignore_empty=False`, region targets ET/TC/WT, and
   our own `postprocess_logits`. nnU-Net's internal metrics do not match. All models are therefore
   re-scored through **our** `scripts/evaluate.py` metric path from saved predictions, so the number
   is computed by one code path for every arm.
3. **Reading only voxel-wise Dice.** BraTS moved to lesion-wise metrics in 2023 because voxel Dice
   favours large lesions. Our model already over-reports multifocality (30.7–41.3% against a true
   22.8%), which voxel Dice cannot see. The lesion-wise endpoint is therefore **co-primary**, not
   supplementary.

---

## Arms

| Arm | Definition | Notes |
|---|---|---|
| `neurovision` | existing checkpoint, `best.pt` (epoch 69) | unchanged, no retraining |
| `baseline_unet3d` | existing checkpoint, 80ep/64³ | the current comparator, retained for continuity |
| `nnunet_v2_3dfullres` | nnU-Net v2, `3d_fullres`, **single fold**, trained on our frozen train split | default nnU-Net recipe, unmodified — the point is the reference configuration, not a tuned one |
| `auto3dseg_segresnet` | MONAI Auto3DSeg, SegResNet, single fold, our frozen train split | second modern comparator; already inside our fixed stack; took 1st place in BraTS-Africa 2023 |

**Controlled:** the training case list (875 cases from `configs/data/splits.yaml`), the test case list
(the same 189), the metric code path, and the postprocessing convention.

**Deliberately not controlled, and stated as such in the paper:** training schedule, patch size,
optimiser, augmentation policy. Equalising those would defeat the purpose — the question is whether
our model competes with the field's *standard practice*, not with a hobbled version of it. The
comparison is therefore explicitly **architecture-plus-recipe against architecture-plus-recipe**, and
any conclusion must be phrased that way.

---

## Endpoints

**Co-primary, both on the 189-case test split, both paired:**

- **E1 — voxel-wise ET Dice**, `neurovision` vs `nnunet_v2_3dfullres`.
- **E2 — lesion-wise ET Dice** (panoptica, BraTS-2023 convention), same pairing.

**Secondary, reported but not gating:** TC and WT for both metric families; HD95; NSD; the same
comparisons against `auto3dseg_segresnet`; and every arm against `baseline_unet3d`.

**Statistics:** `analysis.statistics.compare_models` — paired bootstrap CI over case indices (never
resampling the two score arrays independently), Wilcoxon signed-rank with rank-biserial effect size,
and **Holm correction across the whole comparison table as one family**, fixed before any p-value is
looked at. `verdict` is `inconclusive` if either the CI contains zero or the Holm-adjusted p exceeds
α = 0.05.

**Power statement, fixed now.** At n=189 with the observed per-case paired standard deviation for ET
Dice, the smallest difference this design can resolve is roughly **0.005–0.008 Dice**. Any difference
smaller than that is *not* evidence of equivalence — it is the resolution limit of the instrument.
Equivalence, if we want to claim it, requires a TOST with a margin fixed in advance, as was done for
the MC-dropout comparison in note 39.

---

## Decision rule — fixed before any number exists

| Outcome on **both** E1 and E2 | Verdict | Consequence for the paper |
|---|---|---|
| `neurovision` better, CI excluding zero, Holm-significant | **SURVIVES** | C1 and C2 stand as strong claims and open the paper. The architecture result is real against the field's bar |
| `inconclusive` on either endpoint | **PARITY** | Report as *competitive with nnU-Net at roughly 1/12 the training schedule*, stated as a resolution-limited null, never as equivalence. Architecture stops being the headline; the pipeline contribution leads |
| nnU-Net better, CI excluding zero, Holm-significant | **RETIRED** | **The architecture claim is withdrawn from the paper.** The +0.0267 is reported as measured against a matched-recipe baseline only, with the nnU-Net result printed beside it. The pipeline, guarantee and refusal work becomes the entire contribution |

**Split outcomes** (one endpoint survives, the other does not) resolve to **PARITY**, and the
disagreement between voxel-wise and lesion-wise is itself reported as a finding — it would mean the
gain is concentrated in large-lesion voxel overlap rather than in lesion detection, which is exactly
what the 2023 metric change was introduced to expose.

**No re-running on a smaller metric set.** Selecting a subset after seeing which comparisons failed
destroys the Holm error-rate guarantee. The family is fixed by this document.

---

## What would make this experiment invalid

Recorded now so it cannot be rationalised later:

- nnU-Net trained on any case that appears in our val or test split.
- nnU-Net evaluated on its own internal validation fold instead of our 189 test cases.
- Metrics computed by nnU-Net's own evaluation code rather than ours.
- Any tuning of nnU-Net *downward* (fewer epochs, smaller patch, disabled augmentation) to fit the
  compute budget. If the full recipe cannot be afforded, the run does not happen and the gate stays
  open — a deliberately weakened reference implementation is worse than no reference at all.
- Any tuning of `neurovision` *upward* (TTA, ensembling, longer training) that is not also applied
  to, or already present in, the comparator. Flip TTA from Phase A4 is measured **separately** and
  reported as its own row; it does not enter this comparison.

---

## Cost and abort condition

Estimated 25–40 GPU-h for both new arms on a modern card. A 2-epoch timing probe runs first and must
reach the failure condition rather than merely execute — this project has lost GPU hours twice to
probes that ran cleanly without reaching the state they needed to test. Abort if measured per-epoch
time implies more than 60 GPU-h for the pair.

---

## Result

*(To be completed after the runs. Nothing above this line may be edited once the first number
exists.)*
