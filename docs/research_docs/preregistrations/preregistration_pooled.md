# Pre-registration — D2, pooled multi-cohort training

**Written:** 2026-10-11, before any D2 training starts and before any number from it exists. The git
timestamp is the evidence for that ordering. `master_plan.md` §5 Phase D, item D2 ("Never cut D2");
moved from the college card to Kaggle T4 by author decision 2026-10-11.

---

## Why this run exists

The ET gain does not transfer, and tumour core fails under shift (C7; pooled external `dice_TC`
−0.0333, note 30). Two site-side fixes are already measured: local recalibration restores the bound
except PED · TC (C24), and a cross-fitted fine-tune of the deployed model recovers TC on both cohorts
(C27). D2 asks the **developer-side** question: if the shifted cohorts had been in the training set
from the start, does one model close the gap — and what does it cost in distribution?

The answer is written in advance because review will ask it: if pooled training closes the gap,
why does anyone need a refusal gate? Because no training set covers every deployment shift —
detection and coverage are complements, not substitutes.

## Design

**Arm.** `neurovision_pooled`: `+experiment=neurovision`, **seed 42**, trained **from scratch** with
the deployed recipe unchanged (64³, 80 epochs, `grad_clip_norm` 5.0, current augmentation — D0 was
NULL), on `configs/data/splits_pooled_cf0.yaml`:

| Key | BraTS 2021 | SSA (cf0) | PED (cf0) | Total |
|---|---|---|---|---|
| train | 875 | 25 | 45 | 945 |
| val | 187 | 5 | 5 | 197 |
| test | 189 | 30 | 49 | 268 |

The split is a deterministic concatenation of three frozen files (header of
`splits_pooled_cf0.yaml`), so D2's held-out external cases are **exactly D3 fold 0's held-out
halves** and the two fixes are compared on identical cases. **No oversampling** of the external
cohorts (they are 7.4% of training cases): "pooled" means pooled, and any re-weighting would be a
second free parameter chosen without a validation set big enough to choose it.

**Checkpoint.** `best.pt` by validation `dice_mean` over the pooled 197-case val set — the deployed
selection rule, applied to the pooled val. (Val is 95% BraTS, so selection is close to the deployed
model's own.)

**Data plumbing.** No code change: the notebook symlinks every case directory of the three attached
preprocessed datasets into one `preprocessed/` tree and points `data.splits.path` at the repo's
frozen split file. Case ids are disjoint across cohorts (asserted at build time).

**One fold only.** The second cross-fit fold would double a ~26 GPU-h run; fold 0 alone is what the
quota buys. The held-out external numbers are therefore **cross-fitted on one fold, not external
validation**, n = 30 SSA and 49 PED.

## Endpoints

All via `scripts/evaluate.py` (`save_logits=true`) on the Mac CPU, then `scripts/replay_logits.py`
lesion-wise — the metric path of every other row. Comparator: the frozen deployed `neurovision`
(seed 42) on the **same** cases (its existing rows, restricted to these case ids).

**The family, fixed:** one Holm family of **11**, `scripts/compare_family.py`, paired bootstrap
(n_boot 10000, 95% CI) + Wilcoxon, the usual verdict rule (inconclusive if the CI contains 0 or
family-Holm p > 0.05):

- {`dice_ET`, `dice_TC`, `dice_WT`, `lwdice_ET`} × {SSA held-out n = 30, PED held-out n = 49} = 8
- {`dice_ET`, `dice_TC`, `dice_WT`} on BraTS test n = 189 = 3 (the in-distribution cost)

**Primary: `dice_TC` per external cohort**, pooled − frozen.

**Decision rule, per external cohort, on the primary:**

| Outcome | Verdict |
|---|---|
| pooled better, CI excluding 0, Holm-significant | **RECOVERS** |
| inconclusive | **NO MEASURABLE RECOVERY** (resolution-limited, never "no effect") |
| pooled worse, Holm-significant | **HARMS** |

**In-distribution cost, on BraTS test:** any of the three Holm-significantly worse → **COSTS IN
DISTRIBUTION**, reported beside the external verdicts; otherwise "no measurable in-distribution
cost". The C1 headline is never re-computed from this model.

**Predictions, registered (directional, not gating):**

1. SSA · TC: a smaller gain than D3 fold 0's on the same 30 cases — 25 SSA cases among 945 carry
   less weight than a dedicated fine-tune.
2. PED · TC: improves, because training on PED teaches the paediatric label convention (note 58:
   PED ground truth is ~84% NCR/NET, ~0% ED); the predicted PED TC/WT ratio moves toward 1, but
   less than D3's 0.27 → 0.95.
3. BraTS test: no measurable cost.

**Secondary, descriptive (not in the family):**

1. D2 − D3 fold 0 on the same held-out cases (`dice_TC`, SSA and PED): which fix is larger.
2. PED · TC recalibratable? Conformal curve on D2's held-out PED logits; R(τ_min) reported against
   the 0.20 threshold D3 used.
3. Predicted TC/WT ratio on PED and SSA (the label-convention mechanism of note 58).
4. **Duplicate sensitivity**: the family re-run with `exclude_case_ids` =
   [`BraTS-PED-00121-000`, `BraTS-PED-00137-000`] (byte-identical upstream; one is in train, one in
   held-out test). A verdict that flips under exclusion is reported as fragile.

## Deployment consequence

None. The deployed checkpoint stays `neurovision` seed 42. D2 is evidence about what the developer
could have done, not a model this project ships; it was trained on half of each external cohort.

## What would make this invalid

- Any recipe change other than the split file (seed, epochs, patch, augmentation, LR, selection).
- Editing `splits_pooled_cf0.yaml` after the first session starts.
- Comparing on any external case that is in D2's train or val keys.
- Selecting `last.pt` vs `best.pt` after seeing test numbers (`best.pt`, fixed above).
- Reporting held-out external numbers without "cross-fitted".

## Cost and abort condition

`neurovision` measured 0.272 h/epoch on 875 cases (probe v4); 945 cases → ~0.294 h/epoch, 80 epochs
≈ 24 GPU-h + validation ≈ **26 GPU-h in three chained T4 sessions** (`max_hours` 10.5, resume from
`last.pt`). **Abort** if session 1's measured epoch time implies more than **35 GPU-h** for 80
epochs; an aborted run is reported as aborted, never as a shorter-schedule result.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first number
exists.)*

**Written 2026-11-24, from one evaluation run.** Training: three chained Kaggle T4 sessions
(2026-10-17 → 10-21, ≈ 25 GPU-h, recipe unchanged), `best.pt` = epoch 79, pooled-val `dice_mean`
0.8855 (`docs/data_manifests/checkpoints.sha256`). Evaluation: `scripts/evaluate.py` on the Mac CPU,
268 pooled-test cases, `save_logits=true`, finished 2026-11-24 02:54; lesion-wise replay
(self-consistency |Δ| ≤ 4.4e-17 against the evaluation's own table). Family: `scripts/compare_family.py
analysis=d2_pooled` (`configs/analysis/d2_pooled.yaml`); comparisons run on the `case_id` intersection,
which is exactly the held-out cases — n = 30 / 49 / 189, checked. **CROSS-FITTED ON ONE FOLD, NOT
EXTERNAL VALIDATION.**

**The family of 11 (pooled − frozen, paired bootstrap 95% CI, Holm across all 11):**

| Cohort | Metric | n | Pooled | Frozen | Δ [95% CI] | p_holm | Verdict |
|---|---|---|---|---|---|---|---|
| SSA | `dice_ET` | 30 | 0.8233 | 0.7673 | +0.0560 [+0.0228, +0.0980] | 4.7e-06 | better |
| SSA | **`dice_TC`** | 30 | 0.8092 | 0.8138 | **−0.0046 [−0.0360, +0.0206]** | 1.00 | inconclusive |
| SSA | `dice_WT` | 30 | 0.9272 | 0.9217 | +0.0055 [−0.0028, +0.0156] | 0.73 | inconclusive |
| SSA | `lwdice_ET` | 30 | 0.5529 | 0.4927 | +0.0601 [−0.0016, +0.1262] | 0.014 | inconclusive (CI contains 0) |
| PED | `dice_ET` | 49 | 0.6619 | 0.6365 | +0.0254 [+0.0026, +0.0563] | 1.00 | inconclusive |
| PED | **`dice_TC`** | 49 | 0.7116 | 0.4886 | **+0.2231 [+0.1471, +0.3045]** | 3.0e-10 | **better** |
| PED | `dice_WT` | 49 | 0.8869 | 0.8762 | +0.0107 [+0.0034, +0.0200] | 0.23 | inconclusive |
| PED | `lwdice_ET` | 49 | 0.4839 | 0.4794 | +0.0045 [−0.0366, +0.0399] | 1.00 | inconclusive |
| BraTS test | `dice_ET` | 189 | 0.8718 | 0.8709 | +0.0009 [−0.0018, +0.0039] | 1.00 | inconclusive |
| BraTS test | `dice_TC` | 189 | 0.9118 | 0.9161 | −0.0042 [−0.0138, +0.0043] | 1.00 | inconclusive |
| BraTS test | `dice_WT` | 189 | 0.9319 | 0.9321 | −0.0002 [−0.0025, +0.0018] | 1.00 | inconclusive |

**Verdicts, by the rule fixed above:**

- **SSA · TC: NO MEASURABLE RECOVERY** (resolution-limited at n = 30; the point estimate is −0.005).
- **PED · TC: RECOVERS** (+0.223).
- **BraTS test: no measurable in-distribution cost** (all three |Δ| ≤ 0.0042, every CI contains 0).

Outside the primary: SSA `dice_ET` +0.056 is Holm-significant (better). It is reported, not promoted:
the registered question was tumour core.

**Registered predictions.** (1) SSA · TC smaller than D3 fold 0 on the same 30 cases — **held**
(D2 −0.005 vs frozen; D3 fold 0 0.8441 vs D2 0.8092, D2 − D3 = −0.035). (2) PED · TC improves and the
predicted TC/WT ratio moves toward 1, but less than D3 — **held** (median ratio frozen 0.31 → D2 0.64 →
D3 0.93 on the same 49 cases; cases below 0.5: 73% → 41% → 6%). (3) BraTS test no measurable cost —
**held**.

**Secondaries (descriptive).**

1. *D2 − D3 fold 0, same held-out cases* (`dice_TC`): SSA −0.035, PED −0.130 (D2 0.7116 vs D3 0.8419).
   The dedicated fine-tune is the larger fix on both cohorts.
2. *PED · TC recalibratable?* Curves on D2's held-out PED logits (`scripts/conformal.py`, BraTS-val fit,
   `outputs/conformal/neurovision_pooled_d2/`). R(τ_min), τ_min = 1e-4, same 49 cases: frozen 0.311,
   **D2 0.053**, D3 0.010 — below the 0.20 line, so a local recalibration is feasible. At the frozen
   BraTS-val thresholds the PED · TC miss rate is still 0.246 / 0.388 / 0.507 at α 0.05 / 0.10 / 0.20:
   pooled training moves the floor, not the deployed thresholds.
3. *Predicted TC/WT ratio:* PED above; SSA median 0.41 frozen → 0.43 D2 → 0.33 D3 (no label-convention
   change to learn on SSA, as expected).
4. *Duplicate sensitivity* (`d2_pooled_sens_nodup`, PED n = 48): **no verdict changes**; PED `dice_TC`
   +0.2248 [+0.1468, +0.3045].

**What it means.** Putting a few dozen paediatric cases in the training set teaches the model the
paediatric tumour-core convention without measurable cost on adult test cases — but it recovers less
than a fine-tune on the same cases, and it does not move SSA tumour core at all. Pooling narrows the
label-definition failure; it does not remove the need for a site-side check, because the deployed
thresholds stay miscalibrated on PED. Note 63, claim C31.
