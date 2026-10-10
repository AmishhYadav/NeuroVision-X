# Pre-registration — capacity control, second draw (`capacity_control_unet3d_r2`)

**Written:** 2026-10-11, before the run is launched and before any number from it exists. The git
timestamp is the evidence for that ordering. `master_plan.md` Milestone 5, P4 stretch ("capacity
control (+8)"); moved from the college card to Kaggle by author decision 2026-10-11.

---

## Why this run exists

C2 — "the ET gain is mostly architectural, not a capacity artifact" (~79% architecture, ~21%
capacity) — rests on one run, `capacity_control_unet3d` (seed 42, `GIT_REF f363067`, W&B
`k9j5uba2`, notes 19–21). Its checkpoint, logits and predictions are **permanently lost**; only its
`per_case_metrics.csv` survives (`outputs/eval_test_capacity_control/`, test ET 0.8497). Two
consequences:

1. **C2 cannot be scored lesion-wise.** Every other headline arm has a lesion-wise row (C10/C11);
   the capacity decomposition is the one claim still voxel-only.
2. **C2 is a single draw.** C1 now has a measured seed noise floor (note 49, note 54); C2 does not.

A retrain with the identical recipe fixes the first and gives a second draw for the second. It is
**not a restoration**: GPU training is not bit-deterministic, so the rerun is a new sample from the
same recipe, reported beside the original, never in place of it.

## Design

**Arm.** `capacity_control_unet3d_r2`: `+experiment=capacity_control_unet3d`, **seed 42**, the
recipe unchanged since `f363067` (verified 2026-10-11: the only later commits touching the
composed config add D0's transforms *gated off by default* and the inactive `init_from` key).
64³ patches, 80 epochs, `grad_clip_norm` 5.0, best-val `dice_mean` checkpoint (`best.pt`) — exactly
as the original. One Kaggle T4 session, `max_hours` 10.5; if 80 epochs do not fit, the same
notebook resumes from `last.pt` in a second session (constraint 4), recipe untouched.

**Evaluation.** `scripts/evaluate.py` on the 189 test cases with `save_logits=true`, on the Mac
CPU, then `scripts/replay_logits.py` lesion-wise — the same path as every other arm. Output dirs:
`outputs/eval_test_capacity_control_r2/`, `outputs/replay_lesionwise/eval_test_capacity_control_r2/`.
The original `outputs/eval_test_capacity_control/` is never overwritten.

## Endpoints

**Family, fixed now:** one Holm family of **12**, `scripts/compare_family.py`, paired bootstrap
(n_boot 10000, 95% CI) + Wilcoxon, the project's usual verdict rule (inconclusive if the CI contains
0 or family-Holm p > 0.05):

| Pairing | Metrics |
|---|---|
| `neurovision` − `capacity_control_unet3d_r2` (the **architecture** share) | `dice_ET`, `dice_TC`, `dice_WT`, `lwdice_ET`, `lwdice_TC`, `lwdice_WT` |
| `capacity_control_unet3d_r2` − `baseline_unet3d` (the **capacity** share) | the same six |

**Primary:** `dice_ET` and `lwdice_ET` of `neurovision` − r2.

**Reading, fixed now.** Architecture share = (ET`neurovision` − ET`r2`) / (ET`neurovision` −
ET`baseline_unet3d`), computed on voxel `dice_ET` (and, separately, on `lwdice_ET`).

| Outcome | Wording |
|---|---|
| `neurovision` − r2 on `dice_ET` better, Holm-significant, and architecture share ≥ 0.5 | C2 **stands**, now on two draws; quote the share as a range over the two draws (original 0.79, r2 x) |
| `neurovision` − r2 on `dice_ET` better, Holm-significant, share < 0.5 | C2 **weakens**: "the gain splits between architecture and capacity"; both draws printed |
| `neurovision` − r2 on `dice_ET` inconclusive or worse | C2 **withdrawn** on this draw; both draws printed, and C2 may not open the paper |

The lesion-wise rows are new evidence, not a re-test: they are reported as found under the same
family and become C2's lesion-wise column.

**Secondary, descriptive (not in the family):** r2 − original, paired per case on test
`dice_ET/TC/WT`, printed beside the U-Net seed-to-seed shift (+0.0041 ET, note 54) and the
`neurovision` floor (+0.0021, note 49). A difference larger than both is reported as a
reproducibility finding about the recipe, not explained away.

## What would make this invalid

- Any change to the recipe, seed, epochs, patch size, or checkpoint-selection rule.
- Scoring `last.pt` instead of `best.pt` (the original scored `best.pt`).
- Choosing between the original and r2 after seeing numbers — both are always reported.
- Writing r2's numbers into `outputs/eval_test_capacity_control/`.

## Cost and abort condition

The original took "~8 GPU-h" on a T4 (approximate — its log was never retrieved). Budget **10.5
GPU-h in one session**; if the first session's per-epoch time implies more than **16 GPU-h** for 80
epochs, stop and report rather than chain more than two sessions.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first number
exists.)*
