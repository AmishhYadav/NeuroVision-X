# Pre-registration — flip test-time augmentation (A4 measurement)

**Written:** 2026-09-27, before any TTA number exists. The git timestamp is the evidence for that
ordering. `master_plan.md` Milestone 5, P2.5 ("TTA measurement") and P4.

**Scope.** Flip TTA was wired and tested in Milestone 3 (`inference.tta`, A4) and never measured:
at ~80 s per case per pass on the M4, 8 flips over 189 test cases is ~34 h of Mac time (note 43),
so the measurement moved to a GPU session. This document fixes, before the run, what is measured
and what it is allowed to change.

---

## Why this run exists

1. **Gate A fairness, stated both ways.** nnU-Net's primary arm predicts with mirroring on (its
   standard recipe — `preregistration_strong_baseline.md` Amendment 1 item 7); `neurovision` is
   compared without TTA. That comparison stays exactly as registered. A `neurovision`+TTA row lets
   the reader see the TTA-vs-TTA gap beside it, the same way nnU-Net's `--disable_tta` row lets
   them see the no-TTA-vs-no-TTA gap. Neither enters the Gate A Holm family.
2. The literature prices flip TTA at +0.003–0.008 Dice for BraTS-type models. Our own seed noise
   floor (note 49) is +0.0021 on test ET. Whether TTA clears that floor is a fact the paper needs.

## Design

**Arm.** `neurovision_tta`: the deployed `neurovision` seed-42 `best.pt` (run 2, epoch 69, W&B
`cc2l5j1c`), `scripts/evaluate.py` with `inference.tta.enabled=true`, `inference.tta.axes=[0,1,2]`
(all 8 flip combinations — the config default, fixed a priori: the model trains with independent
flips on each spatial axis and no other spatial augmentation, so these are exactly the invariances
it was asked to learn). Everything else identical to the published `neurovision` test row:
roi 64³, overlap 0.5, Gaussian blending, project default post-processing (threshold 0.5,
`min_component_size` 50, nesting enforced).

**Comparator.** The existing `neurovision` test row, `outputs/neurovision/eval_test/per_case_metrics.csv`
(ET 0.8708593). Paired on the same 189 cases.

**No val step.** The Milestone 3 plan said "measure on val, then test". That order exists to choose
a free parameter on val. There is none: the axes are fixed above and nothing is tuned. Running val
would spend GPU time without informing any decision, so it is not run.

**Cohort.** Test (189) only. SSA and PED are not measured: TTA is not a distribution-shift
intervention and the thesis does not ask the question.

## Endpoints

**Family, fixed:** {`dice_ET`, `dice_TC`, `dice_WT`} on test, `neurovision_tta` − `neurovision`,
`scripts/compare_family.py`, paired bootstrap (n_boot 10000, 95% CI) + Wilcoxon, one Holm family
of 3, the project's usual verdict rule (inconclusive if the CI contains 0 or p_holm > 0.05).

**Primary:** `dice_ET` (the region carrying the project's one positive claim, C1).

**Reading, fixed now:**

| Outcome on the primary | Wording in the paper |
|---|---|
| TTA better, CI excluding 0, Holm-significant | "flip TTA adds +x ET Dice (CI) — reported as its own row, never folded into C1" |
| inconclusive | "flip TTA gain not resolvable at n = 189" (never "no effect") |
| TTA worse, Holm-significant | reported as found |

**Secondary, descriptive:** the size of the TTA gain beside the seed noise floor (+0.0021, note 49)
and beside the Gate A margin, once that exists.

## Deployment consequence

**None, whatever the result.** The clinical tool runs on CPU, where 8x inference turns ~80 s per
study into ~11 min. TTA stays off in `configs/clinical/default.yaml` and in every published row. A
positive result is a sentence in the paper, not a deployment change.

## What would make this invalid

- Scoring any checkpoint other than the deployed `neurovision` `best.pt`.
- Changing axes, overlap, roi, or post-processing after seeing a number.
- Writing the TTA result into `outputs/neurovision/eval_test` (it goes to its own out_dir,
  `outputs/eval_test_neurovision_tta`, as `configs/inference/default.yaml`'s `tta` comment demands).
- Comparing a TTA number against a non-TTA number anywhere without saying so.

## Cost and abort condition

Pure inference, one Kaggle T4 session: 189 cases × 8 sliding-window passes. Estimated ~2 GPU-h.
`save_logits=false`, `save_predictions=false` — only `per_case_metrics.csv` and `summary.csv` come
back (volume caches never cross the wire). **Abort** if the first 5 cases imply more than 5 GPU-h
for the split. Budgeted in the Kaggle ledger (`master_plan.md`, week of 2026-10-17).

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first number
exists.)*
