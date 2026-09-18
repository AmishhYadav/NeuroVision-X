# Pre-registration — D1, a second seed: how much of every number is noise?

**Written:** 2026-09-18, before any second-seed checkpoint of any arm exists, and while D0's third
Kaggle session (`neurovision-d0-s3`) is still training. The git timestamp is the evidence.

**Scope.** `docs/research/master_plan.md` §5 Phase D, item D1, cut to its minimum per the plan's own
cut order ("D1 from 3 seeds to 2, then D3. Never D2."). The Kaggle quota is 30 GPU-h/week and one
80-epoch `neurovision` run costs ~23 GPU-h across three sessions, so **one additional seed of the
deployed architecture** is what fits. The baseline's second seed is registered here as the *next*
run in the queue, not part of this one.

---

## Why this run exists

Every number in this project is single-seed. `claims_and_evidence.md` carries "no claim may rest on
a margin smaller than the between-run noise we cannot measure" as a standing limitation, and the
D0 pre-registration says outright that a single-seed-vs-single-seed comparison "cannot separate
'the augmentation helped' from 'this seed happened to land differently'". D1 is the instrument that
turns that sentence into a number. It is not a hypothesis test; it is a measurement of the
instrument's resolution.

## Arm

| Arm | Definition |
|---|---|
| `neurovision_seed43` | `configs/experiment/neurovision.yaml` recipe **exactly**, with `seed: 43`. 64³, 80 epochs, three Kaggle sessions, `GIT_REF` pinned to one SHA for all three |

**Recipe rule, fixed now, so D0's outcome cannot be used to pick a flattering base.** The
augmentation block is chosen by D0's own pre-registered decision rule
(`preregistration_augmentation.md`): if D0 fires **ADOPT**, this run inherits
`neurovision_heavy_aug`'s `data.augment` (as that rule already prescribes for D1); on **NULL**,
**TRADE-OFF** or **REJECT** it uses the current `_baseline_common.yaml` augmentation. Whichever
applies is recorded in the Result section with D0's verdict beside it. Nothing else changes.

## Endpoints

All on the same case sets and metric code path as the seed-42 run: `scripts/evaluate.py` on BraTS
test (n=189), SSA (n=60), PED (n=99), `save_logits=true`, followed by `scripts/replay_logits.py`
for lesion-wise columns.

**Primary — descriptive, no decision rule:** for each of `dice_ET`, `dice_TC`, `dice_WT`,
`lwdice_ET` on each cohort, the **paired seed-to-seed difference** seed43 − seed42 with a paired
bootstrap CI (`analysis.statistics.compare_models`, Holm across the family below). The absolute
value of the point estimate, and the half-width of its CI, are the "noise floor" every earlier
margin is to be read against.

**The family, fixed:** {`dice_ET`, `dice_TC`, `dice_WT`, `lwdice_ET`} × {test, SSA, PED} = 12
comparisons, Holm-corrected together.

**What the number is for.** Two published margins get re-read against it, and only these two:

1. The headline **+0.0267 ET Dice** over `baseline_unet3d` (test). If |seed43 − seed42| on
   `dice_ET` (test) is of the same order as 0.0267, the headline is stated in the paper as "within
   one seed's spread" and the architecture claim is weakened to *suggestive*; if it is several times
   smaller, the headline stands with the noise floor printed beside it.
2. D0's `dice_TC` pooled-shift result, whatever it turns out to be, is read the same way.

No other claim is re-adjudicated from this run. In particular a seed-43 result that happens to look
better or worse than seed 42 on some cohort is **not** a new finding — it is the noise.

**Power statement.** With two seeds there is no seed-level variance estimate, only one difference.
This is a floor, not a distribution. Three seeds would give the first standard deviation; that is
the next run if quota allows, and it is registered here as `baseline_unet3d_seed43` *first* (the
comparison needs both arms' noise), then `neurovision_seed44`.

## Deployment consequence

None. The deployed clinical checkpoint stays `neurovision` seed 42 regardless of this result; a
second seed is a measurement, not a candidate. (B3's deep ensemble would use both, but B3 is not
started and needs the baseline's seeds too.)

## What would make this invalid

- Any config difference from the seed-42 run other than `seed`, and the augmentation block *only if*
  D0's rule selects it.
- Evaluating on a different case list, or through any path other than `scripts/evaluate.py`.
- Picking which of the two seeds to call "the model" after seeing the numbers.
- Reporting the seed-43 checkpoint's numbers as a second independent confirmation of any claim.

## Cost and abort condition

~23 GPU-h, three sessions of ≤10.5 h (`max_hours` in `_baseline_common.yaml`), same driver notebook
as D0 with `CKPT_SLUG` chained. Abort the run if session 1's per-epoch time implies more than 30
GPU-h total, or if `NVX_HEALTH` reports non-finite metrics at any session exit.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first number
exists.)*
