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

**Completed 2026-09-24.** This is a measurement, not a hypothesis test, so there is no ADOPT/NULL/
REJECT verdict to fire — the result below is the noise floor itself.

### What ran

`neurovision_seed43`, per the recipe rule above: D0 fired **NULL** (`preregistration_augmentation.md`),
so this run trains on the current `_baseline_common.yaml` augmentation, unchanged. Three chained
Kaggle T4 sessions (`neurovision-d1-seed43-s1`, `-s2`, `-s3`, epoch 0→33, 33→70, 70→79), one W&B run
throughout (`iriee13d`), `GIT_REF=9c770ce6a83213931475c05f4850da253a3009de` pinned for all three
sessions, `NVX_HEALTH: OK` and `nonfinite=[]` at every session exit. Final train loss 0.4545, peak
VRAM 7.66 GiB reserved. Checkpoint: `outputs/neurovision_seed43/checkpoints/best.pt` (epoch 79,
val/dice_mean 0.8947, against seed 42's 0.89380).

Evaluation: `scripts/evaluate.py` on BraTS test (189), SSA (60) and PED (99), Mac CPU,
`sw_batch_size=1`, `data.num_workers=0`, `save_logits=true` — the same code path as the seed-42 run
and every other published number:

| Cohort | dice_ET | dice_TC | dice_WT | dice_mean |
|---|---|---|---|---|
| test (n=189) | 0.8730 | 0.9145 | 0.9336 | 0.9070 |
| SSA (n=60) | 0.7894 | 0.7723 | 0.9018 | 0.8212 |
| PED (n=99) | 0.5590 | 0.4647 | 0.8501 | 0.6246 |

Lesion-wise columns came off the saved fp16 logits via `scripts/replay_logits.py`
(`.venv-analysis`, `analysis.replay.lesionwise.enabled=true`), zero further inference. A
self-consistency check against `evaluate.py`'s own output matched exactly (mean absolute delta
~1e-17) on all three splits.

### The family, in full, as declared

`scripts/compare_family.py` (`.venv-analysis`), family `d1_seed_noise_floor`,
`neurovision_seed43` vs `neurovision` (seed 42, deployed), the fixed 12-comparison family, one Holm
correction across all of it (`outputs/compare_family/d1_seed_noise_floor/family.csv`, n_boot=10000).
Paired difference is seed43 − seed42.

| Cohort | Metric | n | Δ | 95% CI | p_holm (m=12) | Verdict |
|---|---|---|---|---|---|---|
| test | dice_ET | 189 | +0.0021 | [-0.0018, +0.0068] | 1 | inconclusive |
| test | dice_TC | 189 | -0.0015 | [-0.0098, +0.0046] | 0.3068 | inconclusive |
| test | dice_WT | 189 | +0.0015 | [-0.0009, +0.0040] | 1 | inconclusive |
| test | lwdice_ET | 189 | -0.0070 | [-0.0269, +0.0133] | 1 | inconclusive |
| SSA | dice_ET | 60 | +0.0111 | [-0.0053, +0.0294] | 1 | inconclusive |
| SSA | dice_TC | 60 | -0.0123 | [-0.0370, +0.0073] | 1 | inconclusive |
| SSA | dice_WT | 60 | +0.0059 | [-0.0006, +0.0137] | 0.2484 | inconclusive |
| SSA | lwdice_ET | 60 | +0.0267 | [-0.0215, +0.0788] | 1 | inconclusive |
| PED | dice_ET | 99 | -0.0044 | [-0.0431, +0.0331] | 1 | inconclusive |
| PED | dice_TC | 99 | +0.0253 | [+0.0042, +0.0472] | 0.2484 | inconclusive |
| PED | dice_WT | 99 | +0.0011 | [-0.0181, +0.0175] | 1 | inconclusive |
| PED | lwdice_ET | 99 | +0.0393 | [-0.0040, +0.0857] | 1 | inconclusive |

**12 of 12 inconclusive.** PED `dice_TC`'s CI clears zero on its own ([+0.0042, +0.0472]) but does
not survive the family-wide Holm correction (p_holm 0.2484) — reported as a near miss, the same
convention D0 used for its own primary endpoint, not rounded into a result.

### What the number is for — the two pre-registered readings

**(1) The headline +0.0267 ET Dice architecture claim (C1, over `baseline_unet3d`, test).** Noise
floor: test `dice_ET` seed43−seed42 = +0.0021, CI [-0.0018, +0.0068] — about 13x smaller than the
published margin. Per this file's own decision rule ("if several times smaller, the headline stands
with the noise floor printed beside it"), the headline **stands**.

**(2) D0's pooled SSA+PED `dice_TC` result (C23, +0.0117, CI [-0.0006, +0.0245], p_holm 0.936,
already inconclusive on its own).** The 12-item family above is deliberately per-cohort, not pooled,
so this reading needs a separate pooled noise-floor number. A descriptive-only comparison (not part
of the 12-item Holm family, no additional Holm correction — `compare_models` called directly on the
pooled SSA+PED case list, n_boot=10000) gives:

**Pooled SSA+PED `dice_TC`, seed43−seed42 = +0.0111, CI [-0.0056, +0.0276], n=159.**

This is essentially the same magnitude as D0's own +0.0117. D0's heavy-augmentation result is
therefore **not distinguishable from seed noise** — this independently reinforces D0's NULL verdict
(which was already inconclusive on its own bootstrap CI) rather than contradicting it.

### Deployment consequence

None, as specified above. The deployed clinical checkpoint stays `neurovision` seed 42 regardless;
this was a measurement, not a model candidate.

### What this does and does not license

- It does **not** license treating `neurovision_seed43` as a second independent confirmation of any
  claim — that is explicitly excluded above ("What would make this invalid").
- It does **not** license picking whichever seed looks better on a given cohort; neither seed is
  "the model" for anything beyond this noise-floor measurement.
- It does license printing a noise floor beside the two margins named above, and it does license
  reading D0's own result in light of that floor.
- With two seeds there is still no seed-level variance estimate, only one difference, exactly as the
  Power statement above says. A third seed (`baseline_unet3d_seed43`, then `neurovision_seed44`)
  remains the next run in the queue if Kaggle quota allows, not started here.

---

## Amendment 1 — `baseline_unet3d_seed43`, fixed 2026-09-26 before the run is launched

**Why.** The Power statement above registered `baseline_unet3d_seed43` as the next run, "first (the
comparison needs both arms' noise)". The D1 result measured only the `neurovision` arm's seed noise.
The headline +0.0267 is a *difference between two architectures*, each trained once, so its honest
noise floor needs a second seed of the comparator too. Milestone 5 (`master_plan.md` P2.5, P4) queues
it. Nothing above the `## Result` line is edited.

**Arm.** `baseline_unet3d_seed43`: `configs/experiment/baseline_unet3d.yaml` **exactly**, with
`seed: 43` and `data.num_workers=2` (as every Kaggle run). 64³, 80 epochs, current augmentation
(D0 was NULL). One Kaggle T4 session (the U-Net costs roughly a tenth of `neurovision` per epoch,
`experiments.md` probe row), `max_hours` 10.5, `GIT_REF` pinned. Checkpoint scored: `best.pt`, the
same selection rule the seed-42 baseline used.

**Evaluation.** `scripts/evaluate.py` on test (189), SSA (60), PED (99), `save_logits=true`, Mac CPU,
then `scripts/replay_logits.py` for the lesion-wise columns — the same path as every other row.

**Endpoints — descriptive, as above.**

1. **Baseline noise floor**: the same 12-comparison family as D1 ({`dice_ET`, `dice_TC`, `dice_WT`,
   `lwdice_ET`} × {test, SSA, PED}), `baseline_unet3d_seed43` − `baseline_unet3d`, one Holm family.
2. **The headline, re-read at seed 43 — the one reading this run exists for.** Test `dice_ET`,
   `neurovision_seed43` − `baseline_unet3d_seed43`, paired bootstrap CI (n_boot = 10000) and Wilcoxon,
   reported beside the published seed-42 margin (+0.0267, CI 0.0166–0.0393).
   - If its CI excludes zero with the same sign, C1 is stated as **replicated across two seed
     pairs**.
   - If its CI contains zero, C1 is weakened to **seed-dependent** in the paper and the claims table,
     whatever Gate A says.
   - If it is significant in the opposite direction, C1 is **withdrawn**.
3. **Seed-averaged margin**, descriptive: per case, mean of the two `neurovision` seeds minus mean of
   the two baseline seeds, test `dice_ET`, paired bootstrap CI. Printed; no decision rides on it.

**Deployment consequence.** None. A second baseline seed is a measurement.

**Abort.** If session 1's per-epoch time implies more than 10 GPU-h, or `NVX_HEALTH` is not OK.
