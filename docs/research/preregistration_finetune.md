# Pre-registration — D3, cross-fitted fine-tuning on the external cohorts

**Written:** 2026-09-26, before the cross-fit split files exist and before any fine-tune has run.
The git timestamp is the evidence for that ordering.

**Scope.** `docs/research/master_plan.md` §5 Phase D, item D3, as re-scoped by Milestone 5 (P2.2,
P2.5, P4): fine-tune on **both** SSA and PED (not SSA only), with two-fold cross-fitting so every
external case is still scored exactly once by a model that never saw it.

---

## Why this run exists

The thesis (`CLAUDE.md`) says the distribution-free bound "fails under distribution shift in
proportion to the shift", and asks "what a new site needs in order to restore it". P1.1 (conformal
Amendment 1) answers the *threshold* half: how many labelled local cases a site needs to recalibrate.
It already predicts that **PED · TC is infeasible at every k** — the paediatric tumour-core miss rate
is 0.356 even at the lowest threshold, so no threshold fixes it. D3 answers the *model* half: if the
site's labelled cases are used to fine-tune the network instead, does segmentation on its unseen
cases recover, and does PED · TC become recalibratable?

## Design

**Cross-fitting.** Each cohort is split once, with a seeded generator, into two halves
(`configs/data/splits_{ssa,ped}_cf{0,1}.yaml`, written by `scripts/make_crossfit_splits.py` from
`analysis.crossfit_splits`, **committed before any training**). Fold *f* fine-tunes on half *f*'s
cases and is scored on the other half. From each training half, **5 cases** are carved off as a
monitoring-only val set (`val_n: 5`); they are never used to select a checkpoint. Pooling the two
folds' held-out predictions gives every case of the cohort exactly one score (SSA n = 60, PED n = 99),
paired against the frozen model on the same case.

**Label.** The numbers are **cross-fitted, not external validation**: each fine-tuned model has seen
half of its own cohort. Every place they are printed says so.

**Arms.**

| Arm | Definition |
|---|---|
| `neurovision` (frozen) | the deployed seed-42 `best.pt`, unchanged — its existing per-case SSA and PED rows |
| `neurovision_ft_{ssa,ped}_cf{0,1}` | four fine-tunes, `training.checkpoint.init_from` = the deployed `neurovision` checkpoint (weights only: fresh optimizer, scheduler, scaler, epoch 0) |

**Fine-tune recipe, fixed now** — `configs/experiment/neurovision.yaml` unchanged except:

- `training.optimizer.lr` **2.0e-5** (one fifth of the from-scratch 1.0e-4, the conventional
  fine-tune reduction; not tuned — there is no legitimate data to tune it on).
- `training.scheduler.warmup_epochs` **2**, cosine to `min_lr` 1.0e-6 as before.
- **Budget: ~3,000 optimizer steps per fine-tune**, so the two cohorts get the same amount of
  optimisation regardless of size: `training.epochs = round(3000 / n_train)` where `n_train` is the
  fold's training-case count after the 5 monitoring cases are removed (one step per case per epoch
  at `batch_size: 1` × `samples_per_volume: 4`). The exact per-fold epoch counts are computed and
  written into each fold's run record from the committed split files **before** launch.
- `data.root_dir` / `data.splits.path` point at the cohort's preprocessed data and the fold's split
  file. Seed 42. Augmentation, loss, patch size (64³), AMP: unchanged.
- **Checkpoint scored: `last.pt`.** Not `best.pt` — selecting on 5 monitoring cases would be a
  noisy, optimistic choice made on cases from the same small cohort.

## Endpoints

All via `scripts/evaluate.py` (`save_logits=true`) on each fold's held-out half, then
`scripts/replay_logits.py` for the lesion-wise columns — the same metric path as every other row.

**Primary: `dice_TC`**, pooled over both folds, per cohort, fine-tuned vs frozen, paired. TC is the
region that fails under shift (pooled external `dice_TC` −0.0333, note 30; PED TC 0.46).

**The family, fixed:** {`dice_ET`, `dice_TC`, `dice_WT`, `lwdice_ET`} × {SSA, PED} = 8 comparisons,
`scripts/compare_family.py`, one Holm correction, paired bootstrap (n_boot 10000) + Wilcoxon, the
project's usual verdict rule (inconclusive if the CI contains 0 or p_holm > 0.05).

**Decision rule, per cohort, on the primary:**

| Outcome | Verdict |
|---|---|
| fine-tuned better, CI excluding 0, Holm-significant | **RECOVERS** |
| inconclusive | **NO MEASURABLE RECOVERY** (resolution-limited, never "no effect") |
| fine-tuned worse, Holm-significant | **HARMS** |

**Secondary, reported, not gating.**

1. **Does fine-tuning make PED · TC recalibratable?** Conformal curves of the pooled cross-fitted
   PED logits (`scripts/conformal.py`), then the P1.1 local-recalibration driver on them at the same
   k grid and α levels, as a counterfactual. Prediction, fixed now: R(τ_min) for PED · TC drops
   below 0.20, so PED · TC becomes feasible at α = 0.20 for some k ≤ 49. If it does not, the paper
   says the paediatric failure survives both fixes a site can make on its own.
2. Per-fold numbers (fold 0 and fold 1 separately), to show the pooled result is not one fold.
3. Monitoring-val curves, to show the fine-tune did not diverge.

**Not measured:** forgetting on BraTS test. Scoring four fine-tunes on 189 test cases is ~18 CPU-hours
for a question the thesis does not ask; it is stated as a limitation.

## Deployment consequence

None. The deployed checkpoint stays `neurovision` seed 42. A cohort-specific fine-tune is evidence
about what a site *could* do, not a model this project ships.

## What would make this invalid

- A split file written or regenerated after any fine-tune number exists.
- Any held-out case appearing in its own fold's training or monitoring set (a test asserts
  disjointness when the split files are written).
- Scoring `best.pt`, or choosing lr / epochs after seeing held-out numbers.
- Reporting the fine-tuned numbers as external validation.

## Cost and abort condition

Measured `neurovision` cost on a T4 is ~1.14 s per optimizer step (D0: 998 s / 875 steps), so ~3,000
steps is ~1 GPU-h per fine-tune plus validation and setup — **~5–6 GPU-h for all four**, run as
single Kaggle sessions (one per fold, `max_hours` 10.5). Abort a fine-tune if its first epoch implies
more than 3 GPU-h for that fold, or if `NVX_HEALTH` is not OK.

---

## Result

*(To be completed after the runs. Nothing above this line may be edited once the first number
exists.)*
