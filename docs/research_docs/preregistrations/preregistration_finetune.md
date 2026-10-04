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

**Filled 2026-10-04.** Full write-up: `experiments.md` note 58; claim C27. Every number below is
**CROSS-FITTED, NOT EXTERNAL VALIDATION**. "Cross-fitted" means each cohort was cut into two halves,
one model was fine-tuned on each half, and each model was scored only on the half it never saw. Every
case is scored once, but each model did see half of its own cohort, so this is not a test on a new
site.

### What ran

Four Kaggle T4 fine-tunes, 2026-10-03/04, `GIT_REF fb8b1b4`. Init: the deployed `neurovision` seed-42
`best.pt` (epoch 69, wandb `cc2l5j1c`, val dice_mean 0.8938), weights only. lr 2e-5, ~3,000 steps per
fold, `last.pt` scored, as registered.

| Fold | Epochs | Held-out cases |
|---|---|---|
| ssa-cf0 | 120 | 30 |
| ssa-cf1 | 120 | 30 |
| ped-cf0 | 67 | 49 |
| ped-cf1 | 68 | 50 |

Every fold: `NVX_HEALTH: OK`, `NVX_EVAL: OK`, peak VRAM 6.16 GiB allocated. About 31 s/epoch on
ssa-cf0 (~1.2 GPU-h including eval); all four ~5 GPU-h, inside the registered ~5–6. Pooled held-out:
SSA 60/60 and PED 99/99, each case exactly once; held-out ∩ training = 0 per fold, by split file.
Evaluation ran on Kaggle with `save_logits=true`; lesion-wise columns came from
`scripts/replay_logits.py`.

### Primary family (8 comparisons, one Holm correction)

`outputs/compare_family/d3_finetune_crossfit/family.csv`. Paired bootstrap (n_boot 10000) plus
Wilcoxon. Fine-tuned minus frozen.

| Cohort | Metric | n | Δ | 95% CI | p_holm | Verdict |
|---|---|---|---|---|---|---|
| SSA | dice_ET | 60 | +0.0497 | [0.0311, 0.0709] | <1e-4 | better |
| SSA | **dice_TC (primary)** | 60 | **+0.0433** | [0.0094, 0.0785] | 0.0007 | better |
| SSA | dice_WT | 60 | +0.0193 | [0.0087, 0.0313] | <1e-4 | better |
| SSA | lwdice_ET | 60 | +0.0727 | [0.0102, 0.1356] | 0.0008 | better |
| PED | dice_ET | 99 | +0.0223 | [−0.0216, 0.0651] | 0.0883 | inconclusive |
| PED | **dice_TC (primary)** | 99 | **+0.3676** | [0.2952, 0.4375] | <1e-4 (Wilcoxon raw 1.8e-13) | better |
| PED | dice_WT | 99 | +0.0032 | [−0.0250, 0.0269] | 0.0883 | inconclusive |
| PED | lwdice_ET | 99 | −0.0259 | [−0.0769, 0.0229] | 0.6230 | inconclusive |

Mean `dice_TC`, frozen to fine-tuned: SSA 0.7846 to 0.8279; PED 0.4394 to 0.8070.

**Verdict by the registered rule: RECOVERS on both cohorts.** Deployment consequence: none. The
deployed model stays `neurovision` seed 42.

**Read the PED TC number with the post-hoc mechanism below.** Most of it is the model adopting the
paediatric label definition.

### Secondary 1: is PED · TC recalibratable after fine-tuning?

Registered prediction: R(τ_min) for PED · TC drops below 0.20. **HELD.** It went from 0.3564 (frozen)
to **0.0476** (fine-tuned) at τ_min = 1e-4. Source: `outputs/conformal/neurovision_ft_ped`
(`scripts/conformal.py` on `outputs/eval_ft_ped_pooled`; the fit is on frozen BraTS val, only the PED
curves are used downstream).

Local recalibration, same k grid, 1000 splits, seed 42, counterfactual
(`outputs/local_recalibration_ft_ped`), PED · TC:

| α | Min feasible k | Structural floor | Note |
|---|---|---|---|
| 0.20 | 6 | 4 | |
| 0.10 | 18 | 9 | mean realised risk at k = 49: 0.0829, feasible 98.1%, RESTORED |
| 0.05 | never | — | feasible at most 15.9% at any k, NOT_RESTORED |

PED · WT min feasible k: 62 / 14 / 5 at α 0.05 / 0.10 / 0.20.

With the frozen BraTS-val thresholds (no recalibration), the fine-tuned PED · TC realised risk at
nominal α 0.10 is 0.1849, CI [0.1470, 0.2287]. The frozen model gave 0.651 (C22). Still violated, much
less.

### Secondary 2: per fold (held-out mean dice)

| Fold | ET | TC | WT |
|---|---|---|---|
| ssa-cf0 | 0.8349 | 0.8441 | 0.9342 |
| ssa-cf1 | 0.8212 | 0.8117 | 0.8963 |
| ped-cf0 | 0.6343 | 0.8419 | 0.8911 |
| ped-cf1 | 0.5380 | 0.7728 | 0.8141 |

SSA cf0 frozen on the same 30 cases: 0.7673 / 0.8138 / 0.9217. Neither PED fold looks like the pooled
result is driven by one fold, but PED cf1 is weaker than cf0 on all three regions.

### Secondary 3: monitoring-val curves

**Not pulled.** Only `NVX_HEALTH: OK` per fold is on record. The claim "did not diverge" rests on that
line and on the held-out scores, not on the curves.

### Post-hoc mechanism (exploratory, NOT pre-registered)

Median share of whole-tumour voxels in the ground truth (`meta.json` `label_voxel_counts`):

| Cohort | n | NCR/NET | ED | ET | TC/WT | Cases with ED < 5% |
|---|---|---|---|---|---|---|
| BraTS 2021 | 1251 | 0.09 | 0.65 | 0.22 | 0.35 | 0% |
| SSA | 60 | 0.05 | 0.66 | 0.23 | 0.34 | 0% |
| PED | 99 | 0.84 | 0.00 | 0.11 | 1.00 | 75% |

Predicted TC/WT voxel ratio (logit > 0), median, with the share of cases below 0.5: PED frozen 0.27
(74%), PED fine-tuned 0.95 (4%). SSA frozen 0.35 (76%), SSA fine-tuned 0.31 (78%).

The paediatric reference counts the non-enhancing, T2-bright tumour as core; the adult convention
labels that region oedema. The frozen adult model draws an adult-shaped core, and fine-tuning teaches
it the paediatric definition. So the PED TC gain is mostly a label-definition change, not better
imaging generalisation. SSA is the control: its label definition matches adult BraTS and its
composition did not move, so SSA TC +0.043 is the cleaner evidence of local adaptation.

### Data finding found after the splits were written

BraTS-PED-00121-000 and BraTS-PED-00137-000 are the same scan twice (identical raw SHA-256 on all five
NIfTIs; byte-identical preprocessed arrays). The split put them in opposite folds, so each PED
fine-tune trained on the twin of one held-out case. Sensitivity family without both cases
(`outputs/compare_family/d3_finetune_crossfit_sens_nodup`, n = 97): PED `dice_TC` +0.3668 [0.2945,
0.4395], p_holm <1e-4. No verdict changed. PED `dice_WT` p_holm 0.0481, still inconclusive because its
CI contains 0. The registered split files were not touched; the primary family above is the
registered one.

### Limitations

Forgetting on BraTS test was not measured (registered as not measured). The four `last.pt`
checkpoints have not been copied off Kaggle (`amishyadav123/neurovision-d3-{ssa,ped}-cf{0,1}`), so
they cannot be re-scored locally yet.
