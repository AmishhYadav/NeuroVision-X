# Experiment Log

Every training run goes in this file. One row per run, appended when the run
**finishes** (or is abandoned), never before. If a run is not in this table it
did not happen and its numbers may not be used in the paper.

A "run" is one `experiment_name` + one seed. A run split across several Kaggle
sessions by resume is still ONE row — sum the GPU hours.

---

## How to fill a row

| Column | Where it comes from |
|---|---|
| **Run** | `experiment_name` from the config, plus `-s<seed>` if not 42 (e.g. `baseline_unet3d-s1`) |
| **Config** | The `+experiment=` file, plus any CLI override that changed the run. `-` if none |
| **Seed** | `cfg.seed` |
| **Model** | `cfg.model.name` and parameter count printed at startup |
| **Epochs** | `epochs_done / epochs_planned`. Not equal means the run was cut short — say why in Notes |
| **GPU h** | Sum of Kaggle session wall-clock across every resume. Kaggle shows this per session |
| **Dice ET/TC/WT** | `summary.csv` from `scripts/evaluate.py`, `mean` row. Test split |
| **HD95 ET/TC/WT** | Same file. **Millimetres** — `evaluate.py` passes spacing; the trainer's val HD95 is in voxels and does NOT go in this table |
| **ECE** | Calibration, once implemented. `-` until then |
| **W&B** | Run id / short URL |
| **Notes** | Anything that would change how the number is read: OOM, restarts, a config edit mid-run, a suspicious loss curve |

**Rules**

- Numbers come from `scripts/evaluate.py` on the **test** split, at
  `inference.sliding_window.overlap: 0.5`. The `val/dice_mean` in W&B is a
  monitoring signal at overlap 0.25 and is not comparable — do not paste it here.
- `ignore_empty=False` (BraTS convention) throughout. On BraTS 2021, 2.6% of
  cases have no enhancing tumor, so this moves ET Dice by well under a point,
  but it must still be stated in the paper.
- `et_min_volume` stays 0. If a row was ever produced with it on, say so in
  Notes — otherwise the number is not comparable to the others.
- If you change a config between two runs you intend to compare, that is a new
  experiment file, not an edit to an existing one.

---

## Runs

| Run | Config | Seed | Model | Epochs | GPU h | Dice ET | Dice TC | Dice WT | HD95 ET | HD95 TC | HD95 WT | ECE | W&B | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_unet3d` | root config, **not** `+experiment=` — see note 1. Overrides: `model=unet3d training.epochs=200 training.val_interval=2 data.dataset_type=dataset data.num_workers=2` | 42 | `unet3d`, 12.87M | 200 / 200 | 16.5 | 0.8587 | 0.9157 | 0.9354 | 4.02 | 5.21 | 5.36 | — | `nz5y7li7` | 1–5 below |

**Notes for `baseline_unet3d` / `nz5y7li7`**

1. **The schedule is not the one `configs/experiment/baseline_unet3d.yaml`
   describes, so this row is not directly comparable to any future run made
   from that file.** `_baseline_common.yaml` was written 2026-08-02, the day
   *after* this run started. Measured differences: `epochs` 200 vs 100,
   `val_interval` 2 vs 5, `scheduler.warmup_epochs` 10 vs 5, and validation
   `sliding_window.overlap` 0.5 vs 0.25 (monitoring only — the reported
   numbers here come from `scripts/evaluate.py` at overlap 0.5 either way).
   The recommendation in `docs/reproducibility.md` §8 is to re-run at 100
   epochs under the experiment file and retire this row, so that the baseline
   table is internally consistent with `baseline_swinunetr` and `neurovision`,
   which will both inherit `_baseline_common`. Until then: a 200-epoch
   baseline against a 100-epoch proposed model favours the baseline, so the
   direction of the bias is at least not self-serving — but it is not a
   controlled comparison and must not be presented as one.
2. **Two Kaggle sessions, one run**, resumed from `last.pt`: 10.93 h
   (2026-08-01, epochs 1–131) + 5.54 h (2026-08-02, epochs 132–200) = 16.47 h,
   rounded to 16.5 above. Tesla T4. Peak VRAM **1.17 GB** — the U-Net leaves
   almost the whole 16 GB card unused.
3. **HD95 ET is over n = 183, not 189.** Six cases are NaN because exactly one
   side of the ET region was empty; `hd95()` deliberately does not collapse
   that into a number. Dice/IoU are over all 189. `gt_empty_ET` on the test
   split is 0.0265 (5 of 189), so those five score ET Dice as empty-vs-empty
   under `ignore_empty=False` rather than by real overlap.
4. **No commit SHA was recorded.** `notebooks/kaggle_train.ipynb` cloned
   `GIT_REF = "main"` and W&B captured no git metadata, so the exact source
   revision is not recoverable — only that it was `main` between 2026-08-01
   16:30 and 2026-08-02 12:16 UTC (commits `3144175`..`9665c70`). Every future
   run pins `GIT_REF` to a SHA.
5. W&B ran in `offline` mode and was synced afterwards; the run id is the same
   across both sessions because it lives in the checkpoint. Full environment,
   seeds and runtimes: `docs/reproducibility.md`.

---

## Planned

Written before the runs start so the plan is on record and cannot be
retrofitted to whatever came out.

| Run | Config | Purpose | Est. GPU h |
|---|---|---|---|
| `baseline_unet3d` | `+experiment=baseline_unet3d` | Milestone-1 baseline A. The number the fusion model must beat. Re-run at the file's own 100-epoch schedule; see note 1 on the row above. | ~8 |
| `baseline_swinunetr` | `+experiment=baseline_swinunetr` | Milestone-1 baseline B. Same schedule, same seed, same data — architecture is the only variable. | ~25 |
| `neurovision` | `+experiment=neurovision` | The proposed model, same `_baseline_common` schedule as both baselines. | ~15 |

The first two estimates are now **re-planned against measurement**, not against
the original paper-FLOP calculation. Measured: 16.47 GPU-h for 200 U-Net epochs
= **0.082 h/epoch**, so 100 epochs is ~8 h — the original `~12` was high by
about 50%. SwinUNETR-B is ~3× the U-Net per epoch (62.19M params plus 20–30%
for gradient checkpointing) → ~25 h. NeuroVision-X at 34.88M sits between them;
that row is still a pure estimate, since nothing of that architecture has run
on a GPU. Re-plan again once each first session reports a real epoch time.

The 6-row fusion ablation grid runs a shortened 40-epoch schedule and is priced
separately by `python scripts/run_ablation_grid.py` — re-run it with a measured
`--sec-per-step` now that one exists.

---

## Abandoned / failed runs

Record these too. A run that OOM'd at epoch 3 or was killed for a config bug is
evidence about the setup, and forgetting it means repeating it.

| Run | Config | GPU h burned | What happened |
|---|---|---|---|
| _(none yet)_ | | | |
