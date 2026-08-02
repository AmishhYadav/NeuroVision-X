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
| _(none yet)_ | | | | | | | | | | | | | | |

---

## Planned

Written before the runs start so the plan is on record and cannot be
retrofitted to whatever came out.

| Run | Config | Purpose | Est. GPU h |
|---|---|---|---|
| `baseline_unet3d` | `+experiment=baseline_unet3d` | Milestone-1 baseline A. The number the fusion model must beat. | ~12 |
| `baseline_swinunetr` | `+experiment=baseline_swinunetr` | Milestone-1 baseline B. Same schedule, same seed, same data — architecture is the only variable. | ~30 |

Estimates are from a paper-FLOP calculation, not measurement. Replace them
with the real figure after the first session and re-plan if they are badly off.

---

## Abandoned / failed runs

Record these too. A run that OOM'd at epoch 3 or was killed for a config bug is
evidence about the setup, and forgetting it means repeating it.

| Run | Config | GPU h burned | What happened |
|---|---|---|---|
| _(none yet)_ | | | |
