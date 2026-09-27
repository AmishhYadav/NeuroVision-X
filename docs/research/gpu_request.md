# College GPU request — NeuroVision-X

**Drafted 2026-09-27** (`master_plan.md` P2.6). To be submitted ~2026-11-05, once every run below
is pre-registered and launch-ready. Every cost here is **measured on a Tesla T4**, not estimated;
the source note is given for each.

## What changed since this request was planned

The core Phase 4 runs (Gate A nnU-Net, D3 fine-tunes, TTA, baseline seed 43) moved to Kaggle T4 on
2026-09-26 (author decision, `master_plan.md` Kaggle ledger), within Kaggle's 30 GPU-h/week cap.
This request therefore covers what does **not** fit that cap: the stretch runs, and a contingency
for a core run that has to be repeated.

## The ask

| Item | Hardware | Hours |
|---|---|---|
| One GPU with **≥ 16 GB** VRAM (T4, V100, A10, A100 or better) | single card, no multi-GPU | **~60 GPU-h** over Dec 2026 – Feb 2027 |
| Disk | persistent scratch | **~40 GB** (preprocessed data 12 GB, checkpoints, logs) |
| Internet | outbound HTTPS | `git clone` from GitHub, `pip install` from PyPI at job start |
| Session length | any; jobs resume from checkpoint | wall limit ≥ 4 h preferred; shorter works, every run resumes |

## Runs, with measured cost

| Run | What it answers | Measured basis | T4-h |
|---|---|---|---|
| **D2** pooled multi-cohort `neurovision` (BraTS + SSA + PED, one seed, 64³, 80 epochs) | Does training on the shifted cohorts close the tumour-core gap? | 0.272 h/epoch on 875 cases (probe v4, `experiments.md` probe table); ~1,000 training cases → ~0.31 h/epoch × 80 + validation | ~30–40 |
| **Capacity control** `unet3d` width-matched, seed 42, 80 epochs | Restores the lost checkpoint behind C2 (79% architecture / 21% capacity), so it can be re-scored lesion-wise | U-Net family at ~0.082 h/epoch (probe table), wider variant ~1.5× | ~8 |
| **Contingency** | One repeat of any core run (a D3 fold, the TTA session, or two Gate A sessions) | as measured on Kaggle | ~12–20 |
| **Total** | | | **~50–68** |

Core runs on Kaggle, for reference: Gate A nnU-Net 1000 epochs at **209 s/epoch** (session 1,
2026-09-27) ≈ 58 GPU-h across ~6 sessions; D3 ~1 GPU-h per fine-tune; TTA ~2 GPU-h.

## Memory

- `neurovision` at 64³, AMP, no gradient checkpointing: **peak 6.17 GiB allocated / 7.4 GiB
  reserved** (probe v4 and every production session since). Fits any ≥ 16 GB card with room.
- U-Net family: peak < 1.2 GiB.
- The code is written against a **16 GB portability floor** (`CLAUDE.md` constraint 1). On a larger
  card nothing needs changing.

## Software stack

Python 3.11, PyTorch (stock CUDA wheels, sm_70+), MONAI, Hydra, Weights & Biases (offline mode
works), NumPy/SciPy/Pandas, Nibabel/SimpleITK. Everything installs from `requirements.txt` in a
fresh virtualenv; no Docker, no root, no system packages beyond a CUDA driver.

## Operating model

- **Launcher:** `scripts/gpu_session.py` — one entry point for Kaggle, SSH and SLURM sessions. It
  enforces a `max_hours` budget below the scheduler's wall limit (there is no SIGTERM handler, so
  the budget is the safety margin).
- **Resume:** every run resumes from `last.pt` (model, optimizer, scheduler, AMP scaler, epoch,
  step, RNG states, W&B run id). A pre-empted job loses at most one epoch.
- **Data out:** only checkpoints and logs leave the cluster. Evaluation runs on the author's laptop.
- **Pre-registration:** every run above has (or will have, before submission) a committed
  pre-registration in `docs/research/`, fixed before any number exists.

## Open questions for the cluster admins

1. Access: SLURM or direct SSH?
2. Per-job wall limit and per-user GPU-hour quota?
3. Is outbound internet available from compute nodes (GitHub, PyPI)?
4. Which card model, and its usable VRAM?
