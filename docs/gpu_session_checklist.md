# GPU session checklist

Every rule here was written against a loss this project has already suffered.
Run through it before starting a session and before letting one end.

## Before the session

- [ ] `pytest` green from the repo root (plain `pytest` — `pyproject.toml` already
      sets `addopts = "-q"`, so an extra `-q` stacks to `-qq` and silently drops
      the pass count).
- [ ] `python scripts/smoke_test.py` exits 0 (~4 s, CPU).
- [ ] Everything committed **and pushed**. A GPU box clones the repo at run time,
      so uncommitted work does not exist for that session.
- [ ] `GIT_REF` pinned to a **commit SHA**, never a branch name. Then clone the
      pinned tree and assert the fix you are relying on is actually in it —
      verifying `origin/main` proves nothing about a pinned SHA. Note `git clone -b`
      accepts a branch or tag only; clone then `git checkout <sha>` as separate
      `subprocess.run(..., check=True)` calls, and do not use `--depth 1`.
- [ ] `grad_clip_norm` identical across every run in a comparison family
      (currently **5.0**). At 1.0, 66–70% of steps clipped, which silently changes
      the effective learning rate and confounds any ablation.
- [ ] W&B in **online** mode if the box has internet. Offline mode plus an
      ephemeral filesystem is how run metrics get lost.
- [ ] On new hardware only: a **2-epoch timing probe** reporting measured step
      time and peak VRAM. Never schedule a long run from an estimated speedup.

## Probe design rule

A probe must reach the **failure condition**, not merely execute. Two GPU-hour
losses in this project share one shape: the pre-flight *read* the changed code
path but never *ran* it at settings where it could fail. Ask what state the real
run reaches that the probe will not, and force it — `probe_saturation` used 10x
LR specifically to drive the branch probes past p = 0.9995 in 20 short epochs.

## Before the session ends

- [ ] Checkpoint written to a **persistent** path and copied off the box.
      `capacity_control` went to `/tmp/capout/` and the 200-epoch baseline to
      `~/Downloads`; both are permanently gone and both runs are now
      unreproducible without retraining.
- [ ] Carry `best.pt` forward as well as `last.pt`. `save_checkpoint` only writes
      `best.pt` when validation improves, so a final session that does not improve
      never creates one in its own output directory.
- [ ] Training log fetched and kept. The capacity control's GPU hours read
      "~8, approximate" purely because its log was never retrieved.
- [ ] Resolved HEAD SHA present in the log.

## What comes back — and what does not

**Bring back:** checkpoints, the log, the W&B run. That is the whole list.

**Leave behind:** `logits/`, `predictions/`, `uncertainty/`. Deterministic
evaluation runs on the Mac CPU at ~15 cases/min — all 189 test cases in
~25 minutes — so a full evaluation is cheaper to redo locally from the checkpoint
than to transfer. This is what keeps the local disk from refilling.

## Getting data onto the box

- Code, configs, frozen splits, docs, knowledge base: `git clone`. All tracked
  (256 files, 8.5 MiB).
- `data/preprocessed/brats` (34 GB): the Kaggle dataset
  `amishyadav123/neurovision-brats-prep`, or `rsync` from the Mac. **This dataset
  is now the only backup of the preprocessed arrays — do not delete it.**
- `data/preprocessed/{brats_ssa,brats_ped}` (4.2 GB): `scripts/package_for_kaggle.py`
  or `rsync`. Needed only for the pooled-cohort run.
- Raw data is no longer on the Mac. Re-download paths and SHA-256 manifests are in
  `docs/data_manifests/`.
