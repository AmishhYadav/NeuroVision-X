# Handoff — wiring the unwired library code

Working note for resuming after a session break. Delete once all five items
are done and `CLAUDE.md`'s status section has absorbed them.

**Started:** 2026-08-06

## The problem this work solves

Five subsystems were implemented and unit-tested but had **no producer** — no
entry point wrote their inputs, so no result file for them existed on disk and
`notebooks/09_paper_figures.ipynb` §10 listed them as blocked. The library code
was never the gap; the wiring was.

## Items

| # | Item | State |
|---|---|---|
| 1 | MC-dropout in `scripts/evaluate.py` | **done** |
| 1b | `save_logits` in `scripts/evaluate.py` | **done** |
| 2 | `scripts/extract_gates.py` — fusion gate maps | code complete, **NOT REVIEWED** |
| 3 | Calibration / ECE producer | not started |
| 4 | Boundary-stratified metrics | not started |
| 5 | Explainability driver script | not started |

## Item 1 — done

`cfg.inference.mc_dropout.enabled` was read by nothing. Now wired.

- `evaluate_case` returns a `CaseOutput` dataclass, not a tuple.
- **The deterministic pass still owns every Dice/HD95 number.** MC adds N
  passes on top rather than replacing the first, costing ~10% over N passes.
  This is what keeps the already-published `baseline_unet3d` row in
  `experiments.md` valid whether or not evaluation is re-run with uncertainty
  on. `predictions_from: mc_mean` flips it, logs a one-time warning, and is an
  ablation — not a default.
- Writes `<eval_dir>/uncertainty/<case>.npy` (mutual information, float16,
  cropped). That exact directory name is a contract
  `notebooks/09_paper_figures.ipynb` reads; do not rename it.
- `save_fields` opts into the other `MCDropoutOutput` tensors. Default is
  MI only: all four fields at float16 is ~79 MB/case, ~15 GB over 189 cases,
  past Kaggle's 20 GB `/kaggle/working` quota on its own.
- `uncertainty_summary.csv` carries per-case MI scalars so the analysis
  survives when the volumes are too large to keep. `mi_mean_fg_<REGION>` is
  **NaN**, not 0.0, when that region's prediction is empty — "nothing to
  average over" and "certain about everything" must not collapse to one number.
- Deliberate asymmetry: under `predictions_from="mc_mean"`, `predictions/` and
  `probabilities/` come from DIFFERENT passes. `probabilities/` always means
  the deterministic pass so one directory name never denotes two quantities.
  Add `"mean_prob"` to `save_fields` for the matching probabilities.

Committed by the implementing subagent as `d434de8` (unrequested; tests were
left uncommitted).

## Item 1b — done

`save_logits` writes float16 raw logits to `<eval_dir>/logits/`.

**Why probabilities cannot substitute.** Measured: fp16's gap below 1.0 is
0.000488, so any probability above ~0.99976 round-trips to exactly 1.0 and its
logit is `+inf`. A trained segmentation net is that confident across most of
the tumor core. Fitting a temperature from saved probabilities would therefore
have to drop or clamp precisely the voxels that drive miscalibration — biasing
the fitted temperature toward 1.0 and understating overconfidence, with nothing
failing anywhere. Logits sit at ~±20, far inside fp16's ±65504.

The saturation is one-sided: near 0, fp16 subnormals reach 6e-8, so `p -> 0`
survives. Only `p -> 1` collapses. ECE/MCE/Brier and the reliability diagram
are unaffected either way — 0.9998 and 1.0 fall in the same bin.

`logits/` is captured before the `mc_mean` branch can rebind `regions`, so it
always means the deterministic pass.

## Item 2 — code complete, NOT REVIEWED

**Resume here.** The implementing subagent was stopped at a session boundary
after writing `scripts/extract_gates.py` (585 lines),
`tests/test_extract_gates.py` (541 lines, 14 passing),
`configs/explainability/default.yaml`, and an edit to `configs/config.yaml`'s
`defaults` list. Verified at the stop: the full suite is green (843 passed),
`scripts/smoke_test.py` passes, and `scripts/show_config.py` composes — so the
tree is consistent, not half-broken.

What did NOT happen: no review pass against the spec, and no check of the four
things a subagent usually gets wrong here (crop-origin off-by-one, `None` and
empty-list gate handling, the `has_label: False` error path, and whether the
manifest columns match what the figure will need). Read the file before
trusting it. The design rationale below is what the spec was built from.


`scripts/extract_gates.py` + `configs/explainability/default.yaml`.

**Deviates from what notebook §10 proposes**, deliberately. §10 asks for a
`--save-gates` path in `scripts/evaluate.py` writing `<eval_dir>/gates/`. That
is the wrong shape: `forward_with_gates` takes a whole volume, evaluation is
sliding-window over 96³ patches, and gates come out as a four-level pyramid at
strides 2/4/8/16 while MONAI's inferer stitches exactly one output.

Instead: one 96³ patch per case, centred on the tumor — exactly what the model
saw during training — through `forward_with_gates` in a single pass.

- ~0.25 MB/case across all four levels. 32 cases is ~8 MB.
- Runs on CPU. The mechanism figure costs no Kaggle hours.
- Makes P1 quantitative rather than decorative: gate value vs.
  distance-to-tumor-boundary is a well-defined correlation over a few dozen
  cases, so "the gate opens at the margin" becomes a number with a CI (via
  `analysis/statistics.py`) instead of three panels.
- Cost: no evidence about gate behaviour in healthy tissue far from the
  lesion. Acceptable — P1 does not claim anything there.

**§10 must be updated to describe this**, not left describing a path that was
never built.

## Item 3 — design decided, not started

`scripts/calibrate.py`, file-driven and CPU-only (no checkpoint, no GPU).
Reads `<eval_dir>/logits/` (preferred) or `probabilities/` plus the
preprocessed labels, streams through `uncertainty/calibration.py`'s
`CalibrationAccumulator`, and writes ECE / MCE / Brier per region, reliability
curves, and risk-coverage curves.

**Temperature must be fit on VAL and applied to TEST.** Fitting on test is a
leak that invalidates the calibration claim, which is this project's headline.
Enforce it structurally — a script that reads one `eval_dir` and fits on it is
the wrong interface.

Risk-coverage needs per-case uncertainty scalars (`case_uncertainty_scalars`)
joined against per-case Dice from `per_case_metrics.csv`. Both already exist.

## Items 4 and 5 — not started

4. Boundary-stratified metrics. `metrics/segmentation.py` scores whole regions
   only. The distance-to-boundary machinery this needs is also what item 2's
   gate-vs-boundary correlation wants, so build it once and reuse.
5. Explainability driver writing attribution maps to disk, so
   `09_paper_figures.ipynb` can stay file-driven and CPU-only.

## Kaggle side — separate track, not blocked on any of the above

Dataset uploaded and live (`amishyadav123/neurovision-brats-prep`). Repo pushed.
One baseline trained (`baseline_unet3d`, 200 epochs, 16.5 GPU h). Still to run:
`baseline_unet3d` re-run at the experiment file's own 100-epoch schedule (~8 h),
`baseline_swinunetr` (~25 h), `neurovision` (~15 h), then the 6-row ablation
grid. See `docs/experiments.md`.

Do items 1–3 before spending GPU on `neurovision`: otherwise that run finishes
and the calibration result still cannot be stated, costing another session to
re-evaluate.
