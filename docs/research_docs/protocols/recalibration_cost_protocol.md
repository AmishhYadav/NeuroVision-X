# Protocol — the cost of local recalibration (MasterPlan W3.5)

**Written:** 2026-11-25, before any RCPS local-recalibration number exists. **Status: DESCRIPTIVE /
EXPLORATORY** — not one of the six confirmatory families (§B3 rule 4). No verdict here is a test of a
hypothesis; it prices a recipe whose feasibility is already established (C24, C27, C34).

## Question

C24 says a new site can restore the conformal bound by recalibrating on its own labelled cases —
"~10–40 of them depending on α" — under CRC (v1 settings, two models). The paper's recipe paragraph
needs the price: how many labelled cases under **CRC and RCPS** (v2 settings, all three models),
what the recalibrated mask looks like (its precision, Dice and volume ratio — the W2.1e "mask
cost"), and what labelling those cases costs a site in annotator time.

## Design — fixed now

- **Inputs:** the conformal v2 per-case loss curves (`results/conformal_v2/curves/`,
  `results/conformal_v2_nnunet/curves/`), v2 settings unchanged: grid per model as in
  `results/conformal_v2/run_meta.json` (`neurovision` and `baseline_unet3d` extended, nnU-Net v1),
  empty handling ET `zero`, TC/WT `exclude`, δ = 0.1 for RCPS (WSR bound). Nothing is re-inferred.
- **Cells:** 3 models × cohorts {test (control), SSA, PED} × regions {WT, TC, ET} × α {0.05, 0.10,
  0.20} × method {CRC, RCPS}.
- **k grid:** {5, 10, 15, 20, 25, 30, 40, 50, half}, keeping only k ≤ n − 10 (at least 10 held-out
  cases). 1,000 seeded within-cohort splits per (cell, method, k), seed 42, as C24.
- **Per split:** fit the threshold on the k calibration cases with the v2 fitter for that method;
  on the held-out cases, realised mean miss rate and the mask quality at that threshold (precision,
  Dice, volume ratio, from `conformal.mask_quality`).
- **Verdict per (cell, method, k)** — the C24 rule, unchanged: `INFEASIBLE` if no split is feasible;
  `BORDERLINE` if < 2 feasible splits; `RESTORED` if the mean held-out risk over feasible splits is
  ≤ α; `NOT_RESTORED` if it exceeds α by more than twice its Monte Carlo SE; otherwise `BORDERLINE`.
- **k needed** (fixed now): the smallest k on the grid with `RESTORED` **and** feasible rate ≥ 0.90
  (a site that recalibrates must get a threshold almost every time, not 4 times in 10). If none:
  "not restorable on this grid".
- **Mask cost:** median over feasible splits, at k needed, of held-out precision, Dice and volume
  ratio; beside the same quantities for the frozen v2 threshold on that cohort (C34), so the table
  shows what recalibration buys and what it costs in over-segmentation.
- **Annotation time:** k needed × minutes per case. Primary: **60 min** per case for manual
  BraTS-style sub-region segmentation (AJNR 2022;43(1):24, "approximately 1 hour" per case,
  verified against the article before quoting). Sensitivity: correcting a model's pre-annotation
  instead of drawing from scratch, **2.21× faster** (Shirokikh et al., arXiv 1909.02799, prospective
  radiosurgery study; a different task — single-target contours — so it is stated as a sensitivity
  bound, not a measured glioma figure).

## What would make this misleading

- Reading `k needed` as a guarantee: it is a Monte Carlo summary over splits of a counterfactual.
- Quoting the annotation hours as measured: they are a literature rate times k.
- Changing the k grid, the 0.90 feasibility rule or the minutes per case after seeing the table.

## Reporting

One numbered note; a table in `results/recalibration_cost/`; the recipe paragraph gains a cost
sentence. C24 is amended with the v2/RCPS numbers if they differ.
