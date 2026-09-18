# Phase G protocol — the end-to-end error budget

**Written:** 2026-09-18, before any pipeline-level number exists. This is not a gated
pre-registration (Phase G is descriptive: it measures, it does not decide between hypotheses), but
the definitions below are fixed *before* the script runs so that no bar can be moved to make a
table look better. The git timestamp is the evidence.

**Scope.** `docs/research/master_plan.md` §5 Phase G, items G1–G3 and G5, on the three cohorts with
ground truth that exist today. G4's fourth cohort (UCSF-PDGM) does not exist — Phase F is not
started — and the DICOM path (ingest → clinical preprocessing) is reported on the n=3 TCIA fixtures
from note 45, with no truth, as what it is: an anecdote, not a rate.

---

## The question

The master plan's principle 2: five stages at 95% each is 77% end to end. Every stage here has been
measured in isolation. Nobody has measured what a study gets *after passing through all of them*,
and what the refusal gate buys at the pipeline level. This measures exactly that.

## Cohorts and inputs

| Cohort | n | Segmentation | Signals |
|---|---|---|---|
| BraTS test | 189 | `outputs/neurovision/eval_test` | same dir (logits) |
| SSA | 60 | `outputs/eval_ssa_neurovision` | same |
| PED | 99 | `outputs/eval_ped_neurovision` | same |

Model: the deployed `neurovision` checkpoint, `best.pt`, unchanged. QC model:
`outputs/neurovision/qc/best.pt`. Conformal: `outputs/conformal/neurovision`, α = 0.10 (the deployed
operating point in `configs/clinical/default.yaml`). Gate thresholds: `outputs/gatekeeper/
thresholds.json`, fitted on val (n=187), **frozen** — nothing is refitted here.

**Input QC is not applicable to these cohorts** and is scored as a pass by construction: they are
curated, skull-stripped, co-registered NIfTI from the challenge, i.e. the output of the stage input
QC exists to guard. Its real-world rate is the fixture anecdote only. The gate therefore runs on
`predicted_dice` and `conformal_band` alone, which is also exactly what it does on a real DICOM
study once E2 has succeeded.

## Definitions, fixed now

- **Usable segmentation (the per-case success bar).** `dice_WT ≥ 0.7` **and** `dice_TC ≥ 0.7`,
  voxel-wise, from the existing `per_case_metrics.csv`. 0.7 is the same bar
  `analysis.qc_validate.bad_dice_threshold` already uses in Gate C, chosen there before this
  document; it is reused, not tuned. ET is excluded from the bar because 2.6% of BraTS 2021 cases
  have no enhancing tumour and the metric is undefined on them; ET is reported alongside.
  A sensitivity sweep over {0.5, 0.6, 0.7, 0.8, 0.9} is reported so the reader can see how much the
  headline depends on the bar.
- **Gate decision.** `run_gatekeeper` with the frozen thresholds and the deployed
  `enabled_signals`. `accepted` = PROCEED or PROCEED_WITH_CAUTION; `refused` = REFUSE.
- **Guarantee met (per cohort, not per case).** Conformal risk control guarantees an *expected*
  miss rate ≤ α on exchangeable data, so "met" is a cohort statement: realised mean miss rate at the
  val-fitted threshold, with a 95% bootstrap CI, on (a) all cases and (b) accepted cases only. The
  per-case miss rate at that threshold is also reported descriptively (fraction of cases with
  miss rate ≤ α) but is *not* the guarantee.
- **The four outcome cells (G5 taxonomy)**, per cohort:

  | | usable | not usable |
  |---|---|---|
  | accepted | correct accept | **silent failure** — the number that matters |
  | refused | over-refusal (cost) | correct refusal |

  Refusals are further broken down by which signal fired (from `GateDecision.verdicts`).
- **End-to-end success (G2).** `P(accepted ∧ usable)` per cohort — the fraction of studies the
  pipeline hands back with a usable mask — and, conditional on acceptance,
  `P(usable | accepted)`, which is what the refusal gate is supposed to raise above the no-gate
  rate `P(usable)`. Both with 95% bootstrap CIs (percentile, case-resampled, seeded).
- **Coverage / accuracy trade-off (G3).** Re-fit the gate's quantile thresholds on the frozen val
  calibration table (`outputs/gatekeeper/calibration_table.csv`) at refuse quantiles
  {0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50} (0.0 is not a fittable quantile; the
  coverage = 1 point is the no-gate rate `P(usable)`), caution quantile held at the deployed
  0.10 (or `q + 0.01` where `q ≥ 0.10`, purely to satisfy the fitter's strict `refuse < caution`
  rule — caution never changes coverage, since cautioned cases are accepted), apply each to each cohort, and
  report coverage (fraction accepted) against `P(usable | accepted)` and mean `dice_TC` among
  accepted. This is the pipeline-level risk–coverage curve of the deployed gate; it is a
  *description* of the operating point, not a new calibration.

## What this deliberately does not do

- Does not refit any threshold used by the deployed gate. The 0.02/0.10 operating point stays.
- Does not run any model on new data; every number derives from artifacts on disk and one
  CPU pass of the QC model over each case's own saved logits (the same identity path
  `scripts/calibrate_gatekeeper.py` used).
- Does not claim OOD detection: `ood_score` is disabled in the gate and stays so.
- Does not touch WT claims in the paper — `claims_and_evidence.md`'s "no claim on WT" stands;
  WT enters the usability bar only as a per-case floor.

## What would make this invalid

- The usability bar, α, the quantile grid, or the accepted/refused mapping changed after the
  first table is seen.
- A cohort's signals computed from a different model's logits than its Dice.
- Reporting `P(usable | accepted)` without `P(accepted)` beside it — a gate that refuses
  everything is trivially perfect.

## Output

`outputs/error_budget/` — `per_case_<cohort>.csv` (signals, decision, dice, usable, cell),
`summary.csv` (one row per cohort × bar), `taxonomy.csv`, `coverage_curve.csv`,
`stage_reliability.csv`, and `error_budget_config.yaml`. Then note 47 in `docs/experiments.md`
and the Phase G table in the master plan.
