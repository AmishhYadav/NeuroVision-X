# Research docs — everything the paper is written from

Gathered here on 2026-09-27 so the paper (MELBA) and the semester report have one folder to work
from. Planning and operations docs stayed in `docs/` and `docs/research/` (listed at the bottom).

## Read in this order

0. **`status_2026-09-27.md`** — start here when picking the project up cold: what is done,
   what is left (with the exact command to restart each open thread), and why the project exists.
1. **`claims_and_evidence.md`** — the gate. Every claim the paper may make, with the artifact
   behind it. A number not in this table does not go in the paper.
2. **`experiments.md`** — every run and every measured result (numbered notes). The source of truth
   for any number `claims_and_evidence.md` cites.
2b. **`project_history.md`** — the same project told in time order, with every GPU run and every
   decision; read it to get the whole journey back.
3. **`project_review_2026-09-24.md`** — outside-view review; why the thesis was rewritten for
   Milestone 5.
4. **`related_work.md`** — where the conformal and refusal claims sit, and what novelty the paper
   must *not* claim.

## Files

| File | What it is | Paper section it feeds |
|---|---|---|
| `status_2026-09-27.md` | Handoff: done / left / how to restart / why / end result | — (orientation) |
| `project_history.md` | The whole project in time order: every build, GPU run, result, dead claim and major decision since 2026-07-31 | Introduction, Discussion (the story arc), Methods (why the schedule is what it is) |
| `claims_and_evidence.md` | Claim table (C1…C25), each tied to an artifact | All — the gate |
| `experiments.md` | Run log + numbered result notes | Results |
| `contribution.md` | The original pre-registered claim, with a status update on what survived | Introduction, Discussion |
| `related_work.md` | Nearest prior work to C14 and the refusal gate | Related work |
| `project_review_2026-09-24.md` | Outside-view critique of the whole project | Discussion, Limitations |
| `model_card.md` | Model card for the deployed `neurovision` seed 42 | Methods, appendix |
| `reproducibility.md` | Seeds, versions, hardware, runtimes, data provenance, rebuild commands | Methods, reproducibility statement |
| `phase0_atlas_findings.md` | Measured SRI24 atlas properties (incl. the mirrored-atlas trap) | Methods (anatomy/atlas) |
| `lessons.md` | Engineering traps with evidence | Limitations, "pitfalls" discussion |
| `preregistrations/` | The 11 pre-registrations, each written before its data existed | Methods (hypotheses + decision rules) |
| `protocols/` | Fixed-definition protocols: end-to-end error budget, real-DICOM validation | Methods |
| `semester_report/` | Semester report draft (Results/Discussion/Limitations) and its figures | Course report; figure source |

### Pre-registrations

| File | Question |
|---|---|
| `preregistration_ambiguity.md` | Does inter-branch disagreement detect failure? (Gate 1) |
| `preregistration_gate2.md` | Gate 2, respecified around voxel-level localisation |
| `preregistration_external_hd95.md` | HD95 direction under distribution shift (SSA/PED) |
| `preregistration_conformal.md` | Distribution-free bound on the mask's miss rate, and where it breaks |
| `preregistration_qc.md` | Can a QC model predict its own error, and survive shift? (Gate C) |
| `preregistration_strong_baseline.md` | Does the architecture gain survive nnU-Net? (Gate A) |
| `preregistration_augmentation.md` | Does heavier augmentation close the SSA/PED gap? (D0) |
| `preregistration_multiseed.md` | Second seed: how much of every number is noise? (D1) |
| `preregistration_finetune.md` | Cross-fitted fine-tuning on the external cohorts (D3) |
| `preregistration_tta.md` | Flip test-time augmentation measurement |
| `preregistration_ood.md` | Input-statistics OOD score for the gate, and its switch-on rule (P1.4) |

## Stayed outside this folder

Some pre-registrations cite these, so they are listed here too:

- `docs/research/master_plan.md` — the active plan and queue (not paper content)
- `docs/research/gpu_request.md` — the college GPU request draft
- `docs/demo_runbook.md` — the semester demo script and day-before checks
- `docs/research/execution_plan.md`, `docs/research/interpretable_pipeline_plan.md` — earlier plans
- `docs/project_state.md` — Milestone 1–3 build record (useful for the Methods section)
- `docs/data_manifests/` — SHA-256 manifests for the SSA/PED raw data
- Result CSVs and figures live under `outputs/` (gitignored): `outputs/paper/`, `outputs/figures/`
