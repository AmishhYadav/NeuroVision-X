# Tool-completion log — status board for the autonomous run

**Started 2026-09-18 00:10 (author away; demo 2026-09-19 ~14:00).** The plan is
`tool_completion_plan.md`; this file is the mutable board — one line per unit, newest finding on top
of the Findings section. A session picking up cold: read this, then the plan, then continue the first
`[ ]` / `[~]` unit in order. T7 is never started.

Legend: `[x]` done + committed · `[~]` in progress · `[ ]` not started · `[-]` blocked / skipped (reason)

## Order of work (demo-first, then the roadmap)

Tier 1 (baseline, no code): `[x]` pytest 2058 pass / 32 skip · smoke 0 · vitest 98 · build OK (00:20)

### T0
- `[x]` T0.1 fixture — Kaggle competition returned 403 (rules not accepted; no browser to accept). Substituted **TCIA UPENN-GBM-00001** (public REST API, no login, CC BY 4.0): T1 axial 1 mm, T1CE axial 1 mm, FLAIR axial **3 mm** (61 slices), T2 **sagittal** SPACE 0.9 mm. `data/fixtures/dicom/UPENN-GBM-00001{,.zip}`; manifest `docs/data_manifests/fixture_dicom_upenn_gbm_00001_sha256.txt`. Author can still fetch `train/00000` after accepting the rules; T0.4 needs that one.
- `[~]` T0.3 run on fixture — run 1 (job c0e5f823…, 00:27): E1 ingest assigned all four roles correctly; pre-E2 input QC **REFUSED** on `geometry_consistency` (see Finding F1). Fix in flight.
- `[ ]` T0.2 `scripts/run_clinical_study.py` + test
- `[-]` T0.4 comparison vs BraTS2021_00000 — needs the Kaggle `train/00000` fixture (author: accept competition rules, rerun `kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification -f train/00000/...`)
- `[ ]` T0.5 job persistence

### T1
- `[ ]` T1.1 `/geometry` route · `[ ]` T1.2 api.ts · `[ ]` T1.3 reportStatus + hook · `[ ]` T1.4 hook geometry · `[ ]` T1.5 BrainTwinScene props · `[ ]` T1.6 viewer twin+report · `[ ]` T1.7 e2e section 12

### T2 · T3 · T4 · T5 · T6
- `[ ]` T2.1–2.5 · `[ ]` T3.1–3.6 · `[ ]` T4.1–4.4 · `[ ]` T5.1–5.6 · `[ ]` T6.1–6.4

## Findings (newest first)

- **F1 (2026-09-18 00:28) — pre-E2 input QC refuses every real study on `geometry_consistency`.**
  `run_input_qc` runs the same check set before and after E2. Before co-registration the four raw
  series legitimately sit on different grids (here: 192×256×192 axial vs 192×256×320 sagittal vs
  192×256×60), so the affine-agreement check fires REFUSE on the pre-E2 pass by construction — the
  exact "resample-then-QC vs relax" risk the plan names. **Decision:** not a threshold change. The
  pre-E2 pass now runs with `stage="pre_registration"`, under which `geometry_consistency` is
  reported at WARN (message says E2 will co-register) and everything else is unchanged; the
  post-E2 pass keeps REFUSE. Default `stage="post_registration"` so no existing caller moves.
- Observed on the fixture pre-E2, all as designed: anisotropy WARN (FLAIR 3.2), expected_shape
  WARN, skull_present WARN (0.67–0.77 nonzero). Spacing 3.0 mm passed the ≤3.0 bound — exactly on it.
