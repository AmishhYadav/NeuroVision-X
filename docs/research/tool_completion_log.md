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
- `[~]` T0.3 run on fixture — run 1 (00:27): E1 assigned all four roles correctly; pre-E2 QC REFUSED on `geometry_consistency` → F1 fixed (8902432). Run 2 (00:32→02:42): stuck in HD-BET accurate+TTA on CPU, 16 GB resident, swapping, killed → F2 fixed (fast + no TTA, config-driven). **Run 3 launched 02:50** (`outputs/clinical_jobs/<id>/t0_summary.json` on finish; driver `scratchpad/t0_drive.py`, log `t0_run3.log`).
- `[ ]` T0.2 `scripts/run_clinical_study.py` + test
- `[-]` T0.4 comparison vs BraTS2021_00000 — needs the Kaggle `train/00000` fixture (author: accept competition rules, rerun `kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification -f train/00000/...`)
- `[x]` T0.5 job persistence (35ebf46) + clinical page lists previous studies / `?job=<id>` (d253b99)

### T1
- `[x]` T1.1 `/geometry` (03f84d0) · `[x]` T1.2+T1.3 (3c54428) · `[x]` T1.4+T1.5 (commit after) · `[x]` T1.6 viewer twin + report drawer · `[ ]` T1.7 e2e section 12 · **`[ ]` T1 verify-by-eye on a real done job (blocked on T0.3 run 3)**
- `[x]` **Twin bug fix (767f0ed)** — see F3. Must be eyeballed on the research viewer too (`/app`, any case): brain should now be crisp, not smeared; patient-left on screen-right when facing the front.

### T2 · T3 · T4 · T5 · T6
- `[ ]` T2.1–2.5 · `[ ]` T3.1–3.6 · `[ ]` T4.1–4.4 · `[ ]` T5.1–5.6 · `[ ]` T6.1–6.4

## Findings (newest first)

- **F3 (02:20) — the shipped twin was geometrically wrong twice over.** `twinMesh.worker.ts` passed dims `[D,H,W]` to an x-fastest `surfaceNets` over the w-fastest buffer, so every real case (D≠W) was surfaced with wrong strides (probe: radius-5 sphere → 28×38×16 smear, 2220 verts). And X=d, Y=w, Z=h is a left-handed anatomical frame — a mirrored brain (trap 3 in 3D). Also surfaceNets' native normals are inward. Fixed in `lib/twinGeometry.ts` (+tests) and the worker; scene is now (Left, Superior, Anterior), right-handed, outward normals. **Nobody has looked at the fixed twin yet.**
- **F2 (02:45) — HD-BET defaults are unusable on the 16 GB CPU floor.** brainles never forwards mode/TTA → accurate (5 folds) + TTA (8 mirrors) = 40 passes. 2 h 10 min, 16 GB resident, state "stuck", 24 CPU-min. Decision: `clinical.preprocess.hd_bet_mode: fast`, `hd_bet_tta: false` (HD-BET's documented CPU setting); GPU deployments opt back in. Not a QC change. Weights dir: brainles_hd_bet's own (not `~/hd-bet_params`) — record in reproducibility.md once run 3 confirms.

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
