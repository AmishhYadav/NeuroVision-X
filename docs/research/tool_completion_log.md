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
- `[x]` T0.3 run on fixture — run 1 (00:27): E1 assigned all four roles correctly; pre-E2 QC REFUSED on `geometry_consistency` → F1 fixed (8902432). Run 2 (00:32→02:42): stuck in HD-BET accurate+TTA on CPU, 16 GB resident, swapping, killed → F2 fixed (fast + no TTA, config-driven). Run 3 (02:50) refused (`predicted_dice` WT/TC) on the old twin geometry; **run 4 (03:01→03:07) `done` / PROCEED in 334 s**, job `a37fcaad`: HD-BET 250 s, inference 67 s, report written; DICOM-SEG refused on stale atlas-spacing assumption → F4 fixed (1dc5192), verified on the real job.
- `[x]` T0.2 `scripts/run_clinical_study.py` + test — real run under `.venv-clinical` surfaced two things the faked test could not (F6): `GlobalHydra is already initialized` (script's `@hydra.main` vs the backend's own `initialize_config_dir`) and `dcm2niix` invisible when the venv is not activated; both fixed in the script.
- `[-]` T0.4 comparison vs BraTS2021_00000 — needs the Kaggle `train/00000` fixture (author: accept competition rules, rerun `kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification -f train/00000/...`)
- `[x]` T0.5 job persistence (35ebf46) + clinical page lists previous studies / `?job=<id>` (d253b99)

### T1
- `[x]` T1.1 `/geometry` (03f84d0) · `[x]` T1.2+T1.3 (3c54428) · `[x]` T1.4+T1.5 (commit after) · `[x]` T1.6 viewer twin + report drawer · `[ ]` T1.7 e2e section 12 · **`[ ]` T1 verify-by-eye on a real done job (blocked on T0.3 run 3)**
- `[x]` **Twin bug fix (767f0ed)** — see F3. Must be eyeballed on the research viewer too (`/app`, any case): brain should now be crisp, not smeared; patient-left on screen-right when facing the front.

### T2 · T3 · T4 · T5 · T6
- `[x]` T2.1+2.2 (78bb119) · `[x]` T2.3 (a5a72f8) · `[x]` T2.4+2.5 (0db9af4)
- `[x]` T3.1 (00c0726) · `[x]` T3.2 (d268daa) · `[ ]` T3.3 · `[ ]` T3.4 · `[ ]` T3.5 · `[ ]` T3.6
- `[x]` T4.1 (2018e9e) · `[x]` T4.2 (6fa6d8d) · `[x]` T4.3 clinical (599993a) · `[~]` T4.3 batch flag · `[ ]` T4.4
- `[x]` T5.1 `knowledge/molecular_markers.yaml` · `[ ]` T5.2 · `[ ]` T5.3 · `[ ]` T5.4 · `[ ]` T5.5 · `[ ]` T5.6
- `[ ]` T6.1 · `[ ]` T6.2 · `[ ]` T6.3 · `[ ]` T6.4
- `[ ]` T1.7 e2e section 12

## Findings (newest first)

- **F6 (07:50) — a script's faked test cannot reach a real-run failure (trap 9, again).** `run_clinical_study.py` was green on a fully faked pipeline and failed twice on the fixture: `@hydra.main` leaves `GlobalHydra` initialised, so the backend's per-job `initialize_config_dir` raised; and `dcm2niix` lives in `.venv-clinical/bin`, which is not on `PATH` unless the venv is activated. Fixed: `release_hydra()` after the script reads its two keys; the interpreter's own bin dir is prepended to `PATH`. Rule stands: a script that drives a real pipeline is verified by running it on the real fixture, not by its unit tests.
- **F5 (05:xx, prior session) — conformal band mask with a restrictive fitted threshold (0f234c8).**
- **F4 (03:40) — DICOM-SEG export refused on the first real study, and would have mis-registered had it not.** `write_dicom_seg` compared the source spacing (1.0, 0.9766, 0.9766) against a hardcoded atlas-space (1,1,1); and the old path reshaped the NIfTI (x,y,z) array straight into DICOM (slice,row,col) frames. New `reporting/dicom_frames.py` samples the native mask onto each DICOM pixel's own world position (IPP/IOP/PixelSpacing, LPS→RAS, through the mask affine) and sorts slices along the normal; `write_dicom_seg` takes `mask_spacing_mm`. Verified on job `a37fcaad`: 183353/183353 voxels sampled; SEG written (156 frames, 3 segments); ET frames land on T1CE mean 691 vs brain 318, slice-/row-flipped controls 435/461.

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
