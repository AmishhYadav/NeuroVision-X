# Tool-completion plan — the digital twin

**Status: ACTIVE. Approved 2026-09-18. Read this before any tool work.** Starting cold? Jump to
"Demo cut" at the bottom if the date is on or before 2026-09-19; otherwise start at T0/T1.
Supersedes nothing in `master_plan.md` — it is Track 1 of that plan, expanded.


## Context

Author's vision, restated 2026-09-17: upload four MRI sequences → accurate tumour outline → an
interactive 3D digital twin of that patient's brain with the tumour in it → the twin carries the
medical meaning (where it is by named structure, what it touches, how big and what shape, how sure
the model is, what the model looked at, an honest biological profile) → analysis and export from
the same screen.

The GPU/model track is parked (author decision 2026-09-15). This is the CPU tool track that
`CLAUDE.md` says must exist before building. Decisions taken 2026-09-17:
- **Four sequences** (T1, T1CE, T2, FLAIR). No missing-sequence synthesis.
- **Phase F (IDH)** is included as the final, gated step; the tool is complete without it.
- **Audience: clinician / researcher.** Decision-support framing stays enforced in code and copy.

## What exploration found (2026-09-17; filesystem is ground truth)

**Live on a clinical upload today:** ingest → input QC → co-reg / SRI24 atlas-reg / HD-BET →
QC → research preprocessing → segmentation (+ logits) → gatekeeper → Grad-CAM (WT, TC) → DICOM-SEG
(native frame via inverse transform) → report JSON (disclaimer, not_claimed, burden, anatomy,
involvement, eloquence, provenance). Job states `queued|running|done|refused|failed`.

**Built but not on the clinical screen:**
- `app/frontend/src/components/BrainTwinScene.tsx` — Surface-Nets meshes (brain L/R, three tumour
  classes) from `workers/twinMesh.worker.ts`; mounted only at `App.tsx:387` (demo) and the landing
  hero. Prop `BrainTwinInput {caseId, shape, spacing, modalityVolumes, tumorMask, tumorSource}`.
- `components/ReportPanel.tsx` — renders the full report JSON; used only at `App.tsx:451`.
  `GET /api/clinical/jobs/{id}/report` exists (`api.py:886`); `api.ts` has no client for it.
- `pages/clinical/ClinicalStudyViewer.tsx` already fetches volume, mask, entropy, conformal band
  (WT/TC), Grad-CAM (WT/TC) via `hooks/useClinicalJobVolumes.ts` and paints them on 2D slices
  through `lib/render.ts::renderSlice` with `lib/colors.ts::entropyColor`.

**Gaps that shape the design:**
- No clinical route sends voxel **spacing** (`_binary_response`, `api.py:74-89`). The twin needs it
  for mL. `prep/<case_id>/meta.json` has `spacing, bbox, affine, cropped_shape`; the existing
  `volumes.py::CaseMeta` (l.40) already models exactly that and `/cases/{id}` returns it.
- After E2 the patient sits on the SRI24 240×240×155 grid; served volumes are that grid cropped by
  `meta.bbox`; `localize.atlas_for_case(atlas, meta, cropped=True)` (`localize.py:360`) gives the
  parcellation in the same frame. **Raw parcellation ids run to 424** (int16; per-plane
  sub-labels) — must be remapped to merged structure index via `AtlasLabels.lookup_array`
  (`atlas.py:347`) before it fits uint8.
- **E2 has never run on a real DICOM study** (`master_plan.md:256`). No `.dcm` fixture on disk;
  HD-BET weights not downloaded.
- Clinical jobs live only in `_CLINICAL_JOBS` (`clinical_jobs.py:234`); a restart forgets every
  study though artifacts under `job_root(settings)/<job_id>/` survive. No PUT/PATCH route exists.
- **Fixture:** `.venv/bin/kaggle competitions files rsna-miccai-brain-tumor-radiogenomic-classification`
  lists `train/<id>/{FLAIR,T1w,T1wCE,T2w}/*.dcm` — native scanner DICOM, not skull-stripped, same
  5-digit IDs as our BraTS 2021 cases. `BraTS2021_00000` is on disk preprocessed with ground truth,
  so the clinical path on `train/00000` can be compared with the research path on the same patient.
  Those patients are BraTS *training* cases: T0 proves plumbing, never performance.
- `anatomy/burden.py::burden_profile` (l.626) already gives volumes, fractions, sphericity, surface
  area, S/V, multifocality, laterality, voxel centroid — **and it is the `burden.csv` row producer**
  (`scripts/burden.py`, `population_stats.py`, `test_burden.py` pin its keys). Not to be extended.
- `reporting/report.py`: optional blocks have a precedent — `involvement=None` (`build_report`,
  l.440; `tests/test_report.py:690` pins key order when absent). `NOT_CLAIMED` hardcoded at l.119.
  **`tests/test_report.py:181` scans every rendered string outside `not_claimed` for `grade`,
  `stage`, `prognosis`, `deficit`, `impair`, `will experience`.**
- Frontend: vitest (node env, pure-function tests only); one pixel-asserting E2E
  `app/frontend/e2e/smoke.mjs`. **Its fingerprint calls `canvas.getContext('2d')` (l.131), which is
  `null` on the twin's WebGL canvas.** Twin assertions need `<Canvas gl={{preserveDrawingBuffer:
  true}}>` + `toDataURL()`, and headless Chrome needs `--use-angle=swiftshader
  --enable-unsafe-swiftshader`.
- Lessons that bind: `docs/lessons.md:26` (verify orientation by eye), `:29` (label layers only from
  `X-Uncertainty-Kind`), `:31` (scope absent-phrase assertions to the claiming element), and trap 3
  (a mirrored atlas scores *higher* on brain-mask Dice — only eyes catch L/R).

## Deliverable

1. **`docs/research/tool_completion_plan.md`** — this plan in `master_plan.md` §4 form
   (orientation, how to tell what is done, queue with `[x]/[ ]`, dependency arrows, verification).
   Markdown, so Opus writes it directly. Committed together with the six docs left uncommitted
   since 2026-09-15 (`CLAUDE.md`, `docs/reproducibility.md`, four `docs/research/*.md`), then
   pushed. `CLAUDE.md` "What is next", `master_plan.md` §4.3 Track 1 and memory
   `gpu-track-parked-tool-first` get a pointer.
2. **The build**, one module per turn through `py-implementer` → `test-runner` → `code-reviewer`,
   for every `.py` **and** `.ts/.tsx` file. Each step below leaves the tool usable.

## The queue

### T0 — Prove the clinical path on one real DICOM study  *(runs in parallel with T1)*

Goal: a raw study goes upload → `done` (or an explained `refused`) with every artifact on disk,
on this Mac under `.venv-clinical`, reproducibly.

| # | Unit | Notes |
|---|---|---|
| T0.1 | Fetch `train/00000/{FLAIR,T1w,T1wCE,T2w}` from the RSNA-MICCAI Kaggle competition into `data/fixtures/dicom/` (gitignored via `/data/`), zip it, SHA-256 manifest in `docs/data_manifests/` | If `kaggle competitions download` returns 403, accept the competition rules once on the Kaggle page. Never commit the data |
| T0.2 | `scripts/run_clinical_study.py` (new; Hydra off `configs/config.yaml`, `+clinical.study_dir=… +clinical.out_dir=…`): zips the folder, calls `create_clinical_job` + `run_clinical_job` synchronously, writes `summary.json` {state, per-stage wall time, gatekeeper decision, which artifacts landed}. Test `tests/test_run_clinical_study_script.py` with the same monkeypatches `test_app_clinical_jobs.py::_wire_full_pipeline_to_gatekeeper` uses | Makes T0 a command, not a story |
| T0.3 | Pre-warm HD-BET once from `.venv-clinical` (weights download on first `HDBetExtractor()`); record the resolved weights dir in `docs/reproducibility.md`. Then run T0.2 on the fixture. Fix what breaks, each fix its own module turn (`clinical_preprocess.py`, `input_qc.py` likely) | Expect: input-QC `anisotropy_refuse_ratio: 6.0` may fire on raw spacing *before* E2 resamples; E1 series naming may not match the rule table; DICOM-SEG geometry refusal. Each is a **finding** to record, and any relaxation is a design decision to flag, not a silent edit |
| T0.4 | Compare the clinical-path prediction on `train/00000` with the research-path prediction on `BraTS2021_00000` (both on the SRI24 grid): Dice per region + burden block side by side. **Note 45** in `docs/experiments.md` | The "preprocessing sensitivity" number — how far our own registration/skull-strip moves the answer from BraTS's. Expected well below 1.0 |
| T0.5 | **Job persistence** — `clinical_jobs.py` (mod): write `<job_dir>/job.json` on every `_update_clinical_job`; rehydrate `_CLINICAL_JOBS` from `job_root(settings)/*/job.json` at startup; `DELETE` removes the dir. Test: create → new store → still listed as `done` | Needed by T5 and by any demo that restarts the server |

Verify: `summary.json` verdict; `report/<id>.json`, `dicom_seg/<id>.dcm` (opens in `pydicom`),
`cache/neurovision/gradcam/{WT,TC}` present; slices right way up by eye.

### T1 — The twin on the clinical screen  *(biggest visible win; no new science)*

| # | Unit | Notes |
|---|---|---|
| T1.1 | `volumes.py` (one-line mod): `bbox` already in `CaseMeta.to_json` — confirm; `api.py` (mod): `GET /api/clinical/jobs/{id}/geometry` → `CaseMeta` from `prep/<case_id>/meta.json` via `to_json()`, byte-compatible with `/cases/{id}`'s `meta`. Test in `tests/test_app_clinical_api.py` using `_fabricate_done_clinical_job`: 200 / 404 / 409 | Header alternative rejected: `/geometry` also feeds T3 and T6 |
| T1.2 | `api.ts` (mod): `getClinicalJobGeometry`, `fetchClinicalReport` (runs `validateReport`, maps 404 → `not_found`). vitest in `api.test.ts` with mocked `fetch` | Reuse the `CaseDetail["meta"]` type |
| T1.3 | `lib/reportStatus.ts` (new, pure): `classifyReportError(err) → ReportPanelStatus`, extracted from `App.tsx:143`; `hooks/useClinicalReport.ts` (new) on top of it. vitest on the pure function | Demo path switches to the same helper — one source of truth |
| T1.4 | `hooks/useClinicalJobVolumes.ts` (mod): add `geometry` | |
| T1.5 | `BrainTwinScene.tsx` (mod): `<Canvas gl={{preserveDrawingBuffer: true}}>`; optional `badge` prop rendered top-left | `preserveDrawingBuffer` is what T1.7 and T6 need; no worker change. Note in spec: the worker uses `spacing` only for mL, geometry assumes the 1 mm SRI24 grid — true on this path, leave it |
| T1.6 | `ClinicalStudyViewer.tsx` (mod): "Slices / Twin" toggle; builds `BrainTwinInput` exactly as `App.tsx:267-294` (`tumorSource: "prediction"`, spacing from geometry); mounts `ReportPanel` (needs a `relative` wrapper — the panel is absolutely positioned); `ClinicalPage.tsx` passes `gatekeeper_decision.decision`; CAUTION → badge on twin and report header | Reuse, never fork, `BrainTwinScene` |
| T1.7 | `e2e/smoke.mjs` (mod): section 12 gated on `NVX_E2E_CLINICAL_JOB` (a done job id); Chrome launched with SwiftShader flags; asserts `toDataURL().length` is non-trivial and changes after a click; report badge check scoped to its element | Pixel assertions per the harness's own rule |

Verify by eye: twin orientation matches the slices (anterior, superior, patient-left on
image-right); tumour sits where the axial slice shows it; CAUTION badge visible on a CAUTION job.

### T2 — Uncertainty and explanation painted on the twin  *(after T1; frontend only)*

| # | Unit | Notes |
|---|---|---|
| T2.1 | `lib/vertexScalars.ts` (new, pure): `sampleScalarsAtVertices(volume, shape, voxelPositions, normals, insetVoxels=1) → Float32Array` — samples **one voxel inward along the normal**, because surface vertices sit exactly on the class boundary where entropy is maximal by construction and would read "all uncertain". vitest on a hand-built 2×2×2 field: constant → constant, ramp → ramp | |
| T2.2 | `lib/vertexColors.ts` (new, pure): `scalarsToColors(scalars, kind)` — `entropyColor` ramp for entropy/gradcam; the 3-value categorical palette `Legend.tsx` already uses for `conformal-band`. vitest | |
| T2.3 | `workers/twinMesh.worker.ts` (mod): request gains optional `scalarLayers: Record<kind, Uint8Array>`; result gains per-class `voxelPositions` and `layerScalars[kind][class]`. Message shape unchanged when absent. Cache scalars per (case, layer) — a layer switch must not re-mesh | Brain shell stays flat; only tumour meshes are painted |
| T2.4 | `BrainTwinScene.tsx` (mod): `activeLayer` prop; `geometry.setAttribute("color", …)`, `vertexColors: true`; `Legend` reused with the exact `X-Uncertainty-Kind` (lesson 29) | Three.js core, no dependency |
| T2.5 | `ClinicalStudyViewer.tsx` (mod): the existing 6-way heat selector drives the twin layer too — slice and twin always show the same layer | |

Verify by eye: pick the most uncertain bit of the ET rim on a slice; the twin is hot there and the
boundary is not uniformly hot; the conformal layer shows three colours.

### T3 — Anatomy on the twin  *(backend units in parallel with T2; frontend after T1)*

| # | Unit | Notes |
|---|---|---|
| T3.1 | `src/neurovision/anatomy/atlas_export.py` (new, pure): `structure_index_volume(atlas) → uint8 (D,H,W)` via `lookup_array` (+1 so 0 stays background; unmapped → 0); `structure_table(atlas, knowledge) → [{index, name, laterality, lobe, eloquence, matched_term}]` from `AtlasLabels` + `knowledge/{aal_lobes,eloquence_map}.yaml`. Tests on `test_atlas.py`'s synthetic atlas: max index < n_structures; round-trip name ↔ index | Not a report producer — no number moves |
| T3.2 | `api.py` (mod): `GET /api/clinical/jobs/{id}/atlas` → `_binary_response` of the bbox-cropped index volume (`atlas_for_case(cropped=True)`), header `X-Uncertainty-Kind: atlas-structure-index`; `GET /api/atlas/structures` JSON; demo twin `GET /api/cases/{id}/atlas`. Tests with fabricated job + monkeypatched `load_atlas`: shape equals the served volume's | |
| T3.3 | `api.ts` + `useClinicalJobVolumes.ts` (mod): `AtlasBuffer` + structure table; `lib/atlasSelection.ts` (new, pure): `selectStructures(report, table)` = indices whose name appears in `report.anatomy.structures[].structure` (≤ `top_n` = 10). vitest | Never mesh all structures |
| T3.4 | `workers/twinMesh.worker.ts` (mod): `atlasSelection: number[]` → one Surface-Nets mesh per index on `atlas === idx`, same transform → `structures: Record<index, MeshBuf>`. Cap 16 | |
| T3.5 | `BrainTwinScene.tsx` + `ClinicalStudyViewer.tsx` (mod): translucent shells; click → name, lobe, eloquence tag, `frac_of_structure` from the report row; `highlightedStructure` state lifted to the viewer; checkbox list to add any structure | Copy stays geometric: "67% of Precentral_L overlaps the mask" |
| T3.6 | `ReportPanel.tsx` (mod): `onHoverStructure` on `StructureTable` rows → highlight on twin | |

Verify by eye, hard: **left/right.** `Precentral_L` must render on the side the twin's left
hemisphere splits to, and match the report's `laterality`. Nothing automated catches a mirrored
atlas (trap 3).

### T4 — Shape descriptors  *(backend in parallel with anything; frontend after T1)*

| # | Unit | Notes |
|---|---|---|
| T4.1 | `src/neurovision/anatomy/shape_descriptors.py` (new): `shape_profile(classes, geom) → dict` — `rim_thickness_ET_median_mm` / `_max_mm` (EDT inward from the ET outer surface, `scipy.ndimage.distance_transform_edt(sampling=spacing)`), `elongation_{R}` / `flatness_{R}` (PCA eigenvalue ratios of voxel coords in mm), `extent_{R}_{i,j,k}_mm`. Tests: sphere → elongation ≈ 1; slab → flatness small; hand-built shell → rim thickness exact | **New module, `burden_profile` untouched** — it feeds the published `burden.csv` |
| T4.2 | `reporting/report.py` (mod): `build_report(..., geometry: Mapping | None = None)` → optional `"geometry"` block after `involvement`, with a caveat string; one appended `not_claimed` tuple ("tumour growth pattern or invasiveness", why). Tests: absent-by-default key order unchanged; forbidden-word scan passes; markdown renders | Block prose must avoid `grade/stage/prognosis/deficit/impair` — say "tumour extent", never "staging" |
| T4.3 | `clinical_jobs._generate_report` (mod) + `scripts/report.py` behind `cfg.analysis.report.geometry`: default **off** for the batch script (so `outputs/report_*` stays byte-identical), **on** for clinical jobs | Additivity by construction |
| T4.4 | `lib/report.ts` formatters for `_mm` / `elongation_` / `flatness_` keys; `ReportPanel.tsx` already iterates block keys (`Object.keys(block).sort()`, l.61) so the section appears with a title line only. vitest on the formatter | Near-zero UI work |

Verify: re-run `scripts/report.py` on three test cases with the flag on — values plausible in mm;
with the flag off — `git status` clean, JSON diff empty.

### T5 — Biological-profile panel, honest shell  *(backend in parallel; frontend after T1)*

| # | Unit | Notes |
|---|---|---|
| T5.1 | `knowledge/molecular_markers.yaml` v1 (Opus writes): IDH, 1p/19q, MGMT, ATRX, TP53, TERT, EGFR, CDKN2A/B — each: one-sentence meaning, CNS5 role, allowed entered values, citation (Louis et al., Neuro-Oncology 2021, WHO CNS5). A `cns5_lookup` table (entered IDH, entered 1p/19q, entered histology) → integrated diagnosis name. **No predictive claim anywhere.** Same source-owns-claim discipline as `eloquence_map.yaml` | Vocabulary: Mutant / Wildtype / Methylated / Unmethylated / Codeleted / Intact / Not tested / Not entered |
| T5.2 | `src/neurovision/reporting/molecular.py` (new, pure): `MARKERS`, `empty_molecular_block()` (each marker `{"confirmed_pathology": "not entered", "ai_estimate": None}`; IDH's `ai_estimate: {"status": "not available — model not trained"}`), `merge_pathology(block, entered)`, `cns5_lookup(entered) → {"name": None | str, "requires": [...]}` returning `None` until the required entries exist. Tests: lookup exhaustive; forbidden-word scan on rendered text ("CNS5 integrated classification", never "WHO grade") | |
| T5.3 | `report.py` (mod): `molecular: Mapping | None = None` optional block; one appended `not_claimed` ("a molecular subtype inferred from imaging", why). `REPORT_VERSION` stays 1 (optional-block precedent). Markdown heading "Confirmed pathology (entered by user)" | |
| T5.4 | `clinical_jobs._generate_report` (mod): passes `empty_molecular_block()` | |
| T5.5 | `api.py` (mod): `PUT /api/clinical/jobs/{id}/pathology` validates against `MARKERS` + vocabulary → `<job_dir>/pathology.json`; `GET /report` merges at read time via `merge_pathology` — the cached report file is never rewritten. Tests: PUT→GET round trip; invalid marker 400. Needs T0.5 | First mutating route on a job; keep it to this one |
| T5.6 | `api.ts` `putClinicalPathology`; `pages/clinical/MolecularPanel.tsx` (new): one select per marker; AI slot greyed with its literal reason; CNS5 line shown **only** when the lookup resolves, labelled "from entered pathology"; `ReportPanel.tsx` mounts it | Copy: "Confirmed pathology, entered by user. Not derived from imaging." |

Verify: enter IDH-wildtype + histology glioblastoma-pattern → line reads "Glioblastoma,
IDH-wildtype — from entered pathology"; reload → persists; AI slot reads "not available" verbatim.

### T6 — Export  *(markdown route any time; bundle after T1–T3)*

Decision: **no PDF dependency.** Markdown + PNG + DICOM-SEG is v1.

| # | Unit | Notes |
|---|---|---|
| T6.1 | `api.py` (mod): `GET /api/clinical/jobs/{id}/report/markdown` — `render_markdown` on the merged JSON, on the fly. Test | |
| T6.2 | `lib/twinSnapshot.ts` (new, pure): `captureCanvas(canvas) → Blob` via `toDataURL` (needs T1.5). vitest on a stub canvas | |
| T6.3 | `api.py` (mod): `POST /api/clinical/jobs/{id}/export` accepts snapshot PNGs, returns a stdlib `zipfile` bundle {report.json, report.md, dicom-seg.dcm, job.json, snapshots/*.png}. Test | |
| T6.4 | `ClinicalStudyViewer.tsx` "Export" button | |

### T7 — Phase F, IDH  *(GPU, gated — author's go/no-go; exactly `master_plan.md` §5 Phase F)*

Pre-registration first: `docs/research/preregistration_idh.md` (design principle 6). Then F1
UCSF-PDGM filtered to four sequences + seg (≈30–40 GB; 113 GiB free; `df -h` before/after —
**TCIA downloader is a dependency ask**) → F2 through the same pipeline, freeze
`configs/data/splits_ucsf.yaml` → F3 tumour-cropped 3D CNN on image + **predicted** mask → F4 age
control (AUC with/without age, age-only baseline; AUC via Mann-Whitney in scipy + numpy bootstrap —
**no scikit-learn** unless asked) → F5 conformal abstention reusing `uncertainty/conformal.py` →
**Gate F:** image model beats the age-only baseline with a bootstrap CI excluding zero, else the
null goes in `experiments.md` and the T5 slot stays "not available" → on GO only, **F6**:
`molecular.py` populates `ai_estimate.IDH = {probability, prediction_set, abstained,
auc_in_distribution, auc_age_controlled}` from a saved per-job inference; the `not_claimed` entry is
rewritten to "IDH estimated from imaging on in-distribution data only".

### T-docs — close the loop
`docs/research/tool_completion_plan.md` · `CLAUDE.md` "What is next" · `master_plan.md` §4.3
Track 1 gains the T-queue · memory `gpu-track-parked-tool-first` points at the plan ·
`docs/experiments.md` note 45 · `docs/reproducibility.md` HD-BET weights + fixture manifest.

## Dependency arrows

```
T0 ∥ T1                      (T1 develops against tests' _fabricate_done_clinical_job; T0 gives a real one)
T1 ──► T2 ──► T3(frontend) ──► T6(bundle)
T3.1–3.2, T4.1–4.3, T5.1–5.5   backend, parallel with anything
T0.5 ──► T5.5 ──► T5.6
T5 ──► T7 (author lifts the park) ──► F6
```
Only one heavy local job at a time (memory `one-local-job-at-a-time`): T0.3/T0.4 runs and T4's
re-scoring never overlap.

## What must NOT change
- Every published number. `burden_profile` untouched; new report blocks optional and absent by
  default in the batch script; `tests/test_report.py`, `tests/test_burden.py` plus a new additivity
  test pin existing keys. New routes extend `tests/test_app_clinical_api.py`.
- `neurovision` stays the only clinical segmenter (`NVX_CLINICAL_CHECKPOINT`).
- The "Do not write" list in `CLAUDE.md` and the forbidden-word scan. New copy is geometric or
  "entered by user" — never grade, stage, prognosis, deficit, or "reliable under shift".
- No new dependency without asking. Asks on the horizon: T7 only (TCIA downloader). T6 PDF: rejected.
- `report_version` stays 1.

## Risks
- **T0 is the unknown.** QC thresholds were set against BraTS-preprocessed data; a raw study may be
  refused before E2 resamples. Resample-then-QC vs relax is a design decision — flag it.
- Trap 3 becomes *visible* for the first time in T3. Verify L/R by eye, every case.
- T2: boundary vertices are maximally uncertain by construction — the one-voxel inset is required,
  not optional.
- The molecular panel is the most diagnosis-shaped surface in the tool: heaviest disclaimer; the
  CNS5 line never appears without entered pathology; the AI slot text is literal.
- Worker memory: four cropped volumes + scalar layers + atlas cloned into the worker — same order
  as the demo today; cap atlas meshes at 16.

## Verification (per step, on top of `master_plan.md` §10)

| Step | pytest | integration | vitest | E2E pixels | by eye |
|---|---|---|---|---|---|
| T0 | script test (faked pipeline); persistence test | real study → `summary.json`; `smoke_test.py` still 0 | — | — | orientation; artifacts present; note 45 |
| T1 | `/geometry` 200/404/409 | — | api mapping, `reportStatus` | twin canvas non-trivial, changes on click; badge text scoped | twin matches 2D mask; CAUTION badge |
| T2 | — | — | `vertexScalars`, `vertexColors` | canvas changes on layer switch | boundary not uniformly hot; band shows 3 colours |
| T3 | `atlas_export`; atlas routes | `scripts/localize.py` output unchanged | `atlasSelection` | click → structure name in DOM | shells align; **L/R correct** |
| T4 | hand-value tests; key order; forbidden scan | `scripts/report.py` flag off → JSON diff empty | formatters | — | mm values plausible |
| T5 | lookup exhaustive; PUT/GET; forbidden scan | put → restart → get | — | entered value survives reload | "not available" verbatim |
| T6 | markdown + zip routes | bundle unzips | `twinSnapshot` | — | PNG not black |
| T7 | CPU shape test | age-only baseline reported | — | slot fills on GO | Gate F in `experiments.md` |

Rough effort (CPU track): T0 ≈ 1 wk (unknowns) · T1 ≈ 4 d · T2 ≈ 3 d · T3 ≈ 1 wk · T4 ≈ 3 d ·
T5 ≈ 1 wk · T6 ≈ 2 d → ~5–6 weeks. T7 ≈ 3–4 weeks on the college GPU, after the author's go.

## Demo cut — presentation 2026-09-19 ~14:00 (added 2026-09-18)

The plan above is the roadmap; this is what is on screen tomorrow. Three tiers, strictly in order.
Nothing in T0–T6 is compute-bound — the limit is module round-trips, ~3–4 in a day.

**Tier 1 — MUST, zero new code, first two hours of today.** Everything here exists and is the story
told on 2026-09-17, on the demo path:
1. Bring up backend (`.venv-clinical`, `uvicorn app.backend.main:app`) and frontend (`npm run dev`)
   on this Mac; confirm `npm test`, `npm run build`, plain `pytest`, `scripts/smoke_test.py` clean.
2. Pick three BraTS test cases with contrasting reports (e.g. `BraTS2021_01520` — motor-cortex,
   near-zero ET; one multifocal; one enhancing-dominant). Open each in the demo browser: slices
   with mask, entropy overlay, the **twin** (orbit, split hemispheres, click sub-structure → mL),
   the **report panel** (burden, anatomy, involvement, eloquence, not_claimed, provenance).
3. NIfTI upload on the demo path (`/api/upload`, four BraTS files) → job → segmentation, as the
   "upload → outline" beat. Live today.
4. Slides straight from real output: the `BraTS2021_01520` report walkthrough (already written
   up in this session), the ten pipeline stages, the honesty list, and the T0–T7 roadmap as the
   "what comes next" slide. Present the gap plainly: "the twin is on the research viewer today;
   wiring it to the clinical upload path is step T1".

**Tier 2 — SHOULD, time-boxed to 3 hours today, stop at the box.** T0.1 + T0.3 by hand (no
script yet): download the Kaggle fixture, pre-warm HD-BET, upload the zip at `/clinical`, watch
the stages. Three possible outcomes, **all demoable**:
- `done` → show the clinical screen: stages, gatekeeper verdict, slices, DICOM-SEG download. Best case.
- `refused` with a named reason → show it as the thesis in action: "a refused job is the pipeline
  working". Keep the screenshot.
- crash (HD-BET / dcm2niix on arm64, geometry) → record the stage it died in, present the
  slide "first real-DICOM run: failed at stage X, that is T0's job", move on. Do not debug past
  the box.

**Tier 3 — COULD, only if Tier 2 reached `done` before ~18:00 today.** T1.1 (`/geometry` route)
+ a lean T1.6 (mount `BrainTwinScene` on `ClinicalStudyViewer` with `tumorSource: "prediction"`,
no report panel, no badge). Two module turns. If either is not green by 22:00, revert to Tier 1's
framing — a broken clinical twin on stage is worse than an honest slide.

**Not attempted before the talk:** T2–T7, job persistence (do not restart the backend during the
demo), the E2E section 12, any threshold change to make the fixture pass.

**Morning of 2026-09-19:** rehearse Tier 1 once end to end on the machine that presents; keep
the backend running; have screenshots of every screen as the fallback if the browser misbehaves.

## First three actions after approval
1. Write `docs/research/tool_completion_plan.md`; commit with the six pending docs; push.
2. T0.1 — author downloads the fixture (Kaggle credentials); T0.2 spec → `py-implementer`.
3. T1.1 spec → `py-implementer`, in parallel.
