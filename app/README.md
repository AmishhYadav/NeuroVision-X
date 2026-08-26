# NeuroVision-X demo

A viewer for the segmentation results this project produces. Pick a case, scrub
through the volume, and see where the model is right, where it is wrong, and
where it is unsure.

It reads **precomputed** artifacts — the output of `scripts/evaluate.py` — so a
session costs no GPU time and needs no checkpoint present. Nothing here is part
of the training path, and none of its dependencies are installed on Kaggle.

## Run it

Two processes. From the repo root:

```bash
# 1. API (leave running)
uv pip install -r app/backend/requirements.txt   # once
uvicorn app.backend.main:app --reload            # http://127.0.0.1:8000

# 2. Frontend
cd app/frontend
npm install                                      # once
npm run dev                                      # http://localhost:5173
```

Open <http://localhost:5173>. The dev server proxies `/api` to the backend, so
CORS is not in play during development.

To serve everything from one process instead, build the frontend once
(`npm run build`) and the API will mount `app/frontend/dist` at `/` — then
<http://127.0.0.1:8000> alone serves the whole app.

## Tests

```bash
pytest tests/test_app_api.py         # API, on synthetic cases (part of the main suite)
cd app/frontend && npm test          # slicing + compositing + error classification
npm run test:e2e                     # drives the real app in headless Chrome
```

`npm test` is pure and fast (~100 ms, no browser). It pins the things that stay
plausible when they are wrong: the plane-to-screen geometry, the exact overlay
blend arithmetic, the entropy alpha ramp, and the rule that a fixed colour scale
is never rescaled per case.

`npm run test:e2e` needs both servers running. It asserts on rendered pixels
rather than on elements existing — a viewport that draws pure black passes every
DOM assertion — and it checks the overlay against `/api/profile`, so "no colour
here" is only a pass when that slice genuinely contains no tumour. **Run it
before showing the demo.**

## Pointing it at different results

Every path comes from the environment, with repo-relative defaults. Nothing is
hardcoded to one machine.

| Variable | Default | What it selects |
|---|---|---|
| `NVX_PREP_DIR` | `data/preprocessed/brats` | Preprocessed cases (`image.npy`, `label.npy`, `meta.json`) |
| `NVX_EVAL_DIR` | `outputs/eval_test_baseline_unet3d` | Which evaluation run to display |
| `NVX_EXPERIMENT` | `baseline_unet3d` | Name shown in the header |
| `NVX_MAX_CASES` | `200` | Upper bound on the case list |
| `NVX_CHECKPOINT` | `outputs/checkpoints/baseline_unet3d/best.pt` | Reserved for live inference |
| `NVX_CLINICAL_CHECKPOINT` | `outputs/neurovision/checkpoints/best.pt` | The clinical pipeline's segmentation checkpoint (see below) — always `neurovision`, independent of `NVX_CHECKPOINT`/`NVX_EXPERIMENT` |
| `NVX_JOB_DIR` | `outputs/demo_jobs` | Where an uploaded case (either job kind below) is unpacked, preprocessed, and cached |
| `NVX_CORS_ORIGINS` | — | Extra allowed origins, comma-separated |

```bash
NVX_EVAL_DIR=outputs/eval_val_baseline_unet3d uvicorn app.backend.main:app
```

`GET /api/health` reports every resolved path and what it can actually see, so
a misconfigured setup explains itself rather than showing an empty list.

## Two kinds of upload job

**`/api/upload`** — the original demo path: four already-registered,
already-skull-stripped BraTS-style NIfTI files in (`t1`/`t1ce`/`t2`/`flair`
form fields), research preprocessing + segmentation out. Uses whatever
`NVX_EXPERIMENT`/`NVX_CHECKPOINT` the server is configured with.

**`/api/clinical/upload`** — the real pipeline (Milestone 4 Phase E): a raw
hospital DICOM study as a `.zip` (arbitrarily nested folder of `.dcm` files,
form field `dicom_zip`). Runs DICOM ingest → input QC → co-registration /
atlas registration / skull-stripping → input QC again → the same research
preprocessing + segmentation → two safety signals (the QC model's own
predicted Dice, the conformal band width) → the refusal gate, and always
segments with the **`neurovision`** checkpoint (`NVX_CLINICAL_CHECKPOINT`),
never whatever `NVX_EXPERIMENT` the rest of the demo is showing. **Requires
`.venv-clinical`** (`pydicom`, `dcm2niix`, `brainles-preprocessing`, `HD-BET`
— see `docs/reproducibility.md`); the plain `/api/upload` path above does not.

| Route | |
|---|---|
| `POST /api/clinical/upload` | 202, queues the job |
| `GET /api/clinical/jobs` / `GET /api/clinical/jobs/{id}` | List / poll one job. `state` is one of `queued`/`running`/`done`/`refused`/`failed` — **`refused` is a normal, successful outcome** (the input QC gate or the refusal gate correctly said no, with a reason in `error` and full detail in `ingest_result`/`input_qc_pre`/`input_qc_post`/`gatekeeper_decision`), never conflated with `failed` (something actually broke) |
| `DELETE /api/clinical/jobs/{id}` | Removes the job and everything it wrote to disk |
| `GET /api/clinical/jobs/{id}/volume/{modality}` / `.../mask/prediction` | Only once `state=="done"` (409 otherwise, including a refused job) |

E6 (DICOM-SEG export) is not wired into this pipeline yet — see
`docs/research/master_plan.md`'s Phase E row for why (the mask lives in atlas
space after co-registration, and the exporter refuses on a geometry mismatch
against the source series until a resample-back step exists).

## What the display means

**Orientation.** BraTS volumes carry the affine `diag(-1, -1, 1)`, so axis 0
runs right→left, axis 1 anterior→posterior, axis 2 inferior→superior. Slices are
drawn with superior up on coronal and sagittal, anterior up on axial and left on
sagittal, and the patient's left on the right of the image (radiological
convention). The mapping lives in one place, `src/lib/slicing.ts` — the naive
"first remaining axis is rows" version passes every shape test and displays the
head lying on its side.

**Geometry.** Saved predictions are in original BraTS geometry (240×240×155)
while images and labels are cropped to the nonzero bounding box. The backend
re-crops predictions with the case's own bbox before serving. Verified rather
than assumed: the whole-tumour Dice recomputed from what the API serves matches
the value `scripts/evaluate.py` reported for the same case to four decimals.

**Metrics** shown in the right panel come from `scripts/evaluate.py` at overlap
0.5. HD95 is in millimetres. A region whose ground truth is empty is marked,
because under the `ignore_empty=False` convention its Dice is scored
empty-vs-empty rather than by real overlap.

**Nearest-neighbour rendering** is deliberate everywhere. A smooth upsample of a
voxel grid implies sub-voxel precision the data does not have.

**Predictive entropy** is the entropy of a *single deterministic pass*:
aleatoric and epistemic combined, with no separable epistemic term. Separating
them needs MC-dropout, which these saved artifacts do not contain. The API
states this in an `X-Uncertainty-Kind` header and the UI labels the layer from
it, so the two cannot drift apart. It is never presented as epistemic
uncertainty.

**Region colours match the paper figures** (`neurovision.visualization.qc`):
necrotic core `#56B4E9`, oedema `#009E73`, enhancing tumour `#D55E00`. The
disagreement mode uses `#F0E442` for false negatives and `#CC79A7` for false
positives — yellow is excluded from every paper figure on purpose, so it can
never be mistaken for a tissue class.

## Layout

```
app/
├── backend/          FastAPI. config.py (paths) · volumes.py (arrays -> bytes) · api.py (routes)
└── frontend/         Vite + React + TypeScript. See app/frontend/README.md
```

Tests for the API live at `tests/test_app_api.py` and run inside the normal
`pytest` suite on synthetic cases — never against real BraTS data.
