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
| `NVX_CORS_ORIGINS` | — | Extra allowed origins, comma-separated |

```bash
NVX_EVAL_DIR=outputs/eval_val_baseline_unet3d uvicorn app.backend.main:app
```

`GET /api/health` reports every resolved path and what it can actually see, so
a misconfigured setup explains itself rather than showing an empty list.

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
