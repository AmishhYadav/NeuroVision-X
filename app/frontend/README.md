# NeuroVision-X frontend

A viewer for 3D brain tumour segmentation results (BraTS MRI): pick a case, scrub
through the volume across three planes, and see where the model is right, where
it is wrong, and where it is unsure.

## Run

```bash
npm install
npm run dev
```

Serves on `http://localhost:5173` and proxies `/api` to `http://127.0.0.1:8000`,
so start the backend separately:

```bash
uvicorn app.backend.main:app --reload
```

## Build

```bash
npm run build
```

Type-checks (`tsc -b`) and builds to `dist/`, which the FastAPI backend mounts
at `/` in production.
