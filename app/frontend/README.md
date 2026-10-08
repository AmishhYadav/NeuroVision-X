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

## Test

```bash
npm test         # unit: geometry, compositing, error classification (~100ms)
npm run test:e2e # end-to-end: drives the real app in headless Chrome
```

`npm test` runs in plain Node with no browser environment — `render.ts`'s only
DOM dependency is the `ImageData` constructor, which `src/test/setup.ts`
polyfills. The tests exist because these two modules fail *plausibly*: a
transposed axis still draws a brain, and a wrong blend still draws a colour.

`npm run test:e2e` needs the backend and the dev server both running. It has no
npm dependencies (Node 22's global `WebSocket` speaks CDP to Chrome directly),
reads pixels back out of the canvases, and cross-checks each viewport's overlay
against `/api/cases/{id}/profile` so an empty panel only passes when that slice
really has no tumour. Set `CHROME_PATH` if Chrome is somewhere unusual.

## Where the tricky parts live

| File | Why it needs care |
|---|---|
| `src/lib/slicing.ts` | The only place volume index arithmetic exists. Encodes the display convention: superior up on coronal/sagittal, anterior up on axial, radiological left-right. |
| `src/lib/render.ts` | Layer composition. Overlay colours are the paper's; the entropy colour scale is fixed to `[0,1]` and must never be normalised per case. |
| `src/api.ts` | `responseError` classifies 502/503/504 as *unreachable*, because a dead backend behind the dev proxy arrives as an HTTP error, not a network failure. |
