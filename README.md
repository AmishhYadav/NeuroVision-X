# NeuroVision-X

3D brain tumour segmentation on BraTS 2021 multi-modal MRI (T1, T1CE, T2, FLAIR),
plus a CPU-only interpretable pipeline that turns a segmentation mask into a
structured, atlas-grounded anatomical report.

Two things live here:

1. **A trained segmentation model** — dual encoder (3D CNN + Swin Transformer),
   adaptive gated cross-attention fusion, U-Net decoder, three heads
   (segmentation, confidence, boundary). Trained, evaluated, and compared against
   two U-Net controls with paired statistics.
2. **An interpretable pipeline** — SRI24 atlas localisation, eloquence lookup,
   tumour burden profile, and a per-case report artifact. All CPU, no GPU hours,
   built on top of already-saved predictions.

Everything reported below is measured on this repo's frozen test split and
recorded run-by-run in [`docs/experiments.md`](docs/experiments.md).

---

## Results, honestly

**The reliability claim this project was designed around did not hold. The
accuracy result did.** Both are stated here because a reader scanning the repo
should not have to reconstruct that from the commit log.

### Test split, 189 cases, `scripts/evaluate.py` at sliding-window overlap 0.5

| Run | Model | Params | Epochs / patch | Dice ET | Dice TC | Dice WT | HD95 ET | HD95 TC | HD95 WT |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_unet3d` | MONAI U-Net | 12.87M | 80 / 64³ | 0.8442 | 0.9058 | 0.9276 | 4.91 | 5.24 | 6.72 |
| `capacity_control_unet3d` | U-Net, widened | 34.83M | 80 / 64³ | 0.8497 | 0.9092 | 0.9295 | 4.84 | 5.67 | 7.09 |
| **`neurovision`** | dual-encoder + gated fusion | 34.91M | 80 / 64³ | **0.8709** | **0.9161** | 0.9321 | **4.20** | **4.98** | 7.09 |
| `baseline_unet3d` (superseded) | MONAI U-Net | 12.87M | 200 / 96³ | 0.8587 | 0.9157 | 0.9354 | 4.02 | 5.21 | 5.36 |

The last row trained on a different schedule *and* a different patch size, so it
is not a controlled comparison and is not used for any claim. The three 80-epoch
/ 64³ rows share every controlled variable except the architecture.

### The paired comparison

Every "A beats B" statement goes through `analysis.statistics.compare_models`:
paired bootstrap CI + Wilcoxon signed-rank + Holm–Bonferroni across the whole
table, `verdict` conservative (inconclusive if *either* the CI contains 0 or the
adjusted p exceeds alpha).

| Comparison | ET | TC | WT |
|---|---|---|---|
| `neurovision` vs `baseline_unet3d` | **+0.0267**, p_holm 7.2e-22 | **+0.0103**, p_holm 4.6e-11 | inconclusive |
| `neurovision` vs `capacity_control` | **+0.0211**, p_holm 7.3e-19 | **+0.0068**, p_holm 5.1e-07 | inconclusive |
| `capacity_control` vs `baseline_unet3d` | **+0.0055**, p_holm 1.6e-09 | inconclusive | inconclusive |

**The +0.0267 ET gain decomposes as ~20.6% capacity, ~79.4% architecture.** The
capacity control is a plain U-Net widened to 34.83M parameters — matched to
`neurovision` within 0.23% — with all fourteen other controlled variables verified
identical from the checkpoint's own stored config. It exists to answer the first
question a reviewer asks, and it was pre-registered with its three possible
outcomes written into `configs/experiment/capacity_control_unet3d.yaml` *before*
the run.

**Whole tumour is inconclusive in every comparison.** WT sits near 0.93 for all
three models and every bootstrap interval straddles zero. No WT claim may be made
from these runs.

### What did not hold

The project's original claim was "competitive Dice with substantially better
calibration and boundary accuracy". All three reliability routes were measured
against a matched baseline and none supports an advantage:

- **Calibration** — split-level ECE 0.0446 uncalibrated / 0.0135 temperature-scaled
  vs the baseline's 0.0395 / 0.0175, but per-case paired statistics return
  `ece_mean` **inconclusive** in both variants. Fitted temperatures are
  comparable (`[2.05, 1.99, 1.63]` vs `[1.92, 2.02, 1.93]`), so this architecture
  is not intrinsically less overconfident.
- **Boundary accuracy** — boundary-stratified error within noise.
- **MC-dropout risk-coverage** — N=10, normalised gain over random 37.6% vs 40.6%
  (CI [−0.176, +0.143]); Spearman(uncertainty, Dice) −0.392 vs −0.463. Both within
  noise. The baseline's own MC-dropout uncertainty is genuinely informative, which
  is exactly why "our model can flag its own failures" is not a claim.

Full numbers and the statistics: `docs/experiments.md` notes 11–17 and 19–21.

---

## Architecture

```
T1, T1CE, T2, FLAIR  (4 channels, 64³ patches)
        │
        ├── CNN encoder      strides 1, 2, 4, 8, 16   (18.85M)
        └── Swin encoder     strides    2, 4, 8, 16   ( 2.04M)
                │
        AdaptiveGatedFusion at strides 2/4/8/16       ( 1.05M)
          • BranchAmbiguity: per-branch region probes on DETACHED features
          • gate sees [cnn, swin, |p_cnn − p_swin|, H(p_cnn), H(p_swin)]
          • windowed cross-attention, CNN branch never attenuated:
            out = cnn + layer_scale * gate * attn
                │
        U-Net decoder with skips                      (12.93M)
                │
        ├── segmentation head(s), deep supervision
        ├── confidence head
        └── boundary head
```

Total 34,911,341 parameters. Output is **3 overlapping sigmoid channels**
(ET ⊂ TC ⊂ WT), never 4 classes and never an argmax.

**The contribution is the ambiguity conditioning, not the gate.** Gated fusion
for dual-branch medical segmentation is heavily published; a gate conditioned on
explicit inter-branch *disagreement* is not. `configs/experiment/ablation_content_only_gate.yaml`
is the one-key ablation that isolates it (parameter-matched to 0.018%). It is
specified, tested, and **not yet trained** — see *Status*. Rationale:
[`docs/research/contribution.md`](docs/research/contribution.md).

---

## The interpretable pipeline

Built after the reliability claim failed, and deliberately all-CPU: it consumes
saved predictions, so it costs zero GPU hours.

| Phase | State | What it does |
|---|---|---|
| 0 — atlas + alignment gate | **PASSED** 2026-08-16 | SRI24 loaded into the BraTS index frame, 122 merged structures; brain-mask Dice + laterality gate |
| 1 — localisation | **built, run on real data** | `frac_of_tumour` / `frac_of_structure` per structure, lobe, eloquence, distance to nearest eloquent structure |
| 2 — eloquence reference (Tier C) | **built** | 23 of 122 structures marked eloquent against the published Sawaya list, in `knowledge/eloquence_map.yaml` |
| 3a — burden profile | **built, run on real data** | volumes, foci, multifocality, sphericity, surface-to-volume, midline crossing, centroids — 57 columns per case |
| 3b — tissue involvement | not started | ependymal / cortical / deep-WM involvement, epicentre naming |
| 4 — report | **library built**, driver + demo panel pending | `reporting/report.py` assembles versioned JSON + Markdown with the non-diagnostic disclaimer, provenance and caveats as required fields |
| 5 — validation + paper | not started | population statistics; agreement between GT-derived and prediction-derived reports |

Two design decisions worth knowing before reading the code:

- **The atlas is not drop-in despite the grid matching.** SRI24 is 240×240×155 at
  1 mm as expected, but anterior–posterior mirrored relative to BraTS voxel
  indexing, and `pbmap_*` is additionally left–right mirrored relative to every
  other file in the same distribution. Both are exact index reversals solved
  *per file from that file's own affine*.
- **Brain-mask Dice cannot detect a left–right flip** — it scores *higher* on the
  mirrored atlas (0.9416 vs 0.9394). Laterality is proved content-wise from 56
  `_L`/`_R` structure pairs instead. See
  [`docs/research/phase0_atlas_findings.md`](docs/research/phase0_atlas_findings.md),
  which supersedes the plan document wherever they disagree.

**Midline shift is declined, not deferred.** The atlas says where a healthy
midline should be, not where this patient's is, and BraTS ships no ground truth
to validate against.

---

## Setup

Python 3.11.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

On Kaggle, install `requirements.txt` **without** `torch` / `torchvision` — the
notebook image ships a CUDA-matched build and pip would replace it with a wheel
that loses the GPU. See the comment at the top of `requirements.txt`.

The demo app's dependencies are separate (`app/backend/requirements.txt`) and are
deliberately not merged into the root file that Kaggle installs every session.

---

## Running the pipeline

Every stage is Hydra-driven; no path is ever hardcoded.

```bash
# 0. Preprocess BraTS (CPU, ~5 min for 1251 cases at num_workers=8)
python scripts/preprocess.py

# 1. Freeze the splits (already checked in as configs/data/splits.yaml)
python scripts/make_splits.py

# 2. Gate before spending GPU hours: real pipeline, synthetic data, ~4 s
python scripts/smoke_test.py

# 3. Train. The SAME command starts and resumes — it auto-resumes from last.pt
python scripts/train.py +experiment=neurovision

# 4. Evaluate on val AND test (calibrate.py refuses to fit and report on one split)
python scripts/evaluate.py inference.evaluation.save_logits=true
```

Post-hoc analysis, all CPU, all consuming artifacts step 4 already wrote:

```bash
python scripts/calibrate.py      # temperature (fit on val, applied to test), ECE,
                                 # reliability, risk-coverage, referral table
python scripts/extract_gates.py  # fusion gate maps for the mechanism figure
python scripts/explain.py        # Integrated Gradients, Grad-CAM, faithfulness
python scripts/fetch_atlas.py    # SRI24 from NITRC, checksum-verified
python scripts/validate_atlas.py # the Phase 0 alignment gate
python scripts/localize.py       # per-structure involvement (Phase 1)
python scripts/burden.py         # burden profile (Phase 3a)
```

`notebooks/09_paper_figures.ipynb` regenerates every paper figure and table in one
run and prints an audit of what it wrote and what it skipped and why.

`scripts/reproduce.sh` with no arguments reports which stages have already
produced output on this machine.

---

## Development

```bash
pytest                  # 1224 tests, CPU only, ~20 s
python scripts/smoke_test.py
ruff check .            # library code is clean; notebooks carry pre-existing E501
black .
```

Run plain `pytest` — `pyproject.toml` already sets `addopts = "-q"`, so an extra
`-q` stacks to `-qq` and silently drops the "N passed" line.

**The Mac is a correctness harness, not a compute device.** Tests run on CPU
(`device="cpu"`, never MPS — MPS 3D conv support is incomplete). Gradient descent
happens on Kaggle T4s. What a green CPU suite does *not* prove: three CUDA-only
faults have shipped past it. Anything touching `.numpy()`, a device transfer, or a
MONAI metric needs reading with "what if this tensor is on CUDA?" in mind.

Frontend tests:

```bash
cd app/frontend && npm test        # 39 vitest, ~100 ms
cd app/frontend && npm run test:e2e # 32 checks against the real backend
```

---

## The demo viewer

`app/` is a FastAPI + Vite/React/TS viewer, entirely outside the training path. It
reads only precomputed `scripts/evaluate.py` output, so it costs no GPU time and
needs no checkpoint present.

```bash
uv pip install -r app/backend/requirements.txt
uvicorn app.backend.main:app --reload    # http://127.0.0.1:8000
cd app/frontend && npm run dev           # http://localhost:5173
```

Details, including the display convention (superior up, radiological left-on-right)
and how the geometry re-crop was verified: [`app/README.md`](app/README.md).

---

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | Hydra configs — `data`, `model`, `training`, `inference`, `calibration`, `explainability`, `anatomy`, `analysis`, `experiment` |
| `src/neurovision/` | Library: `data`, `models` (encoders / fusion / decoder / heads), `losses`, `metrics`, `training`, `inference`, `uncertainty`, `explainability`, `anatomy`, `reporting`, `analysis`, `visualization`, `utils` |
| `scripts/` | CLI entry points, one per pipeline stage |
| `knowledge/` | Versioned knowledge base (eloquence map, lobe map) — **not** `data/`, which is gitignored |
| `notebooks/` | Analysis, paper figures, and the thin Kaggle drivers |
| `app/` | Demo viewer (backend + frontend), separate dependency set |
| `tests/` | pytest suite |
| `docs/` | Experiment log, reproducibility record, Kaggle workflow, research design docs |
| `outputs/` | Run artifacts (gitignored) |

---

## Documentation map

Read in this order depending on what you came for.

| Question | File |
|---|---|
| What was run, what came out, what failed | [`docs/experiments.md`](docs/experiments.md) — the authoritative results record |
| Can I re-derive a number? Seeds, versions, hardware, runtimes | [`docs/reproducibility.md`](docs/reproducibility.md) |
| How do I train on Kaggle and survive the 12-hour cap | [`docs/kaggle_workflow.md`](docs/kaggle_workflow.md) |
| What is the actual research contribution | [`docs/research/contribution.md`](docs/research/contribution.md) |
| Where is the interpretable pipeline going | [`docs/research/interpretable_pipeline_plan.md`](docs/research/interpretable_pipeline_plan.md) |
| What the atlas actually does (supersedes the plan) | [`docs/research/phase0_atlas_findings.md`](docs/research/phase0_atlas_findings.md) |
| Every non-obvious design decision and the bug that motivated it | [`CLAUDE.md`](CLAUDE.md) |

---

## Data, hardware, budget

| | |
|---|---|
| Dataset | BraTS 2021 Task 1 training set, 1251 cases, from Kaggle `dschettler8845/brats-2021-task1` |
| Preprocessing | nonzero z-score, nonzero-bbox crop, label remap, float16 image + uint8 label `.npy` (~38 MB/case, 34.22 GB total) |
| Splits | 875 / 187 / 189, seed 42, **frozen** in `configs/data/splits.yaml` — adding cases later raises rather than reshuffling |
| Training hardware | Kaggle T4 (16 GB nominal, **14.56 GiB usable**), 12-hour session cap, ~30 h/week free tier |
| Atlas | SRI24 (CC-BY-SA) from NITRC, fetched by `scripts/fetch_atlas.py`, never committed |
| GPU hours spent | ~62 against a 60 h plan — the budget is exhausted |

### Attribution

- **SRI24 atlas** — Rohlfing T, Zahr NM, Sullivan EV, Pfefferbaum A. *The SRI24
  multichannel atlas of normal adult human brain structure.* Human Brain Mapping
  31(5):798–819, 2010. Distributed under CC-BY-SA; downloaded at run time, never
  vendored into this repo. The atlas name and version are recorded inside every
  generated report, so a report can be traced to the parcellation that produced it.
- **AAL parcellation labels** (the `_L` / `_R` structure names SRI24 ships) —
  Tzourio-Mazoyer N et al., *NeuroImage* 15(1):273–289, 2002.
- **BraTS 2021** — see the challenge's own citation requirements; the data is not
  redistributed here.

Every training script supports full resume (model, optimizer, scheduler, AMP
scaler, epoch, global step, RNG states, W&B run id) because sessions get killed at
12 hours. Checkpoints are written atomically and stay `weights_only=True`-loadable.

---

## Status

**Milestone 1 — closed.** Baselines and the proposed model trained, evaluated on
val and test, calibrated, MC-dropout risk-coverage measured, capacity control run.
The accuracy result is established and the reliability claim is refuted.

**Milestone 2 — the interpretable pipeline, in progress.** Phases 0, 1, 2 (Tier C)
and 3a are built and have run on real data; Phase 4's report library is built.

Open work, in priority order:

1. `scripts/report.py` (Hydra driver), the `/api/report/<case_id>` route and the
   frontend report panel — Phase 4's remaining thirds.
2. Phase 3b — tissue involvement and epicentre naming, now unblocked by the atlas.
3. Phase 5 — population statistics, and the agreement experiment between reports
   generated from ground truth and from predictions. This is the experiment that
   connects the pipeline to the +0.0267 ET result, and it must control patch size,
   because report agreement is **not** monotonic in Dice (the superseded 96³ U-Net
   has lower ET Dice than `neurovision` and still produces the better report on
   volume and multifocality agreement).
4. `ablation_content_only_gate` (~23 GPU-h) — the load-bearing mechanism
   experiment, unstarted and now gated on finding more GPU hours.

Known limitations, to be stated in the paper rather than discovered in review:

- **Single seed.** No seed-to-seed standard deviation, so no claim may rest on a
  margin smaller than noise that was never measured.
- **No transformer baseline** on our splits. `baseline_swinunetr` was cut for
  compute (~25 h, 42% of the budget). Published SwinUNETR BraTS-2021 numbers are
  not a substitute — those are on the official validation set, ours is a random
  split of the training set.
- **The mechanism ablation is unrun**, so the ambiguity conditioning is argued for
  by design and pre-registration, not yet by measurement.
- `ignore_empty=False` (BraTS convention) throughout; `et_min_volume` is off.

Nothing here is a medical device and nothing it produces is diagnostic. The report
artifact carries that disclaimer inside itself, not only in this file.
