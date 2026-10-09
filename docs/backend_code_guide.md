# NeuroVision-X — Backend Code Guide

A study guide to every non-frontend code file in the repository: `src/neurovision/`,
`scripts/`, `app/backend/`, `configs/`, `knowledge/`, `tests/`, and the shell scripts.
The React frontend (`app/frontend/`) is deliberately out of scope.

Written 2026-10-09 from the code as committed at `dfd2000`. Every number quoted here comes
from `CLAUDE.md`, a module's own docstring, or a result file on disk — if a number is not
here, check `docs/research_docs/claims_and_evidence.md` before saying it aloud.

**How to read this.** Part 0 is the ten-minute version. Part 1 explains the ideas every file
assumes you know. Parts 2–8 go file by file. Part 9 is likely questions with answers.

---

## Part 0 — The ten-minute version

### What the system does, in order

```
 BraTS NIfTI (4 MRI scans + label per patient)
   │  scripts/preprocess.py  → data/preprocessed/brats/<case>/{image.npy,label.npy,meta.json}
   ▼
 Frozen split (configs/data/splits.yaml): 70% train / 15% val / 15% test (189 test cases)
   │  scripts/train.py  (GPU)  → outputs/<experiment>/checkpoints/{last,best}.pt
   ▼
 NeuroVisionX model: CNN encoder + Swin encoder → gated cross-attention fusion
                     → U-Net decoder → segmentation / confidence / boundary heads
   │  scripts/evaluate.py (Mac CPU) → per_case_metrics.csv, summary.csv, logits/, predictions/
   ▼
 Safety layer, all computed from saved logits (no re-running the model):
   uncertainty (entropy, MC dropout)  ·  calibration (ECE)  ·  conformal risk control
   QC model that predicts a mask's own Dice  ·  OOD score  ·  label-free input QC
   ▼
 Gatekeeper: worst of the enabled signals → PROCEED / PROCEED_WITH_CAUTION / REFUSE
   ▼
 Clinical pipeline (app/backend/clinical_jobs.py): DICOM zip → ingest → input QC →
   registration + skull-strip → input QC → research preprocessing → segment →
   gatekeeper → Grad-CAM → DICOM-SEG export → structured anatomical report
   ▼
 FastAPI server (app/backend/api.py) serves volumes, masks, overlays and reports
```

### The thesis (from `CLAUDE.md`)

> A tumour segmentation model wrapped in a distribution-free error bound and a refusal gate:
> the bound holds in distribution, fails under distribution shift in proportion to the
> shift, and the gate cannot see cohort-level shift. We measure where it breaks, and what a
> new site needs in order to restore it.

It is **not** a "state of the art architecture" claim and **not** a "safe to deploy" claim.
Both were tested and refuted; saying so is part of the project's credibility.

### Numbers worth knowing by heart

| What | Value | Where it comes from |
|---|---|---|
| Test-set Dice, `neurovision` (n=189) | ET 0.871 · TC 0.916 · WT 0.932 | `outputs/neurovision/eval_test/summary.csv` |
| Test-set Dice, matched `baseline_unet3d` | ET 0.844 · TC 0.906 · WT 0.928 | `outputs/eval_test_baseline_unet3d/summary.csv` |
| ET Dice gain over matched U-Net | **+0.0267**, p_holm 1.4e-21 | `CLAUDE.md`, note 12 |
| Split of that gain | ~79% architecture / ~21% extra capacity | `CLAUDE.md` |
| Seed noise floor (same model, seed 43 vs 42) | +0.0021 | `CLAUDE.md`, note 49 (D1) |
| Usable mask returned (test / SSA / PED) | 85% / 78% / 24% | note 47 |
| Silent failure (accepted but unusable) | 4.2% / 18.3% / 49.5% | note 47 |
| Confidence-head error-localisation AUROC | ET 0.855 · TC 0.871 · WT 0.477 | `scripts/confidence_diag.py` docstring |
| OOD score flags | 60% of SSA vs 9.5% of test | note 56 |
| Real-DICOM front end | 40% thick-slice refusals, WT −0.23 Dice | note 55 |
| Model size | 34.9M parameters | `scripts/extract_ambiguity_serial.py` |
| CPU evaluation speed on the M4 | ~1.4 min per case | `CLAUDE.md` |
| Test suite | 2,230 passing, 35 skipped, ~60 s | `CLAUDE.md` (2026-09-24) |

SSA = BraTS-Africa (60 cases, a different scanner population). PED = BraTS paediatric
(99 cases, children). Both are **external cohorts**: never trained on.

---

## Part 1 — Ideas every file assumes you know

### 1.1 The data: four scans, three nested regions

Each patient has four co-registered MRI scans ("modalities"), always stacked in this
channel order: **T1, T1CE** (T1 after contrast agent), **T2, FLAIR**. So the model input
is a `(4, D, H, W)` volume.

The BraTS label marks three tissue classes. BraTS 2021 numbers them 1 (necrotic core),
2 (oedema), 4 (enhancing tumour). `preprocessing.remap_labels` renames 4 → 3 so the labels
are contiguous `{0, 1, 2, 3}`.

But BraTS is **scored on three nested regions**, not those classes:

| Region | Built from | Meaning |
|---|---|---|
| **ET** enhancing tumour | label 3 | the part that lights up with contrast |
| **TC** tumour core | labels 1 + 3 | ET plus the necrotic centre |
| **WT** whole tumour | labels 1 + 2 + 3 | everything, including oedema |

ET ⊂ TC ⊂ WT. A voxel in the enhancing tumour is in all three at once. This one fact
drives a lot of code:

- The model has **3 output channels, one per region, each with its own sigmoid**, not a
  4-way softmax. Softmax would force the channels to compete for one probability budget,
  but the regions overlap by definition. Many docstrings repeat "3, never 4" because 4 is
  a tempting wrong number twice over (4 classes, 4 input scans).
- There is **no `argmax` anywhere** in post-processing. Each channel is thresholded at 0.5
  on its own, then nesting is enforced.
- `CLAUDE.md` trap 7 forbids MONAI's `ConvertToMultiChannelBasedOnBratsClassesd` (it looks
  for raw label 4, which no longer exists after the remap) and `DiceCELoss` (it applies
  softmax cross-entropy to channels that overlap). Both fail silently.

### 1.2 Logits, probabilities, masks

- **Logit**: the raw number the network outputs per voxel per channel. Any real number.
- **Probability**: `sigmoid(logit)`, in (0, 1).
- **Mask**: probability ≥ threshold (default 0.5). Binary.

The project saves **fp16 logits** for every evaluated case (`<eval_dir>/logits/*.npy`).
Almost every analysis (thresholds, post-processing, calibration, conformal, QC) re-reads
those instead of re-running the model. That is why the analysis phases needed "zero
inference".

### 1.3 Patches and sliding windows

A whole brain (240×240×155 voxels in raw BraTS, smaller after cropping to the brain) does
not fit through a 3D network on a 16 GB GPU during training. So:

- **Training** uses random **patches** (64³ for every published run; the code default
  is 96³). `RandCropByPosNegLabeld` takes 4 patches per volume, half centred on tumour,
  half random (`pos_neg_ratio: [1, 1]`).
- **Inference** uses a **sliding window**: tile the whole volume with overlapping
  patches (overlap 0.5), run each, and blend the overlaps with Gaussian weights (centre of a
  patch trusted more than its edge). MONAI's `SlidingWindowInferer` does this.

### 1.4 Metrics

- **Dice** = 2·|A∩B| / (|A|+|B|). 1 is perfect overlap. Computed per region.
- **IoU** (Jaccard) = |A∩B| / |A∪B|.
- **HD95**: the 95th percentile of the distance between the two surfaces, in mm. Measures
  boundary error; lower is better.
- **Empty-region convention** (`ignore_empty=False`): if the reference region is empty and
  the prediction is too, Dice = 1; if the prediction has even one false-positive voxel,
  Dice = 0. This matters: 5 test cases have no ET, and two of them carry ~40% of the
  headline ET gain (`scripts/ref_et_comparison.py` exists because of this).
- **Lesion-wise metrics** (BraTS 2023+): score each connected tumour piece separately,
  so a missed small satellite lesion counts as a miss even though it barely moves voxel
  Dice.

### 1.5 Training ideas

- **Deep supervision**: extra segmentation heads on coarser decoder levels (strides 2, 4),
  each with its own loss, weighted 1 : 0.5 : 0.25. Helps gradients reach deep layers.
- **AMP** (automatic mixed precision): runs most maths in fp16 on the GPU for speed and
  memory. Off on CPU. It caused the project's worst bug (trap 2, see §3.4).
- **GroupNorm, never BatchNorm**: batches are 4 patches from *the same brain*, so batch
  statistics would be noise, and BatchNorm's train/eval difference hurts calibration.
- **Resume**: a GPU session can be killed at any moment, so every epoch saves everything
  needed to continue exactly (weights, optimiser, scheduler, AMP scaler, epoch, step, every
  RNG state, W&B run id).

### 1.6 Uncertainty and reliability ideas

- **Predictive entropy**: per voxel, how close the probability is to 0.5.
  `H(p) = −p·log p − (1−p)·log(1−p)`. Free from any single forward pass.
- **MC dropout**: run the model N times with dropout *on*, average. The spread across runs
  splits into aleatoric (noise in the data) and epistemic (the model is unsure).
  **Mutual information** is the epistemic part.
- **Calibration**: does "70% probability" really mean right 70% of the time? Measured by
  **ECE** (expected calibration error) over probability bins. Fixed (partly) by
  **temperature scaling**: divide logits by a fitted T before the sigmoid.
- **Conformal risk control**: pick a lower threshold τ so that, on average over new cases
  from the same population, the mask misses at most α (e.g. 10%) of the true tumour.
  Guaranteed by maths if new cases are *exchangeable* with the calibration cases. Breaks
  under distribution shift, which is the thesis.
- **Selective prediction / risk-coverage**: if the model refers its least-confident cases
  to a human, does quality on the cases it keeps go up?

### 1.7 Statistics ideas

- **Paired comparison**: the same 189 cases scored by two models, compared case by case.
- **Wilcoxon signed-rank test**: paired test that does not assume normal differences.
- **Bootstrap CI**: resample cases with replacement 10,000 times to get a confidence
  interval on the mean difference.
- **Holm–Bonferroni**: corrects p-values when several tests are run at once, so one lucky
  result out of many is not called significant.
- **TOST**: two one-sided tests, the correct way to claim two things are *equivalent*.
  "Not significantly different" is not equivalence.
- **Pre-registration**: the test, endpoints and decision rule are written down in
  `docs/research_docs/preregistrations/` *before* any number is computed. Many scripts say
  "read the pre-registration first" because they implement one verbatim.

### 1.8 Two engineering patterns used everywhere

**Hydra config.** Every script is `@hydra.main(config_path="../configs", config_name="config")`.
`configs/config.yaml` lists defaults (`data: brats`, `model: unet3d`, ...) and an experiment
file such as `configs/experiment/neurovision.yaml` overrides them. You select one with
`+experiment=neurovision`. Any value can be overridden on the command line
(`training.batch_size=2`). No path is ever hardcoded (constraint 2).

**Registries.** Models, losses and fusion blocks are selected by a string in config:

```python
@register_model("neurovision")
def build_neurovision(cfg): ...

model = build_model(cfg)   # looks up cfg.model.name in the registry dict
```

The decorator only runs if the module is imported, which is why each package's
`__init__.py` imports its builders.

**Pure core, thin driver.** Most features are split in two: a pure module in
`src/neurovision/` (no file I/O, no Hydra, fully unit-tested) and a driver in `scripts/`
(reads files, calls the module, writes results). When you are asked "where is X
computed", the answer is usually the `src/` module; "how is it run" is the script.

**Label-free by construction.** Anything that runs at deployment time (input QC, the
gatekeeper, the QC model's degradations) has *no parameter that could receive a ground-truth
label*. Tests introspect function signatures to enforce this. The reason is trap 1: an
earlier calibration mask defined using the label manufactured 41–57% of a reported ECE.

---

## Part 2 — Repository map

```
src/neurovision/          the library (93 files)
  utils/                  device, seeding, logging, json/yaml io
  data/                   BraTS reading, preprocessing, transforms, datasets, DICOM ingest,
                          clinical registration, QC training-pair generation
  models/                 registry, baselines, encoders/, fusion/, decoder/, heads/,
                          the full NeuroVisionX, the SegQC model
  losses/                 registry, Dice+BCE, multi-task
  training/               the Trainer loop and checkpointing
  metrics/                Dice/IoU/HD95, boundary-band errors, lesion-wise
  inference/              sliding window, post-processing, TTA, MC dropout,
                          input QC, gatekeeper
  uncertainty/            calibration, conformal risk control, risk-coverage
  explainability/         Grad-CAM, Integrated Gradients, attention rollout, faithfulness
  anatomy/                SRI24 atlas, alignment checks, burden, localisation,
                          involvement, shape descriptors, atlas export
  reporting/              structured report, molecular block, DICOM-SEG writer
  analysis/               every statistical analysis behind the paper
  visualization/          QC figures, paper figures, paper tables
scripts/                  52 Hydra/argparse entry points + cluster shell scripts
app/backend/              FastAPI server (8 files)
configs/                  Hydra config groups (40 files)
knowledge/                versioned anatomy/pathology knowledge base (4 YAML files)
tests/                    pytest suite (123 files)
notebooks/                Kaggle driver notebooks and figure notebooks
```

Empty `__init__.py` files (they only mark a folder as a Python package):
`src/neurovision/__init__.py`, `analysis/`, `data/`, `explainability/`, `inference/`,
`models/decoder/`, `models/encoders/`, `models/heads/`, `training/`, `utils/`,
`visualization/`, `tests/__init__.py`. The non-empty ones are covered with their package.

---

## Part 3 — `src/neurovision/`, file by file

Ordered the way data flows: utilities → data → model → loss → training → metrics →
inference → uncertainty → explainability → anatomy → reporting → analysis → figures.

### 3.1 `utils/` — the boring foundations

**`utils/device.py`** — the single source of truth for "CPU or GPU".
- `get_device(cfg)` reads `cfg.device`: `"auto"` → CUDA if available, else CPU. Never Apple
  MPS unless asked (MPS 3D convolution is unreliable).
- `amp_enabled(device)` → True only on CUDA. Mixed precision is a GPU feature.
- Why it exists: constraint 3 ("no CUDA-only assumptions"). No other file writes `.cuda()`.

**`utils/seed.py`** — `set_seed(seed, cudnn_benchmark)` seeds Python's `random`, NumPy,
PyTorch and MONAI in one call and returns a seeded `torch.Generator`. Code that needs
randomness takes that generator instead of touching global state, so runs repeat.

**`utils/io.py`** — `ensure_dir`, `read_json`/`write_json`, `read_yaml`/`write_yaml`,
`directory_size_bytes`, `format_size`. Every function takes a path argument and opens files
as UTF-8 explicitly, because macOS and Linux default encodings differ.

**`utils/logging.py`** — `setup_logging(level, log_file)` configures the root logger once,
safely re-callable. Library code uses `logging`, never `print`.

### 3.2 `data/` — from raw files to training batches

**`data/brats.py`** — finds cases on disk. Pure path logic, never opens a volume.
- `scan_brats_root(root)` walks a BraTS folder and returns one frozen `BratsCase` per
  patient, with paths to `t1`, `t1ce`, `t2`, `flair` and (optionally) `seg`.
- Handles both naming styles: BraTS 2020/2021 underscores (`_t1ce.nii.gz`) and 2023+
  hyphens (`-t1c.nii.gz`).
- `BratsCase.modality_paths()` always returns the four paths in the fixed model order.
- `write_case_index` writes a CSV index.

**`data/preprocessing.py`** — runs once, on the Mac CPU, turning NIfTI into `.npy`. No torch.
For each case (`preprocess_case`):
1. `load_case_arrays` reads the four scans and the label with nibabel.
2. `reorient_to_axcodes` puts every volume in the same axis orientation (`[L, P, S]`), so
   voxel index directions mean the same thing for every case.
3. `normalize_nonzero` z-scores each modality using **only brain voxels** (non-zero ones).
   Background is ~0 and would drag the mean down if included.
4. `compute_nonzero_bbox` + `crop_to_bbox` crop to the tight box around the brain.
5. `remap_labels` turns BraTS 2021's `{0,1,2,4}` into `{0,1,2,3}`.
6. Writes `image.npy` (4, D, H, W), `label.npy`, and `meta.json` (the bbox, original shape,
   spacing, affine, label counts). `meta.json` is what lets every later step **uncrop** a
   prediction back to the original 240×240×155 grid.
- `is_case_processed` makes the run resumable.

**`data/transforms.py`** — the MONAI transform pipelines.
- `ConvertToRegionsd` is the project's own transform: integer label → three binary channels
  ET = (label == 3), TC = (label ∈ {1, 3}), WT = (label > 0). Written by hand because MONAI's
  version expects raw label 4 (trap 7).
- `build_train_transforms(cfg)`: load → regions → `RandCropByPosNegLabeld` (4 patches per
  volume, half tumour-centred) → random flips on each axis, random 90° rotations, intensity
  scale ±10% and shift ±0.1, Gaussian noise (p = 0.15). Every number comes from
  `cfg.data.augment`.
- `build_val_transforms(cfg)`: load → regions. Deterministic, no cropping (the sliding
  window handles whole volumes).

**`data/dataset.py`** — glue between case ids and MONAI datasets.
- `build_data_dicts(case_ids, prep_dir)` → `[{"image": path, "label": path, "case_id": id}]`.
- `make_splits` / `load_splits`: a seeded 70/15/15 train/val/test split, written **once**
  to `configs/data/splits.yaml` and frozen. If more cases were preprocessed later, an
  already-reported test set can never be silently reshuffled.
- `build_dataset(..., dataset_type)` picks plain / `CacheDataset` / `PersistentDataset` by
  config string.

**`data/dicom_ingest.py`** — clinical task E1: turn a hospital DICOM folder into four named
NIfTI files, or say it cannot.
- A real study has 4–30 series (scouts, localisers, diffusion, perfusion...). The rule table
  decides which series is T1, T1CE, T2, FLAIR from DICOM header fields.
- Split in two: the **rule table** (`normalise_tokens`, `classify_series`, `assign_roles`)
  works on a plain `SeriesHeader` dataclass with no DICOM library, so it is tested in the main
  suite; the **I/O layer** (`read_series_headers`, `convert_series` via the `dcm2niix`
  binary, `ingest_study`) imports `pydicom` inside each function, so it only needs
  `.venv-clinical`.
- The substring trap: `"_t1" in name` is true for `_t1ce`. So text is split into whole tokens
  and matched by set membership. FLAIR is decided before T2 ("T2 FLAIR" is FLAIR) and T1CE
  before T1 ("T1 POST GD" is T1CE).
- `parse_patient_age` reads DICOM's `PatientAge` ("045Y"), used later by the adults-only
  intended-use check.
- `ingest_study` writes an audit-trail JSON explaining every decision.

**`data/clinical_preprocess.py`** — clinical task E2: make a raw scan look like BraTS.
BraTS images are already co-registered, aligned to the SRI24 atlas and skull-stripped; a
hospital scan is none of those. Feeding it straight to the model gives a confident, wrong
mask with nothing raising.
- Wraps `brainles-preprocessing`'s `AtlasCentricPreprocessor`: register the four scans to
  each other (centred on T1CE) → register to SRI24 → optional N4 bias correction → HD-BET
  skull-stripping.
- **Planning layer** (`build_plan`, `resolve_atlas_name`, `resolve_use_gpu`) is pure and
  tested in `.venv`; **execution layer** (`run_plan`, `preprocess_clinical_study`) imports
  the heavy libraries inside the function.
- Output then goes through the normal research `preprocess_case`, unchanged.

**`data/clinical_resample.py`** — `resample_mask_to_source` takes a predicted mask in atlas
space and maps it back into one scan's original (pre-registration) geometry, using E2's
saved inverse transform. Needed so the DICOM-SEG export lines up with the hospital's own
slices.

**`data/qc_pairs.py`** — makes training examples for the QC model (Phase C).
- The QC model must learn "how good is this mask?" across a wide range of quality, but the
  real model's masks are almost all ~0.87 Dice. So this module **damages** a predicted mask in
  realistic ways: `erode` (shrink), `dilate` (grow), `drop_component` (lose a lesion piece),
  `shift` (registration error), `speckle` (scattered false positives). Each damaged copy is
  scored against the label to get its true Dice: a `(degraded mask, true Dice)` pair.
- Binding rule: damage the **predicted** mask, never the ground truth. Damaged ground truth
  has smooth plausible edges; real predictions fail in this network's own ragged way.
  Training on the wrong kind of damage teaches the wrong failure.
- `degrade_mask` has **no label parameter at all** (a test checks the signature). The label
  is only used in `generate_pairs` to compute the target Dice, via the project's own
  `dice_score`.

### 3.3 `models/` — the networks

**`models/registry.py`** — `register_model(name)` decorator, `build_model(cfg)` looks up
`cfg.model.name`, `available_models()`. `models/__init__.py` imports `baseline`,
`neurovision` and `qc` so their decorators run.

**`models/baseline.py`** — the comparison models, both from MONAI:
- `unet3d` → `monai.networks.nets.UNet`. The matched baseline every claim is measured against.
- `swinunetr` → MONAI's SwinUNETR.
- Both output 3 channels (`cfg.data.num_classes`), take 4 inputs (`cfg.data.in_channels`).

**`models/encoders/cnn.py`** — `CNNEncoder`, half of the dual encoder.
- 5 levels with channels `[32, 64, 128, 256, 320]` at strides 1, 2, 4, 8, 16. Each level is
  `ResidualBlock`s (conv → GroupNorm → activation, plus a skip connection).
- `zero_init_residual`: the second GroupNorm's scale in each residual block starts at zero,
  so each block starts as an exact identity function — a standard trick that stabilises early training.
- Why 5 levels when Swin has 4: Swin cannot produce a stride-1 feature (see next file), so
  CNN level 0 (full resolution) goes to the decoder unfused.
- Why GroupNorm, never BatchNorm: the 4 patches in a step come from one brain, so batch
  statistics are noise; BatchNorm's train/eval switch also hurts calibration.

**`models/encoders/swin.py`** — `SwinEncoder`, the transformer half.
- Wraps MONAI's `SwinTransformer` (the encoder half of SwinUNETR) rather than all of
  SwinUNETR, because this project has its own fusion and decoder.
- Swin first merges every 2×2×2 block of voxels into one token, so its pyramid starts at
  stride 2. A stride-1 Swin feature at 96³ would be 884,736 tokens — impossible.
- `num_levels=4`: MONAI's 5th stage is 75% of the branch's parameters (6.02M of 8.06M) and
  at this patch size operates on a 3×3×3 map, so it buys almost nothing; it is dropped.
- `feature_size=48`, `depths=[2,2,2,2]`, `num_heads=[3,6,12,24]`, `window_size=7`.
- Gradient checkpointing on by default: recompute attention in the backward pass instead of
  storing it — ~20–30% slower, much less memory.

**`models/fusion/registry.py`** — the same registry idea for fusion blocks, except a builder
also receives the two branch widths and the level index, because one block is built per
pyramid level.

**`models/fusion/adaptive_fusion.py`** — **the project's novel component.** Read this one
properly.

Three variants share one interface, so an ablation swaps them by config string:
- `AdaptiveGatedFusion` — the proposed block.
- `ConcatFusion` — the standard baseline: concatenate, 1×1×1 conv back down, norm, activation.
- `AddFusion` — the floor: a 1×1×1 conv to match widths, then add.

What `AdaptiveGatedFusion` does at one pyramid level (`_fuse`):

```
swin_proj  = GroupNorm(Conv1x1(swin_feat))            # project Swin to the CNN's width
ambiguity  = BranchAmbiguity(cnn_feat, swin_proj)     # 9 channels, see below
gate       = GateGenerator(cnn_feat, swin_proj, ambiguity)   # values in (0,1), per voxel
attn_out   = WindowedCrossAttention(q=cnn_feat, k,v=swin_proj)
fused      = cnn_feat + layer_scale * gate * attn_out
```

- **Cross-attention**: every CNN voxel (query) looks at Swin context (keys/values). Attention
  over 48³ = 110,592 tokens would need a 1.2×10¹⁰-entry matrix, so it runs in local windows
  of 4³ tokens; at coarse levels with ≤ 512 tokens it uses full global attention. Decided at
  run time. Uses PyTorch's `scaled_dot_product_attention`, which never materialises the full
  score matrix.
- **`BranchAmbiguity`** — the actual research contribution. Each branch gets a tiny 1×1×1
  conv "probe" that predicts the 3 region logits from that branch's features. Then per voxel:
  `disagreement = |sigmoid(cnn_logits) − sigmoid(swin_logits)|` (3 channels) plus each
  probe's entropy (3 + 3 channels) = 9 channels. Prior gated-fusion work computes the gate
  from branch *content* only; this gate also sees *where the two branches disagree*.
- The probes read **detached** features (`.detach()`): gradients from the probes never
  reach the encoders. Otherwise the training objective would push both encoders to agree
  and destroy the disagreement signal the gate exists to read.
- Entropy is computed **from logits via softplus**, not from probabilities via `log`. The
  probability form with a `1 − 1e-6` clamp is a no-op in fp16 (rounds to exactly 1.0), giving
  `0 × log(0) = NaN`. That bug silently trained `neurovision` session 1 on NaN for 10.5
  GPU-hours (trap 2). `H(p) = p·softplus(−z) + (1−p)·softplus(z)` is finite everywhere.
- **`GateGenerator`**: concat(cnn, swin_proj, ambiguity) → 1×1×1 conv → GroupNorm →
  LeakyReLU → 3×3×3 conv → sigmoid. Output bias starts at 0 so the gate starts at 0.5.
  `gate_channels: scalar` = one gate value per voxel.
- **`layer_scale`** starts at 1e-4, so at initialisation `fused ≈ cnn_feat`: the network
  starts as a plain CNN U-Net and learns how much transformer context to admit.
- `use_ambiguity=False` builds the **content-only ablation** (the P2 test).
- `forward_with_branch_logits` / `forward_with_ambiguity` / `return_gate=True` expose the
  probe logits, the ambiguity map and the gate map for supervision and research scripts.
- `zero_ambiguity`, `mean_ambiguity`, `shuffle_ambiguity` replace the ambiguity input at
  inference time (used by `scripts/ambiguity_intervention.py`).

**`models/decoder/unet_decoder.py`** — `UNetDecoder` walks the skip pyramid from coarse to
fine: upsample (learnable `ConvTranspose3d` by default, `"interp"` trilinear as an option to
avoid checkerboard artifacts), concatenate the skip, residual conv blocks. Returns a list of
**features**, not logits — heads are attached by the caller. `_match_spatial` crops/pads to
fix odd sizes. Optional `AttentionGate` (Oktay-style), off in production.

**`models/heads/segmentation.py`** — `SegmentationHead`: a 1×1×1 conv to 3 channels, raw
logits (sigmoid lives in the loss and post-processing, never here).

**`models/heads/auxiliary.py`** — one class, `AuxiliaryHead`, used twice:
- **Confidence head**: predicts, per voxel per region, "is the segmentation head right here?"
- **Boundary head**: predicts the 1-voxel shell at each region's surface.
- Same architecture; what makes them different is only the training target, built in the
  loss. Both attach only to the full-resolution decoder feature.

**`models/heads/multitask.py`** — `MultiTaskHead` owns one `SegmentationHead` per
deep-supervision level plus the optional auxiliary heads, and returns a `MultiTaskOutput`
dataclass (`seg` list, `confidence`, `boundary`, `branch_logits`). Its docstring estimates the
auxiliary heads' memory cost (~0.45 GB together) against the 16 GB budget.

**`models/neurovision.py`** — `NeuroVisionX`, the whole network assembled:

```
x (B,4,D,H,W)
 ├─ CNNEncoder  → [c0 (stride1), c1, c2, c3, c4]
 └─ SwinEncoder → [s0 (stride2), s1, s2, s3]
 skips = [c0, fuse(c1,s0), fuse(c2,s1), fuse(c3,s2), fuse(c4,s3)]
 UNetDecoder(skips) → features, fine to coarse
 MultiTaskHead(features) → seg logits (3 levels), confidence, boundary
```

- **`forward` returns one of three types, deliberately**: a plain tensor in eval mode
  (always — sliding-window inference needs a tensor); a list of tensors in training with deep
  supervision; a `MultiTaskOutput` in training with auxiliary heads. The MC-dropout trap
  (trap 5) comes from this: calling `model.train()` to enable dropout changes the return type.
- `forward_with_gates`, `forward_with_ambiguity`, `forward_with_auxiliary` are research
  read-outs that always return full-resolution logits plus the extra maps.
- `swin_encoder=None` builds the CNN-only ablation (a genuine removal, not a gate set to 0).
- `build_neurovision(cfg)` builds everything from config and cross-checks config
  consistency (deep supervision on the model ↔ on the loss; auxiliary heads ↔ `multitask`
  loss) and raises if they disagree.

**`models/qc.py`** — `SegQC`, the second, independent network (Phase C). A small 3D CNN
regressor. Input: 3 channels `[MRI modality, predicted mask for one region, entropy map for
that region]`, resized to a fixed shape. Output: one logit; `predicted_dice(logits)` applies
the sigmoid to give a Dice estimate in [0, 1]. Run once per region. Registered as `segqc`.

### 3.4 `losses/` — what training minimises

**`losses/registry.py`** — `register_loss`, `build_loss(cfg)` (reads
`cfg.training.loss.name`), `available_losses()`. `losses/__init__.py` imports both loss
modules.

**`losses/segmentation.py`**
- `DiceBCELoss`: per-channel soft Dice loss + binary cross-entropy (with logits), weighted
  1 : 1. Dice handles the class imbalance (tumour is a few % of voxels); BCE gives smooth
  per-voxel gradients. Sigmoid per channel, because the regions overlap.
- `DeepSupervisionLoss`: applies a base loss to each output level (upsampling the coarse
  ones to the target's size) with weights halving per level (1, 0.5, 0.25), normalised to
  sum to 1.
- `build_dice_ce` registered as `dice_ce` (the baselines' loss).

**`losses/multitask.py`** — `MultiTaskLoss`, registered as `multitask` (the full model's loss):

```
total = 1.0 × seg         (deep-supervised Dice+BCE on the 3 seg levels)
      + 0.3 × boundary    (Dice+BCE of boundary head vs morphological shell of the label)
      + 0.05 × confidence (BCE of confidence head vs "was the seg head right here?")
      + 0.1 × branch      (BCE of each fusion probe vs the label, max-pooled to its level)
      [+ surface term, off]
```

- `morphological_boundary(mask)`: dilate − erode, a thin shell. Computed under `no_grad`
  (it is a fixed function of the label).
- The confidence target is computed from the segmentation head's *current* prediction under
  `no_grad`. If gradient flowed through it, the model could "cheat" by making segmentation
  logits extreme instead of correct.
- The confidence weight is a small fixed 0.05 rather than a warm-up ramp, because a ramp
  would need a step counter the checkpoint does not save — a resumed run would restart it.
- Branch term uses BCE only (no Dice) and `adaptive_max_pool3d` for the target, so a small
  region still counts at coarse levels.
- `last_components` exposes the unweighted terms for W&B logging.

### 3.5 `training/` — the loop and resume

**`training/trainer.py`** — `Trainer` owns the optimiser, scheduler and AMP `GradScaler`
(built inside, so `resume_from` can restore state into the same objects).
- `_build_optimizer`: AdamW, lr 1e-4, weight decay 1e-5.
- `_build_scheduler`: linear warm-up then cosine decay to `min_lr`, stepped once per epoch.
- `train_one_epoch`: for each batch → `autocast` (AMP) → `loss = loss_fn(model(images),
  labels)` → `scaler.scale(loss).backward()` → every `grad_accum_steps`: unscale, clip
  gradient norm (5.0 in the published runs), `scaler.step`, `scaler.update`. Logs the gradient
  norm median / p90 / max and the fraction of clipped steps.
- `validate`: sliding-window inference on the val set, region metrics, returns
  `val/dice_mean` etc.
- `train`: `for epoch in range(start_epoch, epochs)`. Before each epoch it **predicts**
  whether the next epoch would exceed `max_hours` (elapsed + mean epoch time) and stops
  cleanly if so — a GPU session killed mid-epoch cannot save. After each epoch: scheduler
  step, maybe validate, save checkpoint (best if `val/dice_mean` improved), log to W&B.
- `resume_from(path)`: restores everything and sets `start_epoch`.

**`training/checkpoint.py`** — the only code that reads or writes a training checkpoint.
- `save_checkpoint` writes `last.pt` every epoch, `best.pt` when the monitored metric
  improves, and periodic `epoch_NNNN.pt` snapshots (only the newest `keep_last_n` = 2 kept).
- Payload: model, optimiser, scheduler, scaler, epoch, global step, best metric, W&B run id,
  every RNG state (Python, NumPy, torch, CUDA), and the config as plain data.
- **Atomic writes**: write to a temp file in the same folder, then `os.replace`. A kill
  mid-write can never leave a truncated `last.pt` over a good one.
- **Safe loading**: everything is saved so it loads with `torch.load(weights_only=True)`,
  which cannot execute code. That is why the NumPy RNG tuple is converted to a plain dict
  and the OmegaConf config to plain containers.
- `find_resume_checkpoint(dir)` finds what to resume from; `load_checkpoint` restores it.

### 3.6 `metrics/` — scoring

**`metrics/segmentation.py`** — wraps MONAI's metric maths with BraTS conventions.
- `classes_to_regions`, `binarize(logits, 0.5)`.
- `dice_score`, `iou_score` (`ignore_empty=False`, see §1.4), `hd95(pred, target, spacing)`.
- `compute_case_metrics` → one flat dict per case (`dice_ET`, `hd95_WT`, `gt_empty_ET`,
  `dice_mean`, ...).
- `MetricAggregator`: collect cases → `per_case()` table and `summary()` (mean, std, median,
  count, missing). These become `per_case_metrics.csv` and `summary.csv`.
- Trap 8: never pass CUDA tensors to MONAI's HD95 (it routes through CuPy, which breaks on
  Kaggle).

**`metrics/boundary.py`** — error as a function of distance from the true tumour edge.
- `signed_distance_to_boundary(mask)`: per voxel, mm to the surface (negative inside).
- `boundary_band_masks`: split voxels into bands 0–2, 2–5, 5–10, >10 mm.
- `boundary_stratified_errors`: per region, per band, error rate and voxel counts.
- `distance_band_means`: average any per-voxel map (gate, entropy) by band — used to ask
  "does the fusion gate change near the tumour edge?"
- Trap 4: boundary error shares must be weighted by voxel count, not rate.

**`metrics/lesionwise.py`** — official BraTS 2023 lesion-wise metrics via the `panoptica`
library (only in `.venv-analysis`; tests skip without it). Per region: connected components
(26-connectivity) → drop components < 50 voxels from **both** masks → match prediction and
reference lesions one-to-one by IoU → per-lesion Dice and NSD, F1 over tp/fp/fn.

**`metrics/__init__.py`** re-exports the public names of both metric modules.

### 3.7 `inference/` — from a trained model to a deployable decision

**`inference/sliding_window.py`** — `build_inferer(cfg)` builds MONAI's
`SlidingWindowInferer` from `cfg.inference.sliding_window` (overlap 0.5, Gaussian blending —
different from training's faster settings on purpose). `sliding_window_predict` returns raw
logits only. It calls `model.eval()` unless `set_eval=False` (the MC-dropout escape hatch).

**`inference/postprocess.py`** — logits → BraTS-style class map. `postprocess_logits(logits,
cfg)` runs: sigmoid + per-channel threshold (0.5) → `enforce_nesting` (union inner regions
into outer so ET ⊆ TC ⊆ WT) → `remove_small_components` (< 50 voxels) → optional
`keep_largest_component` / `zero_small_et` (off) → `regions_to_classes` (back to
`{0,1,2,3}`, assigning outer to inner so inner wins). `uncrop_to_original` puts the cropped
prediction back into 240×240×155 using `meta.json`'s bbox. No argmax anywhere.

**`inference/tta.py`** — test-time augmentation: run the model on the volume and its 7 other
flipped copies (2³ flip combinations), un-flip, average **probabilities**. Small Dice gain
plus softer, better-calibrated output. Wired but not measured yet.

**`inference/mc_dropout.py`** — Monte Carlo dropout.
- `dropout_enabled(model)`: context manager that switches **only dropout modules** into train
  mode, leaving the model in eval so `forward` still returns a tensor.
- `mc_dropout_predict`: N stochastic sliding-window passes (with `set_eval=False` so
  dropout stays on) → mean probability, predictive entropy, expected entropy (aleatoric),
  mutual information = predictive − expected (epistemic).
- `count_active_dropout` guards against "dropout probability 0, so every pass is identical".
- Trap 5 is documented here in full: without `set_eval=False`, every pass is the same and the
  uncertainty map is exactly zero, with no error.

**`inference/input_qc.py`** — clinical tasks E3/E4: may this study be segmented at all?
- Label-free: no function takes a label (a test inspects every signature).
- `describe_volume` summarises each scan once into a `VolumeInfo`; each `check_*` function
  reads those summaries and returns a `Finding` with `Severity` OK / WARN / REFUSE — even when
  it passes, so a report shows which checks ran.
- Checks: `check_sequence_completeness` (all four scans present — this *is* E4),
  `check_geometry_consistency` (same shape/affine), `check_spacing` (0.3–3.0 mm, anisotropy
  warn at 2×, refuse at 6× — this is what refuses thick-slice clinical scans),
  `check_finite_values` (no NaN/Inf), `check_intensity_sanity`, `check_shape_against_expected`
  (warn only), `check_brain_mask` (brain volume 600–2600 mL, skull still present?).
- `run_input_qc(cfg, volumes, brain_mask, stage)` composes them; runs twice in the clinical
  job — before registration (no brain mask) and after.

**`inference/gatekeeper.py`** — clinical task E5: the refusal gate.
- Reason it exists: errors multiply in a cascade ("five stages at 95% each is 77% end to
  end"), so the pipeline needs an explicit REFUSE state.
- Five signals, each judged by its own function:
  - `judge_input_qc` — input QC severity maps straight onto the decision.
  - `judge_predicted_dice` — the QC model's Dice estimate per region; **low is bad**.
  - `judge_conformal_band` — how much the conformal mask inflates; **high is bad**.
  - `judge_ood_score` — out-of-distribution score; **high is bad**.
  - `judge_intended_use` — patient age < 18 → refuse (adult glioma only); missing age →
    caution.
- `run_gatekeeper` judges every signal and takes the **worst** decision. An enabled signal
  that could not be measured is a REFUSE, never a silent pass.
- `calibrate_thresholds` sets cut points from **quantiles of the val split's own signal
  values** (caution at the 10th percentile, refuse at the 2nd), never from ground-truth
  Dice. `Thresholds` is frozen and saved to `outputs/gatekeeper/thresholds.json`.
- Deployed config enables `input_qc`, `predicted_dice` and `intended_use`. Conformal band
  and OOD score are computed and shown but do not drive decisions — that choice follows the
  measured Gate C and P1.4 results.

### 3.8 `uncertainty/` — reliability measurements

**`uncertainty/calibration.py`** — ECE, MCE, Brier score, reliability diagrams, temperature
scaling, for 3 overlapping regions.
- Convention: bin the predicted probability `p` against the observed frequency of the label
  (not "confidence vs accuracy"), per region.
- `reliability_curve`, `expected_calibration_error`, `maximum_calibration_error`,
  `brier_score`.
- **Masks — trap 1 lives here.** `union_foreground_mask` (predicted ∪ ground truth) is
  labelled DIAGNOSTIC-ONLY and CIRCULAR: using the label to decide which voxels to score
  inflated reported ECE by 41–57%. The legitimate masks are `predicted_foreground_mask`
  (label-free) and `brain_mask`.
- `fit_temperature`: fit T (per channel) minimising BCE of `logits / T`, on a random voxel
  subsample (`subsample_voxels`) for memory. `apply_temperature`.
- `CalibrationAccumulator`: add cases one at a time, pool the bin counts across all voxels,
  report per case and pooled.

**`uncertainty/conformal.py`** — conformal risk control for the mask's miss rate. **Know this
one well; it is the thesis.**
- Question: at what threshold τ does the mask `{p ≥ τ}` miss, **on average**, at most α of
  the true tumour?
- Loss for case i: `L_i(τ) = |true tumour not in mask| / |true tumour|`. Lower τ → bigger
  mask → fewer misses, so the loss only increases with τ.
- `case_loss_curve`: one pass over a case's voxels computes, for 66 fixed thresholds (dense
  near zero: 31 log-spaced in [1e-4, 0.1], 35 linear in [0.1, 0.95], including 0.5), the
  ground-truth count, false-negative count and mask size. That small `CaseLossCurve` is
  everything later steps need, so refitting at another α is cheap arithmetic.
- `fit_threshold(curves, alpha)`: average the miss-rate curves over n calibration cases →
  `R̂(τ)`; check it is non-decreasing (falsifier); pick the **largest** τ with
  `(n·R̂(τ) + 1) / (n + 1) ≤ α`. The `+1` and `n+1` are the finite-sample correction from
  Angelopoulos et al., *Conformal Risk Control* (ICLR 2024).
- Guarantee: expected miss rate ≤ α on new cases **exchangeable** with the calibration set.
  It is an average, not a per-case promise, and says nothing about the mask being useful —
  which is why `band_inflation` (how much bigger the conservative mask is than the 0.5 mask)
  is mandatory to report.
- Empty-ground-truth cases are 0/0: excluded and counted, never scored as 0 or 1.
- `realised_risk` measures the actual miss rate at a fixed τ on another set (test, SSA, PED).
  Holds on test; breaks on SSA and PED in proportion to the shift.
- `load_curves_npz` reloads cached curves; `uncertainty/__init__.py` re-exports the public API.

**`uncertainty/risk_coverage.py`** — selective prediction.
- `case_uncertainty_scalars`: reduce voxel maps to per-case numbers.
- `risk_coverage_curve`: sort cases from most to least confident, mean Dice of the kept
  cases at every coverage level.
- `oracle_curve` (sort by true Dice — the best possible) and `random_curve` (flat — the null).
- `bootstrap_curve_ci`, `referral_table` (model vs oracle vs random at chosen coverages),
  `uncertainty_error_correlation` (Spearman/Pearson).

### 3.9 `explainability/` — what did the model look at?

**`explainability/gradcam.py`** — Seg-Grad-CAM (Vinogradova et al. 2020). Segmentation has no
single class score, so it sums one region's logits over the model's own predicted-positive
voxels, then runs ordinary Grad-CAM on that sum at a chosen layer: average the gradient per
channel → weights → weighted sum of activations → ReLU. `available_layers`, `resolve_layer`
(helpful error on a typo), `center_patch_on_mask`. Used live by the clinical job.

**`explainability/integrated_gradients.py`** — Integrated Gradients via Captum, attributed
to the four **modalities** ("which scan did this prediction rely on?"). Key finding in the
docstring: an all-zeros baseline is wrong for this network — a spatially constant input
collapses every GroupNorm/LayerNorm (std = 0), so 99.9% of the change happens in the first
tenth of the path and completeness recovers 0.2%. A **Gaussian-noise baseline** recovers
99.3%. `modality_ranking` sorts the result.

**`explainability/attention_rollout.py`** — captures the Swin blocks' attention weights
(hooks on MONAI's `WindowAttention`), maps windowed attention back to voxel positions
(handling shifted windows and padding), and composes a stage's two blocks (`attention_rollout`).
Inspection only. Only three attention stages exist in production (strides 2/4/8) because the
4th MONAI stage was dropped.

**`explainability/faithfulness.py`** — tests whether a heatmap is honest instead of just
looking plausible.
- `deletion_curve`: remove the most-attributed voxels first; a faithful map makes the
  prediction collapse quickly. `insertion_curve`: the mirror image.
- `pointing_game`: does the single hottest voxel land on the tumour?
- `attribution_mass_ratio`: share of attribution inside the tumour — the recommended headline.
- `random_attribution_like`: the null baseline every method must beat.
- `compare_methods`: runs everything over several methods.

### 3.10 `anatomy/` — where is the tumour, in brain terms?

BraTS 2021 images are already registered to the **SRI24 atlas** (an average healthy brain
with labelled structures), so the atlas lines up with a case by voxel index — no registration
needed, only a pure axis flip/permute.

**`anatomy/atlas.py`** — loads SRI24 (parcellation + tissue maps) into the BraTS voxel frame.
- `solve_index_transform` / `apply_index_transform` / `reorient_to_target`: pure axis
  permutation + per-axis reversal, solved from the two affines. No interpolation.
- `parse_lut` reads the label table and merges per-plane sub-labels into parent structures.
- `Atlas` holds the arrays; `structure_mask(name)`, `coverage()`. `load_atlas(cfg)`.

**`anatomy/alignment.py`** — Phase 0's gate: prove the atlas really lands on our cases.
- `brain_mask_check` (GATING): Dice of atlas brain vs case brain. 0.9394 correct vs 0.7334
  front-back mirrored. **But blind to a left-right flip**: a mirrored atlas scores 0.9416,
  *higher* (trap 3), because brains are nearly symmetric.
- `laterality_check` (GATING): uses `_L`/`_R` structure centroids to prove left is left —
  exists precisely because of trap 3.
- `lobe_distribution_check` (ADVISORY): tumour lobe distribution vs published epidemiology.
- `run_checks` → `AlignmentReport` (`passed`, `failures`, `summary`).

**`anatomy/burden.py`** — pure arithmetic on a class map, no model, no torch.
`burden_profile(classes, geom)` → one flat CSV row: per-class and per-region volumes (mm³),
composition fractions, connected components (multifocality, pieces ≥ 50 mm³), surface area
(marching cubes), sphericity (1 = perfect sphere), left/right split about the mid-sagittal
plane (`estimate_midline_index` finds the plane of best mirror symmetry), centroid.
`CaseGeometry.from_meta` reads spacing and the left/right convention from `meta.json`.

**`anatomy/localize.py`** — which atlas structures the tumour touches.
- Two separate questions per structure: `frac_of_tumour` (where is the tumour?) and
  `frac_of_structure` (how badly is this structure affected?).
- `load_knowledge` loads `knowledge/eloquence_map.yaml` + `aal_lobes.yaml` and validates
  them against the atlas. `atlas_for_case` crops the atlas to a case's bbox.
- `localize_mask` / `localize_case` → long table; `summarize_case` → one row per case.
- `eloquent_union_mask` + `distance_to_eloquent`: mm from tumour to the nearest "eloquent"
  (function-critical) structure. Known limitation: 100% of BraTS gliomas come out "near
  eloquent", so this field does not discriminate.

**`anatomy/involvement.py`** — coarser questions than single structures, from
`knowledge/involvement_groups.yaml`: does the tumour touch the ventricles? deep white matter?
(`group_overlap`); what share is grey matter / white matter / CSF? (`tissue_overlap`); which
single structure is it centred on? (`epicentre` — a location, not a claimed origin).
`involvement_profile` runs all three.

**`anatomy/shape_descriptors.py`** — purely geometric shape numbers (elongation, flatness,
rim thickness of the enhancing ring). Kept separate from `burden.py` so the published
`burden.csv` keys never change. Key names avoid clinical-judgement words because the report
layer scans text for them.

**`anatomy/atlas_export.py`** — packs the atlas parcellation into a `uint8` structure-index
volume plus a metadata table (`structure_index_volume`, `structure_table`) so the 3D viewer
can draw structure shells. `anatomy/__init__.py` holds only a one-line package docstring.

### 3.11 `reporting/` — outputs a human reads

**`reporting/report.py`** — Phase 4 structured report. Runs no model: it assembles burden,
localisation and summary tables plus provenance into one JSON per case (`build_report`),
renders Markdown (`render_markdown`), writes both (`write_report`). Every report carries a
disclaimer, a "not claimed" block, a mass-effect caveat and eloquence evidence with citation.
`json_safe` converts NaN/numpy types so `json.dumps(allow_nan=False)` never fails.
`Provenance` records which segmentation (prediction or ground truth) and which files each
value came from.

**`reporting/molecular.py`** — the "confirmed pathology" block. A user **enters** marker
results (IDH, 1p/19q, MGMT, ...), and `cns5_lookup` maps them to a WHO CNS5 diagnosis
**name** from `knowledge/molecular_markers.yaml`. Nothing is predicted from imaging.
`empty_molecular_block`, `validate_entered`, `merge_pathology` (returns a new block, never
mutates).

**`reporting/dicom_seg.py`** — clinical task E6: writes the mask as a DICOM Segmentation
object (via `highdicom`) so it opens in hospital viewers (OHIF, PACS).
`check_geometry_against_source` refuses to write when the mask's grid does not match the
source series — it never resamples silently. `segment_masks` splits the class map into one
mask per segment; `read_source_geometry`; `write_dicom_seg`.

**`reporting/dicom_frames.py`** — the geometry fix behind E6. `sort_datasets_along_normal`
orders DICOM slices by physical position; `mask_to_dicom_frames` samples the mask onto the
DICOM series' own pixel grid using real coordinates (nearest neighbour), instead of assuming
array axes match DICOM rows/columns — that assumption mis-registered a real study on
2026-09-18. `reporting/__init__.py` holds only a package docstring.

### 3.12 `analysis/` — the statistics behind every claim

All pure (no file I/O, no Hydra); each has a driver script.

**`analysis/statistics.py`** — every "A beats B" claim goes through here.
`paired_bootstrap_ci` (10,000 resamples), `wilcoxon_signed_rank`, `paired_effect_size`
(Cohen's dz, Hedges' g), `holm_bonferroni`, `metric_direction` (higher-is-better for Dice,
lower for HD95), `load_per_case`, `compare_models` (all of the above per metric),
`format_comparison`.

**`analysis/stratify.py`** — performance by tumour size. SSA tumours are ~1.7× bigger than
BraTS 2021 ones, and big tumours are easier, so a raw cross-cohort comparison confuses
"domain shift" with "size". `ground_truth_volumes`, quantile bins, `stratified_summary`,
`stratified_comparison`, `overlapping_volume_range` + `volume_matched_subset` (compare only
the size range both cohorts share).

**`analysis/replay.py`** — re-score saved logits at a different threshold or post-processing
chain without the model or a GPU. `load_case_logits`, `replay_case`, `threshold_sweep`,
`postprocess_ablation`, `per_case_replay`. Verified to reproduce a published row exactly.
(Small docstring slip: it labels the reproduced ET 0.870859 as `baseline_unet3d`'s; that
value is `neurovision`'s published ET — the baseline's is 0.8442.)

**`analysis/detection.py`** — Gate 1: does inter-branch disagreement know something entropy
does not? `case_entropy_scalars`, `spearman`, `partial_spearman` (correlation after removing
entropy's effect) with bootstrap CI, `auroc` (via Mann-Whitney U), `residualised_auroc`.

**`analysis/localisation.py`** — Gate 2: does entropy + disagreement localise voxel errors
better than entropy alone? `rank_transform` per case, `fit_combiner` (logistic regression
fitted on val only), `case_auroc`, `recall_at_budget` (share of errors inside the top-k% most
flagged voxels).

**`analysis/equivalence.py`** — `paired_tost`: the right test for "as good as MC-dropout at a
tenth of the cost". A non-significant difference is not equivalence.

**`analysis/qc_inference.py`** — shared helpers to turn saved logits into the QC model's input:
`entropy_from_logits`, `resize_packed`, `load_case_arrays`, `pack_sample`.

**`analysis/qc_validate.py`** — Gate C statistics: does the trained QC model beat free entropy
at spotting a bad mask? `falsification_check` (our Dice reconstruction matches the published
numbers), `cell_endpoints` per cohort × region, `mark_family` (Holm), `gate_c_verdict`
(the pre-registered rule, verbatim), `silent_failure_table` (does QC grow over-optimistic
under shift?). Result: MIXED — wins one cell of five.

**`analysis/gatekeeper_calibration.py`** — builds the val-split table the gate's thresholds
are fitted from: `qc_predicted_dice_table` (run SegQC on each val case's own prediction),
`case_conformal_band_widths`, `ood_score_table` (placeholder: mean foreground entropy),
`build_gatekeeper_calibration_table`.

**`analysis/error_budget.py`** — Phase G, the end-to-end error budget. For each cohort:
`decide_cases` (run the frozen gate on every case), `label_usable` (usable = Dice ≥ 0.7 on
both WT and TC), `outcome_cells` (accepted × usable → `correct_accept` / `silent_failure` /
`over_refusal` / `correct_refusal`), `bootstrap_rate_ci`, `summarise_cohort`, `coverage_curve`,
`taxonomy_table`. Produced the 85/78/24% usable and 4.2/18.3/49.5% silent-failure numbers.

**`analysis/local_recalibration.py`** — the "what would a new site need?" counterfactual.
Re-fit the conformal threshold on k cases **from the new cohort itself**, test on the rest:
`draw_split`, `evaluate_split`, `run_local_recalibration` (over α and k), `min_feasible_k`
(smallest k that could possibly meet α — the `(n·R̂+1)/(n+1)` term makes tiny k infeasible),
`summarise`.

**`analysis/ood.py`** — P1.4, an input-statistics out-of-distribution score: 46 label-free
features per study (`extract_features`), a shrinkage-covariance Gaussian fitted on **train**
only (`fit_ood_model` → `OODModel`), Mahalanobis distance as the score (`ood_scores`), cut
points from **val** quantiles (`quantile_cuts`), `flag`. Sees the SSA shift (60% flagged vs
9.5% test) but not which masks fail, so it stays display-only.

**`analysis/population.py`** — Phase 5 cohort anatomy: `structure_involvement_frequency`,
`lobe_burden_distribution`, `eloquence_rates`, `laterality_distribution`,
`summarize_population`.

**`analysis/report_agreement.py`** — does a better mask give a better report? Compares a
ground-truth-derived report with a prediction-derived one: `structure_set`, `top_structure`,
`jaccard`, `compare_reports`, `agreement_table`. Result was negative: better Dice did not
produce a better report.

**`analysis/nnunet_import.py`** — Gate A (vs nnU-Net). nnU-Net writes hard labels on the raw
240×240×155 grid in its own label convention; this converts them into our cropped frame and
labels (`nnunet_to_project_labels` is self-inverse, `import_prediction`), turns hard labels
into saturated ±20 pseudo-logits (`labels_to_pseudo_logits`) so the same `replay_case`
scoring path is used, and `roundtrip_self_test` proves the conversion is lossless before any
real score.

**`analysis/real_dicom_scoring.py`** — P1.2 scoring of the clinical front end against BraTS
ground truth: `uncrop_to_full`, `to_reference_grid` (flip/permute only), `region_masks`,
`score_case` (Dice, HD95), `roundtrip_self_test` (ground truth vs itself must give Dice 1.0),
`lateralisation_check` (detect a left-right flip from tumour content).

### 3.13 `visualization/` — figures and tables

**`visualization/qc.py`** — big, legible figures for catching a broken pipeline by eye:
`plot_case_slices` (mid-slices of all 4 modalities in 3 planes, label overlay),
`plot_intensity_histograms` (before/after normalisation). numpy + matplotlib only.

**`visualization/figures.py`** — paper figures. Never reads files: takes arrays/DataFrames,
returns a matplotlib `Figure`; `save_figure` writes PDF + PNG. House style (`paper_rc`,
`paper_style`), colour-blind-safe palette. Figures: qualitative panel, metric box plots,
reliability diagram, risk-coverage, training curves, comparison forest plot, gate maps, band
profile, structure involvement, lobe distribution, modality attribution, attribution panel.

**`visualization/tables.py`** — paper tables built from result files, rendered to Markdown and
LaTeX so no number is ever retyped by hand: results table, comparison table, boundary table,
`escape_latex`, `write_table`.

---

## Part 4 — `app/backend/`, the FastAPI server

Run from the repo root: `.venv-clinical/bin/uvicorn app.backend.main:app --port 8000`.
Every route lives under `/api`. Two kinds of data are served:

1. **Precomputed research cases** (`/api/cases/*`, `/api/report/*`): output that
   `scripts/evaluate.py` and `scripts/report.py` already wrote. Viewing them costs no GPU and
   needs no torch.
2. **Live clinical jobs** (`/api/clinical/*`): a DICOM study uploaded now, run through the
   whole pipeline on CPU in a background thread.

**`main.py`** (17 lines) — `app = create_app()`. Built at import time because uvicorn's
reloader re-imports the module.

**`config.py`** — `Settings` (frozen dataclass) built once by `get_settings()`. Every path
comes from an environment variable with a repo-relative default, never hardcoded:
`NVX_PREP_DIR`, `NVX_EVAL_DIR`, `NVX_CHECKPOINT`, `NVX_EXPERIMENT`, `NVX_CACHE_DIR`,
`NVX_REPORT_DIR`, `NVX_MAX_CASES`, `NVX_DEMO_OVERLAP`. Derived properties:
`predictions_dir`, `metrics_csv`, `logits_dir`. Defaults point at `baseline_unet3d`; the demo
command overrides them to `neurovision`.

**`volumes.py`** — loads cases and turns volumes into bytes the browser can use.
- Everything served is in the **cropped** preprocessed frame, so image, label, prediction and
  uncertainty share one voxel grid. Saved predictions are in the original 240×240×155 grid, so
  `_crop_to_meta` re-crops them with `meta.json`'s bbox. Skipping this misaligns the overlay
  by the crop offset and still looks plausible.
- Volumes go over the wire as raw `uint8` bytes in `(D, H, W)` C order, with the shape in an
  `X-Volume-Shape` header, so the viewer can slice any plane from one download.
- `list_cases` (cases with both a volume and a prediction, ranked by Dice),
  `load_modality` (z-scored → 0–255 display window), `load_mask` (prediction or ground
  truth), `entropy_from_logits` + `load_uncertainty` (single-pass entropy, scaled to 0–255),
  `region_voxel_counts`, `ground_truth_wt_volume_ml`, `select_showcase` (k cases spread
  evenly by tumour size).

**`api.py`** — the HTTP layer. Handlers are thin; filesystem errors are mapped to HTTP codes
once in `_register_exception_handlers`.
- Research routes: `GET /health`, `/cases` (optional `?showcase=k`), `/cases/{id}`,
  `/cases/{id}/volume/{modality}`, `/cases/{id}/mask/{prediction|label}`,
  `/cases/{id}/uncertainty`, `/cases/{id}/atlas`, `/cases/{id}/profile` (per-slice tumour
  fraction, disagreement, entropy), `/atlas/structures`, `/report/{id}`,
  `/report/{id}/markdown`.
- `_load_verified_report` refuses to serve a ground-truth-derived report next to a
  prediction (checks the report's provenance).
- Clinical routes: `POST /clinical/upload` (202 Accepted, queues a job), `GET
  /clinical/jobs`, `GET|DELETE /clinical/jobs/{id}`, then for a finished job: `volume`,
  `mask/prediction`, `geometry`, `atlas`, `uncertainty`, `conformal-band/{region}`,
  `gradcam/{region}`, `dicom-seg` (download), `report`, `report/markdown`,
  `POST export` (zip of report + DICOM-SEG + job.json + viewer snapshots, with a SHA-256
  manifest), `GET|PUT pathology` (the one mutating route; written atomically).
- Overlay responses carry an `X-Uncertainty-Kind` header (`predictive-entropy-single-pass`,
  conformal band, Grad-CAM, atlas index) so the viewer cannot mislabel what a map is.
- `create_app()` adds CORS, registers the router and exception handlers, and **rehydrates**
  clinical jobs from their `job.json` files so a restart does not lose them.

**`inference.py`** — live CPU segmentation of one preprocessed case. The only backend module
besides `clinical_jobs.py` that uses torch, and it imports torch and Hydra **inside** functions
so `import app.backend.inference` never fails on a machine without them.
- `checkpoint_available`, `inference_status` never raise.
- `_compose_cfg` (Hydra, CPU-only, behind a lock — Hydra's global state is not thread-safe),
  `_load_model`, `_predict_logits` (sliding window), `_postprocess_to_classes`.
- `segment_case(settings, case_id, save_logits, progress)`: cached per case with a per-case
  lock, atomic `.npy` writes. Prediction stays in the cropped frame.
- `explain_case`: Seg-Grad-CAM for one region, cached.

**`jobs.py`** (43 lines) — only `job_root(settings)` remains: the folder for job files
(`NVX_JOB_DIR`). The old NIfTI-upload job code was removed on 2026-09-18 when the DICOM path
replaced it.

**`clinical_jobs.py`** — the real clinical pipeline as a pollable background job. The most
important backend file.
- `create_clinical_job`: validate the zip (size, type, no "zip-slip" paths escaping the
  folder), extract it, register a `ClinicalJob` (state `queued`).
- `start_clinical_job` runs `run_clinical_job` on a daemon thread. Jobs live in an in-memory
  dict behind a lock and are persisted to `job.json` atomically after every update.
- `run_clinical_job` — the stages, with the progress value the UI shows:

| Stage | What runs | Can refuse? |
|---|---|---|
| `ingest` (0.05→0.15) | `dicom_ingest.ingest_study`: pick T1/T1CE/T2/FLAIR series | yes — missing roles |
| `input_qc (pre-preprocessing)` (0.2) | `run_input_qc(stage="pre_registration")` | yes |
| `clinical_preprocessing` (0.5) | `clinical_preprocess`: co-register, SRI24, HD-BET | — |
| `input_qc (post-preprocessing)` (0.55) | `run_input_qc` with the brain mask | yes |
| `research_preprocessing` (0.6) | unchanged `preprocess_case` | yes |
| `segmenting` (0.6→0.85) | `inference.segment_case(save_logits=True)`, `neurovision` model | — |
| `gatekeeper` (0.85) | QC-model predicted Dice + conformal band → `run_gatekeeper` | yes |
| `explaining` (0.9) | Grad-CAM for WT and TC | never fails the job |
| `exporting_dicom_seg` (0.95) | uncrop → resample to native → DICOM-SEG | never fails the job |
| `generating_report` (0.98) | structured anatomical report | — |
| `done` (1.0) | | |

- `_live_predicted_dice`: runs `SegQC` on the job's logits, one region at a time.
- `_live_conformal_band_width`: size of the mask at the fitted conformal threshold divided by
  the size at 0.5. `clinical_conformal_band_mask` makes the per-voxel band for display.
- **`refused` is a successful terminal state, never `failed`.** Refused = a label-free gate
  correctly said no, with a named reason in `error`. Failed = something unexpected broke.
- The model is **pinned**: segmentation always uses `neurovision` via
  `NVX_CLINICAL_CHECKPOINT`, because the QC model and gate thresholds were calibrated only
  against it, whatever the generic `Settings` say.
- Grad-CAM, DICOM-SEG and report are **supplementary**: their failure is logged and the job
  still finishes `done`.
- `rehydrate_clinical_jobs` reloads jobs at startup; a job that was mid-run when the server
  died comes back as `failed`. `delete_clinical_job` removes its folder.

**`__init__.py`** — one-line package docstring.

---

## Part 5 — `scripts/`, every entry point

Almost every script is a Hydra entry point (`python scripts/X.py key=value ...`) composed from
`configs/config.yaml`. Most are thin: read artifacts → call a `src/` module → write CSV/JSON.
Several say "read the pre-registration first": they implement a written-in-advance test
verbatim and must not reinterpret it. `tests/script_loader.py` lets the tests import these
files, since `scripts/` is not a Python package.

### 5.1 Setup and data

**`preprocess.py`** — scans a raw BraTS folder (`brats.scan_brats_root`) and runs
`preprocessing.preprocess_case` on every case in parallel, writing `image.npy`, `label.npy`,
`meta.json` per case plus a `metadata.csv`. Runs once, on the Mac CPU. Skips finished cases.

**`make_splits.py`** — creates (or verifies) the frozen 70/15/15 split in
`configs/data/splits.yaml` (seed 42) via `dataset.make_splits`. Refuses to overwrite unless
told to, because every reported number depends on it.

**`make_crossfit_splits.py`** — for the D3 fine-tune experiment: splits each external cohort
(SSA, PED) once into two halves, so a model fine-tuned on one half is scored on the other and
every case is scored exactly once by a model that never saw it. Writes
`splits_{ssa,ped}_cf{0,1}.yaml`.

**`fetch_atlas.py`** — downloads the SRI24 atlas from NITRC at setup time (it is CC-BY-SA, so it
must never be committed), verifies every archive against a pinned SHA-256 (a silently
different atlas would give plausible reports about the wrong anatomy), extracts it.

**`package_for_kaggle.py`** — plain argparse. Gathers the preprocessed cache + the split file
into one upload folder and size-checks it against Kaggle's dataset limit before a 40-minute
upload.

**`show_config.py`** — prints the fully composed Hydra config and exits. Use it to check
overrides resolve correctly before spending GPU time.

### 5.2 Training and GPU sessions

**`train.py`** — the training entry point. `run_training(cfg)`:
1. `setup_logging`, `set_seed(cfg.seed)` (before any dataset or model exists).
2. `get_device(cfg)`; `build_dataloaders` (frozen split → data dicts → transforms → MONAI
   datasets → `DataLoader`s).
3. `build_model(cfg)` → `.to(device)`; `build_loss(cfg)`; `Trainer(...)` with no W&B yet.
4. `apply_resume_or_init`: if a `last.pt` exists in the checkpoint folder, **resume** it
   (`select_resume_checkpoint`); else if `training.checkpoint.init_from` is set, load those
   weights as a fresh **fine-tune** (`resolve_init_from_checkpoint` refuses a path inside the
   run's own folder, which would silently resume instead).
5. `init_wandb` — after resume, so it can reattach to the same W&B run id.
6. `trainer.train()`, W&B finished in a `finally` block.

Typical command: `python scripts/train.py +experiment=neurovision data.root_dir=...`.

**`smoke_test.py`** — end-to-end CPU test of the real pipeline on two tiny synthetic cases: real
datasets, transforms, registry-built model and loss, `Trainer`, checkpointing, sliding-window
validation. Nothing mocked. Run it before any GPU session.

**`gpu_session.py`** — one safe training session on any GPU host (Kaggle notebook, SSH + tmux,
or SLURM). Lifts the Kaggle notebook's wiring into a CLI: preflight checks, the training call,
post-training checks. It catches a checkpoint from the wrong run before it burns a session
(`--expect-epoch`, `--expect-wandb-id`, scanning for non-finite weights), catches a pinned git
ref that drifted (`--expect-sha`), can insist on a GPU (`--require-cuda`), and reads secrets
only from the environment. Each check answers a loss already suffered
(`docs/gpu_session_checklist.md`).

**`run_ablation_grid.py`** — pure planning tool, no torch. Lists the exact `train.py`
commands for the six-variant fusion ablation grid and estimates its GPU-hour cost against the
weekly quota.

**Shell scripts** (`scripts/cluster/`, tested by `tests/test_cluster_scripts.py`):
- `run.sbatch` — SLURM job: run `gpu_session.py` on one GPU from a pinned checkout.
- `run_tmux.sh` — run `gpu_session.py` in a detached tmux session on an SSH box, logs to
  `NVX_LOG_DIR`.
- `nnunet.sh` — run one budgeted slice of nnU-Net training (Gate A) the same way.
- `_walllimit.sh` — sourced helper: refuse to start unless the requested hours are ≥ 0.5 h
  below the SLURM wall limit.

**`scripts/reproduce.sh`** — the exact command sequence from raw BraTS to every number in the
paper; the executable half of `docs/research_docs/reproducibility.md`.

### 5.3 Evaluation and re-scoring

**`evaluate.py`** — evaluate a checkpoint on a frozen split. Per case (`evaluate_case`):
deterministic sliding-window pass → post-processing → metrics; optionally MC-dropout
(uncertainty maps, Dice still from the deterministic pass) or TTA (replaces the deterministic
pass; never both at once). `run_evaluation` writes `per_case_metrics.csv`, `summary.csv`,
`eval_config.yaml`, uncropped `predictions/`, optionally `logits/` (fp16) and
`probabilities/`, boundary-band errors and lesion-wise metrics. `load_eval_model` checks the
checkpoint's architecture matches the config (`strict_arch_check`).
Example: `python scripts/evaluate.py +experiment=neurovision inference.evaluation.split=test`.

**`rebuild_predictions.py`** — rebuilds the `predictions/` cache (uint8 class maps, original
geometry) from saved `logits/`, so the viewer can browse an eval directory whose predictions
were deleted to save disk.

**`replay_logits.py`** — driver for `analysis.replay`: threshold sweeps and post-processing
ablations on saved logits, written to result files. No model, no GPU.

**`compare_family.py`** — Holm-corrects one comparison across several cohorts and metrics at
once, because a pre-registration often declares one family spanning e.g. {ET, TC, WT} on test
AND on pooled SSA+PED; correcting each cohort separately would under-correct.

**`ref_et_comparison.py`** — regenerates the paper's "reference-ET" column: the ET comparison
restricted to the 184 cases whose reference ET is non-empty, because two of the five empty-ET
cases carry ~40% of the headline ET difference.

**`label_composition.py`** — D3 note 58 mechanism: what share of each tumour is labelled
necrotic / oedema / enhancing, in references and predictions. Asks whether the paediatric
cohort labels nearly the whole tumour as "core". Reads only `meta.json` label counts.

**`silent_failure_by_region.py`** — splits each cohort's silent failures by which region (WT
or TC) made the mask unusable, since PED's tumour-core label means something different from the
adult one.

### 5.4 Calibration, conformal and uncertainty

**`calibrate.py`** — produces the calibration report from two eval directories:
`calibration.fit_dir` (val) and `calibration.apply_dir` (test). Temperature is fitted on val and
applied to test, and the script raises if both point at the same folder (fitting and reporting
on one split makes the number meaningless). Reports ECE/MCE/Brier per region under label-free
masks, reliability tables and risk-coverage. CPU, file-driven, no model.

**`conformal.py`** — driver for `uncertainty.conformal`, per
`preregistration_conformal.md`. Three steps: **extract** a loss curve per case per region from
saved logits (cached to `curves.npz`), **fit** τ on BraTS val at each α (`fit.json`), **apply**
the frozen τ to test, SSA and PED and report realised risk with bootstrap CIs plus mask
inflation.

**`local_recalibration.py`** — driver for `analysis.local_recalibration`: re-fit the conformal
threshold on k cases of SSA or PED itself, over many random splits and several k, to measure
how many local cases a new site needs to restore the bound. Reads saved `curves.npz` only.

**`mc_comparison.py`** — secondary analysis outside the pre-registered families: is
inter-branch disagreement as good an error localiser as MC-dropout mutual information, at a
tenth of the cost? Uses TOST for the equivalence claim.

**`score_confidence.py`** — first-ever scoring of the trained confidence head (master plan A5):
does it localise the segmentation head's own errors better than free single-pass entropy?
Result: ET 0.855, TC 0.871, but WT 0.477 (slightly worse than chance).

**`confidence_diag.py`** — diagnostic R8 for that WT 0.477: tests whether it is a sign bug,
a channel bug, or a real property of the head, on recomputed tensors.

### 5.5 Research on the fusion mechanism

**`extract_gates.py`** — saves fusion gate maps (`forward_with_gates`) for a handful of cases,
one tumour-centred patch each. The primary evidence that the gate fires at all.

**`gate_boundary_profile.py`** — prediction P1, "the mechanism fires": averages gate values by
distance-to-tumour-edge band (`metrics.boundary.distance_band_means`) to test whether the gate
opens and closes with anatomy.

**`gate_failure_detection.py`** — exploratory, not pre-registered: can a label-free read-out of
the gate predict a case's own Dice? Holm across the whole table.

**`extract_ambiguity.py`** — saves the ambiguity maps (`forward_with_ambiguity`) over a whole
split, by sliding window (not patches like `extract_gates.py`), because Gate 1 needs
whole-volume maps.

**`extract_ambiguity_serial.py`** — runs `extract_ambiguity.py` one case at a time, resumably.
Exists because parallel workers on the 16 GB Mac drove it into swap (one worker at
`sw_batch_size=4` peaks at 12.39 GiB — trap 6).

**`detection_stats.py`** — Gate 1 driver: does disagreement carry failure-detection signal that
entropy lacks (partial Spearman, residualised AUROC)? Applies the pre-registered decision table.
Outcome recorded as PARTIAL: localises error out of distribution but does not rank cases.

**`gate2_localisation.py`** — Gate 2 driver: entropy + disagreement (logistic combiner fitted
on val) vs entropy alone for per-voxel error localisation; per-case AUROC and recall at a
flagged-voxel budget; six-test Holm family.

**`ambiguity_intervention.py`** — tests P2 ("the ambiguity conditioning is necessary") at
inference time, without the ~23 GPU-hour retraining: runs each case four times (ambiguity
input unchanged / zeroed / replaced by its mean / spatially shuffled) and records how far the
gate maps move and what happens to Dice, IoU and HD95. Weaker evidence than retraining, and
says so. The retraining ablation
(`ablation_content_only_gate`) was later run and came back null: the content-only gate matches
the full model.

### 5.6 QC model, gatekeeper and error budget

**`train_qc.py`** — trains `SegQC` (Phase C3) on `(degraded predicted mask, true Dice)` pairs
from `data.qc_pairs`, built from the deployed model's own saved logits. Its docstring restates
the binding principle: train on predicted masks, never on damaged ground truth.

**`validate_qc.py`** — Gate C driver (`analysis.qc_validate`): QC model vs entropy at spotting
bad masks, per cohort × region, plus the C5 silent-failure test. Verdict: MIXED.

**`calibrate_gatekeeper.py`** — wires `build_gatekeeper_calibration_table` (val split) to
`calibrate_thresholds` and writes `outputs/gatekeeper/thresholds.json`, the frozen cut points
the live gate loads.

**`ood_score.py`** — P1.4 driver: extract the 46 features for every case, fit on train, set
cuts on val, score test/SSA/PED, in the order the pre-registration fixes.

**`error_budget.py`** — Phase G driver: runs the frozen gate over every cohort using saved
artifacts (logits, QC predictions, conformal fit, thresholds) and writes usable rates, silent
failure rates, the pipeline-level coverage curve and the outcome taxonomy.

### 5.7 Explainability

**`explain.py`** — writes attribution maps (Integrated Gradients, Grad-CAM, attention rollout)
plus faithfulness metrics for a handful of cases to `.npz`, for the paper's explainability
panel.

### 5.8 Anatomy and reports

**`validate_atlas.py`** — Phase 0 gate driver: runs `anatomy.alignment.run_checks` on the real
preprocessed BraTS tree and writes results plus an overlay QC figure.

**`burden.py`** — one `burden_profile` row per case → `burden.csv`. Reads a saved prediction or
the ground-truth label; no model.

**`localize.py`** — intersects each case with the atlas → `anatomy.csv` (one row per case ×
region × structure) and `anatomy_summary.csv`.

**`report.py`** — joins `burden.csv` and `anatomy.csv` per case and calls
`build_report`/`write_report`. Recomputes nothing, so a report can never disagree with the
published CSVs.

**`population_stats.py`** — cohort-level anatomy tables and two figures from `localize.py`
outputs.

**`report_agreement.py`** — Phase 5: does a more accurate segmentation produce a report that
agrees more with the ground-truth report? Came back negative.

### 5.9 nnU-Net comparison (Gate A, in progress)

**`export_nnunet_dataset.py`** — converts our frozen split into nnU-Net v2's raw dataset layout
(`imagesTr`, `labelsTr`, `dataset.json`, `splits_final.json`), so nnU-Net trains on exactly our
training cases.

**`nnunet_session.py`** — runs one ~11 h Kaggle T4 slice of nnU-Net's **default, unmodified**
recipe (3d_fullres, fold "all", 1000 epochs), stopping cleanly between epochs and resuming next
session through nnU-Net's own resume path. Only decides *when* to stop.

**`nnunet_predict_session.py`** — nnU-Net inference on the 189 test cases, from
`checkpoint_final.pth` only (`checkpoint_best` was selected on training cases and would bias
the comparison in nnU-Net's favour).

**`score_nnunet.py`** — scores nnU-Net's predictions through **our** metric path via
`analysis.nnunet_import` (round-trip self-test first).

### 5.10 Clinical pipeline and real DICOM

**`run_clinical_study.py`** — runs the live clinical pipeline (`app.backend.clinical_jobs`) on
one DICOM study from the command line, end to end, and writes a summary JSON. Makes the
pipeline a reproducible command instead of a scratch script.

**`validate_real_dicom.py`** — P1.2: runs the original RSNA-MICCAI DICOM of BraTS test-split
patients through the real pipeline, scores against BraTS ground truth
(`analysis.real_dicom_scoring`), aggregates completion rate, usable rate and front-end cost.
Result: 40% refused for thick slices, WT −0.23 Dice on accepted cases.

**`real_dicom_offsets.py`** — R3: is that WT loss a coordinate bug or registration
disagreement? Measures the integer voxel offsets between our registered output and the BraTS
grid per case. Answer recorded as mostly registration disagreement.

### 5.11 Figures and assets

**`plot_semester_figures.py`** — the two Milestone 5 figures the semester report cites:
`fig_local_recalibration.png` (from `local_recalibration.py`'s summary, note 52) and
`fig_realised_vs_nominal.png` (the conformal bound's realised miss rate per cohort against the
nominal α, from `conformal.py`'s `realised_risk.csv`, note 42). Recomputes nothing.

**`build_hero_brain_mesh.py`** — builds the landing page's 3D cortex mesh from the SRI24 T1
template: blur → marching cubes → largest piece → decimation → Taubin smoothing → normals →
binary files. Offline asset build.

---

## Part 6 — `configs/` (Hydra), every file

`configs/config.yaml` is the root. Its `defaults:` list picks one file per group; `_self_` comes
last so the root's own values win. It also sets `experiment_name`, `seed: 42`, `device: auto`,
`output_dir: outputs/${experiment_name}`, the `wandb:` block (its `run_id` is filled at run
time and saved in checkpoints so a resumed run logs to the same W&B run), and pins
`hydra.job.chdir: false` so relative paths never move mid-run.

### `data/`
| File | What it holds |
|---|---|
| `brats.yaml` | modalities, `in_channels: 4`, `regions: [ET, TC, WT]`, `num_classes: 3`, 1 mm spacing, `patch_size: [96,96,96]` default, `pos_neg_ratio`, `samples_per_volume: 4`, all augmentation probabilities, split settings, dataset caching, and the `preprocessing:` block (out dir, workers, `label_convention: brats2021`, `target_axcodes: [L, P, S]`). `root_dir: ???` means "must be given" — a no-hardcoded-paths rule enforced by Hydra. |
| `splits.yaml` | **the frozen BraTS 2021 train/val/test case lists** (test = 189). Every number depends on it. |
| `splits_ssa.yaml` | BraTS-Africa external cohort; every case in `test`, train/val deliberately empty. |
| `splits_ped.yaml` | BraTS paediatric external cohort, same rule. |
| `splits_ssa_cf0.yaml`, `splits_ssa_cf1.yaml`, `splits_ped_cf0.yaml`, `splits_ped_cf1.yaml` | the two cross-fit folds per cohort for the D3 fine-tune (seed 42, `val_n=5`). |
| `splits_crossfit_summary.yaml` | summary of those folds, frozen before any D3 run. |
| `rsna_validation_cases.yaml` | the 40 sampled BraTS test-split patients used for the real-DICOM validation (P1.2), with the sampling recipe. |

### `model/`
| File | Model |
|---|---|
| `unet3d.yaml` | the plain 3D U-Net baseline. "Keep it honest and reasonably tuned rather than deliberately weak." |
| `unet3d_wide.yaml` | the same U-Net widened to `neurovision`'s parameter count — the capacity control. |
| `swinunetr.yaml` | SwinUNETR baseline. |
| `neurovision.yaml` | the full model: CNN channels `[32,64,128,256,320]`, Swin `feature_size 48`, 4 levels, fusion `adaptive_gated` (4 heads, window 4, full attention ≤ 512 tokens, scalar gate, `layer_scale_init 1e-4`, `use_ambiguity: true`), deconv decoder, head dropout 0.1, confidence + boundary heads on, `deep_supervision_levels: 3`. |
| `segqc.yaml` | the QC model's architecture. |

### `training/default.yaml`
AdamW lr 1e-4 / wd 1e-5, warm-up then cosine, AMP on, `batch_size: 1` (× 4 patches per
volume = 4 patches per step), gradient clipping, the `loss:` block (including the multitask
weights: boundary 0.3, confidence 0.05, branch 0.1 when enabled), training-time sliding window,
checkpoint settings (`keep_last_n: 2`, `monitor: val/dice_mean`, `resume`, `init_from`),
`max_hours`.

### `experiment/` — each file is one training run, selected with `+experiment=<name>`
| File | Purpose |
|---|---|
| `_baseline_common.yaml` | shared body of every comparison run: **64³ patches, 80 epochs**, warm-up 5, grad clip 5.0, val every 10 epochs, `max_hours: 10.5`. Identical data, augmentation, optimiser and schedule, so architecture is the only difference. |
| `baseline_unet3d.yaml` | baseline A, the matched U-Net. |
| `baseline_swinunetr.yaml` | baseline B, SwinUNETR-B (62.19M params). |
| `neurovision.yaml` | the full model; switches the loss to `multitask`, deep supervision on, branch supervision on. |
| `capacity_control_unet3d.yaml` | parameter-matched U-Net — splits the gain into architecture (~79%) vs capacity (~21%). Its checkpoint is lost. |
| `neurovision_heavy_aug.yaml` | D0: heavier augmentation to close the SSA/PED gap. Result: null. |
| `ablation_content_only_gate.yaml` | P2: the gate without the ambiguity signal. Result: matches the full model. |
| `_ablation_schedule.yaml` | shared 40-epoch schedule for the ablation grid (fits the GPU budget). |
| `_ablation_common.yaml` | shared architecture layer for the grid. |
| `ablation_full.yaml` | grid row 1, the reference. |
| `ablation_fusion_concat.yaml` | row 2, concat fusion. |
| `ablation_fusion_add.yaml` | row 3, additive fusion. |
| `ablation_cnn_only.yaml` | row 4, no transformer. |
| `ablation_transformer_only.yaml` | row 5, transformer only. |
| `ablation_no_deep_supervision.yaml` | row 6, deep supervision off. |
| `overfit2.yaml` | sanity check: overfit 2 cases. The first thing to run whenever data, model or loss changes. |

`.gitkeep` keeps the folder in git.

### Other groups
| File | Used by |
|---|---|
| `inference/default.yaml` | `evaluate.py` and the backend: sliding window (overlap 0.5, Gaussian), MC dropout (off, N=10), TTA (off), post-processing (threshold 0.5, nesting on, min component 50), evaluation outputs (`save_logits`, boundary bands, lesion-wise). |
| `calibration/default.yaml` | `calibrate.py`: fit/apply directories, bins, mask mode, temperature fitting. |
| `analysis/default.yaml` | every post-hoc analysis driver (conformal, local recalibration, Gate 1/2/C, OOD, error budget, ...). The largest config, ~1,100 lines. |
| `explainability/default.yaml` | `extract_gates.py`, `extract_ambiguity.py`, `explain.py`. |
| `anatomy/sri24.yaml` | atlas location, label merging, alignment-check thresholds; every value backed by `phase0_atlas_findings.md`. |
| `clinical/default.yaml` | the clinical front end (runs in `.venv-clinical`): `ingest`, `preprocess` (centre on T1CE, SRI24, HD-BET fast), `input_qc` thresholds, `dicom_seg` text, `gatekeeper` (`enabled_signals: [input_qc, predicted_dice, intended_use]`, regions WT and TC, caution quantile 0.10, refuse quantile 0.02, conformal α 0.10, adults ≥ 18), and the P1.2 `validation` block. |
| `hero/default.yaml` | `build_hero_brain_mesh.py`. |

## Part 7 — `knowledge/`, the versioned knowledge base

Plain YAML that the anatomy and reporting code loads and validates against the atlas. Each
file is a compilation from published sources with its own caveats, because there is no
neuroanatomy reviewer on the project.

| File | Content |
|---|---|
| `aal_lobes.yaml` | atlas structure → lobe. "Our own compilation, not an atlas product." |
| `eloquence_map.yaml` | structure → Sawaya eloquence class (a statement about a published list, not about a patient). |
| `involvement_groups.yaml` | which structures and tissue classes make up "ventricles", "deep white matter" etc. |
| `molecular_markers.yaml` | marker vocabulary, what each means in WHO CNS5, the entered-pathology → CNS5 name lookup, and the copy shown to users. |

Also at the root: **`pyproject.toml`** (package metadata so `neurovision` is importable from
`src/`, pytest settings including `addopts = "-q"`, Ruff/Black settings) and the
requirements files (`requirements.txt` fixed stack; `requirements-analysis.txt` adds
`panoptica`; `requirements-clinical.txt` adds `brainles-preprocessing`, `antspyx`, `HD-BET`,
`dcm2niix`, `highdicom`). **`notebooks/`**: `kaggle_train.ipynb` / `kaggle_evaluate.ipynb`
(thin Kaggle drivers — no logic in cells), `01_verify_preprocessing.ipynb`,
`02_baseline_analysis.ipynb`, `09_paper_figures.ipynb`.

---

## Part 8 — `tests/`, every file

Rules (from `CLAUDE.md`): every model component has a CPU shape test on tiny tensors under a
second; losses and metrics are checked against hand-computed values (perfect prediction →
Dice 1.0); data tests use synthetic volumes, never real BraTS; the suite runs on the Mac CPU in
about a minute. Run with plain `pytest`.

### Tests worth naming in a presentation

| Test | What it protects against |
|---|---|
| `test_input_qc.py::test_no_function_in_this_module_takes_a_label` | a deployment gate that secretly uses ground truth (trap 1). Inspects every function signature. |
| `test_gatekeeper.py::test_calibrate_thresholds_is_label_free` | same rule for the gate's threshold fitting. |
| `test_qc_pairs.py::test_degrade_mask_takes_no_label_argument` | the QC model learning from damaged ground truth. |
| `test_adaptive_fusion.py::test_branch_ambiguity_entropy_is_finite_under_fp16_saturation` | the fp16 NaN bug that wasted 10.5 GPU-hours (trap 2). |
| `test_resume.py::test_resume_reproduces_exact_weights_epoch_and_step`, `test_resume_runs_exactly_one_more_epoch_not_zero_or_two`, `test_resume_restores_adam_moment_buffers_exactly` | a killed GPU session resuming wrongly. |
| `test_mc_dropout.py::test_n5_stochastic_model_gives_positive_mutual_information` | MC dropout silently running deterministic passes (trap 5). The only test that would catch it. |
| `test_dicom_ingest.py::test_t1ce_is_not_classified_as_t1` | the substring trap (`"_t1"` in `"_t1ce"`). |
| `test_nnunet_import.py::test_mirrored_header_does_not_silently_pass` | a left-right flipped prediction scoring as fine. |
| `test_transforms.py::test_convert_to_regions_nesting_wt_ge_tc_ge_et` | broken ET ⊂ TC ⊂ WT nesting. |
| `test_report.py::test_forbidden_substrings_only_appear_inside_not_claimed_*` | the report using clinical-judgement words ("grade", "prognosis", ...) outside the "not claimed" block. |

### Full list, grouped by what they cover

**Shared helper:** `script_loader.py` (imports a `scripts/` file as a module for testing);
`__init__.py` (empty).

**utils, data:** `test_utils.py` (seed, device, logging, io — 34 tests) ·
`test_brats_reader.py` (19) · `test_preprocessing.py` (40) · `test_preprocess_script.py` (11) ·
`test_transforms.py` (21) · `test_dataset.py` (23) · `test_make_splits_script.py` (8) ·
`test_make_crossfit_splits_script.py` (17) · `test_dicom_ingest.py` (28) ·
`test_clinical_preprocess.py` (20) · `test_clinical_resample.py` (6) · `test_qc_pairs.py` (18).

**models:** `test_models.py` (registry, unet3d/swinunetr builders — 10) ·
`test_cnn_encoder.py` (18) · `test_swin_encoder.py` (13) · `test_adaptive_fusion.py` (61) ·
`test_decoder.py` (20) · `test_auxiliary_heads.py` (11) · `test_neurovision.py` (the full
network, return-type switch, config guards — 52) · `test_qc_model.py` (9).

**losses, training:** `test_losses.py` (registry, Dice+BCE, deep supervision — 18) ·
`test_multitask_loss.py` (26) · `test_trainer.py` (12) · `test_checkpoint.py` (atomic save, safe
load, pruning, RNG round trip — 22) · `test_resume.py` (8) · `test_train_script.py` (7) ·
`test_train_init_from.py` (fine-tune entry point — 7) · `test_gpu_session.py` (37) ·
`test_cluster_scripts.py` (the shell wrappers — 25) · `test_run_ablation_grid.py` (20) ·
`test_package_script.py` (8).

**metrics, inference:** `test_metrics.py` (25) · `test_boundary_metrics.py` (25) ·
`test_lesionwise_metrics.py` (14, skipped without `panoptica`) · `test_sliding_window.py` (7) ·
`test_postprocess.py` (17) · `test_tta.py` (17) · `test_mc_dropout.py` (22) ·
`test_input_qc.py` (32) · `test_gatekeeper.py` (38) · `test_evaluate_script.py` (36) ·
`test_rebuild_predictions_script.py` (11) · `test_replay.py` (19) · `test_replay_logits.py` (20).

**uncertainty, calibration, conformal:** `test_calibration.py` (50) ·
`test_calibrate_script.py` (32) · `test_conformal.py` (14) · `test_conformal_script.py` (11) ·
`test_risk_coverage.py` (20) · `test_local_recalibration.py` (12) ·
`test_local_recalibration_script.py` (13) · `test_score_confidence.py` (13) ·
`test_confidence_diag.py` (14) · `test_mc_comparison.py` (7) · `test_equivalence.py` (TOST — 9).

**fusion-mechanism research:** `test_extract_gates.py` (15) · `test_extract_ambiguity.py` (25) ·
`test_detection.py` (16) · `test_detection_stats.py` (19) · `test_localisation.py` (19) ·
`test_gate2_localisation.py` (10) · `test_ambiguity_intervention.py` (6).

**QC model, gate, error budget, OOD:** `test_qc_inference.py` (10) · `test_train_qc.py` (24) ·
`test_qc_validate.py` (17) · `test_validate_qc_script.py` (8) ·
`test_gatekeeper_calibration.py` (11) · `test_calibrate_gatekeeper.py` (4) ·
`test_error_budget.py` (14) · `test_error_budget_script.py` (23) · `test_ood.py` (21) ·
`test_ood_score_script.py` (8) · `test_silent_failure_by_region.py` (10).

**statistics:** `test_statistics.py` (39) · `test_stratify.py` (17) · `test_compare_family.py` (7) ·
`test_ref_et_comparison.py` (12) · `test_label_composition.py` (8).

**explainability:** `test_gradcam.py` (26) · `test_integrated_gradients.py` (25) ·
`test_attention_rollout.py` (22) · `test_faithfulness.py` (27) · `test_explain_script.py` (11).

**anatomy, reporting:** `test_atlas.py` (24) · `test_alignment.py` (20) ·
`test_validate_atlas.py` (18) · `test_fetch_atlas.py` (14) · `test_atlas_export.py` (10) ·
`test_burden.py` (24) · `test_burden_script.py` (12) · `test_localize.py` (32) ·
`test_localize_script.py` (28) · `test_involvement.py` (21) · `test_shape_descriptors.py` (11) ·
`test_population.py` (16) · `test_population_stats_script.py` (12) · `test_report.py` (69) ·
`test_report_script.py` (16) · `test_report_agreement.py` (27) ·
`test_report_agreement_script.py` (14) · `test_molecular.py` (29) · `test_dicom_seg.py` (19) ·
`test_dicom_frames.py` (8).

**nnU-Net, real DICOM, clinical:** `test_export_nnunet_dataset.py` (10) ·
`test_nnunet_session.py` (36) · `test_nnunet_predict_session.py` (13) · `test_nnunet_import.py`
(11) · `test_score_nnunet_script.py` (7) · `test_real_dicom_scoring.py` (18) ·
`test_validate_real_dicom_script.py` (30) · `test_real_dicom_offsets.py` (12) ·
`test_run_clinical_study_script.py` (15).

**backend:** `test_app_api.py` (research routes — 36) · `test_app_clinical_api.py` (clinical
routes — 70) · `test_app_clinical_jobs.py` (the job pipeline, refusal vs failure, rehydration —
59) · `test_app_inference.py` (22).

**figures, tables, assets:** `test_visualization_qc.py` (14) · `test_figures.py` (102) ·
`test_tables.py` (62) · `test_plot_semester_figures.py` (11) · `test_build_hero_brain_mesh.py` (8).

One lesson the project learned the hard way (`CLAUDE.md`): **an analysis fix is not verified by
its unit tests.** A commit once claimed a circular-mask bug was fixed with 1,000 green tests
while every reported number stayed circular. Re-run the real analysis and check the number
moved.

---

## Part 9 — Questions you are likely to be asked

**What is new in the architecture?** The fusion gate conditions on *where the CNN and Swin
branches disagree* (per-voxel disagreement + each branch's entropy, from detached linear
probes), not only on branch content. It is gated windowed cross-attention:
`fused = cnn + layer_scale · gate · attention(cnn → swin)`.

**Did the novelty work?** Honestly: the full model beats a matched U-Net on ET Dice by +0.0267
(p_holm 1.4e-21, n=189), ~79% from architecture. But the content-only gate (no ambiguity
signal) matches the full model, so the gain comes from gated fusion itself, not the ambiguity
conditioning. The founding hypothesis was a pre-registered null. And it has not yet been
compared with nnU-Net (Gate A is running).

**Why three sigmoid outputs and not softmax?** The regions overlap (ET ⊂ TC ⊂ WT). Softmax
would make them compete.

**Why train on patches?** A whole 3D brain through this network does not fit in 16 GB with
gradients. Patches train; a Gaussian-blended sliding window predicts whole volumes.

**What does the conformal bound guarantee?** That on new cases exchangeable with the
calibration set, the conservative mask `{p ≥ τ̂}` misses on average at most α (10%) of the true
tumour. It holds on the BraTS test set and breaks on SSA and PED in proportion to the shift —
the thesis. `local_recalibration.py` measures how many local labelled cases restore it.

**What does the gate do?** Combines label-free signals — input QC, the QC model's predicted
Dice, patient age (adults only) — and returns the worst of PROCEED / CAUTION / REFUSE.
Thresholds come from val-split quantiles, never from ground truth. Conformal band width and OOD
score are computed and displayed but do not drive decisions, because their measured results
did not justify it.

**How well does the whole pipeline work?** Usable mask for 85% / 78% / 24% of test / SSA / PED
studies; silent failures 4.2% / 18.3% / 49.5%. The gate cannot see cohort-level shift
(paediatric).

**Why is "refused" not an error?** Because declining a study the pipeline cannot safely handle
is the designed behaviour, with a named reason. `failed` is reserved for real faults.

**How do you know results are not fooling you?** Pre-registered tests, Holm correction, paired
tests, label-free deployment code enforced by signature tests, a measured seed noise floor
(+0.0021) under the +0.0267 effect, re-scoring from saved logits that reproduces published rows
exactly, and a documented list of traps that already cost real time.

**Why the GroupNorm / fp16 / atomic-save details?** Each one answers a measured failure:
correlated small batches (GroupNorm), a NaN run that cost 10.5 GPU-hours (softplus entropy),
GPU sessions killed mid-write (atomic checkpoints and full resume).

**Is it for clinical use?** No. Research and course project; every report, DICOM-SEG and
API response says so.
