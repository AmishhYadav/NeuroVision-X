# NeuroVision-X — Demo-Day Study Guide

Written 2026-10-16, for the project demonstration. This is the **"how it all fits together"**
guide. It does not repeat the file-by-file reference — that is `docs/backend_code_guide.md`
(every `src/`, `scripts/`, `app/backend/`, config and test file, with its functions). Read this one
first; open that one when you need the detail of one file.

Every number below is copied from `docs/research_docs/claims_and_evidence.md`,
`docs/research_docs/experiments.md`, or a result file on disk. If a number is not in this guide or
in `claims_and_evidence.md`, do not say it aloud.

**Reading plan if you have 60 minutes:** Part 0 (10 min) → Part 1 (15 min) → Part 3 (10 min) →
Part 7 (10 min) → Part 8 (15 min). Everything else is for when someone asks a deeper question.

---

## Part 0 — The project in ten minutes

### 0.1 One sentence

A 3D brain-tumour segmentation model, wrapped in (a) a mathematical error bound and (b) a gate that
can refuse a study — and a measurement of **where the bound and the gate break** when the data
comes from a different population.

### 0.2 The thesis (say it this way)

> A tumour segmentation model wrapped in a distribution-free error bound and a refusal gate: the
> bound holds in distribution, fails under distribution shift in proportion to the shift, and the
> gate cannot see cohort-level shift. We measure where it breaks, and what a new site needs in order
> to restore it.

What it is **not**: not "state of the art" (nnU-Net beats it), not "safe to deploy", not clinical.

### 0.3 The whole system on one screen

```
┌─────────────────────────── RESEARCH PIPELINE (offline) ───────────────────────────┐
│                                                                                   │
│  BraTS 2021 NIfTI (1,251 patients × 4 MRI scans + label)                          │
│        │  scripts/preprocess.py            (Mac CPU, once)                         │
│        ▼                                                                          │
│  data/preprocessed/brats/<case>/{image.npy, label.npy, meta.json}                 │
│        │  configs/data/splits.yaml         (frozen: 875 train / 187 val / 189 test)│
│        ▼                                                                          │
│  scripts/train.py  via notebooks/kaggle_train.ipynb   (Kaggle T4 GPU, ~23 GPU-h)  │
│        ▼                                                                          │
│  outputs/neurovision/checkpoints/best.pt   (epoch 69, val Dice 0.8938)            │
│        │  scripts/evaluate.py              (Mac CPU, ~1.4 min/case)               │
│        ▼                                                                          │
│  per_case_metrics.csv · summary.csv · logits/*.npy (fp16)                         │
│        │  ~40 analysis scripts read the SAVED LOGITS, never re-run the model:     │
│        ▼                                                                          │
│  calibration · conformal bound · QC model · OOD score · gate thresholds ·         │
│  error budget · statistics · figures · paper tables                               │
└───────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── CLINICAL PIPELINE (live demo) ─────────────────────────┐
│                                                                                   │
│  Browser /clinical  ──upload DICOM zip──▶  FastAPI (app/backend/api.py)           │
│                                              │ background thread                  │
│                                              ▼ app/backend/clinical_jobs.py        │
│  ingest → input QC → register to SRI24 + skull-strip → input QC → preprocess →    │
│  segment (neurovision) → GATEKEEPER → Grad-CAM → DICOM-SEG → report               │
│                                              │                                    │
│  Browser polls job status, then shows slices, 3D twin, overlays, gate panel,      │
│  report.  Outcome: PROCEED / PROCEED_WITH_CAUTION / REFUSE                        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 0.4 Numbers to know by heart

| What | Value |
|---|---|
| Test-set Dice, `neurovision` (n = 189) | ET **0.871** · TC **0.916** · WT **0.932** |
| Same, matched `baseline_unet3d` | ET 0.844 · TC 0.906 · WT 0.928 |
| ET gain over the matched U-Net | **+0.0267**, CI [0.0166, 0.0393], p_holm 1.4e-21 |
| Replicated at a second seed pair | +0.0247 (seed 43) |
| Seed noise floor (same model, two seeds) | +0.0021 |
| nnU-Net (the field's standard) ET Dice | **0.8879** — beats us by 0.0170 (Gate A RETIRED) |
| Model size | 34.9 M parameters |
| Training cost | 80 epochs, 64³ patches, 23.1 GPU-h on a Kaggle T4, 3 sessions |
| Usable mask returned (test / Africa / paediatric) | 85% / 78% / 24% |
| Silent failure (accepted but unusable) | 4.2% / 18.3% / 49.5% |
| Conformal bound, in distribution | holds: realised risk 0.64–0.96 × nominal, 6/6 cells |
| Conformal bound, paediatric tumour core | 3.5–11.5 × nominal — broken |
| Local cases a new site needs to restore the bound | ~10–40 (except paediatric TC) |
| Real hospital DICOM | 40% refused for thick slices; WT −0.23 Dice on the rest |
| Live clinical job on the laptop | ~6 min (361 s), mostly registration + skull-strip |
| Tests | 2,641 passing, 36 skipped, ~71 s · frontend 362 vitest · 79/79 E2E |

**Cohorts.** *Test* = 189 held-out BraTS 2021 patients. *SSA* = BraTS-Africa, 60 patients (different
scanners and population). *PED* = BraTS paediatric, 99 children. SSA and PED are **never
trained on** — they are the "distribution shift".

---

## Part 1 — The six workflows, with the exact call chain

Each workflow is "what you type" → "which functions run, in order" → "what lands on disk".
`→` below means "calls".

### Workflow 1 — Data preprocessing (once, Mac CPU)

**Why.** Raw BraTS is NIfTI files, 240×240×155 voxels, four scans per patient, mostly black
background. The model wants one array per patient, same orientation, normalised intensities,
cropped to the brain.

**Command.**
```bash
python scripts/preprocess.py data.root_dir=/path/to/BraTS2021
python scripts/make_splits.py            # only ever run once; the split is frozen
```

**Call chain.**
```
scripts/preprocess.py
  → data/brats.py::scan_brats_root(root)          finds each patient's 4 scans + label
  → for each case (in parallel, skips finished ones):
      data/preprocessing.py::preprocess_case
        → load_case_arrays          nibabel reads t1, t1ce, t2, flair, seg
        → reorient_to_axcodes       every volume to the same axis order [L, P, S]
        → normalize_nonzero         z-score each scan using ONLY brain voxels
        → compute_nonzero_bbox      tight box around the brain
        → crop_to_bbox              cut away the empty background
        → remap_labels              BraTS labels {0,1,2,4} → {0,1,2,3}
        → writes image.npy (4,D,H,W) float16, label.npy, meta.json
scripts/make_splits.py
  → data/dataset.py::make_splits  seed 42, 70/15/15 → configs/data/splits.yaml
```

**What to know.**
- `meta.json` stores the crop box and the original shape. That is what lets every later step
  **un-crop** a prediction back to 240×240×155. Forgetting this misaligns overlays by the crop
  offset and *still looks plausible* — the geometry trap, handled in three places.
- Labels: 1 = necrotic core, 2 = oedema, 3 = enhancing (was 4).
- The split is written once and committed. If more data arrives later, the reported test set can
  never be silently reshuffled.
- Verified by eye in `notebooks/01_verify_preprocessing.ipynb` (Part 4).

### Workflow 2 — Training (GPU only)

**Command (on any GPU host).**
```bash
python scripts/train.py +experiment=neurovision data.root_dir=...
```
On Kaggle the same function is called from `notebooks/kaggle_train.ipynb` (Part 4).

**Call chain.**
```
scripts/train.py::run_training(cfg)
  → utils/logging.py::setup_logging
  → utils/seed.py::set_seed(42)                  python, numpy, torch, MONAI all seeded
  → utils/device.py::get_device(cfg)             CUDA if present, else CPU (never MPS)
  → build_dataloaders(cfg)
       → data/dataset.py::load_splits           the frozen case lists
       → data/dataset.py::build_data_dicts      [{"image": path, "label": path, ...}]
       → data/transforms.py::build_train_transforms
            load → ConvertToRegionsd (label → ET/TC/WT channels)
                 → RandCropByPosNegLabeld (4 patches of 64³, half centred on tumour)
                 → random flips, 90° rotations, intensity scale/shift, noise
       → data/transforms.py::build_val_transforms  (no crop, no augmentation)
       → data/dataset.py::build_dataset         MONAI Dataset / CacheDataset
  → models/registry.py::build_model(cfg)        looks up "neurovision" → NeuroVisionX
  → losses/registry.py::build_loss(cfg)         looks up "multitask" → MultiTaskLoss
  → training/trainer.py::Trainer(...)           builds AdamW, warm-up+cosine, AMP GradScaler
  → apply_resume_or_init(cfg, trainer)          last.pt exists? → resume EXACTLY
  → init_wandb                                  reattaches to the same W&B run id on resume
  → Trainer.train()
       for epoch in range(start_epoch, 80):
         predict: would this epoch overrun max_hours (10.5 h)? → stop cleanly
         train_one_epoch:
           for each batch (1 volume × 4 patches):
             autocast (fp16) → model(images) → loss_fn(outputs, labels)
             scaler.scale(loss).backward()
             unscale → clip grad norm at 5.0 → scaler.step → scaler.update
         scheduler.step()
         every 10 epochs: validate()  → sliding-window on 187 val cases → val/dice_mean
         training/checkpoint.py::save_checkpoint → last.pt (+ best.pt if improved)
```

**What lands on disk.** `outputs/<experiment>/checkpoints/{last.pt, best.pt, epoch_NNNN.pt}`.

Part 3 explains every training choice in depth.

### Workflow 3 — Evaluation (Mac CPU)

**Command.**
```bash
python scripts/evaluate.py +experiment=neurovision inference.evaluation.split=test \
  inference.evaluation.checkpoint=outputs/neurovision/checkpoints/best.pt
```

**Call chain.**
```
scripts/evaluate.py::run_evaluation(cfg)
  → build_eval_dataloader(cfg, "test")          189 whole volumes, no augmentation
  → load_eval_model                             refuses if checkpoint architecture ≠ config
  → for each case: evaluate_case
      → inference/sliding_window.py::sliding_window_predict
            tile the whole brain with 64³ windows, 50% overlap,
            Gaussian-blend the overlaps → raw logits (3, D, H, W)
      → inference/postprocess.py::postprocess_logits
            sigmoid → threshold 0.5 per channel → enforce_nesting (ET ⊆ TC ⊆ WT)
            → remove components < 50 voxels → back to class map {0,1,2,3}
      → metrics/segmentation.py::compute_case_metrics   Dice, IoU, HD95 per region
      → (optional) metrics/boundary.py, metrics/lesionwise.py
      → save fp16 logits, uncropped prediction
  → MetricAggregator → per_case_metrics.csv, summary.csv
```

**Why logits are saved.** Almost every analysis afterwards (threshold changes, calibration,
conformal, QC model, gate, error budget) needs only the logits. Saving them once means
**zero re-inference** for the whole analysis phase. ~31 GB of them are on disk.

### Workflow 4 — The safety layer (analysis on saved logits, Mac CPU)

These run in a fixed order because each one's output is the next one's input.

```
1. scripts/calibrate.py
     fit temperature T on VAL logits, apply to TEST, report ECE/Brier
     (uncertainty/calibration.py)

2. scripts/conformal.py                         ← THE THESIS
     extract: per case, per region, miss-rate curve over 66 thresholds → curves.npz
     fit:     on VAL, find threshold τ so mean miss rate ≤ α (0.05 / 0.10 / 0.20)
     apply:   frozen τ on TEST, SSA, PED → realised risk with bootstrap CIs
     (uncertainty/conformal.py::case_loss_curve, fit_threshold, realised_risk)

3. scripts/local_recalibration.py
     "what would a new site need?": re-fit τ on k cases of SSA/PED itself, test on the rest
     (analysis/local_recalibration.py)

4. scripts/train_qc.py                          (QC model, a second small network)
     damage the model's OWN predicted masks (erode, dilate, shift, drop a piece, speckle)
     → (damaged mask, true Dice) pairs → train SegQC to predict Dice from image+mask+entropy
     (data/qc_pairs.py, models/qc.py)
   scripts/validate_qc.py → Gate C verdict MIXED (wins 1 cell of 5)

5. scripts/ood_score.py
     46 label-free input statistics → Mahalanobis distance from the TRAIN distribution
     (analysis/ood.py) → flags 60% of SSA vs 9.5% of test → display-only

6. scripts/calibrate_gatekeeper.py
     run the QC model on every VAL case → its predicted Dice values →
     caution cut = 10th percentile, refuse cut = 2nd percentile
     → outputs/gatekeeper/thresholds.json        (never uses ground truth)

7. scripts/error_budget.py                      (Phase G, end to end)
     run the frozen gate on every case of every cohort →
     accepted × usable → correct accept / SILENT FAILURE / over-refusal / correct refusal
     → the 85/78/24% usable and 4.2/18.3/49.5% silent-failure numbers
```

### Workflow 5 — The live clinical pipeline (the demo)

**Command (two terminals; exact form in `docs/research_docs/reproducibility.md` §5).**
```bash
NVX_EXPERIMENT=neurovision NVX_EVAL_DIR=outputs/neurovision/eval_test \
NVX_CHECKPOINT=outputs/neurovision/checkpoints/best.pt \
NVX_REPORT_DIR=outputs/report_neurovision/reports NVX_JOB_DIR=outputs/clinical_jobs \
.venv-clinical/bin/uvicorn app.backend.main:app --port 8000

cd app/frontend && npm run serve     # production build, http://localhost:4173
```

**Call chain.**
```
Browser /clinical → ClinicalUploadPanel → POST /api/clinical/upload (zip)
  app/backend/api.py → clinical_jobs.py::create_clinical_job
       validate zip (size, no "../" zip-slip paths) → extract → state "queued"
  → start_clinical_job → daemon thread → run_clinical_job:

  stage                          function                                    can refuse?
  ───────────────────────────────────────────────────────────────────────────────────────
  ingest            data/dicom_ingest.py::ingest_study                         yes
                    read every series header → rule table decides which is
                    T1 / T1CE / T2 / FLAIR → dcm2niix converts to NIfTI
  input_qc (pre)    inference/input_qc.py::run_input_qc(stage="pre_registration") yes
                    all 4 present? same geometry? slice spacing OK? NaN? intensities?
  clinical_preproc  data/clinical_preprocess.py::preprocess_clinical_study       —
                    co-register 4 scans (centred on T1CE) → register to SRI24 atlas
                    → HD-BET skull strip.  Makes a hospital scan look like BraTS.
  input_qc (post)   run_input_qc with the brain mask (brain volume 600–2600 mL)  yes
  research_preproc  data/preprocessing.py::preprocess_case  (same as Workflow 1) yes
  segmenting        app/backend/inference.py::segment_case → saves logits        —
  gatekeeper        _live_predicted_dice (SegQC on WT, TC)                        yes
                    _live_conformal_band_width
                    inference/gatekeeper.py::run_gatekeeper → worst of signals
  explaining        explainability/gradcam.py (WT, TC)              never fails the job
  exporting         uncrop → data/clinical_resample.py → reporting/dicom_seg.py   never fails
  generating_report reporting/report.py + anatomy/*                               —
  done

Browser: hooks/useClinicalJob.ts polls GET /api/clinical/jobs/{id} until done/refused/failed
         → ClinicalStudyViewer fetches volume, mask, overlays → renders
```

**Three states that matter.**
- `done` — mask produced; the gate said PROCEED or PROCEED_WITH_CAUTION.
- `refused` — a label-free check said no, with a named reason. **This is a success**, not an error.
- `failed` — something actually broke (exception). Different colour, different meaning.

**The gate, precisely.** Enabled signals: `input_qc`, `predicted_dice` (WT and TC), `intended_use`
(adults only, from DICOM `PatientAge`). The conformal band width and OOD score are computed and
**shown** but do not decide, because their measured results did not justify it. The decision is the
**worst** verdict of any enabled signal. An enabled signal that cannot be measured → REFUSE.

Frozen cut points (`outputs/gatekeeper/thresholds.json`, fitted on 187 val cases):

| Signal | Caution below | Refuse below |
|---|---|---|
| predicted Dice WT | 0.789 | **0.707** |
| predicted Dice TC | 0.660 | 0.465 |

### Workflow 6 — The research viewer (precomputed results)

`/app` in the browser. No model runs. `GET /api/cases` lists test cases that have both a volume and
a prediction; picking one fetches four scans, predicted mask, ground truth and an entropy map as raw
bytes. `?showcase=5` picks five cases evenly spaced by tumour size. `/report/{case_id}` is the
printable plain-language report.

---

## Part 2 — The model, explained for a viva

### 2.1 Input and output

- **Input** `(B, 4, D, H, W)`: four MRI scans stacked as channels in fixed order T1, T1CE, T2, FLAIR.
- **Output** `(B, 3, D, H, W)`: three channels — **ET, TC, WT** — each with its *own* sigmoid.

Why three sigmoids and not a 4-way softmax: the regions are **nested** (ET ⊂ TC ⊂ WT). One
enhancing voxel is in all three at once. Softmax would force them to compete for one probability.

| Region | Labels | Meaning |
|---|---|---|
| ET enhancing tumour | 3 | lights up after contrast (T1CE) |
| TC tumour core | 1 + 3 | ET plus the necrotic centre |
| WT whole tumour | 1 + 2 + 3 | everything, including oedema (FLAIR) |

### 2.2 The architecture (`models/neurovision.py::NeuroVisionX`)

```
x (B,4,64,64,64)
 ├── CNNEncoder (models/encoders/cnn.py)       5 levels, channels 32,64,128,256,320
 │      c0 stride1, c1 stride2, c2 stride4, c3 stride8, c4 stride16
 │      residual blocks, GroupNorm, zero-init residual
 │
 └── SwinEncoder (models/encoders/swin.py)     MONAI SwinTransformer, 4 levels
        s0 stride2, s1 stride4, s2 stride8, s3 stride16
        windowed self-attention, feature_size 48

 fusion at each shared level (models/fusion/adaptive_fusion.py::AdaptiveGatedFusion):
    skips = [ c0,  fuse(c1,s0),  fuse(c2,s1),  fuse(c3,s2),  fuse(c4,s3) ]

 UNetDecoder (models/decoder/unet_decoder.py)  upsample + concat skip + residual conv
 MultiTaskHead (models/heads/multitask.py)
    segmentation head × 3 levels (deep supervision)
    confidence head   "is the segmentation right here?"
    boundary head     "is this voxel on the tumour surface?"
```

Why the CNN has 5 levels and Swin 4: Swin first merges 2×2×2 voxel blocks into tokens, so it cannot
produce a full-resolution feature. CNN level 0 goes to the decoder unfused.

### 2.3 The fusion block — the original idea

At one pyramid level:
```
swin_proj = GroupNorm(Conv1x1(swin_feat))               project Swin to the CNN's width
ambiguity = BranchAmbiguity(cnn_feat, swin_proj)        9 channels:
              tiny probe on each branch predicts ET/TC/WT →
              |p_cnn − p_swin| (3) + entropy of each probe (3 + 3)
gate      = GateGenerator(cnn_feat, swin_proj, ambiguity)   per-voxel value in (0, 1)
attn      = WindowedCrossAttention(query = CNN, key/value = Swin)
fused     = cnn_feat + layer_scale × gate × attn
```

In words: the CNN sees fine local texture; the transformer sees wider context. At every voxel the
gate decides how much transformer context to let in, and the gate is told **where the two branches
disagree**.

Three details worth being able to explain:
1. **`layer_scale` starts at 1e-4**, so at initialisation `fused ≈ cnn_feat`: training starts as a
   plain CNN U-Net and *learns* how much transformer to admit.
2. **The probes read detached features** (`.detach()`). Otherwise training would push the two
   encoders to agree and destroy the disagreement signal the gate reads.
3. **Entropy is computed from logits with softplus**, not from probabilities with `log`. The old
   version used a `1 − 1e-6` clamp; in fp16 that rounds to exactly 1.0 → `log(0)` → NaN. It trained
   on NaN for 10.5 GPU-hours with nothing raising. The test
   `test_branch_ambiguity_entropy_is_finite_under_fp16_saturation` now pins it.

**Did the idea work?** Honestly, no — and that is part of the story:
- The full model beats the matched U-Net on ET (+0.0267), ~80% from architecture, ~20% from extra
  parameters (width-matched capacity control, two draws).
- But the **content-only gate** (same block, ambiguity input removed) matches the full model
  (P2 null). So the gain is gated cross-attention fusion itself, not the disagreement signal.
- And **nnU-Net beats both** (Gate A RETIRED). The architecture claim is withdrawn.

### 2.4 The loss (`losses/multitask.py::MultiTaskLoss`)

```
total = 1.0  × segmentation  (Dice + BCE, deep supervision at 3 levels weighted 1 : 0.5 : 0.25)
      + 0.3  × boundary      (Dice + BCE vs a 1-voxel shell = dilate(label) − erode(label))
      + 0.05 × confidence    (BCE vs "was the seg head right at this voxel?", target under no_grad)
      + 0.1  × branch        (BCE of each fusion probe vs the label)
```
- **Dice** handles class imbalance (tumour is a few % of voxels); **BCE** gives smooth per-voxel
  gradients. Per-channel sigmoid, because regions overlap.
- MONAI's `DiceCELoss` is **forbidden** here: it applies softmax cross-entropy to overlapping
  channels and fails silently (trap 7).
- Confidence target under `no_grad`, else the model could "cheat" by making logits extreme.

The baselines (`unet3d`) use plain `dice_ce` (Dice + BCE), no deep supervision and no auxiliary heads.

---

## Part 3 — Model training in depth

### 3.1 The recipe (identical for every comparison run)

From `configs/experiment/_baseline_common.yaml` + `configs/training/default.yaml`:

| Setting | Value | Why |
|---|---|---|
| Patch size | **64³** (code default 96³) | 96³ cost 0.875 h/epoch → ~91 h per run, over budget. 64³: 0.272 h/epoch |
| Epochs | 80 | fits ~22 GPU-h on a T4 |
| Batch | 1 volume × 4 patches = 4 patches/step | half the patches centred on tumour (`pos_neg_ratio [1,1]`) |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-5 | |
| Schedule | 5-epoch linear warm-up, then cosine decay | |
| Precision | AMP (fp16 on GPU), off on CPU | speed + memory |
| Gradient clipping | norm 5.0 | at 1.0 it clipped 66–70% of steps (measured) — silently changed the learning rate |
| Validation | every 10 epochs, sliding window on 187 val cases | validation costs ~0.23 h each time |
| `max_hours` | 10.5 | Kaggle kills at 12 h; stop cleanly before |
| Normalisation | GroupNorm, never BatchNorm | 4 patches from one brain → batch statistics are noise |
| Seed | 42 (seed 43 for the noise-floor run) | |

"Matched recipe" means the U-Net baseline got exactly this — same data, augmentation, optimiser,
schedule. Only the architecture differs. That is what makes the +0.0267 comparison fair, and also
why nnU-Net (1000 epochs, bigger patches, heavier augmentation, ~59 GPU-h) beats both.

### 3.2 Where it ran

| Machine | Role |
|---|---|
| MacBook Pro M4 | all code, preprocessing, tests, evaluation, analysis, figures, the demo |
| Kaggle T4 GPU (16 GB, 14.56 GiB usable) | gradient descent only, ≤ 12 h sessions, ~28–30 GPU-h/week |

Rule: if it can run on a CPU, it does not belong in a GPU session. Only checkpoints and logs come
back from the GPU — logits are rebuilt locally.

### 3.3 The `neurovision` training run, as it actually happened

| Event | Detail |
|---|---|
| Probe at 96³ | 3.6 s/step, 0.875 h/epoch, peak VRAM 13.59 GiB (93% of the card) → re-plan to 64³ |
| Probe at 64³ | 1.12 s/step, 0.272 h/epoch, peak 6.17 GiB → gradient checkpointing switched off |
| Probe v5 | gradient norms logged → clip raised 1.0 → 5.0 |
| Session 1, attempt 1 | **wasted 10.5 GPU-h**: trained on NaN from ~epoch 10 (the fp16 entropy bug) |
| Run 2, sessions 1–3 | 80/80 epochs, **23.1 GPU-h**, `GIT_REF 7caacfa`, W&B `cc2l5j1c` |
| Best checkpoint | **epoch 69**, val `dice_mean` 0.8938 (session 2's output) |
| Final | epoch 79, `global_step` 70,000 = 80 × 875 (proof every epoch ran) |

A long run spans several Kaggle sessions. Each session resumes from the previous session's `last.pt`
and logs into the **same** W&B run (the run id lives inside the checkpoint).

### 3.4 Resume — why it is a hard constraint

A Kaggle session dies at 12 h; a shared cluster pre-empts. So `training/checkpoint.py` saves
**everything**: model, optimiser (including Adam's moment buffers), scheduler, AMP scaler, epoch,
global step, best metric, W&B run id, every RNG state (Python, NumPy, torch, CUDA), config.

- **Atomic writes**: write to a temp file, then `os.replace`. A kill mid-write never corrupts
  `last.pt`.
- **Safe loading**: `torch.load(weights_only=True)` — cannot execute code from the file.
- **Budget prediction**: before each epoch the trainer asks "elapsed + mean epoch time > 10.5 h?"
  and stops cleanly if so, because a session killed mid-epoch cannot save.
- Tests prove it: `test_resume_reproduces_exact_weights_epoch_and_step`,
  `test_resume_runs_exactly_one_more_epoch_not_zero_or_two`,
  `test_resume_restores_adam_moment_buffers_exactly`.

### 3.5 Every training run in the project

| Run | Config | Purpose | Outcome |
|---|---|---|---|
| `baseline_unet3d` | `+experiment=baseline_unet3d` | the matched comparison | ET 0.844 |
| `neurovision` | `+experiment=neurovision` | the proposed model, **deployed** | ET 0.871 |
| `capacity_control_unet3d` (×2) | widened U-Net, same params | architecture vs capacity | ~80% / ~20% (C2) |
| `ablation_content_only_gate` | `use_ambiguity: false` | does the disagreement signal matter? | **null** — it matches (P2) |
| `neurovision_heavy_aug` (D0) | + rotation, gamma, bias field, elastic | close the shift gap? | **null**, 12/12 inconclusive |
| `neurovision_seed43` (D1) | seed 43 | measure seed noise | +0.0021 noise floor |
| `baseline_unet3d_seed43` | seed 43 | replicate C1 | +0.0247, replicated |
| D3 fine-tunes (×4) | cross-fitted on half of SSA / PED | can local fine-tuning fix shift? | TC recovers (PED +0.37, mostly label definition) |
| TTA (flips) | inference only | does flip-averaging help? | ET +0.0007, inconclusive |
| nnU-Net (Gate A) | default 3d_fullres, 1000 epochs, ~59 GPU-h | the field's bar | **beats us**, ET 0.8879 |
| D2 pooled (running) | BraTS + part of SSA + PED, 945 train cases | does pooled training close the gap? | session 1 done (20/80 epochs, val 0.8642); s2 at the 10-17 reset |

### 3.6 Commands that guard a GPU session

- `python scripts/smoke_test.py` — runs the real pipeline on two synthetic cases on CPU. Before any
  GPU session.
- `+experiment=overfit2` — overfit 2 cases; the first thing to run when data/model/loss changes.
- `python scripts/show_config.py +experiment=neurovision` — print the composed Hydra config.
- `scripts/gpu_session.py` — preflight checks (right checkpoint epoch? right W&B id? pinned SHA?
  real GPU? finite weights?), each written against a loss already suffered
  (`docs/gpu_session_checklist.md`).

---

## Part 4 — The notebooks

All notebooks are **thin drivers**: no logic in cells. They call functions from `src/` and
`scripts/` so the logic is tested on the Mac.

### `notebooks/kaggle_train.ipynb` — training on Kaggle (16 cells)

| Cell | What it does | Why it is written that way |
|---|---|---|
| 1. Session config | `REPO_URL`, `GIT_REF` (a pinned SHA), `EXPERIMENT`, `OVERRIDES`, `WANDB_MODE`, `CKPT_SLUG`; sets `expandable_segments` before torch loads | the only cell you edit; a SHA means the run's source is always recoverable |
| 2. Code + deps | `git clone` then `git checkout <SHA>` with `subprocess(check=True)`; `sys.path.insert` | `git clone -b <sha>` fails; `!git` failures don't stop a cell; Kaggle is Python 3.12 so `pip install -e` refuses |
| 2b. GPU check | `assert torch.cuda.is_available()` **and** runs a real matmul | a P100 reports CUDA available but cannot run kernels (sm_60) |
| 3. W&B | online (Kaggle Secret) / offline (sync later) / disabled | a missing secret must fail at minute 1, not hour 11 |
| 4. Dataset | *discovers* the folder holding `preprocessed/` + `splits.yaml` under `/kaggle/input` | Kaggle mounts one level deeper than documented |
| 5. Resume | copies the previous session's `last.pt` (and `best.pt`) into `/kaggle/working/checkpoints` | `/kaggle/input` is read-only; missing it would silently restart at epoch 0 |
| 6. Train | `hydra.compose(...)` → `run_training(cfg)` in-process | same function as the CLI; first log line says `FRESH:` or `RESUME: ... epoch N` |
| 7. Verify | prints peak VRAM, requires `last.pt`, reports its epoch | the measured VRAM replaced three wrong estimates |

**To chain sessions:** commit the notebook ("Save & Run All"), then attach that version's output as
input to the next session and set `CKPT_SLUG`. Full guide: `docs/kaggle_workflow.md`.

### `notebooks/kaggle_evaluate.ipynb` — evaluation on Kaggle (12 cells)

Same shape: pinned SHA → install → GPU check → discover dataset and **exactly one** `best.pt` →
assert the checkpoint's epoch (69) and W&B id (`cc2l5j1c`) match what you meant to score →
`run_evaluation(cfg)` → print the summary table. The assertion exists because run 2 spanned three
sessions, each with a `last.pt`, and scoring the wrong one gives a complete, plausible table.

### `notebooks/01_verify_preprocessing.ipynb` — eyeball check

Loads 3 preprocessed cases plus their raw NIfTI. Shows a QC table (crop box reproduces shape?
labels only 0–3? mean ≈ 0, std ≈ 1?), mid-slices of all 4 scans with the label overlaid, and
intensity histograms before/after normalisation. Ends with a checklist (T1CE brighter than T1 at the
rim → channels not swapped; FLAIR bright where oedema is; brain not clipped).

### `notebooks/02_baseline_analysis.ipynb` — the U-Net baseline write-up

Training curves from the W&B API, the results table (Markdown + LaTeX), per-case Dice box plots and
cumulative distribution, a best/median/worst qualitative figure (with the prediction re-cropped by
`meta.json`'s box — the geometry trap), and a look at the worst case. Note: it reads
`outputs/eval_test`, which is the **old superseded U-Net**, not `neurovision`.

### `notebooks/09_paper_figures.ipynb` — every figure and table in the paper

18 sections, one per figure/table: case-set consistency check, training curves, main results,
statistical comparison (`compare_models`: bootstrap CI + Wilcoxon + Holm), per-case distributions,
calibration, risk–coverage, qualitative panel, ablation ladder, fusion gate maps, boundary error,
calibration report, explainability, population anatomy, report agreement, error budget, and a final
**audit** that lists every expected artifact as written or skipped-with-reason. Nothing is
recomputed; it only reads result files.

---

## Part 5 — The frontend (`app/frontend/`)

**Stack:** Vite + React 19 + TypeScript + Tailwind 4; `three` / `@react-three/fiber` for the 3D twin.
No router library: `main.tsx` reads `window.location.pathname`.

| URL | Page | What it shows |
|---|---|---|
| `/` | `pages/landing/Landing.tsx` | scroll-driven landing page: SRI24 cortex hologram, measured results ledger, architecture flow, pipeline story, "not claimed" list |
| `/app` | `App.tsx` | research viewer over precomputed test cases |
| `/clinical` | `pages/clinical/ClinicalPage.tsx` | live DICOM upload, job progress, study viewer, gate panel |
| `/report/{case}` | `pages/report/ReportPage.tsx` | printable plain-language report |

### How data reaches the screen

```
api.ts                       typed client; one function per backend route, no reshaping
  → binary routes return raw uint8 bytes + shape from the X-Volume-Shape header
hooks/useCaseData.ts         loads one research case (4 scans, masks, entropy, profile)
hooks/useClinicalJob.ts      polls a clinical job until done / refused / failed
hooks/useClinicalJobVolumes  loads a finished job's volumes; overlays are "supplementary"
                             (lib/supplementaryFetch.ts) so one failing overlay never blanks
                             the whole study
lib/slicing.ts               the ONLY place volume index maths lives: flat index d*H*W + h*W + w
lib/render.ts                composes one 2D slice: greyscale → class overlay →
                             ground-truth outline → entropy heat
workers/twinMesh.worker.ts   builds the 3D meshes off the main thread
  → lib/surfaceNets.ts       turns a voxel mask into a triangle mesh (naive Surface Nets)
  → lib/twinGeometry.ts      voxel axes → anatomical scene axes (left/right easy to get wrong)
  → lib/vertexScalars.ts     paints a mesh by entropy / conformal band / Grad-CAM, sampled one
                             voxel INWARD (on the surface, entropy is maximal by construction)
components/BrainTwinScene    the 3D "digital twin" of whichever case is selected
```

### The clinical page components

| Component | Shows |
|---|---|
| `ClinicalUploadPanel`, `PreviousStudies` | upload a zip; reopen earlier jobs (`?job=<id>` in the URL) |
| `PipelineStepper` | 8 steps mapped from backend stage strings (`lib/pipelineSteps.ts`) |
| `SeriesMatrix` | which DICOM series became T1 / T1CE / T2 / FLAIR |
| `InputQCPanel` | every input-QC finding, before and after registration |
| `GatekeeperPanel` | each signal's value, cut points, verdict; marks display-only signals |
| `RefusalBanner` | the named refusal reason, deliberately in neutral (amber, not red) styling |
| `ClinicalStudyViewer` | slices + 3D twin + overlays (entropy, conformal band, Grad-CAM, atlas) |
| `UncertaintyCard` | numbers for the active overlay (`lib/heatStats.ts`) |
| `MolecularPanel` | **entered** pathology (IDH, MGMT …) → WHO CNS5 name; nothing predicted from imaging |

**Wording rule** (`lib/reportInterpretation.ts`): every sentence is descriptive. Never grade,
prognosis, malignancy, "detects", "diagnoses". Tests scan the copy for forbidden words.

**Commands:** `npm run serve` (fast production build, use for the demo), `npm test` (362 vitest),
`npm run test:e2e` (79 rendered-pixel checks; backend must be running).

---

## Part 6 — Configuration and project patterns

- **Hydra.** `configs/config.yaml` picks one file per group (`data`, `model`, `training`,
  `inference` …). `+experiment=<name>` layers an experiment file on top. Anything can be overridden
  on the command line (`training.epochs=2`). `root_dir: ???` means "must be supplied" — enforces
  **no hardcoded paths**.
- **Registries.** `@register_model("neurovision")`, `@register_loss("multitask")`,
  `@register_fusion("adaptive_gated")` — chosen by a string in config. The decorator only runs if
  the module is imported in its package's `__init__.py`.
- **Pure core, thin driver.** The maths is in `src/neurovision/` (no file I/O, unit-tested); the
  `scripts/` file reads files, calls it, writes results. "Where is X computed?" → `src/`. "How is it
  run?" → `scripts/`.
- **Label-free by construction.** Anything that runs at deployment (input QC, gate, QC-model input)
  has *no parameter that could receive a ground-truth label*. Tests inspect function signatures.
  Reason: an earlier calibration mask that used the label manufactured 41–57% of a reported ECE.
- **Pre-registration.** Before any experiment, the test, endpoints and decision rule are committed
  to `docs/research_docs/preregistrations/`. The git timestamp proves the order.
- **Three virtual environments:** `.venv` (core + tests), `.venv-analysis` (+ `panoptica` for
  lesion-wise metrics), `.venv-clinical` (+ DICOM, ANTs, HD-BET, highdicom — runs the server).

---

## Part 7 — Where the science stands (2026-10-16)

| | Result | Claim |
|---|---|---|
| ✅ | ET Dice +0.0267 over a matched-recipe U-Net, replicated at a second seed | C1 |
| ✅ | ~80% of that is architecture, ~20% capacity (two draws) | C2 |
| ❌ | **nnU-Net beats it**: ET 0.8879 vs 0.8709; lesion-wise −0.0309 — architecture claim withdrawn | C29 |
| ❌ | Founding hypothesis (disagreement-conditioned gate) — pre-registered null | P2 |
| ✅ | Conformal bound holds in distribution (0.64–0.96 × nominal) | C13 |
| ✅ | …and breaks under shift **in proportion to the shift** (SSA WT ~1.1×, PED WT 1.4–1.9×, PED TC 3.5–11.5×) | C14 |
| ✅ | Gate acceptance does not repair it (PED TC 0.573 on accepted cases at nominal 0.10) | C22 |
| ✅ | A new site needs ~10–40 of its own labelled cases to restore the bound; PED TC never (frozen model) | C24 |
| ✅ | Cross-fitted fine-tune recovers TC (SSA +0.043, PED +0.368) — mostly learning the paediatric label definition | C27 |
| ✅ | End to end: usable 85 / 78 / 24%, silent failure 4.2 / 18.3 / 49.5% | C20 |
| ⚠️ | Gate: decent precision, poor recall, blind to cohort shift | C21 |
| ⚠️ | QC model vs free entropy: MIXED (1 of 5 cells) | Gate C |
| ⚠️ | OOD score sees the shifted cohort (60% of SSA flagged) but not the failed study → display-only | C26 |
| ⚠️ | Real DICOM: 40% thick-slice refusals; WT −0.23, mostly registration disagreement with BraTS | C25 |
| ❌ | Heavy augmentation (D0), flip TTA — null / inconclusive | C23, C28 |
| 🔄 | D2 pooled training — running (session 2 at the 10-17 reset) | — |
| 🔄 | Conformal bound on nnU-Net — does the shift failure replicate on the standard model? (queued) | — |

How to frame the negatives: **every one was pre-registered**, so nothing was dropped after seeing the
number. The project's value moved from "a better architecture" to "a measured account of where a
safety layer breaks".

---

## Part 8 — The demo, step by step

Full runbook: `docs/demo_runbook.md`. Short version:

**Before you start (once):** `pytest` green, `scripts/smoke_test.py` passes, backend + `npm run
serve` running, job `9c2cc294` and job `67b67b1d` open at `/clinical`. **Do not upload live** unless
you have to — one job takes ~6 min and ~12 GB of RAM. Have the screen recording ready as a fallback.

| # | Story | What to show | What to say |
|---|---|---|---|
| A | PROCEED (3 min) | job `9c2cc294`: stepper, series matrix, 3D twin, report (WT 176.0 mL), gate: predicted Dice WT 0.901 / TC 0.895 | "This patient was in our training set — this demonstrates the plumbing, not accuracy." |
| B | REFUSE on quality (2 min) | job `67b67b1d`: predicted Dice WT 0.626 < cut 0.707 → REFUSE, reason in banner | "A refusal is a successful outcome. Thresholds are validation-set quantiles, frozen before use, never from ground truth." |
| C | REFUSE on intended use (1 min) | slide: with `intended_use` on, PED accepted 0.798 → 0.000 | "A labelling rule, not detection. It refuses every child by construction." |
| D | Where the bound breaks (4 min) | C13 → C14 → C22 → C24 | "Holds in distribution, breaks in proportion to shift, the gate can't fix it, and a site needs 10–40 local cases — except paediatric tumour core." |
| E | Honest scoreboard (2 min) | C1 beside nnU-Net's 0.8879 | "One positive against a matched recipe; nnU-Net beats us; the contribution is the bound, the gate and the pipeline." |

**Never say:** "safe to deploy", "guaranteed per patient", "clinically validated",
"state of the art", "detects/diagnoses".

---

## Part 9 — Questions you will likely be asked

**What problem does this solve?** Models validated on clean data get used on other hospitals'
scans and fail silently. This measures how a safety layer behaves when that happens, and refuses
studies it cannot handle.

**Why 3D and not 2D slices?** Tumours are 3D structures; 3D convolutions see context across slices.
Cost: memory, hence patches.

**Why patches?** A whole brain through this network with gradients does not fit in a T4's memory.
Train on random 64³ patches (half centred on tumour so the model sees enough tumour); at inference,
tile the whole brain with a Gaussian-blended sliding window.

**Why CNN + transformer?** CNNs are good at local texture and edges; transformers model long-range
context. The fusion gate decides per voxel how much transformer context to use.

**Why GroupNorm?** The 4 patches in a batch come from one brain, so BatchNorm's batch statistics
would be noise and its train/eval switch hurts calibration.

**What is Dice? HD95?** Dice = 2|A∩B| / (|A|+|B|), overlap, 1 is perfect. HD95 = 95th-percentile
distance between the predicted and true surfaces in mm, lower is better.

**What is lesion-wise Dice?** BraTS 2023's metric: score each connected tumour piece separately, so
missing a small satellite lesion counts even though it barely moves voxel Dice.

**What does the conformal bound guarantee?** On new cases *exchangeable* with the calibration set,
the conservative mask `{p ≥ τ}` misses on average at most α (e.g. 10%) of the true tumour. Picked as
the largest τ with `(n·R̂(τ) + 1)/(n + 1) ≤ α` (Angelopoulos et al., Conformal Risk Control, ICLR 2024).
An **average over studies, not a per-patient promise**. It breaks when exchangeability breaks —
which is the thesis.

**Why does it break on children?** Paediatric tumours are labelled differently (reference is ~84%
necrotic/non-enhancing core, ~0% oedema), and the model trained on adults predicts a much smaller
core. No threshold fixes a definition mismatch; fine-tuning does (C27).

**How does the gate decide?** Worst of: input QC, the QC model's predicted Dice for WT and TC, and
patient age (under 18 → refuse; missing → caution). Cut points are the 10th / 2nd percentiles of
the validation set's predicted Dice.

**Why can't the gate see cohort shift?** Its signals are per-study. On PED most studies look
individually fine to the QC model while the whole cohort is mislabelled relative to training. The
OOD score sees the cohort shift (60% of SSA flagged) but does not pick out the failed studies.

**Is it better than nnU-Net?** No. nnU-Net v2, default recipe, same training cases: ET 0.8879 vs
0.8709, lesion-wise ET 0.786 vs 0.755, both significant. nnU-Net also beats the matched U-Net by
+0.044, so the weak link is our short recipe (80 epochs, 64³). nnU-Net had ~59 GPU-h vs our ~22.

**Then why not use nnU-Net?** That is the next experiment: the conformal analysis on nnU-Net's
probabilities is built and queued, to test whether the shift failure (C14) replicates on the
field's standard model. The deployed model stays `neurovision` because the QC model and gate
thresholds were calibrated against it.

**How do you know your results aren't fooling you?** Pre-registered tests, paired comparisons with
Holm correction, a measured seed noise floor (+0.0021) under the +0.0267 effect, label-free
deployment code enforced by tests, re-scoring from saved logits that reproduces published rows
exactly, and a documented list of traps (`docs/research_docs/lessons.md`).

**What was the hardest bug?** The fp16 NaN: an entropy clamp of `1 − 1e-6` is exactly 1.0 in fp16,
so `log(0)` → NaN, 10.5 GPU-hours trained on NaN with nothing raising. Fixed by computing entropy
from logits with softplus. Runner-up: a calibration mask that used the ground-truth label inflated
ECE by 41–57% behind a green test suite.

**What is the 3D twin?** A mesh of *this* patient's brain and tumour, built in the browser from the
same volumes the slice view shows (Surface Nets in a Web Worker), optionally coloured by entropy,
the conformal band or Grad-CAM.

**What is Grad-CAM showing?** Which regions of a chosen network layer most increased the model's
score for one region (WT or TC). It is an explanation of the model, not of the tumour.

**What is DICOM-SEG?** The hospital-standard file format for a segmentation, so the mask opens in
any PACS / OHIF viewer, aligned to the original slices. Writing refuses rather than silently
resampling if the grids don't match.

**Is the molecular panel AI?** No. A user enters lab results (IDH, 1p/19q, MGMT); a lookup table
maps them to a WHO CNS5 diagnosis name with its source. Nothing is predicted from imaging.

**How many tests? Why so many?** 2,641 Python + 362 frontend + 79 E2E. Every model component has a
CPU shape test; losses and metrics are checked against hand-computed values; the most important
tests each pin a bug that already cost real time.

**Is it for clinical use?** No. Course project, capstone and a journal paper (MELBA). Every report,
DICOM-SEG and API response says so.

---

## Part 10 — If you only remember ten things

1. Four MRI scans in, three **nested** regions out (ET ⊂ TC ⊂ WT), three sigmoids, no softmax.
2. Preprocess once → frozen split 875 / 187 / 189 → train on Kaggle → evaluate on the Mac.
3. Dual encoder (CNN + Swin) → gated cross-attention fusion → U-Net decoder → seg / confidence /
   boundary heads. 34.9 M params, 64³ patches, 80 epochs, 23.1 GPU-h.
4. +0.0267 ET over a matched U-Net is real and replicated; nnU-Net beats us; the founding idea was null.
5. Saved fp16 logits feed every analysis — zero re-inference.
6. Conformal risk control: holds in distribution, breaks in proportion to shift.
7. A new site needs ~10–40 labelled cases to restore it — paediatric tumour core never (frozen).
8. The gate: worst of input QC + predicted Dice + age; thresholds from val quantiles; label-free.
9. Usable 85 / 78 / 24%, silent failure 4.2 / 18.3 / 49.5% — the gate is blind to cohort shift.
10. "Refused" is a success. Never say "safe", "per patient", "state of the art", "diagnoses".
