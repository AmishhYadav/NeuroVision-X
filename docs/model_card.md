# Model card — NeuroVision-X (`neurovision`, seed 42)

**Status:** research artifact. **Not a medical device. Not for diagnosis.**
**Last updated:** 2026-09-19 · **Authoritative numbers:** `docs/paper/claims_and_evidence.md`
(a claim not in that table does not belong here either) · **Runs:** `docs/experiments.md`

---

## 1. What this is

A 3D brain-tumour sub-region segmentation model, and the pipeline wrapped around it. The model is one
component; the deliverable is the pipeline, because the model alone has no way to say when it should
not be trusted.

| | |
|---|---|
| Task | Segment enhancing tumour (ET), tumour core (TC) and whole tumour (WT) from four co-registered MRI sequences (T1, T1CE, T2, FLAIR) |
| Architecture | Dual encoder — 3D CNN + Swin Transformer — into adaptive gated cross-attention fusion at four scales, a U-Net decoder, and three heads (segmentation, confidence, boundary) |
| Parameters | 34,911,341 |
| Output | Three overlapping binary region masks, plus per-voxel probabilities |
| Deployed checkpoint | `neurovision`, seed 42, epoch 69 of 80, selected on `val/dice_mean` |
| Training cost | 23.1 GPU-h on a Kaggle T4, three chained sessions |
| Inference | Sliding window 64³, overlap 0.5, Gaussian blending. ~4 s/case on a T4; measured ~1.4 min/case on an M4 CPU at `sw_batch_size=1` |

## 2. Intended use

**In scope.** Research on tumour segmentation and on uncertainty/refusal behaviour; education and
demonstration; a starting point for method work that cites its measured limits.

**Out of scope, explicitly.** Diagnosis, treatment planning, triage, or any use where a clinician or
patient acts on the output. Every cleared product in this space is clinician/PACS-facing and
regulated; this is neither. Patient-facing framing was rejected as a design decision, and the
rejection is enforced in the UI copy, not only in documentation.

**Populations it was not built for.** Paediatric tumours and sub-Saharan African cohorts are both
*measured* here and both are out-of-distribution for this checkpoint — see §5. Paediatric tumour core
in particular fails so completely (voxel Dice 0.4394, lesion-wise 0.234) that the model has no usable
signal there.

## 3. Training data

BraTS 2021 training set, 1,251 cases, split randomly and frozen: **875 train / 187 val / 189 test**
(`configs/data/splits.yaml`). Nothing outside `train` was ever fit on — not the model, not the
calibration temperature, not the conformal threshold, not the refusal-gate thresholds.

Preprocessing: reorient to LPS from each file's own affine, 1 mm isotropic, nonzero z-score
normalisation, crop to the nonzero bounding box. Raw data was deleted after preprocessing; SHA-256
manifests are in `docs/data_manifests/`.

Recipe: 64³ patches, 80 epochs, AdamW 1e-4 (weight decay 1e-5, cosine schedule, 5 warm-up epochs),
batch size 1, AMP on, multitask loss (Dice+CE with deep supervision, plus boundary 0.3, confidence
0.05, branch-consistency 0.1). Augmentation: flips, 90° rotation, ±10% intensity scale/shift, light
Gaussian noise.

**The split is a random split of the BraTS *training* set.** Numbers here are therefore **not
comparable to any published BraTS leaderboard number**, which is scored on the official validation
set.

## 4. In-distribution performance (BraTS test, n=189)

| Metric | `neurovision` | Matched U-Net baseline |
|---|---|---|
| Dice ET | **0.8709** | 0.8442 |
| Dice TC | 0.9161 | 0.9058 |
| Dice WT | 0.9321 | 0.9276 |
| HD95 ET (mm) | 4.20 | 4.91 |

The ET margin is +0.0267 (CI [0.0166, 0.0393], p_holm 1.4e-21, n=189 paired). A width-matched capacity
control attributes ~79% of it to architecture and ~21% to parameter count. Within the architecture,
the gain comes from gated cross-attention fusion; conditioning the gate on inter-branch disagreement —
the project's founding hypothesis — contributes nothing measurable (+0.0022, CI −0.0067 to +0.0152).

**Lesion-wise, the metric BraTS has used since 2023, the same masks score much lower:**
ET **0.755**, TC **0.815**, WT **0.7183**. The WT figure is the one to read: voxel-wise 0.9321 is
dominated by one large component and overstates the result by 0.21 Dice. Report both conventions; the
lesion-wise one is the comparable one.

**No claim is made on WT.** Every WT comparison is inconclusive, voxel-wise and lesion-wise.

## 5. Out-of-distribution performance

Two external cohorts, entirely held out, nothing fit on either.

| Cohort | Dice ET | Dice TC | Dice WT |
|---|---|---|---|
| BraTS test (in distribution) | 0.8709 | 0.9161 | 0.9321 |
| BraTS-Africa (SSA, n=60) | 0.7784 | 0.7846 | 0.8959 |
| BraTS-PED (n=99) | 0.5634 | 0.4394 | 0.8490 |

**The accuracy gain does not transfer.** The ET advantage disappears under shift, and tumour core is
*significantly worse* than the plain U-Net on pooled shifted data (−0.0333, p_holm 0.0132). This is a
pre-registered negative result, not an oversight.

## 6. Uncertainty, and the error bound

**Use single-pass predictive entropy.** It is the strongest per-voxel error localiser in this project
and it is free. Three alternatives were measured against it and none beat it: 10-sample MC-dropout is
statistically equivalent (paired TOST, margin 0.03 AUROC), inter-branch disagreement is worse, and the
model's own trained confidence head is worse on all three regions (on WT it is at chance).

**Conformal risk control** (`scripts/conformal.py`) calibrates a decision threshold on val and applies
it frozen, giving a distribution-free bound on the mask's miss rate.

- **In distribution it holds**: realised risk lands at 0.64×–0.96× of nominal α in 6 of 6 cells, for
  this model and for the baseline. It is a theorem, so it holds for an arbitrarily bad model too.
- **Under shift it breaks, and the size of the break tracks the size of the shift**: SSA WT ~1.1×
  nominal, PED WT 1.4–1.9×, PED TC **3.5×–11.5×**. 7 of 12 cells violated.

Anyone deploying this must read the second bullet as the operating reality, not the first.

## 7. The clinical pipeline, and what the refusal gate actually does

A real DICOM study goes through: ingest → input QC → co-registration + atlas registration +
skull-stripping → input QC again → segmentation (this checkpoint, pinned) → QC-model estimate and
conformal band → refusal gate → **PROCEED / PROCEED_WITH_CAUTION / REFUSE**. A refusal is a successful
outcome of the job, never an error.

Measured end to end with the deployed frozen gate, nothing refit (note 47):

| | BraTS test | SSA | PED |
|---|---|---|---|
| Usable mask returned (WT and TC Dice ≥ 0.7) | **85.2%** [79.9, 90.0] | **78.3%** [66.7, 88.3] | **24.2%** [16.2, 33.3] |
| Silent failure (accepted, but unusable) | 4.2% | 18.3% | 49.5% |

**Do not read the gate as a safety property.** It has decent precision and poor recall, and it is
blind to cohort-level shift: in distribution it refuses 16 usable studies to catch 4 unusable ones; on
SSA it is close to a no-op (2 refusals in 60, 11 of 12 unusable masks passed); on PED its refusals are
mostly correct (precision 0.88) but it catches only 32% of unusable masks, and no operating point on
its coverage curve gets P(usable | accepted) above 0.5. Gate acceptance also does not repair the
conformal guarantee where it is broken (PED TC realised risk 0.651 overall, 0.573 on accepted cases,
at nominal 0.10).

The QC model behind one of those signals beats free entropy on exactly one of five pre-registered
cells (PED·TC, ΔAUROC +0.1686) and loses to it significantly on two others (SSA·TC, PED·WT). Its bias
also turns **more optimistic** under shift in all six external cells — it grows more confident exactly
where the segmentation is least trustworthy.

## 8. Known limitations

1. **Single seed.** No seed-to-seed noise floor exists yet, so no margin below ~0.005 Dice is
   claimable, and every null is "no detectable difference at n=189", not equivalence. A second seed is
   in progress (`preregistration_multiseed.md`).
2. **No strong baseline.** nnU-Net v2 / Auto3DSeg on our exact split was priced at 75.5 GPU-h per fold
   against a 60 GPU-h pre-registered abort bound, and was not run. The comparison the field would
   apply is therefore missing, and the architecture result must be read against a *matched* U-Net, not
   against a properly configured one.
3. **No transformer baseline** on our splits (cut for budget).
4. **Metric convention.** Voxel-wise Dice overstates performance by 0.10–0.32 versus lesion-wise here.
5. **Paediatric tumour core is a failure case**, identically for both models (64 of 99 cases are exact
   ties at lesion-wise Dice 0.234). There is nothing to claim on it in either direction.
6. **The anatomical eloquence layer is degenerate** on this cohort (100% "near eloquent"), so it
   carries no per-case information.
7. **The report's molecular panel is not an AI prediction.** It maps clinician-entered pathology to a
   CNS5 name; the AI slot reads "not available — model not trained".
8. **DICOM-SEG export can legitimately refuse.** It validates geometry against the source series and
   does not resample from atlas space, so a real post-registration case can come back "export
   refused". That is a reported outcome, not a crash.

## 9. Provenance and reproduction

| Artifact | Where |
|---|---|
| Deployed checkpoint | `outputs/neurovision/checkpoints/best.pt` |
| Saved fp16 logits (val / test / SSA / PED) | the table in `CLAUDE.md`; every re-scoring runs off these with zero inference |
| Exact rebuild commands | `docs/reproducibility.md` (§5 for the clinical demo) |
| Every measured result | `docs/experiments.md`, notes 1–47 |
| What may be asserted | `docs/paper/claims_and_evidence.md` |
| Traps that already cost real money | `docs/lessons.md` |

Only three checkpoints survive: `neurovision`, `baseline_unet3d` and `ablation_content_only_gate`.
`capacity_control` is permanently lost and no capacity-control number can be re-scored under a new
metric.
