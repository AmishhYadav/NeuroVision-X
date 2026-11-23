# Pre-registration — severity ladder: conformal risk and the gate under controlled acquisition change

**Written:** 2026-11-23, before any corrupted volume has been generated or scored. MasterPlan W3.2;
analysis-budget family 5 of 6 (§B3 rule 4). **External registration:** *(OSF / Zenodo ID to be
inserted here before the GPU run — MasterPlan W7.3. The probe may run before; the main run does not
start until this line holds an ID.)*

---

## Why this run exists

The thesis used to say the bound "fails in proportion to the shift". With three cohorts (test, SSA,
PED) that cannot be measured: three points do not order shifts by size, and the largest failure
(PED tumour core) is mostly a label-definition change, not an imaging one (note 58, C14). The claim
was retired (MasterPlan §B1). This run is the only way it can come back: hold the patients and the
labels fixed, change only the image, in graded steps, and see whether realised risk rises with the
step.

It also tests the second half of the thesis sentence — "a stronger model shrinks the damage" — on a
controlled axis, where the two real cohorts gave it only two points.

## Design

**Cases.** The 189 BraTS 2021 test cases (`data/preprocessed/brats`, the split used everywhere).
Labels are never altered: what is being measured is how the same truth is segmented from a degraded
image.

**Models.** `neurovision` (seed 42, deployed), `baseline_unet3d` (matched-recipe U-Net), nnU-Net
(Gate A, `checkpoint_final.pth`, TTA off as in note 61's `--disable_tta` arm). Nothing is trained.

**Corruptions** (`src/neurovision/data/corruptions.py`; applied to the preprocessed image in a
positive magnitude domain inside the original brain mask, then re-z-scored over that mask, as the
deployed pipeline normalises every scan). Five kinds × four severities, plus level 0 (clean):

| Kind | Parameter | Levels 1–4 | What it stands for |
|---|---|---|---|
| `thick_slice` | slice thickness, mm (block average along S, linear back to 1 mm) | 2, 3, 5, 7 | thick-slice clinical acquisitions (the front end warns at anisotropy 2, refuses at 6) |
| `rician` | noise σ / brain-intensity SD | 0.05, 0.10, 0.20, 0.30 | low-field or fast, noisy acquisitions |
| `bias` | Legendre coefficient bound, degree 3 | 0.1, 0.2, 0.3, 0.5 | coil inhomogeneity left after (or without) N4 |
| `gamma` | \|log γ\|, direction drawn per case | 0.1, 0.2, 0.4, 0.7 | contrast differences between scanners and protocols |
| `motion` | in-plane translation, mm, on 40% of phase-encode lines (DC line excluded) | 1, 2, 4, 8 | patient motion (ghosting) |

**Seeds.** One per (case, kind, level), `case_seed(42, case_id, kind, level)` (sha256 of the string;
never Python `hash`). Each MRI sequence gets an independent child generator. Corrupted volumes are
not stored: they are regenerated bit-identically from the seed.

**What is saved per (model, case, kind, level)** — scalars and small curves only, never volumes:
Dice / HD95 / volume for ET, TC, WT on the deployed post-processed mask; per-case false-negative
counts |G \ M(τ)|, |G| and |M(τ)| on the conformal v2 threshold grid (v1's 65 points plus the eight
extended points), so any frozen threshold can be applied afterwards without re-inference; mean
foreground entropy; the input-statistics OOD score; and the gate features the deployed gatekeeper
reads.

**Frozen thresholds.** The primary τ̂ is conformal v1's CRC threshold fitted on BraTS validation
for each model (`results/conformal/<model>/fit.json`; nnU-Net from note 62), regions WT and TC,
α ∈ {0.05, 0.10, 0.20}. When conformal v2 (W2.1) exists, its CRC and RCPS thresholds (ET included)
are applied to the same stored curves and reported beside — no re-inference needed. Nothing is
fitted on corrupted data.

**Gate.** The deployed `neurovision` gate (QC model + conformal signals, thresholds unchanged)
applied to `neurovision`. For the other two models the same signals computed on their own outputs
with the same thresholds, labelled "transferred, not refitted". Usable = Dice ≥ 0.7 on WT and TC
(the registered bar); W2.5's volume endpoint is added beside it if it exists by analysis time.

## Endpoints and decision rules

**Primary (one Holm family, 5 kinds × 3 models × 2 regions = 30 tests).** Per (kind, model,
region): the per-case miss rate at α = 0.10's τ̂, across levels 0–4, tested for an increasing trend
with Page's L test (paired, one-sided, `scipy.stats.page_trend_test`). **GRADED** for that cell if
p_holm < 0.05 **and** the paired bootstrap 95% CI (10,000 resamples, seed 42) of mean miss rate at
level 4 minus level 0 excludes zero. **NOT GRADED** otherwise.

The thesis earns the word "graded" back only for the kinds where it is GRADED for all three models
in at least one region; it is written per kind, never as a blanket statement.

**Secondary (one Holm family).** "A stronger model shrinks the damage": per case, the OLS slope of
miss rate on level index; paired difference nnU-Net − `neurovision` and `neurovision` −
`baseline_unet3d`, per kind and region (5 × 2 × 2 = 20 tests), paired bootstrap; a pair is FLATTER
when the CI is below zero and p_holm < 0.05.

**Descriptive (no test).** Per level: Dice, realised risk at each α (with bootstrap CI and the
v1 VIOLATED / holds / inconclusive rule), gate refusal rate, silent-failure rate
(accepted ∧ not usable), OOD flag rate. Dose–response figure: realised risk vs level, one panel per
kind, three model lines, nominal α marked.

**Predictions, registered (directional, not gating):**

1. Realised risk rises with level for `thick_slice`, `rician` and `motion` for every model (GRADED).
2. `bias` and `gamma` move risk least (the per-scan z-score absorbs a global intensity change and
   the training augmentation covered intensity scaling); one or both may be NOT GRADED.
3. nnU-Net has the flattest slope and `baseline_unet3d` the steepest, for most kinds.
4. The gate's refusal rate rises with level but its silent-failure rate also rises: the gate sees
   some bad masks but not all (C21's pattern, now on a controlled axis).
5. The OOD flag rate rises with level for `rician` and `thick_slice` (input statistics move) more
   than for `bias` and `gamma`.
6. At level 0 every number reproduces the saved clean test result (bridge, below).

## Bridge and probe (trap 9)

**Bridge.** Level 0 is re-inferred on the GPU and must reproduce each model's saved test per-case
Dice (`results/…/per_case_metrics.csv`; nnU-Net `results/conformal/nnunet_gatea/`) within 0.003 mean
absolute difference per region. A failed bridge stops the run before any corrupted number is read.

**Probe** (before the main session; counted against the budget): 5 test cases × all 21 conditions ×
3 models, through the exact notebook the main run uses, writing the real output files, running the
bridge on those 5 cases and the motion corruption (the slowest path, a 3-D FFT per channel) and
nnU-Net's own preprocessing. It reports seconds per (model, case, condition). If the projected main
run exceeds 20 GPU-h, the design is cut **before** launch, in this order: drop level 1 of every kind
(the least informative), then drop `baseline_unet3d` from `bias` and `gamma`; the cut is recorded
here as an amendment dated before the main run.

## Power line (§B3 rule 6)

From conformal v1 on clean test (`results/conformal/neurovision/realised_risk.csv`): at α = 0.10 the
95% CI half-width of mean miss rate is 0.012 (WT) and ≈ 0.017 (TC), implying a per-case SD of about
0.08–0.12. A paired level-4-vs-level-0 difference with per-case SD ≤ 0.12 is resolvable at about
0.025 (80% power, two-sided 0.05, n = 189), ≈ 0.03 after Holm over 30. Published corruption studies
on brain-MRI segmentation report Dice losses of 0.05–0.20 at the strongest levels chosen here, and
the miss rate is the false-negative side of the same failure, so the plausible level-4 effect for
`thick_slice`, `rician` and `motion` exceeds the resolvable one. For `bias` and `gamma` it may not —
prediction 2 says so in advance, and a NOT GRADED there is reported as "not resolved at n = 189",
never as "robust". The run launches.

## What would make this invalid

- Changing kinds, levels, seeds, thresholds, the α, the usable bar or the trend test after the probe
  has produced its first corrupted-case number (the probe's 5 cases are not analysed for effects).
- Fitting any threshold, gate cut-point or normalisation on corrupted data.
- Storing corrupted volumes and re-using them across code versions (regenerate from seeds at one
  pinned `GIT_REF`).
- Reporting a kind as "robust" when it is NOT GRADED.

## Cost

GPU: 189 × 21 conditions × 3 models ≈ 11,900 inferences. Measured clean speeds: nnU-Net 1.3 s/case
without TTA on a T4 (note 61); the two in-house models are timed by the probe. Estimate 15–18 GPU-h
over two sessions (resumable per (model, kind, level) shard, each shard's output written on
completion). CPU: the analysis is arithmetic over the saved per-case tables (< 1 CPU-h).

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first corrupted
number exists, except a dated probe amendment as described above.)*
