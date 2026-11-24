# Pre-registration — severity ladder: conformal risk and the gate under controlled acquisition change

**Written:** 2026-11-23, before any corrupted volume has been generated or scored. MasterPlan W3.2;
analysis-budget family 5 of 6 (§B3 rule 4). **External registration:** Zenodo DOI [10.5281/zenodo.24491280](https://doi.org/10.5281/zenodo.24491280), deposited 2026-11-24: snapshot of tag `semester-2026` (commit `894a607`). The protocol as registered is the copy in that snapshot; only this line was added afterwards.

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
(Gate A, `checkpoint_final.pth`, nnU-Net's default mirroring TTA **on** — see Amendment 1). Nothing is
trained.

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

## Amendment 1 — 2026-11-24, before the probe and before any corrupted number

The design above first said nnU-Net would run with TTA **off** (note 61's `--disable_tta` arm, 1.3 s/case).
That contradicts two other fixed parts of this protocol: the bridge reference
(`results/conformal/nnunet_gatea/evals/eval_test/`) and the frozen threshold
(`results/conformal/nnunet_gatea/conformal_nnunet/fit.json`) were both produced with TTA **on**
(`disable_tta: false` in that directory's `nnunet_probs_summary.json`). On clean test the two arms differ
by about 0.006 mean ET Dice — above the 0.003 bridge tolerance — so the nnU-Net bridge would fail by
construction, and a frozen τ fitted on TTA-on probabilities would be applied to a different model
output. Found by reading the driver against its references, before any GPU time was spent.

**Change:** nnU-Net runs with TTA on (`analysis.severity_ladder.nnunet.disable_tta: false`), the same
arm as every nnU-Net number this project quotes (ET 0.8879, C29/C30). **Cost:** measured 7.1–8.2 s/case
on a T4 (notes 61–62) instead of 1.3, about +7 GPU-h over 189 × 21 conditions. The probe measures the
real figure; if the projection exceeds 20 GPU-h the cost-cut order above applies unchanged (drop
level 1 of every kind first). TTA off is **not** a cost-cut option, because it breaks the bridge.

## Amendment 2 — 2026-11-24, after the probe, before any corrupted in-house number

The probe (`neurovision-w32-probe` v1, 0.65 GPU-h, `c4fbfd6`) passed the nnU-Net bridge exactly and **failed both
in-house bridges** (neurovision WT 0.0033; baseline_unet3d ET 0.0178), so by the rule above no corrupted in-house
shard ran. Cause, found on CPU: the driver built the in-house models without their experiment config, so it slid
96³ windows over models trained and evaluated at 64³. Fixed by building each in-house model from its reference
evaluation's own config (`eval_config.yaml`) with a guard that raises on any roi/overlap/mode mismatch; CPU
re-check on the same 5 cases: **PASS**, neurovision ≤ 0.00003, baseline 0.00000. No design element changes
(kinds, levels, seeds, thresholds, α, bars, trend test). The nnU-Net corrupted shards the probe produced were not
looked at for effects and are not reused.

**Cost, from the probe:** nnU-Net 19.4 s/case at 5-case chunks, ≈ 55 s of it fixed per call; at the registered
25-case chunks the measured steady rate is 8.2 s/case (note 62) → ≈ 11–12 GPU-h for 189 × 21 conditions. In-house
timings were measured at the wrong ROI and are re-measured by a second, in-house-only probe (5 cases, all
conditions) before the main run. Projection stays below 20 GPU-h unless that probe says otherwise; no cut yet.

## Amendment 3 — 2026-11-24, after both probes, before the main run and before any corrupted number is read

**Engineering change, no design change.** The driver now overlaps CPU work (corrupting and writing the next
chunk, scoring the last one) with GPU inference (`pipeline_cpu: true`, commit `5962a32`). Outputs are
bit-identical to the serial path (tested); only wall time moves. `s_per_case` is now shard wall time / n.

**In-house timing, probe v3** (`neurovision-w32-probe-v3`, 0.09 GPU-h, `5962a32`; 5 cases × {none, gamma L2,
motion L2}; bridges PASS ≤ 0.00003): `neurovision` 7.4–8.8 s/case, `baseline_unet3d` 3.0–4.4 s/case (serial
probe v2: 11.4–14.1 and 5.2–8.4). Full design, 189 × 21 conditions: in-house ≈ 12.2 GPU-h.

**nnU-Net timing, confirmed on the same route** (W2.6 export, `neurovision-w26-nnunet-probs`, `ca6c9db`, TTA
on, `--save_probabilities`, 25-case chunks — the ladder's own call): the prediction call runs at 9.1 s/case on
test and 8.9 on PED; the whole evaluation including input writing, probability I/O and scoring runs at 14.3
s/case. The ladder's nnU-Net path does the same input writing and probability reading, so its rate lies in
9.1–14.3 s/case → 10.0–15.8 GPU-h for 189 × 21.

**Projection and the registered cut.** Full design 22–28 GPU-h > 20, so the cut order applies:

1. **Cut 1 — drop level 1 of every kind** (21 → 16 conditions: level 0 plus levels 2–4 of five kinds).
   Projection 16.9–21.3 GPU-h. The upper end still exceeds 20, so:
2. **Cut 2 — drop `baseline_unet3d` from `bias` and `gamma`** (−6 baseline conditions, ≈ 1.1 GPU-h).
   Projection **15.8–20.2 GPU-h**; at the central rate (≈ 12 s/case for nnU-Net) ≈ 18.

The cut is decided on the conservative bound, before launch, from cost alone; no corrupted effect from any
probe was looked at. Consequences, stated now: every trend test runs on four points (levels 0, 2, 3, 4)
instead of five; the `baseline_unet3d` rows for `bias` and `gamma` are absent rather than empty, and the
model-ordering contrasts that involve the baseline are reported on `thick_slice`, `rician` and `motion` only.
Kinds, remaining levels, seeds, thresholds, α, bars and the trend test are otherwise unchanged. If the first
nnU-Net shard of the main run measures a rate that would push the total above 28 GPU-h (the weekly cap), the
run pauses between shards and is resumed the following week; no further condition is dropped.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first corrupted
number exists, except a dated probe amendment as described above.)*

**2026-11-24.** Main run 42/42 shards, 189 cases each, bridges PASS (16.57 GPU-h, inside the registered
15.8–20.2). Analysis `scripts/severity_ladder_analysis.py` at `037e95e`, nothing fitted. Full record: note 71,
claim C36, `results/severity_ladder_analysis/`.

- **Primary (26 cells after the Amendment 3 cut):** 18 GRADED, 8 NOT GRADED. Per kind: `bias` and `gamma`
  GRADED in both regions for both measured models (reported as *graded on the two measured models*; the
  baseline was cut); `rician` and `motion` GRADED for all three models on WT; `thick_slice` NOT GRADED for
  every model and region -- the miss rate falls, because the models over-segment thick-slice input and the
  loss counts only missed voxels.
- **Secondary (16 pairs):** FLATTER 3 (nnU-Net vs `neurovision`: rician WT and TC, gamma WT). nnU-Net is
  significantly steeper on motion (WT, TC) and thick-slice WT. `neurovision` vs `baseline_unet3d`: never
  FLATTER.
- **Predictions:** 4 and 6 met; 1, 2, 3, 5 not met (2 and 5 reversed: bias is the most damaging kind and the
  one the OOD score flags most).
- **No deviation** from the registered analysis. The kind-level label for `bias` / `gamma` ("graded on the
  measured models only") follows from Amendment 3's cut and is stated, not promoted to GRADED.
