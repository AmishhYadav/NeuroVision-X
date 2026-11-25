# Protocol — real-DICOM validation of the clinical path, with ground truth (P1.2)

**Written:** 2026-09-26, before any RSNA study other than the T0.4 pilot has been downloaded or
run. The git timestamp is the evidence for that ordering.

**Why.** The clinical path (DICOM zip → ingest → input QC → co-registration / SRI24 / HD-BET →
segmentation → gate) has run on n = 2 real studies with no ground truth (note 45), and both turned out
to be BraTS 2021 *training* patients (note 50). Note 47's error budget starts from BraTS-preprocessed
NIfTI, so it never measured what the front end itself costs. The RSNA-MICCAI 2021 radiogenomic
competition ships the **original DICOM** for BraTS 2021 cases (`train/<5-digit id>` ↔
`BraTS2021_<id>`), and BraTS 2021 ships their tumour labels on the SRI24 grid. That pairing gives
real-DICOM input with ground truth, on cases the model never trained on.

## Case selection — fixed now

- **Eligible:** BraTS 2021 IDs in our frozen **test** split (`configs/data/splits.yaml`) whose 5-digit
  ID appears in RSNA `train_labels.csv`. Counted on 2026-09-26: **90** of 189.
- **Sample:** 40 of those 90, drawn once with `numpy.random.default_rng(42).choice(sorted(eligible),
  40, replace=False)`, the list written to `configs/data/rsna_validation_cases.yaml` and committed
  **before** any of them is downloaded. 40 × ~6 min ≈ 4 CPU-hours, within the one-heavy-job rule.
- **Pilot (not reported as validation):** `train/00000` = `BraTS2021_00000`, a **training** case. It
  checks the geometry and the scorer only — T0.4's original question (preprocessing sensitivity on
  one case). Its number is printed with the label "training case, pilot".
- Nothing is excluded after download. A study the pipeline refuses (e.g. thick slices), or that fails,
  stays in the denominator with its outcome recorded.

## How each study runs

- Through the real clinical path (`scripts/run_clinical_study.py`, the same code the `/clinical` route
  runs), deployed settings (`configs/clinical/default.yaml`), deployed checkpoint (`neurovision`
  seed 42), gate as deployed since note 51 (`enabled_signals: [input_qc, predicted_dice]`).
- **Series roles.** RSNA folders are named `FLAIR`, `T1w`, `T1wCE`, `T2w`. Our ingest does not
  recognise the token "T1wCE" (`dicom_ingest.py`), so roles are assigned from the RSNA folder names
  through `role_overrides`, **stated openly** in the note. No new name tokens are added to the
  ingest: tuning its vocabulary on validation data would leak.

## Scoring — fixed now

- **Grid.** The full 240 × 240 × 155 SRI24 grid. The ground truth is the BraTS 2021 label for the
  same case, uncropped from `data/preprocessed/brats/<case>/` via its `meta.json` bbox and returned to
  the SRI24 orientation. The clinical mask is the job's own mask, uncropped from the job's crop. The
  scorer does not require `source_axcodes` (older `meta.json` files lack it).
- **Self-test first.** The scorer must return Dice exactly 1.0 on a ground-truth round trip (GT
  scored against itself through both uncrop paths), and a left–right mirrored mask must **fail** a
  check, before any study is scored. Brain-mask Dice is **never** used as an alignment check (trap 3:
  it scores higher on a mirrored atlas).
- **Metrics per study:** status (`done` / `refused` / `failed`) and the refusing stage; gate decision
  (PROCEED / CAUTION / REFUSE); voxel Dice ET / TC / WT through the project's metric path
  (`ignore_empty=False`); HD95.
- **Research-path reference:** the same case's existing `neurovision` row from
  `outputs/neurovision/eval_test/per_case_metrics.csv` (BraTS-preprocessed input). The paired
  difference *clinical − research* is the **front-end cost**.

## Endpoints — descriptive, no gate

1. **Completion:** fraction `done`, `refused` (by stage and reason), `failed`.
2. **Usable rate end to end**, using note 47's definition (WT and TC Dice ≥ 0.7), with a bootstrap
   CI: P(accepted AND usable), P(usable | accepted), silent failure (accepted AND unusable).
3. **Front-end cost:** per region, median and mean paired Dice difference *clinical − research*
   over studies that completed, paired bootstrap CI (n_boot 10000) and Wilcoxon, reported once, not
   Holm-corrected (descriptive).
4. A **real-DICOM stage row** added to the error budget (note 47's G-table layout), labelled
   "RSNA DICOM, n = 40, test-split cases".

**Predictions, stated now.** (a) Most completed studies lose a little Dice relative to the research
path — different skull stripping and registration from BraTS's own — with median WT loss ≤ 0.05.
(b) Some studies are refused at input QC for slice thickness or missing series; that is the pipeline
working, not failing. (c) The usable rate is below note 47's in-distribution 0.852, because the
front end adds error that the BraTS-preprocessed path never saw.

## What this is not

- Not external validation: these are BraTS 2021 patients, same distribution as test. It isolates the
  **front end**, which is exactly what n = 2 could not.
- Registration error between our SRI24 alignment and BraTS's own counts against us here. That is
  correct for an end-to-end number and is stated as such.

## Reporting

One numbered note in `docs/research_docs/experiments.md`; the error-budget row; `master_plan.md` P1.2 ticked.

---

## Amendment 1 — Real-DICOM v2 (MasterPlan W3.4), written 2026-11-25

**Written before any of the steps below has run.** Note 55 (C25) and the post-hoc R3 check (note 57)
are known: in atlas space the front end cost a median 0.23 WT Dice, most of it the disagreement
between our SRI24 rigid registration and BraTS's own; 16 of 40 studies were refused at input QC for
thick slices. Nothing about the 50 unsampled eligible studies has been downloaded or looked at.
Analysis-budget family W3.4 (confirmatory, §B3 rule 4). Registered externally before step (c).

### What changes

**(a) Carried reference.** For every study that reaches segmentation, the BraTS ground truth is
carried into *our* atlas space instead of being compared in BraTS's: the BraTS-preprocessed T1CE of
the same case (uncropped to the SRI24 grid exactly as the label is) is rigidly registered (ANTs
`Rigid`, mutual information, fixed seed) to our pipeline's own atlas-space T1CE for that study; the
same transform is applied to the BraTS label (nearest neighbour). Our mask is scored against the
carried label on our grid. Both images are the same acquisition, so the transform removes the
registration disagreement and leaves what the front end itself changes (skull stripping, intensity
handling, resampling of the original DICOM). **Carry QC (fixed now):** Dice between the carried
BraTS brain mask (T1CE > 0) and our skull-stripped brain mask; a study with carry-QC Dice < 0.90 is
reported in its own row and excluded from the carried-reference column only, with the count stated.
Brain-mask Dice is used only as a *registration-failure* screen here, never as the alignment check
for lateralisation (trap 3); the protocol's mirrored-mask self-test still runs first.

**(b) Overridden refusals.** Every study refused at input QC for voxel spacing or anisotropy is run a
second time with only those two refusals disabled (`spacing_max_mm` and `anisotropy_refuse_ratio`
raised so they cannot fire; every other setting deployed). Its mask is scored in both spaces. The
deployed outcome (refused) stays the outcome in every end-to-end endpoint; the overridden run
answers only "was refusing right?".

**(c) Full eligible set.** The remaining **50** eligible test-split studies (90 eligible − 40
sampled; list: `configs/data/rsna_validation_cases_v2.yaml`, committed with this amendment) are
extracted from the RSNA-MICCAI competition data by the same CPU Kaggle kernel as before and run
through (a) and (b). The 40 already-run studies are **re-run** with the atlas-space T1CE kept (it
was deleted in P1.2 to save disk); their re-run masks must reproduce the stored P1.2 masks (WT
Dice ≥ 0.99 per study, else the discrepancy is reported and the re-run mask is used). **n = 90.**
The pilot stays excluded.

**Deployed settings at run time:** `configs/clinical/default.yaml` as of the run commit (since note 55
the live gate also has `intended_use`, note 53). The atlas-space column is therefore recomputed on
all 90, not copied from note 55.

### Endpoints

- **E1 (primary) — front-end cost after removing registration disagreement.** Over studies with a
  mask, carry QC passed, deployed outcome *not* refused at input QC: median paired WT Dice difference
  *carried-reference − research path*, percentile bootstrap 95% CI (n_boot 10000, seed 42).
  **Verdict:** `SMALL` if the CI lower bound > −0.05; `LARGE` if the CI upper bound < −0.05;
  otherwise `INCONCLUSIVE`. TC and ET reported beside it, descriptive.
- **E2 (primary) — was the thick-slice refusal right?** Over all studies refused at input QC for
  spacing/anisotropy, on the overridden run: fraction usable (WT and TC Dice ≥ 0.7) against the
  carried reference, Clopper–Pearson 95%. **Verdict:** `REFUSES USABLE STUDIES` if the lower bound
  > 0.5; `REFUSAL JUSTIFIED` if the upper bound < 0.5; otherwise `INCONCLUSIVE`.
- **E3 — Table XV v2,** three columns per endpoint of the original protocol (completion, P(accepted
  AND usable), P(usable | accepted), silent failure, front-end cost per region): atlas space (as
  P1.2), carried reference, and overridden refusals added in. Descriptive, bootstrap CIs.
- **E4 — (d), registered rule:** only if E2 reads `REFUSES USABLE STUDIES`, a variant in which
  thick-slice studies get CAUTION instead of REFUSE is reported, labelled **post-hoc variant**, with
  its end-to-end numbers on the same 90. The deployed rule changes (an `app/`/config change under
  §B3 rule 5) only in a separate commit after the note, and only on that verdict.

No Holm: two primary endpoints with their own verdicts, read as per-endpoint decisions.

### Predictions (directional, not gating)

1. E1 reads `SMALL`: once registration disagreement is removed, the front end costs < 0.05 WT Dice.
2. On the carried reference, P(usable | accepted) rises from note 55's 0.36 to ≥ 0.75.
3. E2 reads `REFUSES USABLE STUDIES` (the severity ladder, note 71, measured only 0.04–0.08 WT Dice
   lost at 7 mm slabs).
4. Roughly 40% of the 90 are thick-slice refusals again (P1.2's rate is a random-sample estimate).

### Power line (§B3 rule 6)

E1: ~54 studies (90 × 0.6). If the paired WT differences have SD ≈ 0.05 (the severity ladder's and
R3's spreads), the bootstrap CI half-width of the median is ≈ 1.25 × 1.96 × 0.05 / √54 ≈ 0.017, so
a true cost of −0.02 reads `SMALL` and −0.09 reads `LARGE`; the plausible effect (−0.01 to −0.05) is
resolvable except within ~0.02 of the margin. E2: ~36 studies; `REFUSES USABLE STUDIES` needs ≥ 24
of 36 usable (observed ≥ 0.67), plausible if thick slabs cost ≤ 0.08 WT Dice as on the ladder;
`REFUSAL JUSTIFIED` needs ≤ 12 of 36. With fewer refusals the verdict is more often inconclusive;
that is stated now.

### Cost

~6 min/study CPU × (90 + ~36 overridden) ≈ 12.6 CPU-h, run in batches, one heavy job at a time.
Kaggle extraction kernel: CPU only, no GPU quota. Disk: only the atlas-space T1CE, the brain mask
and the masks are kept per study (~15 MB), not the full intermediates (~700 MB).

### What would make this invalid

- Changing the carry method, the carry-QC threshold, the margins or the usable bar after any
  carried-reference Dice exists.
- Excluding a study after download other than by the carry-QC rule above.
- Changing the deployed thick-slice rule on anything other than E2's verdict.

**Erratum to Amendment 1 (2026-11-25, after registration `w3.4-prereg` / Zenodo 10.5281/zenodo.24519277,
before any W3.4 study has run or been downloaded).** The power line's E2 counts were wrong: at n = 36
the Clopper–Pearson 95% lower bound exceeds 0.5 only from **25 of 36** (24/36 gives 0.490), and the
upper bound falls below 0.5 only at **≤ 11 of 36** (12/36 gives 0.510). The decision rule itself
(lower bound > 0.5 / upper bound < 0.5) is unchanged; only these illustrative counts are corrected.
Caught by the endpoints module's hand-value test (`a04c91c`).

**Implementation note to Amendment 1 (2026-11-25, before any valid carried-reference Dice exists).**
On the first real study (BraTS2021_00032, overridden run) the carry registration diverged (identity
rotation, translation ~10⁷ mm, empty carried label, carry-QC 0 — recorded, then discarded as a
computation failure, not a measurement). Cause: the BraTS-preprocessed T1CE is z-scored, so its
background value 0 lies inside the brain's intensity range. Fix, applied to every study: before
registration only, the non-zero voxels of both images are rescaled to [0, 1] by their own 0.5 / 99.5
percentiles (background stays 0); the label and brain mask are carried from the original image. The
transform model (ANTs `Rigid`, mutual information, fixed seed), the carry-QC rule (Dice < 0.90) and
every endpoint are unchanged. A divergence guard now records an implausible transform (translation
> 50 mm, or an empty carried label) as a carry failure, which falls under the existing carry-QC
exclusion. On that study the fixed carry gives translation (2.4, −9.6, −4.6) mm and carry-QC 0.961.
