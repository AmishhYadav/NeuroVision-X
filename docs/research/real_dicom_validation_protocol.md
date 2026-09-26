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

One numbered note in `docs/experiments.md`; the error-budget row; `master_plan.md` P1.2 ticked.
