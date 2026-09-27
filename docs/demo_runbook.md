# Demo runbook — semester presentation (~2026-11-24)

`master_plan.md` P3.2. Written 2026-09-27; **rehearse it once from a cold start before the day,
and run the checks in §2 the day before.** Every claim spoken in the demo must be in
`docs/research_docs/claims_and_evidence.md` — this runbook quotes only rows from there.

---

## 1. What the demo shows, in one sentence

A tumour segmentation model wrapped in a distribution-free error bound and a refusal gate: the bound
holds in distribution, fails under shift in proportion to the shift, and the gate cannot see
cohort-level shift. We show where it breaks, and what a new site needs to restore it.

**Do not say:** "safe to deploy", "guaranteed per patient", "clinically validated", "state of the
art". The bound is an **average over studies, in distribution, not per patient** (P0.5 wording).

---

## 2. The day before (all must be green)

```bash
# from the repo root
.venv/bin/pytest                                   # expect ~2,460 passed, ~37 skipped
.venv/bin/python scripts/smoke_test.py             # SMOKE TEST PASSED
(cd app/frontend && npm test)                      # vitest green

# serve (two terminals) -- exactly as docs/research_docs/reproducibility.md §5
NVX_EXPERIMENT=neurovision NVX_EVAL_DIR=outputs/neurovision/eval_test \
NVX_CHECKPOINT=outputs/neurovision/checkpoints/best.pt \
NVX_REPORT_DIR=outputs/report_neurovision/reports NVX_JOB_DIR=outputs/clinical_jobs \
.venv-clinical/bin/uvicorn app.backend.main:app --port 8000
(cd app/frontend && npm run dev)

(cd app/frontend && npm run test:e2e)              # rendered-pixel E2E, third terminal
```

Also check: jobs `9c2cc294` (PROCEED) and the regenerated UPENN-GBM-00001 REFUSE job (story B) still load at `/clinical` after a
backend restart (T0.5 rehydration), and the 3D twin renders (P0.7).

**Memory.** Memory accumulates across clinical jobs in one backend process; after many jobs the M4
thrashes (`docs/research_docs/lessons.md`, 2026-09-27). If you do upload live, use a small study and restart the backend
afterwards; never queue two uploads.

**Fallback recording.** Once everything is green, screen-record stories A–C end to end (~6 min) and
keep the file on the laptop and on a USB stick. If anything fails live, switch to it without
debugging on stage.

---

## 3. The stories, in order (~12 min)

### A. PROCEED — the pipeline end to end (3 min)

- Open `/clinical`, open job **`9c2cc294`** (UPENN-GBM-00002). Do **not** re-upload live: a fresh
  job takes ~6 min on the laptop (361 s measured), mostly ANTs + HD-BET.
- Walk the stages from `summary.json`: ingest → input QC → co-registration to SRI24 + skull strip
  (279 s) → segmentation (65 s) → gate → report, DICOM-SEG export.
- Show the 3D twin (patient-left on screen-right), the report (WT 176.0 mL, right-sided), the
  gatekeeper panel: predicted Dice WT 0.901 / TC 0.895 → **PROCEED**.
- **Say it:** this study is **a BraTS 2021 training patient** (note 50: UPENN-GBM-00002 =
  `BraTS2021_01202`, train split). It demonstrates the plumbing, not accuracy.

### B. REFUSE on quality — the gate doing its job (2 min)

- Open the REFUSE job for UPENN-GBM-00001 (was `f4a4a754`; that job directory was deleted in the
  2026-09-27 disk reclaim — regenerate it once with `.venv-clinical/bin/python
  scripts/run_clinical_study.py +clinical.study_dir=data/fixtures/dicom/UPENN-GBM-00001
  +clinical.out_dir=outputs/clinical_jobs`, ~6 min, and regenerated 2026-09-27 as job **`67b67b1d`**): predicted Dice WT **0.626** (job `67b67b1d`; 0.656 in the deleted `f4a4a754` — registration is not bit-deterministic) below the calibrated cut 0.707 →
  **REFUSE**, with the reason in the banner.
- **Say it:** a refused study is a successful outcome, not a failure. The gate's thresholds are
  quantiles of the validation set, frozen before use.

### C. REFUSE on intended use — scoping, not detection (1 min, slide)

- Slide: the note 53 table. With the `intended_use` signal on, every paediatric study is refused
  (PED accepted 0.798 → 0.000), test and SSA unchanged.
- **Say it:** this is a labelling rule. It refuses every child *by construction*; it proves nothing
  about detection. A de-identified study with no age is cautioned, not refused.
- Show live only if the gate has `intended_use` enabled by the day (it is switched on after P1.2)
  and an adult study then shows the `intended_use` row as PROCEED in the panel.

### D. Where the bound breaks — the science (4 min, slides)

1. **Holds in distribution** (C13): realised risk 0.64–0.96× nominal on BraTS test, 6/6 cells.
2. **Breaks under shift, graded** (C14): SSA WT ~1.1× nominal, PED WT 1.4–1.9×, PED TC 3.5–11.5×.
3. **The gate can't fix it** (C22): PED TC realised risk 0.573 even on accepted cases at nominal
   0.10.
4. **What a site needs** (C24, counterfactual): ~10–40 of its own labelled cases restore the bound
   for WT (SSA, PED) and TC (SSA). **PED tumour core: no number of local cases** — the miss rate
   at the loosest threshold is already 0.356.
5. Related work (one line): coverage failure under shift is also reported between FeTS sites and
   for CT organs (arXiv 2606.20115, 2608.18193); our contribution is the graded, cross-population
   shape and the infeasible sub-region.

Figure: the P1.1 recalibration curves (`outputs/local_recalibration/`).

### E. The honest scoreboard (2 min, one slide)

- One positive: ET Dice **+0.0267** over a matched U-Net (p_holm 1.4e-21, n = 189), ~13× the seed
  noise floor (C1). Say it is **not yet tested against nnU-Net** (Gate A running; update this line
  if the verdict is in).
- Eight nulls or negatives, all pre-registered — name two: the founding hypothesis (null), heavy
  augmentation D0 (null).

---

## 4. Likely questions

| Question | Answer, with its source |
|---|---|
| "Is it better than nnU-Net?" | Not yet known — Gate A is running on Kaggle, full 1000-epoch recipe, scored on our 189 test cases through our metric path (`preregistration_strong_baseline.md`). |
| "Why refuse instead of just showing the mask?" | Silent failure: on PED, 53/99 accepted masks were unusable before scoping (note 51). |
| "Is the guarantee per patient?" | No. Average over studies, in distribution (model card, P0.5). |
| "Did you train on the demo patient?" | Yes, possibly — note 50. That is why story A is plumbing only. |
| "Real hospital data?" | Real DICOM, public (TCIA UPENN-GBM, RSNA-MICCAI). The P1.2 real-DICOM run measures what the front end costs on BraTS test patients (fill in its note number and headline before the day). |

---

## 5. Before the day — fill in

- [ ] P1.2 note number and headline (usable rate end to end, front-end cost) → §3.D / §4.
- [ ] Gate A verdict if available → §3.E.
- [ ] Multiseed Amendment 1 verdict (headline at seed 43) → §3.E.
- [ ] Cold-start rehearsal done, date: ____
- [ ] Fallback recording made, date: ____
