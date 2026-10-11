# Deep review of rev2: `NeuroVision-X_IEEE_Paper_rev2.docx` (2026-10-04)

**This is the single, complete edit list for rev2.** It replaces the shorter rev2 review committed
earlier today (`34cb525`). It is written so that every remaining change can be made in one pass.

How it was done:
- Every paragraph, table, caption and reference of rev2 was read end to end.
- Every number was traced to its source, including those already checked in earlier rounds. The
  sources are: the per-case CSVs, `outputs/conformal`, `outputs/local_recalibration`,
  `outputs/error_budget`, `experiments.md`, the claims ledger, the pre-registrations and git history.
- Three new reproducibility scripts were run, and one new analysis (R6). Their results change the
  text in a few places.
- Every **Find** string below was matched against rev2 verbatim. Paste it into Word's Find
  (Cmd+F). Subscripts and superscripts do not affect matching.

**Verdict.** The science and the numbers are in very good shape. There are **two factual errors**
(A1, A2), **one new result** that softens a claim (A3: R6), and a set of accuracy, wording and
submission items. Part G lists exactly what was verified, so you need not re-check any of it.

New since rev2:
- Scripts `ref_et_comparison.py`, `real_dicom_offsets.py` and `confidence_diag.py`. These now back
  every post-review number in the paper.
- R6 run (`experiments.md` note 57).
- Ledger C24, and an addendum to the conformal Amendment 1 result.

---

## Part A — Must fix (wrong or overtaken by new results)

### A1. §VIII-C, second caution: the per-split violation claim is inverted

Source: `outputs/local_recalibration/summary.csv`, column `p_violation`, and `experiments.md`
note 52, which correctly says "**at k = half**". The paper says violations reach about 30% "at small
k" and "near the floor". The data show the opposite:
- **Near the floor**, few calibration cases force conservative thresholds. P(violation) is **0 in
  almost every cell at k ≤ 10** (at most 0.09, SSA TC at α = 0.20), but masks are inflated (up to
  3.6× at α = 0.10).
- **At the largest k** (half the cohort), thresholds are tight and P(violation) reaches **25–30%**:
  test TC 0.27/0.30; SSA WT up to 0.25; SSA TC up to 0.29; PED WT up to 0.30.

**Find:**
`Second, the guarantee is on the mean over draws: the fraction of individual splits exceeding α reaches about 30% at small k, on test as well as externally. A site obtains one calibration draw, and near the floor a bound restored in expectation is still violated in roughly 30% of calibration draws.`

**Replace with:**
`Second, the guarantee is on the mean over draws, and the floor trades safety for mask size. Near the floor, the few calibration cases force conservative thresholds: almost no individual split exceeded α at k ≤ 10, but the recalibrated masks were inflated by up to 3.6×. As k grows the thresholds tighten, and at the largest k evaluated (half the cohort) the fraction of individual splits whose held-out miss rate exceeds α reaches 25–30%, on test as well as externally. A site obtains one calibration draw, so a bound restored in expectation can still be violated by the draw it actually has.`

### A2. §V-G and §VIII-C: the "46 and 23" figures and the "twice as many" claim are unverified

Neither figure could be confirmed in [33]. The abstract of [33] reports that its betting bound
re-certifies six organs with 25 local cases, against 30–40 for Hoeffding–Bentkus, and gives no CRC
comparison. Replace both with a bound that can be proven, and is model-independent:
- Under a true risk of α, an all-zero calibration sample has probability (1 − α)^k.
- A valid (1 − δ) upper bound can therefore certify α only if (1 − α)^k ≤ δ, that is
  k ≥ ln δ / ln(1 − α).
- At δ = 0.1 this gives **45** cases (α = 0.05) and **22** cases (α = 0.10).

| # | Location | Find | Replace with |
|---|---|---|---|
| A2a | §V-G, end | `Its minimum calibration size at δ = 0.1 is independent of the model, 46 and 23 cases at α = 0.05 and 0.10 as computed in [33], against 19 and 9 for CRC.` | `Whatever the model, no valid bound of this kind can certify α from fewer than ln δ / ln(1 − α) calibration cases, the size at which even an all-zero sample of losses becomes unlikely under a true risk of α: at δ = 0.1, 45 and 22 cases at α = 0.05 and 0.10, against 19 and 9 for CRC.` |
| A2b | §VIII-C, second caution | `A high-probability bound removes this at the price of more local cases; on CT, [33] needed about twice as many cases for a betting-based RCPS bound as for CRC.` | `A high-probability bound removes this at the price of more local cases; on CT, [33] found that high-probability bounds need more local cases than the expectation bound.` |
| A2c | §VIII-C, second caution | `The model-independent minimum for that bound at δ = 0.1 is 46 cases at α = 0.05 and 23 at α = 0.10 (Section V-G), which an SSA-sized cohort of 60 cannot both calibrate and evaluate at α = 0.05;` | `The model-independent minimum for that bound at δ = 0.1 is 45 cases at α = 0.05 and 22 at α = 0.10 (Section V-G), which an SSA-sized cohort of 60 cannot both calibrate and evaluate at α = 0.05;` |

### A3. R6 result: SSA tumour core at α = 0.05 is not restored even at or above its floor

What was run: `scripts/local_recalibration.py` at k = 39 and k = 45. It uses the same 1,000 seeded
splits as Amendment 1. Output is in `outputs/local_recalibration_r6/` (note 57; pre-registration
addendum; ledger C24).

What it shows:
- **Proposed model:**
  - k = 39: feasible in 43.5% of splits, held-out risk 0.058, **not restored**.
  - k = 45 (15 cases held out): feasible in 45.6%, risk 0.065, **not restored**.
- **Baseline:** restored at both k values (0.027 and 0.030).
- **Why:** the floor (12) uses the *cohort's* R_min. Individual 39-case draws often have a higher
  one, so the floor is **necessary, not sufficient**.
- **Status:** this is exploratory, and the registered verdict stands as registered.

| # | Location | Find | Replace with |
|---|---|---|---|
| A3a | Abstract | `Local recalibration restores the expected-risk bound where the lowest achievable risk allows, with floors of 5–39 labelled cases per site and restoration shown at 30–49, at the cost of larger masks.` | `Local recalibration restores the expected-risk bound in most cells, with floors of 5–39 labelled cases per site and restoration shown at 30–49, at the cost of larger masks; the floor is necessary but not sufficient.` (+6 words, abstract becomes ≈ 254. For IEEE, cut `(0.70–0.91× α; two intervals include α)`.) |
| A3b | §I, contribution 2 | `which follows from the calibration rule and the cohort's lowest achievable risk, together with` | `which follows from the calibration rule and the cohort's lowest achievable risk and proves necessary but not sufficient, together with` |
| A3c | §VIII-C, first paragraph | `SSA TC at α = 0.05 was evaluated only at k = 30, below its floor of 39, and is not restored there; at or above the floor at most 21 cases would remain for evaluation, and that cell has not been run.` | `SSA TC at α = 0.05 is not restored at any k evaluated. Below its floor (k = 30) only 39% of draws are feasible, and at or above it (k = 39 and 45, leaving 21 and 15 held-out cases; post-review, exploratory) only 44–46% are, with held-out miss rates of 0.058 and 0.065. The floor in (12) uses the cohort's lowest achievable risk, so it is necessary but not sufficient: a 39-case draw often has a higher one. The baseline is restored at both k (0.027 and 0.030).` |
| A3d | §VIII-C, same paragraph | `H6 is supported. Its registered predictions` | `H6 is supported as registered, on the registered grid of k. Its registered predictions` |
| A3e | §VIII-C, same paragraph | `the empirical content is the held-out risk at or above the floor.` | `the empirical content is the held-out risk at or above the floor, and it contains one failure that the registered grid could not reach (SSA tumour core at α = 0.05, above).` |
| A3f | Table XI, SSA · TC, "Held-out risk and verdict" cell | `α .05 at k = 30 is below the floor: 39% feasible, 0.054, not restored` | `α .05 not restored: at k = 30 (below the floor) 39% feasible, 0.054; at k = 39 / 45 (post-review) 44% / 46% feasible, 0.058 / 0.065` |
| A3g | Fig. 5 caption | `crosses mark cells not restored: SSA TC at α = 0.05 below its floor,` | `crosses mark cells not restored: SSA TC at α = 0.05 (also not restored at k = 39 and 45, not plotted),` |
| A3h | §VIII-C, first caution | `A site needs at least the floor, not merely a draw that happens to fit.` | `A site needs at least the floor, not merely a draw that happens to fit, and SSA tumour core shows that even the floor may not be enough.` |
| A3i | §X, guarantee paragraph | `restores the mean bound, although a single site's draw can still exceed it and restoration enlarges the mask.` | `restores the mean bound in all but one evaluated cell, although the floor does not guarantee it, a single site's draw can still exceed it, and restoration enlarges the mask.` |
| A3j | §X, recipe paragraph | `recalibrate; and check the size of the recalibrated mask before trusting the bound.` | `recalibrate; confirm the bound on held-out local cases, since the floor is necessary but not sufficient; and check the size of the recalibrated mask before trusting the bound.` |
| A3k | §XII | `it can be restored at a new site, in expectation, with local labelled cases at or above a floor computable from the site's lowest achievable risk, except where the model is already beyond rescue.` | `it can usually be restored at a new site, in expectation, with local labelled cases at or above a floor computable from the site's lowest achievable risk, although that floor is necessary rather than sufficient and is of no help where the model is already beyond rescue.` |
| A3l | Table XVI, H6, Verdict | `Supported in expectation; floors are arithmetic; high-probability bound not fitted` | `Supported as registered; floor necessary, not sufficient (SSA TC at α = 0.05, post-review); high-probability bound not fitted` |
| A3m | §XI, "not run" list | `SSA tumour core at α = 0.05 at or above its floor; ` | delete (it has now been run) |

### A4. §IV-B: "roughly 65% background" is wrong

Measured on 21 cases: background is **83%** of the full 240×240×155 grid, where the z-score is
computed, and 55% of the cropped box.

- **Find:** `so that the roughly 65% background does not distort the statistics;`
- **Replace with:** `so that the background, about 83% of the 240 × 240 × 155 grid, does not distort the statistics;`

### A5. §V-I: wrong toolkit cited, and the registration type is never stated

The code uses `brainles_preprocessing`, the BrainLesion Suite, with **ANTs `Rigid`** (its default).
That is not the 2020 BraTS Toolkit. The "rigid" detail matters, because §IX-E now rests on it.

| # | Location | Find | Replace with |
|---|---|---|---|
| A5a | §V-I | `using ANTs [51] through the BraTS toolkit's preprocessing [52]` | `using rigid ANTs registration [51] through the BrainLesion Suite preprocessing module [52]` |
| A5b | Ref. [52] (cited only once, so numbering is unaffected) | `F. Kofler et al., "BraTS Toolkit: Translating BraTS brain tumor segmentation algorithms into clinical and scientific practice," Front. Neurosci., vol. 14, art. 125, 2020.` | `F. Kofler et al., "BrainLesion Suite: A flexible and user-friendly framework for modular brain lesion image analysis," arXiv:2507.09036, 2025.` |
| A5c | §IX-E | `Table XV therefore measures disagreement between two rigid registrations of the same patient, ours and the reference's,` | `Table XV therefore measures disagreement between two independent registrations of the same patient, our rigid one and the reference's,` (BraTS's own registration type is not established here; do not call it rigid) |

### A6. §III-A and Table A-I: cross-fitted fine-tuning is no longer "not yet run"

The master plan (2026-10-03) shows D3 `d3-ssa-cf0` done, `d3-ssa-cf1` running and the PED folds
queued. TTA is scheduled for about 2026-10-10.

| # | Location | Find | Replace with |
|---|---|---|---|
| A6a | §III-A | `have not yet been run and are listed so that their absence is visible.` | `had not finished at the time of writing and are listed so that their absence is visible.` |
| A6b | Table A-I, "Cross-fitted fine-tuning" row, 3rd column | `Registered; not yet run` (first occurrence) | `Registered; running at the time of writing` |
| A6c | Table A-I, "Flip test-time augmentation" row | `Registered; not yet run` (second occurrence) | `Registered; scheduled` |
| A6d | Table A-I footnote | `TTA and cross-fitted fine-tuning are registered and not yet run.` | `TTA and cross-fitted fine-tuning are registered and were not complete at the time of writing.` |

**If D3 finishes before you submit** (it is the direct test of §X's "only a model trained on the
target label definition can"), see Part E.

### A7. §V-L: the GPU-hour total is stale

"About 170" predates Gate A sessions 3–4 and D3. It is about 186 now, and about 230 projected once
nnU-Net and D3 finish. **Find** `The project used about 170 GPU-hours in total,` and replace the
number with the figure at submission. The `project_history.md` GPU table is the tally.

---

## Part B — Accuracy and wording (true, but imprecise or inconsistent)

| # | Location | Find | Replace with |
|---|---|---|---|
| B1 | §VIII-B (two consecutive sentences say the same thing) | `That is what an adult-trained model should do with non-enhancing and cystic tissue it was taught to leave out of tumour core (Section IV-A), and no threshold can recover it. That an adult-trained model misses tissue outside its training definition is expected; what is measured here is how much, a floor of 35.6% that no threshold removes.` | `That is expected of an adult-trained model facing non-enhancing and cystic tissue it was taught to leave out of tumour core (Section IV-A); what is measured here is how much, a floor of 35.6% that no threshold removes.` |
| B2 | Table X caption (ungrammatical and vague) | `The baseline violates 7 of 12 cells in substantially the same cells.` | `The baseline also violates 7 of 12 cells, 6 of them the same (it violates SSA TC at α = 0.05 instead of PED WT at α = 0.10).` (verified in `outputs/conformal/*/realised_risk.csv`) |
| B3 | §VIII-B | `independently seven of twelve for the baseline, in substantially the same cells.` | `independently seven of twelve for the baseline, six of them the same cells.` |
| B4 | §III-A (rev2 review A1, still open) | `are listed as outstanding in Section XI rather than implied.` | `are listed as outstanding in Section XI rather than implied; those run before submission (the enhancing-tumour comparison on cases with reference enhancing tumour, the registration-offset diagnosis, the confidence-head check and the above-floor recalibration of SSA tumour core) are labelled post-review and exploratory where they are quoted.` |
| B5 | §XII (overstates, given R3) | `On real data the front end, not the model, is where most accuracy was lost, and its cost must be measured rather than assumed.` | `On real data the largest measured loss arose at the front end, mostly as disagreement with the reference's own alignment, and its cost must be measured rather than assumed.` |
| B6 | Data and code availability | `Source code, Hydra configurations, all pre-registrations, measurement protocols, the experiment log and the claims ledger are available at [14].` | `Source code, Hydra configurations, all pre-registrations, measurement protocols, the experiment log and the claims ledger are available at [14], including the scripts that regenerate every post-review analysis (ref_et_comparison.py, real_dicom_offsets.py, confidence_diag.py and local_recalibration.py).` |
| B7 | §VII (now confirmed by script: std 0.016–0.018) | `nearly constant (standard deviation about 0.02) and no different inside and outside the predicted tumour.` | `nearly constant (standard deviation below 0.02) and almost no different inside and outside the predicted tumour.` |
| B8 | Table XI, "Test (control)" row, "Floor k" cell (optional) | `—` (in that row) | `WT 20 / 10 / 5; TC 27 / 11 / 5` (from `floors.csv`; makes the control row comparable) |
| B9 | §IX-E (optional; makes the R3 numbers complete) | `raises the median whole-tumour Dice from 0.66 to 0.71.` | `raises the median whole-tumour Dice from 0.66 to 0.71 (tumour core from 0.58 to 0.66; 16 of 22 studies improve, Wilcoxon p = 0.017).` |

---

## Part C — Check by eye in the exported PDF (not detectable from the .docx XML)

1. **Table IV has 6 columns.** At single-column width it will overflow or wrap badly. Make it a
   two-column-spanning table, or move the ref-ET column into its own small table.
2. **All 16 equations.** Check that the fractions, hats (τ̂, R̂, ŷ) and the ↑ in (6) render. The
   OMML is well-formed, but rendering can still differ.
3. **Tables XI and XIII.** The long cells (A3f, and the OOD rows) must not push the tables across a
   page break mid-row.
4. **Figures 4, 5 and 7.** Check legibility at final column width, especially Fig. 5's three
   panels.

## Part D — Submission items

| Item | Where | Action |
|---|---|---|
| Zenodo DOI | §III-A `[Zenodo DOI, to be minted at submission]` | Create a GitHub release → Zenodo DOI → paste it in |
| Weights and logits | Data availability `[Author to confirm: …]` | Recommended: weights released with the DOI; logits (≈ 31 GB) on request |
| Acknowledgment | `[Author to add: faculty guide, department and any funding.]` | Fill in; keep the LLM disclosure |
| Ref. [61] | §XI | Confirm that arXiv:2404.15009 describes four separate sub-region labels |
| Venue | Whole paper | **16,553 words.** That fits MELBA. IEEE TMI or JBHI would need roughly a third cut (move Appendix A and Tables V and VII to supplementary) |
| Test count | §V-L "about 2,600 automated tests" | Still correct (about 2,620 with the new scripts) |

## Part E — Results that will land soon: where each one goes

These are in flight. When each finishes, edit **all** the listed places in one pass.

**nnU-Net, Gate A** (prediction session planned for about 2026-10-17):
- §VI-E: replace `At the time of writing that run has not finished within the compute budget, so no verdict is reported, Tables III and IV carry no nnU-Net row, and H1 is stated against the matched baseline only.` with the registered verdict (survives / parity / retired) and its numbers.
- Table III: add an nnU-Net row.
- Table IV: add an "nnU-Net − proposed" row, with voxel-wise and lesion-wise ET (the co-primary endpoints).
- Abstract: the architecture sentence.
- §X, first paragraph: `Until the comparison with nnU-Net is complete …`
- Table XVI H1: `nnU-Net comparison not complete`
- §XI: `The comparison with a strong self-configuring baseline is not complete, and the architecture claim is conditional on it.`
- §XI list: `the nnU-Net comparison, and`

**D3, cross-fitted fine-tuning** (SSA and PED):
- New paragraph in §VI-D, after the augmentation paragraph, labelled **cross-fitted, not external validation**.
- §VIII-B or §X: PED TC is where D3 matters most, because §X says "only a model trained on the target label definition can". D3 is that test, so state whether R_min for PED TC falls.
- Table A-I row → `Run; Section VI-D`.
- §III-A (A6a): drop cross-fitted fine-tuning from the "not finished" sentence.

**TTA:** Table A-I row and §III-A, plus a sentence in §VI-A or §VII, according to its pre-registered endpoint.

---

## Part F — Optional analyses still declared "not run" in §XI

All are honest as written, and none needs a GPU. In order of value:
1. **R5, Dice and precision of the recalibrated masks.** This turns "restored" from a statement
   about the miss rate into one a clinician can use. It needs a small extension of
   `analysis/local_recalibration.py` through `py-implementer`, then about an hour of CPU.
2. **RCPS floors and per-draw violation rates.** This completes H6's high-probability side.
3. **R7, pairwise-Dice detector and AURC.** It needs the 10 MC passes per case. Those caches were
   deleted on 2026-09-15 (`reproducibility.md` §11), and rebuilding them on CPU is about 37 h for
   159 external cases. **Not worth it before submission**; keep it in §XI.

---

## Part G — Verified correct in rev2 (no action)

**Numbers recomputed from raw outputs:**
- **Table III:** all 5 arms × 6 metrics, from the per-case CSVs.
- **Table IV:** every Δ, including the ref-ET column (`scripts/ref_et_comparison.py`, 4-decimal match).
- **Table VI:** SSA and PED means and Δ, plus the PED baseline ET of 0.5733.
- **Table VII:** all means.
- **Table VIII:** the TOST differences.
- **Table IX:** τ̂, the risks and every CI from `outputs/conformal/neurovision/realised_risk.csv`.
- **Table X:** all 12 cells, and exactly 7 violated (CI lower bound > α).
- **Baseline:** test ratios 0.64–0.96×; 7 of 12 violated.
- **Table XI:** every value, from `outputs/local_recalibration/{summary,floors}.csv`. This includes
  PED TC, feasible in 0.2–0.9% of splits and missing 0.38–0.41.
- **Table XIII:** every count and ratio.
- **Table XIV:** flag rates.
- **Table XV:** every value.
- **R3 (§IX-E):** `scripts/real_dicom_offsets.py` reproduces it exactly.
- **R8 (§VII):** `scripts/confidence_diag.py` reproduces it; the WT channel std is 0.016–0.018.
- **Bar sensitivity:** silent-failure rates of 1.6/10.0/31.3% (bar 0.5) and 27.0/55.0/68.7% (bar 0.9).
- **PED TC at α = 0.05:** mask inflation 2.29× mean, 1.27× median.
- **Mask inflation at k = 10, α = 0.10:** 1.84 / 3.17 / 3.61×, against 2.07 / 2.42× on test.
- **Inference cost:** FLOPs 267.4 vs 11.3, and 1,085 vs 46 ms (note 28).
- **Lesion-wise (§VI-C):** FP lesions 0.32 vs 0.47 and 0.98 vs 1.38, and the lesion-wise WT 0.7183
  (C11, C12).
- **Report layer:** 33.9 / 40.7 / 22.8% multifocality, and "1 of 25 metrics, 16 identical medians" (C6).
- **QC bias:** −0.070 → +0.228.
- **Real-DICOM counts:** 16/2/22, 8/22 and 14 unusable.
- **Empty-ET share:** 33 of 1,251 cases = 2.6%; 5 of 189 in test.
- **Configuration:** inference (64³, overlap 0.5, Gaussian, τ 0.5, min component 50) and the
  training recipe both match `configs/`.

**Structure:**
- 16 equations; (1)–(16) match the code (verified in the rev1 review).
- 18 tables and 7 figures.
- **References:** all 61 are cited, every one in first-citation order, and none is uncited.
  [32]–[34] and [61] exist and match their titles and authors.
- **Table A-I:** every commit date matches git (`git log --follow --diff-filter=A`), including the
  2026-09-26 date of Amendment 1.

**Claims against the ledger:**
- Every claim in the abstract, §X and §XII has a ledger row: C1, C4, C6, C8, C11, C12, C16, C18,
  C22–C26.
- The ledger was updated today for C1, C16, C24 and C25.

---

## Order of work (one pass)

1. **Part A**, about 40 minutes:
   - A1 and A2 first; they are errors.
   - Then A3a–A3m (R6). These run through the abstract, §I, §VIII-C, Table XI, Fig. 5, §X, §XII,
     Table XVI and §XI.
   - Then A4–A7.
2. **Part B**, about 15 minutes.
3. Export the PDF and do **Part C**.
4. **Part D** at submission.
5. **Part E** as each result lands. If you plan to submit before nnU-Net and D3 finish, the paper
   is consistent as it stands, because it states that both are incomplete.

Sources: [arXiv 2507.09036 (BrainLesion Suite)](https://arxiv.org/abs/2507.09036) ·
[arXiv 2608.18193](https://arxiv.org/abs/2608.18193) · [arXiv 2404.15009](https://arxiv.org/abs/2404.15009v4)
