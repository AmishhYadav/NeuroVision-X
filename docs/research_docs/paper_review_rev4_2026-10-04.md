# Full review of `NeuroVision-X_IEEE_Paper.docx` (rev4, 2026-10-04)

The whole paper was read end to end: every section, all 18 tables (including which cells are
bold) and all 61 references.

How it was checked:
- The three rev3 items are in. Every rev2 edit is still in.
- Every number checked in the earlier reviews was re-derived from the paper's own tables:
  - Table XIII: every count, rate and complement.
  - Tables VII, IX and X: every mean, ratio and CRC admissibility value.
  - The arithmetic of the Table IV and Fig. 3 decomposition.
  - The τ̂ grid points.
- New checks against raw outputs:
  - `outputs/local_recalibration{,_r6}/summary.csv` and `floors.csv`, all cells.
  - `outputs/error_budget/per_case_*.csv`.
  - The OOD and multiseed pre-registrations, git history, and the arXiv records of [33] and [61].

**Verdict.** The numbers hold. There are no arithmetic errors. Five items need fixing, though, and
they are about wording and completeness rather than numbers:
- **M1:** one wording error of mine, which the paper took from the rev2 review.
- **M3:** two pre-registration amendments are missing from Appendix A.
- **M2:** the silent-failure headline is mostly a tumour-core effect, and the paper never says so.
- **M4:** one undisclosed stand-in (the ages behind the intended-use rule).
- **M5:** one undefined phrase, "no threshold".

Every **Find** string below occurs exactly once in the paper.

---

## Part M — Must fix

### M1. "The floor is necessary, not sufficient" is wrong. This is my error from the rev2 review (A3)

The paper's own Table XI contradicts "necessary". Feasibility depends on each draw's own lowest
achievable risk, not on the cohort's:
- **Draws below the floor can be feasible.** PED TC at α = 0.20 has no floor at all ("never"), yet
  it was feasible in 0.2–0.9% of splits at k 5–15. SSA WT at α = 0.05 has a floor of 22, and at
  k = 20 it was feasible in 47% of splits, with a held-out risk of 0.016.
- **Draws at or above the floor can be infeasible.** SSA TC at α = 0.05 was feasible in only 44–46%
  of splits at k = 39 and 45.

So the floor is the calibration size at which a draw *with the cohort's average* lowest achievable
risk becomes feasible. It is a guide for a typical draw, not a guarantee and not a minimum. The same
wording has been corrected today in the ledger (C24), the conformal Amendment 1 addendum and
`experiments.md` note 57.

| # | Location | Find | Replace with |
|---|---|---|---|
| M1a | Abstract | `at the cost of larger masks; the floor is necessary, not sufficient.` | `at the cost of larger masks; the floor is a guide, not a guarantee.` |
| M1b | §I, contribution 2 | `and proves necessary but not sufficient,` | `and proves a guide rather than a guarantee,` |
| M1c | §VIII-C | `The floor in (12) uses the cohort's lowest achievable risk, so it is necessary but not sufficient: a 39-case draw often has a higher one.` | `The floor in (12) uses the cohort's lowest achievable risk, so it describes a typical draw, not every draw: a draw whose own lowest achievable risk is higher can be infeasible above the floor, as here, and one whose risk is lower can be feasible below it, as for PED TC at α = 0.20.` |
| M1d | §X, recipe | `since the floor is necessary but not sufficient;` | `since the floor is a guide, not a guarantee;` |
| M1e | §XII | `although that floor is necessary rather than sufficient and is of no help` | `although that floor is a guide rather than a guarantee and is of no help` |
| M1f | Table XVI, H6 | `floor necessary, not sufficient (SSA TC at α = 0.05, post-review)` | `floor a guide, not a guarantee (SSA TC at α = 0.05, post-review)` |

### M2. The silent-failure rates are almost entirely tumour-core failures. On PED that is mostly the changed label

Source: `outputs/error_budget/per_case_*.csv`, registered gate. Of the accepted-but-unusable
studies, the number whose WT Dice is still ≥ 0.7 (so they fail on TC alone):

| Cohort | Silent failures | Fail on TC alone | Silent failure if usable meant WT ≥ 0.7 only |
|---|---|---|---|
| Test | 8 | **8** | 0 / 189 (0%) |
| SSA | 11 | **9** | 2 / 60 (3.3%) |
| PED | 49 | **44** | 5 / 99 (5.1%) |

Why it matters:
- The abstract's "49.5%" for PED reads as a model failure, but 44 of the 49 cases fail only on the
  region whose label definition differs (Section IV-A).
- §IV-A promises that every PED tumour-core result is read in that light. The error budget is the one
  place where that promise is not kept.
- On SSA the label definition is unchanged, so those are genuine tumour-core failures.

**Now scripted (2026-10-04).** `scripts/silent_failure_by_region.py` regenerates the split from
the saved error-budget CSVs in seconds, and it reproduces every number above. The extra detail it
gives: SSA has 2 cases that fail on both regions; PED has 4 that fail on both and 1 that fails on
WT alone. The result is recorded in `experiments.md` note 57 and ledger C20.

| # | Location | Find | Replace with |
|---|---|---|---|
| M2a | §IX-B | `with silent-failure rates of 4.2%, 18.3% and 49.5% (Table XIII, Fig. 6).` | `with silent-failure rates of 4.2%, 18.3% and 49.5% (Table XIII, Fig. 6). Almost all of these masks fail on tumour core alone: whole tumour clears the bar in all 8 test, 9 of 11 SSA and 44 of 49 PED silent failures (post-review, exploratory). On SSA, where the label definition is unchanged, these are genuine tumour-core failures; on PED the rate largely measures the changed tumour-core definition (Section IV-A).` |
| M2b | Abstract | `with silent-failure rates of 4.2%, 18.3% and 49.5%. An input-statistics` | `with silent-failure rates of 4.2%, 18.3% and 49.5%, almost all on tumour core. An input-statistics` |
| M2c | Abstract (keeps it within 250 words) | `(0.70–0.91× α; two intervals include α)` | delete. The abstract goes from 251 to about 249 words with M1a and M2b. |
| M2d | Data and code availability | `confidence_diag.py and local_recalibration.py)` | `confidence_diag.py, local_recalibration.py and silent_failure_by_region.py)` |
| M2e | §III-A | `the confidence-head check and the above-floor recalibration of SSA tumour core)` | `the confidence-head check, the above-floor recalibration of SSA tumour core and the split of silent failures by region)` |

### M3. Appendix A omits two amendments

Both amendments were committed on 2026-09-26:
- **Gate A Amendment 1** (`2644871`): the full nnU-Net recipe, fold all, and the 95 GPU-h bound.
- **Multiseed Amendment 1** (`cdf4857`): the baseline at seed 43.

The paper says "one amendment" and §III-A says "each is labelled", so as written it misstates the
record.

The multiseed amendment matters more. The replicated headline (+0.0247 at seed 43) depends on it,
and it was committed **after** the proposed model's seed-43 result (D1, note 49) had been seen. In
its favour, the original protocol had already named the baseline's seed 43 as the next run, and that
mitigates the timing. Either way, a reader judging risk needs to see it.

| # | Location | Find | Replace with |
|---|---|---|---|
| M3a | §III-A | `Eleven pre-registrations, one amendment and two measurement protocols were written (Table A-I).` | `Eleven pre-registrations, three amendments and two measurement protocols were written (Table A-I).` |

Add two rows to Table A-I, in date order:
- After "Local recalibration, conformal amendment 1 (H6)", add:
  `Strong baseline, amendment 1 (full recipe, cost bound)` | `2026-09-26` | `No nnU-Net number`
- After that, add:
  `Second seed, amendment 1 (baseline seed 43)` | `2026-09-26` | `Proposed model's seed-43 result (seed noise +0.0021); the baseline's seed 43 was already named in the original protocol`

### M4. The intended-use result rests on stand-in ages, and the paper does not say so

No cohort on disk carries ages. Note 53 assigned each cohort a population stand-in age: 40 for
BraTS 2021 and SSA, 10 for PED. §IX-C reads as though the DICOM age was used.

| # | Location | Find | Replace with |
|---|---|---|---|
| M4 | §IX-C | `Second, an intended-use rule now refuses patients under 18 from the DICOM age. It refuses every paediatric study by construction` | `Second, an intended-use rule now refuses patients under 18 from the DICOM age. The cohorts carry no ages, so each was assigned its population's (adult for test and SSA, paediatric for PED), and the rule therefore refuses every paediatric study by construction` |

### M5. "No threshold" needs defining

Some threshold always satisfies a miss-rate bound: take τ low enough and the mask covers nearly the
whole volume. "Infeasible", "no threshold can restore it" (§I), "a floor … that no threshold
removes" (§VIII-B) and "no threshold can help" (§X) are all true only for the grid, which stops at
10⁻⁴. A conformal reviewer will raise this. One sentence fixes all four places.

| # | Location | Find | Replace with |
|---|---|---|---|
| M5 | §V-G | `and it is infeasible at every k when Rmin ≥ α.` | `and it is infeasible at every k when Rmin ≥ α. "No threshold" in what follows means none on this grid: a threshold low enough to take in nearly the whole volume would satisfy any miss-rate bound, and would be of no clinical use.` |

---

## Part S — Should fix (precision)

| # | Location | Find | Replace with | Why |
|---|---|---|---|---|
| S1 | §IX-D | `All four registered predictions held, including the switch-on failure, and the score stays display-only (H8 supported).` | `All four registered predictions held: test flag rates near their 10% and 2% targets, PED flagged at 50% or more (0.505), SSA above test with separated intervals, and an AUROC of at most 0.65 on both external cohorts (on the point estimates; the upper bounds reach 0.75). The switch-on rule failed, so the score stays display-only (H8 supported).` | The switch-on rule is a decision rule, not one of the four predictions (`preregistration_ood.md`). Also, (d) held only on the point estimates. |
| S2 | §VIII-C | `the fraction of individual splits whose held-out miss rate exceeds α reaches 25–30%, on test as well as externally.` | `the fraction of individual splits whose held-out miss rate exceeds α reaches 25–30%, on test as well as externally; this counts the sampling noise of a 30- to 95-case held-out mean as well as the draw, so it overstates how often a draw's own risk exceeds α.` | The CRC risk of a draw is its expected loss. The held-out mean adds evaluation noise on top: a draw with a true risk of 0.07 at α = 0.10 still "exceeds" on a 30-case mean fairly often. |
| S3 | §VIII-B | `while TC is violated at 1.4–1.7×.` | `while TC exceeds nominal at 1.4–1.7×, significantly at α = 0.10 and 0.20.` | Table X bolds SSA TC at α = 0.10 and 0.20 only. The α = 0.05 cell (1.45×) is inconclusive. |
| S4 | Table XIII, last row | `SSA · + OOD as REFUSE‡ \| — \| — \| — \| SF 11 → 8; OR 1 → 7` | P(acc. ∧ usable) cell: `0.683` | 47 − 6 = 41 of 60. The test row gives its value (0.884), so the SSA row should too. |
| S5 | Table A-I, D3 row | `Registered; running at the time of writing` | `Registered; trained, not yet analysed` | All four D3 folds finished on 2026-10-04 (`f58e075`). Update the footnote to match. Once D3 is analysed, it replaces this row (see the rev2 review, Part E). |
| S6 | Captions of Tables XIII, XIV, XV | — | Add a `Bold:` sentence to each, or remove the bold | These tables bold cells (0.852/0.783/0.242; 0.600/0.505; 0.20/0.36/0.35/−0.232) but never say what bold means. Every other table defines it. |

## Part O — Optional

- **§V-G grid:** 31 + 35 points share τ = 0.1, so the grid has **65** distinct thresholds
  (`curves.npz` confirms this). You could write "65 thresholds: 31 log-spaced … and 35 linear …,
  sharing 0.1".
- **Table VIII footnote:** "pholm ≈ 0" is informal. Quote the actual bound from the source
  (for example "< 10⁻ⁿ").
- **Abstract, first sentence of the method:** "from DICOM input to structured report on BraTS 2021,
  SSA and PED" could suggest that the external cohorts went through DICOM. Only the 40 test patients
  did. §XI states this, so it is not wrong, but a reviewer may still read it that way.
- **§III-B:** the claim that WT has a rejecting rank test with a CI that includes zero "in
  Tables IV and V" cannot be checked from Table IV, which shows no WT p-value. It is true, though
  (note 20: p_holm as low as 1.4×10⁻⁹). Either add the WT p_holm to Table IV or cite Table V only.

## What a reviewer is likely to ask (no edit needed, but have the answer ready)

1. **Does fixing the cohort and splitting it bias the held-out risk?** In a fixed 60-case cohort,
   "easy calibration draws" means "hard held-out cases". A scratch check (cohort as population, no
   held-out coupling) gives SSA TC at α = 0.05 a risk of about 0.037 instead of 0.055–0.065. That
   check is optimistic, though, because it scores calibration cases in-sample, and the truth lies
   between the two. The paper already calls the analysis counterfactual (§XI), so no edit is needed.
   Just do not claim more than "not restored in this design".
2. **What does nnU-Net do?** This is already flagged throughout (§VI-E, §XI). Gate A is at session 5.
3. **Is the PED label effect separable?** That needs [61]'s four-label release. This is already in
   §XI.

## Still open (carried over)

- **PDF by eye:** Table IV width, the equations, page breaks in Tables XI and XIII, and Fig. 5.
- **Placeholders:** the Zenodo DOI, weights and logits, and the Acknowledgment.
- **GPU-hours:** 186 is stale. Gate A s4 and s5 plus D3 add about 27 since then. Update at
  submission.
- **[61]:** the arXiv abstract does not mention the four-label release. Confirm it in the paper body
  before submission.
- **D3 and nnU-Net:** slot them in when they land (rev2 review, Part E).

## Order of work

1. M1 (6 edits), M3, M4 and M5. These are about 20 minutes and need no new evidence.
2. M2 (the script now exists): M2a–M2e.
3. S1–S6.
4. Export the PDF and do the visual checks.
