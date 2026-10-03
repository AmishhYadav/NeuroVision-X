# Review of rev1: `NeuroVision-X_IEEE_Paper_rev1.docx` (2026-10-03)

This reviews the revised draft against the first review (`paper_review_2026-10-03.md`). Each fix
gives **where it is**, the **exact text to find** (Cmd+F; it matches rev1 verbatim) and the **exact
replacement**.

**Verdict: a large improvement. Close to submittable once Part A is done.** Almost every item from
review 1 was applied correctly. All 16 equations are in and match the code. The 16 PENDINGs became
honest "not run" statements, and every number you added that I could check against the data checks
out (Part D).

Two things are new. **Rechecking your empty-ET paragraph against the per-case data turned up a
bigger effect than the paper states** (A1). One overclaim from the original draft was also missed in
review 1 (A2). Both need text changes.

---

## Part A — Must fix

### A1. The two empty-ET cases also inflate the seed-43 replication and the decomposition

Your new §VI-A paragraph is correct. Of the five test cases with no reference ET, the proposed model
predicts empty ET in two (Dice 1) and the baseline in none (Dice 0). That is 2/189 = 0.0106 of
0.0267, or 40%. I extended the check to every arm (project `statistics.py`, paired percentile
bootstrap, 10,000 resamples, seed 42, two-sided Wilcoxon):

| ΔET (test) | all 189 cases | 184 cases with reference ET | Wilcoxon p (184) |
|---|---|---|---|
| Proposed − baseline (seed 42) | +0.0267 | **+0.0165 [0.0125, 0.0211]** | 6.2 × 10⁻²² |
| Proposed − baseline (seed 43) | +0.0247 | **+0.0145 [0.0093, 0.0208]** | 4.9 × 10⁻¹⁸ |
| Capacity − baseline | +0.0055 | +0.0057 [0.0028, 0.0089] | 5.4 × 10⁻¹⁰ |
| Proposed − capacity | +0.0211 | **+0.0109 [0.0069, 0.0147]** | 6.6 × 10⁻¹⁹ |
| Ablation − capacity | +0.0189 | **+0.0140 [0.0087, 0.0208]** | 2.4 × 10⁻²⁰ |
| Proposed − ablation | +0.0022 | **−0.0032 [−0.0083, +0.0006]** | 0.023 (uncorrected) |
| Seed noise, proposed (43 − 42) | +0.0021 | +0.0022 [−0.0018, +0.0072] | 0.29 |
| Seed noise, baseline (43 − 42) | +0.0041 | +0.0042 [0.0016, 0.0075] | 6.2 × 10⁻⁶ |

Seed 42, 184 cases: **159 wins, 1 tie, 24 losses**, Hodges–Lehmann estimate **+0.0151**.

What this means:
- **Every direction holds**, so H1, H2 and the capacity argument stand. But each margin roughly
  halves, and the two cases are **the same two in both seeds**. The proposed model reliably predicts
  "no ET" for them in both seeds; the baseline never does. That is a real, replicated behaviour. It
  is not noise, but it is 2 cases carrying about 40% of the mean.
- "Six times the retraining shift" becomes **about four times** (0.0165 / 0.0042). "Five times" for
  proposed − capacity becomes **about 2.6 times** (0.0109 / 0.0042).
- On cases with reference ET, ambiguity conditioning has a **negative** point estimate (−0.0032). It
  is still inconclusive by your rule, because the CI includes 0. The content-only ablation is the
  numerically best arm there (mean ET 0.8868 against 0.8837). This *strengthens* the H2 null.
- R4 is now done. Delete it from the "not run" list.

This is a post-hoc subset analysis, so label it **exploratory** wherever it appears. Exact edits:

| # | Location | Find | Replace with |
|---|---|---|---|
| A1a | §VI-A | `By arithmetic, the mean difference over the 184 cases with reference enhancing tumour is therefore about +0.017; its interval, win, tie and loss counts and a Hodges–Lehmann estimate have not yet been computed (Section XI). The rank-based test is insensitive to two cases, so the direction of H1 does not rest on them, but its size does.` | `Over the 184 cases with reference enhancing tumour (exploratory), the difference is +0.0165 (95% CI 0.0125–0.0211; Wilcoxon p = 6.2×10−22; 159 wins, 1 tie, 24 losses; Hodges–Lehmann +0.0151), and at seed 43 it is +0.0145 (CI 0.0093–0.0208). The same two cases carry the same share at both seeds: the proposed model consistently returns no enhancing tumour for them and the baseline never does, a replicated behaviour whose weight in the mean is out of proportion to two cases. The direction of H1 does not rest on them, but its size does.` |
| A1b | §VI-A | `The architecture margin is about six times the larger of these two retraining shifts;` | `The architecture margin is about six times the larger of these two retraining shifts, and about four times over the cases with reference enhancing tumour;` |
| A1c | §VI-B, last sentence | `its margin over the capacity control (+0.0211) is five times the larger retraining shift.` | `its margin over the capacity control (+0.0211; +0.0109, CI 0.0069–0.0147, over cases with reference enhancing tumour) is two and a half to five times the larger retraining shift.` |
| A1d | §VI-B | `adds +0.0022 with a CI spanning zero, and the full model and the content-only ablation are inconclusive against each other on every metric (Table IV).` | `adds +0.0022 with a CI spanning zero (−0.0032 over cases with reference enhancing tumour, CI −0.0083 to +0.0006), and the full model and the content-only ablation are inconclusive against each other on every metric (Table IV).` |
| A1e | Fig. 3 caption, end | `Components sum to the total up to rounding.` | `Components sum to the total up to rounding. Over the 184 cases with reference enhancing tumour the components are +0.0057, +0.0140 and −0.0032 (total +0.0165; exploratory, Section VI-A).` |
| A1f | §X, first paragraph | `exceeds each model's retraining shift about six-fold and survives a capacity control.` | `exceeds each model's retraining shift about four- to six-fold and survives a capacity control, although two cases without enhancing tumour carry about 40% of its mean.` |
| A1g | §XI, the "not run" list | `the enhancing-tumour comparison restricted to cases with reference enhancing tumour, with win, tie and loss counts and a Hodges–Lehmann estimate; ` | delete this clause; it has now been run |
| A1h | Table IV (optional, recommended) | — | Add a column "ΔET, reference-ET cases (n = 184)" with the values above. Footnote: `Exploratory; not pre-registered.` |

Optional, for the abstract (249 words now, so it fits at 250 only if you trim elsewhere): after
`An architecture gain in distribution (+0.027 enhancing-tumour Dice)` add `, +0.017 on cases with
enhancing tumour,`.

### A2. "Pre-registered TOST" overclaims. *Missed in review 1, present since the original.*

`scripts/mc_comparison.py` begins: *"Secondary analysis, **explicitly outside** the pre-registered
Gate 1 and Gate 2 families."* Only the **margin** (0.03 AUROC) was fixed in advance, in
`execution_plan.md`, before any external MC map existed. Claims-ledger row C8 says the same:
"Margin fixed in advance".

| # | Location | Find | Replace with |
|---|---|---|---|
| A2a | Table VIII caption | `Entropy and MC-dropout (N = 10) are equivalent by pre-registered TOST (margin 0.03 AUROC):` | `Entropy and MC-dropout (N = 10) are equivalent by TOST against a margin of 0.03 AUROC fixed before any external MC-dropout map existed (secondary analysis):` |
| A2b | §VII | `Under shift it is equivalent to 10-sample MC-dropout by a pre-registered TOST at a 0.03 AUROC margin, at a tenth of the cost.` | `Under shift it is equivalent to 10-sample MC-dropout by a TOST at a 0.03 AUROC margin fixed in advance (a secondary analysis outside the registered families), at a tenth of the cost.` |

### A3. Three bracketed placeholders remain

| Location | Find | Action |
|---|---|---|
| §III-A | `[Zenodo DOI, to be minted at submission]` | Mint the DOI (Zenodo → GitHub integration → make a release) and paste it in |
| Data and code availability | `[Author to confirm: whether trained weights and saved logits are released, and where.]` | Decide. Recommended: `Trained weights for all surviving arms are released at [DOI]; saved logits (≈31 GB) are available on request.` |
| Acknowledgment | `[Author to add: faculty guide, department and any funding.]` | Fill in |

### A4. Venue and length are still undecided

rev1 is **16,356 words** (the original was about 14,500), still in IEEE format. The abstract is
**249 words**, which is OK for IEEE. MELBA (the target in `CLAUDE.md`) accepts this length. IEEE TMI
or JBHI (about 10 pages) do not. Decide before the next pass, because the A1 additions add length.

---

## Part B — Should fix

| # | Location | Find | Replace with / action |
|---|---|---|---|
| B1 | §V-G | `Its minimum calibration size at δ = 0.1 is independent of the model, 46 and 23 cases at α = 0.05 and 0.10 as computed in [33], against 19 and 9 for CRC.` | **Check that [33] states 46 and 23.** Its abstract reports WSR re-certifying six organs with 25 local cases at α = 0.10, which is consistent, but I could not confirm the exact pair. The tightest possible bound with zero observed loss, (1−α)ⁿ ≤ δ, gives 45 and 22, so 46 and 23 for WSR is plausible. If [33] does not print these numbers, write: `Its minimum calibration size at δ = 0.1, reached only if every calibration loss were zero, is about 46 and 23 cases at α = 0.05 and 0.10 for the betting bound, against 19 and 9 for CRC.` and drop "as computed in [33]" |
| B2 | §IX-B | `at bars of 0.5 and 0.9 Dice the silent-failure counts are 3, 6 and 31 and 51, 33 and 68 for test, SSA and PED,` | `the silent-failure rate for test, SSA and PED is 1.6%, 10.0% and 31.3% at a bar of 0.5 Dice and 27.0%, 55.0% and 68.7% at 0.9,` (the same numbers, verified, but as rates and no longer ambiguous) |
| B3 | Captions: Tables IV, VI, VIII, XII | `p_holm` (4 places) | Replace with *p* plus a subscript "holm", matching the body text |
| B4 | Table VIII caption and Table XV footnote | `3.7×10^−5`, `4.8×10^−7` | Replace with a real superscript (10⁻⁵, 10⁻⁷) |
| B5 | §XI | `needs the later BraTS-PEDs release that distributes the four sub-region labels separately.` | Add a citation to the later BraTS-PEDs release paper (the 2024 challenge description). The claim is currently uncited |
| B6 | Fig. 4 | — | Optional: add the CI error bars review 1 asked for. Filled versus hollow already carries the verdict, so this is cosmetic |
| B7 | Table IV | — | Optional: CIs in every row (review 1, C17). The caption note you added is acceptable instead |

---

## Part C — Cheap analyses still listed as "not run"

Section XI now lists seven or more analyses that were not run. That is honest, but a referee reads
it as "unfinished". All of these use only saved artifacts on CPU. **None needs a GPU:**

| Item | Cost | Why it matters |
|---|---|---|
| ~~R4 non-empty ET~~ | **done above** | — |
| R3 signed registration offsets (22 studies) | minutes | **The H9 verdict depends on it** |
| R8 confidence head WT | minutes | Removes the "§ withheld" entry |
| R6 SSA TC α = 0.05 at k ≥ 39 | minutes (saved logits) | Fills a hole in Table XI |
| R5 Dice and precision of the recalibrated masks | about 1 h (saved logits) | Turns "restored" into a usable claim |
| R7 pairwise-Dice detector and AURC | needs the MC passes; check whether `uncertainty/` survived the 2026-09-15 reclaim | Benchmark parity with Zenk et al. |
| R5 RCPS floors | about 1 h | Completes H6 |

If you want, I can spec these for `py-implementer` one at a time.

---

## Part D — Verified correct in rev1 (no action)

**Review-1 items applied correctly:** A1 equations, A2 placeholders (except the three in A3 above),
A3.2 and A3.3 hedges, A4 Index Terms and abstract length, B1–B9, C1–C22. One exception: C17 was met
by a caption note instead of CIs, which is acceptable.

**Equations (1)–(16)** match the code:
- (1) entropy in bits through softplus: `adaptive_fusion.py`.
- (5) weights 0.3, 0.05 and 0.1: `configs/training/default.yaml`.
- (6) halving weights normalised to one, predictions upsampled: `losses/segmentation.py`.
- (8) as min(Dice_WT, Dice_TC) ≥ 0.7: equivalent to the original definition.
- (10) and (12): `conformal.py` `fit_threshold`.
- (16) shrinkage: `ood.py`.

The OMML is properly structured (fractions, accents, sums, radical).

**New numbers you added, all checked against the data:**
- §VI-A empty-ET paragraph: five empty-ET cases; the proposed model is empty in two, the baseline in
  none. Undefined HD95 is 3 against 5. 2/189 = 0.0106, 40%, and +0.0165 on 184 cases.
  (`outputs/neurovision/eval_test/per_case_metrics.csv`,
  `outputs/eval_test_baseline_unet3d/per_case_metrics.csv`)
- §IX-B usable-bar sensitivity: silent failures 3/6/31 at 0.5 and 51/33/68 at 0.9. The test < SSA <
  PED ordering holds at 0.5, 0.6, 0.7, 0.8 and 0.9. (`outputs/error_budget/per_case_*.csv`)
- §V-I "twelve named checks": exactly the 12 `check=` IDs in `inference/input_qc.py`. Your
  correction is better than review 1's B7 suggestion of 7, which counted functions.
- Table VIII caption, "averaged over regions, error in any region": matches the `"ANY"` row in
  `scripts/detection_stats.py` and the region mean in `scripts/mc_comparison.py`.

**References** checked online:
- [32] Shahid, arXiv:2606.20115. Real. FeTS-2022, 1,251 subjects, 20 institutions, pooled CRC
  violates at 40% of sites, risk-curve shrinkage. Your new description is accurate.
- [33] Adhikary, Chabi and Mastmeyer, arXiv:2608.18193. Real, MICCAI-UNSURE 2026; 7 of 12 organs
  exceed α = 0.10.
- [34] Liu, Qiao, Zhang and Chen, arXiv:2608.10893. Real.

**Figures:**
- Fig. 2: E-codes removed; the QC passes match the text.
- Fig. 4: exactly the 7 violated cells are filled, matching Table X.
- Fig. 5: every below-floor point is hollow and agrees with the Table XI floors.
- Fig. 7: six labelled q values per curve, and the deployed q = 0.02 ring sits at the Table XIII
  registered values (0.894/0.953, 0.967/0.810, 0.737/0.329).

Sources: [arXiv 2606.20115](https://arxiv.org/abs/2606.20115) ·
[arXiv 2608.18193](https://arxiv.org/abs/2608.18193) ·
[arXiv 2608.10893](https://arxiv.org/abs/2608.10893v1)

---

## Order of work

1. A1a–A1g (about 20 minutes) → 2. A2 → 3. B1–B5 → 4. decide the venue (A4) →
5. the cheap analyses in Part C, R3 and R8 first → 6. A3 placeholders at submission time.
