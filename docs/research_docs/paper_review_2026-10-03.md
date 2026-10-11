# Paper review and fix list: `NeuroVision-X_IEEE_Paper.docx` (2026-10-03)

This file lists every problem found in the draft paper, and for each one gives **where it is**, the
**exact text to find** (copy it into Word's Find, Cmd+F), and the **exact replacement**. Work from
the top down: Part A blocks submission, Part B contains factual errors, Part C is cleanup.

How the review was done. The full text of the .docx was extracted and every one of its roughly 120
numbers was recomputed or traced to its source: `claims_and_evidence.md`, `experiments.md`, the
pre-registrations, and the raw CSVs in `outputs/`. The equations were reconstructed from the code
that produced the results. Part E lists the sources that were checked and found correct, so you do
not re-check them.

**Verdict: major revision. Not ready to submit.** The science is strong and unusually honest. The
document has mechanical failures (missing equations, open placeholders) plus about 25 fixable
errors.

**Already fixed in the repo (commit of 2026-10-03), so the source documents now agree with this
review:**
- `preregistrations/preregistration_qc.md`: SSA·TC p_holm changed from 0.050 to **0.0504** in the
  result table, with a correction note saying the cell is inconclusive.
- `claims_and_evidence.md` C18: the same correction. C16: added a caveat that WT 0.477 (below
  chance) is probably a bug.
- `experiments.md` note 30 (line ~698) and `research/improvement_plan.md` (line ~103): the HD95 CI
  sign was corrected (see B2).
- **The .docx itself is untouched.** You said you would edit it, so every paper fix below is yours
  to apply.

---

## Part A — Blocking

### A1. All 16 equations are missing from the .docx

**Problem.** Every equation line in the file holds only two tab characters and a number:
"(1)" … "(16)". There is no equation object or image in front of the number. The file was produced
by LibreOffice 24.2 (see File › Properties), and that conversion dropped the math. Section V cannot
be read without them.

**Fix.** Put the cursor before each number and use Insert › Equation. Word's equation box accepts
LaTeX if you switch it to LaTeX mode (Equation tab › Conversions › LaTeX). Paste the LaTeX below,
then convert to "Professional". Every formula below was checked against the code that generated the
results.

| Eq. | Section, and the sentence just before it | LaTeX to insert | Source in code |
|---|---|---|---|
| (1) | V-A, "…With the binary entropy" | `H(z)=\frac{1}{\ln 2}\left[\sigma(z)\,\operatorname{softplus}(-z)+\big(1-\sigma(z)\big)\operatorname{softplus}(z)\right]` | `models/fusion/adaptive_fusion.py` ~l.540 (in bits) |
| (2) | V-A, "…a three-channel ambiguity signal, a gate and a residual update," | `s=\big[\,\lvert p_c-p_s\rvert,\;H(p_c),\;H(p_s)\,\big]` | same file, l.553 |
| (3) | (follows 2) | `g=\sigma\!\big(\varphi([F_c,\,F'_s,\,s])\big)` | `GateGenerator`, l.579–658 |
| (4) | (follows 3) | `F_{\text{out}}=F_c+\gamma\odot g\odot A(F_c,\,F'_s)` | l.888 |
| (5) | V-B, "The loss combines segmentation, boundary, confidence and branch terms," | `\mathcal{L}=\mathcal{L}_{\text{seg}}+0.3\,\mathcal{L}_{\text{bnd}}+0.05\,\mathcal{L}_{\text{conf}}+0.1\,\mathcal{L}_{\text{br}}` | `configs/training/default.yaml` l.77–101 |
| (6) | (follows 5) | `\mathcal{L}_{\text{seg}}=\sum_{k=0}^{2}w_k\,\mathcal{L}_{\text{DB}}\big(\hat y_k^{\uparrow},y\big),\qquad w_k=\frac{2^{-k}}{\sum_{j=0}^{2}2^{-j}}` | `losses/segmentation.py` l.170–195 (predictions upsampled, ↑) |
| (7) | (follows 6) | `\mathcal{L}_{\text{DB}}(\hat y,y)=1-\frac{2\sum\sigma(\hat y)\,y+\epsilon}{\sum\sigma(\hat y)+\sum y+\epsilon}+\operatorname{BCE}(\hat y,y),\quad \epsilon=10^{-5}` | `DiceBCELoss` |
| (8) | V-E, "…a case is usable if both whole tumour and tumour core are segmented to a Dice of at least 0.7," | `\text{usable}_i=\mathbb{1}\big[\operatorname{Dice}_{\text{WT},i}\ge 0.7\;\wedge\;\operatorname{Dice}_{\text{TC},i}\ge 0.7\big]` | QC pre-registration |
| (9) | V-G, "The loss is the miss rate" | `\ell_i(\tau)=\frac{\lvert G_i\setminus M_i(\tau)\rvert}{\lvert G_i\rvert}` | `uncertainty/conformal.py` `miss_rate` |
| (10) | V-G, "…CRC [9] selects" | `\hat\tau=\max\Big\{\tau\in\Lambda:\ \frac{n\,\hat R_n(\tau)+B}{n+1}\le\alpha\Big\}` | `fit_threshold`, l.460 and l.535–540 |
| (11) | V-G, "…and guarantees, for a new exchangeable case," | `\mathbb{E}\big[\ell_{n+1}(\hat\tau)\big]\le\alpha` | — |
| (12) | V-G, "…a fit is feasible only if (k Rmin + 1)/(k + 1) ≤ α, that is" | `k\;\ge\;\frac{1-\alpha}{\alpha-R_{\min}},\qquad R_{\min}<\alpha` | algebra from (10); matches Table XI |
| (13) | V-H, "…global pooling and a linear layer output a logit," | `\widehat{\operatorname{Dice}}=\sigma\big(f_\theta(x,\,m,\,H)\big)` | `models/qc.py` l.192 |
| (14) | V-I, "The gatekeeper (E5) returns" | `v=\max_{e\in E}\,v_e` | — |
| (15) | V-J, "…computes the Mahalanobis distance [36]" | `z=(u-\mu_{\text{train}})\oslash\sigma_{\text{train}}` | `analysis/ood.py` |
| (16) | (follows 15) | `D(u)=\sqrt{z^{\top}\hat\Sigma^{-1}z},\qquad \hat\Sigma=(1-\lambda)\,S+\lambda\,\operatorname{diag}(S)` | `ood.py` l.339–349 |

Notes on the symbols:
- **(1)** is the probability-form entropy −p log₂p − (1−p) log₂(1−p), rewritten in logits. The QC
  model's entropy input (`analysis/qc_inference.py` l.109) is the same formula **without** the
  1/ln 2 factor, so it is in nats. See C11.
- **(5)**: the weights 0.3, 0.05 and 0.1 are not stated anywhere in the paper text. With them in the
  equation they are, which closes a reproducibility gap.
- **(4)**: γ is a **per-channel** vector (`nn.Parameter` of shape (1, C, 1, 1, 1)), so ⊙ is
  broadcast multiplication. g is one value per voxel (`gate_channels: scalar`).

Afterwards, File › Export as PDF and confirm that all 16 equations render.

### A2. Sixteen PENDING/CONFIRM placeholders

Search for `PENDING`. Each must be removed or resolved before submission.

| # | Section | Find text (start) | What to do |
|---|---|---|---|
| P1 | III-A | `[PENDING: Zenodo DOI of the archived repository at submission]` | Archive the repo on Zenodo at submission and paste the DOI |
| P2 | III-B, last sentence | `the Hodges–Lehmann estimate [PENDING: post-review analysis R4].` | Run R4 and give the numbers, or delete the clause |
| P3 | V-G, end | `[PENDING: post-review analysis R5]` (after "δ = 0.1, which limits to δ…") | Run R5 |
| P4 | VI-A, end of the first paragraph | `[PENDING: post-review analysis R4 — ΔET on the test cases with non-empty reference ET…]` | Run R4: ΔET on cases with non-empty ET, the number of empty-ET cases scored differently, win/tie/loss counts, and Hodges–Lehmann with CI |
| P5 | VI-E | `PENDING: Gate A result — nnU-Net training is in progress…` | **Needed before submission.** Insert the verdict (survives / parity / retired) and add an nnU-Net row to Tables III and IV |
| P6 | VI-E, end | `[PENDING: post-review analysis R2; requires nnU-Net inference with saved probabilities]` | Do it, or move it to Future work |
| P7 | VII | `[PENDING: post-review check R8]` | **Needed.** See A3 |
| P8 | VIII-B | `[PENDING: post-review analysis R1 — if the four paediatric sub-region labels…]` | Do it if the labels exist; otherwise delete it and add one sentence to Limitations |
| P9 | VIII-C, first paragraph | `[PENDING: post-review analysis R6 — SSA TC at α = 0.05 evaluated at k ≥ 39…]` | Run it from the saved logits (cheap), then fill in Table XI's SSA·TC row |
| P10 | VIII-C, "Three cautions" | `[PENDING: post-review analysis R5 — RCPS floors and held-out violation rates…]` | Run R5 |
| P11 | VIII-C, "Three cautions", end | `[PENDING: post-review analysis R5 — Dice and precision of the recalibrated masks…]` | Run R5. Without it, "a site can restore the bound" has no clinical-usefulness number |
| P12 | IX-A, end | `[PENDING: post-review analysis R7]` | Run R7 (pairwise Dice of the MC passes, plus AURC) |
| P13 | IX-E | `[PENDING: post-review analysis R3 — signed per-axis offsets…]` | **Needed.** See A3 |
| P14 | Data and code availability | `PENDING: state whether trained weights and saved logits are released.` | Decide, then write one sentence |
| P15 | Appendix A, intro | `[PENDING: add the Zenodo DOI of the archived repository.]` | Same DOI as P1 |
| P16 | Acknowledgment | `PENDING: acknowledgements, faculty guide and any funding. CONFIRM WORDING:` | Name your guide and funding. Keep the LLM-assistance disclosure; check the venue's AI policy wording |

### A3. Three pending items that change claims, not only fill gaps

1. **P5, the nnU-Net comparison (Gate A).** H1 is the paper's only positive result, and it has never
   been compared with the field's default segmenter. Reviewers will ask for this first.
2. **P13 / R3, the registration offset.** §X ("On real data, the front end must be measured…") says
   *the largest single loss measured in this study is … the passage from DICOM to atlas space.* If
   the 5.9 mm anterior–posterior offset turns out to be a template-convention bug, H9 changes from
   "refuted" to "a fixed defect" and that paragraph is wrong. **Run R3 before submission.** If you
   cannot, soften §X now:
   - Find: `The largest single loss measured in this study is neither the model nor the gate but the passage from DICOM to atlas space:`
   - Replace with: `The largest single loss measured in this study, pending a check of the registration convention, is the passage from DICOM to atlas space:`
3. **P7 / R8, the confidence head WT AUROC of 0.477.** A head trained on correctness that scores below
   chance is almost certainly a channel or sign bug. Until it is resolved, quote only ET and TC:
   - §VII, find: `The trained confidence head loses to free entropy in every region on test data, and its WT value (0.477 against 0.904) lies below chance`
   - Keep that sentence only if R8 confirms the value. Otherwise replace it with: `The trained confidence head loses to free entropy on ET and TC (0.855 and 0.871 against 0.912 and 0.908); its WT output was found to [describe the R8 outcome]`.
   - §X, find: `disagreement and the confidence head are worse;` (this is fine as long as ET and TC carry it).

### A4. Choose the venue before formatting further

`CLAUDE.md` names **MELBA** as the journal target, but this .docx is in IEEE format. Measured on this
draft: **16 pages, about 14,500 words, abstract 265 words.**
- **MELBA**: LaTeX template, no hard page limit. Re-template; the length is acceptable.
- **IEEE TMI or JBHI**: about 10 pages, abstract 250 words or fewer, "Index Terms". The draft would
  need to lose about a third; Appendix A and Tables V and VII would go to supplementary material.

If you stay on IEEE:
- Find `Key Words—` and replace it with `Index Terms—`.
- Cut the abstract to 250 words or fewer. The cheapest cut is the sentence beginning
  `The audited dual CNN–Swin encoder gains +0.027…` (shorten it to `An architecture gain in distribution (+0.027 ET Dice) does not transfer.`).

---

## Part B — Factual errors (each one changes a number or a claim)

### B1. The SSA·TC QC loss is not significant (p_holm 0.0504, not 0.050)

Source: `outputs/qc_validation/cells.csv`, `p_holm = 0.0504`. Your rule in §III-B (and in the QC
pre-registration, l.137) is *p_holm **<** 0.05 and the CI excludes 0*, so this cell is
**inconclusive**. H7 stays **Mixed**, because PED·WT (p_holm 0.010) is still a significant
opposite-direction result.

| Location | Find | Replace with |
|---|---|---|
| Table XII, SSA·TC row, p_holm cell | `0.050` | `0.0504` and **un-bold** −0.1951 in that row |
| Table XII caption (add at the end) | `Bootstrap paired on the same resampled cases for both scores.` | `Bootstrap paired on the same resampled cases for both scores. Bold: significant by the rule of Section III-B; SSA·TC (p_holm 0.0504) misses it narrowly. SSA·TC n = 59.` |
| §IX-A, first paragraph | `In the same five-cell family, however, it loses significantly on SSA TC (−0.195) and PED WT (−0.190).` | `In the same five-cell family, however, it loses significantly on PED WT (−0.190), and on SSA TC (−0.195) by a margin that narrowly misses the registered bar (pholm = 0.0504).` |
| §IX-A | `H7 is therefore mixed, and the three cells are always reported together.` | (keep as is; still correct) |
| §X, "Single-pass entropy is a hard baseline" paragraph | `the QC model wins in one external cell and loses in two.` | `the QC model wins in one external cell, loses in one and narrowly misses significance in a third.` |
| Table XVI, H7 row, Verdict | `Mixed (1 win, 2 losses)` | `Mixed (1 win, 1 loss, 1 borderline)` |

### B2. HD95 under shift: the point estimate lies outside its own CI

Source: `outputs/compare_shift/neurovision_vs_baseline_pooled_ssa_ped.csv`. `hd95_ET`
`mean_diff = −1.459` (proposed − baseline, lower is better), CI [−3.942, 0.700], **n = 136** (23 PED
and SSA cases have no ET, so HD95 is undefined for them). The old note printed the "improvement"
sign with the difference's CI.

- §VI-D, find: `at the pooled n = 159 the advantage vanished (ET +1.46 mm, CI −3.94 to +0.70)`
- Replace with: `in the pooled external sample the advantage vanished (ET HD95 −1.46 mm, proposed − baseline, CI −3.94 to +0.70, n = 136 cases with non-empty ET)`

### B3. Per-case storage size does not multiply out

§IV-B says `about 38 MB per case and 34.2 GB in total`. 38 MB × 1,251 = 47.5 GB. On disk,
`data/preprocessed/brats` is **34 GB for 1,251 cases, about 27 MB per case**.
- Find: `about 38 MB per case and 34.2 GB in total`
- Replace with: `about 27 MB per case and 34 GB in total for the 1,251 in-distribution cases`

### B4. Contribution 4 contradicts H7

QC beats entropy on PED·TC, but contribution 4 says no signal does.
- §I, find: `Fourth, it measures four uncertainty and quality-control signals against single-pass predictive entropy, in and out of distribution, and finds none that beats it as an error localiser (Section VII).`
- Replace with: `Fourth, it measures four uncertainty and quality-control signals against single-pass predictive entropy, in and out of distribution: none of the three voxel-level signals beats it as an error localiser, and a learned case-level quality-control model beats it in one external cell and loses in another (Sections VII and IX-A).`

### B5. "Decent precision" is false in distribution

Refusal precision is 4/20 = **0.20 on test**, 1/2 on SSA, and 23/26 = 0.88 on PED.
- §IX-B, find: `The gate has decent precision and poor recall, and it is blind to cohort-level shift.`
- Replace with: `The gate's refusals are mostly wrong in distribution (precision 0.20), decent only on PED (0.88), its recall is poor everywhere, and it is blind to cohort-level shift.`

### B6. The OOD row of Table XIII is measured against a different gate than the label suggests

Test 0.884 = 0.915 − 6/189, which is the **band display-only** gate. Against the registered gate the
same change gives 0.820.
- Table XIII footnote, find: `‡Not adopted (Section IX-D).`
- Replace with: `‡Added to the band display-only gate; not adopted (Section IX-D).`

### B7. "Twelve" input-QC checks

§V-I says the input QC `runs twelve checks in two stages` and then lists nine. The code
(`src/neurovision/inference/input_qc.py`) has seven `check_*` functions: sequence completeness,
geometry consistency, spacing (voxel spacing plus anisotropy), finite values, intensity sanity, shape,
and brain mask (brain volume plus skull remnants). Fig. 2 lists still fewer.
- Count the individual findings that `run_input_qc` can emit, then make the text, Fig. 2 and the
  number agree. If you count functions: find `runs twelve checks in two stages` and replace it with
  `runs seven checks in two stages`.

### B8. Test count

§V-L says `about 2,450 automated tests`. `pytest --collect-only` on 2026-10-03 collects **2,596**
(this includes skipped tests).
- Find: `The code base carries about 2,450 automated tests`
- Replace with: `The code base carries about 2,600 automated tests`

### B9. SegQC architecture text does not match the code

`models/qc.py`: the first conv block has **stride 1**, so only three are strided. The head is
**two linear layers** (128 → 64 → 1, LeakyReLU, dropout before the last), not one.
- §V-H, find: `Four strided Conv3d–GroupNorm–LeakyReLU blocks (16, 32, 64 and 128 channels), dropout 0.1, global pooling and a linear layer output a logit,`
- Replace with: `Four Conv3d–GroupNorm–LeakyReLU blocks (16, 32, 64 and 128 channels; stride 2 from the second block on), global average pooling and a two-layer head (128 → 64 → 1, dropout 0.1) output a logit,`

---

## Part C — Consistency, figures, tables and wording

| # | Location | Find | Replace with / action |
|---|---|---|---|
| C1 | Fig. 5 (image) and caption | — | Several points sit **below their floor** but have no cross: α=.05 panel SSA·TC k=20 (floor 39), SSA·WT k=20 (floor 22), PED·WT k=20 (floor 26); α=.10 panel SSA·TC k=10 (floor 13), PED·WT k=10 (floor 11). Those values are feasibility-conditioned and biased, which is your own caution #1. **Regenerate the figure** with below-floor points greyed or hollow, and change the caption's `Crosses mark cells not restored:` to `Hollow markers lie below the cell's floor (feasibility-conditioned, biased); crosses mark cells not restored:` |
| C2 | Fig. 7 caption vs §IX-B | `On PED no operating point exceeds 0.5.` | `On PED no operating point exceeds 0.44.` Also label the q values on all three curves (only PED has them now) |
| C3 | Fig. 4 caption | `Points above the diagonal exceed the bound; the PED TC points lie far above it.` | `Points above the diagonal exceed nominal α; filled markers are violated by the rule of Section III-B (CI lower bound above α). The PED TC points lie far above it.` Regenerate with CI error bars and filled/hollow markers, and cap the TC panel's x-axis at 0.25 |
| C4 | §III-A | `Eleven pre-registrations and two measurement protocols were written.` | `Eleven pre-registrations, one amendment and two measurement protocols were written (Table A-I).` (Table A-I has 14 rows) |
| C5 | §V-C | `The proposed model's best checkpoint is at epoch 69 (validation mean Dice 0.8938).` | `The proposed model's best checkpoint is the epoch-70 validation (of 80; validation mean Dice 0.8938).` (the code counts from 0) |
| C6 | §V-D | `replayed logits reproduce the committed per-case metrics to within 10−17 Dice.` | `replayed logits reproduce the committed per-case metrics to floating-point precision (≤ 10−15 Dice).` (10⁻¹⁷ is below float64 resolution at 0.9) |
| C7 | §IX-E, second paragraph | `has a median length of 5.9 mm, mostly along the anterior–posterior axis and up to about 15 voxels.` | `has a median length of 5.9 mm, mostly along the anterior–posterior axis, and reaches about 15 mm.` |
| C8 | §VIII-B | `(Rmin, Section VIII-C)` | `(Rmin, Section V-G)` |
| C9 | §V-A | `18.85 M in the CNN, 2.04 M in the Swin encoder, 1.05 M in fusion and 12.93 M in the decoder.` | `18.85 M in the CNN, 2.04 M in the Swin encoder, 1.05 M in fusion, 12.93 M in the decoder and 0.04 M in the heads and probes.` (the parts now sum to 34.91 M) |
| C10 | Fig. 1 caption | `a scalar gate g modulates` | `a per-voxel scalar gate g modulates` |
| C11 | §V-F | `single-pass predictive entropy (1) of the final output;` | `single-pass predictive entropy of the final output (the form of (1));` and add to §V-H: `the entropy channel H is the same quantity in nats.` |
| C12 | §V-J | `under a sample covariance S shrunk towards its diagonal [59], with λ = 0.1 fixed in advance.` | `under a sample covariance S shrunk towards its diagonal with a fixed intensity λ = 0.1 (in the spirit of [59], which instead estimates the intensity), set in advance.` |
| C13 | Throughout | `(E1)`, `(E2)`, `(E3)`, `(E5)` in §V-I; `E1`/`E2`/`E3`/`E5` in Fig. 2 | Delete them. They are internal build labels, and E4 is missing, so readers will look for it. Regenerate Fig. 2 without them |
| C14 | §VI-E and Table A-I | `(Gate A)`, `Strong baseline, nnU-Net (Gate A)`, `Error localisation (Gate 2)` | Replace with descriptive names: `the nnU-Net comparison`, `Error localisation`. Reviewers have no copy of your plan board |
| C15 | All `R1`–`R8` mentions | — | They disappear once the PENDINGs are resolved. Do not leave any "R#" labels in the final paper |
| C16 | Table VIII | — | (a) The SSA and PED rows have no region; write which one (pooled? WT?) in the row label. (b) Add to the caption: `Bold: best in row. —: not measured.` (c) Say why the confidence head is not measured externally and MC-dropout is not measured on test |
| C17 | Table IV | — | (a) Give a 95% CI in every ΔET cell (only 4 of 8 rows have one). (b) Add to the caption: `Capacity − baseline meets the rule but is of the order of the baseline's own retraining shift (+0.0041) and is not attributed to width (Section VI-B).` |
| C18 | Table V | WT row: `+0.0300 [−0.0021, 0.0620]` with `9.6×10−6` | Add a footnote: `WT: the Wilcoxon test rejects but the bootstrap mean CI includes zero; by Section III-B the comparison is inconclusive. The two tests address different estimands (Section III-B).` |
| C19 | File › Properties › Title | `…Dual-Encoder Segmentation Model for Brain-Tumour MRI under Graded Distribution Shift (A Pre-Registered Study…)` | Replace with the current title. "Graded" contradicts §IV-A's "no ordering of shifts … is claimed" |
| C20 | Title (24 words, long for any venue) | `Where the Guarantee Breaks: Conformal Risk Control and Refusal Gating for Brain-Tumour MRI Segmentation, from DICOM to Report, under Distribution Shift` | Suggested: `Where the Guarantee Breaks: Conformal Risk Control and Refusal Gating for Brain-Tumour Segmentation under Distribution Shift` |
| C21 | §IX-B, Fig. 7 sentence | `Sweeping the predicted-Dice refusal quantile from 0.01 to 0.30 (Fig. 7)` | Add `with the registered gate's other signals unchanged`. Also state how the test, SSA and PED curves end up with 5, 6 and 5 points |
| C22 | Table XIV caption | `Cut-points: validation 90th / 98th percentiles.` | `Cut-points: validation 90th / 98th percentiles (7.557 / 9.935). AUROC for flagging studies with an unusable mask.` |

---

## Part D — Reviewer-level points (what a MELBA or TMI referee will raise)

Scores (out of 10): Originality 6 · Methodology 8 · Results 7 · Writing 4 now, 7–8 after Part A ·
**Overall 6, major revision.**

### D1. Originality
- **Strengths.** A pre-registered audit of a full DICOM-to-report pipeline. The label-definition
  finding (PED TC fails at 3.5–11.5× and R_min = 0.356, so no threshold can repair it). An error
  budget measured through a deployed gate rather than as an isolated AUROC.
- 🔴 **Major.** A referee may call the PED TC failure *true by construction*: the label means
  something else, so an adult model must miss it. Lead with what is new: its *size*, the WT
  comparison showing that disease and anatomy also contribute, and R1, the sub-region decomposition.
  R1 is what turns "expected" into "measured".
- 🔴 **Must check by hand.** References **[32] arXiv:2606.20115, [33] arXiv:2608.18193 and
  [34] arXiv:2608.10893** are 2026 preprints and are the closest prior work. Open each one and
  confirm the ID, authors and title. The draft was written with LLM assistance, and one wrong or
  fabricated citation is enough for a desk rejection. All other references matched known
  bibliographic records.

### D2. Methodology
- **Strengths.** Paired bootstrap with Wilcoxon and Holm, TOST for equivalence, a measured seed
  noise floor, verdict labels fixed in advance, post-hoc gate variants kept separate, and p-values
  stated as conditional on the trained weights.
- 🔴 **Major.** There is no strong baseline (A3.1).
- 🟡 **Minor.** The usable bar (Dice ≥ 0.7) drives every number in the error budget. Add a
  sensitivity row to Table XIII at 0.6 and 0.8. It needs no inference; it comes from the saved
  per-case Dice.
- 🟡 **Minor.** The SegQC cut-points are partly in-sample (stated in the paper already). Say how
  much the test-row outcome counts would move with cut-points fitted on the held-out 20%.

### D3. Results
- **Strengths.** Negative results carry full weight (H2 null, H3 not supported, H4 refuted,
  augmentation not demonstrated). Table XVI is excellent.
- 🔴 **Major.** R5: a bound restored by a mask three times the default size is not clinically useful
  as it stands. "A site can restore it" needs Dice and precision at the restored operating points.
- 🟡 **Minor.** The Wilcoxon and mean-CI disagreement (C18) should be explained once, in §III-B,
  with R4.

### D4. Writing
- **Strengths.** Precise prose, hedging calibrated exactly to the evidence, and memorable framing
  ("information is not utility", "the gate sees masks, not cohorts").
- 🔴 Part A1 (equations). 🟡 Title and abstract length (A4, C20). 🟡 Internal jargon (C13–C15).

---

## Part E — Checked and correct (no action)

- **Table XIII.** Every CA/SF/OR/CR row sums to n, and every P(·) equals its count ratio (for example
  161/189 = 0.852, 47/58 = 0.810, 24/73 = 0.329, 173/181 = 0.956, 26/79 = 0.329). The no-gate rates
  0.937, 0.800 and 0.273 are correct; so are the silent-failure rates 4.2%, 18.3% and 49.5%. The gate
  precision and recall sentences for PED (23/26 and 23/72) are correct.
- **Table IX.** All ratios are correct. The τ̂ values lie exactly on the stated grid (10^−1.8 =
  0.0158; 0.1 + 0.025k gives 0.600 and 0.725). The admissible α = 0.20 risk is
  (188·0.2 − 1)/187 = 0.196.
- **Table X.** All twelve ratios are correct. Exactly seven cells are bold, matching "seven of twelve".
- **Table XI.** The floors match (12) for one R_min per cell (SSA WT ≈ 0.007, SSA TC ≈ 0.026, PED WT
  ≈ 0.014). The values 19, 9 and 4 at R_min = 0 are correct.
- **Tables III and IV.** Every delta matches Table III to within rounding. The decomposition
  0.0055 + 0.0189 + 0.0022 = 0.0266 ≈ 0.0267. "Six times" (0.0267/0.0041) and "five times"
  (0.0211/0.0041) are correct, and so is the seed-averaged +0.0257. HD95 WT = 7.09 in two rows is
  real (`experiments.md` l.48–50).
- **Table VII.** All eight means recompute. The duplicate 0.0218 for the baseline is real
  (`experiments.md` l.624–625).
- **Table VIII.** The deltas 0.157 and 0.056, and the TOST differences +0.0214 and −0.0129, are
  correct.
- **Table XV.** 16 + 2 + 22 = 40; 8/40 = 0.20, 8/22 = 0.36 and 14/40 = 0.35. The exact Wilcoxon
  p with 22 of 22 negative is 2/2²² = 4.8 × 10⁻⁷. "Five-fold" (0.232/0.05) is correct.
- **Table XIV.** 18/189, 36/60 and 50/99 are correct.
- **Methods text against config.** Learning rate, weight decay, warm-up, clipping at 5.0, 64³
  patches, 4 crops at 1:1, augmentation, 50% Gaussian overlap, a 50-voxel minimum component, the
  τ = 0.5 threshold, the 2.6% empty-ET share (5/189) and the bound 1/189 = 0.0053 all match
  `configs/` and the code.
- **References.** IEEE first-citation order is respected. [1]–[31] and [35]–[60] match real
  publications (venue, volume and pages look right). Only [32]–[34] need a manual check (D1).

---

## Order of work

1. A1 equations → 2. A4 venue decision → 3. Part B (B1–B9, about 30 minutes of find/replace) →
4. Part C text items → 5. regenerate Figs 2, 4, 5 and 7 (C1–C3, C13) → 6. run R3, R8, R5, R4 and
R6 from saved logits (none needs a GPU) → 7. the nnU-Net verdict once Gate A finishes → 8. verify
references [32]–[34] → 9. clear every PENDING → 10. export the PDF and read it once end to end.
