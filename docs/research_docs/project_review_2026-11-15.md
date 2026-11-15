# Project review — an owner's view (2026-11-15)

Written after reading the whole record. It is the reason `docs/research/MasterPlan.md` exists; every
issue below names the MasterPlan item that fixes it. The earlier outside-view review is
`project_review_2026-09-24.md`.

## What this review is based on

What was read for this review: `master_plan.md` (all 803 lines), `claims_and_evidence.md` (C1–C29),
`project_review_2026-09-24.md`, `paper_review_rev4_2026-10-04.md`, the index of all 61 experiment
notes, `related_work.md`, `status_2026-09-27.md`, the pre-registration list (14 files), the full text
of `NeuroVision-X_IEEE_Paper.docx` (16.9k words, 18 tables, 61 refs), the full text of
`NeuroVision-X_Final_Project_Report.docx` (9.3k words), the git history (463 commits, 2026-07-31 →
2026-10-22), the uninterpreted nnU-Net conformal output on disk, and spot checks in the code.

Facts found that the plan below depends on:

- **Two finished runs have no note, no claim, no interpretation** (idle since 2026-10-22):
  nnU-Net conformal (`outputs/conformal/nnunet_gatea/`, results on disk) and D2 pooled training
  (`outputs/neurovision_pooled/checkpoints/best.pt`, never evaluated).
- **`outputs/` is gitignored and no result CSV is tracked in git.** Every per-case table, verdict
  file and family CSV behind C1–C29 exists only on one laptop (disk 86% full, 63 GiB free).
- **The IEEE paper is frozen at 2026-10-04**, eight days before Gate A retired the architecture
  claim. It still says the nnU-Net run "has not finished".
- **The semester report (2026-10-31) contradicts the paper and the code in several places** (list in
  Part A §4).
- MELBA takes **LaTeX only** (`melba.sty`, via Scholastica, ~20 pages). The paper is an IEEE Word file.
- UCSF-PDGM is **partly inside BraTS 2021** (387/501 reviewed for it), so it is not an unseen cohort.

---

## A0. Verdict in ten lines

1. The engineering and the honesty are far above a normal student project. The science is weak.
2. Twelve weeks, ~210 GPU-hours and ~130k lines produced **one real positive (a small Dice gain over
   a weak baseline) that the field's standard baseline then beat**, plus a set of measurements whose
   headline (the bound breaks under shift) was already published in 2026.
3. The central claim, "fails in proportion to the shift", is **not supported by the project's own
   data** and the paper already retracts it; the ledger, `CLAUDE.md`, the status doc and the semester
   report still assert it.
4. The external evidence is thin: one true covariate-shift cohort (SSA, n = 60, 2 significant cells
   of 6) and one cohort whose big number is mostly a **label-definition mismatch** (PED tumour core).
5. The audited model is not one anyone else uses, and it loses to nnU-Net. Findings about a wrapper
   around a non-standard model have low external value until they are shown on the standard one.
6. Every fitted thing (checkpoint, temperature, conformal threshold, QC model, gate cut-points) came
   from the same 187 validation cases, and the same 189 test cases were read by ~25 analyses, some
   of which then changed the deployed gate. There is no untouched holdout left.
7. The clinical front end does not work on real scans: 8 usable masks from 40 real studies (20%).
8. The pre-registration is real discipline but it is self-registration by one author with post-hoc
   amendments; it does not buy what external registration buys.
9. Effort went to the wrong place: a 3D twin, landing-page hero, molecular panel and launch video
   were built while the paper sat stale and finished GPU runs sat unread.
10. All of this is fixable, mostly on CPU from saved logits, by the April 2027 paper date. Part B is
    the plan.

## A1. What is right (kept, briefly)

Pre-registered decision rules and Holm families; the width-matched capacity control; the claims
ledger as a gate; `lessons.md`; refusal as a successful outcome; resume-safe training; saved fp16
logits that make re-scoring free; the willingness to retire the founding hypothesis and the
architecture claim in writing. None of this changes.

## A2. Major issues — science

Each has evidence and the MasterPlan item that fixes it.

**S1. The research question was chosen by elimination, not by design.** The project had three
identities (architecture → reliability → safety wrapper). Each pivot followed the evidence, which is
correct behaviour, but the final thesis is whatever was left standing, and it was then described as
the plan. Nine hypotheses (H1–H9) in one paper is a list of things tried, not a study. *Fix: one
question, stated before the new data arrives (B2, W5).*

**S2. The central claim is unsupported.** "Fails in proportion to the shift" needs a measure of
shift and more than two shifted points. There are three cohorts, the input-statistics OOD score
does not separate SSA from PED (60% vs 50.5%, overlapping), and PED differs in label definition as
well as images. The paper (§IV-A, §VIII-B) correctly says "no ordering can be established, and none
is claimed". The ledger (C14), `CLAUDE.md` thesis, `status_2026-09-27.md`, `related_work.md` §3 and
the semester report all still claim it. *Fix: retire the wording now (W0.3); earn it properly with
a controlled severity ladder (W3.2).*

**S3. The headline number is label shift, not distribution shift.** PED tumour core "3.5–11.5×
nominal" and "49.5% silent failure" are quoted first everywhere. 44 of 49 PED silent failures fail
on TC alone; the paediatric reference counts non-enhancing tumour as core (median 84% of core). After
a fine-tune the predicted TC/WT ratio moves 0.27 → 0.95. So the dramatic result is "the model was
never taught this label". That is a useful finding about benchmark reuse, but it is a different
finding from the one claimed. *Fix: split the claims — covariate shift (SSA, new cohorts, synthetic)
vs concept shift (PED) — and decompose PED with the four-label release (W3.3).*

**S4. True covariate-shift evidence is one cohort of 60.** On SSA the frozen bound is violated
significantly in 2 of 6 cells (TC at α 0.10 and 0.20). Everything said about "a new site" rests on
that. *Fix: more adult-glioma cohorts held as a lockbox (W3.1), per-institution analysis inside
BraTS 2021 via the FeTS partition (W2.2), synthetic severity ladder (W3.2).*

**S5. The audited model is the wrong model.** nnU-Net beats it on every Dice row (ET −0.0170,
lesion-wise −0.0309) at 1/24 the FLOPs per patch. The reliability findings are supposed to be
model-agnostic, yet the QC model, the gate, the error budget, the OOD switch-on rule and the
real-DICOM run exist for `neurovision` only. The nnU-Net conformal replication is on disk,
unread. Reading it for this review: test holds at all α; **PED TC still fails at 0.55–0.63**
(the label result replicates); SSA TC violated at α 0.10 (0.182, CI 0.135–0.237); PED WT violated
at α 0.05 only. So roughly 5 of 12 cells violated against 7 of 12 — a better model shrinks the
covariate-shift failure and does nothing for the label failure. That is the most interesting
sentence the project currently has and it is not written down. *Fix: W0.2, then make nnU-Net a
co-primary audited model (W2.6).*

**S6. No untouched data remains.** Validation (n = 187) selected every checkpoint and then fitted
the temperature, the conformal thresholds, SegQC (80% of it) and the gate cut-points (partly
in-sample, as the paper admits). Test (n = 189), SSA and PED were read by every analysis, and three
deployment decisions were made after reading them (conformal band → display-only, intended-use
rule added, OOD score not switched on). Reporting those as "labelled post-hoc variants" is honest
but it does not restore a clean estimate. *Fix: a lockbox cohort with the full analysis frozen
before first inference (W3.1); cross-fitted calibration (W2.1).*

**S7. The comparison that produced C1 is under-trained and half artefact.** Every run's best
checkpoint is the last or second-last validation (epoch 69 or 79 of 80): nothing converged, so the
architecture ranking is a ranking of speed-to-80-epochs at 64³. About 40% of the +0.0267 comes from
two empty-ET cases scored 0/1; on the 184 cases with ET the gain is +0.0165. The baseline has a
third of the parameters. nnU-Net beats the "matched" U-Net by +0.0437, so the recipe, not the
architecture, was the weak link — which the plan predicted on 2026-08-23 and then spent ~120 GPU-h
on the architecture anyway. *Fix: stop. Architecture becomes one short section and an appendix
(W5.2). No further GPU on it.*

**S8. The conformal design has gaps a conformal reviewer will find in minutes.**
- Loss is miss rate only. A mask of the whole brain satisfies it. Precision, Dice and volume of the
  conformal masks were never reported (recalibrated masks grew up to 3.6×).
- ET — the clinically central region — is excluded because 2.6% of cases have none. Defining the
  loss as 0 when the reference is empty is standard and removes the exclusion.
- α = 0.20 thresholds sit on the grid ceiling (0.95); nnU-Net's sit there too. The grid, not the
  rule, set them.
- Only the marginal (CRC) guarantee was fitted. The paper itself says a site needs the
  high-probability (RCPS) bound and that it "was not fitted here".
- One calibration draw. The realised-risk tables show one sample from the distribution the theorem
  is about.
- The certified mask is `sigmoid ≥ τ`; the deployed mask is post-processed (component filter,
  nesting). They are different objects.
*Fix: conformal v2 (W2.1).*

**S9. The refusal gate is not a risk-controlled gate.** REFUSE = the 2% quantile of validation
predicted-Dice, set "without labels". By construction it refuses ~2% in distribution regardless of
quality and has no stated target. Precision in distribution is 0.20 (16 usable refused to catch 4).
The strongest published case-level detector (pairwise Dice between ensemble members, Zenk et al.)
was never run although three `neurovision` models with saved logits on all cohorts already form an
ensemble. *Fix: failure-detection benchmark from saved logits (W2.3); a gate whose threshold is
chosen to control P(unusable | accepted) (W2.4).*

**S10. "Usable" has no clinical grounding.** Dice ≥ 0.7 on WT and TC is a convention the author
chose. No clinician, no reader study, no reference to inter-rater agreement or to the volume-change
thresholds response assessment actually uses. *Fix: add a volume-error endpoint and an inter-rater
anchor from the literature; keep 0.7 as the registered one (W2.5).*

**S11. The real-DICOM result measures the wrong thing and exposes a broken front end.** 16 of 40
real studies refused for thick slices — but 2D thick-slice T2/FLAIR is ordinary clinical practice
and BraTS itself resampled these same studies. Of 22 accepted, 8 usable. The −0.23 WT is scored in
atlas space against a reference registered by someone else's pipeline, so most of it is
registration disagreement (median 5.9 mm), not segmentation. The measurement cannot say how good
the front end is. *Fix: carry the reference into our space by rigid same-image registration and
score there; run the 16 refused studies with the refusal overridden; use all ~93 available test
patients (W3.4).*

**S12. Sample sizes are below what the claims need.** n = 60 and 99, single-seed arms, 22 accepted
DICOM studies, QC failure cells with a handful of failures. D0 was run although its own power
statement said the effect was unresolvable (23.7 GPU-h for a known "inconclusive"). The plan's own
principle 1 ("effect-size gate before compute") was not applied to D0 or TTA. *Fix: a written power
line in every new pre-registration, enforced as a launch check (B3 rule 6).*

**S13. Novelty is thin and the neighbours were never read.** The three 2026 preprints that
overlap most (Shahid; Adhikary et al.; Liu et al.) were read "at abstract level" on 2026-09-27 and
the full PDFs are still unread seven weeks later, while the paper cites specific numbers from them.
What is actually new is narrow: (a) label-definition mismatch as the dominant conformal failure and
its persistence on nnU-Net, (b) the end-to-end error budget with a silent-failure column, (c) the
front-end cost on original DICOM. *Fix: full read + citation audit (W5.1); build the paper on (a)–(c).*

## A3. Major issues — process

**P1. Results exist on one disk.** No CSV, verdict, threshold file or checkpoint is in git or any
archive. `capacity_control` was already lost once. A dead laptop ends the paper. *Fix: W0.1, today.*

**P2. Finished work is not closed.** nnU-Net conformal and D2 (≈30 GPU-h between them) have been
sitting since 2026-10-22 with "interpretation pending". The plan board was last ticked 2026-10-22.
P0.7 (a five-minute look at the 3D twin) has been open since 2026-09-18.

**P3. Scope and overhead.** 74k lines of library/scripts, 59k of tests, a 3D twin, hero mesh,
molecular panel with an empty AI slot, a launch video, 60+ `ambiguity_*` output directories, a
35k-token plan with session-by-session Kaggle logs inlined in a table cell. The 2026-09-24 review
said "freeze tool features". The frontend was then rebuilt (2026-10-09).

**P4. Self-registration with amendments.** 14 pre-registrations, 3+ amendments, several written
after related outputs were seen (Table A-I admits it), timestamps set by the committer, the paper
says "eleven". It reads as rigour and a hostile reviewer can discount all of it. *Fix: external,
immutable registration for everything new (OSF or a tagged Zenodo snapshot per protocol), one
protocol for the lockbox (W7.3).*

**P5. Ownership.** The author is new to deep learning; the code, the tests, the plans, most of the
paper and (despite the plan saying otherwise) the semester report were machine-drafted. In a viva
the author must defend conformal risk control, Holm families, TOST and the label-shift argument
alone. *Fix: W8 — the author writes Introduction and Methods; an explain-back gate on every claim.*

**P6. The plan's compute assumption was false for three months.** "200+ GPU-h on a modern card
(college cluster)" headed the plan; every run went to Kaggle. The GPU request due 2026-11-05 has no
recorded outcome.

**P7. Documents disagree with each other.** Ledger vs paper on "ordered by shift"; report vs paper
on inference time, decoder, seed-noise ratio; "eleven" vs fourteen pre-registrations; C1 quoted as
"13× the noise floor" in the report and "~6×" (or ~4× on ET-present cases) in the ledger.

## A4. Review of the semester report (`NeuroVision-X_Final_Project_Report.docx`, 2026-10-31)

Structure and tone are good. It is not yet a professional document because it contains errors and
omissions against the project's own record:

| # | Where | Problem | Correct |
|---|---|---|---|
| R1 | Abstract, §6.7, §7, §10.1 | "breaks … by an amount ordered by the size of the shift" / "in proportion to the shift" | Not established with three cohorts; PED TC is largely label definition. Use the paper's wording |
| R2 | §6.1 | "about thirteen times the seed noise floor" | ~6× the larger retraining shift; ~4× on the 184 ET-present cases |
| R3 | §6.1, abstract | Empty-ET caveat missing | ~40% of +0.0267 comes from two empty-ET cases; +0.0165 on ET-present cases |
| R4 | §4.1 | "U-Net decoder with attention gates" | Attention gates are off by default (`unet_decoder.py`); transposed-conv decoder |
| R5 | §4.4 | "about 4 s per case on a T4" | 11.9 s/case measured (note 15) |
| R6 | Table 3 | Validation "Selection only"; §3.1 "Nothing outside the training split was ever fitted" | Validation fitted temperature, conformal threshold, QC model, gate cut-points |
| R7 | §4.6, §1.6 | "Eleven" pre-registrations | 14 files + amendments; state the count and that some followed related outputs |
| R8 | §6.9 | Silent-failure rates without the region split | 8/8, 9/11, 44/49 fail on TC alone; PED largely the label |
| R9 | §6.2, abstract | "about 80% attributable to architecture" stated as fact | Descriptive; single-seed arms; width term is the size of seed noise |
| R10 | §5.5 | "Tests first for every component" | Tests were written with each module by the implementer; say that |
| R11 | §10.2 | nnU-Net conformal "interpretation pending"; D2 absent | Both must be reported (W0.2) |
| R12 | Front matter | No certificate, declaration, acknowledgement, AI-use statement, supervisor | Required by any college template |
| R13 | Fig. 15 | Example segmentations come from the baseline U-Net | Show the deployed model, plus one failure case per cohort |
| R14 | §8 | No ethics / data-licence paragraph; demo PROCEED case is a training patient (missing from limitations list in this version) | Add |
| R15 | References | 21 refs, several author-less ("Liu, Qiao, Zhang, and Chen"), 2026 arXiv items unverified | Verify each against source |

## A5. Review of the research paper (`NeuroVision-X_IEEE_Paper.docx`, rev of 2026-10-04)

**Summary of the submission.** An audit of one brain-tumour segmentation pipeline (dual-encoder
model, conformal miss-rate bound, QC model, refusal gate, DICOM front end) on BraTS 2021 and two
BraTS 2023 external cohorts, with pre-registered analyses and nine hypotheses.

**Scores (1–5).** Originality 2.5 · Methodology 3 · Results/evidence 2.5 · Writing 3.5 ·
Reproducibility 3 (code public; data, weights, results not archived).

**Recommendation as it stands: reject / major revision.** It cannot be submitted in this state:
its own text says the strong-baseline comparison is unfinished, and that comparison has since gone
against the model.

**Strengths.** Unusually candid; correct statistical rule (CI and Holm, TOST for equivalence);
the label-definition insight in §IV-A and §VIII-B; the four-outcome error budget; Table A-I; the
limitations section.

**Major concerns.**
1. **Stale.** No Gate A result (C29), no D3 (C27), no TTA (C28), no capacity rerun, no nnU-Net
   conformal, no D2. §VI-E, Table A-I and §XI are wrong as written. GPU-hours (186) stale.
2. **Wrong container.** IEEE two-column Word; MELBA needs LaTeX with `melba.sty` through Scholastica.
3. **Too many threads.** 16.9k words, 18 tables, H1–H9. The architecture study (H1–H4), the
   uncertainty comparison, the conformal study, the gate, the OOD score and the DICOM front end are
   six papers' worth of claims at one paper's depth. A reviewer will ask what the contribution is.
4. **The object of the audit is a model nobody uses** and that underperforms the default.
5. **External evidence.** n = 60 for the only clean shift; PED confounded by label definition (the
   paper says so, which then undercuts its own abstract).
6. **Conformal gaps** (A2-S8): FNR-only loss with no precision reported; ET excluded; grid ceiling;
   no high-probability bound; one calibration draw.
7. **Recalibration is same-cohort and counterfactual**, with no cost side (labels a site must draw
   are expert 3D segmentations — tens of hours).
8. **The gate is ad hoc** (quantile cut-points, partly in-sample), and the strongest baseline
   detector from the paper it cites [26] is absent.
9. **Real-DICOM measurement confounds registration with segmentation** and concerns training-
   distribution patients.
10. **Novelty statement is defensive** ("we do not claim to be first…") and rests on preprints the
    author has not read in full; every 2026 arXiv reference must be verified to exist and say what
    is cited.
11. **Pre-registration credibility**: self-hosted timestamps, three amendments, several protocols
    after related outputs.
12. Placeholders (Zenodo DOI, acknowledgement, weights), single author with no clinical co-author,
    AI-assistance statement present (good) but the verification burden it implies is heavy.

**Minor.** rev4 items M1–M5 and S1–S6 (check each was applied); table bold conventions; the
abstract is 250 words of numbers with no sentence a clinician could repeat; Fig. 1 architecture
detail belongs in an appendix; "audit" is used before it is defined.

**What would make it acceptable.** One question; two or three audited models including nnU-Net;
a lockbox cohort; the severity ladder; conformal v2; a principled gate with the ensemble baseline;
the corrected DICOM measurement; LaTeX; ≤ 20 pages plus appendix; archived results. That is W2–W5.

## A6. What should have been done differently

1. Train nnU-Net **first** (week 1), not week 10. Every architecture GPU-hour after 2026-08-23 was
   spent against a risk the plan had already named "moderate–high".
2. Hold out a lockbox before the first evaluation.
3. One question per paper; treat the architecture as a disposable instrument.
4. Read the related work in full before choosing the centrepiece.
5. Check label protocols of external cohorts before calling them "shift".
6. Back up results from day one.
7. Cap tool work to what a number needs.
8. Have the author write the methods section early; build only what the author can explain.

---

Sources checked: [MELBA for authors](https://melba.scholasticahq.com/for-authors) ·
[MELBA template repository](https://github.com/melba-journal) ·
[UCSF-PDGM dataset card](https://atlas.rsna.org/cards/33e4aa54-79dc-4d16-81fe-fd1afa023e5d) ·
[Erasmus Glioma Database](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8203723/) ·
[FeTS challenge data](https://fets-ai.github.io/Challenge/data) ·
[BraTS-Africa](https://arxiv.org/pdf/2305.19369v1.pdf)
