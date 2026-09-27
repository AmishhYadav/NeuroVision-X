# Project review: an outside view (2026-09-24)

This review was written after reading `master_plan.md`, `claims_and_evidence.md`,
`experiments.md` (notes 41–49), `tool_completion_log.md`, `simplification_review.md`, the
pre-registrations and the git history. It also ran the test suite: **2,230 passed and 35 skipped
in 60 s, on 2026-09-24.** It is an opinion document. It changes nothing in the plan by itself.

---

## 0. The short answer

1. **As a research project it is strong. As a "clinical tool" it is not, and it cannot become one
   with these resources.** The value you can realistically get from it is a rigorous paper plus an
   open research prototype. Those two are worth finishing.
2. **Progress is real and fast.** In eight weeks (first commit 2026-07-31) you have 334 commits, 11+
   pre-registered comparisons resolved, and a pipeline that takes a DICOM study to a gated decision
   in about 6 minutes on a laptop. Most of the answers were "no". That is fine, because the claims
   table records them honestly.
3. **Roughly half of the realistic goal is done.** The tool is about 90% done. The experiments are
   about 65% done. The paper is about 10% done, because no manuscript exists yet (§3).
4. **Three things are logically broken right now** (§6). The thesis sentence in `CLAUDE.md` claims
   something the data refuted. The one positive result has never been tested against a strong
   baseline. The novelty claim for the main finding has not been checked against the literature,
   and some 2026 preprints come close to it.
5. **The best next step costs no GPU time.** The per-cohort ("Mondrian") recalibration arm was
   pre-registered in `preregistration_conformal.md` and never run. It asks whether a new site can
   restore the guarantee with a few of its own labelled cases. It runs from saved logits in minutes,
   and it turns the project's main negative finding into a constructive one (§4, item 1).

---

## 1. What this project is now

The project has changed identity three times, and each change followed the evidence:

- **Architecture claim (July–August).** "A disagreement-conditioned fusion gate makes a better
  segmenter." It is dead: the P2 null showed the content-only gate matched the full model.
- **Reliability claim (August).** "The model is more reliable, not just more accurate." This is
  also dead. Calibration, boundary accuracy and uncertainty all came back within noise.
- **Safety wrapper (late August to now).** "A pipeline that bounds its own error and refuses what it
  cannot handle." This part has been measured. The bound holds in distribution. It breaks under
  shift, and it breaks more the larger the shift is. The refusal gate catches some individually bad
  masks but cannot see cohort-level shift (C14, C20–C22).

Pivoting on evidence rather than defending a hypothesis is the correct behaviour. The cost is that
the codebase still carries three projects' worth of machinery, and the story has to be rewritten
around the third one.

---

## 2. Is this a good idea?

### As a clinical product: no

- **Regulation.** Software that segments a tumour for clinical use is a regulated medical device.
  Cleared products already exist (NeuroQuant, Neosoma, Cercare, VUNO; see `master_plan.md` §3).
  Nothing in this project's resources (one person, Kaggle GPUs, no clinical partner, no radiologist
  reviewer; see the memory note on the missing neuroanatomy reviewer) can take it through that
  process.
- **The pipeline is mostly assembled from existing tools.** The front end is brainles-preprocessing,
  HD-BET, ANTs and dcm2niix. Open alternatives do the same assembly (BraTS Toolkit, the BrainLes
  suite, nnU-Net). The part that is new is the refusal gate, the conformal layer and the error
  budget, and those are research contributions, not product features.
- **It fails where a tool would be needed most.** On paediatric scans, only 24% of studies get a
  usable mask, and 49.5% are *accepted with an unusable mask* (note 47). A clinical tool with that
  silent-failure rate on a population it does not refuse would be unsafe. The project's own numbers
  say so, and they deserve credit for saying it.

### As research plus an open prototype: yes

- "Where does a safety wrapper around a segmentation model break under real distribution shift,
  measured end to end?" is a real, publishable question. Most papers benchmark each stage in
  isolation.
- The finding is graded: SSA ~1.1× nominal risk, PED WT 1.4–1.9×, PED TC 3.5–11.5×. A graded result
  like that is more useful than "it works" or "it collapses".
- The methodology findings stand on their own: the circular calibration mask, voxel-weighted
  boundary shares, and the mirrored-atlas Dice blind spot.
- As a portfolio piece, it already shows more rigour than most published student work.

---

## 3. Where it stands: percentages against four finish lines

"Percent done" depends on which finish line you mean, so here are four.

| Finish line | Done | What remains |
|---|---|---|
| **A. Open research prototype**: DICOM in, gated result, report and export, runs on a laptop | **~90%** | T0.4; a human looking at the 3D twin; a cohort-shift or intended-use gate input; a decision on `conformal_band`; wording that says the guarantee is a cohort average; release packaging (MkDocs, DOI) |
| **B. The experiments a paper needs** | **~65%** | Gate A (strong baseline); per-cohort recalibration (never run); D3 fine-tune; an OOD signal; a literature check; real-DICOM validation with n > 2 |
| **C. Paper submitted** | **~10%** | Everything except the claims table, the figure notebook (`09_paper_figures.ipynb`) and a few LaTeX tables. There is no manuscript, no related-work section (`related_work.md` was planned in `contribution.md` and never written), and no chosen venue or deadline |
| **D. Used on real patients** | **not a meaningful %** | Outside this project's scope and resources: regulatory clearance, a clinical partner, prospective validation |

Weighting A, B and C by effort (roughly 25 / 40 / 35), the realistic goal is **about 50% done**. One
of the plan's six months has passed, so the pace is ahead of schedule. The warning is that the
remaining half is the hardest part: GPU-bound experiments and writing, which are exactly the two
kinds of work that have been deferred most often.

---

## 4. What is left, in priority order

Ordered by value divided by cost.

1. **Run the pre-registered Mondrian / per-cohort recalibration arm (B2 secondary).** It needs CPU
   only, uses saved logits, and takes minutes. It asks: if a new site labels half of its cohort, does
   the guarantee come back? Extend it with a curve over k = 5, 10, 20, 30 local cases, as a written
   amendment *before* running. Any outcome is useful:
   - "~15 labelled local cases restore the bound" is a deployable recipe.
   - "PED TC is infeasible at any threshold" proves the model must be retrained there, and motivates
     item 4.

   This matches where the 2026 literature is heading ("local re-certification"; see §6.3). Label it
   as a counterfactual, as the pre-registration already requires.
2. **Validate the clinical path on real DICOM with ground truth.** Today n = 2 with no truth, which
   note 47 calls "an anecdote". The UPENN-GBM collection on TCIA ships tumour segmentations, and the
   clinical path already registers to SRI24. That makes it possible to measure clinical-path Dice on
   20–50 real studies at ~6 min each, all on CPU. It may also unblock T0.4 without Kaggle. **First
   check the overlap with the training split** (§6.4).
3. **Resolve Gate A.** This first needs an honest answer about the college GPU (§7). If that card is
   not usable within about two weeks, write the amendment note 46 already describes: a reduced
   `nnUNetTrainer_250epochs` arm (~19 GPU-h on a T4) that can only return PARITY or RETIRED. Then run
   it.
4. **D3: fine-tune on the external cohorts, with cross-fitting.** Fine-tuning on a slice of SSA
   (n = 60) or PED (n = 99) shrinks the test set. Two-fold cross-fitting fixes that: fine-tune on
   half, test on the other half, swap, and every case still gets scored. Do PED, not just SSA,
   because PED is where the failure is. Cost is about 8 GPU-h per fine-tune. Pre-register it first.
5. **Give the gate a cohort-level signal.** `ood_score` is still a placeholder.
   - The cheapest honest version is an **intended-use rule**: adult glioma only, with patient age read
     from the DICOM header. Cleared products work this way.
   - Report it as *scoping* and never as detection. A rule that refuses every paediatric scan scores
     100% on PED by construction, so it proves nothing about detection.
   - An image-level OOD score (for example, a Mahalanobis distance on encoder features) can follow.
6. **Decide `conformal_band`.** In distribution, all 12 of its refusals were over-refusals. Disable it
   as a refusal signal, keep it as a displayed quantity, and record the decision.
7. **Write the paper**, in parallel with items 1–6 rather than after them. Sections that rest on
   final results can be drafted today: the pipeline, conformal in and out of distribution, the error
   budget, and the methodology traps.
8. **Two author actions of five minutes each, open since 2026-09-18.** Look at the 3D twin on job
   `9c2cc294`. Accept the Kaggle RSNA-MICCAI rules so T0.4 can use the planned patient.

**Recommend cutting or keeping parked:**
- **Phase F (IDH).** It is 3–4 weeks of work and a second project; the paper does not need it.
- The remaining multi-seed grid beyond what Gate A needs.
- B3 (deep ensemble). Two seeds make a weak ensemble, and entropy already wins.
- Any further tool features.

If a fourth cohort is wanted for G4, UCSF-PDGM's structural scans plus segmentations can be used
without the IDH model. Check its overlap with BraTS 2021 first (§6.4).

---

## 5. What is being done right

- **Pre-registration before measurement, Holm correction, and decision rules fixed in advance.** In
  a solo project this is rare, and it is the reason the negative results are publishable rather than
  embarrassing.
- **The width-matched capacity control.** Most architecture papers never run one.
- **`claims_and_evidence.md` as a gate on what may be written.** It has already stopped at least five
  overclaims.
- **`lessons.md`.** The circular mask, the fp16 epsilon, the mirrored atlas and the MC-dropout
  `eval()` recursion are real traps, and writing them down is part of the contribution.
- **The error budget reports the silent-failure column next to the success rate.** That column is
  the honest part of that table.
- **Good cuts.** MGMT, survival, midline shift and the eloquence verdict were each cut for a stated
  reason.
- **Engineering hygiene.** Resume-safe training, config-driven paths, a CPU test suite that runs in
  one minute, and a reproducibility document that lists which artifacts are caches.
- **A refusal is treated as a successful outcome, not an error.** That is correct design for a
  safety pipeline.

---

## 6. What should change: the logic checks

### 6.1 The thesis sentence contradicts the evidence

`CLAUDE.md` states the thesis as a pipeline that "is safe to deploy on data it was not trained on".
`claims_and_evidence.md` lists "the refusal gate makes the pipeline safe to deploy under distribution
shift" as **do not write**, citing note 47. The file that every session reads first is asserting a
refuted claim.

A replacement that matches the evidence:

> A segmentation model wrapped in a distribution-free error bound and a refusal gate: we show the
> bound holds in distribution, fails under shift in proportion to the shift, and that the gate
> cannot see cohort-level shift. We then measure what it takes for a new site to restore it.

The last clause depends on §4 item 1.

### 6.2 The one positive result has never met the real bar

The headline is ET +0.0267 against a MONAI U-Net trained at 64³ for 80 epochs. *nnU-Net Revisited*
(MICCAI 2024) found that gains over weak baselines usually vanish against a properly configured
nnU-Net. Gate A has been open for a month.

There is also a practical reason to settle it. The dual-encoder model costs about 23 GPU-h per
training run, against 3.2 for the baseline, so every GPU experiment on it uses roughly one week of
Kaggle quota. If the gate returns PARITY, a lighter deployed model makes every remaining GPU
experiment about 5× cheaper. **The architecture is now the most expensive part of the project and
the least central to its thesis.**

### 6.3 The novelty claim is unverified, and the field has moved

C14 says: "Nobody has measured conformal coverage under shift for 3D tumour segmentation." No
related-work document exists to back that sentence. A quick search on 2026-09-24 found close
neighbours:

- *When Average Calibration Fails: Site-Conditional Federated Conformal Risk Control* (arXiv
  2606.20115) runs conformal risk control on FeTS-2022 brain-tumour data across 20 institutions and
  reports coverage violations at 40% of sites.
- *Certify or Refuse: Selective Risk Control with Coverage Floors under Covariate Shift* (arXiv
  2608.10893) is close to the "guarantee plus refusal" idea.
- *Bound-Aware Per-Organ Recall Risk Control ... under Clinical Domain Shift* (arXiv 2608.18193) does
  the same kind of work for CT.

These were identified from search results and have not been read in full yet. The project's result
may still be distinct: adult → SSA → paediatric graded shift, an end-to-end pipeline with a real
DICOM front end, and an error budget. But "nobody has measured" has to go, and a related-work pass
is now a blocker for the paper, not a nicety.

### 6.4 Two data-overlap checks were never done

- **UPENN-GBM (the real-DICOM fixtures).** About 400 UPenn patients are inside the BraTS 2021
  training data, which this project's split is drawn from. A 2025 paper reports that the training
  set drops to 848 cases once the UPenn-GBM cohort is excluded. The PROCEED demo study
  (UPENN-GBM-00002) may be a patient the model trained on. That does not affect any current claim,
  because n = 2 carries no performance claim. It does matter before item 2 in §4, and the BraTS ↔
  TCIA mapping would settle it.
- **UCSF-PDGM (the planned "genuinely unseen fourth cohort" in G4).** Its segmentations were produced
  as part of BraTS 2021. Whether those cases landed in the public training set, the hidden
  validation set or the hidden test set decides whether the cohort is unseen. Check this before
  calling it that.

### 6.5 "Bounds its own error" needs precise wording

Conformal risk control bounds the *average* miss rate over a cohort, not the miss rate for any one
patient. Note 47 says this correctly. A clinician reading "guaranteed bound" will assume it applies
to the patient in front of them. The UI copy and the paper should say "on average across studies
like these" wherever the bound appears.

### 6.6 Scope and overhead are the top risk, as the risk register itself says

- **Volume.** About 37k lines of library (14k of it code; 40% of it is docstrings, per
  `simplification_review.md`), 20k of scripts, 50k of tests, 20k of app, and about 8k lines of plans,
  pre-registrations and logs, all in eight weeks. There are five successive plan documents:
  execution, improvement, interpretable pipeline, master and tool completion. `CLAUDE.md` describes
  itself as "deliberately short" and is 329 lines long.
- **Timing of the tool work.** The week of 14 September had 70 commits, mostly tool UI: the 3D twin,
  atlas shells, a landing-page hero, and a molecular panel whose AI slot says "not available". Each
  of these is fine on its own. Together they took the week in which Gate A and the recalibration arm
  sat idle.
- **Recommendation.** Freeze tool features. Allow only changes that a paper number needs. Keep one
  status board (`master_plan.md` §4.3) and move the superseded plans to `docs/archive/`.

### 6.7 Knowledge ownership

The author is learning deep learning, the code is written by subagents, and a memory note records
that long plans cannot be absorbed. The machinery produces more text than one person can read.

In a viva, a review rebuttal or an interview, the author will have to defend conformal risk control,
the Holm family, why the QC model becomes optimistic under shift, and why D1 is a noise floor. The
best test is for the author to write the paper's introduction and methods themselves, from the
claims table, before anyone else drafts them. Any section the author cannot write shows where to
study next.

---

## 7. Resource check: is the rest doable?

These are GPU-hour estimates on a Kaggle T4, from notes 46–49 and the plan's own figures.

| Item | GPU-h |
|---|---|
| Gate A, full pre-registered nnU-Net fold | ~75 (plus Auto3DSeg) |
| Gate A, amended 250-epoch arm | ~19 |
| D3, cross-fitted fine-tunes on SSA and PED | ~30 |
| D2, pooled multi-cohort training | ~40 |
| Rest of the 3 × 3 seed grid | ~50 |

- **Lean path** (amended Gate A, D3, stop): about 50 GPU-h, which is about 2–3 weeks of Kaggle quota.
  This is **doable by the plan's February 2027 horizon.**
- **Full path:** 200+ GPU-h, about 8–10 weeks of quota with no failed runs. This is **only doable if
  the college card is real.**

The plan's header says "Compute: 200+ GPU-h on a modern card (college cluster)", and `CLAUDE.md`
says college GPU access began 2026-08-19. Yet every run since then (D0, D1, the probe) has gone to
Kaggle. **If that card is not actually available, the plan's compute assumption is false and the
lean path is the plan.**

---

## 8. Suggested next six weeks

| Week | CPU (Mac) | GPU (Kaggle) |
|---|---|---|
| 1 | Rewrite the thesis sentence. Run the overlap checks (§6.4). Write the Mondrian amendment, then run it | Decide on the college card; if it is out, write the Gate A amendment |
| 2 | Real-DICOM validation on UPENN-GBM with ground truth (§4 item 2). Decide `conformal_band` | Gate A, reduced arm |
| 3 | Related-work pass on the three preprints. Draft paper sections 1–2 | Pre-register D3, then run the PED fine-tunes |
| 4 | Intended-use gate input. Re-run the error budget with it, reported as scoping | D3, SSA fine-tunes |
| 5–6 | Draft the rest of the paper. Pick a venue and a date | Buffer for failed runs |

---

## 9. Questions only the author can answer

1. **What is this for?** A degree, a portfolio, a publication or a startup? The advice differs. For a
   degree or portfolio, finishing the paper is the only thing that matters from here. For a startup,
   the regulatory path is the first question, before any more features.
2. **Is the college GPU real and usable, yes or no?** Gate A and the size of the rest of the plan
   depend on it.
3. **Which venue, and which deadline?** Without a date, the writing will keep losing to experiments.

---

Sources for §6.3 and §6.4:
[arXiv 2606.20115](https://arxiv.org/pdf/2606.20115) ·
[arXiv 2608.10893](https://arxiv.org/pdf/2608.10893) ·
[arXiv 2608.18193](https://arxiv.org/pdf/2608.18193) ·
[UPenn cohort excluded from BraTS 2021 training, PMC12640913](https://pmc.ncbi.nlm.nih.gov/articles/PMC12640913/) ·
[UCSF-PDGM on TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830) ·
[RSNA-ASNR-MICCAI-BraTS-2021 on TCIA](https://wiki.cancerimagingarchive.net/display/DOI/RSNA-ASNR-MICCAI-BraTS-2021)
