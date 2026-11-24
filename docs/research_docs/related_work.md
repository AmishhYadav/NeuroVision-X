# Related work — where the conformal and refusal claims sit

**Written 2026-09-27** (`master_plan.md` P3.3). The job of this file is to stop the paper claiming
novelty it does not have. It was triggered by the outside-view review (§6.3,
`docs/research_docs/project_review_2026-09-24.md`), which found three close neighbours to C14 that had
not been read. They have now been read at abstract level; each entry says what it does, how it
overlaps with us, and what that forces us to write. **Read the full PDFs before the MELBA draft** —
the abstract-level reading here is enough to retire an overclaim, not to describe their methods in
detail.

---

## 1. The three direct neighbours

### 1.1 Shahid, *When Average Calibration Fails: Site-Conditional Federated Conformal Risk Control* (arXiv 2606.20115, June 2026, rev. July 2026)

- **What.** Conformal risk control on the false-negative rate for brain-tumour segmentation on
  FeTS-2022 (1,251 subjects, 20 institutions). Pooled CRC "protects the average hospital but
  violates coverage at 40% of individual institutions"; the worst site exceeds the target FNR by
  7.8 percentage points. Proposes *risk-curve shrinkage*: each site sends its empirical risk curve
  and one hyperparameter interpolates between local and pooled calibration (2.7/20 violations at a
  2.0× stretch).
- **Overlap with us.** Direct. Same task family (brain-tumour segmentation), same risk (miss rate),
  same finding in kind: a marginal guarantee holds on average and breaks for sub-populations. Their
  shrinkage is a federated relative of our Mondrian local recalibration (C24).
- **What differs.** Their shift is *between hospitals inside one adult glioma population* (FeTS
  institutions are all BraTS-style adult gliomas). Ours is *between populations*: adult glioma →
  sub-Saharan African adult glioma (BraTS-Africa) → paediatric tumours
  (different disease). We report the violation on three models and separate the label-definition failure
  (PED tumour core) from covariate shift (C14, C30; no ordering by shift size is claimed), a
  sub-region that no recalibration can rescue (PED · TC, C24), and what a refusal gate does and
  does not see (C22).
- **Forces.** C14 can no longer say "nobody has measured conformal coverage under shift for 3D
  tumour segmentation". It is measured, on brain tumours, in 2026.

### 1.2 Adhikary, Chabi, Mastmeyer, *Bound-Aware Per-Organ Recall Risk Control for Multi-Organ CT Segmentation under Clinical Domain Shift* (arXiv 2608.18193, Aug 2026)

- **What.** An AMOS-trained nnU-Net, transferred to RAOS. Per-organ recall control passes in
  distribution and fails after transfer for 7/12 organs at α = 0.10. Local re-certification needs
  ~25 local cases with a WSR betting bound, 30–40 with Hoeffding–Bentkus, and 10–15 with CRC (with
  heavier per-case tail risk); no hard ("Tier-2") organ meets their precision criterion at 25 cases.
- **Overlap with us.** Very close to **C24** in design and in the answer: "how many labelled local
  cases restore the bound?" — theirs ~10–40, ours ~10–40 depending on α. Their "Tier-2 organs that
  cannot be re-certified" is the CT analogue of our PED · TC.
- **What differs.** Modality and anatomy (CT organs vs MRI tumour sub-regions); our infeasibility
  result is analytic (R(τ_min) = 0.356 > α at every k) and stated before the run; our shift is
  graded across three cohorts.
- **Forces.** C24 must cite it as the closest prior result and present ours as a replication of the
  same shape in a second modality and disease, not as a new idea.

### 1.3 Liu, Qiao, Zhang, Chen, *Certify or Refuse: A Cross-Model Map for Selective Risk Control with Coverage Floors under Covariate Shift* (arXiv 2608.10893, Aug 2026)

- **What.** Theory for selective prediction under *bounded-ratio* covariate shift: answer at least a
  β-fraction of target data (an automation floor) while keeping risk on answered cases ≤ α, with
  certificates that either fire or refuse. Validated on a synthetic audit and SQuAD → NewsQA (text),
  not imaging.
- **Overlap with us.** Conceptual: "a guarantee plus a refusal". Our gate (PROCEED / CAUTION /
  REFUSE) is the engineering version of the same idea.
- **What differs.** Their refusal is *certified* under an assumed shift bound; ours is not certified
  at all — it is a QC-model and input-QC gate, and our own result (C22) is that it cannot see
  cohort-level shift. Their framework is the principled fix our discussion should point to.
- **Forces.** The discussion names it as the direction a certified gate would take, and does not
  describe our gate as "risk-controlled refusal".

---

## 2. Core references (cite; methods we build on)

Verify every bibliographic detail against the source before the manuscript.

| Topic | Reference |
|---|---|
| Conformal risk control | Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, arXiv 2208.02814 (ICLR 2024) |
| Risk-controlling prediction sets | Bates, Angelopoulos, Lei, Malik, Jordan, *Distribution-Free, Risk-Controlling Prediction Sets*, arXiv 2101.02703 (J. ACM 2021) |
| Conformal prediction under covariate shift | Tibshirani, Barber, Candès, Ramdas, *Conformal Prediction Under Covariate Shift*, arXiv 1904.06019 (NeurIPS 2019) |
| Mondrian / class-conditional conformal | Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World* (2005) |
| Strong baseline | Isensee et al., *nnU-Net*, Nature Methods 18, 203–211 (2021) |
| Data | Baid et al., *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark*, arXiv 2107.02314 · Adewole et al., *BraTS-Africa*, arXiv 2305.19369 · Kazerooni et al., *BraTS-PEDs*, arXiv 2305.17033 |
| Report automation (not part of the conformal claims) | VASARI-auto; BTReport — prior work for the structured-report pipeline, relevant to the Phase 5 negative |

---

## 3. What the paper may now claim as its own

Stated narrowly, each of these survives the three neighbours above:

1. **Graded shift, ordered failure.** One model, one frozen threshold, three cohorts at increasing
   distance from training; the violation grows with the distance (SSA WT ~1.1× nominal, PED WT
   1.4–1.9×, PED TC 3.5–11.5×). Neighbour 1.1 measures between-site heterogeneity inside one
   population; neighbour 1.2 one transfer.
2. **A sub-region no local recalibration fixes, predicted before the run.** PED · TC is infeasible
   at every k because the miss rate at the lowest threshold already exceeds α — an analytic floor,
   stated in the pre-registration amendment before the numbers.
3. **The gate cannot see cohort shift.** An end-to-end, DICOM-in pipeline with a refusal gate,
   measured against the bound: acceptance filters the worst cases but leaves PED · TC at 0.573
   realised risk on accepted cases (C22). Neighbour 1.3 is the principled alternative.
4. **Everything pre-registered, including the nulls.**

What it may **not** claim: first measurement of conformal coverage under shift in tumour
segmentation; first local-recalibration sample-size estimate; a certified refusal.

---

## 4. Reading notes (W5.1, 2026-11-24)

Full-text reading notes on the six sources the paper leans on. Method: PDFs fetched from arxiv.org and
text-extracted locally; section and theorem numbers below are those of the **arXiv version named in each
note**, which can differ from the camera-ready. Quotes are copied from the extracted text. Where a venue
could not be read off the source itself, the note says so. Nothing here changes §1–§3.

### 4.1 Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control*

- **Citation.** arXiv 2208.02814 (v1 4 Aug 2022; v4 13 Jun 2025, the text read here). ICLR 2024 (the ICLR
  proceedings page lists it; the arXiv abstract page carries no journal reference). DOI 10.48550/arXiv.2208.02814.
  Full text read: yes (19 pages, v4).
- **What they did.** Extend split conformal prediction from miscoverage to the expected value of *any*
  monotone bounded loss. Pick the threshold with a one-line rule (§1.1, eq. 4):
  λ̂ = inf{λ : n/(n+1)·R̂ₙ(λ) + B/(n+1) ≤ α}, with λ̂ = λmax if the set is empty.
- **Exact guarantee.** Theorem 1 (§2.1): "Assume that Lᵢ(λ) is non-increasing in λ, right-continuous, and
  Lᵢ(λmax) ≤ α, sup_λ Lᵢ(λ) ≤ B < ∞ almost surely. Then E[Lₙ₊₁(λ̂)] ≤ α." The guarantee is **marginal, in
  expectation, over the draw of calibration and test**. Theorem 2 (§2.2), i.i.d. and no jumps:
  E[Lₙ₊₁(λ̂)] ≥ α − 2B/(n+1) — tight up to O(1/n). The set-up (§1.1) is "an exchangeable collection of
  non-increasing, random functions".
- **Tumour experiment (§3.1, Fig. 1).** Polyp segmentation (not brain), PraNet base model, FNR loss
  L = 1 − |Y ∩ C_λ|/|Y|, n = 1000 calibration, 781 test, α = 0.1: "The mean and standard deviation of the risk
  over 1000 trials are 0.0987 and 0.0114". Risk is controlled on average over splits; individual splits vary
  by ±0.011.
- **Assumptions.** Exchangeability; monotone loss (§2.4, Prop. 1: for a non-monotone loss "there exists a
  non-monotone loss function such that E[Lₙ₊₁(λ̂)] ≥ B − ε", i.e. the bound can fail by an arbitrary amount);
  bounded loss; Lᵢ(λmax) ≤ α.
- **Shift (§4.1).** (a) If the density ratio w(x,y) = dP_test/dP_train is known and bounded, run the
  procedure on the weighted loss. (b) Under pure covariate shift, "If the conditional distribution of Y given
  X remains the same in the training and test domains" a weighted threshold λ̂(x) works (Prop. 2). (c) For
  *arbitrary* shift, Prop. 3: E[Lₙ₊₁(λ̂)] ≤ α + B·Σᵢ₌₁ⁿ TV(Zᵢ, Zₙ₊₁), with a matching lower bound.
- **Does not cover.** Unknown w; shift in P(Y|X); the full weighted procedure with an *estimated* w ("We leave
  the full investigation of weighted conformal risk control with an estimated covariate shift for future
  research", §4.1). No empirical shift experiment. Per-case or conditional (non-marginal) guarantees.
- **Relation to our work.**
  - Cite for the method and for the exact statement we test: the guarantee is marginal and needs
    exchangeability, so "holds in distribution, violated under shift" is *what the theorem predicts*, not a
    surprise. Our contribution is the measurement (7/12 and 5/12 cells), not the failure itself.
  - Prop. 3 is the only theory in the paper that speaks to us, and it is nearly vacuous for our setting: the
    bound adds B·ΣTV over all n calibration cases, so a fixed per-case TV of c gives α + B·n·c (our reading
    of the formula; the paper does not discuss tightness). Worth one sentence: no bound of this form can
    rescue the PED cells.
  - Label-definition change is a change in P(Y|X), outside the covariate-shift case of Prop. 2 (which assumes
    the conditional is unchanged) and not correctable from unlabelled target data. This is the formal reason
    PED tumour core "fails for every model and no threshold repairs it" (§B1). Also check how our τ_min grid
    maps to the Lᵢ(λmax) ≤ α condition: our analytic floor (0.486 / 0.356 > α) is the case where that condition
    fails and λ̂ = λmax carries no guarantee.

### 4.2 Bates, Angelopoulos, Lei, Malik, Jordan — *Distribution-Free, Risk-Controlling Prediction Sets*

- **Citation.** arXiv 2101.02703 (v3, 4 Aug 2021, the text read). *Journal of the ACM* 68(6), Article 43, 2021,
  DOI 10.1145/3478535. The arXiv page lists no journal reference and the ACM page returned 403 to the fetch
  tool; the JACM volume/article/DOI come from a catalogue record found by web search, not from the publisher
  page. Full text read: yes (34 pages incl. appendices).
- **What they did.** Calibrate a nested family of set predictors T_λ on a held-out set so that the *risk*
  R(T) = E[L(Y, T(X))] is below α **with probability ≥ 1 − δ over the calibration draw**. They build an upper
  confidence bound R̂⁺(λ) and take λ̂ = inf{λ : R̂⁺(λ′) < α for all λ′ ≥ λ} (§2.2, eq. 4). Bounds: Hoeffding,
  Hoeffding–Bentkus, Waudby-Smith–Ramdas (their recommended one for bounded losses, §3.1), Pinelis–Utev for
  unbounded, CLT asymptotic (Thm 6).
- **Exact guarantee.** Def. 1 (§1): "T is a (α, δ)-risk-controlling prediction set if, with probability at
  least 1 − δ, we have R(T) ≤ α." Theorem 1 (§2.2): for an i.i.d. sample, a monotone loss, nested sets and a
  pointwise bound, P(R(T_λ̂) ≤ α) ≥ 1 − δ.
- **Segmentation experiment (§5.4, Figs 12–13).** Loss is the per-object missed-pixel fraction, averaged over
  connected components (so missing a small lesion costs as much as missing a large one). Polyp data, 1,781
  examples, 1,000 calibration, PraNet; α = δ = 0.1. "RCPS again control the risk at the desired level", with
  sets about the size of the polyp.
- **Calibration-set size (§3.4).** Rule of thumb: "around 1,000 calibration points suffice" at α = 0.1;
  a few thousand at α = 0.01; about 10,000 at α = 0.001. Validity holds at any n; small n only makes the sets
  larger.
- **Assumptions.** i.i.d. calibration and test (§2.1). Remark 2 (§2.2): "UCB calibration gives an RCPS even
  if the data used to fit the initial predictive model comes from a different distribution. The only
  requirement is that the calibration data and the test data come from the same distribution."
- **Does not cover.** Any shift between calibration and test; the related-work section only points at other
  work for "handling or testing distribution shift". Note the abstract lists "distributionally robust
  learning", but §6.3 is *adversarial* (ℓ∞-perturbation) robustness, not population shift.
- **Relation to our work.**
  - Cite for the high-probability (α, δ) version of the guarantee and for Remark 2: it states in one sentence
    the exact condition our PED/SSA cohorts violate (calibration and test from the same distribution), while
    permitting the *model* to be trained elsewhere — which is why a stronger or different model does not fix
    the problem.
  - Cite for the per-object loss design (§5.4). Our per-region miss-rate risk is a cousin of it; say which
    loss we control and why (an absent region has an undefined rate).
  - §3.4 gives a calibration-size scale (≈1,000 for tight sets at α = 0.1) that is two orders above the
    k ≈ 10–40 local cases of our recalibration result (C24). Validity needs no minimum n, but state that our
    local bound is valid-and-conservative, not tight, at that k.

### 4.3 Isensee, Wald, Ulrich, Baumgartner, Roy, Maier-Hein, Jaeger — *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*

- **Citation.** arXiv 2404.09556 (v1 15 Apr 2024; v2 25 Jul 2024, the text read). arXiv comment: "Accepted at
  MICCAI 2024"; the MICCAI 2024 proceedings page (papers.miccai.org, 562-Paper2847) lists it. Full text
  read: yes (14 pages). Proceedings page numbers not checked.
- **What they did.** A benchmark of CNN-, Transformer- and Mamba-based 3D segmentation methods under a
  controlled protocol: 5-fold cross-validation, one A100 40 GB, standard configuration scheme, six datasets
  (BTCV, ACDC, LiTS, BraTS 2021 with n = 1251, KiTS 2023, AMOS 2022). Metrics are Dice (primary) and NSD,
  averaged over classes — for BraTS, over the hierarchical regions (§3.2).
- **Key results.**
  - Abstract: "the recipe for state-of-the-art performance is 1) employing CNN-based U-Net models … 2) using
    the nnU-Net framework, and 3) scaling models to modern hardware resources."
  - §4: "Transformer-based architectures (SwinUNETR, nnFormer, CoTr) fail to match the performance of CNNs.
    This includes not matching performance of the original nnU-Net."
  - Table 1, BraTS column (n = 1251, Dice %): nnU-Net org. 91.24; ResEnc M/L/XL 91.26 / 91.13 / 91.18;
    MedNeXt L k5 91.50 (best); SwinUNETR 90.68; SwinUNETRv2 90.74; nnFormer 90.22; CoTr 90.73. The whole
    spread of non-Auto3DSeg methods on BraTS is about 1.3 points.
  - §4 and Fig. 1: "scores on BraTS21 are saturated, with minimal variation both between and within methods";
    §2.2 (P3) attributes this to "a low systematic variance (BraTS)", and recommends ACDC, AMOS, KiTS instead.
  - §4: "U-Mamba … the mamba layers actually have no effect on performance, and instead, the originally
    reported gains were due to coupling the method with a residual U-Net" — the P1 pitfall (confounded
    boosters).
- **Assumptions / limits stated.** "the observed superiority of CNNs is tied to the current experimental
  setting of training methods from scratch on benchmarks of limited size" (§4). Six datasets, none with
  tubular structures; no external shift, no uncertainty or calibration.
- **Does not cover.** Anything about reliability, calibration, conformal methods or distribution shift. It
  uses BraTS only as one in-distribution benchmark.
- **Relation to our work.**
  - It is the citation for why Gate A retired: a matched-recipe comparison (their R1/R2) is the only fair
    reading of C1, and a standardised nnU-Net is the baseline that must be beaten (C29: ET 0.8879 vs 0.8709).
  - Caution to quote: their finding that BraTS is *saturated* (low between-method spread) is the context for
    our +0.0267 ET effect. Our effect is on ET alone, where saturation is weaker than on their region-averaged
    Dice, but a reviewer will cite this paper. Table 1 is region-averaged Dice and not comparable to our ET
    numbers; do not put them in one table.
  - Their SwinUNETR rows (below the original nnU-Net on every dataset) fit our result that a Swin-based
    design is not an architecture win over a strong CNN. They never test a dual CNN + Swin encoder, so they
    neither support nor refute the gated-fusion claim.

### 4.4 Zenk, Zimmerer, Isensee, Traub, Norajitra, Jäger, Maier-Hein — *Comparative benchmarking of failure detection methods in medical image segmentation: Unveiling the role of confidence aggregation*

- **Citation.** arXiv 2406.03323 (v1 5 Jun 2024; v2 24 Jun 2024, the text read, whose footer reads "Preprint
  submitted to Elsevier October 23, 2024"). Published in *Medical Image Analysis* 101 (2025), article 103392,
  DOI 10.1016/j.media.2024.103392. Verified: the DOI resolves (302) to Elsevier pii S1361841524003177.
  Not verified at the source: volume/year/article number, which come from a web-search snippet (the DKFZ
  repository record returned an empty page). The arXiv author list is Zenk, Zimmerer, Isensee, Traub,
  Norajitra, Jäger, Maier-Hein. Full text read: yes (43 pages; the PDF extractor warned of skipped figure
  content, so figure-only numbers were not read).
- **What they did.** Image-level failure detection for 3D segmentation: a scalar confidence κ per case, accept
  if κ ≥ τ, evaluated by risk-coverage analysis with AURC as the summary metric (§2.1, §4.1), risk = 1 − Dice
  (mean over classes). Six datasets: 2D brain tumour (artificial corruptions), brain tumour (BraTS 2019, 235
  train; shift = more low-grade cases, 50 + 50 test), heart (unseen scanner vendors), prostate and COVID CT
  (unseen institutions), kidney tumour (no shift) — Table 2. Methods: pixel confidence + 9 aggregation
  variants, plus ensemble/MC-dropout pairwise Dice, quality regression, Mahalanobis, VAE.
- **Key results.** §5.3: "ensemble + pairwise DSC is consistently the best method overall", MC-dropout +
  pairwise Dice second; the AURC gap to the naive "single network + mean PE" baseline "can exceed 0.1
  (Kidney tumor and Covid)". The Mahalanobis OOD detector "is not among the best methods" for failure
  detection but ranks first on OOD detection (fig. C.18).
- **Key conceptual point (§2.2, R1).** OOD ≠ failure: "when applying a segmentation model to a new hospital,
  all samples are technically OOD, but only some of them might turn out to be failures."
- **Assumptions.** Ground-truth masks are needed only to *score* the detectors. Image-level decisions; no
  threshold calibration ("can be improved by threshold calibration", §5.3).
- **Does not cover.** No guarantee of any kind (no conformal or selective-risk bound); no threshold-selection
  procedure; no paediatric or cross-population brain-tumour shift. The only brain-tumour shift is
  LGG-vs-HGG prevalence in BraTS 2019. §6: "Some datasets may have annotation shifts … Covid and Prostate
  … could hence contain annotation shifts", treated as noise that "all methods are affected equally".
- **Relation to our work.**
  - Cite for the framing that justifies our §B1 sentence "the gate sees bad masks, not changed cohorts": their
    OOD-vs-failure argument and the Mahalanobis result are independent evidence that cohort detectors and
    failure detectors are different tools (our P1.4 OOD-score finding agrees).
  - Cite for AURC / risk-coverage as the reporting standard. Our gate is evaluated by accepted-case risk and
    silent-failure rate; add an AURC column so a reader can compare. Their winner (ensemble pairwise Dice)
    is a candidate comparator for the QC model, but costs several forward passes; we have not run it.
  - Where we go beyond it: they never measure *label-definition* shift and call annotation shift a nuisance;
    our PED tumour-core result shows it can be the dominant failure, and no ranking-based detector fixes it.

### 4.5 Adewole et al. — BraTS-Africa (challenge description 2023; journal versions 2025–2026)

- **Citations.**
  (a) Adewole, Rudie, Gbadamosi, … Anazodo, *The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma
  Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa)*, arXiv 2305.19369, **v1 only**, 30 May
  2023; no journal reference on the arXiv page; arXiv admin note of text overlap with arXiv 2107.02314. Full
  text read: yes (10 pages).
  (b) The arXiv page lists, as a *related* DOI, the journal version: Adewole M, Rudie JD, et al., *Brain tumor
  segmentation in Sub-Saharan Africa patient population: The BraTS-Africa challenge*, **Neuro-Oncology
  Advances 2026; 8(1): vdag082**, DOI 10.1093/noajnl/vdag082. Read via the fetch tool's page summary of the
  OUP page (first 100,000 of 134,856 characters, answers checked against requested verbatim quotes), not as a
  PDF.
  (c) A dataset paper, *The BraTS-Africa Dataset: Expanding the Brain Tumor Segmentation (BraTS) Data to
  Capture African Populations*, *Radiology: Artificial Intelligence* 7(4): e240528 (2025), PMID 40237600
  — **ABSTRACT ONLY / not read**: seen only in search-result records (a secondary source says 146 annotated
  cases: 95 adult diffuse gliomas and 51 other CNS neoplasms). Do not cite numbers from it before reading it.
- **What they did.** (a) is a challenge *design* paper: no results. Pre-operative adult glioma mpMRI (T1, T1Gd,
  T2, FLAIR) from routine clinical care, BraTS pipeline (SRI24 registration, 1 mm isotropic, skull-strip).
  §2.1: "The challenge dataset is divided into training (70%), validation (10%), and testing (20%)".
  **(a) states no case counts.** (b) gives them: "115 adults" from "seven (7) diagnostic imaging centers in
  Nigeria", 1.5 T; 2023 split "60 training cases, 15 validation cases, and 20 test cases"; 2024 split 60
  training, 35 validation, 20 new test. Best 2023 team DSC: ET 0.79, NETC 0.84, SNFH 0.91 (2024: 0.90 / 0.91 /
  0.93).
- **Label protocol.** Three labels, "the BraTS tumor annotation protocol" (a, §2.1.2): ET, NETC (non-enhancing
  tumour core: "necrosis, cystic change, calcification …"), SNFH. Labels begin as a STAPLE fusion of nnU-Net,
  DeepScan and DeepMedic (a, §2.1) — (b) says nnU-Net pre-annotation — then manual refinement and neuroradiologist
  approval. So SSA labels follow the *adult* definition; the shift to SSA is acquisition/population, not
  label definition. (b) notes the data show "lower image contrast and resolution".
- **Assumptions / not covered.** No claim about calibration, uncertainty or failure detection. (a) itself
  warns that "other criteria for delineation could be set, resulting in slightly different tumor sub-regions"
  (§2.1.2), and that the sub-regions are "image-based and do not reflect strict biologic entities".
  Because the ground truth began as an nnU-Net-heavy fusion, an nnU-Net comparator on SSA may be
  advantaged (our inference; the papers do not say so).
- **Relation to our work.**
  - Cite (a) for the label protocol and (b) for case counts and for the one-line finding that adult-trained
    models "showed no improvement" (their wording as summarised by the fetch tool; re-check in the PDF
    before quoting). Our SSA n = 60 matches the 60-case labelled training split (our inference: the
    validation/test labels are not public; confirm before the paper states it).
  - Because the SSA label definition equals the adult one, SSA is the clean case for "acquisition and
    population change", and PED (4.6) is the clean case for "label definition changes". The three-cohort
    design only supports this reading if both claims are quoted from these sources.
  - 60 cases is small: every SSA cell has wide intervals, which is part of why we do not order shifts by size.

### 4.6 Kazerooni et al. — BraTS-PEDs (2023 description and 2024 update)

- **Citations.**
  (a) Kazerooni, Khalili, Liu, … (73 authors on the arXiv page), *The Brain Tumor Segmentation (BraTS)
  Challenge 2023: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs)*, arXiv **2305.17033**, v1
  26 May 2023 … v7 23 May 2024 (text read: the current v7). No journal reference. Full text read: yes (8
  pages). **It is an unfinished draft**: §2.4 reads "To be completed", §3 "Overall XXX teams", Table XXX.
  (b) Kazerooni, Khalili, Liu, … (71 authors), *The Brain Tumor Segmentation in Pediatrics (BraTS-PEDs)
  Challenge: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs)*, arXiv **2404.15009**, v1 23 Apr
  2024 … v4 11 Jul 2024 (text read: v4). No journal reference; arXiv admin note flags "substantial text
  overlap" with (a). Full text read: yes (7 pages). Both are design papers: no results.
- **Cohort.** (a) 228 paediatric high-grade gliomas (astrocytoma, diffuse midline glioma incl. DIPG), "training
  (n = 99), validation (n = 45), and testing" (§2.1). (b) 2024 cohort: 464 patients (CBTN 120, DMG/DIPG Registry
  256, Boston Children's 61, Yale 27); the 2024 data are **not skull-stripped** ("only defaced"), unlike 2023.
  Our PED n = 99 matches the 2023 *training* cohort (inference from the number; confirm against the data
  manifest).
- **The label definition — exact text.**
  - Annotated (2023, §2.1.1): four sub-regions recommended by RAPNO: "enhancing tumor (ET), nonenhancing
    tumor (NET), cystic component (CC), and peritumoral edema (ED)".
  - **Distributed to 2023 participants (§2.1.1):** "While four tumor subregions were annotated, in keeping
    with the other BraTS 2023 challenges … the participants are provided with three segmentation labels,
    including enhancing tumor (ET), nonenhancing component (NC - a combination of nonenhancing tumor, cystic
    component, and necrosis), as well as peritumoral edematous area (ED)."
  - **2023 evaluation regions (§2.2):** "i) the 'enhancing tumor' (ET), ii) the 'enhancing tumor/cystic
    component/necrosis' (TC) and iii) the entire tumorous region, or the so-called 'whole tumor' (WT)."
    **Ambiguity to resolve:** this wording lists ET, cyst and necrosis but not NET, while the NC label as
    distributed contains NET. Read the label values in our preprocessed PED volumes, and the BraTS-PEDs
    2023 Synapse documentation, before the paper says which voxels PED "TC" contains.
  - **2024 (b, §2.1.1):** four separate labels, ET = 1, NET = 2, **CC = 3**, ED = 4. Evaluation (b, §2.2):
    "The BraTS-PEDs 2024 evaluation is different from the 2023 challenge … 'tumor core' (TC) as a combination
    of ET, NET, and CC, vi) the entire tumorous region." (b) §1: "In 2024, we continue the BraTS-PEDs challenge
    with modifications in processing pipeline and tumor subregions."
  - **Contrast with adult BraTS.** The adult TC is the BraTS 2021 definition (ET + necrotic/non-enhancing
    core). The adult-protocol text in 4.5(a) §2.1.2 defines NETC to include "necrosis, cystic change,
    calcification". The paediatric 2023 paper (a) §4 states the consequence directly: models trained on adult
    tumours "perform poorly on segmentation of tumor subregions (e.g. in segmentation of enhancing
    tumor/cystic components as a counterpart to tumor core in adult brain tumors)", while whole-tumour Dice
    stays "relatively high".
- **Assumptions / not covered.** No reliability, calibration or failure analysis; no case-level counts for
  the 2023 test set; sub-region labels come from a paediatric nnU-Net/DeepMedic auto-segmenter then manual
  revision by "volunteer neuroradiology experts of varying rank and experience" (§2.1.1).
- **Relation to our work.**
  - This is the primary source for "label definition changes": quote the §2.1.1 sentences above and the §4
    sentence on adult-trained models. It supports reading PED·TC as "mostly a label-definition mismatch"
    (§B1), but does not itself prove it: that is our measurement (C13/C14, floor 0.486).
  - The 2023-to-2024 change shows the paediatric TC definition itself moved within one year. Say exactly
    which release our PED data and our "TC" use; a reviewer will ask. Also the 2024 data are not skull-
    stripped, so PED-2024 would add an acquisition-pipeline shift on top.
  - Both papers are design documents with no results: do not cite them for paediatric Dice values.

### 4.7 Citation audit

| # | Citation as verified | ID/DOI checked against source page | Full text read |
|---|---|---|---|
| 1 | Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.* arXiv 2208.02814 (v4); ICLR 2024 | arXiv abs page yes; ICLR 2024 venue from the ICLR proceedings search hit only (not opened) | yes |
| 2 | Bates, Angelopoulos, Lei, Malik, Jordan. *Distribution-Free, Risk-Controlling Prediction Sets.* arXiv 2101.02703 (v3); J. ACM 68(6), Art. 43 (2021), DOI 10.1145/3478535 | arXiv yes; **JACM volume/DOI no** (ACM page 403; catalogue record via search only) | yes |
| 3 | Isensee, Wald, Ulrich, Baumgartner, Roy, Maier-Hein, Jaeger. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.* arXiv 2404.09556 (v2); MICCAI 2024 | yes (arXiv comment "Accepted at MICCAI 2024"; MICCAI proceedings page seen in search) | yes |
| 4 | Zenk, Zimmerer, Isensee, Traub, Norajitra, Jäger, Maier-Hein. *Comparative Benchmarking of Failure Detection Methods in Medical Image Segmentation: Unveiling the Role of Confidence Aggregation.* arXiv 2406.03323 (v2); Med. Image Anal. 101 (2025) 103392, DOI 10.1016/j.media.2024.103392 | arXiv yes; DOI resolves to Elsevier (302) yes; **volume/year/article no. from a search snippet only** | yes |
| 5a | Adewole, Rudie, Gbadamosi, … Anazodo. *BraTS 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa).* arXiv 2305.19369 (v1 only) | yes | yes (challenge-design paper, no case counts) |
| 5b | Same group. *Brain tumor segmentation in Sub-Saharan Africa patient population: The BraTS-Africa challenge.* Neuro-Oncology Advances 8(1): vdag082 (2026), DOI 10.1093/noajnl/vdag082 | yes (OUP page; also listed on the arXiv page as related DOI) | partial (fetch-tool summary of first 100k of 134,856 chars) |
| 5c | *The BraTS-Africa Dataset: Expanding the BraTS Data to Capture African Populations.* Radiology: AI 7(4): e240528 (2025) | **no** (search records only) | **no, ABSTRACT/record only** |
| 6a | Kazerooni et al. *BraTS 2023: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs).* arXiv 2305.17033 (v7) | yes | yes (unfinished draft, placeholders) |
| 6b | Kazerooni et al. *The Brain Tumor Segmentation in Pediatrics (BraTS-PEDs) Challenge: Focus on Pediatrics (…).* arXiv 2404.15009 (v4) | yes | yes |

**Orchestrator check (2026-11-24) on the 4.6 TC-wording question.** Our on-disk PED labels (2023 release,
values {0,1,2,3}) answer it empirically: median whole-tumour share NCR/NET 0.84, ED 0.00, ET 0.11, TC/WT 1.00,
75% of cases with ED < 5% (experiments.md note 58). So in our data label 1 carries the non-enhancing tumour and
PED "TC" ≈ whole tumour, consistent with "NC = NET + cystic + necrosis" and not with the narrower §2.2 TC wording.
The paper should cite the data composition, not the 2023 prose. Unverified items in 4.1–4.6 (CRC venue, RCPS
JACM details, Zenk MedIA volume, BraTS-Africa journal versions) must be opened by hand before the citation audit
is closed (W5.1).
