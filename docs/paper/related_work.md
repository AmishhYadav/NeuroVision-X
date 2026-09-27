# Related work — where the conformal and refusal claims sit

**Written 2026-09-27** (`master_plan.md` P3.3). The job of this file is to stop the paper claiming
novelty it does not have. It was triggered by the outside-view review (§6.3,
`docs/research/project_review_2026-09-24.md`), which found three close neighbours to C14 that had
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
  (different disease). We report the violation *ordered by the size of the shift* (C14), a
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
