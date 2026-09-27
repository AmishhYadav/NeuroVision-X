# Semester report — Results, Discussion, Limitations (DRAFT)

`master_plan.md` P3.1. **Draft by Claude, 2026-09-27, for the author to edit.** The author writes the
Introduction and Methods (see `README.md` in this folder). Every number below is a row of
`docs/research_docs/claims_and_evidence.md` (claim id in brackets) — do not add a number that is not
there. `[PENDING: …]` marks a result still being produced; fill it or delete the sentence before
submission.

---

## 4. Results

### 4.1 Segmentation accuracy in distribution

On the 189-case BraTS 2021 test split, the dual-encoder model improves enhancing-tumour (ET) Dice
over a matched 3D U-Net by **+0.0267** (95% CI 0.0166–0.0393; Holm-adjusted p = 1.4 × 10⁻²¹,
paired) [C1]. A second training seed of the proposed model moves test ET Dice by only +0.0021
(CI −0.0018 to +0.0068), about thirteen times smaller than the margin [C1]. A width-matched U-Net
(34.83 M vs 34.91 M parameters) recovers only +0.0055 of the gain, attributing roughly 79% of it
to architecture rather than capacity [C2]. Within the architecture, the gain comes from gated
cross-attention fusion; conditioning the gate on inter-branch disagreement — the project's founding
hypothesis — adds nothing measurable (+0.0022, CI −0.0067 to +0.0152) [C3].

The result replicates at a second seed of both models: +0.0247 (CI 0.0116–0.0420), and
+0.0257 averaged over the two seeds [C1, note 54]. The U-Net's own seed-to-seed shift on test ET is
+0.0041 — small, but resolvable at n = 189 — so the margin is about six times the larger seed
effect of either model.
[PENDING: comparison against nnU-Net, the strong baseline — Gate A, running.]

Under lesion-wise scoring (the BraTS convention since 2023; exploratory, not pre-registered) the
in-distribution gain is larger — ET +0.0508, TC +0.0371 [C10] — and comes from fewer spurious
lesions (0.32 vs 0.47 false-positive ET lesions per case) rather than better recall [C11].
Voxel Dice overstates absolute performance for both models by 0.10–0.32 [C12]. No whole-tumour
comparison is conclusive under either metric.

**Table 1** — test-set Dice, both metric conventions, both models. *(Figure/table source:
`outputs/neurovision/eval_test/summary.csv`, `outputs/eval_test_baseline_unet3d/summary.csv`,
`outputs/replay_lesionwise/*/per_case_default.csv`.)*

### 4.2 Out of distribution

The ET gain does not transfer: on BraTS-Africa (SSA, n = 60) and paediatric BraTS (PED, n = 99)
the margin disappears, and tumour-core Dice is significantly worse under shift [C7]. Heavier
simulated augmentation does not close the gap: 12 of 12 pre-registered comparisons are
inconclusive, and the pooled tumour-core change (+0.0117) is the same size as seed noise [C23].

### 4.3 A distribution-free error bound: where it holds and where it breaks

A threshold calibrated on validation with conformal risk control bounds the fraction of tumour
voxels the mask misses. In distribution the bound holds in all six cells, with realised risk at
0.64–0.96× the nominal level [C13], at small or even negative cost in mask size [C15]. Under
shift it fails, and by an amount ordered by the distance of the shift: SSA whole tumour ~1.1× the
nominal risk, PED whole tumour 1.4–1.9×, PED tumour core 3.5–11.5× [C14].

**Figure 2** — realised vs nominal risk, per cohort and region. *(Source: note 42 tables,
`outputs/conformal/`.)*

### 4.4 What a new site would need

As a counterfactual (not external validation), each shifted cohort was split 1,000 times and the
threshold refit on k of its own labelled cases. About 10–40 local cases, depending on α, restore
the bound for whole tumour on SSA and PED and for tumour core on SSA. **Paediatric tumour core
cannot be restored at any k**: even at the loosest threshold the model misses 35.6% of the tumour
core, above every α tested [C24]. This was stated before the run.

**Figure 3** — held-out risk vs number of local calibration cases k. *(Source:
`outputs/local_recalibration/summary.csv`; figure to be generated.)*

### 4.5 The end-to-end pipeline and its refusal gate

Measured through every stage with the deployed, frozen gate, the pipeline returns a usable mask
(WT and TC Dice ≥ 0.7) for 85.2% of in-distribution studies, 78.3% of SSA and 24.2% of PED; the
silent-failure rate — accepted but unusable — is 4.2%, 18.3% and 49.5% [C20]. The gate has decent
precision and poor recall and is blind to cohort-level shift: on SSA it refuses 2 of 60 studies
while passing 11 of 12 unusable masks [C21]. Accepting a study does not restore the bound where it
breaks: PED tumour-core realised risk is 0.573 on accepted cases at a nominal 0.10 [C22]. The QC
model behind the gate becomes *more optimistic* under shift in all six external cells [C19].

An age-based intended-use rule (adults only) removes every paediatric study from the accepted set —
by construction, as a scoping rule, not as detection (note 53).

[PENDING: real-DICOM front-end cost on BraTS test patients — P1.2.]

---

## 5. Discussion

The project's one clean positive is modest and in distribution: a real, attributable ET gain from
gated fusion. It does not transfer, and the mechanism the project was built to test is not what
produces it. The more useful result is about reliability machinery around *any* model. A
distribution-free bound is only as good as the exchangeability behind it: it holds in distribution
and degrades with the size of the shift, and a learned refusal gate does not see that shift
coming. What restores it is local data — tens of labelled cases at a new site — except where the
model's failure is so large that no threshold can meet the target, as for paediatric tumour core;
there, only changing the model (fine-tuning, D3, planned) could help.

These findings sit beside recent work: coverage failure under shift has been reported between
brain-tumour sites within one population and for CT organs, with similar local sample sizes
(`docs/research_docs/related_work.md`). What this project adds is a graded cross-population shift, a
sub-region shown in advance to be unrecoverable, and an end-to-end DICOM-in pipeline measured
against the bound.

---

## 6. Limitations

1. Most models are single-seed; the noise floor is one seed pair, not a variance estimate.
2. The strong baseline (nnU-Net) is still running; C1 is not yet tested against it.
3. The test split is a random split of the BraTS 2021 training set; numbers are not comparable to
   published challenge results, and lesion-wise results are exploratory.
4. The local-recalibration result is a counterfactual on public cohorts, not a site study.
5. The demo's PROCEED study is a BraTS training patient (note 50).
6. Not for clinical use. The bound is an average over studies, in distribution, not per patient.
