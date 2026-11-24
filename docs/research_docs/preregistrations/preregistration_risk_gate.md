# Pre-registration — risk-targeted refusal gate (Learn-then-Test)

**Written:** 2026-11-24, before any gate threshold has been fitted and before any validation-split
value of ENT or ENS-PD has been computed or looked at (the seed-43 and heavy-aug validation logits
that ENS-PD needs were still being generated when this was written). MasterPlan W2.4; analysis-budget
family 4 of 6 (§B3 rule 4). Signals chosen by the author on 2026-11-24 from W2.3 (note 70), which
found no single best detector: **both ENT and ENS-PD, side by side.** **External registration:**
*(to be filled: a new Zenodo version of the repository, deposited before the run)*.

**What is already known, stated so it cannot be re-discovered as a result:** the deployed gate
refuses by quantiles of the validation signal distribution (2% refuse, 10% caution) and makes no
promise about the error rate of the masks it accepts; its end-to-end numbers are notes 47 and 68
(silent failure 4.2% / 18.3% / 49.5% for `neurovision` on test / SSA / PED). W2.3 (note 70) measured
how well ENT and ENS-PD rank bad masks on test, SSA and PED; this protocol does not re-test that.

---

## Why this run exists

The deployed gate's thresholds are quantiles: they say how many studies are refused, not how many
of the accepted masks are bad. A site needs the second number. Learn-then-Test (LTT; Angelopoulos,
Bates, Candès, Jordan, Lei, 2021, "Learn then Test: Calibrating predictive algorithms to achieve risk
control") chooses a threshold with a finite-sample guarantee on that quantity: with probability at
least 1 − δ over the calibration draw, the **selective risk** P(unusable | accepted) is ≤ β. It
holds for exchangeable data. The registered expectation is that it holds in distribution and breaks
under shift; that is the measurement, as for the conformal bound (C14, C34).

## Design

**Audited pipelines** (each gate is a single signal thresholded; no input QC, no conformal band):

| ID | Model whose mask is judged | Signal | Calibration split |
|---|---|---|---|
| NV-ENT | `neurovision` seed 42 (deployed) | ENT | BraTS val (187) |
| NV-ENSPD | `neurovision` seed 42 (deployed) | ENS-PD | BraTS val (187) |
| NN-ENT | nnU-Net Gate A (W2.6 logits) | ENT of nnU-Net's own probabilities | BraTS val (187) |

ENS-PD for nnU-Net is not defined (one nnU-Net model); that cell reads "not measured — no nnU-Net
ensemble".

**Signals.** Exactly W2.3's definitions (`preregistration_failure_detection.md`, `scripts/failure_detection.py`
`case_signals`): ENT = mean foreground entropy of the judged model; ENS-PD = 1 − min over {WT, TC}
of the mean pairwise Dice of the three `neurovision` members (seed 42, seed 43, heavy-aug), each mask
with deployed post-processing. Higher = predicted worse. A case with no predicted foreground has
ENT = +∞ (always refused). The three members' validation logits are produced by
`scripts/evaluate.py` from each member's own saved test `eval_config.yaml` with only the split and
output dir changed (`outputs/eval_val_neurovision_{seed43,heavy_aug}`).

**Target (fixed).** Unusable ⇔ `dice_WT < 0.7` or `dice_TC < 0.7` of the judged model's deployed
mask (T-Dice of W2.3; `per_case_metrics.csv` of each eval dir, which reproduces the error-budget
`usable` column on all 189 test cases for both models — checked 2026-11-24). T-Vol is not used.
Validation base rates (class counts only, checked 2026-11-24 before writing): `neurovision` 18 of
187 unusable, nnU-Net 16 of 187.

**Gate.** Accept a case iff score ≤ λ. Candidate thresholds are fixed by **target coverage**
c ∈ {0.50, 0.55, …, 0.95, 1.00} (11 points): λ_c = the ⌈c·n⌉-th smallest calibration score (c = 1.00
means accept everything). **Fixed-sequence testing** (LTT's FST) from c = 0.50 upward: at each c,
null H_c: selective risk > β; p-value = P(Binomial(n_c, β) ≤ k_c), where n_c = accepted calibration
cases and k_c = unusable among them (exact binomial, valid conditional on the accepted set). Reject
while p ≤ δ; stop at the first non-rejection. λ̂ = λ at the largest rejected c. If H_0.50 is not
rejected, the gate is **infeasible** at that β (reported, not replaced). δ = 0.1; β ∈ {0.05, 0.10}.
Ties in the score at λ_c: every case with score ≤ λ_c is accepted (so n_c may exceed ⌈c·n⌉).

**Frozen application.** λ̂ applied unchanged to BraTS test (189), SSA (60), PED (99). Per cell:
coverage (accepted / n), selective risk (unusable among accepted) with a 95% Clopper–Pearson
interval, silent failure (accepted and unusable / n), over-refusal (refused and usable / n).

**Comparison with the deployed gate** (`neurovision` only; descriptive). The deployed pipeline's
accepted set (`results/error_budget/per_case_*.csv`, column `accepted`) scored the same way:
coverage and selective risk on test, SSA, PED, beside NV-ENT and NV-ENSPD.

## Endpoints and decision rules

**E1 — calibration feasibility** per (pipeline, β): feasible / infeasible, λ̂, target coverage, and
the calibration selective risk. Not a test.

**E2 — frozen application**, per (pipeline, β, cohort): **VIOLATED** if the Clopper–Pearson lower
bound > β; **holds** if the upper bound ≤ β; else **inconclusive**. A cell whose gate accepts no case
reads "accepts nothing" (no risk defined). At most 3 pipelines × 2 β × 3 cohorts = **18 cells**
(fewer if a gate is infeasible). Counted per pipeline. No multiplicity correction: these are
per-cell guarantee checks, read like conformal v2's V2, not a family of comparisons.

**E3 — deployment rule (registered now, applied once):** the deployed gate's predicted-Dice/quantile
refusal is replaced by NV-ENT or NV-ENSPD at β = 0.10 only if, on BraTS test, that gate's coverage
≥ the deployed gate's AND its selective-risk point estimate ≤ the deployed gate's. If both qualify,
the one with higher test coverage. Otherwise the deployed gate stays and the result is reported. Any
replacement is an `app/` change under §B3 rule 5 (a measured reason) and is done in a separate
commit after the note.

**Predictions (registered, directional, not gating):**

1. No test cell is VIOLATED (every feasible pipeline, both β).
2. Every feasible PED cell is VIOLATED (base rate 72 of 99 unusable for `neurovision`).
3. On SSA the selective-risk point estimate exceeds β in every feasible cell.
4. At equal β, NV-ENSPD has higher calibration target coverage than NV-ENT (it was the better
   in-distribution detector in W2.3).
5. β = 0.10 is feasible for all three pipelines; β = 0.05 is feasible for NV-ENSPD.

## Power line (§B3 rule 6)

**Calibration.** With n = 187 and FST from c = 0.50: at c = 0.50 (n_c = 94) rejection at β = 0.05
needs k_c ≤ 1 (P(Bin(94, 0.05) ≤ 1) = 0.05); at c = 0.80 (n_c = 150) it needs k_c ≤ 3; at β = 0.10,
c = 0.80 needs k_c ≤ 9. With 18 unusable cases in total, β = 0.05 at c ≥ 0.80 needs the signal to
push ≥ 15 of 18 into the refused fifth — an AUROC near 0.9 or better, which W2.3 measured on test
for both signals (ENT 0.906, ENS-PD 0.957). So β = 0.05 is plausible for ENS-PD and marginal for
ENT; β = 0.10 is plausible for both. **Application.** Clopper–Pearson half-width at selective risk
0.05 with ~150 accepted test cases ≈ 0.035, so a test cell at β = 0.05 can read "holds" only if
almost no accepted mask is unusable, and will often read "inconclusive" — stated now so it is not
read as failure. SSA (n ≈ 45 accepted): violation is resolvable only above ≈ 0.2 selective risk.
PED (base 73%): any gate accepting ≳ 20 cases resolves a violation. The run launches because the
in-distribution guarantee and the PED break are both resolvable, and the comparison with the deployed
gate is needed for E3 regardless.

## What would make this invalid

- Choosing the signals, β, δ, the coverage grid, the FST order or the target after any validation
  ENT or ENS-PD value has been computed.
- Fitting λ̂ on test, SSA or PED, or re-fitting it after seeing them.
- Using SegQC (trained on these validation masks) as a calibrated signal here.
- Applying E3 with any rule other than the one above.

## Cost

CPU only. Validation logits for seed 43 and heavy-aug (~9 CPU-h, two jobs in sequence); ENS-PD replay
on val (~10 min, W2.3's code); nnU-Net ENT on four splits from the W2.6 logits (~15 min); the gate
itself is arithmetic. One new module (`scripts/risk_gate.py` with its library).

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first λ̂ exists,
except the external-registration line in the header.)*
