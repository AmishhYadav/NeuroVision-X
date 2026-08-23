# Pre-registration — a distribution-free bound on the mask's miss rate, and where it breaks

**Written:** 2026-08-23, **before** `src/neurovision/uncertainty/conformal.py` or
`scripts/conformal.py` exists, before any λ̂ has been fitted, and before any realised risk has been
computed on any split or cohort. The git timestamp on this file is the evidence for that ordering.

**Scope.** This registers **Phase B** of `docs/research/master_plan.md` — B1 (the guarantee) and B2
(the guarantee under distribution shift). Phase B is the project's new scientific centrepiece.

---

## Background, and why this phase is shaped differently from every previous one

Eight of this project's nine resolved comparisons came back null. Every one of them was a *downstream*
claim built on a +0.0267 ET Dice gain — a gain the size of a few voxels at a tumour margin, asked to
move calibration, boundary accuracy, report content and out-of-distribution robustness. Each null was
predictable from an effect-size argument nobody made in advance.

Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control*, ICLR
2024) is not that shape of bet. **B1 cannot fail**: it is a theorem. Given exchangeable calibration
and test data and a loss that is monotone in a threshold, choosing

$$\hat\tau \;=\; \sup\Big\{\tau : \tfrac{n\,\hat R(\tau) + B}{n+1} \le \alpha \Big\}$$

guarantees $\mathbb{E}[L_{\text{test}}(\hat\tau)] \le \alpha$, with no assumption whatsoever about
the model, the data distribution, or whether the model is any good. It ships as a product feature
whatever the numbers say.

**B2 is where the finding lives.** Conformal validity rests on exchangeability, and our two external
cohorts violate it by construction — BraTS-Africa (n=60) and BraTS-PEDs (n=99) are different
scanners, different populations, different tumour biology. Nobody has measured what happens to a
conformal guarantee for 3D tumour segmentation under that shift. We already hold both cohorts with
ground truth, both models' saved fp16 logits, and the calibration machinery. The cost is zero GPU
hours.

Both B2 outcomes are useful and both are publishable. If coverage holds, that is a strong safety
result. If it degrades, we have quantified by how much, and that number directly motivates the
refusal gate of Phase E.

---

## The loss, fixed now

For case $i$ and region $R$, let $p_i$ be the model's per-voxel sigmoid probability for $R$, let
$G_i$ be the ground-truth mask, and let the **conservative mask** at threshold $\tau$ be
$M_i(\tau) = \{v : p_i(v) \ge \tau\}$.

$$L_i(\tau) \;=\; \frac{|G_i \setminus M_i(\tau)|}{|G_i|} \qquad\text{(per-case false-negative rate)}$$

This is the fraction of true tumour the mask misses. It is **non-decreasing in $\tau$** — lowering the
threshold can only grow the mask and can only miss less — which is the monotonicity the theorem
requires, with the direction absorbed into the `sup` above. It is bounded, $B = 1$.

Chosen over per-voxel or Dice-based losses because it is the quantity a clinician actually cares
about ("how much of the tumour could this mask have missed?"), it is bounded without clipping, and it
is monotone without qualification.

**Regions.** Primary endpoints are **WT** and **TC**. **ET is secondary**, because 2.6% of BraTS 2021
cases have no enhancing tumour at all and the loss is undefined on them.

**Empty ground truth.** When $|G_i| = 0$ the loss is $0/0$. Such cases are **excluded** from both the
calibration mean and the realised-risk mean *for that region*, and the excluded count is reported
alongside every table. Fixed now precisely because the alternative conventions (score 0, score 1) are
each defensible and each moves the answer.

**α, fixed now:** $\alpha \in \{0.05,\ 0.10,\ 0.20\}$. Every table reports all three.

---

## Design, fixed now

| | |
|---|---|
| Models | `neurovision` (primary — the deployed model) and `baseline_unet3d` (secondary). Two models, so a finding can be checked for model-specificity |
| Calibration set | the frozen **val** split, n = 187 |
| Evaluation set | the frozen **test** split, n = 189, with $\hat\tau$ applied **frozen** |
| Shift cohorts | BraTS-Africa n = 60, BraTS-PEDs n = 99, same frozen $\hat\tau$ |
| Inputs | saved fp16 logits only. No inference, no GPU, no retraining |
| Randomness | only the bootstrap, seeded at 42 via `utils/seed.py` |

**The threat this design carries, stated before the result: the val split is not clean.** Every
checkpoint in this project was selected on val by `val/dice_mean`. Val is therefore optimistically
biased for these models, so a $\hat\tau$ fitted on it may be **too permissive**, and realised test
risk may exceed $\alpha$ for a reason that has nothing to do with conformal theory. The direction of
that bias is predicted here, in advance: **if the bound is violated on test, it should be violated
upward, and the secondary analysis below should make the violation disappear.**

**Secondary, exchangeable-by-construction check.** Split the 189 test cases at random in half,
calibrate on one half, evaluate on the other, and average the realised risk over 100 random splits
(seed 42). Those two halves *are* exchangeable, so this arm isolates whether any α violation in the
primary arm is caused by checkpoint selection on val rather than by the method.

---

## Endpoints and decision rules, fixed now

**B1 — primary.** For each model, region and α: realised mean FNR on test, with a 95% bootstrap CI
(10,000 resamples, seed 42).

| Outcome | Rule | Reading |
|---|---|---|
| Bound holds | point estimate ≤ α | The guarantee transfers in distribution, as the theorem says it must |
| Bound violated | 95% CI lower bound > α | Something in our pipeline breaks an assumption. Report it, and check the exchangeable arm before blaming the theorem |

**B2 — primary, and the actual open question.** Same quantity on SSA and on PED, with $\hat\tau$
frozen from val. Reported as **excess risk**, realised − α, with a 95% CI.

| Outcome | Rule | Consequence |
|---|---|---|
| Coverage holds under shift | CI upper bound ≤ α for both cohorts | A strong safety result. The guarantee is more robust than exchangeability alone predicts, and that is worth saying |
| Coverage degrades | CI lower bound > α on either cohort | The expected result. Publish the magnitude — the first such number for 3D tumour segmentation — and it becomes the motivation for the Phase E refusal gate |
| Inconclusive | CI straddles α | Report as inconclusive at n = 60 / n = 99 and do not spin it |

**B2 secondary — the Mondrian / weighted variant.** Recalibrate $\hat\tau$ per cohort on a random
half of that cohort, evaluate on the other half, average over 100 splits (seed 42). Pre-registered
claim: **this restores the guarantee within each cohort**, because exchangeability holds inside a
cohort even when it fails across cohorts. If it does not restore it, that is a more interesting and
more alarming result and it gets its own note.

**Mandatory secondary on every arm — the cost of the guarantee.** Mean ratio
$|M_i(\hat\tau)| \,/\, |M_i(0.5)|$, the factor by which the conservative mask inflates the point
mask, and the mean absolute volume of the *uncertain band* $M_i(\hat\tau) \setminus M_i(0.5)$ in
cm³. **A bound bought by predicting the whole brain is worthless, and this table is what stops us
reporting it as a success.** It is registered as mandatory so it cannot be quietly dropped if it
looks bad.

**Reported unconditionally, whatever it says:** the fitted $\hat\tau$ itself. If $\hat\tau \ge 0.5$
the ordinary point-estimate mask already satisfies the bound and the conformal layer is a no-op at
that α — an honest and mildly deflating outcome that must be stated, not buried.

---

## What would falsify the implementation rather than the hypothesis

Registered in advance so that a bug cannot be reported as a finding. `scripts/conformal.py` must
refuse to write results unless all four pass:

1. **Monotonicity.** $\hat R(\tau)$ is non-decreasing in $\tau$ on the calibration set, per region.
   A violation means the loss or the threshold direction is wired backwards.
2. **Degenerate endpoints.** $\alpha \ge 1$ selects the largest candidate $\tau$; $\alpha \to 0$
   drives $\hat\tau \to 0$ (predict everything).
3. **Replay self-consistency.** Recomputing Dice from the saved logits at $\tau = 0.5$ with the
   project's own postprocessing reproduces that run's committed `per_case_metrics.csv`. This is the
   check `scripts/replay_logits.py` already performs, and it is what proves the logits on disk belong
   to the published numbers. Cited precedent: replay reproduced `neurovision` ET **0.870859 vs
   0.870859, delta 0** (`docs/experiments.md` note 223).
4. **Fit/apply separation.** The script raises if the calibration directory and the evaluation
   directory resolve to the same path — copied from `scripts/calibrate.py`, which already enforces
   this, because a convention documented only in a docstring is one CLI override away from being
   silently violated.

---

## What this pre-registration does not license

- Any claim that the model is **better calibrated** than a comparator. That claim is dead
  (`docs/paper/claims_and_evidence.md`) and conformal risk control does not revive it: the guarantee
  is a property of the *procedure*, and it holds for an arbitrarily bad model.
- Any claim about **risk–coverage** or uncertainty quality. Also dead, also not revived.
- Any comparison of `neurovision` against `baseline_unet3d` on realised risk framed as an accuracy
  result. Both models get the guarantee. If their band widths differ, that is a statement about mask
  sharpness, and it is reported as exactly that.

---

## Result

*Empty by construction. Nothing above this line may be edited once a number exists — a prediction
edited after the fact is not a prediction. `docs/research/master_plan.md` §4.6 is the checklist for
filling this in.*
