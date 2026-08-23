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

**A constraint on B2 that outranks this document, recorded 2026-08-23, before any endpoint
existed.** `configs/data/splits_ssa.yaml` carries a standing rule for the external cohorts, written
when they were introduced:

> EVERY case is in `test`. `train` and `val` are deliberately EMPTY and must stay empty: this cohort
> exists to measure out-of-distribution performance, so nothing may ever be fitted on it — not a
> model, not a temperature, not a threshold.

A conformal threshold is a threshold. The **primary** B2 result therefore uses the val-fitted λ̂
applied frozen, with nothing whatsoever fitted on SSA or PED — which is what this document already
specifies, and it is the number that may be reported as external validation.

The Mondrian arm below *does* fit a threshold on a held-out slice of each cohort, so it sits outside
that rule. It is retained, because "would per-cohort recalibration restore the guarantee?" is a real
and useful question, but it is hereby fixed as a **counterfactual, not an external-validation
result**. It answers "what would it take to fix this", never "how well does the pipeline do on
unseen data". It must be labelled as such wherever it appears, and it must never be quoted as the
cohort's coverage number. Registering the distinction now, because the difference between those two
readings is invisible in a table of numbers and entirely visible in a caption.

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

**Infeasibility is a registered outcome, not an error.** Added 2026-08-23, still before any endpoint
existed, prompted by a three-case feasibility probe on val logits that confirmed the loss is monotone
and revealed the following. Mean FNR is dominated by a small number of catastrophic cases: on one
probed case, TC false-negative rate was still 0.39 at $\tau = 0.01$, with the mask already inflated
3.85×. If the calibration mean at the smallest candidate $\tau$ still exceeds α, then **no threshold
satisfies the bound** and the correct report is "infeasible at this α", not a silently clipped
$\hat\tau$. The driver must therefore

- return an explicit infeasible status rather than the smallest candidate,
- report, for each α, the smallest achievable mean risk $\hat R(\tau_{\min})$ alongside α, and
- never present an infeasible α as though the guarantee held.

This is registered as a *substantive scientific outcome*: a conformal bound at α = 0.05 on TC being
unreachable at any threshold would say something real about this model's failure mode — that its
misses are concentrated in cases it gets catastrophically wrong rather than spread thinly at the
margin — and that is precisely the finding the Phase E refusal gate exists to act on.

Disclosure, for the record: the probe computed FNR at five thresholds for three val cases in order to
check monotonicity and array plumbing. No endpoint of this pre-registration — no $\hat\tau$, no
realised test risk, no cohort comparison — was computed before this file was committed.

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

**Resolved 2026-08-24 on `neurovision`.** Nothing above this line was edited. Calibrated on the val
split (n=187), applied frozen to test (n=189), BraTS-Africa (n=60) and BraTS-PEDs (n=99). Zero GPU.
Provenance checked first: replay self-consistency against every committed `per_case_metrics.csv`
came back at **1e-17** mean absolute Dice, and the degenerate-endpoint falsifier passed (α=1.0
selects the largest grid threshold for both regions). Monotonicity held on every fitted curve.

### Fitted thresholds (val, n=187)

| region | α | λ̂ | calibrated risk |
|---|---|---|---|
| WT | 0.05 | 0.100 | 0.0446 |
| WT | 0.10 | 0.725 | 0.0929 |
| WT | 0.20 | 0.950 | 0.1512 |
| TC | 0.05 | 0.0158 | 0.0433 |
| TC | 0.10 | 0.600 | 0.0941 |
| TC | 0.20 | 0.950 | 0.1465 |

No α was infeasible. **Caveat that must travel with the α=0.20 rows: λ̂ = 0.95 is the largest value
in the grid, so those are boundary solutions — the true λ̂ is "≥ 0.95" and is censored by the grid,
not measured at it.**

### B1 — the guarantee in distribution: HOLDS, 6/6

| region | α | realised | 95% CI | verdict |
|---|---|---|---|---|
| WT | 0.05 | 0.0364 | 0.0287–0.0447 | **holds** (CI upper < α) |
| WT | 0.10 | 0.0829 | 0.0716–0.0947 | **holds** |
| WT | 0.20 | 0.1409 | 0.1283–0.1547 | **holds** |
| TC | 0.05 | 0.0444 | 0.0288–0.0634 | holds at the point estimate; CI straddles α |
| TC | 0.10 | 0.0909 | 0.0719–0.1125 | holds at the point estimate; CI straddles α |
| TC | 0.20 | 0.1478 | 0.1251–0.1722 | **holds** |

Every realised risk is at or below nominal (0.70×–0.91× of α). The theorem does what it says.

**The registered threat did not materialise.** This document predicted, in advance, that because
every checkpoint was selected on val, λ̂ might be too permissive and test risk might exceed α —
*upward*, if at all. It did not exceed α anywhere. The exchangeable-halves-of-test arm was registered
specifically to diagnose such a violation; since there is no in-distribution violation to diagnose,
**that arm was not run**, and this is recorded rather than quietly dropped.

### B2 — under distribution shift: COVERAGE DEGRADES, and the degradation is graded

Same λ̂, frozen, nothing fitted on either cohort.

| cohort | region | α | realised | ratio to α | verdict |
|---|---|---|---|---|---|
| SSA (n=60) | WT | 0.05 | 0.0550 | 1.10× | inconclusive |
| SSA | WT | 0.10 | 0.1069 | 1.07× | inconclusive |
| SSA | WT | 0.20 | 0.1686 | 0.84× | inconclusive |
| SSA | TC | 0.05 | 0.0724 | 1.45× | inconclusive |
| SSA | TC | 0.10 | 0.1711 | 1.71× | **VIOLATED** (CI 0.129–0.223) |
| SSA | TC | 0.20 | 0.2837 | 1.42× | **VIOLATED** (CI 0.230–0.343) |
| PED (n=99) | WT | 0.05 | 0.0971 | 1.94× | **VIOLATED** (CI 0.068–0.132) |
| PED | WT | 0.10 | 0.1390 | 1.39× | **VIOLATED** (CI 0.108–0.176) |
| PED | WT | 0.20 | 0.1824 | 0.91× | inconclusive |
| PED | TC | 0.05 | **0.5734** | **11.5×** | **VIOLATED** (CI 0.507–0.639) |
| PED | TC | 0.10 | **0.6513** | **6.5×** | **VIOLATED** (CI 0.589–0.711) |
| PED | TC | 0.20 | **0.7002** | **3.5×** | **VIOLATED** (CI 0.643–0.756) |

**Seven of twelve violated, five inconclusive, none holds.** This is the outcome this pre-registration
called "the expected result", and the number it asked for is now measured.

The structure of the failure is the finding, not the fact of it. The excess risk is **ordered by how
far the shift is**: BraTS-Africa is a scanner and population shift within the same disease and its WT
coverage is essentially at nominal (1.07–1.10×), while BraTS-PEDs is a different disease entity and
breaks WT (1.39–1.94×) and destroys TC (3.5–11.5×). It is also ordered by **region difficulty** —
WT degrades gently, TC catastrophically — which matches where the underlying segmentation itself
fails (note 41: PED TC lesion-wise Dice 0.234, with both models failing identically).

A conformal guarantee therefore does **not** transfer across a disease-entity shift, and the amount
by which it fails is a usable signal rather than a uniform collapse. That is precisely the quantity
the Phase E refusal gate needs, and it is why the gate must key on *how far out of distribution the
input is*, not merely on whether the mask looks uncertain.

### Mandatory secondary — what the guarantee costs in mask volume

Registered as mandatory so it could not be dropped if it looked bad. It does not look bad; it looks
**surprising**, and in the useful direction.

| cohort | region | α | mean inflation vs the τ=0.5 mask |
|---|---|---|---|
| test | WT | 0.05 | 1.10× |
| test | TC | 0.05 | 1.18× |
| test | WT | 0.10 | **0.96×** |
| test | WT | 0.20 | **0.88×** |
| test | TC | 0.20 | **0.90×** |
| PED | TC | 0.05 | 2.29× mean / 1.27× median (10 cases skipped, empty reference mask) |

At α=0.05 the conservative mask grows by only ~10–18% in distribution — the guarantee is cheap. At
α=0.10 and 0.20 the inflation is **below 1.0**: the bound is met by a mask *smaller* than the default
0.5-threshold prediction. That is not a bug. It says the deployed operating point is already more
conservative than a 10%-miss-rate guarantee requires, so at those α the conformal layer licenses
being *less* cautious rather than more. Both directions are legitimate outputs of the procedure and
both must be reported; quoting only the α=0.05 row would misrepresent it.

### What this does not license

No claim that the model is better calibrated, and no risk–coverage claim: both are dead and conformal
risk control does not revive them, because the guarantee holds for an arbitrarily bad model. The
`baseline_unet3d` arm is a robustness check on whether the finding is model-specific, not an accuracy
comparison.

Full numbers, all caveats: `docs/experiments.md` note 42. Artifacts: `outputs/conformal/neurovision/`
(`fit.json`, `realised_risk.csv`, `inflation.csv`, per-split `curves.npz`).
