# Pre-registration — Gate 2, respecified around voxel-level localisation

**Written 2026-08-23, BEFORE any Gate 2 quantity was computed.** It supersedes
the Gate 2 row of `docs/research/execution_plan.md` §Phase 2, which was written
before Gate 1 had numbers. Nothing below may be edited after the first endpoint
is computed; results go in §Result at the bottom.

---

## Why the original Gate 2 must not be run as written

The plan specified Gate 2 as: *referral on disagreement beats entropy and
MC-dropout MI on SSA/PED specifically*. Referral is a **case-level** operation —
it ranks whole cases and hands the worst ones to a human. Gate 1
(`preregistration_ambiguity.md` §Result, 2026-08-23) measured exactly that
ranking and found it **null on both external cohorts**: partial Spearman
−0.173 on SSA (CI −0.432 to +0.090) and +0.000 on PED (CI −0.223 to +0.211),
against −0.393 in distribution.

Running the original Gate 2 would therefore be running a test whose outcome is
already strongly indicated, on the one endpoint the data says does not
transfer. It is respecified here around the endpoint that **did** survive
Gate 1: per-voxel error localisation, which cleared 0.5 on all three cohorts
(p_holm 0.0025) and cleared the pre-registered 0.60 on PED (0.677, CI
0.655–0.698), with ET strongest everywhere (0.733 / 0.689 / 0.784).

## Why the question is "does it ADD to entropy", not "does it beat entropy"

Gate 1's raw numbers make the honest framing unavoidable. Entropy's own voxel
AUROC is **higher** than disagreement's on every cohort (0.888 vs 0.694 on
test, 0.875 vs 0.726 on SSA, 0.785 vs 0.777 on PED). A head-to-head "flag
voxels by disagreement instead of entropy" comparison is therefore expected to
LOSE, and staging one would be theatre.

What Gate 1 actually established is that disagreement carries error information
entropy does **not** have — that is what a residualised AUROC above 0.5
means. The deployable question that follows is whether **entropy plus
disagreement localises error better than entropy alone**, and whether that
still holds under distribution shift. That is what this gate tests.

---

## The combiner, and where it is fitted

A fitted combination needs a fit set that is never reported, or the reported
number is optimistic by construction. This project has the rule already, from
temperature scaling: *fit on val, apply to test.*

- **Combiner:** logistic regression on exactly two features — the per-case
  rank-transformed predictive entropy and the per-case rank-transformed mean
  disagreement, both computed inside the same label-free
  `_predicted_dilated_mask` Gate 1 used (`dilation_mm: 10.0`). Two features, no
  interaction term, no regularisation tuning: fixed now so there is nothing to
  select on later.
- **Fit split:** the frozen **validation split, 187 cases**
  (`outputs/neurovision/eval_val`), which no reported number in this project
  comes from. Ambiguity maps for it do not exist yet and will be extracted the
  same way as every other cohort.
- **Applied frozen** to BraTS test (189), SSA (60), PED (99). **No refitting per
  cohort**, and in particular no refit on an external cohort — a detector that
  needs the shifted data to be fitted is not a detector that transfers.
- **Baseline:** the identical pipeline with the disagreement feature removed —
  a one-feature logistic regression on rank-transformed entropy, fitted on the
  same 187 cases. Both models see the same mask, the same voxels and the same
  fit set, so the only difference between them is the disagreement feature.

---

## Endpoints, fixed before measurement

**Co-primary endpoint C — statistical.** Per-case AUROC for predicting
per-voxel error, computed for both models, and the **paired per-case
difference** Δ_AUROC = AUROC(entropy + disagreement) − AUROC(entropy). Reported
per cohort with a paired bootstrap CI over case indices
(`analysis.statistics.paired_bootstrap_ci`, 10 000 resamples, 95%).

**Co-primary endpoint D — operational.** At a fixed **5% flag budget** — the
5% of predicted-foreground voxels per case with the highest score — the
**recall of error**, i.e. the fraction of that case's erroneous voxels that
land inside the flagged 5%. Reported as the paired per-case difference
Δ_recall@5% between the two models, same bootstrap. This is the number a
deployment argument actually rests on; endpoint C is the number a reviewer
will ask for.

**Secondary, explicitly outside the family — endpoint E.** Component-level
false-positive detection: for each predicted connected component of WT and of
ET, AUROC for "this component has zero overlap with ground truth" from mean
disagreement, and from entropy, and from both. Motivated by a documented and
unaddressed failure mode — the 10 mm+ band holds 1 047 false-positive voxels
against 5 false-negative, in components large enough to survive
`min_component_size: 50`. Reported with CIs, labelled **exploratory**, and it
can never be used to rescue a failed Gate 2.

**Multiplicity.** Holm–Bonferroni across 2 co-primary endpoints × 3 cohorts =
**6 tests**, α = 0.05, fixed now. Endpoint E is not in the family and is not
corrected into it.

---

## Decision rule

| Outcome | Condition | Consequence |
|---|---|---|
| **Pass** | On **at least one external cohort** (SSA or PED): Δ_AUROC ≥ **0.01** with a CI excluding zero, **and** Δ_recall@5% ≥ **0.02** with a CI excluding zero | The paper's contribution is a dual-encoder-only, single-forward-pass error localiser that survives distribution shift. Build the demo layer and the referral figure around voxels, not cases |
| **Partial** | A CI excludes zero on an external cohort but the magnitude misses its threshold | Report as a real but small additive effect, with the magnitude stated in the abstract. No deployment claim |
| **Fail** | No external cohort shows a CI excluding zero on either endpoint | **The uncertainty line stops here.** Disagreement is then a within-distribution diagnostic only, reportable as a mechanism figure (note 32) and nothing more. Fall back to `execution_plan.md` §Fallback |

The thresholds are deliberately modest and the reason is stated in advance:
entropy's baseline AUROC is already ~0.88 in distribution, so the headroom
above it is small by construction, and a 0.01 AUROC gain on top of a 0.88
baseline is a different thing from a 0.01 gain on top of 0.55. The magnitude
will be reported next to the baseline it improves, never alone.

---

## Power, stated in advance

n = 60 (SSA) and 99 (PED) cases, paired. The bootstrap is over cases, so the
relevant n is the case count, not the voxel count — 20 000 sampled voxels per
case make each per-case AUROC precise but do **not** buy cohort-level power.
Every reported quantity carries a CI and no claim rests on a point estimate.
A null result at these sizes is reported as *"CI includes zero at n=60"*, never
as *"no difference"* — this project has produced six underpowered nulls already
and will not add a seventh by mislabelling one.

---

## Pre-committed rules

1. **One run, on complete data.** Every cohort's maps are extracted before the
   family is computed once. No looking at cohort 1 before deciding cohort 2.
2. **No refitting.** The combiner is fitted exactly once, on val, and applied
   byte-identically elsewhere. If it needs a cohort-specific fit to work, it
   has failed this gate.
3. **No re-running on a subset**, and no post-hoc change to the feature set,
   the mask, the budget, the thresholds or the family.
4. **Masks stay label-free.** `_predicted_dilated_mask` only. The
   `union_foreground_mask` failure — a label-derived reporting mask that
   manufactured 41–57% of a reported ECE behind 984 passing tests — is the
   reason this rule is restated in every pre-registration in this project.
5. **Direction is fixed now:** positive Δ means the two-feature model localises
   error better. A negative Δ is reported as a negative result, not reframed.

---

## What this gate deliberately does NOT claim

**MC-dropout equivalence is not tested here, and that is a limitation, not an
omission.** The plan's "equal detection at 1/10 the cost" TOST needs per-voxel
MC-dropout MI maps on all three cohorts. Only a per-case
`uncertainty_summary.csv` survives in distribution; the voxel maps were deleted
as regenerable caches, external cohorts never had them, and regenerating them
is ~10 forward passes per case (order 24 h of CPU). So the cost claim stays
**unmade** until those maps exist. Writing "matches MC-dropout" without them
would be claiming a measurement that was never taken.

---

## Result

**Run 2026-08-23. Verdict: PARTIAL, and in substance mostly negative.** Combiner
fitted once on the 187-case val split (3,740,000 voxels, both arms converged)
and applied frozen. Artifacts: `outputs/gate2/gate2_{per_case,family}.csv` and
`gate2_verdict.json`. Family run once, on complete data.

Fitted coefficients — entropy-only `[-8.446, 7.786]`, entropy+disagreement
`[-8.700, 7.351, 0.952]`. The disagreement feature does get a positive weight,
about 13% of entropy's, so the combiner is genuinely using it.

### The family

| Cohort | endpoint | entropy alone | + disagreement | Δ | 95% CI | p_holm |
|---|---|---|---|---|---|---|
| BraTS test | AUROC | 0.9162 | 0.9150 | **−0.0013** | (−0.0023, −0.0003) | 0.042 |
| BraTS test | recall@5% | 0.5113 | 0.4720 | **−0.0394** | (−0.0453, −0.0337) | 0.0006 |
| SSA | AUROC | 0.8874 | 0.8888 | +0.0013 | (−0.0007, +0.0035) | 0.208 |
| SSA | recall@5% | 0.3432 | 0.3304 | **−0.0128** | (−0.0201, −0.0062) | 0.0006 |
| PED | AUROC | 0.8307 | 0.8428 | **+0.0121** | (+0.0102, +0.0139) | 0.0006 |
| PED | recall@5% | 0.1826 | 0.1812 | −0.0014 | (−0.0033, +0.0003) | 0.208 |

4 of 6 rejected under Holm — but **two of the four rejections are losses.**

### Why this is PARTIAL and not PASS

PASS needed both conjuncts on one external cohort. PED clears the AUROC
threshold cleanly (+0.0121 against a pre-registered 0.01, CI excluding zero)
and shows **nothing** on the operational endpoint. SSA shows nothing on AUROC
and is significantly **worse** on recall. `passed_cohorts` is empty.

### The plain reading, which is blunter than the label

Adding disagreement to entropy does not improve error localisation. It
**degrades** it in distribution on both endpoints, degrades the operational
endpoint on SSA, and helps exactly one cohort on exactly one endpoint. The
operational endpoint — how much of a case's error you catch by flagging 5% of
predicted foreground — gets **worse or unchanged everywhere**: entropy alone
catches 51.1% / 34.3% / 18.3% of error on test / SSA / PED, and adding
disagreement never beats that.

### This does NOT contradict Gate 1, and the gap between them is the finding

Gate 1 asked whether disagreement carries error information entropy lacks, by
residualising, and the answer was yes (voxel AUROC 0.578 / 0.569 / 0.677, all
CIs excluding 0.5). Gate 2 asks whether that information makes a **fitted
operating detector** better, and the answer is no. Both are true at once:
information can be present in a residual and still fail to improve a detector
that already has a very strong feature — entropy's own per-case AUROC here is
0.916 in distribution. **Incremental information is not incremental utility**,
and this project now has a measurement of both.

### A limitation of this gate's own design, stated rather than hidden

The combiner is fitted by **pooled logistic likelihood** while the endpoints are
**per-case AUROC and per-case recall at a budget**. Those objectives are not the
same, and a likelihood-optimal weighting can be worse on a per-case rank metric
— which is the leading explanation for the in-distribution losses. That
mismatch was baked into the pre-registration and cannot now be swapped for a
rank-optimised combiner without the change being post-hoc: fitting directly for
AUROC after seeing these numbers is exactly the move this pre-registration
exists to prevent. It is recorded here as a limitation, and as the one
defensible follow-up if the uncertainty line is ever resumed — pre-registered
in advance, on fresh data.

### Consequence, per the decision rule

Partial: report as a small effect with the magnitude stated, and make **no
deployment claim**. Concretely, for the paper: the disagreement map is a
*mechanism* result (Gate 1, note 31, note 32) and a *localisation* result at
residual level, **not** a better uncertainty estimator. Any sentence implying
users should flag voxels by entropy+disagreement rather than entropy is
contradicted by this table.
