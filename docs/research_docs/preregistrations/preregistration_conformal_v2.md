# Pre-registration — conformal v2: CRC and RCPS, three models, three regions, with the mask cost

**Written:** 2026-11-23, before any v2 number exists (no extended-grid curve, RCPS threshold,
re-split or post-processed risk has been computed). MasterPlan W2.1; analysis-budget family 1 of 6
(§B3 rule 4). **External registration:** Zenodo DOI [10.5281/zenodo.24491280](https://doi.org/10.5281/zenodo.24491280), deposited 2026-11-24: snapshot of tag `semester-2026` (commit `894a607`). The protocol as registered is the copy in that snapshot; only this line was added afterwards.

---

## Why this run exists

Conformal v1 (`preregistration_conformal.md`, note 42; nnU-Net, note 62) left five gaps a reviewer
will find:

1. **ET was excluded** because the miss rate is 0/0 when a case has no enhancing tumour.
2. **α = 0.20 never bound**: τ̂ landed on the grid ceiling (0.95) for every model, so the α 0.20
   cells measure risk at the ceiling, not at a fitted threshold.
3. **Only the expectation guarantee (CRC)** was reported. CRC bounds the *average* over calibration
   draws; a site that calibrates once wants to know how often its one draw is bad. The
   high-probability version (RCPS, Bates et al. 2021) answers that.
4. **One calibration draw.** A single val→test split shows one realisation of a random quantity.
5. **The mask cost was size only** (inflation). Precision, Dice and volume error of the guaranteed
   mask were never shown, and the thresholds were applied to the *raw* mask, not the post-processed
   mask the tool actually deploys.

## Design

**Models.** `neurovision` (seed 42, deployed), `baseline_unet3d` (matched-recipe U-Net), nnU-Net
(Gate A model). Nothing is retrained.

**Data.** Saved fp16 logits for `neurovision` and `baseline_unet3d` on val (187), test (189), SSA
(60), PED (99) — the full rows of the logit table in `CLAUDE.md`. nnU-Net: the saved loss curves
from note 62 only (WT, TC; the v1 grid; raw mask). Its ET curves, extended grid and post-processed
variant need its probabilities, which were discarded on Kaggle; they come from MasterPlan W2.6's
probability export (≤ 3 GPU-h) and are reported when it exists. Until then every nnU-Net cell that
needs them reads "not measured — needs W2.6".

**Loss.** Unchanged: per-case miss rate L_i(τ) = |G_i \ M_i(τ)| / |G_i| of the mask
M_i(τ) = {p ≥ τ}. **(a) ET:** a case with empty reference ET has L_i(τ) = 0 for every τ (nothing
to miss), so ET is included; the v1 convention (exclude, count) is reported beside it for WT/TC,
where no case is empty in BraTS.

**(b) Threshold grid.** The v1 grid (65 points, 1e-4 … 0.95) plus
{0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999}. The v1 grid is a subset, so every v1 number
is reproducible from v2 curves (additivity check below).

**(c) Two guarantees, fitted on BraTS val, applied frozen to test, SSA, PED.**

- **CRC** (Angelopoulos et al., ICLR 2024): τ̂ = max{τ : (n R̂(τ) + 1)/(n + 1) ≤ α} — the v1 rule.
- **RCPS** (Bates et al., JACM 2021) at δ = 0.1, with the Waudby-Smith–Ramdas betting upper
  confidence bound R̂⁺(τ) on the mean loss (losses in [0, 1]; case order a fixed permutation drawn
  once with seed 42). τ̂ = the largest grid τ such that R̂⁺(τ′) < α for every grid τ′ ≤ τ. This
  promises P(realised risk > α) ≤ δ over calibration draws.

α ∈ {0.05, 0.10, 0.20}; regions ET, TC, WT. 3 models × 3 regions × 3 α × 2 guarantees.

**(d) Re-split distribution (in distribution only).** Pool BraTS val + test (376 cases). 1,000
random halvings (188 / 188, seed 42). Each: fit CRC and RCPS on one half, read realised risk on the
other. Report per cell: mean realised risk, P(realised > α) with Monte-Carlo SE, 5/50/95% quantiles.

**(e) Mask cost at each τ̂.** Per case, from the curves' counts (TP = |G| − FN): precision
TP/|M|, Dice 2TP/(|M| + |G|), volume ratio |M|/|G|; mean and median per cohort. Empty-reference
ET cases: precision and Dice defined as 1 if |M| = 0, else precision 0 and Dice 0; volume ratio
undefined (excluded, counted).

**(f) Deployed post-processing.** At each fitted τ̂, the mask is also passed through the deployed
post-processing (`inference.postprocess`: nesting enforcement, components < 50 voxels removed) and
its realised risk and cost are reported beside the raw-mask numbers. τ̂ is fitted on raw curves
only (the theorem applies to them); the post-processed column shows what the tool's mask actually
does.

## Endpoints and decision rules

**V1 — in distribution, re-splits (d).** Per cell: CRC **holds** if mean realised risk ≤ α + 2·MCSE;
RCPS **holds** if P(realised > α) ≤ δ + 2·MCSE. Anything else is **FAILS**, reported as an
implementation-or-assumption alarm, investigated before any other number is read.

**V2 — frozen val threshold on test, SSA, PED.** v1's rule per cell: VIOLATED if the 95% bootstrap
CI lower bound > α (10,000 resamples, seed 42); holds if the upper bound ≤ α; else inconclusive.
Counted per model and guarantee.

**Predictions, registered (directional, not gating):**

1. V1 holds in every cell for both guarantees and all three models.
2. RCPS is never less conservative than CRC: τ̂_RCPS ≤ τ̂_CRC in every cell, with a larger mask cost.
3. With the extended grid, α 0.20 binds below the new ceiling for WT and TC on all models.
4. Under shift, PED TC is VIOLATED at every α for every model under both guarantees (label
   definition, note 58). RCPS violates no more shifted cells than CRC.
5. Deployed post-processing changes realised risk by less than 0.01 in distribution (it removes
   small components; the miss rate is dominated by large structures).
6. ET in distribution is feasible at α ≥ 0.10 for all models.

## Power line (§B3 rule 6)

Bootstrap CI half-widths from v1 at the realised risks seen there: test (n = 189) ≈ 0.008 (WT) to
0.02 (TC); SSA (n = 60) ≈ 0.03–0.05; PED (n = 99) ≈ 0.02–0.06. A shifted cell can be called VIOLATED
only when its excess over α exceeds roughly that half-width. v1's excesses: SSA 0.00–0.08 (so some
SSA cells are resolution-limited and will read "inconclusive", never "holds"), PED TC 0.43–0.50
(resolved). Re-splits: MCSE of P(realised > α) at 1,000 draws ≤ 0.016, enough to separate δ = 0.1
from the ~0.5 an un-adjusted threshold would give. The plausible effects exceed the resolvable ones
where the question is asked; the run launches.

## Additivity check (MasterPlan §B10)

Before any v2 number is read: v2 curves restricted to the v1 grid reproduce v1's `realised_risk.csv`
for `neurovision` and `baseline_unet3d` exactly (same τ̂, same mean miss rate to 1e-9). A mismatch
stops the run.

## What would make this invalid

- Changing the loss, α grid, δ, threshold grid, seeds or calibration split after the first number.
- Fitting any threshold on SSA, PED, or test (test is calibration only inside the re-split
  experiment, where it is never also the evaluation half).
- Fitting τ̂ on post-processed curves and reporting it as conformal.
- Reporting an nnU-Net ET / extended-grid / post-processed cell without the W2.6 probabilities.

## Cost

CPU only. Curve re-extraction from logits for two models × four cohorts (1,070 case-passes, the
v1 extraction path plus post-processing at ≤ 18 τ̂ values per case); fits and re-splits are
arithmetic over stored curves. Estimate ~4 CPU-h, one job at a time.

---

## Result

*(To be completed after the run. Nothing above this line may be edited once the first number
exists.)*
