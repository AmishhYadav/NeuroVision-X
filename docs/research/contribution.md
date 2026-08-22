# Contribution

> **STATUS UPDATE, 2026-08-17 — read this before the text below.** Everything
> from "## The claim, in one paragraph" onward is the **pre-registration**,
> written 2026-08-04 before any of it was measured. It is kept verbatim on
> purpose: predictions that are edited after the fact are not predictions. What
> actually happened is in `## Measured outcomes` near the bottom of this file,
> and in `docs/experiments.md` notes 11–17 and 19–21.
>
> The short version. The prediction that the benefit "should appear as improved
> calibration and boundary accuracy rather than as a uniform Dice gain" is
> **backwards relative to the data**: calibration, boundary accuracy and
> MC-dropout risk-coverage all came back within noise against a matched
> baseline, while Dice improved substantially and significantly (ET +0.0267,
> p_holm 7.2e-22). A parameter-matched capacity control then showed ~79% of that
> ET gain is architecture rather than width. So the model is more **accurate**,
> not more **reliable**, and P2 — the ablation that would say whether the
> *ambiguity conditioning specifically* is what does it — **has not been run**.

> **Status:** draft, 2026-08-04. Written *before* `related_work.md` existed. Every claim
> about what prior work does is marked `[cite]` and must be checked against
> `related_work.md` once written. If any of them turns out to be false — i.e. someone
> already conditions a fusion gate on inter-branch disagreement — this paragraph is the
> thing that has to change, not the experiments.

## The claim, in one paragraph

NeuroVision-X's fusion module differs from prior gated and cross-attention fusion in
*what the gate is a function of*. Prior dual-branch fusion for medical segmentation
derives its mixing weight from the branch content itself — channel-wise
squeeze-excitation weights, spatial attention maps, or cross-attention affinities
between CNN feature maps and transformer tokens `[cite]` — so the weight is a free
latent quantity with no defined semantics, frequently shared across a whole channel or
a whole resolution level rather than resolved per voxel `[cite]`, and supervised only
indirectly through the segmentation loss. Our gate instead takes an *explicit local
ambiguity signal* as input: at each fusion scale, a lightweight auxiliary projection
reads region logits from each branch independently, and the per-voxel disagreement
between those two predictions, together with each branch's own predictive entropy, is
concatenated to the gate's input alongside the features. The gate is therefore a
function of a quantity a content-only gate cannot compute — how much the local-texture
branch and the long-range-context branch disagree about this voxel — and it resolves
per voxel and per scale rather than per channel. The resulting claim is mechanistic and
falsifiable rather than aesthetic: the gate should shift toward the transformer branch
exactly where local evidence is insufficient (tumour boundaries, oedema/infiltration
zones, voxels with high MC-dropout variance) and toward the CNN branch in homogeneous
region interiors, and the benefit of that shift should appear as improved calibration
and boundary accuracy rather than as a uniform Dice gain.

## Why the ambiguity signal is inter-branch disagreement and not the confidence head

The confidence head sits downstream of fusion. Conditioning the gate on it is circular:
the head's output depends on the gate, which would depend on the head. Inter-branch
disagreement is available *at* fusion time from quantities that already exist, costs one
1×1×1 convolution per branch per scale, and is a proxy for exactly the thing we care
about — "the two inductive biases do not agree here."

## Testable predictions

[Outcomes recorded below, in `## Measured outcomes`.]

The contribution is only real if these hold. Each is measured on the held-out test
split, not on validation.

**P1 — Mechanism fires.** The mean gate value (transformer weight) is monotonically
increasing in local ambiguity. Measured two ways, both on held-out cases:
correlation of the upsampled gate map with (a) inverse distance-to-ground-truth-boundary,
and (b) per-voxel MC-dropout variance. Predicted sign: positive, both.
**Null result:** |r| < 0.1. That means the gate collapsed to a near-constant learned
blend and the module is a parameter-matched no-op dressed up as a mechanism. Report it
as such.

**P2 — The ambiguity conditioning is necessary, not decorative.** Ablation ladder, all
parameter-matched, all same seed set:
1. fixed scalar blend (learned, one number per scale)
2. content-only gate — same architecture, disagreement/entropy channels removed
3. full gate (ours)
Predicted: 3 beats 2 on ECE and HD95 by a margin exceeding the seed-to-seed std, and
ties or nearly ties it on Dice.
**Null result:** 2 matches 3. Then the gate's input does not matter, only its
per-voxel spatial resolution does — which is a smaller and different contribution, and
must be reported as that instead.

**P3 — The gain is where the claim says it is.** Metrics stratified by distance to
boundary (e.g. ≤2mm, 2–5mm, >5mm) and by ground-truth-ambiguity proxy. Predicted: the
improvement concentrates in the near-boundary and high-ambiguity strata.
**Null result:** uniform improvement across strata. That is a generic capacity or
regularization gain, not evidence for the stated mechanism, and the paper cannot claim
the mechanism on the strength of it.

**P4 — Faithfulness of the gate as an explanation.** The gate map is itself a
lightweight uncertainty map. Predicted: it is a worse-calibrated but far cheaper
predictor of error than MC-dropout — nonzero AUROC for error detection, well above
chance. This is a bonus result, not load-bearing.

## Measured outcomes

Added 2026-08-17. The predictions above are untouched; this section says what
came back. Sources: `docs/experiments.md` notes 11–17 (the reliability
measurements) and 19–21 (the capacity control).

| Prediction | State | Outcome |
|---|---|---|
| **P1 — mechanism fires** | **MEASURED — fires, but not as predicted** | Measured 2026-08-20 over the full 189-case test split (`scripts/gate_boundary_profile.py`; `docs/experiments.md` note 32). The gate is strongly and monotonically organised by anatomy, with opposite polarity at adjacent fusion scales — but P1 as literally written (the gate peaking at the tumour margin) is **refuted**; what passed is the weaker claim that the gate carries a real, non-decorative spatial signal. See the subsection below |
| **P2 — ambiguity conditioning is necessary** | **RUNNING since 2026-08-22** | The Kaggle weekly ration reset to 30 h and `ablation_content_only_gate` was launched as a 3-session chain (kernel `neurovision-p2ablation-s1`, ~22–23 GPU-h expected, `GIT_REF=7caacfa` — the same tree `neurovision` trained on, so the arms differ by the one key and not by the training code). Parameter-matched to 0.018% (6,360 of 34,911,341), verified from the pinned tree before launch. Until it lands this row is still an assertion, not a result, and the paper must say so |
| **P3 — the gain is where the claim says it is** | **FAILED as stated** | Boundary-stratified error is within noise against the matched baseline. The improvement is a Dice improvement concentrated in ET, not a demonstrated near-boundary effect |
| **P4 — gate as a cheap error predictor** | **MEASURED, exploratory** | `docs/experiments.md` note 34, 2026-08-20. A label-free gate read-out predicts per-case Dice after partialling out predictive entropy and two volume confounds. On PED — where the model is catastrophically worse and the free entropy baseline carries **no** usable signal (partial ρ 0.136, CI containing zero) — `gate1_fg` reaches +0.584 [0.380, 0.725], p_holm 0.013. On SSA no gate feature survives Holm. Not pre-registered, one patch per case, one external cohort out of two. Bonus result, still never load-bearing |
| **P5 — the gain reaches the report** *(added post hoc, 2026-08-19)* | **FAILED** | Not pre-registered, and it should have been. 1 of 25 report-agreement metrics conclusive against the matched baseline, 0 of 25 for the capacity control. See the entry below and `docs/experiments.md` note 23 |

### The gate mechanism, measured (2026-08-20)

- **The gate mechanism is measured, and it does not match the prediction.**
  From `docs/experiments.md` note 32 (`scripts/gate_boundary_profile.py`, gate
  maps from `scripts/extract_gates.py`, full 189-case test split). The gate is
  the transformer weight — the fusion merge is `cnn + layer_scale * gate *
  attn` — and binned by signed distance to the ground-truth whole-tumour
  surface it is strongly and monotonically organised by anatomy, with
  **adjacent scales at opposite polarity**: fusion level 1 (stride 4) runs
  from 0.9814 deep inside the tumour to 0.3252 in surrounding tissue; level 2
  (stride 8) runs the other way, 0.0089 inside to 0.7992 outside; level 3
  weakly follows level 2; level 0 (stride 2) is essentially flat at ~0.87.
  Paired per-case contrasts (inner margin `[-2, 0)` mm minus elsewhere,
  percentile bootstrap over case indices, 10,000 replicates, n=189) are
  conclusive at every level except one: level 1 vs healthy tissue **+0.4316
  [0.4144, 0.4491]**, level 2 vs healthy tissue **-0.7307 [-0.7456, -0.7154]**.
  The only inconclusive contrast in the whole table is level 0 vs healthy
  tissue, -0.0018 [-0.0148, 0.0118].

  **P1 as literally written is refuted.** It predicted the gate would open
  toward *ambiguous zones* — the tumour margin specifically. It does not: at
  every level the margin is an intermediate point on a monotone
  tumour-to-healthy ramp, never a peak, and at level 1 the inner margin is
  significantly LOWER than the tumour interior (-0.2361 [-0.2526, -0.2188]).
  This document is not being rewritten to say "the gate opens at the
  boundary" — it does not.

  **What did pass is the weaker, more important claim underneath P1:** the
  gate is not decoration. It carries a 0.65-wide monotone spatial signal on a
  [0, 1] scale, with a scale-dependent structure — mid-scale context admitted
  inside the lesion, coarse-scale context admitted outside it, and fine-scale
  context admitted everywhere.

  **On SSA (n=60) the same structure is present but measurably weaker**:
  level 1 spans 0.9744 to 0.3534, and level 2's tumour-interior value rises
  from 0.0089 in-distribution to 0.0551. This is a single descriptive cohort
  comparison with no per-case test behind it, so it is reported here as an
  observation, not a claim.

  **Caveats that must travel with the number.** (i) The crop is one 64³
  tumour-centred patch per case, so "healthy tissue" means peritumoral tissue
  inside that crop, not distant brain. (ii) 172 of 189 cases contribute to the
  interior contrast — 17 tumours have no voxel deeper than 10 mm inside their
  own surface. (iii) The gate is conditioned on the inter-branch ambiguity
  signal, so this shows the gate is organised, **not** that the ambiguity
  conditioning is what organises it — that remains P2, which as of 2026-08-22 is
  finally running as a training ablation rather than being argued from design.
  `scripts/ambiguity_intervention.py` tests an inference-time version of the same
  question in the meantime, and an inference-time intervention on a trained gate is
  **not** a substitute for retraining without the signal — a gate that has already
  learned to use disagreement will degrade when it is removed at test time whether or
  not the model could have learned an equally good gate without it.

Two further results that the pre-registration did not anticipate and that any
rewrite of this document must account for:

- **Calibration is not an advantage.** Fitted temperatures are comparable to the
  baseline's (`[2.05, 1.99, 1.63]` vs `[1.92, 2.02, 1.93]`), so the architecture
  is not intrinsically less overconfident, and `ece_mean` is inconclusive under
  paired statistics in both the uncalibrated and the temperature-scaled variant.
- **The capacity control decomposes the gain.** A plain U-Net widened to within
  0.23% of `neurovision`'s parameter count gains +0.0055 ET on its own; the
  remaining +0.0211 is architecture. That keeps an architectural claim alive —
  but "architecture" here means the whole dual-encoder-plus-gated-fusion design,
  **not** the ambiguity conditioning, which is precisely what P2 exists to
  separate and which remains unmeasured.

- **The accuracy gain does not reach the report.** Added 2026-08-19, from the
  Phase 5 experiment (`docs/experiments.md` note 23). Over 189 paired cases and
  25 report-agreement metrics, Holm-corrected and with patch size controlled at
  64³ across all three models, `neurovision` vs `baseline_unet3d` produces
  **exactly one** conclusive improvement (`relerr_vol_TC`) and
  `capacity_control_unet3d` vs `baseline_unet3d` produces **none**.
  Structure-list Jaccard is 0.9074 / 0.9122 / 0.9108 and 16 of the 25 metrics
  share a median across all three models; epicentre-structure match is
  nominally better for the *baseline*.

  This was not pre-registered as a prediction, and it should have been: it is
  the natural downstream test of an accuracy claim, and it comes back negative.
  The mechanism is mechanical rather than surprising — a structured report is
  dominated by *which structures the tumour overlaps*, and a few voxels at a
  margin rarely change whether a structure is involved at all. The honest
  reading is that the interpretable layer is **stable** with respect to
  segmentation quality across this range, which is a deployment property and
  not an accuracy one. Nothing in this document may be rewritten to claim that
  a better model yields a better report.

**What the paper can honestly say today:** a dual-encoder model with gated
cross-attention fusion beats both a matched-schedule U-Net and a
parameter-matched wide U-Net on ET and TC Dice. The fusion gate's *existence
and spatial organisation* are now measured, not merely argued from design: it
is strongly and monotonically organised by anatomy, with a scale-dependent,
opposite-polarity structure across fusion levels (`docs/experiments.md` note
32). What is still argued from design and pre-registration rather than from an
ablation is the gate's *causal dependence on the ambiguity conditioning* —
whether that specific input, rather than the gate's mere existence and
per-voxel resolution, is what produces this organisation. Any sentence
claiming the *disagreement signal* is what produces the Dice gain requires P2
first. And the gain is a segmentation-metric gain only: it does not propagate
to the structured report, measured.

## What this contribution is not

- Not "we added attention to fusion."
- Not a Dice claim. See `CLAUDE.md`: the headline is competitive Dice with substantially
  better calibration and boundary accuracy.
- Not validated by P1 alone. P1 says the gate moves; P2 says the movement is caused by
  the ambiguity signal; P3 says the movement bought something. All three are needed.

## Open questions before implementation

1. Which scales get a fusion module — all decoder levels, or only the coarse ones where
   context actually differs? Cost is per-scale.
2. Are the auxiliary per-branch projections supervised (deep supervision on the region
   labels) or left unsupervised? Supervised makes "disagreement" meaningful much
   earlier in training; unsupervised is one fewer loss term to tune.
3. Gate output shape: one scalar per voxel, or one per region channel per voxel? The
   latter is strictly more expressive and only 3× the gate channels.
