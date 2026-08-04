# Contribution

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
