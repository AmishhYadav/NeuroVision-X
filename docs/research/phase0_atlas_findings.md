# Phase 0 — SRI24 atlas findings

Measured 2026-08-16 against the real atlas files and the real 1,251-case
preprocessed BraTS 2021 tree. Everything here is a measurement, not a citation.
Read alongside `docs/research/interpretable_pipeline_plan.md` §3 (Finding A) and
§5 Phase 0 — **two of that plan's assumptions were wrong and are corrected here.**

---

## Summary of what changed

| Plan said | Measured reality |
|---|---|
| "an SRI24 parcellation drops onto every one of our cases with zero registration and zero resampling" | Grid and spacing are exactly right, **but the atlas is anterior–posterior mirrored relative to BraTS voxel indexing.** A flip is required. It is a pure index reversal — still zero resampling, still zero registration. |
| (not anticipated) | **The `pbmap_*` tissue probability maps are left–right mirrored relative to every other file in the same SRI24 distribution.** |
| "SRI24/TZO … 116 regions" | The volume carries **424 distinct label values**; 1–116 are AAL/TZO, 201+ are manually delineated structures **sliced per anatomical plane**. Merging to parents gives **122 structures**. |
| (not anticipated) | **9 label values present in the volume have no LUT entry**, and 7 LUT entries never appear in the volume. |
| Phase 0 check 1 (brain-mask Dice) treated as a general alignment check | **Brain-mask Dice is structurally blind to a left–right flip** (0.9350 vs 0.9363). It cannot be the laterality check. A content-based replacement is given below. |

---

## Provenance

Downloaded from NITRC `group_id=214`, no login required, 2026-08-16.

| File | SHA-256 | Bytes |
|---|---|---|
| `sri24_labels_nifti.zip` | `ed47fde305baefb687a38e0daa8527896480e4e0159a0a0f56c8cbe09fc97e1f` | 399,821 |
| `sri24_tissue_nifti.zip` | `1375bf8f44a2f6a0704056d87c51ab8dd8d499132d9b06bf9f7e1b6b857673a2` | 15,686,657 |
| `sri24_anatomy_nifti.zip` | `c2a66abd5bb37ba443b935ef4732a3e3a2f34ca57bedcdac6cf5902b50f013b6` | 6,522,201 |

Licence: CC-BY-SA (`LICENSE` ships inside every archive). Downloaded by
`scripts/fetch_atlas.py`, never committed.

---

## Finding A confirmed, and confirmed before loading a single voxel

`tzo116plus.nii` is 17,856,352 bytes. 240 × 240 × 155 = 8,928,000 voxels; at
int16 that is 17,856,000, plus the 352-byte NIfTI-1 header = **17,856,352
exactly**. Every file in the distribution matches the same grid at its own
dtype (`uint8` → 8,928,352; `float32` → 35,712,352).

All files: `shape = (240, 240, 155, 1)`, `zooms = (1, 1, 1)`. The trailing
singleton axis is real and must be squeezed on load.

So the grid claim is right. The *orientation* claim was never checked.

---

## Finding G (new) — the atlas is A–P mirrored relative to BraTS

**Affine algebra.** BraTS `meta.json` carries
`[[-1,0,0,0], [0,-1,0,239], [0,0,1,0]]`, i.e. `y_world = 239 - j`.
SRI24 `tzo116plus`, `lpba40`, `tissues`, `suptent`, `spgr` all carry
`[[-1,0,0,0], [0,1,0,0], [0,0,1,0]]`, i.e. `y_world = +j`.

Setting the two equal: `239 - j_brats = j_sri24`, so `j_brats = 239 - j_sri24`.
Axis 1 has exactly 240 samples, so this is **precisely `atlas[:, ::-1]`** — an
exact index reversal with no offset, no interpolation and no resampling. The
"registration-free by construction" property in the plan's §10 item 1 survives
intact; it just needs one array flip stated out loud.

**Empirical confirmation 1 — brain-mask Dice.** SRI24 `spgr > 0` against a
BraTS brain mask built by uncropping 12 real cases into original geometry:

| Transform | Dice |
|---|---|
| as-is | 0.7341 |
| flip axis 0 (L–R) | 0.7344 |
| **flip axis 1 (A–P)** | **0.9350** |
| flip axis 2 (I–S) | 0.7181 |
| flip axes 0+1 | 0.9363 |

**Empirical confirmation 2 — population lobe distribution**, 400 random cases,
lobe holding the largest share of whole-tumour overlap:

| Lobe | Ours (flipped) | Larjavaara 2007 |
|---|---|---|
| frontal | 30.5% | 40% |
| temporal | 37.8% | 29% |
| parietal | 17.8% | 14% |
| deep | 9.2% | 14% |
| occipital | 4.2% | 3% |

and the **counter-test with the atlas left unflipped**, which is what makes this
proof rather than vibes: frontal **62%**, temporal 24%, deep 12%, parietal
**2%**. An A–P mirror reads occipital and parietal content as frontal, and that
is exactly the signature observed.

### Finding K (new) — the lobe check is NOT the strongest check, and its natural summary statistic prefers the wrong answer

The plan calls the population lobe distribution "the strongest [check]
available". Measured, it is not, and summarising it the obvious way is actively
harmful.

Scored with Spearman rank correlation against the reference ordering:

| Orientation | Spearman rho |
|---|---|
| correct (A–P flipped) | **+0.872** |
| **mirrored (unflipped)** | **+0.975** |

The mirrored atlas scores *higher*. Its ranking is frontal > temporal > deep >
parietal > occipital, which misorders only the adjacent deep/parietal pair,
whereas the correct orientation swaps the top two for the real cohort reason
below. **A Phase 0 gate built on rank correlation would have preferred a
mirrored atlas.** Rank correlation over five bins is close to useless here: it
throws away exactly the magnitudes that carry the signal.

Mean absolute deviation in percentage points is better but still not decisive —
5.7 pp correct vs 7.9 pp mirrored. The single discriminating quantity is
**parietal share: 17.8% correct vs 2.5% mirrored, against a reference of 14%.**
An A–P mirror reads parietal and occipital content as frontal, so parietal
collapses by 5.6x and occipital vanishes from the ranking entirely.

Consequence for the gate, and it is a reordering of the plan's Phase 0:

1. **Primary — brain-mask Dice.** 0.9350 correct vs 0.7341 mirrored. The widest
   margin of any check by far, and the one to gate on.
2. **Primary — laterality from `_L`/`_R` centroid pairs** (Finding J). The only
   check that can see a left–right flip at all.
3. **Supporting — lobe distribution.** Report per-lobe percentages with all
   three caveats. Score by absolute deviation, **never by rank correlation**.
   Advisory, not pass/fail.
4. **Visual QC on ~10 cases**, unchanged.

**Honest deviation to report, not hide:** frontal and temporal are rank-swapped
against the reference. Two reasons, both pre-registered in the plan: BraTS 2021
is overwhelmingly high-grade glioma, and GBM has a documented temporal
predilection relative to gliomas overall, whereas Larjavaara pools all grades;
and the reference counts *tumours per lobe* while we take *largest share of WT
voxel overlap*, and WT includes oedema which tracks white matter. A third
confound is ours alone: the AAL-name → lobe mapping used above is a hand-written
heuristic (Fusiform → temporal, Insula → deep are both arguable). That mapping
therefore ships as an explicit, reviewable config artifact rather than as
heuristics buried in a function. Rough magnitude holds for all five lobes; rank
holds for three of five with the top two swapped.

---

## Finding H (new) — `pbmap_*` disagrees with the rest of SRI24 on left–right

`pbmap_GM/WM/CSF.nii` carry `[[1,0,0,-239], [0,1,0,0], [0,0,1,0]]` — axis 0
sign **inverted** relative to `tissues.nii`, `tzo116plus.nii` and `spgr.nii` in
the same distribution.

Measured, `pbmap_GM > 0.5` against `tissues`:

| `tissues` value | as-is | flip axis 0 |
|---|---|---|
| 1 | 0.1469 | 0.0046 |
| **2** | 0.7018 | **0.9521** |
| 3 | 0.1656 | 0.0090 |

So `tissues` codes **1 = CSF, 2 = GM, 3 = WM**, and the `pbmap_*` volumes need
`[::-1]` on axis 0 to sit on the same voxels as everything else. Solving the
affines gives `i_main = 239 - i_pbmap`, again an exact index reversal.

This is a **left–right flip living inside the atlas distribution itself** — the
plan's own "High — clinical" risk, present in the source data rather than in our
code. It is also the reason the load path must apply a per-file transform
derived from that file's own affine, never one transform for the whole atlas.

Because `tissues.nii` is a hard segmentation on the majority frame and covers
GM/WM/CSF, **`tissues.nii` is the default tissue source and `pbmap_*` is
opt-in.** Fewer moving parts on the path that VASARI F20/F21 will depend on.

---

## Finding I (new) — the LUT is not 116 rows, and it is not tab-separated

`SRI24-tzo116plus.txt` is 422 data rows in `id name R G B A` form. Rows 1–116
use tabs; rows for labels 201+ use **spaces**. A `split("\t")` parser silently
stops mapping at 116 while the volume keeps 424 values — every unmapped voxel
would then read as background. Parse on generic whitespace; all names are
underscore-joined so no name contains a space.

Labels 201+ are the "plus": manually delineated structures **sliced per
anatomical plane** — `LateralVentricle_L_y48 … _y139`, `Pons_x111 … _x114`,
`ThirdVentricle_{L,R}_y*`, `CorpusCallosum_AP_0 … _AP_8`. They are construction
artifacts of the atlas, not anatomical categories. Merging on
`(_[xyz]\d+|_AP_\d+)$` gives **122 parent structures** = 116 AAL + 6:

| Parent | sub-labels | voxels |
|---|---|---|
| `LateralVentricle_L` | 107 | 9,103 |
| `LateralVentricle_R` | 107 | 10,524 |
| `ThirdVentricle_R` | 33 | 1,173 |
| `ThirdVentricle_L` | 32 | 410 |
| `Pons` | 17 | 5,679 |
| `CorpusCallosum` | 9 (`_AP_0..8`) | 1,635 |

**This resolves plan open decision #1** (full TZO vs merge to ~128): merge to
**122**, which is the atlas's own structure count once its per-plane slicing is
undone. Nothing is invented and nothing is thrown away.

Leaving `CorpusCallosum_AP_*` unmerged would give 130 instead, and genu / body /
splenium involvement is genuinely more informative than "corpus callosum". It is
still merged, for a specific reason: **the atlas names those nine sub-labels
`_AP_0` … `_AP_8` and nothing else.** Splitting genu from splenium would mean
assigning anatomical names to bare A–P indices ourselves, which is the kind of
invention this pipeline exists to avoid. "Tumour involves the corpus callosum"
is a claim the atlas supports; "tumour involves the splenium" is not.

It also delivers something the plan did not expect to have: **ventricle labels**.
Lateral and third ventricles are present, so VASARI F19 (ependymal invasion) and
the Phase 3b ventricular-compression / midline-shift measures have a source in
SRI24 native space — no MNI bridge, consistent with Finding D.

**9 values in the volume have no LUT row** (422, 424, 426, 428, 430, 432, 434,
476, 478) and 7 LUT rows never appear in the volume (201, 203, 205, 219, 221,
227, 413). The unmapped values are all even and fall inside the lateral-ventricle
band, where odd = `_L` and even = `_R`, so they are *probably* further
right-lateral-ventricle slices. **They are mapped to `unclassified` anyway**, and
counted in the coverage line. Absorbing them on a numeric pattern would be
exactly the guess this project keeps getting burned by; the volume involved is
small and the coverage line makes the gap visible rather than hidden.

---

## Finding J (new) — brain-mask Dice cannot see a left–right flip

From the table in Finding G: flip-axis-1 scores 0.9350 and flip-axes-0-and-1
scores 0.9363. The brain is very nearly left–right symmetric, so **the plan's
Phase 0 check 1 is structurally incapable of detecting the single most damaging
error this layer can make.** It would have passed a left–right mirrored atlas.

The replacement is content-based and needs no affine, which matters because
Finding H shows the affines within one distribution are not mutually consistent:
**the TZO LUT names carry explicit `_L` / `_R` suffixes**, so laterality can be
proved from the parcellation itself.

Measured, atlas in native frame, BraTS convention (axis 0 low index = patient
right): 54 `_L`/`_R` structure pairs, **0 violations** of "`_L` centroid > 119.5
and `_R` centroid < 119.5". Mean axis-0 centroid `_L` = 143.8, `_R` = 94.4.

Bonus result, and a genuinely independent one: the midpoint of those two means
is **119.1**, against the 119.5 grid centre that `anatomy/burden.py` already
assumes as `midline_index`. That constant was an assumption about SRI24 being
centred in its own grid when it shipped; it is now confirmed from atlas content.

---

---

## Verified end to end, 2026-08-16, on the real atlas through `load_atlas`

Unit tests on synthetic NIfTI files prove the transform algebra, not the atlas.
These are measurements on the loaded `Atlas` object against the real
preprocessed BraTS tree — the standard this project adopted after the
calibration-mask bug shipped behind a green suite.

| Quantity | Measured |
|---|---|
| shape / dtype / contiguity | `(240, 240, 155)`, `int16`, C-contiguous |
| structures after merge | **122** |
| labelled voxels | 1,053,253 |
| unmapped label values | 9, covering **22 voxels** total |
| tissue GM / WM / CSF | 648,426 / 520,983 / 282,297, zero overlap |
| ventricles present | `LateralVentricle_{L,R}`, `ThirdVentricle_{L,R}` |
| `Pons`, `CorpusCallosum` present | yes |
| `_L`/`_R` pairs | 56, **0 violations** |
| midline from `_L`/`_R` centroids | **119.10** (assumed 119.5) |

**Brain-mask gate, 40 cases:**

| Atlas orientation | Dice vs BraTS |
|---|---|
| correct | **0.9394** |
| A–P mirrored | 0.7334 — fails the 0.85 gate |
| L–R mirrored | **0.9416** — *passes*, and scores higher than correct |

The last row is Finding J proved on the shipped object rather than argued:
the primary gate cannot see a left–right flip, which is why the `_L`/`_R`
laterality check is not optional.

**One trap worth naming.** The atlas brain mask is `tissue > 0`, **not**
`parcellation > 0`. AAL parcellates grey matter, so the parcellation covers
1,053,253 of 1,451,706 brain voxels and scores **0.8013** against BraTS — which
reads as a failed gate and is nothing of the kind. `tissue > 0` and `spgr > 0`
are the same 1,451,706 voxels exactly, so the tissue map is the brain mask and
`spgr.nii` need never be loaded.

---

## What the load path must therefore do

1. Squeeze the trailing singleton axis — every file is 4-D.
2. Derive the flip **per file, from that file's own affine**, and assert the
   result is a pure axis reversal with no offset or interpolation. Never apply
   one transform to the whole atlas (Finding H).
3. Parse the LUT on whitespace, not tabs (Finding I).
4. Merge per-plane sub-labels to parents on `(_[xyz]\d+|_AP_\d+)$` (Finding I).
5. Map any volume value with no LUT row to `unclassified` and count it.
6. Prove laterality from `_L`/`_R` centroids, not from the affine (Finding J).
