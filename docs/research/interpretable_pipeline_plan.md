# NeuroVision-X — the interpretable pipeline direction

Planning document for the project's second contribution attempt: an end-to-end,
interpretable pipeline from scan to structured anatomical report.

Written 2026-08-16, after the original reliability claim was measured and did
not hold. Read `docs/experiments.md` notes 11–17 before this file: the numbers
that forced the pivot are there, not here.

**Revised the same day**, after a literature and resource review triggered by a
single question: what happens to Phase 2 when no neuroanatomy reviewer is
available? Three things changed as a result — Phase 2 was rewritten around
claims that need no expertise to check (Finding E, §5), the MNI bridge was
considered and declined (Finding D, §3), and the paper's framing was revised
again because close prior work already exists (Finding F, §3 and §10).

---

## 1. Why the direction changed

The original claim was *"competitive Dice with substantially better calibration
and boundary accuracy."* All three parts have now been measured on the test
split against a controlled baseline:

| Part of the claim | Measured verdict |
|---|---|
| Competitive Dice | **Exceeded** — ET +0.0267 (p_holm 1.4e-21), TC +0.0103 |
| Better calibration | **Not supported** — mean ECE within noise, calibrated or not |
| Better boundary accuracy | **Not supported** — every HD95 difference within noise |
| Better failure ranking (added later) | **Not supported** — risk-coverage within noise |

So the project has one strong, real result — a 17.1% relative reduction in ET
error — and no reliability result. Continuing to write the reliability paper
would mean claiming three things the data refuses to support.

The pivot: stop claiming the model is more *trustworthy*, and build the thing
the accuracy result actually enables — a pipeline that turns a segmentation
into a **structured, anatomically-grounded, quantified report**, with the
calibrated uncertainty we already produce attached to it.

This is an application/systems contribution rather than a methods one. It is a
different and more modest kind of paper, and it has the advantage of being
supported by results we already have rather than results we hoped for.

---

## 2. Scope, and what is deliberately NOT claimed

State these before designing anything, because every one of them is a place
where an over-claim would be easy and damaging.

**In scope**

- Which anatomical structures a predicted tumour overlaps, quantified two ways.
- Which white-matter pathways are involved.
- A literature-grounded, cited mapping from structure to the functions it
  supports and the deficits associated with damage there.
- A quantitative tumour burden profile (volumes, ratios, multifocality,
  mass-effect indices).
- All of the above rendered into a report and into the existing demo viewer,
  with uncertainty and caveats visible in the output itself.

**Explicitly NOT in scope, and why**

| Not claimed | Reason |
|---|---|
| Identifying a **cell** | One voxel is 1 mm³ and contains ~10⁵ neurons plus glia. MRI cannot resolve cells. Glioma cell-of-origin (astrocytic, oligodendroglial) is a histopathology question. The deliverable is *structures*, not cells. |
| A WHO **grade** | WHO CNS5 (2021) grading is *integrated* — it requires histology **plus** molecular markers (IDH, 1p/19q, ATRX, TERT, CDKN2A/B). It is not determinable from MRI, and our dataset carries no grade label. |
| A tumour **stage** | Diffuse gliomas are not staged. There is no TNM system for them; they are graded. "Stage" is the wrong word and using it would signal unfamiliarity with the domain. |
| A **prognosis** or predicted patient outcome | We have no clinical outcomes to validate against (see §3, Finding C). |
| An **eloquence assessment** | VASARI's F3 (eloquent brain involvement) is deliberately not automated. VASARI-auto declined it too — "we did not wish to detract from a gold standard of a neurosurgeon's electrical stimulation assessment for eloquent-sparing resections" — and F3 is among the features with poor inter-rater agreement even between consultants. We report a strictly weaker, purely geometric quantity: overlap with, and millimetre distance to, structures that a **named published classification** designates eloquent. See §5 Phase 2, Tier C. |
| A **deficit** the patient has or will have | Decided 2026-08-16: the shipped pipeline stops at Tier C. No deficit sentences appear in any artifact. |
| Anything **diagnostic** | Framed throughout as a research and educational decision-support demo. |

**The honesty constraint that shapes the whole design:** we can *validate* the
anatomy layer (atlas alignment is checkable), but we **cannot validate a
deficit layer**, because BraTS ships no clinical outcomes. An unvalidatable
layer that we also cannot get reviewed (see Finding E) does not ship. Every
claim in the output must therefore fall into one of exactly two categories:

1. **Geometric** — computed deterministically from a mask and a label map, and
   reproducible by anyone with the same inputs.
2. **Referential** — a lookup into a *named, published, citable* classification,
   where our contribution is the mapping and the source owns the claim.

Anything that is neither does not go in the artifact. This is what replaces the
"reviewed by someone with neuroanatomy background" mitigation the first draft
of this plan relied on.

---

## 3. Enabling findings

Three facts, established by inspecting the data on disk and by literature
search, that determine the entire implementation.

**Finding A — the atlas needs no registration.** BraTS preprocessing
co-registers every case to the **SRI24** template, resamples to 1 mm isotropic
and skull-strips it. SRI24's native grid is **240 × 240 × 155 at 1 mm
isotropic** — *exactly* the BraTS grid. Confirmed against our own data:
`meta.json` reports `original_shape [240, 240, 155]`, `spacing [1, 1, 1]`,
affine `diag(-1, -1, 1)` with a 239 offset on axis 1.

Consequence: **an SRI24 parcellation drops onto every one of our 1,251 cases
with zero registration and zero resampling.** The hardest and least reliable
step in this kind of work — per-patient nonlinear registration to an atlas,
which is exactly the step that fails worst near a lesion — is simply not
required. This is the single biggest reason this feature is tractable.

Re-verified 2026-08-16 over **all 1,251** preprocessed `meta.json` files, not a
sample: `original_shape (240, 240, 155)`, `spacing (1, 1, 1)`, affine diagonal
`(-1, -1, 1)` — 1251/1251 identical, zero variation. BraTS 2021's published
preprocessing is rigid registration to SRI24, resampling to 1 mm isotropic, and
skull-stripping, and the SRI24 paper states the atlas grid as 240 × 240 × 155 at
1 mm. Independent corroboration: MNI152 at 1 mm is 182 × 218 × 182, so the grid
alone rules out the alternative template.

**Finding B — SRI24 ships its own parcellations, openly licensed.** The atlas
includes T1/T2/PD channels, GM/WM/CSF tissue probability maps, DTI-derived
channels (FA, MD), and **two label maps**: cortical regions and subcortical
structures. The cortical map (SRI24/TZO) is built on the Tzourio-Mazoyer AAL
parcellation with additional manually delineated structures. It is distributed
via NITRC under **CC-BY-SA**.

Consequence: no atlas needs to be constructed, and no MNI→SRI24 registration
needs to be performed and validated. CC-BY-SA has share-alike implications if
we redistribute it, so the atlas is **downloaded by a script, never vendored
into the repo** (§7).

**Finding C — there is no clinical metadata in our data.** Verified: our copy
of BraTS 2021 (`dschettler8845/brats-2021-task1`) contains images and
segmentations only. No `name_mapping.csv`, no grade, no survival, no molecular
markers. `metadata.csv` carries only geometry and voxel counts.

Consequence: any grading or outcome claim requires a *different dataset*
(§8). It is not a modelling problem, it is a data problem.

**Finding D — SRI24 is not MNI space, and that cuts both ways.** SRI24 was
built by unbiased, template-free groupwise nonrigid registration of 24 subjects
into its **own population-average coordinate system**. It was deliberately not
registered to MNI152 or ICBM452. Finding A therefore holds only for resources
that ship *in SRI24 space*: the TZO (`tzo116plus`) and LPBA40 parcellations, the
GM/WM/CSF tissue probability maps, the ML tissue segmentation, and the FA/MD/
diffusivity channels. Everything else in the field — JHU ICBM-DTI-81 tract
labels, Brainnetome, Harvard-Oxford, Neurosynth/NeuroQuery meta-analytic maps —
is MNI and would need a one-time nonlinear SRI24↔MNI warp plus a registration
dependency the stack does not have.

**Decision, 2026-08-16: no bridge.** The pipeline stays entirely in SRI24 native
space. Consequences, accepted deliberately:

- No **named** white-matter tracts. The FA/MD channels say *where* white matter
  is, not *which tract* it is; tract identity requires the MNI atlas. Replaced
  by deep-white-matter involvement computed from the tissue maps, which is what
  VASARI F21 asks for anyway and needs no tract atlas.
- No meta-analytic functional decoding. (Also note Neurosynth is **ODbL**, whose
  share-alike applies to derived databases — a licensing question we now avoid.)
- In exchange: zero registration anywhere in the pipeline, zero warp to
  validate, and no new dependency. "No registration at any stage" becomes a
  defensible property of the system rather than an unexamined assumption.

Supporting reason to stay out of MNI tract atlases specifically: ICBM-DTI-81
shipped with a left–right mirrored orientation and 15 mislabelled regions in its
2010 version, and the 2012 "fix" edited the label lookup table rather than the
image — so labels formally aligned while the coordinate frame stayed flipped.
That is precisely this project's recurring failure mode: everything plausible,
nothing raising. Inheriting it on top of our own warp error is a poor trade for
a layer the report does not need.

**Finding E — there is no domain reviewer, and expert review is a weaker
instrument than the first draft assumed.** No one with a neuroanatomy or
neuro-oncology background is available to review this work. Separately, the
published evidence says such review would not have been an oracle: VASARI-auto
measured two consultant neuroradiologists against each other on 100
glioblastomas and found mean Cohen's **κ = 0.49 ± 0.32**, with several features
at or below chance (F24 −0.23, F11 −0.03). Automated-vs-rater was 0.42 ± 0.34 —
statistically indistinguishable from human-vs-human — while the automated method
against itself across different segmentations scored 0.94 ± 0.10.

Consequence, and it is the reason §5 Phase 2 was rewritten: the response to a
missing reviewer is **not** to proceed on model authority and hope, and not to
abandon the layer. It is to restrict every shipped claim to something that does
not need expertise to check — geometry, or a lookup into a published
classification (§2, the two categories).

**Finding F — close prior work exists and is very recent.** Two papers cover
much of what §4 proposes:

- **VASARI-auto** (Ruffle et al., *NeuroImage: Clinical* 2024; arXiv 2404.15318;
  code at `github.com/jamesruffle/vasari-auto`) — deterministic automation of 16
  VASARI features from glioma lesion masks, n = 1172, nonlinearly registered to
  1 mm MNI with enantiomorphic correction. Notably **excludes F3 (eloquence)** on
  principle, and excludes seven further features needing diffusion sequences,
  non-brain-extracted data, or carrying "risk of confabulation".
- **BTReport** (Heras Rivera et al., arXiv 2602.16006, Feb 2026) — deterministic
  feature extraction with an LLM used *only* for narrative structuring, on BraTS,
  grounded in VASARI, releasing a BTReport-BraTS synthetic-report dataset. (Read
  from the abstract; full-text extraction was not available locally.)

Consequence: "we built an automated structured-report pipeline for glioma MRI"
is no longer novel and must not be the paper's claim. See §10.

---

## 4. Pipeline architecture

```
preprocessed case ──► [existing] NeuroVisionX ──► region logits
                                                      │
                                    ┌─────────────────┼──────────────────┐
                                    ▼                 ▼                  ▼
                            postprocess         MC-dropout        calibrated
                            (ET/TC/WT mask)     (uncertainty)     probabilities
                                    │                 │                  │
                                    └────────┬────────┴──────────────────┘
                                             ▼
                        ┌────────────────────────────────────────┐
                        │  NEW: anatomy layer (Phase 1)          │
                        │  SRI24 parcellation ∩ mask             │
                        │  → structure involvement table         │
                        └────────────────────┬───────────────────┘
                                             ▼
                        ┌────────────────────────────────────────┐
                        │  NEW: reference layer (Phase 2)        │
                        │  structure → published eloquence class │
                        │  + mm distance. No deficit text.       │
                        └────────────────────┬───────────────────┘
                                             ▼
                        ┌────────────────────────────────────────┐
                        │  NEW: burden layer (Phase 3)           │
                        │  volumes, ratios, mass effect          │
                        └────────────────────┬───────────────────┘
                                             ▼
                        ┌────────────────────────────────────────┐
                        │  NEW: report layer (Phase 4)           │
                        │  JSON + Markdown + demo panel          │
                        └────────────────────────────────────────┘
```

Every new layer is **CPU-only, deterministic, and consumes artifacts
`scripts/evaluate.py` already writes**. None of it requires a GPU, a training
run, or a new checkpoint. That is a deliberate design property: the entire
contribution can be built and iterated on the Mac, and costs zero Kaggle hours.

---

## 5. Phases

Effort figures are working days for one person, and assume the existing
conventions (type hints, Google docstrings, CPU shape tests under a second,
config through Hydra, no hardcoded paths).

### Phase 0 — Atlas acquisition and alignment proof (2–3 days)

**The gate for everything else.** If the atlas does not provably align, nothing
downstream means anything.

Deliverables:
- `scripts/fetch_atlas.py` — downloads SRI24 from NITRC to a configured path,
  verifies a checksum, never writes into the repo tree.
- `configs/anatomy/sri24.yaml` — atlas paths, label-map selection, version.
- `src/neurovision/anatomy/atlas.py` — `load_atlas`, `AtlasLabels` (id → name,
  laterality, tissue class), `atlas_to_case_geometry`.

**Alignment validation, and this is the part that must not be skimped.** The
project's own history says shape assertions do not catch geometry bugs — the
demo's slice viewer indexed every voxel correctly and still displayed the head
on its side, and that was found by looking at it. Four independent checks:

1. **Brain-mask agreement.** Atlas brain mask vs each case's nonzero-intensity
   mask, Dice over all 1,251 cases. Report the distribution; investigate the
   tail. A systematic misalignment shows up here immediately.
2. **Tissue-channel consistency.** Voxels the atlas calls CSF should be
   dark on T1 and bright on T2 across cases; GM/WM intensities should separate.
   A scrambled or flipped atlas destroys this.
3. **Laterality proof.** Left/right derived from the affine, never assumed, and
   asserted against the known `diag(-1, -1, 1)` orientation. Same class of bug
   as the demo's radiological-convention fix, and a left/right swap in a report
   about which hemisphere is affected is the single most damaging error this
   feature could make.
4. **Population-level plausibility, against published numbers.** Compute the
   tumour-location distribution over all 1,251 cases and compare it with the
   epidemiological literature, which reports gliomas by lobe as **frontal 40%,
   temporal 29%, parietal 14%, occipital 3%, deeper structures 14%**
   (Larjavaara et al., *Neuro-Oncology* 2007). A scrambled, flipped or
   mis-scaled atlas cannot reproduce that ordering by accident. This is a
   cheap, quantitative, independent check and it is the strongest one
   available — and it now has a target to hit rather than a vibe to match.

   Two caveats to state when reporting it. Our cohort is BraTS 2021, which is
   overwhelmingly high-grade glioma, while the reference figures are for gliomas
   generally; and the reference counts *tumours per lobe* while our measure is
   voxel overlap with a parcellation, so exact agreement is not expected. The
   check is on **rank order and rough magnitude**, and that is enough to catch
   every failure it is meant to catch.

Plus visual QC on ~10 cases (atlas boundaries overlaid on T1), because the
lesson from this project's qualitative panels and slice viewer is that some
bugs surface **only** by looking at the render.

Risks: NITRC availability; parcellation may need label merging (a known
approach merges to ~128 regions); atlas may ship in Analyze/NRRD needing
conversion.

### Phase 1 — Anatomical localisation engine (3–4 days)

`src/neurovision/anatomy/localize.py`

Core function: given a binary region mask and the atlas, produce a table.

**Two fractions, both required, and confusing them would be a serious error:**

- `frac_of_tumour` — what share of the tumour sits in this structure. Answers
  *"where is the tumour?"*
- `frac_of_structure` — what share of this structure is occupied by tumour.
  Answers *"how badly is this structure affected?"*

A lesion can place 5% of its volume in the brainstem while that 5% destroys 40%
of the brainstem. Reporting only the first would call that a footnote; it is
the most important line in the report. Both columns ship, and the report sorts
by clinical salience rather than by raw volume.

Geometry note: predictions are saved in **original 240×240×155 geometry**
(`uncrop_to_original`), while preprocessed images and labels are **cropped to
the nonzero bbox**. The atlas is in original geometry. So either uncrop the
mask or crop the atlas with the case's own `meta.json` bbox — and whichever is
chosen must be verified the way the demo verified its re-crop: not by a unit
test, but by recomputing a known quantity end-to-end and matching a number that
was already published.

Outputs: per-case `anatomy.csv`, one row per (region, structure).

Tests: synthetic masks placed in known atlas structures; a mask covering an
entire structure must give `frac_of_structure == 1.0`; laterality round-trip;
empty mask → empty table, not a crash.

### Phase 2 — Eloquence reference layer (2–3 days)

`data/knowledge/eloquence_map.yaml` (versioned, in-repo)

**Rewritten 2026-08-16.** The original design was a knowledge base of functions
and deficits, authored here and mitigated by external expert review. Finding E
removed the mitigation and showed it was weaker than assumed; Finding F showed
that the one team in this exact domain with expert access declined to automate
the same claim. The layer is therefore restricted to what can be checked without
domain expertise.

**Four tiers, separated by how a claim is justified rather than by subject.**

| Tier | Claim | How it is justified | Ships? |
|---|---|---|---|
| A | structure name, laterality, tissue class | SRI24 ships it; Phase 0 validates the alignment | yes |
| B | cortical / deep-WM / ependymal involvement, midline crossing, multifocality, epicentre | deterministic geometry over label + tissue maps; VASARI definitions, with published human-agreement numbers to compare against | yes (Phase 3) |
| C | overlap with, and mm distance to, structures a **named published classification** calls eloquent | the source owns the claim; we own only the mapping | yes |
| D | function and deficit text | needs literature judgement we cannot check | **no — cut** |

Tier C is strictly weaker than VASARI F3 and deliberately so. It is a geometric
statement about a published list, not an assessment of this patient.

**Schema.** One entry per SRI24/TZO label, and `evidence` is the field that
makes the whole layer auditable by a non-expert:

```yaml
version: 1
classification:
  name: Sawaya eloquence grading
  citation: <full citation + PMID/DOI>
  eloquent_structures_verbatim: >
    motor/sensory cortices, visual center, speech center, internal capsule,
    basal ganglia, hypothalamus/thalamus, brainstem, dentate nucleus
near_eloquent_rule:
  # Sawaya's "near-eloquent" is documented as ambiguous in the literature, so we
  # do not guess at it: the report emits the measured distance and applies an
  # explicit, configurable threshold, stated in the artifact.
  distance_mm: 10
entries:
  - structure_id: 1
    structure_name: Precentral_L      # copied verbatim from the SRI24 LUT
    eloquence: eloquent
    matched_term: motor cortex        # the term in the source it matches
    evidence: "<verbatim sentence from the cited source>"
    source: <citation + PMID/DOI>
```

**Hard rules, all enforced by tests:**

- **Every entry stores the verbatim supporting sentence**, not just a citation.
  This is the substitute for expert review, and it changes the verification
  question from *"is this neuroanatomy correct?"* — which needs an expert — to
  *"does this quoted sentence say this thing about this structure?"*, which the
  author can check on every entry in a single sitting.
- **The default is `unclassified`, never `non-eloquent`.** Absence of a source is
  not evidence of functional silence, and defaulting the other way would
  manufacture a reassuring claim out of a gap in our sourcing. Reports render
  `unclassified` explicitly.
- **Structure names are copied from the atlas LUT, never retyped.** A test
  asserts every `structure_id` exists in the loaded label map and that
  `structure_name` matches it exactly — a mismatch means the mapping is against
  a structure that is not the one being measured.
- `eloquence` comes from a closed vocabulary; anything else raises.
- The report prints a **coverage line**: *N of M structures classified, K
  unclassified*, so a thin knowledge base is visible in the output rather than
  hidden by it.

**No white-matter tract identity** (Finding D). "Deep white matter involvement"
comes from the SRI24 tissue maps in Phase 3, VASARI-F21 style. The report does
not name tracts, and must not imply it knows which pathway is involved.

Effort drops from 4–6 days to 2–3 because the sourcing target is a single
published classification over ~116 labels rather than a functional profile per
structure.

**If Tier D is ever revived**, the bar is: restricted to the eloquent set, two
independent sources for `well-established` and one capping at `limited`, the
same verbatim-evidence rule, hedged wording, and no entry without a qualifying
source. That was designed and costed; it was cut on scope, not on feasibility.

### Phase 3 — Tumour characterisation profile (3–4 days)

`src/neurovision/anatomy/burden.py` — the honest replacement for "stage".

- Volumes of ET / TC / WT in mm³ (1 mm isotropic, so voxel count *is* volume)
- Necrotic-core fraction, enhancing fraction, oedema-to-core ratio
- Multifocality: connected-component count and volume distribution
- Shape: sphericity, surface-area-to-volume
- Mass effect: midline shift and ventricular compression, exploiting that
  every case is in a common template where the midline is known a priori
- Anatomical involvement summary from Phase 1

**Use VASARI's definitions wherever one exists, rather than inventing a
parallel vocabulary.** The overlap is large and mostly free: proportion
enhancing (F5), proportion non-enhancing (F6), necrosis proportion (F7),
multifocal/multicentric (F9), oedema proportion (F14), ependymal invasion
(F19), cortical involvement (F20), deep white matter invasion (F21), midline
crossing (F22–23), satellite lesions (F24), epicentre location and side
(F1–F2). Three reasons: the definitions are published and peer-reviewed, so we
are not the authority; VASARI-auto reports per-feature agreement against
consultant neuroradiologists, so our numbers land next to a published human
benchmark instead of floating free; and it makes the output comparable with
other work rather than idiosyncratic.

Note F19/F20/F21 need only the SRI24 tissue maps and ventricle labels — no
tract atlas, which is what makes them the right substitute for the tract layer
Finding D cut.

Reported as a **descriptive profile**, never as a grade or stage. These
quantities are associated with aggressiveness in the literature and are
genuinely informative; they are not a substitute for histopathology and the
output says so.

Tests: hand-computed volumes on synthetic shapes; a perfect sphere must give
sphericity 1.0; midline shift of a deliberately displaced synthetic mask.

### Phase 4 — Report generation and demo integration (4–5 days)

- `src/neurovision/reporting/report.py` — assembles a versioned JSON report
  and renders Markdown/HTML.
- `scripts/report.py` — Hydra entry point, one report per case.
- Demo: new `/api/report/<case_id>` route; a frontend panel showing structure
  involvement, pathways, burden profile, and the existing uncertainty layer.

Presentation requirements, non-negotiable and enforced by tests:
- The non-diagnostic disclaimer is **in the report artifact itself**, not only
  in a README.
- The atlas name, version and the mass-effect caveat appear next to any
  anatomical claim.
- Deficit statements render in hedged language with their citation visible.

### Phase 5 — Validation, documentation, paper (5–7 days)

- Population-level anatomical statistics across all 1,251 cases as a headline
  figure (it is both a validation and a genuinely interesting result).
- Agreement between reports generated from ground-truth masks and from
  predicted masks — this is the one place the *model's* contribution to report
  quality can be measured, and it is a real experiment: how often does a
  segmentation error change the reported structure list?
- That last item is the paper's quantitative core, and it is where the
  accuracy result becomes relevant: a more accurate model should produce
  reports that agree more often with ground-truth-derived reports. **This is
  the experiment that connects Phase 1–4 to the +0.0267 ET Dice result**, and
  it should be designed before implementation, not after.

---

## 6. Total effort and sequencing

| Phase | Days | Blocking? |
|---|---|---|
| 0 — Atlas + alignment proof | 2–3 | **Gate for everything** |
| 1 — Localisation engine | 3–4 | needs 0 |
| 2 — Eloquence reference layer | 2–3 | parallel with 1 |
| 3 — Burden profile | 3–4 | independent, can start now |
| 4 — Report + demo | 4–5 | needs 1, 2, 3 |
| 5 — Validation + paper | 5–7 | needs 4 |

**≈ 19–26 working days**, zero GPU hours (was 21–29; Phase 2's rewrite removed
2–3 days of sourcing). Phase 3 depends on nothing new and could start
immediately if a quick win is wanted.

---

## 7. Data, licensing, reproducibility

- SRI24 is **CC-BY-SA**. It is **downloaded by `scripts/fetch_atlas.py`, never
  committed**, for the same reason BraTS is not committed: licence hygiene and
  repo size. Attribution goes in the README and in every generated report.
- The knowledge base **is** committed — it is our own compilation — with each
  entry's citation.
- Atlas version is recorded in every report, so a report can be traced to the
  parcellation that produced it.
- No new Python dependency. Nibabel handles NIfTI, scipy handles connected
  components and distance transforms, both already pinned. A registration tool
  was considered and **declined** (Finding D), which is what keeps this true.
- Staying in SRI24 native space also keeps the licence surface small: SRI24 is
  the only external data artifact, at CC-BY-SA, downloaded and never vendored.
  The MNI-space route would have added Neurosynth's **ODbL**, whose share-alike
  clause reaches derived *databases* — a live question for a repo that would be
  shipping a derived structure table.

---

## 8. Optional Phase 6 — grading, if it is wanted

Only if a real grade label is obtained. The honest options:

| Route | Reality |
|---|---|
| BraTS 2019 HGG/LGG split (259/76) | Available, but **the label is not WHO grade**: BraTS "LGG" merges grades I–III and "HGG" is grade IV GBM only. A classifier trained on it predicts *GBM vs non-GBM*, and must be named that way. |
| MGMT methylation (BraTS 2021 Task 2) | Poor bet — the RSNA-MICCAI challenge topped out near 0.62 AUC and follow-up work argues it is not recoverable from routine MRI. |
| UPENN-GBM / UCSF-PDGM | Carry molecular labels; new dataset, new preprocessing, new training. |

Cost: a new dataset, a preprocessing run, and a training run. **Not** a
config change. Decide separately, after Phases 0–5.

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Mass effect invalidates the atlas near the lesion** | **High — scientific** | Unavoidable in principle: tumours displace tissue, and a healthy-brain atlas mislabels displaced tissue exactly where the lesion is. Mitigate by reporting involvement as *approximate*, quantifying displacement in Phase 3, and stating the limitation in the report artifact. Do not pretend precision we do not have. |
| Left/right swap in a report | High — clinical | Derive laterality from the affine and assert it; visual QC. Phase 0 check 3. |
| **Prior work already covers the pipeline** | **High — novelty** | VASARI-auto and BTReport exist (Finding F). Reposition the paper around what they do not do; see §10. Cite both prominently rather than being caught by a reviewer who knows them. |
| Knowledge base fabrication | High — credibility | **Mitigation replaced 2026-08-16.** External review is unavailable and, per Finding E, was moderate-agreement anyway. Now: only geometric or referential claims ship, deficit text is cut, every mapping entry stores the verbatim supporting sentence, and unmapped structures default to `unclassified`. |
| Deficit layer is unvalidatable | Resolved by removal | No clinical outcomes exist in our data (Finding C), so the layer does not ship. State in the paper's limitations that this was a deliberate omission and why. |
| Scope drift into clinical claims | High | §2 non-goals; disclaimer enforced by test. |
| Atlas parcellation is a single brain, twice removed | Medium — inherent | SRI24/TZO descends from AAL, parcellated on **one** subject (Colin27), then transferred into SRI24 indirectly via 300 archival 1.5T images by label fusion. So a structure name carries one person's macroanatomy → label fusion → population template → *then* this patient's displaced tissue. Report structure involvement as approximate and say this in the limitations. |
| Atlas parcellation too fine/coarse | Low | Merge to a coarser scheme; a ~128-region merge is precedented. |
| NITRC unavailable | Low | Mirror the checksum; fall back to another parcellation registered once, offline. |

---

## 10. How the paper changes

Old framing (dead): *"A dual-encoder fusion architecture with an
ambiguity-conditioned gate achieves competitive Dice with substantially better
calibration and boundary accuracy."*

Second framing (**also dead, as of Finding F**): *"An end-to-end pipeline
producing anatomically grounded, quantified reports from multi-modal brain
MRI."* VASARI-auto and BTReport already do this, the latter on BraTS
specifically. Claiming it would be caught immediately.

Third framing (the one to test): the pipeline is the **instrument**, not the
result. The question nobody in that prior work answers is *how much the
segmentation actually matters to the report* — and we are unusually well
equipped to ask it, because we have a matched baseline, a capacity control, a
paired-statistics harness, calibrated probabilities, and MC-dropout maps
already built and already honestly measured.

Concretely, three things are ours rather than theirs:

1. **Registration-free by construction.** VASARI-auto nonlinearly registers to
   MNI with enantiomorphic correction; BTReport normalises to MNI with
   SynthMorph. Both spend accuracy on a per-patient warp that is worst exactly
   where the lesion is. We spend none, because BraTS is already on the atlas's
   own grid (Finding A). That is a property to state and quantify, not just
   assert.
2. **Report-level error propagation.** Phase 5's experiment — how often does a
   segmentation error change the reported structure list, and which structures
   are most fragile — turns a Dice difference into a statement about the
   artifact a reader would actually act on. VASARI-auto touched the edge of this
   (κ 0.94 ± 0.10 between reports from manual vs. model masks) but did not
   relate it to segmentation quality or to per-structure fragility.
3. **Uncertainty attached to report fields.** Neither prior system propagates
   calibrated per-voxel uncertainty into the reported quantities. We already
   produce it, and the reliability work — a *negative* result, honestly measured
   against a matched baseline — is itself worth reporting rather than hiding.

Caveat that must be settled first: item 2's force depends on the capacity
control (§11 decision 2). If the wide U-Net matches `neurovision`, the framing
survives — the experiment is about segmentation quality in general, not about
our architecture — but the architectural contribution does not, and the paper
becomes a pipeline-and-analysis paper with a controlled negative result about
architecture. That is still publishable and still honest. It is just a smaller
claim, and it should be planned for rather than discovered.

What carries over unchanged: the segmentation result, the honest statistics
machinery, the calibration analysis (as a *negative* result, which is
publishable and useful), and the demo.

What must be stated as a limitation: no reliability advantage was found; the
functional layer is a literature lookup, not a validated predictor; single
seed; and the absolute Dice is below leaderboard state-of-the-art because both
arms were deliberately cut to a 64³/80-epoch budget.

---

## 11. Open decisions

1. **Cortical parcellation granularity** — full SRI24/TZO, or merged to ~128
   regions? Affects report readability more than correctness. Open.
2. **Does the capacity control change the framing?** If the wide U-Net matches
   `neurovision`, the accuracy result becomes "capacity, not architecture", and
   the paper's model contribution weakens further — the pipeline would then
   have to carry the entire contribution. Running now; decide after. See §10 for
   what survives either way.
3. **Is Phase 6 (grading) wanted at all**, given it needs a new dataset? Open.
4. ~~**Who reviews the knowledge base?**~~ **Resolved 2026-08-16: nobody, and
   the design no longer requires it.** No reviewer with the relevant background
   is available (Finding E). Rather than ship on model authority, the layer was
   restricted to claims that need no expertise to check — see §2's two
   categories and §5 Phase 2's tier table. The deficit layer is cut.
5. **Should the registration-free property be *measured* rather than asserted?**
   §10 item 1 claims an advantage over MNI-registered pipelines. Demonstrating
   it needs the MNI route implemented as a comparison arm — which is exactly the
   dependency Finding D declined. Options: assert it as a design property and
   cite the known failure mode of peri-lesional registration; or reopen the
   bridge decision solely to build a comparison arm. Decide before §10 item 1
   goes in the paper as a claim rather than as a description.
