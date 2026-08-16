# NeuroVision-X — the interpretable pipeline direction

Planning document for the project's second contribution attempt: an end-to-end,
interpretable pipeline from scan to structured anatomical report.

Written 2026-08-16, after the original reliability claim was measured and did
not hold. Read `docs/experiments.md` notes 11–17 before this file: the numbers
that forced the pivot are there, not here.

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
| Anything **diagnostic** | Framed throughout as a research and educational decision-support demo. |

**The honesty constraint that shapes the whole design:** we can *validate* the
anatomy layer (atlas alignment is checkable), but we **cannot validate the
deficit layer**, because BraTS ships no clinical outcomes. The functional layer
is therefore a **cited literature lookup**, not a model output, and must be
presented as such everywhere it appears.

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
                        │  NEW: knowledge layer (Phase 2)        │
                        │  structure → function → deficit, cited │
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
4. **Population-level plausibility.** Compute the tumour-location distribution
   over all 1,251 cases. Literature says gliomas concentrate in frontal and
   temporal white matter and are comparatively rare in brainstem and
   cerebellum. If our distribution reproduces that, the atlas is very unlikely
   to be scrambled — and if it does not, something is wrong. This is a cheap,
   quantitative, independent check and it is the strongest one available.

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

### Phase 2 — Functional knowledge base (4–6 days, mostly not coding)

`data/knowledge/structure_function.yaml` (versioned, in-repo, reviewed)

Schema per structure:

```yaml
- structure_id: 57
  name: Precentral gyrus (left)
  system: motor
  functions: [voluntary motor control of the right side of the body]
  associated_deficits:
    - description: contralateral (right-sided) weakness or paralysis
      confidence: well-established
  pathways: [corticospinal tract]
  sources: [<citation>]
```

Hard rules:
- **Every entry carries a citation.** No entry is generated without a source.
  An uncited neuroanatomical claim in a medical-adjacent tool is not
  acceptable, and this is the layer where fabrication would be easiest and
  least visible.
- `confidence` is an explicit field, because plasticity and individual
  variation mean many associations are probabilistic, not deterministic.
- Deficit text is written in hedged language ("associated with", "may")
  because it is a literature association, not a prediction for this patient.

White-matter pathways come from the SRI24 DTI channels or a tract atlas; tract
involvement is what connects a cortical site to a distant body function (e.g.
corticospinal tract → contralateral limb motor control). This is the honest
version of "what other organs are connected" — the brain relates to the body
through **pathways and functions**, not through organ-to-organ links.

Effort here is dominated by sourcing and review, not code. This layer should be
reviewed by someone with neuroanatomy background before it is shown to anyone.

### Phase 3 — Tumour characterisation profile (3–4 days)

`src/neurovision/anatomy/burden.py` — the honest replacement for "stage".

- Volumes of ET / TC / WT in mm³ (1 mm isotropic, so voxel count *is* volume)
- Necrotic-core fraction, enhancing fraction, oedema-to-core ratio
- Multifocality: connected-component count and volume distribution
- Shape: sphericity, surface-area-to-volume
- Mass effect: midline shift and ventricular compression, exploiting that
  every case is in a common template where the midline is known a priori
- Anatomical involvement summary from Phase 1

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
| 2 — Knowledge base | 4–6 | parallel with 1 |
| 3 — Burden profile | 3–4 | independent, can start now |
| 4 — Report + demo | 4–5 | needs 1, 2, 3 |
| 5 — Validation + paper | 5–7 | needs 4 |

**≈ 21–29 working days**, zero GPU hours. Phase 3 depends on nothing new and
could start immediately if a quick win is wanted.

---

## 7. Data, licensing, reproducibility

- SRI24 is **CC-BY-SA**. It is **downloaded by `scripts/fetch_atlas.py`, never
  committed**, for the same reason BraTS is not committed: licence hygiene and
  repo size. Attribution goes in the README and in every generated report.
- The knowledge base **is** committed — it is our own compilation — with each
  entry's citation.
- Atlas version is recorded in every report, so a report can be traced to the
  parcellation that produced it.
- No new Python dependency is expected. Nibabel handles NIfTI, scipy handles
  connected components and distance transforms, both already pinned. If a
  registration tool turns out to be needed after all (it should not be, per
  Finding A), that is a **new dependency and requires asking first**.

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
| Knowledge base fabrication | High — credibility | Citation required per entry; external review before display. |
| Scope drift into clinical claims | High | §2 non-goals; disclaimer enforced by test. |
| Deficit layer is unvalidatable | Medium — inherent | No clinical outcomes exist in our data (Finding C). Present as cited literature association, never as prediction. Say so in the paper's limitations. |
| Atlas parcellation too fine/coarse | Low | Merge to a coarser scheme; a ~128-region merge is precedented. |
| NITRC unavailable | Low | Mirror the checksum; fall back to another parcellation registered once, offline. |

---

## 10. How the paper changes

Old framing (dead): *"A dual-encoder fusion architecture with an
ambiguity-conditioned gate achieves competitive Dice with substantially better
calibration and boundary accuracy."*

New framing (supported): *"An end-to-end pipeline producing anatomically
grounded, quantified, uncertainty-aware reports from multi-modal brain MRI,
built on a segmentation model that reduces enhancing-tumour error by 17%
relative to a matched baseline."*

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
   regions? Affects report readability more than correctness.
2. **Does the capacity control change the framing?** If the wide U-Net matches
   `neurovision`, the accuracy result becomes "capacity, not architecture", and
   the paper's model contribution weakens further — the pipeline would then
   have to carry the entire contribution. Running now; decide after.
3. **Is Phase 6 (grading) wanted at all**, given it needs a new dataset?
4. **Who reviews the knowledge base?** It should not ship on my authority or
   the author's alone.
