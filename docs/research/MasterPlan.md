# MasterPlan — NeuroVision-X

**Written:** 2026-11-15 · **Status:** ACTIVE — the only live plan · **Horizon:** paper submission
2027-04-15, capstone June 2027

**Supersedes** `docs/archive/master_plan_m4_m5.md` (the Milestone 4–5 plan, kept unchanged as the
historical record). **Why it exists:** `docs/research_docs/project_review_2026-11-15.md` — read its
§A0 first. Authority on results stays with `docs/research_docs/experiments.md`; the gate on what may
be written stays with `docs/research_docs/claims_and_evidence.md`.

---

## B0. How to use this plan

- One live plan: this file. One board: §B5. Session logs go in `docs/research_docs/gpu_ledger.md`,
  never in the board.
- Read order for a cold session: `CLAUDE.md` → this file §B1–B3 → the board → the workstream you
  are about to touch → the `lessons.md` entry for that subsystem.
- A board row is done only when its "Done when" is true **and** its note, claim and board mark are
  committed together and pushed.

## B1. Where the project is going

**The question (one):**
> When a brain-tumour segmentation model is wrapped in a conformal miss-rate bound and a refusal
> gate, which kinds of deployment change break the wrapper — acquisition change, population change,
> or a change in what the label means — does a better model help, and what does a new site need to
> restore it?

**Thesis sentence (replaces the one in `CLAUDE.md`):**
> A conformal bound and a refusal gate around a tumour segmentation model hold where the data are
> exchangeable with the calibration set. Under acquisition and population change the bound degrades
> and a stronger model shrinks the damage; where the label definition changes it fails outright for
> every model and no threshold repairs it. The gate sees bad masks, not changed cohorts. We measure
> each failure, and what local data restores.

"In proportion to the shift" is retired until W3.2 measures it.

**Deliverables and dates.**

| ID | Deliverable | Date | Definition of done |
|---|---|---|---|
| D-A | Semester report + live demo | 2026-11-24 | Corrected report in the college template; cold-start demo rehearsed; tag `semester-2026` |
| D-B | Journal paper (MELBA; LaTeX) | submit 2027-04-15 | W5 exit checklist passed; arXiv preprint posted |
| D-C | Release | 2027-04-15 | Zenodo DOI for code + results + protocols; model card; weights decision recorded |
| D-D | Capstone | 2027-06 | Extension from §B8 delivered with report and demo |

## B2. What the paper will claim when this plan is finished

1. In distribution the bound holds for three models (matched U-Net, dual encoder, nnU-Net), under
   both the expectation (CRC) and high-probability (RCPS) guarantees, for WT, TC and ET, with the
   mask cost (precision, Dice, volume) printed beside it.
2. Under a controlled severity ladder (slice thickness, noise, bias field, contrast, motion) the
   realised risk rises with severity — a measured dose–response, per model.
3. On real adult cohorts from elsewhere (SSA plus a lockbox cohort evaluated once) the bound
   degrades modestly and a stronger model degrades less.
4. Where the label definition changes (PED tumour core) the bound fails for every model, at a floor
   no threshold removes; the four-label release shows which tissue carries the miss.
5. Local recalibration restores the bound with a stated number of labelled cases under each
   guarantee, with the mask cost; fine-tuning repairs the label case.
6. Case-level failure detection: learned QC, entropy, ensemble pairwise Dice and an input-statistics
   score compared by AUROC and risk–coverage, in and out of distribution; a gate with a stated
   risk target and what happens to that target under shift.
7. The front-end cost on original DICOM, with registration disagreement separated from
   segmentation loss.
8. Appendix: the architecture study (C1–C3, C29) as a worked example of a gain that did not
   survive a strong baseline.

Each sentence above becomes a row in the ledger only when its work item is done.

## B3. Binding rules

All existing hard constraints in `CLAUDE.md` stand unchanged (16 GB floor, no hardcoded paths,
`get_device()`, full resume, AMP default, 96³ default patch, no new dependency without asking, CPU
shape test per component; Opus writes no production `.py` — `py-implementer` from a spec, one module
per invocation, then `test-runner`, then `code-reviewer`; never the `code-critic` agent; CPU only on
the Mac, `sw_batch_size=1`, one heavy local job at a time; plain `pytest`; Kaggle T4 ≤ 28 GPU-h/week,
12 h sessions, `GIT_REF` pinned to a SHA; commit and push every change; stage explicit paths, never
`git add docs/`; do not delete `data/preprocessed/brats`). The ten traps in `lessons.md` apply.

New rules, each written against a loss recorded in Part A:

1. **Results are backed up the day they are produced** (W0.1 mechanism). A note may not cite a file
   that exists only under `outputs/`.
2. **A run is not finished until its note and claim are committed.** No new run is launched while a
   finished one is unread.
3. **Lockbox.** New external cohorts are downloaded, hashed and preprocessed, and nothing is run on
   them until their protocol is registered externally. One evaluation pass. No tuning afterwards.
4. **Analysis budget.** At most six new confirmatory families before the paper (W2.1, W2.3, W2.4,
   W3.1, W3.2, W3.4). Anything else is labelled exploratory before it is run.
5. **Feature freeze on the tool.** `app/` changes only to fix a defect a measurement exposed or to
   label something honestly. No new screens, viewers or panels.
6. **Power line.** Every protocol states the smallest effect it can resolve at its n and the
   plausible effect. If plausible < resolvable, the run does not launch (this would have stopped D0).
7. **Fit and evaluate never share cases.** Anything fitted on validation is cross-fitted or fitted
   on a named sub-split; the sub-split file is frozen in `configs/data/`.
8. **No architecture work.** No new model variant, seed or ablation of `neurovision`.
9. **Author explain-back gate.** A section enters the report or paper only after the author has
   explained it back in their own words (W8).
10. **Documents agree.** Any number or wording change lands in the ledger first, then everywhere it
    is quoted, in one commit (`grep` for the old string is part of the change).

## B4. Workstreams

Cost key: CPU-h are Mac wall-clock; GPU-h are Kaggle T4.

### W0 — Emergency and close-out (2026-11-15 → 11-19)

| ID | Task | Done when | Cost |
|---|---|---|---|
| W0.1 | **Back up results.** New tracked directory `results/` mirroring the small files under `outputs/` (per-case CSVs, `summary.csv`, `family.csv`, `*verdict*.json`, `fit.json`, `realised_risk.csv`, `thresholds.json`, `floors.csv`, risk `curves.npz` if < 5 MB each, error-budget CSVs, OOD scores). A script `scripts/sync_results.py` (py-implementer; size cap, allow-list of extensions, refuses volumes) + a `.gitignore` exception. Checkpoints (`neurovision`, seed 43, baseline ×2, content-only, heavy-aug, capacity r2, pooled, 4× D3 `last.pt`, nnU-Net `checkpoint_final.pth`) uploaded to one **private** Kaggle dataset; SHA-256 manifest committed to `docs/data_manifests/checkpoints.sha256` | `git ls-files results | wc -l` > 0; a fresh clone reproduces one table from `results/` alone; manifest committed | 3 CPU-h |
| W0.2a | **nnU-Net conformal → note 62, claim C30**, read strictly against `preregistration_conformal_nnunet.md` (write its Result section; nothing above it edited). Numbers already on disk: test holds; PED TC 0.555/0.624/0.630; SSA TC 0.087/0.182/0.195; PED WT 0.068/0.113/0.123 | Result section, note, C30, board, pushed | 2 h |
| W0.2b | **D2 evaluation.** `scripts/evaluate.py` on the 268 pooled-test cases with `outputs/neurovision_pooled/checkpoints/best.pt`, `save_logits=true`, one job; lesion-wise replay; `compare_family.py` on the registered family of 11; Result section of `preregistration_pooled.md`; note 63; claim C31 | Same closing set as above | ~6.5 CPU-h (overnight) |
| W0.3 | **Truth pass on wording.** C14 → "violated in 7/12 cells; largest where the label definition differs; no ordering by shift size is claimed"; `CLAUDE.md` thesis → §B1 sentence; `status_2026-09-27.md` superseded by a short `status_2026-11-15.md`; `related_work.md` §3 item 1 rewritten; C1 always quoted with the ET-present figure | `grep -rn "in proportion to the shift\|ordered by" docs CLAUDE.md` returns only archived files and quotations marked retired | 1 h |
| W0.4 | Plan hygiene: this file in, old plan archived, `CLAUDE.md` status block rewritten (short), Kaggle session table moved to `gpu_ledger.md`, `project_history.md` gains the Gate A, D2, nnU-Net-conformal and this-review lines | Old plan no longer referenced as live anywhere | 1 h |
| W0.5 | Record the outcome of the college GPU request (P2.6). If not filed or refused: say so; Kaggle is the only GPU and §B6 is the budget | One line in the board | author, 5 min |

### W1 — Semester report and demo, professional standard (11-16 → 11-24)

Source of truth: a Markdown master in `docs/research_docs/semester_report/report.md`; the Word/PDF
is generated from it once, in the college template. The existing `.docx` is the starting text.

| ID | Task | Done when |
|---|---|---|
| W1.1 | **Correction pass**: every row R1–R15 of Part A §4 fixed; W0.2 results added (§6.3 gains the nnU-Net conformal paragraph, §6.8 gains D2) | A table in `semester_report/corrections.md` lists each row with before/after |
| W1.2 | **Number audit**: every number in the report traced to a ledger row and a file in `results/`; a script prints mismatches (`scripts/audit_report_numbers.py`, py-implementer; reads a YAML of `{number, claim id, file, column}`) | Zero mismatches; YAML committed |
| W1.3 | **Front matter to template**: title page, certificate, declaration of originality, acknowledgement, **AI-assistance statement** (what was machine-drafted, what the author verified), abstract ≤ 300 words with one plain-language sentence, ToC, lists of figures/tables/abbreviations, page numbering (roman/arabic) | Matches the college template checklist item by item (author supplies the template) |
| W1.4 | **Figures**: one style (font, palette, 300 dpi, vector where possible) for all 15; every figure regenerated by `scripts/plot_semester_figures.py` from `results/`; captions state n, what bars/intervals are, and the source file; Fig. 15 replaced with deployed-model examples incl. one failure per cohort; add a one-page "what was claimed → what survived" figure | Each PNG has a `.txt` sidecar with its command and source |
| W1.5 | **Writing standard**: past tense for what was done; no claim stronger than its ledger row; each results subsection ends with one sentence saying what it means; limitations as numbered, specific statements; ethics and data-licence paragraph; references in one style with DOI/arXiv ID, each verified against its source | `writing-guidelines` skill pass + author read-through |
| W1.6 | **Author sections**: the author rewrites §1 Introduction and §4.8 Conformal risk control in their own words (explain-back gate) | Author's text in the master |
| W1.7 | **Model-submission documents** (Model 1/2/3 `.docx`): same number audit; Model 3 must state nnU-Net's 0.8879 beside 0.8709 | Three audited files |
| W1.8 | **Demo**: `docs/demo_runbook.md` cold-start rehearsal twice; fallback screen recording; PROCEED case labelled on screen "training-set patient — illustration only"; P0.7 eyeball done; E2E green the day before; a one-slide answer to "why did nnU-Net win?" and "what is actually new?" | Rehearsal log committed |
| W1.9 | Freeze: tag `semester-2026`; submitted PDF stored outside the repo by the author | Tag pushed |

### W2 — CPU science from saved logits (2026-11-25 → 2027-01-10)

All of W2 needs zero inference: `neurovision`, `baseline_unet3d`, seed-43, heavy-aug logits and
nnU-Net risk curves are on disk. Each item: external registration → run → note → claim.

| ID | Task | Method notes | Done when | Cost |
|---|---|---|---|---|
| W2.1 | **Conformal v2** | (a) ET included, loss 0 when the reference is empty; (b) grid extended above 0.95 so α = 0.20 binds; (c) RCPS with a betting (WSR) bound at δ = 0.1 beside CRC; (d) 1,000 random re-splits of the pooled val+test 376 cases to show the distribution of realised risk over calibration draws; (e) for every conformal mask: precision, Dice, volume ratio; (f) thresholds applied to the **post-processed** mask as deployed, and the difference reported; (g) all three models. Extends `uncertainty/conformal.py`, `scripts/conformal.py` (additive; existing columns must not move — additivity test) | Tables for 3 models × 3 regions × 3 α × {CRC, RCPS}; C13/C14/C15 rewritten from it | 2 modules, ~4 CPU-h |
| W2.2 | **Per-institution risk inside BraTS 2021** | Map case IDs to the FeTS 2022 institution partition (verify the ID crosswalk first; if not one-to-one, stop and record). Realised risk per institution on val+test with the pooled threshold. Descriptive, replicates Shahid on our models | A per-site table and a sentence for the limitations ("holds pooled, not per site: x of y sites exceed") | 1 module, 1 CPU-h |
| W2.3 | **Case-level failure-detection benchmark** | Signals: mean foreground entropy; SegQC; pairwise Dice across the three `neurovision` members (seed 42, seed 43, heavy-aug) and their ensemble entropy; OOD score. Targets: unusable (registered 0.7) and W2.5's volume endpoint. Metrics: AUROC and area under the risk–coverage curve, per cohort, paired bootstrap, one Holm family. SegQC cut-points refitted on its held-out 20% only | Table replacing paper Table XII; C18/C19/C21 rewritten | 1 module, ~3 CPU-h |
| W2.4 | **Risk-targeted gate** | Choose the REFUSE threshold on the best W2.3 signal to control P(unusable | accepted) ≤ β on a calibration sub-split (Learn-then-Test style, δ = 0.1); β ∈ {0.05, 0.10}; report coverage; then apply frozen to test, SSA, PED and (later) the lockbox. Expected and registered: holds in distribution, breaks under shift — that is the measurement. Compare with the deployed quantile gate. The deployed gate changes only if this one is no worse in distribution | Error-budget table v2 with both gates; C20–C22 rewritten | 1 module, ~2 CPU-h |
| W2.5 | **Clinically anchored endpoint** | Add "volume error ≤ 25% on WT and on ET" as a second usable definition, justified from response-assessment practice (cite; author verifies the source); report every pipeline table under both; inter-rater Dice from the BraTS literature quoted as the anchor for 0.7 | Second column in every usable table | small change, 1 CPU-h |
| W2.6 | **nnU-Net as a co-primary audited model** | From saved nnU-Net curves and per-case tables: local recalibration (`scripts/local_recalibration.py`, existing), W2.1, W2.3 (entropy + SegQC as transferred, stated), W2.4, error budget. Where nnU-Net voxel probabilities are needed and absent locally, one Kaggle predict session saves them at fp16 (≤ 3 GPU-h) | Every reliability table has a nnU-Net row or an explicit "not measured, because" | ~4 CPU-h (+ ≤ 3 GPU-h) |
| W2.7 | **Patient-level leakage check** | BraTS 2021 name-mapping file + image-hash near-duplicate scan across train/val/test; report; if any patient spans splits, sensitivity analysis excluding them | Note with the count (zero or not) | 1 module, 1 CPU-h |

### W3 — New evidence (2026-12 → 2027-02)

| ID | Task | Method notes | Done when | Cost |
|---|---|---|---|---|
| W3.1 | **Lockbox adult-glioma cohort(s)** | Candidates, in order; each needs licence, overlap and label-protocol checks **before download is counted**: (1) additional BraTS-Africa cases beyond the 60 used (later challenge releases report 95–115+); (2) UCSF-PDGM cases **not** in BraTS 2021 (≥ 114 by the 387/501 figure; needs the ID crosswalk); (3) Erasmus Glioma Database (374 manual segmentations; check access and whether sub-regions are labelled — if WT only, use for WT); (4) a non-glioma disease cohort (BraTS-MEN or BraTS-MET) for WT only, as a second concept-shift point. Choose ≥ 1 from (1)–(3). SHA manifest → research-path preprocessing → external registration of the complete analysis (W2.1, W2.3, W2.4 applied frozen, all three models) → one pass. Dependency ask if a downloader is needed | Result section written from a single run; no re-runs except a documented crash | ~1.4 min/case CPU for `neurovision`; nnU-Net via Kaggle ≈ 2–3 GPU-h |
| W3.2 | **Severity ladder (controlled shift)** | On the 189 test cases: through-plane downsampling to 3/5/7 mm then resampling back (simulates the thick-slice studies the front end refuses), Rician noise, bias field, gamma/contrast, k-space motion — MONAI transforms only, 4 severities each, seeds fixed, corrupted inputs not stored (regenerated deterministically). Inference for 3 models on Kaggle. Outcomes per severity: Dice, realised risk at frozen τ, gate refusal and silent-failure rates, OOD flag rate. Registered prediction: monotone rise in realised risk; a stronger model has a flatter curve | Dose–response figure and table; the "graded" claim is either earned or dropped for good | 1 transform module + 1 driver; ≈ 15–18 GPU-h |
| W3.3 | **PED label decomposition** | Obtain the BraTS-PEDs release with four sub-region labels (check availability and terms); per sub-region miss rate at τ and at τ_min for `neurovision`, nnU-Net, and the fine-tuned model | A table attributing the 0.356 floor to tissue classes; C14 addendum becomes a result | ~2 CPU-h + download |
| W3.4 | **Real-DICOM v2** | (a) Reference carried into our space: rigid-register the BraTS-preprocessed T1CE to our pipeline's T1CE (same image, so near-exact), apply to the label, score there — separates segmentation loss from registration disagreement; (b) the 16 thick-slice refusals re-run with that refusal overridden and scored, to test whether refusing was right; (c) extend from 40 to all available test-split patients with RSNA DICOM (~93); (d) if (a)–(b) show the thick-slice rule refuses usable studies, change the rule to CAUTION and report it as a labelled post-hoc variant. Protocol amendment registered before (c) | Table XV v2 with three columns: atlas-space, carried-reference, overridden refusals; C25 rewritten | ~6 min/study → ~9 CPU-h in batches, one job at a time |
| W3.5 | **Recalibration cost side** | For each restored cell: labelled cases needed under CRC and RCPS, the mask cost (from W2.1e), and an annotation-time estimate with a cited source | One table; the "recipe" paragraph gains a cost sentence | 1 CPU-h |

### W4 — GPU (Kaggle T4, ≤ 28 GPU-h/week)

Order and budget. Nothing else gets GPU time.

| Order | Run | GPU-h | Gate |
|---|---|---|---|
| 1 | W2.6 nnU-Net probability export (only if needed) | ≤ 3 | — |
| 2 | W3.2 severity ladder, 3 models | 15–18 | probe on 5 cases × all corruptions must reach the slowest path first |
| 3 | W3.1 nnU-Net inference on the lockbox | 2–3 | protocol registered |
| 4 | *Stretch:* nnU-Net 5-fold at `nnUNetTrainer_250epochs` on the 1,062 non-test cases, for out-of-fold probabilities → a 1,062-case clean calibration set and real out-of-fold failures to train the QC model on | ~75 | only if 1–3 are closed by 2027-01-31 and the paper draft is on schedule |

Core total ≈ 20–24 GPU-h (one quota week). No `neurovision` training of any kind (rule 8).
Every session: `docs/gpu_session_checklist.md`, pinned SHA, watchdog, probe that reaches the failure
condition (trap 9), results pulled and synced to `results/` the same day.

### W5 — The paper (2026-12-01 → 2027-04-15)

| ID | Task | Done when |
|---|---|---|
| W5.1 | **Read the neighbours in full** (Shahid; Adhikary et al.; Liu et al.; Zenk et al.; CRC; RCPS; nnU-Net Revisited; BraTS-Africa; BraTS-PEDs both versions). One page of notes each in `related_work.md`. **Citation audit**: every reference opened at its source; title, authors, venue, year, and the specific number cited confirmed; anything unverifiable removed | Audit table committed; zero unverified references |
| W5.2 | **New outline** (≤ 20 pages + appendix): 1 Introduction (author) · 2 Related work · 3 Setting: models, cohorts, kinds of change · 4 Methods: CRC/RCPS, recalibration, failure detectors, gate, pipeline · 5 Results in the order of §B2 items 1–7 · 6 Discussion: site-level recipe with costs · 7 Limitations · Appendix A protocols and timeline · B architecture study · C engineering traps. Hypotheses cut from nine to four | Outline approved by the author; old sections mapped to new or to the appendix |
| W5.3 | **Convert to LaTeX** (`melba.sty` from the journal's template repository; BibTeX). Tables generated from `results/` by script, never typed | `paper/` builds to PDF from a clean clone |
| W5.4 | **Draft** in the order: methods → results (as each W2/W3 item closes) → limitations → discussion → related work → introduction (author) → abstract last | Full draft 2027-02-15 |
| W5.5 | **Figures**: failure-case montage per cohort; dose–response; realised-vs-nominal for three models; four-outcome bars for two gates; recalibration curves with cost; DICOM three-column comparison | Each regenerated by one command |
| W5.6 | **Internal review rounds** ×2 with the `paper-review` skill and the number-audit script (W1.2 pattern) | All major items closed; audit zero mismatches |
| W5.7 | **Outside reader**: the faculty guide, plus one person with clinical imaging background if the author can find one (this also answers the single-author / no-clinician weakness) | Comments addressed or answered in writing |
| W5.8 | **Submission pack**: cover letter (journal template), data/code availability with DOI, AI-assistance statement, arXiv preprint | Submitted 2027-04-15 |
| W5.9 | Venue check on 2026-12-01: confirm MELBA's current author instructions; note MIDL 2027 and the 2027 UNSURE workshop dates as fallbacks | Dates written in the board |

### W6 — Tool: freeze and three fixes

| ID | Task | Done when |
|---|---|---|
| W6.1 | Freeze declared in `CLAUDE.md` (rule 5) | Line present |
| W6.2 | Honest labels: demo PROCEED case marked as a training-set patient; guarantee copy re-checked against the W0.3 wording | E2E assertion on the label |
| W6.3 | Front-end policy change **only if** W3.4 shows it (thick-slice REFUSE → CAUTION), as a registered variant | Error budget re-run with and without |
| W6.4 | Disk hygiene: remove the 60+ `outputs/ambiguity_*` directories and other caches listed in `reproducibility.md` §11 as regenerable, **after W0.1** and after listing them to the author; `brag-output-*` out of the repo root | ≥ 100 GiB free; list recorded in `reproducibility.md` |

### W7 — Reproducibility and release

| ID | Task | Done when |
|---|---|---|
| W7.1 | `scripts/reproduce.sh paper` regenerates every table and figure of the report and the paper from `results/` on a fresh clone | Run on a clean clone; diff of tables is empty |
| W7.2 | Environment lock: exact versions for the three requirement files recorded (`pip freeze` snapshots in `docs/research_docs/environments/`); Python 3.11 stated | Fresh-clone rehearsal (repeat of P2.5b) green |
| W7.3 | **External registration** from now on: each new protocol deposited where the timestamp is not the author's (OSF registration, or a Zenodo snapshot of the protocol file) before its run; the ID goes in the protocol header. Existing 14 protocols: one Zenodo snapshot of the repository at the `semester-2026` tag, and Table A-I stays as the honest record of ordering | IDs in headers |
| W7.4 | Release: `LICENSE`, `CITATION.cff`, model card updated (nnU-Net result, label caveat, intended use), weights decision checked against the BraTS data terms and recorded, Zenodo DOI for code + `results/` + protocols | DOI in the paper |
| W7.5 | Confirm the Kaggle datasets holding BraTS-derived data are private (data terms forbid redistribution) | Checked and recorded |
| W7.6 | Raw BraTS 2021 has no SHA manifest: on the next download, write one | Manifest committed |

### W8 — Author ownership (runs alongside everything)

| ID | Task | Done when |
|---|---|---|
| W8.1 | Six study units, one per week, each ending in a half-page the author writes without help: (1) Dice, lesion-wise Dice, empty-region convention; (2) paired bootstrap, Wilcoxon, Holm, why "inconclusive" is not "equal", TOST; (3) conformal risk control from the formula, by hand on ten numbers; (4) exchangeability and the three kinds of change; (5) what nnU-Net does that the project's recipe did not; (6) selective prediction, risk–coverage | Six half-pages in `docs/research_docs/author_notes/` |
| W8.2 | Viva drill before 11-24: the twelve hardest questions (why nnU-Net won; what is new; what "guarantee" means for one patient; why PED is not "shift"; what the gate's 2% means; why 40% of real scans were refused; what was AI-written) answered aloud | Done twice |
| W8.3 | Explain-back gate (rule 9) applied to every paper section | Recorded per section in the board |

## B5. The board

`[ ]` open · `[~]` in progress · `[x]` done. Update on every commit that closes an item.

| Week of | Items |
|---|---|
| 2026-11-15 | W0.1 `[x]` (2026-11-23: `results/` tracked, `eae12ac`; checkpoint manifest committed; 15 checkpoints in private Kaggle dataset `neurovision-checkpoints`; fresh clone reproduced `nnunet_conformal_shift_dice/family.csv` byte-identically from `results/` alone) · W0.2a `[x]` (note 62, C30: REPLICATES) · W0.2b `[x]` (2026-11-24: note 63, C31 — PED TC RECOVERS, SSA TC no measurable recovery, no in-distribution cost) · W0.3 `[x]` (remaining grep hits: archived files, the project reviews, protocol text above a `## Result` line that may not be edited, and passages marked "retired reading") · W0.4 `[x]` (`dbbe651`; `gpu_ledger.md` created) · W0.5 `[ ]` · W1.1 `[~]` (R1–R11, R14 fixed `ac9e82e`; D2, R13, R15 and the C1 CI fixed 2026-11-24; open: R12 only — needs the college template and the author's details) · W1.2 `[~]` (script `47ca799`; registry 116 entries, 0 mismatches, negative control flags a wrong value; ~30 numbers untraceable — `number_registry_untraceable.md`; C1 CI now tracked as `results/compare_family/c1_headline/` and corrected to [0.0143, 0.0432], note 64; 118 entries, 0 mismatches; open: PED R(τ_min) curves into `results/`) |
| 2026-11-20 | W1.3–W1.9 `[ ]` · W8.2 `[ ]` · **D-A due 11-24** |
| 2026-11-25 | W2.7 `[~]` (script `c266da5`, not run) · W2.1 `[~]` (module `739fbe0` + driver `3a6c9ac`; run blocked on external registration) · W2.5 `[x]` measured 2026-11-24 (note 65, C32: volume-usable 0.80 / 0.53 / 0.41 test/SSA/PED vs Dice-usable 0.94 / 0.80 / 0.27; mostly an ET endpoint; open: author verifies the 25% source and the 0.7 inter-rater anchor; second column into the report's usable tables) · W3.2 `[~]` (protocol + corruption module `c1db922`; driver/probe open; GPU blocked on external registration) · W5.9 `[~]` (checked 2026-11-23: MELBA takes submissions via melba.scholasticahq.com/for-authors, cover letter + LaTeX template from github.com/melba-journal — re-read the full author page on 12-01; MIDL 2027 dates not announced (MIDL 2026 full papers were due 2025-12-05, so 2027's may be ~Dec 2026); UNSURE 2027 not announced (past deadlines late June)) · W7.3 set up `[ ]` · W8.1 unit 1 |
| 2026-12-02 | W2.3 `[ ]` · W2.5 `[ ]` · W5.1 `[ ]` · W3.1 cohort checks and downloads `[ ]` |
| 2026-12-09 | W2.4 `[ ]` · W2.2 `[ ]` (2026-11-24 check: FeTS 2022 `partitioning_1.csv` (23 institutions, `Partition_ID`/`Subject_ID`, ids `FeTS2022_###`) is released only through a registered Synapse data request; no public FeTS→BraTS2021 id crosswalk found, so one-to-one cannot be verified from public sources. **Author action:** file the FeTS 2022 data request, or drop W2.2 per §B9 and cite Shahid (arXiv 2606.20115, which uses FeTS ids directly)) · W3.2 transform module + probe `[ ]` · W5.2 `[ ]` |
| 2026-12-16 | W2.6 `[ ]` · W3.2 GPU `[ ]` · W5.3 `[ ]` |
| 2027-01-06 | W3.3 `[ ]` · W3.4 `[ ]` · W3.1 registration `[ ]` |
| 2027-01-13 | W3.1 single pass `[ ]` · W3.5 `[ ]` · W5.4 methods + results |
| 2027-02-01 | W4 stretch decision · W5.4 · W5.5 · W6.3 if triggered · W6.4 |
| 2027-02-15 | **Full draft** · W5.6 round 1 · W7.1–W7.2 |
| 2027-03 | W5.6 round 2 · W5.7 · W7.4–W7.6 |
| 2027-04-15 | **W5.8 submit; D-B, D-C** |
| 2027-04 → 06 | Capstone (§B8) |

Dependencies that bind: W0.1 before everything · W0.2 before W1.1 · W2.1 before W2.4, W2.6, W3.1,
W3.5 · W2.3 before W2.4 · W2.5 before W2.3's second target · W3.1 registration after W2.1/W2.3/W2.4
exist (they are what is frozen) · W3.2 probe before W3.2 GPU · W5.1 before W5.2 · W7.3 before any
new confirmatory run.

## B6. Budget

| Resource | Need | Have |
|---|---|---|
| GPU | 20–24 h core; +75 h stretch | 28 h/week Kaggle |
| Mac CPU | ~45 h of batch jobs, serial | — |
| Disk | lockbox cohorts ≈ 5–15 GB preprocessed | 63 GiB free → ≥ 100 after W6.4 |
| New dependencies | none planned; a dataset downloader or an OSF client would need the author's yes | — |

## B7. Cut list (decided; reopening needs new evidence)

Any further `neurovision` training, seeds or ablations · the remaining 3×3 seed grid · B3 as a
separate phase (folded into W2.3 from existing logits) · 96³ retrain · SwinUNETR baseline · IDH /
Phase F (recommendation: cut permanently — UCSF-PDGM overlaps BraTS 2021, it is a second project,
and nothing in §B2 needs it; the 2027-01-15 go/no-go becomes a no unless the author overrules) ·
LLM report writing · any new viewer, panel or page · MGMT, survival, grade, midline shift, eloquence
verdict (unchanged).

## B8. Capstone direction (decide 2027-02-01, after W2–W3)

Recommended: **"site onboarding kit"** — the W2/W3 machinery packaged as one command a new site
runs on k labelled local cases: OOD flag rate, CRC and RCPS thresholds with their feasibility
floors, mask cost, the risk-targeted gate, and a one-page site report; validated on the lockbox
cohort as a simulated site. It reuses everything, adds no model, and is the constructive end of the
paper. Alternative if the stretch run happens: an out-of-fold-trained QC model. Not recommended:
IDH, new architectures.

## B9. Risk register

| Risk | Likelihood | Response |
|---|---|---|
| Laptop loss before W0.1 | low, fatal | W0.1 is first, today |
| No lockbox cohort passes the licence/overlap/label checks | moderate | W3.2 carries the covariate-shift claim alone; the paper says so; disease-shift cohort (WT only) as the fallback |
| FeTS ID crosswalk not one-to-one | moderate | W2.2 dropped; limitation sentence cites Shahid instead |
| Severity ladder shows no monotone rise | moderate | Report it; "graded" stays retired; that is still a result |
| Risk-targeted gate no better than the quantile gate | moderate–high | Report both; deployed gate unchanged |
| Semester report corrections overrun 11-24 | moderate | Order inside W1: R1–R9 and W0.2 first, front matter second, figures third |
| Scope creep returns | high | Rules 4, 5, 8; the cut list |
| Author cannot defend a section | moderate | W8; a section that fails the gate is cut or simplified, never kept |
| 2026 preprints do not say what is cited | unknown | W5.1 audit; unverifiable references removed |
| Kaggle billing of held kernels | seen twice | Watchdog; judge by the quota counter (lessons.md) |

## B10. Verification — how to know the plan was carried out

- **W0.1:** fresh clone → `results/` present → `scripts/reproduce.sh` regenerates the Gate A family
  table byte-identically; checkpoint hashes match the manifest.
- **W0.2:** two new Result sections, notes 62–63, C30–C31; `pytest` (plain) green; the D2 per-case
  table has 268 rows; nnU-Net bridge deltas match `bridge.json`.
- **W0.3 / rule 10:** the `grep` in W0.3 is clean.
- **W1:** number-audit script prints zero mismatches; the PDF matches the template checklist;
  cold-start demo rehearsal log exists; tag `semester-2026` on the remote.
- **Every W2/W3 analysis:** (a) additivity test — existing columns unchanged; (b) the real analysis
  re-run and the output moved as predicted (not only unit tests); (c) one hand check of a number per
  table; (d) external registration ID predates the output file's mtime.
- **W3.1:** exactly one results directory per cohort, created after the registration date.
- **W5:** `paper/` builds from a clean clone; audit script zero mismatches; citation audit complete.
- **Suite:** plain `pytest` on CPU in about a minute, `python scripts/smoke_test.py` exit 0, before
  every GPU session and every tag.

---

## B11. Files this plan touches most

`docs/research/MasterPlan.md` (new) · `docs/research_docs/project_review_2026-11-15.md`
(new) · `docs/research_docs/claims_and_evidence.md` · `docs/research_docs/experiments.md` ·
`CLAUDE.md` · `docs/research_docs/semester_report/` · `src/neurovision/uncertainty/conformal.py`,
`scripts/conformal.py`, `scripts/local_recalibration.py`, `scripts/error_budget.py`,
`scripts/compare_family.py`, `scripts/evaluate.py`, `scripts/replay_logits.py`,
`src/neurovision/inference/gatekeeper.py`, `scripts/validate_qc.py`,
`scripts/validate_real_dicom.py` (all reused, extended additively) · `.gitignore` (the `results/`
exception).

Sources checked for this plan: [MELBA for authors](https://melba.scholasticahq.com/for-authors) ·
[MELBA template repository](https://github.com/melba-journal) ·
[UCSF-PDGM dataset card](https://atlas.rsna.org/cards/33e4aa54-79dc-4d16-81fe-fd1afa023e5d) ·
[Erasmus Glioma Database](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8203723/) ·
[FeTS challenge data](https://fets-ai.github.io/Challenge/data) ·
[BraTS-Africa](https://arxiv.org/pdf/2305.19369v1.pdf)
