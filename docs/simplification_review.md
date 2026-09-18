# Simplification review — 2026-09-18

A whole-repo pass looking for dead code, duplicated logic, overengineering, and places where a
simpler route existed. **Read-only: nothing was changed.** Baseline before any change: `pytest`
2234 passed, 35 skipped (panoptica / nnunetv2 not in this venv), 57.7 s; `ruff` 1 error (an E501
in a test).

Every finding below was verified by reading the code or by grep, not guessed. Where a claim is
"probably", it says so.

---

## 1. The numbers

| Tree | Files | Lines | Of which code | Docstrings | Comments |
|---|---|---|---|---|---|
| `src/neurovision` | 92 | 36,515 | **14,122 (39%)** | 14,759 (40%) | 2,230 (6%) |
| `scripts/` | 35 | 19,575 | 8,377 (43%) | 6,812 (35%) | 1,327 (7%) |
| `app/backend` | 8 | 5,341 | 1,901 (36%) | 2,152 (40%) | 482 (9%) |
| `tests/` | 100 | 50,360 | — | — | — |
| `app/frontend/src` | 78 | 14,822 | — | — | — |

The library is 36k lines but only 14k of it is code. **87 docstrings are longer than 40 lines and
together are 5,325 lines.** 208 lines of code docstrings reference plan documents, milestone
numbers, task ids (`T4.3`, `Phase E5`) or GPU-hour losses — history that already lives in
`docs/lessons.md` and `docs/experiments.md`.

Tests are 1.4× the size of the library. That ratio is not wrong by itself, but §4 shows where the
test bulk is boilerplate rather than assertions.

---

## 2. Dead code (verified)

### 2.1 The NIfTI-upload job path — `app/backend/jobs.py` and its routes. ~1,120 lines.

`jobs.py` (539 lines) is the 2026-08-19 demo path: four NIfTI files in, segmentation out, polled
as a job. It is served by seven routes in `api.py` (`POST /upload`, `GET/DELETE /jobs/*`,
`/jobs/{id}/volume`, `/jobs/{id}/mask`) and tested by `test_app_jobs.py` (300) and
`test_app_job_routes.py` (282).

**No frontend code calls any of it.** `grep` for `/upload` or `/jobs/` in `app/frontend/src`
returns only the clinical variants. The DICOM clinical path (`clinical_jobs.py`) superseded it
one week later and copied its skeleton rather than reusing it (see §3.9). The only thing anything
else imports from `jobs.py` is `job_root(settings)`, a five-line path helper.

Removing it: delete `jobs.py`, the seven routes, the two test files, move `job_root` into
`config.py`. Zero effect on the clinical pipeline.

### 2.2 Methods only their own test calls

| Symbol | Where | Evidence |
|---|---|---|
| `NeuroVisionX.forward_multitask` | `models/neurovision.py:374` | its own docstring: *"Nothing consumes this yet"*; grep confirms only `tests/test_neurovision.py` |
| `Atlas.tissue_mask` | `anatomy/atlas.py:514` | only `tests/test_atlas.py` |
| `GateDecision.cautions` | `inference/gatekeeper.py:232` | no caller anywhere |

### 2.3 Constants and fields never read (vulture, 60% confidence — each needs a one-line check)

`SIGNAL_NAMES` (gatekeeper), `CHECK_IDS` (input_qc), `AMBIGUITY_CHANNEL_GROUPS` (adaptive_fusion),
`REGION_COLORS` (figures). Dataclass fields `vasari_status` / `vasari_claim` (involvement),
`sequence_variant` / `scan_options` (dicom_ingest), `atlas_source` / `atlas_licence` (report),
`dice_vs_prediction` / `dice_vs_ground_truth` / `auc_vs_ground_truth` (faithfulness),
`mean_difference` (equivalence). Some of these are written into JSON via `asdict()` and so are
"used" by serialisation — those stay. The four module constants are the likely real dead ones.

### 2.4 43 stale `# noqa` comments

`ruff --select RUF100` finds 43 `noqa` directives that suppress nothing. Auto-fixable.

### 2.5 Closed-experiment scripts — NOT dead, but frozen. ~8,500 lines.

`ambiguity_intervention.py`, `gate_failure_detection.py`, `gate_boundary_profile.py`,
`detection_stats.py`, `gate2_localisation.py`, `mc_comparison.py`, `extract_ambiguity.py`,
`extract_ambiguity_serial.py`, `run_ablation_grid.py`, `export_nnunet_dataset.py`.

Each produced a number recorded in `docs/experiments.md` (notes 20–35 region) for an experiment
that has resolved — most of them null. They are reproducibility artifacts, so **do not delete
them**. But they will never be run again unless a reviewer asks, and they are 44% of `scripts/`
by line count. Option: `scripts/archive/` with a README naming the note each one backs. Author's
call; cosmetic.

---

## 3. Duplicated logic (verified, with line references)

### 3.1 Bernoulli entropy from logits — five copies, plus two more

| File | Name | Units |
|---|---|---|
| `analysis/detection.py:66` | `_entropy_from_logits` | numpy, /ln2 |
| `analysis/qc_inference.py:80` | `entropy_from_logits` | torch, nats |
| `scripts/train_qc.py:375` | `entropy_from_logits` | torch, nats — **byte-identical docstring to the previous one** |
| `app/backend/volumes.py:215` | `entropy_from_logits` | numpy, /ln2, summed over channels |
| `models/fusion/adaptive_fusion.py:539` | inner `_entropy_from_logits` | torch, /ln2 |
| `inference/mc_dropout.py:183` | `_bernoulli_entropy` | torch |
| `scripts/score_confidence.py:336` | `bernoulli_entropy_nats` | torch |

Same formula `p·softplus(−z) + (1−p)·softplus(z)`. Each copy carries the same ~15-line docstring
re-telling the fp16 `eps` lesson. One `neurovision/uncertainty/entropy.py` with
`bernoulli_entropy(logits, *, normalise: bool)` accepting torch or numpy replaces all seven;
the lesson gets told once.

### 3.2 `_to_tuple` — four identical copies

`models/neurovision.py:52`, `models/baseline.py:36`, `models/encoders/swin.py:88`,
`models/encoders/cnn.py:59`. One line in `utils/`.

### 3.3 `_zero_small_et` — two copies

`inference/postprocess.py:377` (batched) and `analysis/replay.py:214`. The replay copy's docstring
says it exists because the original is private. Export the original; delete the copy. Same story
for `_binarize_regions` / `_apply_postprocess_steps` in `replay.py`, which re-sequence
`postprocess.py`'s steps by hand.

### 3.4 `load_curves_npz` — two copies

`uncertainty/conformal.py:357` (public) and `scripts/conformal.py:478` (private). The public one's
docstring explains it was added *because* the script one could not be imported. The script should
now call the public one.

### 3.5 Class-map ↔ region-channel conversion — two pairs

`metrics/segmentation.py:51 classes_to_regions` (torch) vs
`app/backend/clinical_jobs.py:1042 _class_map_to_regions` (numpy).
`inference/postprocess.py:253 regions_to_classes` (torch) vs
`reporting/dicom_seg.py:235 classes_from_regions` (numpy). Keep one of each, accepting both types.

### 3.6 `gatekeeper.py` and `input_qc.py` are the same shape twice

Both define: a `StrEnum` severity (`Decision` / `Severity`), a `_worst_*` reducer, a `_jsonable`
converter, a per-signal record (`SignalVerdict` / `Finding`), and a report dataclass with
`refusals()` / `warnings()`-or-`cautions()` / `to_dict()`. ~250 lines of scaffolding that could be
one `Verdict`/`Report` base used by both.

### 3.7 Three registries — 261 lines for one 25-line idea

`models/registry.py`, `models/fusion/registry.py`, `losses/registry.py`: `_REGISTRY` dict,
`register_x(name)` decorator, `build_x(cfg)`, `available_x()`. One generic `Registry[T]` class
instantiated three times.

### 3.8 `api.py` — route families copied

`/cases/*`, `/jobs/*`, `/clinical/jobs/*` each have `volume/{modality}`, `mask/prediction`,
`uncertainty`, `atlas` routes whose bodies differ only in which `Settings` resolver they call
(`get_settings()`, `_job_settings`, `_clinical_job_settings`). After §2.1 it is two families; the
remaining pair can share one helper per route.

### 3.9 `jobs.py` and `clinical_jobs.py` — same skeleton

In-process `dict` store behind a lock, `job_root`, `_update_job(**fields)`, `start_job` on a
daemon thread, `run_job` with the outer `try/except Exception → status="failed"`. `clinical_jobs.py`'s
module docstring says explicitly it copies `jobs.py`'s shape. Removing `jobs.py` (§2.1) resolves
this without a merge.

### 3.10 Upload validation — twice

`jobs._load_and_validate_nifti` / `_validate_consistent_geometry` and
`inference/input_qc.load_volume_infos` / `check_geometry_consistency`. `input_qc.py:44` cites
`jobs.py` as prior art. Resolved by §2.1.

### 3.11 Paired bootstrap — three

`analysis/statistics.py:382 paired_bootstrap_ci` (the library one),
`scripts/gate2_localisation.py:246 _paired_bootstrap`, `scripts/detection_stats.py:332
_bootstrap_two_sided_p`. Frozen scripts (§2.5) — low priority, but they should have called the
library.

### 3.12 Dice — three

`metrics/segmentation.py:146` (torch), `anatomy/alignment.py:156` (numpy), `data/qc_pairs.py:440`
(numpy tuple). One numpy-or-torch function.

### 3.13 Hydra compose — five near-identical sites

`smoke_test.py:206`, `run_ablation_grid.py:206`, `app/backend/inference.py:270`,
`clinical_jobs.py:634`, `clinical_jobs.py:669`. One `compose_config(overrides)` in
`utils/config.py`.

### 3.14 `_atomic_torch_save` / `_atomic_np_save`

Same temp-file-then-`os.replace` body; one `atomic_write(path, writer)`.

### 3.15 29 test files carry the same importlib boilerplate

Because `scripts/` is not a package, every `test_*_script.py` has ~8 lines of
`spec_from_file_location` / `module_from_spec` / `exec_module`, plus a paragraph explaining it.
One `load_script(name)` helper in `tests/conftest.py` removes ~300 lines and 29 paragraphs.

---

## 4. Overengineering and harder routes

Ordered by how much the simpler alternative would have saved.

### 4.1 `NeuroVisionX.forward` returns three different types depending on `self.training`

`models/neurovision.py:298`. Eval → `Tensor`; train with deep supervision → `list[Tensor]`; train
with an auxiliary head → `MultiTaskOutput`. The 60-line docstring calls this "DELIBERATE AND
LOAD-BEARING" and then spends a paragraph on the MC-dropout hazard it creates — which is trap #5
in CLAUDE.md, and it happened.

Simpler: `forward` always returns `MultiTaskOutput` (which already nests the `seg` list); a
five-line `predict(x) -> Tensor` returns `out.seg[0]` for sliding-window and MC-dropout. Trainer
and losses already consume `MultiTaskOutput`. This is the highest-value design fix and the only
one in this document that touches the trained model's contract — so it must be verified by
`scripts/replay_logits.py` reproducing ET 0.870859 on the test split, not by unit tests.

### 4.2 Four forward variants, each with its own encode loop

`forward`, `forward_with_gates`, `forward_with_ambiguity`, `forward_with_auxiliary`.
`_encode_decode`'s docstring admits `forward_with_gates` "has its own near-identical body instead
of reusing this one". One loop with a `collect: set[str]` argument (`{"gates", "branch_logits"}`)
replaces four. Same verification as 4.1.

### 4.3 Library code living in `scripts/` and being copied back into `src/`

`train_qc.py` (1,162), `calibrate.py` (1,283), `extract_ambiguity.py` (1,162), `conformal.py`
(950), `detection_stats.py` (1,117) are libraries with a `main()` at the bottom. Because
`scripts/` cannot be imported by `src/`, §3.1, §3.4 and `analysis/qc_inference.py` (whose module
docstring describes exactly this problem) exist as copies. The simpler route was always: logic in
`src/neurovision/…`, script = 40-line Hydra driver. Doing it now for the two scripts still live
(`train_qc.py`, `conformal.py`) removes the copies; the frozen ones (§2.5) can stay as they are.

### 4.4 Docstrings that narrate project history

Examples: `clinical_jobs._generate_report` (101-line docstring on a private function, citing
T4.3, `scripts/report.py`'s CSV round-trip, and `_export_dicom_seg`'s "philosophy");
`scripts/train_qc.py` module docstring (135 lines); `inference/tta.py::tta_predict` (108 lines).
CLAUDE.md asks that non-obvious *choices* be explained. It does not ask that `docs/lessons.md` be
pasted into every function that touched the lesson. Rule that would halve the prose without losing
a fact: a docstring states **what** and the one-sentence **why**; history, task ids and GPU-hour
costs get a single `See docs/lessons.md §N` pointer. Estimated saving 8–10k lines across
`src/` + `scripts/` + `app/backend`. This is prose, not risk — but it is the single biggest
contributor to "the repository is large".

### 4.5 `clinical_jobs.py` — 2,044 lines, three functions over 300 lines

`run_clinical_job` (≈320 lines) runs ten stages inline: ingest → QC → preprocess → QC → segment →
QC model → conformal → gate → Grad-CAM → DICOM-SEG → report, each wrapped in its own
`try/except` + `_update_clinical_job` + log. A `STAGES: list[Stage]` table (name, callable,
`supplementary: bool`) and a ten-line loop would make the pipeline readable in one screen and
make "a supplementary stage never fails the job" a property of the loop rather than a comment
repeated four times. `_generate_report` (344 lines) and `_export_dicom_seg` (≈190) would move to
`neurovision/reporting/` where their inputs already live.

### 4.6 Nine table formatters in `visualization/tables.py`

`build_results_table` / `format_results_markdown` / `format_results_latex`, then the same trio for
`comparison` and `boundary`. `pandas.DataFrame.to_latex` and `to_markdown` (the latter needs
`tabulate`, which is already a pandas optional in most installs — check before relying on it)
cover the formatting; only the three `build_*` functions carry project logic. ~700 lines → ~250.

### 4.7 Everything reloaded on every clinical job

Atlas, knowledge YAMLs, QC checkpoint, Hydra config — reloaded per request "so a corrected
`knowledge/eloquence_map.yaml` takes effect on the next job". Stated design choice, documented in
five places. For a single-user demo it costs seconds per job and is fine; flagged only because the
justification is repeated more than the code.

### 4.8 Tests

`test_app_clinical_jobs.py` is 2,197 lines. A large share is fixture construction for synthetic
DICOM studies and per-stage monkeypatching, repeated across tests rather than shared. Same
docstring-narration pattern as §4.4 (test docstrings quote the plan). Not wrong; just heavy.
Worth a pass only after the backend simplifications land, since those tests will change anyway.

---

## 5. Bugs

**None confirmed in this pass.** What was checked: the full suite (green), `ruff` on the project
rules (one E501 in a test), and a scratch run of `ruff --select B,SIM,C4,PIE,RET,PERF,PLW,PLE`
(126 hits: 43 stale `noqa`, 18 `zip()` without `strict=` — the library ones at
`losses/segmentation.py:181`, `data/preprocessing.py:303,506`, `reporting/dicom_seg.py:208` were
read and their lengths are guaranteed by surrounding code — 10 loop-variable rebinds, 16 manual
list comprehensions; nothing that is a wrong result). The one runtime warning in the suite
(`lr_scheduler.step()` before `optimizer.step()`) comes from `tests/test_trainer.py:194` calling
the scheduler directly, not from `Trainer`.

A structural pass does not find logic bugs in 36k lines. Bug-finding happens per module while
simplifying it, with the module's own test file open — that is how the circular-mask and fp16
bugs were eventually found, and the review of each simplified module should state what it looked
for.

---

## 6. Recommended order

Safest and largest first. Each step: one module, `py-implementer`, suite green, then the
next. Steps 1–4 are mechanical and reversible; 6–7 change contracts and need the replay check.

| # | Step | Removes | Risk |
|---|---|---|---|
| 1 | Delete the NIfTI `/jobs` path (§2.1) + dead methods (§2.2) + stale `noqa` (§2.4) | ~1,200 lines | none — nothing calls it |
| 2 | Shared helpers: entropy, `_to_tuple`, registry, `_jsonable`/enums, curves loader, class↔region, atomic write, dice (§3.1–3.7, 3.12–3.14) | ~600 lines, 7 copies of one lesson | low — pure refactor, existing tests cover each |
| 3 | `tests/conftest.py::load_script` (§3.15) | ~300 lines | none |
| 4 | `api.py`: share route bodies between `/cases` and `/clinical` (§3.8) | ~200 lines | low — E2E section 12 covers the clinical routes |
| 5 | Docstring diet (§4.4): what + one-line why + pointer | 8–10k lines | none to behaviour; **author must agree**, since CLAUDE.md asked for explanation |
| 6 | Move `train_qc` / `conformal` logic into `src/` (§4.3) | the copies in §3.1/3.4 | low–medium |
| 7 | `clinical_jobs.run_clinical_job` as a stage table (§4.5) | ~400 lines | medium — verified by `scripts/run_clinical_study.py` on UPENN-GBM-00002 reproducing `done`/PROCEED |
| 8 | `forward()` single return type + one encode loop (§4.1, 4.2) | ~300 lines and trap #5 | medium — verified by `replay_logits.py` reproducing ET 0.870859 |
| 9 | `tables.py` via pandas (§4.6) | ~450 lines | low; notebook 09 is the only consumer |
| 10 | `scripts/archive/` for closed experiments (§2.5) | nothing; moves 8.5k lines out of the way | cosmetic — author's call |

Not recommended: touching `adaptive_fusion.py`'s `BranchAmbiguity` / `GateGenerator` beyond §4.2.
The ambiguity conditioning is a measured null, but the trained `neurovision` checkpoint has those
parameters and every saved logit came through them. Removing the mechanism means a new model, not
a simplification.

---

## 7. Outcome — what was actually done (2026-09-18, same day)

Author's instruction: simplify only where there is no chance of a regression; complex-but-working
beats simple-but-risky. Applied in three commits, each verified by the full suite and then by a
real run, not by unit tests alone.

| Commit | Change | Lines | Verification beyond `pytest` |
|---|---|---|---|
| `b73470e` | §2.1 — NIfTI `/upload` + `/jobs/*` path removed; `jobs.py` keeps only `job_root` | −674 | `create_app()` imports; live `uvicorn` in `.venv-clinical` with the documented env: every `/clinical/jobs/{id}/*` route (volume, mask, geometry, atlas, uncertainty, conformal-band, gradcam, report, markdown, pathology, dicom-seg) returns 200 on the real UPENN-GBM `done` job; `/cases/*` 200 on a BraTS case; `/api/jobs` now 404; zero server errors |
| `01594f2` | §3.3, §3.4, part of §3.1 — three private copies replaced by imports (`load_curves_npz`, `zero_small_et`, `entropy_from_logits`) | −74 | `scripts/replay_logits.py` on five real test cases reproduces `outputs/replay/eval_test/per_case_default.csv` with max abs diff **0.0**, matches `evaluate.py` per-case metrics to 1e-17; an `et_min_volume` variant ran through the shared `zero_small_et` on real volumes |
| `ab60429` | §3.15 — `tests/script_loader.py::load_script` replaces 28 boilerplate copies | −186 | test-only; suite count unchanged |

Suite after all three: **2205 passed, 35 skipped** (2234 before, minus the 29 tests of the deleted
path). `scripts/smoke_test.py` passes. Frontend untouched.

### Deliberately NOT done, and why

- **§4.1 / §4.2 `forward()` return-type unification and the four forward variants.** Touches the
  trained model's contract, trainer, losses, sliding-window and MC-dropout. Highest value, highest
  risk. The author chose working-complex over risky-simple. Leave until there is a reason to retrain.
- **§2.2 `forward_multitask`, `Atlas.tissue_mask`, `GateDecision.cautions`.** Dead or test-only,
  but each is ~15 lines inside a critical file (`neurovision.py`, `atlas.py`, `gatekeeper.py`).
  Not worth opening those files for.
- **§2.4 "43 stale `noqa`".** Retracted — the count came from running `ruff --select RUF100`
  alone, which disables the project's own rules. Under the project config the flagged directives
  are `BLE001` / `S310` markers that carry the reason for each broad `except`. They are
  documentation, not lint debt.
- **§3.2 `_to_tuple` ×4, §3.7 registries, §3.6 gatekeeper/input_qc scaffolding, §3.13 Hydra
  compose sites.** All inside model-building or clinical-decision modules; each saves <100 lines.
- **§3.1 remaining entropy copies** (`detection.py`, `volumes.py`, `adaptive_fusion.py`,
  `mc_dropout.py`, `score_confidence.py`). Different units or shapes per copy; the model-internal
  one is part of the trained forward pass. Consolidating means re-deriving five conventions.
- **§4.4 docstring diet.** CLAUDE.md asks for explanation; the author did not opt in. Prose, not
  risk — can be done any time.
- **§4.5 `run_clinical_job` stage table, §4.6 `tables.py` via pandas, §2.5 archiving closed
  scripts.** Rewrites of working code for readability only.

### Bugs found while doing this

None. Every diff was subtractive or an import swap; no behaviour changed, and the real-data
checks above agree with the published numbers to the last digit.
