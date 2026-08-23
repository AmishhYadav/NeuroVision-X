# Experiment Log

Every training run goes in this file. One row per run, appended when the run
**finishes** (or is abandoned), never before. If a run is not in this table it
did not happen and its numbers may not be used in the paper.

A "run" is one `experiment_name` + one seed. A run split across several Kaggle
sessions by resume is still ONE row — sum the GPU hours.

---

## How to fill a row

| Column | Where it comes from |
|---|---|
| **Run** | `experiment_name` from the config, plus `-s<seed>` if not 42 (e.g. `baseline_unet3d-s1`) |
| **Config** | The `+experiment=` file, plus any CLI override that changed the run. `-` if none |
| **Seed** | `cfg.seed` |
| **Model** | `cfg.model.name` and parameter count printed at startup |
| **Epochs** | `epochs_done / epochs_planned`. Not equal means the run was cut short — say why in Notes |
| **GPU h** | Sum of Kaggle session wall-clock across every resume. Kaggle shows this per session |
| **Dice ET/TC/WT** | `summary.csv` from `scripts/evaluate.py`, `mean` row. Test split |
| **HD95 ET/TC/WT** | Same file. **Millimetres** — `evaluate.py` passes spacing; the trainer's val HD95 is in voxels and does NOT go in this table |
| **ECE** | Calibration, once implemented. `-` until then |
| **W&B** | Run id / short URL |
| **Notes** | Anything that would change how the number is read: OOM, restarts, a config edit mid-run, a suspicious loss curve |

**Rules**

- Numbers come from `scripts/evaluate.py` on the **test** split, at
  `inference.sliding_window.overlap: 0.5`. The `val/dice_mean` in W&B is a
  monitoring signal at overlap 0.25 and is not comparable — do not paste it here.
- `ignore_empty=False` (BraTS convention) throughout. On BraTS 2021, 2.6% of
  cases have no enhancing tumor, so this moves ET Dice by well under a point,
  but it must still be stated in the paper.
- `et_min_volume` stays 0. If a row was ever produced with it on, say so in
  Notes — otherwise the number is not comparable to the others.
- If you change a config between two runs you intend to compare, that is a new
  experiment file, not an edit to an existing one.

---

## Runs

| Run | Config | Seed | Model | Epochs | GPU h | Dice ET | Dice TC | Dice WT | HD95 ET | HD95 TC | HD95 WT | ECE | W&B | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_unet3d` (80ep/64³) | `+experiment=baseline_unet3d data.num_workers=2`, `GIT_REF=6ee28a7` | 42 | `unet3d`, 12.87M | 80 / 80 | ~3.2 | 0.8442 | 0.9058 | 0.9276 | 4.91 | 5.24 | 6.72 | — | `pzu8y5fo` (offline) | 6–10 below |
| `neurovision` | `+experiment=neurovision data.num_workers=2`, `GIT_REF=7caacfa` | 42 | `neurovision`, 34.91M | 80 / 80 | 23.1 | 0.8709 | 0.9161 | 0.9321 | 4.20 | 4.98 | 7.09 | 0.0446 → **0.0135** | `cc2l5j1c` (offline) | 11–16 below |
| `ablation_content_only_gate` | `+experiment=ablation_content_only_gate data.num_workers=2`, `GIT_REF=7caacfa` | 42 | `neurovision` with `use_ambiguity: false`, 34.90M | 80 / 80 | **24.16** (10.20 + 10.41 + 3.55 over 3 sessions) | 0.8686 | 0.9158 | 0.9333 | 4.53 | 5.45 | 6.12 | — | `ddkbitjp` (offline) | **P2 rung 2 — THE NULL. Inconclusive vs `neurovision` on every metric.** 38 below |
| `capacity_control_unet3d` | `+experiment=capacity_control_unet3d`, `GIT_REF=f363067` | 42 | `unet3d` widened, 34.83M | 80 / 80 | ~8 (see note 19) | 0.8497 | 0.9092 | 0.9295 | 4.84 | 5.67 | 7.09 | — | `k9j5uba2` (offline) | 19–21 below |
| `baseline_unet3d` | root config, **not** `+experiment=` — see note 1. Overrides: `model=unet3d training.epochs=200 training.val_interval=2 data.dataset_type=dataset data.num_workers=2` | 42 | `unet3d`, 12.87M | 200 / 200 | 16.5 | 0.8587 | 0.9157 | 0.9354 | 4.02 | 5.21 | 5.36 | — | `nz5y7li7` | 1–5 below |

**Notes for `baseline_unet3d` / `nz5y7li7`**

1. **The schedule is not the one `configs/experiment/baseline_unet3d.yaml`
   describes, so this row is not directly comparable to any future run made
   from that file.** `_baseline_common.yaml` was written 2026-08-02, the day
   *after* this run started. Measured differences: `epochs` 200 vs 100,
   `val_interval` 2 vs 5, `scheduler.warmup_epochs` 10 vs 5, and validation
   `sliding_window.overlap` 0.5 vs 0.25 (monitoring only — the reported
   numbers here come from `scripts/evaluate.py` at overlap 0.5 either way).
   The recommendation in `docs/reproducibility.md` §8 is to re-run at 100
   epochs under the experiment file and retire this row, so that the baseline
   table is internally consistent with `baseline_swinunetr` and `neurovision`,
   which will both inherit `_baseline_common`. Until then: a 200-epoch
   baseline against a 100-epoch proposed model favours the baseline, so the
   direction of the bias is at least not self-serving — but it is not a
   controlled comparison and must not be presented as one.
2. **Two Kaggle sessions, one run**, resumed from `last.pt`: 10.93 h
   (2026-08-01, epochs 1–131) + 5.54 h (2026-08-02, epochs 132–200) = 16.47 h,
   rounded to 16.5 above. Tesla T4. Peak VRAM **1.17 GB** — the U-Net leaves
   almost the whole 16 GB card unused.
3. **HD95 ET is over n = 183, not 189.** Six cases are NaN because exactly one
   side of the ET region was empty; `hd95()` deliberately does not collapse
   that into a number. Dice/IoU are over all 189. `gt_empty_ET` on the test
   split is 0.0265 (5 of 189), so those five score ET Dice as empty-vs-empty
   under `ignore_empty=False` rather than by real overlap.
4. **No commit SHA was recorded.** `notebooks/kaggle_train.ipynb` cloned
   `GIT_REF = "main"` and W&B captured no git metadata, so the exact source
   revision is not recoverable — only that it was `main` between 2026-08-01
   16:30 and 2026-08-02 12:16 UTC (commits `3144175`..`9665c70`). Every future
   run pins `GIT_REF` to a SHA.
5. W&B ran in `offline` mode and was synced afterwards; the run id is the same
   across both sessions because it lives in the checkpoint. Full environment,
   seeds and runtimes: `docs/reproducibility.md`.


**Notes for `baseline_unet3d` (80ep/64³) / `pzu8y5fo`**

6. **This is the row the fusion runs are compared against**, not the 200-epoch
   one above it. Same `_baseline_common.yaml` as `neurovision` and
   `ablation_content_only_gate`: 64³ patches, 80 epochs, `val_interval` 10,
   `grad_clip_norm` 5.0, seed 42. One Kaggle T4 session, no resume.
7. **Cost of the re-planned schedule, against the 200-epoch/96³ row.** Dice ET
   0.8587 → 0.8442 (−0.0145), TC 0.9157 → 0.9058 (−0.0099), WT 0.9354 →
   0.9276 (−0.0078). HD95 ET 4.02 → 4.91 mm, TC 5.21 → 5.24 mm, WT 5.36 →
   6.72 mm. So the 64³/80-epoch cut costs roughly **1 Dice point and
   0.9–1.4 mm of HD95**. That was the price of affording the P2 ablation, and
   it applies identically to every arm, so relative comparisons are unaffected
   — but the absolute numbers are not state-of-the-art and the paper must not
   present them as such. WT boundary suffers most, which is expected: whole
   tumour is the largest structure and so loses the most from a smaller
   context window.
8. **Peak VRAM 0.50 GiB of 14.56.** The U-Net leaves 96% of the card unused;
   it is `neurovision` that is memory-bound, not the baseline.
9. **`grad_clip_norm` 5.0 is doing its intended job**, confirmed on real data:
   median gradient norm 0.720, p90 2.373, **max 15.086**, and only **1.6% of
   875 steps clipped**. At the previous value of 1.0 most steps would have
   been rescaled, which is what probe v5 caught before any run started.
10. **HD95 ET is over n = 184**: 5 of 189 cases are NaN because exactly one
    side of the ET region was empty. `gt_empty_ET` on the test split is
    **0.0265** (5 of 189), matching the 2.6% measured over the full BraTS 2021
    training set — an independent check that split and label handling are
    intact. Evaluation ran on the **Mac CPU** in ~35 min for both splits
    (6.49 s/case), consuming zero GPU hours.

**Notes for `neurovision` / `cc2l5j1c`**

11. **Scored from `best.pt` at epoch 69, not `last.pt` at epoch 79.** The run
    peaked at epoch 69 (`val/dice_mean` 0.8938) and the two later validation
    points did not improve on it. `best.pt` therefore lives in **session 2's**
    Kaggle output, not the final session's — see the session-3 row below for
    why that is not where you would first look. Trained over three chained
    sessions, 10.34 + 10.5 + 2.3 = 23.1 GPU-h.
12. **Against `baseline_unet3d` (80ep/64³), paired over the same 189 test
    cases, bootstrap CI + Wilcoxon, Holm-corrected across all six metrics:**
    Dice ET **+0.0267** CI [+0.0143, +0.0432] p_holm 1.4e-21 → **better**;
    Dice TC **+0.0103** CI [+0.0033, +0.0180] p_holm 1.2e-10 → **better**;
    Dice WT +0.0045 CI [−0.0007, +0.0098] → **inconclusive**; HD95 ET −0.67 mm,
    TC −0.27 mm, WT **+0.37 mm (worse)**, all inconclusive. So the Dice half of
    the claim holds on ET and TC, and **no HD95 difference is claimable in any
    region.** `dice_WT` is the case where the CI and the p-value disagree
    (p_holm 5.6e-09 with a CI spanning 0): the sign is consistent across most
    cases while a few large regressions pull the mean CI across zero, and
    `compare_models` resolves that conservatively. 93 of 184 HD95 ET pairs are
    exact ties, so that test runs on a minority of cases.
13. **Boundary-stratified error does NOT show a boundary-specific improvement,
    and this is the note to read before writing the boundary section.** Total
    error voxels (per-band rate × band size, summed over cases — never the mean
    of the rates, see the correction recorded in `CLAUDE.md`) come to
    neurovision/baseline of **0.870 (ET), 0.908 (TC), 0.952 (WT)** — uniformly
    fewer errors. But the *share* falling in the 0–2 mm band is essentially
    unchanged (ET 77.6% vs 78.6%, TC 62.3% vs 63.1%, WT 72.3% vs 73.7%). The
    model is better roughly proportionally at every distance, so on this
    evidence the honest claim is "fewer errors overall", **not** "better
    specifically at the margin". The 10 mm+ band remains almost entirely false
    positives in both models (WT: 166,692 of 171,202 error voxels for
    neurovision), i.e. the confident-prediction-far-from-any-lesion failure
    mode is reduced but not solved.
14. **HD95 ET is over n = 186, not 189** (3 cases NaN from a one-sided-empty ET
    region), against the baseline's n = 184. `gt_empty_ET` is **0.0265** (5 of
    189), identical to the baseline — an independent check that both rows were
    scored on the same split with the same convention.
15. **Evaluation ran on a Kaggle T4 at 11.9 s/case (35:43 for 189 cases), not
    on the Mac.** Measured first on the Mac at **136 s/case**, i.e. ~21× the
    6.49 s/case that `baseline_unet3d` achieves there. The note elsewhere that
    deterministic evaluation is free on CPU was measured on the U-Net and
    **does not transfer to this architecture**; both splits locally would be
    ~14 h. Cost on GPU was ~0.6 h. `save_logits: true`, so temperature scaling
    does not require re-running the split.
16. **CALIBRATION: the headline claim does NOT hold. Mean ECE is within noise
    against the baseline, calibrated or not.** T fit on val, reported on test,
    `predicted` (label-free) mask, `T = [2.05, 1.99, 1.63]` against the
    baseline's `[1.92, 2.02, 1.93]` — i.e. **this architecture is not
    intrinsically less overconfident**; it needs about the same correction.
    Split-level ECE mean: neurovision **0.0446 uncalibrated / 0.0135 scaled**
    vs baseline **0.0395 / 0.0175**. So uncalibrated it is nominally WORSE and
    after scaling nominally better — but per-case paired statistics (bootstrap
    CI + Wilcoxon, Holm over 5 metrics) return **`ece_mean` INCONCLUSIVE in
    both variants** (uncalibrated CI [−0.0006, +0.0105]; scaled CI [+0.0005,
    +0.0089], p_holm 0.38). What IS claimable, in both variants: **`ece_TC`**
    (uncalibrated +0.0120 CI [+0.0047, +0.0199] p_holm 3.6e-10; scaled +0.0085
    CI [+0.0023, +0.0153] p_holm 7.1e-05) and **`brier_mean`** (+0.0062 and
    +0.0065, both p_holm < 1e-5).
    **Consequence for the paper, stated before anyone writes the abstract: of
    the three parts of the stated claim — competitive Dice, better calibration,
    better boundary accuracy — only the Dice part is supported, and it is
    supported more strongly than "competitive" (see note 12). Calibration and
    boundary accuracy are both unsupported (notes 12, 13). The result is a
    more ACCURATE model, not a more RELIABLE one, and the contribution has to
    be rewritten around that or the reliability claim has to be earned by
    something not yet measured (MC-dropout risk-coverage is the remaining
    candidate and has not been run).**
    Note also the earlier figure of **0.0158** recorded in `CLAUDE.md` as "the
    bar" is from the **96³ / epoch-130** baseline, not the 64³ / 80-epoch row
    this run is comparable to. The correct bar is **0.0175**. Comparing against
    the wrong baseline's calibration would have flattered this run.
17. **MC-DROPOUT RISK-COVERAGE: no advantage either. This closes the last route
    to a reliability claim.** N=10, `seed=42`, `predictions_from:
    deterministic` — verified additive: per-case Dice/HD95 are **bit-identical**
    (max diff 0.00000000) to the deterministic run for BOTH models, so the
    published rows are untouched. Cost: `neurovision` 4:12:07 on a T4
    (83 s/case); `baseline_unet3d` 2.5 h on the Mac's CPU (48 s/case), which is
    why the cheap arm belongs on CPU and the expensive one does not.
    Raw AURC is **not comparable across models** — a better model has a lower
    oracle and random curve too — so the comparison is the normalised fraction
    of the achievable gain over random. **neurovision 37.6% vs baseline 40.6%
    (diff −0.029, 95% CI [−0.176, +0.143]); Spearman(uncertainty, Dice)
    −0.392 vs −0.463 (diff +0.071, CI [−0.032, +0.173]). Both WITHIN NOISE**
    under a paired bootstrap resampling case indices. So `neurovision`'s
    epistemic uncertainty ranks its own failures neither better nor worse than
    a plain U-Net's.
    Note the baseline's uncertainty is genuinely informative on its own
    (Spearman −0.463, p 1.9e-11, capturing ~40% of the oracle gain), which is
    exactly why "our model can flag its own failures" is not a claim: MC-dropout
    on a U-Net already does it about as well. **All three reliability routes —
    calibration (note 16), boundary accuracy (notes 12–13), and risk-coverage —
    are now measured and none supports an advantage. The accuracy result
    (note 12) is the finding.**

18. **BURDEN PROFILE, first real run — and the first measurement of how much
    the segmentation actually changes the REPORT, which is the pivoted paper's
    claim.** Local, CPU, zero GPU hours. `scripts/burden.py` over the 189-case
    test split, four times: ground-truth labels and three models. 189/189
    succeeded every time.

    `neurovision` had saved **logits but no `predictions/`**, so they were
    reconstructed from the logits through the project's own `postprocess_logits`
    with that run's own recorded config (identical to the baseline's:
    `threshold 0.5`, `min_component_size 50`, `connectivity 1`,
    `enforce_nesting`). Verified the only way worth trusting — recomputing test
    Dice from the reconstruction and matching the already-published
    `summary.csv`: **ET 0.870859 vs 0.870859, delta 0.00e+00** (TC and WT match
    to 1e-16).

    The three models, and note two of them are U-Nets:

    | Run | test Dice ET | patch | epochs |
    |---|---|---|---|
    | `neurovision` | 0.8709 | 64³ | 80 |
    | `baseline_unet3d` (matched) | 0.8442 | 64³ | 80 |
    | `baseline_unet3d` (superseded) | 0.8587 | 96³ | 200 |

    Agreement of each model's profile with the ground-truth-derived profile,
    per case, over the same 189:

    | Quantity | `neurovision` | U-Net 64³/80ep | U-Net 96³/200ep |
    |---|---|---|---|
    | median rel. err., WT volume | 0.0394 | **0.0386** | 0.0443 |
    | median rel. err., ET volume | 0.0533 | 0.0615 | **0.0500** |
    | median rel. err., sphericity WT | **0.0504** | 0.0633 | 0.0610 |
    | `dominant_side_WT` agreement | **100.0%** | 99.5% | **100.0%** |
    | multifocality agreement | 73.0% | 70.4% | **78.3%** |
    | multifocal rate (GT = 22.8%) | 33.9% | 40.7% | **30.7%** |
    | empty-ET cases found (GT = 5) | **2** | 0 | 1 |

    **Against its matched baseline, `neurovision` wins where it should.** ET
    volume error 0.0533 vs 0.0615, sphericity error 0.0504 vs 0.0633,
    multifocality agreement 73.0% vs 70.4%, and it correctly reports no
    enhancing tumour in 2 of the 5 cases that have none where the baseline
    reports ET in all 189. The ET accuracy result of note 12 does propagate to
    the report. WT volume error is a tie (0.0394 vs 0.0386).

    **But report agreement is NOT monotonic in Dice, and that is the finding
    worth carrying.** The superseded 96³/200-epoch U-Net has *lower* ET Dice
    than `neurovision` (0.8587 vs 0.8709) and yet produces the better report on
    ET volume agreement (0.0500) and on multifocality agreement (78.3%). A
    plausible mechanism is patch size: 96³ gives more global context per
    forward pass, and the failure being measured is *fragmentation of a single
    lesion into several*, which is a context failure rather than a boundary
    one. That run differs in both patch size and epochs, so this is a flag, not
    a conclusion — but it is a flag that lands directly on the Phase 5
    experiment, which must therefore **control patch size** rather than
    treating Dice as the single explanatory variable. Discovering this after
    designing that experiment would have been expensive.

    **All three models over-report multifocality** — 30.7% to 40.7% against a
    true 22.8%. That is a systematic, model-independent property of thresholded
    segmentation output at this component floor, and it is the single least
    reliable field in the whole burden profile. It should be reported with that
    caveat rather than as a finding about any one model.

    **The 100% `dominant_side_WT` agreement is the end-to-end proof of the
    crop-geometry handling**, not a boring row. Ground truth is read cropped to
    the nonzero bbox; predictions are read in original 240×240×155 geometry; the
    midline index therefore differs between them by the crop offset. Mishandle
    it on either side and laterality disagrees on roughly half the cases. It
    disagrees on none. That is the "recompute a known quantity end to end"
    check the plan demanded in place of a unit test.

    Identities verified on all 189 real cases in every run: `vol_TC == vol_NCR +
    vol_ET`, `vol_WT ==` the three-class sum, `vol_right + vol_left == vol_WT`,
    ET ⊆ TC ⊆ WT, and sphericity within [0.232, 0.874] — never above the ~0.92
    ceiling of the marching-cubes estimator.

    **A trap that cost a wrong committed claim:** the first version of this note
    labelled `outputs/eval_test` as `neurovision`. It is the superseded 96³
    U-Net. Three eval directories differ only by suffix and two of them hold a
    U-Net. Read the resolved checkpoint path in `eval_config.yaml` before
    labelling a column; the directory name is not evidence.

    Caveat: this is a *descriptive* profile — no grade, no stage, no prognosis.
    Ventricular/deep-white-matter overlap, the tissue fractions and epicentre
    naming were Phase 3b and are absent from THIS note's tables; they landed
    2026-08-19 and their numbers are in note 25. Midline shift is declined
    outright, not pending; see the plan's Phase 3 note.


19. **CAPACITY CONTROL: the architectural claim SURVIVES. Capacity explains
    about a fifth of the ET gain; architecture explains the rest.** Completed
    2026-08-17, 80/80 epochs (`global_step` 70000 = 80 x 875, the arithmetic
    proof every epoch ran), `best.pt` at epoch 79, `val/dice_mean` 0.8916.
    Params **34,829,696** against `neurovision`'s 34,911,341 — matched to
    0.23%. Every controlled variable verified identical to `baseline_unet3d`
    from the checkpoint's own stored config: seed 42, 64³ patches, 4
    samples/volume, batch 1, 80 epochs, `val_interval` 10, AdamW 1e-4 / wd
    1e-5, warmup 5, `grad_clip_norm` 5.0, `dice_ce`, deep supervision off.
    Only `model.channels` differs.

    Evaluated on the **test** split on the Mac CPU, 189/189 cases, ~24 min at
    7.5 s/case, same postprocessing as every other arm (`threshold` 0.5,
    `min_component_size` 50, `connectivity` 1, `enforce_nesting`), so the row
    is directly comparable.

    Test Dice: **ET 0.8497, TC 0.9092, WT 0.9295.** Against the pre-registered
    outcomes written into `configs/experiment/capacity_control_unet3d.yaml`
    *before* the run — "near 0.844 = architecture", "~0.855 = decomposes",
    "near 0.871 = capacity" — this lands just above the first branch.

    Paired comparisons over the same 189 cases, bootstrap CI + Wilcoxon +
    Holm across the table:

    | Comparison | ET | TC | WT |
    |---|---|---|---|
    | `neurovision` vs `capacity_control` | **+0.0211**, p_holm 7.3e-19, better | **+0.0068**, p_holm 5.1e-07, better | +0.0026, inconclusive |
    | `capacity_control` vs `baseline_unet3d` | **+0.0055**, p_holm 1.6e-09, better | +0.0034, inconclusive | +0.0019, inconclusive |
    | `neurovision` vs `baseline_unet3d` | **+0.0267**, p_holm 7.2e-22, better | **+0.0103**, p_holm 4.6e-11, better | +0.0045, inconclusive |

    **Decomposition of the +0.0267 ET gain: capacity +0.0055 (20.6%),
    architecture +0.0211 (79.4%).** `neurovision` beats a
    parameter-matched U-Net by +0.0211 ET at p_holm 7.3e-19. This is the
    outcome that keeps the architectural contribution alive, and it is the
    single most important number this project has produced since note 12,
    because it is the one a reviewer asks for first.

    Two honest qualifications. Widening the U-Net **did** help ET on its own
    (+0.0055, p_holm 1.6e-09) — capacity is not zero and the paper must report
    the decomposition rather than claiming architecture alone. And TC/WT for
    the capacity control are both **inconclusive** against the baseline, so
    the capacity effect is an ET effect only; the ET-specific reading is that
    extra width helps the smallest, hardest region a little, and the fusion
    design helps it about four times as much.

20. **`dice_WT` is "inconclusive" in every single comparison despite p_holm
    values as low as 1.4e-09.** That is `compare_models` working as designed,
    not a bug: the verdict is conservative and returns `inconclusive` if
    EITHER the bootstrap CI contains 0 or Holm-adjusted p exceeds alpha. Every
    WT interval straddles zero (e.g. `neurovision` vs `capacity_control`:
    +0.0026, CI [-0.0018, +0.0069]). Whole tumour is nearly saturated at
    ~0.93 for all three models, so no WT claim may be made from any of these
    runs. Do not quote a WT difference from this table.

21. **GPU hours are approximate (~8) because the kernel log was not
    retained.** The checkpoint stores epoch, step, config and RNG state but no
    wall-clock, and the `kaggle kernels output` fetch that would have carried
    the log was cancelled to free bandwidth for the local evaluation. The run
    completed in a single session under `max_hours: 10.5`, so the true figure
    is bounded above by 10.5 and the 80-epoch schedule at the measured
    `baseline_unet3d` rate puts it near 8. **Fetch and keep the log next
    time** — this is the same provenance gap as note 4's missing commit SHA,
    in a different field.

22. **The interpretable pipeline is complete and runs end to end on CPU, for
    zero GPU hours.** Phases 0-5 all exist: atlas gate, localisation,
    eloquence (Tier C), burden, involvement (3b), report generation, the demo
    API route and panel, report agreement and population statistics. One
    command per stage; 189 cases localise in ~2.5 min and report in under a
    second on the M4. Five report sets were generated over the test split —
    ground truth plus `neurovision`, `baseline_unet3d`,
    `capacity_control_unet3d` and the superseded 96³ run — and the full
    1251-case cohort was localised for the population figure.

23. **THE PHASE 5 RESULT IS NEGATIVE: the +0.0267 ET Dice gain does NOT
    produce a measurably better structured report.** 16 report-agreement
    metrics, 189 paired cases, Holm-corrected within each comparison:

    | Comparison | conclusive metrics | which |
    |---|---|---|
    | `neurovision` vs `baseline_unet3d` | **1 of 25** | `relerr_vol_TC` +0.3843, p_holm 0.0215 |
    | `neurovision` vs `capacity_control_unet3d` | **1 of 25** | `relerr_vol_TC` +0.1057, p_holm 0.0081 |
    | `capacity_control_unet3d` vs `baseline_unet3d` | **0 of 25** | — |

    Everything else is inconclusive. The agreement metrics themselves are
    nearly identical across models — structure-list Jaccard 0.9074 /
    0.9122 / 0.9108 (`neurovision` / capacity control / baseline), structure
    precision 0.9430 / 0.9463 / 0.9460, recall 0.9497 / 0.9497 / 0.9487 —
    and **16 of the 25 metrics have the same median for all three models**,
    because for a typical case the three reports are identical.

    The Phase 3b involvement fields agree just as tightly and discriminate no
    better: ventricle-contact agreement 0.9841 / 0.9735 / 0.9788,
    deep white-matter contact 0.9630 / 0.9418 / 0.9577, epicentre-structure
    match 0.8571 / 0.8466 / **0.8783**, and epicentre SIDE agreement exactly
    1.0000 for all three models — no model ever puts the tumour's centre on
    the wrong side of the midline, so that field is saturated too.

    Provenance note on the family. An earlier run of this table had 16
    metrics, before the involvement fields were added to `compare_reports`;
    under that smaller Holm family `capacity_control` vs `baseline` also
    showed `relerr_vol_ET` as better (+0.0719, p_holm 0.0427), and it does not
    survive the 25-metric family. The 25-metric table is the one to report:
    the involvement metrics were always intended to be part of the
    report-agreement family and were absent only because Phase 3b landed
    later. Enlarging a Holm family is conservative — it cannot manufacture a
    significant result — which is the opposite of the hazard recorded against
    `compare_models`, where shrinking a family after seeing its p-values
    destroys the error-rate guarantee.

    The direction is not even consistently in `neurovision`'s favour:
    `match_top_structure` is 0.8571 for `neurovision` against **0.8783** for
    the baseline. Inconclusive, but it rules out the reading that the effect
    is real and merely underpowered in one direction.

    **Why this happens, and why it is a result rather than a disappointment.**
    The report is dominated by *which structures the tumour overlaps*, and
    that is robust to boundary-level segmentation differences: a few voxels
    at a margin rarely change whether a structure is involved at all. The
    only metrics that move are the volume ratios, whose tails are driven by a
    handful of small-region cases. So the interpretable layer is **stable**
    with respect to segmentation quality across this range — good for
    deployment, since the report does not swing with the model, and fatal to
    a "better model gives a better report" claim. The paper must state this,
    not bury it: it is the direct answer to the question Phase 5 was designed
    to ask.

    One qualification. `relerr_*` are ratios with a volume denominator and
    are heavy-tailed (measured max 128.6 for `relerr_vol_TC`), so their means
    are outlier-driven — mean `relerr_vol_ET` runs 0.4231 / 0.6103 / 0.6822
    against medians of 0.0533 / 0.0575 / 0.0615. The ordering matches Dice ET
    in both statistics, so the volume signal is real; it is simply too noisy
    for 15 of 16 metrics to survive Holm at n=189.

24. **The eloquence layer is DEGENERATE on BraTS 2021, and this is a finding
    about the layer, not a success rate.** Measured over the full 1251-case
    cohort: `near_eloquent` is `True` for **100%** of cases, the distance to
    the nearest Sawaya-listed structure is exactly **0.0 mm for 98.8%**, and
    98.5% of cases involve at least one eloquent structure. Essentially every
    glioma in this dataset directly touches a structure on that list.

    Consequences. `agree_near_eloquent` and `agree_eloquent_any` are
    constant at 1.0 in every model comparison with a confidence interval of
    exactly [0, 0] — they carry no per-case information and cannot
    discriminate between anything. Reporting "100% agreement on eloquence"
    would be true and completely misleading. `analysis/population.py`'s
    `eloquence_rates` therefore returns a `degenerate_fields` list and
    `scripts/population_stats.py` logs it at WARNING, so the saturation is
    surfaced rather than presented as a result. The 10 mm near-eloquent
    threshold is uninformative on this cohort and any future use of it needs
    a different threshold or a different cohort.

25. **Population anatomy over all 1,251 cases (ground truth).** Lobe shares
    of tumour volume: **unlabelled 34.5%**, frontal 20.3%, temporal 16.7%,
    parietal 8.2%, limbic 6.2%, deep 4.0%, insula 3.3%, occipital 3.3%,
    ventricle 2.7%, cerebellum 0.6%, callosum 0.1%, vermis 0.08%, brainstem
    0.05%. Most-involved structures at `frac_of_structure >= 0.05`:
    `LateralVentricle_R` 45.6% of cases, `LateralVentricle_L` 44.8%,
    `Insula_L`/`Insula_R` 34.3%, `Putamen_R` 31.7%, `CorpusCallosum` 31.3%.
    Median 24 structures involved per case, median unlabelled fraction 34.1%.
    Involvement layer (3b), test split: ventricle contact 81.0% of cases,
    deep white-matter contact 38.1%, median cortical fraction 0.361, epicentre
    resolved exactly on the centroid voxel in 47.6% of cases and via the 10 mm
    nearest-structure fallback otherwise.

    **A bug worth recording: the laterality table read `midline` 34.8% before
    it was fixed, against a true 0.27%.** `localize.py` gives the `unlabelled`
    pseudo-structure a `midline` placeholder laterality — it is not a real
    structure and has no side — and folding that placeholder into the midline
    bucket made it dominate. The table summed to exactly 1.0 and looked
    entirely reasonable, and a reader would have concluded that a third of the
    average glioma sits in midline structures, wrong by two orders of
    magnitude. Corrected: **L 34.7%, R 30.5%, midline 0.27%, unlabelled
    34.5%.** The near-symmetry of L and R is the population-scale check that
    would catch a left-right flipped atlas, which brain-mask Dice provably
    cannot (see the Phase 0 findings).

26. **EXTERNAL VALIDATION ON BraTS-Africa (SSA): the ET Dice advantage does
    NOT transfer.** ASNR-MICCAI BraTS2023-SSA Challenge TrainingData V2, 60
    cases, sub-Saharan African adult glioma, entirely held out —
    `configs/data/splits_ssa.yaml` puts all 60 in `test` with `train`/`val`
    deliberately empty, so nothing — no model, no temperature, no threshold —
    was ever fitted on it. Evaluated on the Mac CPU, `roi_size [64,64,64]`,
    `overlap 0.5`, identical to the published in-domain runs, zero GPU hours.
    Models: `neurovision` and `baseline_unet3d`. **The capacity control could
    NOT be included** — its checkpoint was never retrieved off Kaggle (it
    lived at `/tmp/capout/checkpoints/best.pt`), the same provenance gap as
    note 4's missing commit SHA, now costing an actual comparison.

    Two source-format differences had to be handled, and one of them was
    dangerous. SSA ships BraTS 2023 label values `{0,1,2,3}` (ET = 3), which
    `remap_labels` correctly refused. It also ships voxel axis codes
    `('R','A','S')` against BraTS 2021's `('L','P','S')` — anterior-posterior
    AND left-right reversed. Confirmed from CONTENT, not from headers:
    brain-mask Dice against SRI24 is **0.6772 as-is, 0.9082 A-P flipped**,
    versus **0.8795 as-is** for BraTS 2021. Fixed by deriving the transform
    from each file's own affine (`reorient_to_axcodes`), not by hardcoding a
    flip — which mattered immediately, because BraTS-PED ships LPS and needs
    no reorientation at all. Verified additive: three already-preprocessed
    BraTS 2021 cases re-preprocessed under the change are **bitwise
    identical** in image, label, bbox and affine, so the 1251 cached cases
    stay valid.

    Left-right could not be verified and rests on the affine. Brain-mask Dice
    is structurally blind to mirroring — measured 0.6772 as-is vs 0.6777 L-R
    flipped, and 0.9082 vs 0.9099 after the A-P fix. This is the same
    blindness recorded in the Phase 0 findings. The affine was independently
    confirmed correct on the other two axes. It was deliberately NOT settled
    by running the model both ways and keeping the better score, which would
    be selection on the outcome — the same error that manufactured 41-57% of
    the reported ECE. Bounded risk: training augmentation is `flip_prob: 0.5`
    applied independently on all three axes, so the model is approximately
    mirror-invariant by construction.

    Cohort differences. SSA tumours are ~1.7x larger than BraTS 2021: WT
    median 163,749 voxels vs 96,630 on the BraTS test split; ET median 28,427
    vs 17,337. **0.0%** of SSA cases have empty ground-truth ET, against 2.6%
    of BraTS 2021 — so the `ignore_empty=False` convention hands BraTS free
    Dice 1.0 on cases where SSA gets none, mildly flattering the in-domain ET
    number before any model behaviour is involved. 8.3% of SSA cases have no
    necrotic core, vs 3.4% in BraTS.

    **Result — the ET Dice advantage does not transfer.**

    | metric | in-domain BraTS test (n=189) | external SSA (n=60) |
    |---|---|---|
    | dice_ET | neurovision 0.8709 vs baseline 0.8442, **+0.0267, p_holm 1.4e-21, better** | 0.7784 vs 0.7792, **−0.0008, p_holm 0.3557, inconclusive** |
    | dice_TC | 0.9161 vs 0.9058, +0.0103, better | 0.7846 vs 0.7745, +0.0101, p_holm 1.0000, inconclusive |
    | dice_WT | 0.9321 vs 0.9276, +0.0045, inconclusive | 0.8959 vs 0.8975, −0.0016, inconclusive |
    | hd95_ET | 4.2410 vs 4.9088, +0.67 mm, inconclusive | 5.8353 vs 8.0536, +2.22 mm, inconclusive |
    | hd95_TC | 4.9764 vs 5.2430, +0.27 mm, inconclusive | 13.4473 vs 15.5019, +2.05 mm, inconclusive |
    | hd95_WT | 7.0868 vs 6.7210, **−0.37 mm (worse)**, inconclusive | 16.3541 vs 17.8657, +1.51 mm, inconclusive |

    Positive = `neurovision` better. **Zero of eight comparisons survive Holm
    on SSA. Every effect size is `negligible`.** Absolute drop for the
    baseline: ET −0.0650, TC −0.1313, WT −0.0301. For `neurovision`: ET
    −0.0925. `neurovision` therefore degrades MORE than the baseline —
    direction only, NOT statistically tested, because it compares two gaps
    measured on two different case sets and is unpaired.

    A correction trap worth recording. Uncorrected, `dice_ET` has
    p_wilcoxon = 0.0445 in the BASELINE's favour. Without Holm the conclusion
    would have been "neurovision is significantly worse out of distribution".
    Holm returns it to inconclusive. HD95 rows additionally discard 9-10
    exact ties each under `zero_method="wilcox"`, cutting effective sample
    further.

    Volume-matching makes the shift LARGER, not smaller. Restricting both
    cohorts to their overlapping WT-volume range (10th-90th percentile,
    37,045-175,728 voxels; BraTS n=138, SSA n=28), the BraTS→SSA ET gap
    widens from −0.0650 to **−0.0771** for the baseline and from −0.0925 to
    **−0.1087** for `neurovision`. So the 1.7x size difference was MASKING
    part of the shift rather than causing it. Note n=28 — a point estimate,
    not a tested difference.

    **Do not write the "it generalises" framing.** The headline +0.0267 ET
    gain, the project's single strongest and most rigorously controlled
    in-domain result, is worth nothing on an external cohort.

27. **WHERE THE IN-DOMAIN ET GAIN ACTUALLY COMES FROM.** Boundary-stratified
    error decomposition on the BraTS test split (n=189), summed over cases,
    from the `berr_`/`bfnr_`/`bfpr_`/`bn_` columns already emitted by
    `scripts/evaluate.py`. Reproduces the per-case figures recorded earlier
    exactly (WT 0-2 mm: 1,663,530/189 = 8,802 per case; WT 10 mm+ false
    positives 197,960/189 = 1,047).

    **ET, error voxels by distance to the true boundary:**

    | band | baseline | neurovision | change |
    |---|---|---|---|
    | 0-2 mm | 713,897 (78.6%) | 613,640 (77.6%) | −100,257 |
    | 2-5 mm | 96,295 | 93,826 | −2,469 |
    | 5-10 mm | 48,827 | 42,505 | −6,322 |
    | 10 mm+ | 49,650 | 41,012 | −8,638 |

    The 0-2 mm change is dominated by FALSE NEGATIVES falling 405,168 →
    313,531, a **23% cut**. Far-field (10 mm+) false positives fall 49,511 →
    40,990, a **17% cut**. So the ET gain is chiefly recovered enhancing
    tumour at the margin, plus fewer spurious distant predictions — which is
    what the architecture was designed to do, and is the strongest
    mechanistic support the in-domain claim has.

    **WT is mixed and does not support the same story.** Far-field false
    positives fall 197,960 → 166,692, but far-field false negatives RISE
    948 → 4,510. Net better, trading in both directions, consistent with WT
    being inconclusive in every comparison made in this project.

    Caveat: these are sums over 189 cases, so large tumours dominate. The
    per-case median of the 10 mm+ band is 0.0 — a minority of cases carry all
    of it.

28. **INFERENCE COST: 2.7x the parameters but 23.7x the FLOPs and latency.**
    Measured on CPU with `torch.utils.flop_counter.FlopCounterMode`, one 64³
    patch, batch 1, `torch 2.13.0`:

    | model | params | GFLOPs/patch | ms/patch (CPU) |
    |---|---|---|---|
    | `baseline_unet3d` | 12,870,489 | 11.3 | 45.8 |
    | `neurovision` | 34,911,341 | 267.4 | 1085.5 |

    The gap is far wider than parameter count suggests, because windowed
    cross-attention at stride 2 operates over 110,592 tokens on a 96³ patch.
    Set against an in-domain ET gain of +0.0267 that does not survive
    distribution shift (note 26), this is an unfavourable compute trade and
    the paper must state it plainly rather than leave a reviewer to compute
    it.

29. **TEMPERATURE SCALING DOES NOT TRANSFER UNDER DISTRIBUTION SHIFT, AND ON
    ET IT MAKES CALIBRATION WORSE.** `scripts/calibrate.py`, `mask=predicted`
    (the primary, label-free reporting mask). The temperature is fit on the
    BraTS 2021 **val** split in every column — the same fit, applied to two
    different apply-splits: the in-domain BraTS test split, and the external
    BraTS-Africa cohort. This required `calibration.fit_prep_dir`, added
    because the script previously resolved fit-split and apply-split labels
    from one preprocessed root and could not express a cross-cohort fit at
    all. Fitted temperatures, `baseline_unet3d`: ET 1.9155, TC 2.0250, WT
    1.9272; `neurovision`: ET 2.0531, TC 1.9864, WT 1.6265. Both converged.
    These sit in the normal 1.1-2.0 band for segmentation nets — unlike the
    3.08-4.75 range produced by the circular `union_foreground_mask`, whose
    magnitude was itself the tell that something was wrong.

    **ECE, mask=predicted:**

    | model | region | in-domain uncal | in-domain scaled | SSA uncal | SSA scaled |
    |---|---|---|---|---|---|
    | `baseline_unet3d` | ET | 0.0576 | 0.0218 | 0.0226 | **0.0890** |
    | `baseline_unet3d` | TC | 0.0316 | 0.0218 | 0.0900 | 0.0263 |
    | `baseline_unet3d` | WT | 0.0293 | 0.0090 | 0.0538 | 0.0137 |
    | `baseline_unet3d` | mean | 0.0395 | 0.0175 | 0.0555 | 0.0430 |
    | `neurovision` | ET | 0.0709 | 0.0121 | 0.0247 | **0.0685** |
    | `neurovision` | TC | 0.0327 | 0.0161 | 0.1021 | 0.0416 |
    | `neurovision` | WT | 0.0302 | 0.0122 | 0.0547 | 0.0189 |
    | `neurovision` | mean | 0.0446 | 0.0135 | 0.0605 | 0.0430 |

    **Three findings.**

    1. **In-domain, temperature scaling works: 2.3x better for the baseline
       (0.0395 -> 0.0175), 3.3x for `neurovision` (0.0446 -> 0.0135). Out of
       distribution it barely works: 1.3x and 1.4x, both landing on an
       identical 0.0430.**

    2. **On ET it is actively HARMFUL out of distribution — 0.0226 -> 0.0890
       for the baseline, a 4x degradation, and 0.0247 -> 0.0685 for
       `neurovision`.** The mechanism is coherent and worth stating: in-domain
       the model is overconfident on ET (0.0576), so `T ~ 1.9` helps; on the
       external cohort ET is ALREADY well calibrated before scaling (0.0226),
       because the model is less confident on unfamiliar data, so the same `T`
       pushes it into UNDER-confidence and ECE worsens. A temperature is one
       scalar fit to one distribution; nothing about it is guaranteed to hold
       on another.

    3. **`neurovision` is WORSE calibrated than the baseline before scaling,
       in both settings** — 0.0446 vs 0.0395 in-domain, 0.0605 vs 0.0555 on
       SSA — and after scaling both models land on exactly 0.0430 out of
       distribution. There is no calibration advantage for the proposed
       architecture anywhere in this table. This is the third independent
       measurement to reach that conclusion.

    **Consequence for the paper, and it cuts both ways.** The standing
    objection to any calibration claim in this project was "then
    temperature-scale the baseline too", and the recorded bar was the
    baseline's temperature-scaled in-domain ECE of 0.0158-0.0175. That
    objection is now answerable, but NOT in `neurovision`'s favour: temperature
    scaling is an **in-domain** fix that degrades under exactly the shift
    where calibration matters clinically, and it degrades worst on ET, the
    region a calibration claim would lean on. That is a reportable result
    about the METHOD, not about this architecture. Do not write it as evidence
    for `neurovision`.

    **Caveat.** Under the conservative `brain` mask, SSA ECE is 0.0064
    uncalibrated and 0.0055 scaled — both tiny, and the difference
    uninformative. That is the dilution effect already recorded: hundreds of
    millions of trivially-easy background voxels that were already calibrated
    swamp the statistic. `predicted` is the primary mask and `brain` the
    conservative check, exactly as recorded previously; the finding above
    rests on `predicted`.

30. **EXTERNAL VALIDATION, PART 2: BraTS-PED COMPLETES, AND THE POOLED SHIFT
    RESULT IS NEGATIVE IN THE OTHER DIRECTION — `neurovision` is conclusively
    WORSE than the baseline on tumour core off-distribution.** Measured
    2026-08-19. `outputs/eval_ped_neurovision` finished (99/99 cases; it had
    been interrupted at 19). Paired against `outputs/eval_ped_baseline_unet3d`
    via `compare_models`, Holm across an 8-metric family, saved to
    `outputs/compare_shift/`.

    **PED alone, n=99.** `dice_TC` **-0.0595, p_holm 0.0002, verdict `worse`**;
    `dice_mean` -0.0305, p_holm 0.0036, `worse`. `dice_ET` -0.0099 and
    `dice_WT` -0.0220 are inconclusive. Every HD95 metric is inconclusive.
    Absolute numbers are dire for both models — baseline ET Dice 0.5733
    (std 0.3564), `neurovision` 0.5634 — i.e. bimodal, catastrophic failure on
    a paediatric cohort for a model trained on adult glioma.

    **Pooled SSA + PED, n=159.** `dice_TC` **-0.0333, p_holm 0.0132, `worse`**.
    Everything else inconclusive, including all four HD95 metrics.

    **This kills the boundary-robustness-under-shift hypothesis.** Note 26
    recorded that on SSA alone (n=60) HD95 favoured `neurovision` in all three
    regions (-2.22 / -2.05 / -1.51 mm) with consistent effect size, and flagged
    it as underpowered rather than absent. At n=159 it is absent: `hd95_ET`
    improvement +1.459 mm with CI [-3.942, 0.700], `hd95_mean` +1.003 with CI
    [-2.699, 0.633]. Both straddle zero. The n=60 pattern was noise. **Do not
    write the "better boundaries under shift" framing** — this is the second
    time a promising direction in this project survived only until the sample
    grew, and the first (note 23) has the same shape.

    **Watch the HD95 denominators.** `hd95_ET` pairs only 78 of 99 PED cases
    (21 dropped as one-sided-empty NaN) against 88 for TC and 98 for WT. The
    per-model `summary.csv` means are therefore over DIFFERENT case subsets and
    must not be differenced by hand — doing so on ET reads as a 3.74 mm
    improvement where the paired estimate is 0.895 mm and inconclusive.

    **What this does and does not change.** It removes a claim the project did
    not yet have. It does NOT touch the in-domain results (notes 11-21), which
    are unaffected. And it strengthens the motivation for the failure-detection
    direction in `docs/research/improvement_plan.md`: a model that degrades
    off-distribution *and gives no signal that it has* is exactly what a
    per-case trust score exists for. That hypothesis remains untested.

31. **GATE 1, TEST A -- THE AMBIGUITY MAP IS NOT FLAT, AND IT IS NOT A
    RE-ENCODING OF ENTROPY.** Measured 2026-08-20 on the 10-case probe
    extraction (`outputs/ambiguity_probe10`, `neurovision` best.pt, level 0,
    whole-volume sliding window, logits reused from
    `outputs/neurovision/eval_test/logits`).

    **Not flat.** Mean in-predicted-foreground disagreement varies roughly 2x
    across cases: `amb_dis_mean_fg_mean` spans **0.1405 to 0.2632** over ten
    in-distribution cases, with per-case maxima of 0.48-0.85. The mechanical
    worry stated in advance in `docs/research/preregistration_ambiguity.md` --
    that the branch-supervision term (weight 0.1), which trains both probes
    toward the same label, may have driven the branches to agree everywhere --
    did not happen.

    **The two branch probes are not equally confident, and the asymmetry is
    systematic.** `amb_hswin_mean_fg` (0.58-0.94) is higher than
    `amb_hcnn_mean_fg` (0.31-0.67) in **every region of every one of the ten
    cases**. The Swin probe is markedly less certain than the CNN probe at the
    same voxels. This is consistent with the parameter split -- the Swin branch
    is only 2.0M parameters against the CNN branch's 18.9M -- and it means
    disagreement is driven mostly by the transformer branch being unsure rather
    than by the two branches confidently contradicting each other. Worth saying
    in the paper; it changes how the signal should be described.

    **Redundancy diagnostic (NOT a pre-registered endpoint).** Voxel-wise
    Spearman between mean disagreement and mean single-pass predictive entropy,
    sampled at 20,000 voxels per case inside a label-free 10 mm band around the
    predicted whole tumour, on 5 probe cases: **0.080, 0.168, 0.266, 0.276,
    0.345**. Disagreement and entropy are largely independent quantities, not
    two views of one. That does not by itself say disagreement predicts ERROR
    -- which is what Gate 1's endpoints measure -- but it rules out the
    specific H0 branch ("a redundant re-encoding of predictive entropy") that
    would have killed the pivot outright.

    **No endpoint was computed here, deliberately.** The pre-registration fixes
    the primary endpoints and forbids re-running the family on a subset. These
    are the descriptive checks Test A is specified to be, and no threshold was
    adjusted after seeing them.

32. **P1 IS ANSWERED, AND THE ANSWER IS YES -- BUT NOT IN THE DIRECTION P1
    PREDICTED. The fusion gate is strongly, monotonically and conclusively
    organised by anatomy, and adjacent scales have OPPOSITE polarity.**
    Measured 2026-08-20 over the **full 189-case test split**
    (`scripts/gate_boundary_profile.py`, gate maps from
    `scripts/extract_gates.py` on `neurovision` best.pt, one 64^3
    tumour-centred patch per case). The gate is the transformer weight: the
    merge is `cnn + layer_scale * gate * attn`, so gate near 1 means "admit
    Swin context here", gate near 0 means "this voxel is the CNN branch
    alone". Bands are signed distance in mm to the GROUND-TRUTH whole-tumour
    surface; negative is inside the tumour.

    | level | <-10 | -10..-5 | -5..-2 | -2..0 | 0..2 | 2..5 | 5..10 | >10 |
    |---|---|---|---|---|---|---|---|---|
    | 0 (stride 2)  | 0.8778 | 0.8858 | 0.8546 | 0.8603 | 0.8857 | 0.8971 | 0.8799 | 0.8621 |
    | 1 (stride 4)  | **0.9814** | 0.9645 | 0.8798 | 0.7568 | 0.6479 | 0.4652 | 0.3178 | **0.3252** |
    | 2 (stride 8)  | **0.0089** | 0.0148 | 0.0346 | 0.0685 | 0.1033 | 0.1980 | 0.4699 | **0.7992** |
    | 3 (stride 16) | 0.0023 | 0.0061 | 0.0134 | 0.0185 | 0.0217 | 0.0284 | 0.0447 | 0.1160 |

    Paired per-case contrasts, inner margin `[-2, 0)` mm minus elsewhere,
    percentile bootstrap over case indices, 10,000 replicates:

    | level | vs tumour interior | vs healthy tissue |
    |---|---|---|
    | 0 | -0.0237 [-0.0425, -0.0040] | -0.0018 [-0.0148, 0.0118] (inconclusive) |
    | 1 | **-0.2361 [-0.2526, -0.2188]** | **+0.4316 [0.4144, 0.4491]** |
    | 2 | +0.0550 [0.0411, 0.0698] | **-0.7307 [-0.7456, -0.7154]** |
    | 3 | +0.0152 [0.0118, 0.0188] | -0.0975 [-0.1064, -0.0892] |

    **What this settles.** `docs/research/contribution.md` recorded P1
    ("the mechanism fires") as **undecided** -- the producer and the reducer
    both existed and no number had ever been written down. The gate is not
    decoration: at stride 4 it runs from 0.98 deep inside the tumour to 0.33
    in surrounding tissue, a spread of 0.65 on a [0, 1] scale, monotone
    across all eight bands, with a CI 25 standard errors from zero.

    **What this refutes.** P1 as written predicted the gate would open toward
    *ambiguous zones* -- the tumour MARGIN specifically. It does not. At every
    level the margin is an intermediate value on a monotone tumour-to-healthy
    ramp, not a peak. The inner margin is significantly LOWER than the tumour
    interior at level 1 and only marginally different at level 0. Do not write
    "the gate opens at the boundary".

    **What it appears to be instead: a scale-dependent tissue decomposition,
    with opposite polarity at adjacent scales.** Level 1 admits transformer
    context INSIDE the lesion and shuts it off outside; level 2 does exactly
    the reverse, and level 3 weakly follows level 2. Level 0 (stride 2, the
    finest fused level) is essentially flat at ~0.87 -- fine-scale context is
    admitted everywhere. A reading consistent with the design: mid-scale
    global context is what distinguishes lesion tissue, coarse-scale context
    is what establishes normal brain, and fine-scale context is useful
    everywhere. This is a mechanistic finding the project did not predict and
    could not have got from a single-encoder model.

    **Caveats to keep with the number.** (i) The crop is one 64^3
    tumour-centred patch per case, so "healthy tissue" means peritumoral
    tissue inside that crop, not distant brain. (ii) 172 of 189 cases
    contribute to the interior contrast; 17 tumours have no voxel deeper than
    10 mm inside the surface. (iii) The gate is CONDITIONED on the inter-branch
    ambiguity signal, so this says the gate is organised, not yet that the
    ambiguity conditioning is what organises it -- that is P2, and
    `scripts/ambiguity_intervention.py` tests an inference-time version of it.

33. **FORENSICS: `remove_small_components` does NOT filter the region channels
    independently, and the pinned scikit-image `min_size` is already
    inclusive.** Both measured 2026-08-20, both were documented the other way.

    **Channel merging.** `neurovision.inference.postprocess.remove_small_components`
    calls MONAI's `remove_small_objects` once on the whole `(3, D, H, W)`
    tensor. MONAI's `independent_channels=True` default only decides whether
    the channels are first collapsed to one foreground mask; on that path a
    binary input is cast to `bool` and handed to
    `skimage.morphology.remove_small_objects`, which labels a boolean array
    with `scipy.ndimage.label` **over every axis it has**. The channel axis is
    an adjacency axis. Verified directly: a 27-voxel blob at `min_size=50` is
    removed when it sits in one channel and KEPT (81 voxels) when the same
    blob sits in all three -- which is the common case, since ET, TC and WT
    are nested.

    **Measured impact: negligible, and the "fix" is nominally worse.**
    Replaying `neurovision`'s saved test logits over 10 cases both ways:
    Dice moves by at most 3e-5 in any region (ET -0.00002, TC +0.00003,
    WT -0.00000), and HD95 gets WORSE under true per-channel filtering
    (ET +0.076 mm, TC +0.189 mm) because the extra removals sometimes delete
    a small component nearer the tumour than the ones that survive.
    Per-case voxel differences ranged 0-170 voxels out of 26k-363k.

    **Decision: document, do not change.** Changing the post-processing chain
    would invalidate comparability with `capacity_control_unet3d`, whose
    logits can no longer be regenerated (its checkpoint went to `/tmp/` and is
    gone), for an effect two orders of magnitude below the smallest difference
    this project reports. The docstring now records the real behaviour.

    **`min_size` is inclusive, not strict.** CLAUDE.md recorded the
    scikit-image `min_size` -> `max_size` migration as "dormant because deps
    are pinned". It is not dormant: at the pinned `scikit-image==0.26.0` the
    deprecation shim already applies the new semantics even when called with
    the old positional argument. A component of exactly 50 voxels is REMOVED
    today. Verified directly. This does not change any published number --
    every number this project has ever reported was produced under these
    semantics -- but the note was wrong and would have mispredicted the effect
    of a future dependency bump.

34. **EXPLORATORY: A LABEL-FREE READ-OUT OF THE FUSION GATE PREDICTS A CASE'S
    OWN DICE ON PED -- THE ONE COHORT WHERE THE FREE ENTROPY BASELINE
    COLLAPSES.** Measured 2026-08-20 by `scripts/gate_failure_detection.py`
    over all three cohorts (`outputs/gate_detection/`), from the gate maps
    `scripts/extract_gates.py` wrote in `center_on: prediction` mode. **Not
    pre-registered.** The pre-registered Gate 1 test
    (`docs/research/preregistration_ambiguity.md`) is about the whole-volume
    inter-branch DISAGREEMENT map and is a different quantity; read everything
    here as a hypothesis for that run to confirm or refute.

    Every feature is computed from the model's own output and never from the
    label, and the crop is centred on the PREDICTION, so the label leaks into
    neither the features nor the case selection. Partial Spearman against
    `dice_mean`, rank-residualised on three controls: predictive entropy in the
    predicted foreground, log predicted tumour volume, and mean predicted WT
    probability. Holm is applied across the **whole 51-row table**, entropy
    baseline rows included.

    | cohort | n | entropy baseline | best gate feature | p_holm |
    |---|---|---|---|---|
    | brats_test | 187 | **-0.3479** [-0.4804, -0.1995] | `gate2_bg` **-0.4611** [-0.5734, -0.3226] | 0.0128 |
    | ssa | 58 | **-0.6479** [-0.7865, -0.4167] | `gate1_fg` +0.3828 [0.0542, 0.5995] | 0.528 (n.s.) |
    | ped | 95 | 0.1357 [-0.0589, 0.3641] (**CI contains zero**) | `gate1_fg` **+0.5837** [0.3800, 0.7245] | 0.0128 |

    **The pattern is the interesting part, and it is not "the gate is a better
    detector".** In-distribution, entropy already works and seven gate features
    add signal on top of it (`gate2_mean` -0.4544, `gate2_fg` -0.3867,
    `gate0_agree` +0.3128, `gate1_fg` +0.2727, `gate3_mean` -0.2893,
    `gate3_bg` -0.2818, plus `gate2_bg` above). On SSA, entropy is the single
    strongest predictor anywhere in the table and **no gate feature survives
    Holm at all**. On PED -- the cohort where the model is measurably,
    catastrophically worse (note 30, `dice_TC` -0.0595, p_holm 0.0002) --
    entropy carries **no usable signal**, its CI straddling zero, while
    `gate1_fg` reaches +0.5837 with a CI 0.38 clear of zero, and `gate2_fg`
    reaches -0.4894 (p_holm 0.0128).

    So on one of the two external cohorts, the free baseline that any
    single-encoder model can compute stops working exactly where it is most
    needed, and a dual-encoder-only quantity keeps working. That is the shape
    of result the pivot requires. **It is one cohort out of two, from an
    exploratory family, on a patch rather than a volume -- it is not the
    result.**

    **Direction, which must be stated or the sign inverts the story.**
    `gate1_fg` is POSITIVE against Dice: more mid-scale transformer context
    admitted inside the predicted foreground goes with a BETTER case. Read with
    note 32 -- level 1's gate runs 0.98 deep inside the tumour down to 0.33 in
    surrounding tissue -- the reading is that on PED cases the model segments
    badly, the mid-scale context mechanism did not engage over the lesion. That
    is a post-hoc mechanistic story, not a tested one.

    **Caveats to carry.** (i) One 64^3 tumour-centred patch per case, so this
    says nothing about gate behaviour in distant healthy tissue. (ii) The
    feature set was chosen by the analyst even though Holm covers the family.
    (iii) n = 187 / 58 / 95 after dropping cases with a non-finite feature or
    target; at n=58 and n=95 the CIs are wide, and no claim rests on a point
    estimate. (iv) Nothing here substitutes for the pre-registered endpoint,
    which is scored on mean disagreement over the whole volume and had not been
    computed when this note was written.

35. **GATE 1 IS DECIDED, AND THE VERDICT IS PARTIAL: DISAGREEMENT LOCALISES
    ERROR OUT OF DISTRIBUTION, BUT DOES NOT RANK CASES.** Run 2026-08-23 by
    `scripts/detection_stats.py` over all 348 cases (BraTS test 189, SSA 60,
    PED 99) against `docs/research/preregistration_ambiguity.md`. Every
    threshold, mask, column and the 6-test Holm family were fixed before any
    number existed, and the family was run **once**, on complete data. Full
    tables live in that file's Result section and in
    `outputs/detection/detection_{case_level,voxel_level,family}.csv`.

    **Endpoint A, case level (partial Spearman vs Dice, controlling for
    entropy).** BraTS test **-0.393** (CI -0.512 to -0.265, p_holm 0.0006);
    SSA **-0.173** (CI -0.432 to +0.090); PED **+0.000** (CI -0.223 to
    +0.211). Strong in distribution, **null on both external cohorts.**

    **Endpoint B, voxel level (residualised AUROC for per-voxel error).** ANY
    region: test **0.578** (CI 0.560-0.596), SSA **0.569** (0.545-0.590), PED
    **0.677** (0.655-0.698) -- all three p_holm 0.0025, all three CIs
    excluding 0.5. Per region, ET is the strongest everywhere: **0.733 /
    0.689 / 0.784**.

    **Why PARTIAL.** PASS needed both conjuncts on one EXTERNAL cohort. PED
    clears the voxel threshold decisively and has a case-level correlation of
    exactly nil; SSA clears neither threshold. Not FAIL, because the voxel
    endpoint clears both its CI and its threshold on PED and the map is not
    flat (note 31).

    **The finding, stated so it cannot be inflated later.** Disagreement says
    WHERE a prediction is wrong, beyond entropy, and keeps saying it under
    distribution shift. It does NOT say WHICH CASE will be bad. The case-level
    columns show the mechanism: on SSA and PED the raw disagreement-vs-Dice
    correlation is +0.006 and +0.015, while ENTROPY ALONE is already a strong
    case-quality predictor there (-0.782 and -0.554). There is no case-level
    headroom left for disagreement to occupy, and it occupies none. Do not
    write "better than entropy at flagging bad cases" -- this data refutes it
    externally.

    **Two cases leave the case-level analysis and stay in the voxel one.**
    `BraTS-SSA-00215-000` and `BraTS-PED-00051-000` have an empty predicted WT
    mask, so the predicted-foreground-masked scalar is NaN (n = 59 and 98);
    at voxel level they fall back to whole-volume sampling (n_cases 60 and
    99). Correct per endpoint, but two different denominators -- never report
    them as one.

    **Consequence for Phase 2, decided from these numbers rather than the
    plan's assumption.** Gate 2 as written tests referral, which is a
    CASE-LEVEL operation, on SSA/PED -- exactly the endpoint that came back
    null. It is predicted to fail as specified and must be respecified around
    voxel- and region-level localisation, ET first, BEFORE it is run.

    **Cost: zero GPU hours.** Extraction of all 348 cases ran on the M4 CPU at
    ~94 s/case, one process at a time via `scripts/extract_ambiguity_serial.py`
    (~7 h wall, 5.3 GiB peak RSS, no swap), and the analysis loads only saved
    caches.

36. **PHASE 2.4 IS ANSWERED AND THE INFERENCE-ROI HYPOTHESIS IS REFUTED: A
    BIGGER INFERENCE WINDOW DOES NOT FIX MULTIFOCALITY, AND IT COSTS WT DICE.**
    Run 2026-08-23 on the M4 CPU, zero GPU hours: `neurovision`'s own
    `best.pt` re-evaluated over the same 189 test cases at **ROI 96³** instead
    of the trained 64³ (`outputs/roi_sweep/neurovision_roi96/`), then profiled
    with `scripts/burden.py` and compared paired against the existing 64³ run.

    **Segmentation quality, paired over 189 cases, Holm-corrected.** Every
    region is inconclusive except one, and that one goes the wrong way:
    **`dice_WT` -0.0045, CI (-0.0079, -0.0016), p_holm 0.0090 -- WORSE at
    96³.** `dice_ET` +0.0029 (CI -0.0051 to +0.0154), `dice_TC` -0.0019, all
    three HD95 rows inconclusive with CIs straddling zero.

    **Multifocality, which is the reason 2.4 existed at all.** Ground truth
    22.8% of cases multifocal; `neurovision` at 64³ reports **33.9%**
    (+0.111); at 96³ it reports **41.3%** (+0.185). Mean WT component count
    1.39 (GT) -> 1.65 (64³) -> 1.76 (96³). Per-case agreement with the ground
    truth's multifocal flag falls 0.7302 -> 0.6878, but **that fall is NOT
    established**: paired bootstrap delta -0.0423, CI (-0.1005, +0.0107),
    McNemar exact p = 0.20 on 30 discordant cases. So the honest statement is
    that a larger inference ROI **does not improve** multifocality agreement --
    the point estimate moves the wrong way and no interval supports an
    improvement -- rather than that it significantly harms it.

    **What this closes.** CLAUDE.md recorded patch size as "the candidate
    mechanism" for the superseded 96³/200-epoch U-Net producing a BETTER
    structured report than a higher-Dice model. This experiment separates the
    two things that were confounded in that observation: **inference ROI is not
    the mechanism.** If patch size explains it at all, it must act through the
    TRAINING patch size, which cannot be tested without a GPU run and is not
    currently funded.

    **ROI 128³ was planned and is DECLINED, for a measured reason.** Peak RSS
    at ROI 96³ was **13.7 GiB** on a 16 GiB machine (measured with
    `/usr/bin/time -l`; the 64³ jobs peak at ~5.3 GiB). A 128³ window is
    ~2.4x the voxels of 96³, so the run would page heavily, and the 64->96
    trend already points away from any benefit. Running it would cost ~5 h of
    CPU to make a refuted hypothesis slightly more refuted.

    **Filing note, second instance in one day.** `scripts/burden.py` wrote to
    `cfg.output_dir`, which interpolates `experiment_name` (default
    `baseline_unet3d`), so this profile initially landed in the BASELINE's
    directory. Fixed by `analysis.burden.out_dir` (commit `570d3a2`), the same
    fix `analysis.detection.out_dir` needed hours earlier.

37. **GATE 2 IS DECIDED: PARTIAL BY THE RULE, NEGATIVE IN SUBSTANCE. ADDING
    DISAGREEMENT TO ENTROPY DOES NOT LOCALISE ERROR BETTER.** Run 2026-08-23,
    zero GPU hours. Combiner fitted once on the 187-case val split (3,740,000
    voxels) and applied FROZEN to all three cohorts, per
    `docs/research/preregistration_gate2.md`. Fitted weights:
    `[-8.700, 7.351, 0.952]` -- disagreement does get a positive weight, ~13%
    of entropy's, so the combiner genuinely uses it.

    | Cohort | AUROC: entropy -> +disagreement | recall@5%: entropy -> +disagreement |
    |---|---|---|
    | BraTS test | 0.9162 -> 0.9150 (**-0.0013**, p_holm 0.042) | 0.5113 -> 0.4720 (**-0.0394**, p_holm 0.0006) |
    | SSA | 0.8874 -> 0.8888 (+0.0013, ns) | 0.3432 -> 0.3304 (**-0.0128**, p_holm 0.0006) |
    | PED | 0.8307 -> **0.8428** (+0.0121, p_holm 0.0006) | 0.1826 -> 0.1812 (-0.0014, ns) |

    4 of 6 rejected under Holm, **but two of the four rejections are losses**
    and `passed_cohorts` is empty. The operational endpoint -- how much of a
    case's error you catch by flagging 5% of predicted foreground -- is worse
    or unchanged EVERYWHERE.

    **The finding is the gap against Gate 1, not the table alone.** Gate 1
    asked whether disagreement carries error information entropy lacks
    (residualise, then test): yes, everywhere. Gate 2 asks whether that
    information improves a FITTED OPERATING DETECTOR: no. Both are true.
    **Incremental information is not incremental utility** -- entropy's own
    per-case AUROC is already 0.916 in distribution, and there is no room left
    that this feature can fill.

    **A limitation of the gate's own design, recorded not hidden.** The
    combiner is fitted by pooled logistic likelihood while the endpoints are
    per-case rank metrics; a likelihood-optimal weighting can lose on a rank
    metric, which is the leading explanation for the in-distribution losses.
    That mismatch was baked into the pre-registration, so swapping in a
    rank-optimised combiner now would be post-hoc -- exactly what the
    pre-registration exists to prevent. It is the one defensible follow-up,
    pre-registered in advance, on fresh data.

38. **P2 IS ANSWERED AND IT IS THE PRE-REGISTERED NULL: THE AMBIGUITY
    CONDITIONING BUYS NOTHING MEASURABLE. THE GAIN IS THE GATED FUSION
    ITSELF.** `ablation_content_only_gate` trained 80/80 epochs over three
    Kaggle sessions (24.16 GPU-h, `GIT_REF=7caacfa`, W&B `ddkbitjp`) and was
    evaluated on the test split ON THE GPU -- deliberately the same device as
    `neurovision`'s own evaluation, because the expected effect is tiny and
    CPU-vs-GPU float differences land in the decimals that matter.

    **Test Dice: ET 0.8686 / TC 0.9158 / WT 0.9333.** Paired over the same 189
    cases, `neurovision` vs `ablation_content_only_gate`: **ET +0.0022 (CI
    -0.0067 to +0.0152, p_holm 0.17), TC +0.0003, WT -0.0012 -- every metric
    INCONCLUSIVE**, all three HD95 rows likewise.

    **The decomposition, now complete, in ET Dice against `baseline_unet3d`:**

    | Contribution | ET Dice | Verdict |
    |---|---|---|
    | Total (`neurovision` - baseline) | +0.0267 | better, p_holm 1.4e-21 |
    | Width alone (`capacity_control` - baseline) | +0.0055 | better, p_holm 1.6e-09 |
    | Content-only gated fusion (`ablation` - `capacity_control`) | **+0.0189** | better, p_holm 4.4e-20 |
    | Ambiguity conditioning (`neurovision` - `ablation`) | **+0.0022** | **INCONCLUSIVE** |

    The ablation on its own beats the baseline by **+0.0244** (p_holm 3.9e-22)
    -- i.e. it recovers essentially all of `neurovision`'s advantage without
    ever seeing inter-branch disagreement. **So ~92% of the architectural gain
    is the dual-encoder gated fusion, and the ambiguity conditioning -- the
    part this project claimed as its novelty -- contributes an amount
    indistinguishable from zero.** `docs/research/contribution.md` P2 declared
    this outcome in advance and requires it be reported: the contribution is
    the gate's per-voxel spatial resolution, not the signal it is conditioned
    on. That is a smaller and much more ordinary claim, and gated cross-
    attention fusion for dual-branch medical segmentation is already published.

    **Three caveats that must travel with it.** (i) The CI upper bound is
    +0.0152, so an effect up to ~1.5 Dice points is NOT excluded -- this is
    "no detectable difference at n=189", never "proven identical"; a real
    equivalence claim needs a TOST with a pre-set margin. (ii) Single seed, so
    there is no seed-to-seed noise floor to compare +0.0022 against. (iii) The
    ablation's `best.pt` is its FINAL epoch (79) while `neurovision` peaked at
    69, so the ablation was still improving when the budget ran out -- if
    anything that handicaps the ablation and makes the null more robust, not
    less.

    **This agrees with Gate 2 (note 37), independently.** One line of evidence
    says the disagreement feature adds no utility to an uncertainty detector;
    the other says removing it from the gate costs no accuracy. Two different
    experiments, same conclusion.

39. **THE "1/10 THE COST" CLAIM IS REFUTED FOR DISAGREEMENT AND ESTABLISHED FOR
    ENTROPY. MC-DROPOUT'S 10x COST BUYS NOTHING OVER A SINGLE-PASS ENTROPY
    MAP.** Run 2026-08-23 after generating per-voxel MC mutual-information maps
    for both external cohorts on the T4 (N=10; SSA 1.25 h, PED 2.07 h). All
    three signals scored at the SAME label-free sampled voxels, per case, as
    AUROC for per-voxel error. Secondary analysis, outside the pre-registered
    families; the TOST margin of 0.03 AUROC was fixed in
    `docs/research/execution_plan.md` Phase 2 before any external MC map
    existed.

    | Cohort | entropy | disagreement | MC-dropout (N=10) |
    |---|---|---|---|
    | SSA (n=60) | **0.8874** | 0.7095 | 0.8661 |
    | PED (n=99) | 0.8307 | 0.7879 | **0.8437** |

    **Disagreement vs MC, paired TOST @ 0.03:** SSA **-0.1566** (CI -0.1801 to
    -0.1325), **NOT equivalent**, p 1.0. PED **-0.0558** (CI -0.0750 to
    -0.0367), **NOT equivalent**, p 0.995. These are not underpowered nulls --
    the intervals sit entirely OUTSIDE the margin on the wrong side.
    Disagreement beats MC in only **30 of 159** cases.

    **Entropy vs MC, same test:** SSA **+0.0214** (CI +0.0132 to +0.0300),
    **EQUIVALENT**, p 0.0238. PED **-0.0129** (CI -0.0208 to -0.0048),
    **EQUIVALENT**, p 3.7e-05. Note both are simultaneously *significantly
    different* from MC and *practically equivalent* to it within the
    pre-registered margin -- that is not a contradiction, it is precisely what
    a margin is for, and both facts get reported.

    **What this settles.** The project's cost argument was "inter-branch
    disagreement matches MC-dropout at one tenth the inference cost". It does
    not: it is clearly and consistently worse on both external cohorts. What
    IS true is the more boring and more useful statement -- **single-pass
    predictive entropy, which every segmentation model emits for free, is
    equivalent to 10-sample MC-dropout for voxel-level error localisation on
    both external cohorts.** MC-dropout's tenfold cost buys nothing here. That
    is a real finding, it is just not a finding about this architecture.

    **Consequence for the paper.** Delete the efficiency framing that Gate 1's
    *Partial* row licensed. Combined with notes 37 and 38, three independent
    experiments now agree that the disagreement signal does not carry
    practical value: it does not improve a fitted detector (Gate 2), removing
    it costs no accuracy (P2), and it is worse than both free entropy and
    MC-dropout as a localiser (here).


40. **BraTS 2026 Challenge 3 (generalizability) -- deadline UNRESOLVED from
    public sources; must be checked from a logged-in Synapse session.** Checked
    2026-08-23 while drafting `docs/research/master_plan.md`. The 2026 cluster
    is real and Challenge 3 is *"Generalizability of brain tumor sub-region
    segmentation across tumor entities in MRI scans"* -- an unusually exact fit
    for this project's one conclusively measured weakness (pooled external
    `dice_TC` -0.0333, p_holm 0.0132, note 30). What could NOT be established:
    the validation-phase window or the submission deadline. The challenge portal
    (`syn74274097`, `challenges.synapse.org/brats2026`) is a JavaScript
    application that serves no dates to a plain fetch, the Zenodo record
    (19714728) lists the five challenges but no timeline, and the per-challenge
    GitHub repos defer to Synapse.

    What IS known and bounds the answer: MICCAI 2026 runs **27 Sep - 1 Oct 2026
    in Strasbourg**, with the challenge sessions on 4 and 8 October; historical
    BraTS validation phases run roughly July-August with a containerised
    (MLCube/Docker) submission at the end. Today is 23 Aug 2026. So the 2026
    window is **probably closed or closing**, and the plan assumes BraTS 2027 as
    the realistic target while treating a 2026 entry as an upside to be checked,
    not a dependency. Recorded rather than left implicit because "we could have
    entered the one challenge that matches our result" is exactly the kind of
    thing that gets discovered too late.

    **Action, unchanged by this null result:** log in to Synapse, read the
    Challenge 3 timeline, and replace this note's uncertainty with a date. The
    released data is worth pulling either way -- it is the closest public
    benchmark to the claim this project is now defending.

    ---

    **RESOLVED 2026-08-23, without a Synapse login. The 2026 window is CLOSED --
    it shut on 31 July 2026, 23 days ago.** The dates were never behind the
    login. They are in the challenge's own structured design document, attached
    as a PDF to the same Zenodo record (19714728) whose *landing page* carries
    no timeline -- `284-BraTS_2026_Cluster_of_Challenges_2026-04-22T16-36-39.pdf`,
    100 pages, published 23 April 2026. The earlier check read the HTML and
    stopped. Recording the method as well as the answer, because this is the
    third instance of the same shape of miss in this project: **the artifact
    that answers the question was one link below the page that did not.**

    Official timeline, quoted from p. 13 of that document:

    | Date | Milestone |
    |---|---|
    | 16 March 2026 (or earlier) | Registration opens -- release of training and validation data |
    | **31 July 2026** | **Submission deadline for paper and containerized method** |
    | 7 August 2026 | Initial invitations sent |
    | 14 August 2026 | Paper reviews available |
    | 19 August 2026 | Release of summary testing results to each participant |
    | 21 August 2026 | Camera-ready paper and copyright form deadline |
    | 24 August 2026 | Top-performing methods contacted for oral presentation slides |
    | 4-8 October 2026 | Challenge at MICCAI |

    Registration itself ran "from website opening ... until the short paper
    submission deadline (July 31, 2026)", so registration is closed too.

    One correction to the note above: the challenge sessions are **4-8 October
    2026**, and the "27 Sep - 1 Oct" figure quoted earlier for MICCAI 2026 does
    not match this document. Trust the challenge document for challenge dates.

    Challenge 3's exact title, from p. 3: *"Generalizability of brain tumor
    sub-region segmentation algorithms across tumor entities and appearances in
    pre-operative brain multi-parametric structural MRI scans."*

    **Consequences, and they are all mild.** BraTS 2026 was never a dependency
    of the master plan -- it was flagged as an upside to check in week 1, which
    is what just happened. The plan is unchanged: **target BraTS 2027.** Two
    things are worth doing now rather than in a year:

    1. The 2026 training and validation data was released publicly from 16 March
       2026 and is the closest public benchmark to the claim this project is now
       defending. Pull it when a phase needs it; it does not expire with the
       deadline.
    2. The 2026 cluster added **non-annotated training data for semi-supervised
       learning** (p. 2), which is new this year and is a standing invitation
       for the 2027 iteration.

    Nothing in Phase A-H moves. This note is now closed.


41. **LESION-WISE RE-SCORING OF EVERY SAVED RUN. Voxel Dice was overstating
    this project's performance by 0.10-0.32 Dice, and the architecture gain is
    ~1.9x LARGER under the metric BraTS has actually used since 2023 -- but it
    still does not transfer out of distribution.** Run 2026-08-24, master plan
    item A3. Zero GPU: re-scored from the saved fp16 logits of both surviving
    models on all four splits, 1,070 cases, ~30 min on the M4.

    **Provenance guard passed first.** Every one of the eight replays
    self-consistency-checked against that run's committed `per_case_metrics.csv`
    before anything else was computed. Mean absolute Dice deltas were **1e-17
    to 5e-17** on `dice_ET/TC/WT` across all eight -- i.e. the logits on disk
    reproduce the published numbers to floating-point noise, so these
    lesion-wise numbers are scored on exactly the predictions the paper
    already reports.

    **Finding 1 -- voxel Dice flatters every model, and worst where the paper
    was most confident.** Lesion-wise minus voxel-wise Dice:

    | Run | ET | TC | WT |
    |---|---|---|---|
    | `neurovision` test | -0.116 | -0.101 | **-0.214** |
    | `baseline_unet3d` test | -0.140 | -0.128 | **-0.239** |
    | `neurovision` SSA | -0.233 | -0.225 | **-0.282** |
    | `baseline_unet3d` SSA | -0.266 | -0.230 | **-0.321** |
    | `neurovision` PED | -0.098 | -0.206 | -0.190 |
    | `baseline_unet3d` PED | -0.136 | -0.265 | -0.228 |

    WT is the worst case everywhere. `neurovision`'s WT on test reads **0.9321
    voxel-wise and 0.7183 lesion-wise**. The project's standing instruction
    "make no claim on WT, it is saturated at ~0.93" was correct not to claim it,
    but for the wrong reason: WT is not saturated, it is *mis-measured*. A
    region defined as the union of everything tumour-related is exactly where a
    single large component dominates the voxel count and hides the satellites.

    **Finding 2 -- the mechanism is spurious lesions, and it is now visible.**
    Mean false-positive lesions per case, test split: ET **0.32** for
    `neurovision` vs **0.47** for the baseline (32% fewer), WT 0.56 vs 0.61. On
    SSA, ET 0.98 vs 1.38 (29% fewer). False *negatives* are near-identical
    between the two models on every split (e.g. test ET 0.28 vs 0.30). So the
    architecture's advantage is almost entirely **fewer invented lesions**, not
    better recall -- which is the same over-reporting of multifocality already
    measured from the report side (30.7-41.3% predicted against a true 22.8%,
    notes 22-25), now seen in the segmentation metric itself and attributed to
    the right model.

    **Finding 3 -- in distribution, the gain is bigger lesion-wise than
    voxel-wise.** Paired, Holm-corrected within each cohort's three regions,
    10,000-sample bootstrap CIs, seed 42, via `analysis.statistics.compare_models`:

    | test (n=189) | lesion-wise delta | 95% CI | p_holm | verdict |
    |---|---|---|---|---|
    | `lwdice_ET` | **+0.0508** | +0.0225 to +0.0798 | 1.9e-15 | **better** |
    | `lwdice_TC` | **+0.0371** | +0.0097 to +0.0653 | 2.7e-10 | **better** |
    | `lwdice_WT` | +0.0300 | **-0.0021** to +0.0620 | 9.6e-06 | inconclusive |

    ET lesion-wise **+0.0508 against the voxel-wise +0.0267** -- 1.9x the
    effect. TC is newly separable: voxel-wise TC was +0.0103 and never claimed;
    lesion-wise it is +0.0371 with a CI excluding zero. WT stays inconclusive
    because its CI crosses zero, so the "no claim on WT" rule survives intact
    under the new metric as well.

    Effect sizes stay honest: Cohen's dz is 0.25 (ET), 0.19 (TC), 0.13 (WT) --
    `compare_models` labels these "small" and "negligible". A large p with a
    small dz is what a consistent, modest, per-case advantage looks like at
    n=189, and it should be reported that way rather than as a big win.

    **Finding 4 -- and this is the one that did NOT go the way it looked.
    Nothing transfers out of distribution, lesion-wise either.** Every external
    point estimate favours `neurovision`, and **every single one is
    inconclusive** -- all six confidence intervals straddle zero:

    | Cohort | region | delta | 95% CI | p_holm | verdict |
    |---|---|---|---|---|---|
    | SSA (n=60) | ET | +0.0319 | -0.0184 to +0.0825 | 0.0367 | inconclusive |
    | SSA | TC | +0.0149 | -0.0537 to +0.0797 | 0.3673 | inconclusive |
    | SSA | WT | +0.0368 | -0.0214 to +0.0949 | 0.0718 | inconclusive |
    | PED (n=99) | ET | +0.0282 | -0.0271 to +0.0863 | 0.0315 | inconclusive |
    | PED | TC | -0.0006 | -0.0495 to +0.0486 | 0.6232 | inconclusive |
    | PED | WT | +0.0156 | -0.0382 to +0.0672 | 0.5341 | inconclusive |

    Worth stating plainly, because the table of point estimates invites the
    opposite reading and I nearly took it: **six positive-looking numbers with
    six CIs crossing zero is not evidence of transfer.** It is an underpowered
    null at n=60 and n=99, consistent with note 30's voxel-wise finding rather
    than a correction to it. Three of the six also have p_holm below 0.05 while
    their CIs include zero -- the Wilcoxon is a rank test and the bootstrap CI
    is on the mean, and they disagree here precisely because the per-case
    differences are small, consistent in sign, and heavily tied. The CI is the
    reported quantity, per this project's convention.

    **Finding 5 -- pediatric tumour core is a floor, not a gradient.** PED TC
    lesion-wise Dice is **0.2339** (`neurovision`) and **0.2345** (baseline),
    with lwNSD ~0.15 and **64 of 99 cases exact ties between the two models**.
    Both networks essentially fail to detect pediatric tumour-core lesions at
    all, and they fail identically. No architecture claim of any kind can be
    made on that cell, and the tie count is why.

    **Status of these numbers: EXPLORATORY, not pre-registered.** The
    lesion-wise family was not named in any pre-registration before it was
    computed; only the strong-baseline gate (`preregistration_strong_baseline.md`)
    names lesion-wise ET as a co-primary endpoint, and that gate has not run.
    Holm correction was applied within each cohort's three regions but NOT
    across the four cohorts. These rows enter
    `docs/paper/claims_and_evidence.md` as secondary/exploratory and must be
    labelled as such wherever they appear.

    **Two cells that can never be filled.** `capacity_control` has no
    checkpoint, no logits and no predictions, so the 79% architecture / 21%
    capacity decomposition stays voxel-wise permanently.
    `ablation_content_only_gate` has no saved volumes but its checkpoint
    survives, so it costs one ~15 min CPU inference pass per split to add.

    Artifacts: `outputs/replay_lesionwise/<eval_dir_basename>/per_case_default.csv`,
    eight directories. Rebuild with `scripts/replay_logits.py
    analysis.replay.lesionwise.enabled=true` from `.venv-analysis`.


42. **CONFORMAL RISK CONTROL: THE GUARANTEE HOLDS IN DISTRIBUTION AND BREAKS
    UNDER SHIFT BY AN AMOUNT THAT TRACKS HOW FAR THE SHIFT IS. Both models,
    same pattern, so it is a property of the setting rather than of an
    architecture.** Run 2026-08-24, Phase B, pre-registered in
    `docs/research/preregistration_conformal.md` (result section filled there;
    nothing above its Result line was edited). Zero GPU: calibrated on val
    (n=187), applied **frozen** to test (n=189), BraTS-Africa (n=60) and
    BraTS-PEDs (n=99), for `neurovision` and again for `baseline_unet3d`.

    **Guards first, per the pre-registration.** Replay self-consistency against
    every committed `per_case_metrics.csv`: **1e-17** mean absolute Dice.
    Degenerate-endpoint falsifier: α=1.0 selects the largest grid threshold for
    both regions. Monotonicity held on every fitted risk curve. No α was
    infeasible.

    **B1 -- in distribution the bound holds, 6/6, for both models.** Realised
    risk sits at 0.64x-0.96x of nominal α on the test split. The theorem does
    what it says, and it does it for a model whose calibration claim is dead --
    which is the point: the guarantee is a property of the *procedure*.

    **B2 -- under shift, 7 of 12 cells violated for EACH model, none holds.**
    Ratio of realised risk to nominal α:

    | cohort | region | `neurovision` | `baseline_unet3d` |
    |---|---|---|---|
    | SSA (n=60) | WT | 1.07-1.10x (inconclusive) | 0.85-0.95x (inconclusive) |
    | SSA | TC | 1.42-1.71x (**violated** at α=0.10, 0.20) | 1.48-1.81x (**violated** at all three) |
    | PED (n=99) | WT | 1.39-1.94x (**violated** at α=0.05, 0.10) | 1.18-1.38x (**violated** at α=0.05) |
    | PED | TC | **3.50x-11.47x** (violated at all three) | **3.44x-10.66x** (violated at all three) |

    **The structure of the failure is the finding.** Excess risk is ordered by
    how far the shift is -- BraTS-Africa is a scanner and population shift
    within the same disease and barely dents WT coverage; BraTS-PEDs is a
    different disease entity and breaks WT and destroys TC -- and ordered by
    region difficulty, matching exactly where the segmentation itself fails
    (note 41: PED TC lesion-wise Dice 0.234, both models failing identically).
    A conformal guarantee does not survive a disease-entity shift, and it fails
    *gradedly* rather than collapsing uniformly. That graded excess is the
    quantity the Phase E refusal gate needs, and it says the gate must key on
    **how far out of distribution the input is**, not on whether the mask looks
    uncertain.

    **Replication across models is the robustness check that matters here, and
    it passed.** Both models violate 7 of 12, in substantially the same cells,
    with PED TC at ~10.7x-11.5x for both. Nothing about this is architectural.

    **A mildly counterintuitive detail worth keeping.** The *more accurate*
    model does not have better coverage under shift. On SSA WT the baseline is
    comfortably under nominal (0.85-0.95x) while `neurovision` sits slightly
    over (1.07-1.10x); on PED WT at α=0.10 the baseline is inconclusive at
    1.18x while `neurovision` is violated at 1.39x. Better Dice does not buy
    better conformal coverage, which is exactly what the theory predicts and
    is easy to assume otherwise.

    **Mandatory secondary -- what the guarantee costs in mask volume.**
    Registered as mandatory precisely so it could not be quietly dropped. It
    came back surprising, in the useful direction. In distribution at α=0.05
    the conservative mask grows only **1.10x (WT) / 1.18x (TC)** -- the
    guarantee is cheap. At α=0.10 and 0.20 the inflation is **below 1.0**
    (0.88x-0.96x): the bound is satisfied by a mask *smaller* than the default
    0.5-threshold prediction. That is not a bug. It says the deployed operating
    point is already more conservative than a 10%-miss-rate guarantee requires,
    so at those α the conformal layer licenses being *less* cautious. Both
    directions are legitimate outputs and both must be reported; quoting only
    the α=0.05 row would misrepresent the method. PED TC at α=0.05 is the one
    heavy tail: 2.29x mean against a 1.27x median, with 10 cases skipped for an
    empty reference mask.

    **Caveat that must travel with every α=0.20 row.** λ̂ = 0.95 is the largest
    value in the threshold grid, so those are **boundary solutions**: the true
    λ̂ is "≥ 0.95", censored by the grid rather than measured at it.

    **The registered threat did not materialise, and that was a real test.**
    The pre-registration predicted in advance that because every checkpoint in
    this project was selected on val by `val/dice_mean`, λ̂ fitted on val might
    be too permissive and test risk might exceed α -- *upward*, if at all. It
    did not exceed α anywhere in distribution. The
    exchangeable-halves-of-test arm was registered specifically to diagnose
    such a violation; with no violation to diagnose it **was not run**, and
    that is recorded rather than dropped.

    **What this does not license.** No calibration claim and no risk-coverage
    claim -- both are dead, and conformal risk control does not revive them,
    because the bound holds for an arbitrarily bad model. Also, per
    `configs/data/splits_ssa.yaml`'s standing rule that nothing may ever be
    fitted on the external cohorts, the Mondrian per-cohort recalibration arm
    remains a **counterfactual**, not an external-validation number.

    Artifacts: `outputs/conformal/{neurovision,baseline_unet3d}/` --
    `fit.json`, `realised_risk.csv`, `inflation.csv`, per-split `curves.npz`
    (the sufficient statistic; recalibration at any α is arithmetic on it).

---

## Planned

Written before the runs start so the plan is on record and cannot be
retrofitted to whatever came out.

### The budget this plan is cut to

**60 GPU-h, two weeks, Kaggle free tier (~30 h/week), single seed.** Fixed
2026-08-06. Every cut below follows from that number and from one ranking
decision: **the contribution ablation outranks baseline breadth.**

The reasoning. `docs/research/contribution.md` says the claim is not "we gated
the fusion" — it is that the gate conditions on inter-branch *disagreement*.
Rung 2 of its P2 ladder, the content-only gate, is the only run that can
distinguish those two claims. Without it the paper reduces to "we built a fusion
model", which is not publishable. `baseline_swinunetr` at ~25 h is 42% of the
entire budget for a row that strengthens the results table but proves nothing
about the mechanism. So it is cut, and the hours go to the ablation.

What that costs the paper, stated plainly so it is not discovered in review:
there is no transformer baseline trained under our schedule on our splits.
Published SwinUNETR BraTS-2021 numbers are **not** a substitute — they are on
the official validation set, ours is a random split of the training set, and the
two are not comparable. The paper must say the transformer baseline is absent
for compute reasons rather than quietly implying comparability.

**Cut and not run:** `baseline_swinunetr`; the 6-row architecture ablation grid;
P2 rung 1 (fixed scalar blend); every second and third seed. Single-seed means
no seed-to-seed std, so **no claim may rest on a margin smaller than the
between-run noise we cannot measure.** State it as a limitation.

### Runs

| Run | Config | Purpose | Est. GPU h |
|---|---|---|---|
| _timing probe_ | `+experiment=neurovision data.overfit_n=50` | **DONE — and it fired the abort trigger.** See the four `probe_neurovision` rows under *Abandoned / failed runs*; v3 is the one that produced the number. Measured at the original 96³: **3.6 s/step = 0.875 h/epoch**, i.e. ~91 h for 100 epochs, needed twice. That is >3x the whole budget, so the schedule below is the re-planned one. | ~0.15 spent |
| `baseline_unet3d` | `+experiment=baseline_unet3d` | Milestone-1 baseline. The number the fusion model must be competitive with. Re-run at the shared 80-epoch / 64³ schedule; the existing 200-epoch row cannot serve, both for the reason in note 1 and because it was trained at 96³. | ~3 |
| `neurovision` | `+experiment=neurovision` | The proposed model, same `_baseline_common` schedule as the baseline. Also serves as P2 rung 3, so the ladder's top rung costs nothing extra. **TRAINING DONE 2026-08-16 — 80/80 epochs over THREE sessions (not the two estimated), 23.1 GPU-h, `GIT_REF=7caacfa`, W&B `cc2l5j1c` (offline, three dirs to sync). `best.pt` epoch 69 at `val/dice_mean` 0.8938 in session 2's output; `last.pt` epoch 79 in session 3's. No row in the Runs table yet — that needs `scripts/evaluate.py` on val and test.** | ~23 est / **23.1 actual** |
| `ablation_content_only_gate` | `+experiment=ablation_content_only_gate` | **P2 rung 2 — the load-bearing experiment.** `model.fusion.use_ambiguity: false`, a one-key diff against `neurovision`, parameter-matched to within 0.018% (6,360 of 34,911,341). Isolates the ambiguity conditioning from the gate's mere existence. If it ties `neurovision` on ECE and HD95, the declared null result fires and the contribution must be rewritten as the smaller claim. Its gradient-checkpointing flags must match `neurovision`'s exactly. | ~23 |

**Re-planned 2026-08-08, against measurement rather than arithmetic.** The
original plan (96³, 100 epochs) priced out at ~197 h against 60. The trigger
written into the probe row above — *"if it lands above ~0.20 h/epoch, cut all
three runs together"* — fired at 0.875, more than 4x the threshold.

The cut is **64³ patches and 80 epochs**, set in `_baseline_common.yaml` so
every arm inherits it. Patch volume falls 3.4x and step time falls with it.
It lives in the shared file deliberately: patch size changes what the network
sees, so an architecture comparison in which one arm saw 96³ and another 64³
would be measuring two things at once.

What was **not** cut, and why. The natural instinct was to cut the fusion —
the novel, expensive-looking part. A per-submodule profile says that would
have been exactly wrong: windowed cross-attention across all four levels is
**1.5% of the forward pass**, while the decoder is **69%** and the stride-1
CNN stem another 15.6%. The cost is ordinary full-resolution 3D convolution,
not the contribution. Cutting fusion would have bought ~1% and damaged the
paper. So the architecture, fusion, ambiguity gate, auxiliary heads and the P2
ablation are all untouched; only data and schedule moved.

Cost to state in the paper: less spatial context per training patch, which may
cost a little whole-tumor Dice, and a schedule at ~56% of nnU-Net's reference
budget rather than ~70%. Both apply identically to every arm.

One intended side effect: at 64³ the two coarsest fusion levels (8³ = 512 and
4³ = 64 tokens) fall under `full_attention_max_tokens: 512` and take the
full-attention path instead of the windowed one. That is the documented rule
working as designed, and at 512² score entries it is free.

Evaluation is priced separately at **~7 h total**: `scripts/evaluate.py` on val
**and** test for all three models with `inference.evaluation.save_logits=true`
(~1 h each — `calibrate.py` refuses to fit and report on the same split, and
temperature cannot be fit from fp16 probabilities), plus MC-dropout on **test
only** at N=10 for `neurovision` and `baseline_unet3d` (~2 h each; MC on val as
well would cost ~4 h and buy nothing, since risk-coverage is a test-split
result).

Everything else — calibration, temperature scaling, boundary stratification,
gate extraction, explainability, figures, tables — is CPU and runs on the Mac
for zero GPU hours. None of it belongs in a Kaggle session.

Total: 0.3 (probes, spent) + 3 + 23.7 + 23.7 + 7 = **~58 h**, leaving ~2 h
against 60 for failed sessions, queue time and resumes. Thinner than the
original plan's ~14 h, which is the price of having discovered the real cost
rather than assumed it. Spend it on failures only; the `ablation_fusion_concat`
row the original plan held in reserve is no longer affordable and is cut.

These are no longer projections. `neurovision` is priced from probe v4's
measured 1.12 s/step at 64³ with checkpointing off: 0.272 h/epoch x 80 =
21.8 h of training, plus 8 validation passes at 0.234 h = 1.9 h. The
`ablation_content_only_gate` row is the same architecture to within 0.018% of
its parameters, so it carries the same cost. `baseline_unet3d` is scaled from
its own measured 0.082 h/epoch at 96³ by the 0.296 volume ratio.

Calendar, which is the real constraint rather than the hour count: the two
fusion runs are ~24 h each against a 12 h session cap, so each needs two
chained sessions. Five long sessions total, at a free tier of ~30 h/week, is
roughly two weeks.

The U-Net estimate is **re-planned against measurement**, not against the
original paper-FLOP calculation. Measured: 16.47 GPU-h for 200 U-Net epochs =
**0.082 h/epoch**, so 100 epochs is ~8 h — the original `~12` was high by about
50%. The `neurovision` and `ablation_content_only_gate` rows are still pure
estimates: 34,911,341 parameters sits between `unet3d` (12.87M) and
SwinUNETR-B (62.19M), but parameter count is a poor predictor of step time for
an architecture with windowed cross-attention at four scales. That is exactly
what the timing probe exists to settle, and it is why the probe is the first
thing that runs.

`python scripts/run_ablation_grid.py` and the 6-row architecture grid it prices
are **not part of this budget** — see the cut list above. The script stays for a
future milestone with more hours.

---

## Abandoned / failed runs

Record these too. A run that OOM'd at epoch 3 or was killed for a config bug is
evidence about the setup, and forgetting it means repeating it.

| Run | Config | GPU h burned | What happened |
|---|---|---|---|
| `probe_neurovision` v1 | `+experiment=neurovision training.epochs=2` | ~0.03 (~2 min) | **CUDA OOM on the first training step**, in a `GroupNorm` forward before any optimizer step ran — `Tried to allocate 432.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 346.81 MiB is free`. Everything upstream was correct (875/187 data dicts mounted, model built at 34,911,341 params, multitask loss, `FRESH:` line, W&B offline), so this is purely a memory result. Two corrections follow. A T4's *usable* capacity is **14.56 GiB, not 16** — ~1.4 GiB goes to context and reserve. And the un-checkpointed model does not fit at the default 4-patch step: it reached 14.22 GiB partway through a single forward, so the true peak is well above 14.56 GiB, against a pre-run estimate of 10–12 GB. The AMP conversion factor for this architecture is therefore ~0.75+ of fp32, not the ~0.55 assumed. Fix: `model.encoder.cnn.use_checkpoint=true`. |
| `probe_neurovision` v2 | v1 + `model.encoder.cnn.use_checkpoint=true` | ~0.03 (~2 min) | Checkpointing cleared the forward; **OOM moved to `backward`**, at step 4. `Tried to allocate 216.00 MiB ... 154.81 MiB is free`, with **888 MiB "reserved by PyTorch but unallocated"** — i.e. ~0.9 GB lost to allocator fragmentation against a 216 MiB shortfall, which is what motivated `expandable_segments:True` in v3 rather than paying decoder recompute for the same memory. It ran far enough to read the bar — `3/875 [00:27<1:45:01, 7.23s/it]` — the first sign the ~15 h estimate was badly wrong. Not trusted on its own: a process allocating at the ceiling thrashes, which inflates step time by an unknown amount. |
| `probe_neurovision` v3 | v2 + `expandable_segments`, `data.overfit_n=50`, 3 epochs | ~0.12 (~7 min) | **COMPLETED — the run that re-planned the project.** Steady state `3.69` then `3.59 s/it` over 50-step epochs; loss fell 1.36 → 1.05 and val Dice reached ~0.65 ET, so the architecture trains correctly and this is purely a cost result. (Those metrics are memorization — `overfit_n` sets val = train — and must never be reported.) Scaled up: **0.875 h/epoch** over 875 steps, validation **3.76 s/case → 0.195 h** over 187 cases, so 100 epochs = **~91 h**, versus `baseline_unet3d`'s measured 0.082 h/epoch. `neurovision` is **10.7x the U-Net per epoch**. Peak VRAM **13.59 GiB allocated / 14.39 reserved of 14.56** — 93% of the card, *with* checkpointing on. This is what forced 64³ / 80 epochs. It also calibrated the AMP factor properly: 12.44 GB of fp32 saved tensors predicted vs 13.59 GiB observed, so the right conversion is **~1.0x plus ~1.6 GB** of weights, Adam and workspace — not 0.5–0.6, and not the 0.75 guessed from v1. |
| `probe_neurovision` v4 | 64³ (from the config), **no** gradient checkpointing, `data.overfit_n=50`, 3 epochs | ~0.10 (~6 min) | **COMPLETED — the run the final plan is priced from, and the first estimate today that landed on target.** Steady state `1.12 s/it`; peak VRAM **6.17 GiB allocated / 7.31 GiB reserved of 14.56**, no OOM. Projection beforehand was 1.0 s/step and ~7.4 GiB, so the recalibrated memory model (≈1.0× the fp32 saved-tensor figure, plus ~1.6 GB for weights, Adam and workspace) is confirmed. **Gradient checkpointing is therefore off permanently** — there is ~7 GiB of headroom at 64³ and the ~20-30% recompute is given back. Scaled up: **0.272 h/epoch** over 875 steps = 21.8 h for 80 epochs. Validation measured **4.5 s/case**, i.e. 0.234 h over 187 cases — MORE per case than the 3.76 s at 96³, because smaller windows means more of them to tile the same volume. That overage is what prompted `val_interval` 5 → 10. |
| `probe_neurovision` v5 | v4's config, re-run against the commit that logs gradient norms | ~0.10 (~6 min) | **COMPLETED — settled `grad_clip_norm`.** Per-epoch, stable across all three: `median 1.067 / 1.164 / 1.151`, `p90 ~2.0`, `max 3.894`, and **clipped on 66-70% of steps** at `grad_clip_norm: 1.0`. So the model was training most of the time with the whole gradient rescaled, at an effective LR the config did not describe. Raised to **5.0** for all three runs, above the measured maximum, so clipping returns to guarding against rare spikes. This mattered twice over: `neurovision` and `ablation_content_only_gate` differ in the ambiguity probes and so in gradient magnitude, and would have clipped at *different* rates — confounding the P2 result with an LR difference — and clipping also silently defeated the deliberate choice not to normalize the multi-task loss weights. Step time 1.05-1.10 s and peak VRAM 6.17 / 7.43 GiB reproduced v4, confirming the cost model is stable. |
| `baseline_unet3d` attempt 1 | `+experiment=baseline_unet3d`, `GIT_REF` pinned to `6ee28a7` | ~0.02 (~1 min) | **Died in the clone cell, before any training.** `FileNotFoundError: /kaggle/working/repo/requirements.txt` — but that was the *symptom*. The cause is that `git clone -b` accepts a **branch or tag name only**; given a commit SHA it fails with `fatal: Remote branch <sha> not found in upstream origin`. This run was the first to use the newly-adopted SHA pinning, and the notebook's clone line had never been exercised with one. Compounding it: `!git clone` is a shell magic whose failure does **not** stop a notebook cell, so execution continued for four more lines and reported a missing file rather than a failed clone. Fixed both — clone and checkout are now separate `subprocess.run(..., check=True)` calls (so a failure raises where it happens), `--depth 1` is dropped (a shallow clone fetches only the branch tip, so a pinned earlier SHA cannot be checked out from it), and the resolved HEAD is printed into the run log. Lesson: the pre-flight verified Hydra composition thoroughly but never *executed* the shell command that had changed. The fix was verified by running the clone and checkout against the real remote. |
| `neurovision` session 1 (attempt 1) | `+experiment=neurovision data.num_workers=2`, `GIT_REF=92f404b` | **10.5 (WASTED)** | **Trained on NaN from ~epoch 10-19 onward.** The session itself behaved perfectly — stopped cleanly on its own `max_hours` prediction at epoch 38 (`elapsed 10.4554h plus a predicted 0.2676h ... would exceed 10.5000h`), wrote `last.pt`, peak VRAM 6.17 GiB of 14.56, ~1.05 s/step as measured. But `train/loss_epoch` was `nan`, every `grad_norm` summary from epoch 20 on was `nan`, and `best.pt` was frozen at **epoch 9** — validation never improved again. Cause: `BranchAmbiguity` computed Bernoulli entropy from probabilities as `-(p*log p + (1-p)*log(1-p))`, guarded by `p.clamp(1e-6, 1 - 1e-6)`. That guard is exactly right in fp32 and a **no-op in fp16**, whose epsilon is ~9.8e-4: `1.0 - 1e-6` rounds to exactly 1.0. Under AMP the probes run in fp16, so once a probe passed p≈0.9995 — which real training reaches in ~10 epochs — `(1-p)` was 0, `log(0)` was -inf, and `0 * -inf` was NaN, which flowed through the gate into every fused feature and the loss with nothing raising. Fixed by computing entropy from LOGITS via softplus (`H = p*softplus(-z) + (1-p)*softplus(z)`), which is finite everywhere: a saturated branch gives `0 * finite = 0`, the correct entropy of a certain prediction. Pinned by `test_branch_ambiguity_entropy_is_finite_under_fp16_saturation`, verified to FAIL against the old implementation. **Why no probe caught it:** v4/v5 ran 3 epochs on 50 cases and never let a probe saturate. Note the ablation (`use_ambiguity: false`) has no `BranchAmbiguity` and would NOT have diverged — so had this shipped, the P2 comparison would have been a NaN run against a healthy one. |
| `probe_saturation` | `+experiment=neurovision data.overfit_n=50 training.epochs=20 training.optimizer.lr=1e-3` | ~0.35 (~20 min) | **Proved the entropy fix at the condition that broke run 1.** 20 epochs over 50 cases at 10x production LR — a harsher regime than real training — with **no NaN in any loss**, grad-norm median flat at ~0.58 throughout (a diverging run's median blows up or goes NaN). The one `max inf` at epoch 19 is a single AMP gradient overflow, which `GradScaler` detects and skips by design, and is expected at 10x LR. Its in-notebook check cell crashed on `KeyError: 'model'` (the payload key is `model_state_dict`) — my error, but it cost nothing: saturation lives in the weights, so the check re-ran locally on the downloaded `last.pt`. Result: **max branch-probe p = 0.999790**, which rounds to exactly 1.0 in fp16 (the representable value below 1.0 is 0.99951) — precisely the state that produced `0 * log(0) = NaN` — with **ambiguity finite at all four fused levels**. This is the probe v4/v5 should have been: built to reach the FAILURE CONDITION, not merely to run. |
| `neurovision` run 2, session 1 | `+experiment=neurovision data.num_workers=2`, `GIT_REF=7caacfa` | 10.34 | **HEALTHY — the entropy fix holds on the real run.** Epochs 0–35, clean `max_hours` stop before epoch 36 (`elapsed 10.3360h plus a predicted 0.2865h ... would exceed 10.5000h`). `train/loss_epoch` **0.5372** (finite), grad-norm median **0.714** and stable across every epoch, **no `loss=nan` anywhere**. `best.pt` at **epoch 29** — the latest validation was the best, where attempt 1 was frozen at epoch 9. Peak VRAM 6.16 GiB of 14.56, no OOM. Grad-norm max spikes 17–57 with ~1.4% of steps clipped: exactly the intended behaviour of `grad_clip_norm: 5.0`, and the justification for having raised it from 1.0 — the median is 0.71, so clipping catches genuine spikes rather than rescaling routine steps. 44 epochs remain, ~12.8 h, so two further sessions. NOT a finished run: no numbers from it may be reported until all 80 epochs complete. |
| `neurovision` run 2, session 2 | `+experiment=neurovision data.num_workers=2`, `GIT_REF=7caacfa` | 10.5 | **HEALTHY.** Epochs 36–72, clean `max_hours` stop before epoch 73 (`elapsed 10.4363h plus a predicted 0.2815h ... would exceed 10.5000h`). `train/loss_epoch` **0.4549** (down from 0.5372), grad-norm median **0.689** stable across every epoch, ~1% of steps clipped, `nonfinite=[]`. `best.pt` advanced to **epoch 69** at `val/dice_mean` **0.8938**; epoch 59 also improved, so validation was still climbing at the end of the session. Peak VRAM 6.17 GiB of 14.56. Two log observations worth recording so they are not re-investigated later: the `RESUME:` line is **absent from the saved log** because Kaggle truncates the head of a long log (it begins mid-epoch-56) — the resume is instead proved by the epoch numbering and by `best_metric` carrying forward; and the 8 `nan` mentions are all MONAI HD95 warnings of the form *"the ground truth of class 0 is all 0, this may result in nan/inf distance"*, i.e. the empty-ET cases `hd95()` deliberately returns NaN for, not divergence. |
| `neurovision` run 2, session 3 | `+experiment=neurovision data.num_workers=2`, `GIT_REF=7caacfa` | 2.3 | **TRAINING COMPLETE — 80/80 epochs — but the kernel is marked ERROR, and the error is in the verification cell, not the run.** Epochs 73–79 trained normally (`train/loss_epoch` **0.4591**, grad-norm median **0.680**, ~1% clipped, peak VRAM 6.17 GiB), `last.pt` written at **epoch 79** with `global_step` **70000** = 80 x 875, which is the arithmetic proof every epoch ran. The final cell then raised `FileNotFoundError: /kaggle/working/checkpoints/best.pt missing`. Cause: the resume cell copied only `last.pt` out of the read-only mount, and `save_checkpoint` writes `best.pt` **only when validation improves**. This session resumed at epoch 72 with the run's best already at epoch 69, validated once at epoch 79, did not beat it, and therefore never created a `best.pt` in its own working directory. Nothing was lost — Kaggle **does** persist a failed version's output, verified by downloading `last.pt` (epoch 79) afterwards, and the run's `best.pt` (epoch 69) is intact in session 2's output. Fixed in `8045f49`: the resume cell now carries `best.pt` forward so a final session's output is self-sufficient, and the verification cell requires only `last.pt`. Lesson, and it is the same shape as the `git clone -b` failure: **a guard written for the common case will eventually meet the legitimate uncommon one, and failing a session whose work is already complete is worse than not checking at all.** |
