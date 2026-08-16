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
