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
| `neurovision` | `+experiment=neurovision` | The proposed model, same `_baseline_common` schedule as the baseline. Also serves as P2 rung 3, so the ladder's top rung costs nothing extra. Two sessions. | ~23 |
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

Total: 0.15 (probes, spent) + 3 + 23 + 23 + 7 = **~56 h**, leaving ~4 h against
60 for failed sessions, queue time and resumes. That margin is thin — thinner
than the original plan's ~14 h — which is the price of having discovered the
real cost. Spend it on failures only; the `ablation_fusion_concat` row the
original plan held in reserve is no longer affordable and is cut.

The `neurovision` and ablation figures are still projections until probe v4
reports: 0.875 h/epoch scaled by the 0.296 volume ratio, minus the gradient
checkpointing v4 tests removing. Four separate pre-run estimates of this
model's cost and memory have now been wrong, so treat ~23 h as provisional
until a clean 64³ step time is on record here.

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
