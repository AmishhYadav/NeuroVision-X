# Numbers in report.md that are not in number_registry.yaml

Companion to `number_registry.yaml` (MasterPlan W1.2). Each row is a result number that the audit script cannot recompute from a file under `results/`. Reasons: **no file** (no tracked source found), **outputs-only** (source exists only under `outputs/`, so it may not be cited until mirrored), **derived** (arithmetic on registered numbers or a count that needs a filter the registry cannot express), **prose** (rounded wording), **ledger** (lives in a Markdown ledger, not a result file), **pending D2** (placeholder until note 63).

| Section | Printed text | Why | Claim |
|---|---|---|---|
| Abstract, 6.1 | Holm-adjusted p = 1.4 x 10^-21 | in file since 2026-11-24 (`results/compare_family/c1_headline/family.csv`, `p_holm_family` 1.44e-21) but the audit script compares fixed decimals, not scientific notation | C1 |
| 6.1, Table 8 | TC +0.0103 | now in `results/compare_family/c1_headline/family.csv` (2026-11-24); not yet registered | C1 |
| 6.1 | "about six times", "about four times" the retraining shift; "about 40%" | derived (ratios of registered numbers; 40% = empty_contribution 0.0106 / 0.0267) | C1 |
| 6.1 | "two of the five test cases with no enhancing tumour" | derived (count over `results/ref_et/empty_et_cases.csv`, needs a conditional) | C1 |
| 6.2, Table 8, Table 10 | ~80% architecture; 0.79 / 0.80 share | derived (ratio of `capacity_r2` and `ref_et` differences to +0.0267) | C2 |
| 4.1, 4.2, 6.2, Table 4 | 34.9 M / 34,911,341 parameters; 18.9 M, 12.9 M, 2.0 M, 1.0 M components; 6,360 parameters (0.018%); 34.83 M vs 34.91 M | other: parameter counts come from model code, no results file | C2 / C3 |
| 6.3 | without mirroring nnU-Net still leads by 0.0113 | derived: `results/score_nnunet/gatea_tta_off/summary.csv` dice_ET 0.8822 minus NeuroVision-X 0.8709; two-file difference | C29 |
| 6.3, 5.6, Table 10 | about 2.7 times the GPU time; about 59 vs 22 T4-hours | ledger: `docs/research_docs/gpu_ledger.md` | C29 |
| 6.4 | ET +0.0508, TC +0.0371 lesion-wise gain over matched U-Net | no file: the lesion-wise NeuroVision-X vs U-Net family is not in `results/compare_family/` (note 41) | C10 |
| 6.4 | 0.32 vs 0.47 false-positive ET lesions per case | no file in `results/` (note 41) | C11 |
| 6.4 | voxel Dice overstates by 0.10 to 0.32 | derived (range of lesion-wise minus voxel Dice over runs, note 41) | C12 |
| 6.7 | "7 of 12", "5 of 12", "9 of 12" shifted cells violated / lower risk; "six cells" | derived counts over `realised_risk.csv` against alpha (no threshold-count facility) | C13 / C14 / C30 |
| 6.7 | 0.64 to 0.96 for "both NeuroVision-X and the U-Net" | registered for the U-Net only (0.64 = U-Net WT alpha 0.05, 0.96 = U-Net TC alpha 0.10); NeuroVision-X alone spans 0.70 to 0.91, so the range is the union of the two models | C13 |
| 6.7, 6.8 | 48.6% (nnU-Net) and 35.6% (NeuroVision-X) of PED tumour core missed at the lowest threshold | outputs-only: R(tau_min) comes from the per-case curve arrays under `outputs/conformal/*/eval_ped*/` (only `curves_manifest.json` is mirrored) | C14 / C27 / C30 |
| 6.7 | "At alpha = 0.20 the threshold sits at the top of the grid" | prose; `fit.json` holds threshold 0.95 but the grid maximum is not stored | C13 |
| 6.8 | 10 to 40 local labelled cases | prose summary of `results/local_recalibration/floors.csv` (min_feasible_k by cell); not a single value | C24 |
| 6.9 | 2 of 60 refusals on SSA | derived (60 minus 58 accepted; needs a subtraction) | C21 |
| 6.9 | denominator 26 in "23 of 26 refusals"; recall 32%; precision 0.88 | derived (23 correct + 3 over-refusals; recall 23/72) | C21 |
| 6.9, Table 8 | QC model: 1 of 5 cells, loses in one, borderline in a third | derived counts over `results/qc_validation/cells.csv` | C18 |
| 6.9 | QC model more optimistic in all six external cells | derived (sign of `bias` column vs in-distribution) | C19 |
| 6.10 | 22 accepted, 14 unusable (silent 0.35) | partly registered (22, 36%, 40%); "14 usable" count is derived | C25 |
| 4.3, 4.4, 5.6, Table 10 | 10.7x slower per epoch; best epoch 69 of 80; 23.1 GPU-h; 11.9 s per case, 35 min 43 s; ~1.4 min per case on CPU; ~210 GPU-h over 18 runs; ~30 GPU-h per week | ledger: `gpu_ledger.md` / `experiments.md` notes (no result file) | none |
| 5.2, 5.3, 5.5 | 12 GB peak memory; 122 structures; 56 labelled pairs; ~70 s test suite | other: engineering facts, not results | none |
| 1.6 | about 2,670 tests; fourteen pre-registration documents | other: repository counts | none |
| 6.6 | MC-dropout equivalence margin 0.03 AUROC; 10 samples | design constant (registered rule), not a result | C8 |
| 6.1, 6.2, Table 8 | "+0.0022 seed shift vs +0.0055 width term" comparisons in prose | derived/prose | C2 |
| 6.8, Table 8, Table 10, 10.2 | pooled multi-cohort (D2) training results, "evaluation pending" | pending D2 (note 63, C31); nothing numeric printed yet except GPU-h (~25, ledger) | C31 |
| Table 8 | Front end "costs <= 0.05 WT Dice"; "~80% architecture"; "Per voxel yes, per case no" | design bar / prose | C25 / C2 / C4 |
