"""Analysis of the severity-ladder run (MasterPlan W3.2): endpoints and decision rules.

Implements exactly the "Endpoints and decision rules" of
`docs/research_docs/preregistrations/preregistration_severity_ladder.md` (Amendments 1-3 in force:
levels 0, 2, 3, 4 only; `baseline_unet3d` absent from `bias` and `gamma`). Nothing here is fitted
on corrupted data: every conformal threshold, gate cut-point and OOD model is passed in frozen.

## The one idea

Labels never change across the ladder, only the image does. So for one case the false-negative
rate of the conservative mask at a FROZEN threshold tau (the "miss rate", the same loss conformal
v1 defines) is a number we can follow from level 0 (clean) up to level 4 (worst). Every endpoint
below is arithmetic on those per-case miss rates:

- Primary: does the miss rate rise with level? (Page's L trend test + a level-4-minus-level-0 CI.)
- Secondary: does a stronger model have a flatter per-case slope? (OLS slope, paired bootstrap.)
- Descriptive: Dice, realised risk with the v1 verdict, gate refusal / silent failure, OOD flags.

## Why each test is what it is

- **Page's L test** asks "are the level means ordered 0 < 2 < 3 < 4?". It is paired (every case
  is measured at every level, so we compare a case with itself) and uses only the RANK order of
  each case's four numbers, so it does not care about outliers or scale. It is more powerful than
  a plain Friedman test because it knows the expected order. Direction in scipy 1.17
  (`page_trend_test`): columns are taken in order of increasing predicted rank, a large L means
  "increasing", and the p-value is one-sided in that direction (verified in the tests).
- **Holm** corrects for running many tests at once (26 primary cells): the smallest p-value is
  multiplied by the family size, the next by one less, and so on. It guards against calling one of
  many tests "significant" by luck, and needs no independence assumption.
- **OLS slope per case** compresses a case's four points into one number, "how fast does this
  case's miss rate climb per level step". x is the ACTUAL level index (0, 2, 3, 4), not the
  position (0, 1, 2, 3), because level 1 was dropped and the gap between 0 and 2 is two steps.

Randomness: only `np.random.default_rng(seed)` with seed 42 (the project's bootstrap convention
in `neurovision.analysis.statistics`); no global RNG.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import page_trend_test

from neurovision.analysis.error_budget import bootstrap_rate_ci, decide_cases, label_usable
from neurovision.analysis.ood import OODModel, flag, ood_scores
from neurovision.analysis.statistics import holm_bonferroni, paired_bootstrap_ci
from neurovision.inference.gatekeeper import Decision, Thresholds
from neurovision.uncertainty.conformal import CaseLossCurve, _threshold_index, load_curves_npz

__all__ = [
    "BOOTSTRAP_SEED",
    "N_BOOT",
    "CLEAN_KIND",
    "discover_shards",
    "per_case_miss",
    "load_v1_taus",
    "load_v2_taus",
    "miss_matrix",
    "primary_trend_family",
    "kind_verdicts",
    "secondary_slope_family",
    "descriptive_table",
    "gate_decisions",
    "ood_flags",
]

logger = logging.getLogger(__name__)

BOOTSTRAP_SEED = 42  # registered
N_BOOT = 10_000  # registered
CLEAN_KIND = "none"  # the clean level-0 shards live under <model>/none/level0
USABLE_REGIONS = ("WT", "TC")  # registered usable bar: Dice >= 0.7 on WT and TC
USABLE_BAR = 0.7
_LEVEL_DIR = re.compile(r"level(\d+)")
_REQUIRED_FILES = ("per_case.csv", "curves.npz", "ood_features.csv", "shard_meta.json")

Tau = Mapping[tuple[str, float], float]


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------


def discover_shards(roots: Sequence[Path], *, require_done: bool = True) -> pd.DataFrame:
    """Finds every shard directory `<root>/<model>/<kind>/level<k>/` under the given roots.

    Args:
        roots: Directories that contain one sub-directory per model (e.g. `.../severity_ladder`).
        require_done: If True (the registered default) a level directory without a `DONE`
            marker raises. Pass False for a result mirror that does not carry `DONE` (the
            tracked `results/` mirror does not); the four data files are still required.

    Returns:
        DataFrame with columns `model`, `kind`, `level_index` (int), `path` (Path), sorted by
        those. Clean shards have `kind == "none"` and `level_index == 0`.

    Raises:
        FileNotFoundError: a level directory lacks `DONE` (if required) or a data file.
        ValueError: the same (model, kind, level) appears twice.
    """
    rows: list[dict[str, Any]] = []
    for root in roots:
        for model_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            for kind_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                for level_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
                    match = _LEVEL_DIR.fullmatch(level_dir.name)  # exact, never a substring
                    if match is None:
                        continue
                    needed = _REQUIRED_FILES + (("DONE",) if require_done else ())
                    missing = [f for f in needed if not (level_dir / f).is_file()]
                    if missing:
                        raise FileNotFoundError(f"shard {level_dir} is incomplete: no {missing}.")
                    rows.append(
                        {
                            "model": model_dir.name,
                            "kind": kind_dir.name,
                            "level_index": int(match.group(1)),
                            "path": level_dir,
                        }
                    )
    shards = pd.DataFrame(rows, columns=["model", "kind", "level_index", "path"])
    if shards.duplicated(["model", "kind", "level_index"]).any():
        raise ValueError(
            "discover_shards: a (model, kind, level) appears under more than one root."
        )
    shards = shards.sort_values(["model", "kind", "level_index"]).reset_index(drop=True)

    kinds = sorted(set(shards["kind"]) - {CLEAN_KIND})
    for model in sorted(set(shards["model"])):
        have = set(shards.loc[shards["model"] == model, "kind"])
        absent = [k for k in kinds if k not in have]
        if absent:
            # Absent by design (Amendment 3, cut 2), not empty: no row is ever made for them.
            logger.info("discover_shards: model %s has no shards for kinds %s.", model, absent)
    return shards


@lru_cache(maxsize=512)
def _curves_for(path: str, region: str) -> tuple[CaseLossCurve, ...]:
    """Loads one region's curves from a shard's `curves.npz` (cached; the files are immutable).

    `curves.npz` is exactly the layout `load_curves_npz` reads (`thresholds`, `R__case_ids`,
    `R__gt_voxels`, `R__fn_voxels`, `R__mask_voxels`), so no adapter is needed.
    """
    return tuple(load_curves_npz(Path(path) / "curves.npz", (region,))[region])


def per_case_miss(curves: Sequence[CaseLossCurve], tau: float, empty: str) -> pd.Series:
    """Per-case miss rate `|G \\ M(tau)| / |G|` at a frozen grid threshold.

    Same loss and same empty-ground-truth convention as `conformal.realised_risk_with`
    (conformal v1's `realised_risk` is `empty="exclude"`; v1's config predates the option and
    its reader excludes empty-GT cases).

    Args:
        curves: One `CaseLossCurve` per case, one region, one shared grid.
        tau: Threshold; must be on the grid.
        empty: `"exclude"` drops empty-GT cases from the result; `"zero"` keeps them with loss 0.

    Returns:
        float Series indexed by `case_id`, values in [0, 1].

    Raises:
        ValueError: `tau` not on the grid, duplicate case ids, or bad `empty`.
    """
    if empty not in ("exclude", "zero"):
        raise ValueError(f"empty must be 'exclude' or 'zero', got {empty!r}.")
    if len(curves) == 0:
        raise ValueError("curves must not be empty.")
    idx = _threshold_index(np.asarray(curves[0].thresholds, dtype=np.float64), tau)
    ids, values = [], []
    for c in curves:
        if c.empty_gt and empty == "exclude":
            continue
        ids.append(c.case_id)
        values.append(float(c.miss_rate_with(empty)[idx]))
    out = pd.Series(values, index=pd.Index(ids, name="case_id"), dtype=float)
    if not out.index.is_unique:
        raise ValueError("per_case_miss: duplicate case_id in curves.")
    return out


def load_v1_taus(
    fit_json_path: Path, regions: Sequence[str], alphas: Sequence[float]
) -> dict[tuple[str, float], float]:
    """Reads conformal v1 CRC thresholds from a `fit.json` (keys like `WT__alpha_0.1`).

    Args:
        fit_json_path: Path to `fit.json`.
        regions: Regions to read, e.g. `("WT", "TC")`.
        alphas: Risk levels to read, e.g. `(0.05, 0.10, 0.20)`.

    Returns:
        `{(region, alpha): threshold}`. Infeasible fits (no threshold) are skipped with a log.

    Raises:
        KeyError: a requested `region__alpha_x` key is absent.
    """
    fits = json.loads(Path(fit_json_path).read_text(encoding="utf-8"))
    out: dict[tuple[str, float], float] = {}
    for region in regions:
        for alpha in alphas:
            key = f"{region}__alpha_{alpha}"
            if key not in fits:
                raise KeyError(f"{fit_json_path}: no entry {key!r}.")
            entry = fits[key]
            if not entry.get("feasible", False) or entry.get("threshold") is None:
                logger.warning("load_v1_taus: %s infeasible in %s; skipped.", key, fit_json_path)
                continue
            out[(region, float(alpha))] = float(entry["threshold"])
    return out


def load_v2_taus(
    fits_csv_path: Path, model: str, *, grid: str = "extended"
) -> dict[tuple[str, float, str], float]:
    """Reads conformal v2 thresholds (CRC and RCPS, ET included) from a `fits.csv`.

    The CSV has one row per (model, region, alpha, empty, grid, guarantee); `guarantee` is
    `crc` or `rcps` and becomes the `method` in the key. In the shipped file ET was fitted with
    `empty == "zero"` and WT/TC with `"exclude"`; use the `empty` column (not read here) to
    know which convention a region's tau was fitted under.

    Args:
        fits_csv_path: Path to `fits.csv`.
        model: Value of the `model` column to keep.
        grid: Value of the `grid` column to keep (`"extended"` is the v2 grid).

    Returns:
        `{(region, alpha, method): threshold}`. Infeasible rows are skipped with a logged reason.

    Raises:
        ValueError: two rows map to the same key.
    """
    fits = pd.read_csv(fits_csv_path)
    fits = fits[(fits["model"] == model) & (fits["grid"] == grid)]
    out: dict[tuple[str, float, str], float] = {}
    for row in fits.itertuples(index=False):
        key = (str(row.region), float(row.alpha), str(row.guarantee))
        if not bool(row.feasible) or pd.isna(row.threshold):
            logger.warning("load_v2_taus: %s %s infeasible (no threshold); skipped.", model, key)
            continue
        if key in out:
            raise ValueError(f"load_v2_taus: duplicate fit for {model} {key} in {fits_csv_path}.")
        out[key] = float(row.threshold)
    return out


def miss_matrix(
    shards: pd.DataFrame, model: str, kind: str, region: str, tau: float, empty: str
) -> pd.DataFrame:
    """Per-case miss rate at a frozen `tau`, cases x levels.

    Level 0 is the clean shard `<model>/none/level0`, shared by every kind.

    Args:
        shards: Output of `discover_shards`.
        model: Model directory name.
        kind: Corruption kind (not `"none"`).
        region: `"WT"`, `"TC"` or `"ET"`.
        tau: Frozen threshold on the grid.
        empty: `"exclude"` or `"zero"` (see `per_case_miss`).

    Returns:
        DataFrame `(n_cases, n_levels)`, index `case_id`, columns = `level_index` ascending
        (e.g. 0, 2, 3, 4).

    Raises:
        ValueError: the model has no clean shard or no shard of this kind, or a level's case set
            differs from level 0's.
    """
    clean = shards[(shards["model"] == model) & (shards["kind"] == CLEAN_KIND)]
    ladder = shards[(shards["model"] == model) & (shards["kind"] == kind)]
    if clean.empty or ladder.empty:
        raise ValueError(f"miss_matrix: no clean or {kind!r} shards for model {model!r}.")
    levels = pd.concat([clean[clean["level_index"] == 0], ladder]).sort_values("level_index")
    if levels["level_index"].iloc[0] != 0:
        raise ValueError(f"miss_matrix: model {model!r} has no level-0 shard.")

    columns: dict[int, pd.Series] = {}
    for row in levels.itertuples(index=False):
        curves = _curves_for(str(row.path), region)
        columns[int(row.level_index)] = per_case_miss(curves, tau, empty)
    reference = columns[0].index
    for level, col in columns.items():
        if set(col.index) != set(reference):
            raise ValueError(
                f"miss_matrix: level {level} of {model}/{kind} has a different case set than "
                f"level 0 ({len(col)} vs {len(reference)} cases)."
            )
    return pd.DataFrame({lvl: columns[lvl].reindex(reference) for lvl in sorted(columns)})


# ---------------------------------------------------------------------------
# Primary endpoint
# ---------------------------------------------------------------------------


def _ladder_kinds_models(shards: pd.DataFrame) -> list[tuple[str, str]]:
    """Present (kind, model) pairs with at least one corrupted level, sorted."""
    ladder = shards[shards["kind"] != CLEAN_KIND]
    return sorted({(str(k), str(m)) for k, m in zip(ladder["kind"], ladder["model"])})


def primary_trend_family(
    shards: pd.DataFrame,
    taus_by_model: Mapping[str, Tau],
    *,
    regions: Sequence[str] = ("WT", "TC"),
    alpha: float = 0.10,
    empty: str = "exclude",
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """The primary family: is the miss rate at alpha's frozen v1 tau increasing with level?

    One cell per present (kind, model, region). Each cell: Page's L test (one-sided, increasing;
    columns ascending, predicted ranks 1..k) and a paired bootstrap 95% CI of
    mean(level_max - level_0). A cell is GRADED iff Holm-adjusted p < 0.05 (Holm over ALL
    present cells together) AND the CI lower bound > 0 (the rise is not zero). Both are needed:
    the trend test says "ordered", the CI says "by a non-zero amount at the worst level".

    Args:
        shards: Output of `discover_shards`.
        taus_by_model: `{model: {(region, alpha): tau}}` frozen v1 CRC thresholds.
        regions: Regions tested (registered: WT and TC).
        alpha: Risk level whose tau is used (registered: 0.10).
        empty: Empty-GT convention (v1: `"exclude"`).
        n_boot: Bootstrap resamples (registered: 10,000).
        seed: Bootstrap seed (registered: 42).

    Returns:
        DataFrame with columns kind, model, region, n_cases, n_levels, L, p, p_holm, mean_l0,
        mean_l4, diff, ci_lo, ci_hi, verdict (`GRADED` / `NOT GRADED`); `mean_l4` is the highest
        level present. Empty frame if nothing is present.
    """
    rows: list[dict[str, Any]] = []
    for kind, model in _ladder_kinds_models(shards):
        for region in regions:
            tau = taus_by_model[model][(region, float(alpha))]
            mat = miss_matrix(shards, model, kind, region, tau, empty)
            k = mat.shape[1]
            # Predicted ranks 1..k for the ascending-level columns: Page's L is a statistic on
            # rank order, and the registered alternative is "miss rate rises with level".
            res = page_trend_test(mat.to_numpy(), predicted_ranks=np.arange(1, k + 1))
            top, bottom = mat.iloc[:, -1].to_numpy(), mat.iloc[:, 0].to_numpy()
            ci = paired_bootstrap_ci(
                top, bottom, n_boot=n_boot, generator=np.random.default_rng(seed)
            )
            rows.append(
                {
                    "kind": kind,
                    "model": model,
                    "region": region,
                    "n_cases": mat.shape[0],
                    "n_levels": k,
                    "L": float(res.statistic),
                    "p": float(res.pvalue),
                    "mean_l0": float(bottom.mean()),
                    "mean_l4": float(top.mean()),
                    "diff": ci.point,
                    "ci_lo": ci.lo,
                    "ci_hi": ci.hi,
                }
            )
    columns = [
        "kind", "model", "region", "n_cases", "n_levels", "L", "p", "p_holm",
        "mean_l0", "mean_l4", "diff", "ci_lo", "ci_hi", "verdict",
    ]  # fmt: skip
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out["p_holm"], _ = holm_bonferroni(out["p"].to_numpy())
    out["verdict"] = np.where((out["p_holm"] < 0.05) & (out["ci_lo"] > 0), "GRADED", "NOT GRADED")
    return out[columns]


def kind_verdicts(primary: pd.DataFrame, *, n_models_required: int = 3) -> pd.DataFrame:
    """Per-kind verdict: GRADED if, in some region, every measured model is GRADED.

    The protocol says "GRADED for all three models in at least one region". After Amendment 3
    (cut 2) `bias` and `gamma` were only run on two models, so "all models present" is not "all
    three". Such a kind gets `GRADED_ON_MEASURED_MODELS_ONLY` and is never silently promoted to
    `GRADED`: a reader must see that the third model was not measured.

    Args:
        primary: Output of `primary_trend_family`.
        n_models_required: Models the protocol requires per kind (3).

    Returns:
        DataFrame with columns kind, models_measured (int), graded_regions (comma-joined),
        verdict in {GRADED, NOT GRADED, GRADED_ON_MEASURED_MODELS_ONLY}.
    """
    rows = []
    for kind, sub in primary.groupby("kind", sort=True):
        graded_regions = [
            str(region)
            for region, cell in sub.groupby("region", sort=True)
            if (cell["verdict"] == "GRADED").all()
        ]
        measured = int(sub["model"].nunique())
        if not graded_regions:
            verdict = "NOT GRADED"
        elif measured < n_models_required:
            verdict = "GRADED_ON_MEASURED_MODELS_ONLY"
        else:
            verdict = "GRADED"
        rows.append(
            {
                "kind": kind,
                "models_measured": measured,
                "graded_regions": ",".join(graded_regions),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows, columns=["kind", "models_measured", "graded_regions", "verdict"])


# ---------------------------------------------------------------------------
# Secondary endpoint
# ---------------------------------------------------------------------------


def _ols_slopes(mat: pd.DataFrame) -> pd.Series:
    """Per-case OLS slope of miss rate on the ACTUAL level index (x = 0, 2, 3, 4).

    Closed form `sum((x - x_mean) * y) / sum((x - x_mean)^2)` per row. Using positions
    0..3 instead would understate the slope across the missing level 1.
    """
    x = np.asarray(mat.columns, dtype=float)
    xc = x - x.mean()
    return pd.Series(mat.to_numpy() @ xc / float(xc @ xc), index=mat.index)


def _bootstrap_two_sided_p(diff: np.ndarray, n_boot: int, seed: int) -> float:
    """Two-sided bootstrap p for mean(diff) = 0, from the same resamples `paired_bootstrap_ci` uses.

    `paired_bootstrap_ci` returns no p. It draws `generator.integers(0, n, size=(n_boot, n))`
    first, so a fresh generator with the same seed reproduces its resamples exactly (checked in
    the tests by recomputing its CI). p = min(1, 2 * min((#b<=0 + 1)/(B+1), (#b>=0 + 1)/(B+1))).
    """
    idx = np.random.default_rng(seed).integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    below = (np.sum(boot <= 0) + 1) / (n_boot + 1)
    above = (np.sum(boot >= 0) + 1) / (n_boot + 1)
    return float(min(1.0, 2.0 * min(below, above)))


def secondary_slope_family(
    shards: pd.DataFrame,
    taus_by_model: Mapping[str, Tau],
    *,
    stronger_weaker: Sequence[tuple[str, str]] = (
        ("nnunet_gatea", "neurovision"),
        ("neurovision", "baseline_unet3d"),
    ),
    regions: Sequence[str] = ("WT", "TC"),
    alpha: float = 0.10,
    empty: str = "exclude",
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Does the stronger model have a flatter miss-rate slope (does it "shrink the damage")?

    Per case, the OLS slope of miss rate on level index; then the paired per-case difference
    `slope(stronger) - slope(weaker)`. Negative means the stronger model's miss rate climbs
    more slowly. A pair is evaluated only for kinds where both models have shards (the
    baseline has no `bias` / `gamma`). One Holm family over all present pairs.

    The p-value is the two-sided bootstrap p computed from the same resamples as the CI
    (`paired_bootstrap_ci` returns none): `p = min(1, 2*min((#b<=0+1)/(B+1), (#b>=0+1)/(B+1)))`.

    Args:
        shards: Output of `discover_shards`.
        taus_by_model: `{model: {(region, alpha): tau}}` frozen v1 CRC thresholds.
        stronger_weaker: (stronger, weaker) model pairs; the registered ones are the default.
        regions: Regions tested.
        alpha: Risk level whose tau is used (0.10).
        empty: Empty-GT convention (v1: `"exclude"`).
        n_boot: Bootstrap resamples (10,000).
        seed: Bootstrap seed (42).

    Returns:
        DataFrame with columns kind, region, stronger, weaker, n_cases, slope_stronger,
        slope_weaker, diff, ci_lo, ci_hi, p, p_holm, verdict (`FLATTER` iff ci_hi < 0 and
        p_holm < 0.05, else `NOT FLATTER`).
    """
    present = {(k, m) for k, m in _ladder_kinds_models(shards)}
    kinds = sorted({k for k, _ in present})
    rows: list[dict[str, Any]] = []
    for kind in kinds:
        for stronger, weaker in stronger_weaker:
            if (kind, stronger) not in present or (kind, weaker) not in present:
                continue
            for region in regions:
                s_a = _ols_slopes(
                    miss_matrix(
                        shards,
                        stronger,
                        kind,
                        region,
                        taus_by_model[stronger][(region, alpha)],
                        empty,
                    )
                )
                s_b = _ols_slopes(
                    miss_matrix(
                        shards, weaker, kind, region, taus_by_model[weaker][(region, alpha)], empty
                    )
                )
                joined = pd.concat([s_a, s_b], axis=1, join="inner").dropna()
                a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
                ci = paired_bootstrap_ci(a, b, n_boot=n_boot, generator=np.random.default_rng(seed))
                rows.append(
                    {
                        "kind": kind,
                        "region": region,
                        "stronger": stronger,
                        "weaker": weaker,
                        "n_cases": int(a.size),
                        "slope_stronger": float(a.mean()),
                        "slope_weaker": float(b.mean()),
                        "diff": ci.point,
                        "ci_lo": ci.lo,
                        "ci_hi": ci.hi,
                        "p": _bootstrap_two_sided_p(a - b, n_boot, seed),
                    }
                )
    columns = [
        "kind", "region", "stronger", "weaker", "n_cases", "slope_stronger", "slope_weaker",
        "diff", "ci_lo", "ci_hi", "p", "p_holm", "verdict",
    ]  # fmt: skip
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out["p_holm"], _ = holm_bonferroni(out["p"].to_numpy())
    out["verdict"] = np.where((out["ci_hi"] < 0) & (out["p_holm"] < 0.05), "FLATTER", "NOT FLATTER")
    return out[columns]


# ---------------------------------------------------------------------------
# Gate and OOD (frozen inputs only)
# ---------------------------------------------------------------------------


def gate_decisions(
    cfg: Any,
    per_case: pd.DataFrame,
    thresholds: Thresholds,
    regions: Sequence[str],
    stand_in_age_years: float,
) -> pd.DataFrame:
    """Runs the frozen gate on one shard's `per_case.csv` and joins the usable label.

    Thin wrapper: builds the `signals` frame (`case_id`, `predicted_dice_<R>`,
    `conformal_band_<R>`, `patient_age_years` = the curated-cohort stand-in), calls
    `error_budget.decide_cases` with the curated input-QC stand-in and the thresholds passed in
    (nothing is fitted), and adds `usable` (`label_usable`, Dice >= 0.7 on WT and TC).

    Args:
        cfg: Root config as `decide_cases` expects (`cfg.clinical.gatekeeper...`).
        per_case: One shard's `per_case.csv` (single model), one row per case.
        thresholds: Frozen gate `Thresholds`.
        regions: Regions whose predicted Dice / band the gate reads, e.g. `("WT", "TC")`.
        stand_in_age_years: Age given to every case (curated cohort has no real age).

    Returns:
        `decide_cases`' columns (`case_id`, `decision`, `accepted`, `refusing_signals`,
        `cautioning_signals`) plus `usable` (bool), `silent_failure` (accepted and not usable),
        and `gate_label`: `"deployed"` for `neurovision`, else `"transferred, not refitted"`.

    Raises:
        ValueError: `per_case` does not hold exactly one model.
    """
    models = per_case["model"].unique()
    if len(models) != 1:
        raise ValueError(f"gate_decisions: expected one model in per_case, got {list(models)}.")
    signals = pd.DataFrame({"case_id": per_case["case_id"].to_numpy()})
    for region in regions:
        signals[f"predicted_dice_{region}"] = per_case[f"predicted_dice_{region}"].to_numpy()
        signals[f"conformal_band_{region}"] = per_case[f"conformal_band_{region}"].to_numpy()
    signals["patient_age_years"] = float(stand_in_age_years)

    decisions = decide_cases(cfg, signals, thresholds, regions=tuple(regions))
    usable = label_usable(per_case.reset_index(drop=True), regions=USABLE_REGIONS, bar=USABLE_BAR)
    decisions["usable"] = usable.to_numpy(dtype=bool)
    decisions["silent_failure"] = decisions["accepted"].astype(bool) & ~decisions["usable"]
    decisions["gate_label"] = (
        "deployed" if models[0] == "neurovision" else "transferred, not refitted"
    )
    return decisions


def ood_flags(
    ood_model: OODModel, cuts: tuple[float, float], features: pd.DataFrame
) -> pd.DataFrame:
    """OOD score and flag per case with FROZEN cuts (fitted on clean val; never refitted here).

    Args:
        ood_model: Fitted `OODModel`.
        cuts: `(caution_cut, refuse_cut)`.
        features: One row per case with the model's feature columns (and optionally `case_id`).

    Returns:
        DataFrame with `case_id` (if present), `ood_score`, `ood_flag` (a `Decision` value) and
        `ood_not_ok` (bool, flag is not `proceed`).
    """
    scores = ood_scores(ood_model, features)
    flags = flag(scores, cuts[0], cuts[1])
    out = pd.DataFrame({"ood_score": scores, "ood_flag": flags})
    if "case_id" in features.columns:
        out.insert(0, "case_id", features["case_id"].to_numpy())
    out["ood_not_ok"] = out["ood_flag"] != Decision.PROCEED.value
    return out


# ---------------------------------------------------------------------------
# Descriptive table
# ---------------------------------------------------------------------------


def _risk_verdict(ci_lo: float, ci_hi: float, alpha: float) -> str:
    """v1's rule (`scripts/conformal_v2.py::cell_verdict`, shifted-cohort branch).

    VIOLATED if the CI lower bound is above alpha; "holds" if the upper bound is at or below
    it; otherwise "inconclusive" (also when the CI is undefined).
    """
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "inconclusive"
    if ci_lo > alpha:
        return "VIOLATED"
    if ci_hi <= alpha:
        return "holds"
    return "inconclusive"


def descriptive_table(
    shards: pd.DataFrame,
    taus_by_model: Mapping[str, Mapping[tuple[str, float, str], float]],
    *,
    decisions: Mapping[tuple[str, str, int], pd.DataFrame] | None = None,
    ood: Mapping[tuple[str, str, int], pd.DataFrame] | None = None,
    empty_by_region: Mapping[str, str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Per (model, kind, level) description. No test, no Holm.

    Level 0 appears once per model under `kind == "none"`.

    Long-format schema, one row per measured quantity:
        model, kind, level_index, quantity, region, alpha, method, n, value, ci_lo, ci_hi, verdict

    `quantity` values:
        `dice`: mean Dice of `region` (NaN Dice skipped); n = cases used; no CI.
        `realised_risk`: mean per-case miss at the frozen tau for (`region`, `alpha`, `method`);
            95% percentile bootstrap CI (`bootstrap_rate_ci`, case-resampled, seed 42); `verdict`
            is v1's VIOLATED / holds / inconclusive; n excludes empty-GT cases under "exclude".
        `refusal_rate`, `silent_failure_rate`, `usable_rate`: from `decisions` (denominator is
            all cases; silent failure = accepted and not usable); mean with bootstrap CI.
        `ood_flag_rate`: fraction of cases whose flag is not proceed, from `ood`.
    Columns not applicable to a quantity are NaN / "".

    Args:
        shards: Output of `discover_shards`.
        taus_by_model: `{model: {(region, alpha, method): tau}}`, e.g. v1 CRC as method `"crc"`
            plus v2 CRC/RCPS. Frozen.
        decisions: `{(model, kind, level): gate_decisions(...)}`; gate rows omitted if None.
        ood: `{(model, kind, level): ood_flags(...)}`; OOD row omitted if None.
        empty_by_region: Empty-GT convention per region (default `"exclude"`; v2's ET is `"zero"`).
        n_boot: Bootstrap resamples (10,000).
        seed: Seed for every bootstrap (42; a fresh generator per cell so order never matters).
        progress: Optional callback receiving one message per shard.

    Returns:
        The long-format DataFrame described above.
    """
    empty_by_region = dict(empty_by_region or {})
    rows: list[dict[str, Any]] = []

    def add(base: dict[str, Any], quantity: str, **fields: Any) -> None:
        row = {
            "quantity": quantity, "region": "", "alpha": np.nan, "method": "", "n": 0,
            "value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "verdict": "",
        }  # fmt: skip
        row.update(base)
        row.update(fields)
        rows.append(row)

    def rate(values: np.ndarray) -> tuple[float, float, float]:
        return bootstrap_rate_ci(
            values, n_boot=n_boot, ci=0.95, generator=np.random.default_rng(seed)
        )

    for shard in shards.itertuples(index=False):
        model, kind, level = str(shard.model), str(shard.kind), int(shard.level_index)
        base = {"model": model, "kind": kind, "level_index": level}
        if progress is not None:
            progress(f"{model}/{kind}/level{level}")
        per_case = pd.read_csv(Path(shard.path) / "per_case.csv")
        for region in ("ET", "TC", "WT"):
            dice = per_case[f"dice_{region}"].dropna()
            add(base, "dice", region=region, n=int(dice.size), value=float(dice.mean()))
        for (region, alpha, method), tau in sorted(taus_by_model.get(model, {}).items()):
            curves = _curves_for(str(shard.path), region)
            miss = per_case_miss(curves, tau, empty_by_region.get(region, "exclude"))
            point, lo, hi = rate(miss.to_numpy())
            add(
                base, "realised_risk", region=region, alpha=float(alpha), method=method,
                n=int(miss.size), value=point, ci_lo=lo, ci_hi=hi,
                verdict=_risk_verdict(lo, hi, float(alpha)),
            )  # fmt: skip
        key = (model, kind, level)
        if decisions is not None and key in decisions:
            dec = decisions[key]
            refused = (~dec["accepted"].astype(bool)).to_numpy(dtype=float)
            for name, values in (
                ("refusal_rate", refused),
                ("silent_failure_rate", dec["silent_failure"].to_numpy(dtype=float)),
                ("usable_rate", dec["usable"].to_numpy(dtype=float)),
            ):
                point, lo, hi = rate(values)
                add(base, name, n=int(values.size), value=point, ci_lo=lo, ci_hi=hi)
        if ood is not None and key in ood:
            flagged = ood[key]["ood_not_ok"].to_numpy(dtype=float)
            point, lo, hi = rate(flagged)
            add(base, "ood_flag_rate", n=int(flagged.size), value=point, ci_lo=lo, ci_hi=hi)
    return pd.DataFrame(
        rows,
        columns=[
            "model", "kind", "level_index", "quantity", "region", "alpha", "method", "n",
            "value", "ci_lo", "ci_hi", "verdict",
        ],
    )  # fmt: skip
