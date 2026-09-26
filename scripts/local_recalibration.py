"""Hydra entry point for the local-recalibration (Mondrian) counterfactual.

`src/neurovision/analysis/local_recalibration.py` implements the pure statistics
(`draw_split`, `evaluate_split`, `run_local_recalibration`, `min_feasible_k`,
`summarise`) and is fully tested on its own. This script is the driver that runs it
over this project's ALREADY-SAVED conformal loss curves, per
`docs/research/preregistration_conformal.md`'s "Amendment 1" (read that section
first -- it is the contract this script implements, and it must never be edited
from here). No inference, no GPU: every `curves.npz` this script reads was already
written by `scripts/conformal.py`.

## COUNTERFACTUAL, never external validation

Every number this script produces answers "if a new site labelled `k` of its own
cases, would recalibrating the conformal threshold on exactly those restore the
guarantee on the REST of that site's cases?" It never answers "how well does the
deployed, frozen-threshold pipeline do on unseen data" -- that is the separate,
already-resolved B2 result in `scripts/conformal.py` / the pre-registration's
"Result" section. Losing that distinction is invisible in a table of numbers and
entirely visible in a caption, so every log header this script prints carries the
word COUNTERFACTUAL, and `run_meta.json` carries a fixed `status_label` string
saying so explicitly.

## Why floors are computed and logged BEFORE any split is drawn

`local_recalibration.min_feasible_k` answers a cheap, deterministic, non-random
question -- "could ANY calibration size `k` possibly work here?" -- from the
cohort's own smallest-achievable risk. It costs no random draws and is a lower
bound on what the (expensive, 1000-splits-per-cell) sweep below could ever show, so
it is computed and written to `floors.csv` first: a `k` swept below its floor is a
predictable "won't restore" result, not a surprising one.

## Pairing across models

For every cohort, both models must see the IDENTICAL sequence of random splits, so
that a difference between their summary rows is a statement about the model, not
about which random calibration/held-out partition it happened to be scored on. This
script gets that "for free", not by sharing one `Generator` across models, but
because it constructs a FRESH `np.random.default_rng(seed)` (`seed` read from
`analysis.local_recalibration.seed`) independently for every single `(model,
cohort)` pair. Two fresh generators seeded identically, drawing from ADDITIONALLY
THE SAME sequence of `(region, k)` iterations (`run_local_recalibration` iterates
regions and `ks` in a fixed order) over cohorts of the same size, therefore draw the
exact same sequence of splits -- with no shared mutable generator object between the
two model runs. This depends on both models listing the same cases, in the same
order, per region for a shared cohort; `_validate_loaded` checks that explicitly
(and raises) rather than letting a silent reorder produce same-shaped but silently
unpaired output.

## Deduplicating a literal "half" that coincides with an explicit k

`analysis.local_recalibration.ks` can contain the literal string `"half"`
alongside explicit integers (e.g. SSA has n=60, so `"half"` resolves to k=30,
which is ALSO an explicit entry in the pre-registered sweep). The pre-registration
says such a coincidence is "reported once, labelled both ways" -- so before calling
the library, this script drops a cohort's `"half"` entry whenever it numerically
matches an explicit `k` already in the sweep for that cohort. This is what keeps
that cohort's k=30 splits from being drawn TWICE (once under each label) and
double-counted in `summary.csv`'s `n_splits`. The `is_half` column on every summary
row (`k == n // 2` for that cohort) is what "labelled both ways" means in the
output -- it is set correctly regardless of which of the two config entries
survived the dedup.

## The control-cohort falsifier

Prediction 1 of Amendment 1: the in-distribution BraTS TEST cohort, at k=half,
must be RESTORED for every (model, region, alpha) -- within-cohort splits ARE
exchangeable, so conformal risk control's theorem applies exactly. If any such
cell instead comes back NOT_RESTORED, that means this DRIVER (or the library
underneath it) has a bug, not that the theorem failed -- so this script logs an
ERROR naming the failure and sets `run_meta.json`'s `"control_failed": true`
rather than raising: the pre-registration's falsifier list says "nothing
downstream is valid" when this trips, and a human needs to see every other file
this script wrote anyway to debug it.
"""

from __future__ import annotations

import logging
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis import local_recalibration as lr
from neurovision.uncertainty.conformal import CaseLossCurve, load_curves_npz
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Recorded verbatim into every run_meta.json, and echoed in every log header --
# see the module docstring's "COUNTERFACTUAL, never external validation" section.
_STATUS_LABEL = (
    "COUNTERFACTUAL -- what a new site would need to restore the bound; never external "
    "validation"
)

# The in-distribution cohort the control-cohort falsifier is defined on (Amendment
# 1, prediction 1). Matches the "test" key under analysis.local_recalibration.models.*.cohorts.
_CONTROL_COHORT = "test"


# ---------------------------------------------------------------------------
# classify_restoration: the one piece of decision logic this driver owns
# ---------------------------------------------------------------------------


def classify_restoration(mean_risk: float, mc_se: float, alpha: float, n_feasible: int) -> str:
    """Classifies one (model, cohort, region, alpha, k) cell against Amendment 1's decision rule.

    Args:
        mean_risk: `mean_realised_risk` -- the mean held-out risk over feasible
            splits (`NaN` if none had a defined realised risk).
        mc_se: The Monte Carlo standard error of that mean (`NaN` if fewer than
            two feasible splits had a defined realised risk).
        alpha: The target risk level this cell was fitted at.
        n_feasible: Number of feasible splits in this cell (`draw_split` /
            `fit_threshold` found a threshold on the calibration half).

    Returns:
        `"INFEASIBLE"` if no split was feasible. `"BORDERLINE"` if `n_feasible < 2` --
        `mc_se` is undefined (or estimated from a single draw) at that point, and a lone
        Monte Carlo draw must never be read as `"RESTORED"` or `"NOT_RESTORED"` regardless
        of which side of `alpha` it happens to land on. Otherwise: `"RESTORED"` if
        `mean_risk <= alpha` (equality counts as restored -- the guarantee is `<= alpha`,
        not `< alpha`); `"NOT_RESTORED"` if `mean_risk` exceeds `alpha` by more than twice
        its Monte Carlo standard error; `"BORDERLINE"` otherwise. A `NaN` `mean_risk` never
        satisfies a `<=`/`>` comparison, so such a cell also falls through to
        `"BORDERLINE"` rather than being misclassified as either extreme.
    """
    if n_feasible == 0:
        return "INFEASIBLE"
    if n_feasible < 2:
        return "BORDERLINE"
    if mean_risk <= alpha:
        return "RESTORED"
    if mean_risk - alpha > 2.0 * mc_se:
        return "NOT_RESTORED"
    return "BORDERLINE"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    """Returns the current commit SHA, or `None` if it cannot be determined.

    Never raises -- `git` missing, this not being a repository, or any other
    failure all fall back to `None` rather than aborting a CPU-only analysis run
    over a provenance nicety.
    """
    repo_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except (OSError, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


# ---------------------------------------------------------------------------
# Loading curves (step 1)
# ---------------------------------------------------------------------------


def _load_all_curves(
    models_cfg: Mapping[str, Mapping[str, Any]], regions: Sequence[str]
) -> dict[tuple[str, str], dict[str, list[CaseLossCurve]]]:
    """Loads every configured (model, cohort)'s `curves.npz`, skipping what is missing.

    Args:
        models_cfg: `analysis.local_recalibration.models`, already converted to a
            plain dict: `{model_name: {"conformal_dir": str, "cohorts": {cohort_name:
            subdir}}}`.
        regions: Region names to read back from each `curves.npz`.

    Returns:
        `{(model_name, cohort_name): {region: [CaseLossCurve, ...]}}`, one entry for
        every (model, cohort) whose `curves.npz` exists on disk. A missing file logs
        a WARNING naming the exact path and skips only that (model, cohort) pair --
        it never aborts the whole run, since a Kaggle run's logits for one cohort
        can legitimately still be in flight.
    """
    loaded: dict[tuple[str, str], dict[str, list[CaseLossCurve]]] = {}
    for model_name, model_cfg in models_cfg.items():
        conformal_dir = Path(str(model_cfg["conformal_dir"]))
        for cohort_name, subdir in model_cfg["cohorts"].items():
            path = conformal_dir / str(subdir) / "curves.npz"
            if not path.is_file():
                logger.warning(
                    "local_recalibration: COUNTERFACTUAL -- no curves.npz at %s for "
                    "model=%s cohort=%s; skipping this (model, cohort) pair only.",
                    path,
                    model_name,
                    cohort_name,
                )
                continue
            loaded[(model_name, cohort_name)] = load_curves_npz(path, regions)
    return loaded


def _validate_loaded(
    loaded: Mapping[tuple[str, str], Mapping[str, Sequence[CaseLossCurve]]],
    regions: Sequence[str],
) -> None:
    """Two structural checks, run once, before any floor is computed or any split is drawn.

    1. **Every region within one (model, cohort) must share one case count.** `"half"`
       (`_resolve_ks_for_cohort`) and the cohort's own `n` are only well-defined if every
       region agrees on how many cases there are.
    2. **Every model loaded for the same cohort must list the SAME case_ids, in the SAME
       order, for each region.** The cross-model split pairing this driver relies on (module
       docstring, "Pairing across models") assumes `draw_split`'s index-based partition
       (`rng.permutation(n)` indexed into the curves' own `case_id` order) picks out the
       SAME case for both models at a given draw -- a silent reorder would still produce
       same-shaped output with no error, just silently UNpaired results, so this is checked
       explicitly rather than assumed.

    Args:
        loaded: `_load_all_curves`'s return value.
        regions: Region names to check.

    Raises:
        ValueError: Either check fails. The message names the model(s), cohort and region
            involved.
    """
    for (model_name, cohort_name), curves_by_region in loaded.items():
        counts = {region: len(curves_by_region[region]) for region in regions}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"local_recalibration: model={model_name!r} cohort={cohort_name!r} has a "
                f"different case count per region: {counts}. 'half' and the (model, cohort) "
                "pairing both assume every region of one cohort shares a single case count."
            )

    by_cohort: dict[str, list[str]] = {}
    for model_name, cohort_name in loaded:
        by_cohort.setdefault(cohort_name, []).append(model_name)

    for cohort_name, model_names in by_cohort.items():
        if len(model_names) < 2:
            continue
        reference_model = model_names[0]
        for region in regions:
            reference_ids = [c.case_id for c in loaded[(reference_model, cohort_name)][region]]
            for model_name in model_names[1:]:
                ids = [c.case_id for c in loaded[(model_name, cohort_name)][region]]
                if ids != reference_ids:
                    raise ValueError(
                        f"local_recalibration: case_id order mismatch for "
                        f"cohort={cohort_name!r} region={region!r} between "
                        f"model={reference_model!r} and model={model_name!r} -- the "
                        "cross-model split pairing this driver relies on requires every "
                        "model's curves.npz to list the SAME cases in the SAME order for a "
                        "shared cohort."
                    )


# ---------------------------------------------------------------------------
# Step 2: feasibility floors, before any split is drawn
# ---------------------------------------------------------------------------


def _structural_floor(alpha: float) -> int:
    """The smallest `k` for which `(k * 0 + 1) / (k + 1) <= alpha` can ever hold -- `ceil(1/alpha
    - 1)` -- independent of any data. See `local_recalibration.min_feasible_k`'s docstring for the
    same derivation applied to a cohort's actual smallest achievable risk rather than 0.
    """
    # Nudge below before ceiling, matching min_feasible_k's own convention: an
    # exact-integer ratio can land a hair above it purely from floating-point
    # rounding, which would otherwise bump an exact-equality case up by one.
    return max(0, math.ceil(1.0 / alpha - 1.0 - 1e-9))


def _build_floors(
    loaded: Mapping[tuple[str, str], Mapping[str, Sequence[CaseLossCurve]]],
    regions: Sequence[str],
    alphas: Sequence[float],
) -> pd.DataFrame:
    """Computes `min_feasible_k` for every loaded (model, cohort, region, alpha).

    Args:
        loaded: `_load_all_curves`'s return value.
        regions: Region names, in report order.
        alphas: Target risk levels, in report order.

    Returns:
        A `DataFrame` with columns `model`, `cohort`, `region`, `alpha`, `n_cases`,
        `min_feasible_k` (`"never"` when `local_recalibration.min_feasible_k`
        returns `None` -- no `k` up to `n_cases - 1` could ever work), and
        `structural_floor` (`_structural_floor`'s data-independent bound).
    """
    rows: list[dict[str, Any]] = []
    for (model_name, cohort_name), curves_by_region in loaded.items():
        for region in regions:
            curves = curves_by_region[region]
            n_cases = len(curves)
            for alpha in alphas:
                k = lr.min_feasible_k(curves, alpha, k_max=max(n_cases - 1, 0))
                rows.append(
                    {
                        "model": model_name,
                        "cohort": cohort_name,
                        "region": region,
                        "alpha": alpha,
                        "n_cases": n_cases,
                        "min_feasible_k": k if k is not None else "never",
                        "structural_floor": _structural_floor(alpha),
                    }
                )
    return pd.DataFrame(rows)


def _log_floors_table(floors_df: pd.DataFrame) -> None:
    """Logs `floors.csv`'s contents as a compact table -- see the module docstring for why this
    is computed and logged before any random split is drawn."""
    lines = [
        "=" * 88,
        "local_recalibration: COUNTERFACTUAL -- feasibility floors, computed before any split "
        "is drawn",
        "=" * 88,
    ]
    for _, row in floors_df.iterrows():
        lines.append(
            f"  {row['model']:>24s} | {row['cohort']:>6s} | {row['region']:>3s} | "
            f"alpha={row['alpha']:.2f} | n={int(row['n_cases']):4d} | "
            f"min_feasible_k={row['min_feasible_k']!s:>6s} | "
            f"structural_floor={row['structural_floor']}"
        )
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Step 3: draws
# ---------------------------------------------------------------------------


def _resolve_ks_for_cohort(ks_cfg: Sequence[int | str], n: int) -> list[int | str]:
    """Drops a literal `"half"` entry that numerically coincides with an explicit `k`.

    See the module docstring's "Deduplicating a literal half" section -- this is
    what keeps such a cohort's split from being drawn (and counted) twice under two
    labels.

    Args:
        ks_cfg: The configured `k` sweep for this cohort, ints and/or `"half"`.
        n: This cohort's case count (`floor(n / 2)` is what `"half"` means).

    Returns:
        `ks_cfg` with `"half"` removed if, and only if, `n // 2` is already
        present as an explicit integer entry.
    """
    half_k = n // 2
    has_explicit_half = any((not isinstance(k, str)) and int(k) == half_k for k in ks_cfg)
    resolved: list[int | str] = []
    for k in ks_cfg:
        if isinstance(k, str) and has_explicit_half:
            logger.info(
                "local_recalibration: COUNTERFACTUAL -- 'half' (=%d) coincides with an "
                "explicit k already in this cohort's sweep; drawing it once, not twice.",
                half_k,
            )
            continue
        resolved.append(k)
    return resolved


def _run_draws(
    loaded: Mapping[tuple[str, str], Mapping[str, Sequence[CaseLossCurve]]],
    regions: Sequence[str],
    alphas: Sequence[float],
    ks_cfg: Sequence[int | str],
    n_splits: int,
    seed: int,
) -> dict[str, list[lr.SplitResult]]:
    """Runs `local_recalibration.run_local_recalibration` once per loaded (model, cohort).

    A FRESH `np.random.default_rng(seed)` is constructed for every (model, cohort)
    pair -- see the module docstring's "Pairing across models" section for why this
    makes both models see identical splits on a shared cohort without sharing one
    mutable generator object between them.

    Args:
        loaded: `_load_all_curves`'s return value.
        regions: Region names, in report order.
        alphas: Target risk levels, in report order.
        ks_cfg: The configured `k` sweep (ints and/or the literal `"half"`).
        n_splits: Number of independent random splits per (region, k).
        seed: Seed every fresh generator is constructed from.

    Returns:
        `{model_name: [SplitResult, ...]}`, pooling every cohort's results for that
        model.
    """
    results_by_model: dict[str, list[lr.SplitResult]] = {}
    for (model_name, cohort_name), curves_by_region in loaded.items():
        n = len(curves_by_region[regions[0]])
        ks = _resolve_ks_for_cohort(ks_cfg, n)
        rng = np.random.default_rng(seed)
        split_results = lr.run_local_recalibration(
            curves_by_region,
            cohort=cohort_name,
            alphas=alphas,
            ks=ks,
            n_splits=n_splits,
            rng=rng,
        )
        results_by_model.setdefault(model_name, []).extend(split_results)
    return results_by_model


def _write_splits_csv(results_by_model: Mapping[str, Sequence[lr.SplitResult]], path: Path) -> None:
    """Writes every `SplitResult`, plus a `model` column, to a gzip-compressed CSV."""
    rows: list[dict[str, Any]] = []
    for model_name, results in results_by_model.items():
        for r in results:
            row = asdict(r)
            row["model"] = model_name
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


# ---------------------------------------------------------------------------
# Step 5: summary
# ---------------------------------------------------------------------------


def _augment_summary(results: Sequence[lr.SplitResult]) -> pd.DataFrame:
    """Computes the columns `local_recalibration.summarise` does not: `n_feasible`, `mc_se`,
    `is_half`.

    Grouped identically to `summarise` (by `cohort`, `region`, `alpha`, `k`), so the
    result merges onto it one-to-one.

    Args:
        results: `SplitResult`s for ONE model (pooled across every cohort that
            model ran).

    Returns:
        A `DataFrame` with columns `cohort`, `region`, `alpha`, `k`, `n_feasible`
        (count of feasible splits in the group -- NOTE this can exceed the number
        of splits `mc_se` is actually computed from, see below), `mc_se`
        (`std(ddof=1)` of `realised_risk` over feasible splits WITH A DEFINED
        VALUE, divided by `sqrt` of THAT count -- not `n_feasible` -- since a
        feasible split's `realised_risk` is itself `None` whenever its held-out
        half was entirely empty-ground-truth cases, see
        `local_recalibration.evaluate_split`'s docstring; `NaN` if fewer than two
        splits have a defined value), and `is_half` (`k == n // 2`, `n` read from
        that group's own `n_calibration + n_heldout`).
    """
    groups: dict[tuple[str, str, float, int], list[lr.SplitResult]] = {}
    for r in results:
        groups.setdefault((r.cohort, r.region, r.alpha, r.k), []).append(r)

    rows: list[dict[str, Any]] = []
    for (cohort, region, alpha, k), group in groups.items():
        feasible_rows = [r for r in group if r.feasible]
        n_feasible = len(feasible_rows)
        risks = np.array(
            [r.realised_risk for r in feasible_rows if r.realised_risk is not None],
            dtype=np.float64,
        )
        # Divides by the number of VALUES the std itself was computed from, not by
        # n_feasible -- a feasible split can still have realised_risk=None (its
        # held-out half was all empty-GT), and using n_feasible there would understate
        # the true Monte Carlo standard error whenever that happens.
        if risks.size >= 2:
            mc_se = float(np.std(risks, ddof=1) / math.sqrt(risks.size))
        else:
            mc_se = float("nan")
        n_total = group[0].n_calibration + group[0].n_heldout
        rows.append(
            {
                "cohort": cohort,
                "region": region,
                "alpha": alpha,
                "k": k,
                "n_feasible": n_feasible,
                "mc_se": mc_se,
                "is_half": bool(k == n_total // 2),
            }
        )
    return pd.DataFrame(rows)


def _build_summary(results_by_model: Mapping[str, Sequence[lr.SplitResult]]) -> pd.DataFrame:
    """Builds the full `summary.csv` table: one `local_recalibration.summarise` call per model,
    augmented with `n_feasible`/`mc_se`/`is_half`, a `model` column, and `verdict`.

    Args:
        results_by_model: `_run_draws`'s return value.

    Returns:
        One row per `(model, cohort, region, alpha, k)`. Empty (no rows, no
        columns) if every model had zero results (e.g. every curves.npz was
        missing).
    """
    tables: list[pd.DataFrame] = []
    for model_name, results in results_by_model.items():
        if not results:
            continue
        base = lr.summarise(results)
        augmented = _augment_summary(results)
        merged = base.merge(augmented, on=["cohort", "region", "alpha", "k"], how="left")
        merged["model"] = model_name
        merged["verdict"] = merged.apply(
            lambda row: classify_restoration(
                row["mean_realised_risk"], row["mc_se"], row["alpha"], int(row["n_feasible"])
            ),
            axis=1,
        )
        tables.append(merged)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, axis=0, ignore_index=True)


def _control_cohort_failures(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rows of `summary_df` that trip the control-cohort falsifier (module docstring, "The
    control-cohort falsifier"): the in-distribution cohort, at k=half, verdict `NOT_RESTORED`.

    Args:
        summary_df: `_build_summary`'s return value.

    Returns:
        The (possibly empty) subset of `summary_df` that fails the falsifier.
    """
    if summary_df.empty:
        return summary_df
    mask = (
        (summary_df["cohort"] == _CONTROL_COHORT)
        & summary_df["is_half"]
        & (summary_df["verdict"] == "NOT_RESTORED")
    )
    return summary_df.loc[mask]


def _log_final_table(summary_df: pd.DataFrame) -> None:
    """Logs the final compact per-cell summary -- see the module docstring for why every log
    header here carries the word COUNTERFACTUAL."""
    lines = [
        "=" * 88,
        "local_recalibration: COUNTERFACTUAL -- final summary (never external validation)",
        "=" * 88,
    ]
    if summary_df.empty:
        lines.append("  (no (model, cohort) pair had a curves.npz to read -- nothing ran)")
    else:
        for _, row in summary_df.iterrows():
            lines.append(
                f"  {row['model']:>24s} | {row['cohort']:>6s} | {row['region']:>3s} | "
                f"alpha={row['alpha']:.2f} | k={int(row['k']):4d} | "
                f"feasible_rate={row['feasible_rate']:.3f} | "
                f"mean_realised_risk={row['mean_realised_risk']:.4f} | {row['verdict']}"
            )
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_local_recalibration(cfg: DictConfig) -> dict[str, Path]:
    """Runs the full local-recalibration (Mondrian) counterfactual sweep, per the composed config.

    Args:
        cfg: The full composed Hydra config. Reads `cfg.analysis.local_recalibration`
            (`out_dir`, `models`, `regions`, `alphas`, `ks`, `n_splits`, `seed`).

    Returns:
        A dict mapping a short name to the `Path` each output file was written to:
        `"floors_csv"`, `"splits_csv_gz"`, `"summary_csv"`, `"run_meta_json"`.
    """
    lr_cfg = cfg.analysis.local_recalibration
    out_dir = ensure_dir(str(lr_cfg.out_dir))

    models_cfg: dict[str, Any] = OmegaConf.to_container(lr_cfg.models, resolve=True)
    regions = [str(r) for r in lr_cfg.regions]
    alphas = [float(a) for a in lr_cfg.alphas]
    ks_cfg: list[int | str] = list(OmegaConf.to_container(lr_cfg.ks, resolve=True))
    n_splits = int(lr_cfg.n_splits)
    seed = int(lr_cfg.seed)

    logger.info(
        "local_recalibration: COUNTERFACTUAL run starting, out_dir=%s -- %s",
        out_dir,
        _STATUS_LABEL,
    )

    loaded = _load_all_curves(models_cfg, regions)
    _validate_loaded(loaded, regions)

    floors_df = _build_floors(loaded, regions, alphas)
    _log_floors_table(floors_df)
    floors_path = out_dir / "floors.csv"
    floors_df.to_csv(floors_path, index=False)
    logger.info("local_recalibration: wrote %s", floors_path)

    results_by_model = _run_draws(loaded, regions, alphas, ks_cfg, n_splits, seed)

    splits_path = out_dir / "splits.csv.gz"
    _write_splits_csv(results_by_model, splits_path)
    logger.info("local_recalibration: wrote %s", splits_path)

    summary_df = _build_summary(results_by_model)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("local_recalibration: wrote %s", summary_path)

    failures = _control_cohort_failures(summary_df)
    control_failed = not failures.empty
    if control_failed:
        logger.error(
            "local_recalibration: control failed -- implementation suspect, nothing "
            "downstream is valid. %d row(s) of the %s (in-distribution control) cohort at "
            "k=half are NOT_RESTORED: %s",
            len(failures),
            _CONTROL_COHORT,
            failures[["model", "region", "alpha", "k"]].to_dict("records"),
        )

    run_meta = {
        "git_sha": _git_sha(),
        "seed": seed,
        "n_splits": n_splits,
        "config": OmegaConf.to_container(lr_cfg, resolve=True),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status_label": _STATUS_LABEL,
        "control_failed": control_failed,
    }
    run_meta_path = out_dir / "run_meta.json"
    write_json(run_meta, run_meta_path)
    logger.info("local_recalibration: wrote %s", run_meta_path)

    _log_final_table(summary_df)

    return {
        "floors_csv": floors_path,
        "splits_csv_gz": splits_path,
        "summary_csv": summary_path,
        "run_meta_json": run_meta_path,
    }


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the local-recalibration counterfactual sweep, per the composed config.

    Example:

        python scripts/local_recalibration.py

    (every knob it needs lives at `cfg.analysis.local_recalibration` -- see the
    module docstring and `configs/analysis/default.yaml` for its shape.)

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_local_recalibration(cfg)


if __name__ == "__main__":
    main()
