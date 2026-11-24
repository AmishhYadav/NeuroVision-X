"""Hydra driver for MasterPlan W2.4: a risk-targeted refusal gate (Learn-then-Test).

Protocol (the contract): `docs/research_docs/preregistrations/preregistration_risk_gate.md`.
Statistics live in `neurovision.analysis.risk_gate`; per-case signals reuse
`scripts/failure_detection.py` (same code, same definitions as W2.3).

The idea in plain words: each case gets a score (higher = "probably a bad mask"). A gate accepts a
case when its score is at most a threshold lambda. We pick lambda on the VALIDATION split only,
so that among accepted cases the share of unusable masks stays under beta (with probability
1 - delta). Then lambda is frozen and read on test / SSA / PED. Nothing is ever fitted there.

Stages (all written under `out_dir`):

1. Signals: `signals_nv_val.csv` (computed here), `signals_nn_<split>.csv` (computed here), and
   neurovision test/ssa/ped signals read from W2.3's tables. NaN score -> +inf (empty foreground
   is always refused).
2. Labels: unusable <=> dice_WT < bar or dice_TC < bar, from each eval dir's per_case_metrics.csv.
3. Fit on val -> `fits.csv`, `ltt_steps.csv`.
4. Frozen application -> `cells.csv`.
5. Deployed gate scored the same way (descriptive) -> `deployed.csv`.
6. E3 deployment rule, applied once -> inside `run_summary.json`.

CPU only; one case's logits in memory at a time. Run (only after the protocol is registered):

    python scripts/risk_gate.py
    python scripts/risk_gate.py analysis.risk_gate.case_limit=3   # smoke run
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis import detection, error_budget, risk_gate
from neurovision.analysis.replay import (
    _resolve_postprocess_cfg,
    available_logit_cases,
    load_case_logits,
)
from neurovision.analysis.statistics import load_per_case
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
_SCRIPTS_DIR = Path(__file__).resolve().parent
_COHORTS = ("test", "ssa", "ped")  # the frozen-application cohorts; val is calibration only
_CELL_COLUMNS = [
    "pipeline",
    "model",
    "signal",
    "beta",
    "cohort",
    "lam",
    "n",
    "n_accepted",
    "coverage",
    "k_unusable_accepted",
    "selective_risk",
    "cp_lo",
    "cp_hi",
    "silent_failure",
    "over_refusal",
    "verdict",
]
_FIT_COLUMNS = [
    "pipeline",
    "model",
    "signal",
    "beta",
    "delta",
    "feasible",
    "lam",
    "target_coverage",
    "n_calibration",
    "n_accepted",
    "k_unusable",
    "calib_selective_risk",
]
_STEP_COLUMNS = [
    "pipeline",
    "beta",
    "coverage",
    "lam",
    "n_accepted",
    "k_unusable",
    "p_value",
    "rejected",
]
_METRIC_FIELDS = (
    "n",
    "n_accepted",
    "coverage",
    "k_unusable_accepted",
    "selective_risk",
    "cp_lo",
    "cp_hi",
    "silent_failure",
    "over_refusal",
    "verdict",
)


# ---------------------------------------------------------------------------
# Reusing scripts/failure_detection.py (scripts/ is not a package)
# ---------------------------------------------------------------------------


def _load_failure_detection() -> Any:
    """Loads `scripts/failure_detection.py` by path, reusing an already-loaded copy.

    The module is registered in `sys.modules` BEFORE it runs: dataclasses and Hydra look
    their module up by name while the file is still executing.
    """
    name = "failure_detection_script"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / "failure_detection.py")
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError("risk_gate: cannot import scripts/failure_detection.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_FD = _load_failure_detection()


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Writes a CSV via `<path>.tmp` then a rename, so a kill never leaves a half file."""
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _jsonable(obj: Any) -> Any:
    """Recursively converts numpy scalars / NaN / inf to plain JSON-safe values (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def _git_head(repo_root: Path) -> str | None:
    """Returns `git rev-parse HEAD`, or None if git is unavailable or this is not a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort, never a reason to fail
        return None


# ---------------------------------------------------------------------------
# Stage 1: signals
# ---------------------------------------------------------------------------


def check_member_logits(member_dirs: list[Path]) -> None:
    """Checks every member has logits for every case of the deployed (first) member.

    Args:
        member_dirs: Eval dirs, DEPLOYED FIRST.

    Raises:
        FileNotFoundError: The deployed member has no logits, or another member has none.
        ValueError: Another member lacks some of the deployed member's cases.
    """
    deployed = available_logit_cases(member_dirs[0])
    if not deployed:
        raise FileNotFoundError(f"deployed member {member_dirs[0]} has no logits/ directory.")
    for other in member_dirs[1:]:
        have = set(available_logit_cases(other))
        if not have:
            raise FileNotFoundError(
                f"member {other} has no validation logits (logits/ missing or empty). "
                "Produce them with scripts/evaluate.py before running the risk gate."
            )
        missing = [c for c in deployed if c not in have]
        if missing:
            raise ValueError(
                f"member {other} lacks validation logits for {len(missing)} of the deployed "
                f"member's cases, e.g. {missing[:5]}."
            )


def nnunet_ent_table(
    eval_dir: Path, out_path: Path, case_limit: int | None, overwrite: bool
) -> Path:
    """Writes `signals_nn_<split>.csv` (case_id, ENT) from one nnU-Net eval dir; cached.

    Args:
        eval_dir: Eval dir holding `logits/`.
        out_path: Destination CSV.
        case_limit: If set, only the first N cases (smoke runs).
        overwrite: Recompute even if `out_path` exists.

    Returns:
        `out_path`.
    """
    if out_path.is_file() and not overwrite:
        logger.info("%s exists, skipping (overwrite=false).", out_path)
        return out_path
    case_ids = available_logit_cases(eval_dir)
    if not case_ids:
        raise FileNotFoundError(f"no logits under {eval_dir / 'logits'}.")
    if case_limit is not None:
        case_ids = case_ids[: int(case_limit)]
    rows = []
    for i, case_id in enumerate(case_ids, start=1):
        logits = load_case_logits(eval_dir, case_id)  # float32 (3, D, H, W), one case at a time
        ent = float(detection.case_entropy_scalars(logits)["ent_mean_fg_mean"])
        rows.append({"case_id": case_id, "ENT": ent})
        del logits
        if i % 10 == 0:
            logger.info("%s: %d/%d cases", eval_dir, i, len(case_ids))
    _atomic_csv(pd.DataFrame(rows, columns=["case_id", "ENT"]), out_path)
    return out_path


def nan_to_inf(table: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Replaces NaN scores by +inf in place and returns the count per column.

    Protocol: a case with no predicted foreground has an undefined entropy and is ALWAYS
    refused; +inf is the score that is above every threshold.
    """
    counts: dict[str, int] = {}
    for col in columns:
        nan = table[col].isna()
        counts[col] = int(nan.sum())
        table.loc[nan, col] = np.inf
    return counts


# ---------------------------------------------------------------------------
# Stage 2: labels and joins
# ---------------------------------------------------------------------------


def unusable_labels(metrics: pd.DataFrame, bar: float) -> pd.Series:
    """The T-Dice label: unusable <=> dice_WT < bar or dice_TC < bar (NaN counts as unusable).

    Args:
        metrics: Table indexed by `case_id` with `dice_WT`, `dice_TC`.
        bar: Dice bar (0.7 in the protocol; exactly 0.7 is usable).

    Returns:
        Bool Series, True where the mask is unusable.
    """
    return ~error_budget.label_usable(metrics, regions=("WT", "TC"), bar=bar).astype(bool)


def join_labels(
    signals: pd.DataFrame, unusable: pd.Series, what: str, allow_partial: bool
) -> pd.DataFrame:
    """Attaches the `unusable` label to a signal table indexed by `case_id`.

    Inner-join loss is an error (a silently dropped case would bias the gate), except in a
    smoke run (`allow_partial`).

    Raises:
        ValueError: A signal case has no label, or a labelled case has no signal.
    """
    missing_label = [c for c in signals.index if c not in unusable.index]
    missing_signal = [c for c in unusable.index if c not in signals.index]
    if not allow_partial and (missing_label or missing_signal):
        raise ValueError(
            f"{what}: signal/label case sets differ "
            f"({len(missing_label)} signal cases without a label, e.g. {missing_label[:3]}; "
            f"{len(missing_signal)} labelled cases without a signal, e.g. {missing_signal[:3]})."
        )
    keep = [c for c in signals.index if c in unusable.index]
    out = signals.loc[keep].copy()
    out["unusable"] = unusable.loc[keep].to_numpy(dtype=bool)
    return out


# ---------------------------------------------------------------------------
# Stages 3-5: fit, apply, deployed
# ---------------------------------------------------------------------------


def _metric_row(app: risk_gate.GateApplication) -> dict[str, Any]:
    """The metric columns of one scored gate."""
    return {
        "n": app.n,
        "n_accepted": app.n_accepted,
        "coverage": app.coverage,
        "k_unusable_accepted": app.k_unusable_accepted,
        "selective_risk": app.selective_risk,
        "cp_lo": app.cp_lo,
        "cp_hi": app.cp_hi,
        "silent_failure": app.silent_failure,
        "over_refusal": app.over_refusal,
        "verdict": app.verdict,
    }


def _infeasible_metrics(n: int) -> dict[str, Any]:
    """Metric columns of a cell whose gate is infeasible: nothing was applied."""
    row: dict[str, Any] = {k: math.nan for k in _METRIC_FIELDS}
    row["n"] = n
    row["verdict"] = "gate_infeasible"
    return row


def fit_pipelines(
    tables: dict[tuple[str, str], pd.DataFrame],
    pipelines: dict[str, dict[str, str]],
    betas: list[float],
    delta: float,
) -> tuple[dict[tuple[str, float], risk_gate.LTTFit], pd.DataFrame, pd.DataFrame]:
    """Fits LTT on the VALIDATION tables only.

    Args:
        tables: `(model, split) -> table` with signal columns and bool `unusable`.
        pipelines: id -> {model, signal}.
        betas: Selective-risk targets.
        delta: LTT failure probability.

    Returns:
        `(fits, fits_df, steps_df)`; `fits` is keyed by `(pipeline_id, beta)`.
    """
    fits: dict[tuple[str, float], risk_gate.LTTFit] = {}
    fit_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for pid, spec in pipelines.items():
        val = tables[(spec["model"], "val")]  # the ONLY split a fit ever sees
        scores = val[spec["signal"]].to_numpy(dtype=float)
        unusable = val["unusable"].to_numpy(dtype=bool)
        for beta in betas:
            fit = risk_gate.fit_ltt(scores, unusable, float(beta), float(delta))
            fits[(pid, float(beta))] = fit
            fit_rows.append(
                {
                    "pipeline": pid,
                    "model": spec["model"],
                    "signal": spec["signal"],
                    "beta": beta,
                    "delta": delta,
                    "feasible": fit.feasible,
                    "lam": fit.lam,
                    "target_coverage": fit.target_coverage,
                    "n_calibration": fit.n_calibration,
                    "n_accepted": fit.n_accepted,
                    "k_unusable": fit.k_unusable,
                    "calib_selective_risk": fit.calib_selective_risk,
                }
            )
            for st in fit.steps:
                step_rows.append(
                    {
                        "pipeline": pid,
                        "beta": beta,
                        "coverage": st.coverage,
                        "lam": st.lam,
                        "n_accepted": st.n_accepted,
                        "k_unusable": st.k_unusable,
                        "p_value": st.p_value,
                        "rejected": st.rejected,
                    }
                )
    return (
        fits,
        pd.DataFrame(fit_rows, columns=_FIT_COLUMNS),
        pd.DataFrame(step_rows, columns=_STEP_COLUMNS),
    )


def apply_frozen(
    tables: dict[tuple[str, str], pd.DataFrame],
    pipelines: dict[str, dict[str, str]],
    fits: dict[tuple[str, float], risk_gate.LTTFit],
    conf: float,
) -> pd.DataFrame:
    """Applies each frozen fit to test / SSA / PED. An infeasible fit is never applied."""
    rows: list[dict[str, Any]] = []
    for (pid, beta), fit in fits.items():
        spec = pipelines[pid]
        for cohort in _COHORTS:
            table = tables[(spec["model"], cohort)]
            base = {
                "pipeline": pid,
                "model": spec["model"],
                "signal": spec["signal"],
                "beta": beta,
                "cohort": cohort,
                "lam": fit.lam,
            }
            if fit.feasible:
                app = risk_gate.apply_gate(
                    table[spec["signal"]].to_numpy(dtype=float),
                    table["unusable"].to_numpy(dtype=bool),
                    fit.lam,
                    beta,
                    conf,
                )
                rows.append({**base, **_metric_row(app)})
            else:
                rows.append({**base, **_infeasible_metrics(len(table))})
    return pd.DataFrame(rows, columns=_CELL_COLUMNS)


def deployed_rows(
    model: str,
    cohort: str,
    table_path: Path,
    unusable: pd.Series,
    betas: list[float],
    conf: float,
    allow_partial: bool,
) -> list[dict[str, Any]]:
    """Scores one model's deployed accepted set on one cohort (descriptive).

    Args:
        model: Model name, for the output rows.
        cohort: Cohort name.
        table_path: Deployed per-case table with bool `accepted` and `usable`.
        unusable: T-Dice label from stage 2 for the same model and cohort, by `case_id`.
        betas: Targets for the verdict.
        conf: Clopper-Pearson level.
        allow_partial: Smoke run: compare only on the shared cases.

    Raises:
        ValueError: The table's `usable` disagrees with the T-Dice label on any case, or
            the case sets differ.
    """
    table = load_per_case(table_path)
    if allow_partial:
        ids = [c for c in table.index if c in unusable.index]
        table = table.loc[ids]
    elif set(table.index) != set(unusable.index):
        raise ValueError(
            f"{model}/{cohort}: deployed table {table_path} and per_case_metrics have different "
            f"case sets ({len(table)} vs {len(unusable)})."
        )
    usable_table = table["usable"].astype(bool)
    label_usable = ~unusable.loc[table.index].astype(bool)
    bad = usable_table.index[usable_table.to_numpy() != label_usable.to_numpy()]
    if len(bad):
        raise ValueError(
            f"{model}/{cohort}: deployed table 'usable' disagrees with the T-Dice label on "
            f"{len(bad)} cases, e.g. {list(bad[:5])}. The deployed gate is scored against the "
            "same target as every pipeline, so this must match."
        )
    accepted = table["accepted"].astype(bool).to_numpy()
    u = ~usable_table.to_numpy()
    rows = []
    for beta in betas:
        app = risk_gate.selective_summary(accepted, u, float(beta), conf)
        rows.append(
            {
                "pipeline": "deployed",
                "model": model,
                "signal": "",
                "beta": beta,
                "cohort": cohort,
                "lam": math.nan,
                **_metric_row(app),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Stage 6: E3 deployment rule
# ---------------------------------------------------------------------------

_E3_PIPELINES = ("NV-ENT", "NV-ENSPD")


def deployment_rule(cells: pd.DataFrame, deployed: pd.DataFrame, betas: list[float]) -> dict:
    """E3: does NV-ENT or NV-ENSPD replace the deployed gate? Point estimates, test cohort.

    A pipeline qualifies iff its gate is feasible AND its test coverage >= the deployed
    neurovision coverage AND its test selective risk <= the deployed one. NaN risk (accepts
    nothing) never qualifies. If both qualify the higher coverage wins; equal coverage is a tie
    and there is no winner.

    Args:
        cells: Output of :func:`apply_frozen`.
        deployed: Deployed rows (all models).
        betas: Betas at which the rule is evaluated (protocol: 0.10).

    Returns:
        Dict keyed by beta (as string): deployed numbers, both pipelines' numbers, winner, reason.
    """
    out: dict[str, Any] = {}
    for beta in betas:
        dep = deployed[
            (deployed["model"] == "neurovision")
            & (deployed["cohort"] == "test")
            & np.isclose(deployed["beta"], beta)
        ]
        if dep.empty:
            raise ValueError("deployment rule needs the deployed neurovision table on test.")
        dep_cov = float(dep.iloc[0]["coverage"])
        dep_risk = float(dep.iloc[0]["selective_risk"])
        entry: dict[str, Any] = {
            "deployed": {"coverage": dep_cov, "selective_risk": dep_risk},
            "pipelines": {},
        }
        qualifying: list[tuple[str, float]] = []
        for pid in _E3_PIPELINES:
            row = cells[
                (cells["pipeline"] == pid)
                & (cells["cohort"] == "test")
                & np.isclose(cells["beta"], beta)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            feasible = r["verdict"] != "gate_infeasible"
            cov, risk = float(r["coverage"]), float(r["selective_risk"])
            # NaN compares False both ways, so an infeasible / empty gate never qualifies.
            ok = bool(feasible and cov >= dep_cov and risk <= dep_risk)
            entry["pipelines"][pid] = {
                "feasible": bool(feasible),
                "coverage": cov,
                "selective_risk": risk,
                "qualifies": ok,
            }
            if ok:
                qualifying.append((pid, cov))
        winner: str | None
        if not qualifying:
            winner, reason = None, "none_qualify"
        elif len(qualifying) == 1:
            winner, reason = qualifying[0][0], "only_qualifier"
        elif qualifying[0][1] == qualifying[1][1]:
            winner, reason = None, "tie"
        else:
            winner, reason = max(qualifying, key=lambda t: t[1])[0], "higher_coverage"
        entry["winner"] = winner
        entry["reason"] = reason
        out[str(beta)] = entry
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _to_dict(cfg: Any) -> dict[str, Any]:
    """DictConfig -> plain resolved dict (mappings pass through)."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def run_risk_gate(block_cfg: Any, full_cfg: Any, repo_root: Path) -> dict[str, Path]:
    """Runs all six stages (Hydra-free core).

    Args:
        block_cfg: The `analysis.risk_gate` block.
        full_cfg: Full config; only `inference.postprocess` is read (the deployed chain).
        repo_root: Root against which relative config paths resolve.

    Returns:
        Output name -> path (`fits`, `ltt_steps`, `cells`, `deployed`, `run_summary`).
    """
    block = _to_dict(block_cfg)
    pp_full = _to_dict(full_cfg.inference.postprocess)
    # Same resolution as failure_detection.run_failure_detection: threshold is split off.
    threshold = float(pp_full.pop("threshold"))
    pp_cfg = _resolve_postprocess_cfg(pp_full)

    out_root = ensure_dir(repo_root / block["out_dir"])
    betas = [float(b) for b in block["betas"]]
    delta, bar, conf = float(block["delta"]), float(block["dice_bar"]), float(block["conf"])
    case_limit = block["case_limit"]
    overwrite = bool(block["overwrite"])
    partial = case_limit is not None
    pipelines: dict[str, dict[str, str]] = block["pipelines"]
    models = {"neurovision": block["neurovision"], "nnunet_gatea": block["nnunet_gatea"]}
    signals_used: dict[str, list[str]] = {m: [] for m in models}
    for spec in pipelines.values():
        if spec["signal"] not in signals_used[spec["model"]]:
            signals_used[spec["model"]].append(spec["signal"])

    wall: dict[str, float] = {}
    t0 = time.time()

    # --- Stage 1: signals -------------------------------------------------
    raw: dict[tuple[str, str], pd.DataFrame] = {}
    nv = block["neurovision"]
    val_members = [repo_root / m for m in nv["val_members"]]
    val_csv = out_root / "signals_nv_val.csv"
    if overwrite or not val_csv.is_file():
        check_member_logits(val_members)  # fail early and name the missing member
    _FD.run_stage1(
        "val",
        val_members,
        val_csv,
        threshold,
        pp_cfg,
        case_limit=case_limit,
        overwrite=overwrite,
    )
    raw[("neurovision", "val")] = pd.read_csv(val_csv).set_index("case_id")
    for cohort in _COHORTS:
        sig = pd.read_csv(repo_root / nv["signals_tables"][cohort]).set_index("case_id")
        raw[("neurovision", cohort)] = sig.iloc[: int(case_limit)] if partial else sig
    nn = block["nnunet_gatea"]
    for split in ("val", *_COHORTS):
        path = nnunet_ent_table(
            repo_root / nn["eval_dirs"][split],
            out_root / f"signals_nn_{split}.csv",
            case_limit,
            overwrite,
        )
        raw[("nnunet_gatea", split)] = pd.read_csv(path).set_index("case_id")
    wall["signals_s"] = time.time() - t0

    # --- Stage 2: NaN -> +inf, labels, join --------------------------------
    t0 = time.time()
    inf_counts: dict[str, Any] = {}
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    labels: dict[tuple[str, str], pd.Series] = {}
    for (model, split), sig in raw.items():
        counts = nan_to_inf(sig, signals_used[model])
        inf_counts.setdefault(model, {}).setdefault(split, {}).update(counts)
        for col, k in counts.items():
            logger.info("%s / %s / %s: %d NaN scores -> +inf", model, split, col, k)
        metrics = load_per_case(
            repo_root / models[model]["eval_dirs"][split] / "per_case_metrics.csv"
        )
        labels[(model, split)] = unusable_labels(metrics, bar)
        tables[(model, split)] = join_labels(
            sig, labels[(model, split)], f"{model}/{split}", allow_partial=partial
        )
    wall["labels_s"] = time.time() - t0

    # --- Stage 3: fit on validation only ----------------------------------
    t0 = time.time()
    fits, fits_df, steps_df = fit_pipelines(tables, pipelines, betas, delta)
    wall["fit_s"] = time.time() - t0

    # --- Stage 4: frozen application --------------------------------------
    t0 = time.time()
    cells_df = apply_frozen(tables, pipelines, fits, conf)
    wall["apply_s"] = time.time() - t0

    # --- Stage 5: deployed gate -------------------------------------------
    dep_rows: list[dict[str, Any]] = []
    for model, spec in models.items():
        for cohort, rel in spec.get("deployed_tables", {}).items():
            dep_rows += deployed_rows(
                model, cohort, repo_root / rel, labels[(model, cohort)], betas, conf, partial
            )
    deployed_df = pd.DataFrame(dep_rows, columns=_CELL_COLUMNS)

    # --- Stage 6: E3 ------------------------------------------------------
    e3 = deployment_rule(cells_df, deployed_df, [float(b) for b in block["deployment_rule_betas"]])

    paths = {
        "fits": out_root / "fits.csv",
        "ltt_steps": out_root / "ltt_steps.csv",
        "cells": out_root / "cells.csv",
        "deployed": out_root / "deployed.csv",
        "run_summary": out_root / "run_summary.json",
    }
    _atomic_csv(fits_df, paths["fits"])
    _atomic_csv(steps_df, paths["ltt_steps"])
    _atomic_csv(cells_df, paths["cells"])
    _atomic_csv(deployed_df, paths["deployed"])
    summary = {
        "config": block,
        "git_head": _git_head(repo_root),
        "case_counts": {
            f"{m}/{s}": {"n": len(t), "n_unusable": int(t["unusable"].sum())}
            for (m, s), t in tables.items()
        },
        "inf_replacements": inf_counts,
        "wall_times_s": wall,
        "deployment_rule": e3,
    }
    write_json(_jsonable(summary), paths["run_summary"])
    return paths


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Runs W2.4 per `cfg.analysis.risk_gate`.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_risk_gate(cfg.analysis.risk_gate, cfg, Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    main()
