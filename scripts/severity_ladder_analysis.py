"""Hydra entry point for the W3.2 severity-ladder analysis.

The endpoints are FIXED by `docs/research_docs/preregistrations/preregistration_severity_ladder.md`
(Amendments 1-3) and implemented in `neurovision.analysis.severity_ladder_analysis`. This script
only does the I/O around that library: find the shard outputs, check them, load the FROZEN
thresholds, call the library, write tables and figures.

Nothing is fitted here. Every conformal threshold, gate cut and OOD model is read from disk and
was fitted on clean data long before any corrupted image existed.

Run (CPU, from the repo root; every path comes from `analysis.severity_ladder_analysis`):

    .venv/bin/python scripts/severity_ladder_analysis.py

Override example: `analysis.severity_ladder_analysis.n_boot=200` (any key of that block).
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib as mpl
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis import severity_ladder_analysis as sla
from neurovision.analysis.ood import OODModel
from neurovision.inference.gatekeeper import Thresholds, load_thresholds
from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed
from neurovision.visualization.figures import model_style, paper_style, save_figure

logger = logging.getLogger(__name__)

# Relative to this file so the script works from any working directory.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

_LEVEL0_KIND = sla.CLEAN_KIND
_V1_METHOD = "crc_v1"  # v1 conformal risk control, the primary-endpoint threshold
_PRIMARY_REGIONS_FOR_PLOT = ("WT", "TC")
_TRANSFERRED = "transferred, not refitted"


# ---------------------------------------------------------------------------
# Paths and provenance
# ---------------------------------------------------------------------------


def resolve_path(path: str | Path, base: Path) -> Path:
    """Makes a config path absolute against `base` (an absolute path stays as it is)."""
    p = Path(str(path))
    return p if p.is_absolute() else base / p


def _base_dir() -> Path:
    """The directory relative config paths are read against (Hydra's original cwd)."""
    return Path(get_original_cwd()) if HydraConfig.initialized() else Path.cwd()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    """Current commit SHA, or None if git is unavailable (provenance must never abort a run)."""
    repo_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# ---------------------------------------------------------------------------
# Step 1: discover and validate shards
# ---------------------------------------------------------------------------


def validate_shards(shards: pd.DataFrame, expected_cases: int) -> pd.DataFrame:
    """Checks every shard before any statistic is computed; fails loudly on a violation.

    A shard with a missing or duplicated case would silently shift every paired test, so each
    `per_case.csv` must have exactly `expected_cases` rows with unique `case_id`s, and every
    model must have its clean `none/level0` shard (it is level 0 of every kind).

    Args:
        shards: Output of `sla.discover_shards`.
        expected_cases: Required row count per shard (189 for the BraTS test split).

    Returns:
        A (model, kind) table with the number of levels found (also logged).

    Raises:
        ValueError: on any violation; the message lists them all.
    """
    problems: list[str] = []
    for row in shards.itertuples(index=False):
        ids = pd.read_csv(Path(row.path) / "per_case.csv", usecols=["case_id"])["case_id"]
        if len(ids) != expected_cases:
            problems.append(f"{row.path}: {len(ids)} rows, expected {expected_cases}")
        if not ids.is_unique:
            problems.append(f"{row.path}: duplicate case_id")
    for model in sorted(set(shards["model"])):
        clean = shards[
            (shards["model"] == model)
            & (shards["kind"] == _LEVEL0_KIND)
            & (shards["level_index"] == 0)
        ]
        if clean.empty:
            problems.append(f"model {model}: no {_LEVEL0_KIND}/level0 shard")
    if problems:
        raise ValueError("severity ladder shard validation failed:\n  " + "\n  ".join(problems))
    table = (
        shards.groupby(["model", "kind"])["level_index"]
        .apply(lambda s: ",".join(str(int(v)) for v in sorted(s)))
        .rename("levels")
        .reset_index()
    )
    logger.info("shards found (model, kind, levels):\n%s", table.to_string(index=False))
    return table


def shard_severities(shards: pd.DataFrame) -> dict[tuple[str, int], float]:
    """`{(kind, level_index): severity}` from each shard's `shard_meta.json` (for tick labels).

    Level 0 (the clean shard) has kind `"none"` and severity 0 when the meta stores none.
    """
    out: dict[tuple[str, int], float] = {}
    for row in shards.itertuples(index=False):
        meta = json.loads((Path(row.path) / "shard_meta.json").read_text(encoding="utf-8"))
        severity = meta.get("severity")
        out[(str(row.kind), int(row.level_index))] = 0.0 if severity is None else float(severity)
    return out


# ---------------------------------------------------------------------------
# Step 2: frozen thresholds
# ---------------------------------------------------------------------------


def load_frozen_taus(
    sl_cfg: DictConfig, models: Sequence[str], base: Path
) -> tuple[
    dict[str, dict[tuple[str, float], float]], dict[str, dict[tuple[str, float, str], float]]
]:
    """Reads the frozen v1 (primary) and v2 (descriptive) conformal thresholds for each model.

    v1 comes from `fit.json` (WT/TC only), v2 from `fits.csv` (CRC and RCPS, ET included, kept
    only for the configured alphas). Every tau used is logged.

    Args:
        sl_cfg: `cfg.analysis.severity_ladder_analysis`.
        models: Models present in the shards.
        base: Directory relative paths are resolved against.

    Returns:
        `(v1, v2)`: `{model: {(region, alpha): tau}}` and `{model: {(region, alpha, method): tau}}`
        where v2's `method` is `crc` / `rcps`.

    Raises:
        KeyError: a shard model has no entry in `v1_fits` / `v2_fits`, or its primary-alpha tau
            for a configured region is infeasible (the primary test cannot run without it).
    """
    regions = [str(r) for r in sl_cfg.regions]
    alphas = [float(a) for a in sl_cfg.alphas]
    primary_alpha = float(sl_cfg.primary_alpha)
    if primary_alpha not in alphas:
        alphas.append(primary_alpha)  # the primary tau must always be loaded
    v1: dict[str, dict[tuple[str, float], float]] = {}
    v2: dict[str, dict[tuple[str, float, str], float]] = {}
    for model in models:
        if model not in sl_cfg.v1_fits or model not in sl_cfg.v2_fits:
            raise KeyError(f"model {model!r} has shards but no v1_fits/v2_fits entry in config.")
        v1[model] = sla.load_v1_taus(resolve_path(sl_cfg.v1_fits[model], base), regions, alphas)
        for region in regions:
            if (region, primary_alpha) not in v1[model]:
                raise KeyError(f"{model}: no feasible v1 tau for {region} at alpha {primary_alpha}")
        spec = sl_cfg.v2_fits[model]
        all_v2 = sla.load_v2_taus(
            resolve_path(spec.csv, base), str(spec.model_key), grid=str(sl_cfg.v2_grid)
        )
        v2[model] = {k: t for k, t in all_v2.items() if any(np.isclose(k[1], a) for a in alphas)}
        for key, tau in sorted(v1[model].items()):
            logger.info("tau v1 %s %s = %.6f", model, key, tau)
        for key, tau in sorted(v2[model].items()):
            logger.info("tau v2 %s %s = %.6f", model, key, tau)
    return v1, v2


# ---------------------------------------------------------------------------
# Steps 5-6: gate and OOD (frozen inputs only)
# ---------------------------------------------------------------------------


def run_gate(
    cfg: DictConfig,
    shards: pd.DataFrame,
    thresholds: Thresholds,
    regions: Sequence[str],
    stand_in_age_years: float,
) -> tuple[dict[tuple[str, str, int], pd.DataFrame], pd.DataFrame]:
    """Applies the frozen deployed gate to every shard.

    Returns:
        `({(model, kind, level): decisions}, long table gate_per_case)`.
    """
    logger.warning(
        "Gate uses the curated input-QC stand-in: thick-slice inputs are resampled back to 1 mm "
        "before segmentation, so the real front end's anisotropy input-QC check cannot fire "
        "here. Any refusal below comes from the model-output signals only."
    )
    by_key: dict[tuple[str, str, int], pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []
    for row in shards.itertuples(index=False):
        per_case = pd.read_csv(Path(row.path) / "per_case.csv")
        dec = sla.gate_decisions(cfg, per_case, thresholds, regions, stand_in_age_years)
        key = (str(row.model), str(row.kind), int(row.level_index))
        by_key[key] = dec
        tagged = dec.copy()
        tagged.insert(0, "level", key[2])
        tagged.insert(0, "kind", key[1])
        tagged.insert(0, "model", key[0])
        frames.append(tagged)
    columns = [
        "model", "kind", "level", "case_id", "decision", "accepted", "usable",
        "silent_failure", "gate_label", "refusing_signals",
    ]  # fmt: skip
    return by_key, pd.concat(frames, ignore_index=True)[columns]


def load_ood(ood_dir: Path) -> tuple[OODModel, tuple[float, float]]:
    """Loads the frozen OOD model and cuts exactly as `scripts/ood_score.py` wrote them.

    Raises:
        FileNotFoundError: `model.json` or `thresholds.json` is missing.
        ValueError: `caution_cut > refuse_cut` (high score is bad, caution must cut in first).
    """
    model_path, cuts_path = ood_dir / "model.json", ood_dir / "thresholds.json"
    for path in (model_path, cuts_path):
        if not path.is_file():
            raise FileNotFoundError(f"frozen OOD artifact not found: {path}")
    model = OODModel.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
    payload = json.loads(cuts_path.read_text(encoding="utf-8"))
    cuts = (float(payload["caution_cut"]), float(payload["refuse_cut"]))
    if cuts[0] > cuts[1]:
        raise ValueError(f"{cuts_path}: caution_cut {cuts[0]} > refuse_cut {cuts[1]}")
    return model, cuts


def check_ood_columns(model: OODModel, features: pd.DataFrame, where: str) -> None:
    """Requires the feature CSV's columns to equal the model's feature names, by exact name.

    The library would silently ignore extra columns; here a mismatch means the shard was built
    with a different feature extractor than the frozen model was fitted on, so it must stop.

    Raises:
        ValueError: missing or unexpected feature columns (`case_id` is the only allowed extra).
    """
    have = {c for c in features.columns if c != "case_id"}
    want = set(model.feature_names)
    if have != want:
        raise ValueError(
            f"{where}: OOD feature columns differ from the frozen model's "
            f"(missing {sorted(want - have)}, unexpected {sorted(have - want)})."
        )


def run_ood(
    model: OODModel, cuts: tuple[float, float], shards: pd.DataFrame
) -> tuple[dict[tuple[str, str, int], pd.DataFrame], pd.DataFrame]:
    """Scores every shard's `ood_features.csv` with the frozen model and cuts.

    Returns:
        `({(model, kind, level): ood_flags}, long table ood_per_case)`.
    """
    by_key: dict[tuple[str, str, int], pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []
    for row in shards.itertuples(index=False):
        features = pd.read_csv(Path(row.path) / "ood_features.csv")
        check_ood_columns(model, features, str(row.path))
        flags = sla.ood_flags(model, cuts, features)
        key = (str(row.model), str(row.kind), int(row.level_index))
        by_key[key] = flags
        tagged = flags.copy()
        tagged.insert(0, "level", key[2])
        tagged.insert(0, "kind", key[1])
        tagged.insert(0, "model", key[0])
        frames.append(tagged)
    return by_key, pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 7: descriptive table (v1 and v2 side by side)
# ---------------------------------------------------------------------------


def build_descriptive(
    shards: pd.DataFrame,
    v1: Mapping[str, Mapping[tuple[str, float], float]],
    v2: Mapping[str, Mapping[tuple[str, float, str], float]],
    decisions: Mapping[tuple[str, str, int], pd.DataFrame],
    ood: Mapping[tuple[str, str, int], pd.DataFrame],
    v2_empty_by_region: Mapping[str, str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Descriptive table with v1 (`crc_v1`) and v2 (`crc_v2`, `rcps_v2`) risks in one frame.

    `descriptive_table` takes ONE empty-GT convention per region per call (v1 excludes empty-GT
    cases; v2's ET counts them as zero loss), so it is called twice. Dice, gate and OOD rows come
    from the v1 call only, so they are not duplicated.
    """
    v1_taus = {m: {(r, a, _V1_METHOD): t for (r, a), t in taus.items()} for m, taus in v1.items()}
    v2_taus = {
        m: {(r, a, f"{meth}_v2"): t for (r, a, meth), t in taus.items()} for m, taus in v2.items()
    }
    first = sla.descriptive_table(
        shards, v1_taus, decisions=decisions, ood=ood, empty_by_region={}, n_boot=n_boot, seed=seed
    )
    second = sla.descriptive_table(
        shards, v2_taus, empty_by_region=dict(v2_empty_by_region), n_boot=n_boot, seed=seed
    )
    second = second[second["quantity"] == "realised_risk"]
    return pd.concat([first, second], ignore_index=True)


# ---------------------------------------------------------------------------
# Step 8: outputs
# ---------------------------------------------------------------------------


def _input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(p): sha256_file(p) for p in paths if Path(p).is_file()}


def write_run_meta(
    out_dir: Path,
    cfg_paths: Sequence[Path],
    shards: pd.DataFrame,
    v1: Mapping[str, Any],
    v2: Mapping[str, Any],
    sl_cfg: DictConfig,
) -> Path:
    """Writes `run_meta.json`: git SHA, UTC time, input hashes, every tau, library constants."""
    shard_files = [
        Path(row.path) / name
        for row in shards.itertuples(index=False)
        for name in ("per_case.csv", "curves.npz", "ood_features.csv", "shard_meta.json")
    ]
    meta = {
        "git_head": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "n_shards": int(len(shards)),
        "input_sha256": _input_hashes(list(cfg_paths) + shard_files),
        "taus_v1": {m: {f"{r}__alpha_{a}": t for (r, a), t in d.items()} for m, d in v1.items()},
        "taus_v2": {
            m: {f"{r}__alpha_{a}__{meth}": t for (r, a, meth), t in d.items()}
            for m, d in v2.items()
        },
        "library_constants": {
            "N_BOOT": sla.N_BOOT,
            "BOOTSTRAP_SEED": sla.BOOTSTRAP_SEED,
            "USABLE_REGIONS": list(sla.USABLE_REGIONS),
            "USABLE_BAR": sla.USABLE_BAR,
        },
        "config": OmegaConf.to_container(sl_cfg, resolve=True),
        "nothing_fitted": (
            "every conformal tau, gate threshold and OOD model/cut was read frozen from disk; "
            "no parameter was fitted on corrupted data"
        ),
    }
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def _md_table(df: pd.DataFrame) -> str:
    """A GitHub markdown table without needing `tabulate`."""
    if df.empty:
        return "(none)\n"

    def fmt(v: Any) -> str:
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |"]
    lines.append("|" + "|".join("---" for _ in df.columns) + "|")
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_summary(
    out_dir: Path, primary: pd.DataFrame, verdicts: pd.DataFrame, secondary: pd.DataFrame
) -> Path:
    """Writes `summary.md`: the three endpoint tables, numbers only."""
    cols = ["kind", "model", "region", "n_cases", "L", "p", "p_holm", "diff", "ci_lo", "ci_hi"]
    text = (
        "# Severity ladder: endpoint tables\n\n## Primary (per cell)\n\n"
        + _md_table(primary[cols + ["verdict"]])
        + "\n## Kind verdicts\n\n"
        + _md_table(verdicts)
        + "\n## Secondary (slope pairs)\n\n"
        + _md_table(secondary)
    )
    path = out_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Step 9: figures
# ---------------------------------------------------------------------------


def _level_ticks(
    kind: str, levels: Sequence[int], severities: Mapping[tuple[str, int], float]
) -> list[str]:
    """Tick labels `L<level>` plus the real severity value (level 0 is the clean shard)."""
    labels = []
    for lvl in levels:
        sev = severities.get((_LEVEL0_KIND if lvl == 0 else kind, lvl))
        labels.append(f"L{lvl}\n{'' if sev is None else format(sev, 'g')}")
    return labels


def _series(desc: pd.DataFrame, model: str, kind: str, **where: Any) -> pd.DataFrame:
    """Rows of `desc` for one model along one kind's ladder (clean level 0 prepended), by level."""
    mask = (desc["model"] == model) & desc["kind"].isin([kind, _LEVEL0_KIND])
    for col, val in where.items():
        mask &= desc[col] == val
    return desc[mask].sort_values("level_index")


def plot_dose_response(
    desc: pd.DataFrame,
    models: Sequence[str],
    severities: Mapping[tuple[str, int], float],
    alpha: float,
    out_dir: Path,
) -> list[Path]:
    """One panel per kind: realised v1 risk at `alpha` vs level, WT solid / TC dashed, 95% CI."""
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    kinds = sorted(set(desc["kind"]) - {_LEVEL0_KIND})
    risk = desc[
        (desc["quantity"] == "realised_risk")
        & (desc["method"] == _V1_METHOD)
        & np.isclose(desc["alpha"].astype(float), alpha)
    ]
    with paper_style():
        fig, axes = plt.subplots(
            1, max(len(kinds), 1), figsize=(3.2 * max(len(kinds), 1), 3.2), squeeze=False
        )
        for ax, kind in zip(axes[0], kinds):
            levels = sorted(set(risk[risk["kind"] == kind]["level_index"]) | {0})
            xs = {lvl: i for i, lvl in enumerate(levels)}
            for mi, model in enumerate(models):
                style = model_style(mi)
                for region, ls in zip(_PRIMARY_REGIONS_FOR_PLOT, ("-", "--")):
                    s = _series(risk, model, kind, region=region)
                    if s.empty:
                        continue
                    x = np.array([xs[int(v)] for v in s["level_index"]])
                    y = s["value"].to_numpy(float)
                    err = np.vstack(
                        [y - s["ci_lo"].to_numpy(float), s["ci_hi"].to_numpy(float) - y]
                    )
                    ax.errorbar(
                        x, y, yerr=err, color=style["color"], linestyle=ls, marker="o",
                        markersize=3, capsize=2, label=f"{model} {region}",
                    )  # fmt: skip
            ax.axhline(alpha, color="black", linewidth=0.8, label=f"nominal alpha={alpha:g}")
            ax.set_xticks(list(xs.values()))
            ax.set_xticklabels(_level_ticks(kind, levels, severities))
            ax.set_title(kind)
            ax.set_xlabel("level (severity)")
        axes[0][0].set_ylabel(f"realised risk, v1 CRC alpha={alpha:g}")
        axes[0][-1].legend(fontsize=6, loc="best")
        fig.tight_layout()
        paths = save_figure(fig, out_dir, "dose_response", formats=("png", "pdf"), close=True)
    return paths


def plot_gate_vs_level(
    desc: pd.DataFrame,
    models: Sequence[str],
    severities: Mapping[tuple[str, int], float],
    out_dir: Path,
) -> list[Path]:
    """Refusal rate (top row) and silent-failure rate (bottom row) vs level, per kind and model."""
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    kinds = sorted(set(desc["kind"]) - {_LEVEL0_KIND})
    n = max(len(kinds), 1)
    with paper_style():
        fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.4), squeeze=False)
        for row_i, quantity in enumerate(("refusal_rate", "silent_failure_rate")):
            rows = desc[desc["quantity"] == quantity]
            for ax, kind in zip(axes[row_i], kinds):
                levels = sorted(set(rows[rows["kind"] == kind]["level_index"]) | {0})
                xs = {lvl: i for i, lvl in enumerate(levels)}
                for mi, model in enumerate(models):
                    s = _series(rows, model, kind)
                    if s.empty:
                        continue
                    label = model if model == "neurovision" else f"{model} ({_TRANSFERRED})"
                    ax.plot(
                        [xs[int(v)] for v in s["level_index"]], s["value"].to_numpy(float),
                        marker="o", markersize=3, label=label, **model_style(mi),
                    )  # fmt: skip
                ax.set_xticks(list(xs.values()))
                ax.set_xticklabels(_level_ticks(kind, levels, severities))
                if row_i == 0:
                    ax.set_title(kind)
                ax.set_ylim(-0.02, 1.02)
            axes[row_i][0].set_ylabel(quantity.replace("_", " "))
        axes[0][0].legend(fontsize=5, loc="best")
        fig.tight_layout()
        paths = save_figure(fig, out_dir, "gate_vs_level", formats=("png", "pdf"), close=True)
    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_analysis(cfg: DictConfig, base: Path) -> dict[str, Path]:
    """Runs the whole analysis and writes every output file.

    Args:
        cfg: Root config (needs `analysis.severity_ladder_analysis` and `clinical.gatekeeper`).
        base: Directory relative config paths are resolved against.

    Returns:
        `{output name: path}` for the main outputs.
    """
    sl_cfg = cfg.analysis.severity_ladder_analysis
    out_dir = ensure_dir(resolve_path(sl_cfg.out_dir, base))
    n_boot, seed = int(sl_cfg.n_boot), int(sl_cfg.seed)
    regions = [str(r) for r in sl_cfg.regions]
    primary_alpha = float(sl_cfg.primary_alpha)

    # 1. Find and validate the shards.
    roots = [resolve_path(r, base) for r in sl_cfg.shard_roots]
    shards = sla.discover_shards(roots, require_done=bool(sl_cfg.require_done))
    validate_shards(shards, int(sl_cfg.expected_cases))
    models = sorted(set(shards["model"]))
    severities = shard_severities(shards)

    # 2. Frozen conformal thresholds.
    v1, v2 = load_frozen_taus(sl_cfg, models, base)

    # 3-4. Primary and secondary endpoints (frozen v1 taus).
    primary = sla.primary_trend_family(
        shards, v1, regions=regions, alpha=primary_alpha, n_boot=n_boot, seed=seed
    )
    verdicts = sla.kind_verdicts(primary)
    secondary = sla.secondary_slope_family(
        shards, v1, regions=regions, alpha=primary_alpha, n_boot=n_boot, seed=seed
    )

    # 5. Frozen deployed gate.
    thresholds = load_thresholds(cfg)
    if thresholds is None:
        raise ValueError("clinical.gatekeeper.thresholds is null: the frozen gate is required.")
    gate_regions = [str(r) for r in sl_cfg.gate_regions]
    decisions, gate_table = run_gate(
        cfg, shards, thresholds, gate_regions, float(sl_cfg.stand_in_age_years)
    )

    # 6. Frozen OOD model and cuts.
    ood_dir = resolve_path(sl_cfg.ood_dir, base)
    ood_model, cuts = load_ood(ood_dir)
    ood_by_key, ood_table = run_ood(ood_model, cuts, shards)

    # 7. Descriptive table.
    descriptive = build_descriptive(
        shards, v1, v2, decisions, ood_by_key, dict(sl_cfg.v2_empty_by_region), n_boot, seed
    )

    # 8. Tables and provenance.
    paths: dict[str, Path] = {}
    for name, frame in (
        ("primary", primary),
        ("kind_verdicts", verdicts),
        ("secondary", secondary),
        ("descriptive", descriptive),
        ("gate_per_case", gate_table),
        ("ood_per_case", ood_table),
    ):
        paths[name] = out_dir / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    cfg_inputs = [resolve_path(sl_cfg.v1_fits[m], base) for m in models] + [
        resolve_path(sl_cfg.v2_fits[m].csv, base) for m in models
    ]
    cfg_inputs += [ood_dir / "model.json", ood_dir / "thresholds.json"]
    gate_path = cfg.clinical.gatekeeper.thresholds
    if isinstance(gate_path, str):
        cfg_inputs.append(resolve_path(gate_path, base))
    paths["run_meta"] = write_run_meta(out_dir, sorted(set(cfg_inputs)), shards, v1, v2, sl_cfg)
    paths["summary"] = write_summary(out_dir, primary, verdicts, secondary)

    # 9. Figures.
    plot_dose_response(descriptive, models, severities, primary_alpha, out_dir)
    plot_gate_vs_level(descriptive, models, severities, out_dir)
    logger.info("severity_ladder_analysis: wrote outputs to %s", out_dir)
    return paths


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point; see the module docstring."""
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_analysis(cfg, _base_dir())


if __name__ == "__main__":
    main()
