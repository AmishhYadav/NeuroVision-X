"""Plans the fusion ablation grid and estimates its Kaggle GPU-hour cost.

Pure planning tool: no GPU, no torch import, no data loading, no training.
Its whole job is answering two questions BEFORE any Kaggle session is spent:

  1. What is the exact, ordered list of `python scripts/train.py` commands
     for the six-variant fusion ablation grid?
  2. Roughly how many GPU-hours (and ~30h/week free-tier sessions) does the
     whole grid cost, given a per-step time estimate?

Nothing here executes training. Run it directly:

    python scripts/run_ablation_grid.py

and again after the first real epoch, with a measured `--sec-per-step`, to
replace the initial guess with a real number (the banner in the report tells
you exactly what to divide to get it).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import yaml

logger = logging.getLogger(__name__)

# Repo root derived from this file's location, never from the cwd -- so the
# script works regardless of where it is invoked from. scripts/ has no
# __init__.py sibling (it is not an importable package), matching every other
# script in this repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIGS_DIR = str(_REPO_ROOT / "configs")

# The six ablation grid variants, in the FIXED, canonical order the report
# and the emitted commands must follow regardless of CLI argument order.
GRID_VARIANTS: tuple[str, ...] = (
    "ablation_full",
    "ablation_fusion_concat",
    "ablation_fusion_add",
    "ablation_cnn_only",
    "ablation_transformer_only",
    "ablation_no_deep_supervision",
)

# Per-variant cost multiplier relative to ablation_full (= 1.0). EVERY value
# here except 1.0 is an UNMEASURED estimate -- nobody has run any of these on
# a GPU yet -- reasoned from what each variant removes, not from a profile.
# Recompute from real Kaggle wall-clock times as soon as they exist; treat
# these as "plausible enough to size a GPU-hour budget", not as fact.
DEFAULT_RELATIVE_COST: dict[str, float] = {
    # Reference row. Every other multiplier is relative to this by definition.
    "ablation_full": 1.0,
    # UNMEASURED. Reasoning: concat drops the gate generator and the windowed
    # cross-attention block, keeping a cheap 1x1x1 conv + norm in their place.
    "ablation_fusion_concat": 0.88,
    # UNMEASURED. Reasoning: additive fusion is concat's 1x1x1 conv without
    # even the extra norm/activation stack -- the cheapest merge that exists.
    "ablation_fusion_add": 0.85,
    # UNMEASURED. Reasoning: the entire Swin branch and every fusion block are
    # gone. Swin + fusion is a disproportionate share of runtime (attention is
    # costlier per-parameter than convolution), so this removes more compute
    # than its ~6% share of total params (2.0M Swin + 1.0M fusion / 34.9M).
    "ablation_cnn_only": 0.55,
    # UNMEASURED. Reasoning: the CNN branch (the parameter bulk, 18.9M of
    # 34.9M) is gone, but Swin's shifted-window attention costs more per
    # parameter than the CNN's convolutions, so this is not as cheap as the
    # removed-parameter fraction alone would suggest.
    "ablation_transformer_only": 0.70,
    # UNMEASURED. Reasoning: removes two of three 1x1x1 segmentation heads
    # (681 params total across all three -- negligible) and the deep
    # supervision loss terms. Almost the full architecture still runs.
    "ablation_no_deep_supervision": 0.98,
}


@dataclass(frozen=True)
class CostEstimate:
    """Pure arithmetic result of `estimate_variant_cost` for one variant.

    Attributes:
        steps_per_epoch: Optimizer steps per epoch, `ceil(n_train / batch_size)`.
        n_val_passes: Number of sliding-window validation passes over the run.
        total_seconds: Estimated total wall-clock seconds for the whole run.
        total_hours: `total_seconds / 3600`.
        sessions: `ceil(total_hours / max_hours)`, i.e. how many times
            `scripts/train.py` (start once, then the same command to resume)
            must be invoked to finish this variant.
    """

    steps_per_epoch: int
    n_val_passes: int
    total_seconds: float
    total_hours: float
    sessions: int


@dataclass(frozen=True)
class VariantReport:
    """One row of the ablation grid report.

    Attributes:
        name: Variant name, e.g. `"ablation_full"`.
        status: One of `"ok"`, `"missing_config"`, `"compose_failed"`.
        detail: Human-readable status detail (error message or a fixed note).
            `None` when `status == "ok"`.
        epochs: `cfg.training.epochs` for this variant, or `None` if the
            config never composed.
        relative_cost: The multiplier used for this variant (from the default
            table or a `--relative-cost` override).
        cost: The `CostEstimate` for this variant, or `None` if it never
            composed.
    """

    name: str
    status: str
    detail: str | None
    epochs: int | None
    relative_cost: float
    cost: CostEstimate | None


def load_case_counts(
    splits_path: Path | None,
    n_train_fallback: int | None,
    n_val_fallback: int | None,
) -> tuple[int, int, bool]:
    """Reads train/val case counts from a frozen splits YAML.

    Args:
        splits_path: Path to `splits.yaml` (has `train:`/`val:`/`test:` list
            keys), or `None` if it could not be located.
        n_train_fallback: Value to use if `splits_path` is missing/unreadable.
        n_val_fallback: Value to use if `splits_path` is missing/unreadable.

    Returns:
        `(n_train, n_val, assumed)`. `assumed` is `True` when the fallback
        values were used instead of reading the real file, so callers can
        warn loudly rather than silently reporting a guessed number as fact.

    Raises:
        ValueError: If `splits_path` is missing/unreadable AND either
            fallback is `None`.
    """
    if splits_path is not None and Path(splits_path).is_file():
        raw = yaml.safe_load(Path(splits_path).read_text(encoding="utf-8")) or {}
        n_train = len(raw.get("train") or [])
        n_val = len(raw.get("val") or [])
        logger.debug("Read case counts from %s: train=%d val=%d", splits_path, n_train, n_val)
        return n_train, n_val, False

    if n_train_fallback is None or n_val_fallback is None:
        raise ValueError(
            f"splits.yaml not found at {splits_path!r} and no --n-train/--n-val fallback "
            "was given. Pass --n-train N --n-val N to proceed with assumed counts."
        )
    logger.debug(
        "splits.yaml not found at %s; using fallback counts train=%d val=%d",
        splits_path,
        n_train_fallback,
        n_val_fallback,
    )
    return n_train_fallback, n_val_fallback, True


def compose_variant(configs_dir: Path | str, variant: str, data_root_dir: str) -> Any:
    """Composes the real Hydra config for one ablation grid variant.

    Follows the same programmatic-composition pattern as
    `scripts/smoke_test.py._compose_config`: `hydra.initialize_config_dir` +
    `hydra.compose`, so this exercises the REAL config files (including
    interpolations like `${data.num_classes}`), not a hand-built stand-in.

    Args:
        configs_dir: Path to the `configs/` directory to compose from.
        variant: Experiment name, e.g. `"ablation_full"`. Composed as
            `+experiment=<variant>` since `experiment` has no group default.
        data_root_dir: Value substituted for the mandatory (`???`)
            `data.root_dir`, so composition succeeds without touching any
            real path. This script never reads from it.

    Returns:
        The composed `DictConfig`.

    Raises:
        Exception: Whatever Hydra/OmegaConf raise on a malformed or
            unresolvable config. Callers decide how to report this.
    """
    resolved_dir = str(Path(configs_dir).resolve())
    overrides = [f"+experiment={variant}", f"data.root_dir={data_root_dir}"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=resolved_dir):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def estimate_variant_cost(
    n_train_cases: int,
    n_val_cases: int,
    batch_size: int,
    epochs: int,
    val_interval: int,
    max_hours: float,
    sec_per_step: float,
    sec_per_val_case: float,
    relative_cost: float,
) -> CostEstimate:
    """Pure arithmetic GPU-hour estimate for one variant. No I/O, no Hydra.

    `steps_per_epoch` divides by `batch_size` alone, NOT
    `batch_size * samples_per_volume`. The DataLoader batches CASES;
    `RandCropByPosNegLabeld` returns `samples_per_volume` crops per case,
    which `list_data_collate` flattens AFTER batching. So one optimizer step
    consumes `batch_size` cases (and `batch_size * samples_per_volume`
    patches). Dividing by the product here would understate the number of
    steps -- and therefore the whole grid's cost -- by up to 4x.

    Args:
        n_train_cases: Number of training cases (splits.yaml `train` count).
        n_val_cases: Number of validation cases (splits.yaml `val` count).
        batch_size: `cfg.training.batch_size` (cases per optimizer step).
        epochs: `cfg.training.epochs`.
        val_interval: `cfg.training.val_interval` (epochs between validation
            passes).
        max_hours: `cfg.training.max_hours`, the wall-clock stop per session.
        sec_per_step: Estimated seconds per optimizer step at
            `relative_cost == 1.0`.
        sec_per_val_case: Estimated seconds of sliding-window validation per
            val case at `relative_cost == 1.0`.
        relative_cost: Multiplier applied to both training and validation
            time for this variant, relative to `ablation_full`.

    Returns:
        The `CostEstimate` for this variant.
    """
    steps_per_epoch = math.ceil(n_train_cases / batch_size)
    train_seconds_per_epoch = steps_per_epoch * sec_per_step * relative_cost
    n_val_passes = epochs // val_interval
    val_seconds_per_pass = n_val_cases * sec_per_val_case * relative_cost
    total_seconds = epochs * train_seconds_per_epoch + n_val_passes * val_seconds_per_pass
    total_hours = total_seconds / 3600.0
    sessions = math.ceil(total_hours / max_hours) if total_hours > 0 else 0
    return CostEstimate(
        steps_per_epoch=steps_per_epoch,
        n_val_passes=n_val_passes,
        total_seconds=total_seconds,
        total_hours=total_hours,
        sessions=sessions,
    )


def _canonical_order(variants: Sequence[str]) -> list[str]:
    """Filters `GRID_VARIANTS` down to `variants`, preserving grid order.

    Args:
        variants: Requested variant names, in any order.

    Returns:
        The subset of `GRID_VARIANTS` that appears in `variants`, in
        `GRID_VARIANTS`' fixed order -- never the order `variants` was given in.
    """
    requested = set(variants)
    return [name for name in GRID_VARIANTS if name in requested]


def _resolve_splits_path(configs_dir: Path | str, data_root_dir: str) -> Path | None:
    """Finds `data.splits.path` by composing whichever variant composes first.

    The splits path is a `data` group default, not an experiment-specific
    override, so any variant that composes at all gives the same answer.
    Tried in grid order; a variant with a missing or broken config is skipped
    rather than raising here, since this is only a lookup.

    Args:
        configs_dir: Path to the `configs/` directory.
        data_root_dir: Dummy value for the mandatory `data.root_dir`.

    Returns:
        The splits YAML path resolved against the repo root, or `None` if no
        variant composed at all.
    """
    for name in GRID_VARIANTS:
        if not (Path(configs_dir) / "experiment" / f"{name}.yaml").is_file():
            continue
        try:
            cfg = compose_variant(configs_dir, name, data_root_dir)
        except Exception:  # noqa: BLE001 - deliberately broad, this is a lookup
            continue
        return (_REPO_ROOT / str(cfg.data.splits.path)).resolve()
    return None


def build_variant_reports(
    configs_dir: Path | str,
    variants: Sequence[str],
    data_root_dir: str,
    n_train: int,
    n_val: int,
    sec_per_step: float,
    sec_per_val_case: float,
    relative_cost: dict[str, float],
) -> list[VariantReport]:
    """Builds one `VariantReport` per requested variant, in grid order.

    Never raises on a missing or broken experiment config -- each failure
    mode becomes a row with `status != "ok"` instead.

    Args:
        configs_dir: Path to the `configs/` directory.
        variants: Requested variant names (any order, any subset).
        data_root_dir: Dummy value substituted for `data.root_dir`.
        n_train: Training case count (already resolved by the caller).
        n_val: Validation case count (already resolved by the caller).
        sec_per_step: See `estimate_variant_cost`.
        sec_per_val_case: See `estimate_variant_cost`.
        relative_cost: Per-variant multiplier table (defaults, possibly
            overridden by `--relative-cost`).

    Returns:
        One `VariantReport` per variant in `_canonical_order(variants)`.
    """
    reports: list[VariantReport] = []
    for name in _canonical_order(variants):
        mult = relative_cost.get(name, 1.0)
        yaml_path = Path(configs_dir) / "experiment" / f"{name}.yaml"

        if not yaml_path.is_file():
            reports.append(
                VariantReport(
                    name=name,
                    status="missing_config",
                    detail="MISSING CONFIG (not runnable yet)",
                    epochs=None,
                    relative_cost=mult,
                    cost=None,
                )
            )
            continue

        try:
            cfg = compose_variant(configs_dir, name, data_root_dir)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            short_message = str(exc).strip().splitlines()[0][:200]
            reports.append(
                VariantReport(
                    name=name,
                    status="compose_failed",
                    detail=f"COMPOSE FAILED: {short_message}",
                    epochs=None,
                    relative_cost=mult,
                    cost=None,
                )
            )
            continue

        epochs = int(cfg.training.epochs)
        cost = estimate_variant_cost(
            n_train_cases=n_train,
            n_val_cases=n_val,
            batch_size=int(cfg.training.batch_size),
            epochs=epochs,
            val_interval=int(cfg.training.val_interval),
            max_hours=float(cfg.training.max_hours),
            sec_per_step=sec_per_step,
            sec_per_val_case=sec_per_val_case,
            relative_cost=mult,
        )
        reports.append(
            VariantReport(
                name=name,
                status="ok",
                detail=None,
                epochs=epochs,
                relative_cost=mult,
                cost=cost,
            )
        )
    return reports


def _reference_report(reports: Sequence[VariantReport]) -> VariantReport | None:
    """Returns the `ablation_full` row if present and `ok`, else `None`."""
    for r in reports:
        if r.name == "ablation_full" and r.status == "ok":
            return r
    return None


def _totals_block(reports: Sequence[VariantReport], weekly_budget_hours: float) -> list[str]:
    """Builds the shared totals + excluded-variant lines for both formats.

    Args:
        reports: All variant reports, in grid order.
        weekly_budget_hours: Weekly Kaggle GPU-hour budget for the "weeks at
            budget" line.

    Returns:
        Plain text lines (no markdown/text-specific formatting needed here).
    """
    ok_reports = [r for r in reports if r.status == "ok" and r.cost is not None]
    excluded = [r for r in reports if r.status != "ok"]

    total_hours = sum(r.cost.total_hours for r in ok_reports)  # type: ignore[union-attr]
    total_sessions = sum(r.cost.sessions for r in ok_reports)  # type: ignore[union-attr]
    weeks = total_hours / weekly_budget_hours

    lines = [
        f"Total estimated GPU hours (composable variants only): {total_hours:.1f} h",
        f"Total estimated Kaggle sessions: {total_sessions}",
        f"Weeks at {weekly_budget_hours:.1f} h/week budget: {weeks:.1f}",
    ]

    if excluded:
        names = ", ".join(r.name for r in excluded)
        lines.append("")
        lines.append(
            f"NOTE: {len(excluded)} of {len(reports)} variant(s) excluded from the totals "
            f"above (not composable yet): {names}"
        )
        reference = _reference_report(reports)
        if reference is not None and reference.cost is not None:
            hypothetical_hours = total_hours + len(excluded) * reference.cost.total_hours
            lines.append(
                f"If each excluded variant cost the same as {reference.name} "
                f"({reference.cost.total_hours:.1f} h), grid total would be "
                f"{hypothetical_hours:.1f} h "
                f"({hypothetical_hours / weekly_budget_hours:.1f} weeks at budget)."
            )
        else:
            lines.append(
                "Reference variant ablation_full did not compose either, so no "
                "hypothetical total-with-excluded-variants is available."
            )

    return lines


def _calibration_banner(reports: Sequence[VariantReport]) -> list[str]:
    """Builds the top-of-report calibration warning.

    Args:
        reports: All variant reports, used only to surface a measured
            `steps_per_epoch` value so the recompute instruction is trivial.

    Returns:
        Plain text lines.
    """
    reference = _reference_report(reports)
    if reference is None:
        reference = next((r for r in reports if r.status == "ok"), None)
    steps_per_epoch = reference.cost.steps_per_epoch if reference and reference.cost else None

    lines = [
        "=" * 78,
        "ESTIMATES ONLY. No GPU run has happened yet for this grid.",
        "--sec-per-step defaults to an UNMEASURED guess (1.1s/step, Kaggle T4, AMP, "
        "4 patches of 96^3).",
        "After the first real training epoch, recompute with:",
        "  --sec-per-step <measured epoch wall-clock seconds / steps_per_epoch>",
    ]
    if steps_per_epoch is not None:
        lines.append(f"  steps_per_epoch for this grid: {steps_per_epoch}")
    else:
        lines.append("  (steps_per_epoch unavailable: no variant composed)")
    lines.append("=" * 78)
    return lines


def _commands_block(reports: Sequence[VariantReport], prep_dir: str) -> list[str]:
    """Builds the ordered `scripts/train.py` command list for runnable variants.

    Args:
        reports: All variant reports, in grid order.
        prep_dir: Value substituted for `data.root_dir=` in each command.

    Returns:
        Plain text lines.
    """
    lines = [
        "Commands (grid order, runnable variants only). The SAME command resumes "
        "an interrupted run -- scripts/train.py auto-finds last.pt in the "
        "checkpoint dir -- so a variant needing N sessions means running its line "
        "N times.",
        "",
    ]
    for r in reports:
        if r.status != "ok":
            continue
        sessions_note = f"  # ~{r.cost.sessions} session(s)" if r.cost else ""
        command = f"python scripts/train.py +experiment={r.name} data.root_dir={prep_dir}"
        lines.append(command + sessions_note)
    return lines


def format_report_text(
    reports: Sequence[VariantReport],
    weekly_budget_hours: float,
    prep_dir: str,
    n_train: int,
    n_val: int,
    counts_assumed: bool,
) -> str:
    """Renders the full plain-text ablation grid report.

    Args:
        reports: Variant reports, in grid order.
        weekly_budget_hours: Weekly Kaggle GPU-hour budget.
        prep_dir: Value substituted into the emitted `data.root_dir=` commands.
        n_train: Training case count used for every estimate.
        n_val: Validation case count used for every estimate.
        counts_assumed: Whether `n_train`/`n_val` came from CLI fallbacks
            rather than a real `splits.yaml`.

    Returns:
        The full report as one string, ready to `print`.
    """
    lines: list[str] = []
    lines.extend(_calibration_banner(reports))
    lines.append("")

    counts_suffix = (
        " (ASSUMED, not read from splits.yaml -- pass --n-train/--n-val explicitly to "
        "silence this)"
        if counts_assumed
        else " (read from splits.yaml)"
    )
    lines.append(f"Case counts: n_train={n_train}, n_val={n_val}" + counts_suffix)
    lines.append("")

    header = f"{'variant':<32} {'epochs':>6}  {'status':<40} {'est. hours':>11} {'sessions':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in reports:
        epochs_str = str(r.epochs) if r.epochs is not None else "-"
        hours_str = f"{r.cost.total_hours:.2f}" if r.cost else "-"
        sessions_str = str(r.cost.sessions) if r.cost else "-"
        status_str = r.status if r.status == "ok" else (r.detail or r.status)
        lines.append(
            f"{r.name:<32} {epochs_str:>6}  {status_str:<40} {hours_str:>11} {sessions_str:>9}"
        )
    lines.append("")
    lines.extend(_totals_block(reports, weekly_budget_hours))
    lines.append("")
    lines.extend(_commands_block(reports, prep_dir))
    return "\n".join(lines)


def format_report_markdown(
    reports: Sequence[VariantReport],
    weekly_budget_hours: float,
    prep_dir: str,
    n_train: int,
    n_val: int,
    counts_assumed: bool,
) -> str:
    """Renders the full ablation grid report as Markdown.

    Same content as `format_report_text`, formatted as a Markdown table plus
    a fenced code block for the commands, so it can be pasted directly into
    `docs/experiments.md`.

    Args: see `format_report_text`.

    Returns:
        The full report as one Markdown string.
    """
    lines: list[str] = []
    lines.append("```")
    lines.extend(_calibration_banner(reports))
    lines.append("```")
    lines.append("")

    counts_note = (
        "(ASSUMED, not read from splits.yaml)" if counts_assumed else "(read from splits.yaml)"
    )
    lines.append(f"Case counts: n_train={n_train}, n_val={n_val} {counts_note}")
    lines.append("")

    lines.append("| variant | epochs | status | est. hours | sessions |")
    lines.append("|---|---:|---|---:|---:|")
    for r in reports:
        epochs_str = str(r.epochs) if r.epochs is not None else "-"
        hours_str = f"{r.cost.total_hours:.2f}" if r.cost else "-"
        sessions_str = str(r.cost.sessions) if r.cost else "-"
        status_str = r.status if r.status == "ok" else (r.detail or r.status)
        lines.append(f"| {r.name} | {epochs_str} | {status_str} | {hours_str} | {sessions_str} |")
    lines.append("")
    lines.extend(_totals_block(reports, weekly_budget_hours))
    lines.append("")
    lines.append("```")
    lines.extend(_commands_block(reports, prep_dir))
    lines.append("```")
    return "\n".join(lines)


def parse_relative_cost_overrides(raw: Sequence[str]) -> dict[str, float]:
    """Parses `--relative-cost NAME=FLOAT` entries onto the default table.

    Args:
        raw: Repeated `--relative-cost` values, each `"NAME=FLOAT"`.

    Returns:
        A copy of `DEFAULT_RELATIVE_COST` with the named entries overridden.

    Raises:
        ValueError: If an entry is malformed or names an unknown variant.
    """
    overrides = dict(DEFAULT_RELATIVE_COST)
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--relative-cost expects NAME=FLOAT, got {item!r}")
        name, _, value_str = item.partition("=")
        name = name.strip()
        if name not in overrides:
            raise ValueError(
                f"Unknown ablation variant {name!r} for --relative-cost. "
                f"Known variants: {', '.join(GRID_VARIANTS)}"
            )
        overrides[name] = float(value_str)
    return overrides


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments for the ablation grid planner.

    Plain `argparse`, not Hydra -- this is a planning tool that composes
    Hydra configs internally, it is not itself a Hydra entry point.

    Args:
        argv: Argument list, or `None` to use `sys.argv[1:]`.

    Returns:
        The parsed `argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sec-per-step",
        type=float,
        default=1.1,
        help="Seconds per optimizer step for ablation_full on a Kaggle T4, AMP, "
        "4 patches of 96^3 (UNMEASURED default; recalibrate after the first real epoch).",
    )
    parser.add_argument(
        "--sec-per-val-case",
        type=float,
        default=3.0,
        help="Seconds of sliding-window validation per val case for ablation_full.",
    )
    parser.add_argument(
        "--relative-cost",
        action="append",
        default=[],
        metavar="NAME=FLOAT",
        help="Override one variant's relative-cost multiplier. Repeatable.",
    )
    parser.add_argument("--weekly-budget-hours", type=float, default=30.0)
    parser.add_argument(
        "--prep-dir",
        type=str,
        default="$PREP_DIR",
        help="Substituted into the emitted data.root_dir= argument. A literal shell "
        "variable by default, so no machine-specific path is ever baked in.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Fallback train case count, used only if splits.yaml is unreadable.",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=None,
        help="Fallback val case count, used only if splits.yaml is unreadable.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=list(GRID_VARIANTS),
        help="Restrict the grid to a subset of variants (canonical order is always kept).",
    )
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=_DEFAULT_CONFIGS_DIR,
        help="Path to the configs/ directory (override for tests).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: builds and prints the ablation grid report.

    Args:
        argv: Argument list, or `None` to use `sys.argv[1:]`.

    Returns:
        0 normally, 1 if any variant's config raised while composing (a real
        defect) or if case counts could not be resolved at all.
    """
    args = parse_args(argv)
    configs_dir = Path(args.configs_dir)
    variants = args.variants if args.variants is not None else list(GRID_VARIANTS)

    try:
        relative_cost = parse_relative_cost_overrides(args.relative_cost)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    splits_path = _resolve_splits_path(configs_dir, args.prep_dir)
    try:
        n_train, n_val, counts_assumed = load_case_counts(splits_path, args.n_train, args.n_val)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    reports = build_variant_reports(
        configs_dir=configs_dir,
        variants=variants,
        data_root_dir=args.prep_dir,
        n_train=n_train,
        n_val=n_val,
        sec_per_step=args.sec_per_step,
        sec_per_val_case=args.sec_per_val_case,
        relative_cost=relative_cost,
    )

    formatter = format_report_markdown if args.format == "markdown" else format_report_text
    print(
        formatter(
            reports,
            args.weekly_budget_hours,
            args.prep_dir,
            n_train,
            n_val,
            counts_assumed,
        )
    )

    any_compose_failed = any(r.status == "compose_failed" for r in reports)
    return 1 if any_compose_failed else 0


if __name__ == "__main__":
    sys.exit(main())
