"""Can a label-free read-out of the fusion gate predict a case's own Dice?

**Status: exploratory.** The pre-registered Gate 1 test
(`docs/research/preregistration_ambiguity.md`, run by
`scripts/detection_stats.py`) is about the inter-branch DISAGREEMENT map over
the whole volume. This script asks a cheaper, adjacent question about the
FUSION GATE, from the single tumour-centred patch `scripts/extract_gates.py`
already writes. Nothing here was registered in advance, the family is the
whole table below, and Holm is applied across all of it. Read any number here
as a hypothesis for the registered whole-volume run to confirm or refute, not
as a confirmed result.

## What makes this a detector rather than a curiosity

Every feature is computed from the model's OWN output -- the gate maps and the
segmentation logits saved beside them -- and never from the ground truth. A
score that needs the label is not a failure detector; it is a metric. This
project has already shipped one bug where a reporting mask was built from the
label and manufactured 41-57% of a reported ECE, so the rule is structural
here: no function below takes a label argument.

One caveat the code cannot enforce and the reader must carry:
`scripts/extract_gates.py` chooses WHERE to crop, and its `center_on: label`
mode uses the ground-truth centroid. A run in that mode leaks the label into
crop SELECTION even though no feature reads it. Use `center_on: prediction`
for any run whose numbers are meant to support a deployment claim; this
script records which mode produced each cohort by reading the extraction's
own `gates_config.yaml` when it is present, and warns when it is not.

## The three controls, and why each is there

- **Predictive entropy** in the predicted foreground. The single-pass
  uncertainty any model, including a plain U-Net, produces for free. A gate
  feature that merely reproduces it is worth nothing to the argument.
- **log predicted tumour volume.** Dice rises with lesion size, and a gate
  whose value separates tumour from background will track how much of the
  patch is tumour. Without this control, "the gate predicts Dice" could be
  "big tumours score well", which is already known.
- **Mean predicted whole-tumour probability.** The soft version of the same
  confound, catching what the discretized volume misses.

Partial correlations are rank-residual: both sides are ranked, a linear fit on
the ranks of every control is removed, and Spearman is taken on the residuals
-- monotone-invariant, and the same idea `analysis.detection.residualised_auroc`
uses at voxel level.

Run:

    python scripts/gate_failure_detection.py \\
        --cohort brats_test:outputs/gatespred_test:outputs/neurovision/eval_test \\
        --cohort ssa:outputs/gatespred_ssa:outputs/eval_ssa_neurovision \\
        --cohort ped:outputs/gatespred_ped:outputs/eval_ped_neurovision
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from neurovision.analysis.detection import _entropy_from_logits
from neurovision.analysis.statistics import holm_bonferroni
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

WT_CHANNEL = 2
_SUBSAMPLE = 60000


def case_features(path: Path, generator: np.random.Generator) -> dict[str, float]:
    """Label-free per-case features from one gate-extraction `.npz`.

    Args:
        path: A `<case_id>.npz` written by `scripts/extract_gates.py`, holding
            `gate_level_*` plus the segmentation `logits` for the same patch.
        generator: Seeded generator for the voxel subsample used by the
            gate-versus-prediction agreement feature.

    Returns:
        A flat dict of features and controls. Gate features per fusion level
        `L`: `gate{L}_mean` (whole patch), `gate{L}_fg` / `gate{L}_bg` (inside
        and outside the PREDICTED whole tumour), and `gate{L}_agree`, the
        voxel-wise Spearman correlation between that level's gate map and the
        predicted whole-tumour probability. `gate{L}_agree` is deliberately
        SIGNED and not absolute: its sign is a fixed property of the level
        (level 1's gate opens inside the tumour, level 2's opens outside it),
        so taking a magnitude would fold two opposite mechanisms together.
        Controls: `ent_fg`, `log_pred_vol`, `prob_wt_mean`.
    """
    payload = np.load(path)
    logits = payload["logits"].astype(np.float32)
    prob = expit(logits)
    entropy = _entropy_from_logits(logits)
    foreground = prob[WT_CHANNEL] > 0.5

    row: dict[str, float] = {
        "ent_fg": float(entropy[:, foreground].mean()) if foreground.any() else float("nan"),
        "log_pred_vol": float(np.log1p(foreground.sum())),
        "prob_wt_mean": float(prob[WT_CHANNEL].mean()),
    }

    edge = logits.shape[1]
    flat_prob = prob[WT_CHANNEL].reshape(-1)
    take = generator.choice(flat_prob.size, size=min(_SUBSAMPLE, flat_prob.size), replace=False)

    level = 0
    while f"gate_level_{level}" in payload:
        gate = payload[f"gate_level_{level}"][0].astype(np.float32)
        factor = edge // gate.shape[0]
        upsampled = np.kron(gate, np.ones((factor, factor, factor), np.float32))
        row[f"gate{level}_mean"] = float(upsampled.mean())
        row[f"gate{level}_fg"] = float(upsampled[foreground].mean()) if foreground.any() else np.nan
        row[f"gate{level}_bg"] = (
            float(upsampled[~foreground].mean()) if (~foreground).any() else np.nan
        )
        sampled_gate = upsampled.reshape(-1)[take]
        row[f"gate{level}_agree"] = (
            float(stats.spearmanr(sampled_gate, flat_prob[take]).statistic)
            if sampled_gate.std() > 0
            else float("nan")
        )
        level += 1
    return row


def rank_partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    *,
    generator: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float, float, int]:
    """Spearman of `x` against `y` with a rank-linear fit on `controls` removed.

    Args:
        x: `(n,)` feature values.
        y: `(n,)` target values, same case order.
        controls: `(n, k)` confounds to partial out, same case order.
        generator: Seeded generator for the case-index bootstrap.
        n_boot: Bootstrap replicate count.

    Returns:
        `(rho_partial, ci_lo, ci_hi, p_two_sided, n)`. Cases are dropped
        pairwise-complete across `x`, `y` and every control together, never
        per-variable -- masking each independently would correlate different
        subsets of cases against each other.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = np.atleast_2d(np.asarray(controls, dtype=float))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    x, y, controls = x[keep], y[keep], controls[keep]
    n = int(x.size)

    def statistic(index: np.ndarray) -> float:
        rank_x = stats.rankdata(x[index])
        rank_y = stats.rankdata(y[index])
        design = np.column_stack(
            [np.ones(index.size)]
            + [stats.rankdata(controls[index, j]) for j in range(controls.shape[1])]
        )
        residual_x = rank_x - design @ np.linalg.lstsq(design, rank_x, rcond=None)[0]
        residual_y = rank_y - design @ np.linalg.lstsq(design, rank_y, rcond=None)[0]
        if residual_x.std() == 0 or residual_y.std() == 0:
            return float("nan")
        return float(stats.spearmanr(residual_x, residual_y).statistic)

    point = statistic(np.arange(n))
    replicates = np.array([statistic(generator.integers(0, n, n)) for _ in range(n_boot)])
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        return point, float("nan"), float("nan"), float("nan"), n
    lo, hi = (float(v) for v in np.percentile(replicates, [2.5, 97.5]))
    tail = min(float((replicates <= 0).mean()), float((replicates >= 0).mean()))
    pvalue = float(np.clip(2.0 * tail, 1.0 / max(replicates.size, 1), 1.0))
    return point, lo, hi, pvalue, n


def crop_mode(gate_dir: Path) -> str:
    """Reads which centring mode produced a gate directory.

    Args:
        gate_dir: A `scripts/extract_gates.py` output directory.

    Returns:
        `"label"`, `"prediction"`, or `"unknown"` when no resolved config was
        written beside the maps. `"label"` means the crop position was chosen
        using the ground truth, which is a leak into crop selection even
        though no feature reads the label.
    """
    for name in ("gates_config.yaml", "explainability_config.yaml"):
        path = gate_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for mode in ("prediction", "label"):
                if f"center_on: {mode}" in text:
                    return mode
    return "unknown"


def build_cohort(
    name: str, gate_dir: Path, eval_dir: Path, target: str, generator: np.random.Generator
) -> pd.DataFrame:
    """Joins per-case gate features to the run's published per-case metrics.

    Args:
        name: Cohort label used in the output tables.
        gate_dir: Gate-extraction directory for this cohort.
        eval_dir: Matching evaluation directory (for `per_case_metrics.csv`).
        target: Per-case metric column to predict.
        generator: Seeded generator, forwarded to `case_features`.

    Returns:
        A DataFrame indexed by `case_id`.

    Raises:
        FileNotFoundError: The gate directory holds no `.npz`, or the
            evaluation directory has no `per_case_metrics.csv`.
    """
    paths = sorted(gate_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No gate maps under {gate_dir.resolve()}.")
    metrics_path = eval_dir / "per_case_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No per_case_metrics.csv under {eval_dir.resolve()}.")
    metrics = pd.read_csv(metrics_path).set_index("case_id")

    records = {p.stem: case_features(p, generator) for p in paths if p.stem in metrics.index}
    frame = pd.DataFrame.from_dict(records, orient="index").join(metrics[[target]])
    frame.index.name = "case_id"
    logger.info(
        "cohort %r: %d cases joined (%d gate maps, %d metric rows), crop centring = %s",
        name,
        len(frame),
        len(paths),
        len(metrics),
        crop_mode(gate_dir),
    )
    return frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Argument list, or `None` to read `sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        metavar="NAME:GATE_DIR:EVAL_DIR",
        help="Repeatable. Colon-separated cohort name, gate directory, evaluation directory.",
    )
    parser.add_argument("--target", default="dice_mean")
    parser.add_argument("--out", type=Path, default=Path("outputs/gate_detection"))
    parser.add_argument("--n-boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Builds every cohort's table, applies Holm over the whole family, prints it.

    Args:
        argv: Argument list, or `None` to read `sys.argv`.

    Returns:
        Process exit code.
    """
    setup_logging(level="INFO")
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    for spec in args.cohort:
        name, gate_dir, eval_dir = spec.split(":", 2)
        frame = build_cohort(name, Path(gate_dir), Path(eval_dir), args.target, generator)
        frame.to_csv(args.out / f"features_{name}.csv")

        controls = frame[["ent_fg", "log_pred_vol", "prob_wt_mean"]].to_numpy()
        features = [c for c in frame.columns if c.startswith("gate")]
        # The entropy baseline is reported on the same footing as the gate
        # features, controlling for the two volume confounds only -- it cannot
        # control for itself.
        point, lo, hi, pvalue, n = rank_partial_spearman(
            frame["ent_fg"].to_numpy(),
            frame[args.target].to_numpy(),
            frame[["log_pred_vol", "prob_wt_mean"]].to_numpy(),
            generator=generator,
            n_boot=args.n_boot,
        )
        rows.append(
            dict(
                cohort=name,
                crop=crop_mode(Path(gate_dir)),
                feature="ent_fg (baseline)",
                n=n,
                rho_partial=point,
                ci_lo=lo,
                ci_hi=hi,
                p_raw=pvalue,
            )
        )
        for feature in features:
            point, lo, hi, pvalue, n = rank_partial_spearman(
                frame[feature].to_numpy(),
                frame[args.target].to_numpy(),
                controls,
                generator=generator,
                n_boot=args.n_boot,
            )
            rows.append(
                dict(
                    cohort=name,
                    crop=crop_mode(Path(gate_dir)),
                    feature=feature,
                    n=n,
                    rho_partial=point,
                    ci_lo=lo,
                    ci_hi=hi,
                    p_raw=pvalue,
                )
            )

    table = pd.DataFrame(rows)
    # Holm over the WHOLE table, including the entropy baseline rows: the
    # family is everything reported, fixed by this script's structure rather
    # than chosen after the p-values were seen.
    reject, adjusted = holm_bonferroni(table["p_raw"].to_numpy(), alpha=args.alpha)
    table["p_holm"] = adjusted
    table["reject"] = reject
    table["ci_excludes_zero"] = (table["ci_lo"] * table["ci_hi"]) > 0
    out_path = args.out / f"gate_detection_{args.target}.csv"
    table.to_csv(out_path, index=False)

    print("=" * 100)
    print(
        f"Label-free gate read-out vs per-case {args.target}"
        " -- EXPLORATORY, Holm over the whole table"
    )
    print(
        "partial Spearman controls: predictive entropy, log predicted volume, mean WT probability"
    )
    print("=" * 100)
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
