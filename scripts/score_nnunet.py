"""Hydra entry point that scores nnU-Net's hard-label predictions through our own metric path.

Gate A (`docs/research/preregistration_strong_baseline.md`, Amendment 1
item 8 "Scoring convention") compares our model against nnU-Net -- but
nnU-Net's own prediction files disagree with ours on grid, label
convention and format (hard labels vs logits). The pure functions that
convert one nnU-Net prediction into something `replay_case` can score are
already implemented and tested in `neurovision.analysis.nnunet_import`
(`import_prediction`, `score_imported_case`, `roundtrip_self_test`). This
script is only the thin I/O driver that reads a split, loops over its
cases, and writes the result files every other analysis script in this
project writes.

## Why the round-trip self-test runs before any prediction is scored

The pre-registration is explicit: nnU-Net's predictions must not be scored
until a ground-truth round trip through the SAME import path
(`roundtrip_self_test`) has been shown to be lossless for at least one
case. A missed label swap or a mis-oriented grid would otherwise produce a
plausible-looking but wrong Dice for every case, silently. Running this
check first, and letting it raise before a single real prediction is
touched, is cheaper than discovering the same bug after scoring 189 cases.

## Why a partial prediction set is refused, not scored partially

Scoring only the predictions that happen to exist would let a truncated or
still-running nnU-Net export produce a "result" that quietly excludes its
hardest cases. This script demands every case in the requested split have
a matching `<case_id>.nii.gz` before it scores anything, and raises,
naming the missing ids, if not. An unexpected extra file in `pred_dir`
(e.g. a case from a different split) is not an error -- it is logged and
ignored, so a `pred_dir` shared across scoring runs does not need to be
curated by hand.

Example usage (Gate A, test split):

    python scripts/score_nnunet.py \\
        analysis.score_nnunet.pred_dir=outputs/nnunet_predictions/test \\
        analysis.score_nnunet.out_dir=outputs/score_nnunet/test
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from neurovision.analysis.nnunet_import import (
    import_prediction,
    roundtrip_self_test,
    score_imported_case,
)
from neurovision.data.dataset import load_splits
from neurovision.metrics.lesionwise import require_panoptica
from neurovision.utils.io import ensure_dir, read_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def resolve_lesionwise(score_cfg: DictConfig) -> dict[str, Any] | None:
    """Reads and validates `cfg.analysis.score_nnunet.lesionwise`.

    Mirrors `scripts/replay_logits.py`'s `resolve_lesionwise` exactly --
    same four settings, same range checks -- just read off
    `cfg.analysis.score_nnunet.lesionwise` instead.

    Args:
        score_cfg: `cfg.analysis.score_nnunet`.

    Returns:
        A plain dict of the four keyword arguments `score_imported_case`'s
        `lesionwise` argument forwards to `replay_case`
        (`min_lesion_voxels`, `matching_threshold`, `nsd_tolerance_mm`,
        `connectivity`), or `None` when the block is absent, is `None`, or
        has `enabled: false`.

    Raises:
        ValueError: If `min_lesion_voxels` is negative, `matching_threshold`
            is not in `(0, 1]`, `nsd_tolerance_mm` is not positive, or
            `connectivity` is not one of `{6, 18, 26}`.
    """
    raw = score_cfg.get("lesionwise", None)
    if raw is None or not raw.get("enabled", False):
        return None

    min_lesion_voxels = int(raw.get("min_lesion_voxels", 50))
    matching_threshold = float(raw.get("matching_threshold", 0.5))
    nsd_tolerance_mm = float(raw.get("nsd_tolerance_mm", 1.0))
    connectivity = int(raw.get("connectivity", 26))

    if min_lesion_voxels < 0:
        raise ValueError(
            f"analysis.score_nnunet.lesionwise.min_lesion_voxels must be >= 0, got "
            f"{min_lesion_voxels}."
        )
    if not (0 < matching_threshold <= 1):
        raise ValueError(
            "analysis.score_nnunet.lesionwise.matching_threshold must be in (0, 1], got "
            f"{matching_threshold}."
        )
    if nsd_tolerance_mm <= 0:
        raise ValueError(
            "analysis.score_nnunet.lesionwise.nsd_tolerance_mm must be > 0, got "
            f"{nsd_tolerance_mm}."
        )
    if connectivity not in (6, 18, 26):
        raise ValueError(
            "analysis.score_nnunet.lesionwise.connectivity must be one of {6, 18, 26}, got "
            f"{connectivity}."
        )

    return {
        "min_lesion_voxels": min_lesion_voxels,
        "matching_threshold": matching_threshold,
        "nsd_tolerance_mm": nsd_tolerance_mm,
        "connectivity": connectivity,
    }


def _run_selftest(prep_dir: Path, selftest_case: str, target_axcodes: tuple[str, str, str]) -> None:
    """Runs the pre-registered ground-truth round trip before any real case is scored.

    Args:
        prep_dir: Root of the preprocessed tree.
        selftest_case: Case id to run the round trip on. Must have both
            NCR (label 1) and ED (label 2) present -- see
            `neurovision.analysis.nnunet_import.roundtrip_self_test`.
        target_axcodes: Voxel axis codes cases were normalized to.

    Raises:
        FileNotFoundError: If `selftest_case`'s `label.npy` or `meta.json`
            is missing from `prep_dir`.
        ValueError: If `selftest_case` lacks NCR or ED (propagated from
            `roundtrip_self_test`).
        AssertionError: If the round trip is not lossless (propagated from
            `roundtrip_self_test`).
    """
    case_dir = prep_dir / selftest_case
    label_path = case_dir / "label.npy"
    meta_path = case_dir / "meta.json"
    if not label_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"score_nnunet: selftest_case {selftest_case!r} is missing label.npy or meta.json "
            f"under {case_dir}. The pre-registered ground-truth round trip cannot run without "
            "it, and no nnU-Net prediction may be scored until that round trip passes."
        )
    label_cropped = np.load(label_path)
    meta = read_json(meta_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        roundtrip_self_test(label_cropped, meta, target_axcodes, tmp_dir)
    logger.info("score_nnunet: ground-truth round-trip self-test passed for %s.", selftest_case)


def _check_completeness(case_ids: Sequence[str], pred_dir: Path) -> None:
    """Requires every case in the split to have a matching prediction file.

    Args:
        case_ids: The full list of cases the split names.
        pred_dir: Folder of nnU-Net predictions, `<case_id>.nii.gz` each.

    Raises:
        ValueError: If any `case_id` in `case_ids` has no matching file --
            names every missing id, so a truncated export is never scored
            as if it were complete.
    """
    missing = [case_id for case_id in case_ids if not (pred_dir / f"{case_id}.nii.gz").is_file()]
    if missing:
        raise ValueError(
            f"score_nnunet: {len(missing)} case(s) in the requested split have no matching "
            f"prediction under {pred_dir}: {missing}. Refusing to score a partial set -- see "
            "this script's module docstring for why."
        )

    expected_names = {f"{case_id}.nii.gz" for case_id in case_ids}
    extra = sorted(p.name for p in pred_dir.glob("*.nii.gz") if p.name not in expected_names)
    if extra:
        logger.warning(
            "score_nnunet: %d file(s) in %s are not in the requested split and are ignored: %s",
            len(extra),
            pred_dir,
            extra,
        )


def _score_one_case(
    case_id: str,
    pred_dir: Path,
    prep_dir: Path,
    target_axcodes: tuple[str, str, str],
    lesionwise: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Imports and scores one case's nnU-Net prediction.

    Args:
        case_id: The case identifier.
        pred_dir: Folder of nnU-Net predictions.
        prep_dir: Root of the preprocessed tree (ground truth + spacing).
        target_axcodes: Voxel axis codes cases were normalized to.
        lesionwise: Forwarded to `score_imported_case`.

    Returns:
        The flat metric dict `score_imported_case` returns.

    Raises:
        ValueError: Propagated from `import_prediction` if the prediction's
            geometry does not match this case's `meta.json` (see that
            function's docstring -- the geometry gate that catches a
            mirrored or otherwise mis-oriented file).
    """
    case_dir = prep_dir / case_id
    meta = read_json(case_dir / "meta.json")
    label_gt = np.load(case_dir / "label.npy")
    pred_path = pred_dir / f"{case_id}.nii.gz"

    pred = import_prediction(pred_path, meta, target_axcodes)
    spacing = tuple(float(s) for s in meta["spacing"])
    return score_imported_case(pred, label_gt, spacing=spacing, lesionwise=lesionwise)


def score_predictions(
    case_ids: Sequence[str],
    pred_dir: str | Path,
    prep_dir: str | Path,
    target_axcodes: tuple[str, str, str],
    lesionwise: Mapping[str, Any] | None,
    selftest_case: str,
    out_dir: str | Path,
) -> pd.DataFrame:
    """Scores every case's nnU-Net prediction and writes the per-case and summary tables.

    Runs, in order: the pre-registered ground-truth round-trip self-test
    (must pass before anything else runs), a completeness check over
    `case_ids` (raises on any missing prediction), then one
    import-and-score pass per case.

    Args:
        case_ids: Cases to score (typically one split's case list).
        pred_dir: Folder of nnU-Net predictions, `<case_id>.nii.gz` each,
            on the original raw BraTS grid.
        prep_dir: Root of the preprocessed tree (`<case>/label.npy` +
            `<case>/meta.json`).
        target_axcodes: Voxel axis codes cases were normalized to, e.g.
            `("L", "P", "S")`.
        lesionwise: `None` to skip lesion-wise metrics, or a mapping of
            keyword arguments forwarded to `score_imported_case`.
        selftest_case: Case id the round-trip self-test runs on. Must have
            both NCR and ED present.
        out_dir: Directory `per_case_metrics.csv` and `summary.csv` are
            written into. Created if missing.

    Returns:
        A DataFrame indexed by `case_id`, one column per metric key,
        sorted by `case_id`. Identical to what was written to
        `per_case_metrics.csv`.

    Raises:
        FileNotFoundError: See `_run_selftest`.
        ValueError: See `_run_selftest` (missing NCR/ED) and
            `_check_completeness` (a partial prediction set).
        AssertionError: See `_run_selftest` (a broken round trip).
    """
    pred_dir = Path(pred_dir)
    prep_dir = Path(prep_dir)
    out_dir = ensure_dir(out_dir)

    logger.info(
        "score_nnunet: running the ground-truth round-trip self-test on %r before scoring "
        "anything.",
        selftest_case,
    )
    _run_selftest(prep_dir, selftest_case, target_axcodes)

    sorted_ids = sorted(case_ids)
    _check_completeness(sorted_ids, pred_dir)

    records: dict[str, dict[str, float]] = {}
    for case_id in tqdm(sorted_ids, desc="Scoring nnU-Net predictions"):
        records[case_id] = _score_one_case(case_id, pred_dir, prep_dir, target_axcodes, lesionwise)

    table = pd.DataFrame.from_dict(records, orient="index").rename_axis("case_id")

    per_case_path = out_dir / "per_case_metrics.csv"
    table.to_csv(per_case_path)
    logger.info("score_nnunet: wrote %s (%d case(s))", per_case_path, len(table))

    # One-row summary: the mean of every numeric metric column, across all
    # scored cases. Deliberately not MetricAggregator.summary()'s
    # metric-indexed, multi-statistic table (used by scripts/evaluate.py) --
    # this script's summary is a single row so it can be concatenated
    # across several nnU-Net scoring runs (e.g. test/SSA/PED) with one
    # `pd.concat` and no reshaping.
    summary = table.select_dtypes(include="number").mean().to_frame().T
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("score_nnunet: wrote %s", summary_path)

    return table


def _print_summary(table: pd.DataFrame, pred_dir: Path, out_dir: Path) -> None:
    """Prints (not logs -- matches `scripts/compare_family.py`'s convention) a compact summary.

    Args:
        table: The per-case table `score_predictions` returned.
        pred_dir: The nnU-Net prediction folder that was scored.
        out_dir: Where the result files were written.
    """
    lines = [
        "=" * 70,
        f"score_nnunet: pred_dir={pred_dir}",
        "=" * 70,
        f"  out_dir: {out_dir}",
        f"  cases scored: {len(table)}",
    ]
    for metric in ("dice_ET", "dice_TC", "dice_WT", "dice_mean"):
        if metric in table.columns:
            lines.append(f"  {metric}: mean={float(table[metric].mean()):.4f}")
    print("\n".join(lines))  # print, not logger -- see convention note above


def run_score_nnunet(cfg: DictConfig) -> pd.DataFrame:
    """Resolves `cfg.analysis.score_nnunet` and runs `score_predictions`.

    Writes `per_case_metrics.csv`, `summary.csv`, and `score_config.yaml`
    (the fully resolved config) into the resolved `out_dir`.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The per-case metrics table `score_predictions` returned.

    Raises:
        ValueError: If `analysis.score_nnunet.pred_dir` or `.out_dir` is
            `None` (same null-forces-an-explicit-CLI-override convention as
            `analysis.replay.eval_dir`), or `.split` does not name one of
            the split file's `train`/`val`/`test` keys, or see
            `resolve_lesionwise` / `score_predictions`.
        ImportError: If lesion-wise scoring is enabled and `panoptica` is
            not installed in the current interpreter, raised before any
            case is scored.
    """
    score_cfg = cfg.analysis.score_nnunet

    if score_cfg.pred_dir is None:
        raise ValueError(
            "analysis.score_nnunet.pred_dir is not set. Point it at the folder of nnU-Net "
            "hard-label predictions (<case_id>.nii.gz on the original raw BraTS grid), e.g. "
            "'analysis.score_nnunet.pred_dir=outputs/nnunet_predictions/test'."
        )
    if score_cfg.out_dir is None:
        raise ValueError(
            "analysis.score_nnunet.out_dir is not set. Point it at a directory to write "
            "per_case_metrics.csv / summary.csv / score_config.yaml into, e.g. "
            "'analysis.score_nnunet.out_dir=outputs/score_nnunet/test'."
        )

    pred_dir = Path(str(score_cfg.pred_dir))
    prep_dir = Path(str(score_cfg.prep_dir))
    out_dir = ensure_dir(str(score_cfg.out_dir))

    splits = load_splits(str(score_cfg.splits_path))
    split = str(score_cfg.split)
    if split not in splits:
        raise ValueError(
            f"analysis.score_nnunet.split={split!r} is not one of the split file's keys "
            f"{sorted(splits)}."
        )
    case_ids = list(splits[split])

    target_axcodes = tuple(str(c) for c in cfg.data.preprocessing.target_axcodes)
    lesionwise_cfg = resolve_lesionwise(score_cfg)
    if lesionwise_cfg is not None:
        # Fail fast, before the self-test or a single case is scored --
        # same reasoning as scripts/replay_logits.py's own require_panoptica()
        # call.
        require_panoptica()

    logger.info(
        "score_nnunet: pred_dir=%s prep_dir=%s split=%s (%d case(s)) out_dir=%s",
        pred_dir,
        prep_dir,
        split,
        len(case_ids),
        out_dir,
    )

    table = score_predictions(
        case_ids=case_ids,
        pred_dir=pred_dir,
        prep_dir=prep_dir,
        target_axcodes=target_axcodes,
        lesionwise=lesionwise_cfg,
        selftest_case=str(score_cfg.selftest_case),
        out_dir=out_dir,
    )

    config_path = out_dir / "score_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    logger.info("score_nnunet: wrote %s", config_path)

    _print_summary(table, pred_dir, out_dir)

    return table


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Scores an nnU-Net prediction folder through our metric path, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_score_nnunet(cfg)


if __name__ == "__main__":
    main()
