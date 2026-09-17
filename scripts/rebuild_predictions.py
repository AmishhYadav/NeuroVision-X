"""Hydra entry point that rebuilds `predictions/` from an eval directory's saved `logits/`.

`predictions/` (`<eval_dir>/predictions/<case_id>.npy`, uint8 class maps
`{0,1,2,3}` in ORIGINAL, uncropped BraTS geometry) is a volume-sized cache,
not a result -- see `docs/reproducibility.md` §11. It was deliberately
deleted on 2026-08-19 to reclaim disk, and the demo viewer
(`app/backend/volumes.py::list_cases`) needs it back for any eval directory
someone wants to browse in the clinical/demo UI.

It is rebuilt from `<eval_dir>/logits/<case_id>.npy` (fp16 region logits,
shape `(3, D, H, W)`, CROPPED geometry, channel order ET/TC/WT --
`scripts/evaluate.py`'s `save_logits` option) by re-running exactly the
post-processing chain that run's OWN `<eval_dir>/eval_config.yaml` used --
never the project's current default, in case the two have since diverged.
This module deliberately calls the same private, already-tested building
blocks `neurovision.analysis.replay.replay_case` calls internally
(`_binarize_regions`, `_apply_postprocess_steps`) rather than re-implementing
any part of that sequence, so a rebuilt prediction is byte-identical to what
a full replay (or the original evaluation run itself) would have produced.

## Why every run verifies itself before anything is trusted

Like `scripts/replay_logits.py`'s self-consistency check, this script
recomputes Dice for a handful of cases from the rebuilt prediction and
compares it against the eval directory's own published
`per_case_metrics.csv`. This is the project's standing rule for any rebuild
(`docs/reproducibility.md` §11, CLAUDE.md's testing rules): a cache is only
trustworthy once it has been checked against something independently known,
never merely "the code ran without raising".

Example usage:

    python scripts/rebuild_predictions.py +rebuild.eval_dir=outputs/neurovision/eval_test
    python scripts/rebuild_predictions.py +rebuild.eval_dir=outputs/neurovision/eval_test \\
        +rebuild.limit=20 +rebuild.overwrite=true
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from neurovision.analysis.replay import (
    _apply_postprocess_steps,
    _binarize_regions,
    _resolve_postprocess_cfg,
    available_logit_cases,
    load_case_logits,
)
from neurovision.analysis.statistics import load_per_case
from neurovision.inference.postprocess import regions_to_classes, uncrop_to_original
from neurovision.metrics.segmentation import classes_to_regions, compute_case_metrics
from neurovision.utils.io import ensure_dir, read_json, read_yaml, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/evaluate.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# How often (in cases) progress is logged, matching
# neurovision.analysis.replay's own _LOG_EVERY.
_LOG_EVERY = 25

# Self-verification tolerance, in Dice units. `replay_logits.py` uses 1e-6
# for a replay that shares this same project's float32 arithmetic end to
# end; this script's docstring-mandated tolerance is looser (1e-4) because it
# is set by the task spec, not derived here -- generous enough to absorb any
# summation-order noise while still catching a real configuration mismatch
# (a different threshold or post-processing chain would move Dice by far
# more than 1e-4).
_VERIFY_TOLERANCE = 1e-4

# The only metrics the self-verification check compares, matching
# replay_logits.py's own _CONSISTENCY_METRICS reasoning: HD95 can legitimately
# differ in its NaN handling depending on how the crop/label bbox happens to
# align, which would make an "agree" check noisy for no reason. Dice alone is
# already dispositive.
_VERIFY_METRICS: tuple[str, ...] = ("dice_ET", "dice_TC", "dice_WT")


def _load_run_postprocess_cfg(eval_dir: Path) -> tuple[float | list[float], dict[str, Any]]:
    """Reads the discretization threshold and post-processing chain this run actually used.

    Args:
        eval_dir: An evaluation output directory as written by
            `scripts/evaluate.py`, holding its own `eval_config.yaml`.

    Returns:
        `(threshold, resolved_pp_cfg)` -- `threshold` exactly as recorded
        (a scalar, or a per-channel sequence), and `resolved_pp_cfg` the
        remaining post-processing settings (`enforce_nesting`,
        `min_component_size`, `connectivity`, `keep_largest_only`,
        `et_min_volume`) taken verbatim from that same file. Passing this
        full block through `_resolve_postprocess_cfg` (rather than reading
        the five keys directly) reuses that function's merge-over-default
        logic for free, which is a no-op here since every key is already
        present in a real `eval_config.yaml`.

    Raises:
        FileNotFoundError: If `<eval_dir>/eval_config.yaml` does not exist.
        KeyError: If that file has no `inference.postprocess` block.
    """
    eval_config_path = eval_dir / "eval_config.yaml"
    if not eval_config_path.is_file():
        raise FileNotFoundError(
            f"rebuild_predictions: no eval_config.yaml at {eval_config_path.resolve()}. This "
            "script needs the run's OWN recorded post-processing settings, not the project's "
            "current default -- run scripts/evaluate.py first."
        )
    raw_cfg = read_yaml(eval_config_path)
    pp_block = dict(raw_cfg["inference"]["postprocess"])
    threshold = pp_block.pop("threshold")
    resolved_pp_cfg = _resolve_postprocess_cfg(pp_block)
    return threshold, resolved_pp_cfg


def _rebuild_one_case(
    eval_dir: Path,
    prep_dir: Path,
    case_id: str,
    threshold: float | list[float],
    pp_cfg: dict[str, Any],
) -> np.ndarray:
    """Rebuilds one case's uint8 class map, in ORIGINAL uncropped geometry, from saved logits.

    Args:
        eval_dir: An evaluation output directory with `logits/` saved.
        prep_dir: Root directory of preprocessed cases (for `meta.json`'s
            `bbox` / `original_shape`).
        case_id: The case identifier.
        threshold: Forwarded to `_binarize_regions`.
        pp_cfg: Forwarded to `_apply_postprocess_steps`.

    Returns:
        `uint8` array, shape `meta["original_shape"]`.
    """
    logits = load_case_logits(eval_dir, case_id)  # (3, D, H, W) float32, cropped
    logits_t = torch.from_numpy(logits).unsqueeze(0)  # (1, 3, D, H, W)
    regions = _binarize_regions(logits_t, threshold)
    regions = _apply_postprocess_steps(regions, pp_cfg)
    classes = regions_to_classes(regions)[0].numpy().astype(np.uint8)  # (D, H, W), cropped

    meta = read_json(prep_dir / case_id / "meta.json")
    return uncrop_to_original(classes, meta["bbox"], meta["original_shape"])


def _verify_rebuilt_predictions(
    eval_dir: Path,
    prep_dir: Path,
    case_ids: list[str],
    *,
    verify_n: int,
) -> dict[str, Any] | None:
    """Self-verifies the first `verify_n` labeled cases against the run's own published Dice.

    For each of the first `verify_n` entries of `case_ids` that have a
    `label.npy` (in `prep_dir`) AND a row in `<eval_dir>/per_case_metrics.csv`,
    reads the just-rebuilt prediction back off disk, crops it back to the
    case's `bbox`, and recomputes `dice_ET`/`dice_TC`/`dice_WT` against the
    label -- then compares against the published row.

    Args:
        eval_dir: An evaluation output directory, expected to hold
            `predictions/` (just rebuilt) and its own `per_case_metrics.csv`.
        prep_dir: Root directory of preprocessed cases.
        case_ids: Case ids to consider, in the order they should be checked.
        verify_n: How many labeled, published cases to verify.

    Returns:
        `{"n_verified", "max_abs_delta", "per_case"}` -- written to
        `<eval_dir>/predictions_rebuild_verify.json` on success -- or `None`
        if no case in `case_ids` has both a `label.npy` and a published row
        (logged as a warning: the rebuild is then UNVERIFIED).

    Raises:
        ValueError: If any verified case's recomputed Dice differs from the
            published value by `>= _VERIFY_TOLERANCE` for any region. Names
            the case, the metric and both values -- this rebuild must not be
            trusted (or used) if it raises.
    """
    per_case_path = eval_dir / "per_case_metrics.csv"
    if not per_case_path.is_file():
        logger.warning(
            "rebuild_predictions: no per_case_metrics.csv at %s; skipping self-verification. "
            "The rebuilt predictions/ are therefore UNVERIFIED against anything.",
            per_case_path,
        )
        return None
    published = load_per_case(per_case_path)

    predictions_dir = eval_dir / "predictions"
    per_case: dict[str, dict[str, dict[str, float]]] = {}

    for case_id in case_ids:
        if len(per_case) >= verify_n:
            break
        label_path = prep_dir / case_id / "label.npy"
        if not label_path.is_file() or case_id not in published.index:
            continue

        meta = read_json(prep_dir / case_id / "meta.json")
        bbox = [tuple(pair) for pair in meta["bbox"]]
        crop_slices = tuple(slice(int(start), int(end)) for start, end in bbox)
        spacing = tuple(float(s) for s in meta["spacing"]) if "spacing" in meta else None

        full_pred = np.load(predictions_dir / f"{case_id}.npy")
        cropped_pred = full_pred[crop_slices]
        label = np.load(label_path)

        pred_regions = classes_to_regions(torch.from_numpy(cropped_pred.astype(np.int64)))
        label_regions = classes_to_regions(torch.from_numpy(label.astype(np.int64)))
        recomputed = compute_case_metrics(pred_regions, label_regions, spacing=spacing)

        case_result: dict[str, dict[str, float]] = {}
        for metric in _VERIFY_METRICS:
            rebuilt_value = float(recomputed[metric])
            published_value = float(published.loc[case_id, metric])
            delta = rebuilt_value - published_value
            case_result[metric] = {
                "rebuilt": rebuilt_value,
                "published": published_value,
                "delta": delta,
            }
            if abs(delta) >= _VERIFY_TOLERANCE:
                raise ValueError(
                    f"rebuild_predictions: self-verification FAILED for case {case_id!r}, "
                    f"metric {metric!r}: rebuilt={rebuilt_value}, published={published_value}, "
                    f"|delta|={abs(delta)} >= tolerance {_VERIFY_TOLERANCE}. The rebuilt "
                    f"predictions/ under {eval_dir} do not match its own published "
                    f"{per_case_path} -- do not trust or use them until this is resolved."
                )
        per_case[case_id] = case_result

    if not per_case:
        logger.warning(
            "rebuild_predictions: no case among %s had both a label.npy and a published "
            "per_case_metrics.csv row; skipping self-verification. The rebuilt predictions/ "
            "are therefore UNVERIFIED against anything.",
            case_ids[:verify_n],
        )
        return None

    max_abs_delta = max(
        abs(entry["delta"]) for case in per_case.values() for entry in case.values()
    )
    result: dict[str, Any] = {
        "n_verified": len(per_case),
        "max_abs_delta": max_abs_delta,
        "per_case": per_case,
    }
    write_json(result, eval_dir / "predictions_rebuild_verify.json")
    logger.info(
        "rebuild_predictions: self-verification passed for %d case(s), max |delta| = %.2e.",
        len(per_case),
        max_abs_delta,
    )
    return result


def rebuild_predictions(
    eval_dir: str | Path,
    prep_dir: str | Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    verify_n: int = 5,
) -> dict[str, Any]:
    """Reconstructs `<eval_dir>/predictions/<case>.npy` from `<eval_dir>/logits/<case>.npy`.

    Args:
        eval_dir: An evaluation output directory written by
            `scripts/evaluate.py` with `save_logits=true`, holding `logits/`,
            its own `eval_config.yaml`, and (for self-verification)
            `per_case_metrics.csv`.
        prep_dir: Root directory of preprocessed cases (for `meta.json` and,
            when verifying, `label.npy`).
        limit: Rebuild only the first `limit` case ids (sorted order from
            `available_logit_cases`). `None` rebuilds every case with saved
            logits.
        overwrite: If `False` (the default), a case whose
            `predictions/<case>.npy` already exists is left untouched. If
            `True`, it is recomputed and overwritten.
        verify_n: How many labeled, published cases to self-verify -- see
            `_verify_rebuilt_predictions`.

    Returns:
        `{"n_total", "n_written", "n_skipped", "verify"}`, where `"verify"`
        is `_verify_rebuilt_predictions`'s return value (possibly `None`).

    Raises:
        FileNotFoundError: If `<eval_dir>/logits/` has no cases, or
            `<eval_dir>/eval_config.yaml` is missing (see
            `_load_run_postprocess_cfg`).
        ValueError: If self-verification fails (see
            `_verify_rebuilt_predictions`).
    """
    eval_dir = Path(eval_dir)
    prep_dir = Path(prep_dir)
    predictions_dir = ensure_dir(eval_dir / "predictions")

    case_ids = available_logit_cases(eval_dir)
    if not case_ids:
        raise FileNotFoundError(
            f"rebuild_predictions: no saved logits under {eval_dir / 'logits'}. This script "
            "needs an evaluation run with cfg.inference.evaluation.save_logits=true."
        )
    if limit is not None:
        case_ids = case_ids[:limit]

    threshold, pp_cfg = _load_run_postprocess_cfg(eval_dir)
    logger.info(
        "rebuild_predictions: eval_dir=%s prep_dir=%s, %d case(s), threshold=%s, "
        "postprocess=%s, overwrite=%s",
        eval_dir,
        prep_dir,
        len(case_ids),
        threshold,
        pp_cfg,
        overwrite,
    )

    n_written = 0
    n_skipped = 0
    for i, case_id in enumerate(case_ids, start=1):
        out_path = predictions_dir / f"{case_id}.npy"
        if out_path.is_file() and not overwrite:
            n_skipped += 1
        else:
            uncropped = _rebuild_one_case(eval_dir, prep_dir, case_id, threshold, pp_cfg)
            np.save(out_path, uncropped)
            n_written += 1

        if i % _LOG_EVERY == 0:
            logger.info("rebuild_predictions: processed %d/%d case(s)", i, len(case_ids))

    logger.info(
        "rebuild_predictions: done. %d written, %d skipped (already existed), %d total.",
        n_written,
        n_skipped,
        len(case_ids),
    )

    verify_result = _verify_rebuilt_predictions(eval_dir, prep_dir, case_ids, verify_n=verify_n)

    summary = {
        "n_total": len(case_ids),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "verify": verify_result,
    }

    if verify_result is None:
        verify_line = "verification skipped (no labeled/published case found)"
    else:
        verify_line = (
            f"verified {verify_result['n_verified']} case(s), "
            f"max |delta| = {verify_result['max_abs_delta']:.2e}"
        )
    # print only, not logger.info as well -- setup_logging's StreamHandler
    # already targets stdout, so doing both would print this twice. Matches
    # scripts/replay_logits.py's summary convention.
    print(
        f"rebuild_predictions: {n_written} written, {n_skipped} skipped, "
        f"{len(case_ids)} total; {verify_line}"
    )
    return summary


def run_rebuild(cfg: DictConfig) -> dict[str, Any]:
    """Reads `cfg.rebuild.*` and calls `rebuild_predictions`.

    Args:
        cfg: The full composed Hydra config. `rebuild` is not a config group
            defined anywhere under `configs/`; it exists only for the
            settings this script's CLI overrides add on the fly (e.g.
            `+rebuild.eval_dir=...`), so every key below is read
            defensively with `.get`.

    Returns:
        `rebuild_predictions`'s return value.

    Raises:
        ValueError: If `rebuild.eval_dir` is not set.
    """
    rebuild_cfg = cfg.get("rebuild", None)
    eval_dir_value = rebuild_cfg.get("eval_dir", None) if rebuild_cfg is not None else None
    if eval_dir_value is None:
        raise ValueError(
            "rebuild.eval_dir is not set. Point it at an evaluation output directory written "
            "by scripts/evaluate.py with save_logits=true, e.g. "
            "'+rebuild.eval_dir=outputs/neurovision/eval_test'."
        )
    eval_dir = Path(str(eval_dir_value))

    prep_dir_value = rebuild_cfg.get("prep_dir", None)
    if prep_dir_value is None:
        # Default matches scripts/evaluate.py / scripts/burden.py's own
        # resolution of the preprocessed-case root.
        prep_dir = Path(str(cfg.data.preprocessing.out_dir))
    else:
        prep_dir = Path(str(prep_dir_value))

    limit_value = rebuild_cfg.get("limit", None)
    limit = int(limit_value) if limit_value is not None else None
    overwrite = bool(rebuild_cfg.get("overwrite", False))
    verify_n = int(rebuild_cfg.get("verify_n", 5))

    return rebuild_predictions(
        eval_dir, prep_dir, limit=limit, overwrite=overwrite, verify_n=verify_n
    )


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Rebuilds `predictions/` for one evaluation directory per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_rebuild(cfg)


if __name__ == "__main__":
    main()
