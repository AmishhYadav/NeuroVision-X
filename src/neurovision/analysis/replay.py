"""Re-scores a checkpoint's saved logits at a different threshold/post-processing setting.

`scripts/evaluate.py` can save raw fp16 logits per case
(`<eval_dir>/logits/<case_id>.npy`, shape `(3, D, H, W)`, CROPPED geometry,
channel order ET/TC/WT -- see that script's `save_logits` option). Because
the logits are saved, the discretization threshold and the post-processing
chain (`neurovision.inference.postprocess`) can be re-ablated WITHOUT
re-running the model and WITHOUT a GPU. This has already been verified once
by hand: replaying `baseline_unet3d`'s test-split logits with the run's own
recorded config reproduced its published test Dice exactly (ET 0.870859 vs
0.870859, delta 0).

This module answers two questions a reviewer will ask about any reported
Dice advantage:

1. Is it an artifact of the 0.5 threshold, or does it hold across operating
   points? -- `threshold_sweep`.
2. How much of it comes from the model versus from post-processing (small-
   component removal, largest-component, nesting)? -- `postprocess_ablation`.

## Why this does not simply call `postprocess_logits`

`neurovision.inference.postprocess.postprocess_logits` takes the whole
Hydra config and reads a single scalar `cfg.inference.postprocess.threshold`
inside its own call to `binarize`. That is exactly right for a normal
evaluation run, where every region shares one operating point -- but it
makes two things this module needs impossible without a config object:
varying an individual step (keep-largest, small-component removal, nesting)
independently of the others, and using a DIFFERENT threshold per region
channel (needed to answer question 1 properly: ET, TC and WT do not
necessarily share an optimal operating point). Threading either through
`postprocess_logits`'s cfg-shaped interface would mean building a fake
nested config object just to flip one field, and a per-channel threshold
cannot be expressed by that interface at all (`binarize` compares the whole
tensor against one scalar, with no channel-axis alignment). So this module
calls the ORCHESTRATION steps itself -- `_binarize_regions` (a per-channel-
aware generalization of `binarize`) followed directly by the same public,
already-tested functions `postprocess_logits` calls internally
(`keep_largest_component`, `remove_small_components`, `enforce_nesting`) in
the same fixed order -- rather than reimplementing any of their algorithms.

`postprocess_cfg=None` reproduces the PROJECT DEFAULT chain by reading
`configs/inference/default.yaml`'s `postprocess` block (minus `threshold`,
which `replay_case` takes as its own explicit argument), so a replay with no
arguments matches a standard evaluation run.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from neurovision.inference.postprocess import (
    enforce_nesting,
    keep_largest_component,
    remove_small_components,
)
from neurovision.metrics.segmentation import classes_to_regions, compute_case_metrics
from neurovision.utils.io import read_json, read_yaml

__all__ = [
    "load_case_logits",
    "available_logit_cases",
    "replay_case",
    "threshold_sweep",
    "postprocess_ablation",
    "per_case_replay",
]

logger = logging.getLogger(__name__)

# How often (in cases) the batch functions below log progress. Small enough
# to be useful on a ~189-case test split, large enough not to spam the log on
# a multi-threshold / multi-variant sweep.
_LOG_EVERY = 25

# Repo root, computed from this file's own location -- never from the current
# working directory -- so `configs/inference/default.yaml` resolves the same
# way regardless of where the calling script was invoked from. Same pattern
# scripts/evaluate.py uses for `_CONFIG_DIR`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INFERENCE_CFG_PATH = _REPO_ROOT / "configs" / "inference" / "default.yaml"


def load_case_logits(eval_dir: str | Path, case_id: str) -> np.ndarray:
    """Loads one case's saved fp16 logits.

    Args:
        eval_dir: An evaluation output directory as written by
            `scripts/evaluate.py` (the directory that contains `logits/`,
            `predictions/`, `per_case_metrics.csv`, etc.).
        case_id: The case identifier, matching the `.npy` filename stem.

    Returns:
        Float32 array, shape `(3, D, H, W)`, channel order `(ET, TC, WT)`,
        in CROPPED geometry. Cast from fp16 to float32 immediately on load,
        before any arithmetic touches the values.

    Raises:
        FileNotFoundError: If no saved logits exist for `case_id`, naming
            the path that was expected.
    """
    path = Path(eval_dir) / "logits" / f"{case_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"No saved logits for case {case_id!r} at {path.resolve()}. This replay module "
            "requires an evaluation run with cfg.inference.evaluation.save_logits=true."
        )
    raw = np.load(path)
    return raw.astype(np.float32)


def available_logit_cases(eval_dir: str | Path) -> list[str]:
    """Lists case ids that have saved logits under `<eval_dir>/logits/`.

    Args:
        eval_dir: An evaluation output directory as written by
            `scripts/evaluate.py`.

    Returns:
        Sorted case ids (the `.npy` filename stems). Empty list if
        `<eval_dir>/logits/` does not exist.
    """
    logits_dir = Path(eval_dir) / "logits"
    if not logits_dir.is_dir():
        return []
    return sorted(p.stem for p in logits_dir.glob("*.npy"))


@functools.lru_cache(maxsize=1)
def _project_default_postprocess_cfg() -> dict[str, Any]:
    """Reads `configs/inference/default.yaml`'s `postprocess` block, minus `threshold`.

    Cached (the file never changes mid-process) so a batch sweep over many
    cases and thresholds does not re-read and re-parse this small YAML file
    on every single `replay_case` call.

    Returns:
        A plain dict with keys `enforce_nesting`, `min_component_size`,
        `connectivity`, `keep_largest_only`, `et_min_volume` -- everything
        `postprocess_logits` reads from `cfg.inference.postprocess` except
        `threshold`, which `replay_case` takes as its own explicit argument
        so it can be scalar or per-channel.
    """
    raw = read_yaml(_DEFAULT_INFERENCE_CFG_PATH)
    postprocess_block = dict(raw["postprocess"])
    postprocess_block.pop("threshold", None)
    return postprocess_block


def _resolve_postprocess_cfg(postprocess_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merges a caller-supplied partial post-processing config over the project default.

    Args:
        postprocess_cfg: `None` to use the project default chain unchanged,
            or a mapping of any subset of `enforce_nesting`,
            `min_component_size`, `connectivity`, `keep_largest_only`,
            `et_min_volume` to override.

    Returns:
        A complete dict with all five keys set.
    """
    resolved = dict(_project_default_postprocess_cfg())
    if postprocess_cfg is not None:
        resolved.update(postprocess_cfg)
    return resolved


def _binarize_regions(logits: Tensor, threshold: float | Sequence[float]) -> Tensor:
    """Sigmoid + threshold, generalized to an optional per-channel threshold.

    A superset of `neurovision.metrics.segmentation.binarize`: that function
    only accepts one scalar threshold shared by every channel.  Here
    `threshold` may instead be a length-3 sequence (`ET, TC, WT` order),
    broadcast against the channel axis explicitly so each region gets its
    own operating point.

    Args:
        logits: Raw (pre-sigmoid) logits, shape `(1, 3, D, H, W)`.
        threshold: A single probability threshold applied to every channel,
            or one threshold per channel (`ET, TC, WT` order).

    Returns:
        Binary float tensor, same shape as `logits`.

    Raises:
        ValueError: If `threshold` is a sequence whose length does not match
            the number of region channels.
    """
    probs = torch.sigmoid(logits)
    num_channels = probs.shape[1]
    if isinstance(threshold, (int, float)):
        thresh = torch.full((num_channels,), float(threshold), dtype=probs.dtype)
    else:
        values = [float(t) for t in threshold]
        if len(values) != num_channels:
            raise ValueError(
                f"replay: threshold sequence has {len(values)} entries but logits have "
                f"{num_channels} region channels (ET, TC, WT expected)."
            )
        thresh = torch.tensor(values, dtype=probs.dtype)
    thresh = thresh.view(1, num_channels, 1, 1, 1)
    return (probs >= thresh).to(dtype=torch.float32)


def _zero_small_et(regions: Tensor, et_min_volume: float) -> Tensor:
    """Zeros the ET channel (index 0) if its predicted voxel count is below `et_min_volume`.

    Mirrors `neurovision.inference.postprocess`'s private, single-case
    `_zero_small_et` step exactly (that helper is not exported, and this
    module calls the individual post-processing steps directly rather than
    through `postprocess_logits` -- see the module docstring). `regions` here
    is always a single case (`(1, 3, D, H, W)`), so the per-batch-element
    branching the original needs does not apply.
    """
    regions = regions.clone()
    if regions[:, 0].sum() < et_min_volume:
        regions[:, 0] = 0.0
    return regions


def _apply_postprocess_steps(regions: Tensor, pp_cfg: Mapping[str, Any]) -> Tensor:
    """Runs the non-threshold post-processing steps, in `postprocess_logits`'s fixed order.

    Args:
        regions: Already-binarized region tensor, shape `(1, 3, D, H, W)`.
        pp_cfg: A resolved post-processing config (see
            `_resolve_postprocess_cfg`).

    Returns:
        Binary float tensor, same shape as `regions`.
    """
    if pp_cfg.get("keep_largest_only", False):
        regions = keep_largest_component(regions)

    regions = remove_small_components(
        regions,
        min_size=int(pp_cfg.get("min_component_size", 0)),
        connectivity=int(pp_cfg.get("connectivity", 1)),
    )

    et_min_volume = float(pp_cfg.get("et_min_volume", 0))
    if et_min_volume > 0:
        regions = _zero_small_et(regions, et_min_volume)

    if pp_cfg.get("enforce_nesting", True):
        regions = enforce_nesting(regions)

    return regions


def replay_case(
    logits: np.ndarray,
    label: np.ndarray,
    *,
    threshold: float | Sequence[float] = 0.5,
    postprocess_cfg: Mapping[str, Any] | None = None,
    spacing: tuple[float, float, float] | None = None,
    lesionwise: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Scores one case's saved logits at a given threshold and post-processing setting.

    Args:
        logits: Raw (pre-sigmoid) logits for one case, shape `(3, D, H, W)`,
            channel order `(ET, TC, WT)`, CROPPED geometry.
        label: Integer BraTS class label for the SAME case, values
            `{0, 1, 2, 3}`, shape `(D, H, W)`, same cropped geometry as
            `logits`.
        threshold: A single probability threshold applied to every region
            channel, or a length-3 sequence (`ET, TC, WT` order) giving a
            different threshold per channel.
        postprocess_cfg: `None` for the project default post-processing
            chain (read from `configs/inference/default.yaml`); otherwise a
            mapping overriding any subset of `enforce_nesting`,
            `min_component_size`, `connectivity`, `keep_largest_only`,
            `et_min_volume`.
        spacing: Voxel spacing in mm, `(D, H, W)` order, as stored in the
            case's `meta.json`. Passed straight through to HD95 AND (when
            `lesionwise` is set) to the lesion-wise NSD. When `None`, HD95
            is reported in VOXELS, not millimetres.
        lesionwise: `None` (the default) turns lesion-wise scoring OFF --
            the return value is then byte-identical to what this function
            returned before lesion-wise metrics existed. Otherwise, a
            mapping of keyword arguments for
            `neurovision.metrics.lesionwise.lesionwise_case_metrics`
            (`min_lesion_voxels`, `matching_threshold`, `nsd_tolerance_mm`,
            `connectivity`); its result is merged additively into the
            returned dict, the same pattern `scripts/evaluate.py` uses.
            `lesionwise_case_metrics` needs `panoptica`, imported lazily
            inside this function -- see the module docstring's `.venv` /
            `.venv-analysis` split.

    Returns:
        The same flat metric dict `compute_case_metrics` returns:
        `dice_R`/`iou_R`/`hd95_R`/`gt_empty_R` per region plus
        `dice_mean`/`iou_mean`/`hd95_mean`. When `lesionwise` is not `None`,
        also every key `lesionwise_case_metrics` returns (`lwdice_R`,
        `lwnsd_R`, `lwf1_R`, `lwtp_R`, `lwfp_R`, `lwfn_R` per region, plus
        the `lw*_mean` cross-region means).

    Raises:
        ValueError: If `logits` is not `(3, D, H, W)`, or `threshold` is a
            sequence of the wrong length.
        ImportError: If `lesionwise` is not `None` and `panoptica` is not
            installed in the current interpreter (see
            `neurovision.metrics.lesionwise.require_panoptica`).
    """
    logits_arr = np.asarray(logits, dtype=np.float32)
    if logits_arr.ndim != 4 or logits_arr.shape[0] != 3:
        raise ValueError(
            f"replay_case expects logits of shape (3, D, H, W), got {logits_arr.shape}."
        )
    logits_t = torch.from_numpy(logits_arr).unsqueeze(0)  # (1, 3, D, H, W)

    resolved_pp_cfg = _resolve_postprocess_cfg(postprocess_cfg)
    regions = _binarize_regions(logits_t, threshold)
    regions = _apply_postprocess_steps(regions, resolved_pp_cfg)

    label_t = torch.as_tensor(np.asarray(label))
    target = classes_to_regions(label_t)  # (1, 3, D, H, W)

    metrics = compute_case_metrics(regions, target, spacing=spacing)

    if lesionwise is not None:
        # Imported here, not at module top level: `panoptica` is deliberately
        # absent from the training `.venv` (see this module's docstring), and
        # `neurovision.analysis.replay` must still be importable there. This
        # branch is the only place that needs panoptica, and it is only
        # reached when a caller opts in.
        from neurovision.metrics.lesionwise import lesionwise_case_metrics

        metrics.update(lesionwise_case_metrics(regions, target, spacing=spacing, **lesionwise))

    return metrics


def _load_label_and_spacing(
    prep_dir: str | Path, case_id: str
) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    """Loads one case's ground-truth label and voxel spacing from the preprocessed tree.

    Args:
        prep_dir: Root directory of preprocessed cases (as written by
            `neurovision.data.preprocessing.preprocess_case`).
        case_id: The case identifier.

    Returns:
        `(label, spacing)`, or `None` if `<prep_dir>/<case_id>/label.npy`
        does not exist -- logged as a warning rather than raised, since an
        unlabeled case (e.g. the BraTS validation set) is expected to be
        absent, not a bug.
    """
    case_dir = Path(prep_dir) / case_id
    label_path = case_dir / "label.npy"
    if not label_path.is_file():
        logger.warning(
            "replay: no label.npy for case %r at %s; skipping this case.", case_id, label_path
        )
        return None
    label = np.load(label_path)
    meta = read_json(case_dir / "meta.json")
    spacing = tuple(float(s) for s in meta["spacing"])
    return label, spacing


def _resolve_case_ids(eval_dir: str | Path, case_ids: Sequence[str] | None) -> list[str]:
    """Resolves the case id list a batch function should iterate over."""
    if case_ids is not None:
        return list(case_ids)
    return available_logit_cases(eval_dir)


def _summarize_metric_records(
    records: list[dict[str, float]], metrics: Sequence[str]
) -> dict[str, float]:
    """Builds the `<metric>_mean` / `<metric>_median` columns for one sweep row.

    Args:
        records: One `compute_case_metrics`-style dict per scored case.
        metrics: Metric names to summarize.

    Returns:
        A flat dict with two entries per metric. NaN for both when
        `records` is empty (every case in the split was skipped).
    """
    row: dict[str, float] = {}
    if records:
        table = pd.DataFrame(records)
        for metric in metrics:
            row[f"{metric}_mean"] = float(table[metric].mean())
            row[f"{metric}_median"] = float(table[metric].median())
    else:
        for metric in metrics:
            row[f"{metric}_mean"] = float("nan")
            row[f"{metric}_median"] = float("nan")
    return row


def threshold_sweep(
    eval_dir: str | Path,
    prep_dir: str | Path,
    thresholds: Sequence[float],
    *,
    case_ids: Sequence[str] | None = None,
    postprocess_cfg: Mapping[str, Any] | None = None,
    metrics: Sequence[str] = ("dice_ET", "dice_TC", "dice_WT"),
) -> pd.DataFrame:
    """Replays every case at each threshold and reports the mean/median metrics.

    Post-processing is held fixed (`postprocess_cfg`, project default if
    `None`) so this isolates the effect of the operating point alone.

    Args:
        eval_dir: An evaluation output directory with `logits/` saved.
        prep_dir: Root directory of preprocessed cases (for ground-truth
            labels and spacing).
        thresholds: Thresholds to sweep, applied to every region channel
            equally at each sweep point. Order is preserved in the output.
        case_ids: Cases to replay. `None` means every case with saved
            logits.
        postprocess_cfg: Forwarded to `replay_case` at every threshold.
        metrics: Metric columns to summarize.

    Returns:
        One row per threshold, columns `threshold`, `n` (cases actually
        scored, after skipping any missing labels), and
        `<metric>_mean`/`<metric>_median` for each requested metric, in the
        order `thresholds` was given.
    """
    resolved_ids = _resolve_case_ids(eval_dir, case_ids)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        records: list[dict[str, float]] = []
        for i, case_id in enumerate(resolved_ids, start=1):
            loaded = _load_label_and_spacing(prep_dir, case_id)
            if loaded is None:
                continue
            label, spacing = loaded
            logits = load_case_logits(eval_dir, case_id)
            records.append(
                replay_case(
                    logits,
                    label,
                    threshold=threshold,
                    postprocess_cfg=postprocess_cfg,
                    spacing=spacing,
                )
            )
            if i % _LOG_EVERY == 0:
                logger.info(
                    "threshold_sweep: threshold=%s, processed %d/%d case(s)",
                    threshold,
                    i,
                    len(resolved_ids),
                )
        row: dict[str, Any] = {"threshold": threshold, "n": len(records)}
        row.update(_summarize_metric_records(records, metrics))
        rows.append(row)
    return pd.DataFrame(rows)


def postprocess_ablation(
    eval_dir: str | Path,
    prep_dir: str | Path,
    variants: Mapping[str, Mapping[str, Any]],
    *,
    case_ids: Sequence[str] | None = None,
    threshold: float = 0.5,
    metrics: Sequence[str] = ("dice_ET", "dice_TC", "dice_WT"),
) -> pd.DataFrame:
    """Replays every case under each named post-processing variant.

    Threshold is held fixed so this isolates the effect of post-processing
    alone -- e.g. how much of the reported Dice comes from small-component
    removal or the largest-component filter versus the model itself.

    Args:
        eval_dir: An evaluation output directory with `logits/` saved.
        prep_dir: Root directory of preprocessed cases.
        variants: `{variant_name: postprocess_cfg}`. Each `postprocess_cfg`
            is forwarded to `replay_case` exactly like a direct call --
            a partial mapping is merged over the project default, so a
            variant only needs to name what it changes.
        case_ids: Cases to replay. `None` means every case with saved
            logits.
        threshold: A single threshold, shared by every variant and every
            region channel, so variants differ only in post-processing.
        metrics: Metric columns to summarize.

    Returns:
        One row per variant, in `variants`' iteration order, columns
        `variant`, `n`, and `<metric>_mean`/`<metric>_median` for each
        requested metric.
    """
    resolved_ids = _resolve_case_ids(eval_dir, case_ids)
    rows: list[dict[str, Any]] = []
    for variant_name, variant_cfg in variants.items():
        records: list[dict[str, float]] = []
        for i, case_id in enumerate(resolved_ids, start=1):
            loaded = _load_label_and_spacing(prep_dir, case_id)
            if loaded is None:
                continue
            label, spacing = loaded
            logits = load_case_logits(eval_dir, case_id)
            records.append(
                replay_case(
                    logits,
                    label,
                    threshold=threshold,
                    postprocess_cfg=variant_cfg,
                    spacing=spacing,
                )
            )
            if i % _LOG_EVERY == 0:
                logger.info(
                    "postprocess_ablation: variant=%r, processed %d/%d case(s)",
                    variant_name,
                    i,
                    len(resolved_ids),
                )
        row: dict[str, Any] = {"variant": variant_name, "n": len(records)}
        row.update(_summarize_metric_records(records, metrics))
        rows.append(row)
    return pd.DataFrame(rows)


def per_case_replay(
    eval_dir: str | Path,
    prep_dir: str | Path,
    *,
    case_ids: Sequence[str] | None = None,
    threshold: float | Sequence[float] = 0.5,
    postprocess_cfg: Mapping[str, Any] | None = None,
    lesionwise: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Replays every case once and returns a per-case metrics table.

    Same shape as `per_case_metrics.csv` (indexed by `case_id`, one column
    per `compute_case_metrics` key), so the result can be fed straight into
    `neurovision.analysis.statistics.compare_models` or
    `neurovision.analysis.stratify` with no adaptation.

    Args:
        eval_dir: An evaluation output directory with `logits/` saved.
        prep_dir: Root directory of preprocessed cases.
        case_ids: Cases to replay. `None` means every case with saved
            logits.
        threshold: Forwarded to `replay_case`; scalar or per-channel.
        postprocess_cfg: Forwarded to `replay_case`.
        lesionwise: `None` (the default) turns lesion-wise scoring OFF, and
            the returned table's columns are byte-identical to before this
            parameter existed. Otherwise forwarded to `replay_case` for
            every case, adding the `lw*` columns on top of the existing
            ones -- see `replay_case`'s docstring.

    Returns:
        A DataFrame indexed by `case_id`. Cases whose `label.npy` is missing
        are skipped (warned, not raised) and simply absent from the result.
    """
    resolved_ids = _resolve_case_ids(eval_dir, case_ids)
    records: dict[str, dict[str, float]] = {}
    for i, case_id in enumerate(resolved_ids, start=1):
        loaded = _load_label_and_spacing(prep_dir, case_id)
        if loaded is None:
            continue
        label, spacing = loaded
        logits = load_case_logits(eval_dir, case_id)
        records[case_id] = replay_case(
            logits,
            label,
            threshold=threshold,
            postprocess_cfg=postprocess_cfg,
            spacing=spacing,
            lesionwise=lesionwise,
        )
        if i % _LOG_EVERY == 0:
            logger.info("per_case_replay: processed %d/%d case(s)", i, len(resolved_ids))

    if not records:
        return pd.DataFrame().rename_axis("case_id")
    return pd.DataFrame.from_dict(records, orient="index").rename_axis("case_id")
