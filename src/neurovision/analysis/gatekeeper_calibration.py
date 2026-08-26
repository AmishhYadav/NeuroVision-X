"""Builds the val-split calibration table `gatekeeper.calibrate_thresholds` needs.

Milestone 4, Phase E, the first half of the not-yet-run calibration step:
`neurovision.inference.gatekeeper.Thresholds` has never been fitted --
`cfg.clinical.gatekeeper.thresholds` is currently `null` -- because nothing has ever
assembled one table holding, per calibration case, every signal
`calibrate_thresholds` needs: `predicted_dice_<R>` (Phase C's QC model), `conformal_band_<R>`
(Phase B's conformal risk control) and `ood_score`. This module is exactly that
assembly, and nothing else -- it fits no threshold itself
(`neurovision.inference.gatekeeper.calibrate_thresholds` already does that), runs no
segmentation model, and writes nothing to disk. A separate Hydra driver script (not
this module) calls `build_gatekeeper_calibration_table`, hands its output to
`calibrate_thresholds`, and writes the result to `cfg.clinical.gatekeeper.thresholds`.

Three signals, three small pieces, one composing function:

- `qc_predicted_dice_table` -- runs the trained `SegQC` model (Phase C) on each case's
  OWN deployed prediction (the "identity" path: no mask degradation), reusing
  `neurovision.analysis.qc_inference`'s loading and packing exactly as
  `scripts/validate_qc.py` does.
- `case_conformal_band_widths` / `resolve_fitted_thresholds` -- read a previously
  extracted conformal run's `curves.npz` + `fit.json` (Phase B,
  `scripts/conformal.py`'s own output) and reduce them to one mask-inflation number per
  case per region, at the alpha the deployed gate uses.
- `ood_score_table` -- see the loud warning immediately below. There is no real
  out-of-distribution scorer anywhere in this project yet.

## `ood_score` is a STRUCTURAL PLACEHOLDER, not a validated detector

**`calibrate_thresholds` needs an `ood_score` column to exist so `Thresholds` can be
constructed at all -- it does NOT need `ood_score` to be a good signal.**
`ood_score_table` below computes `ood_score` as
`neurovision.analysis.detection.case_entropy_scalars(...)["ent_mean_fg_mean"]`, read
straight from each case's own saved logits. This is single-pass predictive entropy,
already measured throughout this project to correlate with Dice, and it has **never
been evaluated, validated, or even proposed as an out-of-distribution detector** -- it
is a stand-in chosen only because it is cheap, already computed elsewhere, and
structurally sufficient to fill the column. `cfg.clinical.gatekeeper.enabled_signals`
must keep `ood_score` OUT until a real OOD scorer is designed and measured on a
held-out cohort (see `configs/clinical/default.yaml`'s own comment on this point). Do
not report a number produced through this column as evidence the pipeline detects
out-of-distribution inputs.

## Label-free, by construction of what is NOT read here

Every one of the three tables this module builds is computable with no ground-truth
label at inference time: `qc_predicted_dice_table` reads a case's own prediction and
its own saved logits (the label is loaded internally by
`neurovision.analysis.qc_inference.load_case_arrays` only for a spatial-shape check,
never used to derive a value returned here); `case_conformal_band_widths` reduces an
ALREADY-FITTED conformal artifact (Phase B's own fit/apply separation already governs
how that artifact used ground truth, at calibration time, long before this module
runs); `ood_score_table` reads only logits. No function in this module takes a Dice,
a label array, or a ground-truth mask as an argument.

## Why the three tables are joined INNER, not OUTER

`calibrate_thresholds` computes quantiles per column independently, but a calibration
case that is missing one of the three signals entirely is not a valid calibration
case for THIS gate -- the whole point of one frozen calibration table is that every
row describes one case's full multi-signal picture. `build_gatekeeper_calibration_table`
therefore inner-joins on `case_id` and logs (never raises, unless the join is empty) how
many cases were dropped this way, mirroring `neurovision.inference.gatekeeper.
calibrate_thresholds`'s own "warn on a small calibration_n, do not silently proceed
with a filled-in NaN" discipline.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from neurovision.analysis.detection import case_entropy_scalars
from neurovision.analysis.qc_inference import load_case_arrays, pack_sample
from neurovision.data.transforms import REGION_NAMES
from neurovision.models.qc import build_segqc, predicted_dice
from neurovision.training.checkpoint import load_checkpoint
from neurovision.uncertainty.conformal import CaseLossCurve, load_curves_npz

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared case-id resolution
# ---------------------------------------------------------------------------


def _case_ids_from_logits(eval_dir: Path, case_ids: Sequence[str] | None) -> list[str]:
    """Resolves which case ids to process: the given list, or every case with saved logits.

    Args:
        eval_dir: A `scripts/evaluate.py` output directory holding `logits/*.npy`.
        case_ids: Explicit case ids, or `None` to use every case under
            `eval_dir/logits/*.npy` (the file's stem, e.g. `"BraTS2021_00001"`).

    Returns:
        A sorted, deduplicated list of case ids.

    Raises:
        FileNotFoundError: `case_ids` is `None` and `<eval_dir>/logits` does not exist.
        ValueError: `case_ids` is `None` and no case has saved logits under `eval_dir`.
    """
    if case_ids is not None:
        return sorted({str(c) for c in case_ids})

    logits_dir = Path(eval_dir) / "logits"
    if not logits_dir.is_dir():
        raise FileNotFoundError(
            f"gatekeeper_calibration: no logits/ directory at {logits_dir}. Re-run "
            "scripts/evaluate.py with inference.evaluation.save_logits=true against this "
            "eval_dir."
        )
    resolved = sorted(p.stem for p in logits_dir.glob("*.npy"))
    if not resolved:
        raise ValueError(f"gatekeeper_calibration: no saved logits under {logits_dir}.")
    return resolved


# ---------------------------------------------------------------------------
# Signal 1: the QC model's predicted Dice, on the case's OWN deployed prediction
# ---------------------------------------------------------------------------


def qc_predicted_dice_table(
    cfg: Any,
    checkpoint: Path,
    eval_dir: Path,
    prep_dir: Path,
    regions: Sequence[str],
    case_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scores every case's OWN deployed prediction with the trained `SegQC` model.

    This is the IDENTITY path through `neurovision.analysis.qc_inference`: for each
    case, `arrays.pred_mask` (the model's own, undegraded, deployed prediction) is
    packed and scored directly, via `pack_sample(arrays, mask=arrays.pred_mask,
    region_channel=idx, target_shape=...)` -- no mask degradation from
    `neurovision.data.qc_pairs` is involved, because the number this function reports
    IS the QC model's estimate of the very mask the gatekeeper will actually see.

    `cfg.model` must already be composed as the `segqc` model group (e.g. a Hydra run
    with `model=segqc` on its command line) -- the root config's default model group,
    `unet3d`, has none of the keys `neurovision.models.qc.build_segqc` reads. This
    mirrors `scripts/train_qc.py`'s own module docstring, which states the identical
    requirement for the same reason.

    Args:
        cfg: The full composed Hydra config. Needs `cfg.model` (composed as `segqc`,
            see above), `cfg.analysis.qc.target_shape`, `cfg.analysis.qc.modality_index`
            and `cfg.inference.postprocess` (the last read transitively by
            `neurovision.analysis.qc_inference.load_case_arrays`).
        checkpoint: Path to a trained `SegQC` checkpoint, in the format
            `neurovision.training.checkpoint.save_checkpoint` writes.
        eval_dir: A `scripts/evaluate.py` output directory holding `logits/*.npy`.
        prep_dir: Root of the preprocessed BraTS data, holding
            `<case_id>/{image.npy,label.npy}`.
        regions: Region names to score, e.g. `("WT", "TC")`.
        case_ids: Explicit case ids to score, or `None` (the default) to score every
            case with saved logits under `eval_dir/logits/*.npy`.

    Returns:
        One row per case, with a `case_id` column plus `predicted_dice_<R>` for every
        `R` in `regions`, each in `[0, 1]`.

    Raises:
        FileNotFoundError: `checkpoint` does not exist, `case_ids` is `None` and
            `eval_dir/logits` does not exist, or a case's `image.npy` / `label.npy` /
            saved logits are missing (propagated from `load_case_arrays`).
        ValueError: `case_ids` is `None` and no case has saved logits under `eval_dir`,
            or a region name is not one of `neurovision.data.transforms.REGION_NAMES`.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"qc_predicted_dice_table: no checkpoint at {checkpoint}.")
    for region in regions:
        if region not in REGION_NAMES:
            raise ValueError(
                f"qc_predicted_dice_table: region {region!r} is not one of {REGION_NAMES}."
            )

    resolved_case_ids = _case_ids_from_logits(eval_dir, case_ids)

    # Built and loaded exactly ONCE, never per case: this run can be a few hundred
    # forward passes (n_cases x n_regions). Always CPU -- this calibration step runs on
    # the Mac, never the GPU cluster (CLAUDE.md's machine split), so map_location and
    # the model's device are both hard-coded to "cpu" rather than read from cfg.
    device = torch.device("cpu")
    model = build_segqc(cfg).to(device)
    load_checkpoint(checkpoint, model, optimizer=None, map_location="cpu", restore_rng=False)
    model.eval()

    target_shape = tuple(int(v) for v in cfg.analysis.qc.target_shape)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for case_id in resolved_case_ids:
            arrays = load_case_arrays(cfg, Path(eval_dir), Path(prep_dir), case_id)
            row: dict[str, Any] = {"case_id": case_id}
            for region in regions:
                region_channel = REGION_NAMES.index(region)
                sample = pack_sample(arrays, arrays.pred_mask, region_channel, target_shape)
                sample = sample.unsqueeze(0)  # (1, 3, D', H', W')
                logit = model(sample)  # (1,), a raw logit -- see models.qc's module docstring
                row[f"predicted_dice_{region}"] = float(predicted_dice(logit).item())
            rows.append(row)

    logger.info(
        "qc_predicted_dice_table: scored %d case(s) x %d region(s) from %s against %s.",
        len(rows),
        len(regions),
        eval_dir,
        checkpoint,
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signal 2: conformal band width, at the deployed alpha's fitted threshold
# ---------------------------------------------------------------------------


def resolve_fitted_thresholds(
    fit_payload: Mapping[str, Any],
    regions: Sequence[str],
    alpha: float,
) -> dict[str, float]:
    """Reads region -> fitted conformal threshold, for a fixed `alpha`, out of `fit.json`.

    `scripts/conformal.py::_fit_payload` keys each entry EXACTLY as
    `f"{region}__alpha_{alpha}"`, where `alpha` is Python's default `str()`/f-string
    conversion of the float -- e.g. `alpha=0.1` gives `"WT__alpha_0.1"`, never
    `"WT__alpha_0.10"`. This function builds that same key with plain
    `f"{region}__alpha_{alpha}"` and never with a fixed-decimals format (e.g.
    `f"{alpha:.2f}"`, which would silently miss every entry).

    Args:
        fit_payload: The parsed JSON from a `fit.json` file (a plain
            `{key: {"threshold": float | None, ...}}` mapping).
        regions: Region names to resolve, e.g. `("WT", "TC")`.
        alpha: The target risk level whose fitted threshold to read, e.g. `0.10`.

    Returns:
        `{region: threshold}`, one entry per `regions`.

    Raises:
        ValueError: `fit_payload` has no entry for a region's `f"{region}__alpha_{alpha}"`
            key (names the exact missing key), or that entry's `"threshold"` is `None`
            or absent -- an infeasible fit (see
            `neurovision.uncertainty.conformal.ConformalFit.feasible`) cannot back a
            deployed signal.
    """
    resolved: dict[str, float] = {}
    for region in regions:
        key = f"{region}__alpha_{alpha}"
        if key not in fit_payload:
            raise ValueError(
                f"resolve_fitted_thresholds: fit_payload has no entry {key!r} (region="
                f"{region!r}, alpha={alpha!r})."
            )
        entry = fit_payload[key]
        threshold = entry.get("threshold") if isinstance(entry, Mapping) else None
        if threshold is None:
            raise ValueError(
                f"resolve_fitted_thresholds: entry {key!r} has no usable 'threshold' -- "
                f"the conformal fit for region {region!r} at alpha={alpha!r} was infeasible "
                "(or the entry is malformed). An infeasible fit cannot back a deployed signal."
            )
        resolved[region] = float(threshold)
    return resolved


def _value_at_threshold(curve: CaseLossCurve, values: np.ndarray, threshold: float) -> float:
    """Returns `values[idx]`, where `idx` is `threshold`'s position in `curve.thresholds`.

    Reuses `CaseLossCurve.mask_inflation`'s own threshold-validation instead of
    re-implementing a tolerance-based grid lookup here: calling it with
    `reference_threshold=threshold` raises `ValueError`, naming the nearest grid value,
    exactly when `threshold` is not (to floating-point tolerance) a member of
    `curve.thresholds` -- the same failure `case_conformal_band_widths` must surface
    for an out-of-grid fitted threshold.

    Args:
        curve: The `CaseLossCurve` `threshold` and `values` both index into.
        values: An array already aligned with `curve.thresholds` (e.g.
            `curve.mask_inflation(reference_threshold=0.5)`).
        threshold: The grid value to locate.

    Returns:
        `values` at `threshold`'s position.

    Raises:
        ValueError: `threshold` is not in `curve.thresholds` (message names the
            nearest available value).
    """
    curve.mask_inflation(reference_threshold=threshold)  # validates threshold is in the grid
    idx = int(np.argmin(np.abs(np.asarray(curve.thresholds, dtype=np.float64) - threshold)))
    return float(values[idx])


def case_conformal_band_widths(
    curves_by_region: Mapping[str, Sequence[CaseLossCurve]],
    fitted_thresholds: Mapping[str, float],
    *,
    reference_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Reduces per-region loss curves to one mask-inflation number per case per region.

    For each region, for each `CaseLossCurve` in `curves_by_region[region]`, reads
    `curve.mask_inflation(reference_threshold=reference_threshold)` -- the per-threshold
    ratio `mask_voxels(tau) / mask_voxels(reference_threshold)` -- at the position of
    that region's fitted conformal threshold (`fitted_thresholds[region]`). This is the
    "band width" the gatekeeper's `conformal_band` signal judges: how much larger the
    conformal-guaranteed mask is than the ordinary `p >= 0.5` mask, for this case.

    Args:
        curves_by_region: `{region: [CaseLossCurve, ...]}`, e.g. from
            `neurovision.uncertainty.conformal.load_curves_npz`.
        fitted_thresholds: `{region: threshold}`, e.g. from `resolve_fitted_thresholds`.
            Must have an entry for every key of `curves_by_region`.
        reference_threshold: The point-estimate threshold the ratio is measured
            against. Defaults to `0.5`, matching
            `neurovision.uncertainty.conformal.CaseLossCurve.mask_inflation`'s own
            default.

    Returns:
        `{region: {case_id: mask_inflation_value}}`.

    Raises:
        KeyError: `fitted_thresholds` has no entry for a region present in
            `curves_by_region`.
        ValueError: A region's fitted threshold, or `reference_threshold`, is not in
            some curve's own threshold grid (propagated from `mask_inflation`, naming
            the nearest available value).
    """
    result: dict[str, dict[str, float]] = {}
    for region, curves in curves_by_region.items():
        threshold = fitted_thresholds[region]
        per_case: dict[str, float] = {}
        for curve in curves:
            inflation = curve.mask_inflation(reference_threshold=reference_threshold)
            per_case[curve.case_id] = _value_at_threshold(curve, inflation, threshold)
        result[region] = per_case
    return result


# ---------------------------------------------------------------------------
# Signal 3: ood_score -- a STRUCTURAL PLACEHOLDER, see module docstring
# ---------------------------------------------------------------------------


def ood_score_table(eval_dir: Path, case_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Placeholder case-level `ood_score`: mean predicted-foreground entropy, from saved logits.

    *** THIS IS NOT A VALIDATED OUT-OF-DISTRIBUTION DETECTOR. *** No real OOD scorer
    exists anywhere in this project. This function exists ONLY so
    `neurovision.inference.gatekeeper.calibrate_thresholds` / `Thresholds` can be
    structurally constructed with an `ood_score` column present --
    `cfg.clinical.gatekeeper.enabled_signals` must keep `ood_score` disabled until a
    real scorer is designed and measured against a held-out cohort. Do not cite this
    function, or any number it produces, as evidence the pipeline detects
    out-of-distribution inputs.

    Computed as `neurovision.analysis.detection.case_entropy_scalars(...)
    ["ent_mean_fg_mean"]` -- the NaN-skipping mean, across regions, of each region's
    mean entropy over its own predicted foreground -- read directly from the case's
    saved raw logits (`<eval_dir>/logits/<case_id>.npy`, cast to float32). No
    `neurovision.analysis.qc_inference.CaseArrays` round-trip is needed for this one
    scalar.

    Args:
        eval_dir: A `scripts/evaluate.py` output directory holding `logits/*.npy`.
        case_ids: Explicit case ids to score, or `None` (the default) to score every
            case with saved logits under `eval_dir/logits/*.npy`.

    Returns:
        One row per case, with `case_id` and `ood_score` columns.

    Raises:
        FileNotFoundError: `case_ids` is `None` and `eval_dir/logits` does not exist.
        ValueError: `case_ids` is `None` and no case has saved logits under `eval_dir`.
    """
    resolved_case_ids = _case_ids_from_logits(eval_dir, case_ids)

    rows: list[dict[str, Any]] = []
    for case_id in resolved_case_ids:
        logits_path = Path(eval_dir) / "logits" / f"{case_id}.npy"
        logits = np.load(logits_path).astype(np.float32)  # (3, D, H, W)
        scalars = case_entropy_scalars(logits)
        rows.append({"case_id": case_id, "ood_score": scalars["ent_mean_fg_mean"]})

    logger.info(
        "ood_score_table: computed the PLACEHOLDER ood_score (mean predicted-foreground "
        "entropy -- not a validated OOD detector) for %d case(s) from %s.",
        len(rows),
        eval_dir,
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composing function
# ---------------------------------------------------------------------------


def _inner_join_calibration_tables(
    dice_table: pd.DataFrame,
    conformal_table: pd.DataFrame,
    ood_table: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-joins the three per-case tables on `case_id`; a case needs all three to enter.

    Factored out of `build_gatekeeper_calibration_table` so the join behaviour --
    dropping a case present in only some of the three tables, logging how many, and
    raising only if nothing survives -- is directly testable with small, hand-built
    frames, with no QC checkpoint or conformal artifact involved.

    Args:
        dice_table: A frame with a `case_id` column plus `predicted_dice_<R>` columns
            (e.g. `qc_predicted_dice_table`'s output).
        conformal_table: A frame with a `case_id` column plus `conformal_band_<R>`
            columns.
        ood_table: A frame with `case_id` and `ood_score` columns (e.g.
            `ood_score_table`'s output).

    Returns:
        The inner join of all three on `case_id`.

    Raises:
        ValueError: The join produces zero rows.
    """
    all_ids = (
        set(dice_table["case_id"]) | set(conformal_table["case_id"]) | set(ood_table["case_id"])
    )
    merged = dice_table.merge(conformal_table, on="case_id", how="inner").merge(
        ood_table, on="case_id", how="inner"
    )

    n_dropped = len(all_ids) - len(merged)
    if n_dropped > 0:
        logger.warning(
            "build_gatekeeper_calibration_table: %d case(s) present in only some of the "
            "predicted-Dice (n=%d), conformal-band (n=%d) and ood-score (n=%d) tables were "
            "dropped by the inner join; %d case(s) survive.",
            n_dropped,
            len(dice_table),
            len(conformal_table),
            len(ood_table),
            len(merged),
        )

    if merged.empty:
        raise ValueError(
            "build_gatekeeper_calibration_table: the predicted-Dice, conformal-band and "
            "ood-score tables share no case_id at all; nothing to calibrate on."
        )
    return merged


def build_gatekeeper_calibration_table(cfg: Any) -> pd.DataFrame:
    """Builds the val-split table `neurovision.inference.gatekeeper.calibrate_thresholds` needs.

    Composes `qc_predicted_dice_table`, the conformal band-width machinery
    (`neurovision.uncertainty.conformal.load_curves_npz`, `resolve_fitted_thresholds`,
    `case_conformal_band_widths`) and `ood_score_table` (a STRUCTURAL PLACEHOLDER --
    see that function's docstring) into one table, inner-joined on `case_id`.

    Reads, from `cfg`: `cfg.analysis.qc_validate.checkpoint` (the trained QC
    checkpoint); `cfg.analysis.qc.train_eval_dir` / `cfg.analysis.qc.train_prep_dir`
    (the val split -- the SAME existing keys `scripts/train_qc.py` reads for its own
    training/selection split, not new ones); `cfg.analysis.qc.target_shape` /
    `cfg.analysis.qc.modality_index` (forwarded to `qc_predicted_dice_table`);
    `cfg.clinical.gatekeeper.regions`; `cfg.clinical.gatekeeper.conformal_dir` (the
    directory holding `fit.json` and `eval_val/curves.npz` for the deployed model's
    conformal run, e.g. `outputs/conformal/neurovision`); and
    `cfg.clinical.gatekeeper.conformal_alpha` (the alpha whose fitted threshold backs
    the deployed `conformal_band` signal).

    Args:
        cfg: The full composed Hydra config, exposing every key named above. `cfg.model`
            must already be composed as the `segqc` model group -- see
            `qc_predicted_dice_table`'s docstring for why.

    Returns:
        One row per case surviving the inner join, with columns `case_id`,
        `predicted_dice_<R>` and `conformal_band_<R>` for every `R` in
        `cfg.clinical.gatekeeper.regions`, plus `ood_score` -- exactly what
        `neurovision.inference.gatekeeper.calibrate_thresholds` requires.

    Raises:
        FileNotFoundError: The QC checkpoint, the conformal `curves.npz` / `fit.json`,
            or a case's preprocessed/logits files are missing.
        ValueError: The three component tables share no `case_id` at all
            (propagated from `_inner_join_calibration_tables`), or a fitted conformal
            threshold for a requested region/alpha is missing or infeasible
            (propagated from `resolve_fitted_thresholds`).
    """
    qc_cfg = cfg.analysis.qc
    gk_cfg = cfg.clinical.gatekeeper

    checkpoint = Path(str(cfg.analysis.qc_validate.checkpoint))
    eval_dir = Path(str(qc_cfg.train_eval_dir))
    prep_dir = Path(str(qc_cfg.train_prep_dir))
    regions = [str(r) for r in gk_cfg.regions]

    dice_table = qc_predicted_dice_table(cfg, checkpoint, eval_dir, prep_dir, regions)

    conformal_dir = Path(str(gk_cfg.conformal_dir))
    curves_path = conformal_dir / "eval_val" / "curves.npz"
    fit_path = conformal_dir / "fit.json"
    if not curves_path.is_file():
        raise FileNotFoundError(
            f"build_gatekeeper_calibration_table: no curves.npz at {curves_path}."
        )
    if not fit_path.is_file():
        raise FileNotFoundError(f"build_gatekeeper_calibration_table: no fit.json at {fit_path}.")

    curves_by_region = load_curves_npz(curves_path, regions)
    fit_payload = json.loads(fit_path.read_text())
    alpha = float(gk_cfg.conformal_alpha)
    fitted_thresholds = resolve_fitted_thresholds(fit_payload, regions, alpha)
    band_widths = case_conformal_band_widths(curves_by_region, fitted_thresholds)

    conformal_rows: dict[str, dict[str, Any]] = {}
    for region in regions:
        for case_id, value in band_widths[region].items():
            conformal_rows.setdefault(case_id, {"case_id": case_id})[
                f"conformal_band_{region}"
            ] = value
    conformal_table = pd.DataFrame(list(conformal_rows.values()))

    ood_table = ood_score_table(eval_dir)

    merged = _inner_join_calibration_tables(dice_table, conformal_table, ood_table)

    columns = (
        ["case_id"]
        + [f"predicted_dice_{r}" for r in regions]
        + [f"conformal_band_{r}" for r in regions]
        + ["ood_score"]
    )
    return merged[columns]
