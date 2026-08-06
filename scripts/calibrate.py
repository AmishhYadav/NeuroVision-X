"""Hydra entry point that produces a calibration report from two eval directories.

`src/neurovision/uncertainty/calibration.py` and `risk_coverage.py` are fully
implemented and unit tested, but nothing calls them: no entry point produces
a calibration number, so this project's headline claim ("competitive Dice
with substantially better CALIBRATION") has no result file on disk. This
script is that producer.

It is file-driven and CPU-only. It loads no checkpoint and runs no model.
Its inputs are artifacts `scripts/evaluate.py` already writes (`logits/`,
`probabilities/`, `per_case_metrics.csv`, `uncertainty_summary.csv`) plus the
preprocessed labels (`<prep_dir>/<case_id>/label.npy`).

## The central correctness requirement

Temperature is FIT on the VALIDATION split and APPLIED to the TEST split.
Fitting and reporting on the same split makes the reported number
meaningless -- T would be fit to that split's own noise -- so this is
enforced STRUCTURALLY here, not just documented: `resolve_eval_dirs` takes
two directories (`calibration.fit_dir`, `calibration.apply_dir`) and raises
if they resolve to the same path. There is no single-directory mode.

Example usage:

    python scripts/calibrate.py \\
        calibration.fit_dir=outputs/baseline_unet3d/eval_val \\
        calibration.apply_dir=outputs/baseline_unet3d/eval_test

The wiring is split into small functions (`resolve_eval_dirs`,
`resolve_source`, `load_case`, `subsample_masked_logits`,
`fit_split_temperature`, `accumulate`, `build_risk_coverage`,
`run_calibration`), mirroring `scripts/evaluate.py`'s decomposition, so each
piece can be unit tested without going through Hydra -- see
tests/test_calibrate_script.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor

from neurovision.data.transforms import REGION_NAMES
from neurovision.metrics.segmentation import classes_to_regions
from neurovision.uncertainty.calibration import (
    CalibrationAccumulator,
    TemperatureResult,
    apply_temperature,
    fit_temperature,
    union_foreground_mask,
)
from neurovision.uncertainty.risk_coverage import (
    oracle_curve,
    random_curve,
    referral_table,
    risk_coverage_curve,
    uncertainty_error_correlation,
)
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/evaluate.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

_VALID_SOURCES = ("auto", "logits", "probabilities")


# ---------------------------------------------------------------------------
# Directory / source resolution
# ---------------------------------------------------------------------------


def resolve_eval_dirs(cfg: DictConfig) -> tuple[Path, Path]:
    """Resolves and validates `calibration.fit_dir` / `calibration.apply_dir`.

    This is where the fit-on-val, report-on-test rule is enforced
    structurally: the two directories must both be set and must not resolve
    to the same path.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `(fit_dir, apply_dir)`, both existing directories.

    Raises:
        ValueError: Either key is `None` (names both keys and shows an
            example command line), or the two resolve (via `Path.resolve()`)
            to the same directory.
        FileNotFoundError: Either resolved path does not exist.
    """
    calib_cfg = cfg.calibration
    fit_dir_raw = calib_cfg.fit_dir
    apply_dir_raw = calib_cfg.apply_dir

    missing = []
    if fit_dir_raw is None:
        missing.append("calibration.fit_dir")
    if apply_dir_raw is None:
        missing.append("calibration.apply_dir")
    if missing:
        raise ValueError(
            f"{' and '.join(missing)} must be set. scripts/calibrate.py needs two eval "
            "directories: one to FIT the temperature on (the VAL split's eval_dir) and one to "
            "REPORT calibration numbers on (the TEST split's eval_dir). Example:\n"
            "  python scripts/calibrate.py "
            "calibration.fit_dir=outputs/<experiment>/eval_val "
            "calibration.apply_dir=outputs/<experiment>/eval_test"
        )

    fit_dir = Path(fit_dir_raw)
    apply_dir = Path(apply_dir_raw)

    if not fit_dir.is_dir():
        raise FileNotFoundError(f"calibration.fit_dir does not exist: {fit_dir.resolve()}")
    if not apply_dir.is_dir():
        raise FileNotFoundError(f"calibration.apply_dir does not exist: {apply_dir.resolve()}")

    if fit_dir.resolve() == apply_dir.resolve():
        raise ValueError(
            f"calibration.fit_dir and calibration.apply_dir both resolve to "
            f"{fit_dir.resolve()}. Fitting a temperature on the same split it is then REPORTED "
            "on fits T to that split's own noise -- the reported calibration number would be "
            "meaningless, and this project's headline claim rests on that number. Point "
            "fit_dir at the VAL split's eval_dir and apply_dir at the TEST split's."
        )

    return fit_dir, apply_dir


def _nonempty_npy_dir(path: Path) -> bool:
    """True if `path` is a directory containing at least one `.npy` file."""
    return path.is_dir() and any(path.glob("*.npy"))


def resolve_source(eval_dir: Path, requested: str) -> str:
    """Decides which subdirectory of `eval_dir` to read predictions from.

    Args:
        eval_dir: An eval directory written by `scripts/evaluate.py`.
        requested: `"auto"`, `"logits"`, or `"probabilities"` --
            `cfg.calibration.source`.

    Returns:
        `"logits"` or `"probabilities"`.

    Raises:
        FileNotFoundError: `requested="auto"` and neither `logits/` nor
            `probabilities/` exists (or both are empty) under `eval_dir`; or
            `requested` names one explicitly and it is absent -- an explicit
            request never silently falls back to the other source, since a
            silent logits->probabilities fallback is exactly how a
            temperature fit gets quietly poisoned (see this config's
            `source` comment).
        ValueError: `requested` is not one of `"auto"/"logits"/"probabilities"`.
    """
    if requested not in _VALID_SOURCES:
        raise ValueError(f"calibration.source must be one of {_VALID_SOURCES}, got {requested!r}.")

    logits_dir = eval_dir / "logits"
    probabilities_dir = eval_dir / "probabilities"

    if requested == "logits":
        if not _nonempty_npy_dir(logits_dir):
            raise FileNotFoundError(
                f"calibration.source='logits' but {logits_dir} does not exist or contains no "
                ".npy files. Re-run scripts/evaluate.py with "
                "inference.evaluation.save_logits=true against this eval_dir -- refusing to "
                "silently fall back to probabilities."
            )
        return "logits"

    if requested == "probabilities":
        if not _nonempty_npy_dir(probabilities_dir):
            raise FileNotFoundError(
                f"calibration.source='probabilities' but {probabilities_dir} does not exist or "
                "contains no .npy files. Re-run scripts/evaluate.py with "
                "inference.evaluation.save_probabilities=true against this eval_dir."
            )
        return "probabilities"

    # requested == "auto": prefer logits/, the only source a temperature fit
    # can use (see this config's `source` comment for why).
    if _nonempty_npy_dir(logits_dir):
        return "logits"
    if _nonempty_npy_dir(probabilities_dir):
        return "probabilities"
    raise FileNotFoundError(
        f"Neither {logits_dir} nor {probabilities_dir} exists (or both are empty). Re-run "
        "scripts/evaluate.py against this eval_dir with "
        "inference.evaluation.save_logits=true (preferred -- needed for temperature fitting) "
        "or inference.evaluation.save_probabilities=true (uncalibrated report only)."
    )


def _shared_case_ids(eval_dir: Path, source: str, prep_dir: Path) -> list[str]:
    """The case ids present as BOTH a `<source>/*.npy` file and a `<prep_dir>/<case_id>/` dir.

    Args:
        eval_dir: An eval directory written by `scripts/evaluate.py`.
        source: `"logits"` or `"probabilities"`, as returned by `resolve_source`.
        prep_dir: Root of the preprocessed BraTS data.

    Returns:
        Sorted list of shared case ids.
    """
    source_dir = eval_dir / source
    npy_ids = {p.stem for p in source_dir.glob("*.npy")}
    prep_ids = {p.name for p in prep_dir.iterdir() if p.is_dir()} if prep_dir.is_dir() else set()
    shared = sorted(npy_ids & prep_ids)

    npy_only = sorted(npy_ids - prep_ids)
    prep_only = sorted(prep_ids - npy_ids)
    if npy_only or prep_only:
        message = (
            f"_shared_case_ids: {len(npy_ids)} case(s) under {source_dir}, {len(prep_ids)} "
            f"case(s) under {prep_dir}; using {len(shared)} shared case(s)."
        )
        if npy_only:
            shown = npy_only if len(npy_only) <= 10 else npy_only[:10] + ["..."]
            message += f" Dropped (only in {source_dir}): {shown}."
        if prep_only:
            shown = prep_only if len(prep_only) <= 10 else prep_only[:10] + ["..."]
            message += f" Dropped (only under {prep_dir}): {shown}."
        logger.warning(message)
    else:
        logger.info(
            "_shared_case_ids: %d case(s) shared between %s and %s.",
            len(shared),
            source_dir,
            prep_dir,
        )
    return shared


# ---------------------------------------------------------------------------
# Per-case loading
# ---------------------------------------------------------------------------


def load_case(
    eval_dir: Path, source: str, prep_dir: Path, case_id: str
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Loads one case's prediction and ground-truth regions.

    Args:
        eval_dir: An eval directory written by `scripts/evaluate.py`.
        source: `"logits"` or `"probabilities"`.
        prep_dir: Root of the preprocessed BraTS data.
        case_id: The case identifier.

    Returns:
        `(prob, label, logits)`. `prob` and `label` are float32 CPU tensors
        of shape `(3, D, H, W)`, channel order `(ET, TC, WT)`. `logits` is
        the same shape when `source == "logits"` (the raw pre-sigmoid
        values, with `prob = sigmoid(logits)`), otherwise `None`.

    Raises:
        FileNotFoundError: The prediction `.npy` or `label.npy` is missing.
        ValueError: The loaded prediction array is not shaped `(3, D, H, W)`
            or `(1, 3, D, H, W)`, or its spatial shape disagrees with the
            label's. The latter usually means `predictions/` (ORIGINAL
            uncropped BraTS geometry) was pointed at instead of `logits/` /
            `probabilities/` (CROPPED geometry, matching `label.npy`), or
            that the two directories came from different preprocessing runs
            -- both are silent-misalignment traps, so the message names both
            shapes and both paths.
    """
    pred_path = eval_dir / source / f"{case_id}.npy"
    if not pred_path.is_file():
        raise FileNotFoundError(f"load_case({case_id!r}): {pred_path} does not exist.")

    arr = np.load(pred_path).astype(np.float32)
    x = torch.from_numpy(arr)
    if x.ndim == 5:
        if x.shape[0] != 1:
            raise ValueError(
                f"load_case({case_id!r}): {pred_path} has batch size {x.shape[0]}, expected 1."
            )
        x = x[0]
    if x.ndim != 4:
        raise ValueError(
            f"load_case({case_id!r}): {pred_path} has shape {tuple(arr.shape)}, expected "
            "(3, D, H, W) or (1, 3, D, H, W)."
        )

    if source == "logits":
        logits: Tensor | None = x
        prob = torch.sigmoid(x)
    else:
        logits = None
        prob = x

    label_path = prep_dir / case_id / "label.npy"
    if not label_path.is_file():
        raise FileNotFoundError(f"load_case({case_id!r}): {label_path} does not exist.")
    label_arr = np.load(label_path).astype(np.int64)
    # classes_to_regions always returns a batch axis (adds one of size 1 for
    # a (D, H, W) input); this is a single case, so it is squeezed straight
    # back off.
    label = classes_to_regions(torch.from_numpy(label_arr))[0]

    if prob.shape != label.shape:
        raise ValueError(
            f"load_case({case_id!r}): prediction shape {tuple(prob.shape)} from {pred_path} "
            f"disagrees with label shape {tuple(label.shape)} from {label_path}. This usually "
            "means predictions/ (ORIGINAL uncropped BraTS geometry) was pointed at instead of "
            "logits/ or probabilities/ (CROPPED geometry, matching label.npy) -- or that the "
            "two directories came from different preprocessing runs."
        )

    return prob, label, logits


def subsample_masked_logits(
    logits: Tensor,
    label: Tensor,
    mask: Tensor,
    n_samples: int,
    generator: torch.Generator,
) -> tuple[list[Tensor], list[Tensor]]:
    """Subsamples one case's masked voxels for the temperature fit, per channel.

    `uncertainty.calibration.subsample_voxels` is deliberately NOT reused
    here: it validates its first argument as a probability in `[0, 1]`, and
    this function's first argument is LOGITS, which are not probabilities
    and can be any real number. Loosening that check in the shared library
    function would remove a real guard from every OTHER caller of
    `subsample_voxels`, so this short sampling routine is duplicated here on
    purpose rather than weakening a shared check.

    Sampling is PER CHANNEL and each channel draws from its OWN mask row,
    never from a shared pool of spatial positions. Each region's reporting
    mask (`union_foreground_mask`) covers a different, mostly
    non-overlapping voxel set -- ET's is routinely a small fraction of
    WT's -- so a shared pool (e.g. the union of the three masks) would flood
    the ET and TC columns with WT-only edema voxels lying outside those
    regions' own reporting masks, exactly the dilution
    `union_foreground_mask` exists to prevent.

    The return is therefore RAGGED (a list per channel), not a rectangular
    `(N, C)` tensor. Making it rectangular would mean truncating every
    channel to the smallest count achieved, which has two consequences that
    are invisible once the numbers are in a table: ET's small mask would
    silently dictate the sample count for TC and WT as well, and a case with
    no enhancing tumor at all (2.6% of BraTS 2021, measured) would
    contribute ZERO voxels to the fit -- including its perfectly usable TC
    and WT voxels. `fit_split_temperature` fits each channel separately
    instead, which `fit_temperature`'s own docstring notes is mathematically
    identical to its joint fit (the per-channel losses are additively
    separable) and needs no rectangular input.

    Args:
        logits: Raw logits for one case, shape `(C, D, H, W)`.
        label: Binary region labels, same shape as `logits`.
        mask: Boolean reporting mask, same shape as `logits`.
        n_samples: Maximum number of voxels to draw PER CHANNEL.
        generator: An explicit `torch.Generator`, seeded by the caller.

    Returns:
        `(logits_samples, label_samples)`, each a list of `C` 1-D float32
        CPU tensors. Entry `c` has length `min(n_samples, masked voxels in
        channel c)`, and is empty when that channel's mask selected nothing.
    """
    num_channels = logits.shape[0]
    per_channel_logits: list[Tensor] = []
    per_channel_label: list[Tensor] = []
    for c in range(num_channels):
        flat_logits = logits[c].reshape(-1)
        flat_label = label[c].reshape(-1)
        flat_mask = mask[c].reshape(-1)
        idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        n_available = int(idx.numel())
        if n_available > n_samples:
            perm = torch.randperm(n_available, generator=generator)[:n_samples]
            idx = idx[perm]
        per_channel_logits.append(flat_logits[idx].to(torch.float32))
        per_channel_label.append(flat_label[idx].to(torch.float32))

    return per_channel_logits, per_channel_label


# ---------------------------------------------------------------------------
# Temperature fitting
# ---------------------------------------------------------------------------


def _fit_per_channel(
    channel_logits: list[list[Tensor]],
    channel_labels: list[list[Tensor]],
    per_channel: bool,
) -> TemperatureResult:
    """Fits temperature from RAGGED per-channel samples, via 1-D `fit_temperature` calls.

    `fit_temperature` accepts `(N,)` or `(N, C)`. The 2-D form requires the
    same number of samples in every channel, which this script's per-channel
    reporting masks cannot promise -- see `subsample_masked_logits`. Its own
    docstring notes that a joint `per_channel=True` fit is mathematically
    identical to `C` independent per-channel fits, because the per-channel
    losses are additively separable and a joint minimum is exactly the tuple
    of per-channel minima. So `C` separate 1-D calls give the same answer
    with no rectangular constraint, at the cost of `C` LBFGS runs instead of
    one (negligible: each runs on at most a few hundred thousand scalars).

    Args:
        channel_logits: `channel_logits[c]` is the list of that channel's
            per-case 1-D sample tensors.
        channel_labels: Matching binary labels.
        per_channel: True fits one temperature per channel, returning a
            `(C,)` tensor. False pools every channel's samples into ONE 1-D
            fit, returning a 0-D tensor -- note this weights each masked
            VOXEL equally, so a channel with more masked voxels contributes
            more, rather than weighting each channel equally as a
            rectangular `(N, C)` shared fit would.

    Returns:
        A `TemperatureResult`. `nll_before`/`nll_after` are averaged across
        channels weighted by each channel's sample count (an unweighted mean
        would let a channel with a handful of voxels move the reported NLL as
        much as one with hundreds of thousands). `converged` is True only if
        EVERY channel's fit converged -- one diverged channel makes the whole
        temperature vector unusable.
    """
    if not per_channel:
        pooled_logits = torch.cat([t for per_case in channel_logits for t in per_case])
        pooled_labels = torch.cat([t for per_case in channel_labels for t in per_case])
        pooled = fit_temperature(pooled_logits, pooled_labels, per_channel=False)
        # fit_temperature returns shape (1,) for 1-D input; the shared-T
        # convention elsewhere in this project is a 0-D scalar.
        return TemperatureResult(
            temperature=pooled.temperature.reshape(()),
            nll_before=pooled.nll_before,
            nll_after=pooled.nll_after,
            converged=pooled.converged,
        )

    temperatures: list[float] = []
    weights: list[float] = []
    nll_before: list[float] = []
    nll_after: list[float] = []
    converged = True
    for c, (logits_parts, label_parts) in enumerate(
        zip(channel_logits, channel_labels, strict=True)
    ):
        if not logits_parts:
            # No voxel of this region was ever in the reporting mask across
            # the whole fit split. T = 1.0 is the identity, i.e. "no scaling
            # was learned here" -- never a fitted-looking number.
            logger.warning(
                "_fit_per_channel: channel %d (%s) had no masked voxels anywhere in the fit "
                "split; its temperature stays at 1.0 (identity).",
                c,
                REGION_NAMES[c],
            )
            temperatures.append(1.0)
            weights.append(0.0)
            continue

        logits_c = torch.cat(logits_parts)
        labels_c = torch.cat(label_parts)
        result_c = fit_temperature(logits_c, labels_c, per_channel=False)
        temperatures.append(float(result_c.temperature.reshape(()).item()))
        weights.append(float(logits_c.numel()))
        nll_before.append(result_c.nll_before * logits_c.numel())
        nll_after.append(result_c.nll_after * logits_c.numel())
        converged = converged and result_c.converged

    total_weight = sum(weights)
    return TemperatureResult(
        temperature=torch.tensor(temperatures, dtype=torch.float32),
        nll_before=sum(nll_before) / total_weight if total_weight else float("nan"),
        nll_after=sum(nll_after) / total_weight if total_weight else float("nan"),
        converged=converged,
    )


def fit_split_temperature(
    cfg: DictConfig, fit_dir: Path, prep_dir: Path
) -> TemperatureResult | None:
    """Fits a temperature on `fit_dir`, streaming case by case.

    For each case: loads logits/label, builds the reporting mask with
    `union_foreground_mask(prob, label, threshold=cfg.calibration.threshold)`
    (from the UNCALIBRATED probabilities -- the mask must not itself depend
    on a temperature that has not been fit yet), subsamples
    `cfg.calibration.fit_voxels_per_case` masked voxels per case with a
    `torch.Generator` seeded from `cfg.calibration.seed`, and concatenates
    across cases before calling `fit_temperature`.

    Two calibrate.py-specific attributes, `n_voxels_fit` and `n_cases_fit`,
    are attached to the returned `TemperatureResult` (an ordinary,
    non-frozen dataclass with no `__slots__`) rather than threaded back
    through a wider return type: `TemperatureResult` itself lives in
    `uncertainty/calibration.py`, which this script must not modify, and
    every other caller of `fit_temperature` has no use for this
    provenance -- widening the shared dataclass for one caller's bookkeeping
    would be scope creep.

    Args:
        cfg: The full composed Hydra config.
        fit_dir: The VAL split's eval directory (see `resolve_eval_dirs`).
        prep_dir: Root of the preprocessed BraTS data.

    Returns:
        A `TemperatureResult`, or `None` (with an ERROR logged explaining
        why) when: `fit_dir`'s only available source is `"probabilities"`
        (fitting from fp16 probabilities is the poisoned fit described in
        `configs/calibration/default.yaml`'s `source` comment); no cases are
        found; every case has zero masked voxels; or the LBFGS fit
        diverged (`TemperatureResult.converged is False`). A diverged fit
        must never reach the reported table, so this returns `None` rather
        than raising -- the uncalibrated half of the analysis is still
        valid and should still be written.
    """
    calib_cfg = cfg.calibration

    source = resolve_source(fit_dir, calib_cfg.source)
    if source != "logits":
        logger.error(
            "fit_split_temperature: fit_dir=%s's only available source is %r, not 'logits'. "
            "Fitting a temperature from fp16 probabilities would systematically understate "
            "overconfidence (fp16 saturates any probability above ~0.99976 to exactly 1.0, "
            "whose logit is +inf) -- refusing to fit. Re-run scripts/evaluate.py with "
            "inference.evaluation.save_logits=true against this eval_dir.",
            fit_dir,
            source,
        )
        return None

    case_ids = _shared_case_ids(fit_dir, source, prep_dir)
    if not case_ids:
        logger.error(
            "fit_split_temperature: no cases found in both %s and %s -- cannot fit a "
            "temperature.",
            fit_dir / source,
            prep_dir,
        )
        return None

    generator = torch.Generator().manual_seed(int(calib_cfg.seed))
    num_channels = len(REGION_NAMES)
    channel_logits: list[list[Tensor]] = [[] for _ in range(num_channels)]
    channel_labels: list[list[Tensor]] = [[] for _ in range(num_channels)]
    n_cases_used = 0
    for case_id in case_ids:
        prob, label, logits = load_case(fit_dir, source, prep_dir, case_id)
        assert logits is not None  # guaranteed by source == "logits" above
        mask = union_foreground_mask(prob, label, threshold=float(calib_cfg.threshold))
        sampled_logits, sampled_labels = subsample_masked_logits(
            logits, label, mask, int(calib_cfg.fit_voxels_per_case), generator
        )
        contributed = False
        for c in range(num_channels):
            if sampled_logits[c].numel() > 0:
                channel_logits[c].append(sampled_logits[c])
                channel_labels[c].append(sampled_labels[c])
                contributed = True
        # A case counts as used if ANY channel contributed. A case with no
        # enhancing tumor still has usable TC and WT voxels, and dropping it
        # outright would let the smallest region decide the fit set.
        n_cases_used += int(contributed)

    if not any(channel_logits):
        logger.error(
            "fit_split_temperature: every case in %s had zero masked voxels; cannot fit a "
            "temperature.",
            fit_dir,
        )
        return None

    result = _fit_per_channel(
        channel_logits, channel_labels, per_channel=bool(calib_cfg.per_channel)
    )
    n_voxels = sum(int(t.numel()) for per_case in channel_logits for t in per_case)
    result.n_voxels_fit = n_voxels  # type: ignore[attr-defined]
    result.n_cases_fit = n_cases_used  # type: ignore[attr-defined]

    if not result.converged:
        logger.error(
            "fit_split_temperature: LBFGS diverged (nll_after=%.6f > nll_before=%.6f) on %d "
            "voxel(s) from %d case(s); the fitted temperature is unusable and will NOT be "
            "applied.",
            result.nll_after,
            result.nll_before,
            n_voxels,
            n_cases_used,
        )
        return None

    logger.info(
        "fit_split_temperature: fit on %d voxel(s) from %d case(s) in %s -- nll %.6f -> %.6f, "
        "temperature=%s.",
        n_voxels,
        n_cases_used,
        fit_dir,
        result.nll_before,
        result.nll_after,
        result.temperature.tolist(),
    )
    return result


# ---------------------------------------------------------------------------
# Accumulation (uncalibrated + temperature-scaled)
# ---------------------------------------------------------------------------


def accumulate(
    cfg: DictConfig,
    eval_dir: Path,
    source: str,
    prep_dir: Path,
    case_ids: list[str],
    temperature: Tensor | None = None,
) -> dict[str, CalibrationAccumulator]:
    """ONE pass over `case_ids`, folding each case into every requested variant.

    Both variants are built in a single pass, deliberately. Two passes would
    have to either re-read every case's array from disk (each logits file is
    ~53 MB, so a 189-case split is ~10 GB read twice) or hold every case's
    reporting mask in memory to keep the passes consistent (~9.8 MB per case
    as a 3-channel bool volume, ~1.9 GB across the split). One pass needs
    neither: the case is loaded once, the mask is derived once, both
    accumulators consume it, and nothing survives the iteration.

    Consistency of the mask across variants is the reason it must not simply
    be recomputed per variant. (a) The two variants must cover exactly the
    same voxel set or their ECEs are not comparable. (b) At
    `threshold = 0.5` the mask is provably identical anyway, since
    temperature scaling is strictly monotone -- so recomputing it would be a
    silent no-op that stops being one the moment the threshold moves off
    0.5, and the resulting mismatch would look like a calibration effect.

    Args:
        cfg: The full composed Hydra config.
        eval_dir: The eval directory to read predictions from (normally
            `apply_dir`).
        source: `"logits"` or `"probabilities"`, as returned by
            `resolve_source`.
        prep_dir: Root of the preprocessed BraTS data.
        case_ids: Case ids to accumulate, in order.
        temperature: When given, a second `"temperature_scaled"` variant is
            accumulated alongside the uncalibrated one, applying
            `apply_temperature` to the logits and re-`sigmoid`ing. Requires
            `source == "logits"` -- the caller guarantees that, since
            `load_case` returns `logits=None` for a probabilities source.

    Returns:
        `{"uncalibrated": accumulator}`, plus `"temperature_scaled"` when
        `temperature` was given.
    """
    calib_cfg = cfg.calibration
    accumulators: dict[str, CalibrationAccumulator] = {
        "uncalibrated": CalibrationAccumulator(n_bins=int(calib_cfg.n_bins))
    }
    if temperature is not None:
        accumulators["temperature_scaled"] = CalibrationAccumulator(n_bins=int(calib_cfg.n_bins))

    for case_id in case_ids:
        prob, label, logits = load_case(eval_dir, source, prep_dir, case_id)

        # Computed from the UNCALIBRATED probabilities, never the scaled
        # ones -- see this function's docstring.
        mask = union_foreground_mask(prob, label, threshold=float(calib_cfg.threshold))
        accumulators["uncalibrated"].add_case(case_id, prob, label, mask=mask)

        if temperature is not None:
            assert logits is not None  # guaranteed by source == "logits"
            # apply_temperature expects a batch axis (B, C, D, H, W); add
            # and remove one here since load_case works in single-case
            # (C, D, H, W) terms throughout this script.
            scaled_logits = apply_temperature(logits.unsqueeze(0), temperature)[0]
            accumulators["temperature_scaled"].add_case(
                case_id, torch.sigmoid(scaled_logits), label, mask=mask
            )

    return accumulators


def _summary_with_variant(accumulator: CalibrationAccumulator, variant: str) -> pd.DataFrame:
    """`accumulator.summary()`, tagged with a `variant` column, `metric` as a plain column."""
    df = accumulator.summary().rename_axis("metric").reset_index()
    df["variant"] = variant
    return df


def _per_case_with_variant(accumulator: CalibrationAccumulator, variant: str) -> pd.DataFrame:
    """`accumulator.per_case()`, tagged with a `variant` column, `case_id` as a plain column."""
    df = accumulator.per_case().rename_axis("case_id").reset_index()
    df["variant"] = variant
    return df


# ---------------------------------------------------------------------------
# Risk-coverage
# ---------------------------------------------------------------------------


def build_risk_coverage(cfg: DictConfig, apply_dir: Path) -> dict[str, Any] | None:
    """Builds the risk-coverage curve, referral table, and correlation for `apply_dir`.

    Args:
        cfg: The full composed Hydra config.
        apply_dir: The TEST split's eval directory.

    Returns:
        `None`, with ONE clear reason logged, when: `risk_coverage.enabled`
        is False; `<apply_dir>/uncertainty_summary.csv` is absent (the eval
        run had `inference.mc_dropout.enabled=false`);
        `<apply_dir>/per_case_metrics.csv` is absent; a configured column
        (`uncertainty_column`/`score_column`) is missing after the inner
        join (names the column and lists what columns ARE present); or
        fewer than 2 cases survive the join. Otherwise a dict:
        `"risk_coverage"` (a `pd.DataFrame`, the curve), `"referral_table"`
        (a `pd.DataFrame`), and `"uncertainty_correlation"` (a plain dict of
        JSON-serializable scalars -- Spearman/Pearson correlation plus the
        model/oracle/random AURC and which columns were used).
    """
    rc_cfg = cfg.calibration.risk_coverage
    if not rc_cfg.enabled:
        logger.info("build_risk_coverage: calibration.risk_coverage.enabled=false; skipping.")
        return None

    uncertainty_path = apply_dir / "uncertainty_summary.csv"
    metrics_path = apply_dir / "per_case_metrics.csv"

    if not uncertainty_path.is_file():
        logger.warning(
            "build_risk_coverage: %s does not exist -- the eval run that produced %s likely had "
            "inference.mc_dropout.enabled=false. Skipping risk-coverage analysis.",
            uncertainty_path,
            apply_dir,
        )
        return None
    if not metrics_path.is_file():
        logger.warning(
            "build_risk_coverage: %s does not exist. Skipping risk-coverage analysis.",
            metrics_path,
        )
        return None

    uncertainty_df = pd.read_csv(uncertainty_path, index_col="case_id")
    metrics_df = pd.read_csv(metrics_path, index_col="case_id")
    joined = uncertainty_df.join(metrics_df, how="inner")

    uncertainty_col = str(rc_cfg.uncertainty_column)
    score_col = str(rc_cfg.score_column)
    missing = [c for c in (uncertainty_col, score_col) if c not in joined.columns]
    if missing:
        logger.warning(
            "build_risk_coverage: column(s) %s not found after inner-joining %s and %s. "
            "Columns present: %s. Skipping risk-coverage analysis.",
            missing,
            uncertainty_path,
            metrics_path,
            sorted(joined.columns),
        )
        return None

    if len(joined) < 2:
        logger.warning(
            "build_risk_coverage: only %d case(s) survive the inner join between %s and %s "
            "(need >= 2). Skipping risk-coverage analysis.",
            len(joined),
            uncertainty_path,
            metrics_path,
        )
        return None

    uncertainty = joined[uncertainty_col].to_numpy()
    score = joined[score_col].to_numpy()
    higher_is_better = bool(rc_cfg.score_higher_is_better)

    model_curve = risk_coverage_curve(uncertainty, score, higher_is_better=higher_is_better)
    oracle = oracle_curve(score, higher_is_better=higher_is_better)
    random_ = random_curve(score, higher_is_better=higher_is_better)

    # The oracle and random curves are saved ALONGSIDE the model's, not left in
    # memory, because a risk-coverage figure without them cannot be read: the
    # oracle is the ceiling no uncertainty estimate can beat, and a model curve
    # that hugs the random line means the uncertainty carries no information
    # about case difficulty. Both are computed over the same k = 1..N coverage
    # grid, so the three columns align row for row.
    curve_table = pd.DataFrame(
        {
            "coverage": model_curve.coverage,
            "n_retained": model_curve.n_retained,
            "performance": model_curve.performance,
            "risk": model_curve.risk,
            "oracle_performance": oracle.performance,
            "random_performance": random_.performance,
        }
    )
    ref_table = referral_table(
        uncertainty,
        score,
        coverage_points=list(rc_cfg.coverage_points),
        higher_is_better=higher_is_better,
    )
    correlation = uncertainty_error_correlation(uncertainty, score)

    return {
        "risk_coverage": curve_table,
        "referral_table": ref_table,
        "uncertainty_correlation": {
            **correlation,
            "aurc_model": model_curve.aurc,
            "aurc_oracle": oracle.aurc,
            "aurc_random": random_.aurc,
            "uncertainty_column": uncertainty_col,
            "score_column": score_col,
            "n_cases": int(len(joined)),
        },
    }


# ---------------------------------------------------------------------------
# temperature.json
# ---------------------------------------------------------------------------


def _temperature_payload(
    result: TemperatureResult | None,
    fit_dir: Path,
    apply_dir: Path,
    reason: str | None,
    source: str,
) -> dict[str, Any]:
    """Assembles the `temperature.json` payload -- written even when the fit failed."""
    payload: dict[str, Any] = {
        "fit_dir": str(fit_dir),
        "apply_dir": str(apply_dir),
        "source": source,
        "converged": bool(result.converged) if result is not None else False,
    }
    if result is None:
        payload["reason"] = reason or "no temperature was fit or supplied; see the run log."
        payload["temperature"] = None
        return payload

    temperature = result.temperature
    if temperature.ndim == 0:
        payload["temperature"] = {"shared": float(temperature.item())}
    elif temperature.numel() == len(REGION_NAMES):
        payload["temperature"] = {
            region: float(value) for region, value in zip(REGION_NAMES, temperature.tolist())
        }
    else:
        payload["temperature"] = temperature.tolist()

    payload["nll_before"] = result.nll_before
    payload["nll_after"] = result.nll_after
    payload["n_voxels_fit"] = getattr(result, "n_voxels_fit", None)
    payload["n_cases_fit"] = getattr(result, "n_cases_fit", None)
    if reason:
        payload["reason"] = reason
    return payload


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _print_summary(
    acc_uncalibrated: CalibrationAccumulator,
    acc_scaled: CalibrationAccumulator | None,
    apply_dir: Path,
    n_cases: int,
) -> None:
    """Prints (not logs -- see module docstring) a compact end-of-run summary."""
    lines = [
        "=" * 70,
        f"Calibration summary -- apply_dir={apply_dir}, {n_cases} case(s)",
        "=" * 70,
    ]
    for label, accumulator in (
        ("uncalibrated", acc_uncalibrated),
        ("temperature_scaled", acc_scaled),
    ):
        if accumulator is None:
            continue
        summary = accumulator.summary()
        lines.append(f"  [{label}]")
        for region in accumulator.region_names:
            ece_key, brier_key = f"ece_{region}", f"brier_{region}"
            if ece_key in summary.index:
                lines.append(
                    f"    {region}: ece={summary.loc[ece_key, 'value']:.4f}  "
                    f"brier={summary.loc[brier_key, 'value']:.4f}"
                )
        if "ece_mean" in summary.index:
            lines.append(f"    ece_mean = {summary.loc['ece_mean', 'value']:.4f}")

    # print only, not logger.info as well: setup_logging's StreamHandler
    # already targets stdout, so doing both emits this block twice. Matches
    # scripts/evaluate.py's _log_and_print_summary.
    print("\n".join(lines))


def run_calibration(cfg: DictConfig) -> dict[str, Any]:
    """Orchestrates the full calibration report: fit, score, write, print.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A dict mapping a short name to the `Path` (or list of `Path`s) each
        output was written to, so tests can assert on what was produced
        without re-parsing every CSV.
    """
    calib_cfg = cfg.calibration
    fit_dir, apply_dir = resolve_eval_dirs(cfg)
    prep_dir = Path(cfg.data.preprocessing.out_dir)
    out_dir = ensure_dir(calib_cfg.out_dir)

    apply_source = resolve_source(apply_dir, calib_cfg.source)
    apply_case_ids = _shared_case_ids(apply_dir, apply_source, prep_dir)
    if not apply_case_ids:
        raise ValueError(
            f"No cases found in both {apply_dir / apply_source} and {prep_dir}; nothing to "
            "calibrate."
        )
    logger.info(
        "run_calibration: apply_dir=%s source=%s -- %d case(s).",
        apply_dir,
        apply_source,
        len(apply_case_ids),
    )

    # --- 1. Temperature: explicit override, or fit on fit_dir ---
    explicit = calib_cfg.temperature
    temp_result: TemperatureResult | None
    temp_reason: str | None = None

    if explicit is not None:
        if isinstance(explicit, (list, tuple, ListConfig)):
            temperature = torch.tensor([float(v) for v in explicit], dtype=torch.float32)
        else:
            temperature = torch.tensor(float(explicit), dtype=torch.float32)
        # nll_before/after have no meaning for an explicit override -- NaN
        # rather than 0.0, so a reader cannot mistake this for a measured fit.
        temp_result = TemperatureResult(
            temperature=temperature, nll_before=float("nan"), nll_after=float("nan"), converged=True
        )
        logger.info(
            "run_calibration: calibration.temperature explicitly set to %s -- skipping "
            "fit_split_temperature entirely.",
            temperature.tolist(),
        )
    else:
        temp_result = fit_split_temperature(cfg, fit_dir, prep_dir)
        if temp_result is None:
            temp_reason = (
                "fit_split_temperature returned None (fit_dir has no logits/ source, the LBFGS "
                "fit diverged, or no masked voxels were available) -- see the ERROR log line "
                "above for the exact cause."
            )

    apply_temperature_now = temp_result is not None
    if apply_temperature_now and apply_source != "logits":
        logger.error(
            "run_calibration: a temperature is available but apply_dir's only source is %r, "
            "not 'logits' -- temperature scaling needs raw logits. Skipping the "
            "temperature_scaled variant; the uncalibrated numbers are still written.",
            apply_source,
        )
        apply_temperature_now = False
        temp_reason = f"apply_dir source is {apply_source!r}, not 'logits'; cannot apply."

    temperature_json_path = out_dir / "temperature.json"
    write_json(
        _temperature_payload(temp_result, fit_dir, apply_dir, temp_reason, apply_source),
        temperature_json_path,
    )
    written: dict[str, Any] = {"temperature_json": temperature_json_path}

    # --- 2. Score every variant in ONE pass over the split ---
    accumulators = accumulate(
        cfg,
        apply_dir,
        apply_source,
        prep_dir,
        apply_case_ids,
        temperature=temp_result.temperature if apply_temperature_now else None,
    )
    acc_uncalibrated = accumulators["uncalibrated"]
    acc_scaled = accumulators.get("temperature_scaled")

    summary_frames = []
    per_case_frames = []
    reliability_paths: list[Path] = []
    for variant, accumulator in accumulators.items():
        summary_frames.append(_summary_with_variant(accumulator, variant))
        per_case_frames.append(_per_case_with_variant(accumulator, variant))
        for region in accumulator.region_names:
            path = out_dir / f"reliability_{variant}_{region}.csv"
            accumulator.reliability(region).to_csv(path, index=False)
            reliability_paths.append(path)

    metrics_path = out_dir / "calibration_metrics.csv"
    pd.concat(summary_frames, ignore_index=True).to_csv(metrics_path, index=False)
    written["calibration_metrics_csv"] = metrics_path

    per_case_path = out_dir / "per_case_calibration.csv"
    pd.concat(per_case_frames, ignore_index=True).to_csv(per_case_path, index=False)
    written["per_case_calibration_csv"] = per_case_path
    written["reliability_csvs"] = reliability_paths

    # --- 4. Risk-coverage (independent of the temperature fit) ---
    risk_coverage_result = build_risk_coverage(cfg, apply_dir)
    if risk_coverage_result is not None:
        risk_coverage_csv = out_dir / "risk_coverage.csv"
        referral_csv = out_dir / "referral_table.csv"
        correlation_json = out_dir / "uncertainty_correlation.json"

        risk_coverage_result["risk_coverage"].to_csv(risk_coverage_csv, index=False)
        risk_coverage_result["referral_table"].to_csv(referral_csv)
        write_json(risk_coverage_result["uncertainty_correlation"], correlation_json)

        written["risk_coverage_csv"] = risk_coverage_csv
        written["referral_table_csv"] = referral_csv
        written["uncertainty_correlation_json"] = correlation_json

    # --- 5. Config dump, exactly like scripts/evaluate.py's eval_config.yaml ---
    calibration_config_path = out_dir / "calibration_config.yaml"
    calibration_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    written["calibration_config_yaml"] = calibration_config_path

    _print_summary(acc_uncalibrated, acc_scaled, apply_dir, len(apply_case_ids))

    return written


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the calibration report, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_calibration(cfg)


if __name__ == "__main__":
    main()
