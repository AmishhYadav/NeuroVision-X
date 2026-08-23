"""Hydra entry point for evaluating a trained checkpoint on a frozen split.

Runs sliding-window inference and BraTS-style region metrics (Dice, IoU,
HD95) over every case of a split, and writes everything an evaluation report
needs to disk as it goes: per-case metrics, a summary table, uncropped
prediction volumes, and (optionally) probability maps for later calibration
analysis. Device is resolved once from config via
`neurovision.utils.device.get_device`, exactly like `scripts/train.py`.

Also optionally runs MC-dropout uncertainty estimation
(`neurovision.inference.mc_dropout`), gated behind
`cfg.inference.mc_dropout.enabled` (default off, so a plain evaluation run's
output is byte-identical to before this was wired in). When enabled, it
writes one `.npy` array per case per configured field under
`<out_dir>/<field_dir>/` (see `_MC_FIELD_TO_DIR` below -- `mutual_information`
always lands in `uncertainty/`, which is the fixed contract
`notebooks/09_paper_figures.ipynb` reads from) plus a per-case
`uncertainty_summary.csv` of scalar summaries. Segmentation metrics still
come from the deterministic sliding-window pass unless
`cfg.inference.mc_dropout.predictions_from` is explicitly set to
`"mc_mean"`.

Example usage:

    python scripts/evaluate.py inference.evaluation.split=test
    python scripts/evaluate.py inference.evaluation.split=test inference.mc_dropout.enabled=true

The wiring is split into small functions (`build_eval_dataloader`,
`resolve_checkpoint`, `load_eval_model`, `evaluate_case`, `run_evaluation`)
mirroring `scripts/train.py`'s decomposition, so each piece can be unit
tested without going through Hydra -- see tests/test_evaluate_script.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from monai.data import list_data_collate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurovision.data.dataset import build_data_dicts, build_dataset, load_splits
from neurovision.data.transforms import REGION_NAMES, build_val_transforms
from neurovision.inference.mc_dropout import (
    MCDropoutOutput,
    logits_from_mean_prob,
    mc_dropout_predict,
)
from neurovision.inference.postprocess import (
    postprocess_logits,
    regions_to_classes,
    uncrop_to_original,
)
from neurovision.inference.sliding_window import sliding_window_predict

# Importing these registers the "unet3d"/"swinunetr" and "dice_ce" builders
# (the @register_model / @register_loss decorators run on import) before
# build_model is ever called below. evaluate.py never calls build_loss
# itself, but a checkpoint's stored config (see ResumeState.config) still
# names a loss under cfg.training.loss.name, and importing the module here
# keeps that name resolvable and this script's registry side effects
# identical to scripts/train.py's, rather than depending on train.py having
# been imported first in the same process. Copied from scripts/train.py.
from neurovision.losses import segmentation  # noqa: F401
from neurovision.metrics.boundary import boundary_stratified_errors
from neurovision.metrics.lesionwise import _load_panoptica, lesionwise_case_metrics
from neurovision.metrics.segmentation import MetricAggregator, compute_case_metrics
from neurovision.models import baseline  # noqa: F401
from neurovision.models.registry import build_model
from neurovision.training.checkpoint import ResumeState, load_checkpoint
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, read_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/train.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# One directory name per MCDropoutOutput tensor field. Fixed and explicit
# (never derived from the field name itself), because `mutual_information ->
# "uncertainty"` is a contract `notebooks/09_paper_figures.ipynb` reads from
# directly -- renaming it here would silently break that notebook's fallback
# logic with no error anywhere.
_MC_FIELD_TO_DIR: dict[str, str] = {
    "mutual_information": "uncertainty",
    "predictive_entropy": "entropy_total",
    "expected_entropy": "entropy_aleatoric",
    "mean_prob": "mc_mean_prob",
}

# Storage-guard threshold: above this projected total, the one-time log line
# below calls out Kaggle's 20 GB /kaggle/working quota explicitly.
_MC_STORAGE_WARN_BYTES = 15 * (1024**3)

_VALID_MC_PREDICTIONS_FROM = {"deterministic", "mc_mean"}


def build_eval_dataloader(cfg: DictConfig, split: str) -> tuple[DataLoader, list[str]]:
    """Builds the evaluation `DataLoader` for one frozen split.

    `batch_size=1` is mandatory here, not a tunable default: whole volumes
    have different shapes per case (each was cropped to its own nonzero
    bounding box in preprocessing), and they will not collate at any batch
    size above 1.

    Args:
        cfg: The full composed Hydra config.
        split: Which frozen split to load -- `"train"`, `"val"`, or
            `"test"`, as written by `neurovision.data.dataset.make_splits`.

    Returns:
        `(loader, case_ids)`. `case_ids` is in the SAME order the loader
        yields batches in (the split file's list order, unshuffled), so a
        caller can `zip(case_ids, loader)` to know which case each batch is.

    Raises:
        ValueError: If `split` is not one of the split file's keys, or if
            the requested split has zero cases -- an empty split would
            otherwise silently produce an empty CSV and an all-NaN summary
            that looks like a model failure rather than a config mistake.
    """
    splits = load_splits(cfg.data.splits.path)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}. Available splits: {sorted(splits.keys())}.")

    case_ids = list(splits[split])
    if not case_ids:
        raise ValueError(
            f"Split {split!r} has 0 cases (splits file: {cfg.data.splits.path}). Evaluating "
            "an empty split would silently produce an empty per_case_metrics.csv and an "
            "all-NaN summary.csv that looks like a model failure rather than a config error."
        )

    prep_dir = cfg.data.preprocessing.out_dir
    data_dicts = build_data_dicts(case_ids, prep_dir)
    transform = build_val_transforms(cfg)

    # dataset_type="dataset" (never "cache"/"persistent"): each case in an
    # evaluation pass is visited exactly once, so there is nothing to reuse
    # and caching would only add memory/disk overhead for no benefit.
    dataset = build_dataset(data_dicts, transform, dataset_type="dataset")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=list_data_collate,
    )
    return loader, case_ids


def resolve_checkpoint(cfg: DictConfig) -> Path:
    """Decides which checkpoint file to evaluate.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `Path(cfg.inference.evaluation.checkpoint)` if that is set, else
        `Path(cfg.training.checkpoint.dir) / "best.pt"`.

    Raises:
        FileNotFoundError: If the resolved path does not exist. The message
            lists whatever `.pt` files ARE present in that directory, since
            the common mistake this guards against is evaluating before any
            `best.pt` has been written yet.
    """
    explicit = cfg.inference.evaluation.checkpoint
    if explicit is not None:
        checkpoint_path = Path(explicit)
    else:
        checkpoint_path = Path(cfg.training.checkpoint.dir) / "best.pt"

    if not checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir.is_dir():
            available = sorted(p.name for p in checkpoint_dir.glob("*.pt"))
        else:
            available = []
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path.resolve()}. "
            f".pt files present in {checkpoint_dir.resolve()}: "
            f"{available if available else '(none -- directory is empty or does not exist)'}."
        )
    return checkpoint_path


def load_eval_model(
    cfg: DictConfig, checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, ResumeState]:
    """Builds the model from config and loads a checkpoint's weights into it.

    Args:
        cfg: The full composed Hydra config.
        checkpoint_path: Path to the checkpoint, as returned by
            `resolve_checkpoint`.
        device: The resolved torch device to move the model to.

    Returns:
        `(model, resume_state)`. `resume_state` is returned (not just the
        model) so the caller -- and this function's own architecture check --
        can read the checkpoint's stored training config and metadata.

    Raises:
        ValueError: If `cfg.inference.evaluation.strict_arch_check` is True,
            the checkpoint has a stored config, and that config's
            `model.name` disagrees with the current `cfg.model.name`.
            `load_state_dict(strict=True)` (used internally by
            `load_checkpoint`) only catches a gross architecture swap --
            e.g. `unet3d` weights loaded into a `swinunetr` module raise
            immediately. It does NOT catch a checkpoint trained under one
            `model.*` config (e.g. a different `feature_size` or
            `channels`) being evaluated under a config that merely claims
            the same `model.name`: that can load partially (silently
            dropping mismatched-shape parameters under some MONAI model
            classes) or run to completion and produce numbers that are
            quietly not what the current config says was evaluated.
    """
    model = build_model(cfg)
    model = model.to(device)

    # restore_rng=False: evaluation is deterministic (no training-time random
    # transforms, no optimizer step) and has no reason to perturb the
    # process's RNG state. Restoring a TRAINING run's RNG state here would
    # just be a confusing side effect with no benefit to inference.
    resume_state = load_checkpoint(
        checkpoint_path, model, map_location=str(device), restore_rng=False
    )

    eval_cfg = cfg.inference.evaluation
    if eval_cfg.strict_arch_check and resume_state.config is not None:
        checkpoint_model_name = resume_state.config.model.name
        current_model_name = cfg.model.name
        if checkpoint_model_name != current_model_name:
            raise ValueError(
                f"Checkpoint {checkpoint_path} was trained with model.name="
                f"{checkpoint_model_name!r}, but the current config has model.name="
                f"{current_model_name!r}. Refusing to evaluate: load_state_dict(strict=True) "
                "only catches a gross architecture swap, not a checkpoint trained under "
                "different model settings being scored under a config that claims something "
                "else. Set inference.evaluation.strict_arch_check=false to override."
            )

    # This line is how a user confirms in the log which checkpoint actually
    # got scored: epoch, and the metric/value it was selected as "best" on.
    logger.info(
        "Loaded checkpoint %s for evaluation: epoch=%d, best_metric_name=%s, best_metric=%s",
        checkpoint_path,
        resume_state.start_epoch - 1,
        resume_state.best_metric_name,
        resume_state.best_metric,
    )
    return model, resume_state


@dataclass
class CaseOutput:
    """Everything `evaluate_case` produces for one case.

    A dataclass rather than a tuple: `mc` is sometimes `None` and a 3-tuple
    whose last element is sometimes absent is exactly the kind of positional
    API that gets mis-unpacked at a call site.

    Attributes:
        regions: Binary float tensor, shape `(1, 3, D, H, W)`, channel order
            `(ET, TC, WT)`. This is what `run_evaluation` saves as the
            prediction and scores against the ground truth. Comes from the
            deterministic sliding-window pass unless
            `cfg.inference.mc_dropout.predictions_from == "mc_mean"`, in
            which case it is re-derived from the MC-dropout mean pass
            instead (see `evaluate_case`'s docstring).
        probabilities: `torch.sigmoid(logits)` from the deterministic pass,
            same shape as `regions`, when
            `cfg.inference.evaluation.save_probabilities` is True;
            otherwise `None` so a ~53 MB-per-case tensor is never
            materialized when it is not going to be written to disk.
        logits: The RAW logits from the deterministic pass, same shape as
            `regions`, when `cfg.inference.evaluation.save_logits` is True;
            otherwise `None`. Kept separately from `probabilities` because
            temperature scaling needs logits and cannot recover them from
            fp16-saved probabilities -- see that config key's comment.
        mc: The `MCDropoutOutput` from the extra stochastic passes, when
            `cfg.inference.mc_dropout.enabled` is True; otherwise `None`.
    """

    regions: Tensor
    probabilities: Tensor | None
    logits: Tensor | None
    mc: MCDropoutOutput | None


def _validate_mc_dropout_config(mc_cfg: DictConfig) -> None:
    """Validates `cfg.inference.mc_dropout` before any inference has run.

    Called once, at the top of `run_evaluation`, before the checkpoint is
    loaded or `out_dir` is created -- a config typo here should fail
    immediately, not after minutes of sliding-window inference on the first
    case.

    Args:
        mc_cfg: `cfg.inference.mc_dropout`. A no-op when `mc_cfg.enabled` is
            False, since neither field this checks is read anywhere in that
            case.

    Raises:
        ValueError: If `mc_cfg.predictions_from` is not one of
            `{"deterministic", "mc_mean"}`, or if `mc_cfg.save_fields`
            names a field `MCDropoutOutput` does not have.
    """
    if not mc_cfg.enabled:
        return

    if mc_cfg.predictions_from not in _VALID_MC_PREDICTIONS_FROM:
        raise ValueError(
            f"cfg.inference.mc_dropout.predictions_from={mc_cfg.predictions_from!r} is not "
            f"valid. Choose one of {sorted(_VALID_MC_PREDICTIONS_FROM)}."
        )

    # Derived from the dataclass itself (minus num_samples, which is a count,
    # not a per-voxel field) rather than a second hardcoded list -- so this
    # check cannot silently drift out of sync with MCDropoutOutput.
    valid_fields = {f.name for f in dataclass_fields(MCDropoutOutput) if f.name != "num_samples"}
    unknown = [name for name in mc_cfg.save_fields if name not in valid_fields]
    if unknown:
        raise ValueError(
            f"cfg.inference.mc_dropout.save_fields contains unknown field(s) {unknown}. "
            f"Valid MCDropoutOutput fields are {sorted(valid_fields)}."
        )


def resolve_boundary_bands(eval_cfg: DictConfig) -> tuple[tuple[float, float], ...] | None:
    """Reads and validates `cfg.inference.evaluation.boundary_bands`.

    Args:
        eval_cfg: `cfg.inference.evaluation`. The key is read with a default
            of `None` rather than attribute access, so a config composed
            before this key existed (an older saved `eval_config.yaml`, a
            minimal test config) still runs with the analysis simply off.

    Returns:
        The bands as a tuple of `(lo, hi)` float pairs, or `None` when the
        analysis is disabled.

    Raises:
        ValueError: If an entry is not a 2-element pair. Band ordering and
            overlap are validated downstream by
            `neurovision.metrics.boundary.boundary_band_masks`, which owns
            that rule -- duplicating it here would let the two drift apart.
    """
    raw = eval_cfg.get("boundary_bands", None)
    if raw is None:
        return None

    bands: list[tuple[float, float]] = []
    for i, entry in enumerate(raw):
        pair = list(entry)
        if len(pair) != 2:
            raise ValueError(
                f"cfg.inference.evaluation.boundary_bands[{i}] must be a [lo, hi] pair, got "
                f"{pair!r}. Use `.inf` for an unbounded final band."
            )
        bands.append((float(pair[0]), float(pair[1])))
    return tuple(bands)


def resolve_lesionwise(eval_cfg: DictConfig) -> dict[str, Any] | None:
    """Reads and validates `cfg.inference.evaluation.lesionwise`.

    Args:
        eval_cfg: `cfg.inference.evaluation`. The key is read with a default
            of `None` rather than attribute access -- same reasoning as
            `resolve_boundary_bands`: a config composed before this key
            existed (an older saved `eval_config.yaml`, a minimal test
            config) must still run, with the analysis simply off.

    Returns:
        A plain dict of the four settings `lesionwise_case_metrics` accepts
        as keyword arguments (`min_lesion_voxels`, `matching_threshold`,
        `nsd_tolerance_mm`, `connectivity`), ready to `**`-splat into that
        call -- or `None` when the block is absent, is `None`, or has
        `enabled: false`. Unset sub-keys fall back to
        `lesionwise_case_metrics`'s own defaults, so a partially-specified
        block still works.

    Raises:
        ValueError: If `min_lesion_voxels` is negative, `matching_threshold`
            is not in `(0, 1]`, `nsd_tolerance_mm` is not positive, or
            `connectivity` is not one of `{6, 18, 26}` (the only
            neighbourhoods `cc3d.connected_components` accepts in 3-D).
    """
    raw = eval_cfg.get("lesionwise", None)
    if raw is None or not raw.get("enabled", False):
        return None

    min_lesion_voxels = int(raw.get("min_lesion_voxels", 50))
    matching_threshold = float(raw.get("matching_threshold", 0.5))
    nsd_tolerance_mm = float(raw.get("nsd_tolerance_mm", 1.0))
    connectivity = int(raw.get("connectivity", 26))

    if min_lesion_voxels < 0:
        raise ValueError(
            "cfg.inference.evaluation.lesionwise.min_lesion_voxels must be >= 0, got "
            f"{min_lesion_voxels}."
        )
    if not (0 < matching_threshold <= 1):
        raise ValueError(
            "cfg.inference.evaluation.lesionwise.matching_threshold must be in (0, 1], "
            f"got {matching_threshold}."
        )
    if nsd_tolerance_mm <= 0:
        raise ValueError(
            "cfg.inference.evaluation.lesionwise.nsd_tolerance_mm must be > 0, got "
            f"{nsd_tolerance_mm}."
        )
    if connectivity not in (6, 18, 26):
        raise ValueError(
            "cfg.inference.evaluation.lesionwise.connectivity must be one of {6, 18, 26}, "
            f"got {connectivity}."
        )

    return {
        "min_lesion_voxels": min_lesion_voxels,
        "matching_threshold": matching_threshold,
        "nsd_tolerance_mm": nsd_tolerance_mm,
        "connectivity": connectivity,
    }


def evaluate_case(
    model: nn.Module, batch: dict[str, Any], cfg: DictConfig, device: torch.device
) -> CaseOutput:
    """Runs sliding-window inference (and, optionally, MC-dropout) for one case.

    Always runs the deterministic `sliding_window_predict` pass first, and
    `CaseOutput.regions` comes from THAT pass by default: turning on
    MC-dropout uncertainty must never silently move a Dice number that is
    already reported elsewhere (e.g. `docs/experiments.md`). The only
    exception is `cfg.inference.mc_dropout.predictions_from == "mc_mean"`, an
    explicit opt-in that re-derives `regions` from the MC-dropout mean
    prediction instead -- `run_evaluation` logs a one-time warning when this
    is active, since it does make segmentation metrics incomparable to a
    deterministic-pass run.

    Args:
        model: The segmentation model, already on `device`.
        batch: One collated batch from `build_eval_dataloader`'s loader,
            containing `"image"` of shape `(1, 4, D, H, W)`.
        cfg: The full composed Hydra config.
        device: The resolved torch device.

    Returns:
        A `CaseOutput`. See its docstring for what each field holds.
    """
    image = batch["image"]
    logits = sliding_window_predict(model, image, cfg, device)
    regions = postprocess_logits(logits, cfg)

    eval_cfg = cfg.inference.evaluation
    probabilities = torch.sigmoid(logits) if eval_cfg.save_probabilities else None
    # The deterministic logits, kept before any `mc_mean` branch below can
    # rebind `regions`: `logits/` always means the deterministic pass.
    saved_logits = logits if eval_cfg.save_logits else None

    mc_cfg = cfg.inference.mc_dropout
    mc_output: MCDropoutOutput | None = None
    if mc_cfg.enabled:
        # num_samples/seed/require_dropout are read by mc_dropout_predict
        # itself from cfg.inference.mc_dropout, so they are not repeated here.
        mc_output = mc_dropout_predict(model, image, cfg, device)
        if mc_cfg.predictions_from == "mc_mean":
            mc_logits = logits_from_mean_prob(mc_output.mean_prob)
            regions = postprocess_logits(mc_logits, cfg)
            # NOTE: `probabilities` above is deliberately NOT replaced by
            # mc_output.mean_prob. `probabilities/` always means "the
            # deterministic single pass", in every run, so one directory name
            # never denotes two different quantities depending on a config
            # flag. The consequence to be aware of: under this branch,
            # `predictions/` and `probabilities/` come from DIFFERENT passes,
            # so a calibration analysis must not pair them. Add "mean_prob"
            # to `mc_dropout.save_fields` to get the matching probabilities
            # in `mc_mean_prob/` instead.

    return CaseOutput(
        regions=regions, probabilities=probabilities, logits=saved_logits, mc=mc_output
    )


def _log_and_print_summary(
    summary_df: pd.DataFrame, split: str, n_cases: int, n_skipped_unlabeled: int
) -> None:
    """Logs and prints a compact end-of-run summary table.

    Args:
        summary_df: The `MetricAggregator.summary()` DataFrame.
        split: The split name that was evaluated.
        n_cases: Total number of cases in the split.
        n_skipped_unlabeled: Number of cases skipped for metrics because
            `meta["has_label"]` was False.
    """
    lines = [
        "=" * 70,
        f"Evaluation summary -- split={split!r}, {n_cases} case(s), "
        f"{n_skipped_unlabeled} skipped (unlabeled)",
        "=" * 70,
    ]

    if summary_df.empty:
        lines.append("No cases were scored (every case was unlabeled or the split was empty).")
    else:
        for region in ("ET", "TC", "WT"):
            dice_key, iou_key, hd95_key = f"dice_{region}", f"iou_{region}", f"hd95_{region}"
            if dice_key in summary_df.index:
                lines.append(
                    f"  {region}: dice={summary_df.loc[dice_key, 'mean']:.4f}  "
                    f"iou={summary_df.loc[iou_key, 'mean']:.4f}  "
                    f"hd95={summary_df.loc[hd95_key, 'mean']:.4f} mm"
                )
        for mean_key in ("dice_mean", "iou_mean", "hd95_mean"):
            if mean_key in summary_df.index:
                lines.append(f"  {mean_key} = {summary_df.loc[mean_key, 'mean']:.4f}")

        # gt_empty_ET moves headline ET Dice by several points and is rarely
        # reported -- printed explicitly so it is never invisible.
        if "gt_empty_ET" in summary_df.index:
            lines.append(
                "  fraction of cases with empty ground-truth ET (gt_empty_ET mean) = "
                f"{summary_df.loc['gt_empty_ET', 'mean']:.4f}"
            )

    # print only, not logger.info as well: setup_logging's StreamHandler
    # already targets stdout, so doing both emits this block twice and reads
    # like the evaluation ran twice. Matches scripts/preprocess.py's summary.
    print("\n".join(lines))


def run_evaluation(cfg: DictConfig) -> pd.DataFrame:
    """Evaluates a checkpoint over one split and writes every result to disk.

    Per case: read `meta.json` (for `bbox`/`original_shape`/`spacing`/
    `has_label`), run `evaluate_case`, optionally save an uncropped
    prediction and/or a cropped probability map, and -- for labeled cases
    only -- score it against the ground truth and record the metrics.
    Metrics are computed in CROPPED space: the ground-truth `label.npy` on
    disk is already cropped, and uncropping both prediction and target
    would add identical background to both, changing nothing about Dice or
    HD95 except making the computation slower. This is a deliberate choice,
    not an oversight.

    The per-case CSV is rewritten after EVERY case (not once at the end):
    a full-split sliding-window evaluation can take minutes, and a Kaggle
    session dying partway through should not lose everything that had
    already been scored.

    When `cfg.inference.mc_dropout.enabled` is True, also writes one `.npy`
    array per case per field in `cfg.inference.mc_dropout.save_fields`
    (directory names via `_MC_FIELD_TO_DIR`) plus a per-case
    `uncertainty_summary.csv`. Both are entirely absent when MC-dropout is
    off -- a plain evaluation run's output is unchanged from before this was
    wired in.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The per-case metrics DataFrame (also written to
        `<out_dir>/per_case_metrics.csv`), indexed by `case_id`. Cases
        skipped because `meta["has_label"]` is False do not appear in it.
    """
    device = get_device(cfg)

    # Validated before anything else runs: a config typo here should fail
    # immediately, not after a checkpoint load and minutes of sliding-window
    # inference on the first case.
    mc_cfg = cfg.inference.mc_dropout
    _validate_mc_dropout_config(mc_cfg)

    eval_cfg = cfg.inference.evaluation
    lesionwise_cfg = resolve_lesionwise(eval_cfg)
    if lesionwise_cfg is not None:
        # Same reasoning as _validate_mc_dropout_config above, checked right
        # next to it: fail before the checkpoint even loads, not after a
        # ~25-minute sliding-window pass over 189 cases discovers on the
        # last one that panoptica is missing. _load_panoptica() is cheap (an
        # import plus one-time log setup) and is exactly the call that
        # raises the ImportError this analysis needs -- naming
        # requirements-analysis.txt and .venv-analysis -- when panoptica is
        # not installed in the current interpreter. Its return value is
        # unused here; only the import-succeeded check matters.
        _load_panoptica()

    checkpoint_path = resolve_checkpoint(cfg)
    model, _resume_state = load_eval_model(cfg, checkpoint_path, device)

    split = eval_cfg.split
    boundary_bands = resolve_boundary_bands(eval_cfg)
    loader, case_ids = build_eval_dataloader(cfg, split)

    out_dir = ensure_dir(eval_cfg.out_dir)
    prep_dir = Path(cfg.data.preprocessing.out_dir)

    predictions_dir = out_dir / "predictions"
    probabilities_dir = out_dir / "probabilities"
    logits_dir = out_dir / "logits"
    if eval_cfg.save_predictions:
        ensure_dir(predictions_dir)
    if eval_cfg.save_probabilities:
        ensure_dir(probabilities_dir)
    if eval_cfg.save_logits:
        ensure_dir(logits_dir)

    # One output directory per requested MC-dropout field, built up front so
    # the per-case loop below never has to check "does this dir exist yet".
    # Empty (mc_field_dirs stays {}) when mc_dropout is disabled or
    # save_fields is empty -- no directory is created in either case.
    mc_field_dirs: dict[str, Path] = {}
    if mc_cfg.enabled:
        for field_name in mc_cfg.save_fields:
            mc_field_dirs[field_name] = ensure_dir(out_dir / _MC_FIELD_TO_DIR[field_name])

        if mc_cfg.predictions_from == "mc_mean":
            # Logged once here (before the loop), not per case: this is a
            # fact about the run's CONFIG, not about any individual case.
            logger.warning(
                "cfg.inference.mc_dropout.predictions_from='mc_mean': every case's saved "
                "prediction and reported segmentation metrics (Dice/IoU/HD95) come from the "
                "MC-dropout mean pass, not the deterministic single pass. These numbers are "
                "NOT comparable to a deterministic-pass evaluation run -- e.g. an "
                "already-reported baseline row in docs/experiments.md."
            )

    per_case_csv_path = out_dir / "per_case_metrics.csv"
    summary_csv_path = out_dir / "summary.csv"
    uncertainty_csv_path = out_dir / "uncertainty_summary.csv"

    aggregator = MetricAggregator()
    n_skipped_unlabeled = 0
    uncertainty_rows: dict[str, dict[str, float]] = {}
    mc_storage_logged = False
    lesionwise_cost_logged = False

    model.eval()
    with torch.no_grad():
        progress = tqdm(zip(case_ids, loader), total=len(case_ids), desc=f"Evaluating ({split})")
        for case_id, batch in progress:
            meta = read_json(prep_dir / case_id / "meta.json")

            case_output = evaluate_case(model, batch, cfg, device)
            regions = case_output.regions
            probabilities = case_output.probabilities

            if eval_cfg.save_predictions:
                classes = regions_to_classes(regions)  # (1, D, H, W)
                classes_np = classes[0].cpu().numpy().astype(np.uint8)
                uncropped = uncrop_to_original(classes_np, meta["bbox"], meta["original_shape"])
                np.save(predictions_dir / f"{case_id}.npy", uncropped)

            if eval_cfg.save_probabilities:
                # Saved in CROPPED geometry, deliberately not uncropped:
                # these exist for calibration/ECE analysis, which must be
                # done in the frame the model actually predicted in.
                probs_np = probabilities[0].cpu().numpy().astype(np.float16)
                np.save(probabilities_dir / f"{case_id}.npy", probs_np)

            if eval_cfg.save_logits:
                # CROPPED geometry, like probabilities/ above. These are the
                # input to temperature scaling: fp16 probabilities saturate to
                # exactly 1.0 above ~0.99976 (logit +inf), which would silently
                # exclude the most-confident voxels from the fit. Logits are
                # ~+/-20 and round-trip through fp16 intact.
                logits_np = case_output.logits[0].cpu().numpy().astype(np.float16)
                np.save(logits_dir / f"{case_id}.npy", logits_np)

            if case_output.mc is not None:
                mc = case_output.mc

                # Saved in CROPPED geometry, same convention as
                # probabilities/ above -- these back calibration-style
                # analysis, which needs the frame the model predicted in.
                first_field_bytes: int | None = None
                for field_name, field_dir in mc_field_dirs.items():
                    field_np = getattr(mc, field_name)[0].cpu().numpy().astype(np.float16)
                    np.save(field_dir / f"{case_id}.npy", field_np)
                    if first_field_bytes is None:
                        first_field_bytes = field_np.nbytes

                if not mc_storage_logged and first_field_bytes is not None:
                    # Only the first case's array size is measured directly;
                    # the rest is a projection from it, since every case's
                    # array is the same dtype/channel-count (only D, H, W
                    # differ case to case, so this is an estimate, not exact).
                    projected_bytes = first_field_bytes * len(case_ids) * len(mc_field_dirs)
                    message = (
                        f"MC-dropout save_fields={list(mc_field_dirs)}: first case's array is "
                        f"{first_field_bytes / 1e6:.2f} MB, projecting to "
                        f"~{projected_bytes / 1e9:.2f} GB across {len(case_ids)} case(s)."
                    )
                    if projected_bytes > _MC_STORAGE_WARN_BYTES:
                        message += (
                            " This exceeds 15 GB -- Kaggle's /kaggle/working output quota is "
                            "20 GB. Consider a smaller save_fields list or a subset split."
                        )
                    logger.warning(message)
                    mc_storage_logged = True

                # Uncertainty summary row: independent of has_label, since it
                # says nothing about ground truth. mi/entropy moved to CPU
                # for the same device-mismatch reason regions/label are below.
                mi = mc.mutual_information.cpu()
                entropy = mc.predictive_entropy.cpu()
                regions_cpu = regions.cpu()
                row: dict[str, float] = {}
                for i, region in enumerate(REGION_NAMES):
                    mi_channel = mi[:, i]
                    row[f"mi_mean_{region}"] = mi_channel.mean().item()
                    row[f"mi_max_{region}"] = mi_channel.max().item()
                    fg_mask = regions_cpu[:, i] > 0.5
                    if fg_mask.any():
                        row[f"mi_mean_fg_{region}"] = mi_channel[fg_mask].mean().item()
                    else:
                        # NaN, not 0.0: an empty prediction and a confidently
                        # certain prediction are different states and must
                        # not collapse to the same number.
                        row[f"mi_mean_fg_{region}"] = float("nan")
                    row[f"entropy_mean_{region}"] = entropy[:, i].mean().item()
                row["num_samples"] = float(mc.num_samples)
                uncertainty_rows[case_id] = row

                # Rewritten every iteration, same reasoning as the per-case
                # metrics CSV below: a killed run keeps every already-scored
                # case's uncertainty summary instead of losing all of them.
                pd.DataFrame.from_dict(uncertainty_rows, orient="index").rename_axis(
                    "case_id"
                ).to_csv(uncertainty_csv_path)

            if not meta["has_label"]:
                # Real: the BraTS validation set ships without segmentations.
                # Scoring such a case against an all-zero label would drag
                # the reported mean toward whatever the model happens to
                # predict there, so it gets inference and a saved
                # prediction but never reaches the metric aggregator.
                n_skipped_unlabeled += 1
                logger.info(
                    "Case %s has no ground-truth label (has_label=False); skipping metrics.",
                    case_id,
                )
            else:
                # Both moved to CPU explicitly. `regions` is on `device`
                # while `batch["label"]` came straight off the DataLoader and
                # is on CPU, so passing them as-is raises a device mismatch
                # inside MONAI's compute_dice -- but only on CUDA. On a CPU
                # box the two agree by accident and every test passes, which
                # is exactly why this is pinned by a comment rather than left
                # to the suite to catch.
                #
                # CPU rather than `device` because HD95 over a full ~240^3
                # volume allocates several intermediate buffers, and the
                # 16 GB VRAM budget is already committed to the model and
                # the sliding-window output. The per-case transfer is
                # negligible against the sliding-window pass itself.
                pred_cpu = regions.cpu()
                label_cpu = batch["label"].cpu()
                case_metrics = compute_case_metrics(pred_cpu, label_cpu, spacing=meta["spacing"])
                if boundary_bands is not None:
                    # Merged into the SAME per-case record rather than added
                    # through a second aggregator: these are columns of the
                    # same table, and MetricAggregator.update raises on a
                    # duplicate case_id, so add_case + a second write is not
                    # available. Additive only -- no existing Dice/HD95 column
                    # moves when this is on, which is what keeps an already
                    # published results row valid.
                    case_metrics.update(
                        boundary_stratified_errors(
                            pred_cpu,
                            label_cpu,
                            spacing=meta["spacing"],
                            bands=boundary_bands,
                        )
                    )
                if lesionwise_cfg is not None:
                    # Same additivity reasoning as boundary_bands right above
                    # -- columns merged into the same per-case record, not a
                    # second aggregator. Timed because panoptica's connected-
                    # component labelling plus per-region surface-distance
                    # computation is noticeably more expensive than the
                    # metrics above it (see the one-time log line below).
                    start_time = time.perf_counter()
                    lw_metrics = lesionwise_case_metrics(
                        pred_cpu, label_cpu, spacing=meta["spacing"], **lesionwise_cfg
                    )
                    elapsed = time.perf_counter() - start_time
                    case_metrics.update(lw_metrics)

                    if not lesionwise_cost_logged:
                        # Only the first scored case's wall time is measured
                        # and logged, same "one-time, not per-case" pattern
                        # as the MC-dropout storage-guard warning above --
                        # every case pays roughly this cost, so one number
                        # tells a user watching a 189-case run what the
                        # lesion-wise pass adds on top of sliding-window
                        # inference (measured ~2.4s/case on a median-sized,
                        # multi-lesion 137x171x140 synthetic volume).
                        projected_minutes = elapsed * len(case_ids) / 60
                        logger.info(
                            "Lesion-wise scoring (panoptica: connected-component "
                            "labelling + per-region surface-distance) took %.2fs for "
                            "case %s. Projecting to ~%.1f min across %d case(s), on top "
                            "of the sliding-window pass, which takes far longer per case.",
                            elapsed,
                            case_id,
                            projected_minutes,
                            len(case_ids),
                        )
                        lesionwise_cost_logged = True
                aggregator.update(case_id, case_metrics)

            # Rewritten every iteration, not just at the end: cheap for a
            # per-case table this small, and it means a killed run still has
            # every already-scored case on disk instead of losing all of them.
            aggregator.per_case().to_csv(per_case_csv_path)

    per_case_df = aggregator.per_case()
    summary_df = aggregator.summary()
    summary_df.to_csv(summary_csv_path)

    eval_config_path = out_dir / "eval_config.yaml"
    eval_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    _log_and_print_summary(summary_df, split, len(case_ids), n_skipped_unlabeled)

    return per_case_df


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate a checkpoint against a frozen split, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
