"""Hydra entry point that scores the confidence head against free predictive entropy.

Master plan item A5. `neurovision`'s confidence head is a small auxiliary head, trained
alongside segmentation at loss weight 0.05 (see `neurovision.losses.multitask.MultiTaskLoss`),
to predict PER VOXEL AND PER REGION whether the segmentation head's own prediction is right.
It has never once been scored. This script does that, for the first time, by asking the one
question that matters for a failure-detector: does it beat single-pass predictive entropy --
which is FREE from any model, including a plain U-Net -- at localising the segmentation head's
own errors? "It beats entropy", "it matches entropy", and "it learned nothing" are all
publishable outcomes here; the third is the most likely one (see this project's other
pre-registered ablations, all but one of which came back null).

## The polarity trap this script exists to avoid

`neurovision.losses.multitask.MultiTaskLoss.forward`'s confidence-term comment is the
AUTHORITY on this: the training target is `correct = (predicted_positive == target)`, so
`sigmoid(confidence_logits)` is P(the segmentation head is CORRECT), not P(it is wrong).
Every AUROC computed below is against an "error" event, so this script uses
`1 - sigmoid(confidence_logits)` throughout -- getting this backwards inverts every result
while producing entirely plausible-looking numbers, with nothing that raises. See
`docs/lessons.md` and CLAUDE.md's trap list; this is exactly that shape of bug.

## Why a wrapper module, like scripts/extract_ambiguity.py

`NeuroVisionX.forward_with_auxiliary(x) -> (logits, confidence_logits, boundary_logits)`
already returns everything this script needs from a single forward call, at full
resolution, in eval mode -- but MONAI's `SlidingWindowInferer` (which this script needs,
because a per-case scalar must be computed over the WHOLE volume, not one patch) only
stitches ONE output tensor per call. `_ConfidenceWrapper` below solves this exactly the way
`_AmbiguityAtLevel` in `scripts/extract_ambiguity.py` solves the analogous problem for the
fusion-ambiguity read-out: pack everything into one channel-concatenated tensor, split it
back apart after the sliding-window pass.

## The measurement, per case per region

1. One sliding-window pass gives full-volume segmentation logits and confidence logits.
2. `error` = `(sigmoid(seg) > threshold) != ground_truth` -- the event being detected. STRICT
   `>`, deliberately, not this project's usual `>=` (`binarize`, `postprocess_logits`):
   `neurovision.losses.multitask.MultiTaskLoss.forward` builds the confidence head's TRAINING
   target with a strict `>`, so scoring must ask the same question the head was trained to
   answer. See `score_case`'s comment at the point of use.
3. `confidence_error_score` = `1 - sigmoid(confidence_logits)` -- see the polarity note above.
4. `entropy` = free single-pass Bernoulli predictive entropy, in nats, computed from LOGITS via
   softplus (never from clamped probabilities -- an `eps` clamp sized for fp32 is a no-op in
   fp16 and cost this project 10.5 GPU-hours of training on silent NaN; see
   `bernoulli_entropy_nats`).
5. A seeded, LABEL-FREE voxel sample is drawn per region (`label_free_sample_mask` /
   `draw_voxel_indices`) -- the predicted foreground for that region (`binarize`, `>=` -- this
   ONE use is not the training target and does not need to match it, see `score_case`),
   dilated by a margin in mm, so the sample straddles the decision boundary where errors
   actually live. The mask is never built from the ground-truth label: this project has
   already shipped a reporting mask that WAS built from the label, and it manufactured 41-57%
   of a reported ECE behind 984 green tests (CLAUDE.md trap #1).
6. Three AUROCs on that sample: the confidence head's own detection power
   (`auroc_confidence`), free entropy's detection power (`auroc_entropy`), and the confidence
   head's power over and above entropy (`auroc_confidence_residual`, via
   `neurovision.analysis.detection.residualised_auroc`). A region/case with only one class in
   the sample (all-error or all-correct) is SKIPPED, never silently scored as 0.5.

Example usage:

    python scripts/score_confidence.py analysis.confidence.split=test
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import list_data_collate
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import distance_transform_edt
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurovision.analysis.detection import residualised_auroc
from neurovision.analysis.statistics import compare_models
from neurovision.data.dataset import build_data_dicts, build_dataset, load_splits
from neurovision.data.transforms import REGION_NAMES, build_val_transforms
from neurovision.inference.sliding_window import sliding_window_predict
from neurovision.metrics.segmentation import binarize

# Importing this registers "unet3d"/"swinunetr" (from baseline.py) AND "neurovision"
# (models/__init__.py imports neurovision.py too) before build_model is ever called below.
# Copied from scripts/evaluate.py / scripts/extract_ambiguity.py.
from neurovision.models import baseline  # noqa: F401
from neurovision.models.registry import build_model
from neurovision.training.checkpoint import load_checkpoint
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, read_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on any machine --
# no absolute paths. Copied from scripts/evaluate.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


class _ConfidenceWrapper(nn.Module):
    """Adapts a model so ONE sliding-window pass yields segmentation AND confidence logits.

    `NeuroVisionX.forward_with_auxiliary` already returns both full-resolution tensors from a
    single forward call -- MONAI's `SlidingWindowInferer` still only stitches ONE output
    tensor per call, though, so this wrapper channel-concatenates the two. Solves exactly the
    same problem `_AmbiguityAtLevel` in `scripts/extract_ambiguity.py` solves for the
    fusion-ambiguity read-out; this is the confidence-head analogue (read that class's
    docstring for the shared reasoning, not repeated here).

    Raises at CONSTRUCTION time -- before any inference, before `out_dir` is created -- if the
    wrapped model has no confidence head. This is this script's fail-fast contract: someone
    WILL eventually point it at `baseline_unet3d`, which has no such head at all.
    """

    def __init__(self, model: nn.Module, num_regions: int = len(REGION_NAMES)) -> None:
        """Wraps `model`, checking up front that it has a confidence head to score.

        Args:
            model: The loaded segmentation model. Must expose
                `forward_with_auxiliary(x) -> (seg_logits, confidence_logits, boundary_logits)`
                and `model.heads.confidence` -- only `NeuroVisionX` does.
            num_regions: Channel count of one of the two stacked outputs. Documents `forward`'s
                output shape only; never used to slice anything (the model's own output decides
                that).

        Raises:
            ValueError: `model` has no `heads` attribute at all, or `model.heads.confidence` is
                `None` -- e.g. a `unet3d`/`swinunetr` checkpoint, or a `neurovision` checkpoint
                trained with `model.head.confidence.enabled=false`. Names
                `model.heads.confidence` explicitly.
        """
        super().__init__()
        confidence_head = getattr(getattr(model, "heads", None), "confidence", None)
        if confidence_head is None:
            raise ValueError(
                "scripts/score_confidence.py requires model.heads.confidence to be a real "
                "head, but it is None (or the loaded model has no 'heads' attribute at all -- "
                "e.g. a 'unet3d'/'swinunetr' checkpoint). The confidence head is scored only "
                "for a 'neurovision' checkpoint trained with "
                "model.head.confidence.enabled=true. Point analysis.confidence.checkpoint at "
                "such a checkpoint."
            )
        self.model = model
        self.num_regions = num_regions

    def forward(self, x: Tensor) -> Tensor:
        """Returns segmentation logits concatenated with confidence logits.

        Args:
            x: Input MRI volume/patch, shape `(B, in_channels, D, H, W)`.

        Returns:
            `(B, 2 * num_regions, D, H, W)`: channels `0 : num_regions` are the segmentation
            logits, channels `num_regions : 2 * num_regions` are the confidence logits.
        """
        logits, confidence_logits, _boundary_logits = self.model.forward_with_auxiliary(x)
        return torch.cat([logits, confidence_logits], dim=1)


def resolve_confidence_checkpoint(cfg: DictConfig) -> Path:
    """Decides which checkpoint file to score.

    Same convention as `scripts/evaluate.py`'s `resolve_checkpoint` /
    `scripts/extract_ambiguity.py`'s `resolve_ambiguity_checkpoint`, reading from
    `cfg.analysis.confidence.checkpoint` instead.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `Path(cfg.analysis.confidence.checkpoint)` if set, else
        `Path(cfg.training.checkpoint.dir) / "best.pt"`.

    Raises:
        FileNotFoundError: If the resolved path does not exist. The message lists whatever
            `.pt` files ARE present in that directory.
    """
    explicit = cfg.analysis.confidence.checkpoint
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


def load_confidence_model(
    cfg: DictConfig, checkpoint_path: Path, device: torch.device
) -> nn.Module:
    """Builds the model from config and loads a checkpoint's weights into it.

    Args:
        cfg: The full composed Hydra config.
        checkpoint_path: Path to the checkpoint, as returned by `resolve_confidence_checkpoint`.
        device: The resolved torch device to move the model to.

    Returns:
        The loaded model, in eval mode.
    """
    model = build_model(cfg)
    model = model.to(device)
    # restore_rng=False: scoring is deterministic inference, with no reason to perturb the
    # process's RNG state -- same reasoning as scripts/evaluate.py's load_eval_model.
    load_checkpoint(checkpoint_path, model, map_location=str(device), restore_rng=False)
    model.eval()
    return model


def build_confidence_dataloader(cfg: DictConfig, split: str) -> tuple[DataLoader, list[str]]:
    """Builds the whole-volume `DataLoader` for one frozen split.

    `batch_size=1` is mandatory: whole volumes have per-case shapes and do not collate at any
    larger batch size. Same convention as `scripts/evaluate.py`'s `build_eval_dataloader`.

    Args:
        cfg: The full composed Hydra config.
        split: Which frozen split to load -- `"train"`, `"val"`, or `"test"`.

    Returns:
        `(loader, case_ids)`. `case_ids` is in the SAME order the loader yields batches in.

    Raises:
        ValueError: If `split` is not one of the split file's keys, or the requested split has
            zero cases.
    """
    splits = load_splits(cfg.data.splits.path)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}. Available splits: {sorted(splits.keys())}.")

    case_ids = list(splits[split])
    if not case_ids:
        raise ValueError(
            f"Split {split!r} has 0 cases (splits file: {cfg.data.splits.path}). Scoring an "
            "empty split would silently produce an empty per_case_confidence.csv."
        )

    prep_dir = cfg.data.preprocessing.out_dir
    data_dicts = build_data_dicts(case_ids, prep_dir)
    transform = build_val_transforms(cfg)
    dataset = build_dataset(data_dicts, transform, dataset_type="dataset")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=list_data_collate,
    )
    return loader, case_ids


def label_free_sample_mask(
    pred_region: np.ndarray, spacing: Sequence[float], dilation_mm: float
) -> np.ndarray:
    """Label-free voxel-sampling mask: the predicted foreground dilated by `dilation_mm`.

    Built from the PREDICTED region mask alone -- this function has NO `label` parameter,
    structurally, not by convention. See CLAUDE.md's "a calibration reporting mask must never
    be defined using the ground-truth label" trap: a reporting mask built from the label
    elsewhere in this project manufactured 41-57% of a reported ECE behind 984 green tests. A
    mask that decides WHICH voxels get scored must be computable at deployment time, when no
    ground truth exists at all.

    Args:
        pred_region: Boolean array, shape `(D, H, W)` -- one region's predicted foreground.
            `score_case` passes `binarize(seg_logits, threshold)` here (this project's usual
            `>=` discretization convention), NOT the strict-`>` comparison it uses for `error`
            -- see `score_case`'s comment on why those two deliberately differ.
        spacing: `(D, H, W)` physical voxel size in mm, from `meta.json`. Passed straight to
            `distance_transform_edt`'s `sampling`, so dilation is a real Euclidean distance in
            mm, not an isotropic voxel-count approximation.
        dilation_mm: Dilation radius in millimetres. The sample straddles the predicted
            decision boundary -- where errors actually live -- rather than sitting only deep
            inside a large confident blob or only far outside it.

    Returns:
        Boolean array, same shape as `pred_region`. All-True if `pred_region` is entirely
        foreground (nothing left to dilate into). All-False if `pred_region` is entirely
        empty (there is no predicted foreground to dilate from at all).
    """
    if pred_region.all():
        return np.ones_like(pred_region, dtype=bool)
    if not pred_region.any():
        return np.zeros_like(pred_region, dtype=bool)
    distance_mm = distance_transform_edt(~pred_region, sampling=spacing)
    return distance_mm <= dilation_mm


def draw_voxel_indices(
    mask: np.ndarray, max_voxels: int, generator: np.random.Generator
) -> np.ndarray:
    """Draws up to `max_voxels` FLAT indices, without replacement, from where `mask` is True.

    No `label` parameter -- see `label_free_sample_mask`'s docstring for why that is
    load-bearing here, not a style choice.

    Args:
        mask: Boolean array, any shape (flattened internally, row-major).
        max_voxels: Upper bound on how many indices are drawn. Fewer are drawn if the mask has
            fewer than this many `True` voxels.
        generator: A seeded `np.random.Generator` (see `neurovision.utils.seed.set_seed`),
            consumed once per call -- a caller drawing per case per region must reuse the SAME
            generator across every call, in a fixed order, for a whole run to be reproducible
            from one `cfg.seed`.

    Returns:
        1-D int array of flat indices into `mask.reshape(-1)`, length
        `min(max_voxels, mask.sum())`. Empty when `mask` has no `True` voxel at all.
    """
    flat_idx = np.flatnonzero(mask.reshape(-1))
    n_draw = min(int(max_voxels), flat_idx.size)
    if n_draw == 0:
        return flat_idx
    return generator.choice(flat_idx, size=n_draw, replace=False)


def bernoulli_entropy_nats(logits: Tensor) -> Tensor:
    """Per-voxel Bernoulli predictive entropy, in NATS, computed directly from LOGITS.

    `H = p * softplus(-z) + (1 - p) * softplus(z)` -- algebraically equal to the textbook
    `-(p*log(p) + (1-p)*log(1-p))`, but numerically safe: `softplus` is finite for any finite
    input, unlike `log(sigmoid(z).clamp(eps, 1-eps))`. That clamp is sized for fp32 and is a
    NO-OP in fp16 (`1.0 - 1e-6` rounds to exactly `1.0` in fp16 arithmetic), which silently gave
    `log(0) = NaN` and cost this project 10.5 GPU-hours of training on NaN with nothing raising
    (see `docs/lessons.md`). Mirrors
    `neurovision.models.fusion.adaptive_fusion.BranchAmbiguity._entropy_from_logits`'s formula
    exactly, except UNNORMALISED (in nats, not divided by `ln(2)` into `[0, 1]`) -- AUROC only
    depends on rank order, so the normalisation would not change any result here, but this
    function follows this script's specification literally.

    Args:
        logits: Raw (pre-sigmoid) values, any shape.

    Returns:
        Same shape as `logits`, values in `[0, ln(2)] ~= [0, 0.693]` nats.
    """
    p = torch.sigmoid(logits)
    return p * F.softplus(-logits) + (1.0 - p) * F.softplus(logits)


def score_case(
    seg_logits: Tensor,
    confidence_logits: Tensor,
    target: Tensor,
    spacing: Sequence[float],
    threshold: float,
    dilation_mm: float,
    max_voxels: int,
    generator: np.random.Generator,
    region_names: Sequence[str] = REGION_NAMES,
) -> dict[str, float | str]:
    """Scores one case's confidence head against free predictive entropy, per region.

    Draws a seeded, label-free voxel sample per region (`label_free_sample_mask` /
    `draw_voxel_indices`) and computes three AUROCs on it: the confidence head's own
    detection power, free single-pass entropy's detection power, and the confidence head's
    RESIDUAL power once entropy's own contribution is removed
    (`neurovision.analysis.detection.residualised_auroc`).

    Args:
        seg_logits: Raw segmentation logits, shape `(num_regions, D, H, W)`.
        confidence_logits: Raw confidence-head logits, same shape.
            `sigmoid(confidence_logits)` is P(the segmentation head is CORRECT) -- see
            `neurovision.losses.multitask.MultiTaskLoss.forward`'s confidence-term comment,
            the authority for this polarity. This function scores `1 - sigmoid(...)` (computed
            below), since AUROC needs a "higher = more likely POSITIVE (error)" score.
        target: Binary ground-truth region indicator, same shape as `seg_logits`.
        spacing: `(D, H, W)` physical voxel size in mm, from `meta.json`.
        threshold: Sigmoid threshold defining "predicted positive"
            (`cfg.inference.postprocess.threshold`). Used TWICE below, with two DELIBERATELY
            DIFFERENT comparisons -- see the comment at the point of use.
        dilation_mm: Forwarded to `label_free_sample_mask`.
        max_voxels: Forwarded to `draw_voxel_indices`, per region.
        generator: A seeded `np.random.Generator`, consumed once per region IN `region_names`
            order -- a caller processing several cases must reuse the SAME generator across
            cases, in a fixed case order, for a whole run to be reproducible from one
            `cfg.seed`.
        region_names: Region channel names, in channel order.

    Returns:
        Flat dict with, per region `R`: `auroc_confidence_R`, `auroc_entropy_R`,
        `auroc_confidence_residual_R` (NaN when skipped), `n_voxels_R` (voxels actually drawn,
        possibly 0), `skip_reason_R` (`""` when scored, `"empty_mask"` when the dilated
        predicted-foreground sample was empty, `"single_class"` when the drawn sample was
        all-error or all-correct -- AUROC is undefined either way and this function never
        substitutes 0.5 for it).
    """
    # "error" uses a STRICT `>`, deliberately NOT neurovision.metrics.segmentation.binarize
    # (`>=`), even though `>=` is this project's usual discretization convention everywhere
    # else (binarize, postprocess_logits). This is the one place that convention is
    # overridden, and on purpose: neurovision.losses.multitask.MultiTaskLoss.forward builds
    # the confidence head's TRAINING target with a strict `>`
    # (`correct = ((prob > self.confidence_threshold).float() == target.float())`), so the
    # head was trained against a strict-`>` notion of "correct". Scoring it must ask the exact
    # question it was trained to answer -- an `error` definition that silently switched to
    # `>=` would score the head against a target it never saw, off by whatever mass sits at
    # exactly p == threshold (measure zero for continuous logits, but real for the exact ties
    # a saturated/near-binary head produces).
    error = (torch.sigmoid(seg_logits) > threshold) != (target > 0.5)
    # 1 - sigmoid(...): predicted ERROR probability, not predicted correctness -- see this
    # function's confidence_logits arg docstring. neurovision.losses.multitask.MultiTaskLoss is
    # the authority for the opposite (P(correct)) polarity of the raw logits.
    confidence_error_score = 1.0 - torch.sigmoid(confidence_logits)
    entropy = bernoulli_entropy_nats(seg_logits)

    # The SAMPLING MASK, unlike `error` above, uses `binarize` (`>=`) -- this project's
    # standard discretization convention (also used by postprocess_logits). It is not the
    # training target and does not need to match it: the mask only decides WHICH voxels get
    # drawn into the sample, never what counts as an error, so there is no reason to inherit
    # the loss module's strict `>` here. Reusing `binarize` keeps this one definition rather
    # than a second hand-written comparison that could silently drift from it.
    pred_for_mask = binarize(seg_logits, threshold=threshold).bool()

    row: dict[str, float | str] = {}
    for i, region in enumerate(region_names):
        pred_region_np = pred_for_mask[i].cpu().numpy()
        mask = label_free_sample_mask(pred_region_np, spacing, dilation_mm)
        drawn = draw_voxel_indices(mask, max_voxels, generator)
        row[f"n_voxels_{region}"] = int(drawn.size)

        if drawn.size == 0:
            row[f"auroc_confidence_{region}"] = float("nan")
            row[f"auroc_entropy_{region}"] = float("nan")
            row[f"auroc_confidence_residual_{region}"] = float("nan")
            row[f"skip_reason_{region}"] = "empty_mask"
            continue

        err_i = error[i].reshape(-1).cpu().numpy()[drawn]
        if bool(err_i.all()) or not bool(err_i.any()):
            # AUROC is undefined with only one class present -- counted and reported as a
            # skip, never silently emitted as 0.5 (see CLAUDE.md's substring/silent-default
            # family of traps: a 0.5 here would look exactly like a real, uninformative score).
            row[f"auroc_confidence_{region}"] = float("nan")
            row[f"auroc_entropy_{region}"] = float("nan")
            row[f"auroc_confidence_residual_{region}"] = float("nan")
            row[f"skip_reason_{region}"] = "single_class"
            continue

        conf_i = confidence_error_score[i].reshape(-1).cpu().numpy()[drawn]
        ent_i = entropy[i].reshape(-1).cpu().numpy()[drawn]

        residual = residualised_auroc(conf_i, ent_i, err_i)
        row[f"auroc_confidence_{region}"] = residual["auroc_score"]
        row[f"auroc_entropy_{region}"] = residual["auroc_control"]
        row[f"auroc_confidence_residual_{region}"] = residual["auroc_residual"]
        row[f"skip_reason_{region}"] = ""

    return row


def _summarize(per_case_df: pd.DataFrame) -> pd.DataFrame:
    """Reduces the per-case table to mean/median/std per column, with skip counts.

    Mirrors `neurovision.metrics.segmentation.MetricAggregator.summary()`'s exact convention
    (mean/std/median/count/n_missing, all NaN-skipping). `n_missing` on an `auroc_*` column IS
    the skip count for that region: every skip (`empty_mask` or `single_class`) writes NaN into
    the three AUROC columns, never 0.5, so a skip can never look like a real computed value
    here.

    Args:
        per_case_df: `run_scoring`'s per-case table.

    Returns:
        A `DataFrame` indexed by column name, columns `mean`, `std`, `median`, `count`,
        `n_missing`. Empty if `per_case_df` is empty.
    """
    if per_case_df.empty:
        return pd.DataFrame()
    numeric = per_case_df.select_dtypes(include="number")
    return pd.DataFrame(
        {
            "mean": numeric.mean(skipna=True),
            "std": numeric.std(skipna=True, ddof=1),
            "median": numeric.median(skipna=True),
            "count": numeric.count(),
            "n_missing": numeric.isna().sum(),
        }
    )


def _build_comparison_frames(
    per_case_df: pd.DataFrame, region_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the two per-case tables `compare_models` needs to compare confidence vs entropy.

    `compare_models` compares SAME-NAMED columns across two tables -- so both returned frames
    use identical column names (`auroc_<region>`), one holding the confidence head's AUROC per
    case, the other holding entropy's, which is what makes each region its own compared metric.

    Args:
        per_case_df: `run_scoring`'s per-case table, columns `auroc_confidence_<region>` /
            `auroc_entropy_<region>` among others.
        region_names: Region names to compare, in the same order used throughout this script.

    Returns:
        `(a_df, b_df)`, both indexed the same as `per_case_df`.
    """
    a_df = per_case_df[[f"auroc_confidence_{r}" for r in region_names]].rename(
        columns={f"auroc_confidence_{r}": f"auroc_{r}" for r in region_names}
    )
    b_df = per_case_df[[f"auroc_entropy_{r}" for r in region_names]].rename(
        columns={f"auroc_entropy_{r}": f"auroc_{r}" for r in region_names}
    )
    return a_df, b_df


def _log_and_print_summary(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    split: str,
    n_cases: int,
    n_skipped_unlabeled: int,
) -> None:
    """Logs and prints a compact end-of-run summary, in the style of `scripts/evaluate.py`'s.

    Args:
        summary_df: `_summarize`'s output.
        comparison_df: `compare_models`'s output (confidence vs entropy, per region).
        split: The split name that was scored.
        n_cases: Total number of cases in the split.
        n_skipped_unlabeled: Number of cases skipped because `meta["has_label"]` was False.
    """
    lines = [
        "=" * 70,
        f"Confidence-head scoring summary -- split={split!r}, {n_cases} case(s), "
        f"{n_skipped_unlabeled} skipped (unlabeled)",
        "=" * 70,
    ]
    if summary_df.empty:
        lines.append("No cases were scored (every case was unlabeled or the split was empty).")
    else:
        for col in summary_df.index:
            if col.startswith("auroc_"):
                lines.append(
                    f"  {col}: mean={summary_df.loc[col, 'mean']:.4f}  "
                    f"n_missing(skipped)={int(summary_df.loc[col, 'n_missing'])}"
                )
        if not comparison_df.empty:
            lines.append("")
            lines.append("confidence vs entropy (paired, Holm-corrected across regions):")
            for metric, row in comparison_df.iterrows():
                lines.append(
                    f"  {metric}: improvement={row['improvement']:.4f}  "
                    f"p_holm={row['p_holm']:.4g}  verdict={row['verdict']}"
                )

    # print only, not logger.info as well: setup_logging's StreamHandler already targets
    # stdout, so doing both would print this block twice. Matches scripts/evaluate.py's /
    # scripts/extract_ambiguity.py's summary.
    print("\n".join(lines))


def run_scoring(cfg: DictConfig) -> pd.DataFrame:
    """Scores a checkpoint's confidence head against free entropy over one split.

    Per labeled case: one sliding-window pass through `_ConfidenceWrapper` yields both
    segmentation and confidence logits; `score_case` draws a seeded, label-free voxel sample
    per region and computes the three AUROCs. Writes `per_case_confidence.csv` after EVERY
    case (not just at the end) so a killed run keeps every already-scored case, then
    `summary.csv` and `confidence_vs_entropy.csv` once scoring finishes.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The per-case table (also written to `<out_dir>/per_case_confidence.csv`), indexed by
        `case_id`. Cases skipped because `meta["has_label"]` is False do not appear in it.

    Raises:
        ValueError: See `_ConfidenceWrapper.__init__` (no confidence head) and
            `build_confidence_dataloader` (unknown or empty split).
        FileNotFoundError: See `resolve_confidence_checkpoint`.
    """
    device = get_device(cfg)
    conf_cfg = cfg.analysis.confidence
    prep_dir = Path(cfg.data.preprocessing.out_dir)
    region_names = tuple(conf_cfg.regions)
    threshold = float(cfg.inference.postprocess.threshold)
    dilation_mm = float(conf_cfg.dilation_mm)
    max_voxels = int(conf_cfg.max_voxels_per_case)
    num_regions = len(region_names)

    checkpoint_path = resolve_confidence_checkpoint(cfg)
    model = load_confidence_model(cfg, checkpoint_path, device)
    # Raises immediately here -- before out_dir is created, before any inference -- if the
    # loaded model has no confidence head. See _ConfidenceWrapper.__init__ and this script's
    # fail-fast contract.
    wrapped_model = _ConfidenceWrapper(model, num_regions=num_regions).to(device)
    wrapped_model.eval()

    split = str(conf_cfg.split)
    loader, case_ids = build_confidence_dataloader(cfg, split)

    out_dir = ensure_dir(conf_cfg.out_dir)
    per_case_csv_path = out_dir / "per_case_confidence.csv"
    summary_csv_path = out_dir / "summary.csv"
    comparison_csv_path = out_dir / "confidence_vs_entropy.csv"

    # Shared across every voxel draw AND the confidence-vs-entropy bootstrap below, so the
    # whole run is reproducible from one cfg.seed -- same convention as
    # scripts/detection_stats.py's voxel_level_table.
    generator = np.random.default_rng(int(cfg.seed))

    per_case_rows: dict[str, dict[str, float | str]] = {}
    n_skipped_unlabeled = 0

    model.eval()
    with torch.no_grad():
        progress = tqdm(zip(case_ids, loader), total=len(case_ids), desc=f"Scoring ({split})")
        for case_id, batch in progress:
            meta = read_json(prep_dir / case_id / "meta.json")
            if not meta["has_label"]:
                # Real: the BraTS validation set ships without segmentations, and "error"
                # cannot be defined without a ground truth to compare against.
                n_skipped_unlabeled += 1
                logger.info(
                    "Case %s has no ground-truth label (has_label=False); skipping.", case_id
                )
                continue

            image = batch["image"]  # (1, 4, D, H, W)
            combined = sliding_window_predict(wrapped_model, image, cfg, device)
            seg_logits = combined[0, :num_regions].cpu()
            confidence_logits = combined[0, num_regions : 2 * num_regions].cpu()
            target = batch["label"][0].cpu()  # (num_regions, D, H, W)

            row = score_case(
                seg_logits,
                confidence_logits,
                target,
                meta["spacing"],
                threshold,
                dilation_mm,
                max_voxels,
                generator,
                region_names,
            )
            per_case_rows[case_id] = row

            # Rewritten every iteration, not just at the end -- a full-split sliding-window
            # run can take minutes, and a killed run should keep every already-scored case.
            # Same reasoning as scripts/evaluate.py's / scripts/extract_ambiguity.py's per-case
            # CSVs.
            pd.DataFrame.from_dict(per_case_rows, orient="index").rename_axis("case_id").to_csv(
                per_case_csv_path
            )

    per_case_df = pd.DataFrame.from_dict(per_case_rows, orient="index").rename_axis("case_id")
    summary_df = _summarize(per_case_df)
    summary_df.to_csv(summary_csv_path)

    a_df, b_df = _build_comparison_frames(per_case_df, region_names)
    comparison_df = compare_models(
        a_df, b_df, generator=generator, name_a="confidence", name_b="entropy"
    )
    comparison_df.to_csv(comparison_csv_path)

    config_path = out_dir / "confidence_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    _log_and_print_summary(summary_df, comparison_df, split, len(case_ids), n_skipped_unlabeled)

    return per_case_df


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Scores a checkpoint's confidence head against free entropy, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_scoring(cfg)


if __name__ == "__main__":
    main()
