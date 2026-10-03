"""Hydra entry point for diagnostic R8: why is the confidence head's WT AUROC only 0.477?

Reproducibility script for paper result "R8" (experiments.md note 57, ledger claim C16,
paper Section VII / Table VIII). `scripts/score_confidence.py` found that `neurovision`'s
trained confidence head localises segmentation errors well for ET and TC (mean per-case AUROC
0.855 / 0.871) but scores 0.477 on WT -- slightly WORSE than a coin flip. That could be:

  1. a sign bug (we scored P(correct) where we wanted P(error)),
  2. a channel bug (the WT error was scored against the wrong confidence channel), or
  3. a genuinely uninformative WT confidence channel.

The reported finding is (3): the WT confidence channel is nearly constant, and identical inside
and outside the predicted tumour, and NO confidence channel predicts WT errors (so it is not a
channel swap either; ET/TC channels do rank their own errors well, which also rules out a
global sign bug). This script reproduces that evidence in two parts.

Part 1 (no model): summarises the per-case AUROC table `score_confidence.py` already wrote
(`analysis.confidence_diag.per_case_csv`) -- n, mean, median, IQR, min, max, and the fraction of
cases below 0.5 -- per region. Written to `per_case_auroc_distribution.csv`.

Part 2 (needs the model, CPU): for a few named test cases, one sliding-window pass gives
segmentation logits and confidence logits (3 channels each: ET, TC, WT). On the label-free WT
sample (predicted WT dilated by `analysis.confidence.dilation_mm`, the same sampling rule the
scorer uses) it reports
  * each confidence channel's spread (mean/std/min/max of sigmoid) and its mean inside vs
    outside the predicted WT (`channel_stats.csv`);
  * a 3x3 matrix: AUROC of channel c's error score for predicting region r's errors, for every
    (r, c). A channel swap would put the high values OFF the diagonal
    (`cross_channel_auroc.csv`).

Polarity and error definition are copied from `score_confidence.py`: error score =
`1 - sigmoid(confidence_logit)` (the head predicts P(correct)), and an error is
`(sigmoid(seg) > threshold) != target` with a STRICT `>`, matching the head's training target.

Comparability caveat: the 3x3 matrix uses the WT sample mask (predicted WT, dilated) for EVERY
error region, and keeps every voxel of it (no `max_voxels_per_case` cap). `score_confidence.py`
instead samples a per-region mask, capped at 20000 voxels. So only the (WT error, WT channel)
cell is comparable with that scorer's per-case table; the ET/TC cells are not expected to
reproduce it.

On the Mac, run whole-volume jobs at one window per batch (CLAUDE.md trap #6):

    python scripts/confidence_diag.py +experiment=neurovision \
        inference.sliding_window.sw_batch_size=1 data.num_workers=0

Omitting `+experiment=neurovision` selects the default baseline U-Net config, which has no
confidence head and no checkpoint at the resolved path: the run fails (FileNotFoundError, or
ValueError if a baseline checkpoint exists).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from neurovision.analysis.localisation import case_auroc
from neurovision.inference.sliding_window import sliding_window_predict
from neurovision.metrics.segmentation import binarize
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, read_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
_SCORE_PATH = Path(__file__).resolve().parent / "score_confidence.py"

# Load score_confidence.py by path (scripts/ is not a package) -- same pattern as
# scripts/mc_comparison.py's import of detection_stats.py.
_spec = importlib.util.spec_from_file_location("score_confidence_script", _SCORE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - import plumbing
    raise ImportError(f"confidence_diag: cannot import {_SCORE_PATH}.")
_SCORE = importlib.util.module_from_spec(_spec)
sys.modules["score_confidence_script"] = _SCORE
_spec.loader.exec_module(_SCORE)

# Channel order of the segmentation head, confidence head and label.
REGIONS: tuple[str, ...] = ("ET", "TC", "WT")
_WT_INDEX = REGIONS.index("WT")


def distribution_summary(df: pd.DataFrame, regions: Sequence[str] = REGIONS) -> pd.DataFrame:
    """Summarises the per-case confidence AUROC distribution for each region (Part 1).

    Args:
        df: `score_confidence.py`'s per-case table, with columns `auroc_confidence_<region>`.
            NaN entries (skipped cases) are ignored and not counted in `n`.
        regions: Region names to summarise.

    Returns:
        One row per region, columns `region, n, mean, median, q25, q75, min, max,
        frac_below_0_5` (`frac_below_0_5` = fraction of cases with AUROC < 0.5, i.e. worse
        than chance).
    """
    rows = []
    for region in regions:
        values = df[f"auroc_confidence_{region}"].dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            raise ValueError(
                f"distribution_summary: column auroc_confidence_{region} is empty or all NaN."
            )
        rows.append(
            {
                "region": region,
                "n": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "min": float(values.min()),
                "max": float(values.max()),
                "frac_below_0_5": float((values < 0.5).mean()),
            }
        )
    return pd.DataFrame(rows)


def channel_stats(
    conf_logits: Tensor,
    sample_mask: Tensor | np.ndarray,
    pred_wt: Tensor | np.ndarray,
    region_names: Sequence[str] = REGIONS,
) -> list[dict[str, float | str]]:
    """Spread of each confidence channel over the sample, and inside vs outside predicted WT.

    Args:
        conf_logits: Raw confidence logits, shape `(C, D, H, W)`.
        sample_mask: Boolean `(D, H, W)` sampling mask (predicted WT, dilated).
        pred_wt: Boolean `(D, H, W)` predicted WT foreground. "Inside" means
            `sample_mask & pred_wt`, "outside" means `sample_mask & ~pred_wt`.
        region_names: Channel names, in channel order.

    Returns:
        One dict per channel with `channel, mean, std, min, max` of `sigmoid(conf)` over the
        sample, plus `mean_in_pred_wt` and `mean_out_pred_wt` (NaN if that part is empty).
    """
    sample = torch.as_tensor(sample_mask).bool()
    inside = sample & torch.as_tensor(pred_wt).bool()
    outside = sample & ~torch.as_tensor(pred_wt).bool()
    probs = torch.sigmoid(conf_logits.float())  # P(segmentation is correct)
    out: list[dict[str, float | str]] = []
    for c, name in enumerate(region_names):
        p = probs[c]
        in_vals, out_vals, all_vals = p[inside], p[outside], p[sample]
        out.append(
            {
                "channel": name,
                "mean": float(all_vals.mean()) if all_vals.numel() else float("nan"),
                # unbiased=False: describing this sample, not estimating a population
                "std": float(all_vals.std(unbiased=False)) if all_vals.numel() else float("nan"),
                "min": float(all_vals.min()) if all_vals.numel() else float("nan"),
                "max": float(all_vals.max()) if all_vals.numel() else float("nan"),
                "mean_in_pred_wt": float(in_vals.mean()) if in_vals.numel() else float("nan"),
                "mean_out_pred_wt": float(out_vals.mean()) if out_vals.numel() else float("nan"),
            }
        )
    return out


def cross_channel_auroc(
    seg_logits: Tensor,
    conf_logits: Tensor,
    target: Tensor,
    sample_mask: Tensor | np.ndarray,
    threshold: float,
    region_names: Sequence[str] = REGIONS,
) -> list[dict[str, float | str]]:
    """AUROC of every confidence channel against every region's errors, on one sample.

    Args:
        seg_logits: Raw segmentation logits, shape `(C, D, H, W)`.
        conf_logits: Raw confidence logits, shape `(C, D, H, W)`. `sigmoid` of these is
            P(correct), so the error score is `1 - sigmoid`.
        target: Binary ground truth, shape `(C, D, H, W)`.
        sample_mask: Boolean `(D, H, W)` sampling mask.
        threshold: Sigmoid threshold for "predicted positive" (strict `>`, as in the head's
            training target).
        region_names: Channel names, in channel order.

    Returns:
        One dict per error region (rows of the matrix): `error_region`, `auroc_ch_<c>` for
        every channel `c` (NaN if that region's errors are single-class in the sample), and
        `error_rate` (fraction of sample voxels that are errors for this region).
    """
    sample = torch.as_tensor(sample_mask).bool()
    error = (torch.sigmoid(seg_logits.float()) > threshold) != (target > 0.5)
    error_score = 1.0 - torch.sigmoid(conf_logits.float())
    out: list[dict[str, float | str]] = []
    for r, region in enumerate(region_names):
        err = error[r][sample].numpy()
        row: dict[str, float | str] = {
            "error_region": region,
            "error_rate": float(err.mean()) if err.size else float("nan"),
        }
        for c, channel in enumerate(region_names):
            # case_auroc already returns NaN for single-class input (never a fake 0.5).
            row[f"auroc_ch_{channel}"] = case_auroc(error_score[c][sample].numpy(), err)
        out.append(row)
    return out


def select_case_indices(case_ids: Sequence[str], requested: Sequence[str]) -> list[int]:
    """Finds the position of each requested case id in the split's id list.

    Args:
        case_ids: The split's case ids, in loader order.
        requested: Case ids to diagnose.

    Returns:
        Indices into `case_ids`, in `requested` order.

    Raises:
        ValueError: If any requested id is not in `case_ids`.
    """
    missing = [c for c in requested if c not in case_ids]
    if missing:
        raise ValueError(f"Requested case id(s) not in the split: {missing}.")
    return [list(case_ids).index(c) for c in requested]


def run_diagnostic(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs Part 1 and Part 2 and writes every output file.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `(distribution, channel_stats_table, cross_channel_table)`.

    Raises:
        ValueError: If a requested case is not in the split, or the model has no
            confidence head.
        FileNotFoundError: If the per-case CSV or the checkpoint is missing.
    """
    diag_cfg = cfg.analysis.confidence_diag
    conf_cfg = cfg.analysis.confidence
    # Channel indices below ([:3], [3:6], WT = 2) assume this exact order.
    if tuple(conf_cfg.regions) != REGIONS:
        raise ValueError(
            f"analysis.confidence.regions must be {list(REGIONS)} (channel order), "
            f"got {list(conf_cfg.regions)}."
        )
    out_dir = ensure_dir(diag_cfg.out_dir)

    # ---- Part 1: no model ----
    per_case_df = pd.read_csv(diag_cfg.per_case_csv)
    distribution = distribution_summary(per_case_df, REGIONS)
    distribution.to_csv(out_dir / "per_case_auroc_distribution.csv", index=False)

    # ---- Part 2: model ----
    device = get_device(cfg)
    threshold = float(cfg.inference.postprocess.threshold)
    dilation_mm = float(conf_cfg.dilation_mm)
    prep_dir = Path(cfg.data.preprocessing.out_dir)

    model = _SCORE.load_confidence_model(cfg, _SCORE.resolve_confidence_checkpoint(cfg), device)
    wrapped = _SCORE._ConfidenceWrapper(model, num_regions=len(REGIONS)).to(device)
    wrapped.eval()

    loader, case_ids = _SCORE.build_confidence_dataloader(cfg, str(conf_cfg.split))
    requested = [str(c) for c in diag_cfg.cases]
    indices = select_case_indices(case_ids, requested)

    stats_rows: list[dict[str, float | str]] = []
    cross_rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for case_id, idx in zip(requested, indices):
            # Index the dataset directly: iterating the loader would run the transform
            # pipeline on every one of the split's cases just to discard all but a few.
            item = loader.dataset[idx]
            image = item["image"].unsqueeze(0)  # (1, 4, D, H, W)
            target = item["label"].cpu()  # (3, D, H, W)
            meta = read_json(prep_dir / case_id / "meta.json")
            if not meta["has_label"]:
                raise ValueError(f"Case {case_id} has no ground-truth label; cannot score errors.")

            combined = sliding_window_predict(wrapped, image, cfg, device)
            seg_logits = combined[0, :3].cpu()
            conf_logits = combined[0, 3:6].cpu()

            # Label-free sample: predicted WT (>=, the project's binarize), dilated.
            pred_wt = binarize(seg_logits[_WT_INDEX : _WT_INDEX + 1], threshold=threshold)
            pred_wt_np = pred_wt[0].bool().numpy()
            sample = _SCORE.label_free_sample_mask(pred_wt_np, meta["spacing"], dilation_mm)

            for row in channel_stats(conf_logits, sample, pred_wt_np):
                stats_rows.append({"case": case_id, **row})
            for row in cross_channel_auroc(seg_logits, conf_logits, target, sample, threshold):
                cross_rows.append({"case": case_id, **row})
            logger.info("Diagnosed %s (sample voxels: %d).", case_id, int(sample.sum()))

    stats_df = pd.DataFrame(stats_rows)
    cross_df = pd.DataFrame(cross_rows)
    stats_df.to_csv(out_dir / "channel_stats.csv", index=False)
    cross_df.to_csv(out_dir / "cross_channel_auroc.csv", index=False)

    resolved = {
        "confidence_diag": OmegaConf.to_container(diag_cfg, resolve=True),
        "confidence": OmegaConf.to_container(conf_cfg, resolve=True),
    }
    (out_dir / "confidence_diag_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(resolved)), encoding="utf-8"
    )

    logger.info("Per-case AUROC distribution:\n%s", distribution.to_string(index=False))
    logger.info("Channel stats:\n%s", stats_df.to_string(index=False))
    logger.info("Cross-channel AUROC:\n%s", cross_df.to_string(index=False))
    return distribution, stats_df, cross_df


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs diagnostic R8 per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_diagnostic(cfg)


if __name__ == "__main__":
    main()
