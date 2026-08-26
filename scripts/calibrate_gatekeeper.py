"""Hydra entry point that fits the gatekeeper's `Thresholds` from the val split.

Milestone 4, Phase E -- the second half of E5's calibration step. Everything the
gate needs to combine numbers into PROCEED / PROCEED_WITH_CAUTION / REFUSE
already exists: `neurovision.analysis.gatekeeper_calibration.
build_gatekeeper_calibration_table` assembles one calibration-case table, and
`neurovision.inference.gatekeeper.calibrate_thresholds` fits absolute cut
points from that table's own quantiles. Nothing wired those two together and
wrote the result to disk -- this script is exactly that wiring, and nothing
else. It fits no threshold itself, builds no calibration signal itself, and
runs no model itself (that all happens inside
`build_gatekeeper_calibration_table`).

Example usage (the `model=segqc` override is REQUIRED -- see below):

    python scripts/calibrate_gatekeeper.py model=segqc

## Why `model=segqc` is required

`build_gatekeeper_calibration_table` internally calls
`neurovision.analysis.gatekeeper_calibration.qc_predicted_dice_table`, which
builds and loads a trained `SegQC` checkpoint via `neurovision.models.qc.
build_segqc(cfg)`. The root config's default model group is `unet3d`, which
has none of the keys `build_segqc` reads -- this is the SAME requirement
`scripts/train_qc.py`'s own module docstring states, for the same reason.

## A deliberate, important omission: this script does not edit any config file

`configs/clinical/default.yaml`'s `gatekeeper.thresholds` key starts at
`null`. Running this script writes a `thresholds.json` file -- it does
**not** write that path back into `configs/clinical/default.yaml`, and it
does not mutate any config file on disk. Pointing `gatekeeper.thresholds` at
the written `thresholds.json` path is a separate, deliberate step someone
(a human, or a different task) takes only after reviewing the actual
numbers `thresholds.json` holds -- a threshold that flows into the deployed
refusal gate without a human looking at it first is exactly the "made-up
threshold that looks plausible" `configs/clinical/default.yaml`'s own comment
on this block warns against.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import hydra
from omegaconf import DictConfig

from neurovision.analysis.gatekeeper_calibration import build_gatekeeper_calibration_table
from neurovision.inference.gatekeeper import Thresholds, calibrate_thresholds
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _print_summary(thresholds: Thresholds, regions: Sequence[str], out_dir: Path) -> None:
    """Prints (not logs -- see `scripts/validate_qc.py`'s identical convention) a compact summary.

    Args:
        thresholds: The just-fitted `Thresholds`.
        regions: The regions `thresholds.predicted_dice` / `thresholds.conformal_band`
            were fit for, in the order to print them.
        out_dir: Where `calibration_table.csv` and `thresholds.json` were written,
            named in the summary's header line only.
    """
    lines = [
        "=" * 78,
        f"Gatekeeper calibration summary -- out_dir={out_dir}",
        "=" * 78,
        f"  calibration_n={thresholds.calibration_n} (caution_quantile="
        f"{thresholds.caution_quantile}, refuse_quantile={thresholds.refuse_quantile})",
        "  predicted_dice: region -> (refuse_below, caution_below); LOW is bad",
    ]
    for region in regions:
        refuse_below, caution_below = thresholds.predicted_dice[region]
        lines.append(
            f"    {region:>3s}: refuse_below={refuse_below:.4f} caution_below={caution_below:.4f}"
        )
    lines.append("  conformal_band: region -> (caution_above, refuse_above); HIGH is bad")
    for region in regions:
        caution_above, refuse_above = thresholds.conformal_band[region]
        lines.append(
            f"    {region:>3s}: caution_above={caution_above:.4f} refuse_above={refuse_above:.4f}"
        )
    ood_caution_above, ood_refuse_above = thresholds.ood_score
    lines.append(
        "  ood_score (STRUCTURAL PLACEHOLDER -- not a validated detector, not enabled by "
        f"default): caution_above={ood_caution_above:.4f} refuse_above={ood_refuse_above:.4f}"
    )
    lines.append("-" * 78)
    # print only, not logger.info as well -- matches scripts/validate_qc.py's /
    # scripts/train_qc.py's own end-of-run summary convention.
    print("\n".join(lines))


def run_calibration(cfg: DictConfig) -> dict[str, Path]:
    """Builds the calibration table, fits `Thresholds`, and writes both to disk.

    Args:
        cfg: The full composed Hydra config. `cfg.model` must already be
            composed as the `segqc` model group -- see this module's docstring.
            Reads `cfg.clinical.gatekeeper.{regions,caution_quantile,
            refuse_quantile,out_dir}` directly; every other key
            `build_gatekeeper_calibration_table` needs (`cfg.analysis.qc`,
            `cfg.analysis.qc_validate.checkpoint`, `cfg.clinical.gatekeeper.
            conformal_dir` / `conformal_alpha`, `cfg.inference.postprocess`) is
            read by that function itself.

    Returns:
        `{"calibration_table": <csv path>, "thresholds": <json path>}`.

    Raises:
        FileNotFoundError: Propagated from `build_gatekeeper_calibration_table`
            -- the QC checkpoint, the conformal `curves.npz` / `fit.json`, or a
            case's preprocessed/logits files are missing.
        ValueError: Propagated from `build_gatekeeper_calibration_table` (the
            three component tables share no case_id) or from
            `calibrate_thresholds` (an empty `regions`, an out-of-range
            quantile, or a required signal column missing/all-NaN).
    """
    gk_cfg = cfg.clinical.gatekeeper

    # Resolved and logged purely so a run's console output always states which
    # device this session thought it was on -- never a hardcoded "cpu" literal
    # with no log line. This has no effect on the actual work below:
    # qc_predicted_dice_table (inside build_gatekeeper_calibration_table)
    # always runs its SegQC forward pass on CPU regardless of cfg.device, per
    # that function's own module docstring -- this calibration step is a
    # Mac-only CPU job by design (CLAUDE.md's machine split).
    device = get_device(cfg)
    logger.info(
        "calibrate_gatekeeper: resolved device=%s (informational only -- this run's actual "
        "SegQC forward passes are hard-coded to CPU inside qc_predicted_dice_table).",
        device,
    )

    table = build_gatekeeper_calibration_table(cfg)

    out_dir = ensure_dir(str(gk_cfg.out_dir))
    calibration_table_path = out_dir / "calibration_table.csv"
    table.to_csv(calibration_table_path, index=False)
    logger.info(
        "calibrate_gatekeeper: wrote %s (%d calibration case(s)) -- open this file to see "
        "exactly what the thresholds below were fit from.",
        calibration_table_path,
        len(table),
    )

    regions = [str(r) for r in gk_cfg.regions]
    thresholds = calibrate_thresholds(
        table,
        regions=regions,
        caution_quantile=float(gk_cfg.caution_quantile),
        refuse_quantile=float(gk_cfg.refuse_quantile),
    )

    thresholds_path = out_dir / "thresholds.json"
    write_json(thresholds.to_dict(), thresholds_path, indent=2)
    logger.info(
        "calibrate_gatekeeper: wrote %s. configs/clinical/default.yaml's "
        "gatekeeper.thresholds is NOT edited by this script -- that must be done "
        "manually, after reviewing this file's actual numbers.",
        thresholds_path,
    )

    _print_summary(thresholds, regions, out_dir)

    return {"calibration_table": calibration_table_path, "thresholds": thresholds_path}


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Runs gatekeeper threshold calibration, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
            Must include `model=segqc` -- see this module's docstring.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_calibration(cfg)


if __name__ == "__main__":
    main()
