"""Hydra entry point for the tumour-burden profile driver.

Walks a frozen split, loads each case's class map (either a saved prediction
or the preprocessed ground-truth label) plus its `meta.json`, computes one
`neurovision.anatomy.burden.burden_profile` row per case, and writes the
result to `burden.csv`. CPU-only and deterministic: this script loads no
checkpoint and runs no model -- it only reads artifacts that
`scripts/evaluate.py` and `neurovision.data.preprocessing` already wrote to
disk.

Example usage:

    python scripts/burden.py analysis.burden.source=label
    python scripts/burden.py analysis.burden.source=prediction \\
        analysis.burden.eval_dir=outputs/eval_test analysis.burden.split=test

The geometry rule this script exists to get right: saved predictions
(`<eval_dir>/predictions/<case>.npy`) are in ORIGINAL BraTS geometry
(240x240x155), while preprocessed labels (`<preprocessed_dir>/<case>/label.npy`)
are CROPPED to the nonzero bounding box. `CaseGeometry.from_meta`'s `cropped`
flag differs between the two by the crop offset, so this script derives
`cropped` from `analysis.burden.source` and never accepts it as an
independent config value -- see `_resolve_source_root`.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from neurovision.anatomy.burden import CaseGeometry, burden_profile
from neurovision.data.dataset import load_splits
from neurovision.utils.io import ensure_dir, read_json, write_yaml
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/evaluate.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

_VALID_SOURCES = ("prediction", "label")


@dataclass(frozen=True)
class BurdenSource:
    """Where one case's class map and its meta.json come from."""

    case_id: str
    array_path: Path
    meta_path: Path
    cropped: bool


def _resolve_source_root(burden_cfg: DictConfig) -> tuple[Path, bool]:
    """Resolves the root directory holding the class-map arrays, and `cropped`.

    `cropped` is derived here, from `burden_cfg.source`, and nowhere else --
    it must never be taken as an independent flag, because a mismatch reports
    the wrong hemisphere for every case while every shape check still passes.

    Args:
        burden_cfg: `cfg.analysis.burden`.

    Returns:
        `(root_dir, cropped)`. For `source="prediction"`, `root_dir` is
        `<eval_dir>/predictions` and `cropped` is False (saved predictions
        are in original, uncropped BraTS geometry). For `source="label"`,
        `root_dir` is `preprocessed_dir` and `cropped` is True.

    Raises:
        ValueError: If `source == "prediction"` and `eval_dir` is null, or if
            `source` is not one of `_VALID_SOURCES`.
    """
    source = burden_cfg.source
    if source == "prediction":
        eval_dir = burden_cfg.eval_dir
        if eval_dir is None:
            raise ValueError(
                "cfg.analysis.burden.eval_dir is null, but analysis.burden.source='prediction' "
                "requires it -- the array comes from <eval_dir>/predictions/<case_id>.npy."
            )
        return Path(eval_dir) / "predictions", False
    if source == "label":
        return Path(burden_cfg.preprocessed_dir), True
    raise ValueError(
        f"cfg.analysis.burden.source={source!r} is not valid. Valid values: {_VALID_SOURCES}."
    )


def resolve_sources(cfg: DictConfig) -> list[BurdenSource]:
    """Resolves the list of `BurdenSource`s for one split.

    Cases whose array or meta.json is missing on disk are logged at WARNING
    (as a group, with the count) and excluded from the result rather than
    silently dropped -- a missing-file bug should be visible in the run log,
    not just in a shorter-than-expected CSV.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        One `BurdenSource` per case that has both files present, in the
        split file's order.

    Raises:
        ValueError: See `_resolve_source_root`, or if
            `cfg.analysis.burden.split` is not a key of the loaded splits
            file.
        FileNotFoundError: If every case in the split is missing its array
            and/or meta.json. This almost always means a wrong `eval_dir` or
            `preprocessed_dir` -- returning an empty list here would let the
            rest of the pipeline silently write an empty CSV that looks like
            a successful run over zero cases.
    """
    burden_cfg = cfg.analysis.burden
    root_dir, cropped = _resolve_source_root(burden_cfg)
    preprocessed_dir = Path(burden_cfg.preprocessed_dir)

    split = burden_cfg.split
    splits = load_splits(cfg.data.splits.path)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}. Available splits: {sorted(splits.keys())}.")
    case_ids = list(splits[split])

    sources: list[BurdenSource] = []
    missing: list[str] = []
    for case_id in case_ids:
        if burden_cfg.source == "prediction":
            array_path = root_dir / f"{case_id}.npy"
        else:
            array_path = root_dir / case_id / "label.npy"
        meta_path = preprocessed_dir / case_id / "meta.json"

        if not array_path.is_file() or not meta_path.is_file():
            missing.append(case_id)
            continue
        sources.append(
            BurdenSource(
                case_id=case_id, array_path=array_path, meta_path=meta_path, cropped=cropped
            )
        )

    if missing:
        logger.warning(
            "resolve_sources: %d/%d case(s) in split %r are missing their array and/or "
            "meta.json and were excluded: %s",
            len(missing),
            len(case_ids),
            split,
            missing,
        )

    if not sources:
        raise FileNotFoundError(
            f"resolve_sources: no usable case(s) for split={split!r}, "
            f"source={burden_cfg.source!r}. Resolved directory: {root_dir.resolve()}. All "
            f"{len(case_ids)} case(s) were missing their array and/or meta.json -- check "
            "cfg.analysis.burden.eval_dir / cfg.analysis.burden.preprocessed_dir."
        )

    return sources


def load_case(source: BurdenSource) -> tuple[np.ndarray, dict]:
    """Loads and shape-validates one case's class map and meta.json.

    Args:
        source: A resolved `BurdenSource`.

    Returns:
        `(classes, meta)`: the loaded class-map array, `(D, H, W)`, and the
        parsed meta.json mapping.

    Raises:
        ValueError: If the array is not 3-D, or if its shape does not match
            the expected shape from meta.json (`original_shape` when
            `source.cropped` is False, `cropped_shape` when True). This is
            the guard against a prediction/label mix-up, an `eval_dir` from a
            different preprocessing run, or a `cropped` flag out of step
            with `source` -- see the module docstring.
    """
    array = np.load(source.array_path)
    meta = read_json(source.meta_path)

    if array.ndim != 3:
        raise ValueError(
            f"load_case({source.case_id!r}): expected a 3-D class map at {source.array_path}, "
            f"got shape {array.shape} (ndim={array.ndim})."
        )

    if source.cropped:
        expected_key = "cropped_shape"
    else:
        expected_key = "original_shape"
    expected = tuple(int(s) for s in meta[expected_key])

    if tuple(array.shape) != expected:
        raise ValueError(
            f"load_case({source.case_id!r}): array at {source.array_path} has shape "
            f"{tuple(array.shape)}, expected {expected} (meta['{expected_key}'], "
            f"cropped={source.cropped}). This usually means a prediction/label mix-up, an "
            "eval_dir from a different preprocessing run, or `cropped` out of step with "
            "`source`."
        )

    return array, meta


def profile_case(source: BurdenSource, cfg: DictConfig) -> dict[str, float | int | str]:
    """Computes one burden-profile row for a single case.

    Args:
        source: A resolved `BurdenSource`.
        cfg: The full composed Hydra config.

    Returns:
        A flat dict: `burden_profile`'s keys, with `case_id` inserted as the
        FIRST key.

    Raises:
        ValueError: See `load_case` and `neurovision.anatomy.burden.burden_profile`.
    """
    classes, meta = load_case(source)
    burden_cfg = cfg.analysis.burden

    geom = CaseGeometry.from_meta(
        meta, cropped=source.cropped, midline_index=burden_cfg.midline_index
    )
    profile = burden_profile(
        classes,
        geom,
        min_volume_mm3=float(burden_cfg.min_volume_mm3),
        connectivity=int(burden_cfg.connectivity),
    )
    return {"case_id": source.case_id, **profile}


def _log_sanity_summary(rows: list[dict[str, float | int | str]]) -> None:
    """Logs a few known-plausible-range quantities so a broken measure is visible in the log.

    Args:
        rows: The successfully-computed burden-profile rows.
    """
    df = pd.DataFrame.from_records(rows)
    logger.info(
        "Sanity summary over %d case(s): median vol_WT_mm3=%.1f, median sphericity_WT=%.3f, "
        "fraction with n_components_WT > 1 = %.3f.",
        len(df),
        df["vol_WT_mm3"].median(),
        df["sphericity_WT"].median(),
        (df["n_components_WT"] > 1).mean(),
    )


def run_burden(cfg: DictConfig) -> Path:
    """Profiles every resolvable case of a split and writes the result to disk.

    The CSV is rewritten after EVERY case (not once at the end), the same
    convention `scripts/evaluate.py` uses for `per_case_metrics.csv`, so a
    killed run leaves usable partial output. A case that raises while being
    profiled does not kill the run -- its traceback is logged at ERROR, it is
    counted as a failure, and the loop continues.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        Path to the written `burden.csv`.

    Raises:
        Whatever `resolve_sources` raises (a config problem, checked before
        any work starts). Also:
        RuntimeError: If every resolved case failed to profile -- a CSV of
            zero real rows would look like a successful run over an empty
            split rather than a run where nothing actually worked.
    """
    burden_cfg = cfg.analysis.burden
    sources = resolve_sources(cfg)
    root_dir, _cropped = _resolve_source_root(burden_cfg)

    # cfg.output_dir interpolates experiment_name, whose default is
    # baseline_unet3d -- so a burden profile computed from ANOTHER model's
    # predictions lands in the baseline's directory and reads as the
    # baseline's result. That has now happened twice in this project (see
    # analysis.detection.out_dir). An explicit analysis.burden.out_dir wins
    # when set; the old behaviour is the fallback so existing commands are
    # unchanged.
    out_dir = ensure_dir(burden_cfg.get("out_dir") or cfg.output_dir)
    csv_path = out_dir / burden_cfg.out_name

    rows: dict[str, dict[str, float | int | str]] = {}
    n_failed = 0

    for source in tqdm(sources, desc=f"Burden profile ({burden_cfg.split})"):
        try:
            rows[source.case_id] = profile_case(source, cfg)
        except Exception:
            n_failed += 1
            logger.error(
                "profile_case failed for case %s:\n%s", source.case_id, traceback.format_exc()
            )
            continue

        # Rewritten after every case, not just at the end: a killed run keeps
        # every already-profiled case's row instead of losing all of them.
        pd.DataFrame.from_records(list(rows.values())).to_csv(csv_path, index=False)

    n_succeeded = len(rows)
    logger.info(
        "Burden profile complete: %d succeeded, %d failed (of %d resolved case(s)).",
        n_succeeded,
        n_failed,
        len(sources),
    )

    if n_succeeded == 0:
        raise RuntimeError(
            f"run_burden: 0/{len(sources)} resolved case(s) succeeded -- see the ERROR-level "
            "log lines above for per-case tracebacks."
        )

    pd.DataFrame.from_records(list(rows.values())).to_csv(csv_path, index=False)

    # A CSV whose provenance is only in a terminal log nobody kept cannot be
    # traced months later -- same reasoning as eval_config.yaml.
    config_record = OmegaConf.to_container(burden_cfg, resolve=True)
    config_record["resolved_source_dir"] = str(root_dir.resolve())
    write_yaml(config_record, out_dir / "burden_config.yaml")

    _log_sanity_summary(list(rows.values()))

    return csv_path


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Profile a frozen split's tumour burden, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    csv_path = run_burden(cfg)
    print(f"Burden profile written to {csv_path}")


if __name__ == "__main__":
    main()
