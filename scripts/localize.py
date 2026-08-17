"""Hydra entry point for the anatomical-localisation driver.

Phase 1 of `docs/research/interpretable_pipeline_plan.md` §5. Walks a frozen
split, loads each case's class map (either a saved prediction or the
preprocessed ground-truth label) plus its `meta.json`, intersects it with the
SRI24/TZO atlas, and writes two CSVs: one long per-structure table
(`anatomy.csv`) and one per-case summary (`anatomy_summary.csv`). CPU-only and
deterministic -- this script loads no checkpoint and runs no model, it only
reads artifacts that `scripts/evaluate.py` and `neurovision.data.preprocessing`
already wrote to disk, plus the atlas `scripts/fetch_atlas.py` downloaded.

Example usage:

    python scripts/localize.py analysis.localize.source=label
    python scripts/localize.py analysis.localize.source=prediction \\
        analysis.localize.eval_dir=outputs/eval_test analysis.localize.split=test

The geometry rule this script exists to get right -- identical to
`scripts/burden.py`'s: saved predictions (`<eval_dir>/predictions/<case>.npy`)
are in ORIGINAL BraTS geometry (240x240x155), while preprocessed labels
(`<preprocessed_dir>/<case>/label.npy`) are CROPPED to the nonzero bounding
box. `neurovision.anatomy.localize.atlas_for_case`'s `cropped` flag differs
between the two by the crop offset, so this script derives `cropped` from
`analysis.localize.source` and never accepts it as an independent value --
see `_resolve_source_root`.
"""

from __future__ import annotations

import logging
import math
import traceback
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from neurovision.anatomy import burden
from neurovision.anatomy.atlas import Atlas, load_atlas
from neurovision.anatomy.localize import (
    KnowledgeBase,
    distance_to_eloquent,
    eloquent_union_mask,
    load_knowledge,
    localize_case,
    summarize_case,
)
from neurovision.data.dataset import load_splits
from neurovision.utils.io import ensure_dir, read_json, read_yaml, write_yaml
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/burden.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

_VALID_SOURCES = ("prediction", "label")

# Must match neurovision.anatomy.localize's internal `_UNLABELLED_NAME`
# exactly -- it is not exported, since it is an implementation detail of that
# module's table, but this script has to name it too when deciding which rows
# `min_frac` filtering may never drop.
_UNLABELLED_STRUCTURE_NAME = "unlabelled"

# Below this, `min_frac` is discarding so much of the tumour that the per-case
# table no longer describes it. Expected retention is high; a little filtering
# is by design.
_MIN_RETAINED_FRAC_WARN = 0.5


@dataclass(frozen=True)
class LocalizeSource:
    """Where one case's class map and its meta.json come from."""

    case_id: str
    array_path: Path
    meta_path: Path
    cropped: bool


def _resolve_source_root(localize_cfg: DictConfig) -> tuple[Path, bool]:
    """Resolves the root directory holding the class-map arrays, and `cropped`.

    `cropped` is derived here, from `localize_cfg.source`, and nowhere else --
    it must never be taken as an independent flag, because a mismatch shifts
    every structure assignment by the crop offset and produces an entirely
    plausible, entirely wrong table while every shape check still passes.

    Args:
        localize_cfg: `cfg.analysis.localize`.

    Returns:
        `(root_dir, cropped)`. For `source="prediction"`, `root_dir` is
        `<eval_dir>/predictions` and `cropped` is False (saved predictions
        are in original, uncropped BraTS geometry). For `source="label"`,
        `root_dir` is `preprocessed_dir` and `cropped` is True.

    Raises:
        ValueError: If `source == "prediction"` and `eval_dir` is null, or if
            `source` is not one of `_VALID_SOURCES`.
    """
    source = localize_cfg.source
    if source == "prediction":
        eval_dir = localize_cfg.eval_dir
        if eval_dir is None:
            raise ValueError(
                "cfg.analysis.localize.eval_dir is null, but analysis.localize.source="
                "'prediction' requires it -- the array comes from "
                "<eval_dir>/predictions/<case_id>.npy."
            )
        return Path(eval_dir) / "predictions", False
    if source == "label":
        return Path(localize_cfg.preprocessed_dir), True
    raise ValueError(
        f"cfg.analysis.localize.source={source!r} is not valid. Valid values: {_VALID_SOURCES}."
    )


def resolve_sources(cfg: DictConfig) -> list[LocalizeSource]:
    """Resolves the list of `LocalizeSource`s for one split.

    Cases whose array or meta.json is missing on disk are logged at WARNING
    (as a group, with the count) and excluded from the result rather than
    silently dropped -- a missing-file bug should be visible in the run log,
    not just in a shorter-than-expected CSV.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        One `LocalizeSource` per case that has both files present, in the
        split file's order.

    Raises:
        ValueError: See `_resolve_source_root`, or if
            `cfg.analysis.localize.split` is not a key of the loaded splits
            file.
        FileNotFoundError: If every case in the split is missing its array
            and/or meta.json. This almost always means a wrong `eval_dir` or
            `preprocessed_dir` -- returning an empty list here would let the
            rest of the pipeline silently write an empty CSV that looks like
            a successful run over zero cases.
    """
    localize_cfg = cfg.analysis.localize
    root_dir, cropped = _resolve_source_root(localize_cfg)
    preprocessed_dir = Path(localize_cfg.preprocessed_dir)

    split = localize_cfg.split
    splits = load_splits(cfg.data.splits.path)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}. Available splits: {sorted(splits.keys())}.")
    case_ids = list(splits[split])

    sources: list[LocalizeSource] = []
    missing: list[str] = []
    for case_id in case_ids:
        if localize_cfg.source == "prediction":
            array_path = root_dir / f"{case_id}.npy"
        else:
            array_path = root_dir / case_id / "label.npy"
        meta_path = preprocessed_dir / case_id / "meta.json"

        if not array_path.is_file() or not meta_path.is_file():
            missing.append(case_id)
            continue
        sources.append(
            LocalizeSource(
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
            f"source={localize_cfg.source!r}. Resolved directory: {root_dir.resolve()}. All "
            f"{len(case_ids)} case(s) were missing their array and/or meta.json -- check "
            "cfg.analysis.localize.eval_dir / cfg.analysis.localize.preprocessed_dir."
        )

    return sources


def load_case(source: LocalizeSource) -> tuple[np.ndarray, dict]:
    """Loads and shape-validates one case's class map and meta.json.

    Args:
        source: A resolved `LocalizeSource`.

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


def _crop_eloquent_mask(eloquent_mask: np.ndarray, meta: dict, *, cropped: bool) -> np.ndarray:
    """Crops the whole-atlas eloquent-union mask to one case's bbox, or returns it unchanged.

    Mirrors `neurovision.anatomy.localize.atlas_for_case`'s bbox slicing
    exactly, so the eloquent mask lines up with the parcellation and the
    class map at whatever geometry the case's mask is in. `eloquent_mask` is
    not part of the `Atlas` dataclass (it is derived from the knowledge
    base), so it cannot go through `atlas_for_case` itself.

    Args:
        eloquent_mask: `(D, H, W)` boolean array, full atlas geometry.
        meta: The case's `meta.json` mapping.
        cropped: Whether to crop to `meta["bbox"]`.

    Returns:
        A `(D, H, W)` boolean array view, cropped when `cropped=True`.
    """
    if not cropped:
        return eloquent_mask
    bbox = tuple(tuple(int(v) for v in pair) for pair in meta["bbox"])
    slices = tuple(slice(start, end) for start, end in bbox)
    return eloquent_mask[slices]


def _summary_region_scope(table: pd.DataFrame) -> pd.DataFrame:
    """Restricts `table` to whatever rows `summarize_case` itself would summarize.

    Mirrors `neurovision.anatomy.localize.summarize_case`'s own region
    selection exactly: when `table` carries a `region` column, only the
    `"WT"` rows count (falling back to an empty slice if there are none);
    without a `region` column, the whole table counts. Kept in lockstep with
    `summarize_case` on purpose -- `frac_of_tumour_retained` has to describe
    the same slice of the table that the rest of the summary row describes,
    or it would be a number about a different region sitting next to one
    about WT.

    Args:
        table: A `localize_case` (or `localize_mask`) output table.

    Returns:
        The scoped `DataFrame` (a view/slice, possibly empty).
    """
    if "region" in table.columns:
        wt_only = table[table["region"] == "WT"]
        return wt_only if not wt_only.empty else table.iloc[0:0]
    return table


def localize_one(
    source: LocalizeSource,
    atlas: Atlas,
    knowledge: KnowledgeBase,
    eloquent_mask: np.ndarray,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Localises one case: the per-structure table plus its case-level summary row.

    Args:
        source: A resolved `LocalizeSource`.
        atlas: The loaded `Atlas` (built ONCE by the caller, not per case).
        knowledge: The loaded `KnowledgeBase` (built ONCE by the caller).
        eloquent_mask: The whole-atlas eloquent-union mask (built ONCE by the
            caller, via `eloquent_union_mask`).
        cfg: The full composed Hydra config.

    Returns:
        `(table, summary_row)`. `table` has `case_id` as its FIRST column,
        then `region` and `neurovision.anatomy.localize`'s columns, filtered
        by `cfg.analysis.localize.min_frac` (the `"unlabelled"` row is never
        dropped by that filter). `summary_row` is `summarize_case`'s output
        (computed on the FILTERED table) with `case_id` first,
        `distance_to_eloquent_mm` / `near_eloquent` overwritten with the real
        measured distance -- `summarize_case` alone cannot compute it, since a
        tidy table carries no coordinates -- and an added
        `frac_of_tumour_retained` field: the fraction of the WT-scoped tumour
        (the same scope `summarize_case` itself summarizes, see
        `_summary_region_scope`) that survived `min_frac` filtering.

    Raises:
        ValueError: See `load_case` and `neurovision.anatomy.localize`.
    """
    classes, meta = load_case(source)
    localize_cfg = cfg.analysis.localize
    regions = [str(r) for r in localize_cfg.regions]

    table = localize_case(
        classes, atlas, meta, cropped=source.cropped, regions=regions, knowledge=knowledge
    )
    table.insert(0, "case_id", source.case_id)

    # frac_of_tumour is documented to sum to 1.0 per region -- checked here,
    # on the table as localize_case returned it, BEFORE any filtering.
    # min_frac is designed to drop small non-zero rows, so checking this
    # identity AFTER filtering would fire on essentially every case of every
    # run; a violation HERE instead means localize_case's own output is
    # wrong, not that the filter did its job.
    for region, group in table.groupby("region"):
        total = float(group["frac_of_tumour"].sum())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            logger.warning(
                "localize_one(%s): localize_case's UNFILTERED frac_of_tumour sums to %.9f "
                "(expected 1.0) for region %s. This indicates a problem in localize_case's own "
                "output, not in min_frac filtering.",
                source.case_id,
                total,
                region,
            )

    min_frac = float(localize_cfg.min_frac)
    drop_mask = (
        (table["frac_of_structure"] < min_frac)
        & (table["frac_of_tumour"] < min_frac)
        & (table["structure"] != _UNLABELLED_STRUCTURE_NAME)
    )
    filtered = table.loc[~drop_mask].reset_index(drop=True)

    # How much of the tumour survived filtering, scoped to exactly what
    # summarize_case summarizes below (WT, when the table has a region
    # column) -- a diagnostic quantified in the summary CSV rather than
    # announced in a log nobody keeps.
    unfiltered_scope = _summary_region_scope(table)
    filtered_scope = _summary_region_scope(filtered)
    unfiltered_total = float(unfiltered_scope["frac_of_tumour"].sum())
    if unfiltered_total > 0.0:
        frac_of_tumour_retained = float(filtered_scope["frac_of_tumour"].sum()) / unfiltered_total
    else:
        # No tumour in scope to begin with -- vacuously fully "retained",
        # rather than a division by zero.
        frac_of_tumour_retained = 1.0

    if frac_of_tumour_retained < _MIN_RETAINED_FRAC_WARN:
        logger.warning(
            "localize_one(%s): min_frac filtering retained only %.3f of the tumour "
            "(threshold %.3f) at min_frac=%s. Consider lowering min_frac.",
            source.case_id,
            frac_of_tumour_retained,
            _MIN_RETAINED_FRAC_WARN,
            min_frac,
        )

    # summarize_case can only report 0.0-on-overlap / NaN-otherwise for
    # distance_to_eloquent_mm, since a tidy table carries no coordinates.
    # Compute the real measured distance here, from the mask and the atlas.
    wt_mask = burden.region_mask(classes, "WT")
    spacing = tuple(float(s) for s in meta["spacing"])
    cropped_eloquent = _crop_eloquent_mask(eloquent_mask, meta, cropped=source.cropped)
    distance_mm = distance_to_eloquent(wt_mask, cropped_eloquent, spacing=spacing)
    near_eloquent_mm = float(knowledge.near_eloquent_mm)
    # False (never NaN-propagated) when the distance is NaN: "near an
    # eloquent structure" cannot be true of an undefined distance.
    near_eloquent = bool(not math.isnan(distance_mm) and distance_mm <= near_eloquent_mm)

    summary_row: dict[str, object] = {
        "case_id": source.case_id,
        **summarize_case(filtered, knowledge),
    }
    summary_row["distance_to_eloquent_mm"] = distance_mm
    summary_row["near_eloquent"] = near_eloquent
    summary_row["frac_of_tumour_retained"] = frac_of_tumour_retained

    return filtered, summary_row


def _log_sanity_summary(summary_rows: list[dict[str, object]]) -> None:
    """Logs a few known-plausible-range quantities so a broken measure is visible in the log.

    Args:
        summary_rows: The successfully-computed per-case summary rows.
    """
    df = pd.DataFrame.from_records(summary_rows)
    n_low_retention = int((df["frac_of_tumour_retained"] < _MIN_RETAINED_FRAC_WARN).sum())
    logger.info(
        "Sanity summary over %d case(s): median n_structures_involved=%.1f, median "
        "frac_unlabelled=%.3f, fraction with any eloquent involvement=%.3f, median "
        "frac_of_tumour_retained=%.3f, %d case(s) below the %.3f retention warn threshold.",
        len(df),
        df["n_structures_involved"].median(),
        df["frac_unlabelled"].median(),
        (df["n_eloquent_structures"] > 0).mean(),
        df["frac_of_tumour_retained"].median(),
        n_low_retention,
        _MIN_RETAINED_FRAC_WARN,
    )


def run_localize(cfg: DictConfig) -> tuple[Path, Path]:
    """Localises every resolvable case of a split and writes the result to disk.

    The atlas, the knowledge base, and the eloquent-union mask are each
    expensive to build and are loaded exactly ONCE, before the per-case loop
    -- building them per case would dominate runtime. Both output CSVs are
    rewritten after EVERY case (not once at the end), the same convention
    `scripts/evaluate.py` and `scripts/burden.py` use, so a killed run leaves
    usable partial output. A case that raises while being localised does not
    kill the run -- its traceback is logged at ERROR, it is counted as a
    failure, and the loop continues.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `(anatomy_csv_path, summary_csv_path)`.

    Raises:
        Whatever `resolve_sources` raises (a config problem, checked before
        any work starts). Also:
        RuntimeError: If every resolved case failed to localise -- CSVs of
            zero real rows would look like a successful run over an empty
            split rather than a run where nothing actually worked.
    """
    localize_cfg = cfg.analysis.localize
    sources = resolve_sources(cfg)
    root_dir, _cropped = _resolve_source_root(localize_cfg)

    atlas = load_atlas(cfg.anatomy)
    knowledge = load_knowledge(localize_cfg.eloquence_map, localize_cfg.lobe_map, atlas)
    eloquent_mask = eloquent_union_mask(atlas, knowledge)

    out_dir = ensure_dir(cfg.output_dir)
    anatomy_csv_path = out_dir / localize_cfg.out_name
    summary_csv_path = out_dir / localize_cfg.summary_name

    tables: list[pd.DataFrame] = []
    summary_rows: dict[str, dict[str, object]] = {}
    n_failed = 0

    for source in tqdm(sources, desc=f"Localize ({localize_cfg.split})"):
        try:
            table, summary_row = localize_one(source, atlas, knowledge, eloquent_mask, cfg)
        except Exception:
            n_failed += 1
            logger.error(
                "localize_one failed for case %s:\n%s", source.case_id, traceback.format_exc()
            )
            continue

        tables.append(table)
        summary_rows[source.case_id] = summary_row

        # Rewritten after every case, not just at the end: a killed run keeps
        # every already-localised case's rows instead of losing all of them.
        pd.concat(tables, ignore_index=True).to_csv(anatomy_csv_path, index=False)
        pd.DataFrame.from_records(list(summary_rows.values())).to_csv(summary_csv_path, index=False)

    n_succeeded = len(summary_rows)
    logger.info(
        "Localize complete: %d succeeded, %d failed (of %d resolved case(s)).",
        n_succeeded,
        n_failed,
        len(sources),
    )

    if n_succeeded == 0:
        raise RuntimeError(
            f"run_localize: 0/{len(sources)} resolved case(s) succeeded -- see the ERROR-level "
            "log lines above for per-case tracebacks."
        )

    pd.concat(tables, ignore_index=True).to_csv(anatomy_csv_path, index=False)
    pd.DataFrame.from_records(list(summary_rows.values())).to_csv(summary_csv_path, index=False)

    # A CSV whose provenance is only in a terminal log nobody kept cannot be
    # traced months later -- same reasoning as eval_config.yaml /
    # burden_config.yaml. The coverage line especially: a report built on a
    # thin knowledge base must carry that fact with it.
    coverage_line = knowledge.coverage_line(len(knowledge.eloquence))
    config_record = OmegaConf.to_container(localize_cfg, resolve=True)
    config_record["resolved_source_dir"] = str(root_dir.resolve())
    config_record["atlas"] = {
        "name": atlas.name,
        "version": atlas.version,
        "source": atlas.source,
        "licence": str(cfg.anatomy.licence),
    }
    config_record["coverage_line"] = coverage_line
    write_yaml(config_record, out_dir / "localize_config.yaml")

    _log_sanity_summary(list(summary_rows.values()))

    return anatomy_csv_path, summary_csv_path


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Localise a frozen split's tumour masks against the SRI24 atlas, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    anatomy_csv_path, summary_csv_path = run_localize(cfg)
    # localize_config.yaml was just written by run_localize, alongside the
    # two CSVs -- read the coverage line back from it rather than
    # recomputing it, so there is exactly one place that assembles it.
    config_record = read_yaml(anatomy_csv_path.parent / "localize_config.yaml")
    print(f"Per-structure table written to {anatomy_csv_path}")
    print(f"Per-case summary written to {summary_csv_path}")
    print(f"Knowledge coverage: {config_record['coverage_line']}")


if __name__ == "__main__":
    main()
