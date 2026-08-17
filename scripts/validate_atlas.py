"""Phase 0's gate driver: proves the SRI24 atlas lands on our BraTS cases.

Runs `neurovision.anatomy.alignment.run_checks` against the real preprocessed
BraTS tree -- brain-mask Dice, `_L`/`_R` laterality, and the (advisory)
population lobe distribution -- and writes every result to disk, plus a
visual QC overlay figure. CPU-only: no model, no checkpoint, no GPU. See
`docs/research/phase0_atlas_findings.md` for why each check exists and what
the real atlas measured.

Cost asymmetry this script is built around: brain masks come from
`image.npy` (`(4, D, H, W)` float16, expensive to read), tumour masks from
`label.npy` (`uint8`, cheap). The lobe check wants many cases, the brain
check does not -- so both checks are nested samples of ONE draw
(`sample_cases`), never two independent draws, so the written report
describes a single cohort.

Example usage:

    python scripts/validate_atlas.py
    python scripts/validate_atlas.py anatomy.validation.n_cases=100

Exits non-zero when a GATING check fails (`brain_mask_dice`,
`laterality_pairs`, `midline_estimate`); the advisory `lobe_distribution`
check never affects the exit code.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra

# The Agg backend MUST be selected before pyplot is imported anywhere in this
# process, or matplotlib will try (and, headlessly, fail or hang) to pick an
# interactive backend that can open a window. This is the only file in the
# project allowed to import matplotlib -- see CLAUDE.md's constraint on this
# script. The two lines below intentionally break up the import block, which
# is why the second one needs a noqa.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from neurovision.anatomy.alignment import (  # noqa: E402
    AlignmentReport,
    load_lobe_map,
    run_checks,
    uncrop,
)
from neurovision.anatomy.atlas import Atlas, load_atlas  # noqa: E402
from neurovision.utils.io import ensure_dir, read_json, write_json  # noqa: E402
from neurovision.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/evaluate.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# `cfg.anatomy.validation.n_brain_cases` does not exist in
# configs/anatomy/sri24.yaml yet (owned by another agent this session), so it
# is read with this default rather than added to the YAML -- the same pattern
# scripts/evaluate.py's `resolve_boundary_bands` uses for an equally-absent key.
_DEFAULT_N_BRAIN_CASES = 60

# Fixed artifact filenames written by write_report / qc_overlay_figure. Named
# once here so run_validation and main can never disagree about them.
_REPORT_JSON = "alignment_report.json"
_PER_CASE_DICE_CSV = "alignment_per_case_dice.csv"
_LATERALITY_CSV = "alignment_laterality_pairs.csv"
_LOBE_DISTRIBUTION_CSV = "alignment_lobe_distribution.csv"
_SUMMARY_TXT = "alignment_summary.txt"
_QC_PNG = "alignment_qc_overlay.png"


@dataclass(frozen=True)
class CaseSample:
    """One reproducible draw of case ids, nested for the two mask-cost tiers.

    Attributes:
        case_ids: The full sample -- used for the (cheap) tumour masks.
        brain_case_ids: A PREFIX of `case_ids` -- used for the (expensive)
            brain masks. Being a prefix, not an independent draw, is what
            keeps the report describing one cohort rather than two.
    """

    case_ids: tuple[str, ...]
    brain_case_ids: tuple[str, ...]


def sample_cases(cfg: Any, preprocessed_dir: str | Path) -> CaseSample:
    """Draws one reproducible sample of case ids from a preprocessed tree.

    Case ids are directory names directly under `preprocessed_dir` that
    contain a `meta.json`. The candidate list is sorted before sampling so
    the draw is reproducible across filesystems (directory iteration order
    is not guaranteed), and the sample itself is drawn with a LOCAL
    `random.Random` instance, never the global `random` module state, so
    calling this cannot perturb anything else in the process.

    Args:
        cfg: The `anatomy.validation` config node (attribute access to
            `n_cases`, `seed`, and -- via `.get()` with a default -- the
            not-yet-configured `n_brain_cases`).
        preprocessed_dir: Root of the preprocessed BraTS tree.

    Returns:
        A `CaseSample`. `brain_case_ids` is `case_ids[:n_brain_cases]`,
        clamped to `n_cases` if `n_brain_cases` is larger.

    Raises:
        FileNotFoundError: If `preprocessed_dir` does not exist, or exists
            but holds no case (a subdirectory with a `meta.json`).
        ValueError: If fewer cases exist than `cfg.n_cases` requests --
            silently sampling fewer would change the cohort a reported
            number describes.
    """
    preprocessed_dir = Path(preprocessed_dir)
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(
            f"sample_cases: preprocessed directory {preprocessed_dir.resolve()} does not exist."
        )

    available = sorted(
        p.name for p in preprocessed_dir.iterdir() if p.is_dir() and (p / "meta.json").is_file()
    )
    if not available:
        raise FileNotFoundError(
            "sample_cases: no cases (a subdirectory holding a meta.json) found under "
            f"{preprocessed_dir.resolve()}."
        )

    n_cases = int(cfg.n_cases)
    if len(available) < n_cases:
        raise ValueError(
            f"sample_cases: requested n_cases={n_cases}, but only {len(available)} case(s) "
            f"with a meta.json were found under {preprocessed_dir.resolve()}."
        )

    n_brain_cases = int(cfg.get("n_brain_cases", _DEFAULT_N_BRAIN_CASES))
    n_brain_cases = min(n_brain_cases, n_cases)

    rng = random.Random(int(cfg.seed))
    sampled = rng.sample(available, n_cases)

    return CaseSample(case_ids=tuple(sampled), brain_case_ids=tuple(sampled[:n_brain_cases]))


def brain_mask_iter(
    case_ids: Iterable[str], preprocessed_dir: str | Path
) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yields `(case_id, brain_mask)` pairs from `image.npy`, one case at a time.

    A generator, not a list: 400 uncropped `(240, 240, 155)` boolean arrays
    would be ~2.7 GB and pointless when only `brain_case_ids` (default 60)
    of them are actually consumed by `brain_mask_check`. `image.npy` is
    memory-mapped and only channel 0 is materialized, so the other three
    modalities' bytes are never read.

    Args:
        case_ids: Case ids to iterate, in order.
        preprocessed_dir: Root of the preprocessed BraTS tree.

    Yields:
        `(case_id, mask)`: `mask` is a boolean `(D, H, W)` array in
        ORIGINAL (uncropped) BraTS geometry.
    """
    preprocessed_dir = Path(preprocessed_dir)
    for case_id in case_ids:
        case_dir = preprocessed_dir / case_id
        meta = read_json(case_dir / "meta.json")
        image = np.load(case_dir / "image.npy", mmap_mode="r")
        channel0 = np.asarray(image[0])
        # abs(), never `> 0`: preprocessing z-scores each modality over its
        # nonzero voxels, so brain interiors are routinely NEGATIVE. `> 0`
        # would silently keep only the bright half of the brain -- the same
        # trap already documented for the qualitative panel's intensity
        # scaling (see CLAUDE.md).
        brain_mask = np.abs(channel0) != 0
        yield case_id, uncrop(brain_mask, meta["bbox"], meta["original_shape"])


def tumour_mask_iter(
    case_ids: Iterable[str], preprocessed_dir: str | Path
) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yields `(case_id, tumour_mask)` pairs from `label.npy`, one case at a time.

    A generator for the same reason as `brain_mask_iter`, though `label.npy`
    is cheap (`uint8`) so the memory pressure this avoids is smaller.

    Args:
        case_ids: Case ids to iterate, in order.
        preprocessed_dir: Root of the preprocessed BraTS tree.

    Yields:
        `(case_id, mask)`: `mask` is a boolean `(D, H, W)` array in
        ORIGINAL (uncropped) BraTS geometry.
    """
    preprocessed_dir = Path(preprocessed_dir)
    for case_id in case_ids:
        case_dir = preprocessed_dir / case_id
        meta = read_json(case_dir / "meta.json")
        label = np.load(case_dir / "label.npy")
        tumour_mask = label > 0
        yield case_id, uncrop(tumour_mask, meta["bbox"], meta["original_shape"])


def axial_display_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """Extracts one axial slice from a `(D, H, W)` volume for on-screen display.

    Matches the radiological display convention pinned in
    `app/frontend/src/lib/slicing.ts`, which fixed exactly this class of bug
    for the demo viewer: under the BraTS affine, axis 0 (d) runs
    right -> left and axis 1 (h) runs anterior -> posterior, so a naive
    "rows = the first remaining axis, columns = the second" slice puts
    anterior sideways -- every voxel is correct and the picture is still
    recognisably a brain, which is exactly why a shape test cannot catch it.

    After this transform: row 0 is anterior (top of the image, since no
    reversal is applied to axis 1 -- low `h` is anterior), and column `D-1`
    is the patient's LEFT, displayed on the RIGHT of the image (radiological
    convention -- axis 0's low index is patient right, and no reversal is
    applied to it either).

    Args:
        volume: `(D, H, W)` array.
        index: Axial slice index along axis 2 (w, inferior -> superior).

    Returns:
        `(H, D)` 2-D array.
    """
    return volume[:, :, index].T


def _label_boundary(labels_2d: np.ndarray) -> np.ndarray:
    """Boolean mask: True where a voxel's label differs from a 4-neighbour.

    Args:
        labels_2d: 2-D integer label array.

    Returns:
        Boolean array, same shape as `labels_2d`.
    """
    boundary = np.zeros_like(labels_2d, dtype=bool)
    diff_rows = labels_2d[:-1, :] != labels_2d[1:, :]
    boundary[:-1, :] |= diff_rows
    boundary[1:, :] |= diff_rows
    diff_cols = labels_2d[:, :-1] != labels_2d[:, 1:]
    boundary[:, :-1] |= diff_cols
    boundary[:, 1:] |= diff_cols
    return boundary


def _display_window(slice_2d: np.ndarray) -> tuple[float, float]:
    """1st/99th percentile of the NONZERO voxels, for a legible grey window.

    Z-scored MRI has a negative mean brain interior against an exact-zero
    background; a plain min-max (or including the background in the
    percentile) would wash the whole panel out to mid-grey. See CLAUDE.md's
    note on the qualitative panel's identical trap.
    """
    nonzero = slice_2d[slice_2d != 0]
    if nonzero.size == 0:
        lo, hi = float(slice_2d.min()), float(slice_2d.max())
    else:
        lo, hi = (float(v) for v in np.percentile(nonzero, [1.0, 99.0]))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def qc_overlay_figure(
    atlas: Atlas,
    case_ids: Iterable[str],
    preprocessed_dir: str | Path,
    out_path: str | Path,
    *,
    n_slices: int = 3,
) -> Path:
    """Renders a T1 + atlas-parcellation-boundary QC grid: rows = cases, columns = slices.

    This is the fourth Phase 0 check and the one that cannot be automated
    away -- this project's geometry bugs have repeatedly passed every
    assertion and been caught only by looking at the render (see CLAUDE.md's
    decisions on the demo's sideways head and the qualitative panel).

    Each panel shows the case's own T1 slice in grey with the atlas
    parcellation's BOUNDARIES overlaid (never filled -- a filled
    parcellation hides the anatomy it is meant to align with).

    Args:
        atlas: A loaded `Atlas`.
        case_ids: Cases to render, one row each.
        preprocessed_dir: Root of the preprocessed BraTS tree.
        out_path: Destination `.png` path.
        n_slices: Axial slices per case, centred on the tumour centroid.

    Returns:
        `out_path`, as a `Path`.

    Raises:
        ValueError: If `case_ids` is empty.
    """
    preprocessed_dir = Path(preprocessed_dir)
    out_path = Path(out_path)
    case_ids = list(case_ids)
    if not case_ids:
        raise ValueError("qc_overlay_figure: case_ids is empty; nothing to render.")

    fig, axes = plt.subplots(
        len(case_ids), n_slices, figsize=(3.2 * n_slices, 3.2 * len(case_ids)), squeeze=False
    )

    for row, case_id in enumerate(case_ids):
        case_dir = preprocessed_dir / case_id
        meta = read_json(case_dir / "meta.json")

        image = np.load(case_dir / "image.npy", mmap_mode="r")
        t1 = np.asarray(image[0]).astype(np.float32)
        t1_full = uncrop(t1, meta["bbox"], meta["original_shape"])

        label = np.load(case_dir / "label.npy")
        tumour_full = uncrop(label > 0, meta["bbox"], meta["original_shape"])

        w_size = t1_full.shape[2]
        w_coords = np.flatnonzero(tumour_full.any(axis=(0, 1)))
        centroid_w = int(round(float(w_coords.mean()))) if w_coords.size else w_size // 2

        if n_slices > 1:
            spread = max(1, w_size // 10)
            offsets = np.linspace(-spread, spread, n_slices)
        else:
            offsets = [0.0]
        slice_indices = [int(np.clip(round(centroid_w + off), 0, w_size - 1)) for off in offsets]

        for col, w_index in enumerate(slice_indices):
            ax = axes[row][col]
            t1_slice = axial_display_slice(t1_full, w_index)
            parc_slice = axial_display_slice(atlas.parcellation, w_index)
            boundary = _label_boundary(parc_slice)

            lo, hi = _display_window(t1_slice)
            ax.imshow(t1_slice, cmap="gray", vmin=lo, vmax=hi, origin="upper")

            overlay = np.zeros((*boundary.shape, 4), dtype=np.float32)
            overlay[boundary] = (1.0, 0.9, 0.0, 1.0)  # opaque yellow, boundary voxels only
            ax.imshow(overlay, origin="upper")

            ax.set_title(f"{case_id}\nw={w_index}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"SRI24 atlas alignment QC -- {atlas.name} v{atlas.version}")
    fig.tight_layout()

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_report(report: AlignmentReport, out_dir: str | Path, cfg: Any) -> dict[str, Path]:
    """Writes every Phase 0 artifact this project can trace a claim back to.

    Args:
        report: An `AlignmentReport` from `run_checks`.
        out_dir: Destination directory, created if missing.
        cfg: The `anatomy` config node (attribute access to `version`,
            `source`, `parcellation.name`, and the `validation.*`
            thresholds) -- i.e. `cfg.anatomy` of the full composed config.

    Returns:
        Dict mapping a short artifact name (`"report"`, `"per_case_dice"`,
        `"laterality_pairs"`, `"lobe_distribution"`, `"summary"`) to its
        written `Path`.
    """
    out_dir = ensure_dir(out_dir)
    validation_cfg = cfg.validation

    report_payload = {
        "passed": report.passed,
        "checks": [asdict(check) for check in report.checks],
        "atlas": {
            "name": str(cfg.parcellation.name),
            "version": str(cfg.version),
            "source": str(cfg.source),
        },
        "thresholds": {
            "min_brain_dice": float(validation_cfg.min_brain_dice),
            "min_laterality_pairs_correct": float(validation_cfg.min_laterality_pairs_correct),
            "midline_index": float(validation_cfg.midline_index),
            "max_midline_deviation": float(validation_cfg.max_midline_deviation),
        },
        "case_counts": {
            "n_brain_cases_scored": int(len(report.per_case_dice)),
            "n_laterality_pairs": int(len(report.laterality_pairs)),
            "n_lobe_cases_attributed": int(report.lobe_distribution["n_cases"].sum()),
        },
    }

    paths: dict[str, Path] = {}

    report_path = out_dir / _REPORT_JSON
    write_json(report_payload, report_path)
    paths["report"] = report_path

    per_case_path = out_dir / _PER_CASE_DICE_CSV
    report.per_case_dice.to_csv(per_case_path, index=False)
    paths["per_case_dice"] = per_case_path

    laterality_path = out_dir / _LATERALITY_CSV
    report.laterality_pairs.to_csv(laterality_path, index=False)
    paths["laterality_pairs"] = laterality_path

    lobe_path = out_dir / _LOBE_DISTRIBUTION_CSV
    report.lobe_distribution.to_csv(lobe_path, index=False)
    paths["lobe_distribution"] = lobe_path

    summary_path = out_dir / _SUMMARY_TXT
    summary_path.write_text(report.summary(), encoding="utf-8")
    paths["summary"] = summary_path

    return paths


def run_validation(cfg: DictConfig) -> AlignmentReport:
    """Loads the atlas, samples cases, runs every check, and writes the report.

    Never raises because a gating check FAILED -- a failed gate must still
    write its artifacts so the failure is diagnosable. It can still raise
    for a genuine setup problem (a missing atlas file, an empty preprocessed
    directory, a lobe map missing a structure) -- those come from
    `load_atlas` / `sample_cases` / `run_checks` and are not swallowed here.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The `AlignmentReport`. `report.passed` reflects only the GATING
        checks.
    """
    anatomy_cfg = cfg.anatomy
    validation_cfg = anatomy_cfg.validation
    preprocessed_dir = Path(cfg.data.preprocessing.out_dir)

    atlas = load_atlas(anatomy_cfg)
    lobe_map = load_lobe_map(validation_cfg.lobe_map)
    case_sample = sample_cases(validation_cfg, preprocessed_dir)

    logger.info(
        "run_validation: sampled %d case(s) (%d for the brain-mask check) from %s.",
        len(case_sample.case_ids),
        len(case_sample.brain_case_ids),
        preprocessed_dir,
    )

    brain_masks = brain_mask_iter(case_sample.brain_case_ids, preprocessed_dir)
    tumour_masks = tumour_mask_iter(case_sample.case_ids, preprocessed_dir)
    report = run_checks(atlas, brain_masks, tumour_masks, validation_cfg, lobe_map)

    out_dir = Path(cfg.output_dir)
    write_report(report, out_dir, anatomy_cfg)

    qc_case_ids = case_sample.case_ids[: int(validation_cfg.qc_cases)]
    qc_overlay_figure(atlas, qc_case_ids, preprocessed_dir, out_dir / _QC_PNG)

    logger.info("run_validation: %s", report.summary())
    return report


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the Phase 0 atlas-alignment gate, per the composed config.

    Exits non-zero (`SystemExit(1)`) when a GATING check failed, so a shell
    `&&` chain or a CI job can tell. The advisory `lobe_distribution` check
    never affects the exit code.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    report = run_validation(cfg)

    print(report.summary())

    out_dir = Path(cfg.output_dir)
    print("Artifacts written:")
    for name in (
        _REPORT_JSON,
        _PER_CASE_DICE_CSV,
        _LATERALITY_CSV,
        _LOBE_DISTRIBUTION_CSV,
        _SUMMARY_TXT,
        _QC_PNG,
    ):
        print(f"  {out_dir / name}")

    if not report.passed:
        failing = ", ".join(check.name for check in report.failures())
        print(
            f"GATE FAILED: gating check(s) failed: {failing}. The advisory lobe_distribution "
            "check never affects this exit code."
        )
        raise SystemExit(1)

    print("GATE PASSED (the advisory lobe_distribution check never affects the exit code).")


if __name__ == "__main__":
    main()
