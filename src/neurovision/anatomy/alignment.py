"""Phase 0's gate: proves the SRI24 atlas actually lands on our BraTS cases.

Three checks, of very different strength, measured against the real atlas in
`docs/research/phase0_atlas_findings.md`:

    1. Brain-mask Dice (`brain_mask_check`) -- the widest-margin check
       (0.9394 correct vs 0.7334 A-P mirrored), and GATING. But it is
       structurally blind to a left-right flip: a brain is nearly
       left-right symmetric, so an L-R mirrored atlas scores 0.9416 --
       *higher* than the correct orientation.
    2. Laterality from `_L`/`_R` structure-pair centroids
       (`laterality_check`) -- the ONLY check that can see a left-right
       flip (0/56 violations correct, 56/56 violations mirrored), and
       GATING for exactly that reason.
    3. Population lobe distribution (`lobe_distribution_check`) --
       ADVISORY only, never gating. Its natural summary statistic (rank
       correlation) scores a MIRRORED atlas higher than the correct one
       (+0.975 vs +0.872), so it must never be used as a pass/fail gate;
       it is scored here by mean absolute deviation in percentage points
       instead, and reported for its epidemiological plausibility value
       only.

This module is pure array + text arithmetic: no model, no checkpoint, and no
dependency on the deep-learning stack, so it (and anything that imports it)
stays importable in an environment with none of that installed -- see
`tests/test_alignment.py::test_alignment_module_does_not_import_torch`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from neurovision.anatomy.atlas import Atlas

__all__ = [
    "CheckResult",
    "AlignmentReport",
    "atlas_brain_mask",
    "dice",
    "uncrop",
    "brain_mask_check",
    "laterality_check",
    "load_lobe_map",
    "lobe_distribution_check",
    "run_checks",
]

logger = logging.getLogger(__name__)

# Larjavaara et al., Neuro-Oncology 2007 -- population lobe distribution of
# supratentorial glioma, used as the reference for the advisory lobe check.
DEFAULT_REFERENCE_PCT: dict[str, float] = {
    "frontal": 40.0,
    "temporal": 29.0,
    "parietal": 14.0,
    "deep": 14.0,
    "occipital": 3.0,
}

_MIN_STRUCTURE_VOXELS = 50


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckResult:
    """One pass/fail (or report-only) check result.

    Attributes:
        name: Short identifier, e.g. `"brain_mask_dice"`.
        gating: Whether this check can fail `AlignmentReport.passed`.
        passed: Whether the check passed. Always `True` for an advisory
            check -- it reports, it does not gate.
        value: The measured statistic.
        threshold: The threshold `value` was compared against, or `None`
            for a check with no single threshold (e.g. an advisory check).
        detail: Human-readable detail, meant to make a failure diagnosable
            without re-running anything.
    """

    name: str
    gating: bool
    passed: bool
    value: float
    threshold: float | None
    detail: str


@dataclass(frozen=True)
class AlignmentReport:
    """The full Phase 0 alignment report: gating checks plus advisory ones.

    Attributes:
        checks: Every `CheckResult`, gating and advisory.
        per_case_dice: Columns `case_id`, `dice` -- one row per case scored
            by `brain_mask_check`.
        laterality_pairs: Columns `base`, `centroid_l`, `centroid_r`, `ok`
            -- one row per `_L`/`_R` structure pair scored by
            `laterality_check`.
        lobe_distribution: Columns `lobe`, `n_cases`, `pct`,
            `reference_pct`, `abs_deviation_pp` -- one row per lobe scored
            by `lobe_distribution_check`.
    """

    checks: tuple[CheckResult, ...]
    per_case_dice: pd.DataFrame
    laterality_pairs: pd.DataFrame
    lobe_distribution: pd.DataFrame

    @property
    def passed(self) -> bool:
        """Whether every GATING check passed. Advisory checks are ignored."""
        return all(c.passed for c in self.checks if c.gating)

    def failures(self) -> tuple[CheckResult, ...]:
        """The gating checks that failed, in `self.checks` order."""
        return tuple(c for c in self.checks if c.gating and not c.passed)

    def summary(self) -> str:
        """A short human-readable table: gating checks first, each marked PASS/FAIL."""
        gating = [c for c in self.checks if c.gating]
        advisory = [c for c in self.checks if not c.gating]
        lines = ["Alignment report:"]
        for check in (*gating, *advisory):
            status = "PASS" if check.passed else "FAIL"
            kind = "gating" if check.gating else "advisory"
            threshold_str = (
                f", threshold={check.threshold:.4f}" if check.threshold is not None else ""
            )
            lines.append(
                f"  [{status}] ({kind}) {check.name}: value={check.value:.4f}"
                f"{threshold_str} -- {check.detail}"
            )
        overall = "PASS" if self.passed else "FAIL"
        lines.append(f"Overall: {overall} (gating checks only; advisory checks are informational)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Small array primitives
# --------------------------------------------------------------------------- #


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient between two boolean masks.

    Args:
        a: `(D, H, W)` array, cast to boolean.
        b: `(D, H, W)` array, cast to boolean, same shape as `a`.

    Returns:
        `2 * |a & b| / (|a| + |b|)`. `float("nan")` when both masks are
        empty -- never `1.0` and never a raise, since "both empty" is not a
        measurement of agreement.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_sum = float(a.sum())
    b_sum = float(b.sum())
    if a_sum == 0.0 and b_sum == 0.0:
        return float("nan")
    intersection = float(np.logical_and(a, b).sum())
    return 2.0 * intersection / (a_sum + b_sum)


def uncrop(
    array: np.ndarray,
    bbox: Sequence[Sequence[int]],
    original_shape: Sequence[int],
) -> np.ndarray:
    """Places a cropped array back into its original, uncropped geometry.

    Args:
        array: `(D, H, W)` cropped array.
        bbox: Three `[start, stop]` pairs, one per axis (the same
            convention as `meta.json`'s `bbox`).
        original_shape: The `(D, H, W)` shape of the uncropped volume.

    Returns:
        An `original_shape` array of `array`'s dtype, zero-filled outside
        `bbox`, holding `array` inside it.

    Raises:
        ValueError: If `array.shape` does not equal the extents implied by
            `bbox` -- meaning `array` and `bbox` came from different
            preprocessing runs. Same reasoning as
            `inference.postprocess.uncrop_to_original`.
    """
    bbox_int = tuple((int(start), int(stop)) for start, stop in bbox)
    extents = tuple(stop - start for start, stop in bbox_int)
    if tuple(array.shape) != extents:
        raise ValueError(
            f"uncrop: array shape {tuple(array.shape)} does not match the extents implied "
            f"by bbox {bbox_int} ({extents}). This means the array and the bbox came from "
            "different preprocessing runs."
        )
    out = np.zeros(tuple(int(s) for s in original_shape), dtype=array.dtype)
    slices = tuple(slice(start, stop) for start, stop in bbox_int)
    out[slices] = array
    return out


# --------------------------------------------------------------------------- #
# Check 1 -- brain-mask Dice (gating, widest margin, blind to L-R flip)
# --------------------------------------------------------------------------- #


def atlas_brain_mask(atlas: Atlas, source: str = "tissue") -> np.ndarray:
    """The atlas's brain mask.

    Deliberately offers no `"parcellation"` option. AAL/TZO parcellates
    grey matter only, so `atlas.parcellation > 0` covers a fraction of the
    true brain extent (measured 1,053,253 of 1,451,706 voxels, Dice 0.8013
    against a real brain mask) -- a number that reads as a failed gate and
    is nothing of the kind. `atlas.tissue > 0` and `spgr.nii > 0` are the
    same 1,451,706 voxels exactly.

    Args:
        atlas: A loaded `Atlas`.
        source: Must be `"tissue"`.

    Returns:
        `(D, H, W)` boolean array.

    Raises:
        ValueError: If `source` is not `"tissue"`, or if `source ==
            "tissue"` but `atlas.tissue is None`.
    """
    if source != "tissue":
        raise ValueError(
            f"atlas_brain_mask: unsupported source '{source}'; only 'tissue' is supported. "
            "The parcellation is NOT a valid brain mask -- AAL parcellates grey matter only "
            "and covers a fraction of true brain extent (measured Dice 0.80 against a real "
            "brain mask); see docs/research/phase0_atlas_findings.md."
        )
    if atlas.tissue is None:
        raise ValueError(
            "atlas_brain_mask: atlas.tissue is None -- no tissue map was loaded, so the "
            "brain mask (tissue > 0) cannot be built."
        )
    return atlas.tissue > 0


def brain_mask_check(
    atlas: Atlas,
    case_brain_masks: Iterable[tuple[str, np.ndarray]],
    *,
    min_dice: float,
    source: str = "tissue",
) -> tuple[CheckResult, pd.DataFrame]:
    """Dice of the atlas brain mask against each case's brain mask. GATING.

    Args:
        atlas: A loaded `Atlas`.
        case_brain_masks: `(case_id, brain_mask)` pairs, `brain_mask` a
            boolean `(D, H, W)` array in the atlas's own (original BraTS)
            geometry.
        min_dice: The gate: `value >= min_dice` passes.
        source: Passed to `atlas_brain_mask`.

    Returns:
        `(CheckResult, DataFrame[case_id, dice])`. `CheckResult.value` is
        the MEDIAN Dice across cases (robust to a few odd cases);
        `detail` additionally reports min, max, and the count of cases
        below `min_dice`, so a bad tail is not hidden by the median.

    Raises:
        ValueError: If `case_brain_masks` is empty.
    """
    atlas_mask = atlas_brain_mask(atlas, source=source)

    rows = [
        {"case_id": case_id, "dice": dice(atlas_mask, mask)} for case_id, mask in case_brain_masks
    ]
    if not rows:
        raise ValueError(
            "brain_mask_check: case_brain_masks is empty; need at least one case to compute "
            "the median Dice."
        )
    per_case = pd.DataFrame(rows, columns=["case_id", "dice"])

    values = per_case["dice"].to_numpy(dtype=float)
    median_dice = float(np.nanmedian(values))
    n_below = int(np.sum(values < min_dice))
    detail = (
        f"brain-mask Dice vs atlas ({source}): median={median_dice:.4f}, "
        f"min={float(np.nanmin(values)):.4f}, max={float(np.nanmax(values)):.4f} over "
        f"{len(values)} case(s); {n_below} case(s) below threshold {min_dice:.4f}."
    )
    check = CheckResult(
        name="brain_mask_dice",
        gating=True,
        passed=median_dice >= min_dice,
        value=median_dice,
        threshold=float(min_dice),
        detail=detail,
    )
    return check, per_case


# --------------------------------------------------------------------------- #
# Check 2 -- laterality from _L/_R centroids (gating, sees an L-R flip)
# --------------------------------------------------------------------------- #


def laterality_check(
    atlas: Atlas,
    *,
    midline_index: float,
    min_fraction_correct: float,
    max_midline_deviation: float,
) -> tuple[tuple[CheckResult, CheckResult], pd.DataFrame]:
    """Proves laterality from `_L`/`_R` structure-pair centroids. GATING.

    Under the BraTS convention axis 0 runs right -> left, so LOW index =
    patient RIGHT: a correctly oriented pair has `centroid_l > midline_index`
    and `centroid_r < midline_index`. This is the only check in the project
    that can see a left-right flip -- brain-mask Dice cannot, because a
    brain is nearly left-right symmetric.

    Structures with fewer than 50 voxels are skipped (too small for a
    stable centroid).

    Args:
        atlas: A loaded `Atlas`.
        midline_index: The assumed axis-0 midline (voxel index).
        min_fraction_correct: The gate for the `laterality_pairs` check:
            fraction of pairs satisfying the orientation rule.
        max_midline_deviation: The gate for the `midline_estimate` check:
            max allowed `|independent midline estimate - midline_index|`.

    Returns:
        `((laterality_pairs_check, midline_estimate_check), DataFrame[base,
        centroid_l, centroid_r, ok])`. Both checks are gating.

    Raises:
        ValueError: If the atlas has no `_L`/`_R` structure pairs at all --
            meaning this parcellation is not one this check understands.
            Reporting "0 of 0 correct, 100%" would pass the gate on no
            evidence, which is worse than raising.
    """
    structures_by_name = {s.name: s for s in atlas.labels.structures}

    pairs = []
    for structure in atlas.labels.structures:
        if structure.laterality != "L":
            continue
        base = structure.name[:-2]
        r_name = f"{base}_R"
        if r_name in structures_by_name:
            pairs.append((base, structure, structures_by_name[r_name]))

    if not pairs:
        raise ValueError(
            "laterality_check: no _L/_R structure pairs found in atlas.labels; this "
            "parcellation is not one this check understands, and reporting '0 of 0 correct' "
            "would pass the gate on no evidence."
        )

    rows = []
    for base, l_struct, r_struct in pairs:
        l_mask = atlas.structure_mask(l_struct.name)
        r_mask = atlas.structure_mask(r_struct.name)
        if int(l_mask.sum()) < _MIN_STRUCTURE_VOXELS or int(r_mask.sum()) < _MIN_STRUCTURE_VOXELS:
            continue
        centroid_l = float(np.argwhere(l_mask)[:, 0].mean())
        centroid_r = float(np.argwhere(r_mask)[:, 0].mean())
        ok = bool(centroid_l > midline_index and centroid_r < midline_index)
        rows.append({"base": base, "centroid_l": centroid_l, "centroid_r": centroid_r, "ok": ok})

    if not rows:
        raise ValueError(
            f"laterality_check: every _L/_R pair had fewer than {_MIN_STRUCTURE_VOXELS} "
            "voxels on at least one side; nothing to check."
        )

    laterality_pairs = pd.DataFrame(rows, columns=["base", "centroid_l", "centroid_r", "ok"])

    n_total = len(laterality_pairs)
    n_ok = int(laterality_pairs["ok"].sum())
    n_bad = n_total - n_ok
    fraction_ok = n_ok / n_total

    if n_bad == 0:
        pairs_detail = (
            f"{n_ok}/{n_total} _L/_R pairs satisfy (centroid_L > {midline_index}) and "
            f"(centroid_R < {midline_index})."
        )
    else:
        bad_rows = laterality_pairs[~laterality_pairs["ok"]].head(3)
        # A partial failure is a different diagnosis from a total one (a total
        # failure is a mirrored atlas; a couple of failures is a near-midline
        # structure) -- name up to three violating pairs so the two are
        # distinguishable without re-running anything.
        named = ", ".join(
            f"{r.base} (L={r.centroid_l:.1f}, R={r.centroid_r:.1f})" for r in bad_rows.itertuples()
        )
        pairs_detail = (
            f"{n_ok}/{n_total} _L/_R pairs satisfy (centroid_L > {midline_index}) and "
            f"(centroid_R < {midline_index}); {n_bad} violate, e.g.: {named}."
        )

    pairs_check = CheckResult(
        name="laterality_pairs",
        gating=True,
        passed=fraction_ok >= min_fraction_correct,
        value=fraction_ok,
        threshold=float(min_fraction_correct),
        detail=pairs_detail,
    )

    all_centroids = np.concatenate(
        [
            laterality_pairs["centroid_l"].to_numpy(dtype=float),
            laterality_pairs["centroid_r"].to_numpy(dtype=float),
        ]
    )
    midline_value = float(all_centroids.mean())
    deviation = abs(midline_value - midline_index)
    midline_detail = (
        f"independent midline estimate from {n_total} _L/_R pair centroid(s) = "
        f"{midline_value:.2f}, vs assumed midline_index={midline_index}: "
        f"deviation={deviation:.2f} (max allowed {max_midline_deviation})."
    )
    midline_check = CheckResult(
        name="midline_estimate",
        gating=True,
        passed=deviation <= max_midline_deviation,
        value=midline_value,
        threshold=float(max_midline_deviation),
        detail=midline_detail,
    )

    return (pairs_check, midline_check), laterality_pairs


# --------------------------------------------------------------------------- #
# Check 3 -- population lobe distribution (advisory only, never gates)
# --------------------------------------------------------------------------- #


def load_lobe_map(path: str | Path) -> dict[str, dict[str, Any]]:
    """Loads and validates the AAL-structure -> lobe compilation.

    Args:
        path: Path to a YAML file shaped like `knowledge/aal_lobes.yaml`
            (top-level `lobes`, `epidemiology_lobes`, `structures`).

    Returns:
        `data["structures"]`: base structure name -> `{"lobe": ..., "
        epidemiology_lobe": ..., ...}`.

    Raises:
        ValueError: If any entry is missing `lobe` or `epidemiology_lobe`,
            or names a value outside the file's own declared vocabularies.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    valid_lobes = set(data["lobes"])
    valid_epi_lobes = set(data["epidemiology_lobes"])
    structures: dict[str, dict[str, Any]] = data["structures"]

    for name, entry in structures.items():
        if "lobe" not in entry or "epidemiology_lobe" not in entry:
            raise ValueError(
                f"load_lobe_map: structure '{name}' is missing 'lobe' or 'epidemiology_lobe'."
            )
        if entry["lobe"] not in valid_lobes:
            raise ValueError(
                f"load_lobe_map: structure '{name}' has lobe '{entry['lobe']}' outside the "
                f"declared vocabulary {sorted(valid_lobes)}."
            )
        if entry["epidemiology_lobe"] not in valid_epi_lobes:
            raise ValueError(
                f"load_lobe_map: structure '{name}' has epidemiology_lobe "
                f"'{entry['epidemiology_lobe']}' outside the declared vocabulary "
                f"{sorted(valid_epi_lobes)}."
            )

    return structures


def _base_name(atlas: Atlas, structure_name: str) -> str:
    """The lobe-map key for a structure: its name with any `_L`/`_R` suffix stripped."""
    structure = atlas.labels.by_name(structure_name)
    if structure.laterality in ("L", "R"):
        return structure_name[:-2]
    return structure_name


def lobe_distribution_check(
    atlas: Atlas,
    lobe_map: dict[str, dict[str, Any]],
    case_tumour_masks: Iterable[tuple[str, np.ndarray]],
    *,
    reference_pct: dict[str, float] | None = None,
) -> tuple[CheckResult, pd.DataFrame]:
    """Population lobe distribution vs published epidemiology. ADVISORY, never gates.

    For each case: intersects the tumour mask with the parcellation,
    excludes background, unmapped structures, and any lobe whose
    `epidemiology_lobe` is `"excluded"` (the infratentorial structures --
    the reference figures are for supratentorial glioma), and attributes
    the case to the `epidemiology_lobe` holding the largest remaining
    voxel share.

    Scored by MEAN ABSOLUTE DEVIATION in percentage points against
    `reference_pct`, never by rank correlation: measured on the real atlas,
    Spearman rank correlation scores a left-right MIRRORED atlas higher
    (+0.975) than the correctly oriented one (+0.872) -- see
    `docs/research/phase0_atlas_findings.md` Finding K. A gate built on
    that statistic would prefer the wrong orientation, which is why this
    check reports a deviation instead and never gates at all.

    Args:
        atlas: A loaded `Atlas`.
        lobe_map: From `load_lobe_map`, keyed on BASE structure name (any
            `_L`/`_R` suffix stripped).
        case_tumour_masks: `(case_id, tumour_mask)` pairs, `tumour_mask` a
            boolean `(D, H, W)` array in the atlas's own geometry.
        reference_pct: Epidemiology reference, lobe -> percent. Defaults to
            `DEFAULT_REFERENCE_PCT` (Larjavaara et al., Neuro-Oncology
            2007).

    Returns:
        `(CheckResult, DataFrame[lobe, n_cases, pct, reference_pct,
        abs_deviation_pp])`. `CheckResult.gating` is `False` and
        `CheckResult.passed` is unconditionally `True`.

    Raises:
        ValueError: If a structure overlapping a tumour mask (and not the
            atlas's own "unmapped" placeholder) has no entry in `lobe_map`
            -- silently dropping it would be exactly the kind of guess this
            check exists to avoid.
    """
    if reference_pct is None:
        reference_pct = DEFAULT_REFERENCE_PCT

    unmapped_name = atlas.labels.unmapped_name
    tally: dict[str, int] = {}
    n_total_cases = 0
    n_attributed_cases = 0

    for _case_id, mask in case_tumour_masks:
        n_total_cases += 1
        mask = np.asarray(mask, dtype=bool)
        overlap = mask & (atlas.parcellation != 0)
        if not overlap.any():
            continue

        label_ids, counts = np.unique(atlas.parcellation[overlap], return_counts=True)
        lobe_counts: dict[str, int] = {}
        for label_id, count in zip(label_ids.tolist(), counts.tolist(), strict=True):
            name = atlas.labels.name_for_id(int(label_id))
            if name in ("", unmapped_name):
                continue
            base = _base_name(atlas, name)
            if base not in lobe_map:
                raise ValueError(
                    f"lobe_distribution_check: structure '{base}' (from raw label "
                    f"'{name}') has no entry in lobe_map."
                )
            epi_lobe = lobe_map[base]["epidemiology_lobe"]
            if epi_lobe == "excluded":
                continue
            lobe_counts[epi_lobe] = lobe_counts.get(epi_lobe, 0) + int(count)

        if not lobe_counts:
            continue

        dominant_lobe = max(lobe_counts, key=lobe_counts.__getitem__)
        tally[dominant_lobe] = tally.get(dominant_lobe, 0) + 1
        n_attributed_cases += 1

    rows: list[dict[str, Any]] = []
    if n_attributed_cases == 0:
        for lobe, reference in reference_pct.items():
            rows.append(
                {
                    "lobe": lobe,
                    "n_cases": 0,
                    "pct": float("nan"),
                    "reference_pct": float(reference),
                    "abs_deviation_pp": float("nan"),
                }
            )
        value = float("nan")
    else:
        deviations = []
        for lobe, reference in reference_pct.items():
            n_cases = tally.get(lobe, 0)
            pct = 100.0 * n_cases / n_attributed_cases
            abs_deviation_pp = abs(pct - reference)
            deviations.append(abs_deviation_pp)
            rows.append(
                {
                    "lobe": lobe,
                    "n_cases": n_cases,
                    "pct": pct,
                    "reference_pct": float(reference),
                    "abs_deviation_pp": abs_deviation_pp,
                }
            )
        value = float(np.mean(deviations))

    lobe_distribution = pd.DataFrame(
        rows, columns=["lobe", "n_cases", "pct", "reference_pct", "abs_deviation_pp"]
    )

    if math.isnan(value):
        per_lobe_str = "no case contributed any mapped, non-excluded voxels"
    else:
        per_lobe_str = "; ".join(
            f"{r['lobe']}={r['pct']:.1f}% (ref {r['reference_pct']:.0f}%, "
            f"delta {r['abs_deviation_pp']:.1f}pp)"
            for r in rows
        )
    detail = (
        "ADVISORY ONLY -- never gates (rank correlation on this check scores a MIRRORED "
        "atlas higher than the correct one, see docs/research/phase0_atlas_findings.md "
        f"Finding K). Attributed {n_attributed_cases}/{n_total_cases} case(s) a dominant "
        f"lobe. Mean absolute deviation = {value:.2f}pp. Per-lobe: {per_lobe_str}."
    )

    check = CheckResult(
        name="lobe_distribution",
        gating=False,
        passed=True,
        value=value,
        threshold=None,
        detail=detail,
    )
    return check, lobe_distribution


# --------------------------------------------------------------------------- #
# Everything together
# --------------------------------------------------------------------------- #


def run_checks(
    atlas: Atlas,
    case_brain_masks: Iterable[tuple[str, np.ndarray]],
    case_tumour_masks: Iterable[tuple[str, np.ndarray]],
    cfg: Any,
    lobe_map: dict[str, dict[str, Any]],
) -> AlignmentReport:
    """Runs all three checks and assembles the `AlignmentReport`.

    Args:
        atlas: A loaded `Atlas`.
        case_brain_masks: `(case_id, brain_mask)` pairs for
            `brain_mask_check`.
        case_tumour_masks: `(case_id, tumour_mask)` pairs for
            `lobe_distribution_check`.
        cfg: The `validation:` config node of `configs/anatomy/sri24.yaml`
            (attribute access to `min_brain_dice`, `brain_mask_source`,
            `min_laterality_pairs_correct`, `midline_index`,
            `max_midline_deviation`).
        lobe_map: From `load_lobe_map`.

    Returns:
        An `AlignmentReport` combining all three checks.
    """
    brain_check, per_case_dice = brain_mask_check(
        atlas,
        case_brain_masks,
        min_dice=float(cfg.min_brain_dice),
        source=str(cfg.brain_mask_source),
    )
    (pairs_check, midline_check), laterality_pairs = laterality_check(
        atlas,
        midline_index=float(cfg.midline_index),
        min_fraction_correct=float(cfg.min_laterality_pairs_correct),
        max_midline_deviation=float(cfg.max_midline_deviation),
    )
    lobe_check, lobe_distribution = lobe_distribution_check(atlas, lobe_map, case_tumour_masks)

    report = AlignmentReport(
        checks=(brain_check, pairs_check, midline_check, lobe_check),
        per_case_dice=per_case_dice,
        laterality_pairs=laterality_pairs,
        lobe_distribution=lobe_distribution,
    )
    logger.info("run_checks: %s", report.summary())
    return report
