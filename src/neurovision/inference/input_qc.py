"""Input QC gate: decide whether an uploaded study may be segmented at all.

Milestone 4, Phase E, tasks **E3** (the input QC gate) and **E4** (missing-
sequence refusal). E4 is not a separate module: "detect a missing sequence
and refuse with a named reason" IS one of E3's checks
(`check_sequence_completeness`). Splitting it out would put one refusal rule
in two places, which is how two copies of a rule drift apart -- see
`configs/clinical/default.yaml`'s `input_qc:` block, which says the same
thing.

**The binding principle.** Every check here is label-free BY CONSTRUCTION,
not by convention: no function in this module takes a label argument, full
stop -- `tests/test_input_qc.py::test_no_function_in_this_module_takes_a_label`
introspects every public callable's signature and enforces this
structurally. There is no ground truth at deployment time, and this project
has already shipped a calibration reporting mask that WAS defined using the
ground-truth label; it manufactured 41-57% of a reported ECE behind a fully
green test suite because the code did exactly what it said. A deployment
gate that consulted a label would be the same bug at higher stakes.

**Reading this module.** `describe_volume` turns one raw NIfTI array into a
small, label-free `VolumeInfo` summary -- built once, checked many times.
The `check_*` functions are individually public and individually tested;
`run_input_qc` composes them into one `InputQCReport`. That split is
deliberate: a gate whose composite verdict is the only testable thing is a
gate nobody can debug. Every check function returns a `Finding` (or a small
tuple of them) even when it PASSES, carrying `Severity.OK` and the measured
numbers -- a check that is silent when it passes cannot be audited, and
"which check did not run" is the question that matters when something goes
wrong. `CHECK_IDS` is the single, machine-checkable list of what a complete
report must contain (see `tests/test_input_qc.py`'s check-id coverage test).

**Why `VolumeInfo` carries summary statistics, not the raw array.** A QC
gate that re-reads full volumes on every check is needlessly heavy, and
threading raw arrays through every check function invites exactly the kind
of accidental voxel-wise coupling this module wants to avoid. `describe_volume`
computes everything once, while the array is still in hand -- including,
when a brain mask is supplied, the true voxel-wise intersection of that
modality's nonzero voxels with the mask (`VolumeInfo.n_nonzero_in_mask`) --
and every check works off `VolumeInfo`'s summary counts afterwards, never
the raw array again.

**Read before touching this file:**
- `app/backend/jobs.py` (`_validate_roles`, `_load_and_validate_nifti`,
  `_validate_consistent_geometry`, `_AFFINE_ATOL`) -- this module generalises
  those checks into a reusable, richer gate. Wiring it into the backend is a
  separate task; nothing in `jobs.py` is imported or modified here.
- `configs/clinical/default.yaml`'s `input_qc:` block -- every threshold
  this module reads comes from there, reachable at `cfg.clinical.input_qc`
  (NOT `cfg.input_qc`; see `test_config_block_is_reachable_at_the_composed_path`).
- `src/neurovision/data/dicom_ingest.py` -- `ROLES` is imported from there
  rather than re-declared, so the two modules cannot silently disagree about
  the modality vocabulary.

No CUDA, no torch: this module runs no model. numpy and nibabel only, both
already in `requirements.txt`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neurovision.data.dicom_ingest import ROLES

logger = logging.getLogger(__name__)

# The reserved key `load_volume_infos` treats as "this path is the brain
# mask, not a modality" -- kept out of the returned VolumeInfo mapping and
# returned as this function's second value instead.
_BRAIN_MASK_KEY = "brain_mask"

# A real affine's determinant is a voxel volume in mm^3 -- for the
# spacing_min_mm/spacing_max_mm range in configs/clinical/default.yaml
# (0.3..3.0mm) that is roughly 0.03..27. 1e-6 is comfortably below any real
# spacing yet catches a genuinely singular (rank-deficient) affine, which has
# determinant exactly 0. Not a clinical threshold, so -- unlike every number
# in configs/clinical/default.yaml's input_qc block -- it is a numerical
# tolerance local to this module, the same status as jobs.py's own
# `_AFFINE_ATOL`.
_SINGULAR_DET_ATOL = 1e-6

# Every check id this module can produce, and the single source of truth a
# "did every check run" test can check itself against (see
# tests/test_input_qc.py::test_every_check_emits_a_finding_even_when_it_passes).
# A gate whose completeness is only enforced by a hand-maintained second list
# is exactly the kind of drift CLAUDE.md's traps warn about.
CHECK_IDS: frozenset[str] = frozenset(
    {
        "sequence_completeness",
        "unexpected_role",
        "geometry_consistency",
        "finite_values",
        "affine_invertible",
        "spacing_range",
        "anisotropy",
        "expected_shape",
        "dynamic_range",
        "brain_volume",
        "nonzero_in_brain",
        "skull_present",
    }
)


class Severity(StrEnum):
    """How serious a `Finding` is. Ordered OK < WARN < REFUSE.

    A plain member index (not string comparison -- "REFUSE" < "WARN"
    alphabetically, which would silently invert the ordering) decides the
    worst-of relationship `InputQCReport.verdict` needs; see
    `_SEVERITY_RANK` below.
    """

    OK = "ok"
    WARN = "warn"
    REFUSE = "refuse"


# Explicit rank, never derived from string order -- see `Severity`'s
# docstring for why that matters.
_SEVERITY_RANK: dict[Severity, int] = {Severity.OK: 0, Severity.WARN: 1, Severity.REFUSE: 2}


def _worst_severity(severities: Iterable[Severity]) -> Severity:
    """The worst (highest-ranked) severity in `severities`, or OK if empty."""
    worst = Severity.OK
    for severity in severities:
        if _SEVERITY_RANK[severity] > _SEVERITY_RANK[worst]:
            worst = severity
    return worst


def _jsonable(value: Any) -> Any:
    """Recursively convert `value` to plain, `json.dumps`-safe Python types.

    A numpy `bool_` or `float32` surviving into a `Finding.detail` dict is
    exactly how a long pipeline dies at its very last line (CLAUDE.md's
    reporting-mask trap is a cousin of this one) -- so this is applied to
    every `detail` dict at `InputQCReport.to_dict()` time, not left to
    whichever check happened to remember to call `float(...)`.
    """
    if isinstance(value, Severity):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):  # a numpy scalar, e.g. np.float32, np.bool_
        return value.item()
    return value


@dataclass(frozen=True)
class Finding:
    """One check's verdict, always emitted -- even when the check passes.

    Attributes:
        check: Stable, machine-readable check id, e.g.
            `"sequence_completeness"`. One of `CHECK_IDS`.
        severity: How serious this finding is.
        message: Actionable prose, for a human operator.
        detail: The numbers behind the decision. Must contain only
            JSON-serialisable values once passed through `InputQCReport.to_dict`
            (plain floats/ints/strings/lists/dicts -- convert numpy scalars
            before building this).
    """

    check: str
    severity: Severity
    message: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class InputQCReport:
    """The composed outcome of `run_input_qc`.

    Attributes:
        verdict: The worst severity among `findings`.
        findings: Every check's `Finding`, in the order `run_input_qc`
            produced them.
    """

    verdict: Severity
    findings: tuple[Finding, ...]

    def refusals(self) -> tuple[Finding, ...]:
        """Findings with `severity is Severity.REFUSE`."""
        return tuple(f for f in self.findings if f.severity is Severity.REFUSE)

    def warnings(self) -> tuple[Finding, ...]:
        """Findings with `severity is Severity.WARN`."""
        return tuple(f for f in self.findings if f.severity is Severity.WARN)

    def to_dict(self) -> dict[str, Any]:
        """This report as a plain, JSON-serialisable dict.

        For the job manifest and the UI. Every value is a native Python
        type -- see `_jsonable`.
        """
        return {
            "verdict": self.verdict.value,
            "findings": [
                {
                    "check": finding.check,
                    "severity": finding.severity.value,
                    "message": finding.message,
                    "detail": _jsonable(finding.detail),
                }
                for finding in self.findings
            ],
        }


@dataclass(frozen=True)
class VolumeInfo:
    """One modality's label-free summary. Built once by `describe_volume`, checked many times.

    Attributes:
        role: The modality role this summarises (e.g. `"t1ce"`), or any
            other supplied key -- `describe_volume` does not itself require
            `role` to be one of `ROLES`.
        shape: `(D, H, W)`, the volume's voxel grid shape.
        affine: 4x4 float64 voxel-to-world affine, as read from the NIfTI
            header.
        spacing_mm: Voxel size in mm along each of the three axes, computed
            from the affine's column norms (`np.linalg.norm(affine[:3, i])`
            for `i` in `0, 1, 2`) -- NOT from a header field. The affine is
            the geometric ground truth; a header spacing field can silently
            disagree with it after a resample.
        n_nonzero: Count of voxels `!= 0`.
        n_nonzero_in_mask: Count of voxels that are BOTH `!= 0` in this
            modality AND nonzero in the brain mask supplied to
            `describe_volume` -- a true voxel-wise intersection, computed
            once while the raw array was in hand. `None` when no mask was
            supplied to `describe_volume` for this volume.
        n_voxels: Total voxel count (`shape` product).
        n_nonfinite: Count of voxels that are NaN or +/-Inf.
        p005: 0.5th percentile, computed over the FINITE, NONZERO voxels
            only. `0.0` if there are none (see `describe_volume`).
        p995: 99.5th percentile, same population as `p005`.
        robust_scale: The median absolute deviation (MAD) of the finite,
            nonzero voxels -- `median(|x - median(x)|)`. `0.0` if there are
            no finite, nonzero voxels, or if they are all equal. See
            `describe_volume`'s docstring for why MAD was chosen over, say,
            `p995` itself.
    """

    role: str
    shape: tuple[int, int, int]
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]
    n_nonzero: int
    n_nonzero_in_mask: int | None
    n_voxels: int
    n_nonfinite: int
    p005: float
    p995: float
    robust_scale: float


def describe_volume(
    role: str,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    brain_mask: np.ndarray | None = None,
) -> VolumeInfo:
    """Summarise one modality's raw array into a label-free `VolumeInfo`.

    Percentiles and the robust scale are computed over NONZERO voxels only:
    a background-dominated brain volume's 0.5th percentile over ALL voxels
    is otherwise always 0 (background dominates by voxel count), which would
    make the dynamic-range check meaningless. NaN/Inf voxels are excluded
    from that same population (they would otherwise poison the percentile
    computation or a MAD built on top of it) but are still counted, via
    `n_nonfinite` -- `check_finite_values` is what acts on that count.

    `robust_scale` is the median absolute deviation (MAD) of the finite,
    nonzero voxels, chosen over an alternative like `p995` itself for three
    reasons: it is scale-EQUIVARIANT (`MAD(k * x) == abs(k) * MAD(x)` for any
    real `k`), so a ratio built on it is scale-invariant regardless of
    whether a volume is stored in raw scanner units or already z-scored; it
    is robust to a single outlier voxel in a way a raw percentile is not;
    and it is exactly `0.0` for a constant (or empty) volume, which is
    precisely the "no dynamic range" case `check_intensity_sanity` needs to
    catch without a separate branch.

    When `brain_mask` is supplied, this also computes the TRUE voxel-wise
    intersection of `data`'s nonzero voxels with the mask's nonzero voxels,
    while `data` is still in hand, and stores the count as
    `VolumeInfo.n_nonzero_in_mask` -- this is exactly the quantity
    `check_brain_mask`'s `nonzero_in_brain` check reports on.

    Args:
        role: A label for this volume (e.g. `"t1ce"`). Not required to be
            one of `ROLES` -- `check_geometry_consistency` handles an
            unrecognised role as its own (WARN) finding.
        data: The raw voxel array, `(D, H, W)`.
        affine: The 4x4 voxel-to-world affine.
        brain_mask: The brain mask array, `(D, H, W)`, or `None` if
            unavailable. Must have the same shape as `data`.

    Returns:
        The summary. See `VolumeInfo`'s docstring for each field.

    Raises:
        ValueError: If `brain_mask` is supplied and its shape does not
            match `data`'s shape. Names both shapes.
    """
    data = np.asarray(data)
    affine = np.asarray(affine, dtype=np.float64)
    shape = tuple(int(s) for s in data.shape)
    spacing_mm = tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))

    finite_mask = np.isfinite(data)
    n_nonfinite = int(data.size - int(np.count_nonzero(finite_mask)))
    nonzero_mask = data != 0
    n_nonzero = int(np.count_nonzero(nonzero_mask))
    n_voxels = int(data.size)

    n_nonzero_in_mask: int | None = None
    if brain_mask is not None:
        brain_mask = np.asarray(brain_mask)
        if brain_mask.shape != data.shape:
            raise ValueError(
                f"describe_volume: brain_mask shape {brain_mask.shape!r} does not match "
                f"data shape {data.shape!r} for role {role!r}"
            )
        n_nonzero_in_mask = int(np.count_nonzero(nonzero_mask & brain_mask.astype(bool)))

    valid = data[finite_mask & nonzero_mask]
    if valid.size == 0:
        p005 = 0.0
        p995 = 0.0
        robust_scale = 0.0
    else:
        p005 = float(np.percentile(valid, 0.5))
        p995 = float(np.percentile(valid, 99.5))
        median = np.median(valid)
        robust_scale = float(np.median(np.abs(valid - median)))

    return VolumeInfo(
        role=role,
        shape=shape,  # type: ignore[arg-type]
        affine=affine,
        spacing_mm=spacing_mm,  # type: ignore[arg-type]
        n_nonzero=n_nonzero,
        n_nonzero_in_mask=n_nonzero_in_mask,
        n_voxels=n_voxels,
        n_nonfinite=n_nonfinite,
        p005=p005,
        p995=p995,
        robust_scale=robust_scale,
    )


def _reference_role(volumes: Mapping[str, VolumeInfo]) -> str | None:
    """The deterministic reference role for geometry comparisons.

    Always the first entry of `ROLES` that is present in `volumes` --
    NEVER `next(iter(volumes))`, which would follow the mapping's insertion
    order and make the "reference" depend on the order the caller happened
    to build the dict in. Falls back to the alphabetically-first key when
    `volumes` holds only roles outside `ROLES` (e.g. only an unrecognised
    modality was supplied), so this always returns a role when `volumes` is
    non-empty.

    Args:
        volumes: role -> `VolumeInfo`.

    Returns:
        The reference role, or `None` if `volumes` is empty.
    """
    for role in ROLES:
        if role in volumes:
            return role
    if volumes:
        return sorted(volumes)[0]
    return None


def check_sequence_completeness(
    present_roles: Iterable[str], required_roles: Iterable[str]
) -> Finding:
    """REFUSE if any `required_roles` entry is missing from `present_roles`.

    This check id IS Milestone 4's E4 (missing-sequence refusal) -- see the
    module docstring for why that is not a separate function.

    Args:
        present_roles: Roles actually supplied (e.g. `volumes.keys()`).
        required_roles: Roles that must be present, from
            `cfg.clinical.input_qc.required_roles`.

    Returns:
        A `"sequence_completeness"` `Finding`. REFUSE, naming every missing
        role, if any required role is absent; OK otherwise.
    """
    present = set(present_roles)
    required = set(required_roles)
    missing = sorted(required - present)

    detail = {"required_roles": sorted(required), "present_roles": sorted(present)}
    if missing:
        detail["missing_roles"] = missing
        return Finding(
            check="sequence_completeness",
            severity=Severity.REFUSE,
            message=f"Missing required sequence(s): {', '.join(missing)}.",
            detail=detail,
        )
    return Finding(
        check="sequence_completeness",
        severity=Severity.OK,
        message="All required sequences are present.",
        detail=detail,
    )


def check_geometry_consistency(
    volumes: Mapping[str, VolumeInfo], affine_atol: float
) -> tuple[Finding, ...]:
    """Cross-volume geometry checks: unrecognised roles, singular affines, shape/affine agreement.

    Grouped into one function because all three need exactly one pass over
    `volumes`: whether each supplied role is one this project knows about
    (`ROLES`), whether each modality's own affine is invertible, and whether
    every modality agrees with a single deterministic reference
    (`_reference_role`) on shape and affine. Always returns exactly three
    findings, in this order: `unexpected_role`, `affine_invertible`,
    `geometry_consistency`.

    Args:
        volumes: role -> `VolumeInfo`. May be empty, or hold exactly one
            entry -- both handled without raising.
        affine_atol: Absolute tolerance (mm) for affine agreement with the
            reference, from `cfg.clinical.input_qc.affine_atol`.

    Returns:
        `(unexpected_role, affine_invertible, geometry_consistency)`.
    """
    # -- unexpected_role: WARN only -----------------------------------
    unexpected = sorted(role for role in volumes if role not in ROLES)
    if unexpected:
        unexpected_finding = Finding(
            check="unexpected_role",
            severity=Severity.WARN,
            message=f"Supplied role(s) not recognised: {', '.join(unexpected)}.",
            detail={"unexpected_roles": unexpected, "known_roles": list(ROLES)},
        )
    else:
        unexpected_finding = Finding(
            check="unexpected_role",
            severity=Severity.OK,
            message="All supplied roles are recognised.",
            detail={"known_roles": list(ROLES)},
        )

    # -- affine_invertible: REFUSE -------------------------------------
    determinants = {role: float(np.linalg.det(vi.affine)) for role, vi in volumes.items()}
    singular = sorted(role for role, det in determinants.items() if abs(det) < _SINGULAR_DET_ATOL)
    if singular:
        affine_finding = Finding(
            check="affine_invertible",
            severity=Severity.REFUSE,
            message=f"Singular (non-invertible) affine for: {', '.join(singular)}.",
            detail={"determinants": determinants},
        )
    else:
        affine_finding = Finding(
            check="affine_invertible",
            severity=Severity.OK,
            message="All affines are invertible.",
            detail={"determinants": determinants},
        )

    # -- geometry_consistency: REFUSE ----------------------------------
    if not volumes:
        geometry_finding = Finding(
            check="geometry_consistency",
            severity=Severity.OK,
            message="No modalities supplied; nothing to compare.",
            detail={},
        )
    elif len(volumes) == 1:
        (only_role,) = volumes.keys()
        geometry_finding = Finding(
            check="geometry_consistency",
            severity=Severity.OK,
            message=f"Only one modality ({only_role!r}); nothing to compare.",
            detail={"reference_role": only_role},
        )
    else:
        ref_role = _reference_role(volumes)
        assert ref_role is not None  # volumes is non-empty here
        ref = volumes[ref_role]
        mismatched = []
        for role, vi in volumes.items():
            if role == ref_role:
                continue
            shape_ok = vi.shape == ref.shape
            affine_ok = shape_ok and np.allclose(vi.affine, ref.affine, atol=affine_atol)
            if not (shape_ok and affine_ok):
                mismatched.append(role)
        mismatched.sort()
        if mismatched:
            geometry_finding = Finding(
                check="geometry_consistency",
                severity=Severity.REFUSE,
                message=(
                    f"Geometry mismatch against reference {ref_role!r} for: "
                    f"{', '.join(mismatched)}."
                ),
                detail={"reference_role": ref_role, "mismatched_roles": mismatched},
            )
        else:
            geometry_finding = Finding(
                check="geometry_consistency",
                severity=Severity.OK,
                message=f"All modalities agree with reference {ref_role!r} on shape and affine.",
                detail={"reference_role": ref_role},
            )

    return (unexpected_finding, affine_finding, geometry_finding)


def check_spacing(volumes: Mapping[str, VolumeInfo], cfg_block: Any) -> tuple[Finding, ...]:
    """Voxel spacing range and anisotropy, read from `cfg_block`.

    Args:
        volumes: role -> `VolumeInfo`.
        cfg_block: `cfg.clinical.input_qc` (or anything exposing
            `spacing_min_mm`, `spacing_max_mm`, `anisotropy_warn_ratio`,
            `anisotropy_refuse_ratio`).

    Returns:
        `(spacing_range, anisotropy)`.
    """
    spacing_min = float(cfg_block.spacing_min_mm)
    spacing_max = float(cfg_block.spacing_max_mm)
    warn_ratio = float(cfg_block.anisotropy_warn_ratio)
    refuse_ratio = float(cfg_block.anisotropy_refuse_ratio)

    out_of_range: dict[str, list[float]] = {}
    ratios: dict[str, float] = {}
    for role, vi in volumes.items():
        dims = vi.spacing_mm
        if any(d < spacing_min or d > spacing_max for d in dims):
            out_of_range[role] = list(dims)
        ratios[role] = (max(dims) / min(dims)) if min(dims) > 0 else math.inf

    bounds = [spacing_min, spacing_max]
    if out_of_range:
        spacing_finding = Finding(
            check="spacing_range",
            severity=Severity.REFUSE,
            message=(
                f"Voxel spacing outside [{spacing_min}, {spacing_max}] mm for: "
                f"{', '.join(sorted(out_of_range))}."
            ),
            detail={"bounds_mm": bounds, "spacing_mm": out_of_range},
        )
    elif not volumes:
        spacing_finding = Finding(
            check="spacing_range",
            severity=Severity.OK,
            message="No modalities supplied; nothing to check.",
            detail={"bounds_mm": bounds},
        )
    else:
        spacing_finding = Finding(
            check="spacing_range",
            severity=Severity.OK,
            message=f"All voxel spacings are within [{spacing_min}, {spacing_max}] mm.",
            detail={
                "bounds_mm": bounds,
                "spacing_mm": {r: list(v.spacing_mm) for r, v in volumes.items()},
            },
        )

    refuse_roles = sorted(role for role, ratio in ratios.items() if ratio >= refuse_ratio)
    warn_roles = sorted(
        role for role, ratio in ratios.items() if warn_ratio <= ratio < refuse_ratio
    )
    if refuse_roles:
        aniso_finding = Finding(
            check="anisotropy",
            severity=Severity.REFUSE,
            message=f"Voxel anisotropy ratio >= {refuse_ratio} for: {', '.join(refuse_roles)}.",
            detail={"ratios": ratios, "warn_ratio": warn_ratio, "refuse_ratio": refuse_ratio},
        )
    elif warn_roles:
        aniso_finding = Finding(
            check="anisotropy",
            severity=Severity.WARN,
            message=f"Voxel anisotropy ratio >= {warn_ratio} for: {', '.join(warn_roles)}.",
            detail={"ratios": ratios, "warn_ratio": warn_ratio, "refuse_ratio": refuse_ratio},
        )
    elif not volumes:
        aniso_finding = Finding(
            check="anisotropy",
            severity=Severity.OK,
            message="No modalities supplied; nothing to check.",
            detail={"warn_ratio": warn_ratio, "refuse_ratio": refuse_ratio},
        )
    else:
        aniso_finding = Finding(
            check="anisotropy",
            severity=Severity.OK,
            message="Voxel anisotropy is within limits for all modalities.",
            detail={"ratios": ratios, "warn_ratio": warn_ratio, "refuse_ratio": refuse_ratio},
        )

    return (spacing_finding, aniso_finding)


def check_finite_values(volumes: Mapping[str, VolumeInfo]) -> tuple[Finding, ...]:
    """REFUSE if any modality contains a NaN or Inf voxel.

    Args:
        volumes: role -> `VolumeInfo`.

    Returns:
        `(finite_values,)`.
    """
    bad = {role: vi.n_nonfinite for role, vi in volumes.items() if vi.n_nonfinite > 0}
    if bad:
        finding = Finding(
            check="finite_values",
            severity=Severity.REFUSE,
            message=f"Non-finite (NaN/Inf) voxels present in: {', '.join(sorted(bad))}.",
            detail={"nonfinite_counts": bad},
        )
    elif not volumes:
        finding = Finding(
            check="finite_values",
            severity=Severity.OK,
            message="No modalities supplied; nothing to check.",
            detail={},
        )
    else:
        finding = Finding(
            check="finite_values",
            severity=Severity.OK,
            message="No non-finite voxels found in any modality.",
            detail={},
        )
    return (finding,)


def check_intensity_sanity(
    volumes: Mapping[str, VolumeInfo], cfg_block: Any
) -> tuple[Finding, ...]:
    """REFUSE if any modality's dynamic range is too low relative to its own scale.

    Reads `p005` / `p995` / `robust_scale` off each `VolumeInfo` and REFUSEs
    a role if `(p995 - p005) / robust_scale < cfg_block.min_dynamic_range`,
    OR if `robust_scale == 0.0` -- which `describe_volume` sets for both an
    entirely-zero volume and a constant nonzero one, so both collapse into
    the same branch here rather than needing separate handling (and without
    a zero-division).

    Args:
        volumes: role -> `VolumeInfo`.
        cfg_block: `cfg.clinical.input_qc` (or anything exposing
            `min_dynamic_range`).

    Returns:
        `(dynamic_range,)`.
    """
    min_dynamic_range = float(cfg_block.min_dynamic_range)
    bad: list[str] = []
    per_role: dict[str, dict[str, float]] = {}
    for role, vi in volumes.items():
        if vi.robust_scale == 0.0:
            ratio = 0.0
        else:
            ratio = (vi.p995 - vi.p005) / vi.robust_scale
        per_role[role] = {
            "ratio": ratio,
            "p005": vi.p005,
            "p995": vi.p995,
            "robust_scale": vi.robust_scale,
        }
        if ratio < min_dynamic_range:
            bad.append(role)
    bad.sort()

    if bad:
        finding = Finding(
            check="dynamic_range",
            severity=Severity.REFUSE,
            message=(
                f"Dynamic range below {min_dynamic_range} (relative to the volume's own "
                f"scale), or the volume is empty/constant, for: {', '.join(bad)}."
            ),
            detail={"min_dynamic_range": min_dynamic_range, "per_role": per_role},
        )
    elif not volumes:
        finding = Finding(
            check="dynamic_range",
            severity=Severity.OK,
            message="No modalities supplied; nothing to check.",
            detail={"min_dynamic_range": min_dynamic_range},
        )
    else:
        finding = Finding(
            check="dynamic_range",
            severity=Severity.OK,
            message="All modalities show adequate dynamic range.",
            detail={"min_dynamic_range": min_dynamic_range, "per_role": per_role},
        )
    return (finding,)


def check_shape_against_expected(
    volumes: Mapping[str, VolumeInfo], expected_shape: tuple[int, int, int]
) -> tuple[Finding, ...]:
    """WARN (never REFUSE) if a modality's shape differs from `expected_shape`.

    A WARN, not a REFUSE, because a different-but-self-consistent atlas is a
    legal configuration (see `configs/clinical/default.yaml`'s comment on
    this exact key) -- this check exists to notice that clinical
    preprocessing's atlas registration did not run, not to enforce one atlas.

    Args:
        volumes: role -> `VolumeInfo`.
        expected_shape: The expected `(D, H, W)`, from
            `cfg.clinical.input_qc.expected_shape`.

    Returns:
        `(expected_shape,)`.
    """
    expected = tuple(int(s) for s in expected_shape)
    mismatched = {role: vi.shape for role, vi in volumes.items() if vi.shape != expected}
    if mismatched:
        finding = Finding(
            check="expected_shape",
            severity=Severity.WARN,
            message=(
                f"Shape differs from expected {expected} for: {', '.join(sorted(mismatched))}."
            ),
            detail={
                "expected_shape": list(expected),
                "actual_shapes": {r: list(s) for r, s in mismatched.items()},
            },
        )
    elif not volumes:
        finding = Finding(
            check="expected_shape",
            severity=Severity.OK,
            message="No modalities supplied; nothing to check.",
            detail={"expected_shape": list(expected)},
        )
    else:
        finding = Finding(
            check="expected_shape",
            severity=Severity.OK,
            message=f"All modalities match the expected shape {expected}.",
            detail={"expected_shape": list(expected)},
        )
    return (finding,)


def check_brain_mask(
    mask: np.ndarray | None,
    spacing_mm: tuple[float, float, float],
    volumes: Mapping[str, VolumeInfo],
    cfg_block: Any,
) -> tuple[Finding, ...]:
    """Brain-mask-derived sanity: mask volume, in-mask signal, and whole-volume skull signal.

    `skull_present` needs no mask -- it reads each modality's OWN
    whole-volume nonzero fraction off `VolumeInfo`, so it always runs.
    `brain_volume` and `nonzero_in_brain` genuinely need `mask`; when
    `mask is None` they are each skipped with a visible `Severity.OK`
    finding saying so, rather than being silently absent (see the module
    docstring's "silent skip" concern).

    `nonzero_in_brain` reports the FRACTION of the mask's voxels for which a
    modality is also nonzero: `VolumeInfo.n_nonzero_in_mask / mask_voxel_count`,
    a true voxel-wise intersection computed once by `describe_volume` while
    the raw array was still in hand (see that function's docstring). This
    function does not itself see the raw arrays -- it only reads the counts
    `describe_volume` already computed, so `volumes` must have been built
    with this SAME `mask` for that count to be present; see the `Raises`
    section.

    Args:
        mask: The brain mask array, or `None` if unavailable.
        spacing_mm: Voxel size in mm, used to convert the mask's voxel count
            to millilitres.
        volumes: role -> `VolumeInfo`. When `mask` is not `None`, every
            entry must carry a non-`None` `n_nonzero_in_mask` (i.e. must
            have been built by `describe_volume(..., brain_mask=mask)`).
        cfg_block: `cfg.clinical.input_qc` (or anything exposing
            `brain_volume_min_ml`, `brain_volume_max_ml`,
            `min_nonzero_fraction_in_brain`, `skull_present_warn_fraction`).

    Returns:
        `(brain_volume, nonzero_in_brain, skull_present)`.

    Raises:
        ValueError: If `mask` is not `None`, the mask has at least one
            voxel, and some `VolumeInfo` in `volumes` has
            `n_nonzero_in_mask is None`. That is a programming error in the
            caller -- `describe_volume` was not given this same mask -- not
            a property of the data, so this is raised rather than silently
            falling back to a whole-volume approximation.
    """
    # -- skull_present: WARN, no mask needed ---------------------------
    warn_fraction = float(cfg_block.skull_present_warn_fraction)
    whole_fractions = {
        role: (vi.n_nonzero / vi.n_voxels if vi.n_voxels else 0.0) for role, vi in volumes.items()
    }
    skull_roles = sorted(role for role, frac in whole_fractions.items() if frac > warn_fraction)
    if skull_roles:
        skull_finding = Finding(
            check="skull_present",
            severity=Severity.WARN,
            message=(
                f"Whole-volume nonzero fraction above {warn_fraction} for: "
                f"{', '.join(skull_roles)}; a skull may still be present."
            ),
            detail={"whole_volume_fractions": whole_fractions, "warn_fraction": warn_fraction},
        )
    elif not volumes:
        skull_finding = Finding(
            check="skull_present",
            severity=Severity.OK,
            message="No modalities supplied; nothing to check.",
            detail={"warn_fraction": warn_fraction},
        )
    else:
        skull_finding = Finding(
            check="skull_present",
            severity=Severity.OK,
            message="Whole-volume nonzero fraction is within limits for all modalities.",
            detail={"whole_volume_fractions": whole_fractions, "warn_fraction": warn_fraction},
        )

    # -- brain_volume / nonzero_in_brain: need the mask -----------------
    if mask is None:
        volume_finding = Finding(
            check="brain_volume",
            severity=Severity.OK,
            message="Brain mask unavailable; skipping the brain-volume check.",
            detail={},
        )
        fraction_finding = Finding(
            check="nonzero_in_brain",
            severity=Severity.OK,
            message="Brain mask unavailable; skipping the nonzero-in-brain check.",
            detail={},
        )
        return (volume_finding, fraction_finding, skull_finding)

    mask_bool = np.asarray(mask).astype(bool)
    mask_voxel_count = int(mask_bool.sum())

    vol_min = float(cfg_block.brain_volume_min_ml)
    vol_max = float(cfg_block.brain_volume_max_ml)
    # Pure multiplication, never a division -- a zero-voxel mask therefore
    # yields 0.0 mL cleanly rather than a ZeroDivisionError, and 0.0 mL is
    # then simply outside [vol_min, vol_max] like any other out-of-range
    # volume.
    voxel_volume_mm3 = math.prod(spacing_mm)
    volume_ml = mask_voxel_count * voxel_volume_mm3 / 1000.0
    bounds_ml = [vol_min, vol_max]
    if volume_ml < vol_min or volume_ml > vol_max:
        volume_finding = Finding(
            check="brain_volume",
            severity=Severity.REFUSE,
            message=f"Brain mask volume {volume_ml:.1f} mL is outside [{vol_min}, {vol_max}] mL.",
            detail={
                "volume_ml": volume_ml,
                "bounds_ml": bounds_ml,
                "mask_voxel_count": mask_voxel_count,
            },
        )
    else:
        volume_finding = Finding(
            check="brain_volume",
            severity=Severity.OK,
            message=f"Brain mask volume {volume_ml:.1f} mL is within [{vol_min}, {vol_max}] mL.",
            detail={
                "volume_ml": volume_ml,
                "bounds_ml": bounds_ml,
                "mask_voxel_count": mask_voxel_count,
            },
        )

    min_fraction = float(cfg_block.min_nonzero_fraction_in_brain)
    if mask_voxel_count == 0:
        # Division by mask_voxel_count would be a ZeroDivisionError; the
        # empty-mask case is already flagged above via brain_volume (0 mL),
        # so this is a visible, non-crashing skip rather than a duplicate
        # refusal.
        fraction_finding = Finding(
            check="nonzero_in_brain",
            severity=Severity.OK,
            message=(
                "Brain mask has zero voxels; skipping the nonzero-in-brain fraction check "
                "(see brain_volume)."
            ),
            detail={"mask_voxel_count": 0},
        )
    else:
        missing = sorted(role for role, vi in volumes.items() if vi.n_nonzero_in_mask is None)
        if missing:
            raise ValueError(
                "check_brain_mask: a brain_mask was supplied but VolumeInfo for "
                f"{', '.join(missing)} has n_nonzero_in_mask=None. describe_volume must be "
                "given this same mask (describe_volume(..., brain_mask=mask)) so the true "
                "voxel-wise intersection is available -- this is a caller error, not a data "
                "condition, so it is not silently approximated."
            )
        in_brain_fractions = {
            role: vi.n_nonzero_in_mask / mask_voxel_count for role, vi in volumes.items()
        }
        low_roles = sorted(role for role, frac in in_brain_fractions.items() if frac < min_fraction)
        if low_roles:
            fraction_finding = Finding(
                check="nonzero_in_brain",
                severity=Severity.REFUSE,
                message=(
                    f"Nonzero-in-brain fraction below {min_fraction} for: "
                    f"{', '.join(low_roles)}."
                ),
                detail={"in_brain_fractions": in_brain_fractions, "min_fraction": min_fraction},
            )
        elif not volumes:
            fraction_finding = Finding(
                check="nonzero_in_brain",
                severity=Severity.OK,
                message="No modalities supplied; nothing to check.",
                detail={"min_fraction": min_fraction},
            )
        else:
            fraction_finding = Finding(
                check="nonzero_in_brain",
                severity=Severity.OK,
                message="Nonzero-in-brain fraction is within limits for all modalities.",
                detail={"in_brain_fractions": in_brain_fractions, "min_fraction": min_fraction},
            )

    return (volume_finding, fraction_finding, skull_finding)


def run_input_qc(
    cfg: Any,
    volumes: Mapping[str, VolumeInfo],
    *,
    brain_mask: np.ndarray | None = None,
) -> InputQCReport:
    """Run every check and compose one `InputQCReport`.

    Reads `cfg.clinical.input_qc` (see `configs/clinical/default.yaml`) --
    NOT `cfg.input_qc`, which does not exist at the composed config path.

    A supplied `brain_mask` whose shape disagrees with the reference
    modality's shape is folded into the `geometry_consistency` finding
    (REFUSE), rather than raising its own separate check id: it is exactly
    the same failure mode `check_geometry_consistency` already reports for
    two mismatched modalities, just with the mask standing in for one of
    them, and this module's caller should be able to look at ONE
    `geometry_consistency` finding to learn everything about geometry
    disagreement in the study.

    Args:
        cfg: The root config (or anything exposing `cfg.clinical.input_qc`
            with the fields `configs/clinical/default.yaml` documents).
        volumes: role -> `VolumeInfo`, typically from `load_volume_infos`.
        brain_mask: The brain mask array, if available. `None` -> the
            brain-mask-dependent checks are skipped, visibly (see
            `check_brain_mask`).

    Returns:
        The composed report. `verdict` is REFUSE if any finding refused,
        else WARN if any warned, else OK.
    """
    qc_cfg = cfg.clinical.input_qc

    findings: list[Finding] = [check_sequence_completeness(volumes.keys(), qc_cfg.required_roles)]

    geometry_findings = list(check_geometry_consistency(volumes, float(qc_cfg.affine_atol)))
    if brain_mask is not None:
        ref_role = _reference_role(volumes)
        if ref_role is not None:
            ref_shape = volumes[ref_role].shape
            mask_shape = tuple(int(s) for s in np.asarray(brain_mask).shape)
            if mask_shape != ref_shape:
                index = next(
                    i for i, f in enumerate(geometry_findings) if f.check == "geometry_consistency"
                )
                old = geometry_findings[index]
                geometry_findings[index] = Finding(
                    check="geometry_consistency",
                    severity=Severity.REFUSE,
                    message=(
                        f"{old.message} Brain mask shape {mask_shape} disagrees with "
                        f"reference {ref_role!r} shape {ref_shape}."
                    ),
                    detail={**old.detail, "brain_mask_shape": list(mask_shape)},
                )
    findings.extend(geometry_findings)

    findings.extend(check_finite_values(volumes))
    findings.extend(check_spacing(volumes, qc_cfg))
    findings.extend(check_shape_against_expected(volumes, tuple(qc_cfg.expected_shape)))
    findings.extend(check_intensity_sanity(volumes, qc_cfg))

    ref_role = _reference_role(volumes)
    reference_spacing = volumes[ref_role].spacing_mm if ref_role is not None else (1.0, 1.0, 1.0)
    findings.extend(check_brain_mask(brain_mask, reference_spacing, volumes, qc_cfg))

    verdict = _worst_severity(f.severity for f in findings)
    if verdict is Severity.REFUSE:
        logger.warning(
            "run_input_qc: REFUSE (%d refusal finding(s)): %s",
            sum(1 for f in findings if f.severity is Severity.REFUSE),
            "; ".join(f"{f.check}: {f.message}" for f in findings if f.severity is Severity.REFUSE),
        )
    else:
        logger.debug("run_input_qc: verdict=%s", verdict.value)

    return InputQCReport(verdict=verdict, findings=tuple(findings))


def load_volume_infos(
    paths: Mapping[str, Path],
) -> tuple[dict[str, VolumeInfo], np.ndarray | None]:
    """Load one `VolumeInfo` per path, plus an optional brain mask array.

    A path keyed `"brain_mask"` (see `_BRAIN_MASK_KEY`) is treated
    specially: it is loaded as a plain array and returned separately, not as
    a `VolumeInfo` -- it is not a modality, and `run_input_qc` takes it
    through its own `brain_mask` keyword.

    Args:
        paths: role -> NIfTI path. May include the reserved key
            `"brain_mask"`.

    Returns:
        `(volumes, brain_mask)`: `volumes` is role -> `VolumeInfo` for every
        key other than `"brain_mask"`; `brain_mask` is the loaded array, or
        `None` if no `"brain_mask"` key was supplied.

    Raises:
        ValueError: If a path does not parse as a readable NIfTI volume, or
            parses to something other than a 3D volume. The message names
            the role and wraps the underlying error, matching the shape
            `app/backend/jobs.py::_load_and_validate_nifti` already uses.
    """
    volumes: dict[str, VolumeInfo] = {}
    brain_mask: np.ndarray | None = None

    # The mask is loaded first, and in full, so it can be passed into every
    # describe_volume call below -- describe_volume needs the mask array
    # while it still has the modality's own array in hand, to compute the
    # true voxel-wise intersection (VolumeInfo.n_nonzero_in_mask). Iteration
    # order of `paths` must not matter for this, so the mask is not loaded
    # inline within the main loop.
    if _BRAIN_MASK_KEY in paths:
        mask_path = Path(paths[_BRAIN_MASK_KEY])
        try:
            mask_img = nib.load(str(mask_path))
            brain_mask = np.asarray(mask_img.dataobj)
        except Exception as exc:  # noqa: BLE001 - re-raised as ValueError, naming the role
            raise ValueError(
                f"load_volume_infos: input for {_BRAIN_MASK_KEY!r} ({mask_path}) is not a "
                f"valid NIfTI volume: {exc}"
            ) from exc

        if brain_mask.ndim != 3:
            raise ValueError(
                f"load_volume_infos: input for {_BRAIN_MASK_KEY!r} ({mask_path}) must be a "
                f"3D volume, got shape {brain_mask.shape!r}"
            )

    for role, path in paths.items():
        if role == _BRAIN_MASK_KEY:
            continue
        path = Path(path)
        try:
            img = nib.load(str(path))
            data = np.asarray(img.dataobj)
            affine = np.asarray(img.affine, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - re-raised as ValueError, naming the role
            raise ValueError(
                f"load_volume_infos: input for {role!r} ({path}) is not a valid NIfTI "
                f"volume: {exc}"
            ) from exc

        if data.ndim != 3:
            raise ValueError(
                f"load_volume_infos: input for {role!r} ({path}) must be a 3D volume, "
                f"got shape {data.shape!r}"
            )

        volumes[role] = describe_volume(role, data, affine, brain_mask=brain_mask)

    return volumes, brain_mask
