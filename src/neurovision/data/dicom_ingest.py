"""DICOM ingest: turn a real hospital study folder into four named NIfTIs.

Milestone 4, Phase E, task E1. Today `app/backend/jobs.py` accepts four
NIfTI files the caller has already named `t1` / `t1ce` / `t2` / `flair`. A
real hospital study is a folder of DICOM files containing anywhere from four
to thirty series -- localisers, scouts, derived maps, diffusion, perfusion --
and this module works out which four of them are the sequences the model was
trained on, or says it cannot.

**The central design constraint.** `pydicom` and the `dcm2niix` binary live
only in `.venv-clinical` (see `requirements-clinical.txt`); the project's
main test suite runs in `.venv` and must stay green there. So this file is
split cleanly in two:

- The **rule table and role assignment** (`normalise_tokens`,
  `classify_series`, `assign_roles`) operate on a plain `SeriesHeader`
  dataclass and use no DICOM library at all. They are fully tested in the
  main suite with synthetic headers.
- The **I/O layer** (`read_series_headers`, `resolve_dcm2niix`,
  `convert_series`, `ingest_study`) is isolated into a few thin functions.
  Every one of them imports `pydicom` *inside the function body*, never at
  module scope, so importing this module never requires `.venv-clinical`.

**The trap this rule table is built to avoid** (see `CLAUDE.md`'s ten
traps): a short token being a substring of a longer word. `"_t1" in name` is
true for `_t1ce`; a naive T1 rule would misclassify every post-contrast
series. So every description match here goes through `normalise_tokens`,
which splits on non-alphanumeric characters and compares whole tokens via
set membership -- never `in` on a raw string. FLAIR is decided before T2
("T2 FLAIR" is FLAIR, not T2) and T1CE is decided before T1 ("T1 POST GD" is
T1CE, not T1), both by construction of the scoring rules below rather than
by rule ordering, which is what makes them robust to any evaluation order.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from neurovision.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)

# Fixed channel order the model expects -- matches
# `neurovision.data.preprocessing`'s `_MODALITY_ROLES` and
# `app/backend/jobs.py`'s upload role names.
ROLES: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")


# ---------------------------------------------------------------------------
# Plain dataclasses -- no DICOM library involved.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesHeader:
    """The DICOM fields the rule table reads, and nothing else.

    Attributes:
        series_uid: SeriesInstanceUID (0020,000E).
        series_number: SeriesNumber (0020,0011), or None if absent.
        series_description: (0008,103E), empty string when absent.
        protocol_name: (0018,1030), empty string when absent.
        sequence_name: (0018,0024), empty string when absent.
        scanning_sequence: (0018,0020), e.g. `("SE",)`, `("GR",)`,
            `("IR", "SE")`.
        sequence_variant: (0018,0021).
        scan_options: (0018,0022).
        image_type: (0008,0008).
        echo_time: TE in ms, or None if absent.
        repetition_time: TR in ms, or None if absent.
        inversion_time: TI in ms, or None if absent.
        contrast_agent: (0018,0010), empty string when absent.
        n_instances: Number of DICOM instances (files) in this series.
    """

    series_uid: str
    series_number: int | None
    series_description: str
    protocol_name: str
    sequence_name: str
    scanning_sequence: tuple[str, ...]
    sequence_variant: tuple[str, ...]
    scan_options: tuple[str, ...]
    image_type: tuple[str, ...]
    echo_time: float | None
    repetition_time: float | None
    inversion_time: float | None
    contrast_agent: str
    n_instances: int


class SeriesOutcome(StrEnum):
    """Why `classify_series` (or `assign_roles`) landed where it did.

    This used to be encoded as a string prefix on `reasons[0]` (e.g.
    `"rejected: "`), parsed back out with `str.startswith` -- exactly the
    "a short token is a substring of a longer word" trap CLAUDE.md warns
    about, since a reworded reason string would silently break the parser.
    This enum replaces that: it is the one and only place routing decisions
    are made, and `reasons` is now pure human-readable prose.

    Attributes:
        ASSIGNED: A role was picked; `role` is one of `ROLES`.
        REJECTED: The series was thrown out outright (localiser/scout/
            derived image type, a rejection token, a diffusion/perfusion/
            angio token, or too few instances).
        NO_EVIDENCE: No rule matched any role at all (every score was 0.0).
        AMBIGUOUS: The top two candidate roles scored within
            `AMBIGUITY_MARGIN` of each other, so nothing was guessed.
    """

    ASSIGNED = "assigned"
    REJECTED = "rejected"
    NO_EVIDENCE = "no_evidence"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class RoleAssignment:
    """What the rule table decided for one series.

    Attributes:
        role: One of `ROLES`, or `None` if this series is not one of ours
            (rejected outright, no rule matched, or the top two candidates
            were too close to call).
        score: The winning role's score, or the top score seen if `role` is
            `None` because of ambiguity (0.0 if nothing matched at all).
        reasons: Human-readable strings, one per rule that fired -- this is
            a product requirement, not decoration: the UI has to tell a
            radiographer why a series was picked (or refused).
        outcome: The structural reason `role` is (or is not) set. See
            `SeriesOutcome`. Required (no default) so that every call site
            that builds a `None`-role assignment must state deliberately
            which of the three `None` cases it is -- a default here would
            let a rejected/no-evidence/ambiguous path silently fall back to
            `ASSIGNED`. Invariant: `outcome is SeriesOutcome.ASSIGNED` iff
            `role is not None`.
    """

    role: str | None
    score: float
    reasons: tuple[str, ...]
    outcome: SeriesOutcome


@dataclass(frozen=True)
class IngestResult:
    """The outcome of ingesting one DICOM study.

    Attributes:
        paths: role -> the NIfTI file written for it.
        assignments: series_uid -> what the rule table decided for every
            series seen, including rejected, ambiguous and losing ones.
        missing_roles: Roles in `ROLES` that no series was assigned to.
        rejected: `(series_uid, why)` pairs for series the rule table threw
            out outright (localiser, wrong modality, too few instances).
        warnings: Non-fatal notes -- an ambiguous series, an override that
            displaced an automatic winner, an empty study.
    """

    paths: dict[str, Path]
    assignments: dict[str, RoleAssignment]
    missing_roles: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# The rule table.
# ---------------------------------------------------------------------------

# Rejection vocabulary. Localisers/scouts/derived series and non-structural
# (diffusion/perfusion/angio) series are never one of our four target roles,
# whatever their description otherwise looks like.
_REJECT_IMAGE_TYPES = {"LOCALIZER", "SCOUT", "DERIVED", "SECONDARY"}
_REJECT_TOKENS = frozenset({"localizer", "localiser", "scout", "survey", "calibration", "ref"})
_NONSTRUCTURAL_TOKENS = frozenset(
    {"dwi", "adc", "dti", "trace", "dsc", "dce", "perfusion", "asl", "swi", "mra", "tof", "bold"}
)

# Role vocabulary. Every entry here is a WHOLE token (see `normalise_tokens`)
# -- never a substring pattern -- which is what keeps "t1ce" from ever being
# read as containing "t1".
_FLAIR_TOKENS = frozenset({"flair"})
_T1_TOKENS = frozenset({"t1", "t1w", "mprage", "mpr", "bravo", "tfl", "spgr", "fspgr"})
_T2_TOKENS = frozenset({"t2", "t2w", "tse", "frfse", "fse"})
_CONTRAST_TOKENS = frozenset({"gd", "gad", "gadolinium", "post", "ce", "contrast", "c"})
# "T1CE" with no separator tokenises to ONE token, "t1ce" -- it can never be
# decomposed into "t1" + "ce" by `normalise_tokens`, so it needs its own
# direct rule rather than relying on the T1 + contrast combination below.
_T1CE_DIRECT_TOKENS = frozenset({"t1ce"})

# --- Rule weights -----------------------------------------------------
# Every rule fires at most once per (header, role) and adds a fixed,
# documented amount to that role's score. `classify_series` reports every
# rule that fired as a human-readable reason and awards the role with the
# highest total. Weights are graded by how specific the evidence is: an
# explicit sequence-name token or a diagnostic TE/TR/TI combination is
# "primary" evidence and can decide a role on its own; TE/TR alone that
# merely leans one way is "support" and is deliberately too weak to decide
# anything by itself (see AMBIGUITY_MARGIN).
W_TOKEN = 3.0  # an explicit, unambiguous sequence-name token
W_TOKEN_T1CE_DIRECT = 5.0  # "t1ce" as a single token names the role outright
W_GEOMETRY_PRIMARY = 3.0  # a TE/TR/TI combination that is diagnostic alone
W_GEOMETRY_T1_SHORT = 2.0  # short TE + short TR: fairly specific to T1, weaker than a token
W_GEOMETRY_SUPPORT = 1.0  # TE/TR that reinforces other evidence but can't decide alone
W_DARK_FLUID = 2.0  # the "dark fluid" phrase, a plain-language name for FLAIR
W_CONTRAST_AGENT_FIELD = 3.0  # a non-empty DICOM contrast-agent field: hard evidence
W_CONTRAST_TOKEN = 2.0  # a post-contrast token in the description

# A role must beat its nearest rival by at least this much to win outright.
# Chosen strictly between W_GEOMETRY_SUPPORT (1.0) and W_TOKEN /
# W_GEOMETRY_PRIMARY (3.0): a lone "support" rule can never by itself put a
# role 1.5 points clear of a zero-scoring rival, so weak geometry-only
# leanings correctly come out ambiguous rather than confidently wrong, while
# a single strong token or primary geometry match always clears the bar.
AMBIGUITY_MARGIN = 1.5

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def normalise_tokens(text: str) -> tuple[str, ...]:
    """Lowercase and split free text into whole alphanumeric tokens.

    Splits on runs of non-alphanumeric characters, so every separator --
    space, underscore, hyphen, plus, slash -- breaks a token, and no rule
    anywhere in this module ever matches a short token with `in` on the raw
    string (see the module docstring for why that is banned here). A word
    with no separator inside it survives as ONE token: `"T1CE"` returns
    `("t1ce",)`, never `("t1", "ce")`.

    Args:
        text: Free-text DICOM field, e.g. a series description.

    Returns:
        Lowercase tokens in order, with empty strings dropped.
    """
    if not text:
        return ()
    return tuple(part for part in _TOKEN_SPLIT_RE.split(text.lower()) if part)


def _rejection_reason(header: SeriesHeader, tokens: frozenset[str]) -> str | None:
    """Return why a series is outright not one of our four roles, or None."""
    image_types = {str(t).upper() for t in header.image_type}
    hit_types = image_types & _REJECT_IMAGE_TYPES
    if hit_types:
        return f"image_type contains {sorted(hit_types)}"

    hit_reject = tokens & _REJECT_TOKENS
    if hit_reject:
        return f"description token(s) {sorted(hit_reject)} name a non-diagnostic series"

    hit_nonstructural = tokens & _NONSTRUCTURAL_TOKENS
    if hit_nonstructural:
        return (
            f"description token(s) {sorted(hit_nonstructural)} "
            "name a diffusion/perfusion/angio series"
        )

    return None


def _score_flair(header: SeriesHeader, tokens: frozenset[str]) -> tuple[float, tuple[str, ...]]:
    """FLAIR evidence. Decided independently of T2 -- see `_score_t2`."""
    score = 0.0
    reasons: list[str] = []

    if tokens & _FLAIR_TOKENS:
        score += W_TOKEN
        reasons.append("description contains the token 'flair'")

    if (
        header.inversion_time is not None
        and header.inversion_time > 1500
        and "IR" in header.scanning_sequence
    ):
        score += W_GEOMETRY_PRIMARY
        reasons.append(
            f"inversion-recovery sequence with TI={header.inversion_time:.0f}ms (>1500ms)"
        )

    if {"dark", "fluid"} <= tokens:
        score += W_DARK_FLUID
        reasons.append("description mentions 'dark fluid'")

    if (
        header.echo_time is not None
        and header.echo_time >= 80
        and header.repetition_time is not None
        and header.repetition_time >= 4000
    ):
        score += W_GEOMETRY_SUPPORT
        reasons.append(
            f"long TE={header.echo_time:.0f}ms/TR={header.repetition_time:.0f}ms supports FLAIR"
        )

    return score, tuple(reasons)


def _score_t1_weighted(
    header: SeriesHeader, tokens: frozenset[str]
) -> tuple[float, tuple[str, ...]]:
    """'Is this series T1-weighted at all' -- shared by the T1 and T1CE rules."""
    score = 0.0
    reasons: list[str] = []

    matched = tokens & _T1_TOKENS
    if matched:
        score += W_TOKEN
        reasons.append(f"description token(s) {sorted(matched)} indicate T1-weighted")

    if (
        header.echo_time is not None
        and header.echo_time < 30
        and header.repetition_time is not None
        and header.repetition_time < 1000
    ):
        score += W_GEOMETRY_T1_SHORT
        reasons.append(
            f"short TE={header.echo_time:.0f}ms/TR={header.repetition_time:.0f}ms "
            "indicates T1-weighted geometry"
        )

    return score, tuple(reasons)


def _score_contrast(header: SeriesHeader, tokens: frozenset[str]) -> tuple[float, tuple[str, ...]]:
    """Post-contrast evidence -- shared by the T1CE rule and T1's reduction."""
    score = 0.0
    reasons: list[str] = []

    if header.contrast_agent:
        score += W_CONTRAST_AGENT_FIELD
        reasons.append(f"contrast agent field is set: {header.contrast_agent!r}")

    matched = tokens & _CONTRAST_TOKENS
    if matched:
        score += W_CONTRAST_TOKEN
        reasons.append(f"description token(s) {sorted(matched)} indicate post-contrast")

    return score, tuple(reasons)


def _score_t2(
    header: SeriesHeader, tokens: frozenset[str], *, flair_evidence: bool
) -> tuple[float, tuple[str, ...]]:
    """T2 evidence. FLAIR is decided first: no T2 credit once FLAIR evidence exists."""
    if flair_evidence:
        return 0.0, ()

    score = 0.0
    reasons: list[str] = []

    matched = tokens & _T2_TOKENS
    if matched:
        score += W_TOKEN
        reasons.append(f"description token(s) {sorted(matched)} indicate T2-weighted")

    if (
        header.echo_time is not None
        and header.echo_time >= 80
        and header.repetition_time is not None
        and header.repetition_time >= 2000
        and header.inversion_time is None
    ):
        score += W_GEOMETRY_PRIMARY
        reasons.append(
            f"long TE={header.echo_time:.0f}ms/TR={header.repetition_time:.0f}ms "
            "without an inversion time indicates T2-weighted"
        )

    return score, tuple(reasons)


def _score_t1(
    t1_evidence: float, t1_evidence_reasons: tuple[str, ...], contrast_evidence: float
) -> tuple[float, tuple[str, ...]]:
    """T1 evidence, reduced to zero once post-contrast evidence exists.

    T1CE is decided first, mirroring FLAIR-before-T2: a series that is both
    T1-weighted and post-contrast is T1CE's territory, not plain T1's.
    """
    if t1_evidence > 0 and contrast_evidence > 0:
        return 0.0, ("post-contrast evidence present; suppressing plain-T1 in favor of T1CE",)
    return t1_evidence, t1_evidence_reasons


def _score_t1ce(
    tokens: frozenset[str],
    t1_evidence: float,
    t1_evidence_reasons: tuple[str, ...],
    contrast_evidence: float,
    contrast_reasons: tuple[str, ...],
) -> tuple[float, tuple[str, ...]]:
    """T1CE evidence: the direct 't1ce' token, or T1-weighted + post-contrast together."""
    score = 0.0
    reasons: list[str] = []

    if tokens & _T1CE_DIRECT_TOKENS:
        score += W_TOKEN_T1CE_DIRECT
        reasons.append("description token 't1ce' names the role directly")

    if t1_evidence > 0 and contrast_evidence > 0:
        score += t1_evidence + contrast_evidence
        reasons.extend(t1_evidence_reasons)
        reasons.extend(contrast_reasons)

    return score, tuple(reasons)


def classify_series(header: SeriesHeader) -> RoleAssignment:
    """Score one DICOM series against the four-role rule table.

    Applies the absolute rejection rules first (LOCALIZER/SCOUT/
    DERIVED/SECONDARY image type, a rejection token, a diffusion/
    perfusion/angio token), then scores `t1` / `t1ce` / `t2` / `flair` and
    returns the highest scorer -- or `role=None` if nothing scored at all,
    or if the top two candidates are within `AMBIGUITY_MARGIN` of each
    other, in which case the series is reported as ambiguous rather than
    guessed.

    Args:
        header: The DICOM fields this rule table reads.

    Returns:
        The winning role (or `None`), its score, and the reasons that fired.
    """
    text = " ".join((header.series_description, header.protocol_name, header.sequence_name))
    tokens = frozenset(normalise_tokens(text))

    reject_reason = _rejection_reason(header, tokens)
    if reject_reason is not None:
        return RoleAssignment(
            role=None, score=0.0, reasons=(reject_reason,), outcome=SeriesOutcome.REJECTED
        )

    flair_score, flair_reasons = _score_flair(header, tokens)
    t1_evidence, t1_evidence_reasons = _score_t1_weighted(header, tokens)
    contrast_evidence, contrast_reasons = _score_contrast(header, tokens)
    t2_score, t2_reasons = _score_t2(header, tokens, flair_evidence=flair_score > 0)
    t1_score, t1_reasons = _score_t1(t1_evidence, t1_evidence_reasons, contrast_evidence)
    t1ce_score, t1ce_reasons = _score_t1ce(
        tokens, t1_evidence, t1_evidence_reasons, contrast_evidence, contrast_reasons
    )

    role_scores: dict[str, float] = {
        "t1": t1_score,
        "t1ce": t1ce_score,
        "t2": t2_score,
        "flair": flair_score,
    }
    role_reasons: dict[str, tuple[str, ...]] = {
        "t1": t1_reasons,
        "t1ce": t1ce_reasons,
        "t2": t2_reasons,
        "flair": flair_reasons,
    }

    # Highest score first; ROLES order breaks an exact score tie so this is
    # fully deterministic regardless of dict iteration order.
    ranked = sorted(role_scores.items(), key=lambda kv: (-kv[1], ROLES.index(kv[0])))
    top_role, top_score = ranked[0]
    second_role, second_score = ranked[1]

    if top_score == 0.0:
        return RoleAssignment(
            role=None,
            score=0.0,
            reasons=(f"no rule matched any of {ROLES}",),
            outcome=SeriesOutcome.NO_EVIDENCE,
        )

    if top_score - second_score < AMBIGUITY_MARGIN:
        ambiguity = (
            f"{top_role} (score={top_score:.2f}) vs "
            f"{second_role} (score={second_score:.2f}); margin "
            f"{top_score - second_score:.2f} < {AMBIGUITY_MARGIN}"
        )
        return RoleAssignment(
            role=None,
            score=top_score,
            reasons=(ambiguity, *role_reasons[top_role], *role_reasons[second_role]),
            outcome=SeriesOutcome.AMBIGUOUS,
        )

    return RoleAssignment(
        role=top_role,
        score=top_score,
        reasons=role_reasons[top_role],
        outcome=SeriesOutcome.ASSIGNED,
    )


def assign_roles(
    headers: Sequence[SeriesHeader],
    *,
    min_instances: int,
    overrides: Mapping[str, str] | None = None,
) -> tuple[
    dict[str, SeriesHeader],
    dict[str, RoleAssignment],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
]:
    """Pick one series per role out of a study's series headers.

    Rejects first (too few instances, then everything `classify_series`
    rejects outright), classifies the survivors, then for each role picks
    the highest-scoring series among those classified into it. Ties break
    by most instances, then lowest series number, then lowest series_uid --
    fully deterministic, independent of input order or of dict/set
    iteration order. A series that wins no role stays visible in the
    returned `assignments` dict (the "leftovers") rather than being
    silently dropped.

    `overrides` is applied AFTER automatic assignment and takes precedence.
    Missing roles are never raised on here -- refusing a study for a
    missing sequence is E4's job, not this function's; check the returned
    dict's keys against `ROLES` for what is missing.

    Args:
        headers: One `SeriesHeader` per series in the study. May be empty.
        min_instances: A series with fewer instances than this is rejected
            as a non-diagnostic (localiser/scout/derived) volume.
        overrides: series_uid -> role, from a manual UI correction. Applied
            after the automatic pass and overrides it.

    Returns:
        `(role_to_header, assignments, rejected, warnings)`:
        - `role_to_header`: role -> the winning `SeriesHeader`. Missing keys
          are missing roles.
        - `assignments`: series_uid -> what was decided for every header
          given, including rejected, ambiguous and losing series.
        - `rejected`: `(series_uid, why)` for series thrown out outright.
        - `warnings`: non-fatal notes (ambiguous series, an override that
          displaced an automatic winner, an empty input).

    Raises:
        ValueError: If an override names a `series_uid` not present in
            `headers`, or a role outside `ROLES`.
    """
    assignments: dict[str, RoleAssignment] = {}
    rejected: list[tuple[str, str]] = []
    warnings: list[str] = []
    headers_by_uid: dict[str, SeriesHeader] = {h.series_uid: h for h in headers}

    if not headers:
        warnings.append("no DICOM series provided; nothing to classify")

    survivors: list[SeriesHeader] = []
    for header in headers:
        if header.n_instances < min_instances:
            reason = f"too few instances ({header.n_instances} < {min_instances})"
            assignments[header.series_uid] = RoleAssignment(
                role=None, score=0.0, reasons=(reason,), outcome=SeriesOutcome.REJECTED
            )
            rejected.append((header.series_uid, reason))
            continue

        assignment = classify_series(header)
        assignments[header.series_uid] = assignment
        if assignment.role is None:
            reason = assignment.reasons[0] if assignment.reasons else "unclassified"
            # Route on the structural outcome, never on the text of a
            # reason -- see `SeriesOutcome`'s docstring for why.
            if assignment.outcome is SeriesOutcome.REJECTED:
                rejected.append((header.series_uid, reason))
            else:
                warnings.append(f"series {header.series_uid}: {reason}")
            continue

        survivors.append(header)

    role_winner_uid: dict[str, str] = {}
    for role in ROLES:
        candidates = [h for h in survivors if assignments[h.series_uid].role == role]
        if not candidates:
            continue

        def _sort_key(h: SeriesHeader) -> tuple[float, int, float, str]:
            candidate_score = assignments[h.series_uid].score
            series_number = h.series_number if h.series_number is not None else math.inf
            return (-candidate_score, -h.n_instances, series_number, h.series_uid)

        candidates.sort(key=_sort_key)
        role_winner_uid[role] = candidates[0].series_uid

    if overrides:
        for uid, role in overrides.items():
            if uid not in headers_by_uid:
                raise ValueError(f"assign_roles: override names unknown series_uid {uid!r}")
            if role not in ROLES:
                raise ValueError(
                    f"assign_roles: override names unknown role {role!r}; expected one of {ROLES}"
                )
        for uid, role in overrides.items():
            previous_uid = role_winner_uid.get(role)
            if previous_uid is not None and previous_uid != uid:
                warnings.append(
                    f"override displaced automatic winner for role {role!r}: "
                    f"{previous_uid} -> {uid}"
                )
            role_winner_uid[role] = uid

    role_to_header = {role: headers_by_uid[uid] for role, uid in role_winner_uid.items()}
    return role_to_header, assignments, tuple(rejected), tuple(warnings)


# ---------------------------------------------------------------------------
# I/O layer. Every function below imports pydicom INSIDE its body, so this
# module is importable in the main .venv without pydicom installed at all.
# Only these functions (and their tests, via `pytest.importorskip`) need
# `.venv-clinical`.
# ---------------------------------------------------------------------------

# Generous but bounded: a single series' worth of slices should never take
# this long to convert. Chosen so a hung dcm2niix process cannot silently
# stall an ingest job forever.
_DCM2NIIX_TIMEOUT_S = 300


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a pydicom multi-valued element (or a scalar) to a str tuple."""
    if value is None:
        return ()
    if isinstance(value, list | tuple):
        return tuple(str(v) for v in value)
    return (str(value),)


def _safe_float(value: Any) -> float | None:
    """Coerce a pydicom numeric element to a plain float, or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Coerce a pydicom numeric element to a plain int, or None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_series_headers(study_dir: Path) -> list[SeriesHeader]:
    """Read one `SeriesHeader` per DICOM series found under a study folder.

    Walks `study_dir` recursively, reads each file's header only (never
    pixel data -- `stop_before_pixels=True`, since this only needs
    metadata), groups files by SeriesInstanceUID, and takes the header
    fields from the first instance of each series (a clinical series shares
    one protocol across its instances). Files that are not readable DICOM
    (a DICOMDIR index, a stray README) are skipped rather than failing the
    whole study.

    Args:
        study_dir: Root folder of one DICOM study. May contain nested
            subdirectories, e.g. one per series.

    Returns:
        One `SeriesHeader` per distinct SeriesInstanceUID found, sorted by
        series_uid for a deterministic order.
    """
    import pydicom

    study_dir = Path(study_dir)
    series_datasets: dict[str, list[Any]] = {}
    for path in sorted(study_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception as exc:  # noqa: BLE001 - a study folder may hold non-DICOM files
            logger.debug("read_series_headers: skipping non-DICOM file %s: %s", path, exc)
            continue

        uid = getattr(dataset, "SeriesInstanceUID", None)
        if not uid:
            logger.debug("read_series_headers: skipping %s: no SeriesInstanceUID.", path)
            continue
        series_datasets.setdefault(str(uid), []).append(dataset)

    headers: list[SeriesHeader] = []
    for uid in sorted(series_datasets):
        datasets = series_datasets[uid]
        first = datasets[0]
        headers.append(
            SeriesHeader(
                series_uid=uid,
                series_number=_safe_int(getattr(first, "SeriesNumber", None)),
                series_description=str(getattr(first, "SeriesDescription", "") or ""),
                protocol_name=str(getattr(first, "ProtocolName", "") or ""),
                sequence_name=str(getattr(first, "SequenceName", "") or ""),
                scanning_sequence=_as_str_tuple(getattr(first, "ScanningSequence", None)),
                sequence_variant=_as_str_tuple(getattr(first, "SequenceVariant", None)),
                scan_options=_as_str_tuple(getattr(first, "ScanOptions", None)),
                image_type=_as_str_tuple(getattr(first, "ImageType", None)),
                echo_time=_safe_float(getattr(first, "EchoTime", None)),
                repetition_time=_safe_float(getattr(first, "RepetitionTime", None)),
                inversion_time=_safe_float(getattr(first, "InversionTime", None)),
                contrast_agent=str(getattr(first, "ContrastBolusAgent", "") or ""),
                n_instances=len(datasets),
            )
        )
    return headers


def resolve_dcm2niix(configured_path: str | None) -> Path:
    """Resolve the `dcm2niix` binary from config, or from `PATH`.

    Args:
        configured_path: An explicit path to pin a specific build, or
            `None` to resolve from `PATH` via `shutil.which` -- what makes
            the same config work unchanged on macOS and Linux.

    Returns:
        Path to the resolved `dcm2niix` executable.

    Raises:
        FileNotFoundError: If `configured_path` is set but does not exist,
            or if `configured_path` is `None` and no `dcm2niix` is on
            `PATH`. The message names the fix: build `.venv-clinical` from
            `requirements-clinical.txt`.
    """
    if configured_path:
        resolved = Path(configured_path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"resolve_dcm2niix: configured dcm2niix_path {resolved} does not exist."
            )
        return resolved

    found = shutil.which("dcm2niix")
    if found is None:
        raise FileNotFoundError(
            "resolve_dcm2niix: no 'dcm2niix' binary found on PATH. Build the clinical "
            "environment first: `uv venv --python 3.11 .venv-clinical && "
            ".venv-clinical/bin/pip install -r requirements-clinical.txt -e .`, or set "
            "clinical.ingest.dcm2niix_path explicitly in config."
        )
    return Path(found)


def convert_series(study_dir: Path, header: SeriesHeader, out_path: Path, dcm2niix: Path) -> Path:
    """Convert one DICOM series to a NIfTI file at an exact path.

    `dcm2niix` converts everything in a directory it is pointed at, and a
    real study folder mixes many series together, so this re-reads headers
    to find exactly the files belonging to `header.series_uid`, copies just
    those into a private temp directory, and runs `dcm2niix` on that
    directory alone. That also sidesteps the fact that `dcm2niix` names its
    output from a filename template rather than an exact path: with only
    one series as input, it should produce exactly one volume, which this
    function then locates and renames to `out_path`.

    Args:
        study_dir: Root folder of the DICOM study `header` was read from.
        header: The series to convert.
        out_path: Exact destination path, e.g. `<out_dir>/t1ce.nii.gz`.
        dcm2niix: Resolved path to the `dcm2niix` binary (see
            `resolve_dcm2niix`).

    Returns:
        `out_path`, once the converted file has been moved there.

    Raises:
        RuntimeError: If no file under `study_dir` matches
            `header.series_uid`, if `dcm2niix` exits non-zero (message
            includes its stderr), if it times out, or if it produces zero
            or more than one candidate volume for the series.
    """
    import pydicom

    study_dir = Path(study_dir)
    out_path = Path(out_path)

    with tempfile.TemporaryDirectory(prefix="dicom_ingest_") as tmp:
        tmp_path = Path(tmp)
        series_input_dir = ensure_dir(tmp_path / "series_input")
        series_output_dir = ensure_dir(tmp_path / "series_output")

        matched = 0
        for path in sorted(study_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            except Exception:  # noqa: BLE001 - non-DICOM files are simply not this series
                continue
            if str(getattr(dataset, "SeriesInstanceUID", "")) != header.series_uid:
                continue
            shutil.copy2(path, series_input_dir / path.name)
            matched += 1

        if matched == 0:
            raise RuntimeError(
                f"convert_series: no files under {study_dir} match series_uid "
                f"{header.series_uid!r} ({header.series_description!r})."
            )

        cmd = [
            str(dcm2niix),
            "-z",
            "y",  # gzip the output, matching the project's .nii.gz convention
            "-b",
            "n",  # no BIDS JSON sidecar -- we only want the volume
            "-f",
            "series",  # fixed stem; dcm2niix appends a suffix if it splits the series
            "-o",
            str(series_output_dir),
            str(series_input_dir),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=_DCM2NIIX_TIMEOUT_S,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"convert_series: dcm2niix failed for series {header.series_uid!r} "
                f"({header.series_description!r}): {exc.stderr}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"convert_series: dcm2niix timed out after {_DCM2NIIX_TIMEOUT_S}s for series "
                f"{header.series_uid!r} ({header.series_description!r})."
            ) from exc

        candidates = sorted(series_output_dir.glob("series*.nii.gz"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"convert_series: dcm2niix produced {len(candidates)} candidate volume(s) for "
                f"series {header.series_uid!r} ({header.series_description!r}): "
                f"{[c.name for c in candidates]}; expected exactly 1."
            )

        ensure_dir(out_path.parent)
        shutil.move(str(candidates[0]), str(out_path))

    return out_path


def _build_manifest(
    study_dir: Path,
    out_dir: Path,
    paths: Mapping[str, Path],
    missing_roles: Sequence[str],
    assignments: Mapping[str, RoleAssignment],
    rejected: Sequence[tuple[str, str]],
    warnings: Sequence[str],
) -> dict[str, Any]:
    """Build the plain-dict audit trail `ingest_study` writes as JSON.

    Pulled out of `ingest_study` as its own function so the manifest's
    shape -- in particular, that every series entry carries its
    `SeriesOutcome`, not just its role -- can be tested with plain
    `RoleAssignment` values, without needing `pydicom` or a real
    `dcm2niix` binary.

    Args:
        study_dir: Root folder of the DICOM study that was ingested.
        out_dir: Directory the converted NIfTIs and manifest were written to.
        paths: role -> the NIfTI file written for it.
        missing_roles: Roles in `ROLES` that no series was assigned to.
        assignments: series_uid -> what the rule table decided for every
            series seen, including rejected, ambiguous and losing ones.
        rejected: `(series_uid, why)` pairs for series thrown out outright.
        warnings: Non-fatal notes (ambiguous series, an override, etc.).

    Returns:
        A JSON-serialisable dict: the manifest.
    """
    return {
        "study_dir": str(study_dir),
        "out_dir": str(out_dir),
        "roles_written": {role: str(path) for role, path in paths.items()},
        "missing_roles": list(missing_roles),
        "series": {
            uid: {
                "role": a.role,
                "score": a.score,
                "reasons": list(a.reasons),
                # `.value`, not `str(...)` or `repr(...)`: SeriesOutcome is a
                # str Enum, so `str(a.outcome)` would give "SeriesOutcome.REJECTED"
                # -- `.value` is the plain "rejected" the manifest should read.
                "outcome": a.outcome.value,
            }
            for uid, a in assignments.items()
        },
        "rejected": [{"series_uid": uid, "reason": reason} for uid, reason in rejected],
        "warnings": list(warnings),
    }


def ingest_study(cfg: Any, study_dir: Path, out_dir: Path) -> IngestResult:
    """Turn one DICOM study folder into four named NIfTIs, plus an audit trail.

    Reads `cfg.clinical.ingest` (`dcm2niix_path`, `min_instances`,
    `role_overrides`) as written in `configs/clinical/default.yaml`. Reads
    headers, assigns roles, converts each assigned series to
    `<out_dir>/<role>.nii.gz`, and writes `<out_dir>/ingest_manifest.json`
    recording every series seen, its assignment, its score and its reasons,
    plus the missing roles -- the manifest is the audit trail that must
    make a wrong assignment explainable after the fact.

    Args:
        cfg: The root config (or anything exposing `cfg.clinical.ingest`
            with the fields above).
        study_dir: Root folder of the DICOM study to ingest.
        out_dir: Directory to write the converted NIfTIs and the manifest
            into. Created if it does not exist.

    Returns:
        An `IngestResult` describing what was written and decided. Missing
        roles are reported in it, never raised (refusing a study for a
        missing sequence is E4's job).

    Raises:
        FileNotFoundError: If `dcm2niix` cannot be resolved (see
            `resolve_dcm2niix`).
        RuntimeError: If converting an assigned series fails (see
            `convert_series`).
        ValueError: If `cfg.clinical.ingest.role_overrides` names an
            unknown series_uid or role (see `assign_roles`).
    """
    ingest_cfg = cfg.clinical.ingest
    dcm2niix = resolve_dcm2niix(ingest_cfg.dcm2niix_path)

    study_dir = Path(study_dir)
    headers = read_series_headers(study_dir)

    overrides = dict(ingest_cfg.role_overrides) if ingest_cfg.role_overrides else None
    role_headers, assignments, rejected, warnings = assign_roles(
        headers, min_instances=ingest_cfg.min_instances, overrides=overrides
    )
    missing_roles = tuple(role for role in ROLES if role not in role_headers)

    out_dir = ensure_dir(out_dir)
    paths: dict[str, Path] = {}
    for role, header in role_headers.items():
        out_path = out_dir / f"{role}.nii.gz"
        paths[role] = convert_series(study_dir, header, out_path, dcm2niix)

    manifest = _build_manifest(
        study_dir, out_dir, paths, missing_roles, assignments, rejected, warnings
    )
    write_json(manifest, out_dir / "ingest_manifest.json")

    return IngestResult(
        paths=paths,
        assignments=assignments,
        missing_roles=missing_roles,
        rejected=rejected,
        warnings=warnings,
    )
