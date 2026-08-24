"""The refusal gate: E5. Combine every safety signal into PROCEED / CAUTION / REFUSE.

Milestone 4, Phase E, task **E5**. `docs/research/master_plan.md` principle 2 is the
reason this module exists at all: "errors multiply in a cascade -- five stages at 95%
each is 77% end-to-end -- so every stage emits a confidence and the pipeline has an
explicit REFUSE state." This module IS that state, for a pipeline whose stages are (a)
`input_qc` (E3/E4, this project's only label-free, always-on gate), (b) the QC model's
own estimate of the mask's Dice (Phase C), (c) the conformal band width (Phase B), and
(d) a case-level out-of-distribution score. Each is judged independently by its own
`judge_*` function and then combined by `run_gatekeeper` -- the same "individually
public, individually tested checks composed by one entry point" shape as
`neurovision.inference.input_qc`, and for the same reason: a composite verdict that is
the only testable thing is a gate nobody can debug.

**`Severity`, `Finding` and `InputQCReport` are reused from `input_qc`, not redefined.**
Redefining them here would let the two modules silently drift about what "REFUSE" means.

**Whether the QC-model and conformal signals may drive a REFUSE is a *scientific*
question, not an engineering one**, and it is answered in
`docs/research/preregistration_qc.md` and `configs/clinical/default.yaml`'s
`gatekeeper.enabled_signals` comment: those signals are wired in (or left out) by
config, from the *measured* result of Gate C, never from the hope that they will work.
This module enforces the mechanics of that decision -- an enabled signal that could not
be measured is a REFUSE, never a silent pass -- but does not itself decide which
signals are enabled.

**Label-free by construction.** `calibrate_thresholds` fits absolute cut points from
QUANTILES of the calibration split's own signal distribution -- never from a
ground-truth Dice. This project has already shipped a calibration reporting mask that
WAS defined from the label; it manufactured 41-57% of a reported ECE behind a green
test suite. `tests/test_gatekeeper.py::test_calibrate_thresholds_is_label_free`
introspects every public callable's signature and enforces this structurally, the same
way `input_qc`'s label-free test does.

**Direction is not configurable and lives one place per signal.** Predicted Dice is
LOW-is-bad (a low estimated Dice is a bad mask); conformal band width and the OOD score
are HIGH-is-bad (a wide band or a high unlikeness-to-calibration score is a bad case).
Getting one backwards silently inverts the gate for that signal, so each direction is
encoded in exactly one `judge_*` function, each with its own directional test.

No CUDA, no torch: this module runs no model, reads no image, and does no inference --
it only combines numbers that were computed elsewhere. numpy, pandas and this repo's
own `input_qc` module only, all already in `requirements.txt`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurovision.inference.input_qc import InputQCReport, Severity

logger = logging.getLogger(__name__)

# The four signals this gate knows how to judge. `run_gatekeeper` only ever
# consults these; an unrecognised name in `cfg.clinical.gatekeeper.enabled_signals`
# is simply never looked at, rather than raising, since a typo in a config that
# has not been re-run through calibration yet should not crash a demo session.
SIGNAL_NAMES: tuple[str, ...] = ("input_qc", "predicted_dice", "conformal_band", "ood_score")

# `calibrate_thresholds` warns (does not raise) when the surviving calibration
# count is at or below this -- "a handful of cases" per the spec this module was
# written against. A quantile estimated from that few points is not wrong, but
# reporting it as a frozen operating point without flagging the sample size is
# how a shaky threshold becomes an invisible one.
_MIN_STABLE_CALIBRATION_N = 10


class Decision(StrEnum):
    """The gate's three possible outcomes for one case.

    Ordered PROCEED < PROCEED_WITH_CAUTION < REFUSE. Combining verdicts (in
    `run_gatekeeper` and `GateDecision`) uses the explicit `_DECISION_RANK` map
    below, never `max()`/`sorted()` on the plain string values -- see
    `_worst_decision`'s docstring for why relying on that coincidence would be
    fragile even where it happens to hold.
    """

    PROCEED = "proceed"
    PROCEED_WITH_CAUTION = "proceed_with_caution"
    REFUSE = "refuse"


# Explicit rank, never derived from string order. `input_qc.Severity` has the
# analogous table because "REFUSE" < "WARN" alphabetically actually inverts that
# enum's true order; these three `Decision` values happen not to invert under
# plain string comparison (see the module's tests), but the fix is the same
# either way -- an ordering this important must be declared, not incidental.
_DECISION_RANK: dict[Decision, int] = {
    Decision.PROCEED: 0,
    Decision.PROCEED_WITH_CAUTION: 1,
    Decision.REFUSE: 2,
}

# input_qc's Severity maps onto Decision one level for level: OK is a pass,
# WARN is a caution, REFUSE is a refusal. Declared once here so judge_input_qc
# cannot drift from this mapping across two branches.
_SEVERITY_TO_DECISION: dict[Severity, Decision] = {
    Severity.OK: Decision.PROCEED,
    Severity.WARN: Decision.PROCEED_WITH_CAUTION,
    Severity.REFUSE: Decision.REFUSE,
}


def _worst_decision(decisions: Iterable[Decision]) -> Decision:
    """The worst (highest-ranked) `Decision` in `decisions`, or PROCEED if empty.

    Args:
        decisions: Any iterable of `Decision` values, e.g. one per `SignalVerdict`.

    Returns:
        The single worst decision, by `_DECISION_RANK`.
    """
    worst = Decision.PROCEED
    for decision in decisions:
        if _DECISION_RANK[decision] > _DECISION_RANK[worst]:
            worst = decision
    return worst


def _jsonable(value: Any) -> Any:
    """Recursively convert `value` to plain, `json.dumps`-safe Python types.

    Mirrors `input_qc._jsonable` -- a numpy `bool_` or `float32` surviving into
    a `SignalVerdict.detail` dict is exactly how a long pipeline dies at its
    very last line, so every `detail` is passed through this at `to_dict` time
    rather than trusting each `judge_*` function to remember to call `float(...)`.
    """
    if isinstance(value, Decision | Severity):
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


def _is_bad_number(value: Any) -> bool:
    """True if `value` is `None`, or a float that is NaN or +/-Inf.

    A NaN compared against a threshold with `<` or `>` is `False` either way --
    the exact silent-pass trap this module must not fall into (see CLAUDE.md's
    calibration-mask trap, of which this is a cousin). Every numeric `judge_*`
    check routes a value through this before comparing it to a threshold.
    """
    if value is None:
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


@dataclass(frozen=True)
class GateSignals:
    """Everything `run_gatekeeper` reads for one case. Any optional field may be `None`.

    Attributes:
        input_qc: The `InputQCReport` from `neurovision.inference.input_qc.run_input_qc`,
            or `None` if it could not be produced.
        predicted_dice: region -> the QC model's estimated Dice for that region's mask
            (Phase C), or `None` if unavailable. LOW is bad.
        conformal_band: region -> the conformal prediction band's width for that region
            (Phase B), or `None` if unavailable. HIGH is bad.
        ood_score: A single case-level out-of-distribution score, higher meaning more
            unlike the calibration set, or `None` if unavailable. HIGH is bad.
    """

    input_qc: InputQCReport | None = None
    predicted_dice: Mapping[str, float] | None = None
    conformal_band: Mapping[str, float] | None = None
    ood_score: float | None = None


@dataclass(frozen=True)
class SignalVerdict:
    """One signal's judgement, always emitted -- enabled or not, measured or not.

    Attributes:
        signal: One of `SIGNAL_NAMES`.
        decision: This signal's own `Decision`. Always `Decision.PROCEED` when
            `enabled` is `False` -- a disabled signal never contributes a caution
            or a refusal, it is simply not consulted.
        available: Whether the raw signal data was actually supplied, independent
            of `enabled` -- a signal can be available but disabled (measured, not
            trusted yet) or enabled but unavailable (trusted, but missing here).
        enabled: Whether `cfg.clinical.gatekeeper.enabled_signals` names this signal
            (always `True` for `input_qc`; see `run_gatekeeper`).
        message: Actionable prose, for a human operator or the UI's refusal banner.
        detail: The numbers behind the decision, JSON-serialisable once passed
            through `GateDecision.to_dict`.
    """

    signal: str
    decision: Decision
    available: bool
    enabled: bool
    message: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class GateDecision:
    """The composed outcome of `run_gatekeeper`.

    Attributes:
        decision: The worst `Decision` among `verdicts`.
        verdicts: One `SignalVerdict` per signal `run_gatekeeper` judged, in the
            order `SIGNAL_NAMES` lists them.
    """

    decision: Decision
    verdicts: tuple[SignalVerdict, ...]

    def refusals(self) -> tuple[SignalVerdict, ...]:
        """Verdicts with `decision is Decision.REFUSE`."""
        return tuple(v for v in self.verdicts if v.decision is Decision.REFUSE)

    def cautions(self) -> tuple[SignalVerdict, ...]:
        """Verdicts with `decision is Decision.PROCEED_WITH_CAUTION`."""
        return tuple(v for v in self.verdicts if v.decision is Decision.PROCEED_WITH_CAUTION)

    def to_dict(self) -> dict[str, Any]:
        """This decision as a plain, JSON-serialisable dict, for the job manifest and the UI."""
        return {
            "decision": self.decision.value,
            "verdicts": [
                {
                    "signal": v.signal,
                    "decision": v.decision.value,
                    "available": bool(v.available),
                    "enabled": bool(v.enabled),
                    "message": v.message,
                    "detail": _jsonable(v.detail),
                }
                for v in self.verdicts
            ],
        }


@dataclass(frozen=True)
class Thresholds:
    """Absolute cut points for the numeric signals, derived from calibration quantiles and FROZEN.

    Produced once by `calibrate_thresholds` on the frozen calibration split, written to
    `cfg.clinical.gatekeeper.thresholds` (a JSON file path or an inline mapping), and
    read back by `load_thresholds`. Never recomputed inside `run_gatekeeper` itself --
    a threshold invented at inference time is exactly the "made-up threshold that looks
    plausible" `configs/clinical/default.yaml`'s comment on this block warns against.

    Attributes:
        predicted_dice: region -> `(refuse_below, caution_below)`. A value below
            `refuse_below` REFUSEs; below `caution_below` (and at or above
            `refuse_below`) CAUTIONs; at or above `caution_below` PROCEEDs.
        conformal_band: region -> `(caution_above, refuse_above)`. A value above
            `refuse_above` REFUSEs; above `caution_above` (and at or below
            `refuse_above`) CAUTIONs; at or below `caution_above` PROCEEDs.
        ood_score: `(caution_above, refuse_above)`, same reading as `conformal_band`
            but case-level rather than per-region.
        calibration_n: The number of calibration cases the *smallest* contributing
            column actually had (after dropping NaNs) -- see `calibrate_thresholds`.
        caution_quantile: The quantile used for every CAUTION cut point.
        refuse_quantile: The quantile used for every REFUSE cut point.
    """

    predicted_dice: dict[str, tuple[float, float]]
    conformal_band: dict[str, tuple[float, float]]
    ood_score: tuple[float, float]
    calibration_n: int
    caution_quantile: float
    refuse_quantile: float

    def to_dict(self) -> dict[str, Any]:
        """This object as a plain, JSON-serialisable dict. See `from_dict` for the inverse."""
        return {
            "predicted_dice": {
                region: [float(pair[0]), float(pair[1])]
                for region, pair in self.predicted_dice.items()
            },
            "conformal_band": {
                region: [float(pair[0]), float(pair[1])]
                for region, pair in self.conformal_band.items()
            },
            "ood_score": [float(self.ood_score[0]), float(self.ood_score[1])],
            "calibration_n": int(self.calibration_n),
            "caution_quantile": float(self.caution_quantile),
            "refuse_quantile": float(self.refuse_quantile),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Thresholds:
        """Rebuild a `Thresholds` from `to_dict`'s output (or an equivalent JSON payload)."""

        def _pair(raw: Sequence[float]) -> tuple[float, float]:
            return (float(raw[0]), float(raw[1]))

        return cls(
            predicted_dice={
                str(region): _pair(pair) for region, pair in payload["predicted_dice"].items()
            },
            conformal_band={
                str(region): _pair(pair) for region, pair in payload["conformal_band"].items()
            },
            ood_score=_pair(payload["ood_score"]),
            calibration_n=int(payload["calibration_n"]),
            caution_quantile=float(payload["caution_quantile"]),
            refuse_quantile=float(payload["refuse_quantile"]),
        )


def calibrate_thresholds(
    table: pd.DataFrame,
    *,
    regions: Sequence[str],
    caution_quantile: float,
    refuse_quantile: float,
) -> Thresholds:
    """Fit absolute `Thresholds` from quantiles of a calibration split's own signal distribution.

    Label-free by construction: this function takes no ground-truth Dice or any other
    label column, only the calibration table's OWN signal values --
    `tests/test_gatekeeper.py::test_calibrate_thresholds_is_label_free` enforces this
    structurally by introspecting every public callable's signature. Reads columns
    `predicted_dice_<R>` and `conformal_band_<R>` for every `R` in `regions`, plus a
    single `ood_score` column, all required to be present in `table`.

    Args:
        table: One row per calibration case. Must contain `predicted_dice_<R>` and
            `conformal_band_<R>` for every region in `regions`, plus `ood_score`.
        regions: The regions to fit `predicted_dice` and `conformal_band` cut points
            for, e.g. `("WT", "TC")` from `cfg.clinical.gatekeeper.regions`. Must be
            non-empty.
        caution_quantile: The quantile defining the CAUTION cut point, e.g. `0.10`.
        refuse_quantile: The quantile defining the REFUSE cut point, e.g. `0.02`. Must
            be strictly less than `caution_quantile` -- the REFUSE band is the more
            extreme tail, so a rarer quantile.

    Returns:
        The fitted, frozen `Thresholds`.

    Raises:
        ValueError: If `regions` is empty; if `caution_quantile` or `refuse_quantile`
            is outside `(0, 1)`; if `refuse_quantile >= caution_quantile` (the REFUSE
            band would swallow CAUTION); if a required column for a requested region
            (or `ood_score`) is absent from `table`; or if a required column is
            entirely NaN (a NaN threshold would make every future comparison silently
            `False` -- see `_is_bad_number`).
    """
    if not regions:
        raise ValueError("calibrate_thresholds: `regions` must not be empty.")
    if not 0.0 < caution_quantile < 1.0:
        raise ValueError(
            f"calibrate_thresholds: caution_quantile must be in (0, 1), got {caution_quantile!r}."
        )
    if not 0.0 < refuse_quantile < 1.0:
        raise ValueError(
            f"calibrate_thresholds: refuse_quantile must be in (0, 1), got {refuse_quantile!r}."
        )
    if refuse_quantile >= caution_quantile:
        raise ValueError(
            "calibrate_thresholds: refuse_quantile must be strictly less than "
            f"caution_quantile (got refuse_quantile={refuse_quantile!r}, "
            f"caution_quantile={caution_quantile!r}); otherwise the REFUSE band would "
            "swallow the CAUTION band."
        )

    def _column(name: str) -> pd.Series:
        if name not in table.columns:
            raise ValueError(f"calibrate_thresholds: required column {name!r} is absent.")
        series = table[name].dropna()
        if series.empty:
            raise ValueError(
                f"calibrate_thresholds: column {name!r} is entirely NaN; a threshold "
                "cannot be calibrated from it."
            )
        return series

    calibration_ns: list[int] = []

    predicted_dice: dict[str, tuple[float, float]] = {}
    for region in regions:
        series = _column(f"predicted_dice_{region}")
        calibration_ns.append(len(series))
        # Low tail: the REFUSE cut is the rarer, more extreme (lower) quantile.
        predicted_dice[region] = (
            float(series.quantile(refuse_quantile)),
            float(series.quantile(caution_quantile)),
        )

    conformal_band: dict[str, tuple[float, float]] = {}
    for region in regions:
        series = _column(f"conformal_band_{region}")
        calibration_ns.append(len(series))
        # High tail: mirror of predicted_dice, so the REFUSE cut is 1 - refuse_quantile.
        conformal_band[region] = (
            float(series.quantile(1.0 - caution_quantile)),
            float(series.quantile(1.0 - refuse_quantile)),
        )

    ood_series = _column("ood_score")
    calibration_ns.append(len(ood_series))
    ood_score = (
        float(ood_series.quantile(1.0 - caution_quantile)),
        float(ood_series.quantile(1.0 - refuse_quantile)),
    )

    # The smallest surviving column, not the largest: a threshold is only as
    # trustworthy as its worst-supported column, and reporting the max here
    # would hide exactly the column that most needs the "tiny calibration set"
    # warning below.
    calibration_n = min(calibration_ns)
    if calibration_n <= _MIN_STABLE_CALIBRATION_N:
        logger.warning(
            "calibrate_thresholds: calibration_n=%d is a handful of cases; a quantile "
            "estimated from that few is not a stable operating point.",
            calibration_n,
        )

    return Thresholds(
        predicted_dice=predicted_dice,
        conformal_band=conformal_band,
        ood_score=ood_score,
        calibration_n=calibration_n,
        caution_quantile=float(caution_quantile),
        refuse_quantile=float(refuse_quantile),
    )


def judge_input_qc(report: InputQCReport | None, *, enabled: bool) -> SignalVerdict:
    """Judge the input QC report: `input_qc.Severity` maps directly onto `Decision`.

    Args:
        report: The `InputQCReport` from `run_input_qc`, or `None` if it could not be
            produced.
        enabled: Whether this signal is consulted. `run_gatekeeper` always passes
            `True` here -- input QC is the only signal that can detect a study the
            model must never see at all, and cannot be turned off by config.

    Returns:
        A `"input_qc"` `SignalVerdict`. REFUSE if `enabled` and `report is None`;
        otherwise `report.verdict` mapped through `_SEVERITY_TO_DECISION`. Every
        REFUSE and WARN finding's check id and message is carried into `detail` so
        the reason survives into `GateDecision.to_dict()` for the UI's refusal banner.
    """
    if not enabled:
        return SignalVerdict(
            signal="input_qc",
            decision=Decision.PROCEED,
            available=report is not None,
            enabled=False,
            message="input_qc was not enabled; not consulted.",
            detail={},
        )
    if report is None:
        return SignalVerdict(
            signal="input_qc",
            decision=Decision.REFUSE,
            available=False,
            enabled=True,
            message=(
                "input_qc is enabled but no InputQCReport was supplied; a study cannot "
                "be admitted without it."
            ),
            detail={},
        )

    refusal_reasons = [f"{f.check}: {f.message}" for f in report.refusals()]
    warning_reasons = [f"{f.check}: {f.message}" for f in report.warnings()]
    detail = {
        "input_qc_verdict": report.verdict.value,
        "refusal_reasons": refusal_reasons,
        "warning_reasons": warning_reasons,
    }
    if refusal_reasons:
        message = f"Input QC refused: {'; '.join(refusal_reasons)}"
    elif warning_reasons:
        message = f"Input QC warned: {'; '.join(warning_reasons)}"
    else:
        message = "Input QC passed all checks."

    return SignalVerdict(
        signal="input_qc",
        decision=_SEVERITY_TO_DECISION[report.verdict],
        available=True,
        enabled=True,
        message=message,
        detail=detail,
    )


def _judge_region_signal(
    *,
    signal: str,
    values: Mapping[str, float] | None,
    thresholds: Mapping[str, tuple[float, float]] | None,
    regions: Sequence[str],
    enabled: bool,
    low_is_bad: bool,
) -> SignalVerdict:
    """Shared machinery for `judge_predicted_dice` and `judge_conformal_band`.

    Both are "the worst across a per-region mapping, judged against a per-region
    absolute threshold pair" -- the only difference is which side of the pair is
    the REFUSE cut and which direction is bad. That direction is the ONE thing
    kept out of this shared function and fixed in each public `judge_*` wrapper
    instead, so a reader auditing the low-is-bad/high-is-bad question never has
    to trace it through a shared branch -- see the module docstring.

    Args:
        signal: `"predicted_dice"` or `"conformal_band"`, for messages and the
            returned `SignalVerdict.signal`.
        values: region -> measured value, or `None` if unmeasured.
        thresholds: region -> `(low_cut, high_cut)`, or `None` if uncalibrated.
        regions: The regions to judge. Must be non-empty when `enabled`.
        enabled: Whether this signal is consulted.
        low_is_bad: `True` for predicted_dice (a REFUSE below `low_cut`, i.e. the
            first element of the pair is `refuse_below`); `False` for
            conformal_band and ood_score (a REFUSE above `high_cut`, the second
            element of the pair is `refuse_above`).

    Returns:
        A `SignalVerdict` whose decision is the worst across `regions`.

    Raises:
        ValueError: If `enabled` and `regions` is empty, or if `enabled` and any
            region in `regions` has no entry in `thresholds` (or `thresholds` is
            `None` outright) -- a missing calibrated threshold is never invented.
    """
    if not enabled:
        return SignalVerdict(
            signal=signal,
            decision=Decision.PROCEED,
            available=values is not None,
            enabled=False,
            message=f"{signal} was not enabled; not consulted.",
            detail={},
        )
    if not regions:
        raise ValueError(f"{signal}: `regions` must not be empty when this signal is enabled.")

    for region in regions:
        if thresholds is None or region not in thresholds:
            raise ValueError(
                f"{signal}: no calibrated threshold for region {region!r} (thresholds is "
                "None, or missing this region). Run calibrate_thresholds / "
                "scripts/calibrate_gatekeeper.py before enabling this signal -- a "
                "threshold is never invented at inference time."
            )

    if values is None:
        return SignalVerdict(
            signal=signal,
            decision=Decision.REFUSE,
            available=False,
            enabled=True,
            message=f"{signal} is enabled but was not measured for this case.",
            detail={"regions": list(regions)},
        )

    per_region_decision: dict[str, Decision] = {}
    per_region_detail: dict[str, dict[str, Any]] = {}
    for region in regions:
        low_cut, high_cut = thresholds[region]
        if region not in values:
            per_region_decision[region] = Decision.REFUSE
            per_region_detail[region] = {"reason": "region missing from measured values"}
            continue
        value = values[region]
        if _is_bad_number(value):
            per_region_decision[region] = Decision.REFUSE
            per_region_detail[region] = {"value": None, "reason": "value is missing or non-finite"}
            continue
        value = float(value)
        if low_is_bad:
            # predicted_dice: low_cut = refuse_below, high_cut = caution_below.
            if value < low_cut:
                decision = Decision.REFUSE
            elif value < high_cut:
                decision = Decision.PROCEED_WITH_CAUTION
            else:
                decision = Decision.PROCEED
        else:
            # conformal_band: low_cut = caution_above, high_cut = refuse_above.
            if value > high_cut:
                decision = Decision.REFUSE
            elif value > low_cut:
                decision = Decision.PROCEED_WITH_CAUTION
            else:
                decision = Decision.PROCEED
        per_region_decision[region] = decision
        per_region_detail[region] = {
            "value": value,
            "low_cut": float(low_cut),
            "high_cut": float(high_cut),
            "decision": decision.value,
        }

    worst = _worst_decision(per_region_decision.values())
    bad_regions = [r for r, d in per_region_decision.items() if d is not Decision.PROCEED]
    message = f"{signal}: worst region decision is {worst.value}"
    if bad_regions:
        message += f" ({', '.join(bad_regions)})"
    return SignalVerdict(
        signal=signal,
        decision=worst,
        available=True,
        enabled=True,
        message=message,
        detail={"per_region": per_region_detail},
    )


def judge_predicted_dice(
    values: Mapping[str, float] | None,
    thresholds: Mapping[str, tuple[float, float]] | None,
    regions: Sequence[str],
    *,
    enabled: bool,
) -> SignalVerdict:
    """Judge the QC model's per-region predicted Dice. LOW is bad.

    Args:
        values: region -> the QC model's predicted Dice, or `None` if unmeasured.
        thresholds: region -> `(refuse_below, caution_below)`, i.e.
            `Thresholds.predicted_dice`, or `None` if uncalibrated.
        regions: The regions to judge, e.g. `cfg.clinical.gatekeeper.regions`.
        enabled: Whether this signal is consulted.

    Returns:
        A `"predicted_dice"` `SignalVerdict`, worst across `regions`.
    """
    return _judge_region_signal(
        signal="predicted_dice",
        values=values,
        thresholds=thresholds,
        regions=regions,
        enabled=enabled,
        low_is_bad=True,
    )


def judge_conformal_band(
    values: Mapping[str, float] | None,
    thresholds: Mapping[str, tuple[float, float]] | None,
    regions: Sequence[str],
    *,
    enabled: bool,
) -> SignalVerdict:
    """Judge the conformal prediction band's per-region width. HIGH is bad.

    Args:
        values: region -> the band width, or `None` if unmeasured.
        thresholds: region -> `(caution_above, refuse_above)`, i.e.
            `Thresholds.conformal_band`, or `None` if uncalibrated.
        regions: The regions to judge, e.g. `cfg.clinical.gatekeeper.regions`.
        enabled: Whether this signal is consulted.

    Returns:
        A `"conformal_band"` `SignalVerdict`, worst across `regions`.
    """
    return _judge_region_signal(
        signal="conformal_band",
        values=values,
        thresholds=thresholds,
        regions=regions,
        enabled=enabled,
        low_is_bad=False,
    )


def judge_ood_score(
    value: float | None,
    thresholds: tuple[float, float] | None,
    *,
    enabled: bool,
) -> SignalVerdict:
    """Judge the case-level out-of-distribution score. HIGH is bad.

    Args:
        value: The OOD score for this case, or `None` if unmeasured.
        thresholds: `(caution_above, refuse_above)`, i.e. `Thresholds.ood_score`, or
            `None` if uncalibrated.
        enabled: Whether this signal is consulted.

    Returns:
        A `"ood_score"` `SignalVerdict`.

    Raises:
        ValueError: If `enabled` and `thresholds is None` -- a threshold is never
            invented at inference time.
    """
    signal = "ood_score"
    if not enabled:
        return SignalVerdict(
            signal=signal,
            decision=Decision.PROCEED,
            available=value is not None,
            enabled=False,
            message=f"{signal} was not enabled; not consulted.",
            detail={},
        )
    if thresholds is None:
        raise ValueError(
            f"{signal}: no calibrated threshold (thresholds is None). Run "
            "calibrate_thresholds / scripts/calibrate_gatekeeper.py before enabling "
            "this signal -- a threshold is never invented at inference time."
        )
    if value is None:
        return SignalVerdict(
            signal=signal,
            decision=Decision.REFUSE,
            available=False,
            enabled=True,
            message=f"{signal} is enabled but was not measured for this case.",
            detail={},
        )
    if _is_bad_number(value):
        return SignalVerdict(
            signal=signal,
            decision=Decision.REFUSE,
            available=True,
            enabled=True,
            message=f"{signal} is enabled but its value is non-finite.",
            detail={"value": None},
        )

    value = float(value)
    caution_above, refuse_above = thresholds
    if value > refuse_above:
        decision = Decision.REFUSE
    elif value > caution_above:
        decision = Decision.PROCEED_WITH_CAUTION
    else:
        decision = Decision.PROCEED

    return SignalVerdict(
        signal=signal,
        decision=decision,
        available=True,
        enabled=True,
        message=f"{signal}: {decision.value} (value={value}).",
        detail={
            "value": value,
            "caution_above": float(caution_above),
            "refuse_above": float(refuse_above),
            "decision": decision.value,
        },
    )


def load_thresholds(cfg: Any) -> Thresholds | None:
    """Load calibrated `Thresholds` from `cfg.clinical.gatekeeper.thresholds`.

    Args:
        cfg: The root config, exposing `cfg.clinical.gatekeeper.thresholds` as one
            of: `None` (uncalibrated), a mapping (as `Thresholds.to_dict()` produces),
            or a string path to a JSON file holding that same mapping.

    Returns:
        The loaded `Thresholds`, or `None` if `cfg.clinical.gatekeeper.thresholds`
        is `None`.

    Raises:
        FileNotFoundError: If `cfg.clinical.gatekeeper.thresholds` is a string path
            and no file exists there. Names the path.
        TypeError: If `cfg.clinical.gatekeeper.thresholds` is neither `None`, a
            mapping, nor a string.
    """
    raw = cfg.clinical.gatekeeper.thresholds
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return Thresholds.from_dict(raw)
    if isinstance(raw, str):
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"load_thresholds: thresholds file not found: {path}")
        payload = json.loads(path.read_text())
        return Thresholds.from_dict(payload)
    raise TypeError(
        "load_thresholds: cfg.clinical.gatekeeper.thresholds must be None, a mapping, "
        f"or a path string, got {type(raw)!r}."
    )


def run_gatekeeper(
    cfg: Any,
    signals: GateSignals,
    thresholds: Thresholds | None = None,
) -> GateDecision:
    """Judge every signal and combine the worst into one `GateDecision`.

    Reads `cfg.clinical.gatekeeper` (see `configs/clinical/default.yaml`) -- NOT
    `cfg.gatekeeper`, which does not exist at the composed config path.

    `input_qc` is always judged as enabled, regardless of
    `cfg.clinical.gatekeeper.enabled_signals`: it is the only signal that can detect
    a study the model must never see at all, so omitting it from config is treated
    as an oversight, logged, and overridden rather than honoured.

    Args:
        cfg: The root config, exposing `cfg.clinical.gatekeeper` with
            `enabled_signals` and `regions`.
        signals: The measured (or unmeasured) signals for this case.
        thresholds: The calibrated `Thresholds` to judge the numeric signals
            against. If `None` (the default), loaded from `cfg` via
            `load_thresholds` -- pass an explicit `Thresholds` to bypass `cfg`
            entirely (e.g. in a test, or a calibration dry run).

    Returns:
        The combined `GateDecision`: `decision` is the worst across all four
        signals' `SignalVerdict`s, `verdicts` holds one per signal in
        `SIGNAL_NAMES` order.

    Raises:
        ValueError: Propagated from a `judge_*` call -- an enabled numeric signal
            with an empty `regions` list, or with no calibrated threshold.
    """
    gk_cfg = cfg.clinical.gatekeeper
    configured = {str(name) for name in gk_cfg.enabled_signals}
    if "input_qc" not in configured:
        logger.warning(
            "run_gatekeeper: cfg.clinical.gatekeeper.enabled_signals omits 'input_qc'; "
            "enabling it anyway -- it is the only signal that can detect a study the "
            "model must never see at all."
        )

    regions = [str(r) for r in gk_cfg.regions]

    if thresholds is None:
        thresholds = load_thresholds(cfg)

    pd_thresholds = thresholds.predicted_dice if thresholds is not None else None
    cb_thresholds = thresholds.conformal_band if thresholds is not None else None
    ood_thresholds = thresholds.ood_score if thresholds is not None else None

    verdicts = (
        judge_input_qc(signals.input_qc, enabled=True),
        judge_predicted_dice(
            signals.predicted_dice,
            pd_thresholds,
            regions,
            enabled="predicted_dice" in configured,
        ),
        judge_conformal_band(
            signals.conformal_band,
            cb_thresholds,
            regions,
            enabled="conformal_band" in configured,
        ),
        judge_ood_score(
            signals.ood_score,
            ood_thresholds,
            enabled="ood_score" in configured,
        ),
    )

    overall = _worst_decision(v.decision for v in verdicts)
    if overall is Decision.REFUSE:
        logger.warning(
            "run_gatekeeper: REFUSE -- %s",
            "; ".join(
                f"{v.signal}: {v.message}" for v in verdicts if v.decision is Decision.REFUSE
            ),
        )
    else:
        logger.debug("run_gatekeeper: decision=%s", overall.value)

    return GateDecision(decision=overall, verdicts=verdicts)
