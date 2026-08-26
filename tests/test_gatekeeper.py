"""Tests for `neurovision.inference.gatekeeper`.

Milestone 4, Phase E, task E5 (the refusal gate). Everything here is synthetic --
`SimpleNamespace` stand-ins for config (the same pattern `tests/test_input_qc.py`
uses) and tiny hand-built `pandas.DataFrame`s -- never real patient or BraTS data,
and every test runs in well under a second on CPU.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neurovision.inference import gatekeeper
from neurovision.inference.gatekeeper import (
    Decision,
    GateDecision,
    GateSignals,
    SignalVerdict,
    Thresholds,
    calibrate_thresholds,
    judge_conformal_band,
    judge_input_qc,
    judge_ood_score,
    judge_predicted_dice,
    load_thresholds,
    run_gatekeeper,
)
from neurovision.inference.input_qc import Finding, InputQCReport, Severity

# Real configs/ directory, resolved relative to this file -- same pattern as
# tests/test_input_qc.py, so the "reachable at the composed path" test composes
# the PROJECT's actual config, not a hand-built stand-in.
_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    enabled_signals: list[str],
    regions: list[str],
    thresholds: object = None,
) -> SimpleNamespace:
    """A minimal stand-in for the real config, exposing exactly `cfg.clinical.gatekeeper`."""
    gk = SimpleNamespace(
        enabled_signals=list(enabled_signals),
        regions=list(regions),
        thresholds=thresholds,
    )
    return SimpleNamespace(clinical=SimpleNamespace(gatekeeper=gk))


def _report(severity: Severity, message: str = "finding message") -> InputQCReport:
    """A one-finding `InputQCReport` at the given severity."""
    finding = Finding(check="sequence_completeness", severity=severity, message=message, detail={})
    return InputQCReport(verdict=severity, findings=(finding,))


def _thresholds(regions: list[str]) -> Thresholds:
    """A simple, hand-computed `Thresholds` fixture, one cut-point pair per region."""
    return Thresholds(
        predicted_dice={r: (0.5, 0.7) for r in regions},
        conformal_band={r: (0.1, 0.3) for r in regions},
        ood_score=(0.1, 0.3),
        calibration_n=187,
        caution_quantile=0.10,
        refuse_quantile=0.02,
    )


# ---------------------------------------------------------------------------
# 1. Label-free by construction
# ---------------------------------------------------------------------------


def test_calibrate_thresholds_is_label_free() -> None:
    """No public callable in this module may take a ground-truth-shaped parameter.

    Mirrors `tests/test_input_qc.py::test_no_function_in_this_module_takes_a_label` --
    this project has already shipped a calibration mask defined from the label, and
    the fix that stuck was making it structurally impossible, not a code-review rule.
    """
    forbidden = {"label", "labels", "gt", "ground_truth", "y_true", "target", "true_dice"}
    checked = 0
    for name, obj in inspect.getmembers(gatekeeper, inspect.isfunction):
        if name.startswith("_") or obj.__module__ != gatekeeper.__name__:
            continue
        checked += 1
        params = set(inspect.signature(obj).parameters)
        offending = params & forbidden
        assert not offending, f"{name} takes forbidden parameter(s) {offending}"
    assert checked > 0


# ---------------------------------------------------------------------------
# 2. calibrate_thresholds arithmetic
# ---------------------------------------------------------------------------


def test_calibrate_thresholds_quantiles() -> None:
    """A known table gives the arithmetically expected cut points for all three signals.

    101 evenly spaced values `i / 100` for `i in 0..100` make `quantile(q)` land
    exactly on `q * 100 / 100 = q` under pandas' default linear interpolation
    (index = q * (n - 1) = q * 100, an integer for every quantile used here), so the
    expected cut points are just the quantiles themselves -- no numerical fudging.
    """
    base = np.arange(101) / 100.0  # 0.00, 0.01, ..., 1.00
    table = pd.DataFrame(
        {
            "predicted_dice_WT": base,
            "predicted_dice_TC": base + 5.0,  # shifted, to prove regions are independent
            "conformal_band_WT": base,
            "conformal_band_TC": base + 5.0,
            "ood_score": base,
        }
    )

    thresholds = calibrate_thresholds(
        table, regions=["WT", "TC"], caution_quantile=0.10, refuse_quantile=0.02
    )

    assert thresholds.predicted_dice["WT"] == pytest.approx((0.02, 0.10))
    assert thresholds.predicted_dice["TC"] == pytest.approx((5.02, 5.10))
    assert thresholds.conformal_band["WT"] == pytest.approx((0.90, 0.98))
    assert thresholds.conformal_band["TC"] == pytest.approx((5.90, 5.98))
    assert thresholds.ood_score == pytest.approx((0.90, 0.98))
    assert thresholds.calibration_n == 101
    assert thresholds.caution_quantile == pytest.approx(0.10)
    assert thresholds.refuse_quantile == pytest.approx(0.02)


def test_calibrate_thresholds_rejects_crossed_quantiles() -> None:
    table = pd.DataFrame(
        {"predicted_dice_WT": np.arange(20) / 20.0, "ood_score": np.arange(20) / 20.0}
    )
    with pytest.raises(ValueError, match="refuse_quantile"):
        calibrate_thresholds(table, regions=["WT"], caution_quantile=0.10, refuse_quantile=0.10)
    with pytest.raises(ValueError, match="refuse_quantile"):
        # Crossed the other way: refuse rarer than caution is required, not just unequal.
        calibrate_thresholds(table, regions=["WT"], caution_quantile=0.05, refuse_quantile=0.10)


def test_calibrate_thresholds_rejects_quantiles_out_of_range() -> None:
    table = pd.DataFrame(
        {"predicted_dice_WT": np.arange(20) / 20.0, "ood_score": np.arange(20) / 20.0}
    )
    with pytest.raises(ValueError, match="caution_quantile"):
        calibrate_thresholds(table, regions=["WT"], caution_quantile=1.5, refuse_quantile=0.02)
    with pytest.raises(ValueError, match="refuse_quantile"):
        calibrate_thresholds(table, regions=["WT"], caution_quantile=0.10, refuse_quantile=0.0)
    with pytest.raises(ValueError, match="refuse_quantile"):
        calibrate_thresholds(table, regions=["WT"], caution_quantile=0.10, refuse_quantile=-0.1)


def test_calibrate_thresholds_rejects_an_all_nan_column() -> None:
    table = pd.DataFrame(
        {
            "predicted_dice_WT": [float("nan")] * 10,
            "conformal_band_WT": np.arange(10) / 10.0,
            "ood_score": np.arange(10) / 10.0,
        }
    )
    with pytest.raises(ValueError, match="predicted_dice_WT"):
        calibrate_thresholds(table, regions=["WT"], caution_quantile=0.10, refuse_quantile=0.02)


def test_calibrate_thresholds_warns_on_a_tiny_calibration_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = pd.DataFrame(
        {
            "predicted_dice_WT": np.linspace(0.0, 1.0, 5),
            "conformal_band_WT": np.linspace(0.0, 1.0, 5),
            "ood_score": np.linspace(0.0, 1.0, 5),
        }
    )
    with caplog.at_level(logging.WARNING, logger="neurovision.inference.gatekeeper"):
        thresholds = calibrate_thresholds(
            table, regions=["WT"], caution_quantile=0.10, refuse_quantile=0.02
        )
    assert thresholds.calibration_n == 5
    assert any("calibration_n=5" in rec.message for rec in caplog.records)


def test_calibrate_thresholds_rejects_empty_regions() -> None:
    table = pd.DataFrame({"ood_score": np.arange(10) / 10.0})
    with pytest.raises(ValueError, match="regions"):
        calibrate_thresholds(table, regions=[], caution_quantile=0.10, refuse_quantile=0.02)


def test_calibrate_thresholds_rejects_a_missing_region_column() -> None:
    table = pd.DataFrame({"ood_score": np.arange(10) / 10.0})
    with pytest.raises(ValueError, match="predicted_dice_WT"):
        calibrate_thresholds(table, regions=["WT"], caution_quantile=0.10, refuse_quantile=0.02)


# ---------------------------------------------------------------------------
# 3. Directions -- the decisive test
# ---------------------------------------------------------------------------


def test_signal_directions_are_not_flipped() -> None:
    """For each numeric signal, a bad-side value REFUSEs and a good-side value PROCEEDs.

    Construction, so that flipping either comparison operator inside any ONE of
    `judge_predicted_dice`, `judge_conformal_band` or `judge_ood_score` breaks this
    test:

    - `predicted_dice` thresholds are `(refuse_below=0.5, caution_below=0.7)`.
      `0.3` is REFUSE because `0.3 < 0.5`; flip that `<` to `>` and `0.3` falls
      through to the CAUTION branch (`0.3 < 0.7`) instead, so the REFUSE assertion
      fails. `0.95` is PROCEED because it is below neither cut; flip the CAUTION
      branch's `<` to `>` and `0.95 > 0.7` fires, so the PROCEED assertion fails.
    - `conformal_band` and `ood_score` are the mirror image, thresholds
      `(caution_above=0.1, refuse_above=0.3)`. `0.5` is REFUSE because `0.5 > 0.3`;
      flip that `>` to `<` and it falls through to CAUTION instead. `0.02` is
      PROCEED because it is above neither cut; flip the CAUTION branch's `>` to
      `<` and `0.02 < 0.1` fires, so the PROCEED assertion fails.

    A direction swapped wholesale (e.g. `judge_conformal_band` accidentally using
    predicted_dice's low-is-bad logic) is also caught here: under that logic `0.5`
    (the conformal_band bad value) would be neither below `0.1` nor below `0.3`,
    so it would PROCEED instead of the REFUSE this test asserts.
    """
    pd_thresholds = {"WT": (0.5, 0.7)}
    pd_bad = judge_predicted_dice({"WT": 0.3}, pd_thresholds, ["WT"], enabled=True)
    pd_good = judge_predicted_dice({"WT": 0.95}, pd_thresholds, ["WT"], enabled=True)
    assert pd_bad.decision is Decision.REFUSE
    assert pd_good.decision is Decision.PROCEED

    cb_thresholds = {"WT": (0.1, 0.3)}
    cb_bad = judge_conformal_band({"WT": 0.5}, cb_thresholds, ["WT"], enabled=True)
    cb_good = judge_conformal_band({"WT": 0.02}, cb_thresholds, ["WT"], enabled=True)
    assert cb_bad.decision is Decision.REFUSE
    assert cb_good.decision is Decision.PROCEED

    ood_thresholds = (0.1, 0.3)
    ood_bad = judge_ood_score(0.5, ood_thresholds, enabled=True)
    ood_good = judge_ood_score(0.02, ood_thresholds, enabled=True)
    assert ood_bad.decision is Decision.REFUSE
    assert ood_good.decision is Decision.PROCEED


# ---------------------------------------------------------------------------
# 7-9, 11-12. judge_* edge cases
# ---------------------------------------------------------------------------


def test_enabled_signal_that_is_none_refuses() -> None:
    verdict = judge_predicted_dice(None, {"WT": (0.5, 0.7)}, ["WT"], enabled=True)
    assert verdict.decision is Decision.REFUSE
    assert verdict.available is False
    assert verdict.enabled is True

    assert (
        judge_conformal_band(None, {"WT": (0.1, 0.3)}, ["WT"], enabled=True).decision
        is Decision.REFUSE
    )
    assert judge_ood_score(None, (0.1, 0.3), enabled=True).decision is Decision.REFUSE
    assert judge_input_qc(None, enabled=True).decision is Decision.REFUSE


def test_disabled_signal_is_reported_not_omitted() -> None:
    cfg = _make_cfg(enabled_signals=["input_qc"], regions=["WT", "TC"])
    signals = GateSignals(
        input_qc=_report(Severity.OK),
        predicted_dice=None,
        conformal_band={"WT": 0.9, "TC": 0.9},  # would REFUSE if consulted
        ood_score=None,
    )
    decision = run_gatekeeper(cfg, signals, thresholds=_thresholds(["WT", "TC"]))

    assert decision.decision is Decision.PROCEED  # the disabled REFUSE-shaped data changes nothing
    by_signal = {v.signal: v for v in decision.verdicts}
    assert set(by_signal) == {"input_qc", "predicted_dice", "conformal_band", "ood_score"}
    cb = by_signal["conformal_band"]
    assert cb.enabled is False
    assert cb.decision is Decision.PROCEED
    assert cb.available is True  # data WAS supplied, just not consulted


def test_input_qc_is_enabled_even_when_the_config_omits_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _make_cfg(enabled_signals=[], regions=["WT"])  # omits input_qc entirely
    signals = GateSignals(input_qc=_report(Severity.REFUSE, "Missing required sequence(s): flair."))
    with caplog.at_level(logging.WARNING, logger="neurovision.inference.gatekeeper"):
        decision = run_gatekeeper(cfg, signals)

    by_signal = {v.signal: v for v in decision.verdicts}
    assert by_signal["input_qc"].enabled is True
    assert decision.decision is Decision.REFUSE
    assert any("input_qc" in rec.message for rec in caplog.records)


def test_missing_threshold_raises_rather_than_inventing_one() -> None:
    cfg = _make_cfg(enabled_signals=["input_qc", "predicted_dice"], regions=["WT"], thresholds=None)
    signals = GateSignals(input_qc=_report(Severity.OK), predicted_dice={"WT": 0.9})
    with pytest.raises(ValueError, match="predicted_dice"):
        run_gatekeeper(cfg, signals, thresholds=None)

    # Same trap at the judge_* level directly.
    with pytest.raises(ValueError, match="ood_score"):
        judge_ood_score(0.5, None, enabled=True)
    with pytest.raises(ValueError, match="TC"):
        judge_predicted_dice({"WT": 0.9, "TC": 0.9}, {"WT": (0.5, 0.7)}, ["WT", "TC"], enabled=True)


def test_nan_predicted_dice_refuses() -> None:
    """The silent-pass trap: NaN compared with `<` is `False` on both sides."""
    verdict = judge_predicted_dice({"WT": float("nan")}, {"WT": (0.5, 0.7)}, ["WT"], enabled=True)
    assert verdict.decision is Decision.REFUSE
    assert verdict.detail["per_region"]["WT"]["value"] is None


def test_missing_region_refuses_naming_it() -> None:
    verdict = judge_predicted_dice(
        {"TC": 0.95}, {"WT": (0.5, 0.7), "TC": (0.5, 0.7)}, ["WT", "TC"], enabled=True
    )
    assert verdict.decision is Decision.REFUSE
    assert "WT" in verdict.message
    assert "missing" in verdict.detail["per_region"]["WT"]["reason"]
    # TC itself was fine -- only WT should be blamed.
    assert verdict.detail["per_region"]["TC"]["decision"] == Decision.PROCEED.value


def test_judge_region_signal_rejects_empty_regions() -> None:
    with pytest.raises(ValueError, match="regions"):
        judge_predicted_dice({"WT": 0.9}, {"WT": (0.5, 0.7)}, [], enabled=True)
    with pytest.raises(ValueError, match="regions"):
        judge_conformal_band({"WT": 0.1}, {"WT": (0.1, 0.3)}, [], enabled=True)


# ---------------------------------------------------------------------------
# 13. Worst-of ordering
# ---------------------------------------------------------------------------


def test_decision_is_the_worst_severity() -> None:
    """The overall decision is the worst across signals, not whichever sorts last by name.

    `SIGNAL_NAMES` sorted alphabetically is `conformal_band, input_qc, ood_score,
    predicted_dice` -- `predicted_dice` is alphabetically LAST. This case makes
    `predicted_dice` PROCEED (good) while `conformal_band` (alphabetically FIRST)
    REFUSEs, so an implementation that mistakenly took "whichever signal sorts
    last" as the overall verdict would report PROCEED instead of the correct
    REFUSE -- a string-ordering bug this test would catch even though, as
    `Decision`'s docstring notes, the three `Decision` *values* themselves do not
    happen to invert under plain string comparison.
    """
    cfg = _make_cfg(
        enabled_signals=["input_qc", "predicted_dice", "conformal_band", "ood_score"],
        regions=["WT"],
    )
    signals = GateSignals(
        input_qc=_report(Severity.OK),
        predicted_dice={"WT": 0.95},  # PROCEED
        conformal_band={"WT": 0.5},  # REFUSE (> 0.3)
        ood_score=0.2,  # PROCEED_WITH_CAUTION (> 0.1, <= 0.3)
    )
    decision = run_gatekeeper(cfg, signals, thresholds=_thresholds(["WT"]))
    assert decision.decision is Decision.REFUSE

    by_signal = {v.signal: v.decision for v in decision.verdicts}
    assert by_signal["predicted_dice"] is Decision.PROCEED
    assert by_signal["conformal_band"] is Decision.REFUSE
    assert by_signal["ood_score"] is Decision.PROCEED_WITH_CAUTION


# ---------------------------------------------------------------------------
# 14-15. input_qc mapping and refusal reasons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        (Severity.OK, Decision.PROCEED),
        (Severity.WARN, Decision.PROCEED_WITH_CAUTION),
        (Severity.REFUSE, Decision.REFUSE),
    ],
)
def test_input_qc_severity_mapping(severity: Severity, expected: Decision) -> None:
    verdict = judge_input_qc(_report(severity), enabled=True)
    assert verdict.decision is expected


def test_refusal_reasons_survive_into_the_decision() -> None:
    cfg = _make_cfg(enabled_signals=["input_qc"], regions=["WT"])
    report = _report(Severity.REFUSE, "Missing required sequence(s): flair.")
    decision = run_gatekeeper(cfg, GateSignals(input_qc=report))
    assert decision.decision is Decision.REFUSE
    payload = json.dumps(decision.to_dict())
    assert "flair" in payload


# ---------------------------------------------------------------------------
# 16-17. JSON round trips
# ---------------------------------------------------------------------------


def test_to_dict_is_json_serialisable() -> None:
    verdict = SignalVerdict(
        signal="ood_score",
        decision=Decision.REFUSE,
        available=True,
        enabled=True,
        message="numpy-tainted",
        detail={
            "value": np.float32(1.5),
            "flag": np.bool_(True),
            "nested": {"arr": np.array([1, 2, 3])},
        },
    )
    decision = GateDecision(decision=Decision.REFUSE, verdicts=(verdict,))
    payload = json.dumps(decision.to_dict())  # must not raise
    restored = json.loads(payload)
    assert restored["decision"] == "refuse"
    assert restored["verdicts"][0]["detail"]["value"] == pytest.approx(1.5)
    assert restored["verdicts"][0]["detail"]["flag"] is True
    assert restored["verdicts"][0]["detail"]["nested"]["arr"] == [1, 2, 3]


def test_thresholds_round_trip() -> None:
    thresholds = _thresholds(["WT", "TC"])
    assert Thresholds.from_dict(thresholds.to_dict()) == thresholds
    # And the dict itself must be plain JSON -- no tuples, no numpy.
    json.dumps(thresholds.to_dict())


# ---------------------------------------------------------------------------
# 18. load_thresholds
# ---------------------------------------------------------------------------


def test_load_thresholds_none_and_path_and_mapping(tmp_path: Path) -> None:
    thresholds = _thresholds(["WT"])

    cfg_none = _make_cfg(enabled_signals=["input_qc"], regions=["WT"], thresholds=None)
    assert load_thresholds(cfg_none) is None

    cfg_mapping = _make_cfg(
        enabled_signals=["input_qc"], regions=["WT"], thresholds=thresholds.to_dict()
    )
    assert load_thresholds(cfg_mapping) == thresholds

    json_path = tmp_path / "thresholds.json"
    json_path.write_text(json.dumps(thresholds.to_dict()))
    cfg_path = _make_cfg(enabled_signals=["input_qc"], regions=["WT"], thresholds=str(json_path))
    assert load_thresholds(cfg_path) == thresholds

    missing_path = tmp_path / "does_not_exist.json"
    cfg_missing = _make_cfg(
        enabled_signals=["input_qc"], regions=["WT"], thresholds=str(missing_path)
    )
    with pytest.raises(FileNotFoundError, match=re_escape_path(missing_path)):
        load_thresholds(cfg_missing)


def re_escape_path(path: Path) -> str:
    """`re.escape` a path for use as a `pytest.raises(match=...)` pattern."""
    return re.escape(str(path))


# ---------------------------------------------------------------------------
# 19. Real config, composed through Hydra
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_at_the_composed_path() -> None:
    """The REAL project config, composed through Hydra, must expose the gatekeeper
    block at `cfg.clinical.gatekeeper` -- the exact path `run_gatekeeper` reads.
    """
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config")

    assert "clinical" in cfg
    assert "gatekeeper" in cfg.clinical
    assert "gatekeeper" not in cfg  # NOT cfg.gatekeeper

    gk_cfg = cfg.clinical.gatekeeper
    expected_keys = {
        "enabled_signals",
        "regions",
        "caution_quantile",
        "refuse_quantile",
        "thresholds",
        "out_dir",
    }
    assert expected_keys <= set(gk_cfg.keys())

    # run_gatekeeper must actually work against the real composed config, not just
    # expose the right keys -- see CLAUDE.md's driver-whose-tests-passed-against-a-
    # hand-built-fixture trap. Gate C fired POSITIVE 2026-08-26, so the real config
    # now enables predicted_dice and conformal_band too (not just input_qc) -- a
    # signal that is enabled but unmeasured is a REFUSE by design (see
    # `_judge_region_signal`), so a case that should PROCEED must supply values for
    # every region on the safe side of the real, calibrated thresholds.json this
    # config now points at (scripts/calibrate_gatekeeper.py, 2026-08-26).
    report = _report(Severity.OK)
    regions = list(gk_cfg.regions)
    decision = run_gatekeeper(
        cfg,
        GateSignals(
            input_qc=report,
            predicted_dice=dict.fromkeys(regions, 0.95),
            conformal_band=dict.fromkeys(regions, 0.5),
        ),
    )
    assert isinstance(decision, GateDecision)
    assert decision.decision is Decision.PROCEED
