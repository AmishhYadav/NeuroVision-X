"""Tests for scripts/calibrate_gatekeeper.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` -- the same pattern
tests/test_validate_qc_script.py and tests/test_calibrate_script.py already
use for their own sibling scripts.

`build_gatekeeper_calibration_table` is monkeypatched with a small synthetic
`pd.DataFrame` throughout: no real SegQC checkpoint, no real conformal
artifact, no network access and no BraTS data is needed to test this script's
own plumbing (write the table, fit thresholds, write the JSON). The fitting
math itself (`calibrate_thresholds`) is already tested in
tests/test_gatekeeper.py -- these tests only check that this driver wires it
up and writes what it claims to.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from omegaconf import OmegaConf

from neurovision.inference.gatekeeper import Thresholds, calibrate_thresholds

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_gatekeeper.py"
_spec = importlib.util.spec_from_file_location("calibrate_gatekeeper_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
calibrate_gatekeeper_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_gatekeeper_script"] = calibrate_gatekeeper_script
_spec.loader.exec_module(calibrate_gatekeeper_script)

run_calibration = calibrate_gatekeeper_script.run_calibration

REGIONS = ("WT", "TC")


def _synthetic_table(n: int = 3) -> pd.DataFrame:
    """A small, hand-built table with exactly the columns `calibrate_thresholds` needs."""
    return pd.DataFrame(
        {
            "case_id": [f"case_{i}" for i in range(n)],
            "predicted_dice_WT": [0.9, 0.7, 0.8][:n],
            "predicted_dice_TC": [0.85, 0.6, 0.75][:n],
            "conformal_band_WT": [1.1, 1.4, 1.2][:n],
            "conformal_band_TC": [1.05, 1.3, 1.15][:n],
            "ood_score": [0.2, 0.5, 0.3][:n],
        }
    )


def _make_cfg(out_dir: Path, **gatekeeper_overrides: object) -> OmegaConf:
    gatekeeper = {
        "regions": list(REGIONS),
        "caution_quantile": 0.5,
        "refuse_quantile": 0.1,
        "out_dir": str(out_dir),
    }
    gatekeeper.update(gatekeeper_overrides)
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "clinical": {"gatekeeper": gatekeeper},
        }
    )


# ---------------------------------------------------------------------------
# 1. Both outputs are written, and are valid CSV / JSON
# ---------------------------------------------------------------------------


def test_run_calibration_writes_both_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _synthetic_table()
    monkeypatch.setattr(
        calibrate_gatekeeper_script, "build_gatekeeper_calibration_table", lambda cfg: table
    )

    out_dir = tmp_path / "gatekeeper"
    cfg = _make_cfg(out_dir)

    result = run_calibration(cfg)

    assert set(result.keys()) == {"calibration_table", "thresholds"}
    calibration_table_path = out_dir / "calibration_table.csv"
    thresholds_path = out_dir / "thresholds.json"
    assert result["calibration_table"] == calibration_table_path
    assert result["thresholds"] == thresholds_path
    assert calibration_table_path.is_file()
    assert thresholds_path.is_file()

    written_table = pd.read_csv(calibration_table_path)
    pd.testing.assert_frame_equal(written_table, table)

    payload = json.loads(thresholds_path.read_text())
    assert set(payload.keys()) == {
        "predicted_dice",
        "conformal_band",
        "ood_score",
        "calibration_n",
        "caution_quantile",
        "refuse_quantile",
    }


# ---------------------------------------------------------------------------
# 2. The written JSON round-trips through Thresholds.from_dict
# ---------------------------------------------------------------------------


def test_thresholds_json_round_trips_through_from_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _synthetic_table()
    monkeypatch.setattr(
        calibrate_gatekeeper_script, "build_gatekeeper_calibration_table", lambda cfg: table
    )

    out_dir = tmp_path / "gatekeeper"
    cfg = _make_cfg(out_dir)
    result = run_calibration(cfg)

    payload = json.loads(result["thresholds"].read_text())
    rebuilt = Thresholds.from_dict(payload)

    assert isinstance(rebuilt, Thresholds)
    assert set(rebuilt.predicted_dice.keys()) == set(REGIONS)
    assert set(rebuilt.conformal_band.keys()) == set(REGIONS)
    assert rebuilt.calibration_n == len(table)
    assert rebuilt.caution_quantile == pytest.approx(0.5)
    assert rebuilt.refuse_quantile == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 3. The fitted numbers match calling calibrate_thresholds directly
# ---------------------------------------------------------------------------


def test_thresholds_match_calibrate_thresholds_called_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _synthetic_table()
    monkeypatch.setattr(
        calibrate_gatekeeper_script, "build_gatekeeper_calibration_table", lambda cfg: table
    )

    out_dir = tmp_path / "gatekeeper"
    cfg = _make_cfg(out_dir, caution_quantile=0.4, refuse_quantile=0.05)
    result = run_calibration(cfg)

    payload = json.loads(result["thresholds"].read_text())
    written = Thresholds.from_dict(payload)

    expected = calibrate_thresholds(
        table, regions=list(REGIONS), caution_quantile=0.4, refuse_quantile=0.05
    )

    assert written.to_dict() == expected.to_dict()


# ---------------------------------------------------------------------------
# 4. out_dir is created even when it does not exist yet
# ---------------------------------------------------------------------------


def test_run_calibration_creates_out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = _synthetic_table()
    monkeypatch.setattr(
        calibrate_gatekeeper_script, "build_gatekeeper_calibration_table", lambda cfg: table
    )

    out_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not out_dir.exists()
    cfg = _make_cfg(out_dir)

    run_calibration(cfg)

    assert out_dir.is_dir()
    assert (out_dir / "calibration_table.csv").is_file()
    assert (out_dir / "thresholds.json").is_file()
