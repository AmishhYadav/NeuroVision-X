"""Tests for neurovision.analysis.volume_endpoint (tiny synthetic arrays, CPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from neurovision.analysis.volume_endpoint import (
    label_usable_volume,
    label_volumes,
    region_volumes,
    usable_both,
    volume_error_table,
)

SPACING = (1.0, 1.0, 2.0)  # voxel = 2 mm^3 = 0.002 ml


def _regions_10_voxels() -> np.ndarray:
    r = np.zeros((3, 4, 4, 4), dtype=bool)
    r[0, 0, 0, :4] = True  # ET: 4 voxels
    r[1, 0, 0, :4] = True  # TC: 4 + 3 = 7
    r[1, 1, 0, :3] = True
    r[2, 0, 0, :4] = True  # WT: 10
    r[2, 1, 0, :3] = True
    r[2, 2, 0, :3] = True
    return r


def test_region_volumes_hand_computed_numpy_and_torch() -> None:
    r = _regions_10_voxels()
    expected = {"vol_ET_ml": 0.008, "vol_TC_ml": 0.014, "vol_WT_ml": 0.020}
    for arr in (r, torch.from_numpy(r.astype(np.float32))):
        got = region_volumes(arr, SPACING)
        assert got == pytest.approx(expected)


def test_region_volumes_bad_shape() -> None:
    with pytest.raises(ValueError):
        region_volumes(np.zeros((2, 3, 3, 3)), SPACING)


def test_label_volumes_nesting_and_values() -> None:
    lab = np.zeros((4, 4, 4), dtype=np.int64)
    lab[0, 0, :2] = 3  # 2 ET
    lab[0, 1, :3] = 1  # 3 necrotic
    lab[0, 2, : 5 - 1] = 2  # 4 edema
    v = label_volumes(lab, (1.0, 1.0, 1.0))
    assert v["vol_ET_ml"] == pytest.approx(0.002)
    assert v["vol_TC_ml"] == pytest.approx(0.005)
    assert v["vol_WT_ml"] == pytest.approx(0.009)
    assert v["vol_ET_ml"] <= v["vol_TC_ml"] <= v["vol_WT_ml"]


def _frames(pred_vals: dict, ref_vals: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.Index(["a", "b", "c"], name="case_id")
    pred = pd.DataFrame({f"vol_{r}_ml": pred_vals[r] for r in ("ET", "TC", "WT")}, index=idx)
    ref = pd.DataFrame({f"vol_{r}_ml": ref_vals[r] for r in ("ET", "TC", "WT")}, index=idx)
    return pred, ref


def test_volume_error_table_exact_and_empty_ref() -> None:
    pred, ref = _frames(
        {"ET": [1.5, 0.0, 2.0], "TC": [1, 1, 1], "WT": [3, 3, 3]},
        {"ET": [1.0, 0.0, 0.0], "TC": [1, 1, 1], "WT": [3, 3, 3]},
    )
    t = volume_error_table(pred, ref)
    assert t.loc["a", "vol_rel_err_ET"] == pytest.approx(0.5)
    assert t.loc["a", "vol_pred_ET_ml"] == 1.5
    assert t.loc["a", "vol_ref_ET_ml"] == 1.0
    assert not t.loc["a", "ref_empty_ET"]
    assert np.isnan(t.loc["b", "vol_rel_err_ET"]) and t.loc["b", "ref_empty_ET"]
    assert np.isnan(t.loc["c", "vol_rel_err_ET"]) and t.loc["c", "ref_empty_ET"]


def test_volume_error_table_index_mismatch() -> None:
    pred, ref = _frames(
        {r: [1, 1, 1] for r in ("ET", "TC", "WT")}, {r: [1, 1, 1] for r in ("ET", "TC", "WT")}
    )
    with pytest.raises(ValueError, match="index"):
        volume_error_table(pred, ref.iloc[::-1])


def _usable(pred_et: list[float], ref_et: list[float], **kw) -> pd.Series:
    n = len(pred_et)
    idx = pd.Index([f"c{i}" for i in range(n)])
    pred = pd.DataFrame(
        {"vol_ET_ml": pred_et, "vol_TC_ml": [1.0] * n, "vol_WT_ml": [1.0] * n}, index=idx
    )
    ref = pd.DataFrame(
        {"vol_ET_ml": ref_et, "vol_TC_ml": [1.0] * n, "vol_WT_ml": [1.0] * n}, index=idx
    )
    return label_usable_volume(volume_error_table(pred, ref), **kw)


def test_tol_boundary() -> None:
    # 1.25 vs 1.0 -> exactly 0.25 (exact in binary floating point); 1.2501 -> 0.2501.
    u = _usable([1.25, 1.2501, 0.75], [1.0, 1.0, 1.0])
    assert list(u) == [True, False, True]


def test_empty_reference_conventions() -> None:
    u = _usable([0.0, 0.5], [0.0, 0.0])
    assert list(u) == [True, False]
    u2 = _usable([0.5, 0.6], [0.0, 0.0], empty_pred_max_ml=0.5)
    assert list(u2) == [True, False]


def test_nan_is_not_usable() -> None:
    u = _usable([np.nan, 1.0], [1.0, np.nan])
    assert list(u) == [False, False]


def test_validation_errors() -> None:
    idx = pd.Index(["a"])
    table = pd.DataFrame({"vol_pred_WT_ml": [1.0], "ref_empty_WT": [False]}, index=idx)
    with pytest.raises(ValueError, match="vol_rel_err_WT"):
        label_usable_volume(table, regions=("WT",))
    full = pd.DataFrame(
        {"vol_pred_WT_ml": [1.0], "ref_empty_WT": [False], "vol_rel_err_WT": [0.0]}, index=idx
    )
    with pytest.raises(ValueError, match="tol"):
        label_usable_volume(full, regions=("WT",), tol=0.0)


def test_usable_both() -> None:
    idx = pd.Index(["a", "b", "c", "d"])
    d = pd.Series([True, True, False, False], index=idx)
    v = pd.Series([True, False, True, False], index=idx)
    t = usable_both(d, v)
    assert list(t.columns) == ["usable_dice", "usable_volume", "usable_both", "usable_either"]
    assert list(t["usable_both"]) == [True, False, False, False]
    assert list(t["usable_either"]) == [True, True, True, False]
    with pytest.raises(ValueError, match="index"):
        usable_both(d, v.iloc[::-1])
