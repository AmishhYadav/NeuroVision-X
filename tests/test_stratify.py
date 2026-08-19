"""Tests for `neurovision.analysis.stratify`.

CPU only, synthetic data (hand-built label arrays and small tmp_path
directories, no real BraTS data), whole file well under a few seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neurovision.analysis.statistics import compare_models
from neurovision.analysis.stratify import (
    assign_bins,
    ground_truth_volumes,
    overlapping_volume_range,
    quantile_bin_edges,
    stratified_comparison,
    stratified_summary,
    volume_matched_subset,
)


def _write_case(prep_dir, case_id: str, label: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True)
    np.save(case_dir / "label.npy", label.astype(np.uint8))
    meta = {
        "case_id": case_id,
        "original_shape": list(label.shape),
        "affine": np.diag([-1.0, -1.0, 1.0, 1.0]).tolist(),
        "spacing": list(spacing),
    }
    (case_dir / "meta.json").write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# 1-3. ground_truth_volumes
# ---------------------------------------------------------------------------


def test_ground_truth_volumes_recovers_exact_counts_and_nesting(tmp_path) -> None:
    label = np.zeros((10, 10, 10), dtype=np.uint8)
    label[0:5, 0:5, 0:5] = 2  # edema, 125 voxels
    label[0:3, 0:3, 0:3] = 1  # necrotic core, 27 voxels (inside the edema block)
    label[0:2, 0:2, 0:2] = 3  # enhancing tumor, 8 voxels (inside the core block)
    _write_case(tmp_path, "case_a", label)

    volumes = ground_truth_volumes(tmp_path, case_ids=["case_a"])

    assert volumes.loc["case_a", "vol_ET_vox"] == 8
    assert volumes.loc["case_a", "vol_TC_vox"] == 27
    assert volumes.loc["case_a", "vol_WT_vox"] == 125

    row = volumes.loc["case_a"]
    assert row["vol_ET_vox"] <= row["vol_TC_vox"] <= row["vol_WT_vox"]


def test_ground_truth_volumes_mm3_uses_spacing(tmp_path) -> None:
    label = np.zeros((10, 10, 10), dtype=np.uint8)
    label[0:2, 0:2, 0:2] = 3  # ET, 8 voxels
    spacing = (1.5, 2.0, 0.5)
    _write_case(tmp_path, "case_a", label, spacing=spacing)

    volumes = ground_truth_volumes(tmp_path, case_ids=["case_a"])

    voxel_mm3 = spacing[0] * spacing[1] * spacing[2]
    assert volumes.loc["case_a", "vol_ET_mm3"] == pytest.approx(8 * voxel_mm3)
    assert volumes.loc["case_a", "vol_WT_mm3"] == pytest.approx(8 * voxel_mm3)


def test_ground_truth_volumes_skips_unlabeled_case_with_warning(tmp_path, caplog) -> None:
    label = np.zeros((6, 6, 6), dtype=np.uint8)
    label[0:2, 0:2, 0:2] = 3
    _write_case(tmp_path, "case_labeled", label)

    unlabeled_dir = tmp_path / "case_unlabeled"
    unlabeled_dir.mkdir()
    (unlabeled_dir / "image.npy").touch()  # no label.npy

    with caplog.at_level("WARNING"):
        volumes = ground_truth_volumes(tmp_path, case_ids=["case_labeled", "case_unlabeled"])

    assert list(volumes.index) == ["case_labeled"]
    assert any("case_unlabeled" in r.message for r in caplog.records)


def test_ground_truth_volumes_raises_on_missing_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ground_truth_volumes(tmp_path / "does_not_exist")


def test_ground_truth_volumes_discovers_cases_when_ids_none(tmp_path) -> None:
    label = np.zeros((6, 6, 6), dtype=np.uint8)
    label[0:2, 0:2, 0:2] = 3
    _write_case(tmp_path, "case_x", label)
    _write_case(tmp_path, "case_y", label)

    volumes = ground_truth_volumes(tmp_path)

    assert set(volumes.index) == {"case_x", "case_y"}


# ---------------------------------------------------------------------------
# 4-5. quantile_bin_edges
# ---------------------------------------------------------------------------


def test_quantile_bin_edges_equal_count_bins() -> None:
    values = np.arange(100, dtype=np.float64)
    edges = quantile_bin_edges(values, n_bins=4)

    assert len(edges) == 5
    assert edges[0] == 0.0
    assert edges[-1] == float("inf")

    labels = assign_bins(values, edges)
    counts = pd.Series(labels).value_counts()
    # Equal-count quantile bins on evenly spaced data: each bin ~25 values.
    assert counts.min() >= 20
    assert counts.max() <= 30


def test_quantile_bin_edges_dedupes_and_warns_on_duplicate_heavy_input(caplog) -> None:
    values = np.full(50, 7.0)

    with caplog.at_level("WARNING"):
        edges = quantile_bin_edges(values, n_bins=4)

    # All values identical -> every quantile is 7.0 -> after dedup only
    # [7.0, inf] survive -> exactly one (degenerate-width-free) bin.
    assert edges == [7.0, float("inf")]
    assert any("quantile_bin_edges" in r.message for r in caplog.records)


def test_quantile_bin_edges_raises_on_empty_or_bad_n_bins() -> None:
    with pytest.raises(ValueError):
        quantile_bin_edges([], n_bins=4)
    with pytest.raises(ValueError):
        quantile_bin_edges([1.0, 2.0], n_bins=0)


# ---------------------------------------------------------------------------
# 6-7. assign_bins
# ---------------------------------------------------------------------------


def test_assign_bins_half_open_upper_bin_on_boundary() -> None:
    edges = [0.0, 10.0, 20.0, float("inf")]
    labels = assign_bins([0.0, 9.9, 10.0, 19.9, 20.0, 100.0], edges)

    assert labels == [
        "0-10",
        "0-10",
        "10-20",  # exactly on the interior edge -> UPPER bin
        "10-20",
        "20-inf",  # exactly on the interior edge -> UPPER bin
        "20-inf",
    ]


def test_assign_bins_raises_below_first_edge() -> None:
    edges = [10.0, 20.0, float("inf")]
    with pytest.raises(ValueError):
        assign_bins([5.0], edges)


# ---------------------------------------------------------------------------
# 8-9. stratified_summary
# ---------------------------------------------------------------------------


def test_stratified_summary_bin_order_is_numeric_not_lexicographic() -> None:
    # Edges chosen so the bin labels sort differently lexicographically
    # ("10000-inf" < "2000-5000" as strings) than numerically.
    edges = [0.0, 2000.0, 10000.0, float("inf")]
    case_ids = [f"c{i}" for i in range(9)]
    volumes = pd.DataFrame(
        {"vol_WT_vox": [500, 800, 1500, 3000, 4000, 7000, 15000, 20000, 30000]},
        index=pd.Index(case_ids, name="case_id"),
    )
    per_case = pd.DataFrame({"dice_WT": np.linspace(0.5, 0.9, 9)}, index=case_ids)
    per_case.index.name = "case_id"

    result = stratified_summary(per_case, volumes, metric_cols=["dice_WT"], edges=edges)

    assert list(result["bin"]) == ["0-2000", "2000-10000", "10000-inf"]
    assert list(result["bin_lo"]) == [0.0, 2000.0, 10000.0]


def test_stratified_summary_accepts_index_or_column_case_id() -> None:
    case_ids = [f"c{i}" for i in range(6)]
    volumes_as_index = pd.DataFrame(
        {"vol_WT_vox": [100, 200, 300, 400, 500, 600]},
        index=pd.Index(case_ids, name="case_id"),
    )
    per_case_as_column = pd.DataFrame(
        {"case_id": case_ids, "dice_WT": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    )

    result = stratified_summary(
        per_case_as_column, volumes_as_index, metric_cols=["dice_WT"], n_bins=2
    )

    assert result["n"].sum() == 6
    assert "dice_WT_mean" in result.columns
    assert "dice_WT_median" in result.columns
    assert "dice_WT_n_missing" in result.columns


# ---------------------------------------------------------------------------
# 10-11. stratified_comparison
# ---------------------------------------------------------------------------


def _paired_tables(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    case_ids = [f"c{i}" for i in range(n)]
    a = pd.DataFrame(
        {"dice_WT": np.clip(rng.normal(0.85, 0.05, n), 0, 1)},
        index=pd.Index(case_ids, name="case_id"),
    )
    b = pd.DataFrame(
        {"dice_WT": np.clip(rng.normal(0.80, 0.05, n), 0, 1)},
        index=pd.Index(case_ids, name="case_id"),
    )
    volumes = pd.DataFrame(
        {"vol_WT_vox": rng.integers(1000, 200000, n)},
        index=pd.Index(case_ids, name="case_id"),
    )
    return a, b, volumes


def test_stratified_comparison_matches_compare_models_at_n_bins_1() -> None:
    a, b, volumes = _paired_tables(30, seed=1)

    plain = compare_models(
        a, b, generator=np.random.default_rng(42), metrics=["dice_WT"], name_a="a", name_b="b"
    )
    stratified = stratified_comparison(
        a,
        b,
        volumes,
        metric_cols=["dice_WT"],
        generator=np.random.default_rng(42),
        name_a="a",
        name_b="b",
        n_bins=1,
    )

    assert len(stratified) == 1
    row = stratified.iloc[0]
    plain_row = plain.loc["dice_WT"]
    assert row["mean_diff"] == pytest.approx(plain_row["mean_diff"])
    assert row["n"] == plain_row["n"]
    assert row["p_holm"] == pytest.approx(plain_row["p_holm"])
    assert row["holm_family"] == row["bin"]


def test_stratified_comparison_skips_small_bin_with_warning(caplog) -> None:
    a, b, volumes = _paired_tables(20, seed=2)
    # Force two bins via explicit edges, one of which will end up tiny.
    vols = volumes["vol_WT_vox"].to_numpy()
    median = float(np.median(vols))
    edges = [float(vols.min()), median, float("inf")]

    # Manually shrink one bin to below 5 cases by editing volumes so almost
    # every case falls in the upper bin.
    volumes = volumes.copy()
    volumes.iloc[3:, volumes.columns.get_loc("vol_WT_vox")] = vols.max() + 1000

    with caplog.at_level("WARNING"):
        result = stratified_comparison(
            a,
            b,
            volumes,
            metric_cols=["dice_WT"],
            generator=np.random.default_rng(7),
            edges=edges,
        )

    assert result["bin"].nunique() <= 1
    assert any("stratified_comparison" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 12-13. overlapping_volume_range / volume_matched_subset
# ---------------------------------------------------------------------------


def test_overlapping_volume_range_raises_on_disjoint_cohorts() -> None:
    volumes_a = pd.DataFrame(
        {"vol_WT_vox": np.arange(1000, 2000, 10)}, index=pd.RangeIndex(100, name="case_id")
    )
    volumes_b = pd.DataFrame(
        {"vol_WT_vox": np.arange(50000, 51000, 10)}, index=pd.RangeIndex(100, name="case_id")
    )

    with pytest.raises(ValueError):
        overlapping_volume_range(volumes_a, volumes_b)


def test_overlapping_volume_range_on_overlapping_cohorts() -> None:
    rng = np.random.default_rng(3)
    volumes_a = pd.DataFrame(
        {"vol_WT_vox": rng.uniform(1000, 5000, 200)}, index=pd.RangeIndex(200, name="case_id")
    )
    volumes_b = pd.DataFrame(
        {"vol_WT_vox": rng.uniform(3000, 9000, 200)}, index=pd.RangeIndex(200, name="case_id")
    )

    lo, hi = overlapping_volume_range(volumes_a, volumes_b, quantile=0.05)
    assert lo < hi


def test_volume_matched_subset_is_inclusive_at_both_ends() -> None:
    volumes = pd.DataFrame(
        {"vol_WT_vox": [100, 200, 300, 400, 500]},
        index=pd.Index(["c0", "c1", "c2", "c3", "c4"], name="case_id"),
    )

    subset = volume_matched_subset(volumes, lo=200, hi=400)

    assert subset == ["c1", "c2", "c3"]
