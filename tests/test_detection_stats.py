"""Tests for scripts/detection_stats.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location`, the same pattern
`tests/test_report_agreement_script.py` and `scripts/smoke_test.py` use.

CPU only, tiny synthetic arrays built in the test (no real BraTS data),
everything well under a second per test. Case-level tests (3-4) build the
already-joined table directly, sidestepping `load_cohort`/`entropy_table`,
which are exercised separately by tests 1-2. Voxel-level test (5) writes a
minimal real preprocessed-tree case (`label.npy` + `meta.json`) and a real
`.npz` ambiguity file, because the property under test -- the sampling mask
never depending on the label -- can only be demonstrated by actually varying
the label on disk between two runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neurovision.utils.io import write_json

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detection_stats.py"
_spec = importlib.util.spec_from_file_location("detection_stats_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
detection_stats: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["detection_stats_script"] = detection_stats
_spec.loader.exec_module(detection_stats)

load_cohort = detection_stats.load_cohort
entropy_table = detection_stats.entropy_table
case_level_table = detection_stats.case_level_table
voxel_level_table = detection_stats.voxel_level_table
_process_voxel_case = detection_stats._process_voxel_case
build_family_table = detection_stats.build_family_table
build_verdict = detection_stats.build_verdict


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _write_ambiguity_summary(shard_dir: Path, case_ids: list[str]) -> None:
    """Writes a minimal ambiguity_summary.csv covering `case_ids`."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        case_id: {
            "amb_dis_mean_fg_mean": 0.1,
            "level": 0.0,
            "n_windows": 1.0,
        }
        for case_id in case_ids
    }
    pd.DataFrame.from_dict(rows, orient="index").rename_axis("case_id").to_csv(
        shard_dir / "ambiguity_summary.csv"
    )


def _write_logits(eval_dir: Path, case_id: str, logits: np.ndarray) -> None:
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / f"{case_id}.npy", logits.astype(np.float16))


def _write_preprocessed_case(
    prep_dir: Path, case_id: str, label: np.ndarray, spacing: tuple[float, float, float]
) -> None:
    """Writes `<prep_dir>/<case_id>/{label.npy, meta.json}`, matching the real schema.

    Field names copied from `neurovision.data.preprocessing.preprocess_case`
    (`case_id`, `original_shape`, `cropped_shape`, `bbox`, `affine`,
    `spacing`, `has_label`, `label_voxel_counts`, `source_axcodes`,
    `target_axcodes`). `_load_label_and_spacing` only reads `spacing`, but
    the rest is included so a fixture built this way cannot silently drift
    from the real on-disk schema.
    """
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", label.astype(np.uint8))
    shape = list(label.shape)
    write_json(
        {
            "case_id": case_id,
            "original_shape": shape,
            "cropped_shape": shape,
            "bbox": [[0, s] for s in shape],
            "affine": np.eye(4).tolist(),
            "spacing": list(spacing),
            "has_label": True,
            "label_voxel_counts": None,
            "source_axcodes": "LPS",
            "target_axcodes": "LPS",
        },
        case_dir / "meta.json",
    )


def _write_ambiguity_npz(
    shard_dir: Path, case_id: str, disagreement: np.ndarray, logits: np.ndarray
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    entropy_cnn = np.zeros_like(disagreement, dtype=np.float16)
    entropy_swin = np.zeros_like(disagreement, dtype=np.float16)
    np.savez_compressed(
        shard_dir / f"{case_id}.npz",
        disagreement=disagreement.astype(np.float16),
        entropy_cnn=entropy_cnn,
        entropy_swin=entropy_swin,
        logits=logits.astype(np.float16),
    )


def _detection_cfg(**overrides: object) -> OmegaConf:
    """Builds a minimal `cfg.analysis.detection`-shaped config for direct function calls."""
    base = {
        "case": {
            "score_column": "score",
            "control_column": "control",
            "metric_column": "metric",
        },
        "voxel": {
            "enabled": True,
            "mask": "predicted_dilated",
            "dilation_mm": 2.0,
            "max_voxels_per_case": 50,
            "max_cases": None,
        },
        "bootstrap": {"n_boot": 200, "ci": 0.95},
        "alpha": 0.05,
    }
    base.update(overrides)
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# 1. load_cohort raises on a duplicated case_id across shards
# ---------------------------------------------------------------------------


def test_load_cohort_raises_on_duplicate_case_id_across_shards(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_ambiguity_summary(shard0, ["CASE_A", "CASE_B"])
    _write_ambiguity_summary(shard1, ["CASE_B", "CASE_C"])  # CASE_B overlaps

    cohort_cfg = {"ambiguity_dirs": [str(shard0), str(shard1)]}
    with pytest.raises(ValueError, match="CASE_B"):
        load_cohort(cohort_cfg)


def test_load_cohort_concatenates_disjoint_shards(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_ambiguity_summary(shard0, ["CASE_A", "CASE_B"])
    _write_ambiguity_summary(shard1, ["CASE_C"])

    cohort_cfg = {"ambiguity_dirs": [str(shard0), str(shard1)]}
    summary, npz_paths = load_cohort(cohort_cfg)

    assert sorted(summary.index) == ["CASE_A", "CASE_B", "CASE_C"]
    assert npz_paths["CASE_A"] == shard0 / "CASE_A.npz"
    assert npz_paths["CASE_C"] == shard1 / "CASE_C.npz"


# ---------------------------------------------------------------------------
# 2. entropy_table reuses its cache
# ---------------------------------------------------------------------------


def test_entropy_table_reuses_cache_on_second_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    eval_dir = tmp_path / "eval"
    case_ids = ["CASE_A", "CASE_B"]
    rng = np.random.default_rng(0)
    for case_id in case_ids:
        _write_logits(eval_dir, case_id, rng.normal(size=(3, 4, 4, 4)).astype(np.float32))

    cache_path = tmp_path / "entropy_cache.csv"
    first = entropy_table(eval_dir, case_ids, cache_path)
    assert cache_path.is_file()
    mtime_after_first = cache_path.stat().st_mtime_ns

    # Delete the source logits: if the second call re-read them it would
    # crash with FileNotFoundError, proving the cache was actually used
    # rather than merely present.
    for case_id in case_ids:
        (eval_dir / "logits" / f"{case_id}.npy").unlink()

    with caplog.at_level("INFO"):
        second = entropy_table(eval_dir, case_ids, cache_path)

    assert cache_path.stat().st_mtime_ns == mtime_after_first  # file was not rewritten
    pd.testing.assert_frame_equal(first, second)
    assert any("reusing" in record.message.lower() for record in caplog.records)


def test_entropy_table_recomputes_when_cache_is_missing_a_case(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    rng = np.random.default_rng(1)
    for case_id in ("CASE_A", "CASE_B"):
        _write_logits(eval_dir, case_id, rng.normal(size=(3, 4, 4, 4)).astype(np.float32))

    cache_path = tmp_path / "entropy_cache.csv"
    entropy_table(eval_dir, ["CASE_A"], cache_path)

    # Second call asks for a case the cache does not have -- must recompute,
    # not raise and not silently drop CASE_B.
    result = entropy_table(eval_dir, ["CASE_A", "CASE_B"], cache_path)
    assert sorted(result.index) == ["CASE_A", "CASE_B"]


# ---------------------------------------------------------------------------
# 3-4. case_level_table: "just entropy again" vs. genuine independent signal
# ---------------------------------------------------------------------------


def test_case_level_table_pure_reproduction_of_entropy_gives_near_zero_partial_rho() -> None:
    rng = np.random.default_rng(4)
    n = 200
    case_ids = [f"CASE_{i:03d}" for i in range(n)]

    control = rng.normal(size=n)  # entropy
    score = control + 0.02 * rng.normal(size=n)  # disagreement: almost pure fn of entropy
    metric = control + rng.normal(size=n)  # Dice: correlated with entropy only

    joined = pd.DataFrame(
        {"score": score, "control": control, "metric": metric}, index=case_ids
    ).rename_axis("case_id")

    cfg = _detection_cfg()
    generator = np.random.default_rng(7)
    result = case_level_table("brats_test", joined, cfg, generator)

    row = result.iloc[0]
    assert abs(row["rho_score"]) > 0.3  # raw correlation looks real
    assert abs(row["rho_partial"]) < 0.15  # nothing survives controlling for entropy
    assert row["contains_zero"]


def test_case_level_table_independent_signal_gives_large_partial_rho() -> None:
    rng = np.random.default_rng(123)
    n = 80
    case_ids = [f"CASE_{i:03d}" for i in range(n)]

    control = rng.normal(size=n)  # entropy, unrelated to the real signal
    shared = rng.normal(size=n)  # the real, independent signal
    score = shared + 0.1 * rng.normal(size=n) + 0.2 * control
    metric = shared + 0.1 * rng.normal(size=n) + 0.2 * control

    joined = pd.DataFrame(
        {"score": score, "control": control, "metric": metric}, index=case_ids
    ).rename_axis("case_id")

    cfg = _detection_cfg()
    generator = np.random.default_rng(9)
    result = case_level_table("ssa", joined, cfg, generator)

    row = result.iloc[0]
    assert abs(row["rho_partial"]) >= 0.5  # clearly above the 0.20 pass threshold
    assert not row["contains_zero"]
    assert row["p_boot"] < 0.05


# ---------------------------------------------------------------------------
# 5. voxel sampling mask never depends on the label
# ---------------------------------------------------------------------------


def _build_voxel_case(tmp_path: Path, case_id: str) -> tuple[Path, Path]:
    """Writes one case's npz (fixed) under `shard_dir`; caller writes/rewrites the label."""
    shape = (8, 8, 8)
    disagreement = np.zeros((3, *shape), dtype=np.float32)
    logits = np.full((3, *shape), -10.0, dtype=np.float32)

    # A 4x4x4 = 64-voxel foreground block (>= inference.postprocess.min_component_size=50,
    # so it survives small-component removal) with varied disagreement values, so the
    # sampled 'score' array is not trivially constant.
    rng = np.random.default_rng(0)
    block = (slice(2, 6), slice(2, 6), slice(2, 6))
    logits[:, block[0], block[1], block[2]] = 10.0
    disagreement[:, block[0], block[1], block[2]] = rng.uniform(0.0, 1.0, size=(3, 4, 4, 4))

    shard_dir = tmp_path / "shard0"
    _write_ambiguity_npz(shard_dir, case_id, disagreement, logits)
    prep_dir = tmp_path / "prep"
    return shard_dir, prep_dir


def test_voxel_sampling_indices_do_not_depend_on_the_label(tmp_path: Path) -> None:
    case_id = "CASE_A"
    shard_dir, prep_dir = _build_voxel_case(tmp_path, case_id)
    npz_path = shard_dir / f"{case_id}.npz"
    shape = (8, 8, 8)
    spacing = (1.0, 1.0, 1.0)

    # Run 1: an empty label.
    _write_preprocessed_case(prep_dir, case_id, np.zeros(shape, dtype=np.uint8), spacing)
    result_1 = _process_voxel_case(
        case_id,
        npz_path,
        prep_dir,
        mask_mode="predicted_dilated",
        dilation_mm=2.0,
        max_voxels=50,
        generator=np.random.default_rng(11),
        cohort_name="brats_test",
    )
    assert result_1 is not None

    # Run 2: a COMPLETELY different label (everything is enhancing tumor).
    _write_preprocessed_case(prep_dir, case_id, np.full(shape, 3, dtype=np.uint8), spacing)
    result_2 = _process_voxel_case(
        case_id,
        npz_path,
        prep_dir,
        mask_mode="predicted_dilated",
        dilation_mm=2.0,
        max_voxels=50,
        generator=np.random.default_rng(11),  # same seed -> same draw, if mask is unchanged
        cohort_name="brats_test",
    )
    assert result_2 is not None

    for region in ("ET", "TC", "WT", "ANY"):
        # score/control come only from disagreement/logits, never the label --
        # a wrong (label-derived) mask would very likely draw a different
        # voxel subset and break this equality.
        np.testing.assert_array_equal(result_1[region]["score"], result_2[region]["score"])
        np.testing.assert_array_equal(result_1[region]["control"], result_2[region]["control"])

    # The label really was different, and it really did change something --
    # otherwise this test would pass trivially even with a broken pipeline.
    assert not np.array_equal(result_1["WT"]["positive"], result_2["WT"]["positive"])


# ---------------------------------------------------------------------------
# 6. build_verdict: collapse to 0.5 -> fail; strong external signal -> pass
# ---------------------------------------------------------------------------


def test_build_verdict_collapsed_auroc_gives_fail() -> None:
    case_rows = {"ssa": {"rho_partial": 0.05, "contains_zero": True}}
    voxel_any_rows = {"ssa": {"auroc_residual": 0.50, "resid_ci_lo": 0.44, "resid_ci_hi": 0.56}}
    verdict = build_verdict(case_rows, voxel_any_rows)
    assert verdict["verdict"] == "fail"
    assert verdict["passed_cohorts"] == []
    assert verdict["partial_cohorts"] == []


def test_build_verdict_strong_external_signal_gives_pass() -> None:
    case_rows = {"ssa": {"rho_partial": 0.35, "contains_zero": False}}
    voxel_any_rows = {"ssa": {"auroc_residual": 0.72, "resid_ci_lo": 0.65, "resid_ci_hi": 0.80}}
    verdict = build_verdict(case_rows, voxel_any_rows)
    assert verdict["verdict"] == "pass"
    assert verdict["passed_cohorts"] == ["ssa"]
    assert verdict["preregistration"] == "docs/research/preregistration_ambiguity.md"


def test_build_verdict_in_distribution_only_signal_is_not_a_pass() -> None:
    # The pass condition names SSA/PED explicitly -- a strong signal on
    # brats_test alone (not external) must not pass.
    case_rows = {"brats_test": {"rho_partial": 0.35, "contains_zero": False}}
    voxel_any_rows = {
        "brats_test": {"auroc_residual": 0.72, "resid_ci_lo": 0.65, "resid_ci_hi": 0.80}
    }
    verdict = build_verdict(case_rows, voxel_any_rows)
    assert verdict["verdict"] != "pass"
    assert verdict["verdict"] == "partial"  # its CI still excludes both nulls


# ---------------------------------------------------------------------------
# 7. build_family_table: exactly 6 rows for 3 cohorts, Holm-corrected once
# ---------------------------------------------------------------------------


def test_build_family_table_has_exactly_six_rows_for_three_cohorts() -> None:
    case_rows = {
        "brats_test": {"p_boot": 0.20},
        "ssa": {"p_boot": 0.01},
        "ped": {"p_boot": 0.40},
    }
    voxel_any_rows = {
        "brats_test": {"p_boot": 0.15},
        "ssa": {"p_boot": 0.005},
        "ped": {"p_boot": 0.30},
    }
    family = build_family_table(case_rows, voxel_any_rows, alpha=0.05)

    assert len(family) == 6
    assert set(family["endpoint"]) == {"case", "voxel_any"}
    assert sorted(family["cohort"].unique()) == ["brats_test", "ped", "ssa"]
    assert list(family.columns) == ["cohort", "endpoint", "statistic", "p_raw", "p_holm", "reject"]
    # Holm-adjusted p-values are never smaller than the raw ones.
    assert (family["p_holm"] >= family["p_raw"] - 1e-12).all()
    # The smallest raw p-value (ssa/voxel_any, 0.005) gets multiplied by the
    # full family size (6) under Holm's step-down rule.
    smallest = family.loc[family["p_raw"].idxmin()]
    assert smallest["p_holm"] == pytest.approx(0.005 * 6)


def test_build_family_table_drops_cohort_missing_one_endpoint() -> None:
    case_rows = {"brats_test": {"p_boot": 0.2}, "ssa": {"p_boot": 0.1}}
    voxel_any_rows = {"brats_test": {"p_boot": 0.2}}  # ssa has no voxel-level row yet
    family = build_family_table(case_rows, voxel_any_rows, alpha=0.05)
    assert len(family) == 2  # only brats_test contributes, at both endpoints
    assert set(family["cohort"]) == {"brats_test"}


def test_build_family_table_raises_when_no_cohort_has_both_endpoints() -> None:
    with pytest.raises(ValueError, match="no cohort"):
        build_family_table({"ssa": {"p_boot": 0.1}}, {"ped": {"p_boot": 0.1}}, alpha=0.05)


# ---------------------------------------------------------------------------
# voxel_level_table: end-to-end over the fixture case, sanity on the shape
# ---------------------------------------------------------------------------


def test_voxel_level_table_end_to_end_shape(tmp_path: Path) -> None:
    case_id = "CASE_A"
    shard_dir, prep_dir = _build_voxel_case(tmp_path, case_id)
    _write_preprocessed_case(
        prep_dir, case_id, np.zeros((8, 8, 8), dtype=np.uint8), (1.0, 1.0, 1.0)
    )
    npz_paths = {case_id: shard_dir / f"{case_id}.npz"}

    cfg = _detection_cfg()
    generator = np.random.default_rng(3)
    table, diagnostics = voxel_level_table("brats_test", npz_paths, prep_dir, cfg, generator)

    assert sorted(table["region"]) == ["ANY", "ET", "TC", "WT"]
    assert (table["cohort"] == "brats_test").all()
    for col in (
        "auroc_score",
        "auroc_control",
        "auroc_residual",
        "resid_ci_lo",
        "resid_ci_hi",
        "p_boot",
    ):
        assert col in table.columns
    assert diagnostics["enabled"] is True
    assert diagnostics["n_cases_used"] == 1


def test_voxel_level_table_disabled_returns_empty(tmp_path: Path) -> None:
    case_id = "CASE_A"
    shard_dir, prep_dir = _build_voxel_case(tmp_path, case_id)
    _write_preprocessed_case(
        prep_dir, case_id, np.zeros((8, 8, 8), dtype=np.uint8), (1.0, 1.0, 1.0)
    )
    npz_paths = {case_id: shard_dir / f"{case_id}.npz"}

    cfg = _detection_cfg(voxel={"enabled": False})
    table, diagnostics = voxel_level_table(
        "brats_test", npz_paths, prep_dir, cfg, np.random.default_rng(3)
    )
    assert table.empty
    assert diagnostics == {"enabled": False}
