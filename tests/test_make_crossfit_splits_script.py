"""Tests for scripts/make_crossfit_splits.py.

The script lives under scripts/, not src/, so it is loaded via
`tests.script_loader.load_script`, the same pattern as
tests/test_local_recalibration_script.py. `run_make_crossfit_splits` is
exercised with a small, hand-built `OmegaConf` config holding only
`analysis.crossfit_splits` -- the same pattern
`tests/test_local_recalibration_script.py::_make_cfg` uses -- rather than a
full Hydra compose, since composing the real `configs/config.yaml` would
require an unrelated, unused `data.root_dir` (see the script's own module
docstring). One test at the bottom composes the real Hydra config directly,
to guard the `cfg.analysis.crossfit_splits` config path itself.

Everything here is synthetic case ids under `tmp_path` -- no real BraTS data,
no GPU, and the real `configs/data/splits_ssa.yaml` / `splits_ped.yaml` are
never read or written by any test.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from neurovision.data.dataset import load_splits
from neurovision.utils.io import write_yaml
from tests.script_loader import load_script

make_crossfit_splits_script = load_script("make_crossfit_splits")

load_cohort_test_ids = make_crossfit_splits_script.load_cohort_test_ids
build_cohort_folds = make_crossfit_splits_script.build_cohort_folds
run_make_crossfit_splits = make_crossfit_splits_script.run_make_crossfit_splits

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_source(path: Path, case_ids: list[str], train: list[str] | None = None) -> None:
    """Writes a cohort source split file: every case in test, train/val empty
    (unless a non-empty `train` is passed, to exercise the validation error)."""
    write_yaml({"train": train or [], "val": [], "test": case_ids}, path)


def _make_cfg(
    tmp_path: Path,
    cohorts: dict[str, str],
    *,
    n_folds: int = 2,
    val_n: int = 5,
    seed: int = 42,
    overwrite: bool = False,
    out_pattern: str | None = None,
) -> OmegaConf:
    """Minimal composed-looking cfg with only what run_make_crossfit_splits reads."""
    return OmegaConf.create(
        {
            "analysis": {
                "crossfit_splits": {
                    "n_folds": n_folds,
                    "seed": seed,
                    "val_n": val_n,
                    "cohorts": cohorts,
                    "out_pattern": out_pattern or str(tmp_path / "splits_{cohort}_cf{fold}.yaml"),
                    "overwrite": overwrite,
                }
            }
        }
    )


def _ssa_ids(n: int = 60) -> list[str]:
    return [f"BraTS-SSA-{i:05d}-000" for i in range(n)]


def _ped_ids(n: int = 99) -> list[str]:
    return [f"BraTS-PED-{i:05d}-000" for i in range(n)]


# ---------------------------------------------------------------------------
# 1. load_cohort_test_ids: happy path + both validation errors
# ---------------------------------------------------------------------------


def test_load_cohort_test_ids_returns_sorted_test_list(tmp_path: Path):
    source = tmp_path / "splits_ssa.yaml"
    ids = _ssa_ids(10)
    _write_source(source, list(reversed(ids)))

    result = load_cohort_test_ids(source)

    assert result == sorted(ids)


def test_load_cohort_test_ids_raises_on_nonempty_train(tmp_path: Path):
    source = tmp_path / "splits_ssa.yaml"
    _write_source(source, _ssa_ids(10), train=["stray_case"])

    with pytest.raises(ValueError, match="train/val"):
        load_cohort_test_ids(source)


def test_load_cohort_test_ids_raises_on_empty_test(tmp_path: Path):
    source = tmp_path / "splits_ssa.yaml"
    _write_source(source, [])

    with pytest.raises(ValueError, match="empty test"):
        load_cohort_test_ids(source)


# ---------------------------------------------------------------------------
# 2. build_cohort_folds: partition, disjointness, val size, epoch rounding
# ---------------------------------------------------------------------------


def test_folds_partition_the_cohort_and_are_pairwise_disjoint():
    ids = [f"case_{i:03d}" for i in range(20)]
    folds = build_cohort_folds(ids, n_folds=2, val_n=5, seed=42)

    assert len(folds) == 2
    for fold in folds:
        train_set = set(fold["train"])
        val_set = set(fold["val"])
        test_set = set(fold["test"])
        assert train_set & val_set == set()
        assert train_set & test_set == set()
        assert val_set & test_set == set()
        # Every fold's train U val U test covers the whole cohort exactly once.
        assert train_set | val_set | test_set == set(ids)

    # Every case is in exactly one fold's train-or-val (its "home" half) and
    # exactly one fold's test (every OTHER fold's half).
    home_counts = {case_id: 0 for case_id in ids}
    test_counts = {case_id: 0 for case_id in ids}
    for fold in folds:
        for case_id in fold["train"] + fold["val"]:
            home_counts[case_id] += 1
        for case_id in fold["test"]:
            test_counts[case_id] += 1
    assert all(count == 1 for count in home_counts.values())
    assert all(count == 1 for count in test_counts.values())


def test_fold_test_equals_union_of_other_folds_train_and_val():
    ids = [f"case_{i:03d}" for i in range(20)]
    folds = build_cohort_folds(ids, n_folds=2, val_n=5, seed=42)

    for f, fold in enumerate(folds):
        other_home = set()
        for k, other_fold in enumerate(folds):
            if k == f:
                continue
            other_home |= set(other_fold["train"]) | set(other_fold["val"])
        assert set(fold["test"]) == other_home


def test_val_size_matches_val_n():
    ids = [f"case_{i:03d}" for i in range(20)]
    folds = build_cohort_folds(ids, n_folds=2, val_n=5, seed=42)
    for fold in folds:
        assert len(fold["val"]) == 5


def test_build_cohort_folds_is_deterministic():
    ids = [f"case_{i:03d}" for i in range(20)]
    folds_a = build_cohort_folds(ids, n_folds=2, val_n=5, seed=42)
    folds_b = build_cohort_folds(ids, n_folds=2, val_n=5, seed=42)
    assert folds_a == folds_b


def test_build_cohort_folds_output_lists_are_sorted():
    ids = [f"case_{i:03d}" for i in range(20)]
    folds = build_cohort_folds(ids, n_folds=2, val_n=5, seed=42)
    for fold in folds:
        assert fold["train"] == sorted(fold["train"])
        assert fold["val"] == sorted(fold["val"])
        assert fold["test"] == sorted(fold["test"])


# ---------------------------------------------------------------------------
# 3. run_make_crossfit_splits: writes files, summary, overwrite rule
# ---------------------------------------------------------------------------


def test_run_writes_fold_files_loadable_through_project_split_loader(tmp_path: Path):
    ssa_source = tmp_path / "splits_ssa.yaml"
    ped_source = tmp_path / "splits_ped.yaml"
    _write_source(ssa_source, _ssa_ids(20))
    _write_source(ped_source, _ped_ids(20))

    cfg = _make_cfg(tmp_path, {"ssa": str(ssa_source), "ped": str(ped_source)})
    result = run_make_crossfit_splits(cfg)

    for cohort in ("ssa", "ped"):
        for fold in (0, 1):
            out_path = tmp_path / f"splits_{cohort}_cf{fold}.yaml"
            assert out_path.is_file()
            loaded = load_splits(out_path)
            expected = result["folds"][cohort][fold]
            assert loaded["train"] == expected["train"]
            assert loaded["val"] == expected["val"]
            assert loaded["test"] == expected["test"]
            # The header comment block must actually be there, not silently
            # dropped by a yaml.safe_dump round-trip.
            text = out_path.read_text(encoding="utf-8")
            assert "FROZEN" in text
            assert "cross-fitted, NOT external validation" in text or "cross-fitted" in text


def test_run_writes_summary_sidecar_next_to_out_pattern(tmp_path: Path):
    ssa_source = tmp_path / "splits_ssa.yaml"
    _write_source(ssa_source, _ssa_ids(20))
    cfg = _make_cfg(tmp_path, {"ssa": str(ssa_source)})

    result = run_make_crossfit_splits(cfg)

    summary_path = tmp_path / "splits_crossfit_summary.yaml"
    assert summary_path.is_file()
    text = summary_path.read_text(encoding="utf-8")
    assert "FROZEN" in text
    assert len(result["summary"]) == 2  # 2 folds for the one configured cohort


def test_run_refuses_to_overwrite_existing_files(tmp_path: Path):
    ssa_source = tmp_path / "splits_ssa.yaml"
    _write_source(ssa_source, _ssa_ids(20))
    cfg = _make_cfg(tmp_path, {"ssa": str(ssa_source)})

    run_make_crossfit_splits(cfg)
    with pytest.raises(FileExistsError):
        run_make_crossfit_splits(cfg)


def test_run_allows_overwrite_when_flagged(tmp_path: Path):
    ssa_source = tmp_path / "splits_ssa.yaml"
    _write_source(ssa_source, _ssa_ids(20))
    cfg = _make_cfg(tmp_path, {"ssa": str(ssa_source)})
    cfg_overwrite = _make_cfg(tmp_path, {"ssa": str(ssa_source)}, overwrite=True)

    first = run_make_crossfit_splits(cfg)
    second = run_make_crossfit_splits(cfg_overwrite)

    assert first["folds"] == second["folds"]


def test_run_never_touches_source_split_files(tmp_path: Path):
    ssa_source = tmp_path / "splits_ssa.yaml"
    _write_source(ssa_source, _ssa_ids(20))
    before = ssa_source.read_text(encoding="utf-8")
    cfg = _make_cfg(tmp_path, {"ssa": str(ssa_source)})

    run_make_crossfit_splits(cfg)

    after = ssa_source.read_text(encoding="utf-8")
    assert before == after


def test_run_raises_on_nonempty_source_train_before_writing_anything(tmp_path: Path):
    ssa_source = tmp_path / "splits_ssa.yaml"
    _write_source(ssa_source, _ssa_ids(20), train=["stray"])
    cfg = _make_cfg(tmp_path, {"ssa": str(ssa_source)})

    with pytest.raises(ValueError):
        run_make_crossfit_splits(cfg)

    assert not (tmp_path / "splits_ssa_cf0.yaml").is_file()


def test_cohort_split_independent_of_other_cohorts_configured(tmp_path: Path):
    ssa_source = tmp_path / "splits_ssa.yaml"
    ped_source = tmp_path / "splits_ped.yaml"
    ssa_ids = _ssa_ids(20)
    _write_source(ssa_source, ssa_ids)
    _write_source(ped_source, _ped_ids(20))

    solo_dir = tmp_path / "solo"
    combo_dir = tmp_path / "combo"
    cfg_solo = _make_cfg(
        tmp_path,
        {"ssa": str(ssa_source)},
        out_pattern=str(solo_dir / "splits_{cohort}_cf{fold}.yaml"),
    )
    cfg_combo = _make_cfg(
        tmp_path,
        {"ssa": str(ssa_source), "ped": str(ped_source)},
        out_pattern=str(combo_dir / "splits_{cohort}_cf{fold}.yaml"),
    )

    solo_result = run_make_crossfit_splits(cfg_solo)
    combo_result = run_make_crossfit_splits(cfg_combo)

    assert solo_result["folds"]["ssa"] == combo_result["folds"]["ssa"]


def test_epoch_rule_rounding_matches_round_3000_over_n_train(tmp_path: Path):
    # 20 cases, 2 folds -> halves of 10, minus val_n=5 -> n_train=5 per fold.
    # round(3000 / 5) = 600, hand-computed.
    ssa_source = tmp_path / "splits_ssa.yaml"
    _write_source(ssa_source, _ssa_ids(20))
    cfg = _make_cfg(tmp_path, {"ssa": str(ssa_source)}, val_n=5)

    result = run_make_crossfit_splits(cfg)

    for row in result["summary"]:
        assert row["n_train"] == 5
        assert row["epochs"] == 600


# ---------------------------------------------------------------------------
# 4. Regression: cfg.analysis.crossfit_splits is really where the real
#    composed config puts this block (mirrors qc_validate's own regression
#    test for cfg.analysis.qc vs cfg.qc).
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_at_the_composed_path(tmp_path: Path):
    ssa_source = tmp_path / "splits_ssa.yaml"
    ped_source = tmp_path / "splits_ped.yaml"
    # 20 cases, not 10: the real config's val_n=5 must leave a non-empty
    # training half (10-case halves - 5 monitoring cases = 0 would divide by
    # zero in the epoch rule).
    _write_source(ssa_source, _ssa_ids(20))
    _write_source(ped_source, _ped_ids(20))

    overrides = [
        f"data.root_dir={tmp_path}",  # mandatory ??? in configs/data/brats.yaml, unused here
        f"analysis.crossfit_splits.cohorts.ssa={ssa_source}",
        f"analysis.crossfit_splits.cohorts.ped={ped_source}",
        f"analysis.crossfit_splits.out_pattern='{tmp_path}/real_splits_{{cohort}}_cf{{fold}}.yaml'",
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    # Real config values survive the compose (n_folds=2, seed=42, val_n=5).
    assert cfg.analysis.crossfit_splits.n_folds == 2
    assert cfg.analysis.crossfit_splits.seed == 42
    assert cfg.analysis.crossfit_splits.val_n == 5

    result = run_make_crossfit_splits(cfg)
    assert set(result["folds"].keys()) == {"ssa", "ped"}
    assert (tmp_path / "real_splits_ssa_cf0.yaml").is_file()
