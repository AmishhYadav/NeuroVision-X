"""Tests for scripts/run_ablation_grid.py.

Pure planning tool: no torch, no GPU, no real BraTS data, no real Hydra
composition of the full config tree (that would pull in the whole `configs/`
directory, which is also being edited concurrently by other work in this
repo -- these tests must not depend on its exact contents). Instead,
`compose_variant` is monkeypatched with a tiny stand-in `SimpleNamespace` cfg
wherever a composed config is needed, exactly the way `tests/test_train_script.py`
loads `scripts/train.py` via `importlib.util.spec_from_file_location` since
`scripts/` has no `__init__.py`.

One test (`test_real_ablation_full_composes_against_the_real_repo_configs`)
does compose the real `ablation_full` config, as a light integration check
that the script's Hydra usage actually works -- it is skipped if that
experiment config does not exist yet, since the module docstring is explicit
that `ablation_cnn_only`/`ablation_transformer_only` (and by extension, other
in-flight ablation configs) may not exist when this script runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation_grid.py"
_spec = importlib.util.spec_from_file_location("run_ablation_grid_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
grid_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["run_ablation_grid_script"] = grid_script
_spec.loader.exec_module(grid_script)

GRID_VARIANTS = grid_script.GRID_VARIANTS
CostEstimate = grid_script.CostEstimate
VariantReport = grid_script.VariantReport
estimate_variant_cost = grid_script.estimate_variant_cost
load_case_counts = grid_script.load_case_counts
compose_variant = grid_script.compose_variant
build_variant_reports = grid_script.build_variant_reports
format_report_text = grid_script.format_report_text
format_report_markdown = grid_script.format_report_markdown
parse_relative_cost_overrides = grid_script.parse_relative_cost_overrides
parse_args = grid_script.parse_args
main = grid_script.main


def _fake_cfg(
    epochs: int = 40,
    batch_size: int = 1,
    val_interval: int = 5,
    max_hours: float = 11.0,
) -> SimpleNamespace:
    """Builds a minimal stand-in for a composed Hydra config.

    Only exposes the attributes `build_variant_reports` actually reads, so
    tests never depend on the shape of the real config tree.
    """
    training = SimpleNamespace(
        epochs=epochs, batch_size=batch_size, val_interval=val_interval, max_hours=max_hours
    )
    return SimpleNamespace(training=training)


def _touch_experiment_configs(configs_dir: Path, names: list[str]) -> None:
    """Creates empty placeholder `experiment/<name>.yaml` files.

    Content is irrelevant in tests that monkeypatch `compose_variant` --
    only the file's *existence* is read directly by `build_variant_reports`.
    """
    exp_dir = configs_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (exp_dir / f"{name}.yaml").write_text("# placeholder\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# estimate_variant_cost
# ---------------------------------------------------------------------------


def test_estimate_variant_cost_hand_computed():
    """100 train cases, batch 1, 1.0s/step, 10 epochs, val every 5 epochs,
    10 val cases, 2.0s/val-case -> 100*1.0*10 + 2*10*2.0 = 1040 s.
    """
    cost = estimate_variant_cost(
        n_train_cases=100,
        n_val_cases=10,
        batch_size=1,
        epochs=10,
        val_interval=5,
        max_hours=100.0,
        sec_per_step=1.0,
        sec_per_val_case=2.0,
        relative_cost=1.0,
    )
    assert cost.steps_per_epoch == 100
    assert cost.n_val_passes == 2
    assert cost.total_seconds == pytest.approx(1040.0)
    assert cost.total_hours == pytest.approx(1040.0 / 3600.0)


def test_steps_per_epoch_uses_ceil_division_and_ignores_samples_per_volume():
    """10 train cases at batch_size 3 -> ceil(10/3) = 4 steps, not 3.33 or 10/3/anything.

    `estimate_variant_cost` takes no `samples_per_volume` argument at all --
    dividing by it (on top of batch_size) would understate cost by up to 4x,
    since RandCropByPosNegLabeld's crops are flattened into the batch AFTER
    the DataLoader already batched by case count.
    """
    cost = estimate_variant_cost(
        n_train_cases=10,
        n_val_cases=1,
        batch_size=3,
        epochs=1,
        val_interval=1,
        max_hours=100.0,
        sec_per_step=1.0,
        sec_per_val_case=0.0,
        relative_cost=1.0,
    )
    assert cost.steps_per_epoch == 4  # ceil(10 / 3), not floor (3) or 10/3 (3.33)


def test_sessions_ceil_against_max_hours():
    """total_hours slightly over one max_hours window must round UP to 2 sessions."""
    # steps_per_epoch=1, 1 epoch, sec_per_step chosen so total_hours is just over 1.0.
    cost = estimate_variant_cost(
        n_train_cases=1,
        n_val_cases=0,
        batch_size=1,
        epochs=1,
        val_interval=1,
        max_hours=1.0,
        sec_per_step=3601.0,  # exactly 1 step * 3601s = 1h + 1s
        sec_per_val_case=0.0,
        relative_cost=1.0,
    )
    assert cost.total_hours > 1.0
    assert cost.sessions == 2

    # And exactly at the boundary, ceil(1.0) == 1, not 2.
    cost_exact = estimate_variant_cost(
        n_train_cases=1,
        n_val_cases=0,
        batch_size=1,
        epochs=1,
        val_interval=1,
        max_hours=1.0,
        sec_per_step=3600.0,
        sec_per_val_case=0.0,
        relative_cost=1.0,
    )
    assert cost_exact.total_hours == pytest.approx(1.0)
    assert cost_exact.sessions == 1


def test_relative_cost_multiplier_halves_the_estimate():
    base = estimate_variant_cost(
        n_train_cases=100,
        n_val_cases=10,
        batch_size=1,
        epochs=10,
        val_interval=5,
        max_hours=100.0,
        sec_per_step=1.0,
        sec_per_val_case=2.0,
        relative_cost=1.0,
    )
    halved = estimate_variant_cost(
        n_train_cases=100,
        n_val_cases=10,
        batch_size=1,
        epochs=10,
        val_interval=5,
        max_hours=100.0,
        sec_per_step=1.0,
        sec_per_val_case=2.0,
        relative_cost=0.5,
    )
    assert halved.total_seconds == pytest.approx(base.total_seconds * 0.5)


# ---------------------------------------------------------------------------
# parse_relative_cost_overrides
# ---------------------------------------------------------------------------


def test_relative_cost_cli_override_updates_only_named_variant():
    table = parse_relative_cost_overrides(["ablation_full=0.5"])
    assert table["ablation_full"] == 0.5
    # Every other variant keeps its default value.
    assert (
        table["ablation_fusion_concat"]
        == grid_script.DEFAULT_RELATIVE_COST["ablation_fusion_concat"]
    )


def test_relative_cost_cli_override_rejects_unknown_variant():
    with pytest.raises(ValueError):
        parse_relative_cost_overrides(["not_a_real_variant=0.5"])


def test_relative_cost_cli_override_rejects_malformed_entry():
    with pytest.raises(ValueError):
        parse_relative_cost_overrides(["ablation_full_without_equals_sign"])


# ---------------------------------------------------------------------------
# build_variant_reports: missing config, compose failure, canonical order
# ---------------------------------------------------------------------------


def test_missing_experiment_config_is_reported_and_excluded(tmp_path: Path, monkeypatch):
    configs_dir = tmp_path / "configs"
    # Only ablation_full's file exists; the other five are absent, matching
    # the real repo state the spec warns about (cnn_only / transformer_only
    # "may not exist when this script runs").
    _touch_experiment_configs(configs_dir, ["ablation_full"])

    monkeypatch.setattr(grid_script, "compose_variant", lambda *a, **k: _fake_cfg())

    reports = build_variant_reports(
        configs_dir=configs_dir,
        variants=GRID_VARIANTS,
        data_root_dir="$PREP_DIR",
        n_train=100,
        n_val=10,
        sec_per_step=1.0,
        sec_per_val_case=1.0,
        relative_cost=dict(grid_script.DEFAULT_RELATIVE_COST),
    )

    by_name = {r.name: r for r in reports}
    assert by_name["ablation_full"].status == "ok"
    assert by_name["ablation_full"].cost is not None

    for missing_name in (
        "ablation_fusion_concat",
        "ablation_fusion_add",
        "ablation_cnn_only",
        "ablation_transformer_only",
        "ablation_no_deep_supervision",
    ):
        row = by_name[missing_name]
        assert row.status == "missing_config"
        assert row.cost is None
        assert "MISSING CONFIG" in (row.detail or "")

    # Missing rows never crash report formatting, and are excluded from totals.
    text = format_report_text(
        reports,
        weekly_budget_hours=30.0,
        prep_dir="$PREP_DIR",
        n_train=100,
        n_val=10,
        counts_assumed=False,
    )
    assert "MISSING CONFIG" in text
    assert "excluded from the totals" in text
    # Only the composable variant's total appears in the "Total estimated" line.
    assert by_name["ablation_full"].cost is not None


def test_compose_failure_is_reported_and_excluded_not_raised(tmp_path: Path, monkeypatch):
    configs_dir = tmp_path / "configs"
    _touch_experiment_configs(configs_dir, ["ablation_full"])

    def _raise(*args, **kwargs):
        raise RuntimeError("boom: unresolvable interpolation")

    monkeypatch.setattr(grid_script, "compose_variant", _raise)

    reports = build_variant_reports(
        configs_dir=configs_dir,
        variants=["ablation_full"],
        data_root_dir="$PREP_DIR",
        n_train=100,
        n_val=10,
        sec_per_step=1.0,
        sec_per_val_case=1.0,
        relative_cost=dict(grid_script.DEFAULT_RELATIVE_COST),
    )
    assert len(reports) == 1
    assert reports[0].status == "compose_failed"
    assert reports[0].cost is None
    assert "COMPOSE FAILED" in (reports[0].detail or "")
    assert "boom" in (reports[0].detail or "")


def test_variants_are_emitted_in_canonical_grid_order_regardless_of_cli_order(
    tmp_path: Path, monkeypatch
):
    configs_dir = tmp_path / "configs"
    _touch_experiment_configs(configs_dir, list(GRID_VARIANTS))
    monkeypatch.setattr(grid_script, "compose_variant", lambda *a, **k: _fake_cfg())

    reversed_request = list(reversed(GRID_VARIANTS))
    reports = build_variant_reports(
        configs_dir=configs_dir,
        variants=reversed_request,
        data_root_dir="$PREP_DIR",
        n_train=100,
        n_val=10,
        sec_per_step=1.0,
        sec_per_val_case=1.0,
        relative_cost=dict(grid_script.DEFAULT_RELATIVE_COST),
    )
    assert [r.name for r in reports] == list(GRID_VARIANTS)

    # A subset, still requested out of order, still comes back in grid order.
    subset_reversed = ["ablation_no_deep_supervision", "ablation_full"]
    subset_reports = build_variant_reports(
        configs_dir=configs_dir,
        variants=subset_reversed,
        data_root_dir="$PREP_DIR",
        n_train=100,
        n_val=10,
        sec_per_step=1.0,
        sec_per_val_case=1.0,
        relative_cost=dict(grid_script.DEFAULT_RELATIVE_COST),
    )
    assert [r.name for r in subset_reports] == ["ablation_full", "ablation_no_deep_supervision"]


# ---------------------------------------------------------------------------
# load_case_counts
# ---------------------------------------------------------------------------


def test_load_case_counts_reads_real_splits_yaml(tmp_path: Path):
    splits_path = tmp_path / "splits.yaml"
    splits_path.write_text(
        "train:\n- case_a\n- case_b\n- case_c\nval:\n- case_d\ntest: []\n", encoding="utf-8"
    )
    n_train, n_val, assumed = load_case_counts(splits_path, None, None)
    assert (n_train, n_val, assumed) == (3, 1, False)


def test_load_case_counts_falls_back_when_splits_missing(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.yaml"
    n_train, n_val, assumed = load_case_counts(missing_path, 875, 187)
    assert (n_train, n_val, assumed) == (875, 187, True)


def test_load_case_counts_raises_without_fallback_when_splits_missing(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ValueError):
        load_case_counts(missing_path, None, None)


# ---------------------------------------------------------------------------
# format_report_text / format_report_markdown: basic shape checks
# ---------------------------------------------------------------------------


def test_format_report_text_contains_commands_in_grid_order():
    cost = CostEstimate(
        steps_per_epoch=10,
        n_val_passes=2,
        total_seconds=100.0,
        total_hours=100.0 / 3600,
        sessions=1,
    )
    reports = [
        VariantReport("ablation_full", "ok", None, 40, 1.0, cost),
        VariantReport("ablation_fusion_concat", "ok", None, 40, 0.88, cost),
    ]
    text = format_report_text(
        reports,
        weekly_budget_hours=30.0,
        prep_dir="/data/prep",
        n_train=10,
        n_val=2,
        counts_assumed=False,
    )
    idx_full = text.index("+experiment=ablation_full")
    idx_concat = text.index("+experiment=ablation_fusion_concat")
    assert idx_full < idx_concat
    assert "data.root_dir=/data/prep" in text


def test_format_report_markdown_contains_a_pipe_table():
    cost = CostEstimate(
        steps_per_epoch=10,
        n_val_passes=2,
        total_seconds=100.0,
        total_hours=100.0 / 3600,
        sessions=1,
    )
    reports = [VariantReport("ablation_full", "ok", None, 40, 1.0, cost)]
    text = format_report_markdown(
        reports,
        weekly_budget_hours=30.0,
        prep_dir="$PREP_DIR",
        n_train=10,
        n_val=2,
        counts_assumed=True,
    )
    assert "| ablation_full |" in text
    assert "ASSUMED" in text


# ---------------------------------------------------------------------------
# main(): CLI wiring, exit codes
# ---------------------------------------------------------------------------


def test_main_exits_zero_and_prints_report(tmp_path: Path, monkeypatch, capsys):
    configs_dir = tmp_path / "configs"
    _touch_experiment_configs(configs_dir, list(GRID_VARIANTS))
    monkeypatch.setattr(grid_script, "compose_variant", lambda *a, **k: _fake_cfg())
    # Bypass splits.yaml resolution entirely so counts come from --n-train/--n-val,
    # not from whatever the real repo's splits.yaml happens to contain.
    monkeypatch.setattr(grid_script, "_resolve_splits_path", lambda *a, **k: None)

    exit_code = main(
        [
            "--configs-dir",
            str(configs_dir),
            "--n-train",
            "8",
            "--n-val",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ablation_full" in captured.out
    assert "n_train=8, n_val=2" in captured.out


def test_main_exits_one_when_a_variant_fails_to_compose(tmp_path: Path, monkeypatch, capsys):
    configs_dir = tmp_path / "configs"
    _touch_experiment_configs(configs_dir, list(GRID_VARIANTS))
    monkeypatch.setattr(grid_script, "_resolve_splits_path", lambda *a, **k: None)

    def _flaky_compose(configs_dir, variant, data_root_dir):
        if variant == "ablation_full":
            raise RuntimeError("bad interpolation")
        return _fake_cfg()

    monkeypatch.setattr(grid_script, "compose_variant", _flaky_compose)

    exit_code = main(["--configs-dir", str(configs_dir), "--n-train", "8", "--n-val", "2"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "COMPOSE FAILED" in captured.out


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args([])
    assert args.sec_per_step == 1.1
    assert args.sec_per_val_case == 3.0
    assert args.weekly_budget_hours == 30.0
    assert args.prep_dir == "$PREP_DIR"
    assert args.variants is None
    assert args.format == "text"


def test_parse_args_relative_cost_is_repeatable():
    args = parse_args(
        ["--relative-cost", "ablation_full=0.9", "--relative-cost", "ablation_fusion_add=0.4"]
    )
    assert args.relative_cost == ["ablation_full=0.9", "ablation_fusion_add=0.4"]


# ---------------------------------------------------------------------------
# Light real-config integration check
# ---------------------------------------------------------------------------


def test_real_ablation_full_composes_against_the_real_repo_configs():
    """One integration check against the real configs/ tree.

    Skipped rather than failing if ablation_full.yaml is not present yet --
    it is not this test's job to assert the ablation configs exist, only that
    `compose_variant` correctly drives Hydra when they do.
    """
    real_configs_dir = Path(grid_script._DEFAULT_CONFIGS_DIR)
    if not (real_configs_dir / "experiment" / "ablation_full.yaml").is_file():
        pytest.skip("configs/experiment/ablation_full.yaml does not exist yet")

    cfg = compose_variant(real_configs_dir, "ablation_full", data_root_dir="/tmp/dummy_prep")
    assert cfg.training.batch_size >= 1
    assert cfg.training.epochs >= 1
    assert cfg.training.max_hours > 0
