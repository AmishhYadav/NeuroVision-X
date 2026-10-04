"""Tests for scripts/silent_failure_by_region.py, on tiny synthetic CSVs in tmp_path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.script_loader import load_script

sfr = load_script("silent_failure_by_region")

REGIONS = ["WT", "TC"]
BAR = 0.7

# (case_id, accepted, dice_WT, dice_TC)
#  c1 accepted, both fine            -> correct_accept
#  c2 accepted, TC fails             -> silent, TC only
#  c3 accepted, WT fails             -> silent, WT only
#  c4 accepted, both fail            -> silent, both
#  c5 refused, TC fails              -> correct_refusal
#  c6 refused, both fine             -> over_refusal
#  c7 accepted, Dice exactly 0.7     -> usable (>=), correct_accept
ROWS = [
    ("c1", True, 0.9, 0.9),
    ("c2", True, 0.9, 0.3),
    ("c3", True, 0.4, 0.9),
    ("c4", True, 0.2, 0.1),
    ("c5", False, 0.9, 0.2),
    ("c6", False, 0.8, 0.8),
    ("c7", True, 0.7, 0.7),
]


def _frame(rows: list = ROWS) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["case_id", "accepted", "dice_WT", "dice_TC"])
    df["usable"] = (df["dice_WT"] >= BAR) & (df["dice_TC"] >= BAR)

    def cell(r: pd.Series) -> str:
        if r["accepted"]:
            return "correct_accept" if r["usable"] else "silent_failure"
        return "over_refusal" if r["usable"] else "correct_refusal"

    df["cell"] = df.apply(cell, axis=1)
    # Decoys with different values: must never be picked up.
    df["predicted_dice_WT"] = 0.0
    df["predicted_dice_TC"] = 1.0
    return df


def test_hand_computed_counts() -> None:
    row, cases = sfr.analyse_cohort(_frame(), "toy", REGIONS, BAR)
    assert row["n"] == 7
    assert row["n_accepted"] == 5
    assert row["n_silent"] == 3
    assert row["silent_rate"] == pytest.approx(3 / 7)
    assert row["only_TC"] == 1
    assert row["only_WT"] == 1
    assert row["multiple"] == 1
    # WT alone below bar among accepted: c3, c4. TC alone: c2, c4.
    assert row["silent_if_WT_only"] == 2
    assert row["silent_if_TC_only"] == 2
    assert row["silent_rate_if_WT_only"] == pytest.approx(2 / 7)
    assert list(cases["case_id"]) == ["c2", "c3", "c4"]
    assert list(cases["failing_regions"]) == ["TC", "WT", "WT+TC"]


def test_exactly_at_bar_is_usable() -> None:
    df = _frame([("a", True, 0.7, 0.7)])
    row, cases = sfr.analyse_cohort(df, "toy", REGIONS, BAR)
    assert row["n_silent"] == 0
    assert len(cases) == 0


def test_inconsistent_usable_raises() -> None:
    df = _frame()
    df.loc[df["case_id"] == "c2", "usable"] = True
    with pytest.raises(ValueError, match="toy.*c2"):
        sfr.analyse_cohort(df, "toy", REGIONS, BAR)


def test_inconsistent_cell_raises() -> None:
    df = _frame()
    df.loc[df["case_id"] == "c3", "cell"] = "correct_accept"
    with pytest.raises(ValueError, match="toy.*c3"):
        sfr.analyse_cohort(df, "toy", REGIONS, BAR)


def test_nan_dice_raises() -> None:
    df = _frame()
    df.loc[0, "dice_TC"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        sfr.analyse_cohort(df, "toy", REGIONS, BAR)


def test_string_booleans_parse_and_junk_raises() -> None:
    df = _frame()
    for col in ("accepted", "usable"):
        df[col] = df[col].map({True: "True", False: "False"})
    row, _ = sfr.analyse_cohort(df, "toy", REGIONS, BAR)
    assert row["n_silent"] == 3

    df.loc[0, "accepted"] = "yes"
    with pytest.raises(ValueError, match="cannot parse"):
        sfr.analyse_cohort(df, "toy", REGIONS, BAR)


def test_run_writes_files(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _frame().to_csv(in_dir / "per_case_toy.csv", index=False)
    out = tmp_path / "out"
    cfg = {
        "out_dir": str(out),
        "per_case_dir": str(in_dir),
        "cohorts": ["toy"],
        "usable_regions": REGIONS,
        "usable_bar": BAR,
    }
    table = sfr.run(cfg)
    assert list(table["cohort"]) == ["toy"]
    written = pd.read_csv(out / "silent_failure_by_region.csv")
    assert written.loc[0, "only_TC"] == 1
    cases = pd.read_csv(out / "silent_failure_cases.csv")
    assert list(cases.columns) == [
        "cohort",
        "case_id",
        "dice_WT",
        "dice_TC",
        "failing_regions",
    ]
    assert (out / "silent_failure_by_region_config.yaml").exists()


def test_empty_cohort_raises() -> None:
    df = _frame().iloc[0:0]
    with pytest.raises(ValueError, match="toy.*empty"):
        sfr.analyse_cohort(df, "toy", REGIONS, BAR)


def test_no_silent_failures_gives_object_column() -> None:
    df = _frame([("a", True, 0.9, 0.9), ("b", False, 0.1, 0.1)])
    row, cases = sfr.analyse_cohort(df, "toy", REGIONS, BAR)
    assert row["n_silent"] == 0
    assert row["only_WT"] == row["only_TC"] == row["multiple"] == 0
    assert len(cases) == 0
    assert cases["failing_regions"].dtype == object


def test_single_region() -> None:
    df = _frame()
    # Re-derive the file's usable/cell from WT alone, as a WT-only budget would.
    df["usable"] = df["dice_WT"] >= BAR
    df["cell"] = [
        (
            ("correct_accept" if u else "silent_failure")
            if a
            else ("over_refusal" if u else "correct_refusal")
        )
        for a, u in zip(df["accepted"], df["usable"], strict=True)
    ]
    row, cases = sfr.analyse_cohort(df, "toy", ["WT"], BAR)
    # Silent = accepted and WT < bar: c3, c4.
    assert row["n_silent"] == 2
    assert row["only_WT"] == row["n_silent"]
    assert row["multiple"] == 0
    assert list(cases["failing_regions"]) == ["WT", "WT"]
