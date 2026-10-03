"""Tests for scripts/ref_et_comparison.py, on tiny synthetic CSVs in tmp_path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.script_loader import load_script

ref = load_script("ref_et_comparison")

IDS = ["c1", "c2", "c3", "c4"]
FLAG = [True, False, False, False]  # c1 has an empty reference ET


def _write(path: Path, dice: list[float], flag: list = FLAG, ids: list[str] = IDS) -> Path:
    pd.DataFrame({"case_id": ids, "dice_ET": dice, "gt_empty_ET": flag}).to_csv(path, index=False)
    return path


@pytest.fixture
def arms(tmp_path: Path) -> dict[str, Path]:
    # A - B = [1.0, 0.1, 0.0, -0.1]
    return {
        "A": _write(tmp_path / "a.csv", [1.0, 0.5, 0.6, 0.2]),
        "B": _write(tmp_path / "b.csv", [0.0, 0.4, 0.6, 0.3]),
    }


def _cfg(arms: dict[str, Path], out_dir: Path) -> dict:
    return {
        "out_dir": str(out_dir),
        "metric": "dice_ET",
        "empty_flag_column": "gt_empty_ET",
        "n_boot": 200,
        "ci": 0.95,
        "seed": 1,
        "arms": {k: str(v) for k, v in arms.items()},
        "comparisons": [["A", "B"]],
    }


def test_hodges_lehmann_hand_values() -> None:
    # Walsh averages 1, 1.5, 2, 2, 2.5, 3 -> median 2.0
    assert ref.hodges_lehmann(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)
    # [1,2,3,4]: Walsh sorted 1,1.5,2,2,2.5,2.5,3,3,3.5,4 -> (2.5 + 2.5) / 2
    assert ref.hodges_lehmann(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(2.5)
    # [1,3]: Walsh 1, 2, 3 -> 2
    assert ref.hodges_lehmann(np.array([1.0, 3.0])) == pytest.approx(2.0)
    # Diagonal (i == j) matters: with it 0,0,0,5,5,10 -> 2.5; without it 0,5,5 -> 5
    assert ref.hodges_lehmann(np.array([0.0, 0.0, 10.0])) == pytest.approx(2.5)


def test_run_hand_values_and_files(arms: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "out"
    table = ref.run(_cfg(arms, out))
    all_row = table[table["subset"] == "all"].iloc[0]
    ref_row = table[table["subset"] == "ref_et"].iloc[0]

    assert all_row["n"] == 4 and ref_row["n"] == 3  # ref_et drops exactly c1
    assert all_row["mean_diff"] == pytest.approx(0.25)
    assert ref_row["mean_diff"] == pytest.approx(0.0)
    assert (all_row["wins"], all_row["ties"], all_row["losses"]) == (2, 1, 1)
    assert (ref_row["wins"], ref_row["ties"], ref_row["losses"]) == (1, 1, 1)
    # empty-case diff sum = 1.0, divided by all n = 4
    assert all_row["empty_contribution"] == pytest.approx(0.25)
    assert np.isnan(ref_row["empty_contribution"])

    on_disk = pd.read_csv(out / "ref_et_comparisons.csv")
    assert list(on_disk.columns) == [
        "a",
        "b",
        "subset",
        "n",
        "mean_diff",
        "ci_lo",
        "ci_hi",
        "p_wilcoxon",
        "wins",
        "ties",
        "losses",
        "hodges_lehmann",
        "empty_contribution",
    ]
    empty = pd.read_csv(out / "empty_et_cases.csv")
    assert empty["case_id"].tolist() == ["c1"]
    assert empty.loc[0, "A"] == 1.0 and empty.loc[0, "B"] == 0.0
    assert (out / "ref_et_config.yaml").is_file()


def test_ci_independent_of_row_order(arms: dict[str, Path], tmp_path: Path) -> None:
    cfg = _cfg(arms, tmp_path / "o1")
    first = ref.run(cfg)
    cfg2 = _cfg(arms, tmp_path / "o2")
    cfg2["comparisons"] = [["B", "A"], ["A", "B"]]  # A-B now runs second
    second = ref.run(cfg2)
    a = first[first["subset"] == "ref_et"].iloc[0]
    b = second[(second["a"] == "A") & (second["subset"] == "ref_et")].iloc[0]
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])


def test_string_flags_parsed(tmp_path: Path) -> None:
    cases = {
        "words": ["True", "False", "false", "FALSE"],
        "digits": ["1", "0", "0", "0"],  # object dtype once mixed with a non-numeric row below
    }
    for name, flags in cases.items():
        df = pd.DataFrame({"case_id": IDS, "dice_ET": [1.0] * 4, "gt_empty_ET": flags})
        path = tmp_path / f"{name}.csv"
        df.to_csv(path, index=False)
        got = ref.load_arms({"x": path}, "dice_ET", "gt_empty_ET")["x"]["gt_empty_ET"]
        assert got.tolist() == FLAG
    # Direct call with an object-dtype "1"/"0" series (what a mixed column becomes).
    series = pd.Series(["1", "0", "0", "0"], index=IDS, dtype=object)
    assert ref._parse_flag(series, "x", "p").tolist() == FLAG
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {"case_id": IDS, "dice_ET": [1.0] * 4, "gt_empty_ET": ["True", "x", "no", "False"]}
    ).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="bad.csv"):
        ref.load_arms({"x": bad}, "dice_ET", "gt_empty_ET")


def test_numeric_flag_outside_0_1_raises(tmp_path: Path) -> None:
    p = _write(tmp_path / "two.csv", [1.0] * 4, flag=[2.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="two.csv"):
        ref.load_arms({"x": p}, "dice_ET", "gt_empty_ET")


def test_missing_case_id_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "noid.csv"
    pd.DataFrame({"dice_ET": [1.0] * 4, "gt_empty_ET": FLAG}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="noid.csv"):
        ref.load_arms({"x": p}, "dice_ET", "gt_empty_ET")


def test_alignment_is_by_case_id_not_row_order(arms: dict[str, Path], tmp_path: Path) -> None:
    base = ref.run(_cfg(arms, tmp_path / "o1"))
    order = [3, 1, 0, 2]
    b_vals = [0.0, 0.4, 0.6, 0.3]
    shuffled = _write(
        tmp_path / "b_shuf.csv",
        [b_vals[i] for i in order],
        flag=[FLAG[i] for i in order],
        ids=[IDS[i] for i in order],
    )
    shuf = ref.run(_cfg({"A": arms["A"], "B": shuffled}, tmp_path / "o2"))
    cols = ["n", "mean_diff", "wins", "ties", "losses", "hodges_lehmann"]
    pd.testing.assert_frame_equal(base[cols], shuf[cols])
    assert base["empty_contribution"].equals(shuf["empty_contribution"])


def test_empty_contribution_divides_by_n_all(tmp_path: Path) -> None:
    flag = [True, True, False, False, False]
    ids = ["c1", "c2", "c3", "c4", "c5"]
    # diffs: 1.0, 0.5 (empty cases), 0, 0, 0 -> empty sum 1.5; / n_all(5) = 0.3, not / 3 = 0.5
    a = _write(tmp_path / "a.csv", [1.0, 0.5, 0.2, 0.2, 0.2], flag=flag, ids=ids)
    b = _write(tmp_path / "b.csv", [0.0, 0.0, 0.2, 0.2, 0.2], flag=flag, ids=ids)
    table = ref.run(_cfg({"A": a, "B": b}, tmp_path / "o"))
    row = table[table["subset"] == "all"].iloc[0]
    assert row["empty_contribution"] == pytest.approx(0.3)


def test_numeric_flags_parsed(tmp_path: Path) -> None:
    p = _write(tmp_path / "n.csv", [1.0] * 4, flag=[1.0, 0.0, 0.0, 0.0])
    got = ref.load_arms({"x": p}, "dice_ET", "gt_empty_ET")["x"]["gt_empty_ET"]
    assert got.tolist() == FLAG


def test_mismatched_case_sets_raise(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", [0.1] * 4)
    b = _write(tmp_path / "b_other.csv", [0.1] * 4, ids=["c1", "c2", "c3", "zz"])
    with pytest.raises(ValueError, match="b_other.csv"):
        ref.load_arms({"A": a, "B": b}, "dice_ET", "gt_empty_ET")


def test_disagreeing_flags_raise(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", [0.1] * 4)
    b = _write(tmp_path / "b_flag.csv", [0.1] * 4, flag=[False, True, False, False])
    with pytest.raises(ValueError, match="b_flag.csv"):
        ref.load_arms({"A": a, "B": b}, "dice_ET", "gt_empty_ET")


def test_nan_metric_raises(tmp_path: Path) -> None:
    a = _write(tmp_path / "a_nan.csv", [0.1, float("nan"), 0.1, 0.1])
    with pytest.raises(ValueError, match="a_nan.csv"):
        ref.load_arms({"A": a}, "dice_ET", "gt_empty_ET")
