"""Tests for real_dicom_v2_endpoints, on small synthetic frames (hand-computed values)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from neurovision.analysis import real_dicom_v2_endpoints as ep

NAN = float("nan")
INPUT_QC = "input_qc (pre-preprocessing)"


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Full-schema frame; unspecified fields are NaN / False / None."""
    base = {c: NAN for c in ep.REQUIRED_COLUMNS}
    base.update(
        status="done",
        refusing_stage=None,
        thick_slice_refusal=False,
        accepted=False,
        gate_decision="REFUSE",
        override_status=None,
        override_accepted=False,
        override_gate_decision=None,
    )
    out = []
    for i, r in enumerate(rows):
        row = dict(base, case_id=f"c{i}")
        row.update(r)
        out.append(row)
    return pd.DataFrame(out)


def dice(prefix: str, wt: float, tc: float = 0.9, et: float = 0.9) -> dict:
    return {f"{prefix}_WT": wt, f"{prefix}_TC": tc, f"{prefix}_ET": et}


def done_row(diff: float, qc: float = 0.95, **kw) -> dict:
    """A completed study whose carried WT Dice is research WT + diff."""
    return {
        "status": "done",
        "carry_qc_dice": qc,
        **dice("research_dice", 0.80),
        **dice("carried_dice", 0.80 + diff),
        **kw,
    }


def test_usable_truth_table() -> None:
    wt = pd.Series([0.8, 0.7, 0.69, 0.9, NAN, 0.9])
    tc = pd.Series([0.8, 0.7, 0.9, 0.69, 0.9, NAN])
    got = ep.usable(wt, tc, 0.7).tolist()
    assert got == [True, True, False, False, False, False]


def test_e1_small_large_inconclusive() -> None:
    kw = dict(margin=0.05, carry_qc_threshold=0.9, n_boot=200, seed=1)
    small = ep.e1_front_end_cost(make_df([done_row(-0.01) for _ in range(10)]), **kw)
    assert small["verdict"] == "SMALL"
    assert small["per_region"][2]["median_diff"] == pytest.approx(-0.01)
    large = ep.e1_front_end_cost(make_df([done_row(-0.2) for _ in range(10)]), **kw)
    assert large["verdict"] == "LARGE"
    # Half at -0.01, half at -0.2: the CI of the median straddles the margin.
    mixed = [done_row(-0.01) for _ in range(5)] + [done_row(-0.2) for _ in range(5)]
    assert ep.e1_front_end_cost(make_df(mixed), **kw)["verdict"] == "INCONCLUSIVE"


def test_e1_excludes_carry_qc_failures_and_lists_them() -> None:
    rows = [done_row(-0.01) for _ in range(4)] + [done_row(-0.5, qc=0.5), done_row(-0.5, qc=NAN)]
    res = ep.e1_front_end_cost(
        make_df(rows), margin=0.05, carry_qc_threshold=0.9, n_boot=100, seed=1
    )
    assert res["n_population"] == 4
    assert res["n_excluded_carry_qc"] == 2
    assert res["excluded_carry_qc_case_ids"] == ["c4", "c5"]
    assert res["verdict"] == "SMALL"


def test_e1_population_gatekeeper_in_input_qc_out() -> None:
    rows = [
        done_row(-0.01),
        done_row(-0.01, status="refused", refusing_stage="gatekeeper"),
        # Input-QC refusal: dice present (as if scored) but must NOT be in the population.
        done_row(-0.9, status="refused", refusing_stage=INPUT_QC),
        done_row(-0.9, status="failed"),
    ]
    res = ep.e1_front_end_cost(
        make_df(rows), margin=0.05, carry_qc_threshold=0.9, n_boot=100, seed=1
    )
    assert res["n_population"] == 2
    assert res["n_excluded_carry_qc"] == 0
    assert res["per_region"][2]["median_diff"] == pytest.approx(-0.01)


def thick_row(wt: float, tc: float = 0.9, qc: float = 0.95) -> dict:
    return {
        "status": "refused",
        "refusing_stage": INPUT_QC,
        "thick_slice_refusal": True,
        "override_status": "done",
        "override_carry_qc_dice": qc,
        **dice("override_carried_dice", wt, tc),
        **dice("override_atlas_dice", 0.3, 0.3),
    }


def e2(k: int, n: int) -> dict:
    rows = [thick_row(0.9) for _ in range(k)] + [thick_row(0.2) for _ in range(n - k)]
    return ep.e2_refusal_check(make_df(rows), usable_threshold=0.7, carry_qc_threshold=0.9)


def test_e2_clopper_pearson_matches_scipy() -> None:
    res = e2(24, 36)
    c = res["carried"]
    assert (c["k"], c["n"]) == (24, 36)
    assert c["fraction"] == pytest.approx(24 / 36)
    assert c["ci_lo"] == pytest.approx(stats.beta.ppf(0.025, 24, 13))
    assert c["ci_hi"] == pytest.approx(stats.beta.ppf(0.975, 25, 12))
    assert c["ci_lo"] == pytest.approx(0.4903, abs=1e-3)  # lower bound just under 0.5: INCONCLUSIVE
    assert c["ci_hi"] == pytest.approx(0.8144, abs=1e-3)


def test_e2_verdict_boundaries() -> None:
    assert e2(30, 36)["verdict"] == "REFUSES USABLE STUDIES"
    assert e2(5, 36)["verdict"] == "REFUSAL JUSTIFIED"
    assert e2(18, 36)["verdict"] == "INCONCLUSIVE"
    assert e2(36, 36)["carried"]["ci_hi"] == 1.0
    assert e2(0, 36)["carried"]["ci_lo"] == 0.0


def test_e2_excludes_carry_failures_and_reports_atlas() -> None:
    rows = [thick_row(0.9) for _ in range(3)] + [thick_row(0.9, qc=0.5), thick_row(NAN, NAN, NAN)]
    rows.append(done_row(-0.01))  # not a thick-slice refusal: ignored
    res = ep.e2_refusal_check(make_df(rows), usable_threshold=0.7, carry_qc_threshold=0.9)
    assert res["n_thick"] == 5
    assert res["n_excluded"] == 2
    assert res["excluded_case_ids"] == ["c3", "c4"]
    assert (res["carried"]["k"], res["carried"]["n"]) == (3, 3)
    # Atlas-space override Dice (0.3) is unusable; the NaN row is unscored and skipped.
    assert res["atlas_descriptive"]["n"] == 5 or res["atlas_descriptive"]["k"] == 0


def six_rows() -> pd.DataFrame:
    def r(accepted: bool, wt: float, tc: float, **kw) -> dict:
        return {
            "accepted": accepted,
            "carry_qc_dice": 0.95,
            **dice("atlas_dice", wt, tc),
            **dice("carried_dice", wt, tc),
            **dice("research_dice", 0.8),
            **kw,
        }

    return make_df(
        [
            r(True, 0.9, 0.8),  # correct accept
            r(True, 0.9, 0.5),  # silent failure
            r(False, 0.9, 0.8),  # over-refusal
            r(True, 0.9, 0.8, carry_qc_dice=0.5),  # carry-QC failure
            {  # thick-slice refusal at input QC, overridden run accepted and usable
                "status": "refused",
                "refusing_stage": INPUT_QC,
                "thick_slice_refusal": True,
                "override_status": "done",
                "override_accepted": True,
                "override_carry_qc_dice": 0.95,
                **dice("override_carried_dice", 0.8, 0.8),
            },
            {"status": "failed"},
        ]
    )


def table_value(tab: pd.DataFrame, row: str, column: str) -> pd.Series:
    sel = tab[(tab["row"] == row) & (tab["column"] == column)]
    assert len(sel) == 1
    return sel.iloc[0]


def test_e3_hand_example() -> None:
    tab = ep.e3_table(six_rows(), usable_threshold=0.7, carry_qc_threshold=0.9, n_boot=100, seed=3)
    assert list(tab.columns) == ["row", "column", "value", "ci_lo", "ci_hi", "n"]
    # atlas: accepted = rows 0,1,3; usable & accepted = 0,3 -> 2/6; 2/3; silent 1/6.
    a = lambda row: table_value(tab, row, "atlas")  # noqa: E731
    assert a("p_accepted_and_usable")["value"] == pytest.approx(2 / 6)
    assert a("p_usable_given_accepted")["value"] == pytest.approx(2 / 3)
    assert a("p_usable_given_accepted")["n"] == 3
    assert a("p_silent_failure")["value"] == pytest.approx(1 / 6)
    assert a("completion: done")["value"] == pytest.approx(4 / 6)
    assert a("completion: failed")["value"] == pytest.approx(1 / 6)
    assert a("completion: refused (" + INPUT_QC + ")")["value"] == pytest.approx(1 / 6)
    # carried: row 3 excluded (n=5); accepted = 0,1 -> AU 1/5, 1/2, silent 1/5.
    c = lambda row: table_value(tab, row, "carried")  # noqa: E731
    assert c("carry_qc_excluded")["value"] == 1
    assert c("p_accepted_and_usable")["value"] == pytest.approx(1 / 5)
    assert c("p_accepted_and_usable")["n"] == 5
    assert c("p_usable_given_accepted")["value"] == pytest.approx(1 / 2)
    assert c("p_silent_failure")["value"] == pytest.approx(1 / 5)
    # overridden: row 4 now accepted and usable -> AU 2/5, 2/3, silent 1/5.
    o = lambda row: table_value(tab, row, "overridden")  # noqa: E731
    assert o("p_accepted_and_usable")["value"] == pytest.approx(2 / 5)
    assert o("p_usable_given_accepted")["value"] == pytest.approx(2 / 3)
    assert o("p_silent_failure")["value"] == pytest.approx(1 / 5)
    assert o("completion: done")["value"] == pytest.approx(5 / 6)  # thick row now done
    # Front-end cost: carried WT 0.9 vs research 0.8 on the 3 non-excluded masked rows.
    assert c("front_end_cost_median_WT")["value"] == pytest.approx(0.1)
    assert c("front_end_cost_median_WT")["n"] == 3


def test_e4_uses_override_accepted_for_thick_rows() -> None:
    df = six_rows()
    kw = dict(usable_threshold=0.7, carry_qc_threshold=0.9, n_boot=100, seed=3)
    res = ep.e4_caution_variant(df, **kw)
    assert res["label"] == "post-hoc variant"
    assert res["n_thick_slice"] == 1
    assert res["n_thick_slice_accepted"] == 1
    assert res["p_accepted_and_usable"][0] == pytest.approx(2 / 5)
    # If the overridden run was NOT accepted by the gate, the thick row does not count.
    df.loc[4, "override_accepted"] = False
    res2 = ep.e4_caution_variant(df, **kw)
    assert res2["n_thick_slice_accepted"] == 0
    assert res2["p_accepted_and_usable"][0] == pytest.approx(1 / 5)
    assert res2["p_usable_given_accepted"][0] == pytest.approx(1 / 2)


def test_bridge_check_boundary() -> None:
    s = pd.Series({"a": 0.99, "b": 0.995, "c": 0.9899, "d": NAN})
    res = ep.bridge_check(s, min_dice=0.99)
    assert res["below_min_dice"] == ["c", "d"]
    assert res["all_pass"] is False
    ok = ep.bridge_check(pd.Series({"a": 0.99, "b": 1.0}), min_dice=0.99)
    assert ok["all_pass"] is True and ok["below_min_dice"] == []
    assert ep.bridge_check(pd.Series(dtype=float), min_dice=0.99)["all_pass"] is False


def test_missing_columns_raise() -> None:
    df = make_df([done_row(-0.01)]).drop(columns=["carry_qc_dice", "override_status"])
    for fn, kw in [
        (ep.e1_front_end_cost, dict(margin=0.05, carry_qc_threshold=0.9, n_boot=10, seed=1)),
        (ep.e2_refusal_check, dict(usable_threshold=0.7, carry_qc_threshold=0.9)),
        (ep.e3_table, dict(usable_threshold=0.7, carry_qc_threshold=0.9, n_boot=10, seed=1)),
        (
            ep.e4_caution_variant,
            dict(usable_threshold=0.7, carry_qc_threshold=0.9, n_boot=10, seed=1),
        ),
    ]:
        with pytest.raises(ValueError, match="carry_qc_dice.*override_status"):
            fn(df, **kw)


def test_same_seed_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    rows = [done_row(float(d)) for d in rng.normal(-0.02, 0.05, size=15)]
    df = make_df(rows)
    kw = dict(margin=0.05, carry_qc_threshold=0.9, n_boot=200, seed=7)
    assert ep.e1_front_end_cost(df, **kw) == ep.e1_front_end_cost(df, **kw)
    t_kw = dict(usable_threshold=0.7, carry_qc_threshold=0.9, n_boot=100, seed=7)
    pd.testing.assert_frame_equal(ep.e3_table(df, **t_kw), ep.e3_table(df, **t_kw))
