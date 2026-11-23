"""Tests for scripts/audit_report_numbers.py (synthetic files only)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_SPEC = importlib.util.spec_from_file_location(
    "audit_report_numbers",
    Path(__file__).resolve().parent.parent / "scripts" / "audit_report_numbers.py",
)
ar = importlib.util.module_from_spec(_SPEC)
sys.modules["audit_report_numbers"] = ar  # dataclasses needs this
_SPEC.loader.exec_module(ar)


def make_entry(**kw):
    base = dict(
        id="a",
        text="+0.0267",
        claim="C1",
        file="f.csv",
        decimals=4,
        column="v",
        where={"m": "x"},
        sign=True,
    )
    base.update(kw)
    return ar.Entry(**base)


def write_csv(tmp_path, rows):
    lines = ["m,alpha,v"] + [",".join(r) for r in rows]
    (tmp_path / "f.csv").write_text("\n".join(lines) + "\n")


def kinds(doc, entries, root):
    return [m.kind for m in ar.audit(doc, entries, root)]


def test_csv_exact_match(tmp_path):
    write_csv(tmp_path, [["x", "0.1", "0.02671"], ["y", "0.1", "9"]])
    assert kinds("gain +0.0267 here", [make_entry()], tmp_path) == []


def test_json_key_path(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"a": {"b": [1.0, 0.5]}}))
    e = make_entry(
        file="r.json", column=None, key="a.b.1", where={}, text="0.50", sign=False, decimals=2
    )
    assert kinds("is 0.50", [e], tmp_path) == []
    e2 = make_entry(file="r.json", column=None, key="a.z", where={}, text="0.50")
    assert kinds("is 0.50", [e2], tmp_path) == ["bad_selector"]


def test_value_mismatch_reports_both(tmp_path):
    write_csv(tmp_path, [["x", "0.1", "0.0300"]])
    (m,) = ar.audit("gain +0.0267", [make_entry()], tmp_path)
    assert m.kind == "value" and "0.0267" in m.detail and "+0.0300" in m.detail


def test_missing_in_doc_and_substring_trap(tmp_path):
    write_csv(tmp_path, [["x", "0.1", "0.8709"]])
    e = make_entry(text="0.8709", sign=False)
    assert kinds("nothing", [e], tmp_path) == ["missing_in_doc"]
    assert kinds("value 0.87093 only", [e], tmp_path) == ["missing_in_doc"]
    assert kinds("value 10.8709 only", [e], tmp_path) == ["missing_in_doc"]
    assert kinds("value 0.8709.", [e], tmp_path) == []


def test_occurrences(tmp_path):
    write_csv(tmp_path, [["x", "0.1", "0.8709"]])
    e = make_entry(text="0.8709", sign=False, occurrences=2)
    assert kinds("0.8709 and 0.87093", [e], tmp_path) == ["missing_in_doc"]
    assert kinds("0.8709 and 0.8709", [e], tmp_path) == []


def test_missing_file(tmp_path):
    assert kinds("+0.0267", [make_entry()], tmp_path) == ["missing_file"]


def test_selector_zero_two_rows_and_agg(tmp_path):
    write_csv(tmp_path, [["x", "0.1", "0.02"], ["x", "0.2", "0.04"]])
    assert kinds("+0.0267", [make_entry()], tmp_path) == ["bad_selector"]
    assert kinds("+0.0267", [make_entry(where={"m": "nope"})], tmp_path) == ["bad_selector"]
    assert kinds("+0.0267", [make_entry(where={"nocol": "x"})], tmp_path) == ["bad_selector"]
    e = make_entry(agg="mean", text="+0.0300")
    assert kinds("+0.0300", [e], tmp_path) == []
    c = make_entry(agg="count", text="2", decimals=0, sign=False)
    assert kinds("n = 2", [c], tmp_path) == []


def test_numeric_where(tmp_path):
    write_csv(tmp_path, [["x", "0.10", "0.0267"]])
    e = make_entry(where={"m": "x", "alpha": 0.1})
    assert kinds("+0.0267", [e], tmp_path) == []


def test_scale_percent_and_unicode_minus(tmp_path):
    write_csv(tmp_path, [["x", "0.1", "0.855"], ["y", "0.1", "-0.2312"]])
    pct = make_entry(text="85.5%", scale=100, decimals=1, sign=False)
    assert kinds("85.5% of cases", [pct], tmp_path) == []
    neg = make_entry(text="−0.2312", where={"m": "y"}, sign=True)
    assert kinds("WT −0.2312", [neg], tmp_path) == []


def test_negative_zero(tmp_path):
    write_csv(tmp_path, [["x", "0.1", "-0.00001"]])
    assert ar.format_value(-0.00001, make_entry()) == "+0.0000"
    assert ar.format_value(-0.00001, make_entry(sign=False)) == "0.0000"
    assert ar.normalize_printed("−0.0000") == "0.0000"
    assert kinds("+0.0000", [make_entry(text="+0.0000")], tmp_path) == []
    assert kinds("-0.0000", [make_entry(text="-0.0000")], tmp_path) == []


def test_unregistered_skips_code_and_comments():
    doc = (
        "reg 0.1234 and new 0.5678\n"
        "```\ncode 0.9999\n```\n"
        "<!-- hidden 0.7777\n still hidden 0.8888 -->\n"
        "last −0.4321 ok 0.12\n"
    )
    found = ar.find_unregistered(doc, ["0.1234"])
    assert found == [(1, "0.5678"), (7, "−0.4321")]


def test_main_strict_and_invalid_registry(tmp_path):
    write_csv(tmp_path, [["x", "0.1", "0.0267"]])
    doc = tmp_path / "d.md"
    doc.write_text("gain +0.0267 and stray 0.5555\n")
    good = {
        "numbers": [
            dict(
                id="a",
                text="+0.0267",
                claim="C1",
                file="f.csv",
                decimals=4,
                column="v",
                where={"m": "x"},
                sign=True,
            )
        ]
    }
    reg = tmp_path / "r.yaml"
    reg.write_text(yaml.safe_dump(good))
    base = ["--doc", str(doc), "--registry", str(reg), "--root", str(tmp_path)]
    assert ar.main(base) == 0
    assert ar.main(base + ["--strict-unregistered"]) == 1

    dup = {"numbers": good["numbers"] * 2}
    reg.write_text(yaml.safe_dump(dup))
    assert ar.main(base) == 2

    missing = {"numbers": [{k: v for k, v in good["numbers"][0].items() if k != "claim"}]}
    reg.write_text(yaml.safe_dump(missing))
    assert ar.main(base) == 2

    nosel = {"numbers": [{k: v for k, v in good["numbers"][0].items() if k != "column"}]}
    reg.write_text(yaml.safe_dump(nosel))
    assert ar.main(base) == 2


def test_main_value_mismatch_exit_1(tmp_path, capsys):
    write_csv(tmp_path, [["x", "0.1", "0.05"]])
    doc = tmp_path / "d.md"
    doc.write_text("gain +0.0267\n")
    reg = tmp_path / "r.yaml"
    reg.write_text(
        yaml.safe_dump(
            {
                "numbers": [
                    dict(
                        id="a",
                        text="+0.0267",
                        claim="C1",
                        file="f.csv",
                        decimals=4,
                        column="v",
                        where={"m": "x"},
                        sign=True,
                    )
                ]
            }
        )
    )
    code = ar.main(["--doc", str(doc), "--registry", str(reg), "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1 and "value  a  C1" in out and "1 entries, 1 mismatches" in out


def test_registry_error_type(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("numbers: 3\n")
    with pytest.raises(ar.RegistryError):
        ar.load_registry(p)
