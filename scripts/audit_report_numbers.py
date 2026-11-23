"""Audit the numbers quoted in a Markdown document against tracked result files.

Every number printed in the semester report must trace to a claims-ledger row
and to a file under ``results/``. This script reads a Markdown document and a
YAML registry of the numbers it quotes, recomputes each number from its source
file, and prints every mismatch. It is a check, not a generator: it never edits
the document. Exit code 0 only when there are zero mismatches (2 = bad registry).

argparse is used instead of Hydra on purpose: this is a standalone checker with
two path arguments and no project config.

Registry format::

    numbers:
      - id: c1_et_gain              # unique
        text: "+0.0267"             # exact string as printed in the document
        claim: C1                   # claims_and_evidence.md row id (informational)
        file: results/compare_family/gatea/family.csv   # CSV or JSON, relative to --root
        where: {cohort: test, metric: dice_ET}   # CSV: equality filters
        column: improvement                      # CSV: column to read
        # JSON instead of where/column:  key: a.b.c  (integer parts index lists)
        agg: null                   # optional: mean | median | sum | count
        scale: 1.0                  # multiplied in before formatting (100 for %)
        decimals: 4
        sign: true                  # explicit '+' for non-negative values
        occurrences: 1              # minimum times `text` must appear (default 1)

Usage::

    python scripts/audit_report_numbers.py --doc report.md --registry reg.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

AGGS = ("mean", "median", "sum", "count")
REQUIRED_KEYS = ("id", "text", "claim", "file", "decimals")


class RegistryError(ValueError):
    """The registry file is malformed."""


class SelectorError(ValueError):
    """The entry's selector did not pick out a usable value."""


@dataclass
class Entry:
    """One registered number."""

    id: str
    text: str
    claim: str
    file: str
    decimals: int
    column: str | None = None
    key: str | None = None
    where: dict[str, Any] = field(default_factory=dict)
    agg: str | None = None
    scale: float = 1.0
    sign: bool = False
    occurrences: int = 1


@dataclass
class Mismatch:
    """One failed check."""

    kind: str
    id: str
    claim: str
    detail: str


def load_registry(path: Path) -> list[Entry]:
    """Load and validate a registry YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The entries, in file order.

    Raises:
        RegistryError: On any structural problem.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("numbers"), list):
        raise RegistryError("registry must be a mapping with a 'numbers' list")
    entries: list[Entry] = []
    seen: set[str] = set()
    for i, raw in enumerate(data["numbers"]):
        if not isinstance(raw, dict):
            raise RegistryError(f"entry #{i} is not a mapping")
        label = raw.get("id", f"#{i}")
        for k in REQUIRED_KEYS:
            if k not in raw:
                raise RegistryError(f"entry {label}: missing required key '{k}'")
        if raw.get("column") is None and raw.get("key") is None:
            raise RegistryError(f"entry {label}: needs 'column' (CSV) or 'key' (JSON)")
        if raw["id"] in seen:
            raise RegistryError(f"duplicate id '{raw['id']}'")
        seen.add(raw["id"])
        agg = raw.get("agg")
        if agg is not None and agg not in AGGS:
            raise RegistryError(f"entry {label}: agg must be one of {AGGS}")
        where = raw.get("where") or {}
        if not isinstance(where, dict):
            raise RegistryError(f"entry {label}: 'where' must be a mapping")
        entries.append(
            Entry(
                id=str(raw["id"]),
                text=str(raw["text"]),
                claim=str(raw["claim"]),
                file=str(raw["file"]),
                decimals=int(raw["decimals"]),
                column=raw.get("column"),
                key=raw.get("key"),
                where=where,
                agg=agg,
                scale=float(raw.get("scale", 1.0)),
                sign=bool(raw.get("sign", False)),
                occurrences=int(raw.get("occurrences", 1)),
            )
        )
    return entries


def _cell_matches(cell: str, wanted: Any) -> bool:
    """String equality on the cell text, or numeric equality if both parse."""
    if cell.strip() == str(wanted).strip():
        return True
    try:
        return float(cell) == float(wanted)
    except (TypeError, ValueError):
        return False


def _extract_json(entry: Entry, path: Path) -> float:
    obj: Any = json.loads(path.read_text(encoding="utf-8"))
    for part in str(entry.key).split("."):
        try:
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        except (KeyError, IndexError, ValueError, TypeError):
            raise SelectorError(f"JSON key path '{entry.key}' not found") from None
    try:
        return float(obj)
    except (TypeError, ValueError):
        raise SelectorError(f"value at '{entry.key}' is not numeric") from None


def _extract_csv(entry: Entry, path: Path) -> float:
    # Read everything as text so "0.10" and "0.1" can be compared numerically.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in [*entry.where, entry.column]:
        if col not in df.columns:
            raise SelectorError(f"column '{col}' not in {path.name}")
    mask = pd.Series(True, index=df.index)
    for col, wanted in entry.where.items():
        mask &= df[col].map(lambda c, w=wanted: _cell_matches(c, w))
    rows = df[mask]
    n = len(rows)
    if n == 0 or (n > 1 and entry.agg is None):
        raise SelectorError(f"where matched {n} rows")
    try:
        vals = pd.to_numeric(rows[entry.column])
    except ValueError:
        raise SelectorError(f"column '{entry.column}' is not numeric") from None
    if entry.agg is None:
        return float(vals.iloc[0])
    if entry.agg == "count":
        return float(n)
    return float(getattr(vals, entry.agg)())


def extract_value(entry: Entry, root: Path) -> float:
    """Recompute an entry's value from its source file.

    Args:
        entry: The registry entry.
        root: Directory that ``entry.file`` is relative to.

    Returns:
        The value before scaling.

    Raises:
        FileNotFoundError: Source file missing.
        SelectorError: Selector matched nothing usable.
    """
    path = Path(root) / entry.file
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if entry.key is not None:
        return _extract_json(entry, path)
    return _extract_csv(entry, path)


def format_value(v: float, entry: Entry) -> str:
    """Scale and format a value the way the document should print it.

    ``-0.0000`` is treated as zero, so it prints as ``0.0000`` (``+0.0000``
    when ``entry.sign``).
    """
    s = f"{v * entry.scale:.{entry.decimals}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    if entry.sign and not s.startswith("-"):
        s = "+" + s
    return s


_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


def normalize_printed(text: str) -> str:
    """Pull the numeric part out of printed text (``'−12.5%'`` -> ``'-12.5'``).

    Returns an empty string if there is no number. ``-0.00`` becomes ``0.00``.
    """
    m = _NUM.search(text.replace("−", "-"))
    if not m:
        return ""
    s = m.group(0)
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def _occurrence_spans(doc: str, text: str) -> list[tuple[int, int]]:
    """Non-overlapping matches of ``text`` that are not part of a longer number."""
    spans: list[tuple[int, int]] = []
    if not text:
        return spans
    start = 0
    while True:
        i = doc.find(text, start)
        if i < 0:
            return spans
        j = i + len(text)
        start = i + 1  # advance by one so a rejected match cannot hide a later one
        before = doc[i - 1] if i > 0 else ""
        before2 = doc[i - 2] if i > 1 else ""
        after = doc[j] if j < len(doc) else ""
        after2 = doc[j + 1] if j + 1 < len(doc) else ""
        # Substring trap: "0.8709" must not match inside "0.87093" or "10.8709".
        if text[0].isdigit() and (before.isdigit() or (before == "." and before2.isdigit())):
            continue
        if text[-1].isdigit() and (after.isdigit() or (after == "." and after2.isdigit())):
            continue
        if spans and i < spans[-1][1]:
            continue
        spans.append((i, j))
        start = j


def _blank_code_and_comments(doc: str) -> str:
    """Replace fenced code and HTML comments with spaces, keeping line numbers."""

    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    doc = re.sub(r"<!--.*?-->", blank, doc, flags=re.S)
    return re.sub(r"^(```|~~~).*?^\1[^\n]*$", blank, doc, flags=re.S | re.M)


_DECIMAL = re.compile(r"(?<![\d.])[+\-−]?\d+\.\d{3,}")


def find_unregistered(doc_text: str, registered_texts: list[str]) -> list[tuple[int, str]]:
    """Find decimals with >= 3 places that no registry ``text`` covers.

    Args:
        doc_text: Markdown source.
        registered_texts: The ``text`` of every registry entry.

    Returns:
        ``(line_number, number_as_printed)`` pairs, code blocks and HTML
        comments skipped. A number is covered if its digits (ignoring sign)
        equal those of a registered text.
    """

    def core(s: str) -> str:
        return normalize_printed(s).lstrip("+-")

    covered = {core(t) for t in registered_texts} - {""}
    clean = _blank_code_and_comments(doc_text)
    out: list[tuple[int, str]] = []
    for m in _DECIMAL.finditer(clean):
        if core(m.group(0)) in covered:
            continue
        out.append((clean.count("\n", 0, m.start()) + 1, m.group(0)))
    return out


def audit(doc_text: str, entries: list[Entry], root: Path) -> list[Mismatch]:
    """Run checks 1-4 for every entry.

    Args:
        doc_text: Markdown source.
        entries: Validated registry entries.
        root: Directory source files are relative to.

    Returns:
        All mismatches (empty means the document agrees with the files).
    """
    out: list[Mismatch] = []
    for e in entries:
        n = len(_occurrence_spans(doc_text, e.text))
        if n < e.occurrences:
            out.append(
                Mismatch(
                    "missing_in_doc",
                    e.id,
                    e.claim,
                    f"'{e.text}' found {n}x, need {e.occurrences}",
                )
            )
        try:
            v = extract_value(e, root)
        except FileNotFoundError as exc:
            out.append(Mismatch("missing_file", e.id, e.claim, str(exc)))
            continue
        except SelectorError as exc:
            out.append(Mismatch("bad_selector", e.id, e.claim, str(exc)))
            continue
        want = format_value(v, e)
        got = normalize_printed(e.text)
        # A printed zero may carry either sign; treat +0.0000 and 0.0000 alike
        # only when the recomputed value is also zero.
        if want != got and not (
            got and float(got) == 0.0 and float(want) == 0.0 and want.lstrip("+") == got.lstrip("+")
        ):
            out.append(Mismatch("value", e.id, e.claim, f"printed {got}, recomputed {want}"))
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the exit code (0 ok, 1 mismatches, 2 bad registry)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--strict-unregistered", action="store_true")
    args = parser.parse_args(argv)

    try:
        entries = load_registry(args.registry)
    except (RegistryError, OSError, yaml.YAMLError) as exc:
        print(f"invalid registry: {exc}", file=sys.stderr)
        return 2

    doc_text = args.doc.read_text(encoding="utf-8")
    mismatches = audit(doc_text, entries, args.root)
    unreg = find_unregistered(doc_text, [e.text for e in entries])
    if args.strict_unregistered:
        mismatches += [Mismatch("unregistered", "-", "-", f"line {ln}: {txt}") for ln, txt in unreg]
    for m in mismatches:
        print(f"{m.kind}  {m.id}  {m.claim}  {m.detail}")
    if unreg and not args.strict_unregistered:
        print("unregistered (first 20):")
        for ln, txt in unreg[:20]:
            print(f"  line {ln}: {txt}")
    print(f"{len(entries)} entries, {len(mismatches)} mismatches, {len(unreg)} unregistered")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
