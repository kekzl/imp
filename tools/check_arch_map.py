#!/usr/bin/env python3
"""Architecture map gate: every directory under src/ has a row, and a layer rank.

WHY THIS EXISTS
---------------
docs/internals/ARCHITECTURE.md is the map a model port starts from. On 2026-11-24 it
named neither src/audio (12 files) nor src/lora (4) nor the Inkling forward (6345 lines
in 28 files): AUDIT_ARCH_2026_11_24 X-9. Nothing failed when a directory arrived.

A date rule does not catch this. The `verified:` rule of scripts/docs_lint.py compares
the header with the document's OWN last commit; a new src/ directory touches no doc.
(Applied to docs/internals/ as it stands for the CLAUDE.md tree, it would also have
failed 6 of the 10 files on 2026-11-25 for edits that never bumped the header.)

WHAT IT CHECKS
--------------
1. Each top-level directory under src/ has one row in the "Source directories" table of
   docs/internals/ARCHITECTURE.md: a table line whose first cell is `src/<dir>/` in
   backticks. A directory without a row fails; a row whose directory is gone fails; a
   row with an empty second cell fails.
2. Each of those directories has a rank in tools/check_layering.py RANK. A directory
   without one is skipped by the layering gate, so its includes are unchecked.

Usage:
    python3 tools/check_arch_map.py             # check (CI)
    python3 tools/check_arch_map.py --selftest  # planted cases
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
DOC = REPO / "docs" / "internals" / "ARCHITECTURE.md"
LAYERING = REPO / "tools" / "check_layering.py"

ROW = re.compile(r"^\|\s*`src/([A-Za-z0-9_]+)/`\s*\|([^|]*)\|")
RANK_BLOCK = re.compile(r"^RANK\s*=\s*\{(.*?)\}", re.S | re.M)


def rows(doc: str) -> dict[str, str]:
    """Directory -> second cell, for each `src/<dir>/` row of the document."""
    out: dict[str, str] = {}
    for line in doc.splitlines():
        m = ROW.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def ranked(layering: str) -> set[str]:
    m = RANK_BLOCK.search(layering)
    return set(re.findall(r'"(\w+)"\s*:', m.group(1))) if m else set()


def verdict(dirs: list[str], doc: str, layering: str) -> list[str]:
    fails = []
    have = rows(doc)
    rank = ranked(layering)
    for d in sorted(dirs):
        if d not in have:
            fails.append(f"FAIL src/{d}/: no row in the Source directories table of "
                         f"docs/internals/ARCHITECTURE.md")
        elif not have[d]:
            fails.append(f"FAIL src/{d}/: its row in docs/internals/ARCHITECTURE.md says nothing")
        if d not in rank:
            fails.append(f"FAIL src/{d}/: no rank in tools/check_layering.py RANK (its includes are unchecked)")
    for d in sorted(set(have) - set(dirs)):
        fails.append(f"FAIL docs/internals/ARCHITECTURE.md: row for src/{d}/, which does not exist")
    return fails


def selftest() -> int:
    doc = "| Directory | Holds |\n|---|---|\n| `src/core/` | leaf headers |\n| `src/exec/` | forward |\n"
    lay = 'RANK = {\n    "core": 0, "exec": 6,\n}\n'
    cases = [
        ("every directory has a row and a rank", ["core", "exec"], doc, lay, 0),
        ("a new directory without a row", ["core", "exec", "audio"], doc, 'RANK = {"core": 0, "exec": 6, "audio": 7}', 1),
        ("a new directory without a row and without a rank", ["core", "exec", "audio"], doc, lay, 2),
        ("a row for a directory that is gone", ["core"], doc, lay, 1),
        ("a row that says nothing", ["core", "exec"], doc.replace("forward", " "), lay, 1),
        ("a directory named only in prose does not count",
         ["core", "exec", "lora"], doc + "\nAdapters live in `src/lora/`.\n",
         'RANK = {"core": 0, "exec": 6, "lora": 5}', 1),
        ("a file path in another table does not count",
         ["core", "exec", "lora"], doc + "| LoRA | `src/lora/lora_adapter.cpp` | deltas |\n",
         'RANK = {"core": 0, "exec": 6, "lora": 5}', 1),
    ]
    failures = 0
    for label, dirs, d, layering, want in cases:
        got = len(verdict(dirs, d, layering))
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got} failure(s)" + ("" if ok else f" (want {want})"))
    print(f"selftest: {len(cases) - failures}/{len(cases)} cases")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true", help="run the planted cases")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    dirs = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    fails = verdict(dirs, DOC.read_text(encoding="utf-8"), LAYERING.read_text(encoding="utf-8"))
    for f in fails:
        print("  " + f)
    if fails:
        print(f"FAIL: {len(fails)} failure(s); add the row (one line: what the directory holds) or the rank.")
        return 1
    print(f"OK: {len(dirs)} directories under src/, each with a row in docs/internals/ARCHITECTURE.md "
          f"and a layer rank.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
