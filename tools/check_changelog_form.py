#!/usr/bin/env python3
"""Every entry under `## [Unreleased]` in CHANGELOG.md is at most three lines.

WHY THIS EXISTS
---------------
Root CLAUDE.md: "The CHANGELOG is a changelog, not a journal. One to three lines
per entry". On 2026-09-05, 19 of the 26 entries in the last three releases broke
that, the longest at 13 lines, and nothing checked it (AUDIT_arch_2026 J-8).
Released sections are records and are left alone; the gate reads the block a
PR appends to, so the rule is enforced where the entry is written.

WHAT IT CHECKS
--------------
Inside the `## [Unreleased]` section (up to the next `## ` heading) an entry
starts with `- ` at column 0 and continues on every following non-empty line
that starts with whitespace; a `### ` heading or a blank line ends it. Entries
longer than MAX_LINES fail.

Usage:
    python3 tools/check_changelog_form.py             # check (CI)
    python3 tools/check_changelog_form.py --list      # every entry with its length
    python3 tools/check_changelog_form.py --selftest  # planted cases
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
MAX_LINES = 3


def unreleased_entries(text: str) -> list[tuple[int, int, str]]:
    """(1-based line number, line count, first line) per entry under [Unreleased]."""
    out: list[tuple[int, int, str]] = []
    inside = False
    start = None
    count = 0
    first = ""

    def flush() -> None:
        nonlocal start, count
        if start is not None:
            out.append((start, count, first))
        start, count = None, 0

    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            flush()
            inside = line.startswith("## [Unreleased]")
            continue
        if not inside:
            continue
        if line.startswith("- "):
            flush()
            start, count, first = i, 1, line
        elif start is not None and line.strip() and line[0].isspace():
            count += 1
        else:
            flush()
    flush()
    return out


def check(list_all: bool) -> int:
    entries = unreleased_entries(CHANGELOG.read_text(encoding="utf-8"))
    long = [e for e in entries if e[1] > MAX_LINES]
    if list_all:
        for ln, n, first in entries:
            print(f"CHANGELOG.md:{ln}: {n} line(s): {first[:70]}")
    for ln, n, first in long:
        print(f"FAIL: CHANGELOG.md:{ln}: entry is {n} lines, max {MAX_LINES}: {first[:70]}")
    if long:
        print(f"check_changelog_form: {len(long)} of {len(entries)} [Unreleased] entries over {MAX_LINES} lines")
        return 1
    print(f"PASS: {len(entries)} [Unreleased] entries, none over {MAX_LINES} lines")
    return 0


def selftest() -> int:
    text = """# Changelog

## [Unreleased]

### Added

- one line entry (#1)
- three line entry
  continues here
  and ends here (#2)
- four line entry
  two
  three
  four (#3)
- nested list entry
  - sub one
  - sub two
  - sub three
  - sub four (#4)

### Changed

- after a blank line (#5)

## [0.1.0] - 2026-01-01

- released entry that is
  very
  very
  very long (#6)
"""
    entries = unreleased_entries(text)
    lengths = {first.split("(#")[1].rstrip(")"): n for _, n, first in entries if "(#" in first}
    by_no = {}
    for ln, n, first in entries:
        by_no[ln] = n
    cases = [
        ("five entries under [Unreleased]", len(entries) == 5),
        ("a one-line entry counts 1", lengths.get("1") == 1),
        ("continuation lines are counted", entries[1][1] == 3),
        ("a four-line entry is over the limit", entries[2][1] == 4),
        ("nested bullets count as continuation", entries[3][1] == 5),
        ("a blank line and a ### heading end an entry", entries[4][1] == 1),
        ("released sections are not read", all("(#6)" not in first for _, _, first in entries)),
    ]
    bad = [name for name, ok in cases if not ok]
    for name, ok in cases:
        print(f"  {'ok ' if ok else 'BAD'} {name}")
    if bad:
        print(f"selftest: {len(bad)} of {len(cases)} planted cases failed")
        return 1
    print(f"selftest: {len(cases)}/{len(cases)} planted cases pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return check(a.list)


if __name__ == "__main__":
    sys.exit(main())
