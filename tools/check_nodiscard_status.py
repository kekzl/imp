#!/usr/bin/env python3
"""#2211 gate: every bool function declared in src/**/*.{h,cuh} is [[nodiscard]] or predicate-named.

Matcher = the header-bool rg in #2211. A non-predicate bool is a status a caller can drop;
it needs [[nodiscard]] (same line or the line above) or an allowlist entry with a reason.
The allowlist may only shrink (CEILING).

Usage:
    python3 tools/check_nodiscard_status.py             # check (CI)
    python3 tools/check_nodiscard_status.py --list      # print every violation
    python3 tools/check_nodiscard_status.py --selftest  # the matcher still matches
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ALLOWLIST = REPO / "tools" / "nodiscard_status_allowlist.txt"
SUFFIXES = (".h", ".cuh")
# Ratchet: lower with every fix, never raise.
CEILING = 0

DECL = re.compile(r"^\s*(\[\[nodiscard\]\]\s*)?(static\s+|virtual\s+|inline\s+)*bool\s+([\w:]+)\s*\(")
PREDICATE = re.compile(
    r"^(is_|has_|can_|should_|empty|valid|ok|enabled|use_|needs_|supports?_|operator)"
)
COMMENT = re.compile(r"^\s*(//|\*|/\*)")
NODISCARD_LINE = re.compile(r"^\s*\[\[nodiscard\]\]\s*$")


def violations(text: str) -> list[tuple[int, str]]:
    out = []
    prev = ""
    for lineno, line in enumerate(text.splitlines(), 1):
        m = DECL.match(line)
        if m and not COMMENT.match(line):
            name = m.group(3)
            if not m.group(1) and not NODISCARD_LINE.match(prev) and not PREDICATE.match(name):
                out.append((lineno, name))
        if line.strip():
            prev = line
    return out


def scan() -> dict[str, list[tuple[int, str]]]:
    hits = {}
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        found = violations(path.read_text(encoding="utf-8", errors="replace"))
        if found:
            hits[path.relative_to(REPO).as_posix()] = found
    return hits


def load_allowlist(text: str) -> tuple[set[str], list[str]]:
    allow, errors = set(), []
    for n, raw in enumerate(text.splitlines(), 1):
        body, _, reason = raw.partition("#")
        if not body.strip():
            continue
        if len(body.split()) != 1 or ":" not in body or not reason.strip():
            errors.append(f"allowlist:{n}: want '<path>:<function>  # reason': {raw!r}")
            continue
        allow.add(body.strip())
    return allow, errors


def check(hits: dict[str, list[tuple[int, str]]], allow: set[str]) -> list[str]:
    errors = []
    found = set()
    for rel, sites in hits.items():
        for ln, name in sites:
            key = f"{rel}:{name}"
            found.add(key)
            if key not in allow:
                errors.append(f"{rel}:{ln}: bool {name}(...) without [[nodiscard]] (status a caller can drop)")
    for key in sorted(allow - found):
        errors.append(f"{key}: stale allowlist entry, remove it")
    if len(allow) > CEILING:
        errors.append(f"allowlist size {len(allow)} > CEILING {CEILING}: the allowlist may only shrink")
    return errors


def selftest() -> int:
    cases = {
        "bool load(const std::string& p);": 1,
        "  static bool stage(int x) {": 1,
        "inline bool fits(int n) { return n > 0; }": 1,
        "[[nodiscard]] bool load(int);": 0,
        "[[nodiscard]] static inline bool stage(int);": 0,
        "[[nodiscard]]\nbool load(int);": 0,
        "bool is_ready() const;": 0,
        "bool has_moe() const;": 0,
        "bool operator==(const A&) const;": 0,
        "// bool load(int);": 0,
        " * bool load(int);": 0,
        "bool ok_ = false;": 0,
        "void f(bool load);": 0,
        "virtual bool run(int) = 0;": 1,
    }
    bad = 0
    for text, want in cases.items():
        got = len(violations(text))
        if got != want:
            print(f"selftest: {text!r}: want {want}, got {got}")
            bad += 1
    hit = {"src/a.h": [(3, "load")]}
    allow_cases = [
        (hit, "", 1),
        ({}, "", 0),
        (hit, "src/a.h:load\n", 2),  # no reason: bad entry + unallowed hit
        ({}, "src/a.h:load  # r\n", 1 + (1 if CEILING < 1 else 0)),  # stale
    ]
    for hits, text, want in allow_cases:
        allow, errs = load_allowlist(text)
        got = len(errs) + len(check(hits, allow))
        if got != want:
            print(f"selftest: allowlist {text!r}: want {want} errors, got {got}")
            bad += 1
    total = len(cases) + len(allow_cases)
    print(f"selftest: {total - bad}/{total} cases ok")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    hits = scan()
    if args.list:
        for rel, sites in hits.items():
            for ln, name in sites:
                print(f"{rel}:{ln}: {name}")
        return 0
    allow, errors = load_allowlist(ALLOWLIST.read_text(encoding="utf-8"))
    errors += check(hits, allow)
    for e in errors:
        print(e)
    total = sum(len(v) for v in hits.values())
    print(f"header bool status functions without [[nodiscard]]: {total} in {len(hits)} files "
          f"(allowlist {len(allow)}, ceiling {CEILING})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
