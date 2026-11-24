#!/usr/bin/env python3
"""AUDIT_ARCH_2026_11_24 X-5 gate: the Sanitizers path filter cannot miss a parser.

tools/untrusted_input_sources.txt holds the path prefixes whose change runs the Sanitizers
job on a PR. Until 2026-11-25 that list was a regex in ci.yml and missed 4 files with a
fuzz target or an "Untrusted input" header. This gate demands a prefix for:
  1. every src/ file in the CPU fuzz build (cmake/imp-fuzzers-cpu.cmake)
  2. every src/ or tools/imp-server/ file a fuzz/ file #includes, and its .cpp/.cu sibling
  3. every src/ or tools/ file with "Untrusted input" in its first 15 lines, and its sibling
and fails on a prefix that matches no tracked file.

Usage:
    python3 tools/check_untrusted_sources.py             # gate (CI)
    python3 tools/check_untrusted_sources.py --match     # paths on stdin; exit 0 = one matches, 1 = none
    python3 tools/check_untrusted_sources.py --selftest
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
LIST = "tools/untrusted_input_sources.txt"
FUZZ_CMAKE = "cmake/imp-fuzzers-cpu.cmake"
TAG = "Untrusted input"
TAG_LINES = 15
SOURCE_SUFFIXES = (".cpp", ".cu", ".h", ".cuh", ".hpp")
INCLUDE_ROOTS = ("src", "tools/imp-server")

CMAKE_SRC = re.compile(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/(src/[\w./-]+)")
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def parse_list(text: str) -> tuple[list[str], list[str]]:
    """(prefixes, exempt prefixes)."""
    prefixes, exempt = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if token.startswith("!"):
            exempt.append(token[1:])
        else:
            prefixes.append(token)
    return prefixes, exempt


def covered(path: str, prefixes: list[str]) -> bool:
    return any(path.startswith(p) for p in prefixes)


def cmake_sources(text: str) -> list[str]:
    return CMAKE_SRC.findall(text)


def includes(text: str) -> list[str]:
    return INCLUDE.findall(text)


def tagged(text: str) -> bool:
    return any(TAG in line for line in text.splitlines()[:TAG_LINES])


def with_siblings(path: str, tree: set[str]) -> list[str]:
    """The file plus the same-stem .cpp/.cu beside a header."""
    out = [path]
    p = pathlib.PurePosixPath(path)
    if p.suffix in (".h", ".cuh", ".hpp"):
        out += [s for s in (str(p.with_suffix(e)) for e in (".cpp", ".cu")) if s in tree]
    return out


def demanded(tree: set[str], read) -> dict[str, str]:
    """path -> why the list has to cover it."""
    want: dict[str, str] = {}
    for src in cmake_sources(read(FUZZ_CMAKE)):
        want.setdefault(src, f"compiled into the CPU fuzz build ({FUZZ_CMAKE})")
    for f in sorted(tree):
        if f.startswith("fuzz/") and f.endswith(SOURCE_SUFFIXES):
            for inc in includes(read(f)):
                for root in INCLUDE_ROOTS:
                    cand = f"{root}/{inc}"
                    if cand in tree:
                        for s in with_siblings(cand, tree):
                            want.setdefault(s, f"#include'd by {f}")
        elif f.startswith(("src/", "tools/")) and f.endswith(SOURCE_SUFFIXES) and tagged(read(f)):
            for s in with_siblings(f, tree):
                want.setdefault(s, f'"{TAG}" in the header of {f}')
    return want


def check(tree: set[str], read) -> list[str]:
    prefixes, exempt = parse_list(read(LIST))
    errors = []
    for path, why in sorted(demanded(tree, read).items()):
        if not covered(path, prefixes) and not covered(path, exempt):
            errors.append(f"{path}: {why}, but no prefix in {LIST} matches it")
    for p in prefixes + exempt:
        if not any(f.startswith(p) for f in tree):
            errors.append(f"{LIST}: prefix '{p}' matches no tracked file")
    return errors


def repo_tree() -> set[str]:
    # Walk, not `git ls-files`: the Actions container may refuse git on ownership.
    tree = set()
    for top in ("src", "tools", "fuzz", "tests", "cmake", ".github/workflows"):
        for path in (REPO / top).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                tree.add(path.relative_to(REPO).as_posix())
    tree.add("CMakeLists.txt")
    return tree


def repo_read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8", errors="replace")


def selftest() -> int:
    files = {
        LIST: "# c\nsrc/a/wav\nfuzz/\n!src/core/logging  reason\n",
        FUZZ_CMAKE: "add_library(x OBJECT\n    ${CMAKE_CURRENT_SOURCE_DIR}/src/a/wav.cpp\n"
        "    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/logging.cpp)\n",
        "src/a/wav.cpp": "int x;\n",
        "src/a/wav.h": "// Untrusted input: sizes are checked\n",
        "src/core/logging.cpp": "int y;\n",
        "fuzz/fuzz_wav.cpp": '#include "fuzz_common.h"\n#include "a/wav.h"\n',
    }

    def run(extra: dict[str, str], drop: tuple[str, ...] = ()) -> list[str]:
        fs = {k: v for k, v in {**files, **extra}.items() if k not in drop}
        return check(set(fs), fs.__getitem__)

    late_tag = "\n" * TAG_LINES + "// Untrusted input\n"
    cases = [
        ("clean tree", run({}), 0),
        ("fuzz cmake source not listed", run({LIST: "fuzz/\n!src/core/logging x\n"}), 2),
        ("tagged header and its .cpp not listed", run({"src/b/p.h": "// Untrusted input\n", "src/b/p.cpp": ""}), 2),
        ("tagged .cu not listed", run({"src/b/k.cu": "// x\n// Untrusted input here\n"}), 1),
        ("tag past the header lines is not a tag", run({"src/b/late.cpp": late_tag}), 0),
        ("fuzz include not listed", run({"fuzz/fuzz_q.cpp": '#include "b/q.h"\n', "src/b/q.h": "", "src/b/q.cpp": ""}), 2),
        ("server include resolved", run({"fuzz/fuzz_t.cpp": '#include "t.h"\n', "tools/imp-server/t.h": ""}), 1),
        ("exemption dropped", run({LIST: "src/a/wav\nfuzz/\n"}), 1),
        ("dead prefix", run({LIST: files[LIST] + "src/gone/parser\n"}), 1),
        ("dead exemption", run({FUZZ_CMAKE: "${CMAKE_CURRENT_SOURCE_DIR}/src/a/wav.cpp\n"},
                               drop=("src/core/logging.cpp",)), 1),
    ]
    bad = 0
    for name, errors, want in cases:
        if len(errors) != want:
            print(f"selftest: {name}: want {want} error(s), got {len(errors)}: {errors}")
            bad += 1
    prefixes, _ = parse_list(files[LIST])
    for path, want in (("src/a/wav.cpp", True), ("fuzz/x", True), ("xsrc/a/wav.cpp", False), ("src/a/wa", False)):
        if covered(path, prefixes) != want:
            print(f"selftest: covered({path!r}) != {want}")
            bad += 1
    n = len(cases) + 4
    print(f"selftest: {n - bad}/{n} cases ok")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--match", action="store_true", help="paths on stdin; exit 0 when one matches a prefix")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    prefixes, _ = parse_list(repo_read(LIST))
    if args.match:
        if not prefixes:
            print(f"{LIST} holds no prefix", file=sys.stderr)
            return 2
        hits = [p for p in (line.strip() for line in sys.stdin) if p and covered(p, prefixes)]
        for h in hits:
            print(h)
        return 0 if hits else 1
    tree = repo_tree()
    errors = check(tree, repo_read)
    for e in errors:
        print(e)
    n_files = sum(1 for f in tree if covered(f, prefixes))
    print(f"untrusted-input surface: {len(prefixes)} prefixes cover {n_files} files, "
          f"{len(demanded(tree, repo_read))} demanded, {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
