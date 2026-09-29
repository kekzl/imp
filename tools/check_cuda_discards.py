#!/usr/bin/env python3
"""#2211 gate: statements that call a CUDA runtime API and discard the cudaError_t.

Matcher = the awk in #2211 (statement starts with cuda<Upper>( after a finished
statement; cudaGetLastError and type names excluded). Per-file counts must equal
tools/cuda_discard_allowlist.txt; the allowlist total may not exceed CEILING.

Usage:
    python3 tools/check_cuda_discards.py             # check (CI)
    python3 tools/check_cuda_discards.py --list      # print every site
    python3 tools/check_cuda_discards.py --counts    # per-file counts, allowlist format
    python3 tools/check_cuda_discards.py --selftest  # the matcher still matches
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ALLOWLIST = REPO / "tools" / "cuda_discard_allowlist.txt"
SUFFIXES = (".cpp", ".cu", ".h", ".cuh")
# Ratchet: lower with every fix, never raise.
CEILING = 105
# Entries here need a "# reason" (only class c: an intentional discard).
ZERO_DIRS = (
    "src/runtime/", "src/memory/",
    "src/core/", "src/quant/", "src/model/", "src/vision/", "src/lora/", "src/api/",
    "src/compute/",
)

CALL = re.compile(r"^[ \t]*cuda[A-Z][A-Za-z0-9_]*[ \t]*\(")
EXCLUDED = re.compile(
    r"^[ \t]*cuda(Error_t|Stream_t|Event_t|Graph_t|GraphExec_t|DeviceProp"
    r"|GetErrorString|GetErrorName|GetLastError)"
)
PREV_END = re.compile(r"[;{}][ \t]*(//.*)?$")
PREV_LEAD = re.compile(r"^[ \t]*(//|#|else|do)")
PREV_CTRL = re.compile(r"^[ \t]*(if|for|while|else)")
BLANK = re.compile(r"^[ \t]*$")
COMMENT = re.compile(r"^[ \t]*//")


def sites(text: str) -> list[tuple[int, str]]:
    out = []
    prev = "{"
    for lineno, line in enumerate(text.splitlines(), 1):
        if CALL.match(line) and not EXCLUDED.match(line) and (
            PREV_END.search(prev)
            or PREV_LEAD.match(prev)
            or (re.search(r"\)[ \t]*$", prev) and PREV_CTRL.match(prev))
        ):
            out.append((lineno, line.strip().split("(", 1)[0].strip()))
        if not BLANK.match(line) and not COMMENT.match(line):
            prev = line
    return out


def scan() -> dict[str, list[tuple[int, str]]]:
    hits = {}
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        found = sites(path.read_text(encoding="utf-8", errors="replace"))
        if found:
            hits[path.relative_to(REPO).as_posix()] = found
    return hits


def load_allowlist(text: str) -> tuple[dict[str, tuple[int, str]], list[str]]:
    allow, errors = {}, []
    for n, raw in enumerate(text.splitlines(), 1):
        body, _, reason = raw.partition("#")
        if not body.strip():
            continue
        parts = body.split()
        if len(parts) != 2 or not parts[0].isdigit():
            errors.append(f"allowlist:{n}: want '<count> <path> [# reason]': {raw!r}")
            continue
        allow[parts[1]] = (int(parts[0]), reason.strip())
    return allow, errors


def check(hits: dict[str, list], allow: dict[str, tuple[int, str]]) -> list[str]:
    errors = []
    for rel in sorted(set(hits) | set(allow)):
        got = len(hits.get(rel, []))
        want, reason = allow.get(rel, (0, ""))
        if got > want:
            lines = ",".join(str(ln) for ln, _ in hits[rel])
            errors.append(f"{rel}: {got} discarded cudaError_t > allowed {want} (lines {lines})")
        elif got < want:
            errors.append(f"{rel}: {got} < allowed {want}: lower the allowlist entry to {got}")
        if want and rel.startswith(ZERO_DIRS) and not reason:
            errors.append(f"{rel}: allowlist entry in {', '.join(ZERO_DIRS)} needs '# reason'")
    total = sum(c for c, _ in allow.values())
    if total > CEILING:
        errors.append(f"allowlist total {total} > CEILING {CEILING}: the allowlist may only shrink")
    return errors


def selftest() -> int:
    cases = {
        "cudaFree(p);": 1,
        "  cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToHost, st);": 1,
        "x = 1;\ncudaStreamSynchronize(s);": 1,
        "if (p)\n    cudaFree(p);": 1,
        "else\n    cudaFree(p);": 1,
        "} else\n    cudaFree(p);": 0,
        "// note\ncudaFree(p);": 1,
        "cudaGetLastError();": 0,
        "cudaError_t e = cudaFree(p);": 0,
        "IMP_CUDA_CHECK_LOG(cudaFree(p));": 0,
        "(void)cudaFree(p);": 0,
        "return cudaFree(p);": 0,
        "foo(a,\n    cudaFree(p));": 0,
        "cudaStream_t s;": 0,
        "// cudaFree(p);": 0,
    }
    bad = 0
    for text, want in cases.items():
        got = len(sites(text))
        if got != want:
            print(f"selftest: {text!r}: want {want}, got {got}")
            bad += 1
    allow_cases = [
        ({"src/a.cu": [(1, "cudaFree")]}, "1 src/a.cu\n", 0),
        ({"src/a.cu": [(1, "cudaFree")] * 2}, "1 src/a.cu\n", 1),
        ({"src/a.cu": [(1, "cudaFree")]}, "2 src/a.cu\n", 1),
        ({}, "", 0),
        ({"src/memory/m.cpp": [(1, "cudaFree")]}, "1 src/memory/m.cpp\n", 1),
        ({"src/memory/m.cpp": [(1, "cudaFree")]}, "1 src/memory/m.cpp # sticky clear\n", 0),
        ({"src/b.cu": [(1, "cudaFree")] * (CEILING + 1)}, f"{CEILING + 1} src/b.cu\n", 1),
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
    ap.add_argument("--counts", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    hits = scan()
    if args.list:
        for rel, found in hits.items():
            for ln, api in found:
                print(f"{rel}:{ln}: {api}")
        return 0
    if args.counts:
        for rel, found in hits.items():
            print(f"{len(found)} {rel}")
        return 0
    allow, errors = load_allowlist(ALLOWLIST.read_text(encoding="utf-8"))
    errors += check(hits, allow)
    for e in errors:
        print(e)
    total = sum(len(v) for v in hits.values())
    print(f"discarded cudaError_t: {total} in {len(hits)} files (allowlist ceiling {CEILING})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
