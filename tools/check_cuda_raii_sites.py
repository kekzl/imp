#!/usr/bin/env python3
"""#2213 gate: CUDA stream/event create/destroy only inside src/core/cuda_raii.h.

Everything else owns streams and events through CudaStream / CudaEvent.
Allowlist is empty: any raw site in src/ outside cuda_raii.h fails.

Usage:
    python3 tools/check_cuda_raii_sites.py             # check (CI)
    python3 tools/check_cuda_raii_sites.py --selftest  # the matcher still matches
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
OWNER = "src/core/cuda_raii.h"
SUFFIXES = (".cpp", ".cu", ".h", ".cuh", ".hpp")

# cudaExecutionCtxStreamCreate: green-context streams (CudaStream::create_on_ctx).
PATTERN = re.compile(
    r"\b(cudaStreamCreate\w*|cudaEventCreate\w*|cudaExecutionCtxStreamCreate"
    r"|cudaStreamDestroy|cudaEventDestroy)\s*\("
)
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT = re.compile(r"//[^\n]*")


def strip_comments(text: str) -> str:
    # Keep newlines so reported line numbers stay right.
    text = BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return LINE_COMMENT.sub("", text)


def sites(text: str) -> list[tuple[int, str]]:
    out = []
    for lineno, line in enumerate(strip_comments(text).splitlines(), 1):
        for m in PATTERN.finditer(line):
            out.append((lineno, m.group(1)))
    return out


def scan() -> list[str]:
    hits = []
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel == OWNER:
            continue
        for lineno, api in sites(path.read_text(encoding="utf-8", errors="replace")):
            hits.append(f"{rel}:{lineno}: {api}")
    return hits


def selftest() -> int:
    cases = {
        "cudaStreamCreate(&s);": 1,
        "cudaStreamCreateWithPriority(&s, f, p);": 1,
        "cudaEventCreateWithFlags (&e, cudaEventDisableTiming);": 1,
        "cudaExecutionCtxStreamCreate(&s, ctx, f, p);": 1,
        "IMP_CUDA_CHECK_LOG(cudaEventDestroy(e)); cudaStreamDestroy(s);": 2,
        "// cudaEventDestroy(e) in a comment": 0,
        "/* cudaStreamDestroy(s)\n cudaEventCreate(&e) */": 0,
        "my_cudaEventCreate(&e);": 0,
        "ev.create(cudaEventDisableTiming);": 0,
    }
    bad = 0
    for text, want in cases.items():
        got = len(sites(text))
        if got != want:
            print(f"selftest: {text!r}: want {want}, got {got}")
            bad += 1
    print(f"selftest: {len(cases) - bad}/{len(cases)} cases ok")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    hits = scan()
    for h in hits:
        print(h)
    print(f"raw CUDA stream/event sites outside {OWNER}: {len(hits)} (allowed: 0)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
