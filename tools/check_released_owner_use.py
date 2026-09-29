#!/usr/bin/env python3
"""#2294 gate: no dereference of an owner after its .release() in the same block.

`p.release()` nulls p; a later `p->x`, `*p`, `p.get()` or `p[i]` in the same block
reads null (#2278: enable_mtp_spec_decode logged ws->n_kv_slots after ws.release()).
Re-seating (`p = ...`, `p.reset(...)`) ends the scan for that owner.

Usage:
    python3 tools/check_released_owner_use.py             # check (CI)
    python3 tools/check_released_owner_use.py --selftest  # the matcher still matches
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
SUFFIXES = (".cpp", ".cu", ".h", ".cuh", ".hpp")

RELEASE = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*release\s*\(\s*\)")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT = re.compile(r"//[^\n]*")
STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')


def strip(text: str) -> str:
    # Keep newlines so reported line numbers stay right.
    text = BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    text = LINE_COMMENT.sub("", text)
    return STRING.sub('""', text)


def block_tail(text: str, pos: int) -> str:
    """Text from pos to the close of the block that contains pos."""
    depth = 0
    for i in range(pos, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return text[pos:i]
    return text[pos:]


def uses(text: str) -> list[tuple[int, str]]:
    out = []
    for m in RELEASE.finditer(text):
        var = re.escape(m.group(1))
        tail = block_tail(text, m.end())
        reseat = re.compile(rf"\b{var}\s*(=[^=]|\.\s*reset\s*\()")
        deref = re.compile(
            rf"\b{var}\s*->|(?<![\w)\]])\s*\*\s*{var}\b|\b{var}\s*\.\s*get\s*\(|\b{var}\s*\["
        )
        stop = reseat.search(tail)
        limit = stop.start() if stop else len(tail)
        d = deref.search(tail, 0, limit)
        if d:
            lineno = text.count("\n", 0, m.end() + d.start()) + 1
            out.append((lineno, m.group(1)))
    return out


def check() -> int:
    bad = []
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        text = strip(path.read_text(encoding="utf-8", errors="replace"))
        for lineno, var in uses(text):
            bad.append(f"{path.relative_to(REPO)}:{lineno}: '{var}' dereferenced after {var}.release()")
    for line in bad:
        print(line)
    if bad:
        print(f"{len(bad)} use(s) of a released owner; read through the new owner instead")
        return 1
    return 0


CASES = [
    # (source, expected hits)
    ("void f(){ auto ws = mk(); o.reset(ws.release()); log(ws->n); }", 1),
    ("void f(){ auto ws = mk(); o.reset(ws.release()); use(*ws); }", 1),
    ("void f(){ auto ws = mk(); o.reset(ws.release()); g(ws.get()); }", 1),
    ("void f(){ auto ws = mk(); o.reset(ws.release()); g(o->n); }", 0),
    ("void f(){ auto ws = mk(); o.reset(ws.release()); ws = mk(); g(ws->n); }", 0),
    ("void f(){ if (a) { o.reset(ws.release()); } g(ws->n); }", 0),
    ("int f(){ return ref ? ref.release() : -1; }", 0),
    ("void f(){ o.reset(ws.release()); int k = a * ws_count; }", 0),
    ("void f(){ o.reset(ws.release()); log(\"ws->n\"); }", 0),
]


def selftest() -> int:
    fails = 0
    for src, want in CASES:
        got = len(uses(strip(src)))
        if got != want:
            print(f"selftest FAIL: want {want} got {got}: {src}")
            fails += 1
    print(f"selftest: {len(CASES) - fails}/{len(CASES)} cases")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    return selftest() if args.selftest else check()


if __name__ == "__main__":
    sys.exit(main())
