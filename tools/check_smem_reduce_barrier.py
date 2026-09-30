#!/usr/bin/env python3
"""Gate: a shared-memory reduction slot read back by every thread is behind a
__syncthreads() before the next write to that array.

Pattern: `x = f(s_red[0]);` then `s_red[tid] = y;` with no barrier between. A
slow warp then reads thread 0's new value instead of the reduced sum (WAR race,
results differ run to run). #1750 fixed it in gdn_scan_fused_kernel; #2254 found
it in 5 sibling GDN scan kernels.

Scope: src/**/*.cu, src/**/*.cuh, comments stripped. A read is `<lhs> = <expr
containing NAME[0]>` where NAME is __shared__-style (s_* or red*). The next
write to NAME[...] before the enclosing block closes must come after a
__syncthreads(). Loop-carried pairs (write at the top of the next iteration) are
not checked.

Usage:
    python3 tools/check_smem_reduce_barrier.py             # gate
    python3 tools/check_smem_reduce_barrier.py --selftest  # planted cases
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
SUFFIXES = (".cu", ".cuh")

NAME = r"(?:s_\w+|red\w*)"
READ0 = re.compile(r"^[^=;]*[^=!<>+\-*/]=(?!=)[^;]*\b(?P<name>" + NAME + r")\[0\][^;]*;")
BARRIER = re.compile(r"\b__syncthreads\s*\(|\b(?:cg::)?sync\s*\(\s*block\b|\.sync\s*\(\s*\)")


def write_re(name: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(name) + r"\s*\[[^\]]*\]\s*(?:[+\-*/]?=)(?!=)")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def findings(text: str) -> list[tuple[int, str]]:
    """-> [(1-based line of the unguarded write, array name)]."""
    lines = strip_comments(text).splitlines()
    out = []
    for i, line in enumerate(lines):
        m = READ0.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        # The read line must not itself be the write (`s_red[0] = v;`).
        if re.match(r"^\s*" + re.escape(name) + r"\s*\[", line):
            continue
        wr = write_re(name)
        depth = 0
        for j in range(i + 1, len(lines)):
            nxt = lines[j]
            if BARRIER.search(nxt):
                break
            if wr.search(nxt):
                out.append((j + 1, name))
                break
            depth += nxt.count("{") - nxt.count("}")
            if depth < 0:  # left the enclosing block (not loop-carried: the next iteration is unchecked)
                break
    return out


def scan() -> list[str]:
    bad = []
    for p in sorted(SRC.rglob("*")):
        if p.suffix not in SUFFIXES:
            continue
        for ln, name in findings(p.read_text(encoding="utf-8", errors="replace")):
            bad.append(f"{p.relative_to(REPO)}:{ln}: write to {name}[] after a {name}[0] broadcast read "
                       f"with no __syncthreads() between")
    return bad


SELFTEST = {
    "race": ("float a = rsqrtf(s_reduce[0]);\n\ns_reduce[d] = q;\n__syncthreads();\n", 1),
    "guarded": ("float a = rsqrtf(s_reduce[0]);\n__syncthreads();\ns_reduce[d] = q;\n", 0),
    "compound": ("const float m = red[0] * 2.f;\nred[tid] += v;\n", 1),
    "comment_barrier_is_not_one": ("x = s_red[0];\n// __syncthreads();\ns_red[t] = y;\n", 1),
    "self_write_not_a_read": ("s_red[0] = v;\ns_red[1] = w;\n", 0),
    "compare_not_write": ("x = s_red[0];\nif (s_red[1] == 0) {}\n", 0),
    "next_kernel_is_out_of_scope": ("  y = s_buf[0];\n}\n__global__ void k() {\n  s_buf[d] = v;\n", 0),
    "write_in_nested_block": ("x = s_red[0];\nif (d < 4) {\n  s_red[d] = y;\n}\n", 1),
}


def selftest() -> int:
    rc = 0
    for label, (src, want) in SELFTEST.items():
        got = len(findings(src))
        ok = got == want
        rc |= 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {got} finding(s), want {want}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    bad = scan()
    for b in bad:
        print(b)
    print(f"check_smem_reduce_barrier: {len(bad)} unguarded broadcast-read/write pair(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
