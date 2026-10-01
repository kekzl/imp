#!/usr/bin/env python3
"""#2275 gate: synchronous device writes (legacy default stream) outside an allowlist.

Rule: a synchronous cudaMemcpy (H2D/D2D/default), cudaMemcpy2D, cudaMemcpyToSymbol or
cudaMemset runs on the legacy default stream, which does not order with the engine's
cudaStreamNonBlocking streams. A write that a stream kernel consumes goes
cudaMemcpyAsync/cudaMemsetAsync on the consumer stream, or is followed by an explicit sync.
D2H copies are reads and out of scope.

Allowlist: tools/sync_device_write_allowlist.txt, one site key per line:
    <path> <call>(<first arg> <class>  # reason
class: init (engine/model construction, before serving), debug (diagnostics-gated),
offline (quantizer, probe, tool), synced (explicit legacy-stream or device sync before
any consumer). Per-key counts must match the scan exactly.

Usage:
    python3 tools/check_sync_device_writes.py             # check (CI)
    python3 tools/check_sync_device_writes.py --list      # every site with line numbers
    python3 tools/check_sync_device_writes.py --selftest  # matcher cases
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ALLOWLIST = REPO / "tools" / "sync_device_write_allowlist.txt"
SUFFIXES = (".cpp", ".cu", ".h", ".cuh")
CLASSES = {"init", "debug", "offline", "synced"}
CALL = re.compile(r"\b(cudaMemcpy|cudaMemcpy2D|cudaMemcpyToSymbol|cudaMemset)\s*\(")


def strip_comments(text: str) -> str:
    """Blank // and /* */ comments and string literals, keeping offsets and newlines."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            j = n if j < 0 else j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j += 1
            for k in range(i + 1, min(j - 1, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        else:
            i += 1
            continue
        for k in range(i, j):
            if out[k] != "\n":
                out[k] = " "
        i = j
    return "".join(out)


def call_args(text: str, open_paren: int) -> str:
    depth, i = 0, open_paren
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1:i]
        i += 1
    return text[open_paren + 1:]


def first_arg(args: str) -> str:
    depth = 0
    for i, c in enumerate(args):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            return " ".join(args[:i].split())
    return " ".join(args.split())


def sites(text: str) -> list[tuple[int, str]]:
    """(line, key) per synchronous device write; key = call(first-arg."""
    code = strip_comments(text)
    out = []
    for m in CALL.finditer(code):
        args = call_args(code, m.end() - 1)
        if m.group(1) in ("cudaMemcpy", "cudaMemcpy2D") and "cudaMemcpyDeviceToHost" in args:
            continue
        line = code.count("\n", 0, m.start()) + 1
        out.append((line, f"{m.group(1)}({first_arg(args)}".replace(" ", "")))
    return out


def scan(root: pathlib.Path = SRC) -> Counter:
    found: Counter = Counter()
    for path in sorted(root.rglob("*")):
        if path.suffix in SUFFIXES and path.is_file():
            rel = path.relative_to(REPO).as_posix() if root == SRC else path.relative_to(root).as_posix()
            for _, key in sites(path.read_text(errors="replace")):
                found[(rel, key)] += 1
    return found


def load_allowlist(path: pathlib.Path = ALLOWLIST) -> tuple[Counter, list[str]]:
    allowed: Counter = Counter()
    problems = []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        body, _, reason = raw.partition("#")
        parts = body.split()
        if not parts:
            continue
        if len(parts) != 3 or parts[2] not in CLASSES or not reason.strip():
            problems.append(f"{path.name}:{n}: want '<path> <key> <{'|'.join(sorted(CLASSES))}>  # reason'")
            continue
        allowed[(parts[0], parts[1])] += 1
    return allowed, problems


def evaluate(found: Counter, allowed: Counter) -> list[str]:
    problems = []
    for k in sorted(found.keys() | allowed.keys()):
        f, a = found.get(k, 0), allowed.get(k, 0)
        if f > a:
            problems.append(f"{k[0]}: {k[1]} x{f - a} not allowlisted: synchronous legacy-stream write; "
                            "use cudaMemcpyAsync/cudaMemsetAsync on the consumer stream")
        elif a > f:
            problems.append(f"{k[0]}: {k[1]} allowlisted x{a} but found x{f}: drop the stale entry")
    return problems


def selftest() -> int:
    cases = [
        ("cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);", ["cudaMemcpy(d"]),
        ("cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);", []),
        ("cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s);", []),
        ("cudaMemsetAsync(p, 0, n, s);", []),
        ("IMP_CUDA_CHECK_LOG(cudaMemset(pool_ + off, 0,\n    total));", ["cudaMemset(pool_+off"]),
        ("// cudaMemset(p, 0, n);", []),
        ('log("cudaMemset(p, 0, n)");', []),
        ("cudaMemcpyToSymbol(sym, &v, 4);", ["cudaMemcpyToSymbol(sym"]),
        ("cudaMemcpyFromSymbol(&v, sym, 4);", []),
    ]
    bad = 0
    for text, want in cases:
        got = [k for _, k in sites(text)]
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {text.splitlines()[0]!r}: want {want}, got {got}")
    print(f"selftest: {len(cases) - bad}/{len(cases)} cases")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.list:
        for path in sorted(SRC.rglob("*")):
            if path.suffix in SUFFIXES and path.is_file():
                for line, key in sites(path.read_text(errors="replace")):
                    print(f"{path.relative_to(REPO).as_posix()}:{line} {key}")
        return 0
    allowed, problems = load_allowlist()
    problems += evaluate(scan(), allowed)
    for p in problems:
        print(f"check_sync_device_writes: {p}")
    if problems:
        print(f"check_sync_device_writes: FAIL, {len(problems)} problem(s); rule in the script docstring (#2275)")
        return 1
    print(f"check_sync_device_writes: OK, {sum(allowed.values())} allowlisted synchronous device writes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
