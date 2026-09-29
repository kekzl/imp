#!/usr/bin/env python3
"""clang-tidy findings gate for imp (#2210).

Error checks: the `WarningsAsErrors` globs of .clang-tidy (bugprone-*, clang-analyzer-*).
A finding of an error check in a file fails unless tools/tidy_baseline.toml pins that
(file, check) at >= the count found. Other checks stay advisory.

Input: the per-TU logs of scripts/tidy_lane.sh. Findings are deduplicated on
(file, line, col, check): a header finding seen by N TUs counts once.

Exit codes: 0 pass, 1 violation, 2 malformed input.

Usage:
  python3 tools/check_tidy_findings.py --logs DIR [--list]
  python3 tools/check_tidy_findings.py --update --logs DIR   # rewrite [baseline] (full lane only)
  python3 tools/check_tidy_findings.py --selftest
"""
import argparse
import fnmatch
import glob
import json
import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.stderr.write("check_tidy_findings.py needs Python 3.11+ (tomllib)\n")
    sys.exit(2)

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS)
DEFAULT_CONFIG = os.path.join(TOOLS, "tidy_baseline.toml")
DEFAULT_TIDY = os.path.join(REPO_ROOT, ".clang-tidy")
FINDING = re.compile(r"^(?P<path>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+): (?:warning|error): .* \[(?P<checks>[A-Za-z0-9_.,-]+)\]$")
REPO_DIRS = ("src/", "include/", "tools/", "tests/")


def error_globs(tidy_path):
    """The WarningsAsErrors globs of .clang-tidy, as a list."""
    text = open(tidy_path, encoding="utf-8").read()
    m = re.search(r"^WarningsAsErrors:\s*'([^']*)'", text, re.M)
    if not m:
        raise ValueError(f"{tidy_path}: no single-quoted WarningsAsErrors line")
    return [g.strip() for g in m.group(1).split(",") if g.strip()]


def is_error(check, globs):
    hit = False
    for g in globs:
        if g.startswith("-"):
            if fnmatch.fnmatchcase(check, g[1:]):
                hit = False
        elif fnmatch.fnmatchcase(check, g):
            hit = True
    return hit


def rel(path, root):
    """Path relative to `root` if it is a repo source file, else None (system, build/_deps)."""
    p = os.path.normpath(path)
    if os.path.isabs(p):
        if not p.startswith(root.rstrip("/") + "/"):
            return None
        p = p[len(root.rstrip("/")) + 1:]
    return p if p.startswith(REPO_DIRS) else None


def parse(lines, globs, root=REPO_ROOT):
    """Set of (file, line, col, check) for error-check findings in repo files."""
    out = set()
    for ln in lines:
        m = FINDING.match(ln.rstrip("\n"))
        if not m:
            continue
        f = rel(m.group("path"), root)
        if f is None:
            continue
        for check in m.group("checks").split(","):
            if check != "clang-diagnostic-error" and is_error(check, globs):
                out.add((f, int(m.group("line")), int(m.group("col")), check))
    return out


def counts(findings):
    out = {}
    for f, _, _, check in findings:
        k = f"{f}::{check}"
        out[k] = out.get(k, 0) + 1
    return out


def evaluate(found, baseline):
    """(over, notes): over = [(key, pin, n)] above pin; notes = [(key, pin, n)] below pin."""
    over = sorted((k, baseline.get(k, 0), n) for k, n in found.items() if n > baseline.get(k, 0))
    notes = sorted((k, p, found.get(k, 0)) for k, p in baseline.items() if found.get(k, 0) < p)
    return over, notes


def write_baseline(path, found):
    text = open(path, encoding="utf-8").read()
    head = text[:text.index("[baseline]\n") + len("[baseline]\n")]
    body = "".join(f"{json.dumps(k)} = {n}\n" for k, n in sorted(found.items()))
    open(path, "w", encoding="utf-8").write(head + body)


def selftest():
    """Every rule the gate relies on, planted (#1858)."""
    globs = ["bugprone-*", "clang-analyzer-*"]
    w = "/work/src/a.cu:10:5: warning: result of multiplication [bugprone-implicit-widening-of-multiplication-result]"
    cases = [
        ("bugprone finding counts", [w], {"src/a.cu::bugprone-implicit-widening-of-multiplication-result": 1}),
        ("same finding from two TU logs counts once", [w, w],
         {"src/a.cu::bugprone-implicit-widening-of-multiplication-result": 1}),
        ("readability finding is advisory", ["/work/src/a.cu:3:1: warning: x [readability-foo]"], {}),
        ("analyzer finding counts", ["/work/src/b.cpp:7:2: warning: null [clang-analyzer-core.NullDereference]"],
         {"src/b.cpp::clang-analyzer-core.NullDereference": 1}),
        ("alias list: the error check in it counts",
         ["src/c.h:1:1: warning: x [cert-err33-c,bugprone-unused-return-value]"],
         {"src/c.h::bugprone-unused-return-value": 1}),
        ("CUTLASS header under _deps is not ours",
         ["/work/build/_deps/cutlass-src/include/cute/x.hpp:1:1: warning: x [bugprone-foo]"], {}),
        ("system header is not ours", ["/usr/local/cuda/include/cuda_fp16.hpp:9:9: warning: x [bugprone-foo]"], {}),
        ("build/ under the repo root is not ours", ["/work/build/generated/w.h:1:1: warning: x [bugprone-foo]"], {}),
        ("path outside the root is not ours", ["/other/src/a.cu:1:1: warning: x [bugprone-foo]"], {}),
        ("note lines are not findings", ["/work/src/a.cu:10:5: note: expanded from macro [bugprone-foo]"], {}),
    ]
    failures = 0
    for name, lines, want in cases:
        got = counts(parse(lines, globs, "/work"))
        ok = got == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: expected {want}, got {got}")
    neg = ["bugprone-*", "-bugprone-foo"]
    ok = not is_error("bugprone-foo", neg) and is_error("bugprone-bar", neg)
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  '-glob' in WarningsAsErrors excludes a check")
    gate_cases = [
        ("unpinned finding fails", {"a::c": 1}, {}, 1),
        ("finding at its pin passes", {"a::c": 3}, {"a::c": 3}, 0),
        ("pinned (file, check) +1 fails", {"a::c": 4}, {"a::c": 3}, 1),
        ("fewer than the pin passes", {"a::c": 2}, {"a::c": 3}, 0),
        ("pin in another file does not cover this one", {"b::c": 1}, {"a::c": 3}, 1),
    ]
    for name, found, base, want in gate_cases:
        over, _ = evaluate(found, base)
        ok = len(over) == want
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: expected {want}, got {len(over)}")
    total = len(cases) + 1 + len(gate_cases)
    print(f"selftest: {total - failures}/{total} cases")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description="imp clang-tidy findings gate")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--clang-tidy", default=DEFAULT_TIDY)
    ap.add_argument("--logs", help="directory of per-TU clang-tidy logs (*.log)")
    ap.add_argument("--list", action="store_true", help="print every error-check finding")
    ap.add_argument("--update", action="store_true", help="rewrite [baseline] from the logs")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.logs:
        ap.error("--logs is required")
    try:
        globs = error_globs(args.clang_tidy)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2
    if not globs:
        print("ERROR: WarningsAsErrors is empty: no check would be gated")
        return 2
    logs = sorted(glob.glob(os.path.join(args.logs, "*.log")))
    if not logs:
        print(f"ERROR: no *.log in {args.logs}: a gate over zero logs passes nothing (#1626)")
        return 2
    lines = []
    for p in logs:
        with open(p, encoding="utf-8", errors="replace") as f:
            lines.extend(f)
    findings = parse(lines, globs)
    found = counts(findings)
    if args.update:
        write_baseline(args.config, found)
        print(f"baseline rewritten: {len(found)} (file, check) entries, {len(findings)} findings")
        return 0
    with open(args.config, "rb") as f:
        baseline = tomllib.load(f).get("baseline", {})
    bad = [k for k, v in baseline.items() if not isinstance(v, int) or v < 1 or "::" not in k]
    if bad:
        print("ERROR: [baseline] keys are `path::check`, values integers >= 1. Offenders:")
        for k in bad:
            print(f"  {k}")
        return 2
    if args.list:
        for f, line, col, check in sorted(findings):
            print(f"  {f}:{line}:{col}  {check}")
    over, notes = evaluate(found, baseline)
    # A pin is only "shrunk" when its file was itself a linted TU in this run.
    linted = {os.path.basename(p)[:-4] for p in logs}
    notes = [n for n in notes if n[0].split("::")[0].replace("/", "_") in linted]
    print(f"error checks {','.join(globs)} | {len(logs)} TU log(s) | {len(findings)} finding(s) "
          f"| baseline {sum(baseline.values())} in {len(baseline)} pin(s) | over {len(over)}")
    for k, p, n in notes:
        print(f"NOTE  {k}: {n} < pin {p}; lower the pin")
    for k, p, n in over:
        print(f"FAIL  {k}: {n} finding(s), pin {p}")
    if over:
        print("\nFAIL: fix the finding, or NOLINT(check) with a reason on the line.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
