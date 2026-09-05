#!/usr/bin/env python3
"""Every ImpError entry point of the C API runs its body under a try/catch.

WHY THIS EXISTS
---------------
include/imp/imp.h is a C ABI: an exception escaping an `ImpError imp_*()` is
undefined behaviour at the boundary, and std::bad_alloc / std::system_error are
reachable from the container and CUDA-handle work every entry point does. The
convention was 20 hand-copied try/catch blocks in src/api/*.cpp; on 2026-09-05
four entry points had none (AUDIT_arch_2026 G-10). `imp::api_guard()`
(src/api/imp_internal.h) is the shared form; an inline `try {` still counts.

WHAT IT CHECKS
--------------
For each function definition `ImpError imp_<name>(...) {` in src/api/*.cpp the
brace-matched body must contain `try {` or `api_guard(`. There is no allowlist:
a body that "only returns error codes" still allocates through the handles it
touches.

Usage:
    python3 tools/check_api_guard.py             # check (CI)
    python3 tools/check_api_guard.py --list      # every entry point + verdict
    python3 tools/check_api_guard.py --selftest  # planted cases
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_GLOB = os.path.join(ROOT, "src", "api", "*.cpp")

_DEF = re.compile(r"^ImpError\s+(imp_\w+)\s*\([^;{)]*\)\s*\{", re.M)


def strip_comments_and_strings(src: str) -> str:
    """Blank out comments and string/char literals, keeping line structure."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(c + " " * (j - i - 2) + c if j - i >= 2 else c)
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


_DELEGATE = re.compile(r"^\s*return\s+(imp_\w+)\s*\(.*\)\s*;\s*$", re.S)


def entry_points(src: str) -> list[tuple[str, bool]]:
    """(name, guarded) for every `ImpError imp_*(...) {` definition in `src`.

    Guarded = the body holds `try {` or `api_guard(`, or is a pure delegation
    (`return imp_other(...);` and nothing else) to an entry point that is.
    """
    clean = strip_comments_and_strings(src)
    bodies = []
    for m in _DEF.finditer(clean):
        depth, i = 0, m.end() - 1
        while i < len(clean):
            if clean[i] == "{":
                depth += 1
            elif clean[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        bodies.append((m.group(1), clean[m.end():i]))
    direct = {
        name: (re.search(r"\btry\s*\{", body) is not None or "api_guard(" in body)
        for name, body in bodies
    }
    found = []
    for name, body in bodies:
        ok = direct[name]
        if not ok:
            d = _DELEGATE.match(body)
            ok = d is not None and direct.get(d.group(1), False)
        found.append((name, ok))
    return found


def scan() -> list[tuple[str, str, bool]]:
    rows = []
    for path in sorted(glob.glob(API_GLOB)):
        with open(path, encoding="utf-8") as f:
            for name, ok in entry_points(f.read()):
                rows.append((os.path.relpath(path, ROOT), name, ok))
    return rows


def selftest() -> int:
    cases = [
        ("inline try", "ImpError imp_a(int x) {\n  if (!x) return IMP_ERROR_INVALID_ARG;\n"
                       "  try { work(); return IMP_SUCCESS; } catch (...) { return IMP_ERROR_INTERNAL; }\n}\n",
         [("imp_a", True)]),
        ("api_guard", "ImpError imp_b(void) {\n  return imp::api_guard(\"imp_b\", [&]() -> ImpError {"
                      " work(); return IMP_SUCCESS; });\n}\n",
         [("imp_b", True)]),
        ("naked body", "ImpError imp_c(ImpContext ctx) {\n  if (!ctx) return IMP_ERROR_INVALID_ARG;\n"
                       "  ctx->engine->reset();\n  return IMP_SUCCESS;\n}\n",
         [("imp_c", False)]),
        ("try only in a comment", "ImpError imp_d(void) {\n  // try { } catch is what this needs\n"
                                  "  work(\"try {\");\n  return IMP_SUCCESS;\n}\n",
         [("imp_d", False)]),
        ("nested braces before the guard", "ImpError imp_e(int k) {\n  if (k) { if (k > 2) { return IMP_ERROR_INVALID_ARG; } }\n"
                                           "  return imp::api_guard(\"imp_e\", [&]() -> ImpError { return IMP_SUCCESS; });\n}\n",
         [("imp_e", True)]),
        ("two functions, second naked", "ImpError imp_f(void) {\n  try { return IMP_SUCCESS; } catch (...) { return IMP_ERROR_INTERNAL; }\n}\n"
                                        "ImpError imp_g(void) {\n  return IMP_SUCCESS;\n}\n",
         [("imp_f", True), ("imp_g", False)]),
        ("declaration is not a definition", "ImpError imp_h(int x);\nvoid other() { try { } catch (...) { } }\n", []),
        ("pure delegation to a guarded one", "ImpError imp_i(int x) {\n  return imp_j(x, 0);\n}\n"
                                             "ImpError imp_j(int x, int y) {\n  try { return IMP_SUCCESS; } catch (...) { return IMP_ERROR_INTERNAL; }\n}\n",
         [("imp_i", True), ("imp_j", True)]),
        ("delegation to a naked one, or with work before it", "ImpError imp_k(int x) {\n  return imp_l(x);\n}\n"
                                                              "ImpError imp_l(int x) {\n  return IMP_SUCCESS;\n}\n"
                                                              "ImpError imp_m(int x) {\n  prep(x);\n  return imp_j(x, 1);\n}\n",
         [("imp_k", False), ("imp_l", False), ("imp_m", False)]),
        ("non-ImpError return is out of scope", "void imp_free(void* p) {\n  delete p;\n}\n", []),
    ]
    fails = 0
    for label, src, want in cases:
        got = entry_points(src)
        ok = got == want
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}: {got}")
    print(f"selftest: {len(cases) - fails}/{len(cases)}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print every entry point with its verdict")
    ap.add_argument("--selftest", action="store_true", help="run the planted classifier cases")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    rows = scan()
    if not rows:
        print("check_api_guard: no `ImpError imp_*` definitions found under src/api/ - the scan is broken")
        return 1
    bad = [(p, n) for p, n, ok in rows if not ok]
    if args.list:
        for p, n, ok in rows:
            print(f"  {'guarded' if ok else 'NAKED  '}  {n}  ({p})")
    print(f"check_api_guard: {len(rows) - len(bad)}/{len(rows)} ImpError entry points guarded")
    for p, n in bad:
        print(f"  NAKED: {n} in {p} - wrap the body in imp::api_guard() or an inline try/catch")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
