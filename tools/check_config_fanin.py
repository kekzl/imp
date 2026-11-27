#!/usr/bin/env python3
"""AUDIT_ARCH_2026_11_24 X-4 / B-123 gate: config sections stay behind a narrow include fan-in.

A header that includes core/dispatch_policy.h or runtime/config.h hands every config/core section
(core/config/*.h) to each of its includers: engine.h did, so a gemm key recompiled 70 more TUs
(139 TUs reached core/config/gemm.h on 2026-11-27, 95 after). Fails on:
  1. a header under src/ or tools/ that includes either one and is not in ALLOW
  2. an ALLOW row whose include is gone (the list only shrinks)
A TU that reads one section includes core/config/<section>.h and core/dispatch_policy_fwd.h.

Usage:
    python3 tools/check_config_fanin.py
    python3 tools/check_config_fanin.py --selftest
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADERS = (".h", ".cuh", ".hpp")
WIDE = ("core/dispatch_policy.h", "runtime/config.h")
# (header, included) -> reason.
ALLOW = {
    ("src/runtime/config.h", "core/dispatch_policy.h"): "RuntimeConfig holds every section by value",
    ("src/runtime/snapshot_host_default.h", "runtime/config.h"): "inline sizing over RuntimeConfig, 3 TUs",
    ("src/runtime/rope_override.h", "runtime/config.h"): "takes RuntimeConfig::Rope, 2 TUs",
    ("src/runtime/offload_spec_defaults.h", "runtime/config.h"): "inline defaults over RuntimeConfig, 3 TUs",
}
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def check(files: dict[str, str]) -> list[str]:
    errors, seen = [], set()
    for path in sorted(files):
        if not path.endswith(HEADERS):
            continue
        for inc in INCLUDE.findall(files[path]):
            if inc not in WIDE:
                continue
            seen.add((path, inc))
            if (path, inc) not in ALLOW:
                errors.append(f"{path}: includes {inc}; forward-declare (core/dispatch_policy_fwd.h, "
                              f"struct RuntimeConfig) and include core/config/<section>.h where read")
    for key in sorted(ALLOW):
        if key not in seen:
            errors.append(f"ALLOW row {key[0]} -> {key[1]} matches nothing; drop it")
    return errors


def repo_files() -> dict[str, str]:
    out = {}
    for top in ("src", "tools"):
        for path in (REPO / top).rglob("*"):
            if path.is_file() and path.suffix in HEADERS:
                out[path.relative_to(REPO).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return out


def selftest() -> int:
    base = {k[0]: f'#include "{k[1]}"\n' for k in ALLOW}
    cases = [
        ("clean tree", {}, 0),
        ("engine.h includes the policy again", {"src/runtime/engine.h": '#include "core/dispatch_policy.h"\n'}, 1),
        ("a tool header includes runtime/config.h", {"tools/imp-server/x.h": ' # include "runtime/config.h"\n'}, 1),
        ("a .cpp may include it", {"src/exec/a.cpp": '#include "core/dispatch_policy.h"\n'}, 0),
        ("the forward header is fine", {"src/exec/b.h": '#include "core/dispatch_policy_fwd.h"\n'}, 0),
        ("one section header is fine", {"src/exec/c.cuh": '#include "core/config/attention.h"\n'}, 0),
        ("stale allow row", {"src/runtime/rope_override.h": "int x;\n"}, 1),
    ]
    bad = 0
    for name, extra, want in cases:
        errors = check({**base, **extra})
        if len(errors) != want:
            print(f"selftest: {name}: want {want} error(s), got {len(errors)}: {errors}")
            bad += 1
    print(f"selftest: {len(cases) - bad}/{len(cases)} cases ok")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return selftest()
    files = repo_files()
    errors = check(files)
    for e in errors:
        print(e)
    print(f"config fan-in: {len(files)} headers, {len(ALLOW)} allowed wide includes, {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
