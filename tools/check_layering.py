#!/usr/bin/env python3
"""Backward #include edges between src/ layers stay on the pinned list.

WHY THIS EXISTS
---------------
docs/audit/ARCHMAP.md draws the intended layer order (api above runtime above
exec ... above core). The tree never was that DAG: on 2026-09-05 the layer graph
was one 8-node cycle with 88 backward include lines, 64 of them four
dependency-free headers that sat in runtime/ and compute/ by accident
(AUDIT_arch_2026 G-1, P0-1..P0-5). Dispatch #14 moved those to core/, deleted
the one dead `runtime/config.h` include in exec/, and left 24 backward lines:
six pre-dequant TUs that read the Engine-planned VRAMBudget (executor.h
forward-declares it to cut its fan-out) and 18 placement decisions the audit's
SCC simulation showed no move repairs. Nothing kept the count from drifting back up
(the `runtime/config.h` include had returned once already, #1388, after F-10
had closed it at zero).

WHAT IT CHECKS
--------------
Every `#include "<layer>/..."` line under src/<layer>/ is an edge. The layer
order is

    api > runtime > vision > exec > lora > model > compute > quant > memory > core

and an edge from a lower to a higher layer is backward. Each backward
(includer layer, included header) pair must be listed in
tools/layering_pins.txt with a ceiling and a reason. The gate fails on

  * a backward pair that is not pinned,
  * a pinned pair whose line count exceeds its ceiling,
  * a pinned pair with no remaining line (stale pin; the list cannot rot).

A count below its ceiling is reported so the pin can be tightened; --update
rewrites every ceiling to the current count and drops stale pins.

Usage:
    python3 tools/check_layering.py             # check (CI)
    python3 tools/check_layering.py --list      # every backward include line
    python3 tools/check_layering.py --update    # re-pin ceilings to today's counts
    python3 tools/check_layering.py --selftest  # planted cases
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PINS = ROOT / "tools" / "layering_pins.txt"

RANK = {
    "core": 0, "memory": 1, "quant": 2, "compute": 3, "model": 4,
    "lora": 5, "exec": 6, "vision": 7, "runtime": 8, "api": 9,
}
EXTS = (".h", ".cuh", ".cu", ".cpp", ".inl")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([a-z_]+)/([^"]+)"', re.M)


def edges_in(rel_path: str, text: str) -> list[tuple[str, str, str, int]]:
    """(includer layer, includee layer, included header, line) for every layer include."""
    parts = rel_path.split("/")
    if len(parts) < 3 or parts[0] != "src" or parts[1] not in RANK:
        return []
    layer = parts[1]
    out = []
    for m in INCLUDE_RE.finditer(text):
        to_layer = m.group(1)
        if to_layer not in RANK:
            continue
        line = text.count("\n", 0, m.start()) + 1
        out.append((layer, to_layer, f"{to_layer}/{m.group(2)}", line))
    return out


def is_backward(from_layer: str, to_layer: str) -> bool:
    return RANK[from_layer] < RANK[to_layer]


def scan() -> tuple[int, dict[tuple[str, str], list[tuple[str, int]]]]:
    """Returns (forward line count, {(layer, header): [(file, line), ...]} for backward lines)."""
    forward = 0
    back: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for p in sorted(SRC.rglob("*")):
        if not p.is_file() or p.suffix not in EXTS:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for layer, to_layer, header, line in edges_in(rel, p.read_text(errors="replace")):
            if is_backward(layer, to_layer):
                back.setdefault((layer, header), []).append((rel, line))
            else:
                forward += 1
    return forward, back


def read_pins(path: pathlib.Path) -> dict[tuple[str, str], tuple[int, str]]:
    pins: dict[tuple[str, str], tuple[int, str]] = {}
    if not path.exists():
        return pins
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split(None, 3)
        if len(cols) < 4:
            raise SystemExit(f"{path.name}: malformed line: {raw!r} (want: layer header ceiling reason)")
        pins[(cols[0], cols[1])] = (int(cols[2]), cols[3])
    return pins


def verdicts(back: dict[tuple[str, str], list], pins: dict[tuple[str, str], tuple[int, str]]):
    """Yields (status, layer, header, count, ceiling). status in ok / below / unpinned / over / stale."""
    for key in sorted(set(back) | set(pins)):
        n = len(back.get(key, []))
        if key not in pins:
            yield "unpinned", key[0], key[1], n, None
        elif n == 0:
            yield "stale", key[0], key[1], 0, pins[key][0]
        elif n > pins[key][0]:
            yield "over", key[0], key[1], n, pins[key][0]
        elif n < pins[key][0]:
            yield "below", key[0], key[1], n, pins[key][0]
        else:
            yield "ok", key[0], key[1], n, pins[key][0]


def write_pins(back: dict[tuple[str, str], list], pins: dict[tuple[str, str], tuple[int, str]]) -> None:
    head = [
        "# Backward #include edges between src/ layers, each with its ceiling and the",
        "# reason it is tolerated. tools/check_layering.py fails on an unlisted backward",
        "# edge, on a count above its ceiling, and on a stale line (edge gone).",
        "# Layer order: api > runtime > vision > exec > lora > model > compute > quant > memory > core.",
        "# Baseline 2026-09-06 (AUDIT_arch_2026 dispatch #14): 88 lines -> 24, four headers",
        "# moved to core/, storage_planner to exec/. What is left is the vram_budget.h",
        "# forward-declare trade and placement the audit's SCC simulation showed no move",
        "# repairs (memory <-> model, model <-> vision, vision <-> runtime,",
        "# compute -> model -> quant -> compute).",
        "#",
        "# Format: <includer layer> <included header> <ceiling> <reason>",
        "",
    ]
    rows = []
    for key in sorted(set(back) | set(pins)):
        n = len(back.get(key, []))
        if n == 0:
            continue
        reason = pins.get(key, (0, "TODO: reason"))[1]
        rows.append(f"{key[0]:<8}{key[1]:<40}{n:<4}{reason}")
    PINS.write_text("\n".join(head + rows) + "\n")


def selftest() -> int:
    pins = {("exec", "runtime/vram_budget.h"): (1, "r"), ("model", "exec/gone.h"): (1, "r")}
    files = {
        "src/exec/a.cu": '#include "runtime/vram_budget.h"\n#include "compute/gemm.h"\n',
        "src/exec/b.cu": '#include "runtime/vram_budget.h"\n',
        "src/quant/c.cu": '#include "compute/pdl_device.cuh"\n#include "core/tensor.h"\n',
        "src/core/d.h": '#include <string>\n#include "core/logging.h"\n',
        "tests/e.cu": '#include "runtime/engine.h"\n',
    }
    back: dict[tuple[str, str], list] = {}
    forward = 0
    for rel, text in files.items():
        for layer, to_layer, header, line in edges_in(rel, text):
            if is_backward(layer, to_layer):
                back.setdefault((layer, header), []).append((rel, line))
            else:
                forward += 1
    got = {(s, l, h): (n, c) for s, l, h, n, c in verdicts(back, pins)}
    want = {
        ("over", "exec", "runtime/vram_budget.h"): (2, 1),
        ("unpinned", "quant", "compute/pdl_device.cuh"): (1, None),
        ("stale", "model", "exec/gone.h"): (0, 1),
    }
    cases = [
        ("forward edges counted (exec->compute, quant->core, core->core), tests/ ignored", forward == 3),
        ("over-ceiling backward edge fails", got.get(("over", "exec", "runtime/vram_budget.h")) == (2, 1)),
        ("unpinned backward edge fails", got.get(("unpinned", "quant", "compute/pdl_device.cuh")) == (1, None)),
        ("stale pin fails", got.get(("stale", "model", "exec/gone.h")) == (0, 1)),
        ("no other verdicts", got == want),
    ]
    fails = 0
    for label, ok in cases:
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    print(f"selftest: {len(cases) - fails}/{len(cases)}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print every backward include line")
    ap.add_argument("--update", action="store_true", help="re-pin every ceiling to the current count")
    ap.add_argument("--selftest", action="store_true", help="run the planted cases")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    forward, back = scan()
    if forward == 0:
        print("check_layering: no layer includes found under src/ - the scan is broken")
        return 1
    pins = read_pins(PINS)
    if args.update:
        write_pins(back, pins)
        print(f"check_layering: re-pinned {len([k for k in back if back[k]])} edges in {PINS.relative_to(ROOT)}")
        return 0
    if args.list:
        for key in sorted(back):
            for rel, line in back[key]:
                print(f"  {key[0]:<8}-> {key[1]:<40}{rel}:{line}")
    bad = 0
    lines = sum(len(v) for v in back.values())
    for status, layer, header, n, ceiling in verdicts(back, pins):
        if status == "ok":
            continue
        if status == "below":
            print(f"  below   {layer} -> {header}: {n} < ceiling {ceiling} (run --update to tighten)")
            continue
        bad += 1
        if status == "unpinned":
            print(f"  UNPINNED {layer} -> {header}: {n} line(s); a lower layer now includes a higher one. "
                  f"Move the header down or pin it in {PINS.name} with a reason")
        elif status == "over":
            print(f"  OVER     {layer} -> {header}: {n} > ceiling {ceiling}")
        else:
            print(f"  STALE    {layer} -> {header}: pinned at {ceiling}, 0 lines left; drop the pin")
    print(f"check_layering: {forward} forward lines, {lines} backward lines over "
          f"{len([k for k in back if back[k]])} pinned edges, {bad} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
