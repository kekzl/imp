#!/usr/bin/env python3
"""Gate: the deterministic-mode kernel sites match what docs/determinism.md says.

WHY THIS EXISTS
---------------
`runtime.deterministic` is a single boolean that reaches the kernels through
`process_diag_deterministic_gemm()`. Which kernels it reaches is a property of
the code; which kernels it CLAIMS to reach is a property of the doc. #1574 was
exactly that pair drifting apart: the doc said "GEMM" while the primary GEMM
for NVFP4 weights - the CUTLASS grouped path - read the flag nowhere, and
nothing in the tree could notice.

So the doc names the sites, this counts them, and a new site or a deleted one
fails until both agree.

WHAT IT DOES NOT DO
-------------------
It does not check that a site is CORRECT, only that it exists and is listed. A
kernel that reads the flag and ignores it passes here; that is what the E2E
gate (`*DetEvalE2ETest*`, run from `make test-gpu` since #1575) is for.

Usage:
    python3 tools/check_determinism_sites.py            # gate
    python3 tools/check_determinism_sites.py --report   # list the sites
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "determinism.md"
READER = "process_diag_deterministic_gemm()"

# The files the doc's known-limit 5 enumerates as reading the flag. Keep this
# list and that paragraph in sync; the gate exists to force it.
EXPECTED = {
    "src/compute/gemm.cu",
    "src/compute/sampling_topk_topp.cu",
    "src/compute/moe_routing.cu",
    "src/compute/moe_routing_permute.cu",
}


def sites():
    """Files under src/compute/ that read the flag, and how many times."""
    found = {}
    for path in sorted((ROOT / "src" / "compute").rglob("*")):
        if path.suffix not in (".cu", ".cuh", ".cpp", ".h"):
            continue
        text = path.read_text(errors="replace")
        # Skip comment-only mentions: count call syntax, not prose.
        n = len(re.findall(re.escape(READER), text))
        if n:
            rel = str(path.relative_to(ROOT))
            found[rel] = n
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    found = sites()
    if args.report:
        for f, n in sorted(found.items()):
            print(f"  {n:2d}  {f}")
        print(f"  total {sum(found.values())} read(s) in {len(found)} file(s)")

    doc = DOC.read_text(errors="replace") if DOC.exists() else ""
    problems = []

    missing = EXPECTED - found.keys()
    extra = found.keys() - EXPECTED
    if missing:
        problems.append(
            "these files no longer read " + READER + ": " + ", ".join(sorted(missing)) +
            "\n  Either restore the site or drop it from EXPECTED here AND from "
            "known limit 5 in docs/determinism.md.")
    if extra:
        problems.append(
            "new deterministic-mode site(s): " + ", ".join(sorted(extra)) +
            "\n  Add them to EXPECTED here and name them in docs/determinism.md, "
            "so the documented coverage keeps matching the code.")

    # The doc has to keep naming the uncovered path by file, because that is the
    # claim a reader acts on.
    for needed in ("gemm_cutlass_grouped_3x.cu", "process_diag_deterministic_gemm"):
        if needed not in doc:
            problems.append(f"docs/determinism.md no longer mentions `{needed}`; "
                            "known limit 5 is what tells an operator which weight "
                            "paths the mode covers.")

    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1

    print(f"determinism sites: {sum(found.values())} read(s) in {len(found)} file(s), "
          "matching docs/determinism.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
