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

SCOPE
-----
All of `src/`, comments stripped, minus the accessor's own definition. It used
to scan `src/compute/` only, which left the gate blind exactly where its
founding defect lived: the MoE CUTLASS dispatch is `src/exec/`, so a new reader
there passed unnoticed (verified 2026-09-02 by planting one). Comments are
stripped because two prose mentions in `src/runtime/` would otherwise count as
sites.

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
# file -> number of reads. The count is pinned, not just the file: gemm.cu
# reads the flag in two dispatch branches and losing one of them is the same
# drift as losing the file (a branch that stops honouring the mode), which the
# file-level check passed silently until 2026-09-02.
EXPECTED = {
    "src/compute/gemm.cu": 2,
    "src/compute/sampling_topk_topp.cu": 1,
    "src/compute/moe_routing.cu": 2,
    "src/compute/moe_routing_permute.cu": 1,
}

# Where the accessor is declared and defined. Counting these as sites would be
# counting the switch as one of the things it switches.
SELF = {
    "src/runtime/process_diag.cpp",
    "src/runtime/process_diag.h",
}


def strip_comments(text):
    """Blank out // and /* */ comments, keeping line count and other text."""
    out = []
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(c if c == "\n" else " " for c in text[i:j]))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def sites(root=ROOT):
    """Files under src/ that read the flag in code, and how many times."""
    found = {}
    for path in sorted((root / "src").rglob("*")):
        if path.suffix not in (".cu", ".cuh", ".cpp", ".h"):
            continue
        rel = str(path.relative_to(root))
        if rel in SELF:
            continue
        n = len(re.findall(re.escape(READER), strip_comments(path.read_text(errors="replace"))))
        if n:
            found[rel] = n
    return found


def evaluate(found, expected, doc):
    """The gate's whole judgement, as a list of problems."""
    problems = []

    missing = expected.keys() - found.keys()
    extra = found.keys() - expected.keys()
    moved = {f: (expected[f], found[f]) for f in expected.keys() & found.keys()
             if expected[f] != found[f]}
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
    if moved:
        problems.append(
            "read count changed: " +
            ", ".join(f"{f} {was} -> {now}" for f, (was, now) in sorted(moved.items())) +
            "\n  A branch gained or lost the flag. Re-pin EXPECTED here once the "
            "change is deliberate; a lost read is a path that stops honouring the mode.")

    # The doc has to keep naming the uncovered path by file, because that is the
    # claim a reader acts on.
    for needed in ("gemm_cutlass_grouped_3x.cu", "process_diag_deterministic_gemm"):
        if needed not in doc:
            problems.append(f"docs/determinism.md no longer mentions `{needed}`; "
                            "known limit 5 is what tells an operator which weight "
                            "paths the mode covers.")

    return problems


def selftest():
    """Plant each drift this gate exists to catch, on a fixture tree.

    The gate scanned src/compute/ only until 2026-09-02 and pinned files rather
    than read counts, so two of the four cases below passed silently: a reader
    in src/exec (where the MoE CUTLASS dispatch lives, the founding defect's own
    neighbourhood) and a branch inside gemm.cu that stopped reading the flag.
    """
    import tempfile

    call = f"return {READER};"
    doc = "gemm_cutlass_grouped_3x.cu process_diag_deterministic_gemm"
    base = {
        "src/compute/gemm.cu": 2,
        "src/compute/sampling_topk_topp.cu": 1,
        "src/compute/moe_routing.cu": 2,
        "src/compute/moe_routing_permute.cu": 1,
    }

    def tree(root, extra_files=(), counts=None):
        for rel, n in (counts or base).items():
            f = root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("".join(f"bool f{i}() {{ {call} }}\n" for i in range(n)))
        for rel, text in extra_files:
            f = root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)

    cases = [
        ("clean tree", (), None, 0),
        ("reader in src/exec", (("src/exec/executor_forward_moe_cutlass.cu",
                                 f"bool m() {{ {call} }}\n"),), None, 1),
        ("reader in src/quant", (("src/quant/nvfp4_quant.cu",
                                  f"bool m() {{ {call} }}\n"),), None, 1),
        ("comment-only mention", (("src/runtime/engine.cpp",
                                   f"// {READER} in prose\n"),), None, 0),
        ("accessor's own definition", (("src/runtime/process_diag.cpp",
                                        f"bool {READER[:-2]}() {{ return true; }}\n"),), None, 0),
        ("branch lost the flag", (), {**base, "src/compute/gemm.cu": 1}, 1),
        ("branch gained the flag", (), {**base, "src/compute/gemm.cu": 3}, 1),
        ("file lost the flag", (), {k: v for k, v in base.items()
                                    if k != "src/compute/moe_routing.cu"}, 1),
    ]

    failures = 0
    for name, extra, counts, want in cases:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tree(root, extra, counts)
            got = 1 if evaluate(sites(root), EXPECTED, doc) else 0
            ok = got == want
            failures += not ok
            print(f"  {'ok  ' if ok else 'FAIL'}  {name}: expected rc={want}, got rc={got}")
    print(f"selftest: {len(cases) - failures}/{len(cases)} cases")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="check the gate still catches the drift it exists for")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    found = sites()
    if args.report:
        for f, n in sorted(found.items()):
            print(f"  {n:2d}  {f}")
        print(f"  total {sum(found.values())} read(s) in {len(found)} file(s)")

    doc = DOC.read_text(errors="replace") if DOC.exists() else ""
    problems = evaluate(found, EXPECTED, doc)

    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1

    print(f"determinism sites: {sum(found.values())} read(s) in {len(found)} file(s), "
          "matching docs/determinism.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
