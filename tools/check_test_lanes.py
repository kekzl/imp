#!/usr/bin/env python3
"""Pin the number of GTest cases that no CI lane runs, and fail when it moves.

WHY THIS EXISTS
---------------
`docs/DESIGN_DECISIONS.md` "No GPU runner in CI" is a decision with a stated
reason, and this file does not argue with it. What was missing is that the
decision's cost was never a number anywhere in the repo. The required check is
the job named `Build`, whose test step is `ctest -L unit`; five whole binaries
(test-compute, test-attention, test-quant, test-kv, test-moe-gdn) and the
complement of the e2e unit filter carry the label `gpu` and execute only when a
human runs `make verify-fast` or `make test-gpu` on a real card.

So this asserts the size of that hole. Growing it stays allowed. Growing it
without anyone noticing does not.

WHY IT READS SOURCES AND NOT A BUILD DIRECTORY
----------------------------------------------
The obvious implementation counts `--gtest_list_tests` on the built binaries.
That was the first implementation and it was wrong, for a reason worth writing
down: a gate that counts by reading `build-dev/` inherits that directory's
provenance, and a build directory is exactly the artefact whose provenance
nobody checks. Measured 2026-08-21: an uncommitted file belonging to another
session, registered in `CMakeLists.txt` but never committed, had been compiled
into `test-quant`, and the count read 998 where the clean tree gives 995. Pinning
998 would have baked a stranger's local diagnostic into a gate on `main`, where
it would then fail on every clean checkout with a message pointing at the
contributor's tests rather than at the bad pin.

Reading sources removes the failure mode rather than mitigating it. A stray file
that was never registered cannot move the number; a stray file that WAS
registered shows up as a `CMakeLists.txt` diff in review, which is where it
belongs.

WHAT IT COUNTS, EXACTLY
-----------------------
**`TEST` / `TEST_F` / `TEST_P` macros in the test sources of each module**, which
is not the same quantity as `--gtest_list_tests`. A `TEST_P` is one macro and
runs once per instantiated value row, so the listed-test figure is larger: 995
listed against 829 macros for the unlaned set on 2026-08-21. Both are honest
readings of "tests no CI lane runs". This file pins the macro count because it is
the one derivable from sources, and it says so in its own failure message so the
two can never be read against each other by accident. The listed-test figure is
recorded beside it in `docs/audit/DEBT_LEDGER_2026_08_21.md`.

Usage:
    python3 tools/check_test_lanes.py           # gate
    python3 tools/check_test_lanes.py --report  # the full per-module breakdown
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CMAKE = ROOT / "CMakeLists.txt"

# Binaries whose every test carries the ctest label `gpu` (CMakeLists.txt
# add_test/set_tests_properties). test-e2e is split by a filter and handled
# separately below.
GPU_ONLY = ("test-compute", "test-attention", "test-quant", "test-kv", "test-moe-gdn")
UNIT_ONLY = ("test-core", "test-text")

TEST_RE = re.compile(r"^\s*(TEST|TEST_F|TEST_P)\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)", re.M)


def module_sources(text):
    """-> {module: [tests/<file>, ...]}, from both registration forms.

    `imp_add_test_module(<name> ... SOURCES ...)` is the common one, and
    `target_sources(<name> PRIVATE ...)` is the one that is easy to miss: it adds
    13 files to test-core, 271 macros' worth. A parse that only handles the first
    form undercounts test-core and silently inflates the unlaned share.
    """
    mods = {}
    for m in re.finditer(r"imp_add_test_module\(\s*(test-[a-z0-9-]+)", text):
        name = m.group(1)
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        mods.setdefault(name, []).extend(re.findall(r"tests/([A-Za-z0-9_]+\.(?:cpp|cu))", text[m.end():i]))
    for m in re.finditer(r"target_sources\(\s*(test-[a-z0-9-]+)", text):
        name = m.group(1)
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        mods.setdefault(name, []).extend(re.findall(r"tests/([A-Za-z0-9_]+\.(?:cpp|cu))", text[m.end():i]))
    return {k: sorted(set(v)) for k, v in mods.items()}


def unit_filter(text):
    m = re.search(r'set\(_unit_e2e_filter\s+"([^"]+)"\)', text)
    if not m:
        print("ERROR: could not find _unit_e2e_filter in CMakeLists.txt", file=sys.stderr)
        sys.exit(2)
    return m.group(1).split(":")


def matches(patterns, fixture, name):
    full = f"{fixture}.{name}"
    for p in patterns:
        if p.endswith(".*"):
            if fixture == p[:-2]:
                return True
        elif p == full:
            return True
    return False


def macros(path):
    return TEST_RE.findall(path.read_text(errors="ignore"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--pin", type=int, default=None,
                    help="expected unlaned macro count (default: read PINNED below)")
    args = ap.parse_args()

    PINNED = 975

    text = CMAKE.read_text()
    mods = module_sources(text)
    patterns = unit_filter(text)

    counts, unlaned, laned = {}, 0, 0
    for mod, files in sorted(mods.items()):
        n = 0
        for f in files:
            p = ROOT / "tests" / f
            if p.exists():
                n += len(macros(p))
        counts[mod] = n
        if mod in GPU_ONLY:
            unlaned += n
        elif mod in UNIT_ONLY:
            laned += n

    e2e_unit = e2e_gpu = 0
    for f in mods.get("test-e2e", []):
        p = ROOT / "tests" / f
        if not p.exists():
            continue
        for _, fixture, name in macros(p):
            if matches(patterns, fixture, name):
                e2e_unit += 1
            else:
                e2e_gpu += 1
    unlaned += e2e_gpu
    laned += e2e_unit

    if args.report:
        print(f"{'module':<16} {'macros':>7}  lane")
        for mod, n in counts.items():
            lane = "gpu only" if mod in GPU_ONLY else ("unit" if mod in UNIT_ONLY else "split")
            print(f"{mod:<16} {n:>7}  {lane}")
        print(f"{'  test-e2e unit':<16} {e2e_unit:>7}  unit (filter)")
        print(f"{'  test-e2e gpu':<16} {e2e_gpu:>7}  gpu only")
        print(f"\n{'in a CI lane':<16} {laned:>7}")
        print(f"{'in no CI lane':<16} {unlaned:>7}")
        print(f"{'total':<16} {laned + unlaned:>7}")

    pin = args.pin if args.pin is not None else PINNED
    print(f"test-lanes: {unlaned} GTest macro(s) run in no CI lane (pinned {pin}); "
          f"{laned} run in `ctest -L unit`")
    if unlaned != pin:
        print(f"\nFAIL: the unlaned GTest MACRO count is {unlaned}, pinned at {pin}.")
        print("This counts TEST/TEST_F/TEST_P macros in sources. It is NOT the")
        print("`--gtest_list_tests` figure, which is larger because a TEST_P runs")
        print("once per instantiated value row. Do not compare the two.")
        print("\nNot automatically a regression: it is the number of tests whose only")
        print("execution is a human running `make verify-fast` / `make test-gpu` on a")
        print("real card. If you added GPU tests, re-pin PINNED in this file and say")
        print("so in the PR. If it DROPPED, a test moved into the CPU lane -- re-pin")
        print("and say that too, because that is the direction worth celebrating.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
