#!/usr/bin/env python3
"""Every GTest macro runs in a declared ctest lane, and no CPU-lane test leaves the CPU lane.

WHY THIS EXISTS
---------------
`docs/DESIGN_DECISIONS.md` "No GPU runner in CI": the required check runs `ctest -L unit`;
tests labelled `gpu` run only when a human runs `make verify-fast` / `make test-gpu`.
This gate keeps that split explicit and fails on the two ways a test silently stops running in CI:

  1. a test runs in NO ctest entry: its source is in no test module, its module has no add_test
     (and is not in NO_CTEST below), or a filtered module's filter complement is not registered;
  2. a test that ran in `ctest -L unit` at the base commit still exists but no longer does
     (moved into a GPU module, dropped from `_unit_e2e_filter`). Deleting a test is not this.

Lanes are derived from CMakeLists.txt (`add_test` + `set_tests_properties(... LABELS ...)`), so a
new GPU test added to a `gpu`-labelled module needs no edit here. The unlaned count (tests whose
only execution is a human on a card) is printed with its delta against the base, not pinned:
the pin it replaced was hand-bumped by 16 of 35 commits on main since 2026-10-01
and caught neither failure above.

WHY IT READS SOURCES AND NOT A BUILD DIRECTORY
----------------------------------------------
A build directory inherits whatever was compiled into it: on 2026-08-21 an uncommitted test
registered only locally moved `--gtest_list_tests` from 995 to 998. Sources are what review sees.

WHAT IT COUNTS
--------------
`TEST` / `TEST_F` / `TEST_P` macros, not `--gtest_list_tests` rows (a TEST_P runs once per value).

Usage:
    python3 tools/check_test_lanes.py             # gate (base: IMP_GATE_BASE, else merge-base origin/main)
    python3 tools/check_test_lanes.py --report    # per-module breakdown
    python3 tools/check_test_lanes.py --selftest  # the gate still fails on both defect classes
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
import io

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Test modules with no ctest entry on purpose: module -> reason. Anything else without one fails.
NO_CTEST = {
    "test-hf-network": "opt-in real huggingface.co fetch (IMP_TEST_NETWORK=1); CI has no network",
}

TEST_RE = re.compile(r"^\s*(TEST|TEST_F|TEST_P)\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)", re.M)
SRC_RE = re.compile(r"tests/([A-Za-z0-9_/]+\.(?:cpp|cu))")


def _call_body(text, start):
    """Text of a CMake call from `start` (just past its open paren) to its matching close paren."""
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    return text[start:i - 1]


def module_sources(text):
    """-> {module: [tests-relative source, ...]} from imp_add_test_module and target_sources."""
    mods = {}
    for kw in ("imp_add_test_module", "target_sources"):
        for m in re.finditer(kw + r"\(\s*(test-[a-z0-9-]+)", text):
            mods.setdefault(m.group(1), []).extend(SRC_RE.findall(_call_body(text, m.end())))
    return {k: sorted(set(v)) for k, v in mods.items()}


def cmake_vars(text):
    return {m.group(1): m.group(2) for m in re.finditer(r'set\(\s*(\w+)\s+"([^"]*)"\s*\)', text)}


def ctest_entries(text):
    """-> [(module, labels, filter)] for each add_test whose COMMAND is a test module.

    filter: None (whole binary), ("+", patterns) or ("-", patterns) from --gtest_filter.
    """
    labels = {}
    for m in re.finditer(r"set_tests_properties\(", text):
        body = _call_body(text, m.end())
        lm = re.search(r'PROPERTIES\s+LABELS\s+"([^"]+)"', body)
        if lm:
            for name in body[:body.index("PROPERTIES")].split():
                labels.setdefault(name, set()).update(lm.group(1).split(";"))
    var = cmake_vars(text)
    out = []
    for m in re.finditer(r"add_test\(\s*NAME\s+(\S+)\s+COMMAND\s+(test-[a-z0-9-]+)\b", text):
        body = _call_body(text, m.start() + len("add_test("))
        filt = None
        fm = re.search(r'--gtest_filter=(-?)([^"\s]+)', body)
        if fm:
            pats = re.sub(r"\$\{(\w+)\}", lambda v: var.get(v.group(1), v.group(0)), fm.group(2))
            filt = ("-" if fm.group(1) else "+", pats.split(":"))
        out.append((m.group(2), labels.get(m.group(1), set()), filt))
    return out


def glob_match(pattern, full):
    rx = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
    return re.match(rx, full) is not None


def runs(filt, full):
    if filt is None:
        return True
    sign, pats = filt
    hit = any(glob_match(p, full) for p in pats)
    return hit if sign == "+" else not hit


def macros(path):
    return TEST_RE.findall(path.read_text(errors="ignore"))


def classify(root):
    """-> (tests, errors, mods). tests: [(module, 'Fixture.Name', lane)], lane unit|gpu|none."""
    text = (root / "CMakeLists.txt").read_text()
    mods = module_sources(text)
    entries = ctest_entries(text)
    errors, tests = [], []

    registered = {f for files in mods.values() for f in files}
    included = set()
    for f in registered:
        p = root / "tests" / f
        if p.exists():
            included.update(re.findall(r'#include\s+"([^"]+\.(?:cpp|cu))"', p.read_text(errors="ignore")))
    for p in sorted((root / "tests").rglob("*")):
        if p.suffix not in (".cpp", ".cu") or not p.is_file():
            continue
        rel = p.relative_to(root / "tests").as_posix()
        n = len(macros(p))
        if n and rel not in registered and rel not in included and p.name not in included:
            errors.append(f"tests/{rel}: {n} TEST macro(s) in no test module (CMakeLists.txt)")

    for mod, files in sorted(mods.items()):
        mine = [(lbl, flt) for m, lbl, flt in entries if m == mod]
        if not mine and mod not in NO_CTEST:
            errors.append(f"{mod}: test module with no add_test; label it unit or gpu in CMakeLists.txt "
                          f"or list it in NO_CTEST in tools/check_test_lanes.py with a reason")
        for f in files:
            p = root / "tests" / f
            if not p.exists():
                continue
            for _, fixture, name in macros(p):
                full = f"{fixture}.{name}"
                lanes = [lbl for lbl, flt in mine if runs(flt, full)]
                if any("unit" in lbl for lbl in lanes):
                    lane = "unit"
                elif lanes:
                    lane = "gpu"
                else:
                    lane = "none"
                    if mine:
                        errors.append(f"{mod}: {full} ({f}) matches no add_test filter of its module")
                tests.append((mod, full, lane))
    return tests, errors, mods


def base_tree(base):
    """Extract CMakeLists.txt + tests/ at `base` into a temp dir; None when git cannot."""
    try:
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", base, "CMakeLists.txt", "tests"],
                              check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"test-lanes: base {base} unreadable ({e})", file=sys.stderr)
        return None
    d = tempfile.TemporaryDirectory()
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        t.extractall(d.name, filter="data")
    return d


def resolve_base():
    base = os.environ.get("IMP_GATE_BASE", "")
    if base and set(base) != {"0"}:
        return base, True
    try:
        return subprocess.run(["git", "-C", str(ROOT), "merge-base", "HEAD", "origin/main"], check=True,
                              capture_output=True, text=True).stdout.strip(), False
    except (OSError, subprocess.CalledProcessError):
        return None, False


def demoted(base_tests, head_tests):
    """Tests in the unit lane at base that still exist at HEAD outside the unit lane."""
    base_unit = {t for _, t, lane in base_tests if lane == "unit"}
    head_unit = {t for _, t, lane in head_tests if lane == "unit"}
    head_all = {t for _, t, _ in head_tests}
    return sorted((base_unit & head_all) - head_unit)


def gate(root, base_root, report):
    tests, errors, mods = classify(root)
    unit = sum(1 for t in tests if t[2] == "unit")
    gpu = sum(1 for t in tests if t[2] != "unit")
    if report:
        print(f"{'module':<16} {'unit':>6} {'no-CI':>6}")
        for mod in mods:
            u = sum(1 for m, _, lane in tests if m == mod and lane == "unit")
            g = sum(1 for m, _, lane in tests if m == mod and lane != "unit")
            print(f"{mod:<16} {u:>6} {g:>6}" + (f"  (NO_CTEST: {NO_CTEST[mod]})" if mod in NO_CTEST else ""))
    delta = ""
    if base_root is not None:
        btests, _, _ = classify(base_root)
        bgpu = sum(1 for t in btests if t[2] != "unit")
        delta = f" (base {bgpu}, {gpu - bgpu:+d})"
        for t in demoted(btests, tests):
            errors.append(f"{t}: ran in `ctest -L unit` at the base, runs only outside CI now")
    print(f"test-lanes: {unit} GTest macro(s) in `ctest -L unit`, {gpu} in no CI lane{delta}")
    for e in errors:
        print(f"FAIL: {e}")
    return 1 if errors else 0


def selftest():
    cmake = """
imp_add_test_module(test-core SOURCES tests/a.cpp)
imp_add_test_module(test-gpu SOURCES tests/g.cu)
imp_add_test_module(test-e2e SOURCES tests/e.cpp)
set(_f "Keep.*")
add_test(NAME unit_core COMMAND test-core)
add_test(NAME unit_e2e COMMAND test-e2e "--gtest_filter=${_f}")
add_test(NAME gpu_gpu COMMAND test-gpu)
add_test(NAME gpu_e2e COMMAND test-e2e "--gtest_filter=-${_f}")
set_tests_properties(unit_core unit_e2e PROPERTIES LABELS "unit")
set_tests_properties(gpu_gpu gpu_e2e PROPERTIES LABELS "gpu")
"""
    files = {"a.cpp": "TEST(A, x) {}\n", "g.cu": "TEST(G, y) {}\n", "e.cpp": "TEST(Keep, z) {}\nTEST(Gpu, w) {}\n"}

    def tree(cm, fs):
        d = tempfile.TemporaryDirectory()
        r = pathlib.Path(d.name)
        (r / "tests").mkdir()
        (r / "CMakeLists.txt").write_text(cm)
        for k, v in fs.items():
            (r / "tests" / k).write_text(v)
        return d

    cases = [
        ("clean tree passes", cmake, files, 0),
        ("new GPU test in a gpu module passes", cmake, {**files, "g.cu": files["g.cu"] + "TEST(G, n) {}\n"}, 0),
        ("orphan test source fails", cmake, {**files, "o.cpp": "TEST(O, q) {}\n"}, 1),
        ("module without add_test fails",
         cmake + "imp_add_test_module(test-new SOURCES tests/n.cpp)\n", {**files, "n.cpp": "TEST(N, q) {}\n"}, 1),
        ("CPU test moved to a gpu module fails", cmake, {**files, "a.cpp": "", "g.cu": files["g.cu"] + "TEST(A, x) {}\n"}, 1),
        ("CPU fixture dropped from the e2e filter fails", cmake.replace('"Keep.*"', '"None.*"'), files, 1),
    ]
    bad = 0
    with tree(cmake, files) as base:
        for label, cm, fs, want in cases:
            with tree(cm, fs) as head:
                with open(os.devnull, "w") as null:
                    old, sys.stdout = sys.stdout, null
                    try:
                        got = gate(pathlib.Path(head), pathlib.Path(base), False)
                    finally:
                        sys.stdout = old
            ok = got == want
            bad += not ok
            print(f"  {'ok' if ok else 'FAIL'}  {label}: expected {want}, got {got}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    base, explicit = resolve_base()
    tmp = base_tree(base) if base else None
    if base is None:
        print("test-lanes: skip base comparison: no IMP_GATE_BASE and no origin/main")
    elif tmp is None and explicit:
        return 1
    try:
        return gate(ROOT, pathlib.Path(tmp.name) if tmp else None, args.report)
    finally:
        if tmp:
            tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
