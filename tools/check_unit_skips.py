#!/usr/bin/env python3
"""Pin, by name, the GTest cases that `ctest -L unit` starts and then skips.

WHY THIS EXISTS
---------------
`tools/check_test_lanes.py` counts TEST macros per binary. A test whose body
opens with `SKIP_IF_NO_CUDA()` or a checkpoint-on-disk check reads there as
"in a CI lane" while never executing in it: before #1860/#1861, 62 of
test-core's cases were in that state, green in every CI run. The merge gate's
honest figure is what a GPU-less run skips, and that number lived in a doc.
This pins it, by name, so a test cannot sit in the lane without running there
and nobody notices.

HOW
---
The three lane commands (`unit_core`, `unit_text`, `unit_e2e_subset` in
CMakeLists.txt) write GTest JSON reports into <build>/gtest-reports/.
`guard_unit_skips` runs after them (ctest DEPENDS) and compares each lane's
skipped set with tools/unit_lane_skips.txt.

RULES
-----
* skipped but not allowlisted   -> FAIL. The defect this exists for.
* allowlisted but ran, or gone  -> FAIL when the run had no CUDA device:
                                   delete the line (the direction worth
                                   celebrating), or fix the rename.
* the run had a device          -> the GPU-gated entries legitimately ran, so
                                   only the first rule is judged.

"The run had no device" is read off the reports: SKIP_IF_NO_CUDA's message
(tests/test_cuda_skip.h) is on at least one skip. Should every GPU-gated test
ever leave the lanes, that signal goes with them and the second rule stops
firing; the allowlist is then model-gated only.

A model env var (IMP_TEST_GGUF, IMP_TEST_MODEL_DEEPSEEK, ...) set on a GPU-less
run makes a model-gated entry run and trips the second rule. CI sets none and
`make dev-test` mounts nothing, so that is a local-setup message, not a gate
defect.

Usage:
    python3 tools/check_unit_skips.py <reports-dir> [--allowlist FILE] [--report]
    make dev-test                      # runs it as guard_unit_skips
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ALLOWLIST = ROOT / "tools" / "unit_lane_skips.txt"
LANES = ("unit_core", "unit_text", "unit_e2e_subset")
NO_DEVICE_MESSAGE = "No CUDA device available"  # tests/test_cuda_skip.h
REASONS = ("cuda", "vmm", "model")


def load_report(path):
    """{Suite.Test: skip message, or None when the test ran} for one lane."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    tests = {}
    for suite in doc.get("testsuites", []):
        for case in suite.get("testsuite", []):
            name = f"{case.get('classname', suite.get('name'))}.{case['name']}"
            if case.get("result") != "SKIPPED":
                tests[name] = None
                continue
            entries = case.get("skipped") or []
            raw = entries[0].get("message", "") if entries else ""
            # GTEST_SKIP() puts "file:line" on the first line, the text after it.
            lines = [ln for ln in raw.splitlines() if ln.strip()]
            tests[name] = " ".join(lines[1:]) if len(lines) > 1 else (lines[0] if lines else "(no message)")
    return tests


def load_allowlist(path):
    """{lane: {Suite.Test: reason}}; lines are `lane test reason`, `#` comments."""
    allow = {lane: {} for lane in LANES}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3 or parts[0] not in LANES or parts[2] not in REASONS:
            sys.exit(f"{path}:{lineno}: expected `<lane> <Suite.Test> <{'|'.join(REASONS)}>`, got: {raw}")
        allow[parts[0]][parts[1]] = parts[2]
    return allow


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reports_dir", type=pathlib.Path)
    ap.add_argument("--allowlist", type=pathlib.Path, default=DEFAULT_ALLOWLIST)
    ap.add_argument("--report", action="store_true", help="list every skipped test with its reason")
    args = ap.parse_args()

    allow = load_allowlist(args.allowlist)
    lanes = {}
    for lane in LANES:
        path = args.reports_dir / f"{lane}.json"
        if not path.is_file():
            print(f"FAIL: no report at {path}. The lane command writes it "
                  f"(--gtest_output=json:, CMakeLists.txt); run `ctest -L unit`, not the binary alone.")
            return 1
        lanes[lane] = load_report(path)

    device_present = not any(msg and NO_DEVICE_MESSAGE in msg
                             for tests in lanes.values() for msg in tests.values())

    failures, ran_with_device, summary = [], [], []
    for lane in LANES:
        tests = lanes[lane]
        skipped = {n: m for n, m in tests.items() if m is not None}
        summary.append(f"{lane} {len(skipped)} skipped of {len(tests)}")
        if args.report:
            print(f"{lane}: {len(skipped)} skipped of {len(tests)}")
            for name, msg in sorted(skipped.items()):
                print(f"  {name:<64} {allow[lane].get(name, 'NOT ALLOWLISTED'):<8} {msg}")
        for name, msg in sorted(skipped.items()):
            if name not in allow[lane]:
                failures.append(f"{lane}: {name} skipped and is not allowlisted ({msg})")
        for name, reason in sorted(allow[lane].items()):
            if name in skipped:
                continue
            if name not in tests:
                failures.append(f"{lane}: allowlisted {name} ({reason}) is not in the run: "
                                "renamed or deleted, fix the line")
            elif device_present:
                ran_with_device.append(f"{lane}: {name} ({reason})")
            else:
                failures.append(f"{lane}: allowlisted {name} ({reason}) RAN in a run without a "
                                "CUDA device: delete its line")

    print(f"unit-skips: {'; '.join(summary)}; "
          f"{'a CUDA device was present' if device_present else 'no CUDA device'}")
    if ran_with_device:
        print(f"  {len(ran_with_device)} allowlisted test(s) ran because a device was present, not judged:")
        for line in ran_with_device:
            print(f"    {line}")
    if failures:
        print(f"\nFAIL: {len(failures)} difference(s) between the skipped set and {args.allowlist}:")
        for line in failures:
            print(f"  {line}")
        print("\nA test that skips in `ctest -L unit` counts as laned for check_test_lanes.py")
        print("and never runs in CI. Make it run without a device or a checkpoint, or")
        print("allowlist it with its reason:  <lane> <Suite.Test> <cuda|vmm|model>")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
