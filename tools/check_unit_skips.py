#!/usr/bin/env python3
"""The unit lane (`ctest -L unit`: test-core, test-text, the e2e subset) has no
skips. A test that needs a GPU or a model file lives in a GPU-lane binary
(test-kv, or test-e2e outside the unit filter), not behind a runtime skip in a
lane that CI runs without either. tools/check_test_lanes.py counts TEST
macros per binary and cannot see a skip; this reads the lanes' GTest JSON
reports (CMakeLists.txt, --gtest_output) and fails on any skipped test.

Usage: python3 tools/check_unit_skips.py <reports-dir>   (ctest: guard_unit_skips)
"""
import json
import pathlib
import sys

LANES = ("unit_core", "unit_text", "unit_e2e_subset")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    reports = pathlib.Path(sys.argv[1])
    skipped, total = [], 0
    for lane in LANES:
        path = reports / f"{lane}.json"
        if not path.is_file():
            print(f"FAIL: no report at {path}; run `ctest -L unit`, not the binary alone")
            return 1
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for suite in doc.get("testsuites", []):
            for case in suite.get("testsuite", []):
                total += 1
                if case.get("result") == "SKIPPED":
                    skipped.append(f"{lane}: {case.get('classname', suite.get('name'))}.{case['name']}")
    print(f"unit-skips: {len(skipped)} skipped of {total}")
    if skipped:
        print("\n".join("  " + s for s in skipped))
        print("FAIL: the unit lane has no skips. Move the test to a GPU-lane binary "
              "(test-kv, or test-e2e outside _unit_e2e_filter) or make it run here.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
