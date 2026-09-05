#!/bin/sh
# scripts/verify.sh carries a literal copy of CMake's `_unit_e2e_filter`
# (the container path has no configured build/ dir to read it from). The two
# copies must be byte-identical (AUDIT_arch_2026 I-3).
#
# verify.sh's own comment said going stale "is caught by the primary ctest
# guard plus the frozen EXPECTED list". Neither reads verify.sh's string:
# guard_e2e_lane_split gets the CMake variable, check_verify_filter.sh greps
# `^ *FILTER="` only. So #1795 added two suites to CMake on 2026-08-27, the
# copy kept the old list, and every full `make verify` failed its lane-split
# check for nine days.
#
# Usage: check_lane_filter_copy.sh <repo-root>
# Exit 0 = identical; 1 = drift; 2 = a literal could not be read.
set -eu
ROOT="${1:?usage: check_lane_filter_copy.sh <repo-root>}"
CMAKE_COPY=$(sed -n 's/^ *set(_unit_e2e_filter "\([^"]*\)").*/\1/p' "$ROOT/CMakeLists.txt" | head -1)
VERIFY_COPY=$(sed -n 's/^ *_LANE_FILTER="\([^"]*\)".*/\1/p' "$ROOT/scripts/verify.sh" | head -1)
if [ -z "$CMAKE_COPY" ] || [ -z "$VERIFY_COPY" ]; then
    echo "check_lane_filter_copy: could not read both literals" >&2
    echo "  CMakeLists.txt _unit_e2e_filter='$CMAKE_COPY'" >&2
    echo "  scripts/verify.sh _LANE_FILTER='$VERIFY_COPY'" >&2
    exit 2
fi
if [ "$CMAKE_COPY" != "$VERIFY_COPY" ]; then
    echo "check_lane_filter_copy: FAIL - scripts/verify.sh _LANE_FILTER drifted from CMakeLists.txt _unit_e2e_filter" >&2
    echo "  CMakeLists.txt: $CMAKE_COPY" >&2
    echo "  verify.sh:      $VERIFY_COPY" >&2
    echo "  Copy the CMake string into verify.sh (every full 'make verify' fails its lane split otherwise)." >&2
    exit 1
fi
echo "check_lane_filter_copy: OK (verify.sh _LANE_FILTER matches CMakeLists.txt _unit_e2e_filter)"
