#!/bin/sh
# Can a change to this shell script change what verify.sh does? The pre-push hook gates on
# `^scripts/`, right for verify.sh and the CMake-registered guard ctests, wrong for the other 17
# scripts (ci_static_gates, check-release, bench harnesses, server smoke drivers) that nothing
# in verify.sh's path reaches; editing one used to cost the full GPU suite for a run that can't
# change its own outcome (same class as #1723/#1825).
# Membership test is the reference, not a copy: a script counts as reachable when verify.sh or
# CMakeLists.txt names it, so adding a call re-arms its gate with no edit here.
# Usage: verify_touches.sh <path>. Exit 0 = reachable, 1 = not.
set -eu

P="${1:?usage: verify_touches.sh <path>}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$(basename "$P")

case "$P" in
    scripts/*.sh) ;;
    *) exit 0 ;;              # only shell scripts under scripts/ are in question
esac

grep -q -- "$B" "$ROOT/scripts/verify.sh" "$ROOT/CMakeLists.txt" 2>/dev/null
