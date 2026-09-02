#!/bin/sh
# Can a change to this shell script change what `scripts/verify.sh` does?
#
# The pre-push hook gates on `^scripts/`, which is right for verify.sh itself
# and for the guard scripts CMake registers as ctests (they run inside
# `ctest -L unit`, which verify.sh runs). It is wrong for the other 17 scripts
# in there: ci_static_gates.sh, check-release.sh, the bench harnesses and the
# server smoke drivers are invoked by CI jobs, by the Makefile or by hand, and
# nothing in verify.sh's path reaches them. Editing one used to cost the full
# GPU suite on a shared card for a run that cannot change its own outcome -
# the same over-gating .md/.py/.hook were excluded for (#1723, #1825).
#
# The membership test is the reference, not a copied list: a script counts as
# reachable when verify.sh or CMakeLists.txt names it. Adding a call to a script
# therefore re-arms its gate with no edit here.
#
# Usage: verify_touches.sh <path>     exit 0 = reachable, 1 = not
set -eu

P="${1:?usage: verify_touches.sh <path>}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$(basename "$P")

case "$P" in
    scripts/*.sh) ;;
    *) exit 0 ;;              # only shell scripts under scripts/ are in question
esac

grep -q -- "$B" "$ROOT/scripts/verify.sh" "$ROOT/CMakeLists.txt" 2>/dev/null
