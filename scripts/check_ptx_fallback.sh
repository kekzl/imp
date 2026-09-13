#!/bin/bash
# Assembles the compute_120f PTX fallback (#1650): emitted as code=compute_120f (PTX-only,
# ptxas never runs over it here or in CI); the driver's JIT on a GB203 is the first thing that
# assembles it, so a rejected image is a field failure on a card nobody here owns.
# Needs no GPU (ptxas is a compiler).

set -uo pipefail

BIN="${1:?usage: check_ptx_fallback.sh <binary> [arch]}"
ARCH="${2:-sm_120}"

if ! command -v cuobjdump >/dev/null || ! command -v ptxas >/dev/null; then
    echo "check_ptx_fallback: SKIP (cuobjdump/ptxas not on PATH)"
    exit 0
fi
if [ ! -f "$BIN" ]; then
    echo "check_ptx_fallback: $BIN not found" >&2
    exit 1
fi
# Absolute, because the extraction below runs from a temp directory: a relative
# path silently extracted nothing there, and the guard reported "listed, none
# extracted" against a perfectly good binary.
BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Whether a fallback should exist is a build property, not inferred from image count:
# IMP_DISABLE_120F_FALLBACK=ON emits sm_120a SASS only (0 PTX images is correct there).
# Read from CMakeCache.txt next to the binary to tell that apart from a build that lost it.
CACHE="$(dirname "$BIN")/CMakeCache.txt"
if [ -f "$CACHE" ] && grep -q '^IMP_DISABLE_120F_FALLBACK:BOOL=ON$' "$CACHE"; then
    echo "check_ptx_fallback: SKIP - $CACHE sets IMP_DISABLE_120F_FALLBACK=ON"
    echo "  That build is sm_120a SASS only, by request. No PTX image exists to assemble."
    exit 0
fi

n_listed=$(cuobjdump -all -lptx "$BIN" 2>/dev/null | grep -c 'PTX file' || true)
if [ "$n_listed" -eq 0 ]; then
    # Not "nothing to check": the fallback is on by default and this build did
    # not opt out, so zero images means it stopped emitting the fallback and
    # every non-5090 Blackwell lost its only path.
    echo "check_ptx_fallback: FAIL - $BIN carries no PTX image at all." >&2
    echo "  The compute_120f fallback is on by default (CMakeLists.txt) and" >&2
    echo "  ${CACHE} does not set IMP_DISABLE_120F_FALLBACK=ON." >&2
    echo "  A binary without it runs on sm_120a only." >&2
    exit 1
fi

( cd "$WORK" && cuobjdump -all -xptx all "$BIN" >/dev/null 2>&1 )
mapfile -t files < <(find "$WORK" -name '*.ptx' | sort)
if [ "${#files[@]}" -eq 0 ]; then
    echo "check_ptx_fallback: FAIL - $n_listed PTX image(s) listed, none extracted" >&2
    exit 1
fi

failed=0
for f in "${files[@]}"; do
    if ! err=$(ptxas -arch="$ARCH" -o /dev/null "$f" 2>&1); then
        failed=$((failed + 1))
        echo "check_ptx_fallback: ptxas rejected $(basename "$f")" >&2
        echo "$err" | head -5 >&2
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "check_ptx_fallback: FAIL - $failed of ${#files[@]} PTX image(s) do not assemble for $ARCH" >&2
    exit 1
fi

echo "check_ptx_fallback: PASS (${#files[@]} PTX images assemble for $ARCH)"
