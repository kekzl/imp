#!/bin/bash
# Assemble the compute_120f PTX fallback (#1650).
#
# The build emits the fallback as `code=compute_120f`, which is the PTX-only
# form: ptxas never runs over it here or in CI, and the first thing that would
# assemble it is the driver's JIT on the user's GB203. A PTX image that ptxas
# rejects is therefore a runtime failure on a card nobody in this project owns,
# discovered by whoever bought one.
#
# This extracts every PTX image from a built binary and assembles it. It needs
# no GPU: ptxas is a compiler.
#
# Usage: check_ptx_fallback.sh <binary> [arch]
# Exit 0 = every PTX image assembles. 1 = one did not, or there were none.

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

n_listed=$(cuobjdump -all -lptx "$BIN" 2>/dev/null | grep -c 'PTX file' || true)
if [ "$n_listed" -eq 0 ]; then
    # Not "nothing to check": the fallback is on by default, so zero images
    # means the build stopped emitting it and every non-5090 Blackwell lost its
    # only path. IMP_DISABLE_120F_FALLBACK=ON is the deliberate way to opt out,
    # and that build should not be running this check.
    echo "check_ptx_fallback: FAIL — $BIN carries no PTX image at all." >&2
    echo "  The compute_120f fallback is on by default (CMakeLists.txt)." >&2
    echo "  A binary without it runs on sm_120a only." >&2
    exit 1
fi

( cd "$WORK" && cuobjdump -all -xptx all "$BIN" >/dev/null 2>&1 )
mapfile -t files < <(find "$WORK" -name '*.ptx' | sort)
if [ "${#files[@]}" -eq 0 ]; then
    echo "check_ptx_fallback: FAIL — $n_listed PTX image(s) listed, none extracted" >&2
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
    echo "check_ptx_fallback: FAIL — $failed of ${#files[@]} PTX image(s) do not assemble for $ARCH" >&2
    exit 1
fi

echo "check_ptx_fallback: PASS (${#files[@]} PTX images assemble for $ARCH)"
