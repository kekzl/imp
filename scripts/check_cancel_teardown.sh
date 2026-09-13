#!/bin/sh
# Guard: every abnormal request end must go through Engine::cancel_sequence_ (#1632).
# A site calling kv_manager_->free_sequence() directly frees KV but not the recurrent-state
# slot (fixed-size pool); once empty, later sequences alias via id%cap and share one SSM state.
# Usage: check_cancel_teardown.sh <repo-root>. Exit 0 = no direct call.

set -eu

ROOT="${1:-.}"
FILES="src/runtime/engine_scheduler.cpp \
       src/runtime/engine_prefill.cpp \
       src/runtime/engine_prefill_ragged.cpp \
       src/runtime/engine_decode_pipeline.cpp"

CALLS=0
HITS=""
for f in $FILES; do
    P="$ROOT/$f"
    if [ ! -f "$P" ]; then
        echo "check_cancel_teardown: $P not found" >&2
        exit 2
    fi
    # free_sequence on the request being torn down. The KV manager's own
    # callers elsewhere (eviction, reset) are not request teardown and are
    # not in scope.
    H=$(grep -n 'free_sequence(req->id)' "$P" || true)
    if [ -n "$H" ]; then
        HITS="$HITS$f: $H
"
    fi
    C=$(grep -c 'cancel_sequence_(' "$P" || true)
    CALLS=$((CALLS + C))
done

# Asserts the positive too: the cancel sites the helper replaced must still call it, so
# renaming/moving the code or deleting the helper fails here instead of passing on "found nothing".
MIN_CALLS=6

if [ "$CALLS" -lt "$MIN_CALLS" ]; then
    echo "check_cancel_teardown: FAIL" >&2
    echo "" >&2
    echo "Found $CALLS call(s) to cancel_sequence_(...), expected at least $MIN_CALLS." >&2
    echo "Either a cancel path was removed, or this guard is looking at the wrong" >&2
    echo "files and would have passed without checking anything. See #1632." >&2
    exit 1
fi

if [ -n "$HITS" ]; then
    echo "check_cancel_teardown: FAIL" >&2
    echo "" >&2
    echo "These sites free KV directly instead of calling cancel_sequence_(req):" >&2
    echo "$HITS" >&2
    echo "" >&2
    echo "cancel_sequence_ (src/runtime/engine.cpp) also releases the recurrent-state" >&2
    echo "slot, which free_sequence does not. See #1632." >&2
    exit 1
fi

echo "check_cancel_teardown: PASS ($CALLS teardown calls, no direct free_sequence(req->id))"
