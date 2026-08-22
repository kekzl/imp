#!/bin/sh
# Guard: every abnormal end of a request goes through Engine::cancel_sequence_.
#
# #1632. Six sites in engine_scheduler.cpp called kv_manager_->free_sequence()
# and nothing else. KV came back; the recurrent-state slot did not, and it is a
# fixed-size pool - once empty, every later sequence falls back to `id % cap`
# aliasing and two live sequences share one SSM state.
#
# The compiler cannot catch a seventh site, because calling free_sequence() is
# a perfectly valid thing to write. This asserts that the scheduler reaches it
# only through the teardown helper, so adding a cancel path without the slot
# release fails here rather than in someone's output six months later.
#
# Usage: check_cancel_teardown.sh <repo-root>
# Exit 0 = no direct call; non-zero = a site bypasses the helper.

set -eu

ROOT="${1:-.}"
FILE="$ROOT/src/runtime/engine_scheduler.cpp"

if [ ! -f "$FILE" ]; then
    echo "check_cancel_teardown: $FILE not found" >&2
    exit 2
fi

# free_sequence on the request being torn down. The KV manager's own callers
# elsewhere (eviction, reset) are not request teardown and are not in scope.
HITS=$(grep -n 'free_sequence(req->id)' "$FILE" || true)

# A guard that only ever says "I found nothing bad" is indistinguishable from
# one pointed at the wrong file. This one asserts the positive too: the six
# cancel sites the helper replaced must still be calling it. Rename the file,
# move the code, or delete the helper, and this fails instead of passing.
CALLS=$(grep -c 'cancel_sequence_(req)' "$FILE" || true)
MIN_CALLS=6

if [ "$CALLS" -lt "$MIN_CALLS" ]; then
    echo "check_cancel_teardown: FAIL" >&2
    echo "" >&2
    echo "Found $CALLS call(s) to cancel_sequence_(req), expected at least $MIN_CALLS." >&2
    echo "Either a cancel path was removed, or this guard is looking at the wrong" >&2
    echo "file and would have passed without checking anything. See #1632." >&2
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
