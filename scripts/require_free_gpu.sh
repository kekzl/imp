#!/bin/bash
# Refuse a GPU gate that would judge the host instead of the change.
#
# Both local gates end in a CUDA run: pre-commit executes the full suite, and
# pre-push executes verify-fast. A co-tenant on the card turns either of them
# into a statement about the box. VramQueryTest sizes its budget against free
# VRAM and fails outright, the perf gate reads a card it does not have to
# itself, and the report points at the diff, which is the wrong place to look.
# Before this check that cost a Docker build plus the whole suite to find out.
#
# THIS ASKS TWO QUESTIONS AND USED TO ANSWER BOTH WITH ONE NUMBER (2026-08-21).
# The old rule was `memory.used > 2000 MiB`, and its header argued "the occupied
# memory is the tell, not the process list". That was true of the tenant it was
# written against - an imp server holding weights shows up as gigabytes - and
# false of everything else:
#
#   1. "Is someone COMPUTING?" is utilisation, not memory. A tenant that is
#      compute-heavy and VRAM-light passes a memory-only check while corrupting
#      every number the gate is protecting. Measured on this box: a full Unreal
#      Engine render peaks at 2385 MiB and 71 % utilisation, so its own share is
#      ~700 MiB - invisible to a memory rule, ruinous to a perf measurement.
#
#   2. "Is there ROOM for my model?" is free VRAM. Still a real question, with a
#      different failure and a different message, and it is not contention.
#
# And the memory threshold was picked, not derived. This box idles at 1675 MiB
# (seen as low as 1435), so 2000 left ~325 MiB of margin against a card doing
# nothing. It misfired on 2026-08-21 during a container teardown and cost a
# round trip. Worse, the misfire rate is highest exactly when iterating fastest,
# because back-to-back runs are when a previous container is most likely to be
# unwinding as the next gate looks.
#
# So: SAMPLE, and take the MINIMUM memory rather than one reading. A teardown
# flicker is not the minimum of five samples; a real tenant is.
#
# Silent and successful when there is no GPU at all: that case is the caller's
# to decide, and the two callers decide it differently.
#
# Usage: scripts/require_free_gpu.sh "<gate name>"
set -uo pipefail

GATE="${1:-gpu gate}"
# Derived from this box, not rounded: idle 1675 (1435 low), plus a ~700 MiB
# non-imp tenant, is 2400. imp's own models are 8000+ MiB. 4000 separates them
# with room on both sides.
THRESHOLD="${IMP_GPU_FREE_MIB:-4000}"
# Idle here reads 3-8 %. Sustained load above this is somebody computing.
BUSY_PCT="${IMP_GPU_BUSY_PCT:-25}"
SAMPLES="${IMP_GPU_SAMPLES:-5}"

command -v nvidia-smi >/dev/null 2>&1 || exit 0

MEMS=() UTILS=() LINES=""
for _ in $(seq 1 "$SAMPLES"); do
    row=$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
                     --format=csv,noheader,nounits 2>/dev/null | head -1)
    m="${row%%,*}"; u="${row##*,}"
    m="${m// /}"; u="${u// /}"
    case "$m$u" in '' | *[!0-9]*) exit 0 ;; esac  # unreadable: do not block on a parse
    MEMS+=("$m"); UTILS+=("$u")
    LINES="${LINES}${m} MiB/${u}%  "
    sleep 0.4
done

min_of() { printf '%s\n' "$@" | sort -n | head -1; }
med_of() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }

MEM_MIN=$(min_of "${MEMS[@]}")
UTIL_MED=$(med_of "${UTILS[@]}")

BUSY=0
[ "$UTIL_MED" -ge "$BUSY_PCT" ] && BUSY=1
FULL=0
[ "$MEM_MIN" -gt "$THRESHOLD" ] && FULL=1
[ "$BUSY" -eq 0 ] && [ "$FULL" -eq 0 ] && exit 0

if [ "$BUSY" -eq 1 ]; then
    echo "$GATE: something is computing on the card (${UTIL_MED}% median utilisation" >&2
    echo "         over $SAMPLES samples, threshold ${BUSY_PCT}%)." >&2
else
    echo "$GATE: the card is occupied (${MEM_MIN} MiB held across all $SAMPLES samples," >&2
    echo "         threshold ${THRESHOLD})." >&2
fi
echo "         Running it now would report the host rather than the change." >&2
# The samples themselves, so a reader can tell a flicker from a tenant. A single
# number invites "raise the threshold" as the fix, which is the wrong fix.
echo "         samples: ${LINES}" >&2
# Order matters: >&2 first, so stdout follows the real stderr rather than the
# /dev/null that 2>/dev/null would have just installed.
HOLDERS=$(docker ps --format '{{.Names}} ({{.Image}})' 2>/dev/null)
if [ -n "$HOLDERS" ]; then
    # Line by line, NOT `printf ... $HOLDERS`: unquoted word splitting turned
    # one container into two lines, "name" and "(image)", which reads as two
    # tenants where there is one.
    while IFS= read -r h; do
        [ -n "$h" ] && echo "         running container: $h" >&2
    done <<< "$HOLDERS"
else
    echo "         No container holds it. On WSL2 that does NOT mean nobody does:" >&2
    echo "         a Windows-side process (an Unreal Engine run from /mnt/c, a game," >&2
    echo "         a browser) never appears in 'docker ps' and never will." >&2
fi
echo "         Wait for the card and try again." >&2
exit 1
