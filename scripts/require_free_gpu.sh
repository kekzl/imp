#!/bin/bash
# Refuse a GPU gate that would judge the host instead of the change: both local gates end in a
# CUDA run (pre-commit full suite, pre-push verify-fast), and a co-tenant on the card corrupts
# the result. Two distinct questions: utilization = "is someone computing" (a compute-heavy,
# VRAM-light tenant passes a memory-only check while corrupting the measurement); free VRAM =
# "is there room for my model" (a different failure, different message).
# Sample and take the MINIMUM memory reading, not one reading: a teardown flicker is not the
# minimum of five samples, a real tenant is. Silent success when there is no GPU at all.
# Usage: scripts/require_free_gpu.sh "<gate name>".
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
