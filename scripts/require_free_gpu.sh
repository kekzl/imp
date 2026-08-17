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
# The occupied memory is the tell, not the process list: on WSL2 nvidia-smi does
# not show a container holding the card, so a busy GPU can look idle there.
#
# Silent and successful when there is no GPU at all: that case is the caller's
# to decide, and the two callers decide it differently.
#
# Usage: scripts/require_free_gpu.sh "<gate name>"
set -uo pipefail

GATE="${1:-gpu gate}"
THRESHOLD="${IMP_GPU_FREE_MIB:-2000}"

command -v nvidia-smi >/dev/null 2>&1 || exit 0

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
USED="${USED// /}"
case "$USED" in
    '' | *[!0-9]*) exit 0 ;;  # unreadable: do not block on a parse
esac
[ "$USED" -le "$THRESHOLD" ] && exit 0

echo "$GATE: the card is not free (${USED} MiB in use, threshold ${THRESHOLD})." >&2
echo "         Running it now would report the host rather than the change." >&2
# Order matters: >&2 first, so stdout follows the real stderr rather than the
# /dev/null that 2>/dev/null would have just installed.
docker ps --format '         running container: {{.Names}} ({{.Image}})' >&2 2>/dev/null
echo "         Wait for the card and try again. On WSL2 nvidia-smi does not list a" >&2
echo "         container holding the GPU, so 'docker ps' above is part of the answer." >&2
exit 1
