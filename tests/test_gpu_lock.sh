#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2034 # check() evals its condition later; vars are read there
# scripts/gpu_lock.sh and its use in scripts/require_free_gpu.sh, CPU only (no Docker, no GPU).
# Wired into scripts/ci_static_gates.sh (group gpulock). Owners are simulated with
# IMP_GPU_LOCK_OWNER; the lock file lives in a temp dir.
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
LOCK_SH="$PWD/scripts/gpu_lock.sh"
REQ_SH="$PWD/scripts/require_free_gpu.sh"

T=$(mktemp -d)
SLEEPERS=()
cleanup() {
    IMP_GPU_LOCK="$T/lock" bash "$LOCK_SH" release --force >/dev/null 2>&1
    for p in "${SLEEPERS[@]}"; do kill "$p" 2>/dev/null; done
    rm -rf "$T"
}
trap cleanup EXIT
export IMP_GPU_LOCK="$T/lock"

PASS=0 FAIL=0
ok()  { PASS=$((PASS + 1)); echo "  ok    $1"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL  $1"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
as() { local o="$1"; shift; IMP_GPU_LOCK_OWNER="$o" bash "$LOCK_SH" "$@"; }
sleeper() { sleep 300 & SLEEPERS+=("$!"); LAST=$!; }
# Alive and not a zombie (a CI PID 1 may never reap the killed keeper).
running() { [ -r "/proc/$1/stat" ] && [ "$(sed "s/.*) //" "/proc/$1/stat" | cut -c1)" != Z ]; }

echo "== concurrent acquire: exactly one wins (20 rounds, 2 racers each) =="
wins_ok=0
for round in $(seq 1 20); do
    rm -f "$T/lock"
    if [ $((round % 2)) -eq 0 ]; then
        sleeper; pa=$LAST; sleeper; pb=$LAST
        as /wt/a acquire --pid "$pa" "race a" >"$T/a.out" 2>&1 & ja=$!
        as /wt/b acquire --pid "$pb" "race b" >"$T/b.out" 2>&1 & jb=$!
    else
        as /wt/a acquire "race a" >"$T/a.out" 2>&1 & ja=$!
        as /wt/b acquire "race b" >"$T/b.out" 2>&1 & jb=$!
    fi
    wait "$ja"; ra=$?; wait "$jb"; rb=$?
    loser_out=""
    if [ "$ra" = 0 ] && [ "$rb" = 1 ]; then loser_out="$T/b.out"; fi
    if [ "$ra" = 1 ] && [ "$rb" = 0 ]; then loser_out="$T/a.out"; fi
    if [ -n "$loser_out" ] && grep -q 'held by another session: pid=[0-9]* worktree=/wt/[ab] purpose="race [ab]" since=' "$loser_out"; then
        wins_ok=$((wins_ok + 1))
    else
        echo "    round $round: rc a=$ra b=$rb"; cat "$T/a.out" "$T/b.out"
    fi
    as /wt/a release --force >/dev/null 2>&1
done
check "20/20 rounds: one acquire exit 0, the other exit 1 with the holder line ($wins_ok/20)" \
    '[ "$wins_ok" = 20 ]'
echo "  last loser said: $(cat "$loser_out")"

echo "== status =="
check "status on a free lock prints 'free'" '[ "$(bash "$LOCK_SH" status)" = free ]'
sleeper; holder=$LAST
as /wt/a acquire --pid "$holder" "bench qwen3" >/dev/null
st=$(bash "$LOCK_SH" status)
echo "  status: $st"
check "status prints the holder line" \
    '[[ "$st" == "held: pid=$holder worktree=/wt/a purpose=\"bench qwen3\" since="* ]]'

echo "== ownership =="
check "same worktree re-acquires (reentrant, exit 0)" 'as /wt/a acquire "nested" >/dev/null'
check "check: same worktree passes" 'as /wt/a check'
check "check: other worktree refused (exit 1)" '! as /wt/b check 2>/dev/null'
check "release by another worktree refused" '! as /wt/b release 2>/dev/null && [ -s "$T/lock" ]'
check "release --pid of a non-holder is a no-op" 'as /wt/a release --pid 1 && [ -s "$T/lock" ]'
check "release --pid of the holder drops it" 'as /wt/a release --pid "$holder" >/dev/null && [ ! -e "$T/lock" ]'

echo "== stale holder =="
sleep 0 & dead=$!; wait "$dead"
printf 'pid=%s\nstart=1\nkeeper=0\nworktree=/wt/gone\npurpose=crashed\nsince=2026-01-01T00:00:00Z\n' \
    "$dead" > "$T/lock"
out=$(as /wt/b acquire "after crash" 2>&1); rc=$?
echo "  $out" | head -2
check "dead pid is cleared, new owner acquires (exit $rc)" \
    '[ "$rc" = 0 ] && grep -q "cleared stale lock" <<<"$out" && grep -q "worktree=/wt/b" "$T/lock"'
as /wt/b release >/dev/null
sleeper; alive=$LAST
printf 'pid=%s\nstart=999999999999\nkeeper=0\nworktree=/wt/gone\npurpose=recycled\nsince=x\n' \
    "$alive" > "$T/lock"
check "recycled pid (start time differs) counts as stale" '[ "$(bash "$LOCK_SH" status 2>/dev/null)" = free ]'

echo "== keeper (session hold) =="
IMP_GPU_LOCK_TTL=1 as /wt/a acquire "session" >/dev/null
kpid=$(sed -n 's/^pid=//p' "$T/lock")
check "keeper pid $kpid alive after the acquiring shell exited" 'running "$kpid"'
sleep 1.5
check "keeper past TTL: lock reads free" '[ "$(bash "$LOCK_SH" status 2>/dev/null)" = free ]'
as /wt/a acquire "session" >/dev/null
kpid=$(sed -n 's/^pid=//p' "$T/lock")
as /wt/a release >/dev/null
sleep 0.1
check "release kills the keeper" '! running "$kpid"'

echo "== run =="
out=$(as /wt/a run "make test-gpu" -- bash "$LOCK_SH" status)
check "run holds the lock while cmd runs ($out)" '[[ "$out" == "held: pid="*"purpose=\"make test-gpu\""* ]]'
check "run released it afterwards" '[ ! -e "$T/lock" ]'
as /wt/a run x -- false; rc=$?
check "run passes the command's exit code (1)" '[ "$rc" = 1 ]'
sleeper; as /wt/b acquire --pid "$LAST" "other session" >/dev/null
as /wt/a run x -- touch "$T/ran" 2>/dev/null; rc=$?
check "run refused under another holder, cmd not started (exit $rc)" '[ "$rc" = 1 ] && [ ! -e "$T/ran" ]'

echo "== require_free_gpu.sh honours the lock (no nvidia-smi on PATH) =="
out=$(cd "$T" && PATH=/usr/bin:/bin IMP_GPU_LOCK_OWNER=/wt/a bash "$REQ_SH" "pre-commit" 2>&1); rc=$?
echo "  $out" | head -1
check "held by another worktree: exit 1 with holder line (exit $rc)" \
    '[ "$rc" = 1 ] && grep -q "worktree=/wt/b" <<<"$out"'
out=$(PATH=/usr/bin:/bin IMP_GPU_LOCK_OWNER=/wt/b bash "$REQ_SH" "pre-commit" 2>&1); rc=$?
check "held by this worktree: passes (exit $rc)" '[ "$rc" = 0 ]'

echo "gpu_lock: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
