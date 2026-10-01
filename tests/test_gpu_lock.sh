#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2034 # check() evals its condition later; vars are read there
# scripts/gpu_lock.sh and its use in scripts/require_free_gpu.sh, CPU only (no Docker, no GPU).
# Wired into scripts/ci_static_gates.sh (group gpulock). Owners are simulated with
# IMP_GPU_LOCK_OWNER; the lock file lives in a temp dir. Cross-script cases need the skill
# script (GPU_LOCK_SKILL_SH, default ~/.claude/skills/gpu-stats/gpu-lock.sh); absent = skip.
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
export IMP_GPU_LOCK_LEGACY="$T/legacy"
unset GPU_LOCK_FILE
SKILL_SH="${GPU_LOCK_SKILL_SH:-$HOME/.claude/skills/gpu-stats/gpu-lock.sh}"

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

as /wt/b release --force >/dev/null 2>&1

# One lock with the gpu-stats skill script: same file, mutex and record format.
startof() { sed 's/.*) //' "/proc/$1/stat" | awk '{print $20}'; }
sk() { GPU_LOCK_FILE="$T/lock" GPU_BUSY_CHECK=/bin/true bash "$SKILL_SH" "$@"; }
deadpid() { sleep 0 & DEAD=$!; wait "$DEAD"; }
if [ ! -f "$SKILL_SH" ]; then
    echo "== cross-script cases: SKIP, $SKILL_SH absent (set GPU_LOCK_SKILL_SH) =="
else
    echo "== cross A: repo holds, skill refuses =="
    rm -f "$T/lock"; sleeper
    as /wt/r acquire --pid "$LAST" "repo job" >/dev/null; rc=$?
    out=$(sk acquire s p 5 1 fragen 2>&1); src=$?
    echo "  ${out%%$'\n'*}"
    check "repo acquire exit 0, skill acquire exit 1 REFUSED (exit $rc/$src)" \
        '[ "$rc" = 0 ] && [ "$src" = 1 ] && grep -q "^REFUSED: held by /wt/r" <<<"$out"'
    as /wt/r release >/dev/null

    echo "== cross B: skill holds, repo refuses =="
    sk acquire s p 5 1 fragen >/dev/null; rc=$?
    as /wt/other check 2>/dev/null; crc=$?
    as /wt/other run x -- touch "$T/ranB" 2>/dev/null; rrc=$?
    check "skill acquire exit 0, repo check exit 1, repo run exit 1 (exit $rc/$crc/$rrc)" \
        '[ "$rc" = 0 ] && [ "$crc" = 1 ] && [ "$rrc" = 1 ] && [ ! -e "$T/ranB" ]'
    sk release s >/dev/null
    check "skill release appends to the shared history" 'grep -q "released by s\] holder=s" "$T/lock.history"'

    echo "== cross stale/expiry =="
    sleeper; p=$LAST
    as /wt/r acquire --pid "$p" "crashed" >/dev/null
    kill "$p"; wait "$p" 2>/dev/null
    out=$(sk acquire s2 p 5 1 fragen 2>&1); rc=$?
    check "repo record with dead pid: skill acquire takes it (exit $rc)" \
        '[ "$rc" = 0 ] && grep -q "^holder=s2$" "$T/lock" && ! grep -q "^pid=" "$T/lock"'
    sk release s2 >/dev/null
    sk acquire s3 p 0 1 fragen >/dev/null
    sleeper
    out=$(as /wt/r acquire --pid "$LAST" "after expiry" 2>&1); rc=$?
    check "skill record with expected_minutes=0: repo acquire passes (exit $rc)" \
        '[ "$rc" = 0 ] && grep -q "^worktree=/wt/r$" "$T/lock" && grep -q "cleared stale" <<<"$out"'
    as /wt/r release >/dev/null

    echo "== legacy $IMP_GPU_LOCK_LEGACY: live pid blocks both, file never written =="
    sleeper
    printf 'pid=%s\nstart=%s\nkeeper=0\nworktree=/wt/old\npurpose=old checkout\nsince=x\n' \
        "$LAST" "$(startof "$LAST")" > "$IMP_GPU_LOCK_LEGACY"
    sum0=$(cksum < "$IMP_GPU_LOCK_LEGACY")
    as /wt/r acquire --pid "$LAST" "new" 2>/dev/null; arc=$?
    as /wt/r check 2>/dev/null; crc=$?
    out=$(sk acquire s p 5 1 fragen 2>&1); src=$?
    check "repo acquire 1, repo check 1, skill acquire 1 (exit $arc/$crc/$src)" \
        '[ "$arc" = 1 ] && [ "$crc" = 1 ] && [ "$src" = 1 ] && grep -q "legacy" <<<"$out" && [ ! -e "$T/lock" ]'
    check "status names the legacy holder" '[[ "$(bash "$LOCK_SH" status)" == "held (legacy "*"worktree=/wt/old"* ]]'
    check "legacy file unchanged" '[ "$(cksum < "$IMP_GPU_LOCK_LEGACY")" = "$sum0" ]'
    deadpid
    printf 'pid=%s\nstart=1\nworktree=/wt/old\n' "$DEAD" > "$IMP_GPU_LOCK_LEGACY"
    sleeper
    check "legacy with dead pid does not block" 'as /wt/r acquire --pid "$LAST" x >/dev/null'
    as /wt/r release >/dev/null; rm -f "$IMP_GPU_LOCK_LEGACY"

    echo "== mutex: 20 parallel acquires, 10 per script, exactly one wins (5 rounds) =="
    rounds_ok=0
    for round in 1 2 3 4 5; do
        rm -f "$T/lock" "$T/go"; jobs_=()
        gate() { while [ ! -e "$T/go" ]; do sleep 0.01; done; "$@"; }  # start all 20 together
        for i in $(seq 1 10); do
            sleeper
            gate as "/wt/m$i" acquire --pid "$LAST" "race $i" >/dev/null 2>&1 & jobs_+=("$!")
            gate sk acquire "m$i" "race $i" 5 1 fragen >/dev/null 2>&1 & jobs_+=("$!")
        done
        sleep 0.2; touch "$T/go"
        zero=0 one=0
        for j in "${jobs_[@]}"; do wait "$j"; case $? in 0) zero=$((zero + 1)) ;; 1) one=$((one + 1)) ;; esac; done
        echo "    round $round: exit0=$zero exit1=$one holder=$(sed -n 's/^holder=//p' "$T/lock")"
        [ "$zero" = 1 ] && [ "$one" = 19 ] && rounds_ok=$((rounds_ok + 1))
        as /wt/x release --force >/dev/null 2>&1
    done
    check "5/5 rounds: 1 acquire exit 0, 19 exit 1 ($rounds_ok/5)" '[ "$rounds_ok" = 5 ]'
fi

echo "gpu_lock: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
