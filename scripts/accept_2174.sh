#!/usr/bin/env bash
# GPU acceptance for #2174: a hybrid resend restores its snapshot only with the KV of the same forward.
# C1 Qwen3.6-35B-A3B-NVFP4, prefix_resend_probe.py (turn-1 request first, 3 sends), fresh server per
#    round, IMP_ACCEPT_ROUNDS (10) rounds: every round exit 0 (identical + restored), server log
#    shows >= 1 snapshot-chain rebind and >= 1 restored state (the #2174 path ran).
# C2 test-e2e hybrid filter of `make test-e2e` (*HybridRestoreChainTest* + hybrid restore tests) on
#    Qwen3.5-4B-mxfp4: exit 0, 0 failed, 0 skipped.
# INFO Qwen3.8-Flash-Next-NVFP4, same probe, IMP_ACCEPT_FLASH_ROUNDS (3) rounds (issue: a second,
#    unlocalized difference remains there).
# Usage: bash scripts/accept_2174.sh (make build first, or unset IMP_ACCEPT_SKIP_BUILD). Exit 0 iff all PASS.
# Env: IMP_MODELS_DIR (~/models), IMP_ACCEPT_PORT (18174), IMP_ACCEPT_ROUNDS, IMP_ACCEPT_FLASH_ROUNDS,
#      IMP_ACCEPT_SKIP_BUILD=1, IMP_ACCEPT_BUSY_CHECK (host busy-check path), IMP_ACCEPT_PY_IMG (python:3.12-slim).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
MODEL="Qwen3.6-35B-A3B-NVFP4"
FLASH="Qwen3.8-Flash-Next-NVFP4"
GDN="Qwen3.5-4B-mxfp4.gguf"
ROUNDS="${IMP_ACCEPT_ROUNDS:-10}"
FLASH_ROUNDS="${IMP_ACCEPT_FLASH_ROUNDS:-3}"
PORT="${IMP_ACCEPT_PORT:-18174}"
PY_IMG="${IMP_ACCEPT_PY_IMG:-python:3.12-slim}"
BUSY="${IMP_ACCEPT_BUSY_CHECK:-$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh}"
CTR=imp_accept_2174
URL="http://127.0.0.1:$PORT"

for tool in docker curl jq timeout; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
for m in "$MODEL" "$FLASH" "$GDN"; do
    [ -e "$MODELS_DIR/$m" ] || { echo "FAIL setup: $MODELS_DIR/$m missing"; exit 1; }
done

gpu_free() {
    if [ -x "$BUSY" ] && ! "$BUSY"; then
        echo "FAIL setup: $BUSY reports the card busy; not started"
        return 1
    fi
    bash scripts/require_free_gpu.sh accept_2174 || { echo "FAIL setup: require_free_gpu.sh refused"; return 1; }
}

if [ -z "${IMP_ACCEPT_2174_LOCKED:-}" ]; then
    gpu_free || exit 1
    if [ "${IMP_ACCEPT_SKIP_BUILD:-0}" != 1 ]; then
        make build || { echo "FAIL setup: make build"; exit 1; }
    fi
    IMP_ACCEPT_2174_LOCKED=1 exec bash scripts/gpu_lock.sh run accept_2174 -- bash "$ROOT/scripts/accept_2174.sh" "$@"
fi

gpu_free || exit 1
IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/accept_2174.XXXXXX")"
echo "logs: $WORK"
# docker rm -f can return before the name is free (Flash-Next teardown): wait until it is gone.
rm_ctr() {
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    for s in $(seq 0 120); do
        if [ -z "$(docker ps -aq --filter "name=^/$CTR\$")" ]; then
            [ "$s" -gt 0 ] && echo "  container $CTR gone after $s s"
            return 0
        fi
        sleep 1
    done
    echo "  container $CTR still present after 120 s"; return 1
}
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { rm_ctr >/dev/null 2>&1 || true; }
trap cleanup EXIT

FAILS=0
pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }

start_server() {  # start_server <model in MODELS_DIR>: deterministic, FP16 KV, debug log; 1 on timeout
    rm_ctr || return 1
    docker run -d --name "$CTR" --gpus all -p "127.0.0.1:$PORT:8080" -v "$MODELS_DIR:/models:ro" \
        -e IMP_DETERMINISTIC=1 "$IMG" imp-server --host 0.0.0.0 --port 8080 --model "/models/$1" \
        --set runtime.deterministic=true --set kv_cache.dtype=fp16 --set diagnostics.log_level=debug \
        >/dev/null || { echo "  docker run $1 failed"; return 1; }
    for _ in $(seq 1 300); do
        [ "$(curl -s -m 5 "$URL/health" | jq -r '.model_loaded // false' 2>/dev/null)" = true ] && return 0
        sleep 2
    done
    echo "  $1 not healthy after 600 s"; docker logs --tail 30 "$CTR" 2>&1
    return 1
}

probe_round() {  # probe_round <model> <label> <k>: 0 iff probe exit 0, >= 1 rebind, >= 1 restore
    local out="$WORK/$2_$3" rc rebinds restores
    start_server "$1" > "$out.start.log" 2>&1 || { cat "$out.start.log"; return 1; }
    timeout 1800 docker run --rm --network host -v "$ROOT/tools/analysis/prefix_resend:/probe:ro" "$PY_IMG" \
        python /probe/prefix_resend_probe.py --url "$URL" --sends 3 > "$out.probe.log" 2>&1
    rc=$?
    docker logs "$CTR" > "$out.server.log" 2>&1
    rebinds="$(grep -c 'snapshot chain of seq .* rebinds block' "$out.server.log")"
    restores="$(grep -c 'RecurrentSnapshot: restored' "$out.server.log")"
    echo "  $2 round $3: probe exit $rc, rebinds $rebinds, restores $restores; $(grep -E '^(PASS|FAIL) ' "$out.probe.log" | paste -sd ';' -)"
    [ "$rc" -eq 0 ] && [ "$rebinds" -ge 1 ] && [ "$restores" -ge 1 ]
}

# ---- C1: the issue's reproducer on the model it names ----
ok=0
for k in $(seq 1 "$ROUNDS"); do
    probe_round "$MODEL" qwen36 "$k" && ok=$((ok + 1))
done
if [ "$ok" -eq "$ROUNDS" ]; then pass "C1 resend $MODEL: $ok/$ROUNDS rounds identical + restored + rebind seen"
else fail "C1 resend $MODEL: $ok/$ROUNDS rounds ($WORK/qwen36_*)"; fi

# ---- INFO: Flash-Next ----
ok=0
for k in $(seq 1 "$FLASH_ROUNDS"); do
    probe_round "$FLASH" flash "$k" && ok=$((ok + 1))
done
echo "INFO flash $FLASH: $ok/$FLASH_ROUNDS rounds identical + restored + rebind seen"
rm_ctr || true

# ---- C2: hybrid restore suites (the hybrid container of `make test-e2e`) ----
# HybridBatchedDecodeTest.* excluded: pre-existing IMA on main, #2275.
timeout 3600 docker run --rm --gpus all -v "$MODELS_DIR:/models:ro" \
    -e IMP_TEST_MODEL="/models/$GDN" -e IMP_TEST_MODEL_GDN="/models/$GDN" "$IMG" test-e2e \
    --gtest_filter="PrefixCacheE2ETest.HybridSnapshotRestoreMatchesFresh:PrefixCacheE2ETest.HybridTranscriptRestoreContinuesAtReplyEnd:GdnGraphBucketTest.*:*HybridRestoreChainTest*" \
    > "$WORK/c2.log" 2>&1
rc=$?
nok="$(grep -c '^\[       OK \]' "$WORK/c2.log")"
nchain="$(grep -c '^\[       OK \] .*HybridRestoreChainTest' "$WORK/c2.log")"
nskip="$(grep -c '^\[  SKIPPED \]' "$WORK/c2.log")"
nfail="$(grep -c '^\[  FAILED  \] .*(' "$WORK/c2.log")"
if [ "$rc" -eq 0 ] && [ "$nchain" -ge 1 ] && [ "$nskip" -eq 0 ] && [ "$nfail" -eq 0 ]; then
    pass "C2 test-e2e hybrid: exit 0, $nok OK ($nchain HybridRestoreChainTest), 0 skipped"
else
    fail "C2 test-e2e hybrid: exit $rc, $nok OK ($nchain HybridRestoreChainTest), $nskip skipped, $nfail failed ($WORK/c2.log)"
fi

echo "checks failed: $FAILS"
[ "$FAILS" -eq 0 ]
