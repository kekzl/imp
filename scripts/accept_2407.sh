#!/usr/bin/env bash
# GPU acceptance for #2407: session_id keeps an agent's turn-2 prefix hit under KV pressure.
# Fixed KV pool; per arm: turn 1, filler prompts totalling 2x the pool, turn 2 (turn 1 + reply +
# new message). Control arm: same prompt shape, no filler = the turn-2 hit without pressure.
# C1 pinned arm (session_id): turn-2 cached_tokens >= control. C2 unpinned arm: below control.
# C3 close: imp_kv_sessions 1 -> 0. Exit 0 only if all pass.
# Usage: bash scripts/accept_2407.sh
# Env: IMP_ACCEPT_MODEL (default Qwen3-8B-Q8_0.gguf), IMP_MODELS_DIR (default ~/models),
#      IMP_TEST_IMG, IMP_ACCEPT_PORT (default 8407).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Busy check first, then hold the card for the whole run (scripts/gpu_lock.sh).
if [ -z "${IMP_ACCEPT_2407_LOCKED:-}" ]; then
    HOST_CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
    if [ -x "$HOST_CHECK" ]; then
        "$HOST_CHECK" || { echo "FAIL setup: GPU busy ($HOST_CHECK)"; exit 1; }
    fi
    bash scripts/require_free_gpu.sh "accept_2407" || exit 1
    IMP_ACCEPT_2407_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2407" -- bash "$0" "$@"
fi

MODEL="${IMP_ACCEPT_MODEL:-Qwen3-8B-Q8_0.gguf}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_ACCEPT_PORT:-8407}"
CTR=imp_accept_2407
BASE="http://127.0.0.1:$PORT"

for tool in docker curl jq; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
[ -e "$MODELS_DIR/$MODEL" ] || { echo "FAIL setup: $MODELS_DIR/$MODEL missing"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

ADIR="$(mktemp -d "${TMPDIR:-/tmp}/accept_2407.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$ADIR"; }
trap cleanup EXIT

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}
metric() { curl -s "$BASE/metrics" | awk -v n="$1" '$1 == n { print $2 }'; }
words() { seq -s ' ' "$1" "$2"; }  # distinct text per range

docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT":"$PORT" "$IMG" \
    imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" \
    --set kv_cache.growable=false --set runtime.max_seq_len=16384 >/dev/null || exit 1
for _ in $(seq 1 180); do
    curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && break
    sleep 1
done
CAP=$(curl -s "$BASE/health" | jq -r '.kv_capacity_tokens // empty')
BS=$(curl -s "$BASE/health" | jq -r '.kv_block_size // empty')
[ -n "$CAP" ] && [ -n "$BS" ] || { echo "FAIL setup: server not ready"; docker logs "$CTR" 2>&1 | tail -20; exit 1; }
MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.loaded == true)][0].id')
echo "pool $CAP tokens, block $BS"

chat() {  # chat <messages-json> <session-or-empty> <out>
    jq -nc --arg m "$MODEL_ID" --argjson msgs "$1" --arg s "$2" \
        '{model: $m, messages: $msgs, max_tokens: 8, temperature: 0} + (if $s == "" then {} else {session_id: $s} end)' |
        curl -s -o "$3" -H 'Content-Type: application/json' -d @- "$BASE/v1/chat/completions"
}

# arm <base> <session-or-empty> <fill 0|1>: prints "turn1_full_block_tokens turn2_cached_tokens".
# Every base is 7 digits: all arms render the same token count (Qwen3: 12017).
arm() {
    local base=$1 sid=$2 fill=$3 m1 m2 reply p1 pf k=0 filled=0
    m1=$(jq -nc --arg t "$(words "$base" $((base + 1500)))" '[{role: "user", content: $t}]')
    chat "$m1" "$sid" "$ADIR/t1.json"
    p1=$(jq -r '.usage.prompt_tokens' "$ADIR/t1.json")
    reply=$(jq -r '.choices[0].message.content' "$ADIR/t1.json")
    # Filler: distinct ~12k-token prompts (Qwen3 splits digits: 1501 numbers of 7 digits),
    # under max_seq_len 16384, no session. A rejected filler aborts: no pressure, no verdict.
    while [ "$fill" = 1 ] && [ "$filled" -lt $((2 * CAP)) ]; do
        k=$((k + 1))
        chat "$(jq -nc --arg t "$(words $((base + 100000 * k)) $((base + 100000 * k + 1500)))" \
            '[{role: "user", content: $t}]')" "" "$ADIR/f.json"
        pf=$(jq -r '.usage.prompt_tokens // 0' "$ADIR/f.json")
        [ "$pf" -gt 0 ] || { echo "FAIL setup: filler $k rejected: $(head -c 300 "$ADIR/f.json")" >&2; exit 1; }
        filled=$((filled + pf))
    done
    [ "$fill" = 0 ] || echo "arm '$sid': $k fillers, $filled tokens" >&2
    # Turn-2 message >= 33 tokens past turn 1: reuse leaves kMinPromptChunkRows (33) rows to
    # re-prefill (src/runtime/prompt_tail.h), so a shorter tail caps the hit below turn 1.
    m2=$(jq -c --arg r "$reply" --arg n "Next. $(words 1 40)" \
        '. + [{role: "assistant", content: $r}, {role: "user", content: $n}]' <<<"$m1")
    chat "$m2" "$sid" "$ADIR/t2.json"
    echo "$(((p1 / BS) * BS)) $(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$ADIR/t2.json")"
}

# Control, not turn-1 full blocks: a thinking prompt ends in <think>\n (2 tokens,
# tools/imp-server/handlers_chat_core.cpp) that turn 2 does not render, so the shared prefix is shorter.
out_c=$(arm 8000000 "" 0) || exit 1
out_u=$(arm 3000000 "" 1) || exit 1
out_p=$(arm 5000000 "accept-2407" 1) || exit 1
read -r need_c ctrl <<<"$out_c"
read -r _ got_u <<<"$out_u"
read -r _ got_p <<<"$out_p"
echo "control: turn-1 full blocks $need_c, turn-2 cached $ctrl without filler"
[ "$ctrl" -gt 0 ] || { echo "FAIL setup: control turn 2 cached 0, prefix cache not reusing"; exit 1; }
if [ "$got_p" -ge "$ctrl" ]; then
    verdict C1-pinned-turn2-hit PASS "cached $got_p >= control $ctrl (turn-1 full blocks $need_c)"
else
    verdict C1-pinned-turn2-hit FAIL "cached $got_p < control $ctrl (turn-1 full blocks $need_c)"
fi
if [ "$got_u" -lt "$ctrl" ]; then
    verdict C2-unpinned-lower PASS "cached $got_u < control $ctrl (filler evicted it)"
else
    verdict C2-unpinned-lower FAIL "cached $got_u >= control $ctrl: filler did not evict, pressure not shown"
fi
s0=$(metric imp_kv_sessions)
curl -s -X POST "$BASE/v1/sessions/accept-2407/close" >/dev/null
sleep 2
s1=$(metric imp_kv_sessions)
if [ "$s0" = 1 ] && [ "$s1" = 0 ]; then
    verdict C3-close-releases PASS "imp_kv_sessions $s0 -> $s1"
else
    verdict C3-close-releases FAIL "imp_kv_sessions '$s0' -> '$s1'"
fi

echo "== accept_2407 summary ($MODEL, image $IMG) =="
printf '%s\n' "${RESULTS[@]}"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')
[ ${#RESULTS[@]} -eq 3 ] || { echo "FAIL: ${#RESULTS[@]}/3 criteria reported"; exit 1; }
[ "$fails" = 0 ] && { echo "ALL PASS"; exit 0; }
echo "$fails FAILED"
exit 1
