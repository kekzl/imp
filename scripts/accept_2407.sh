#!/usr/bin/env bash
# GPU acceptance for #2407: session_id keeps an agent's turn-2 prefix hit under KV pressure.
# Fixed KV pool; per arm: turn 1, filler prompts totalling 2x the pool, turn 2 (turn 1 + reply +
# new message). C1 pinned arm (session_id): turn-2 cached_tokens >= turn-1 full-block tokens.
# C2 unpinned arm: below that. C3 close: imp_kv_sessions 1 -> 0. Exit 0 only if all pass.
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

arm() {  # arm <base> <session-or-empty>: prints "turn1_full_block_tokens turn2_cached_tokens"
    local base=$1 sid=$2 m1 m2 reply p1 k=0 filled=0
    m1=$(jq -nc --arg t "$(words "$base" $((base + 1500)))" '[{role: "user", content: $t}]')
    chat "$m1" "$sid" "$ADIR/t1.json"
    p1=$(jq -r '.usage.prompt_tokens' "$ADIR/t1.json")
    reply=$(jq -r '.choices[0].message.content' "$ADIR/t1.json")
    while [ "$filled" -lt $((2 * CAP)) ]; do  # filler: distinct ~8k-token prompts, no session
        k=$((k + 1))
        chat "$(jq -nc --arg t "$(words $((base + 100000 * k)) $((base + 100000 * k + 4000)))" \
            '[{role: "user", content: $t}]')" "" "$ADIR/f.json"
        filled=$((filled + $(jq -r '.usage.prompt_tokens // 0' "$ADIR/f.json")))
        [ "$k" -lt 400 ] || break
    done
    m2=$(jq -c --arg r "$reply" '. + [{role: "assistant", content: $r}, {role: "user", content: "Next."}]' <<<"$m1")
    chat "$m2" "$sid" "$ADIR/t2.json"
    echo "$(((p1 / BS) * BS)) $(jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$ADIR/t2.json")"
}

read -r need_u got_u < <(arm 1 "")
read -r need_p got_p < <(arm 5000000 "accept-2407")
if [ "$got_p" -ge "$need_p" ] && [ "$need_p" -gt 0 ]; then
    verdict C1-pinned-turn2-hit PASS "cached $got_p >= turn-1 full blocks $need_p"
else
    verdict C1-pinned-turn2-hit FAIL "cached $got_p < turn-1 full blocks $need_p"
fi
if [ "$got_u" -lt "$need_u" ]; then
    verdict C2-unpinned-lower PASS "cached $got_u < $need_u (filler evicted it)"
else
    verdict C2-unpinned-lower FAIL "cached $got_u >= $need_u: filler did not evict, pressure not shown"
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
