#!/usr/bin/env bash
# GPU acceptance for #2206: Responses API store=true / previous_response_id.
# Prints PASS/FAIL per criterion, exit 0 only if every criterion passes. Needs a GPU, the
# worktree image (make build) and the model under IMP_MODELS_DIR.
# Usage: bash scripts/accept_2206.sh
# Env: IMP_ACCEPT_MODEL (default Qwen3-8B-Q8_0.gguf), IMP_MODELS_DIR (default ~/models),
#      IMP_TEST_IMG, IMP_ACCEPT_PORT (default 8206).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Busy check first, then hold the card for the whole run (scripts/gpu_lock.sh).
if [ -z "${IMP_ACCEPT_2206_LOCKED:-}" ]; then
    HOST_CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
    if [ -x "$HOST_CHECK" ]; then
        "$HOST_CHECK" || { echo "FAIL setup: GPU busy ($HOST_CHECK)"; exit 1; }
    fi
    bash scripts/require_free_gpu.sh "accept_2206" || exit 1
    IMP_ACCEPT_2206_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2206" -- bash "$0" "$@"
fi

MODEL="${IMP_ACCEPT_MODEL:-Qwen3-8B-Q8_0.gguf}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_ACCEPT_PORT:-8206}"
TTL=10
CTR=imp_accept_2206
BASE="http://127.0.0.1:$PORT"

for tool in docker curl jq; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
[ -e "$MODELS_DIR/$MODEL" ] || { echo "FAIL setup: $MODELS_DIR/$MODEL missing"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

ADIR="$(mktemp -d "${TMPDIR:-/tmp}/accept_2206.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$ADIR"; }
trap cleanup EXIT

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

start_server() {  # start_server <extra imp-server args...>
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT":"$PORT" "$IMG" \
        imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" "$@" >/dev/null || return 1
    local i
    for i in $(seq 1 180); do
        curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && return 0
        docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || break
        sleep 1
    done
    echo "server did not load within 180 s:"
    docker logs "$CTR" 2>&1 | tail -20
    return 1
}

MODEL_ID=""
post() {  # post <json-body> <out-file> -> HTTP code on stdout
    curl -s -o "$2" -w '%{http_code}' -H 'Content-Type: application/json' -d "$1" "$BASE/v1/responses"
}
req() {  # req <input-json> [extra jq object] -> request body
    jq -nc --arg m "$MODEL_ID" --argjson in "$1" --argjson x "${2:-{\}}" \
        '{model: $m, input: $in, temperature: 0, max_output_tokens: 64,
          reasoning: {effort: "none"}} + $x'
}
text_of() { jq -r '[.output[] | select(.type == "message") | .content[] | select(.type == "output_text") | .text] | join("")' "$1"; }
metric() { curl -s "$BASE/metrics" | awk -v n="$1" '$1 == n { print $2 }'; }

# ================= Phase A: default store limits =================
if ! start_server; then
    verdict C1-two-turn-identity FAIL "server did not start"
    verdict C2-get-stored FAIL "server did not start"
    verdict C3-stream-stored FAIL "server did not start"
else
    MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.loaded == true)][0].id')
    Q1='"Name three primary colors. One line."'
    Q2='[{"role": "user", "content": "Now name one secondary color. One word."}]'

    c1=$(post "$(req "$Q1" '{"store": true}')" "$ADIR/t1.json")
    id1=$(jq -r '.id' "$ADIR/t1.json")
    c2=$(post "$(req "$Q2" "$(jq -nc --arg p "$id1" '{previous_response_id: $p}')")" "$ADIR/t2.json")
    # Stateless: the same conversation resent in full, store=false.
    full=$(jq -c --argjson q1 "$Q1" --argjson q2 "$Q2" \
        '[{role: "user", content: $q1}] + .output + $q2' "$ADIR/t1.json")
    c3=$(post "$(req "$full" '{"store": false}')" "$ADIR/s2.json")
    t2=$(text_of "$ADIR/t2.json")
    s2=$(text_of "$ADIR/s2.json")
    u2=$(jq -c '[.usage.input_tokens, .usage.output_tokens]' "$ADIR/t2.json")
    us=$(jq -c '[.usage.input_tokens, .usage.output_tokens]' "$ADIR/s2.json")
    echo "turn2 stateful : $u2 $(printf '%q' "$t2")"
    echo "turn2 stateless: $us $(printf '%q' "$s2")"
    if [ "$c1$c2$c3" = 200200200 ] && [ -n "$t2" ] && [ "$t2" = "$s2" ] && [ "$u2" = "$us" ]; then
        verdict C1-two-turn-identity PASS "text and [input,output] tokens $u2 identical"
    else
        verdict C1-two-turn-identity FAIL "HTTP $c1/$c2/$c3, usage $u2 vs $us, texts differ or empty"
    fi

    g=$(curl -s -o "$ADIR/g1.json" -w '%{http_code}' "$BASE/v1/responses/$id1")
    if [ "$g" = 200 ] && [ "$(jq -c .output "$ADIR/g1.json")" = "$(jq -c .output "$ADIR/t1.json")" ]; then
        verdict C2-get-stored PASS "GET $id1 returns the turn-1 output"
    else
        verdict C2-get-stored FAIL "GET $id1 -> HTTP $g or output differs"
    fi

    curl -s -N -H 'Content-Type: application/json' \
        -d "$(req "$Q1" '{"store": true, "stream": true}')" "$BASE/v1/responses" >"$ADIR/st.sse"
    sid=$(grep '^data: ' "$ADIR/st.sse" | sed 's/^data: //' |
        jq -r 'select(.type == "response.completed") | .response.id' | head -1)
    gs=$(curl -s -o "$ADIR/gs.json" -w '%{http_code}' "$BASE/v1/responses/${sid:-none}")
    cs=$(post "$(req "$Q2" "$(jq -nc --arg p "${sid:-none}" '{previous_response_id: $p}')")" "$ADIR/cs.json")
    if [ -n "$sid" ] && [ "$gs" = 200 ] && [ "$cs" = 200 ]; then
        verdict C3-stream-stored PASS "streamed $sid stored (GET 200), continuation 200"
    else
        verdict C3-stream-stored FAIL "stream id '${sid}', GET $gs, continuation $cs"
    fi
fi

# ================= Phase B: --responses-store-ttl 10, --responses-store-max-entries 2 =================
if ! start_server --responses-store-ttl "$TTL" --responses-store-max-entries 2; then
    verdict C4-eviction-counter FAIL "server did not start"
    verdict C5-expiry-404 FAIL "server did not start"
else
    MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.loaded == true)][0].id')
    ev0=$(metric imp_responses_store_evictions_total)
    ids=()
    for i in 1 2 3; do
        post "$(req "\"Say $i.\"" '{"store": true, "max_output_tokens": 4}')" "$ADIR/e$i.json" >/dev/null
        ids+=("$(jq -r .id "$ADIR/e$i.json")")
    done
    ev1=$(metric imp_responses_store_evictions_total)
    n=$(metric imp_responses_store_entries)
    g1=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/responses/${ids[0]}")
    g3=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/responses/${ids[2]}")
    if [ "$ev0" = 0 ] && [ "$ev1" = 1 ] && [ "$n" = 2 ] && [ "$g1" = 404 ] && [ "$g3" = 200 ]; then
        verdict C4-eviction-counter PASS "evictions_total $ev0 -> $ev1, entries $n, oldest 404, newest 200"
    else
        verdict C4-eviction-counter FAIL "evictions_total $ev0 -> $ev1, entries $n, oldest $g1, newest $g3"
    fi

    sleep $((TTL + 2))
    ce=$(post "$(req '"Continue."' "$(jq -nc --arg p "${ids[2]}" '{previous_response_id: $p}')")" "$ADIR/x.json")
    code=$(jq -r '.error.code // empty' "$ADIR/x.json")
    exp=$(metric imp_responses_store_expired_total)
    if [ "$ce" = 404 ] && [ "$code" = response_not_found ] && [ "${exp:-0}" -ge 1 ]; then
        verdict C5-expiry-404 PASS "after $((TTL + 2)) s: HTTP 404 $code, expired_total $exp"
    else
        verdict C5-expiry-404 FAIL "after $((TTL + 2)) s: HTTP $ce code '$code', expired_total '$exp'"
    fi
fi

echo "== accept_2206 summary ($MODEL, image $IMG) =="
printf '%s\n' "${RESULTS[@]}"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')
[ ${#RESULTS[@]} -eq 5 ] || { echo "FAIL: ${#RESULTS[@]}/5 criteria reported"; exit 1; }
[ "$fails" = 0 ] && { echo "ALL PASS"; exit 0; }
echo "$fails FAILED"
exit 1
