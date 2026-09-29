#!/usr/bin/env bash
# GPU acceptance for #2201: fill-in-the-middle via POST /infill and the /v1/completions `suffix`.
# Prints PASS/FAIL per criterion, exit 0 only if every criterion passes. Needs a GPU and the
# worktree image (make build).
# Usage: bash scripts/accept_2201.sh
# Env: IMP_ACCEPT_FIM_MODEL (default Qwen3-Coder-30B-A3B-Instruct-FP4, FIM-trained),
#      IMP_ACCEPT_NOFIM_MODEL (default Llama-3.2-3B-Instruct-Q8_0.gguf, no FIM tokens),
#      IMP_MODELS_DIR (default ~/models), IMP_TEST_IMG, IMP_ACCEPT_PORT (default 8201).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Busy check first, then hold the card for the whole run (scripts/gpu_lock.sh).
if [ -z "${IMP_ACCEPT_2201_LOCKED:-}" ]; then
    bash scripts/require_free_gpu.sh "accept_2201" || exit 1
    IMP_ACCEPT_2201_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2201" -- bash "$0" "$@"
fi

FIM_MODEL="${IMP_ACCEPT_FIM_MODEL:-Qwen3-Coder-30B-A3B-Instruct-FP4}"
NOFIM_MODEL="${IMP_ACCEPT_NOFIM_MODEL:-Llama-3.2-3B-Instruct-Q8_0.gguf}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
PORT="${IMP_ACCEPT_PORT:-8201}"
CTR=imp_accept_2201
BASE="http://127.0.0.1:$PORT"

for tool in docker curl jq; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
for m in "$FIM_MODEL" "$NOFIM_MODEL"; do
    [ -e "$MODELS_DIR/$m" ] || { echo "FAIL setup: $MODELS_DIR/$m missing"; exit 1; }
done
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/accept_2201.XXXXXX")"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$TMP"; }
trap cleanup EXIT

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

start_server() {  # start_server <model file or dir under MODELS_DIR>
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -p "$PORT":"$PORT" "$IMG" \
        imp-server --model "/models/$1" --host 0.0.0.0 --port "$PORT" >/dev/null || return 1
    for _ in $(seq 1 300); do
        curl -s "$BASE/health" 2>/dev/null | grep -q '"model_loaded":true' && return 0
        docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || break
        sleep 1
    done
    echo "server did not load $1 within 300 s:"
    docker logs "$CTR" 2>&1 | tail -20
    return 1
}

# A known function with its body missing; the suffix calls it, so the body is unambiguous.
PREFIX=$'def add(a, b):\n    '
SUFFIX=$'\n\n\ndef sub(a, b):\n    return a - b\n\n\nprint(add(1, 2), sub(3, 1))\n'
WANT='return a + b'

# fill_ok <text> <label>: the body is the add, and no FIM marker leaked into it.
fill_ok() {
    if [[ "$1" == *"$WANT"* && "$1" != *"<|file_sep|>"* && "$1" != *"<|fim_"* ]]; then
        verdict "$2" PASS "fill $(jq -Rn --arg s "$1" '$s')"
    else
        verdict "$2" FAIL "fill $(jq -Rn --arg s "$1" '$s') lacks '$WANT' or carries a FIM marker"
    fi
}

echo "== FIM model: $FIM_MODEL =="
if ! start_server "$FIM_MODEL"; then
    for c in C1-infill C2-completions-suffix C3-infill-stream; do verdict "$c" FAIL "server did not start"; done
else
    MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.loaded == true)][0].id')

    # C1: POST /infill, llama.cpp shape, no model field.
    code=$(curl -s -o "$TMP/c1.json" -w '%{http_code}' "$BASE/infill" -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$PREFIX" --arg s "$SUFFIX" \
            '{input_prefix: $p, input_suffix: $s, max_tokens: 48, temperature: 0}')")
    if [ "$code" = 200 ] && [ "$(jq -r '.content == .choices[0].text' "$TMP/c1.json")" = true ]; then
        fill_ok "$(jq -r '.content' "$TMP/c1.json")" C1-infill
    else
        verdict C1-infill FAIL "HTTP $code: $(head -c 400 "$TMP/c1.json")"
    fi

    # C2: /v1/completions with `suffix`.
    code=$(curl -s -o "$TMP/c2.json" -w '%{http_code}' "$BASE/v1/completions" -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg m "$MODEL_ID" --arg p "$PREFIX" --arg s "$SUFFIX" \
            '{model: $m, prompt: $p, suffix: $s, max_tokens: 48, temperature: 0}')")
    if [ "$code" = 200 ]; then
        fill_ok "$(jq -r '.choices[0].text' "$TMP/c2.json")" C2-completions-suffix
    else
        verdict C2-completions-suffix FAIL "HTTP $code: $(head -c 400 "$TMP/c2.json")"
    fi

    # C3: streaming /infill: several chunks, `content` per chunk, [DONE], same fill.
    curl -sN "$BASE/infill" -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$PREFIX" --arg s "$SUFFIX" \
            '{input_prefix: $p, input_suffix: $s, max_tokens: 48, temperature: 0, stream: true}')" >"$TMP/c3.sse"
    grep '^data: {' "$TMP/c3.sse" | sed 's/^data: //' >"$TMP/c3.jsonl"
    n=$(grep -c . "$TMP/c3.jsonl")
    text=$(jq -rj 'select(.choices | length > 0) | .content' "$TMP/c3.jsonl")
    if [ "$n" -ge 3 ] && grep -q '^data: \[DONE\]' "$TMP/c3.sse"; then
        fill_ok "$text" "C3-infill-stream($n chunks)"
    else
        verdict C3-infill-stream FAIL "$n chunks, [DONE] $(grep -c 'DONE' "$TMP/c3.sse"): $(head -c 400 "$TMP/c3.sse")"
    fi
fi

echo "== non-FIM model: $NOFIM_MODEL =="
if ! start_server "$NOFIM_MODEL"; then
    verdict C4-nofim-400 FAIL "server did not start"
else
    MODEL_ID=$(curl -s "$BASE/v1/models" | jq -r '[.data[] | select(.loaded == true)][0].id')
    c_inf=$(curl -s -o "$TMP/c4a.json" -w '%{http_code}' "$BASE/infill" -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$PREFIX" --arg s "$SUFFIX" '{input_prefix: $p, input_suffix: $s}')")
    c_cmp=$(curl -s -o "$TMP/c4b.json" -w '%{http_code}' "$BASE/v1/completions" -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg m "$MODEL_ID" --arg p "$PREFIX" --arg s "$SUFFIX" '{model: $m, prompt: $p, suffix: $s}')")
    k_inf=$(jq -r '.error.code' "$TMP/c4a.json" 2>/dev/null)
    k_cmp=$(jq -r '.error.code' "$TMP/c4b.json" 2>/dev/null)
    if [ "$c_inf" = 400 ] && [ "$c_cmp" = 400 ] && [ "$k_inf" = fim_not_supported ] && [ "$k_cmp" = fim_not_supported ]; then
        verdict C4-nofim-400 PASS "/infill $c_inf $k_inf, /v1/completions+suffix $c_cmp $k_cmp"
    else
        verdict C4-nofim-400 FAIL "/infill $c_inf $k_inf, /v1/completions+suffix $c_cmp $k_cmp"
    fi
fi

echo
echo "== accept_2201 summary (FIM $FIM_MODEL, non-FIM $NOFIM_MODEL, image $IMG) =="
printf '%s\n' "${RESULTS[@]}"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')
[ ${#RESULTS[@]} -eq 4 ] || { echo "FAIL: ${#RESULTS[@]}/4 criteria reported"; exit 1; }
[ "$fails" = 0 ] && { echo "ALL PASS"; exit 0; }
echo "$fails FAIL"
exit 1
