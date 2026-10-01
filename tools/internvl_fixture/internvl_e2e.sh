#!/usr/bin/env bash
# InternVL3.5-2B end to end: imp-cli --image and imp-server name the bus, the cat and the pizza
# (~/models/gemma-3-4b-vl/test_{bus,cat,pizza}.jpg), then tools/analysis/vision_sight_check.py
# against the server (python:3.12-slim, host network). GPU. Exit 0 only if every check passes.
#   tools/internvl_fixture/internvl_e2e.sh [IMAGE]   (default imp:test)
# Env: IMP_MODELS_DIR (default $HOME/models), IMP_E2E_PORT (default 8095), IMP_E2E_LOGDIR (keep raw outputs).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
IMG="${1:-imp:test}"
MODELS="${IMP_MODELS_DIR:-$HOME/models}"
MODEL=InternVL3_5-2B-HF
PICS="$MODELS/gemma-3-4b-vl"
PORT="${IMP_E2E_PORT:-8095}"
PROMPT="What is shown in this image? Answer in one short sentence."
WORK="$(mktemp -d)"
CTR="imp-internvl-e2e-$$"
cleanup() {
    docker rm -f "$CTR" >/dev/null 2>&1 || true
    [ -n "${IMP_E2E_LOGDIR:-}" ] && mkdir -p "$IMP_E2E_LOGDIR" && cp "$WORK"/*.out "$WORK"/*.err "$WORK"/*.json "$WORK"/sight.log "$IMP_E2E_LOGDIR"/ 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT
rc=0

for pair in bus:bus cat:cat pizza:pizza; do
    name=${pair%%:*} want=${pair##*:}
    docker run --rm --gpus all -v "$MODELS":/models:ro -v "$PICS":/pics:ro "$IMG" \
        imp-cli --model "/models/$MODEL" --image "/pics/test_$name.jpg" --prompt "$PROMPT" \
        --max-tokens 40 --temperature 0 >"$WORK/cli_$name.out" 2>"$WORK/cli_$name.err"
    ans=$(tr '\n' ' ' <"$WORK/cli_$name.out" | grep -oiE "[A-Za-z ,'-]*\b${want}(es|s)?\b[^.[]*\.?" | head -1)
    echo "CLI $name: ${ans:-<none>}"
    [ -n "$ans" ] || { echo "FAIL cli $name"; tail -3 "$WORK/cli_$name.err"; rc=1; }
done
grep -hE "InternVL: [0-9]+ image|InternVL vision tower|Vision tower \(InternVL\)" "$WORK"/cli_cat.err | head -3

docker run -d --name "$CTR" --gpus all -v "$MODELS":/models:ro -p "$PORT:$PORT" "$IMG" \
    imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" >/dev/null
for _ in $(seq 1 90); do
    curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"model_loaded":true' && break
    sleep 2
done
for pair in bus:bus cat:cat pizza:pizza; do
    name=${pair%%:*} want=${pair##*:}
    printf 'data:image/jpeg;base64,%s' "$(base64 -w0 "$PICS/test_$name.jpg")" >"$WORK/u.txt"
    jq -Rs . "$WORK/u.txt" >"$WORK/u.json"
    jq -n --slurpfile u "$WORK/u.json" --arg p "$PROMPT" --arg m "$MODEL" '{model: $m, max_tokens: 40,
        temperature: 0, messages: [{role: "user", content: [{type: "image_url", image_url: {url: $u[0]}},
        {type: "text", text: $p}]}]}' >"$WORK/req.json"
    curl -s "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        --data-binary @"$WORK/req.json" >"$WORK/resp_$name.json"
    ans=$(jq -r '.choices[0].message.content // .error.message // "no response"' "$WORK/resp_$name.json")
    echo "SERVER $name: $ans"
    grep -qiE "\b${want}(es|s)?\b" <<<"$ans" || { echo "FAIL server $name"; rc=1; }
done
docker logs "$CTR" 2>&1 | grep -E "InternVL: [0-9]+ image" | tail -1

docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$TREE":"$TREE":ro -w "$TREE" \
    python:3.12-slim python3 tools/analysis/vision_sight_check.py --url "http://127.0.0.1:$PORT" --model "$MODEL" \
    >"$WORK/sight.log" 2>&1
sight=$?
tail -6 "$WORK/sight.log"
echo "vision_sight_check exit $sight"
[ "$sight" = 0 ] || rc=1
[ "$rc" = 0 ] && echo "PASS internvl e2e: cli + server + sight check"
exit "$rc"
