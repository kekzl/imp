#!/usr/bin/env bash
# Qwen3-VL video end to end: imp-cli and imp-server each answer about the 8 committed red-panda
# frames (tests/fixtures/qwen3vl_video/red_panda). GPU. Exit 0 only if both answers name a red panda.
#   tools/qwen3vl_video_fixture/red_panda_e2e.sh [IMAGE]   (default imp:test)
# Env: IMP_MODELS_DIR (default $HOME/models), IMP_E2E_PORT (default 8093).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/../.." && pwd)"
IMG="${1:-imp:test}"
MODELS="${IMP_MODELS_DIR:-$HOME/models}"
MODEL=Qwen3-VL-4B-Instruct
PORT="${IMP_E2E_PORT:-8093}"
FR="$TREE/tests/fixtures/qwen3vl_video/red_panda"
# Source frame index / 30 fps for indices 0 128 255 383 511 639 766 894.
TS="0,4.266667,8.5,12.766667,17.033333,21.3,25.533333,29.8"
PROMPT="What animal is shown in this video? Answer in one sentence."
WORK="$(mktemp -d)"
CTR="imp-red-panda-e2e-$$"
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

frames=""
for k in 1 2 3 4 5 6 7 8; do frames="${frames:+$frames,}/fx/frame_$k.png"; done

rc=0
docker run --rm --gpus all -v "$MODELS":/models:ro -v "$FR":/fx:ro "$IMG" \
    imp-cli --model "/models/$MODEL" --video-frames "$frames" --video-timestamps "$TS" \
    --prompt "$PROMPT" --max-tokens 48 --temperature 0 >"$WORK/cli.out" 2>"$WORK/cli.err" || rc=1
grep -E "Video loaded|Qwen3-VL: .*video" "$WORK/cli.err" || true
echo "CLI answer: $(tr '\n' ' ' <"$WORK/cli.out")"
grep -qi "red panda" "$WORK/cli.out" || { echo "FAIL cli: no 'red panda'"; tail -5 "$WORK/cli.err"; rc=1; }

docker run -d --name "$CTR" --gpus all -v "$MODELS":/models:ro -p "$PORT:$PORT" "$IMG" \
    imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" >/dev/null
for _ in $(seq 1 90); do
    curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"model_loaded":true' && break
    sleep 2
done
parts="[]"
for k in 1 2 3 4 5 6 7 8; do
    printf 'data:image/png;base64,%s' "$(base64 -w0 "$FR/frame_$k.png")" >"$WORK/f$k.txt"
    parts=$(jq -c --rawfile u "$WORK/f$k.txt" '. + [$u]' <<<"$parts")
done
jq -n --argjson frames "$parts" --arg p "$PROMPT" --arg ts "$TS" '{
    max_tokens: 48, temperature: 0,
    messages: [{role: "user", content: [
        {type: "video", video: $frames, timestamps: ($ts | split(",") | map(tonumber))},
        {type: "text", text: $p}]}]}' >"$WORK/req.json"
curl -s "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary @"$WORK/req.json" >"$WORK/resp.json" || true
docker logs "$CTR" 2>&1 | grep -E "Qwen3-VL: .*video" | tail -1 || true
answer="$(jq -r '.choices[0].message.content // .error.message // "no response"' "$WORK/resp.json")"
echo "SERVER answer: $answer"
grep -qi "red panda" <<<"$answer" || { echo "FAIL server: no 'red panda'"; rc=1; }
[ "$rc" = 0 ] && echo "PASS red-panda video: cli + server"
exit "$rc"
