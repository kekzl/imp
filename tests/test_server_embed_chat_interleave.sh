#!/bin/bash
# MANUAL tool — not wired into ctest/CI/verify.sh (like test_server_concurrent.sh).
# Needs a running imp-server on :8080 with a model that supports /v1/embeddings.
#
# Regression for the "0 completion tokens" wedge (DEBUG-0tokens.md): the
# /v1/embeddings handler used to stop() the batching engine for exclusive
# C-API access, which CANCELLED every in-flight generation. Under interleaved
# embed+chat load each chat then returned an empty `finish_reason:"cancelled"`
# completion. The fix drains in-flight work (pause/resume) instead of cancelling.
#
# This test hammers chat and embeddings CONCURRENTLY and asserts that no chat
# completion comes back empty or cancelled.
set -u

BASE="http://localhost:8080"
MODEL="${IMP_TEST_MODEL:-Qwen3-8B-NVFP4-cortecs}"
DURATION="${1:-30}"

# A prompt long enough that generation overlaps incoming embeddings calls.
PROMPT="Write a detailed multi-paragraph essay about the history of cartography."

RESULT_DIR="$(mktemp -d)"
EMPTY_FLAG="$RESULT_DIR/empty"
STOP_FLAG="$RESULT_DIR/stop"

embed_loop() {
  while [ ! -f "$STOP_FLAG" ]; do
    curl -s "$BASE/v1/embeddings" -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL\",\"input\":\"the quick brown fox jumps over the lazy dog repeatedly\"}" >/dev/null 2>&1
  done
}

chat_loop() {
  local n=0
  while [ ! -f "$STOP_FLAG" ]; do
    n=$((n + 1))
    resp=$(curl -s "$BASE/v1/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":64,\"temperature\":0.7}")
    content=$(echo "$resp" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
    finish=$(echo "$resp" | jq -r '.choices[0].finish_reason // empty' 2>/dev/null)
    reasoning=$(echo "$resp" | jq -r '.choices[0].message.reasoning_content // empty' 2>/dev/null)
    if { [ -z "$content" ] && [ -z "$reasoning" ]; } || [ "$finish" = "cancelled" ]; then
      echo "FAIL: chat #$n empty/cancelled (finish=$finish): $resp" | head -c 400
      echo
      touch "$EMPTY_FLAG"
      touch "$STOP_FLAG"
      return
    fi
  done
  echo "chat issued $n requests, all non-empty"
}

echo "=== interleaved embed+chat for ${DURATION}s (model=$MODEL) ==="
embed_loop & E1=$!
embed_loop & E2=$!
chat_loop  & C1=$!

sleep "$DURATION"
touch "$STOP_FLAG"
wait "$C1" 2>/dev/null
kill "$E1" "$E2" 2>/dev/null
wait "$E1" "$E2" 2>/dev/null

if [ -f "$EMPTY_FLAG" ]; then
  echo "FAIL: a chat completion was cancelled/empty during interleaved embeddings load"
  rm -rf "$RESULT_DIR"
  exit 1
fi
echo "PASS: all interleaved chat completions stayed non-empty"
rm -rf "$RESULT_DIR"
exit 0
