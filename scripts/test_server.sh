#!/usr/bin/env bash
# Stage 3 of the test gate — the SERVER stage (local, GPU-only).
#
# CI has no GPU runner, so the real imp-server (handlers.cpp / batching_engine /
# the OpenAI+Anthropic wire protocols) can only be exercised here, against a
# live model on the 5090. Stage 1 (pre-commit `make test-gpu`) covers GPU kernel
# correctness; Stage 2 (CI `ctest -L unit`) covers the CPU helper units; neither
# boots a server. This script does, and GATES on the result (every battery's
# exit code is propagated — unlike scripts/coverage_server.sh, which runs the
# same batteries with `|| true` purely to measure coverage).
#
# Boots imp-server once, then runs, as hard gates:
#   - exercise_all_endpoints.py        every endpoint/mode returns non-5xx
#   - test_server_robustness.py        bad input -> 4xx+envelope, valid -> 2xx (#712)
#   - test_server_0token_battery.py    no empty-completion wedge under mixed load (#710)
#   - test_server_embed_chat_interleave.sh   embeddings don't cancel in-flight chat
#   - test_server_logprobs.py          logprob sum + top-k descending order
#   - test_server_messages_stream.py   Anthropic /v1/messages event sequence
#   - test_server_vision_and_utf8.py   images refused when unusable (#1198),
#                                     non-ASCII survives json_schema (#1197)
#
# Usage:   make test-server   (or: scripts/test_server.sh)
# Env:     IMP_SRV_MODEL    (default Qwen3-8B-NVFP4-cortecs) — needs chat+tools+embeddings
#          IMP_MODELS_DIR   (default $HOME/models)      — host dir mounted at /models
#          IMP_SRV_PORT     (default 8080)
#          IMP_TEST_IMG     (default imp:test)               — built by `make build`
#          IMP_SRV_BUILD    (set to 1 to force a docker build first)
set -uo pipefail

MODEL="${IMP_SRV_MODEL:-Qwen3-8B-NVFP4-cortecs}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
PORT="${IMP_SRV_PORT:-8080}"
IMG="${IMP_TEST_IMG:-imp:test}"
CTR=imp_test_server
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "test-server: no GPU on this host — the server stage is GPU-only, skipping." >&2
    echo "             Run on the 5090 box before relying on server changes." >&2
    exit 0
fi

if [ "${IMP_SRV_BUILD:-0}" = "1" ] || ! docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "== build $IMG =="
    docker build --build-arg IMP_BUILD_TESTS=ON -t "$IMG" .
fi

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== launch imp-server ($MODEL) =="
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT":"$PORT" "$IMG" \
    imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" \
    --max-concurrent 8 --rate-limit 100000 --max-input-tokens 100000 >/dev/null

ok=0
for i in $(seq 1 90); do
    if curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"model_loaded":true'; then ok=1; break; fi
    if ! docker ps -q --no-trunc | grep -q "$(docker inspect -f '{{.Id}}' "$CTR" 2>/dev/null)"; then
        echo "test-server: server container exited during startup. Logs:" >&2
        docker logs "$CTR" 2>&1 | tail -30 >&2
        exit 1
    fi
    sleep 2
done
if [ "$ok" != "1" ]; then
    echo "test-server: server did not become healthy within 180s. Logs:" >&2
    docker logs "$CTR" 2>&1 | tail -30 >&2
    exit 1
fi

export IMP_BASE="http://localhost:$PORT" IMP_MODEL="$MODEL"
export IMP_HOST=localhost IMP_PORT="$PORT" IMP_TEST_MODEL="$MODEL"

fails=()
run() {  # run <label> <cmd...>
    local label="$1"; shift
    echo; echo "== $label =="
    if "$@"; then echo "   -> PASS ($label)"; else echo "   -> FAIL ($label)"; fails+=("$label"); fi
}

run "endpoints smoke"     python3 tests/exercise_all_endpoints.py
run "robustness (#712)"   python3 tests/test_server_robustness.py
run "logprobs"            python3 tests/test_server_logprobs.py
run "messages stream"     python3 tests/test_server_messages_stream.py
run "thinking toggle"     python3 tests/test_server_thinking_toggle.py
run "vision refusal + utf8 (#1197/#1198)" python3 tests/test_server_vision_and_utf8.py
run "embed/chat interleave" bash tests/test_server_embed_chat_interleave.sh 15
run "0-token battery (#710)" env N=8 LOAD=80 FAIL_THRESHOLD=0.10 python3 tests/test_server_0token_battery.py

echo
if [ "${#fails[@]}" -ne 0 ]; then
    echo "test-server: FAIL — ${#fails[@]} batterie(s) regressed: ${fails[*]}"
    exit 1
fi
echo "test-server: PASS — all server batteries green"
