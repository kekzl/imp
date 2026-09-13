#!/usr/bin/env bash
# Stage 3 of the test gate (local, GPU-only): the only place handlers.cpp/batching_engine and
# the OpenAI+Anthropic wire protocols run end to end against a live model. Every battery's exit
# code gates (unlike coverage_server.sh, which runs the same batteries with || true).
# Hard gates: exercise_all_endpoints.py, test_server_robustness.py (#712),
# test_server_0token_battery.py (#710), test_server_embed_chat_interleave.sh,
# test_server_logprobs.py, test_server_ignore_eos.py, test_server_messages_stream.py,
# test_server_vision_and_utf8.py (#1198/#1197), test_server_metrics.py.
# Usage: make test-server (or scripts/test_server.sh). Env: IMP_SRV_MODEL (default
# Qwen3-8B-NVFP4-cortecs), IMP_MODELS_DIR, IMP_SRV_PORT, IMP_TEST_IMG, IMP_SRV_BUILD.
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

# Stage 3a: does the server start with NO capacity flags at all? Every battery below boots
# with four of them, so the config a first-time reader runs was the one nothing covered (#1631).
echo "== default-start gate (#1631) =="
if ! bash scripts/test_server_default_start.sh; then
    echo "test-server: FAIL - imp-server does not start on shipped defaults"
    exit 1
fi

echo "== launch imp-server ($MODEL) =="
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT":"$PORT" \
    --add-host=host.docker.internal:host-gateway "$IMG" \
    imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" \
    --max-concurrent 8 --rate-limit 100000 --max-input-tokens 100000 \
    --set server.otlp_endpoint=http://host.docker.internal:4318/v1/traces >/dev/null

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
run "ignore_eos"          python3 tests/test_server_ignore_eos.py
run "messages stream"     python3 tests/test_server_messages_stream.py
run "thinking toggle"     python3 tests/test_server_thinking_toggle.py
# Budget sweep the toggle test can't see: pins max_tokens 512, the answer-headroom force-close
# only bites below it. multiturn_deep.py had zero invocation sites (#1573 class).
# --assert-answered fails an empty reply unless the server labels it reasoning_budget_exhausted.
run "reasoning budget reaches the answer" python3 tools/analysis/multiturn_deep.py \
    --url "http://localhost:$PORT" --model "$MODEL" --max-tokens 200,260,400,600 \
    --assert-answered
run "tracing (OTLP spans)" python3 tests/test_server_tracing.py
run "metrics (every path feeds the histograms)" python3 tests/test_server_metrics.py
run "vision refusal + utf8 (#1197/#1198)" python3 tests/test_server_vision_and_utf8.py
run "embed/chat interleave" bash tests/test_server_embed_chat_interleave.sh 15
run "0-token battery (#710)" env N=8 LOAD=80 FAIL_THRESHOLD=0.10 python3 tests/test_server_0token_battery.py
# #1573: tools/analysis/degen_suite.py had ZERO invocation sites, 41 checks nothing ever ran.
# Categories, not --corpus (the ~250-prompt corpus belongs in a longer lane): server-protocol
# failure classes the C-API GTests structurally cannot see (think-leak, special tokens, stream
# consistency).
run "degeneration suite (#1573)" python3 tools/analysis/degen_suite.py --url "http://localhost:$PORT"

echo
if [ "${#fails[@]}" -ne 0 ]; then
    echo "test-server: FAIL — ${#fails[@]} batterie(s) regressed: ${fails[*]}"
    exit 1
fi
echo "test-server: PASS — all server batteries green"
