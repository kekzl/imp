#!/usr/bin/env bash
# Does imp-server start on SHIPPED DEFAULTS? (#1631)
#
# Every other server battery passes flags: scripts/test_server.sh boots with
# --max-concurrent / --rate-limit / --max-input-tokens, and the audit's own
# runtime pass used --set runtime.max_seq_len=8192 --max-batch 4 because the
# defaults did not start. So the one configuration a first-time reader actually
# runs was the one nothing exercised, and it exited 1 with 537 CUDA OOM lines on
# the repo's own perf-baseline model, on an idle 32 GB card.
#
# The gate is deliberately narrow: `imp-server --model <path>` and nothing else,
# then /health, then one generation. A start that reaches /health but cannot
# answer is not a pass.
#
# Usage:   bash scripts/test_server_default_start.sh
# Env:     IMP_DEFAULT_START_MODEL  file name under the models dir
#                                   (default: the perf-baseline model)
#          IMP_MODELS_DIR           host dir mounted at /models (default $HOME/models)
#          IMP_SRV_PORT             (default 8080)
#          IMP_TEST_IMG             (default imp:test)
set -uo pipefail

MODEL="${IMP_DEFAULT_START_MODEL:-Qwen3-8B-Q8_0.gguf}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
PORT="${IMP_SRV_PORT:-8080}"
IMG="${IMP_TEST_IMG:-imp:test}"
CTR=imp_default_start_test
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "default-start: no GPU on this host, skipping (this stage is GPU-only)." >&2
    exit 0
fi

# unset -> skip, set-but-missing -> red. A dangling path that silently skips is
# how make test-e2e reported green while loading no model at all.
if [ ! -r "$MODELS_DIR/$MODEL" ]; then
    if [ -n "${IMP_DEFAULT_START_MODEL:-}" ]; then
        echo "default-start: IMP_DEFAULT_START_MODEL=$MODEL not readable under $MODELS_DIR" >&2
        exit 1
    fi
    echo "default-start: $MODEL not in $MODELS_DIR, skipping." >&2
    exit 0
fi

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker rm -f "$CTR" >/dev/null 2>&1 || true

echo "== imp-server on shipped defaults ($MODEL) =="
# --host/--port only: without them the container's server is unreachable, and
# they are not capacity knobs, which is what this test is about.
docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT":"$PORT" "$IMG" \
    imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" >/dev/null

ok=0
for _ in $(seq 1 90); do
    if curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"model_loaded":true'; then
        ok=1
        break
    fi
    if ! docker ps -q --no-trunc | grep -q "$(docker inspect -f '{{.Id}}' "$CTR" 2>/dev/null)"; then
        echo "default-start: FAIL - the container exited during startup." >&2
        echo "               This is #1631: the planner commits more than the card has left" >&2
        echo "               and the first cuBLASLt call OOMs. Logs:" >&2
        docker logs "$CTR" 2>&1 | tail -25 >&2
        exit 1
    fi
    sleep 2
done
if [ "$ok" != "1" ]; then
    echo "default-start: FAIL - not healthy within 180s. Logs:" >&2
    docker logs "$CTR" 2>&1 | tail -25 >&2
    exit 1
fi

# A start that cannot answer is not a start. The OOM this guards against fires
# on the first forward, which /health alone never triggers.
body=$(curl -s -m 180 "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France.\"}],\"max_tokens\":600}")
if ! grep -q "Paris" <<<"$body"; then
    echo "default-start: FAIL - server is healthy but the first generation did not answer." >&2
    echo "               response: ${body:0:400}" >&2
    docker logs "$CTR" 2>&1 | tail -25 >&2
    exit 1
fi

# grep -q exits on the first match, the producer dies on EPIPE, and pipefail
# turns a HIT into a non-zero pipeline (#1499). Feed it from a variable.
oom=$(docker logs "$CTR" 2>&1 | grep -c "out of memory")
if [ "$oom" != "0" ]; then
    echo "default-start: FAIL - server answered but logged $oom 'out of memory' lines." >&2
    exit 1
fi

kv=$(curl -s "localhost:$PORT/health" | grep -oP '"kv_capacity_tokens":\K[0-9]+')
echo "default-start: PASS - healthy, answered, 0 OOM lines, kv_capacity_tokens=$kv"
