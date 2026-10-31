#!/usr/bin/env bash
# Measures real gcov LINE coverage of tools/imp-server/ end to end: CI only unit-tests pure
# helpers plus a Python mock contract, never handlers.cpp/main.cpp/batching_engine.cpp for real.
# Builds imp-server with gcov instrumentation (server TUs only), drives every endpoint against
# a real model on GPU. Usage: scripts/coverage_server.sh. Env: IMP_COV_MODEL, IMP_MODELS_DIR,
# IMP_COV_PORT, IMP_COV_KEEP. Output: build/coverage/ (txt+html).
set -euo pipefail

MODEL="${IMP_COV_MODEL:-Qwen3-8B-NVFP4-cortecs}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
PORT="${IMP_COV_PORT:-8080}"
IMG=imp:cov
CTR=imp_cov
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cleanup() { [ "${IMP_COV_KEEP:-0}" = "1" ] || docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== 1/5 build instrumented imp-server (gcov, server TUs only) =="
docker build --target builder --build-arg IMP_BUILD_TESTS=OFF \
    --build-arg IMP_EXTRA_CMAKE="-DIMP_COVERAGE=ON" -t "$IMG" .

echo "== 2/5 launch container + server =="
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT":"$PORT" -w /src "$IMG" sleep infinity >/dev/null
docker exec "$CTR" bash -c "apt-get update -qq && apt-get install -y -qq gcovr >/dev/null 2>&1"
# Rich flag set so args.cpp / main.cpp pre-routing (rate-limit/concurrency) are exercised too.
docker exec -d "$CTR" bash -c "cd /src && ./build/imp-server --model /models/$MODEL \
    --host 0.0.0.0 --port $PORT --max-concurrent 8 --rate-limit 100000 --max-input-tokens 100000 \
    > /tmp/srv.log 2>&1"
for i in $(seq 1 90); do
    curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"model_loaded":true' && break
    sleep 2
done

echo "== 3/5 exercise endpoints + manual batteries =="
# Batteries in python:3.13-slim (stdlib only), never host python3 (clean host).
py() { docker run --rm --network host -u "$(id -u):$(id -g)" -v "$ROOT":"$ROOT":ro -w "$ROOT" "$@"; }
py -e IMP_BASE="http://localhost:$PORT" -e IMP_MODEL="$MODEL" python:3.13-slim python3 tests/exercise_all_endpoints.py || true
py -e IMP_HOST=localhost -e IMP_PORT="$PORT" -e IMP_MODEL="$MODEL" python:3.13-slim python3 tests/test_server_robustness.py || true
IMP_TEST_MODEL="$MODEL" bash tests/test_server_embed_chat_interleave.sh 15 || true
py -e IMP_MODEL="$MODEL" -e N=4 -e LOAD=40 -e FAIL_THRESHOLD=0.5 python:3.13-slim python3 tests/test_server_0token_battery.py || true

echo "== 4/5 stop server (flush gcov) =="
docker exec "$CTR" bash -c "kill -TERM \$(pgrep imp-server); for i in \$(seq 1 30); do pgrep imp-server >/dev/null || break; sleep 1; done"

echo "== 5/5 gcovr report (tools/imp-server) =="
docker exec "$CTR" bash -c "cd /src && mkdir -p build/coverage && \
    gcovr --root . --filter 'tools/imp-server/' --txt --html-details build/coverage/index.html 2>/dev/null"
docker cp "$CTR":/src/build/coverage "$ROOT/build/coverage" 2>/dev/null || true
echo "HTML report (if produced): build/coverage/index.html"
