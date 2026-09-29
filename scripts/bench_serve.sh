#!/usr/bin/env bash
# Server-level serving benchmark (#2202): imp-server in Docker, streaming concurrency sweep via
# tools/analysis/serving_kpi.py (also in Docker), JSON + table, server stopped on exit.
# Usage: scripts/bench_serve.sh [--mock]
# Env: MODEL (file under ~/models, required unless --mock), CONC="1 8 32", PROMPT_LEN=128,
#      OUT_LEN=128, N (requests per level, default max(32, 2*C)), SEED=0, PORT=8093,
#      IMG (server image, default: make's DOCKER_IMG), SERVER_ARGS (extra imp-server flags),
#      RESULTS_DIR=results, KPI_ARGS (extra serving_kpi.py flags).
# --mock: tests/api/mock_server.py on python:3.12-slim instead of imp-server (no GPU, no model).
# Exit 0 = sweep done with 0 request errors; 1 = errors or server not ready; 2 = usage.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
ROOT=$PWD

MOCK=0
[ "${1:-}" = "--mock" ] && MOCK=1
CONC="${CONC:-1 8 32}"
PROMPT_LEN="${PROMPT_LEN:-128}"
OUT_LEN="${OUT_LEN:-128}"
N="${N:-0}"
SEED="${SEED:-0}"
PORT="${PORT:-8093}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PYIMG="${PYIMG:-python:3.12-slim}"
CTR="imp-bench-serve-$$"

if [ "$MOCK" = 1 ]; then
    MODEL="${MODEL:-mock-model-v1}"
elif [ -z "${MODEL:-}" ]; then
    echo "bench-serve: MODEL=<file under ~/models> is required" >&2
    exit 2
fi
LEVELS=$(echo "$CONC" | tr -s ' ,' ',' | sed 's/^,//;s/,$//')
[[ "$LEVELS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "bench-serve: bad CONC='$CONC'" >&2; exit 2; }

SHA=$(git rev-parse --short=8 HEAD 2>/dev/null || echo nogit)
git diff --quiet HEAD 2>/dev/null || SHA="${SHA}-dirty"
SLUG=$(basename "$MODEL" .gguf | tr -c 'A-Za-z0-9._\n-' '_')
mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/bench_serve_${SHA}_${SLUG}.json"

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1; }
trap cleanup EXIT

if [ "$MOCK" = 1 ]; then
    # mock binds 127.0.0.1 only: host network, not a published port
    docker run -d --name "$CTR" --network host -v "$ROOT/tests/api:/mock:ro" "$PYIMG" \
        python -u /mock/mock_server.py --port "$PORT" --latency-ms "${MOCK_LATENCY_MS:-5}" >/dev/null || exit 1
else
    IMG="${IMG:-${DOCKER_IMG:-$(bash scripts/image_tag.sh)}}"
    # shellcheck disable=SC2086  # SERVER_ARGS is a flag list, word splitting intended
    docker run -d --name "$CTR" --gpus all -p "$PORT:8080" -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --model "/models/$MODEL" ${SERVER_ARGS:-} >/dev/null || exit 1
fi

echo "bench-serve: waiting for /ready on :$PORT" >&2
ready=0
for _ in $(seq 1 "${READY_TIMEOUT_S:-300}"); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/ready")" = 200 ]; then
        ready=1
        break
    fi
    docker ps -q -f "name=$CTR" | grep -q . || break
    sleep 1
done
if [ "$ready" != 1 ]; then
    echo "bench-serve: server not ready; last log lines:" >&2
    docker logs --tail 20 "$CTR" >&2
    exit 1
fi

# Client in Docker too (host python is not a dev runtime). --network host reaches the published port.
# shellcheck disable=SC2086  # KPI_ARGS is a flag list
docker run --rm --network host --user "$(id -u):$(id -g)" -v "$ROOT:/src" -w /src "$PYIMG" \
    python tools/analysis/serving_kpi.py --url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --levels "$LEVELS" --max-tokens "$OUT_LEN" --prompt-tokens "$PROMPT_LEN" \
    --requests-per-level "$N" --seed "$SEED" --tag "bs$SEED" --ignore-eos --brief \
    --json "$RESULTS_DIR/.bench_serve_$$.json" ${KPI_ARGS:-}
rc=$?
if [ ! -s "$RESULTS_DIR/.bench_serve_$$.json" ]; then
    echo "bench-serve: client wrote no JSON (exit $rc)" >&2
    exit 1
fi
mv "$RESULTS_DIR/.bench_serve_$$.json" "$OUT"
errs=$(jq '[.results[].client.err] | add' "$OUT")
echo "bench-serve: $OUT (errors: $errs)" >&2
[ "$rc" = 0 ] && [ "$errs" = 0 ]
