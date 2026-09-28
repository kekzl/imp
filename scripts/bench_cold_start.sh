#!/usr/bin/env bash
# Cold-start KPI: docker run -> container start -> model load start -> /health ready -> first token.
# Prints each phase in ms for N repeats plus a median row; page cache is never dropped (needs root),
# so run 1 is the coldest and the cache column reads fincore's resident % or "unknown".
# Usage: bash scripts/bench_cold_start.sh [--repeats N] [--mock] [--dry-run]
# Env: IMP_CS_MODEL (under IMP_MODELS_DIR), IMP_MODELS_DIR, IMP_TEST_IMG, IMP_SRV_PORT,
#      IMP_CS_TIMEOUT (s), IMP_CS_ARGS (extra imp-server flags).
set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
MODEL="${IMP_CS_MODEL:-Qwen3.8-Flash-Next-NVFP4}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"
IMG="${IMP_TEST_IMG:-imp:test}"
PORT="${IMP_SRV_PORT:-18080}"
TIMEOUT_S="${IMP_CS_TIMEOUT:-600}"
EXTRA="${IMP_CS_ARGS:-}"
CTR=imp_cold_start_bench
REPEATS=3
MOCK=0
DRY=0
# stdout is block-buffered until the listen banner; the logger's per-line fflush releases this
# line with the first engine log line, so load start is late by at most that gap.
LOAD_MARK="Loading model:"

while [ $# -gt 0 ]; do
    case "$1" in
        --repeats) REPEATS="$2"; shift 2 ;;
        --mock) MOCK=1; shift ;;
        --dry-run) DRY=1; shift ;;
        *) echo "cold-start: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ "$MOCK" = 1 ]; then
    # CPU plumbing check: tests/api/mock_server.py, no GPU, no model, binds 127.0.0.1.
    IMG=python:3.12-slim
    MODEL=mock-model-v1
    LOAD_MARK="Mock imp server listening"
    RUN=(docker run -d --name "$CTR" --network host -v "$ROOT/tests/api:/mock:ro" "$IMG"
         python -u /mock/mock_server.py --port "$PORT")
else
    # shellcheck disable=SC2206  # IMP_CS_ARGS is a flag list, word splitting intended
    RUN=(docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT":"$PORT" "$IMG"
         imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" $EXTRA)
fi
REQ="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France.\"}],\"max_tokens\":1,\"temperature\":0}"
FIRST=(curl -s -o /dev/null -w '%{http_code}' -m 300 "localhost:$PORT/v1/chat/completions"
       -H 'Content-Type: application/json' -d "$REQ")

now_ns() { date +%s%N; }
iso_ns() { date -d "$1" +%s%N; }  # RFC3339Nano from dockerd -> epoch ns
ms() { echo $(( ($2 - $1) / 1000000 )); }

cache_state() {  # resident % of the model files, or "unknown" without fincore
    local path="$MODELS_DIR/$MODEL"
    if [ "$MOCK" = 1 ]; then echo "n/a"; return; fi
    if ! command -v fincore >/dev/null 2>&1 || [ ! -e "$path" ]; then echo "unknown"; return; fi
    find "$path" -type f -print0 | xargs -0 fincore -b -n -o RES,SIZE 2>/dev/null |
        awk '{r += $1; s += $2} END {if (s > 0) printf "%d%%\n", 100 * r / s; else print "unknown"}'
}

if [ "$DRY" = 1 ]; then
    echo "dry-run: per repeat, $REPEATS repeats"
    echo "  docker rm -f $CTR"
    echo "  ${RUN[*]}"
    echo "  docker inspect -f '{{.State.StartedAt}}' $CTR"
    echo "  poll every 50 ms: curl -s localhost:$PORT/health | grep '\"model_loaded\":true'"
    echo "  docker logs --timestamps $CTR | first line matching '$LOAD_MARK'"
    echo "  ${FIRST[*]}"
    exit 0
fi

if [ "$MOCK" = 0 ] && [ ! -e "$MODELS_DIR/$MODEL" ]; then
    echo "cold-start: $MODELS_DIR/$MODEL does not exist" >&2
    exit 1
fi

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

img_id=$(docker image inspect -f '{{.Id}}' "$IMG" 2>/dev/null | cut -c8-19)
img_rev=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMG" 2>/dev/null)
echo "# cold-start  image=$IMG id=${img_id:-?} revision=${img_rev:-none}  repo=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"
echo "# model=$MODEL  repeats=$REPEATS  ready = /health model_loaded:true (50 ms poll)  first token = max_tokens=1"
printf '%-4s %-8s %10s %10s %10s %10s %10s\n' run cache run_ms to_load_ms load_ms ttft_ms total_ms
rows=()
for i in $(seq 1 "$REPEATS"); do
    cleanup
    cache=$(cache_state)
    t0=$(now_ns)
    if ! "${RUN[@]}" >/dev/null; then echo "cold-start: docker run failed" >&2; exit 1; fi
    t_start=$(iso_ns "$(docker inspect -f '{{.State.StartedAt}}' "$CTR")")
    deadline=$(( t0 + TIMEOUT_S * 1000000000 ))
    t_ready=0
    while [ "$(now_ns)" -lt "$deadline" ]; do
        if curl -s -m 1 "localhost:$PORT/health" 2>/dev/null | grep -q '"model_loaded": *true'; then
            t_ready=$(now_ns); break
        fi
        if [ "$(docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null)" != "true" ]; then
            echo "cold-start: container exited during startup. Logs:" >&2
            docker logs "$CTR" 2>&1 | tail -25 >&2; exit 1
        fi
        sleep 0.05
    done
    if [ "$t_ready" = 0 ]; then
        echo "cold-start: not ready within ${TIMEOUT_S}s. Logs:" >&2
        docker logs "$CTR" 2>&1 | tail -25 >&2; exit 1
    fi
    code=$("${FIRST[@]}")
    t_first=$(now_ns)
    if [ "$code" != "200" ]; then
        echo "cold-start: first request returned HTTP $code" >&2; exit 1
    fi
    # Read logs into a variable first: grep -m1 in a pipe EPIPEs the producer under pipefail.
    logs=$(docker logs --timestamps "$CTR" 2>&1)
    mark=$(grep -m1 -F "$LOAD_MARK" <<<"$logs" | cut -d' ' -f1)
    if [ -z "$mark" ]; then echo "cold-start: no '$LOAD_MARK' line in the logs" >&2; exit 1; fi
    t_load=$(iso_ns "$mark")
    row=$(printf '%-4s %-8s %10s %10s %10s %10s %10s' "$i" "$cache" "$(ms "$t0" "$t_start")" \
        "$(ms "$t_start" "$t_load")" "$(ms "$t_load" "$t_ready")" "$(ms "$t_ready" "$t_first")" \
        "$(ms "$t0" "$t_first")")
    echo "$row"
    rows+=("$row")
done
printf '%s\n' "${rows[@]}" | awk -v n="$REPEATS" '
    { for (c = 3; c <= 7; c++) v[c, NR] = $c }
    END {
        printf "%-4s %-8s", "med", "-"
        for (c = 3; c <= 7; c++) {
            m = 0; for (r = 1; r <= n; r++) a[r] = v[c, r]
            for (r = 1; r <= n; r++) for (s = r + 1; s <= n; s++) if (a[s] < a[r]) { t = a[r]; a[r] = a[s]; a[s] = t }
            printf " %10s", (n % 2) ? a[(n + 1) / 2] : int((a[n / 2] + a[n / 2 + 1]) / 2)
        }
        printf "\n"
    }'
