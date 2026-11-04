#!/usr/bin/env bash
# Per-step kernel time of a DEFAULT imp-server with MTP at long context: a fresh server under nsys (graph-node
# trace; fresh because a prefix-cache hit would skip a prefill), mtp_longctx_client.py with CONC streams over
# CTX, then mtp_step_profile.py per client window (decode phase only).
# Usage: IMAGE=imp:x OUT=/mnt/data/scratch/dir [CONC=1] [CTX="20000 80000"] [GEN=384]
#        [CORPUS=dir holding docs/internals + src/] [SETS="k=v,k=v"] [PROFILE_ARGS=...] mtp_longctx_nsys.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IMAGE=${IMAGE:?IMAGE=imp:tag}
OUT=${OUT:?OUT=dir}
CONC=${CONC:-1}
NVFP4=${NVFP4:-Qwen3.8-27B-NVFP4-vllm}
PORT=${PORT:-8212}
NAME=${NAME:-mtplong-nsys-$PORT}
SETS=${SETS:-}
NSYS="${NSYS:-$(ls -d /opt/nvidia/nsight-systems/*/ | sort | tail -1)}"
mkdir -p "$OUT" && chmod 777 "$OUT"
echo "harness: md5=$(md5sum "$0" | cut -c1-12) profile_md5=$(md5sum "$HERE/mtp_step_profile.py" | cut -c1-12)" \
    "image=$IMAGE tree=$(docker inspect "$IMAGE" --format '{{index .Config.Labels "imp.tree"}}') conc=$CONC" \
    "sets=$SETS" | tee "$OUT/summary.txt"
"$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > /dev/null || { echo "GPU busy" >&2; exit 2; }
docker rm -f "$NAME" >/dev/null 2>&1
# shellcheck disable=SC2046
docker run -d --name "$NAME" --gpus all -v "$HOME/models":/models:ro -v "$OUT":/out -v "$NSYS":/nsys -w /tmp \
    -p "$PORT:$PORT" --entrypoint /nsys/target-linux-x64/nsys "$IMAGE" \
    profile --sample=none --cpuctxsw=none --backtrace=none -t cuda --cuda-graph-trace=node \
    -o /out/serving --force-overwrite=true \
    imp-server --model "/models/$NVFP4" --port "$PORT" --host 0.0.0.0 \
    $(for s in ${SETS//,/ }; do printf -- "--set %s " "$s"; done) >/dev/null
for _ in $(seq 1 600); do
    sleep 2
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    [ -z "$(docker ps -q -f name="$NAME")" ] && { echo "server died"; docker logs "$NAME" 2>&1 | tail -20; exit 3; }
done
mkdir -p "$OUT/client"
docker run --rm --network host -e CTX="${CTX:-20000 80000}" -e GEN="${GEN:-384}" -e CONC="$CONC" \
    -e OUT_DIR=/o -v "$OUT/client":/o -e REPO_DIR=/corpus -v "${CORPUS:-$REPO}":/corpus:ro \
    -v "$HERE":/h:ro python:3.13-slim python -I /h/mtp_longctx_client.py "$PORT" "c$CONC" 2>&1 |
    tee "$OUT/client.txt"
grep -E '^(REQ|AGG)' "$OUT/client.txt" >> "$OUT/summary.txt"
docker kill --signal=SIGINT "$NAME" >/dev/null 2>&1
for _ in $(seq 1 180); do
    sleep 2
    [ -z "$(docker ps -q -f name="$NAME")" ] && break
done
docker logs "$NAME" > "$OUT/server.log" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
grep -E 'spec-batch price|mtp-batched-feed' "$OUT/server.log" | tail -12 | tee -a "$OUT/summary.txt"
nsys export --type sqlite --force-overwrite=true -o "$OUT/serving.sqlite" "$OUT/serving.nsys-rep" >/dev/null 2>&1
# shellcheck disable=SC2086
docker run --rm -v "$OUT":/d:ro -v "$HERE":/h:ro python:3.13-slim \
    python -I /h/mtp_step_profile.py /d/serving.sqlite ${PROFILE_ARGS:-} | tee -a "$OUT/summary.txt"
