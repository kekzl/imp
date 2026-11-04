#!/usr/bin/env bash
# Kernel-time share of the batched NVFP4 verify GEMV family on a serving run: imp-server under
# nsys with graph-node trace, one client level after another (8 s idle between, the window split),
# then tools/analysis/nsys_kernel_windows.py per level.
# Usage: tools/analysis/verify_gemv_share_nsys.sh OUTDIR. Env: MODEL, BIN_DIR (holds imp-server),
# LEVELS ("1 4 32"), WAVES (2), PLEN (0), GEN (300), MAXSEQ (4096), KVB (2387, empty = auto),
# EXTRA (extra imp-server args, e.g. "--set speculative.capture=false"), PORT.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?OUTDIR}"
MODEL=${MODEL:-/models/Qwen3.8-27B-NVFP4-vllm}
BIN_DIR=${BIN_DIR:?BIN_DIR}
LEVELS=${LEVELS:-1 4 32}
WAVES=${WAVES:-2}
PLEN=${PLEN:-0}
GEN=${GEN:-300}
MAXSEQ=${MAXSEQ:-4096}
KVB=${KVB-2387}
EXTRA=${EXTRA:-}
PORT=${PORT:-8371}
NAME=imp-kparf16-nsys
NSYS="${NSYS:-$(ls -d /opt/nvidia/nsight-systems/*/ | sort | tail -1)}"
mkdir -p "$OUT" && chmod 777 "$OUT"
"$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > /dev/null || { echo "GPU busy" >&2; exit 1; }
KVARG=()
[ -n "$KVB" ] && KVARG=(--set kv_cache.max_blocks="$KVB")
docker rm -f $NAME >/dev/null 2>&1
# shellcheck disable=SC2086
docker run -d --name $NAME --gpus all -v "$HOME/models":/models:ro -v "$BIN_DIR":/bin_arm:ro -v "$OUT":/out \
    -v "$NSYS":/nsys -w /tmp -p ${PORT}:${PORT} --entrypoint /nsys/target-linux-x64/nsys imp:toolchain \
    profile --sample=none --cpuctxsw=none --backtrace=none -t cuda --cuda-graph-trace=node \
    -o /out/serving --force-overwrite=true \
    /bin_arm/imp-server --model "$MODEL" --port $PORT --host 0.0.0.0 --max-concurrent 32 \
    --set runtime.max_batch_size=32 --set runtime.max_seq_len="$MAXSEQ" "${KVARG[@]}" $EXTRA >/dev/null
for _ in $(seq 1 300); do
    sleep 2
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    [ -z "$(docker ps -q -f name=$NAME)" ] && { echo "server died"; docker logs $NAME 2>&1 | tail -20; exit 3; }
done
sleep 8
for C in $LEVELS; do
    echo "== level c$C ==" | tee -a "$OUT/client.log"
    docker run --rm --network host -e MODEL_NAME="$(basename "$MODEL")" -e GEN="$GEN" \
        -v "$ROOT/tools/analysis:/h:ro" python:3.13-slim \
        python -I /h/conc_client.py $PORT "$C" "$WAVES" "share" "$PLEN" 2>&1 | tee -a "$OUT/client.log"
    sleep 8
done
docker kill --signal=SIGINT $NAME >/dev/null 2>&1
for _ in $(seq 1 120); do
    sleep 2
    [ -z "$(docker ps -q -f name=$NAME)" ] && break
done
docker logs $NAME > "$OUT/server.log" 2>&1
docker rm -f $NAME >/dev/null 2>&1
nsys export --type sqlite --force-overwrite=true -o "$OUT/serving.sqlite" "$OUT/serving.nsys-rep" >/dev/null 2>&1
docker run --rm -v "$OUT":/d:ro -v "$ROOT/tools/analysis:/h:ro" python:3.13-slim \
    python -I /h/nsys_kernel_windows.py /d/serving.sqlite | tee "$OUT/windows.txt"
