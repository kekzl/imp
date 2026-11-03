#!/usr/bin/env bash
# imp-server under nsys (--cuda-graph-trace=node) with the vllm_gap_sweep.sh flags (max_batch MAXC,
# spec off), a CONC-stream conc_client burst (WAVES waves, PLEN filler tokens), graceful stop,
# sqlite export, decode-step breakdown via vllm_gap_nsys.py.
# Usage: bash tools/analysis/vllm_gap_nsys.sh MODEL_DIR [OUT=/mnt/data/scratch/vllmgap/nsys]
# Env: CONC (16), WAVES (2), PLEN (0), GEN (300), MAXC (16), IMG (imp:test), NSYS (/opt/nvidia/nsight-systems/2026.1.3),
#      ENGINE=vllm (profile vLLM instead; MARK=<regex of its per-step sampler kernel>),
#      IMP_SET="k=v ..." (extra imp-server --set pairs, e.g. kv_cache.dtype=fp8)
set -uo pipefail
EXTRA=()
for kv in ${IMP_SET:-}; do EXTRA+=(--set "$kv"); done
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=$1
CONC=${CONC:-16}; WAVES=${WAVES:-2}; PLEN=${PLEN:-0}; MAXC=${MAXC:-16}
IMG=${IMG:-imp:test}
NSYS=${NSYS:-/opt/nvidia/nsight-systems/2026.1.3}
OUT="${OUT:-/mnt/data/scratch/vllmgap/nsys}/$MODEL-c$CONC-p$PLEN${ENGINE:+-$ENGINE}"
PORT=${PORT:-8098}
NAME=vllmgap_nsys
mkdir -p "$OUT" && chmod 777 "$OUT"
"$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" >/dev/null || { echo "GPU busy" >&2; exit 2; }
docker rm -f $NAME >/dev/null 2>&1
NSYS_ARGS=(profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --backtrace=none
    --stats=false --force-overwrite=true -o /out/serving)
if [ "${ENGINE:-imp}" = vllm ]; then  # ENGINE=vllm: same flags as the sweep's vLLM arm; IMG only runs the export
    docker run -d --name $NAME --gpus all --ipc=host -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 -w /tmp \
        -v "$HOME/models":/models:ro -v "$OUT":/out -v /opt/nvidia:/opt/nvidia:ro -p "$PORT:$PORT" \
        --entrypoint "$NSYS/target-linux-x64/nsys" "${IMG_VLLM:-vllm/vllm-openai:v0.31.0}" "${NSYS_ARGS[@]}" \
        vllm serve "/models/$MODEL" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT" \
        --max-model-len 16384 --max-num-seqs "$MAXC" >/dev/null
else
    docker run -d --name $NAME --gpus all -u "$(id -u):$(id -g)" -w /tmp -v "$HOME/models":/models:ro \
        -v "$OUT":/out -v /opt/nvidia:/opt/nvidia:ro -p "$PORT:$PORT" \
        --entrypoint "$NSYS/target-linux-x64/nsys" "$IMG" "${NSYS_ARGS[@]}" \
        imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" --max-concurrent "$MAXC" \
        --set "runtime.max_batch_size=$MAXC" --set speculative.mtp_k=0 --set speculative.ngram=false \
        "${EXTRA[@]}" >/dev/null
fi
for _ in $(seq 1 300); do
    sleep 2
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    [ -n "$(docker ps -q -f name=$NAME)" ] || { echo "server died" >&2; docker logs $NAME 2>&1 | tail -20; exit 3; }
done
docker run --rm --network host -e IGNORE_EOS=1 -e MODEL_NAME="$MODEL" -e GEN="${GEN:-300}" -v "$HERE":/t:ro imp:toolchain \
    python3 /t/conc_client.py "$PORT" "$CONC" "$WAVES" nsys "$PLEN" > "$OUT/client.log" 2>&1
grep -E "wave|MEDIAN" "$OUT/client.log"
docker kill --signal=SIGINT $NAME >/dev/null 2>&1
for _ in $(seq 1 120); do
    sleep 2
    [ -n "$(docker ps -q -f name=$NAME)" ] || break
done
docker logs $NAME > "$OUT/server.log" 2>&1
docker rm -f $NAME >/dev/null 2>&1
docker run --rm -u "$(id -u):$(id -g)" -v "$OUT":/out -v /opt/nvidia:/opt/nvidia:ro \
    --entrypoint "$NSYS/target-linux-x64/nsys" "$IMG" export --type sqlite --force-overwrite=true \
    -o /out/serving.sqlite /out/serving.nsys-rep >/dev/null 2>&1
docker run --rm -e MARK="${MARK:-argmax_reduce}" -e SIG="${SIG:-}" -v "$OUT":/d:ro -v "$HERE":/t:ro python:3.13-slim \
    python -I /t/vllm_gap_nsys.py /d/serving.sqlite 30 | tee "$OUT/breakdown.txt"
