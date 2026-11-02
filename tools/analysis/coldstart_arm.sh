#!/usr/bin/env bash
# One fresh imp-server arm for first-request TTFT and KV-growth admission (gap sweep levers 1-2).
# Server flags as vllm_gap_sweep.sh (16 streams, spec off); binaries from BIN_DIR (a make dev copy).
# Writes ready_s (docker run -> /health), base/ready/end_mib (nvidia-smi), server.log, client output.
# MODE ttft: ttft_probe.py (phases of exact-length token prompts); ppbase: pp512 pp4096 pp8192 pp15000
# in one server (serving_kpi.py, max_tokens 1); pplong: pp8192 pp15000 on a fresh server.
# Usage: bash tools/analysis/coldstart_arm.sh BIN_DIR MODEL MODE OUT_DIR [imp-server args...]
# Env: PORT (8197), DENV (extra docker run args, e.g. "-e CUDA_MODULE_LOADING=EAGER").
set -u
BIN=$1 MODEL=$2 MODE=$3 D=$4
shift 4
EXTRA=("$@")
PORT=${PORT:-8197}
NAME=imp-coldstart-arm
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$D"; chmod 777 "$D"
docker rm -f $NAME >/dev/null 2>&1
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$D/base_mib"
t0=$(date +%s.%N)
# shellcheck disable=SC2086 # DENV is a list of docker flags
docker run -d --name $NAME --gpus all ${DENV:-} -v "$HOME/models":/models:ro -v "$BIN":/bin_arm:ro -p "$PORT:$PORT" \
    imp:toolchain /bin_arm/imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" \
    --max-concurrent 16 --set runtime.max_batch_size=16 --set speculative.mtp_k=0 \
    --set speculative.ngram=false "${EXTRA[@]}" >/dev/null
ok=0
for _ in $(seq 1 1200); do
    sleep 0.25
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
    [ -n "$(docker ps -q -f name=$NAME)" ] || break
done
t1=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f\n", b - a }' > "$D/ready_s"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$D/ready_mib"
if [ $ok != 1 ]; then
    echo "SERVER FAILED"; docker logs $NAME > "$D/server.log" 2>&1; docker rm -f $NAME >/dev/null; exit 1
fi
client() {
    docker run --rm --network host -e IGNORE_EOS=1 -e MODEL_NAME="$MODEL" -e GEN="${GEN:-300}" \
        -v "$HERE":/t:ro -v "$D":/o imp:toolchain "$@"
}
GEN=32 client python3 /t/conc_client.py "$PORT" 1 1 warm 0 > "$D/warm.txt" 2>&1
kpi() {  # $1 = prompt tokens
    client python3 /t/serving_kpi.py --url "http://127.0.0.1:$PORT" --model "$MODEL" \
        --endpoint completions --levels 1 --max-tokens 1 --prompt-tokens "$1" \
        --requests-per-level "$(( $1 > 4096 ? 4 : 8 ))" --warmup 0 --no-power --tag "pp$1" \
        --json "/o/pp$1.json" > "$D/pp$1.txt" 2>&1
}
case "$MODE" in
    ttft) client python3 /t/ttft_probe.py "$PORT" "$MODEL" /o/ttft.json > "$D/ttft.txt" 2>&1 ;;
    ppbase) for pp in 512 4096 8192 15000; do kpi $pp; done ;;
    pplong) for pp in 8192 15000; do kpi $pp; done ;;
esac
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$D/end_mib"
docker logs $NAME > "$D/server.log" 2>&1
docker rm -f $NAME >/dev/null 2>&1
echo "done $MODE $MODEL ready=$(cat "$D/ready_s")s"
