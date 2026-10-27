#!/usr/bin/env bash
# MTP one-stream decode, imp vs vLLM, split into acceptance and step cost: seq_stream_client.py,
# 6 identical real chat prompts x 400 greedy tokens x ROUNDS (2), ignore_eos, fresh server per arm,
# spec counters from /metrics around the timed rounds. Per arm: median decode tok/s, accepted per verify, ms per verify.
# Usage: tools/analysis/mtp_vs_vllm.sh [ARMS="imp:2 imp:2:speculative.mtp_nvfp4_head=false vllm:2"] [ROUNDS=2] [IMG_IMP=ghcr.io/kekzl/imp:latest]
#        [IMG_VLLM=vllm/vllm-openai:v0.31.0] [IMP_SET="k=v ..."] [NVFP4=Qwen3.8-27B-NVFP4-vllm]
set -uo pipefail
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
ARMS=${ARMS:-"imp:2 vllm:2"}
ROUNDS=${ROUNDS:-2}
IMG_IMP=${IMG_IMP:-ghcr.io/kekzl/imp:latest}
IMG_VLLM=${IMG_VLLM:-vllm/vllm-openai:v0.31.0}
NVFP4=${NVFP4:-Qwen3.8-27B-NVFP4-vllm}
PORT=${PORT:-8096}
ctr() {  # $1 = metrics text, $2 = metric name: sum of its samples (labels summed)
    grep -E "^$2(\{|[[:space:]])" <<< "$1" | awk '{s += $NF} END {printf "%.0f", s}'
}
for arm in $ARMS; do
    IFS=: read -r eng k sets <<< "$arm"   # imp:2:speculative.x=y,runtime.z=w
    docker rm -f mtpvs >/dev/null 2>&1; sleep 5
    bash "$HERE/../../scripts/require_free_gpu.sh" "mtp_vs_vllm $arm" >/dev/null || { echo "GPU BUSY before $arm"; exit 2; }
    if [ "$eng" = imp ]; then
        docker run -d --name mtpvs --gpus all -v "$MODELS_DIR":/models -p "$PORT:$PORT" "$IMG_IMP" imp-server \
            --model "/models/$NVFP4" --host 0.0.0.0 --port "$PORT" --max-concurrent 1 --set runtime.max_batch_size=1 \
            --set "speculative.mtp_k=$k" $(for s in ${IMP_SET:-} ${sets//,/ }; do printf -- "--set %s " "$s"; done) >/dev/null
        D='imp_spec_drafted_total'; A='imp_spec_accepted_total'; V='imp_spec_verify_steps_total'
    else
        docker run -d --name mtpvs --gpus all --ipc=host -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 -v "$MODELS_DIR":/models \
            -p "$PORT:$PORT" "$IMG_VLLM" --model "/models/$NVFP4" --served-model-name "$NVFP4" --host 0.0.0.0 \
            --port "$PORT" --max-model-len 16384 --max-num-seqs 1 \
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$k}" >/dev/null
        D='vllm:spec_decode_num_draft_tokens_total'; A='vllm:spec_decode_num_accepted_tokens_total'
        V='vllm:spec_decode_num_drafts_total'
    fi
    for _ in $(seq 1 600); do sleep 2; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
        [ -n "$(docker ps -q -f name=mtpvs)" ] || { echo "$arm: server died"; docker logs mtpvs 2>&1 | tail -5; exit 3; }; done
    # warm wave (graph capture, cuBLAS selection) outside the counters
    docker run --rm --network host -e IGNORE_EOS=1 -e MODEL_NAME="$NVFP4" -v "$HERE":/t:ro imp:toolchain \
        python3 /t/conc_client.py "$PORT" 1 1 warm 0 >/dev/null 2>&1   # graph capture, cuBLAS picks
    m0=$(curl -s "http://127.0.0.1:$PORT/metrics")
    out=$(docker run --rm --network host -e ROUNDS="$ROUNDS" -e MODEL_NAME="$NVFP4" -v "$HERE":/t:ro imp:toolchain \
        python3 /t/seq_stream_client.py "$PORT" "$arm" 2>&1 | grep -E "^$arm ")
    m1=$(curl -s "http://127.0.0.1:$PORT/metrics")
    d=$(( $(ctr "$m1" "$D") - $(ctr "$m0" "$D") )); a=$(( $(ctr "$m1" "$A") - $(ctr "$m0" "$A") ))
    v=$(( $(ctr "$m1" "$V") - $(ctr "$m0" "$V") ))
    tps=$(awk '/MEDIAN/ {print $4}' <<< "$out")
    awk -v arm="$arm" -v t="$tps" -v d="$d" -v a="$a" -v v="$v" 'BEGIN {
        tpv = v ? (a + v) / v : 0; printf "%s tok/s %s drafted %d accepted %d verifies %d accept %.1f%% tokens/verify %.3f ms/verify %.2f\n",
        arm, t, d, a, v, d ? 100 * a / d : 0, tpv, (t > 0 && tpv > 0) ? 1000 * tpv / t : 0 }'
    docker logs mtpvs > "${TMPDIR:-/tmp}/mtpvs_${arm//[:=,]/_}.log" 2>&1
    docker rm -f mtpvs >/dev/null 2>&1
done
