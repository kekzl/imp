#!/usr/bin/env bash
# Throughput A/B for the sparse page score, same image and model as the NIAH
# run. Arms alternate across rounds because the host drifts over an evening.
# The metadata pass changes too (Welford instead of min/max), and a ~77k-token
# prompt is prefill-dominated, so this shape prices both passes.
#
# Usage: bash scratch/speed_score_ab.sh <budget_tokens> <rounds> <gen>
set -u

BUDGET="${1:-4096}"
ROUNDS="${2:-2}"
GEN="${3:-128}"
MODEL=Qwen3.8-27B-NVFP4-vllm
IMG=imp:test
NAME=imp-speed-ab
export TARGET_CHARS=310000   # ~77k prompt tokens

start_arm() {
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all -p 8080:8080 -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --model "/models/$MODEL" \
        --set kv_cache.dtype=nvfp4 \
        --set runtime.max_seq_len=131072 \
        --set runtime.max_batch_size=8 \
        --set kv_cache.growable=false \
        --set "attention.sparse_topk_tokens=$BUDGET" \
        --set "attention.sparse_score_meanstd=$1" >/dev/null
    for i in $(seq 1 180); do
        curl -sf http://localhost:8080/health >/dev/null 2>&1 && break
        sleep 2
    done
}

check_active() {
    local n
    n=$(docker logs "$NAME" 2>&1 | grep -c "sparse decode attention ACTIVE")
    if [ "$n" -lt 1 ]; then
        echo "ABORT: arm $1 never engaged sparse selection"
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        exit 2
    fi
}

for r in $(seq 1 "$ROUNDS"); do
    for arm in false true; do
        tag=$([ "$arm" = "true" ] && echo meanstd || echo minmax)
        start_arm "$arm"
        python3 tools/analysis/longctx_conc_client.py 8080 1 "$GEN" "warm-$tag" >/dev/null 2>&1
        for w in 1 2; do
            python3 tools/analysis/longctx_conc_client.py 8080 1 "$GEN" "r${r}-${tag}-w${w}"
        done
        check_active "$tag"
        docker rm -f "$NAME" >/dev/null 2>&1 || true
    done
done
