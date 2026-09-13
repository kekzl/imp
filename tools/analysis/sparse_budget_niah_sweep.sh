#!/usr/bin/env bash
# Single-arm NIAH sweep over sparse budgets: where does the page score start losing needles?
# Used to find a target budget for further selection work (4096 already reads 10/10 under
# mean+std).
# Usage: bash scratch/niah_budget_sweep.sh <meanstd true|false> <budget...>.
set -u

MEANSTD="$1"; shift
MODEL=Qwen3.8-27B-NVFP4-vllm
IMG=imp:test
NAME=imp-niah-sweep
OUT=/tmp/claude-1000/-home-kekz-github-com-kekzl-imp/7af69718-7814-41c2-b16c-5995fc182a78/scratchpad

for BUDGET in "$@"; do
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all -p 8080:8080 -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --model "/models/$MODEL" \
        --set kv_cache.dtype=nvfp4 \
        --set runtime.max_seq_len=131072 \
        --set runtime.max_batch_size=8 \
        --set kv_cache.growable=false \
        --set "attention.sparse_topk_tokens=$BUDGET" \
        --set "attention.sparse_score_meanstd=$MEANSTD" >/dev/null
    for i in $(seq 1 180); do
        curl -sf http://localhost:8080/health >/dev/null 2>&1 && break
        sleep 2
    done
    res=$(python3 tools/analysis/niah_check.py --url http://localhost:8080 --model "$MODEL" \
        --lengths 100000,155000 --depths 0.05,0.25,0.5,0.75,0.95 2>&1 | tee "$OUT/sweep_${MEANSTD}_${BUDGET}.log" \
        | sed 's/\x1b\[[0-9;]*m//g' | grep "niah_check:")
    active=$(docker logs "$NAME" 2>&1 | grep -c "sparse decode attention ACTIVE")
    echo "budget=$BUDGET meanstd=$MEANSTD ACTIVE=$active  $res"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    [ "$active" -lt 1 ] && { echo "ABORT: selection never engaged at budget $BUDGET"; exit 2; }
done
