#!/usr/bin/env bash
# NIAH A/B for the sparse page score: min/max corner bound vs mean+std.
# One image, one model, one budget; the only difference between arms is
# attention.sparse_score_meanstd. Prints the per-arm hit count.
#
# Usage: bash scratch/niah_score_ab.sh <budget_tokens> [std_coef]
set -u

BUDGET="${1:-4096}"
COEF="${2:-1.0}"
MODEL=Qwen3.8-27B-NVFP4-vllm
IMG=imp:test
NAME=imp-niah-ab
OUT=/tmp/claude-1000/-home-kekz-github-com-kekzl-imp/7af69718-7814-41c2-b16c-5995fc182a78/scratchpad
DEPTHS=0.05,0.25,0.5,0.75,0.95
LENGTHS=100000,155000

run_arm() {
    local meanstd="$1" tag="$2"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all -p 8080:8080 -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --model "/models/$MODEL" \
        --set kv_cache.dtype=nvfp4 \
        --set runtime.max_seq_len=131072 \
        --set runtime.max_batch_size=8 \
        --set kv_cache.growable=false \
        --set "attention.sparse_topk_tokens=$BUDGET" \
        --set "attention.sparse_score_meanstd=$meanstd" \
        --set "attention.sparse_score_std_coef=$COEF" >/dev/null
    echo "--- arm $tag (meanstd=$meanstd budget=$BUDGET coef=$COEF)"
    for i in $(seq 1 180); do
        curl -sf http://localhost:8080/health >/dev/null 2>&1 && break
        sleep 2
    done
    # Positive control: the feature has to actually engage in both arms, and
    # the resolved score has to be the one the arm asked for.
    docker logs "$NAME" 2>&1 | grep -E "Sparse decode attention:" | tail -1
    python3 tools/analysis/niah_check.py --url http://localhost:8080 --model "$MODEL" \
        --lengths "$LENGTHS" --depths "$DEPTHS" 2>&1 | tee "$OUT/niah_${tag}_${BUDGET}.log" | tail -14
    # The positive control is a GATE, not a footnote: a 10/10 from an arm whose
    # selection never engaged is a dense run wearing a sparse label. The growable
    # KV pool refuses the metadata pool and produced exactly that on 2026-09-12.
    local active
    active=$(docker logs "$NAME" 2>&1 | grep -c "sparse decode attention ACTIVE")
    docker logs "$NAME" 2>&1 | grep -E "sparse_topk_tokens=.* ignored" | head -2
    echo "ACTIVE lines: $active"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    if [ "$active" -lt 1 ]; then
        echo "ABORT: arm $tag never engaged sparse selection - the result would be a dense run"
        exit 2
    fi
}

run_arm false minmax
run_arm true meanstd
echo "=== logs in $OUT/niah_{minmax,meanstd}_${BUDGET}.log"
