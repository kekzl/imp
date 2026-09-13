#!/usr/bin/env bash
# Decode tok/s across sparse budgets, prefill cancelled by construction: same long prompt at
# two generation lengths, slope (n2-n1)/(wall2-wall1) is decode throughput with the prefill
# term removed. A single wall reading at a long prompt can't price a decode-side knob when
# prefill dominates short generations.
set -u

MODEL=Qwen3.8-27B-NVFP4-vllm
IMG=imp:test
NAME=imp-slope
N1=256
N2=1024
export TARGET_CHARS=310000

# Returns "<emitted_tokens> <wall_s>". The slope MUST use the tokens actually
# emitted, not the requested max_tokens: the model can stop early, and assuming
# the request value made an 8192 budget read faster than a 1024 one (2026-09-12).
run_of() {  # $1 = gen, $2 = tag
    python3 tools/analysis/longctx_conc_client.py 8080 1 "$1" "$2" \
      | awk '{n="";w="";for(i=1;i<=NF;i++){if($i ~ /^sum_completion=/){sub("sum_completion=","",$i);n=$i} if($i ~ /^wall_s=/){sub("wall_s=","",$i);w=$i}} print n, w}'
}

for BUDGET in "$@"; do
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all -p 8080:8080 -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --model "/models/$MODEL" \
        --set kv_cache.dtype=nvfp4 \
        --set runtime.max_seq_len=131072 \
        --set runtime.max_batch_size=8 \
        --set kv_cache.growable=false \
        --set speculative.mtp_k=0 \
        --set speculative.ngram=false \
        --set "attention.sparse_topk_tokens=$BUDGET" >/dev/null
    for i in $(seq 1 180); do
        curl -sf http://localhost:8080/health >/dev/null 2>&1 && break
        sleep 2
    done
    python3 tools/analysis/longctx_conc_client.py 8080 1 "$N1" warm >/dev/null 2>&1
    for r in 1 2 3; do
        read -r n1 w1 <<<"$(run_of "$N1" "b${BUDGET}-n1-r${r}")"
        read -r n2 w2 <<<"$(run_of "$N2" "b${BUDGET}-n2-r${r}")"
        echo "budget=$BUDGET round=$r n=$n1/$n2 wall=$w1/$w2 decode_tok_s=$(python3 -c "
dn = $n2 - $n1
dw = $w2 - $w1
print(f'{dn/dw:.1f}' if dn > 0 and dw > 0 else 'INVALID')")"
    done
    active=$(docker logs "$NAME" 2>&1 | grep -c "sparse decode attention ACTIVE")
    echo "  budget=$BUDGET ACTIVE=$active"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
done
