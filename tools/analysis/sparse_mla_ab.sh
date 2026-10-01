#!/usr/bin/env bash
# Sparse decode A/B on any model (roadmap row 3, MLA): NIAH hits and decode slope, dense vs
# sparse budgets, one fresh imp-server per arm, speculation and prefix cache off.
# Decode slope: same prompt at two generation lengths (ignore_eos), tok/s = (n2-n1)/(t2-t1),
# so prefill cancels (docs/plans/2026-08-28-sparse-decode-attention.md, budget section).
# A sparse arm without a `sparse decode attention ACTIVE` line aborts.
# NIAH_ARMS=none skips NIAH. Usage: MODEL=DeepSeek-V2-Lite-NVFP4-imp IMG=imp:test tools/analysis/sparse_mla_ab.sh <outdir>
set -u
OUT=${1:?outdir}
MODEL=${MODEL:?MODEL}
IMG=${IMG:-imp:test}
PORT=${PORT:-18362}
ROUNDS=${ROUNDS:-3}
SLOPE_CTX=${SLOPE_CTX:-16000,32000}
SLOPE_ARMS=${SLOPE_ARMS:-0,4096}
NIAH_ARMS=${NIAH_ARMS:-0,1024,4096}
NIAH_LENGTHS=${NIAH_LENGTHS:-28000,30000}
DEPTHS=${DEPTHS:-0.05,0.25,0.5,0.75,0.95}
CORPUS=${CORPUS:-$HOME/models/calib_corpus.txt}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$OUT"

start() {  # start <budget> <tag>
    docker rm -f imp-sparse-ab >/dev/null 2>&1
    docker run -d --name imp-sparse-ab --gpus all -p "127.0.0.1:$PORT:$PORT" -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --port "$PORT" --model "/models/$MODEL" \
        --set runtime.max_seq_len=${MAX_SEQ_LEN:-40960} --set runtime.max_batch_size=4 \
        --set speculative.mtp_k=0 --set speculative.ngram=false --set server.prefix_cache=false \
        --set "attention.sparse_topk_tokens=$1" >/dev/null
    for _ in $(seq 1 120); do
        docker logs imp-sparse-ab > "$OUT/server_$2.log" 2>&1
        grep -q "Server listening" "$OUT/server_$2.log" && return 0
        docker ps -q -f name=imp-sparse-ab | grep -q . || return 1
        sleep 5
    done
    return 1
}
stop() {  # stop <budget> <tag>: keeps the log, asserts ACTIVE on sparse arms
    docker logs imp-sparse-ab > "$OUT/server_$2.log" 2>&1
    docker rm -f imp-sparse-ab >/dev/null 2>&1
    local n
    n=$(grep -c "sparse decode attention ACTIVE" "$OUT/server_$2.log")
    echo "  ACTIVE lines: $n"
    if [ "$1" -gt 0 ] && [ "$n" -eq 0 ]; then
        echo "ABORT: sparse arm $2 never selected"
        exit 2
    fi
}
gen() {  # gen <prompt-json-file> <max_tokens> -> "completion_tokens seconds"
    local t0 t1 n
    # Body via stdin: a 77k-token prompt as one argv entry exceeds the kernel's argument limit.
    jq -c --argjson m "$2" '.max_tokens = $m' "$1" > "$1.body"
    t0=$(date +%s.%N)
    n=$(curl -s -m 1800 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d @"$1.body" | jq -r '.usage.completion_tokens // 0')
    t1=$(date +%s.%N)
    echo "$n $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.4f", b - a}')"
}

echo "### decode slope, rounds=$ROUNDS ctx=$SLOPE_CTX arms=$SLOPE_ARMS"
for ctx in ${SLOPE_CTX//,/ }; do
    head -c $((ctx * 4)) "$CORPUS" > "$OUT/prompt_$ctx.txt"
    jq -n --rawfile c "$OUT/prompt_$ctx.txt" --arg m "$MODEL" \
        '{model:$m,prompt:$c,temperature:0,ignore_eos:true,max_tokens:1}' > "$OUT/req_$ctx.json"
done
for r in $(seq 1 "$ROUNDS"); do
    for b in ${SLOPE_ARMS//,/ }; do
        tag="slope_b${b}_r$r"
        start "$b" "$tag" || { echo "ABORT: boot $tag"; exit 3; }
        for ctx in ${SLOPE_CTX//,/ }; do
            gen "$OUT/req_$ctx.json" 8 >/dev/null  # warm: graphs, clocks
            read -r n1 t1 < <(gen "$OUT/req_$ctx.json" 64)
            read -r n2 t2 < <(gen "$OUT/req_$ctx.json" 320)
            tps=$(awk -v n1="$n1" -v n2="$n2" -v a="$t1" -v b="$t2" 'BEGIN{printf "%.4f", (n2 - n1) / (b - a)}')
            printf "SLOPE round=%d budget=%d ctx=%d n=%d/%d t=%.2f/%.2f decode_tok_s=%.2f\n" \
                "$r" "$b" "$ctx" "$n1" "$n2" "$t1" "$t2" "$tps"
        done
        stop "$b" "$tag"
    done
done

[ "$NIAH_ARMS" = none ] && { echo ALLDONE; exit 0; }
echo "### NIAH lengths=$NIAH_LENGTHS depths=$DEPTHS arms=$NIAH_ARMS"
for b in ${NIAH_ARMS//,/ }; do
    tag="niah_b$b"
    start "$b" "$tag" || { echo "ABORT: boot $tag"; exit 3; }
    docker run --rm --network host -v "$REPO/tools/analysis:/a:ro" python:3.13-slim \
        python3 /a/niah_check.py --url "http://127.0.0.1:$PORT" --model "$MODEL" \
        --lengths "$NIAH_LENGTHS" --depths "$DEPTHS" > "$OUT/niah_$tag.log" 2>&1
    echo "NIAH budget=$b rc=$? $(tail -1 "$OUT/niah_$tag.log")"
    stop "$b" "$tag"
done
echo ALLDONE
