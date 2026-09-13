#!/usr/bin/env bash
# Equivalence A/B for speculative.factored_spare: one image, one model, greedy and
# deterministic; the slot-swapping verify and the factored one must emit the SAME tokens.
# Also prints each arm's resolved SSM/GDN state pool and batch.
# Usage: bash tools/analysis/factored_spare_equiv.sh [max_batch] [n_prompts].
set -u

# Both arms must run the SAME geometry: the factored arm frees ~2.5 GiB and left to itself
# keeps a larger batch and KV pool, making greedy output differ for batch-shape reasons alone
# (SETTLED D-2). BATCH is pinned below the slot-swapping arm's clamp and the KV pool is pinned
# outright, so the only remaining difference is how the drafted row is carried.
BATCH="${1:-16}"
NPROMPT="${2:-8}"
KVBLOCKS="${KVBLOCKS:-2000}"
MODEL=Qwen3.8-27B-NVFP4-vllm
IMG=imp:test
NAME=imp-fspare
OUT="${OUT_DIR:-/tmp/factored_spare}"
mkdir -p "$OUT"

run_arm() {
    local fac="$1" tag="$2"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all -p 8080:8080 -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --model "/models/$MODEL" \
        --set runtime.max_batch_size="$BATCH" \
        --set kv_cache.max_blocks="$KVBLOCKS" \
        --set runtime.deterministic=true \
        --set speculative.batch_verify=true \
        --set speculative.mtp_k=1 \
        --set "speculative.factored_spare=$fac" >/dev/null
    for i in $(seq 1 180); do
        curl -sf http://localhost:8080/health >/dev/null 2>&1 && break
        sleep 2
    done
    docker logs "$NAME" 2>&1 | grep -E "SSM/GDN state|clamped|factored_spare" | head -4
    # CONCURRENT on purpose: the batched verify needs a batch of at least two
    # with drafts. Fired sequentially every step declined 'no_draft_in_batch'
    # and the arms agreed because neither path ran (2026-09-12).
    rm -f "$OUT/$tag."*.part
    for p in $(seq 1 "$NPROMPT"); do
        curl -s http://localhost:8080/v1/completions -H 'Content-Type: application/json' -d "$(printf '{"model":"%s","prompt":"Write a long detailed essay about the history of steam engines, part %d.","max_tokens":400,"temperature":0,"stream":false}' "$MODEL" "$p")" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])' > "$OUT/$tag.$p.part" &
    done
    wait
    cat "$OUT/$tag."*.part > "$OUT/$tag.txt"
    rm -f "$OUT/$tag."*.part
    # Positive control, as a GATE: an arm that refused the batched verify emits the same tokens as
    # any dense arm, so "IDENTICAL" from a refused arm proves nothing. Guards against the factored
    # arm silently reserving no spare slots and refusing every step with "no_spare_slots".
    local inits batched
    inits=$(docker logs "$NAME" 2>&1 | grep -c "speculative.factored_spare:")
    # The BATCHED verify's one-shot proof line. Its periodic counter only logs
    # every 100 steps, and a workload that declines most steps never reaches
    # that, so reading the counter called a working arm dead (2026-09-12).
    batched=$(docker logs "$NAME" 2>&1 | grep -c "spec-batch: first batched verify")
    echo "  factored init lines: $inits, batched verify reached: $batched"
    docker logs "$NAME" 2>&1 | grep -oE "spec-batch: first batched verify.*" | sed "s/^/  /"
    docker logs "$NAME" 2>&1 | grep -oE "spec-batch: (step declined|batched verify refused) by '[a-z_]+'" |
        sort | uniq -c | sed 's/^/  /'
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    if [ "$batched" -lt 1 ]; then
        echo "ABORT: arm $tag ran no BATCHED verify at all - the comparison would be dense vs dense"
        exit 2
    fi
    if [ "$fac" = "true" ] && [ "$inits" -lt 1 ]; then
        echo "ABORT: arm $tag never built the factored rows"
        exit 2
    fi
}

echo "=== arm A: slot-swapping spare"
run_arm false swap
echo "=== arm B: factored spare"
run_arm true factored
echo "=== diff"
if diff -q "$OUT/swap.txt" "$OUT/factored.txt" >/dev/null; then
    echo "IDENTICAL: $(wc -l < "$OUT/swap.txt") lines, $(wc -c < "$OUT/swap.txt") bytes"
else
    echo "DIFFERS"
    diff "$OUT/swap.txt" "$OUT/factored.txt" | head -20
    exit 1
fi
