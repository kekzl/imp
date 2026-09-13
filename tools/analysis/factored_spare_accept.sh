#!/usr/bin/env bash
# Acceptance-rate oracle for the factored verify spare: byte equality is not available since
# batched verify concurrency makes which requests carry a draft a timing race (SETTLED D-2), so
# greedy text diverges for reasons unrelated to state. Acceptance rate is decisive instead: the
# verify accepts a draft only when it matches the model's own next token, so a wrong recurrent
# state or conv window collapses acceptance toward zero.
# Usage: bash tools/analysis/factored_spare_accept.sh [batch] [streams] [tokens].
set -u

BATCH="${1:-16}"
STREAMS="${2:-8}"
TOKENS="${3:-400}"
MODEL=Qwen3.8-27B-NVFP4-vllm
IMG=imp:test
NAME=imp-fspare-acc
KVBLOCKS="${KVBLOCKS:-2000}"

run_arm() {
    local fac="$1" tag="$2"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all -p 8080:8080 -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --model "/models/$MODEL" \
        --set runtime.max_batch_size="$BATCH" \
        --set kv_cache.max_blocks="$KVBLOCKS" \
        --set speculative.batch_verify=true \
        --set speculative.mtp_k=1 \
        --set "speculative.factored_spare=$fac" >/dev/null
    for i in $(seq 1 180); do
        curl -sf http://localhost:8080/health >/dev/null 2>&1 && break
        sleep 2
    done
    for p in $(seq 1 "$STREAMS"); do
        curl -s http://localhost:8080/v1/completions -H 'Content-Type: application/json' \
            -d "$(printf '{"model":"%s","prompt":"Write a long detailed essay about the history of steam engines, part %d.","max_tokens":%d,"temperature":0}' "$MODEL" "$p" "$TOKENS")" \
            > /dev/null &
    done
    wait
    local reached
    reached=$(docker logs "$NAME" 2>&1 | grep -c "spec-batch: first batched verify")
    echo "--- arm $tag (factored=$fac), batched verify reached: $reached"
    docker logs "$NAME" 2>&1 | grep -oE "\[spec-ngram\] verify_steps=[0-9]+ .*accepted=[0-9]+ \([0-9.]+%\)" |
        tail -1 | sed 's/^/  /'
    docker logs "$NAME" 2>&1 | grep -oE "spec-batch: first batched verify.*" | sed 's/^/  /'
    docker logs "$NAME" 2>&1 | grep -E "SSM/GDN state +[0-9]+ MiB" | head -1 | sed 's/^/  /'
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    if [ "$reached" -lt 1 ]; then
        echo "ABORT: arm $tag never ran a batched verify"
        exit 2
    fi
}

run_arm false swap
run_arm true factored
