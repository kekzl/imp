#!/usr/bin/env bash
# Q8_0 dense prefill route A/B (#2416): alternating imp-cli --bench runs per prompt length, one process per
# run, arms = binary dir x gemm.q8_imma_enabled. Prints one TSV row per run: model, pp, arm, pp tok/s.
# Usage: tools/analysis/q8_imma_route_ab.sh ARM_SPEC... where ARM_SPEC = name:bindir:on|off.
# Env: MODEL (default Qwen3-8B-Q8_0.gguf), MODELS_DIR, PPS (default "1536 2048 4096 8192"), ROUNDS (3).
set -euo pipefail
MODEL="${MODEL:-Qwen3-8B-Q8_0.gguf}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PPS="${PPS:-1536 2048 4096 8192}"
ROUNDS="${ROUNDS:-3}"
CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
for r in $(seq 1 "$ROUNDS"); do
    for pp in $PPS; do
        for spec in "$@"; do
            IFS=: read -r name dir imma <<< "$spec"
            [ "$imma" = on ] && val=true || val=false
            "$CHECK" > /dev/null || { echo "GPU busy, abort" >&2; exit 1; }
            log=$(docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v "$MODELS_DIR":/models:ro \
                -v "$dir":/arm:ro imp:toolchain /arm/imp-cli --model "/models/$MODEL" --bench \
                --bench-pp "$pp" --bench-reps 5 --max-tokens 8 --temperature 0 --seed 42 \
                --set speculative.ngram=false --set speculative.mtp_k=0 \
                --set "gemm.q8_imma_enabled=$val" 2>&1 || true)
            tps=$(grep -oE "^pp +${pp} tokens +avg +[0-9.]+ ms +\( *[0-9.]+ tok/s" <<< "$log" \
                | grep -oE "[0-9]+\.[0-9]+ tok/s" | grep -oE "[0-9]+\.[0-9]+" | head -1)
            printf '%s\t%s\t%s\t%s\t%s\n' "$MODEL" "$pp" "$name" "${tps:-FAIL}" "$r"
        done
    done
done
