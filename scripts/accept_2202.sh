#!/usr/bin/env bash
# GPU acceptance for #2202: bench-serve at c=1 agrees with the single-stream decode rate within
# TOL_PCT (default 10). Reference is `imp-cli --bench --json` decode_tps (real model, same
# command as scripts/gen_perf_baseline.sh); `imp-bench e2e` is a synthetic 256-dim model and takes
# no --model, so it cannot be the reference for Qwen3-8B.
# Bench-serve side: 1000 / TPOT p50 at c=1 (decode only, TTFT excluded), prompt 512, 128 tokens,
# speculative.ngram=false on both sides.
# Run on the GPU runner, one job at a time. Prints PASS/FAIL, exit 0 = PASS, 1 = FAIL, 2 = busy/usage.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

MODEL="${MODEL:-Qwen3-8B-Q8_0.gguf}"
TOL_PCT="${TOL_PCT:-10}"
BUSY_CHECK="${BUSY_CHECK:-$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh}"
IMG="${IMG:-${DOCKER_IMG:-$(bash scripts/image_tag.sh)}}"

busy() {  # a busy card corrupts both arms: refuse, never wait
    if ! bash "$BUSY_CHECK"; then
        echo "accept_2202: GPU busy, ask the owner before running" >&2
        exit 2
    fi
}

busy
CLI_JSON=$(docker run --rm --gpus all -v "$HOME/models:/models" "$IMG" imp-cli \
    --model "/models/$MODEL" --bench --bench-pp 512 --bench-reps 5 --prefill-chunk-size 0 \
    --max-tokens 128 --temperature 0 --json --set speculative.ngram=false 2>/dev/null)
CLI_TPS=$(echo "$CLI_JSON" | jq -er '.decode_tps') || { echo "accept_2202: no decode_tps from imp-cli" >&2; exit 1; }

busy
IMG="$IMG" MODEL="$MODEL" CONC="1" PROMPT_LEN=512 OUT_LEN=128 SERVER_ARGS="--set speculative.ngram=false" \
    bash scripts/bench_serve.sh || { echo "accept_2202: FAIL (bench-serve exit $?)" >&2; exit 1; }
SHA=$(git rev-parse --short=8 HEAD)
git diff --quiet HEAD 2>/dev/null || SHA="${SHA}-dirty"
OUT="results/bench_serve_${SHA}_$(basename "$MODEL" .gguf | tr -c 'A-Za-z0-9._\n-' '_').json"
TPOT_MS=$(jq -er '.results[0].client.tpot_ms.p50' "$OUT") || { echo "accept_2202: no TPOT in $OUT" >&2; exit 1; }
SERVE_TPS=$(awk -v t="$TPOT_MS" 'BEGIN { printf "%.2f", 1000 / t }')

awk -v a="$SERVE_TPS" -v b="$CLI_TPS" -v tol="$TOL_PCT" -v m="$MODEL" 'BEGIN {
    d = (a - b) / b * 100; if (d < 0) d = -d
    printf "accept_2202: model=%s bench-serve c=1 decode %.2f tok/s, imp-cli decode %.2f tok/s, delta %.2f %% (tol %s %%)\n", m, a, b, d, tol
    if (d <= tol) { print "accept_2202: PASS"; exit 0 }
    print "accept_2202: FAIL"; exit 1 }'
