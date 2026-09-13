#!/bin/bash
# Generates tests/perf_baseline.json: N independent cli invocations (default 5) per metric,
# median (resists cuBLAS-algo-state drift over long sessions, up to -10% decode with no code change).
# Run inside imp:test: docker run --rm --gpus all -v $HOME/models:/models -v $PWD:/src -w /src
# -u $(id -u):$(id -g) -e CUBLAS_WORKSPACE_CONFIG=:4096:8 --entrypoint bash imp:test
# scripts/gen_perf_baseline.sh (-u required so the container can write perf_baseline.json back).
# Args: $1 model path (default /models/Qwen3-8B-Q8_0.gguf), $2 trials (default 5).
set -euo pipefail

MODEL="${1:-/models/Qwen3-8B-Q8_0.gguf}"
N_TRIALS="${2:-5}"
REPS=5
OUTPUT="tests/perf_baseline.json"
CLI="imp-cli"
COOLDOWN_SEC=15
# Chunk size for the gated pp512+tg128 invocation. Must match what verify.sh
# benches against this baseline (verify-fast legacy path: 0 = single-chunk;
# set 512 when regenerating tests/perf_baseline_chunked.json).
CHUNK_SIZE="${IMP_BASELINE_CHUNK_SIZE:-0}"

echo "Generating performance baseline..."
echo "Model: $MODEL"
echo "Trials: $N_TRIALS × $REPS reps, $COOLDOWN_SEC s cooldown between trials"

# Reads one throughput number from imp-cli --bench --json (#1583) rather than regexing the
# human table (a format change there used to silently produce an empty sample).
# $1 = "prefill"|"decode"; stdin = the run's stdout. jq -e aborts the whole run on a missing key.
extract_tps() {
    local v
    v=$(jq -er ".${1}_tps") || {
        echo "gen_perf_baseline: no .${1}_tps in the bench output - is --json supported by this build?" >&2
        exit 1
    }
    printf '%s\n' "$v"
}

# Median of stdin (one number per line). Sort + pick middle.
median() {
    sort -g | awk '{a[NR]=$1} END {if (NR%2) print a[(NR+1)/2]; else printf "%.4f\n", (a[NR/2]+a[NR/2+1])/2}'
}

# Run one standalone bench invocation (pp128 / pp4096). Each is its own
# `imp-cli --bench` invocation so cuBLAS algo selection resets between.
# (pp512 + tg128 are measured together in the loop below, gate-matched.)
run_trial() {
    local pp_size="$1"
    local prefix="$2"      # "prefill" or "decode"
    local chunk="${3:-}"   # optional --prefill-chunk-size
    local chunk_arg=()
    [ -n "$chunk" ] && chunk_arg=(--prefill-chunk-size "$chunk")
    # 2>/dev/null, not 2>&1: the human tables go to stderr under --json and
    # would otherwise land in jq's stdin.
    $CLI --model "$MODEL" --bench --bench-pp "$pp_size" --bench-reps "$REPS" \
        "${chunk_arg[@]}" --max-tokens 128 --temperature 0 --json \
        --set speculative.ngram=false 2>/dev/null | extract_tps "$prefix"
}

pp128_samples=$(mktemp)
pp512_samples=$(mktemp)
pp4096_samples=$(mktemp)
tg128_samples=$(mktemp)
trap 'rm -f "$pp128_samples" "$pp512_samples" "$pp4096_samples" "$tg128_samples"' EXIT

for trial in $(seq 1 "$N_TRIALS"); do
    echo "  trial $trial/$N_TRIALS..."
    run_trial 128 prefill >> "$pp128_samples"
    # pp512 and tg128 come from ONE invocation (tg128 = decode at ctx~512 after pp512 prefill,
    # single-chunk), matching how verify.sh measures the gate; a pp128 prefill pins a
    # systematically higher decode rate (~2.5% on 8B, ~5% on 14B).
    # speculative.ngram=false matches verify.sh: the self-repetitive bench prompt (~99.9% accept)
    # would otherwise measure the batched verify GEMMs instead (restart-volatile).
    gate_out=$($CLI --model "$MODEL" --bench --bench-pp 512 --bench-reps "$REPS" \
        --prefill-chunk-size "$CHUNK_SIZE" --max-tokens 128 --temperature 0 --json \
        --set speculative.ngram=false 2>/dev/null)
    echo "$gate_out" | extract_tps prefill >> "$pp512_samples"
    echo "$gate_out" | extract_tps decode >> "$tg128_samples"
    # pp4096 with single-chunk (--prefill-chunk-size 0) so it crosses the
    # cuBLAS→FMHA threshold and exercises the register-resident FA2 kernel
    # (attention.fmha_fa2=on default). verify.sh benches it the same way.
    run_trial 4096 prefill 0 >> "$pp4096_samples"
    if [ "$trial" -lt "$N_TRIALS" ]; then
        sleep "$COOLDOWN_SEC"
    fi
done

pp128=$(median < "$pp128_samples")
pp512=$(median < "$pp512_samples")
pp4096=$(median < "$pp4096_samples")
tg128=$(median < "$tg128_samples")

echo "  pp128 samples: $(paste -sd, "$pp128_samples")  → median $pp128"
echo "  pp512 samples: $(paste -sd, "$pp512_samples")  → median $pp512"
echo "  pp4096 samples: $(paste -sd, "$pp4096_samples")  → median $pp4096"
echo "  tg128 samples: $(paste -sd, "$tg128_samples")  → median $tg128"

# Get GPU info. Try nvcc first, then fall back to nvidia-smi cuda_version
# (the runtime image has no nvcc, only the devel image does).
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
# `a | b | c || fallback` never reaches the fallback: sed exits 0 on empty input, so the
# pipeline succeeds with nothing. Test the value, not the exit code (#1684).
CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' | head -1)
[ -n "$CUDA" ] || CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9.]+' | head -1)
[ -n "$CUDA" ] || CUDA="unknown"
# The commit the numbers were measured at. Absent from every baseline until
# #1684, which is why the generated PROV block had to invent one.
COMMIT=$(git rev-parse --short=8 HEAD 2>/dev/null)
[ -n "$COMMIT" ] || COMMIT="unknown"
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    COMMIT="${COMMIT}-dirty"
fi
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

# Get model VRAM from benchmark output (independent quick run).
vram_line=$($CLI --model "$MODEL" --bench --bench-pp 128 --bench-reps 1 --max-tokens 1 --temperature 0 2>&1 | grep "GPU memory after weight upload" | tail -1)
vram_weights=$(echo "$vram_line" | grep -oP 'weights ~\K[0-9]+' || echo "0")

# Peak VRAM for the verify.sh gate: same invocation the gate uses, so pinned and measured are
# comparable. own_peak (this process's allocations since engine init), not device peak_used
# (which also carries the CUDA context and any neighbour process).
own_peak=$($CLI --model "$MODEL" --bench --bench-pp 128 --bench-reps 1 --max-tokens 8 \
              --temperature 0 --set speculative.ngram=false --mem-report 2>&1 \
           | grep -oP 'own_peak=\K[0-9]+' | tail -1)
own_peak=${own_peak:-0}

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$OUTPUT" << EOF
{
  "schema_version": "legacy-v1",
  "model": "$(basename "$MODEL")",
  "gpu": "$GPU",
  "cuda": "$CUDA",
  "commit": "$COMMIT",
  "vram_total_mb": $VRAM_TOTAL,
  "timestamp": "$TIMESTAMP",
  "methodology": "median of $N_TRIALS trials × $REPS reps, ${COOLDOWN_SEC}s cooldown between trials (cuBLAS algo drift resistant); tg128 from the pp512 run (gate-matched, chunk=$CHUNK_SIZE); speculative.ngram=false",
  "reps": $REPS,
  "n_trials": $N_TRIALS,
  "metrics": {
    "prefill_tps": {
      "pp128": $pp128,
      "pp512": $pp512,
      "pp4096": $pp4096
    },
    "decode_tps": {
      "tg128": $tg128
    },
    "memory_mb": {
      "model_weights": $vram_weights,
      "own_peak_mb": $own_peak
    }
  },
  "thresholds": {
    "decode_regression_pct": 8,
    "prefill_regression_pct": 8,
    "vram_increase_pct": 10
  }
}
EOF

echo "Baseline written to $OUTPUT"
cat "$OUTPUT"
