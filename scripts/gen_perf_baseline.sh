#!/bin/bash
# Generate performance baseline JSON for regression testing.
#
# Methodology: N independent cli invocations (default 5) per metric, with a
# short cooldown between, then take the **median**. Resists cuBLAS-algo-state
# drift over long sessions — see
# memory/bench_sustained_load_cublas_algo_drift_2026_05_23.md for the failure
# mode this guards against (multi-hour bench session can drift decode -10 %
# even with no code change). The old script used a single invocation; CI was
# then sensitive to whatever cuBLAS state happened to be cached at that moment.
#
# Usage (always inside the imp:test container — uses bare `imp-cli`):
#   docker run --rm --gpus all \
#     -v /home/kekz/models:/models \
#     -v $PWD:/src -w /src \
#     -u $(id -u):$(id -g) \
#     -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
#     --entrypoint bash imp:test scripts/gen_perf_baseline.sh
#
# The `-u $(id -u):$(id -g)` is required so the container can write the new
# tests/perf_baseline.json back to the host bind mount. Without it the script
# completes the benches but fails at the final `cat > $OUTPUT` step.
#
# Optional positional args:
#   $1  model path (default /models/Qwen3-8B-Q8_0.gguf)
#   $2  number of trials (default 5)
#
set -euo pipefail

MODEL="${1:-/models/Qwen3-8B-Q8_0.gguf}"
N_TRIALS="${2:-5}"
REPS=5
OUTPUT="tests/perf_baseline.json"
CLI="imp-cli"
COOLDOWN_SEC=15

echo "Generating performance baseline..."
echo "Model: $MODEL"
echo "Trials: $N_TRIALS × $REPS reps, $COOLDOWN_SEC s cooldown between trials"

# Extract the tok/s value inside parens. Bench line format:
#   `pp   512 tokens  avg    33.95 ms  (15083.10 tok/s)  [5 reps]`
extract_tps() {
    grep -oP "$1"'\s+\d+\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' | head -1
}

# Median of stdin (one number per line). Sort + pick middle.
median() {
    sort -g | awk '{a[NR]=$1} END {if (NR%2) print a[(NR+1)/2]; else printf "%.4f\n", (a[NR/2]+a[NR/2+1])/2}'
}

# Run one full trial: pp128 + pp512 + tg128 measurements. Each is its own
# `imp-cli --bench` invocation so cuBLAS algo selection resets between.
run_trial() {
    local pp_size="$1"
    local prefix="$2"  # "pp" or "tg"
    $CLI --model "$MODEL" --bench --bench-pp "$pp_size" --bench-reps "$REPS" \
        --max-tokens 128 --temperature 0 2>&1 | extract_tps "^$prefix"
}

pp128_samples=$(mktemp)
pp512_samples=$(mktemp)
tg128_samples=$(mktemp)
trap 'rm -f "$pp128_samples" "$pp512_samples" "$tg128_samples"' EXIT

for trial in $(seq 1 "$N_TRIALS"); do
    echo "  trial $trial/$N_TRIALS..."
    run_trial 128 pp >> "$pp128_samples"
    run_trial 512 pp >> "$pp512_samples"
    run_trial 128 tg >> "$tg128_samples"
    if [ "$trial" -lt "$N_TRIALS" ]; then
        sleep "$COOLDOWN_SEC"
    fi
done

pp128=$(median < "$pp128_samples")
pp512=$(median < "$pp512_samples")
tg128=$(median < "$tg128_samples")

echo "  pp128 samples: $(paste -sd, "$pp128_samples")  → median $pp128"
echo "  pp512 samples: $(paste -sd, "$pp512_samples")  → median $pp512"
echo "  tg128 samples: $(paste -sd, "$tg128_samples")  → median $tg128"

# Get GPU info. Try nvcc first, then fall back to nvidia-smi cuda_version
# (the runtime image has no nvcc, only the devel image does).
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
CUDA=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' \
       || nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9.]+' \
       || echo "unknown")
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

# Get model VRAM from benchmark output (independent quick run).
vram_line=$($CLI --model "$MODEL" --bench --bench-pp 128 --bench-reps 1 --max-tokens 1 --temperature 0 2>&1 | grep "GPU memory after weight upload" | tail -1)
vram_weights=$(echo "$vram_line" | grep -oP 'weights ~\K[0-9]+' || echo "0")

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$OUTPUT" << EOF
{
  "model": "$(basename "$MODEL")",
  "gpu": "$GPU",
  "cuda": "$CUDA",
  "vram_total_mb": $VRAM_TOTAL,
  "timestamp": "$TIMESTAMP",
  "methodology": "median of $N_TRIALS trials × $REPS reps, ${COOLDOWN_SEC}s cooldown between trials (cuBLAS algo drift resistant)",
  "reps": $REPS,
  "n_trials": $N_TRIALS,
  "metrics": {
    "prefill_tps": {
      "pp128": $pp128,
      "pp512": $pp512
    },
    "decode_tps": {
      "tg128": $tg128
    },
    "memory_mb": {
      "model_weights": $vram_weights
    }
  },
  "thresholds": {
    "decode_regression_pct": 3,
    "prefill_regression_pct": 5,
    "vram_increase_pct": 10
  }
}
EOF

echo "Baseline written to $OUTPUT"
cat "$OUTPUT"
