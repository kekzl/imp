#!/bin/bash
# Generate performance baseline JSON for regression testing.
# Usage: ./scripts/gen_perf_baseline.sh [model_path]
# Default: /models/Qwen3-8B-Q8_0.gguf
set -euo pipefail

MODEL="${1:-/models/Qwen3-8B-Q8_0.gguf}"
REPS=5
OUTPUT="tests/perf_baseline.json"
CLI="imp-cli"

echo "Generating performance baseline..."
echo "Model: $MODEL"
echo "Reps: $REPS"

# Collect metrics — extract the tok/s value inside parens, not the avg-ms field.
# Bench line format: `pp   512 tokens  avg    33.95 ms  (15083.10 tok/s)  [5 reps]`
# Same parser shape as scripts/verify.sh — keep them in sync.
extract_tps() {
    grep -oP "$1"'\s+\d+\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' | head -1
}
pp128=$($CLI --model "$MODEL" --bench --bench-pp 128 --bench-reps "$REPS" --max-tokens 128 --temperature 0 2>&1 | extract_tps "^pp")
pp512=$($CLI --model "$MODEL" --bench --bench-pp 512 --bench-reps "$REPS" --max-tokens 128 --temperature 0 2>&1 | extract_tps "^pp")
tg128=$($CLI --model "$MODEL" --bench --bench-pp 128 --bench-reps "$REPS" --max-tokens 128 --temperature 0 2>&1 | extract_tps "^tg")

# Get GPU info. Try nvcc first, then fall back to nvidia-smi cuda_version
# (the runtime image has no nvcc, only the devel image does).
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
CUDA=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' \
       || nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9.]+' \
       || echo "unknown")
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

# Get model VRAM from benchmark output
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
  "reps": $REPS,
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
