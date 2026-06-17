#!/usr/bin/env bash
# Perf regression gate: run imp-cli --bench and compare decode/prefill against a
# baseline json. Single-session, warmed clocks (see BENCHMARKING.md). Decode
# tg128 is the headline. Extracted from ci.yml so the same logic runs locally
# (scripts/verify.sh) and in the GPU CI job.
#
# Usage: bench_gate.sh <imp-cli> [baseline.json] [model_path]
#   IMP_MODELS_DIR overrides the model search dir (default $HOME/models).
# Skips (exit 0) when the baseline model isn't present, so CPU-only runners pass.
set -euo pipefail

CLI="${1:?usage: bench_gate.sh <imp-cli> [baseline.json] [model_path]}"
BASELINE="${2:-tests/perf_baseline.json}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"

BL_MODEL=$(jq -r '.model' "$BASELINE")
MODEL_PATH="${3:-$MODELS_DIR/$BL_MODEL}"
if [ ! -f "$MODEL_PATH" ]; then
  echo "::warning::perf gate skipped — baseline model $MODEL_PATH not found (set IMP_MODELS_DIR)"
  exit 0
fi

BL_TG=$(jq -r '.metrics.decode_tps.tg128' "$BASELINE")
BL_PP=$(jq -r '.metrics.prefill_tps.pp512' "$BASELINE")
DEC_THR=$(jq -r '.thresholds.decode_regression_pct' "$BASELINE")
PRE_THR=$(jq -r '.thresholds.prefill_regression_pct' "$BASELINE")

run_bench() {
  # NOTE: imp-cli prints the "pp 512 …" / "tg 128 …" result lines to STDERR, so
  # do NOT merge 2>&1 here — the measured call below captures stderr into $ERR and
  # parses it. (Merging would send the result lines to the caller's /dev/null and
  # leave $ERR empty, which set -e then turns into a silent early exit.)
  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CLI" --model "$MODEL_PATH" --bench \
    --bench-pp 512 --bench-reps 3 --prefill-chunk-size 0 --max-tokens 128 \
    --temperature 0
}

# Warm the clocks: the GPU downclocks at idle and the first ~1s reads low. One
# discarded run before the measured run (BENCHMARKING.md).
echo "warming clocks (discarded run)..."
run_bench >/dev/null 2>&1 || true

ERR=$(mktemp)
run_bench >/dev/null 2>"$ERR"
PP=$(grep -oP '^pp\s+512\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR" | head -1)
TG=$(grep -oP '^tg\s+128\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR" | head -1)
if [ -z "$PP" ] || [ -z "$TG" ]; then echo "could not parse bench output"; tail -15 "$ERR"; exit 1; fi

DEC_DELTA=$(awk -v c="$TG" -v b="$BL_TG" 'BEGIN{printf "%.2f",(c-b)/b*100}')
PRE_DELTA=$(awk -v c="$PP" -v b="$BL_PP" 'BEGIN{printf "%.2f",(c-b)/b*100}')
echo "decode tg128=$TG (baseline $BL_TG, delta ${DEC_DELTA}%); prefill pp512=$PP (baseline $BL_PP, delta ${PRE_DELTA}%)"

if awk -v d="$DEC_DELTA" -v t="$DEC_THR" 'BEGIN{exit !(-d > t)}'; then
  echo "::error::decode regression > ${DEC_THR}% (if intended: scripts/gen_perf_baseline.sh)"; exit 1
fi
if awk -v d="$PRE_DELTA" -v t="$PRE_THR" 'BEGIN{exit !(-d > t)}'; then
  echo "::warning::prefill regression > ${PRE_THR}% (often cuBLAS variance)"
fi
echo "perf gate passed"
