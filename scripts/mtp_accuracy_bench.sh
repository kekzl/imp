#!/usr/bin/env bash
# =============================================================================
# mtp_accuracy_bench.sh — Phase 5.5 validation harness
# =============================================================================
#
# Runs Qwen3.6-NVFP4 generation with --mtp-spec-decode 1 across 4 prompt
# classes and reports the MTP draft accuracy for each. Acceptance rate
# tells us whether the current MTP forward (Phase 2.2.MoE + 2.2.Attn MVP)
# can produce drafts that match the main model's predictions.
#
# Run:
#   bash scripts/mtp_accuracy_bench.sh
#
# Decision thresholds (per docs/superpowers/specs/2026-05-14-mtp-wiring-design.md):
#   ≥ 60% on ≥ 3/4 prompt classes  → batched-verify Phase 3.5 is ROI-worthy
#   < 30% across the board         → Phase 2.2.Attn+KV (real attention with
#                                    MTP-side KV cache) is the blocker
# =============================================================================

set -euo pipefail

MODEL=${MTP_MODEL:-/home/kekz/models/Qwen3.6-35B-A3B-NVFP4}
MAX_TOKENS=${MTP_MAX_TOKENS:-128}
K=${MTP_K:-1}

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: model directory not found: $MODEL" >&2
    echo "Set MTP_MODEL or place Qwen3.6-NVFP4 at the default location." >&2
    exit 1
fi

if ! docker image inspect imp:test >/dev/null 2>&1; then
    echo "Building imp:test ..." >&2
    make build
fi

# Prompt classes (per the spec). Kept short so the test runs in reasonable time.
declare -A PROMPTS=(
    [factual]="What is the chemical formula for water, and what are its boiling and freezing points at standard atmospheric pressure?"
    [verbose-think]="Explain why the sky appears blue during the day but red during sunset. Walk me through the physics step by step."
    [code]="Write a Python function that computes the nth Fibonacci number using dynamic programming. Include a docstring."
    [instruction]="Compose a polite email to a colleague asking them to review a draft document by end of week. Keep it under 80 words."
)

echo
echo "=== MTP accuracy bench (Phase 5.5) ==="
echo "  model:         $MODEL"
echo "  max_tokens:    $MAX_TOKENS"
echo "  K:             $K"
echo "  prompt classes: ${!PROMPTS[*]}"
echo

declare -A RATES=()
declare -A MATCHES=()
declare -A TOTALS=()

for class in factual verbose-think code instruction; do
    prompt="${PROMPTS[$class]}"
    echo "--- $class ---"
    # Capture full output (don't tail-truncate) because the "mtp" summary
    # line is at the very end of generation, after decoded tokens may have
    # produced many lines of output.
    raw=$(docker run --rm --gpus all \
        -v /home/kekz/models:/home/kekz/models \
        imp:test imp-cli \
            --model "$MODEL" \
            --mtp-spec-decode "$K" \
            --prompt "$prompt" \
            --max-tokens "$MAX_TOKENS" \
            --temperature 0 \
            2>&1)
    # Parse "mtp     M / T drafts matched (P.P% accept rate)"
    line=$(echo "$raw" | grep -E "^mtp\s+[0-9]+ / [0-9]+ drafts matched" || true)
    if [[ -z "$line" ]]; then
        echo "  WARN: no mtp line in output"
        echo "$raw" | tail -5
        continue
    fi
    echo "  $line"
    m=$(echo "$line" | awk '{print $2}')
    t=$(echo "$line" | awk '{print $4}')
    r=$(echo "$line" | grep -oE '[0-9.]+%' | tr -d '%')
    MATCHES[$class]=$m
    TOTALS[$class]=$t
    RATES[$class]=$r
done

echo
echo "=== Summary ==="
printf '  %-15s  %8s  %8s  %s\n' "class" "matches" "total" "rate"
above_60=0
for class in factual verbose-think code instruction; do
    if [[ -n "${RATES[$class]:-}" ]]; then
        printf '  %-15s  %8s  %8s  %5s%%\n' \
            "$class" "${MATCHES[$class]}" "${TOTALS[$class]}" "${RATES[$class]}"
        # bash float comparison via awk
        if awk -v r="${RATES[$class]}" 'BEGIN { exit !(r >= 60) }'; then
            above_60=$((above_60 + 1))
        fi
    fi
done

echo
if [[ $above_60 -ge 3 ]]; then
    echo "RESULT: ≥ 3/4 prompt classes at ≥ 60% — batched-verify Phase 3.5 ROI-justified"
elif [[ $above_60 -ge 1 ]]; then
    echo "RESULT: $above_60/4 prompt classes at ≥ 60% — borderline; investigate dataset-specific behavior"
else
    echo "RESULT: 0/4 prompt classes at ≥ 60% — Phase 2.2.Attn+KV (proper MTP attention + KV cache)"
    echo "         is the prerequisite blocker before Phase 3.5 batched-verify pays off."
fi
