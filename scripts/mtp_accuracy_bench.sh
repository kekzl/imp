#!/usr/bin/env bash
# =============================================================================
# mtp_accuracy_bench.sh — Phase 5.5 validation harness
# =============================================================================
#
# Runs the released MTP checkpoint with --mtp-spec-decode 1 across 4 prompt
# classes and reports the MTP draft accuracy for each. Acceptance rate
# tells us whether the current MTP forward (Phase 2.2.MoE + 2.2.Attn MVP)
# can produce drafts that match the main model's predictions.
#
# Run:
#   bash scripts/mtp_accuracy_bench.sh
#
# Decision thresholds:
#   ≥ 60% on ≥ 3/4 prompt classes  → batched-verify Phase 3.5 is ROI-worthy
#   < 30% across the board         → Phase 2.2.Attn+KV (real attention with
#                                    MTP-side KV cache) is the blocker
# =============================================================================

set -euo pipefail

# Default: the checkpoint MTP is released for (docs/LIMITATIONS.md). It was
# Qwen3.6-35B-A3B-NVFP4, which is also on this box and also released, but the
# number this harness produces belongs beside the +21.3 % decode figure, and
# that figure is Qwen3.8-27B's.
MODEL=${MTP_MODEL:-$HOME/models/Qwen3.8-27B-NVFP4}
MAX_TOKENS=${MTP_MAX_TOKENS:-128}
K=${MTP_K:-1}

# The telemetry this harness reads (Engine::mtp_accuracy_, incremented in
# engine_scheduler.cpp's per-step block) only scores EAGER decode steps: it asks
# whether the head's depth-1 draft equals the token the main model then emits.
# With the verify loop running, most steps are verify steps and the counter sees
# almost nothing — and once the economics guard unbinds the head it sees nothing
# at all, which is how this script came to print "WARN: no mtp line" on
# Nemotron-3.5 while the head was drafting fine. So the verify loop is pinned
# off and the guard with it: teacher-forced accuracy is what the number means.
# speculative.hybrid=false is what removes the verify loop on a hybrid; on a
# non-hybrid the drafts still reach a verify chunk and the sample will be small,
# which the diagnosis below names rather than hides.
SPEC_PINS=(--set speculative.hybrid=false --set speculative.mtp_econ_min_emit=0)

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: model directory not found: $MODEL" >&2
    echo "Set MTP_MODEL, or stage the released checkpoint at the default location." >&2
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

FAILED_CLASSES=0
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
        -v $HOME/models:$HOME/models \
        imp:test imp-cli \
            --model "$MODEL" \
            --mtp-spec-decode "$K" \
            "${SPEC_PINS[@]}" \
            --prompt "$prompt" \
            --max-tokens "$MAX_TOKENS" \
            --temperature 0 \
            2>&1)
    # Parse "mtp     M / T drafts matched (P.P% accept rate)"
    line=$(echo "$raw" | grep -E "^mtp\s+[0-9]+ / [0-9]+ drafts matched" || true)
    if [[ -z "$line" ]]; then
        # "no mtp line" is a symptom with several causes and they need different
        # fixes. Name the one that fired instead of printing the last five lines
        # of a 400-line log and leaving the reader to guess.
        echo "  NO ACCURACY MEASURED — reason:"
        if grep -q "enable_mtp_spec_decode: model has no MTP head loaded" <<< "$raw"; then
            echo "    the checkpoint carries no MTP head, so --mtp-spec-decode did nothing."
        elif grep -q "MTP head present in this checkpoint but not loaded" <<< "$raw"; then
            echo "    the head was present but not loaded — --mtp-spec-decode did not reach the"
            echo "    loader (check the flag order and that this build accepts it)."
        elif grep -q "mtp-spec: drafting off" <<< "$raw"; then
            echo "    the head was UNBOUND mid-generation, so the eager telemetry stopped"
            echo "    counting. The engine says why:"
            grep -oE "mtp-spec: drafting off for req [0-9]+ \([^)]*\)" <<< "$raw" | head -1 |
                sed 's/^/      /'
            echo "    SPEC_PINS should prevent this; if it fired anyway the pin did not take."
        elif grep -q "spec-ngram: req .* gave up" <<< "$raw"; then
            echo "    speculation gave up for the request:"
            grep -oE "spec-ngram: req [0-9]+ gave up \([^)]*\)" <<< "$raw" | head -1 |
                sed 's/^/      /'
        elif grep -q "MTP spec-decode enabled" <<< "$raw"; then
            echo "    the head was enabled and stayed bound, but no eager decode step ever"
            echo "    scored a draft. On a non-hybrid the verify loop consumes every step and"
            echo "    speculative.hybrid=false does not remove it — measure this class with a"
            echo "    serving-path accept rate (/metrics) instead."
        else
            echo "    the engine never reported MTP at all — the run probably failed before"
            echo "    generation. Last lines:"
            tail -5 <<< "$raw" | sed 's/^/      /'
        fi
        FAILED_CLASSES=$((FAILED_CLASSES + 1))
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
if [[ $FAILED_CLASSES -gt 0 ]]; then
    echo "  ($FAILED_CLASSES of 4 classes measured nothing — see the reasons above;"
    echo "   the average below covers only the classes that produced a rate)"
    echo
fi
# Mean over the classes that actually produced a rate. Dividing a partial result
# by four would report a number lower than anything measured and read like a
# result rather than a gap.
measured=$((4 - FAILED_CLASSES))
if [[ $measured -eq 0 ]]; then
    echo "RESULT: nothing measured on any prompt class — the reasons are above."
    exit 1
fi
mean=$(awk -v a="${RATES[factual]:-0}" -v b="${RATES[verbose-think]:-0}" \
            -v c="${RATES[code]:-0}"    -v d="${RATES[instruction]:-0}" \
            -v n="$measured" 'BEGIN { printf "%.1f", (a + b + c + d) / n }')
if [[ $above_60 -ge 3 ]]; then
    echo "RESULT: ≥ 3/4 prompt classes at ≥ 60% (avg ${mean}%) — batched-verify Phase 3.5 ROI-justified"
elif awk -v m="$mean" 'BEGIN { exit !(m >= 15) }'; then
    echo "RESULT: avg ${mean}% accept rate (${above_60}/4 ≥ 60%) — real signal present."
    echo "         Below the ≥ 60%-on-3/4 default-on threshold, but batched-verify (Phase 3.5)"
    echo "         could still be a net win if the verify-forward cost is amortized across K drafts."
    echo "         Improving acceptance: RoPE, Q/K norm, multi-step (K>1) MTP forward chaining."
else
    echo "RESULT: 0/4 prompt classes at ≥ 60% (avg ${mean}%) — Phase 2.2.Attn+KV / RoPE / Q-K-norm"
    echo "         improvements needed before Phase 3.5 batched-verify pays off."
fi
