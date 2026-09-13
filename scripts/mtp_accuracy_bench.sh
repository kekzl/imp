#!/usr/bin/env bash
# Runs the released MTP checkpoint with --mtp-spec-decode 1 across 4 prompt classes, reports
# MTP draft acceptance rate (measures whether the current MTP forward matches the main model).
# Decision: >=60% on >=3/4 classes -> batched-verify is ROI-worthy; <30% across the board ->
# real attention with MTP-side KV cache is the blocker.

set -euo pipefail

# Default checkpoint MTP is released for (docs/LIMITATIONS.md): Qwen3.8-27B-NVFP4-vllm.
# The decode-lift figure this harness's number sits beside belongs to that model, not
# Qwen3.6-35B-A3B-NVFP4 (also on this box, also released).
MODEL=${MTP_MODEL:-$HOME/models/Qwen3.8-27B-NVFP4-vllm}
MAX_TOKENS=${MTP_MAX_TOKENS:-128}
K=${MTP_K:-1}

# mtp_accuracy_ (Engine::mtp_accuracy_, engine_scheduler.cpp) scores only EAGER decode steps
# (head's depth-1 draft vs the main model's next token); with the verify loop running, or the
# economics guard unbinding the head, it sees almost nothing.
# speculative.hybrid=false removes the verify loop on a hybrid so this measures teacher-forced
# accuracy; on a non-hybrid the sample stays small (diagnosed, not hidden).
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
