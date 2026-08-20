#!/usr/bin/env bash
# Compare imp vs llama.cpp output quality for complex structured prompts.
# Tests multiple prompts with different penalty settings.
# Usage: ./test_repetition_compare.sh [imp|llama|both]
set -euo pipefail

MODEL="/models/Qwen3-32B-Q4_K_M.gguf"
MAX_TOKENS=2000
TEMP=0.8

# Test prompts (complex structured output requests)
declare -a PROMPTS=(
  '{"role":"system","content":"Du bist ein YouTube-Shorts-Skript-Experte. Antworte AUSSCHLIESSLICH mit validem JSON: {\"hook\":\"...\",\"body_segments\":[\"s1\",\"s2\",\"s3\"],\"cta\":\"...\",\"tags\":[\"t1\"],\"title_options\":[\"t1\",\"t2\",\"t3\"],\"hook_score\":0.8}"}|{"role":"user","content":"Topic: Microsoft verspricht weniger KI-Nerverei"}'
  '{"role":"system","content":"Du bist ein SEO-Experte. Antworte AUSSCHLIESSLICH mit validem JSON: {\"title\":\"...\",\"meta_description\":\"...\",\"h1\":\"...\",\"keywords\":[\"k1\",\"k2\"],\"word_count\":1500,\"outline\":[\"section1\",\"section2\"]}"}|{"role":"user","content":"Topic: Beste Gaming-Monitore 2026"}'
  '{"role":"system","content":"Du bist ein Rezept-Generator. Antworte AUSSCHLIESSLICH mit validem JSON: {\"name\":\"...\",\"servings\":4,\"prep_time\":\"30min\",\"ingredients\":[{\"item\":\"...\",\"amount\":\"...\"}],\"steps\":[\"step1\",\"step2\"],\"nutrition\":{\"calories\":500}}"}|{"role":"user","content":"Erstelle ein veganes Curry-Rezept"}'
)

PROMPT_NAMES=("youtube-shorts" "seo-article" "recipe-json")

# Penalty configurations to test
declare -a CONFIGS=(
  "none|1.0|0.0|0.0|0.0"
  "rep1.1|1.1|0.0|0.0|0.0"
  "rep1.2|1.2|0.0|0.0|0.0"
  "freq0.5|1.0|0.5|0.0|0.0"
  "pres0.5|1.0|0.0|0.5|0.0"
  "dry0.8|1.0|0.0|0.0|0.8"
  "combo|1.1|0.3|0.3|0.5"
)

run_imp_test() {
    local prompt_pair="$1"
    local config="$2"
    local name="$3"

    IFS='|' read -r label rep_pen freq_pen pres_pen dry_mul <<< "$config"
    IFS='|' read -r sys_msg user_msg <<< "$prompt_pair"

    local body
    body=$(cat <<ENDJSON
{
  "model": "Qwen3-32B-Q4_K_M.gguf",
  "messages": [$sys_msg, $user_msg],
  "temperature": $TEMP,
  "max_tokens": $MAX_TOKENS,
  "think_budget": 0,
  "repetition_penalty": $rep_pen,
  "frequency_penalty": $freq_pen,
  "presence_penalty": $pres_pen,
  "dry_multiplier": $dry_mul
}
ENDJSON
)

    local t0 resp content finish tokens
    t0=$(date +%s%3N)
    resp=$(curl -s http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "$body" 2>&1)
    local elapsed=$(( $(date +%s%3N) - t0 ))

    finish=$(echo "$resp" | jq -r '.choices[0].finish_reason // "error"')
    tokens=$(echo "$resp" | jq -r '.usage.completion_tokens // 0')
    content=$(echo "$resp" | jq -r '.choices[0].message.content // "ERROR"')

    # Check if content is valid JSON (strip markdown code fences)
    local clean_content
    clean_content=$(echo "$content" | sed 's/^```json//;s/^```//;s/```$//' | sed '/^$/d')
    local is_valid_json="NO"
    if echo "$clean_content" | jq . >/dev/null 2>&1; then
        is_valid_json="YES"
    fi

    # Check for repetition (same 3+ char pattern repeated 5+ times)
    local has_repetition="NO"
    # Herestring: grep -q closes the pipe at the first match, echo dies of EPIPE,
    # and pipefail turns a MATCH into a miss. $content is a whole completion, so
    # here that would report NO repetition on exactly the output that has it.
    if grep -qP '(.{3,})\1{4,}' <<< "$content"; then
        has_repetition="YES"
    fi

    printf "%-15s %-10s | finish=%-8s tokens=%-5s json=%-3s repeat=%-3s %5dms\n" \
        "$name" "$label" "$finish" "$tokens" "$is_valid_json" "$has_repetition" "$elapsed"

    # Save output for inspection
    local outdir="test_outputs"
    mkdir -p "$outdir"
    echo "$content" > "$outdir/${name}_${label}.txt"
}

echo "=== imp Server Repetition Test ==="
echo "Model: Qwen3-32B-Q4_K_M  Temp: $TEMP  MaxTokens: $MAX_TOKENS"
echo "================================================================"
printf "%-15s %-10s | %-14s %-11s %-9s %-10s %s\n" \
    "PROMPT" "CONFIG" "FINISH" "TOKENS" "JSON?" "REPEAT?" "TIME"
echo "----------------------------------------------------------------"

for i in "${!PROMPTS[@]}"; do
    for config in "${CONFIGS[@]}"; do
        run_imp_test "${PROMPTS[$i]}" "$config" "${PROMPT_NAMES[$i]}"
    done
    echo "---"
done

echo ""
echo "Outputs saved to test_outputs/"
echo "Check: ls test_outputs/*.txt"
