#!/usr/bin/env bash
# chat.sh — Interactive chat or smoke-test for imp models.
#
# Usage:
#   ./tools/chat.sh                        Select a model and chat
#   ./tools/chat.sh --test                 Smoke-test all models
#   ./tools/chat.sh --test --model gemma   Test only Gemma models
#
# Options:
#   --test               Run each model with diverse prompts (non-interactive)
#   --model <substr>     Only show/test models matching substring
#   --max-tokens <n>     Max tokens in test mode (default: 128)
#   --help               Show this help
#   --                   Pass remaining args to imp-cli

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMP_CLI="$ROOT_DIR/build/imp-cli"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"

# ── Options ──────────────────────────────────────────────────────────────────
TEST_MODE=0
FILTER=""
MAX_TOKENS=128

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)       TEST_MODE=1; shift ;;
        --model)      FILTER="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# \?//; p }' "$0"
            exit 0
            ;;
        --)           shift; break ;;
        *)            break ;;
    esac
done
IMP_EXTRA=("$@")

if [[ ! -x "$IMP_CLI" ]]; then
    echo "ERROR: imp-cli not found at $IMP_CLI"
    echo "Build with: cmake --build build -j\$(nproc)"
    exit 1
fi

# ── Template family name (informational — actual detection is engine-side) ───
template_hint() {
    local lower="${1,,}"
    if   [[ "$lower" == *gemma* ]];    then echo "gemma"
    elif [[ "$lower" == *nemotron* ]]; then echo "nemotron"
    elif [[ "$lower" == *llama-3* || "$lower" == *llama3* ]]; then echo "llama3"
    elif [[ "$lower" == *llama*   || "$lower" == *mistral* || "$lower" == *mixtral* ]]; then echo "llama2"
    elif [[ "$lower" == *deepseek* ]]; then echo "deepseek_r1"
    elif [[ "$lower" == *phi* ]]; then echo "phi"
    else echo "chatml"
    fi
}

# ── Discover GGUF models ────────────────────────────────────────────────────
declare -a M_NAMES=() M_PATHS=() M_SIZES=() M_HINTS=()
declare -A SEEN=()

while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    [[ "$path" == */.no_exist/* ]] && continue          # HF cache phantom entries

    real="$(readlink -f "$path" 2>/dev/null)" || continue
    [[ ! -f "$real" ]] && continue
    [[ -n "${SEEN[$real]+x}" ]] && continue             # deduplicate
    SEEN["$real"]=1

    name="$(basename "$path" .gguf)"                    # symlink name, not blob hash
    hint=$(template_hint "$name")
    size=$(du -h "$real" 2>/dev/null | cut -f1)

    if [[ -n "$FILTER" && "${name,,}" != *"${FILTER,,}"* ]]; then
        continue
    fi

    M_NAMES+=("$name")
    M_PATHS+=("$real")
    M_SIZES+=("$size")
    M_HINTS+=("$hint")
done < <(find "$ROOT_DIR/models" "$HF_CACHE" -name "*.gguf" \( -type f -o -type l \) 2>/dev/null | sort -u)

if [[ ${#M_NAMES[@]} -eq 0 ]]; then
    echo "No GGUF models found in models/ or HF cache."
    [[ -n "$FILTER" ]] && echo "  (filter: '$FILTER')"
    exit 1
fi

# ── Extract generated text from imp-cli stdout ──────────────────────────────
# IMP_LOG_INFO goes to stdout alongside generated text.  Strip log lines.
extract_response() {
    sed 's/\[[-0-9:]*\]\[\(INFO\|DEBUG\|WARN\|ERROR\)\] [^ ]*: .*//g' \
    | grep -vE '^\[.*\]\[(INFO|DEBUG|WARN|ERROR)\]' \
    | grep -v '^IMP Inference Engine' \
    | grep -v '^Loading model:' \
    | sed '/^$/d'
}

# ══════════════════════════════════════════════════════════════════════════════
# Interactive mode
# ══════════════════════════════════════════════════════════════════════════════
if [[ $TEST_MODE -eq 0 ]]; then
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────────┐"
    echo "│  imp chat                                                           │"
    echo "└──────────────────────────────────────────────────────────────────────┘"
    echo ""
    printf "  \e[1m%-3s %-48s %6s  %-8s\e[0m\n" "#" "Model" "Size" "Template"
    printf "  %-3s %-48s %6s  %-8s\n" "---" "------------------------------------------------" "------" "--------"
    for i in "${!M_NAMES[@]}"; do
        printf "  %2d) %-48s %6s  %s\n" \
            $((i+1)) "${M_NAMES[$i]}" "${M_SIZES[$i]}" "${M_HINTS[$i]}"
    done
    echo ""
    read -rp "  Model [1-${#M_NAMES[@]}]: " choice

    if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#M_NAMES[@]} )); then
        echo "Invalid choice."
        exit 1
    fi

    idx=$((choice - 1))
    echo ""
    echo "  Model:    ${M_NAMES[$idx]}"
    echo "  Template: ${M_HINTS[$idx]} (auto-detected from GGUF)"
    echo "  Path:     ${M_PATHS[$idx]}"
    echo ""

    # Use --chat-template auto: the engine reads the Jinja template from GGUF
    # metadata and resolves special token IDs from the vocabulary.
    # Explicit template names (chatml, gemma, ...) can fail when the tokenizer
    # uses non-standard token strings.
    exec "$IMP_CLI" \
        --model "${M_PATHS[$idx]}" \
        --interactive \
        --chat-template auto \
        ${IMP_EXTRA[@]+"${IMP_EXTRA[@]}"}
fi

# ══════════════════════════════════════════════════════════════════════════════
# Test mode: smoke-test each model with diverse prompts
# ══════════════════════════════════════════════════════════════════════════════
TEST_PROMPTS=(
    "What is 2+2? Answer with just the number."
    "Write a short Python function that checks if a number is prime."
    "Explain why the sky is blue in one sentence."
    "Write a haiku about programming."
)

SEP="════════════════════════════════════════════════════════════════════════════"
THIN="────────────────────────────────────────────────────────────────────────────"

echo ""
echo "$SEP"
printf "  imp model smoke-test — %d models, %d prompts, max_tokens=%d\n" \
    "${#M_NAMES[@]}" "${#TEST_PROMPTS[@]}" "$MAX_TOKENS"
echo "$SEP"

pass=0
fail=0
tmpout=$(mktemp)
tmperr=$(mktemp)
trap "rm -f '$tmpout' '$tmperr'" EXIT

for i in "${!M_NAMES[@]}"; do
    echo ""
    echo "$THIN"
    printf "  \e[1m[%d/%d] %s\e[0m  (%s, %s)\n" \
        $((i+1)) "${#M_NAMES[@]}" "${M_NAMES[$i]}" "${M_HINTS[$i]}" "${M_SIZES[$i]}"
    echo "$THIN"

    model_ok=1

    for j in "${!TEST_PROMPTS[@]}"; do
        prompt="${TEST_PROMPTS[$j]}"
        echo ""
        printf "  \e[36mPrompt %d:\e[0m %s\n" $((j+1)) "$prompt"

        if "$IMP_CLI" \
            --model "${M_PATHS[$i]}" \
            --prompt "$prompt" \
            --chat-template auto \
            --max-tokens "$MAX_TOKENS" \
            --temperature 0 \
            --seed 42 \
            ${IMP_EXTRA[@]+"${IMP_EXTRA[@]}"} \
            >"$tmpout" 2>"$tmperr"; then

            # Extract just the generated text (strip log lines from stdout)
            response=$(extract_response < "$tmpout")

            # Show warnings from stderr (template fallbacks, etc.)
            warnings=$(grep -i '\[WARN\]' "$tmperr" 2>/dev/null || true)
            if [[ -n "$warnings" ]]; then
                printf "  \e[33mWarning:\e[0m %s\n" "$(echo "$warnings" | head -1 | sed 's/.*\] //')"
            fi

            if [[ -z "$response" ]]; then
                printf "  \e[31m✗ Empty response\e[0m\n"
                model_ok=0
            else
                printf "  \e[32mResponse:\e[0m\n"
                echo "$response" | head -20 | sed 's/^/    /'
                lines=$(echo "$response" | wc -l)
                (( lines > 20 )) && echo "    ... ($lines lines total)"
            fi

            # Show timing from stderr
            timing=$(grep -oP '(pp|tg)\s+\d+.*tok/s\)' "$tmperr" 2>/dev/null | tail -2 || true)
            if [[ -n "$timing" ]]; then
                echo ""
                echo "$timing" | sed 's/^/    /'
            fi
        else
            printf "  \e[31m✗ FAILED (exit code $?)\e[0m\n"
            grep -iE 'error|failed|cannot|CUDA' "$tmperr" 2>/dev/null | head -3 | sed 's/^/    /' || true
            model_ok=0
        fi
    done

    echo ""
    if [[ $model_ok -eq 1 ]]; then
        printf "  \e[32m→ PASS\e[0m\n"
        pass=$((pass + 1))
    else
        printf "  \e[31m→ FAIL\e[0m\n"
        fail=$((fail + 1))
    fi
done

echo ""
echo "$SEP"
if [[ $fail -eq 0 ]]; then
    printf "  \e[32mAll %d models passed.\e[0m\n" "$pass"
else
    printf "  \e[32m%d passed\e[0m, \e[31m%d failed\e[0m (of %d models)\n" \
        "$pass" "$fail" "${#M_NAMES[@]}"
fi
echo "$SEP"

exit $((fail > 0 ? 1 : 0))
