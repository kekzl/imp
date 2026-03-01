#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMP_CLI="$ROOT_DIR/build/imp-cli"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"

if [[ ! -x "$IMP_CLI" ]]; then
    echo "ERROR: imp-cli not found. Build with: cmake --build build -j\$(nproc)"
    exit 1
fi

# ── Find all GGUF models ─────────────────────────────────────────────────────
declare -a NAMES=()
declare -a PATHS=()

while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    real="$(readlink -f "$path")"
    name="$(basename "$real" .gguf)"
    NAMES+=("$name")
    PATHS+=("$real")
done < <(find "$ROOT_DIR/models" "$HF_CACHE" -name "*.gguf" \( -type f -o -type l \) 2>/dev/null | sort)

if [[ ${#NAMES[@]} -eq 0 ]]; then
    echo "No GGUF models found in models/ or HF cache."
    exit 1
fi

# ── Model picker ──────────────────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────┐"
echo "│  imp chat — select a model                         │"
echo "└─────────────────────────────────────────────────────┘"
echo ""
for i in "${!NAMES[@]}"; do
    size=$(du -h "${PATHS[$i]}" 2>/dev/null | cut -f1)
    printf "  %2d) %-50s [%s]\n" $((i+1)) "${NAMES[$i]}" "$size"
done
echo ""
read -rp "Model [1-${#NAMES[@]}]: " choice

if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#NAMES[@]} )); then
    echo "Invalid choice."
    exit 1
fi

MODEL_PATH="${PATHS[$((choice-1))]}"
MODEL_NAME="${NAMES[$((choice-1))]}"

echo ""
echo "Loading: $MODEL_NAME"
echo "Path:    $MODEL_PATH"
echo ""

# ── Launch interactive chat ───────────────────────────────────────────────────
exec "$IMP_CLI" \
    --model "$MODEL_PATH" \
    --interactive \
    --chat-template auto \
    "$@"
