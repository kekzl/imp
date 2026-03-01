#!/usr/bin/env bash
# benchmark.sh — Run decode throughput benchmarks across all available models.
# Compares imp (baseline vs NVFP4 decode) and optionally llama.cpp.
#
# Usage: ./tools/benchmark.sh [options]
#
# Options:
#   --reps N       Benchmark repetitions (default: 3)
#   --tg N         Decode tokens to generate (default: 128)
#   --pp N         Prefill token count (default: 128)
#   --model NAME   Run only models matching NAME (substring match)
#   --no-llama     Skip llama-bench even if available
#   --quick        Quick mode: 1 rep, 64 tg tokens

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$ROOT_DIR/build/imp-cli"
LLAMA_BENCH="${LLAMA_BENCH:-$(command -v llama-bench 2>/dev/null || echo "${HOME}/llama.cpp/build/bin/llama-bench")}"
MODELS_DIR="$ROOT_DIR/models"

# Defaults
REPS=3
TG=128
PP=128
FILTER=""
NO_LLAMA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps)     REPS="$2"; shift 2 ;;
        --tg)       TG="$2"; shift 2 ;;
        --pp)       PP="$2"; shift 2 ;;
        --model)    FILTER="$2"; shift 2 ;;
        --no-llama) NO_LLAMA=1; shift ;;
        --quick)    REPS=1; TG=64; shift ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$CLI" ]]; then
    echo "ERROR: imp-cli not found at $CLI — run cmake --build build first"
    exit 1
fi

# Auto-detect llama-bench
HAS_LLAMA=0
if [[ $NO_LLAMA -eq 0 && -x "$LLAMA_BENCH" ]]; then
    HAS_LLAMA=1
fi

# ── Model registry ──────────────────────────────────────────────────────────
# Format: short_name|dir_glob|extra_flags
# Ordered by model size (smallest first).
MODELS=(
    "Phi-4-mini Q8_0|models--unsloth--Phi-4-mini-instruct-GGUF|"
    "Qwen3-4B Q8_0|models--unsloth--Qwen3-4B-Instruct-2507-GGUF|"
    "DS-R1-7B Q8_0|models--unsloth--DeepSeek-R1-Distill-Qwen-7B-GGUF|"
    "Gemma-3-12B Q8_0|models--unsloth--gemma-3-12b-it-GGUF|--chat-template gemma"
    "DS-R1-14B Q6_K|models--unsloth--DeepSeek-R1-Distill-Qwen-14B-GGUF|"
    "Qwen3-Coder-30B-A3B Q6_K|models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF|"
    "Nemotron-30B-A3B Q6_K|models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF|"
)

# ── Helpers ─────────────────────────────────────────────────────────────────
find_gguf() {
    find "$MODELS_DIR/$1" -name "*.gguf" -not -path "*/.no_exist/*" 2>/dev/null | head -1
}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Run imp-cli --bench and parse results.
# Sets: R_PP R_TG R_FP16 R_NV4 R_VRAM R_FP16_N R_NV4_N R_ERR
run_imp() {
    local gguf="$1"; shift
    local extra_flags="$1"; shift
    local mode_flags=("$@")
    local out="$TMPDIR/run.out"

    R_PP="—" R_TG="—" R_FP16="—" R_NV4="—" R_VRAM="—"
    R_FP16_N="—" R_NV4_N="—" R_ERR=""

    if ! "$CLI" --model "$gguf" --bench --bench-pp "$PP" --max-tokens "$TG" \
         --bench-reps "$REPS" --seed 42 $extra_flags "${mode_flags[@]}" \
         >"$out" 2>&1; then
        R_ERR=$(grep -iE "error|failed|cannot|CUDA" "$out" | head -1)
        R_ERR="${R_ERR:-FAILED}"
        return 1
    fi

    R_PP=$(grep -oP  'pp\s+\d+.*?\(\s*\K[\d.]+(?=\s*tok/s\))'       "$out" | head -1)
    R_TG=$(grep -oP  'tg\s+\d+.*?\(\s*\K[\d.]+(?=\s*tok/s\))'       "$out" | head -1)
    R_FP16=$(grep -oP 'FP16 weight cache: \d+ tensors, \K[\d.]+'     "$out" | head -1)
    R_NV4=$(grep -oP  'NVFP4 decode cache: \d+ tensors, \K[\d.]+'    "$out" | head -1)
    R_VRAM=$(grep -oP 'GPU memory: \K\d+(?= MiB used)'               "$out" | head -1)
    R_FP16_N=$(grep -oP 'FP16 weight cache: \K\d+'                   "$out" | head -1)
    R_NV4_N=$(grep -oP  'NVFP4 decode cache: \K\d+'                  "$out" | head -1)

    R_PP="${R_PP:-—}"; R_TG="${R_TG:-—}"; R_FP16="${R_FP16:-—}"
    R_NV4="${R_NV4:-—}"; R_VRAM="${R_VRAM:-?}"
    R_FP16_N="${R_FP16_N:-0}"; R_NV4_N="${R_NV4_N:-0}"
    return 0
}

# Run llama-bench and parse results.
# Sets: LL_PP LL_TG
run_llama() {
    local gguf="$1"
    local out="$TMPDIR/llama.json"
    LL_PP="—" LL_TG="—"

    if ! "$LLAMA_BENCH" -m "$gguf" -p "$PP" -n "$TG" -ngl 99 -fa 1 \
         -r "$REPS" -o json > "$out" 2>/dev/null; then
        return 1
    fi

    LL_PP=$(python3 -c "
import json
data = json.load(open('$out'))
pp = [r for r in data if r.get('n_prompt',0) > 0 and r.get('n_gen',0) == 0]
print(f'{pp[0][\"avg_ts\"]:.2f}' if pp else '—')
" 2>/dev/null || echo "—")
    LL_TG=$(python3 -c "
import json
data = json.load(open('$out'))
tg = [r for r in data if r.get('n_gen',0) > 0 and r.get('n_prompt',0) == 0]
print(f'{tg[0][\"avg_ts\"]:.2f}' if tg else '—')
" 2>/dev/null || echo "—")
}

fmt_delta() {
    local base="$1" new="$2"
    [[ "$base" == "—" || "$new" == "—" ]] && return
    python3 -c "
b, n = float('$base'), float('$new')
if b > 0:
    d = (n - b) / b * 100
    print(f'+{d:.1f}%' if d >= 0 else f'{d:.1f}%')
" 2>/dev/null
}

# ── Table formatting ────────────────────────────────────────────────────────
SEP="───────────────────────────────────────────────────────────────────────────────────────────────────────────────"

print_header() {
    echo "$SEP"
    printf "%-28s  %-9s  %9s %9s  %14s %14s  %8s  %s\n" \
        "Model" "Engine" "pp tok/s" "tg tok/s" "FP16 cache" "NVFP4 cache" "VRAM" "vs base"
    echo "$SEP"
}

print_row() {
    local name="$1" engine="$2" pp="$3" tg="$4"
    local fp16_n="$5" fp16_m="$6" nv4_n="$7" nv4_m="$8"
    local vram="$9" delta="${10:-}"

    local fp16_str="—" nv4_str="—" vram_str="—"
    [[ "$fp16_m" != "—" ]] && fp16_str="${fp16_n}t ${fp16_m}M"
    [[ "$nv4_m"  != "—" ]] && nv4_str="${nv4_n}t ${nv4_m}M"
    [[ "$vram" != "?" && "$vram" != "—" ]] && vram_str="${vram}M"

    printf "%-28s  %-9s  %9s %9s  %14s %14s  %8s  %s\n" \
        "$name" "$engine" "$pp" "$tg" "$fp16_str" "$nv4_str" "$vram_str" "$delta"
}

# ── Main ────────────────────────────────────────────────────────────────────
echo "imp benchmark — $(date '+%Y-%m-%d %H:%M') — $GPU_NAME (${GPU_VRAM} MiB)"
echo "Settings: pp=$PP tg=$TG reps=$REPS"
if [[ $HAS_LLAMA -eq 1 ]]; then
    echo "llama-bench: $LLAMA_BENCH"
else
    echo "llama-bench: not found (use LLAMA_BENCH= to set path)"
fi
echo ""
print_header

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name dir_pattern extra_flags <<< "$entry"

    # Filter
    if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
        continue
    fi

    gguf=$(find_gguf "$dir_pattern")
    if [[ -z "$gguf" ]]; then
        printf "%-28s  SKIP (not found)\n" "$name"
        echo ""
        continue
    fi

    # ── llama-bench ──
    if [[ $HAS_LLAMA -eq 1 ]]; then
        if run_llama "$gguf"; then
            print_row "$name" "llama.cpp" "$LL_PP" "$LL_TG" "—" "—" "—" "—" "—" ""
        else
            print_row "$name" "llama.cpp" "ERR" "ERR" "—" "—" "—" "—" "—" ""
        fi
    fi

    # ── imp modes ──
    base_tg="—"
    for mode_entry in "base|" "nvfp4|--decode-nvfp4"; do
        IFS='|' read -r mode_label mode_flags <<< "$mode_entry"

        local_engine="imp"
        [[ "$mode_label" != "base" ]] && local_engine="imp+$mode_label"

        # shellcheck disable=SC2086
        if run_imp "$gguf" "$extra_flags" $mode_flags; then
            delta=""
            if [[ "$mode_label" == "base" ]]; then
                base_tg="$R_TG"
                # Show vs llama delta on base row
                if [[ $HAS_LLAMA -eq 1 && "$LL_TG" != "—" && "$R_TG" != "—" ]]; then
                    delta=$(fmt_delta "$LL_TG" "$R_TG")
                fi
            else
                # Show vs imp base delta
                delta=$(fmt_delta "$base_tg" "$R_TG")
            fi
            print_row "$name" "$local_engine" "$R_PP" "$R_TG" \
                "$R_FP16_N" "$R_FP16" "$R_NV4_N" "$R_NV4" "$R_VRAM" "$delta"
        else
            printf "%-28s  %-9s  %s\n" "$name" "$local_engine" "$R_ERR"
        fi
    done
    echo ""
done

echo "$SEP"
echo "FP16 cache = prefill weight cache  |  NVFP4 cache = decode weight cache"
echo "vs base: imp row = vs llama.cpp tg  |  imp+nvfp4 row = vs imp base tg"
echo "Done."
