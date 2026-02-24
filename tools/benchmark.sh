#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMP_CLI="$ROOT_DIR/build/imp-cli"
LLAMA_BENCH="/home/kekz/llama.cpp/build/bin/llama-bench"
REPORT_DIR="$ROOT_DIR/benchmarks"
REPORT="$REPORT_DIR/report.md"

PP_TOKENS=512          # prompt tokens for llama-bench (imp uses actual prompt length)
TG_TOKENS=128          # tokens to generate
LLAMA_REPS=3           # repetitions for llama-bench
TEMPERATURE=0          # greedy decoding for reproducibility

# Models: name | path | quant
declare -a MODEL_NAMES=(
    "Qwen3-4B-Instruct"
    "Qwen3-Coder-30B-A3B-Instruct"
    "Nemotron-3-Nano-30B-A3B"
)
declare -a MODEL_PATHS=(
    "$ROOT_DIR/models/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q8_0.gguf"
    "$ROOT_DIR/models/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/b17cb02dd882d5b6ab62fc777ad2995f19668350/Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf"
    "$ROOT_DIR/models/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/9ad8b366c308f931b2a96b9306f0b41aef9cd405/Nemotron-3-Nano-30B-A3B-Q6_K.gguf"
)
declare -a MODEL_QUANTS=(
    "Q8_0"
    "Q6_K"
    "Q6_K"
)

# Long prompt (~500 tokens) for imp-cli
PROMPT="Explain the complete history of the development of the transformer architecture in deep learning, starting from the original 2017 paper 'Attention Is All You Need' by Vaswani et al. Cover the key innovations including self-attention mechanisms, multi-head attention, positional encoding, and the encoder-decoder structure. Then discuss how this architecture evolved through BERT, GPT, GPT-2, GPT-3, T5, and other notable models. Explain the scaling laws discovered by Kaplan et al. and how they influenced the development of increasingly large language models. Discuss the emergence of capabilities like in-context learning, chain-of-thought reasoning, and instruction following. Cover the technical challenges of training large transformers including distributed training, mixed precision, gradient checkpointing, and data parallelism. Explain the role of RLHF in aligning language models with human preferences. Discuss the hardware implications including the shift toward specialized AI accelerators and the memory bandwidth bottleneck in inference. Finally, analyze current trends in efficient inference including quantization, speculative decoding, KV cache optimization, paged attention, and continuous batching. Provide specific examples and technical details throughout your explanation."

# ─── Preflight ───────────────────────────────────────────────────────────────
echo "=== imp vs llama.cpp Benchmark ==="
echo ""

if [[ ! -x "$IMP_CLI" ]]; then
    echo "ERROR: imp-cli not found at $IMP_CLI"
    echo "Build with: cmake --build build -j\$(nproc)"
    exit 1
fi

if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "WARNING: llama-bench not found at $LLAMA_BENCH"
    echo "Will skip llama.cpp benchmarks."
    HAS_LLAMA=0
else
    HAS_LLAMA=1
fi

mkdir -p "$REPORT_DIR"

# ─── Gather system info ─────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' || /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' || echo "Unknown")

echo "GPU: $GPU_NAME ($GPU_VRAM)"
echo "Driver: $GPU_DRIVER | CUDA: $CUDA_VER"
echo ""

# ─── Helper: parse imp-cli stderr output ─────────────────────────────────────
# Expects lines like:
#   pp   523 tokens in  1234.56 ms  ( 423.88 tok/s)
#   tg   128 tokens in  5678.90 ms  (  22.54 tok/s)
parse_imp_output() {
    local stderr_file="$1"
    IMP_PP_TOKS=$(grep "^pp " "$stderr_file" | sed 's/.*(\s*\([0-9.]*\) tok\/s).*/\1/' || echo "0")
    IMP_TG_TOKS=$(grep "^tg " "$stderr_file" | sed 's/.*(\s*\([0-9.]*\) tok\/s).*/\1/' || echo "0")
    IMP_PP_N=$(grep "^pp " "$stderr_file" | awk '{print $2}' || echo "0")
    IMP_TG_N=$(grep "^tg " "$stderr_file" | awk '{print $2}' || echo "0")
}

# ─── Helper: parse llama-bench JSON output ────────────────────────────────────
parse_llama_output() {
    local json_file="$1"
    # llama-bench -o json outputs a JSON array of results
    # Each entry has "test" (pp or tg), "avg_ts" (tokens/sec)
    LLAMA_PP_TOKS=$(python3 -c "
import json, sys
data = json.load(open('$json_file'))
pp = [r for r in data if r.get('n_prompt', 0) > 0 and r.get('n_gen', 0) == 0]
if pp: print(f\"{pp[0]['avg_ts']:.2f}\")
else: print('N/A')
" 2>/dev/null || echo "N/A")
    LLAMA_TG_TOKS=$(python3 -c "
import json, sys
data = json.load(open('$json_file'))
tg = [r for r in data if r.get('n_gen', 0) > 0 and r.get('n_prompt', 0) == 0]
if tg: print(f\"{tg[0]['avg_ts']:.2f}\")
else: print('N/A')
" 2>/dev/null || echo "N/A")
}

# ─── Begin report ────────────────────────────────────────────────────────────
cat > "$REPORT" << EOF
# imp vs llama.cpp Benchmark Report

**Date:** $(date -u '+%Y-%m-%d %H:%M UTC')
**GPU:** $GPU_NAME ($GPU_VRAM)
**Driver:** $GPU_DRIVER | **CUDA:** $CUDA_VER
**OS:** $(uname -sr)

## Methodology

- **Prompt processing (pp):** llama-bench uses synthetic $PP_TOKENS-token prompt; imp uses a real ~500-token text prompt
- **Text generation (tg):** $TG_TOKENS tokens, temperature=$TEMPERATURE (greedy)
- **llama-bench:** $LLAMA_REPS repetitions, flash attention enabled, all layers on GPU
- **imp:** single run, all layers on GPU, chat template disabled for fair comparison
- Both engines: batch size 1, single sequence

## Results

| Model | Quant | Engine | pp tok/s | tg tok/s |
|-------|-------|--------|----------|----------|
EOF

# ─── Run benchmarks ──────────────────────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

for i in "${!MODEL_NAMES[@]}"; do
    name="${MODEL_NAMES[$i]}"
    path="${MODEL_PATHS[$i]}"
    quant="${MODEL_QUANTS[$i]}"

    echo "━━━ $name ($quant) ━━━"

    if [[ ! -f "$path" ]]; then
        echo "  SKIP: model file not found at $path"
        echo "| $name | $quant | imp | N/A | N/A |" >> "$REPORT"
        echo "| $name | $quant | llama.cpp | N/A | N/A |" >> "$REPORT"
        continue
    fi

    # ── llama-bench ──
    if [[ $HAS_LLAMA -eq 1 ]]; then
        echo "  Running llama-bench..."
        LLAMA_JSON="$TMPDIR/llama_${i}.json"
        if "$LLAMA_BENCH" \
            -m "$path" \
            -p "$PP_TOKENS" -n "$TG_TOKENS" \
            -ngl 99 -fa 1 \
            -r "$LLAMA_REPS" \
            -o json > "$LLAMA_JSON" 2>/dev/null; then
            parse_llama_output "$LLAMA_JSON"
            echo "  llama.cpp: pp=$LLAMA_PP_TOKS tok/s, tg=$LLAMA_TG_TOKS tok/s"
        else
            LLAMA_PP_TOKS="ERR"
            LLAMA_TG_TOKS="ERR"
            echo "  llama-bench failed!"
        fi
    else
        LLAMA_PP_TOKS="N/A"
        LLAMA_TG_TOKS="N/A"
    fi

    # ── imp-cli ──
    echo "  Running imp-cli..."
    IMP_STDERR="$TMPDIR/imp_${i}.stderr"
    IMP_STDOUT="$TMPDIR/imp_${i}.stdout"
    if "$IMP_CLI" \
        --model "$path" \
        --prompt "$PROMPT" \
        --max-tokens "$TG_TOKENS" \
        --temperature "$TEMPERATURE" \
        --chat-template none \
        > "$IMP_STDOUT" 2> "$IMP_STDERR"; then
        parse_imp_output "$IMP_STDERR"
        echo "  imp:       pp=$IMP_PP_TOKS tok/s (${IMP_PP_N} tokens), tg=$IMP_TG_TOKS tok/s (${IMP_TG_N} tokens)"
    else
        IMP_PP_TOKS="ERR"
        IMP_TG_TOKS="ERR"
        echo "  imp-cli failed! stderr:"
        cat "$IMP_STDERR" | tail -5
    fi

    # Write to report
    echo "| $name | $quant | llama.cpp | $LLAMA_PP_TOKS | $LLAMA_TG_TOKS |" >> "$REPORT"
    echo "| $name | $quant | imp | $IMP_PP_TOKS | $IMP_TG_TOKS |" >> "$REPORT"
    echo ""
done

# ─── Finalize report ─────────────────────────────────────────────────────────
cat >> "$REPORT" << 'EOF'

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- llama.cpp uses synthetic prompt tokens; imp tokenizes a real text prompt, so pp token counts may differ slightly
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp uses Blackwell-optimized TCGEN05 attention on sm_120; llama.cpp uses its own Flash Attention
EOF

echo "━━━ Done ━━━"
echo "Report saved to: $REPORT"
