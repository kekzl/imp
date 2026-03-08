#!/bin/bash
# Download all GGUF models needed for imp benchmarks and testing
set -euo pipefail

MODELS=(
    "unsloth/Phi-4-mini-instruct-GGUF                Phi-4-mini-instruct.Q8_0.gguf"
    "unsloth/Qwen3-4B-Instruct-2507-GGUF             Qwen3-4B-Instruct-2507-Q8_0.gguf"
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF        DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"
    "unsloth/gemma-3-12b-it-GGUF                      gemma-3-12b-it-Q8_0.gguf"
    "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF        DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf"
    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF       Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf"
    "unsloth/Nemotron-3-Nano-30B-A3B-GGUF             Nemotron-3-Nano-30B-A3B-Q6_K.gguf"
)

echo "Downloading ${#MODELS[@]} models..."
echo ""

for entry in "${MODELS[@]}"; do
    repo=$(echo "$entry" | awk '{print $1}')
    file=$(echo "$entry" | awk '{print $2}')
    echo "── $repo ($file)"
    dev python hf download "$repo" "$file"
    echo ""
done

echo "All models downloaded."
