#!/bin/bash
# FP8 vs FP16 prefill weight cache benchmark
# Runs pp512 benchmark on all models, comparing default (FP16) vs --prefill-fp8

set -e

CLI=./build/imp-cli
PP=512
TG=32
REPS=3

MODELS=(
  "Phi-4-mini Q8_0|/home/kekz/github.com/kekzl/imp/models/models--unsloth--Phi-4-mini-instruct-GGUF/snapshots/78eb92a46fc37e6b524df991ed9aca9bc6aa7b80/Phi-4-mini-instruct.Q8_0.gguf"
  "Qwen3-4B Q8_0|/home/kekz/github.com/kekzl/imp/models/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q8_0.gguf"
  "DS-R1-7B Q8_0|/home/kekz/github.com/kekzl/imp/models/models--unsloth--DeepSeek-R1-Distill-Qwen-7B-GGUF/snapshots/097680e4eed7a83b3df6b0bb5e5134099cadf1b0/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"
  "DS-R1-14B Q6_K|/home/kekz/github.com/kekzl/imp/models/models--unsloth--DeepSeek-R1-Distill-Qwen-14B-GGUF/snapshots/7b05b58b41f623e66fc74cd27b35475267b2f3e3/DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf"
  "Gemma-3-12B Q8_0|/home/kekz/github.com/kekzl/imp/models/models--unsloth--gemma-3-12b-it-GGUF/snapshots/d15e4c7dc21dc55d56bf8549db57a71ad8a2a35d/gemma-3-12b-it-Q8_0.gguf"
  "Qwen3-Coder-30B Q6_K|/home/kekz/github.com/kekzl/imp/models/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/b17cb02dd882d5b6ab62fc777ad2995f19668350/Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf"
  "Nemotron-30B Q6_K|/home/kekz/github.com/kekzl/imp/models/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/9ad8b366c308f931b2a96b9306f0b41aef9cd405/Nemotron-3-Nano-30B-A3B-Q6_K.gguf"
)

# Extract tok/s from bench output line like: "pp   512 tokens  avg    25.31 ms  (20226.14 tok/s)  [1 reps]"
extract_pp() { echo "$1" | grep "^pp " | sed 's/.*(\s*//' | awk '{print $1}'; }
extract_tg() { echo "$1" | grep "^tg " | sed 's/.*(\s*//' | awk '{print $1}'; }

echo "================================================================="
echo " FP8 vs FP16 Prefill Weight Cache Benchmark"
echo " pp=$PP  tg=$TG  reps=$REPS"
echo "================================================================="
echo ""

# Collect results
declare -a RESULTS

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name path <<< "$entry"
  echo "-----------------------------------------------------------------"
  echo " $name"
  echo "-----------------------------------------------------------------"

  # FP16 (default)
  echo "  [FP16 cache]"
  fp16_out=$($CLI --model "$path" --bench --bench-pp $PP --bench-reps $REPS --max-tokens $TG --no-cuda-graphs 2>&1)
  fp16_pp=$(extract_pp "$fp16_out")
  fp16_tg=$(extract_tg "$fp16_out")
  fp16_cache=$(echo "$fp16_out" | grep -oP "FP16 weight cache: .+" | head -1)
  echo "    pp: $fp16_pp tok/s   tg: $fp16_tg tok/s"
  echo "    $fp16_cache"

  # FP8
  echo "  [FP8 cache]"
  fp8_out=$($CLI --model "$path" --bench --bench-pp $PP --bench-reps $REPS --max-tokens $TG --no-cuda-graphs --prefill-fp8 2>&1)
  fp8_pp=$(extract_pp "$fp8_out")
  fp8_tg=$(extract_tg "$fp8_out")
  fp8_cache=$(echo "$fp8_out" | grep -oP "FP8 weight cache: .+" | head -1)
  echo "    pp: $fp8_pp tok/s   tg: $fp8_tg tok/s"
  echo "    $fp8_cache"

  # Speedup
  if [[ -n "$fp16_pp" && -n "$fp8_pp" ]]; then
    speedup=$(awk "BEGIN{printf \"%.2f\", $fp8_pp / $fp16_pp}")
    echo "  => PP speedup: ${speedup}x"
  fi

  RESULTS+=("$name|$fp16_pp|$fp8_pp|$fp16_tg|$fp8_tg")
  echo ""
done

echo ""
echo "================================================================="
echo " Summary: pp=$PP tok/s"
echo "================================================================="
printf "%-25s %12s %12s %8s %10s %10s\n" "Model" "FP16 pp" "FP8 pp" "Speedup" "FP16 tg" "FP8 tg"
echo "-------------------------------------------------------------------------------"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r name fp16_pp fp8_pp fp16_tg fp8_tg <<< "$r"
  if [[ -n "$fp16_pp" && -n "$fp8_pp" && "$fp16_pp" != "0" ]]; then
    speedup=$(awk "BEGIN{printf \"%.2f\", $fp8_pp / $fp16_pp}")
  else
    speedup="N/A"
  fi
  printf "%-25s %12s %12s %7sx %10s %10s\n" "$name" "$fp16_pp" "$fp8_pp" "$speedup" "$fp16_tg" "$fp8_tg"
done
