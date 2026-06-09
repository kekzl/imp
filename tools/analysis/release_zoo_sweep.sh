#!/bin/bash
# v0.10.0 release benchmark sweep — full local zoo, isolated (one process per
# model), decode (tg256) is the hero signal; prefill (pp512) reported but noisy
# (2.6x restart variance). Samples GPU clocks throughout (healthy host =
# mem 13801 MHz / ~500 W under load). Continues past OOM/load failures.
set -uo pipefail
IMG=imp:test; MODELS=/home/kekz/models
R() { docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v "$MODELS":/models "$IMG" \
        imp-cli --model "$1" --bench --bench-pp 512 --bench-reps 10 --max-tokens 256 \
        --temperature 0 --seed 42 2>&1; }

# label | model path  (NVFP4 dirs + GGUF files; broken/too-big/vision excluded)
MODELS_LIST=(
  "NVFP4 Qwen3-8B-cortecs|/models/Qwen3-8B-NVFP4-cortecs"
  "NVFP4 Qwen3-14B (dense)|/models/Qwen3-14B-NVFP4"
  "NVFP4 Qwen3-30B-A3B-Modelopt|/models/Qwen3-30B-A3B-NVFP4-Modelopt"
  "NVFP4 Qwen3-Coder-30B-A3B|/models/Qwen3-Coder-30B-A3B-Instruct-FP4"
  "NVFP4 Qwen3.6-35B-A3B|/models/Qwen3.6-35B-A3B-NVFP4"
  "NVFP4 Gemma-4-26B-A4B|/models/Gemma-4-26B-A4B-it-NVFP4"
  "NVFP4 Nemotron-3-Nano-30B|/models/Nemotron-3-Nano-30B-A3B-NVFP4"
  "NVFP4 Nemotron-Elastic-30B|/models/Nemotron-Labs-3-Elastic-30B-A3B-NVFP4"
  "NVFP4 Phi-4-reasoning-plus|/models/Phi-4-reasoning-plus-NVFP4"
  "NVFP4 gpt-oss-20b|/models/gpt-oss-20b"
  "NVFP4 Qwen3.5-4B (GDN)|/models/Qwen3.5-4B"
  "GGUF Qwen3-8B-Q8_0|/models/Qwen3-8B-Q8_0.gguf"
  "GGUF Qwen3-14B-Q6_K|/models/Qwen3-14B-Q6_K.gguf"
  "GGUF Qwen3-30B-A3B-Q4_K_M|/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf"
  "GGUF Qwen3-4B-2507-Q8_0|/models/Qwen3-4B-Instruct-2507-Q8_0.gguf"
  "GGUF Llama-3.2-3B-Q8_0|/models/Llama-3.2-3B-Instruct-Q8_0.gguf"
  "GGUF Gemma-4-26B-A4B-Q8_0|/models/gemma-4-26B-A4B-it-Q8_0.gguf"
  "GGUF Gemma-4-26B-A4B-Q4_K_M|/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
  "GGUF Qwen3.6-35B-A3B-Q4_K_M|/models/qwen3.6-35B-A3B-gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)

# Background clock sampler for the whole sweep.
( while true; do
    echo "[clk $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv,noheader | tr -d '\n')"
    sleep 20
  done > /tmp/out/zoo_clocks.log 2>&1 ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

echo "######## v0.10.0 zoo sweep ($(git -C /home/kekz/github.com/kekzl/imp rev-parse --short HEAD)) ########"
echo "[warmup, discarded]"; R /models/Qwen3-8B-Q8_0.gguf >/dev/null 2>&1
for entry in "${MODELS_LIST[@]}"; do
  lab="${entry%%|*}"; path="${entry#*|}"
  echo "===== $lab ====="
  out="$(R "$path")"
  echo "$out" | grep -E '^pp|^tg' | sed 's/^/  /'
  err="$(echo "$out" | grep -iE 'error|fail|insufficient|OOM|out of memory|abort' | head -2)"
  [ -n "$err" ] && echo "  !! $err"
done
echo "==== clocks during sweep ===="; cat /tmp/out/zoo_clocks.log
echo "DONE"
