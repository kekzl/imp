#!/usr/bin/env bash
# Roadmap row 32 (#2421): code-agent turns with speculative.ngram_corpus off / on, one client,
# greedy. Per turn: tokens, tok/s, md5 of the text; per arm: the cumulative [spec-ngram] line.
# Usage: spec_corpus_replay.sh [model] [corpus_dir] [runs]; IMG, MODELS_DIR
set -uo pipefail
MODEL="${1:-Qwen3-14B-NVFP4}"
CORPUS="$(realpath "${2:-$(dirname "$0")/../../src}")"
RUNS="${3:-3}"
PROMPTS=(
  "Write the full text of the MIT License, with the copyright line 'Copyright (c) 2026 kekzl'."
  "Write a CUDA kernel that applies RMSNorm to FP16 rows (one block per row, warp-shuffle reduction) and a host launcher that checks the launch. C++23, namespace imp."
  "Write a C++23 header that declares a struct with eight int configuration fields, each with a one-line comment, inside namespace imp, using #pragma once."
  "Write a 300-word essay about the history of bridges."
)
start_server() {
  docker rm -f spc32 >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --init --name spc32 --gpus all --network host -v "${MODELS_DIR:-$HOME/models}:/models:ro" \
    -v "$CORPUS:/corpus:ro" "${IMG:-imp:test}" \
    imp-server --host 127.0.0.1 --port 8132 --model "/models/$MODEL" $1 >/dev/null
  for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8132/health >/dev/null 2>&1 && break; sleep 1; done
}
for run in $(seq 1 "$RUNS"); do
  for arm in off on; do
    bash "$(dirname "$0")/../../scripts/require_free_gpu.sh" >/dev/null || { echo BUSY; exit 3; }
    if [ "$arm" = on ]; then start_server "--set speculative.ngram_corpus=/corpus"; else start_server ""; fi
    M=$(curl -s http://127.0.0.1:8132/v1/models | jq -r '.data[0].id')
    tot_n=0; tot_ms=0
    for i in "${!PROMPTS[@]}"; do
      body=$(jq -nc --arg m "$M" --arg p "${PROMPTS[$i]}" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:600,temperature:0,chat_template_kwargs:{enable_thinking:false}}')
      t0=$(date +%s%N)
      r=$(curl -s -m 600 http://127.0.0.1:8132/v1/chat/completions -H 'Content-Type: application/json' -d "$body")
      ms=$(( ($(date +%s%N) - t0) / 1000000 ))
      n=$(jq '.usage.completion_tokens' <<<"$r")
      tot_n=$((tot_n + n)); tot_ms=$((tot_ms + ms))
      echo "run$run $arm p$i: $n tok $ms ms $(awk -v n="$n" -v ms="$ms" 'BEGIN{printf "%.1f", n*1000/ms}') tok/s md5=$(jq -r '.choices[0].message.content' <<<"$r" | md5sum | cut -c1-8)"
    done
    echo "run$run $arm total: $tot_n tok $tot_ms ms $(awk -v n="$tot_n" -v ms="$tot_ms" 'BEGIN{printf "%.1f", n*1000/ms}') tok/s"
    docker logs spc32 2>&1 | grep -E "ngram_corpus:" | sed "s/^.*INFO\\] //"
    docker logs spc32 2>&1 | grep -E "\\[spec-ngram\\] verify_steps" | tail -1 | sed "s/^.*INFO\\] //"
    docker rm -f spc32 >/dev/null 2>&1
  done
done
