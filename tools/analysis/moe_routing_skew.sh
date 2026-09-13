#!/usr/bin/env bash
# Records MoE expert-activation histograms and reports routing skew: the measurement
# docs/roadmap.md and docs/GOAL.md leave open (bandwidth half is measured, skew half wasn't).
# Skew decides whether a resident/host expert split streams most active experts every token or
# far fewer. NEEDS A FREE GPU (busy card slows but doesn't corrupt the histogram).
# Prompts are deliberately short with long generation: the cold-expert question is about
# DECODE traffic, so decode decisions must outnumber prefill ones; a long-prompt/short-gen run
# measures mostly prefill and answers a different question.
# Usage: bash tools/analysis/moe_routing_skew.sh [output-dir].
set -uo pipefail

OUT="${1:-/tmp/moe_skew}"
mkdir -p "$OUT"
cd "$(git rev-parse --show-toplevel)"

MAX_TOKENS="${MAX_TOKENS:-512}"
# Where the checkpoints live. Not hardcoded to a home directory: check-release.sh
# rejects maintainer paths in tracked files, and rightly so.
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

# Short prompts, long answers — see the methodology note above. Three different
# subjects, because a single prompt measures one trajectory's expert taste and
# the question is about a workload.
PROMPTS=(
  "Write a short essay about why the sea is salty."
  "Explain step by step how a bicycle gear system works."
  "List ten prime numbers and say why each is prime."
)

# 128-expert/top-8 first: the structure the 80B-120B class has, where a resident subset is a
# meaningful choice. 32-expert gpt-oss is the control: if skew matches at both expert counts
# the result generalises, otherwise expert count is a variable and the 30B answer doesn't transfer.
MODELS=(
  "qwen3-30b-a3b:/models/Qwen3-30B-A3B-NVFP4-Modelopt"
  "gpt-oss-20b:/models/gpt-oss-20b-mxfp4.gguf"
)

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  path="${entry#*:}"
  log="$OUT/${name}.log"
  : >"$log"
  echo "== $name =="
  hists=()
  # One process AND one histogram file per prompt: the file is written whole at
  # executor teardown, so a second run would overwrite the first rather than add
  # to it. The analysis sums them back into one workload.
  for i in "${!PROMPTS[@]}"; do
    hist="$OUT/${name}.p${i}.json"
    # $OUT is mounted at /out rather than reached through /src: it defaults to an
    # absolute path outside the repo, and "/src/$OUT" would silently become
    # "/src//tmp/..." — a path the container happily creates and nobody reads.
    docker run --rm --gpus all -v "$PWD":/src -v "$MODELS_DIR":/models \
      -v "$(cd "$OUT" && pwd)":/out -w /src/build-dev \
      imp:toolchain ./imp-cli --model "$path" \
      --set "diagnostics.moe_expert_hist=/out/${name}.p${i}.json" \
      --prompt "${PROMPTS[$i]}" --max-tokens "$MAX_TOKENS" --temperature 0 \
      >>"$log" 2>&1
    if [ -s "$hist" ]; then hists+=("$hist"); else echo "  prompt $i: no histogram"; fi
  done
  grep -E "moe expert histogram" "$log" | tail -3
  if [ "${#hists[@]}" -gt 0 ]; then
    python3 tools/analysis/moe_routing_skew.py "${hists[@]}" | tee "$OUT/${name}.report.txt"
  else
    echo "  nothing recorded — see $log"
  fi
  echo

done

echo "artifacts in $OUT"
