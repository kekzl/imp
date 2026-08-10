#!/usr/bin/env bash
# Record MoE expert-activation histograms and report the routing skew.
#
# This is the measurement docs/roadmap.md ("CPU-resident cold experts") and
# docs/GOAL.md leave open: the bandwidth half of the budget is measured
# (62.5 GB/s streaming read on this host), the skew half was not. Skew decides
# whether a resident/host expert split streams (1-f) of the active experts every
# token or far less.
#
# NEEDS A FREE GPU. Check `nvidia-smi` and `docker ps` first — a busy card makes
# the run slow but does not corrupt the histogram, since counts are exact.
#
# Usage:  bash tools/analysis/moe_routing_skew.sh [output-dir]
#
# METHODOLOGY NOTE, and it matters: the histogram counts EVERY routing decision,
# prefill and decode alike. The cold-expert question is about DECODE, where the
# per-token weight traffic is paid. So the prompts below are deliberately short
# and the generation long — decode decisions then outnumber prefill ones by
# roughly max_tokens/prompt_tokens. Re-running this with a long prompt and a
# short generation measures mostly prefill and answers a different question.
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

# 128-expert / top-8 first: it is the structure the 80B-120B class has, and the
# one where a resident subset is a meaningful choice at all. The 32-expert
# gpt-oss is the control — if skew looks identical at both expert counts, the
# result generalises; if not, the expert count is a variable and the 30B answer
# does not transfer.
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
