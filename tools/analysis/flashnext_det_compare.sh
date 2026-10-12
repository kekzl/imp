#!/usr/bin/env bash
# Deterministic greedy run per arm (runtime.deterministic, temperature 0): prints the generated text md5 and
# the device expert-cache hit/miss counters, so two kernels or config arms can be checked for identical
# routing, cache decisions and output. Usage: tools/analysis/flashnext_det_compare.sh OUTDIR NAME BIN_DIR [args...]
set -uo pipefail
OUT="$1"; NAME="$2"; BIN_DIR="$3"; shift 3
MODEL=${MODEL:-/models/Qwen3.8-Flash-Next-NVFP4}
TOKENS=${TOKENS:-256}
PROMPT=${PROMPT:-"Implement a lock-free single-producer single-consumer ring buffer in C++20 with std::atomic, explain the memory orderings, and add a small benchmark main()."}
mkdir -p "$OUT" && chmod 777 "$OUT"
"$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > /dev/null || { echo "GPU busy" >&2; exit 1; }
docker run --rm --gpus all -v "$HOME/models":/models:ro -v "$BIN_DIR":/bin_arm:ro imp:toolchain /bin_arm/imp-cli \
    --model "$MODEL" --prompt "$PROMPT" --max-tokens "$TOKENS" --temperature 0 --seed 1 \
    --set runtime.deterministic=true "$@" > "$OUT/$NAME.out" 2> "$OUT/$NAME.err"
md5=$(grep -vE "^(pp|tg) " "$OUT/$NAME.out" | md5sum | cut -c1-12)
stats=$(sed -nE 's/.*Device expert cache stats: (.*)/\1/p' "$OUT/$NAME.err" "$OUT/$NAME.out" | tail -1)
tg=$(sed -nE 's/^tg .*\( *([0-9.]+) tok\/s\)/\1/p' "$OUT/$NAME.out" | tail -1)
echo "$NAME text_md5=$md5 tg=$tg stats: $stats"
