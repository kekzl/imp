#!/usr/bin/env bash
# Mission scoreboard harness — measures imp across the hero model matrix on RTX 5090.
# Reproducible: same weights, pp512 + tg128, canonical bench methodology.
# Output: docs/scoreboard.tsv (append) + prints a table.
#
# Usage: scripts/scoreboard.sh [reps] [extra imp-cli flags...]
#   reps default 10. GPU must be free (checked).
set -uo pipefail

REPS="${1:-10}"; shift || true
EXTRA="$*"
IMG="${DOCKER_IMG:-imp:test}"
MODELS_HOST="/home/kekz/models"
OUT="$(cd "$(dirname "$0")/.." && pwd)/docs/scoreboard.tsv"
PP="${BENCH_PP:-512}"
TG="${BENCH_TG:-128}"

DOCKER="docker run --rm --gpus all -v ${MODELS_HOST}:/models -e CUBLAS_WORKSPACE_CONFIG=:4096:8 ${IMG}"

# name <TAB> relative_path <TAB> family <TAB> quant <TAB> extra_flags
MATRIX=$(cat <<'EOF'
Qwen3-8B	Qwen3-8B-Q8_0.gguf	dense	Q8_0
Qwen3-8B	Qwen3-8B-NVFP4-cortecs	dense	NVFP4
Qwen3-14B	Qwen3-14B-Q6_K.gguf	dense	Q6_K
Qwen3-14B	Qwen3-14B-NVFP4	dense	NVFP4
Qwen3-30B-A3B	Qwen3-30B-A3B-NVFP4-Modelopt	moe	NVFP4
Qwen3-30B-A3B	Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf	moe	Q4_K_M
Qwen3-Coder-30B-A3B	Qwen3-Coder-30B-A3B-Instruct-FP4	moe	NVFP4
Qwen3.6-35B-A3B	Qwen3.6-35B-A3B-NVFP4	hybrid	NVFP4
Qwen3.6-35B-A3B	qwen3.6-35B-A3B-gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf	hybrid	Q4_K_M
Gemma-4-26B-A4B	Gemma-4-26B-A4B-it-NVFP4	moe	NVFP4
Gemma-4-26B-A4B	gemma-4-26B-A4B-it-UD-Q4_K_M.gguf	moe	Q4_K_M
Gemma-3-12B	gemma-3-12b-it-Q4_K_M.gguf	dense	Q4_K_M
Phi-4-reasoning	Phi-4-reasoning-plus-NVFP4	dense	NVFP4
Nemotron-3-Nano-30B	Nemotron-3-Nano-30B-A3B-NVFP4	hybrid	NVFP4
EOF
)

[ -f "$OUT" ] || echo -e "timestamp\tname\tfamily\tquant\tpath\tpp${PP}_tps\ttg${TG}_tps\tstatus" > "$OUT"

printf "%-22s %-7s %-7s %12s %10s  %s\n" "MODEL" "FAMILY" "QUANT" "pp${PP}" "tg${TG}" "STATUS"
echo "$MATRIX" | while IFS=$'\t' read -r name path family quant flags; do
  [ -z "$name" ] && continue
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  log=$(timeout 600 $DOCKER imp-cli --model "/models/$path" --bench --bench-pp "$PP" \
        --bench-reps "$REPS" --max-tokens "$TG" --temperature 0 $flags $EXTRA 2>&1)
  pp=$(echo "$log" | grep -oP '^pp\s+\d+.*\(\s*\K[0-9.]+(?=\s+tok/s)' | tail -1)
  tg=$(echo "$log" | grep -oP '^tg\s+\d+.*\(\s*\K[0-9.]+(?=\s+tok/s)' | tail -1)
  if [ -n "$pp" ] && [ -n "$tg" ]; then status="ok"; else
    status="FAIL: $(echo "$log" | grep -iE 'error|abort|throw|what\(\)|OOM|out of memory' | head -1 | cut -c1-80)"
    pp=${pp:-NA}; tg=${tg:-NA}
  fi
  printf "%-22s %-7s %-7s %12s %10s  %s\n" "$name" "$family" "$quant" "$pp" "$tg" "$status"
  echo -e "${ts}\t${name}\t${family}\t${quant}\t${path}\t${pp}\t${tg}\t${status}" >> "$OUT"
done
