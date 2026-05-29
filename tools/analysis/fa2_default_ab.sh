#!/bin/bash
# Does attention.fmha_fa2=on actually help end-to-end under DEFAULT settings
# (chunking on, auto threshold)? Decides whether to flip the default.
# Config-only A/B (no rebuild). Detects FA2 activation via its INFO log.
set -uo pipefail
cd /src/build-ciq
MODEL="${MODEL:-/models/Qwen3-14B-NVFP4}"
REPS="${REPS:-10}"

run() {  # $1=pp  $2=fa2(never|on) -> "tok/s | activated?"
  local out
  out=$(./imp-cli --model "$MODEL" --bench --bench-pp "$1" --bench-reps "$REPS" \
        --max-tokens 1 --temperature 0 --set attention.fmha_fa2="$2" 2>&1)
  local tps act
  tps=$(echo "$out" | grep -oP '^pp\s+'"$1"'\s.*\(\s*\K[0-9.]+' | head -1)
  act=$(echo "$out" | grep -qE "FMHA FA2 register-resident kernel ACTIVE" && echo "FA2-ACTIVE" || echo "no-FA2")
  echo "${tps:-FAIL} ($act)"
}

echo "model=$MODEL reps=$REPS  (default settings: chunking on, auto threshold)"
printf "%-7s | %-26s | %-26s\n" "pp" "fa2=never (fp8 FMHA)" "fa2=on"
for PP in 2048 4096 8192 16384; do
  a=$(run "$PP" never); sleep 8
  b=$(run "$PP" on);    sleep 8
  printf "%-7s | %-26s | %-26s\n" "$PP" "$a" "$b"
done
