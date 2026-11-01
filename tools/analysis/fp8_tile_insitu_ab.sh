#!/usr/bin/env bash
# In-situ GEMM A/B on a native FP8 checkpoint: N arms (each a list of --set key=value), rounds in rotating
# order (odd rounds forward, even rounds reversed), imp-cli --bench under nsys per arm, then per arm the pp
# tok/s and a (kernel, grid) table of the own tile GEMM and cuBLASLt (nvjet) launches with the GEMM sum.
# Roadmap row 123 (FP8 below M 2048). Runs a COPY of build-dev/imp-cli inside imp:toolchain.
# Usage: tools/analysis/fp8_tile_insitu_ab.sh OUT ROUNDS PP name1 "k=v k=v" name2 "k=v" ...
# Env: MODEL (default Qwen3-8B-FP8), MODELS_DIR.
set -uo pipefail
OUT="$1"; ROUNDS="$2"; PP="$3"; shift 3
NAMES=(); SETS=()
while [ $# -gt 0 ]; do NAMES+=("$1"); SETS+=("$2"); shift 2; done
MODEL="${MODEL:-Qwen3-8B-FP8}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
mkdir -p "$OUT/bin" && cp "$ROOT/build-dev/imp-cli" "$OUT/bin/" && chmod -R 777 "$OUT"

# (kernel, grid) -> launches, avg us, total us; name = everything after the 20th csv comma.
ksum() {
    tail -n +2 "$1" | awk '{
        n = split($0, a, ","); if (n < 21) next
        name = a[21]; for (i = 22; i <= n; ++i) name = name "," a[i]
        if (name !~ /gemm_nvfp4_tile|nvjet/) next
        gsub(/"/, "", name); sub(/^void /, "", name); sub(/\(.*$/, "", name)
        sub(/imp::<unnamed>::gemm_nvfp4_tile_kernel<imp::nvfp4_tile::Cfg</, "own<", name)
        key = substr(name, 1, 110) " grid=" a[4] "x" a[5] "x" a[6] " blk=" a[7] " reg=" a[10]
        cnt[key]++; sum[key] += a[2]; tot += a[2]
    } END {
        for (k in cnt) printf "%6d %9.2f %11.1f  %s\n", cnt[k], sum[k] / cnt[k] / 1000.0, sum[k] / 1000.0, k
        printf "   GEMM sum %.1f us\n", tot / 1000.0
    }' | sort -k3 -n -r
}

run_arm() {
    local idx="$1" r="$2" tag="${NAMES[$1]}_r$2" extra=() pp
    "$CHECK" >/dev/null || { echo "GPU busy, abort at $tag" >&2; exit 1; }
    for s in ${SETS[$idx]}; do extra+=(--set "$s"); done
    docker run --rm --gpus all -v "$MODELS_DIR":/models -v "$OUT":/out -w /tmp imp:toolchain \
        nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda --cuda-graph-trace=node \
        -o "/out/$tag" --force-overwrite=true \
        /out/bin/imp-cli --model "/models/$MODEL" --bench --bench-pp "$PP" --bench-reps 3 --max-tokens 8 \
        --temperature 0 --seed 42 --set speculative.ngram=false --set speculative.mtp_k=0 "${extra[@]}" \
        > "$OUT/$tag.log" 2>&1
    docker run --rm -v "$OUT":/out imp:toolchain nsys stats --report cuda_gpu_trace --format csv \
        --force-export=true -o "/out/$tag" "/out/$tag.nsys-rep" > /dev/null 2>&1
    rm -f "$OUT/$tag.sqlite" "$OUT/$tag.nsys-rep"
    pp=$(grep -oE "^pp +${PP} tokens +avg +[0-9.]+ ms +\( *[0-9.]+ tok/s" "$OUT/$tag.log" \
         | grep -oE "[0-9]+\.[0-9]+ tok/s" | head -1)
    echo "== $tag pp${PP}=${pp:-?}"
    ksum "$OUT/${tag}_cuda_gpu_trace.csv"
}

echo "binary=build-dev/imp-cli ($(git -C "$ROOT" rev-parse --short HEAD)) model=$MODEL pp=$PP rounds=$ROUNDS"
for r in $(seq 1 "$ROUNDS"); do
    if (( r % 2 )); then order=$(seq 0 $(( ${#NAMES[@]} - 1 ))); else order=$(seq $(( ${#NAMES[@]} - 1 )) -1 0); fi
    for i in $order; do run_arm "$i" "$r"; done
done
