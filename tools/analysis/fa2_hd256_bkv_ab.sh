#!/usr/bin/env bash
# A/B of attention.fa2_hd256_bkv (64 vs 32) on a hd=256 hybrid checkpoint:
# alternating `imp-cli --bench` pairs under nsys, reporting the FA2 kernel
# time sum per arm (the only number that resolves a <5% prefill-kernel delta,
# docs/internals/BENCHMARKING.md) next to the e2e pp tok/s.
#
#   tools/analysis/fa2_hd256_bkv_ab.sh [PP] [PAIRS]
#
# Runs build-dev/imp-cli inside imp:toolchain (the only image with nsys, same
# recipe as serving_idle_profile.sh); the .qdstrm import and `nsys stats` run
# on the host nsys. Env: MODEL (default Qwen3.8-27B-NVFP4-vllm), MODELS_DIR,
# OUT (results dir).
set -euo pipefail
PP="${1:-4096}"
PAIRS="${2:-3}"
MODEL="${MODEL:-Qwen3.8-27B-NVFP4-vllm}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OUT="${OUT:-/tmp/fa2_hd256_bkv_ab}"
mkdir -p "$OUT" && chmod 777 "$OUT"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
IMPORTER=$(ls /opt/nvidia/nsight-systems/*/host-linux-x64/QdstrmImporter 2>/dev/null | head -1)

run_arm() {  # $1 = bkv, $2 = pair index
    local bkv="$1" i="$2" tag="pp${PP}_bkv$1_p$2"
    [ -x "$CHECK" ] && "$CHECK" >/dev/null || { echo "GPU busy, abort" >&2; exit 1; }
    docker run --rm --gpus all -v "$MODELS_DIR":/models -v "$ROOT":/src -v "$OUT":/out -w /tmp \
        imp:toolchain \
        nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda \
        --cuda-graph-trace=node -o "/out/$tag" --force-overwrite=true \
        /src/build-dev/imp-cli --model "/models/$MODEL" --bench --bench-pp "$PP" --bench-reps 3 \
            --max-tokens 8 --temperature 0 --seed 42 \
            --set speculative.ngram=false --set speculative.mtp_k=0 \
            --set attention.fa2_hd256_bkv="$bkv" > "$OUT/$tag.log" 2>&1 || true
    if [ ! -f "$OUT/$tag.nsys-rep" ] && [ -f "$OUT/$tag.qdstrm" ] && [ -n "$IMPORTER" ]; then
        "$IMPORTER" -i "$OUT/$tag.qdstrm" -o "$OUT/$tag.nsys-rep" >/dev/null 2>&1
    fi
    nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/$tag.nsys-rep" \
        > "$OUT/$tag.kern.csv" 2>/dev/null || true
    local pp fa2
    pp=$(grep -oE "^pp +${PP} tokens +avg +[0-9.]+ ms +\( *[0-9.]+ tok/s" "$OUT/$tag.log" \
         | grep -oE "[0-9]+\.[0-9]+ tok/s" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    # cuda_gpu_kern_sum csv: Time (%),Total Time (ns),Instances,Avg,Med,Min,Max,StdDev,Name
    # nsys prints the template args as "<(int)64, (int)256, ...>"
    fa2=$(grep "fmha_sm120_fa2_kernel<(int)64, (int)256" "$OUT/$tag.kern.csv" \
          | awk -F, '{s+=$2; n+=$3} END {if (n) printf "%.3f ms / %d launches", s/1e6, n}' || true)
    echo "pair $i bkv=$bkv pp${PP}=${pp:-?} tok/s  fa2_hd256: ${fa2:-none}"
}

echo "binary=build-dev/imp-cli ($(git -C "$ROOT" rev-parse --short HEAD)) model=$MODEL pp=$PP pairs=$PAIRS out=$OUT"
for i in $(seq 1 "$PAIRS"); do
    if (( i % 2 )); then run_arm 64 "$i"; run_arm 32 "$i"; else run_arm 32 "$i"; run_arm 64 "$i"; fi
done
