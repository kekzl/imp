#!/usr/bin/env bash
# Alternating `imp-cli --bench` pairs under nsys for one config key with two
# values, reporting the nsys time sum of a kernel-name regex per arm next to
# the e2e pp tok/s (a <5% prefill-kernel delta only resolves in the kernel
# sum, docs/internals/BENCHMARKING.md).
#
#   tools/analysis/prefill_kernel_ab.sh KEY VAL_A VAL_B KERNEL_REGEX [PP] [PAIRS]
#   e.g. tools/analysis/prefill_kernel_ab.sh gemm.nvfp4_cutlass_streamk 0 1 \
#          'cutlass.*BlockScaled' 512 3
#
# Runs build-dev/imp-cli inside imp:toolchain (the only image with nsys, same
# recipe as serving_idle_profile.sh); the .qdstrm import and `nsys stats` run
# on the host nsys. Env: MODEL (default Qwen3-14B-NVFP4), MODELS_DIR, OUT,
# EXTRA_SET (extra `--set k=v` args applied to BOTH arms).
set -euo pipefail
KEY="$1"; VAL_A="$2"; VAL_B="$3"; KREGEX="$4"
PP="${5:-512}"
PAIRS="${6:-3}"
MODEL="${MODEL:-Qwen3-14B-NVFP4}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OUT="${OUT:-/tmp/prefill_kernel_ab}"
mkdir -p "$OUT" && chmod 777 "$OUT"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
IMPORTER=$(ls /opt/nvidia/nsight-systems/*/host-linux-x64/QdstrmImporter 2>/dev/null | head -1)

run_arm() {  # $1 = value, $2 = pair index
    local val="$1" i="$2" tag="pp${PP}_${KEY//./_}=$1_p$2"
    [ -x "$CHECK" ] && "$CHECK" >/dev/null || { echo "GPU busy, abort" >&2; exit 1; }
    docker run --rm --gpus all -v "$MODELS_DIR":/models -v "$ROOT":/src -v "$OUT":/out -w /tmp \
        imp:toolchain \
        nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda \
        --cuda-graph-trace=node -o "/out/$tag" --force-overwrite=true \
        /src/build-dev/imp-cli --model "/models/$MODEL" --bench --bench-pp "$PP" --bench-reps 3 \
            --max-tokens 8 --temperature 0 --seed 42 \
            --set speculative.ngram=false --set speculative.mtp_k=0 \
            --set "$KEY=$val" ${EXTRA_SET:-} > "$OUT/$tag.log" 2>&1 || true
    if [ ! -f "$OUT/$tag.nsys-rep" ] && [ -f "$OUT/$tag.qdstrm" ] && [ -n "$IMPORTER" ]; then
        "$IMPORTER" -i "$OUT/$tag.qdstrm" -o "$OUT/$tag.nsys-rep" >/dev/null 2>&1
    fi
    nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/$tag.nsys-rep" \
        > "$OUT/$tag.kern.csv" 2>/dev/null || true
    local pp ksum
    pp=$(grep -oE "^pp +${PP} tokens +avg +[0-9.]+ ms +\( *[0-9.]+ tok/s" "$OUT/$tag.log" \
         | grep -oE "[0-9]+\.[0-9]+ tok/s" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    # cuda_gpu_kern_sum csv: Time (%),Total Time (ns),Instances,...,Name (template args as "(int)64")
    ksum=$(grep -E "$KREGEX" "$OUT/$tag.kern.csv" \
           | awk -F, '{s+=$2; n+=$3} END {if (n) printf "%.3f ms / %d launches", s/1e6, n}' || true)
    echo "pair $i $KEY=$val pp${PP}=${pp:-?} tok/s  kernels[$KREGEX]: ${ksum:-none}"
}

echo "binary=build-dev/imp-cli ($(git -C "$ROOT" rev-parse --short HEAD)) model=$MODEL pp=$PP pairs=$PAIRS out=$OUT"
for i in $(seq 1 "$PAIRS"); do
    if (( i % 2 )); then run_arm "$VAL_A" "$i"; run_arm "$VAL_B" "$i"; else run_arm "$VAL_B" "$i"; run_arm "$VAL_A" "$i"; fi
done
