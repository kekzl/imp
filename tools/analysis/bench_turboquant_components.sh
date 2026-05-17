#!/bin/bash
# Phase 1 of the TurboQuant–FP8 gap design memo.
# Captures nsys + ncu for {TQ-full, TQ-stripped, FP8, NVFP4} on Qwen3-8B Q8_0
# at pp=512 and pp=4096 with tg=256, and reports the per-token kernel-time
# fraction attributable to QJL XNOR+popcount + Q-side sketch precompute.
#
# Acceptance per design memo §5:
#   - (TQ-full − TQ-stripped) / TQ-full >= 15%  → Path A bottleneck-targeted.
#   - (NVFP4 − FP8) / FP8 <= 5%                 → Path A perf ceiling confirmed.
#
# Usage:
#   tools/analysis/bench_turboquant_components.sh [model_path]
# Default: /m/Qwen3-8B-Q8_0.gguf (mounted from $MODELS_DIR/Qwen3-8B-Q8_0.gguf)
#
# Requires: imp:test docker image (built via `make build`), nsight-systems
# on host at /opt/nvidia/nsight-systems, nsight-compute available inside
# the imp:test container at /usr/local/cuda/bin/ncu.

set -e
MODEL_PATH="${1:-/m/Qwen3-8B-Q8_0.gguf}"
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
OUT_DIR="${OUT_DIR:-/tmp/tq_phase1}"

mkdir -p "$OUT_DIR"
chmod 777 "$OUT_DIR"

run_nsys() {
    local label="$1"
    local kv_flag="$2"
    local env_extra="$3"
    local pp="$4"
    echo "=== nsys: $label  kv=$kv_flag  env=$env_extra  pp=$pp ==="
    docker run --rm --gpus all \
        -v "$MODELS_DIR":/m \
        -v /usr/local/cuda:/usr/local/cuda:ro \
        -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
        -v "$OUT_DIR":/out \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        $env_extra \
        imp:test bash -c "
            /usr/local/cuda/bin/nsys profile -t cuda,nvtx \
                -o /out/${label}_pp${pp} --force-overwrite=true \
                imp-cli --model '$MODEL_PATH' $kv_flag \
                --bench --bench-pp $pp --bench-reps 3 --max-tokens 256 \
                --temperature 0 --no-cuda-graphs 2>&1 | tail -5
        "
    echo ""
    echo "    Top kernels for $label pp=$pp:"
    docker run --rm \
        -v /usr/local/cuda:/usr/local/cuda:ro \
        -v "$OUT_DIR":/out \
        imp:test \
        /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum \
            --format csv --force-export=true \
            "/out/${label}_pp${pp}.nsys-rep" 2>/dev/null \
        | grep -iE "paged_attention|cublas" | head -8 || true
    echo ""
}

run_ncu() {
    local label="$1"
    local kv_flag="$2"
    local env_extra="$3"
    echo "=== ncu ComputeWorkloadAnalysis: $label  kv=$kv_flag ==="
    docker run --rm --gpus all \
        -v "$MODELS_DIR":/m \
        -v /usr/local/cuda:/usr/local/cuda:ro \
        -v "$OUT_DIR":/out \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        $env_extra \
        imp:test bash -c "
            /usr/local/cuda/bin/ncu \
                --section ComputeWorkloadAnalysis \
                --section MemoryWorkloadAnalysis \
                --kernel-name 'regex:paged_attention_(decode|splitk)_turboquant' \
                --kernel-name 'regex:paged_attention_(decode|splitk)_(fp8|nvfp4)' \
                --launch-skip 5 --launch-count 3 \
                --csv --log-file /out/${label}_ncu.log \
                imp-cli --model '$MODEL_PATH' $kv_flag \
                --bench --bench-pp 512 --bench-reps 1 --max-tokens 32 \
                --temperature 0 --no-cuda-graphs 2>/dev/null | tail -3
        "
    echo "    ncu log: $OUT_DIR/${label}_ncu.log"
    echo ""
}

echo "============================================================="
echo "TurboQuant Phase 1 bench — model: $MODEL_PATH"
echo "============================================================="
echo ""

# 1. Full TurboQuant
run_nsys tq_full     "--kv-turboquant" ""                      512
run_nsys tq_full     "--kv-turboquant" ""                      4096
run_ncu  tq_full     "--kv-turboquant" ""

# 2. TurboQuant with QJL stripped
run_nsys tq_stripped "--kv-turboquant" "-e IMP_TQ_SKIP_QJL=1"   512
run_nsys tq_stripped "--kv-turboquant" "-e IMP_TQ_SKIP_QJL=1"   4096
run_ncu  tq_stripped "--kv-turboquant" "-e IMP_TQ_SKIP_QJL=1"

# 3. FP8 (the perf target)
run_nsys fp8         "--kv-fp8"        ""                      512
run_nsys fp8         "--kv-fp8"        ""                      4096
run_ncu  fp8         "--kv-fp8"        ""

# 4. NVFP4 (proxy for post-Path-A MXFP4-K decode cost)
run_nsys nvfp4       "--kv-nvfp4"      ""                      512
run_nsys nvfp4       "--kv-nvfp4"      ""                      4096
run_ncu  nvfp4       "--kv-nvfp4"      ""

echo ""
echo "============================================================="
echo "Summary"
echo "============================================================="
echo ""
echo "nsys reports: $OUT_DIR/*.nsys-rep"
echo "ncu CSV logs: $OUT_DIR/*_ncu.log"
echo ""
echo "To compute acceptance criteria, extract the Avg(ns) of"
echo "paged_attention_decode_turboquant_kernel from each report and:"
echo "  qjl_fraction = (avg_tq_full - avg_tq_stripped) / avg_tq_full"
echo "  ceiling_gap  = (avg_nvfp4 - avg_fp8) / avg_fp8"
echo ""
echo "Acceptance per design memo §5:"
echo "  qjl_fraction >= 0.15  → Path A bottleneck-targeted (PROCEED)"
echo "  ceiling_gap  <= 0.05  → Path A perf ceiling confirmed (PROCEED)"
echo ""
echo "Findings memo template:"
echo "  docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md"
