#!/bin/bash
# Profile pp=512 on a large dense Q4_K_M model and dump the top-kernel +
# host-API breakdown. Re-run after any prefill-perf change to track the
# Q4_K_M dequant ratio + cudaMalloc overhead documented in
# `pp512_large_dense_perf_2026_05_15.md`.
#
# Usage:
#   tools/analysis/profile_pp512_large_dense.sh [model_path]
#
# Default model: models/Qwen3-32B-Q4_K_M.gguf. Output report:
#   /tmp/pp512_analysis/qwen32b_q4.nsys-rep
#
# Requires: imp:test docker image + /opt/nvidia/nsight-systems on host.

set -e
MODEL_PATH="${1:-/m/Qwen3-32B-Q4_K_M.gguf}"
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
OUT_DIR="${OUT_DIR:-/tmp/pp512_analysis}"

mkdir -p "$OUT_DIR"
chmod 777 "$OUT_DIR"

echo "=== Profile: $MODEL_PATH at pp=512, no-cuda-graphs ==="

docker run --rm --gpus all \
  -v "$MODELS_DIR":/m \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v "$OUT_DIR":/out \
  imp:test bash -c "
    /usr/local/cuda/bin/nsys profile -t cuda,nvtx -o /out/qwen32b_q4 --force-overwrite=true \
      imp-cli --model '$MODEL_PATH' \
      --bench --bench-pp 512 --bench-reps 1 --max-tokens 1 --temperature 0 --no-cuda-graphs 2>&1 | tail -5
  "

echo ""
echo "=== Top GPU kernels by total time ==="
docker run --rm \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v "$OUT_DIR":/out \
  imp:test \
  /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum --format csv \
    --force-export=true /out/qwen32b_q4.nsys-rep 2>/dev/null | head -12

echo ""
echo "=== Top host-side API calls ==="
docker run --rm \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v "$OUT_DIR":/out \
  imp:test \
  /usr/local/cuda/bin/nsys stats --report cuda_api_sum --format csv \
    /out/qwen32b_q4.nsys-rep 2>/dev/null | head -10

echo ""
echo "Full report: $OUT_DIR/qwen32b_q4.nsys-rep"
echo "Open in Nsight Systems UI for the timeline view."
