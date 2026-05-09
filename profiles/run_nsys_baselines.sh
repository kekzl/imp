#!/usr/bin/env bash
# Capture nsys baselines for the small dense models (transferable findings).
# W1: long-context prefill (pp=8192, tg=64)
# W2: decode-heavy             (pp=256,  tg=2048)
#
# Each model run twice: --no-cuda-graphs (per-kernel attribution)
# Plus one model (Qwen3-4B) re-run with graphs ON for graph-behavior verification.
#
# Output: profiles/baselines/<model>_<workload>_<graphs>.nsys-rep
set -euo pipefail

REPO="/home/kekz/github.com/kekzl/imp"
PROFILES="${REPO}/profiles/baselines"
mkdir -p "${PROFILES}"

NSYS=/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys
IMG=imp:profile

# Mount layout: host /opt/nvidia → container /opt/nvidia (host's standalone nsys works
# inside container; CUDA-bundled nsys at /usr/local/cuda/bin/nsys hits a manifest check
# error when its install root isn't visible).
DOCKER_BASE="docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -w /tmp \
  -v ${REPO}/models:/models \
  -v ${PROFILES}:/profiles \
  -v /opt/nvidia:/opt/nvidia:ro \
  --entrypoint=${NSYS} ${IMG}"

# nsys flags — measurement-first capture.
# --gpu-metrics-devices skipped: needs CAP_SYS_ADMIN or
# RestrictProfilingToAdminUsers=0 on consumer driver. We pull HBM/SM throughput
# from ncu in Phase 3 instead.
NSYS_FLAGS=(
  profile
  --trace=cuda,nvtx,osrt,cublas
  --cuda-memory-usage=true
  --force-overwrite=true
  --stats=false
)

# Model registry: tag → file
declare -A MODELS=(
  [qwen3-4b-q8]=Qwen3-4B-Instruct-2507-Q8_0.gguf
  [llama32-3b-q8]=Llama-3.2-3B-Instruct-Q8_0.gguf
  [qwen35-4b-gdn-q8]=Qwen3.5-4B-Q8_0.gguf
)

run_one() {
  local tag="$1" file="$2" wl="$3" pp="$4" tg="$5" graphs="$6"
  local out="/profiles/${tag}_${wl}_${graphs}"
  local extra=()
  [ "${graphs}" = "ng" ] && extra+=(--no-cuda-graphs)

  echo ""
  echo "=== ${tag} | ${wl} (pp=${pp} tg=${tg}) | graphs=${graphs} ==="
  echo "    out: ${out}.nsys-rep"

  ${DOCKER_BASE} \
    "${NSYS_FLAGS[@]}" -o "${out}" \
    /usr/local/bin/imp-cli \
      --model "/models/${file}" \
      --bench --bench-pp "${pp}" --bench-reps 1 \
      --max-tokens "${tg}" --temperature 0 --seed 42 \
      "${extra[@]}" \
    2>&1 | tail -8
}

# All small dense models, both workloads, graphs OFF (per-kernel attribution)
for tag in "${!MODELS[@]}"; do
  file="${MODELS[$tag]}"
  run_one "${tag}" "${file}" "W1" 8192 64   "ng"
  run_one "${tag}" "${file}" "W2" 256  2048 "ng"
done

# Bonus: Qwen3-4B with graphs ON to verify graph capture / launch collapse
run_one "qwen3-4b-q8" "Qwen3-4B-Instruct-2507-Q8_0.gguf" "W2" 256 2048 "g"

echo ""
echo "=== Done. Profiles in ${PROFILES} ==="
ls -la "${PROFILES}/"
