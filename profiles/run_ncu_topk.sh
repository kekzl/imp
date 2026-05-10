#!/usr/bin/env bash
# Phase 3: ncu deep-dive on top kernels identified from Phase 2 timeline analysis.
# Args: $1 = kernel-name regex, $2 = optional model file (default Qwen3-4B Q8_0),
#       $3 = workload tag (default decode-heavy), $4 = output suffix
#
# ncu may also hit ERR_NVGPUCTRPERM on consumer driver — set
# `nvidia-modprobe -u -c=0 --persistence-mode=1` and configure
# /etc/modprobe.d/nvidia.conf with 'options nvidia NVreg_RestrictProfilingToAdminUsers=0'
# then reboot driver. As a fallback, run with sudo.
set -euo pipefail

REPO="$REPO"
NCU_DIR="${REPO}/profiles/ncu"
mkdir -p "${NCU_DIR}"

KERNEL_REGEX="${1:?Usage: $0 <kernel-regex> [model] [workload] [tag]}"
MODEL="${2:-Qwen3-4B-Instruct-2507-Q8_0.gguf}"
WL="${3:-W2}"   # W1=long-prefill (pp=8192 tg=64) | W2=decode-heavy (pp=256 tg=2048)
TAG="${4:-$(echo "${KERNEL_REGEX}" | tr -c '[:alnum:]' '_' | head -c 40)}"

case "${WL}" in
  W1) PP=8192; TG=64 ;;
  W2) PP=256;  TG=2048 ;;
  *)  echo "Unknown workload ${WL}"; exit 2 ;;
esac

OUT="/profiles/../ncu/$(basename "${MODEL}" .gguf)_${WL}_${TAG}"

docker run --rm --gpus all --privileged \
  --user $(id -u):$(id -g) -w /tmp \
  -v "${REPO}/models:/models" \
  -v "${REPO}/profiles:/profiles" \
  -v /opt/nvidia:/opt/nvidia:ro \
  --entrypoint=/opt/nvidia/nsight-compute/2026.1.1/host/target-linux-x64/ncu \
  imp:profile \
    --target-processes all \
    --kernel-name "regex:${KERNEL_REGEX}" \
    --launch-skip 50 --launch-count 10 \
    --set full --import-source yes \
    --force-overwrite \
    -o "${OUT}" \
    /usr/local/bin/imp-cli --model "/models/${MODEL}" \
      --bench --bench-pp "${PP}" --bench-reps 1 --max-tokens "${TG}" \
      --temperature 0 --seed 42 --no-cuda-graphs 2>&1 | tail -30
