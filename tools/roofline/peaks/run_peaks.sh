#!/usr/bin/env bash
# Builds imp-peaks in imp:toolchain and runs it on the GPU.
# Usage: tools/roofline/peaks/run_peaks.sh [--build-only] [suite...]   (suites: mem tc simt launch)
# Output: tools/roofline/peaks/out/peaks_<utc>.json + .log; refuses a busy card.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/out"
mkdir -p "$out"
chmod 777 "$out"

docker run --rm --user "$(id -u):$(id -g)" -v "$here:/src" -w /src imp:toolchain \
    nvcc -std=c++20 -O3 -lineinfo --generate-code=arch=compute_120a,code=sm_120a \
    -Xptxas -v -o /src/out/imp-peaks \
    peaks_main.cu peaks_mem.cu peaks_tc.cu peaks_simt.cu peaks_launch.cu \
    -L/usr/local/cuda/lib64/stubs -lnvidia-ml >"$out/build.log" 2>&1 \
    || { tail -40 "$out/build.log"; exit 1; }
grep -E 'error|spill' "$out/build.log" | grep -v ' 0 bytes spill' || true
[[ "${1:-}" == "--build-only" ]] && exit 0

if ! "$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"; then
    echo "GPU busy, not running" >&2
    exit 3
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
{
    echo "# commit $(git -C "$here" rev-parse --short HEAD)$(git -C "$here" diff --quiet || echo -dirty)"
    nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,clocks.max.mem \
        --format=csv,noheader | sed 's/^/# /'
    nvidia-smi -q -d CLOCK | grep -A3 -iE 'locked|applications clocks' | sed 's/^/# /' || true
} >"$out/peaks_$stamp.log"

# Clock sampler for the whole run (100 ms): SM MHz, mem MHz, W, util.
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu \
    --format=csv,noheader -lms 100 >"$out/peaks_$stamp.clocks.csv" &
sampler=$!
trap 'kill $sampler 2>/dev/null || true' EXIT

docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$here/out:/out" imp:toolchain \
    /out/imp-peaks --json "/out/peaks_$stamp.json" "$@" 2>&1 | grep -v '^==\|^$\|NVIDIA\|license\|Container\|CUDA Version\|^WARNING\|docs.nvidia\|Use the' \
    | tee -a "$out/peaks_$stamp.log"
echo "json: $out/peaks_$stamp.json"
