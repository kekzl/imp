#!/usr/bin/env bash
# Phase-0 microbench for the BitDecoding NVFP4 port. Builds + runs the
# scalar-FFMA vs HMMA-MMA Q.K dot comparison on synthetic input, then
# SASS-audits the binary to confirm HMMA dispatch.
#
# Re-run after each kernel-layout change to verify numerical equivalence
# stays within tolerance and HMMA dispatch is preserved.
#
# Usage: bash tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC="$REPO_ROOT/tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu"
BIN=/tmp/bench_nvfp4_qk_tc_vs_scalar

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}

echo "=== build ==="
"$NVCC" -O2 --generate-code=arch=compute_120a,code=sm_120a "$SRC" -o "$BIN"

echo
echo "=== run ==="
"$BIN"

echo
echo "=== SASS HMMA check ==="
HMMA=$("$CUOBJDUMP" --dump-sass "$BIN" 2>/dev/null | grep -cE "HMMA" || true)
echo "  HMMA instruction count: $HMMA"
if [ "$HMMA" -eq 0 ]; then
    echo "  ⚠ Zero HMMA — TC dispatch did not survive ptxas. Bench result is invalid."
    exit 1
fi
echo "  ✓ HMMA dispatch confirmed."
