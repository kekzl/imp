#!/usr/bin/env bash
# Nsight Compute profiling for imp inference engine.
#
# Usage:
#   ./bench/profile.sh <model_path> [prompt_tokens] [gen_tokens]
#
# Examples:
#   ./bench/profile.sh models/Qwen3-8B-Q8_0.gguf
#   ./bench/profile.sh models/Qwen3-8B-Q8_0.gguf 512 64
#
# Output: bench/results/imp_profile.ncu-rep
#
# Requires: Nsight Compute (ncu) installed and accessible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

MODEL="${1:?Usage: $0 <model_path> [pp_tokens] [gen_tokens]}"
PP="${2:-512}"
TG="${3:-64}"
IMP_BIN="${IMP_BIN:-./build/imp-cli}"
NCU="${NCU:-ncu}"
OUTPUT="${RESULTS_DIR}/imp_profile"

if ! command -v "${NCU}" &>/dev/null; then
    echo "Error: ncu (Nsight Compute) not found."
    echo "Install: https://developer.nvidia.com/nsight-compute"
    echo "Or set NCU=/path/to/ncu"
    exit 1
fi

if [ ! -f "${IMP_BIN}" ]; then
    echo "Error: imp-cli not found at ${IMP_BIN}"
    echo "Build first: cmake --build build -j\$(nproc)"
    exit 1
fi

echo "=== Nsight Compute Profiling ==="
echo "  Model:   ${MODEL}"
echo "  Prefill: ${PP} tokens"
echo "  Decode:  ${TG} tokens"
echo "  Output:  ${OUTPUT}.ncu-rep"
echo ""

# Key metrics for inference performance analysis
METRICS=(
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__warps_active.avg.pct_of_peak_sustained_active"
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__inst_executed_pipe_tensor.sum"
)

METRICS_STR=$(IFS=,; echo "${METRICS[*]}")

echo "Profiling (this may take several minutes)..."
echo ""

# Profile with limited kernel count to keep report manageable.
# --launch-count 50: profile first 50 kernel launches
# --launch-skip 10: skip warmup kernels
sudo "${NCU}" \
    --set full \
    --metrics "${METRICS_STR}" \
    --launch-skip 10 \
    --launch-count 50 \
    --target-processes all \
    -o "${OUTPUT}" \
    -f \
    "${IMP_BIN}" \
        --model "${MODEL}" \
        --bench \
        --bench-pp "${PP}" \
        --bench-reps 1 \
        --max-tokens "${TG}" \
        --temperature 0

echo ""
echo "=== Profile complete ==="
echo "  Report: ${OUTPUT}.ncu-rep"
echo ""
echo "View report:"
echo "  ncu-ui ${OUTPUT}.ncu-rep"
echo ""

# Extract summary if ncu supports CSV export
if "${NCU}" --help 2>&1 | grep -q "csv"; then
    echo "Extracting kernel summary..."
    "${NCU}" --import "${OUTPUT}.ncu-rep" \
        --csv \
        --page raw \
        2>/dev/null | head -60 > "${RESULTS_DIR}/imp_profile_summary.csv" \
        && echo "  CSV summary: ${RESULTS_DIR}/imp_profile_summary.csv"
fi

# Quick analysis: parse top kernels by time
echo ""
echo "=== Quick Kernel Analysis ==="
"${NCU}" --import "${OUTPUT}.ncu-rep" \
    --csv \
    --page raw \
    2>/dev/null | python3 -c "
import csv
import sys

reader = csv.DictReader(sys.stdin)
kernels = []
for row in reader:
    name = row.get('Kernel Name', '')
    duration = row.get('Duration', '0')
    dram = row.get('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', '')
    sm = row.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', '')
    occ = row.get('sm__warps_active.avg.pct_of_peak_sustained_active', '')
    try:
        dur_us = float(duration) / 1000  # ns -> us
    except (ValueError, TypeError):
        dur_us = 0
    kernels.append((name, dur_us, dram, sm, occ))

kernels.sort(key=lambda x: -x[1])

print(f\"{'Kernel':<60} {'Time us':>9} {'DRAM%':>6} {'SM%':>6} {'Occ%':>6}\")
print('-' * 93)
for name, dur, dram, sm, occ in kernels[:20]:
    short = name[:58] if len(name) > 58 else name
    print(f'{short:<60} {dur:>9.1f} {dram:>6} {sm:>6} {occ:>6}')
" 2>/dev/null || echo "(CSV parsing not available — open the .ncu-rep in ncu-ui)"
