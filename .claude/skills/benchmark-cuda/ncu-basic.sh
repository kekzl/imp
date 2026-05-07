#!/usr/bin/env bash
# ncu-basic.sh — canonical Nsight Compute metric set for imp kernels on sm_120.
#
# Usage:
#   ./ncu-basic.sh "<kernel-regex>" <binary> [<binary-args>...]
#
# Example:
#   ./ncu-basic.sh "fmha.*" ./build/imp-bench --model models/Qwen3-4B-Q8_0.gguf
#
# Skips 3 warmup launches, captures 10. Compile with `-lineinfo` and add
# `--set detailed --import-source yes` for source-correlated stalls.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <kernel-regex> <binary> [<binary-args>...]" >&2
  exit 1
fi

KERNEL_REGEX="$1"
shift

METRICS=(
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  gpu__time_duration.sum
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes.sum
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
  l1tex__t_sector_hit_rate
  smsp__inst_executed_pipe_tensor_op_hmma.sum
  smsp__inst_executed_pipe_tensor_op_qmma.sum
  sm__warps_active.avg.pct_of_peak_sustained_active
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
  smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
)

# Join metrics with comma
IFS=,
METRIC_ARG="${METRICS[*]}"
unset IFS

exec ncu \
  --kernel-name "regex:${KERNEL_REGEX}" \
  --launch-skip 3 \
  --launch-count 10 \
  --metrics "${METRIC_ARG}" \
  "$@"
