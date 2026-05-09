#!/usr/bin/env bash
# Export nsys stats CSVs from each .nsys-rep in profiles/baselines/
# Reports: cuda_gpu_kern_sum (kernel time), cuda_gpu_mem_time_sum (mem ops),
#          cuda_api_sum (host API calls), cuda_gpu_sum (per-stream summary)
set -euo pipefail

REPO="/home/kekz/github.com/kekzl/imp"
BASE="${REPO}/profiles/baselines"
CSV="${REPO}/profiles/csv"
mkdir -p "${CSV}"

NSYS=/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys

REPORTS="cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,cuda_gpu_sum"

for rep in "${BASE}"/*.nsys-rep; do
  base="$(basename "${rep}" .nsys-rep)"
  echo "=== ${base} ==="
  ${NSYS} stats --report "${REPORTS}" \
    --format csv --output "${CSV}/${base}" \
    "${rep}" 2>&1 | tail -3
done

echo ""
echo "CSVs:"
ls -la "${CSV}/" | head
