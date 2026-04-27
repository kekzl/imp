#!/usr/bin/env bash
# Master PTX feature survey for sm_120f.
# Runs every individual survey in tools/analysis/ptx_*_survey.sh and
# concatenates the markdown output. Use after CUDA toolkit upgrades to refresh
# the dead-end status of every documented PTX instruction we care about.
#
# Usage:
#   tools/analysis/ptx_survey_all.sh                  # full
#   tools/analysis/ptx_survey_all.sh > docs/ptx-status-$(date +%Y-%m-%d).md
#   tools/analysis/ptx_survey_all.sh --image custom   # custom CUDA image

set -e

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SURVEYS=(
    ptx_cvt_survey.sh
    ptx_mma_survey.sh
    ptx_async_survey.sh
    ptx_atomic_survey.sh
    ptx_special_survey.sh
    ptx_cluster_survey.sh
)

echo "# PTX feature acceptance survey for sm_120f"
echo ""
echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Toolkit:   $(docker run --rm nvidia/cuda:13.2.1-devel-ubuntu24.04 nvcc --version 2>/dev/null | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
echo "Arch:      compute_120f / sm_120 (RTX 5090 GB202)"
echo ""
echo "Each section is a separate ptxas-acceptance test for one PTX instruction"
echo "family. ✅ = ptxas accepts on sm_120f (instruction is callable; runtime"
echo "behavior may still differ). ❌ = ptxas rejects with the cited reason."
echo ""
echo "Re-run \`tools/analysis/ptx_survey_all.sh\` after every CUDA toolkit"
echo "upgrade — newly-supported instructions surface here as ✅ flips."
echo ""

for s in "${SURVEYS[@]}"; do
    if [[ -x "$DIR/$s" ]]; then
        echo ""
        echo "---"
        bash "$DIR/$s" "$@"
    else
        echo ""
        echo "⚠️  $s not found or not executable — skipping"
    fi
done

echo ""
echo "---"
echo ""
echo "Full survey complete."
