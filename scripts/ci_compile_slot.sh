#!/bin/sh
# CI CUDA compiler launcher. Peak RSS, cold build, nvcc 13.4, 727 TUs: the 5 TUs below
# 4.97-8.56 GB, all others <= 1.27 GB. Lock A + lock B: <= 2 heavy at once, worst pair
# 7.44 + 5.22 = 12.7 GB; the 8.56 GB bench holds both and runs alone.
lock=/tmp/imp-heavy-tu
for a in "$@"; do
    case "${a##*/}" in
        gemm_cutlass_sm120.cu) exec flock "$lock.A" "$@" ;;
        gemm_cutlass_grouped_3x.cu | gemm_cutlass_mxfp4_sm120.cu | gemm_cutlass_mxfp8_sm120.cu)
            exec flock "$lock.B" "$@" ;;
        test_cutlass_grouped_tile_bench.cu) exec flock "$lock.A" flock "$lock.B" "$@" ;;
    esac
done
exec "$@"
