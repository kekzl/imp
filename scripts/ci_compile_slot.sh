#!/bin/sh
# CI CUDA compiler launcher. Peak RSS, cold build, nvcc 13.4: the CUTLASS TU below is the one heavy
# compile (all others <= 1.27 GB); lock A runs it alone.
lock=/tmp/imp-heavy-tu
for a in "$@"; do
    case "${a##*/}" in
        gemm_cutlass_sm120.cu) exec flock "$lock.A" "$@" ;;
    esac
done
exec "$@"
