#!/bin/sh
# CI CUDA compiler launcher: a *cutlass*.cu TU holds one global lock, all other TUs run free.
# Peak RSS, cold build, nvcc 13.4, 727 TUs: 5 cutlass TUs 4.97-8.56 GB, all others <= 1.27 GB.
# -j4 with one heavy at a time <= 12.4 GB on the 16 GB runner; two heavy reach 16.0 GB.
for a in "$@"; do
    case "${a##*/}" in
        *cutlass*.cu) exec flock "${IMP_HEAVY_TU_LOCK:-/tmp/imp-heavy-tu.lock}" "$@" ;;
    esac
done
exec "$@"
