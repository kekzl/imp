#!/usr/bin/env bash
# Parallel compile jobs that fit in RAM, printed for `cmake --build -j`.
#
# The four CUTLASS TUs peak at 5.2-7.9 GB resident each (nvcc 13.4, measured
# 2026-09-19: gemm_cutlass_sm120 7.85 GB, grouped_3x 5.49, mxfp4 5.36, mxfp8
# 5.24) and the build starts all four within 7 s of each other. On a 16 GB CI
# runner -j4 is an OOM kill (exit 137 plus "runner has received a shutdown
# signal", no compiler diagnostic). A warm ccache hides it - the objects come
# back without running cicc - so only a cold cache, e.g. the first build after
# a CUDA_VERSION bump rotates the cache key, hits it.
#
# 6 GB per job: two of the heaviest TUs (7.85 + 5.49) stay inside a 16 GB
# runner, and hosts with real memory still get every core.
set -u

cpus=$(nproc)
mem_mb=$(awk '/MemTotal/ {printf "%d", $2 / 1024}' /proc/meminfo)

# /proc/meminfo reports the host inside a container: a cgroup limit is the real
# ceiling there (v2 prints "max" when there is none).
cg=/sys/fs/cgroup/memory.max
if [ -r "$cg" ]; then
    lim=$(cat "$cg")
    if [ "$lim" != "max" ]; then
        lim_mb=$((lim / 1024 / 1024))
        [ "$lim_mb" -lt "$mem_mb" ] && mem_mb=$lim_mb
    fi
fi

by_ram=$((mem_mb / 6144))
[ "$by_ram" -lt 1 ] && by_ram=1

if [ "$by_ram" -lt "$cpus" ]; then echo "$by_ram"; else echo "$cpus"; fi
