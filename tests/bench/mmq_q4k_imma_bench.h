#pragma once

#include <cuda_runtime.h>

namespace imp {

// Raw-MMA throughput microbench for INT8 TC candidates for a Q4_K_M direct GEMM kernel
// (docs/plans/2026-05-28-q4k-mmq-kernel-design.md).
// Ratio (INT8 IMMA / FP16 HMMA) is the gate: sm_120 theoretical ~2.0x; <1.5x means INT8 is
// throttled to FP16-peak and the project should be DEFERRED.

struct ImmaBenchResult {
    static constexpr int kMaxEntries = 8;
    struct Entry {
        const char* label;
        float ms;
        double tops;  // 10^12 ops/sec; FMAs counted as 2 ops
        double ops_per_mma;
    };
    Entry entries[kMaxEntries];
    int count = 0;
};

ImmaBenchResult bench_mmq_q4k_imma(int warps, int iterations, cudaStream_t stream);

}  // namespace imp
