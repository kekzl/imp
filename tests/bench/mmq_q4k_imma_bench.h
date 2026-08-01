#pragma once

#include <cuda_runtime.h>

namespace imp {

// Raw-MMA throughput microbench for the INT8 Tensor-Core variants candidate
// for a future Q4_K_M direct GEMM kernel on sm_120a. Phase 1 of the INT8 IMMA
// direct-GEMM experiment (outcome: docs/plans/2026-05-28-q4k-mmq-kernel-design.md).
//
// Each entry runs a tight per-warp loop of one MMA opcode with all operands
// alive (so ptxas can't constant-fold the loop body), measures wall-clock,
// and divides by ops_per_mma * total_mma_calls to get TOPS / TFLOPS.
//
// The relative ratio (INT8 IMMA TOPS / FP16 HMMA TFLOPS) is the gate:
// theoretical sm_120 should show ~2.0×. If the measured ratio is < 1.5×,
// INT8 is throttled to FP16-peak on consumer Blackwell and the Q4_K_M IMMA
// project should be DEFERRED.

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
