#pragma once

#include <cuda_runtime.h>

namespace imp {

// Discriminates whether MMA-level FP4 PV gain survives realistic post-softmax data before
// committing to Phase 3b/3c integration.
// Accuracy: >~50% rel error vs an FP32 ref mandates the two-level accumulator; <5% needs none.
// Throughput: mxf4nvf4 m16n8k64 vs the missing HMMA PV reference.

struct Fp4PvAccuracyResult {
    int n_rows;
    int K;          // P width / V height (must be a multiple of 16)
    int head_dim;   // V width
    // Relative error percentiles (vs FP32 reference). Cell value is
    // |O_fp4 - O_ref| / (|O_ref| + 1e-9).
    float rel_err_median;
    float rel_err_p90;
    float rel_err_p99;
    float rel_err_max;
    // Absolute error percentiles (same indices).
    float abs_err_median;
    float abs_err_p99;
    float abs_err_max;
    // Fraction of outputs with relative error > 0.5 (i.e. catastrophic
    // tail truncation). High value confirms Phase 3b is mandatory.
    float frac_rel_err_above_50pct;
};

struct Fp4PvThroughputResult {
    float hmma_ms;          // wmma m16n16k16 PV reference, avg ms per rep
    float blockscale_ms;    // mxf4nvf4.block_scale m16n8k64 PV, avg ms per rep
    double hmma_tops;
    double blockscale_tops;
    double speedup;         // blockscale / hmma
};

// Generate synthetic post-softmax rows + quantise + measure relative error
// vs FP32 reference dot product. Deterministic given `seed`.
Fp4PvAccuracyResult bench_fp4_pv_accuracy(int n_rows, int K, int head_dim,
                                          unsigned seed);

// Raw-instruction throughput comparison: HMMA m16n16k16 (PV reference)
// vs mxf4nvf4 m16n8k64. Same warps × iterations pattern as
// mxf4nvf4_mma_bench so numbers compose with that existing surface.
Fp4PvThroughputResult bench_fp4_pv_throughput(int warps, int iterations,
                                              cudaStream_t stream);

// O_2L = P_coarse_fp4@V_fp4 + (P-P_coarse_fp4)@V_fp16 = P_lossy@V_lossy + P_residual@V_orig.
// 2L-A (sparse residual) and 2L-B (full HMMA residual) are numerically equivalent, differing
// only in wall time. Answers accuracy; throughput is bench_fp4_pv_2level_throughput_estimate.
Fp4PvAccuracyResult bench_fp4_pv_accuracy_2level(int n_rows, int K, int head_dim,
                                                 unsigned seed);

struct Fp4PvTwoLevelThroughputEstimate {
    float coarse_ms;         // mxf4nvf4 m16n8k64 alone — same as throughput.blockscale_ms
    float residual_full_ms;  // HMMA m16n8k16 alone — same as throughput.hmma_ms
    // 2L-A sparse-residual ~10% of full HMMA cost is a SageAttention3 PAPER PROJECTION, not
    // measured; implementing the sparse residual is itself a research item.
    float estimated_2l_a_ms;
    // 2L-B (full HMMA on residual): the FP4 + HMMA can overlap on
    // independent MMA pipes, so combined wall time ≈ max(coarse, residual),
    // not coarse + residual. This is the easier-to-implement variant.
    float estimated_2l_b_ms;
    // Throughput ratio vs HMMA-alone baseline (today's PV path).
    double speedup_2l_a;
    double speedup_2l_b;
};

// Throughput estimate for the two-level paths. Uses the same per-kernel
// timings as bench_fp4_pv_throughput plus model coefficients for the
// sparse residual cost.
Fp4PvTwoLevelThroughputEstimate bench_fp4_pv_2level_throughput_estimate(
    const Fp4PvThroughputResult& single);

}  // namespace imp
