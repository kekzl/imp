#pragma once

#include <cuda_runtime.h>

namespace imp {

// =============================================================================
// fp4_pv_bench.h — Phase 3a FP4 PV microbench
// =============================================================================
//
// Discriminates "MMA-level gain is real on realistic post-softmax data" vs
// "evaporates" before committing to the multi-week Phase 3b/3c production
// integration. See `docs/plans/fp4_pv_phase3_design_2026_05_17.md` §3 + §4.
//
// Two questions answered:
//
//   1. ACCURACY — single-level FP4 quant of post-softmax probabilities.
//      Generates N synthetic attention rows (softmax over Gaussian logits +
//      spike), quantises each row to FP4 + UE4M3 per-16 scales, and
//      compares the "FP4 P @ FP4 V" dot product against an FP32 reference.
//      Reports relative-error percentiles. THIS IS THE KEY DATA POINT —
//      if single-level FP4 PV has catastrophic relative error (>~50 %),
//      Phase 3b's two-level accumulator is mandatory; if it's bounded
//      (<5 %), Phase 3 could ship without the accumulator complexity.
//
//   2. THROUGHPUT — raw MMA-pipeline TOPS for mxf4nvf4 m16n8k64 vs the
//      WMMA m16n16k16 PV reference. The existing mxf4nvf4_mma_bench
//      compared mxf4nvf4 vs f8f6f4; this adds the missing HMMA reference
//      that the PV path actually replaces.
//
// Both run on the GPU and emit results to stdout. The accuracy half does
// the quantisation in software (host-side reconstruction matches what the
// PTX `cvt.rn.satfinite.e2m1x2.f32` + `cvt.rn.satfinite.e4m3.f32` produce
// modulo IEEE rounding at midpoints; the numerical signal of interest —
// long-tail truncation in softmax distributions — is independent of the
// rounding mode).
// =============================================================================

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

// Phase 3b: two-level accumulator simulation.
//
// O_2L = P_coarse_fp4 @ V_fp4 + (P - P_coarse_fp4) @ V_fp16
//      = P_lossy @ V_lossy + P_residual @ V_orig
//
// The residual MMA uses FP16 P (the residual) and FP16 V (unquantised).
// Mathematically equivalent for 2L-A (sparse residual exploiting
// structural near-zero entries) and 2L-B (full HMMA residual) — they
// only differ in WALL TIME, not in numerical result. This function
// answers the accuracy half ("does the residual recover the long tail
// FP4 truncates?"); throughput is the separate question Phase 3b's gate
// uses bench_fp4_pv_2level_throughput_estimate for.
//
// Returns the same struct as the Phase 3a single-level accuracy bench
// so percentile-by-percentile A/B is direct.
Fp4PvAccuracyResult bench_fp4_pv_accuracy_2level(int n_rows, int K, int head_dim,
                                                 unsigned seed);

struct Fp4PvTwoLevelThroughputEstimate {
    float coarse_ms;         // mxf4nvf4 m16n8k64 alone — same as throughput.blockscale_ms
    float residual_full_ms;  // HMMA m16n8k16 alone — same as throughput.hmma_ms
    // 2L-A (structural-sparse residual): the SageAttention3 paper claims
    // ~10 % of full HMMA cost for the sparse path because P_residual is
    // mostly near-zero after FP4 quant. THIS IS A PAPER PROJECTION, not
    // a measurement — implementing the sparse residual is itself a real
    // research item. The microbench reports the projection so Phase 3c
    // decisions are based on a known number, not a guess.
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
