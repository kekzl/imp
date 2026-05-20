#pragma once
// =============================================================================
// attention_prefill_paths_bench.h — Säule 1+2 of Track E gating bench
// =============================================================================
//
// Empirically benches the two prefill attention paths on the production shape
// matrix to decide if Track E (tiled streaming softmax) is worth the 10-15 dev
// days of new-kernel work, or whether existing FMHA can replace cuBLAS without
// a new kernel (path b). See docs/superpowers/specs/2026-05-20-track-e-tiled-
// streaming-softmax-design.md for context.
//
// Säule 1: attention_cublas_prefill   (cuBLAS QKᵀ + softmax + PV, 1 GiB S-mat)
// Säule 2: fmha_sm120_prefill         (tiled FA2-style, no S-matrix)
//
// Methodology (per memory file bench_methodology_2026_05_15):
//   - CUBLAS_WORKSPACE_CONFIG=:4096:8 caller responsibility
//   - 3 warmup + 10 reps per measurement, report median
//   - all FP16 inputs (prefill paths don't accept other dtypes)
// =============================================================================

namespace imp {

struct AttnPrefillBenchResult {
    // Per-path median elapsed ms (across kReps reps). NaN if path failed.
    double cublas_ms;
    double fmha_ms;

    // GFLOPS achieved (forward attention: 4 * nh * seq² * hd for causal half).
    double cublas_gflops;
    double fmha_gflops;

    // Workspace bytes used by S-matrix (cuBLAS only; FMHA is O(1) per CTA).
    long long cublas_s_workspace_bytes;
};

// Run both paths on a single shape and fill `out`.
// Returns false on allocation / launch failure of EITHER path; partial result
// (NaN ms for the failing path) is still written.
bool attention_prefill_paths_bench(int seq, int n_heads, int n_kv_heads,
                                   int head_dim, AttnPrefillBenchResult* out);

}  // namespace imp
