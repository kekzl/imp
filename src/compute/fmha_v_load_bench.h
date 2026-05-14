#pragma once
// =============================================================================
// fmha_v_load_bench.h — cp.async vs TMA bulk microbench for FMHA V-tile loads
// =============================================================================
//
// Empirically validates whether TMA bulk (`cp.async.bulk.tensor.2d`) beats
// per-thread cp.async (`cp.async.ca.shared.global`) for the FMHA V-prefetch
// load pattern (Bkv × head_dim FP16 tile). Phase 1 gate for the Tier-2 lever
// "LDGSTS → TMA migration for hand-rolled attention kernels" documented in
// memory file `hw_capability_audit_complete_2026_05_10` (expected 5-15%
// kernel speed, 3-5% E2E decode).
//
// Output: prints per-variant bandwidth (GB/s) and speedup ratio. Returns
// `true` on success, `false` if TMA descriptor build failed.
// =============================================================================

namespace imp {

struct FmhaVLoadBenchResult {
    double cp_async_ms;
    double tma_bulk_ms;
    double speedup;
    double cp_async_gb_per_s;
    double tma_bulk_gb_per_s;
};

// Bench: load `iters` tiles of (Bkv × head_dim FP16) into shared memory.
// One CTA per SM. 128 threads per CTA.
bool fmha_v_load_bench(int Bkv, int head_dim, FmhaVLoadBenchResult* out);

}  // namespace imp
