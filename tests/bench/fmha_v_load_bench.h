#pragma once
// Validates whether TMA bulk beats per-thread cp.async for the FMHA V-prefetch load pattern
// (Bkv x head_dim FP16 tile). Phase 1 gate for the LDGSTS->TMA lever (memory file
// hw_capability_audit_complete_2026_05_10), expected 5-15% kernel / 3-5% E2E decode.

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
