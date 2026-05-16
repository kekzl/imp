// =============================================================================
// test_fmha_v_load_bench.cu — gtest harness for fmha_v_load_bench
// =============================================================================
// Phase 1 gate for the LDGSTS→TMA conversion lever. The bench runs both
// variants on the FMHA V-load shapes (HD=64/128, Bkv=64/128) and prints
// per-variant bandwidth + speedup. Test passes if both variants run cleanly.
// The threshold for committing to the multi-week integration is in the memo —
// not enforced as a gate here so we can see all data even on a regression.
// =============================================================================

#include "bench/fmha_v_load_bench.h"
#include <gtest/gtest.h>
#include <cstdio>

TEST(FmhaVLoadBench, CpAsyncVsTmaBulk_HD128_Bkv128) {
    imp::FmhaVLoadBenchResult r{};
    bool ok = imp::fmha_v_load_bench(/*Bkv=*/128, /*head_dim=*/128, &r);
    ASSERT_TRUE(ok) << "TMA descriptor build or kernel launch failed";

    std::printf(
        "\n=== FMHA V-load bench HD=128 Bkv=128 (32 KiB tile) ===\n"
        "  cp.async  : %8.3f ms  %7.1f GB/s\n"
        "  TMA bulk  : %8.3f ms  %7.1f GB/s\n"
        "  speedup   : %6.3fx (cp.async / TMA)\n",
        r.cp_async_ms, r.cp_async_gb_per_s,
        r.tma_bulk_ms, r.tma_bulk_gb_per_s,
        r.speedup);
    std::fflush(stdout);
}

TEST(FmhaVLoadBench, CpAsyncVsTmaBulk_HD128_Bkv64) {
    imp::FmhaVLoadBenchResult r{};
    bool ok = imp::fmha_v_load_bench(/*Bkv=*/64, /*head_dim=*/128, &r);
    ASSERT_TRUE(ok);

    std::printf(
        "\n=== FMHA V-load bench HD=128 Bkv=64 (16 KiB tile) ===\n"
        "  cp.async  : %8.3f ms  %7.1f GB/s\n"
        "  TMA bulk  : %8.3f ms  %7.1f GB/s\n"
        "  speedup   : %6.3fx\n",
        r.cp_async_ms, r.cp_async_gb_per_s,
        r.tma_bulk_ms, r.tma_bulk_gb_per_s,
        r.speedup);
    std::fflush(stdout);
}

TEST(FmhaVLoadBench, CpAsyncVsTmaBulk_HD64_Bkv128) {
    imp::FmhaVLoadBenchResult r{};
    bool ok = imp::fmha_v_load_bench(/*Bkv=*/128, /*head_dim=*/64, &r);
    ASSERT_TRUE(ok);

    std::printf(
        "\n=== FMHA V-load bench HD=64 Bkv=128 (16 KiB tile) ===\n"
        "  cp.async  : %8.3f ms  %7.1f GB/s\n"
        "  TMA bulk  : %8.3f ms  %7.1f GB/s\n"
        "  speedup   : %6.3fx\n",
        r.cp_async_ms, r.cp_async_gb_per_s,
        r.tma_bulk_ms, r.tma_bulk_gb_per_s,
        r.speedup);
    std::fflush(stdout);
}

TEST(FmhaVLoadBench, CpAsyncVsTmaBulk_HD256_Bkv64) {
    imp::FmhaVLoadBenchResult r{};
    bool ok = imp::fmha_v_load_bench(/*Bkv=*/64, /*head_dim=*/256, &r);
    ASSERT_TRUE(ok);

    std::printf(
        "\n=== FMHA V-load bench HD=256 Bkv=64 (32 KiB tile) ===\n"
        "  cp.async  : %8.3f ms  %7.1f GB/s\n"
        "  TMA bulk  : %8.3f ms  %7.1f GB/s\n"
        "  speedup   : %6.3fx\n",
        r.cp_async_ms, r.cp_async_gb_per_s,
        r.tma_bulk_ms, r.tma_bulk_gb_per_s,
        r.speedup);
    std::fflush(stdout);
}
