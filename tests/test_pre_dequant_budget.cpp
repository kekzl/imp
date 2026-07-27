// Pre-dequant budget split (#1100): who pays for the NVFP4 decode cache.
//
// The cache build runs AFTER the KV pool is allocated, so the KV bytes are
// already gone from free_vram. On top of that, the NVFP4 decode cache's own
// reservation is withheld from the early phases (Phase 1 FP16 / Phase 2 FP8),
// which would otherwise hand out VRAM the not-yet-allocated cache needs.
//
// The bug: that reservation was subtracted from the SHARED budget, which is
// also the budget Phase 3 — the NVFP4 decode cache itself — spends from. The
// cache paid for itself twice, so every byte the KV pool took came out of it a
// second time. On Qwen3-14B-Q6_K at the server's full-context default
// (max_seq_len=40960 → 5.9 GiB KV pool) the cache fell from 278/280 to 100/280
// tensors and decode dropped 38%, while ~11 GiB of VRAM sat free.
//
// CPU-only by construction: split_pre_dequant_budget is pure arithmetic.

#include <gtest/gtest.h>

#include "runtime/vram_budget.h"

using namespace imp;

namespace {
constexpr size_t MiB = 1024ull * 1024;
}  // namespace

// The regression itself, in the numbers the server actually produced: 11.3 GiB
// free after the KV pool, a 3261 MiB reserve, a 7505 MiB NVFP4 reservation.
// The cache needs ~7021 MiB and MUST fit — before the fix it saw 833 MiB.
TEST(PreDequantBudget, Nvfp4ReservationIsChargedToEarlyPhasesOnly) {
    const size_t free_vram = 11599 * MiB;
    const size_t reserve = 3261 * MiB;
    const size_t nvfp4 = 7505 * MiB;

    PreDequantBudget b = split_pre_dequant_budget(free_vram, reserve, nvfp4);

    // Phase 3 sees the real post-reserve ceiling...
    EXPECT_EQ(b.shared, free_vram - reserve);
    // ...while Phases 1/2 still withhold the reservation.
    EXPECT_EQ(b.early, free_vram - reserve - nvfp4);
    // The property that regressed: the reservation fits in the budget the
    // reserved-for phase actually spends from.
    EXPECT_GE(b.shared, nvfp4) << "decode cache cannot afford its own reservation";
}

// Growing the KV pool must cost the decode cache 1:1, not 2:1. This is the
// mechanism behind the capacity tax: raising runtime.max_seq_len grows the KV
// pool, which shrinks free_vram, which used to be double-charged.
TEST(PreDequantBudget, KvGrowthCostsTheSharedBudgetExactlyOnce) {
    const size_t reserve = 3261 * MiB;
    const size_t nvfp4 = 7505 * MiB;
    const size_t kv_growth = 5609 * MiB;  // 320 MiB pool → 5929 MiB pool
    const size_t free_small_kv = 17787 * MiB;

    PreDequantBudget small = split_pre_dequant_budget(free_small_kv, reserve, nvfp4);
    PreDequantBudget large = split_pre_dequant_budget(free_small_kv - kv_growth, reserve, nvfp4);

    EXPECT_EQ(small.shared - large.shared, kv_growth);
    EXPECT_EQ(small.early - large.early, kv_growth);
}

// Both budgets clamp at zero instead of wrapping: free_vram below the reserve,
// and a reservation larger than the post-reserve budget, are both reachable on
// a full card and must not underflow into a multi-exabyte budget.
TEST(PreDequantBudget, ClampsAtZeroInsteadOfWrapping) {
    PreDequantBudget starved = split_pre_dequant_budget(1 * MiB, 512 * MiB, 256 * MiB);
    EXPECT_EQ(starved.shared, 0u);
    EXPECT_EQ(starved.early, 0u);

    // Post-reserve budget smaller than the reservation: early phases get
    // nothing, but the shared budget stays intact for the cache itself.
    PreDequantBudget tight = split_pre_dequant_budget(4096 * MiB, 1024 * MiB, 8192 * MiB);
    EXPECT_EQ(tight.shared, 3072u * MiB);
    EXPECT_EQ(tight.early, 0u);
}

// With no NVFP4 cache planned (FP16_ONLY strategy) the two budgets coincide —
// nothing is withheld from the early phases.
TEST(PreDequantBudget, NoNvfp4ReservationLeavesEarlyBudgetWhole) {
    PreDequantBudget b = split_pre_dequant_budget(8192 * MiB, 1024 * MiB, 0);
    EXPECT_EQ(b.shared, 7168u * MiB);
    EXPECT_EQ(b.early, b.shared);
}
