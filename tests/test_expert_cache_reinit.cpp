// ExpertLRUCache::reinit_or_disable with both budgets refused before any CUDA call:
// the cache must end disabled (pool_ == nullptr, n_slots_ == 0), the state every consumer checks.

#include <gtest/gtest.h>
#include "exec/expert_cache.h"

namespace imp {
namespace {

TEST(ExpertCacheReinitCpu, BothBudgetsRefusedLeavesCacheDisabled) {
    constexpr size_t kSlot = 1 << 20;
    ExpertLRUCache cache;
    // Grown: 0 slots/layer. Original: 1 slot total, init's n_slots_ < 2 exit, which alone
    // leaves n_slots_ = 1 and slots_per_layer_ = 1 with pool_ == nullptr.
    EXPECT_EQ(cache.reinit_or_disable(kSlot, kSlot / 2, kSlot, /*alloc=*/nullptr, /*n_layers=*/1,
                                      /*n_experts=*/8, /*debug_parity=*/false, /*nvfp4_slots=*/true),
              0u);
    EXPECT_EQ(cache.pool_, nullptr);
    EXPECT_EQ(cache.n_slots_, 0);
    EXPECT_EQ(cache.slots_per_layer_, 0);
    EXPECT_FALSE(cache.nvfp4_slots_);
    EXPECT_EQ(cache.layer_slot_scales(0), nullptr);
}

}  // namespace
}  // namespace imp
