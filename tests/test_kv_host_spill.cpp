// KVHostSpill slot bookkeeping (#2203), host-only: plain malloc stands in for the pinned arena.
#include "memory/kv_host_spill.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

namespace imp {
namespace {

KVHostSpill make(size_t slots, size_t slot_bytes = 64) {
    return KVHostSpill(
        slots * slot_bytes, slot_bytes, [](size_t n) { return std::malloc(n); },
        [](void* p) { std::free(p); });
}

TEST(KVHostSpillTest, BudgetBelowOneSlotIsOff) {
    KVHostSpill s(63, 64, [](size_t n) { return std::malloc(n); }, [](void* p) { std::free(p); });
    EXPECT_FALSE(s.enabled());
    EXPECT_EQ(s.reserve(1), nullptr);
}

TEST(KVHostSpillTest, CommittedEntryRoundTrips) {
    auto s = make(2);
    void* slot = s.reserve(7);
    ASSERT_NE(slot, nullptr);
    std::memset(slot, 0x5C, 64);
    EXPECT_EQ(s.find(7), nullptr) << "an uncommitted entry is not a hit";
    s.commit(7);
    const auto* got = static_cast<const unsigned char*>(s.find(7));
    ASSERT_NE(got, nullptr);
    EXPECT_EQ(got[63], 0x5C);
    EXPECT_EQ(s.saves(), 1u);
}

TEST(KVHostSpillTest, FullTierReplacesTheLeastRecentlyUsed) {
    auto s = make(2);
    s.reserve(1), s.commit(1);
    s.reserve(2), s.commit(2);
    ASSERT_NE(s.find(1), nullptr);  // 1 becomes most recent, 2 is the LRU
    s.reserve(3), s.commit(3);
    EXPECT_NE(s.find(1), nullptr);
    EXPECT_EQ(s.find(2), nullptr);
    EXPECT_NE(s.find(3), nullptr);
    EXPECT_EQ(s.replacements(), 1u);
    EXPECT_EQ(s.size(), 2);
}

TEST(KVHostSpillTest, SameHashReusesItsSlotAndEraseFreesIt) {
    auto s = make(1);
    void* a = s.reserve(9);
    s.commit(9);
    EXPECT_EQ(s.reserve(9), a);
    s.commit(9);
    EXPECT_EQ(s.replacements(), 0u);
    s.erase(9);
    EXPECT_EQ(s.size(), 0);
    EXPECT_NE(s.reserve(10), nullptr);
    s.abandon(10);
    EXPECT_EQ(s.find(10), nullptr);
    s.reserve(11), s.commit(11);
    s.clear();
    EXPECT_EQ(s.find(11), nullptr);
}

}  // namespace
}  // namespace imp
