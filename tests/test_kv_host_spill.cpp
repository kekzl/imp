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

// A slot remembers the binding its KV came from: only that (block, serial) pairs with it (#2174).
TEST(KVHostSpillTest, SlotHoldsOnlyItsOriginBinding) {
    auto s = make(2);
    const KvChainLink a{5, 3, 42}, b{5, 7, 50};  // two forwards of hash 5
    s.reserve(KVHostSpill::slot_key(a));
    EXPECT_FALSE(s.contains(5)) << "uncommitted";
    s.commit(KVHostSpill::slot_key(a), a);
    EXPECT_TRUE(s.contains(5));
    EXPECT_TRUE(s.holds(a));
    EXPECT_FALSE(s.holds(KvChainLink{5, 3, 43})) << "a later binding of block 3";
    EXPECT_FALSE(s.holds(KvChainLink{6, 3, 42}));
    EXPECT_EQ(s.origin(5).serial, 42u);
    EXPECT_EQ(s.origin(6).block_id, -1);

    s.reserve(KVHostSpill::slot_key(b)), s.commit(KVHostSpill::slot_key(b), b);
    EXPECT_TRUE(s.holds(a)) << "each forward keeps its own slot";
    EXPECT_TRUE(s.holds(b));
    EXPECT_EQ(s.origin(5).serial, 50u) << "the hash names its newest slot";
    s.erase(KVHostSpill::slot_key(b));
    EXPECT_FALSE(s.contains(5)) << "the alias goes with its slot";
    EXPECT_TRUE(s.holds(a));
}

// The arena is pinned a chunk at a time, as slots are first used.
TEST(KVHostSpillTest, ChunksArePinnedOnFirstUse) {
    static int allocs;
    allocs = 0;
    KVHostSpill s(
        5 * 64, 64,
        [](size_t n) {
            ++allocs;
            return std::malloc(n);
        },
        [](void* p) { std::free(p); }, /*chunk_bytes=*/2 * 64);
    ASSERT_TRUE(s.enabled());
    EXPECT_EQ(s.slots(), 5);
    EXPECT_EQ(s.pinned_slots(), 2);
    EXPECT_EQ(allocs, 1);
    for (size_t k = 1; k <= 5; ++k) {
        void* p = s.reserve(k);
        ASSERT_NE(p, nullptr);
        std::memset(p, static_cast<int>(k), 64);
        s.commit(k);
    }
    EXPECT_EQ(allocs, 3) << "2 + 2 + 1 slots";
    EXPECT_EQ(s.pinned_slots(), 5);
    EXPECT_EQ(s.replacements(), 0u);
    for (size_t k = 1; k <= 5; ++k) {
        const auto* got = static_cast<const unsigned char*>(s.find(k));
        ASSERT_NE(got, nullptr);
        EXPECT_EQ(got[0], k) << "slots of different chunks do not overlap";
        EXPECT_EQ(got[63], k);
    }
    s.reserve(6), s.commit(6);
    EXPECT_EQ(allocs, 3);
    EXPECT_EQ(s.replacements(), 1u) << "at the ceiling the LRU slot is replaced";
}

// A refused chunk leaves the tier at what it has: it replaces among the pinned slots.
TEST(KVHostSpillTest, RefusedChunkCapsTheTier) {
    static int allocs;
    allocs = 0;
    KVHostSpill s(
        6 * 64, 64, [](size_t n) { return ++allocs <= 1 ? std::malloc(n) : nullptr; },
        [](void* p) { std::free(p); }, /*chunk_bytes=*/2 * 64);
    ASSERT_TRUE(s.enabled());
    s.reserve(1), s.commit(1);
    s.reserve(2), s.commit(2);
    ASSERT_NE(s.reserve(3), nullptr);
    s.commit(3);
    EXPECT_EQ(s.slots(), 2);
    EXPECT_EQ(s.replacements(), 1u);
    EXPECT_EQ(s.find(1), nullptr);
    s.reserve(4), s.commit(4);
    EXPECT_EQ(allocs, 2) << "no second try after a refusal";
}

TEST(KVHostSpillTest, RefusedFirstChunkIsOff) {
    KVHostSpill s(4 * 64, 64, [](size_t) -> void* { return nullptr; }, [](void* p) { std::free(p); });
    EXPECT_FALSE(s.enabled());
    EXPECT_EQ(s.reserve(1), nullptr);
}

}  // namespace
}  // namespace imp
