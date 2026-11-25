// InklingSlotTable (exec/inkling_slot_table.h): an Inkling row finds its slot by its request id.
// Before (#3272) the key was the sequence's first KV block id, which the pool hands out again.

#include <gtest/gtest.h>

#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "exec/inkling_slot_table.h"

using imp::InklingSlotTable;

namespace {

// Stand-in for InklingRuntime::bind's rank: longest kept prefix first.
struct Kept {
    std::vector<std::vector<int>> seq;
    auto rank(std::span<const int> prompt) const {
        return [this, prompt](size_t s) {
            size_t m = 0;
            while (m < seq[s].size() && m < prompt.size() && seq[s][m] == prompt[m])
                ++m;
            return std::tuple(-static_cast<int>(m));
        };
    }
};

std::vector<int> slots_of(const InklingSlotTable& t, const std::vector<int>& ids) {
    std::vector<int> out;
    t.rows_to_slots(ids.data(), static_cast<int>(ids.size()), out);
    return out;
}

}  // namespace

TEST(InklingSlotTable, RowsFindTheirSlotByRequestId) {
    InklingSlotTable t(3);
    Kept k{{{}, {}, {}}};
    const std::vector<int> p = {1, 2, 3};
    ASSERT_EQ(t.bind(40, k.rank(p)), 0);
    ASSERT_EQ(t.bind(41, k.rank(p)), 1);
    ASSERT_EQ(t.bind(42, k.rank(p)), 2);
    EXPECT_EQ(t.bind(43, k.rank(p)), -1);  // every slot bound: busy = bound
    // Rows arrive in any order; each finds the slot its request was bound to.
    EXPECT_EQ(slots_of(t, {42, 40, 41}), (std::vector<int>{2, 0, 1}));
    EXPECT_EQ(slots_of(t, {41}), (std::vector<int>{1}));
    EXPECT_EQ(t.slot_of(40), 0);
    EXPECT_EQ(t.req_of(2), 42);
}

// The #3272 shape: request 7 decoded on slot 0 with first KV block 0 and finished; the pool hands
// block 0 to request 9, which binds to slot 1 while request 8 holds slot 0. Rows carry 8 and 9, both
// of whose first blocks are 0 at some point: neither may land on the other's slot.
TEST(InklingSlotTable, ARepeatedFirstBlockIdCannotMisroute) {
    InklingSlotTable t(2);
    Kept k{{{5, 5, 5}, {}}};
    const std::vector<int> a = {5, 5, 5, 6}, b = {7};
    ASSERT_EQ(t.bind(7, k.rank(a)), 0);
    t.release(7);
    ASSERT_EQ(t.bind(8, k.rank(a)), 0);  // the kept prefix pulls 8 onto slot 0
    ASSERT_EQ(t.bind(9, k.rank(b)), 1);
    const std::vector<int> first_block = {0, 0};  // what the block-id key saw for both rows
    ASSERT_EQ(first_block[0], first_block[1]);
    EXPECT_EQ(slots_of(t, {9, 8}), (std::vector<int>{1, 0}));
    EXPECT_EQ(slots_of(t, {8, 9}), (std::vector<int>{0, 1}));
    EXPECT_THROW((void)slots_of(t, {7}), std::runtime_error);  // the finished request has no slot
}

TEST(InklingSlotTable, UnknownIdThrows) {
    InklingSlotTable t(2);
    Kept k{{{}, {}}};
    ASSERT_EQ(t.bind(3, k.rank({})), 0);
    EXPECT_THROW((void)t.slot_of(4), std::runtime_error);   // never bound, not slot 0
    EXPECT_THROW((void)t.slot_of(-1), std::runtime_error);  // -1 marks a free slot, never a key
    EXPECT_THROW((void)slots_of(t, {3, 4}), std::runtime_error);
    EXPECT_THROW((void)slots_of(t, {3, 3}), std::runtime_error);  // two rows of one sequence
    std::vector<int> out;
    EXPECT_THROW(t.rows_to_slots(nullptr, 1, out), std::runtime_error);
    EXPECT_THROW((void)t.bind(-1, k.rank({})), std::invalid_argument);
    EXPECT_EQ(t.find(4), -1);
}

// Release frees the slot and keeps its sequence: the next prompt with that prefix binds there,
// although the other free slot holds the older binding.
TEST(InklingSlotTable, ReleaseKeepsPrefixUnbinds) {
    InklingSlotTable t(2);
    Kept k{{{}, {}}};
    const std::vector<int> a = {1, 2, 3, 4}, other = {9};
    ASSERT_EQ(t.bind(11, k.rank(other)), 0);  // slot 0: the older binding
    ASSERT_EQ(t.bind(10, k.rank(a)), 1);
    k.seq[0] = other;
    k.seq[1] = a;
    t.release(10);
    t.release(10);  // idempotent
    EXPECT_FALSE(t.bound(1));
    EXPECT_EQ(t.slot_of(11), 0);  // another request's release leaves 11 bound
    t.release(11);
    EXPECT_FALSE(t.bound(0));
    EXPECT_EQ(t.find(10), -1);
    EXPECT_THROW((void)t.slot_of(10), std::runtime_error);
    EXPECT_EQ(t.bind(12, k.rank(a)), 1);  // slot 1 kept a (4 positions), slot 0 none
    EXPECT_EQ(t.bind(13, k.rank(a)), 0);  // slot 1 busy
    t.release(12);
    // A request bound again (re-prefill) drops its old binding first: one slot per id.
    EXPECT_EQ(t.bind(13, k.rank(a)), 1);
    EXPECT_FALSE(t.bound(0));
    EXPECT_EQ(slots_of(t, {13}), (std::vector<int>{1}));
}
