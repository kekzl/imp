// inkling_claim_block (exec/inkling_slot_map.h): a slot is found by its sequence's first KV block.

#include <gtest/gtest.h>

#include <vector>

#include "exec/inkling_slot_map.h"

// Slot 0 held a sequence that started at block 0 and finished; its blocks are free again. A new
// request bound to slot 1 gets block 0 as its first block (the pool hands out the lowest free id).
// Decode rows of the new sequence carry block 0 and have to find slot 1, not slot 0.
TEST(InklingSlotMap, AReusedFirstBlockLeavesTheOldSlot) {
    std::vector<int> slot_block = {0, -1};
    imp::inkling_claim_block(slot_block, 1, 0);
    EXPECT_EQ(slot_block[0], -1);
    EXPECT_EQ(slot_block[1], 0);
    int found = -1;
    for (size_t i = 0; i < slot_block.size() && found < 0; ++i)
        if (slot_block[i] == 0)
            found = static_cast<int>(i);
    EXPECT_EQ(found, 1);
}

// The same slot taking its block again, and another slot's different block, change nothing else.
TEST(InklingSlotMap, OtherBlocksStay) {
    std::vector<int> slot_block = {7, 3, -1};
    imp::inkling_claim_block(slot_block, 0, 7);
    EXPECT_EQ(slot_block, (std::vector<int>{7, 3, -1}));
    imp::inkling_claim_block(slot_block, 2, 9);
    EXPECT_EQ(slot_block, (std::vector<int>{7, 3, 9}));
    imp::inkling_claim_block(slot_block, 2, 3);
    EXPECT_EQ(slot_block, (std::vector<int>{7, -1, 3}));
}
