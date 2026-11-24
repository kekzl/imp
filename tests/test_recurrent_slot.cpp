// Regression: Engine::reset_ssm_state fell back to slot `seq_id % cap` for an id without a
// slot. The server cancels requests that still wait for admission (no prefill, no slot), and
// that slot belonged to another running sequence, whose GDN/SSM state was zeroed.

#include "runtime/recurrent_slot.h"

#include <gtest/gtest.h>

using imp::recurrent_slot_for_reset;
using imp::RecurrentSlotMap;

TEST(RecurrentSlotForReset, KnownIdReturnsItsSlot) {
    const RecurrentSlotMap slot_of{{7, 3}, {8, 0}};
    EXPECT_EQ(recurrent_slot_for_reset(slot_of, 7), 3);
    EXPECT_EQ(recurrent_slot_for_reset(slot_of, 8), 0);
}

TEST(RecurrentSlotForReset, UnknownIdResetsNothing) {
    EXPECT_FALSE(recurrent_slot_for_reset({}, 0).has_value());
    EXPECT_FALSE(recurrent_slot_for_reset({{7, 3}}, 8).has_value());
}

TEST(RecurrentSlotForReset, UnknownIdNeverAliasesAnotherSequencesSlot) {
    // cap 4: request 5 runs in slot 1; request 9 waits for admission and 9 % 4 == 1.
    const int cap = 4;
    const RecurrentSlotMap slot_of{{5, 1}};
    ASSERT_EQ(9 % cap, slot_of.at(5));
    EXPECT_EQ(recurrent_slot_for_reset(slot_of, 9), std::nullopt);
    // Pool order, not the id, picks the slot: the first request pops slot 0 whatever its id.
    const RecurrentSlotMap first{{2, 0}};
    EXPECT_EQ(recurrent_slot_for_reset(first, 4), std::nullopt);  // 4 % 4 == 0
}
