#pragma once
// Inkling slots are found by the first KV block of the sequence they hold (decode rows carry block
// tables, not request ids). A block id belongs to one live sequence: when slot s takes a sequence
// whose first block is `block`, any other slot still naming that id held a sequence that is gone.
// Without this a freed id handed out again (BlockPool gives the lowest free id) sent the new
// sequence's decode rows to the old slot.

#include <cstddef>
#include <vector>

namespace imp {

inline void inkling_claim_block(std::vector<int>& slot_block, size_t s, int block) {
    for (size_t i = 0; i < slot_block.size(); ++i)
        if (i != s && slot_block[i] == block)
            slot_block[i] = -1;
    slot_block[s] = block;
}

}  // namespace imp
