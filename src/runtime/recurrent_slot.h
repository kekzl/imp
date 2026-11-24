#pragma once

// Slot lookup for the recurrent (SSM/GDN) state pool. Header-only and CUDA-free so the CPU
// lane pins it: an id without an entry holds no slot, and no slot is derived from the id.

#include <optional>
#include <unordered_map>

namespace imp {

using RecurrentSlotMap = std::unordered_map<int, int>;  // req.id -> slot

// Slot a teardown may zero: only the one seq_id holds. nullopt = reset nothing; `seq_id % cap`
// names a slot another live sequence can hold (slots are assigned at prefill, not by id).
[[nodiscard]] inline std::optional<int> recurrent_slot_for_reset(const RecurrentSlotMap& slot_of,
                                                                 int seq_id) {
    const auto it = slot_of.find(seq_id);
    if (it == slot_of.end())
        return std::nullopt;
    return it->second;
}

}  // namespace imp
