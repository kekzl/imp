#pragma once

#include <cstddef>
#include <cstdint>

namespace imp {

// One KV block a recurrent snapshot was computed against: the prefix-cache hash, the
// block bound to it, and the binding's serial (new on every bind, #2174).
struct KvChainLink {
    size_t hash = 0;
    int block_id = -1;
    uint64_t serial = 0;
};

}  // namespace imp
