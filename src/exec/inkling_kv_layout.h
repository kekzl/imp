#pragma once
// Inkling2-Loop global K/V in the engine's paged KV pool (B-111 S2b): the pool layer of each layer
// application, the rows a decode step writes past its position, the prefix a request may reuse and the
// pool's size. CUDA-free for the CPU lane (tests/test_inkling_kv_layout.cpp).

#include "model/inkling_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace imp {

// Positions a decode step writes past its own: the lookahead call's draft rows, in the target's layers and
// in the draft model's (InklingForward::kGraphRows - 1).
inline constexpr int kInklingKvAhead = 7;

struct InklingKvLayout {
    std::vector<int> app_layer;  // per layer application: its pool layer, -1 = sliding (a per-slot ring)
    int n_global = 0;            // full-attention applications: pool layers 0 .. n_global - 1
    int draft_layer = -1;        // the DSpark draft's full-context layer, -1 = none
    [[nodiscard]] int n_layers() const { return n_global + (draft_layer >= 0 ? 1 : 0); }
};

// One pool layer per full-attention application in application order, then the draft's when draft is set.
inline InklingKvLayout inkling_kv_apps(const InklingConfig& c, bool draft) {
    InklingKvLayout l;
    l.app_layer.assign(static_cast<size_t>(c.n_apps()), -1);
    for (int a = 0; a < c.n_apps(); ++a) {
        const size_t layer = static_cast<size_t>(c.app_layer(a));
        if (layer < c.global.size() && c.global[layer])
            l.app_layer[static_cast<size_t>(a)] = l.n_global++;
    }
    l.draft_layer = draft ? l.n_global : -1;
    return l;
}

// Blocks a request reuses from the prefix cache: no more than its slot can resume (slot_match: the ring and
// conv state are the slot's), the device holds (cached_blocks) and the prompt cap allows, floored to the
// checkpoint grid so InklingForward::rewind lands on exactly that position.
inline int inkling_prefix_cap_blocks(int slot_match, int cached_blocks, int cap_blocks, int bs, int grid) {
    if (bs <= 0 || grid <= 0 || grid % bs != 0)
        return 0;
    const int64_t tokens = std::min({static_cast<int64_t>(slot_match),
                                     static_cast<int64_t>(cached_blocks) * bs,
                                     static_cast<int64_t>(cap_blocks) * bs});
    return tokens <= 0 ? 0 : static_cast<int>(tokens / grid * grid / bs);
}

// Snapshot key of the first pos tokens (B-111 S4): the chain hash of their blocks seeded with salt, the hash
// the prefix cache files block pos / bs - 1 under. 0 = pos off the block grid. hash:
// KVCacheManager::compute_block_hash.
using InklingBlockHash = size_t (*)(std::span<const int32_t>, size_t);
[[nodiscard]] inline size_t inkling_snapshot_key(std::span<const int32_t> tokens, int pos, int bs,
                                                 size_t salt, InklingBlockHash hash) {
    if (bs <= 0 || pos <= 0 || pos % bs != 0 || static_cast<size_t>(pos) > tokens.size() || !hash)
        return 0;
    size_t key = salt;
    for (int b = 0; b < pos / bs; ++b)
        key = hash(tokens.subspan(static_cast<size_t>(b) * bs, static_cast<size_t>(bs)), key);
    return key;
}

// The decode steps' I8B32 dense projections fit when free_bytes keeps kInklingDense8Spare behind them. The
// pool is not charged: its ceiling commits as sequences grow (2 x 81920: 9137 MiB free, 655 MiB needed;
// charging its 5761 MiB dropped them, other texts and -12 % decode).
inline constexpr size_t kInklingDense8Spare = size_t{3} << 30;
[[nodiscard]] inline bool inkling_dense8_fits(size_t free_bytes, size_t need) {
    return free_bytes >= need + kInklingDense8Spare;
}

// Positions the pool commits per growth step: what the slots' own buffers committed at a time before S2b.
inline constexpr int kInklingKvGrowTokens = 2048;

// The pool's ceiling: every slot at max_seq_len plus the rows a step writes past it.
inline int inkling_pool_blocks(int slots, int max_seq_len, int bs) {
    return bs <= 0 ? 0 : slots * ((max_seq_len + kInklingKvAhead + bs - 1) / bs);
}

}  // namespace imp
