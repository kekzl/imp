#pragma once

// Speculation defaults that layer offload (--gpu-layers) changes. A verify there costs one streamed
// pass like a decode step (Qwen3-14B Q6_K --gpu-layers 0: 189.71 ms at width 64 vs 188.6 ms at 1).

#include "core/logging.h"
#include "runtime/config.h"
#include "runtime/graph_eligibility.h"

#include <algorithm>

namespace imp {

// 64-row budget: width 8 prose 7.22, code 9.80 tok/s (width 4: 6.12, 8.90).
inline constexpr int kOffloadRecycleWidth = 8;

// Multi-candidate verify rows, width * (1 + depth): resident 17 (mid capture bucket, per-row
// context walks at linear-chunk cost); offload 64 (one streamed pass is flat to 64 rows).
inline constexpr int kVerifyRowsResident = 17;
inline constexpr int kVerifyRowsOffload = 64;

struct RecycleMcShape {
    int width = 1;
    int depth = 1;
};

// Width capped at the 8 recycle slots, depth at the row budget and k.
inline RecycleMcShape recycle_mc_shape(int recycle_width, int recycle_depth, int k, bool offload) {
    const int rows = offload ? kVerifyRowsOffload : kVerifyRowsResident;
    RecycleMcShape s;
    s.width = std::min(8, std::max(1, recycle_width));
    const int depth_cap = std::max(1, (rows / std::max(2, s.width)) - 1);
    s.depth = std::min({std::max(1, recycle_depth), depth_cap, k});
    return s;
}

// Token recycling on (width 8) under offload unless the operator set the key.
inline void apply_offload_recycling_default(RuntimeConfig& rc, int gpu_layers) {
    if (!layer_offload_blocks_graphs(gpu_layers) || rc.was_set("speculative.token_recycling"))
        return;
    rc.speculative.token_recycling = true;
    if (!rc.was_set("speculative.recycle_width"))
        rc.speculative.recycle_width = kOffloadRecycleWidth;
    IMP_LOG_INFO("speculative.token_recycling: auto -> on, width %d (layer offload)",
                 rc.speculative.recycle_width);
}

}  // namespace imp
