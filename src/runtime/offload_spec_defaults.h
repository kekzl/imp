#pragma once

// Speculation defaults that layer offload (--gpu-layers) changes. A verify there costs one streamed
// pass like a decode step (Qwen3-14B Q6_K --gpu-layers 0: 189.71 ms at width 64 vs 188.6 ms at 1).

#include "core/logging.h"
#include "runtime/config.h"
#include "runtime/graph_eligibility.h"

namespace imp {

inline constexpr int kOffloadRecycleWidth = 4;  // width 4: prose 5.30 -> 6.13, code 6.85 -> 8.87 tok/s

// Token recycling on (width 4) under offload unless the operator set the key.
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
