#pragma once

#include <cstddef>

namespace imp {

// GDN layer workspaces of GraphExecutor (allocated in executor_workspace_buffers.cpp for has_gdn_ models) and
// the one-shot activity log flags of the GDN fast paths. Moved verbatim out of executor.h.
struct GdnWorkspace {
    // Chunk-parallel GDN prefill scan workspace (gdn.chunkpar_scan): five
    // per-(chunk,head) strip arrays + FP32 inter-strip state. Engine lifetime,
    // allocated when the model has GDN layers; nullptr degrades to the fused scan.
    void* chunkpar = nullptr;
    size_t chunkpar_bytes = 0;

    // GDN alpha/beta narrow GEMM workspace (gdn.alpha_beta_smallm): split-K
    // partials + per-tile tickets, ~100 KiB, zeroed once at allocation.
    // nullptr keeps the two cuBLAS calls.
    void* ab = nullptr;
    size_t ab_bytes = 0;
    void* ab_prefill = nullptr;  // split-K partials of the prefill launch (own tickets)
    size_t ab_prefill_bytes = 0;
    bool ab_narrow_logged = false;
    bool ab_prefill_logged = false;
    bool m1_input_logged = false;
    bool m1_out_logged = false;
};

}  // namespace imp
