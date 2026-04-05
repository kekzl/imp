#pragma once

#include "graph/weight_cache_manager.h"
#include "graph/quant_scratch.h"
#include <cuda_runtime.h>

namespace imp {

// ---------------------------------------------------------------------------
// GemmContext: bundles all state needed by gemm_dispatch into a single struct.
//
// Replaces 21 loose parameters with one context object. Created once per
// forward pass step (or per layer), passed to all GEMM dispatch calls.
// All members are non-owning references — lifetime managed by GraphExecutor.
// ---------------------------------------------------------------------------
struct GemmContext {
    cudaStream_t stream = nullptr;

    // Output mode: 0.0 = overwrite (C = A@B), 1.0 = residual add (C += A@B)
    float beta = 0.0f;

    // Weight caches (non-owning)
    const WeightCacheManager* wcache = nullptr;
    bool force_fp16 = false;

    // Quantization scratch buffers (non-owning)
    const QuantScratch* qscratch = nullptr;

    // Helper: create from executor state
    static GemmContext make(cudaStream_t s, const WeightCacheManager& wc,
                            const QuantScratch& qs, bool force_fp16 = false) {
        GemmContext ctx;
        ctx.stream = s;
        ctx.wcache = &wc;
        ctx.qscratch = &qs;
        ctx.force_fp16 = force_fp16;
        return ctx;
    }

    // Convenience: set beta for residual-add pattern
    GemmContext with_beta(float b) const {
        GemmContext c = *this;
        c.beta = b;
        return c;
    }
};

} // namespace imp
