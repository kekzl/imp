#pragma once

#include "exec/quant_scratch.h"
#include "runtime/config.h"
#include <cuda_runtime.h>

namespace imp {

struct WeightCaches;  // defined in executor.h

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
    const WeightCaches* wcache = nullptr;
    bool force_fp16 = false;

    // Per-model override: when set, the GGUF small-M dispatch path prefers
    // the mmvq backend over dp4a for eligible qtypes. Sourced from
    // ModelConfig::Overrides::Gemma4::force_mmvq (Phase 5 Track A). Defaults
    // to false so non-Gemma-4 models behave identically.
    bool force_mmvq = false;

    // Phase 5 Track D follow-up: per-Engine knobs; were read from
    // RuntimeConfig::current() in gemm_dispatch / gemm_kernel_gguf hot paths.
    // Wired by the executor's GemmContext::make caller from runtime_config().
    bool q4k_imma_enabled = false;
    bool q4k_hmma_enabled = false;
    bool q8_imma_enabled = false;
    bool gemm_no_mmvq = false;
    bool gemm_no_mmvq_q8_0 = false;
    bool gemm_no_dp4a_gemv = false;

    // Quantization scratch buffers (non-owning)
    const QuantScratch* qscratch = nullptr;

    // Helper: create from executor state. `rcfg` is the per-Engine RuntimeConfig
    // — its gemm.* flags are mirrored into the context once at construction
    // (replaces the former RuntimeConfig::current() reads in gemm_dispatch
    // and gemm_kernel_gguf).
    static GemmContext make(cudaStream_t s, const WeightCaches& wc, const QuantScratch& qs,
                            const RuntimeConfig& rcfg, bool force_fp16 = false,
                            bool force_mmvq = false) {
        GemmContext ctx;
        ctx.stream = s;
        ctx.wcache = &wc;
        ctx.qscratch = &qs;
        ctx.force_fp16 = force_fp16;
        ctx.force_mmvq = force_mmvq;
        ctx.q4k_imma_enabled = rcfg.gemm.q4k_imma_enabled;
        ctx.q4k_hmma_enabled = rcfg.gemm.q4k_hmma_enabled;
        ctx.q8_imma_enabled = rcfg.gemm.q8_imma_enabled;
        ctx.gemm_no_mmvq = rcfg.gemm.no_mmvq;
        ctx.gemm_no_mmvq_q8_0 = rcfg.gemm.no_mmvq_q8_0;
        ctx.gemm_no_dp4a_gemv = rcfg.gemm.no_dp4a_gemv;
        return ctx;
    }

    // Convenience: set beta for residual-add pattern
    GemmContext with_beta(float b) const {
        GemmContext c = *this;
        c.beta = b;
        return c;
    }
};

}  // namespace imp
