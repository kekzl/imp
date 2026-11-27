#pragma once

#include "exec/gemm_mode.h"
#include "exec/quant_scratch.h"
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

    // This forward's mode (audit D-7): GEMM dispatch reads it here, not from executor members.
    GemmMode mode;

    // Per-model override: when set, the GGUF small-M dispatch path prefers
    // the mmvq backend over dp4a for eligible qtypes. Sourced from
    // ModelConfig::Overrides::Gemma4::force_mmvq (Phase 5 Track A). Defaults
    // to false so non-Gemma-4 models behave identically.
    bool force_mmvq = false;

    // Spec-verify chunk forward (#998): small-M GEMMs (M <= largest capture
    // bucket) prefer the NVFP4 decode overlay over the M>1 prefill dequant
    // path. mode.spec_verify of a site that opts in; the LM head and the MoE
    // shared expert do not. Never set for real prefills.
    bool spec_verify_small_m = false;

    // Quantization scratch buffers (non-owning)
    const QuantScratch* qscratch = nullptr;

    // Helper: create from executor state. verify_small_m_routes=false keeps a site off the
    // spec-verify small-M routes while it still carries the forward's mode.
    static GemmContext make(cudaStream_t s, const WeightCaches& wc, const QuantScratch& qs,
                            const GemmMode& mode = {}, bool force_mmvq = false,
                            bool verify_small_m_routes = true) {
        GemmContext ctx;
        ctx.stream = s;
        ctx.wcache = &wc;
        ctx.qscratch = &qs;
        ctx.mode = mode;
        ctx.force_mmvq = force_mmvq;
        ctx.spec_verify_small_m = mode.spec_verify && verify_small_m_routes;
        return ctx;
    }

    // gemm_nvfp4's M>1 fallback workspace of this call (audit D-9); null without a qscratch.
    Nvfp4DequantWorkspace* nvfp4_dequant_ws() const { return qscratch ? qscratch->nvfp4_dequant : nullptr; }

    // Convenience: set beta for residual-add pattern
    GemmContext with_beta(float b) const {
        GemmContext c = *this;
        c.beta = b;
        return c;
    }

    // Act-quant hint (NVFP4 prefill QKV / gate-up dedupe): the SfAtom NVFP4
    // activation scratch already holds quantize(input) for exactly this
    // (data, M, K) triple — a prior dispatch on the SAME input quantized it.
    // gemm_via_handle_ forwards the match as args.act_prequantized so the
    // handler skips its quantize step. Scoped per call site (QKV, gate/up);
    // never cached across ops — the scratch is clobbered by the next
    // dispatch on a different input.
    const void* act_quant_hint_data = nullptr;
    int act_quant_hint_m = 0;
    int act_quant_hint_k = 0;

    GemmContext with_act_quant_hint(const void* data, int m, int k) const {
        GemmContext c = *this;
        c.act_quant_hint_data = data;
        c.act_quant_hint_m = m;
        c.act_quant_hint_k = k;
        return c;
    }

    // True when the call site marked exactly (data, m, k) as already quantized.
    [[nodiscard]] bool act_quant_hint_matches(const void* data, int m, int k) const {
        return act_quant_hint_data != nullptr && act_quant_hint_data == data && act_quant_hint_m == m &&
               act_quant_hint_k == k;
    }
};

}  // namespace imp
