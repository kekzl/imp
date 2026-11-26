#pragma once

// The LM-head route: one pure decision for serving, perplexity, prompt logprobs and the draft
// argmax head (audit D-5, ROADMAP B-112). The callers keep their kernels; they switch on this.

#include "core/storage_tier.h"

#include <cstdint>

namespace imp {

// What the loaded head offers. Built once per call by GraphExecutor::lm_head_facts_().
struct LmHeadFacts {
    StorageTier tier = StorageTier::Undefined;  // registry primary tier of out_proj
    bool fp8_cache = false;                     // wcache_.fp8 holds out_proj
    bool nvfp4_cache = false;                   // wcache_.nvfp4 holds out_proj (Phase-3 secondary)
    bool mxfp4_linear = false;                  // MXFP4 payload with linear scales
    bool dp4a_ready = false;                    // Q8_1 scratch, FP16 compute, dp4a source qtype
};

// ServeRow: prefill last row or decode n == 1. ServeBatch: decode n > 1.
// Eval: perplexity, spec verify and prompt logprobs (for_each_lm_head_batch_).
enum class LmHeadUse : uint8_t { ServeRow, ServeBatch, Eval };

enum class LmHeadRoute : uint8_t {
    Mxfp4Gemv,  // gemv_mxfp4_kpar_fp32
    Fp8Cache,   // cuBLASLt on the wcache_.fp8 copy
    Nvfp4,      // NVFP4 GEMV (batched or kpar), small-M and tile GEMM first where offered
    Dp4a,       // RMSNorm -> Q8_1 + dp4a GEMV on the raw GGUF head
    Handle,     // RMSNorm + gemm_via_handle_ (ServeBatch tries the FP8 tier first)
};

// Not FP8-cached and NVFP4 by tier or by the secondary cache.
constexpr bool lm_head_is_nvfp4(const LmHeadFacts& f) {
    return !f.fp8_cache && (f.tier == StorageTier::NVFP4 || f.nvfp4_cache);
}

constexpr LmHeadRoute lm_head_route(const LmHeadFacts& f, LmHeadUse use) {
    const bool nvfp4 = lm_head_is_nvfp4(f);
    switch (use) {
        case LmHeadUse::ServeRow:
            if (f.tier == StorageTier::MXFP4 && f.mxfp4_linear)
                return LmHeadRoute::Mxfp4Gemv;
            if (f.fp8_cache)
                return LmHeadRoute::Fp8Cache;
            if (nvfp4)
                return LmHeadRoute::Nvfp4;
            return f.dp4a_ready ? LmHeadRoute::Dp4a : LmHeadRoute::Handle;
        case LmHeadUse::ServeBatch:
            if (nvfp4)
                return LmHeadRoute::Nvfp4;
            // An FP16-tier copy of a dp4a head takes the cuBLAS GEMM over per-row GEMVs.
            return (f.dp4a_ready && f.tier != StorageTier::FP16) ? LmHeadRoute::Dp4a : LmHeadRoute::Handle;
        case LmHeadUse::Eval:
            if (nvfp4)
                return LmHeadRoute::Nvfp4;
            // gemm_via_handle_ at M=1 on a GGUF head with FP16 input faulted (gemma-3-12b PPL=1.0000).
            return (f.dp4a_ready && f.tier != StorageTier::MXFP4 && !f.fp8_cache) ? LmHeadRoute::Dp4a
                                                                                  : LmHeadRoute::Handle;
    }
    return LmHeadRoute::Handle;
}

// The draft argmax head (lm_head_rows_argmax, MTP draws): a W4A16 draft copy first, the FP8
// head only while the tile GEMM head is not built, else the tile GEMM; None = no route.
enum class DraftHeadRoute : uint8_t { W4a16Draft, Fp8Head, TileGemm, None };

constexpr DraftHeadRoute draft_head_route(bool draft_copy, bool fp8_head, bool tile_ready,
                                          bool tile_scratch) {
    if (draft_copy)
        return DraftHeadRoute::W4a16Draft;
    if (!tile_ready && fp8_head)
        return DraftHeadRoute::Fp8Head;
    return (tile_ready && tile_scratch) ? DraftHeadRoute::TileGemm : DraftHeadRoute::None;
}

}  // namespace imp
