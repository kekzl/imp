#pragma once

// GemmKernel registry, the (Strategy -> handler) table behind the 3 dispatch sites:
//   {FP16, NONE, false}          generic dequant catch-all (M>1, uncached weight)
//   {FP16, <gguf qtype>, true}   GGUF small-M (mmvq/dp4a/fused gemv), 8 qtypes
//   {CUTLASS_NVFP4, F16, false}  CUTLASS NVFP4 prefill GEMM
// Every key has one producer and vice versa; RegistryHoldsExactlyTheProducedKeys pins the
// count at 10. A new tier registers here only together with its dispatch site.

#include "core/storage_tier.h"
#include "core/tensor.h"
#include "model/model_config.h"  // QType

#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

// Forward-declared payload structs (defined in their owning TUs).
struct CutlassNvFP4Weight;
struct CutlassMxFP4Weight;

// Strategy key: fully describes which kernel handles a dispatch request. qtype is the
// WEIGHT qtype (input is always FP16 on the hot path today). m_is_one separates GEMV
// (M==1, decode) from GEMM (M>1, prefill): the two reach different kernels even within
// the same tier (e.g. NVFP4 GEMV vs NVFP4 dequant->GEMM).
struct GemmStrategy {
    StorageTier tier = StorageTier::Undefined;
    QType weight_qtype = QType::F16;
    bool m_is_one = false;

    bool operator==(const GemmStrategy& o) const noexcept {
        return tier == o.tier && weight_qtype == o.weight_qtype && m_is_one == o.m_is_one;
    }
};

// Args struct replacing the 21 trailing parameters of the legacy gemm_dispatch_impl.
// Per-tier handlers cast weight_payload to the tier-appropriate pointer type (Tensor* for
// FP16, FP8CacheEntry* for FP8, NvFP4QuantResult* for NVFP4, etc.).
struct GemmKernelArgs {
    // Required across every tier
    const Tensor* input = nullptr;        // [M, K] FP16
    Tensor* output = nullptr;             // [M, N] FP16 (or FP32 for some LM-head paths)
    cudaStream_t stream = nullptr;
    float beta = 0.0f;                    // residual-fused GEMM on FP16 paths only

    // Tier-specific weight handle. Casts:
    //   FP16            -> const Tensor* (F16 weight or GGUF source tensor)
    //   CUTLASS_NVFP4   -> const CutlassNvFP4Weight*
    const void* weight_payload = nullptr;

    // Workspace pointers — only some tiers consume them. nullptr means the
    // caller did not supply that workspace; the kernel must either degrade
    // (fall back to a non-workspace path) or fail loud via IMP_CHECK.
    void* dequant_scratch = nullptr;
    void* cutlass_act_data = nullptr;
    void* cutlass_act_sf = nullptr;
    void* cutlass_workspace = nullptr;
    size_t cutlass_workspace_size = 0;
    void* mxfp4_act_sf = nullptr;
    void* mxfp4_workspace = nullptr;
    size_t mxfp4_workspace_size = 0;
    // QW7 dual-cache CUTLASS MXFP4 hand-off: when CUTLASS_NVFP4 fires and weight.data is also
    // in cutlass_mxfp4 (--mxfp4-prefill on), the dispatch forwards the MXFP4 payload so the
    // handler tries MXFP4 CUTLASS before falling back to NVFP4 CUTLASS. nullptr = no
    // dual-cache hit. See cutlass_nvfp4_gemm_kernel (gemm_kernel_cutlass_nvfp4.cu).
    const void* mxfp4_payload = nullptr;

    // dp4a/Q8_1 activation quantization scratch (GGUF dp4a tier): per-call scratch pre-sized
    // by engine init (QuantScratch::q8_1_buf/d8_buf). nullptr means no scratch supplied; the
    // kernel must PreconditionFail so the dispatch site falls back to legacy.
    void* q8_1_buf = nullptr;
    float* d8_buf = nullptr;

    // Act-quant dedupe (CUTLASS_NVFP4 prefill): cutlass_act_data/_sf already hold the
    // quantized form of input (a prior dispatch on the same input quantized it); the handler
    // skips quantize_fp16_to_nvfp4_cutlass. Set when the GemmContext act-quant hint matches
    // (input.data, M, K).
    bool act_prequantized = false;

    // Per-model gemma4.force_mmvq override (Phase 5 Track A). Used by the
    // GGUF small-M kernels to decide between the mmvq and dp4a backends.
    // Sourced from ModelConfig::Overrides::Gemma4::force_mmvq via GemmContext.
    bool force_mmvq = false;

    // Phase 5 Track D follow-up: per-Engine RuntimeConfig::gemm flags
    // that gemm_kernel_gguf used to read directly via RuntimeConfig::current().
    // Threaded from GemmContext at dispatch time.
    bool no_mmvq = false;
    bool no_mmvq_q8_0 = false;
    bool no_dp4a_gemv = false;
};

// Result of a dispatch attempt.
enum class GemmDispatchResult {
    Ok = 0,             // kernel ran; caller is done
    NoMatch = 1,        // no registered kernel for this strategy; caller falls back
    PreconditionFail = 2, // kernel rejected the args (e.g. workspace nullptr); caller falls back
};

// Per-kernel function signature.
using GemmKernelFn = GemmDispatchResult (*)(const GemmKernelArgs& args);

// Registry singleton: register_kernel populates the table at static-init time (each
// tier's .cu calls it via a constructor attribute or static-init idiom). dispatch is the
// hot-path entry: linear scan over the small table to the matching kernel.
class GemmKernelRegistry {
public:
    static GemmKernelRegistry& instance();

    // Idempotent: re-registering the same strategy overrides the prior
    // entry and logs IMP_LOG_WARN. Slice 1 expects at most one registration
    // per strategy.
    void register_kernel(GemmStrategy strategy, GemmKernelFn fn);

    // Returns NoMatch if no registered handler covers `strategy`. Otherwise
    // invokes the kernel and propagates its result. Hot-path safe (no
    // allocation, no mutex acquisition after init).
    GemmDispatchResult dispatch(const GemmStrategy& strategy, const GemmKernelArgs& args) const;

    // Diagnostic: returns the number of registered strategies. Useful for
    // tests asserting that the static registration ran.
    std::size_t size() const noexcept;

private:
    GemmKernelRegistry() = default;
    GemmKernelRegistry(const GemmKernelRegistry&) = delete;
    GemmKernelRegistry& operator=(const GemmKernelRegistry&) = delete;

    // Small-N linear scan: the production table holds 10 entries (see the
    // file header); a linear scan beats a hash map at this size and avoids
    // the std::unordered_map alloc.
    struct Entry {
        GemmStrategy strategy;
        GemmKernelFn fn;
    };
    // 10 live entries + headroom for the GGUF qtypes not yet on the small-M
    // path (Q4_1, IQ4). register_kernel IMP_CHECKs the capacity.
    Entry entries_[16] = {};
    std::size_t count_ = 0;
};

}  // namespace imp
