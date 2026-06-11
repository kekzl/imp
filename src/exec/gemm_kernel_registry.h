#pragma once

// GemmKernel registry — R5 first-domino cross-axis refactor (slice 1).
//
// Replaces the 21-parameter `gemm_dispatch_impl` god-dispatcher (review
// phase3_maint.md §2 #1) with a single small args struct and a (Strategy
// -> function pointer) registry. Adding a new qtype/quantization tier
// becomes a single-file change: register an entry; nothing else moves.
//
// Slice 1 scope: define the interface, register only the FP16 tier as
// proof-of-concept. Live dispatch tries the registry first when the
// runtime config flag `gemm.use_kernel_registry` is true and falls back
// to the legacy `gemm_dispatch_impl` for tiers not yet registered.
// Default flag is OFF so production behavior is unchanged.
//
// Future slices migrate FP8, NVFP4, CUTLASS_NVFP4, MXFP4 tiers one at a
// time; once every dispatch site is covered the legacy path can be
// deleted (the cross-axis maintainability + extensibility win from
// review phase5_synthesis.md §5).

#include "core/storage_tier.h"
#include "core/tensor.h"
#include "model/model_config.h"  // QType

#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

// Forward-declared payload structs (defined in their owning TUs).
struct FP8CacheEntry;
struct NvFP4QuantResult;
struct CutlassNvFP4Weight;
struct CutlassMxFP4Weight;

// ---------------------------------------------------------------------------
// Strategy key — fully describes which kernel handles a dispatch request.
// `qtype` is the WEIGHT qtype (input qtype is always FP16 on the dispatch
// hot path today). `m_is_one` separates GEMV (M==1, decode) from GEMM
// (M>1, prefill) because the two reach different kernels even within the
// same tier (e.g. NVFP4 GEMV vs NVFP4 dequant->GEMM).
// ---------------------------------------------------------------------------
struct GemmStrategy {
    StorageTier tier = StorageTier::Undefined;
    QType weight_qtype = QType::F16;
    bool m_is_one = false;

    bool operator==(const GemmStrategy& o) const noexcept {
        return tier == o.tier && weight_qtype == o.weight_qtype && m_is_one == o.m_is_one;
    }
};

// ---------------------------------------------------------------------------
// Args struct — replaces the 21 trailing parameters of the legacy
// gemm_dispatch_impl. Per-tier handlers cast `weight_payload` to the
// tier-appropriate pointer type (Tensor* for FP16, FP8CacheEntry* for
// FP8, NvFP4QuantResult* for NVFP4, etc.).
// ---------------------------------------------------------------------------
struct GemmKernelArgs {
    // Required across every tier
    const Tensor* input = nullptr;        // [M, K] FP16
    Tensor* output = nullptr;             // [M, N] FP16 (or FP32 for some LM-head paths)
    cudaStream_t stream = nullptr;
    float beta = 0.0f;                    // residual-fused GEMM on FP16 paths only

    // Tier-specific weight handle. Casts:
    //   FP16            -> const Tensor*
    //   FP8             -> const FP8CacheEntry*
    //   NVFP4           -> const NvFP4QuantResult*
    //   CUTLASS_NVFP4   -> const CutlassNvFP4Weight*
    //   MXFP4           -> const CutlassMxFP4Weight*
    const void* weight_payload = nullptr;

    // Workspace pointers — only some tiers consume them. nullptr means the
    // caller did not supply that workspace; the kernel must either degrade
    // (fall back to a non-workspace path) or fail loud via IMP_CHECK.
    void* dequant_scratch = nullptr;
    size_t dequant_scratch_size = 0;
    void* cutlass_act_data = nullptr;
    void* cutlass_act_sf = nullptr;
    void* cutlass_workspace = nullptr;
    size_t cutlass_workspace_size = 0;
    void* mxfp4_act_sf = nullptr;
    void* mxfp4_workspace = nullptr;
    size_t mxfp4_workspace_size = 0;
    // R5 Slice 8.6 — QW7 dual-cache CUTLASS MXFP4 hand-off. When the
    // CUTLASS_NVFP4 strategy fires AND the same `weight.data` is also present
    // in the `cutlass_mxfp4` cache (only happens when `--mxfp4-prefill` is on,
    // because executor_pre_dequant.cu builds the mxfp4 cache by iterating
    // every NVFP4 entry), the dispatch site forwards the MXFP4 payload here
    // so the handler can try the MXFP4 CUTLASS GEMM before falling back to
    // the NVFP4 CUTLASS GEMM. nullptr = no dual-cache hit, take the NVFP4
    // path directly. See cutlass_nvfp4_gemm_kernel in
    // gemm_kernel_cutlass_nvfp4.cu for the branching logic.
    const void* mxfp4_payload = nullptr;
    void* fp8_act_buf = nullptr;
    float* d_act_scale = nullptr;
    float* d_fp8_block_maxes = nullptr;
    float* d_fp8_absmax = nullptr;
    int fp8_max_grid = 0;

    // dp4a / Q8_1 activation quantization scratch (slice 7 — GGUF dp4a tier).
    // Same role as fp8_act_buf + d_act_scale above: a per-call activation
    // scratch pre-sized by engine init (QuantScratch::q8_1_buf / d8_buf).
    // The dp4a kernels quantize the FP16 activation into `q8_1_buf` then run
    // dispatch_dp4a_gemv with the block scales in `d8_buf`. nullptr means the
    // caller did not supply scratch and the kernel must PreconditionFail so
    // the dispatch site can fall back to legacy. Typed as void* / float* to
    // mirror the legacy gemm_dispatch_impl signature; the kernel reinterpret-
    // casts to block_q8_1*.
    void* q8_1_buf = nullptr;
    float* d8_buf = nullptr;

    // Act-quant dedupe (CUTLASS_NVFP4 prefill): cutlass_act_data/_sf already
    // hold the quantized form of `input` (a prior dispatch on the same input
    // quantized it). The handler skips quantize_fp16_to_nvfp4_cutlass. Set
    // by gemm_via_handle_ when the GemmContext act-quant hint matches the
    // (input.data, M, K) of this call.
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

// ---------------------------------------------------------------------------
// Registry singleton.
//
// `register_kernel` populates the table at static-init time (each tier's
// .cu file calls it from a `__attribute__((constructor))` or
// `static const int _reg = (register_kernel(...), 0);` idiom). `dispatch`
// is the hot-path entry — does a linear scan over the (small) table and
// invokes the matching kernel.
// ---------------------------------------------------------------------------
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

    // Small-N linear scan — production table holds at most ~8 entries
    // even after the full migration; std::vector + linear scan beats a
    // hash map for this size and avoids the std::unordered_map alloc.
    struct Entry {
        GemmStrategy strategy;
        GemmKernelFn fn;
    };
    // Capacity sized for Slice 7: 8 entries through Slice 6 + 11 new GGUF
    // strategies (4 mmvq qtypes + 7 dp4a qtypes) = 19; bumped to 32 for
    // headroom (next slices may register additional GGUF qtype/backend
    // combinations or the remaining Q4_1 / IQ4 paths).
    Entry entries_[32] = {};
    std::size_t count_ = 0;
};

}  // namespace imp
