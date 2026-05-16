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
    void* fp8_act_buf = nullptr;
    float* d_act_scale = nullptr;
    float* d_fp8_block_maxes = nullptr;
    float* d_fp8_absmax = nullptr;
    int fp8_max_grid = 0;
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
    Entry entries_[16] = {};
    std::size_t count_ = 0;
};

}  // namespace imp
