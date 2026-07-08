#pragma once

#include "core/logging.h"
#include "core/qtype.h"
#include "core/tensor_kind.h"
#include "core/storage_tier.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>

namespace imp {

struct WeightHandle {
    TensorID id = kInvalidTensorID;
    TensorKind kind = TensorKind::UNKNOWN;
    StorageTier primary_tier = StorageTier::Undefined;
    StorageTier prefill_tier = StorageTier::Undefined;  // M>1 GEMM dispatch
    StorageTier decode_tier = StorageTier::Undefined;   // M=1 GEMV dispatch
    int64_t shape[2] = {0, 0};
    // Size in bytes of VRAM owned by this handle. Zero means storage is
    // BORROWED (e.g. via the Phase-2 shim that points handles at wcache_
    // entries). A non-zero value means this handle's PlanExecutor (Phase 4+)
    // allocated the storage and is responsible for freeing it in the
    // registry destructor. Never mix borrowed and owned storage on the same
    // handle — the freer would double-free or leak.
    int64_t owned_bytes = 0;

    // Pointer to the ORIGINAL quantized weight bytes (in Model::gpu_allocations_),
    // and its source qtype. Always borrowed (never freed by the handle).
    // Used by weight_dispatch for the M=1 dp4a/mmvq fallback when primary_tier
    // is FP16/FP8/NVFP4 but the source is a GGUF block-quant format and the
    // small-M path on the original is faster than cuBLAS on the cached overlay.
    // Phase 5 PR #1 Commit 5.1.3.a — used by the upcoming weight_dispatch shim.
    const void* source_data = nullptr;
    QType source_qtype = QType::NONE;
    void* source_scales = nullptr;
    float source_tensor_scale = 1.0f;

    union {
        struct {
            float* data;
        } fp32;
        struct {
            half* data;
        } fp16;
        struct {
            __nv_fp8_e4m3* data;
            float* d_scale;
        } fp8;
        struct {
            uint8_t* data;
            uint8_t* block_scales;
            float* tensor_scale;
            float* tensor_scale_2;
        } nvfp4;
        struct {
            void* weight;
            void* sf;
            float* global_scale;
        } cutlass_nvfp4;
        struct {
            void* weight;
            void* scales;
            void* linear_scales;
            int hadamard_bs;
        } mxfp4;
    } payload{};

    bool is_populated() const { return primary_tier != StorageTier::Undefined; }

    // Phase 5 PR #1 Commit 5.1.4.a: can the original GGUF source bytes
    // (source_data, owned by Model::gpu_allocations_) be safely freed once
    // this handle is populated?
    //
    // Safe to drop when primary_tier provides BOTH a decode-fast kernel AND
    // a prefill kernel that consume the overlay payload (not the original).
    // Tiers that qualify: NVFP4 (gemv_nvfp4_kpar decode + dequant→cuBLAS prefill),
    // CUTLASS_NVFP4 (NVFP4 GEMV decode + CUTLASS NVFP4 GEMM prefill), FP8
    // (gemv_fp8 decode + cuBLAS FP8 GEMM prefill), MXFP4 (gemv_mxfp4 +
    // CUTLASS MXFP4 GEMM).
    //
    // Tiers that do NOT qualify: FP16 (decode prefers dp4a on the original
    // quant — see 5.1.3.c), FP32 (LM-head policy keeps original), Undefined.
    //
    // Predicate is conservative: returns true only when both source_data is
    // present AND the tier covers both M=1 and M>1 paths. Actual freeing
    // additionally requires dispatch-site audits — done in 5.1.4.b.
    bool can_drop_source() const {
        if (source_data == nullptr)
            return false;
        switch (primary_tier) {
            case StorageTier::NVFP4:
            case StorageTier::CUTLASS_NVFP4:
            case StorageTier::FP8:
            case StorageTier::MXFP4:
                return true;
            case StorageTier::FP16:
            case StorageTier::FP32:
            case StorageTier::Undefined:
                return false;
        }
        return false;
    }
};

class VRAMAllocator;

class WeightRegistry {
public:
    TensorID reserve(TensorKind kind, int64_t rows, int64_t cols);

    // C++23 deducing this: one overload serves const and non-const callers.
    template <typename Self>
    auto&& handle(this Self&& self, TensorID id) {
        if (id < 0 || id >= static_cast<TensorID>(self.handles_.size())) {
            IMP_LOG_FATAL("WeightRegistry::handle: id %d out of range [0, %zu)", id, self.handles_.size());
        }
        return self.handles_[id];
    }
    size_t size() const { return handles_.size(); }

    void clear();

    // Lookup by the source GGUF data pointer that Phase 4 stamped into
    // `source_data`. Returns nullptr if no handle was registered for this
    // pointer. Used by the migrating dispatch helpers to find the
    // pre-registered handle when callers still pass raw Tensor (Phase 5
    // Commit 5.1.3.b/c — see weight_dispatch.cu).
    const WeightHandle* find_by_source_data(const void* p) const;

    // Free VRAM for any handle whose `owned_bytes > 0`. Borrowed handles
    // (owned_bytes == 0) are left alone — their storage is managed by the
    // original allocator (e.g. wcache_ maps or Model::gpu_allocations_).
    // Idempotent: each freed handle's `owned_bytes` is reset to 0 and the
    // payload pointer to nullptr, so a second call is a no-op.
    size_t free_owned_storage(VRAMAllocator* alloc);

private:
    std::vector<WeightHandle> handles_;
};

}  // namespace imp
