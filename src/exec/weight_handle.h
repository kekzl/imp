#pragma once

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
    } payload;

    bool is_populated() const { return primary_tier != StorageTier::Undefined; }
};

class VRAMAllocator;

class WeightRegistry {
public:
    TensorID reserve(TensorKind kind, int64_t rows, int64_t cols);
    WeightHandle& handle(TensorID id);
    const WeightHandle& handle(TensorID id) const;
    size_t size() const { return handles_.size(); }

    void clear();

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
