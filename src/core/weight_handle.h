#pragma once

// WeightHandle (per-weight storage/tier descriptor) lives in core/ because
// compute/weight_dispatch.h takes it by reference; keeping it in exec/
// would make compute/ include a higher layer. WeightRegistry stays in exec/.

#include "core/logging.h"
#include "core/qtype.h"
#include "core/tensor_kind.h"
#include "core/storage_tier.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace imp {

struct WeightHandle {
    TensorID id = kInvalidTensorID;
    TensorKind kind = TensorKind::UNKNOWN;
    StorageTier primary_tier = StorageTier::Undefined;
    StorageTier prefill_tier = StorageTier::Undefined;  // M>1 GEMM dispatch
    StorageTier decode_tier = StorageTier::Undefined;   // M=1 GEMV dispatch
    int64_t shape[2] = {0, 0};
    // VRAM bytes owned by this handle. Zero = storage is BORROWED (e.g. points
    // at a wcache_ entry). Non-zero = this handle's PlanExecutor allocated it
    // and frees it in the registry destructor. Never mix borrowed and owned on
    // the same handle: the freer would double-free or leak.
    int64_t owned_bytes = 0;

    // Pointer to the ORIGINAL quantized weight bytes (Model::gpu_allocations_)
    // plus its source qtype. Always borrowed, never freed here. Used by
    // weight_dispatch for the M=1 dp4a/mmvq fallback when the small-M path on
    // the original is faster than cuBLAS on the cached overlay.
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

    // Can the original GGUF source bytes (source_data) be safely freed once
    // this handle is populated? Safe only when primary_tier has BOTH a
    // decode-fast kernel AND a prefill kernel that consume the overlay payload:
    // NVFP4, CUTLASS_NVFP4, FP8, MXFP4. NOT safe: FP16 (decode prefers dp4a on
    // the original), FP32 (LM-head policy keeps original), Undefined.
    // Conservative: true only when source_data is present AND the tier covers
    // both M=1 and M>1 paths; actual freeing needs a dispatch-site audit too.
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

}  // namespace imp
