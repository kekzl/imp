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

struct Fp16CacheEntry;          // exec/weight_cache_entries.h
struct FP8CacheEntry;           // exec/weight_caches.h
struct NvFP4CacheEntry;         // exec/weight_cache_entries.h
struct NvFP4SfAtomCacheEntry;   // exec/weight_cache_entries.h
struct MxFP8SfAtomCacheEntry;   // exec/weight_cache_entries.h
struct MxFP4SfAtomCacheEntry;   // exec/weight_cache_entries.h

// The WeightCaches entries keyed by a handle's source_data (audit D-3), bound by
// bind_cache_entries (exec/weight_tier_entries.h); null = that map holds none.
struct WeightCacheRefs {
    const Fp16CacheEntry* fp16 = nullptr;
    const FP8CacheEntry* fp8 = nullptr;
    const NvFP4CacheEntry* nvfp4 = nullptr;
    const NvFP4SfAtomCacheEntry* nvfp4_sfatom = nullptr;
    const NvFP4SfAtomCacheEntry* nvfp4_prefill = nullptr;  // nvfp4_sfatom_prefill
    const MxFP8SfAtomCacheEntry* mxfp8_prefill = nullptr;  // mxfp8_sfatom_prefill
    const MxFP4SfAtomCacheEntry* mxfp4_sfatom = nullptr;
    const FP8CacheEntry* released_fp8 = nullptr;  // released_fp8_view (Phase 4b)
    bool q8_planes_only = false;                  // Phase 4c freed the Q8_0 source
};

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
    // NVFP4 activation tensor scale (dequant form) the W4A4 GEMMs quantize this weight's input with;
    // alpha = weight tensor scale x act_scale. 1.0 unless the checkpoint carries a scale.
    float act_scale = 1.0f;
    // Phase 4b freed the source bytes (source_data stays as the cache key): paths not covered
    // by a copy rebuild the F16 weight from wcache (released_source_gemm_), never read source_data.
    bool source_released = false;
    // Phase 4 binds them, rebinding source_data rebinds them; dispatch reads them, not the maps.
    WeightCacheRefs entries;

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
            const float* d_act_scale;  // static input scale, null = dynamic (FP8CacheEntry)
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
        } nvfp4_sfatom;
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
    // NVFP4_LINEAR, NVFP4_SFATOM, FP8, MXFP4. NOT safe: FP16 (decode prefers dp4a on
    // the original), FP32 (LM-head policy keeps original), Undefined.
    // Conservative: true only when source_data is present AND the tier covers
    // both M=1 and M>1 paths; actual freeing needs a dispatch-site audit too.
    // W4A4 GEMM alpha of an NVFP4_SFATOM payload: weight tensor scale (1.0 when absent) x `act_scale`.
    [[nodiscard]] float nvfp4_sfatom_alpha(float act) const {
        return (payload.nvfp4_sfatom.global_scale ? *payload.nvfp4_sfatom.global_scale : 1.0f) * act;
    }

    bool can_drop_source() const {
        if (source_data == nullptr)
            return false;
        switch (primary_tier) {
            case StorageTier::NVFP4_LINEAR:
            case StorageTier::NVFP4_SFATOM:
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
