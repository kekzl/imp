#include "exec/weight_handle.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"

#include <cstring>
#include <utility>

namespace imp {

TensorID WeightRegistry::reserve(TensorKind kind, int64_t rows, int64_t cols) {
    TensorID id = static_cast<TensorID>(handles_.size());
    WeightHandle h;
    h.id = id;
    h.kind = kind;
    h.primary_tier = StorageTier::Undefined;
    h.shape[0] = rows;
    h.shape[1] = cols;
    h.owned_bytes = 0;
    std::memset(&h.payload, 0, sizeof(h.payload));
    handles_.push_back(h);
    return id;
}

void WeightRegistry::clear() { handles_.clear(); }

size_t WeightRegistry::free_owned_storage(VRAMAllocator* alloc) {
    if (!alloc)
        return 0;
    size_t freed_bytes = 0;
    for (auto& h : handles_) {
        if (h.owned_bytes == 0)
            continue;
        // Free the primary-tier payload pointer. Tiers with auxiliary buffers
        // (NVFP4 scales, CUTLASS sf, MXFP4 scales) are freed by their native
        // free_nvfp4_result / free_cutlass_nvfp4_weight / etc. helpers in
        // executor_workspace_buffers.cu — this registry helper is currently
        // only used by FP16 overlay tensors (fused_kv, fused_gate_up) which
        // have a single contiguous storage pointer.
        switch (h.primary_tier) {
            case StorageTier::FP16:
                if (h.payload.fp16.data) {
                    alloc->free(h.payload.fp16.data);
                    h.payload.fp16.data = nullptr;
                    freed_bytes += static_cast<size_t>(h.owned_bytes);
                    h.owned_bytes = 0;
                }
                break;
            case StorageTier::FP32:
            case StorageTier::FP8:
            case StorageTier::NVFP4:
            case StorageTier::CUTLASS_NVFP4:
            case StorageTier::MXFP4:
            case StorageTier::Undefined:
                // Not currently owned by the registry — fall through to
                // legacy free paths.
                IMP_LOG_WARN(
                    "WeightRegistry::free_owned_storage: handle id=%d "
                    "kind=%d tier=%d has owned_bytes=%lld but no "
                    "type-specific free is wired; skipping.",
                    h.id, std::to_underlying(h.kind), std::to_underlying(h.primary_tier),
                    static_cast<long long>(h.owned_bytes));
                break;
        }
    }
    return freed_bytes;
}

}  // namespace imp
