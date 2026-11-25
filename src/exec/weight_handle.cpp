#include "exec/weight_handle.h"
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

void WeightRegistry::adopt_fp16_storage(TensorID id, DeviceOwned<void>&& storage) {
    auto& h = handle(id);
    h.primary_tier = StorageTier::FP16;
    h.payload.fp16.data = static_cast<half*>(storage.get());
    h.owned_bytes = storage ? static_cast<int64_t>(storage.bytes()) : 0;
    if (storage)
        owned_storage_.push_back(std::move(storage));
}

size_t WeightRegistry::free_owned_storage() {
    size_t freed_bytes = 0;
    // Only adopt_fp16_storage sets owned_bytes: an owning handle is an FP16 overlay.
    for (auto& h : handles_) {
        if (h.owned_bytes == 0)
            continue;
        if (h.primary_tier == StorageTier::FP16)
            h.payload.fp16.data = nullptr;
        freed_bytes += static_cast<size_t>(h.owned_bytes);
        h.owned_bytes = 0;
    }
    owned_storage_.clear();  // each buffer once, through the family that allocated it
    return freed_bytes;
}

}  // namespace imp
