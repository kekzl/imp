#include "graph/weight_handle.h"
#include "core/logging.h"

#include <cstring>

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

WeightHandle& WeightRegistry::handle(TensorID id) {
    if (id < 0 || id >= static_cast<TensorID>(handles_.size())) {
        IMP_LOG_FATAL("WeightRegistry::handle: id %d out of range [0, %zu)", id, handles_.size());
    }
    return handles_[id];
}

const WeightHandle& WeightRegistry::handle(TensorID id) const {
    if (id < 0 || id >= static_cast<TensorID>(handles_.size())) {
        IMP_LOG_FATAL("WeightRegistry::handle: id %d out of range [0, %zu)", id, handles_.size());
    }
    return handles_[id];
}

void WeightRegistry::clear() {
    handles_.clear();
}

} // namespace imp
