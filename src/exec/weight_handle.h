#pragma once

#include "core/logging.h"
#include "core/weight_handle.h"  // WeightHandle — moved down so compute/ need not include exec/
#include "core/qtype.h"
#include "core/tensor_kind.h"
#include "core/storage_tier.h"
#include "memory/device_owned.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>

namespace imp {

class WeightRegistry {
public:
    TensorID reserve(TensorKind kind, int64_t rows, int64_t cols);

    // C++23 deducing this: one overload serves const and non-const callers.
    template <typename Self>
    auto&& handle(this Self&& self, TensorID id) {
        // IMP_CHECK, not IMP_LOG_FATAL: the latter only logs (logging.h:56), so this used to
        // report "out of range" and then index out of range anyway. Abort rather than throw: this
        // is a precondition inside CUDA-graph capture regions (unwinding isn't a recovery), and
        // an out-of-range TensorID means the registry and its caller disagree about identity.
        IMP_CHECK(id >= 0 && id < static_cast<TensorID>(self.handles_.size()),
                  "WeightRegistry::handle: id %d out of range [0, %zu)", id, self.handles_.size());
        return self.handles_[id];
    }
    size_t size() const { return handles_.size(); }

    void clear();

    // An FP16 overlay (fused KV / gate+up) hands its buffer to handle `id`: payload.fp16.data and
    // owned_bytes describe it, the owner is kept here. WeightHandle stays a copyable view.
    void adopt_fp16_storage(TensorID id, DeviceOwned<void>&& storage);

    // Releases every adopted buffer once and nulls the payload of its handle (owned_bytes back
    // to 0). Handles that borrow (owned_bytes == 0) are left alone. Idempotent. Returns bytes.
    size_t free_owned_storage();

private:
    std::vector<WeightHandle> handles_;
    std::vector<DeviceOwned<void>> owned_storage_;  // survives clear(): released only above
};

}  // namespace imp
