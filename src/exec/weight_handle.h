#pragma once

#include "core/logging.h"
#include "core/weight_handle.h"  // WeightHandle — moved down so compute/ need not include exec/
#include "core/qtype.h"
#include "core/tensor_kind.h"
#include "core/storage_tier.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>

namespace imp {

class VRAMAllocator;

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

    // Free VRAM for any handle whose owned_bytes > 0. Borrowed handles (owned_bytes == 0)
    // are left alone; their storage is managed by the original allocator. Idempotent: each
    // freed handle's owned_bytes resets to 0 and payload to nullptr, so a second call is a
    // no-op.
    size_t free_owned_storage(VRAMAllocator* alloc);

private:
    std::vector<WeightHandle> handles_;
};

}  // namespace imp
