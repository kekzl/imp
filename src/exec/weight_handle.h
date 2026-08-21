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
        // IMP_CHECK, not IMP_LOG_FATAL: the latter only LOGS (logging.h:58),
        // so this used to report "out of range" and then index out of range
        // anyway. Abort rather than throw, for two reasons specific to this
        // site: it is a precondition on an accessor called from inside
        // CUDA-graph capture regions, where unwinding is not a recovery, and
        // an out-of-range TensorID means the registry and its caller disagree
        // about identity - there is nothing to return.
        IMP_CHECK(id >= 0 && id < static_cast<TensorID>(self.handles_.size()),
                  "WeightRegistry::handle: id %d out of range [0, %zu)", id, self.handles_.size());
        return self.handles_[id];
    }
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
