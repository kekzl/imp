#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace imp {

// Per-expert device F16 tensors of one kind (gate, up or down) -> one [ne, rows, cols] buffer, so
// the n == 1 decode GEMV and the grouped prefill read them as a packed 3-D tensor (Ling-3.0 BF16:
// per-expert tensors took the host-args legacy MoE path, which aborts the decode graph capture).

// Packable: >= 2 experts, all on the device, F16, 2-D, one shape. Writes {ne, rows, cols}.
[[nodiscard]] bool expert_pack_shape(const std::vector<Tensor>& experts, int64_t out_shape[3]);

using ExpertPackAlloc = std::function<void*(size_t bytes)>;              // nullptr = no room
using ExpertPackRelease = std::function<void(void* ptr, size_t bytes)>;  // one expert's buffer

// Experts all in `allocs` (own allocations): copies them into one buffer (tracked in `allocs`),
// releases each expert buffer, re-points the experts. False = nothing changed.
[[nodiscard]] bool pack_device_experts(std::vector<Tensor>& experts, Tensor& packed,
                                       std::vector<void*>& allocs, const ExpertPackAlloc& alloc,
                                       const ExpertPackRelease& release, cudaStream_t stream);

}  // namespace imp
