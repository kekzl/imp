#pragma once

#include "core/tensor.h"
#include "memory/host_pinned.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace imp {

// Token embedding in mapped pinned host memory (#2484): embedding_lookup reads the mapped view,
// one row per token per step. F32/F16/BF16 tables are stored FP16, Q8_0/Q6_K raw.
[[nodiscard]] bool host_embedding_supported(QType q);

// FP16 bit patterns of n source elements (F32, F16 or BF16), split over `threads` workers.
void embedding_to_fp16(const void* src, QType q, size_t n, uint16_t* dst, unsigned threads);

// Moves `emb` into a new mapped pinned buffer appended to `pins`; emb.data becomes its device
// view. False (emb untouched) on an unsupported qtype or a failed pinned allocation.
[[nodiscard]] bool place_embedding_on_host(Tensor& emb, std::vector<PinnedBuffer>& pins);

}  // namespace imp
