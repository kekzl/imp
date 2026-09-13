#pragma once

// MMVQ (Q8_1-input GEMV) scratch, file-scope, shared between the mmvq dispatch
// (gemm_kernel_gguf.cu) and engine workspace init (executor_workspace_buffers.cu).
// prewarm_mmvq_scratch(max_tokens, max_K) is called once at engine init; idempotent
// (smaller size no-ops, larger grows). Without prewarm, the first hot-path call triggers
// a cudaMalloc, unsafe under CUDA graph capture (logged as one ERROR).

#include <cstddef>

namespace imp {

void prewarm_mmvq_scratch(int max_tokens, int max_K);

// Hot-path getter. Returns the cached buffer if `g_mmvq_scratch_size >= need`,
// otherwise grows (capture-unsafe; emits a one-shot ERROR log).
void mmvq_scratch_get_or_grow(std::size_t need, void** out_buf, std::size_t* out_size);

}  // namespace imp
