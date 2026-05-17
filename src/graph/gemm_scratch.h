#pragma once

// MMVQ (Q8_1-input GEMV) scratch — single file-scope buffer shared between
// the mmvq dispatch in `gemm_kernel_gguf.cu` and engine workspace init in
// `executor_workspace_buffers.cu`. R5 Slice 8.6 hoisted these out of
// `executor_kernels.cu` into their own TU now that the legacy
// `gemm_dispatch_impl` switch is retired and the only remaining consumer of
// `mmvq_scratch_get_or_grow` is the GGUF registry kernel.
//
// `prewarm_mmvq_scratch(max_tokens, max_K)` is called once at engine init
// with the model's largest expected dims. Idempotent: re-call with a smaller
// size is a no-op; re-call with a larger size grows the buffer. Without a
// prewarm, the first hot-path call into `mmvq_scratch_get_or_grow` triggers
// a `cudaMalloc` which is unsafe under CUDA graph capture — the cold path
// emits one ERROR log so the missing prewarm is visible.

#include <cstddef>

namespace imp {

void prewarm_mmvq_scratch(int max_tokens, int max_K);

// Hot-path getter. Returns the cached buffer if `g_mmvq_scratch_size >= need`,
// otherwise grows (capture-unsafe; emits a one-shot ERROR log).
void mmvq_scratch_get_or_grow(std::size_t need, void** out_buf, std::size_t* out_size);

}  // namespace imp
