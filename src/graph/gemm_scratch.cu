#include "graph/gemm_scratch.h"

#include "core/logging.h"

#include <cuda_runtime.h>
#include <atomic>

namespace imp {

// ---------------------------------------------------------------------------
// MMVQ (Q8_1-input GEMV) scratch — file-scope, sized once at workspace init
// via prewarm_mmvq_scratch(). The hot-path mmvq_scratch_get_or_grow() reads
// the cached size; the grow branch (cudaFree+cudaMalloc) is the cold-path
// fallback and capture-unsafe — engine init MUST prewarm to model max dims.
//
// R5 Slice 8.6: hoisted out of executor_kernels.cu now that the legacy
// `gemm_dispatch_impl` switch is retired. Single-TU ownership keeps the
// global state private to this unit and the public header declares only
// the two entry points.
// ---------------------------------------------------------------------------
namespace {
void* g_mmvq_scratch = nullptr;
size_t g_mmvq_scratch_size = 0;
}  // namespace

void prewarm_mmvq_scratch(int max_tokens, int max_K) {
    if (max_tokens <= 0 || max_K <= 0)
        return;
    const size_t per_call = static_cast<size_t>(max_tokens) * ((max_K + 31) / 32) * 36;
    const size_t need = per_call * 2;
    if (g_mmvq_scratch && g_mmvq_scratch_size >= need)
        return;
    if (g_mmvq_scratch)
        IMP_CUDA_CHECK_LOG(cudaFree(g_mmvq_scratch));
    cudaError_t err = cudaMalloc(&g_mmvq_scratch, need);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("prewarm_mmvq_scratch: cudaMalloc(%zu) failed: %s", need,
                      cudaGetErrorString(err));
        g_mmvq_scratch = nullptr;
        g_mmvq_scratch_size = 0;
        return;
    }
    g_mmvq_scratch_size = need;
    IMP_LOG_INFO("MMVQ scratch pre-warmed: %.2f KiB (max_tokens=%d, max_K=%d)",
                 need / 1024.0, max_tokens, max_K);
}

void mmvq_scratch_get_or_grow(std::size_t need, void** out_buf, std::size_t* out_size) {
    if (g_mmvq_scratch && g_mmvq_scratch_size >= need) {
        *out_buf = g_mmvq_scratch;
        *out_size = g_mmvq_scratch_size;
        return;
    }
    // Cold path: prewarm missed (or model dim changed mid-run). Re-grow.
    // Capture-unsafe; emits one ERROR log so the missing prewarm is visible.
    static std::atomic<bool> s_warned{false};
    if (!s_warned.exchange(true)) {
        IMP_LOG_ERROR(
            "mmvq_scratch_get_or_grow: hot-path grow fired (need=%zu, have=%zu) — "
            "engine init did not call prewarm_mmvq_scratch() with the model's "
            "(max_tokens, max_K). cudaMalloc inside graph capture will fail.",
            need, g_mmvq_scratch_size);
    }
    if (g_mmvq_scratch)
        IMP_CUDA_CHECK_LOG(cudaFree(g_mmvq_scratch));
    cudaError_t err = cudaMalloc(&g_mmvq_scratch, need * 2);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("mmvq_scratch_get_or_grow: cudaMalloc(%zu) failed: %s", need * 2,
                      cudaGetErrorString(err));
        g_mmvq_scratch = nullptr;
        g_mmvq_scratch_size = 0;
        *out_buf = nullptr;
        *out_size = 0;
        return;
    }
    g_mmvq_scratch_size = need * 2;
    *out_buf = g_mmvq_scratch;
    *out_size = g_mmvq_scratch_size;
}

}  // namespace imp
