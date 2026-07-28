#include "exec/gemm_scratch.h"

#include "core/logging.h"
#include "memory/engine_arena.h"

#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// MMVQ (Q8_1-input GEMV) scratch — file-scope, sized once at workspace init
// via prewarm_mmvq_scratch(). The hot-path mmvq_scratch_get_or_grow() reads
// the cached size.
//
// A7 step 4: both slabs prefer the engine-persistent arena, so neither path
// calls the driver when the arena can serve it. That matters for the grow
// branch, which this comment used to describe as "capture-unsafe": a
// cudaMalloc inside a captured region fails the capture outright, whereas an
// arena take is a pointer bump and is safe anywhere.
//
// The demand is max_tokens x ceil(K/32) x 36 x 2, and max_tokens is the
// EXECUTOR's, not something the engine can bound when it opens the arena
// (896 on a bench run, 4096 on a server default — 108 MiB at K=12288). So a
// shortfall falls back to a direct allocation with a WARN rather than
// failing: the tier is not planner-sized yet (A4), and a guessed reservation
// that is too small must not be what breaks a model. When the planner sizes
// T2 from measured high-water marks, the fallback goes.
//
// A superseded slab taken from the arena stays stranded there — bounded,
// one-time, and visible in the arena's high-water mark.
//
// R5 Slice 8.6: hoisted out of executor_kernels.cu now that the legacy
// `gemm_dispatch_impl` switch is retired. Single-TU ownership keeps the
// global state private to this unit and the public header declares only
// the two entry points.
// ---------------------------------------------------------------------------
namespace {
void* g_mmvq_scratch = nullptr;
size_t g_mmvq_scratch_size = 0;
// The arena generation the cached pointer belongs to. A close (engine
// teardown) or reset invalidates every span it handed out; without this the
// static would survive into the next engine still pointing at the old region.
// The pre-arena code had the same hazard against a cudaFree'd pointer — this
// closes it rather than reproducing it.
uint64_t g_mmvq_scratch_gen = 0;
// Sentinel generation for a slab we allocated ourselves (arena shortfall).
constexpr uint64_t kFallbackGen = ~0ull;

bool scratch_is_live() {
    if (!g_mmvq_scratch)
        return false;
    // A fallback slab (gen 0 sentinel) is ours and outlives arena resets.
    return g_mmvq_scratch_gen == kFallbackGen || g_mmvq_scratch_gen == engine_arena().generation();
}

// Take `bytes` from the arena, falling back to a direct allocation. Returns
// false only if BOTH fail. Sets g_mmvq_scratch/_size/_gen on success.
bool take_scratch(size_t bytes) {
    auto slab = engine_arena().take_bytes(bytes);
    if (!slab.empty()) {
        g_mmvq_scratch = slab.data();
        g_mmvq_scratch_size = bytes;
        g_mmvq_scratch_gen = engine_arena().generation();
        return true;
    }
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("mmvq scratch: arena short by %zu B AND cudaMalloc failed: %s", bytes,
                      cudaGetErrorString(err));
        return false;
    }
    IMP_LOG_WARN("mmvq scratch: engine arena could not supply %.2f MiB (remaining %.2f MiB) — "
                 "fell back to a direct allocation. The T2 tier is not planner-sized yet.",
                 bytes / (1024.0 * 1024.0), engine_arena().remaining() / (1024.0 * 1024.0));
    g_mmvq_scratch = p;
    g_mmvq_scratch_size = bytes;
    g_mmvq_scratch_gen = kFallbackGen;
    return true;
}
}  // namespace

void prewarm_mmvq_scratch(int max_tokens, int max_K) {
    if (max_tokens <= 0 || max_K <= 0)
        return;
    const size_t per_call = static_cast<size_t>(max_tokens) * ((max_K + 31) / 32) * 36;
    const size_t need = per_call * 2;
    if (scratch_is_live() && g_mmvq_scratch_size >= need)
        return;
    if (!take_scratch(need)) {
        g_mmvq_scratch = nullptr;
        g_mmvq_scratch_size = 0;
        return;
    }
    IMP_LOG_INFO("MMVQ scratch pre-warmed: %.2f KiB (max_tokens=%d, max_K=%d)",
                 need / 1024.0, max_tokens, max_K);
}

void mmvq_scratch_get_or_grow(std::size_t need, void** out_buf, std::size_t* out_size) {
    if (scratch_is_live() && g_mmvq_scratch_size >= need) {
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
            "(max_tokens, max_K).",
            need, g_mmvq_scratch_size);
    }
    // The superseded slab is not reclaimable — that is the arena's contract.
    if (!take_scratch(need * 2)) {
        g_mmvq_scratch = nullptr;
        g_mmvq_scratch_size = 0;
        *out_buf = nullptr;
        *out_size = 0;
        return;
    }
    *out_buf = g_mmvq_scratch;
    *out_size = g_mmvq_scratch_size;
}

}  // namespace imp
