#include "memory/vram_query.h"
#include "core/logging.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>

namespace imp {

namespace {
// Process-wide budget state. Written once by vram_budget_install (engine
// init), read from every sizing site afterwards.
std::atomic<size_t> g_budget_bytes{0};
std::atomic<size_t> g_free_at_install{0};
std::atomic<size_t> g_total_at_install{0};
std::atomic<size_t> g_own_peak{0};
}  // namespace

void vram_budget_install(size_t budget_mb) {
    size_t budget = budget_mb << 20;
    // Snapshot the baseline even when uncapped. It is what separates "this
    // process's allocations" from the CUDA context and any neighbour already
    // on the card, and both --mem-report and the peak-VRAM gate need that
    // split to say anything useful about a budget being respected.
    size_t free_b = 0, total_b = 0;
    const bool have_info = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    if (have_info) {
        g_free_at_install.store(free_b, std::memory_order_relaxed);
        g_total_at_install.store(total_b, std::memory_order_relaxed);
    }
    if (budget == 0) {
        g_budget_bytes.store(0, std::memory_order_relaxed);
        return;
    }
    if (!have_info) {
        IMP_LOG_WARN("vram_budget: cudaMemGetInfo failed — budget disabled");
        g_budget_bytes.store(0, std::memory_order_relaxed);
        return;
    }
    if (budget > total_b) {
        IMP_LOG_WARN("vram_budget: %zu MiB exceeds device total %zu MiB — clamping", budget_mb,
                     total_b >> 20);
        budget = total_b;
    }
    g_free_at_install.store(free_b, std::memory_order_relaxed);
    g_budget_bytes.store(budget, std::memory_order_relaxed);
    IMP_LOG_INFO("VRAM budget: %.0f MiB (device free at install: %.0f MiB) — sizing sees a "
                 "virtual %.0f MiB GPU",
                 budget / (1024.0 * 1024.0), free_b / (1024.0 * 1024.0),
                 budget / (1024.0 * 1024.0));
}

size_t vram_budget_bytes() { return g_budget_bytes.load(std::memory_order_relaxed); }

size_t vram_used_at_install_bytes() {
    const size_t total = g_total_at_install.load(std::memory_order_relaxed);
    const size_t free_b = g_free_at_install.load(std::memory_order_relaxed);
    return total > free_b ? total - free_b : 0;
}

size_t vram_own_peak_bytes() { return g_own_peak.load(std::memory_order_relaxed); }

size_t vram_own_used_bytes() {
    const size_t baseline = g_free_at_install.load(std::memory_order_relaxed);
    if (baseline == 0)
        return 0;  // install never ran
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return baseline > free_b ? baseline - free_b : 0;
}

bool vram_budget_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        if (free_bytes)
            *free_bytes = 0;
        if (total_bytes)
            *total_bytes = 0;
        return false;
    }
    // Track own-usage high water here rather than in a sampler thread: this
    // function is called at every sizing site, which is exactly the phase in
    // which the peak forms. Once serving starts imp allocates nothing (I2), so
    // the peak cannot move behind our back.
    const size_t baseline = g_free_at_install.load(std::memory_order_relaxed);
    if (baseline > 0) {
        const size_t own = (baseline > free_b) ? (baseline - free_b) : 0;
        size_t prev = g_own_peak.load(std::memory_order_relaxed);
        while (own > prev && !g_own_peak.compare_exchange_weak(prev, own, std::memory_order_relaxed))
            ;
    }
    const size_t budget = g_budget_bytes.load(std::memory_order_relaxed);
    if (budget > 0) {
        const size_t my_used = (baseline > free_b) ? (baseline - free_b) : 0;
        const size_t budget_left = (budget > my_used) ? (budget - my_used) : 0;
        free_b = std::min(free_b, budget_left);
        total_b = budget;
    }
    if (free_bytes)
        *free_bytes = free_b;
    if (total_bytes)
        *total_bytes = total_b;
    return true;
}

}  // namespace imp
