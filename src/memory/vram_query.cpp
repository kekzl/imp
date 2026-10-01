#include "memory/vram_query.h"
#include "core/cuda_raii.h"
#include "core/logging.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>

namespace imp {

namespace {
// Process-wide budget state. Written once by vram_budget_install (engine
// init), read from every sizing site afterwards.
std::atomic<size_t> g_budget_bytes{0};
std::atomic<size_t> g_free_at_install{0};
std::atomic<size_t> g_total_at_install{0};
std::atomic<size_t> g_own_peak{0};
std::atomic<std::ptrdiff_t> g_reserved_uncommitted{0};
}  // namespace

void vram_reserved_uncommitted_add(std::ptrdiff_t delta_bytes) {
    g_reserved_uncommitted.fetch_add(delta_bytes, std::memory_order_relaxed);
}

size_t vram_reserved_uncommitted_bytes() {
    const std::ptrdiff_t v = g_reserved_uncommitted.load(std::memory_order_relaxed);
    return v > 0 ? static_cast<size_t>(v) : 0;
}

void vram_budget_install(size_t budget_mb) {
    size_t budget = budget_mb << 20;
    // Snapshot the baseline even when uncapped: it separates "this process's allocations"
    // from the CUDA context and any neighbour already on the card, and both the VRAM audit table and
    // the peak-VRAM gate need that split to say anything useful about a budget being
    // respected.
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

namespace {

bool issue_copies(const DeviceCopy* copies, size_t n, cudaStream_t stream) {
    for (size_t i = 0; i < n; i++)
        if (cudaMemcpyAsync(copies[i].dst, copies[i].src, copies[i].bytes, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess)
            return false;
    return true;
}

// One graph launch, not n copy submissions: 32 x 2 MiB issued one by one read 128-605 GB/s
// on a card with 30.8 GiB free (WSL2 submission gaps), 1218-1270 GB/s as a graph (#2366).
// Empty exec = capture failed, the caller issues the copies plainly.
CudaGraphExec capture_copies(const DeviceCopy* copies, size_t n, cudaStream_t stream) {
    CudaGraphExec exec;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        (void)cudaGetLastError();
        return exec;
    }
    const bool captured = issue_copies(copies, n, stream);
    cudaGraph_t g = nullptr;
    const bool ended = cudaStreamEndCapture(stream, &g) == cudaSuccess;
    CudaGraph graph;
    graph.reset(g);
    cudaGraphExec_t e = nullptr;
    if (captured && ended && g && cudaGraphInstantiate(&e, graph, 0) == cudaSuccess)
        exec.reset(e);
    else
        (void)cudaGetLastError();
    return exec;
}

}  // namespace

double device_copy_bandwidth_gbps(const DeviceCopy* copies, size_t n, int warm_ms) {
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        if (copies[i].dst == nullptr || copies[i].src == nullptr)
            return 0.0;
        total += copies[i].bytes;
    }
    if (total == 0)
        return 0.0;
    CudaEvent t0, t1;  // timing enabled for cudaEventElapsedTime
    CudaStream stream;
    if (!t0.create(cudaEventDefault) || !t1.create(cudaEventDefault) || !stream.create(cudaStreamNonBlocking))
        return 0.0;
    // The copies used to run on the legacy stream, ordered after its pending work; keep that order.
    if (cudaStreamSynchronize(nullptr) != cudaSuccess)
        return 0.0;
    const CudaGraphExec exec = capture_copies(copies, n, stream);
    auto launch = [&]() {
        return exec ? cudaGraphLaunch(exec, stream) == cudaSuccess : issue_copies(copies, n, stream);
    };
    // Warm the clocks: repeat the set until warm_ms of wall time has passed.
    bool ok = true;
    const auto warm_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(0, warm_ms));
    for (int iter = 0; ok && iter < 100000 && std::chrono::steady_clock::now() < warm_end; iter++)
        ok = launch() && cudaStreamSynchronize(stream) == cudaSuccess;
    // Timed window >= 1 GiB of traffic (max 16 launches), best of 3: one 64 MiB launch read 365-1562
    // GB/s on the same resident pool (#2366). A spilled pool cannot read high, so the max is safe.
    constexpr size_t kTimedTraffic = size_t{1} << 30;
    const int reps = static_cast<int>(std::clamp<size_t>(kTimedTraffic / (2 * total), 1, 16));
    double best = 0.0;
    for (int pass = 0; ok && pass < 3; pass++) {
        ok = cudaEventRecord(t0, stream) == cudaSuccess;
        for (int r = 0; ok && r < reps; r++)
            ok = launch();
        float ms = 0.0f;
        ok = ok && cudaEventRecord(t1, stream) == cudaSuccess && cudaEventSynchronize(t1) == cudaSuccess &&
             cudaEventElapsedTime(&ms, t0, t1) == cudaSuccess && ms > 0.0f;
        if (ok)
            best = std::max(best,
                            2.0 * static_cast<double>(total) * reps / (static_cast<double>(ms) * 1e-3) / 1e9);
    }
    t0.reset();
    t1.reset();
    if (!ok || best <= 0.0) {
        (void)cudaGetLastError();  // leave nothing sticky for the next caller
        return 0.0;
    }
    return best;
}

bool vram_budget_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    return vram_budget_mem_get_info_ex(free_bytes, total_bytes, /*exclude_pending=*/true);
}

bool vram_budget_mem_get_info_ex(size_t* free_bytes, size_t* total_bytes, bool exclude_pending) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        if (free_bytes)
            *free_bytes = 0;
        if (total_bytes)
            *total_bytes = 0;
        return false;
    }
    // Track own-usage high water here rather than in a sampler thread: this function is
    // called at every sizing site, exactly the phase in which the peak forms. Once serving
    // starts imp allocates nothing (I2), so the peak cannot move behind our back.
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
    // What the lazy pools were charged for and have not committed is not free for anyone who
    // sizes from this reading: without excluding it, the KV plan took the arena's deferred
    // charge as free and a later commit (e.g. the vision tower) would spill into a pool that
    // had already spent it.
    if (exclude_pending) {
        const size_t pending = vram_reserved_uncommitted_bytes();
        free_b = free_b > pending ? free_b - pending : 0;
    }
    if (free_bytes)
        *free_bytes = free_b;
    if (total_bytes)
        *total_bytes = total_b;
    return true;
}

}  // namespace imp
