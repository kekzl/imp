#include "memory/mem_account.h"
#include "core/logging.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

namespace imp {

namespace {
constexpr double kMiB = 1024.0 * 1024.0;

void query_free_total(size_t& free_b, size_t& total_b) {
    free_b = 0;
    total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
}
}  // namespace

MemAccount& MemAccount::instance() {
    static MemAccount inst;
    return inst;
}

MemAccount::~MemAccount() {
    sampler_stop();
}

void MemAccount::set_dump_path(std::string path) {
    std::lock_guard<std::mutex> lock(mu_);
    dump_path_ = std::move(path);
}

MemAccount::Pool& MemAccount::pool_locked(const char* name) {
    for (auto& p : pools_) {
        if (p.name == name)
            return p;
    }
    pools_.push_back(Pool{name, 0, 0, 0});
    return pools_.back();
}

void MemAccount::note(const char* pool, std::ptrdiff_t delta_bytes) {
    if (!enabled_.load(std::memory_order_relaxed) || delta_bytes == 0)
        return;
    std::lock_guard<std::mutex> lock(mu_);
    Pool& p = pool_locked(pool);
    p.current += delta_bytes;
    if (delta_bytes > 0)
        p.alloc_count++;
    if (p.current > p.peak)
        p.peak = p.current;
}

void MemAccount::checkpoint(const char* name) {
    size_t free_b = 0, total_b = 0;
    query_free_total(free_b, total_b);
    sample_once();  // fold the checkpoint instant into the peak too
    if (!enabled_.load(std::memory_order_relaxed))
        return;
    std::lock_guard<std::mutex> lock(mu_);
    checkpoints_.push_back(Checkpoint{name, free_b, total_b - free_b});
}

void MemAccount::sample_once() {
    size_t free_b = 0, total_b = 0;
    query_free_total(free_b, total_b);
    if (!total_b)
        return;
    size_t used = total_b - free_b;
    size_t prev = peak_used_.load(std::memory_order_relaxed);
    while (used > prev && !peak_used_.compare_exchange_weak(prev, used, std::memory_order_relaxed)) {
    }
}

void MemAccount::set_named_charges(size_t context_bytes, size_t library_bytes, size_t arena_bytes,
                                   size_t arena_high_water) {
    std::lock_guard<std::mutex> lock(mu_);
    named_context_ = context_bytes;
    named_library_ = library_bytes;
    named_arena_ = arena_bytes;
    named_arena_high_ = arena_high_water;
}

void MemAccount::arm_steady_state_watermarks() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return;
    // Writing a *High attribute resets it to the current value.
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess && pool) {
        unsigned long long cur = 0;
        if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &cur) == cudaSuccess)
            (void)cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &cur);
        if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &cur) == cudaSuccess)
            (void)cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &cur);
    }
    unsigned long long zero = 0;
    (void)cudaDeviceSetGraphMemAttribute(dev, cudaGraphMemAttrUsedMemHigh, &zero);
    (void)cudaGetLastError();  // graph mem attrs are unsupported on some stacks
}

void MemAccount::sampler_start(int interval_us) {
    if (!enabled_.load(std::memory_order_relaxed))
        return;
    if (sampler_run_.exchange(true))
        return;  // already running
    sampler_interval_us_ = interval_us > 0 ? interval_us : 2000;
    sampler_ = std::thread([this] {
        while (sampler_run_.load(std::memory_order_relaxed)) {
            sample_once();
            std::this_thread::sleep_for(std::chrono::microseconds(sampler_interval_us_));
        }
    });
    IMP_LOG_INFO("MemAccount: peak sampler started (interval=%d us)", sampler_interval_us_);
}

void MemAccount::sampler_stop() {
    if (!sampler_run_.exchange(false))
        return;
    if (sampler_.joinable())
        sampler_.join();
}

void MemAccount::report(const char* phase_label) {
    sample_once();

    size_t free_b = 0, total_b = 0;
    query_free_total(free_b, total_b);
    size_t used = total_b ? (total_b - free_b) : 0;
    size_t peak = peak_used_.load(std::memory_order_relaxed);
    peak = std::max(peak, used);

    std::lock_guard<std::mutex> lock(mu_);

    // Build the table into a buffer so it lands atomically in the log and the
    // append-only dump file.
    std::string out;
    char line[256];
    auto emit = [&](const char* fmt, auto... args) {
        // fmt is a compile-time literal at every call site below; the forwarding
        // through a const char* parameter is what trips -Wformat-security.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
        std::snprintf(line, sizeof(line), fmt, args...);
#pragma GCC diagnostic pop
        out += line;
        out += '\n';
    };

    emit("===== VRAM AUDIT [%s] =====", phase_label ? phase_label : "?");
    emit("device: total=%.0f MiB  used=%.0f MiB  free=%.0f MiB  peak_used=%.0f MiB",
         total_b / kMiB, used / kMiB, free_b / kMiB, peak / kMiB);

    // cudaMallocAsync default-pool slack: freed-but-not-returned-to-OS memory
    // (e.g. the #679 ms_ref cudaFree's) sits here as reserved and still counts
    // as device-used. reserved - used = trimmable headroom (cudaMemPoolTrimTo).
    {
        cudaMemPool_t pool = nullptr;
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess && pool) {
            unsigned long long rsv = 0, usd = 0;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &rsv);
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &usd);
            emit("mempool(async): reserved=%.0f MiB  used=%.0f MiB  trimmable=%.0f MiB",
                 rsv / kMiB, usd / kMiB, (double(rsv) - double(usd)) / kMiB);
            // I2 / criterion 3. Armed at the Serving transition, so anything
            // above the value it was armed at was allocated while serving —
            // and unlike steady_state_allocations() this sees allocations that
            // never touched Backend.
            unsigned long long usd_hi = 0, rsv_hi = 0;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &usd_hi);
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &rsv_hi);
            emit("mempool(async) high-water since serving began: used=%.0f MiB  reserved=%.0f MiB "
                 "(delta vs now: used %+.0f MiB)",
                 usd_hi / kMiB, rsv_hi / kMiB, (double(usd_hi) - double(usd)) / kMiB);
        }
        // Graph-owned memory: stream-ordered allocations captured INSIDE a
        // graph land here, not in the default pool. A5.2 predicts this should
        // reach zero once step 5 removes per-request cudaMallocAsync from the
        // captured regions — at which point the cudaDeviceGraphMemTrim calls
        // in cuda_graph.cu become provably dead.
        {
            int gdev = 0;
            cudaGetDevice(&gdev);
            unsigned long long g_used = 0, g_high = 0, g_rsv = 0;
            const bool ok =
                cudaDeviceGetGraphMemAttribute(gdev, cudaGraphMemAttrUsedMemCurrent, &g_used) ==
                cudaSuccess;
            cudaDeviceGetGraphMemAttribute(gdev, cudaGraphMemAttrUsedMemHigh, &g_high);
            cudaDeviceGetGraphMemAttribute(gdev, cudaGraphMemAttrReservedMemCurrent, &g_rsv);
            (void)cudaGetLastError();
            if (ok) {
                emit("graphmem: used=%.1f MiB  reserved=%.1f MiB  high_since_serving=%.1f MiB",
                     g_used / kMiB, g_rsv / kMiB, g_high / kMiB);
            }
        }
    }

    if (!checkpoints_.empty()) {
        emit("--- lifecycle checkpoints (phase delta = measured cost) ---");
        emit("%-26s %12s %12s %12s", "checkpoint", "used_MiB", "free_MiB", "delta_MiB");
        size_t prev_used = 0;
        bool first = true;
        for (const auto& c : checkpoints_) {
            double delta = first ? 0.0 : (double(c.used_bytes) - double(prev_used)) / kMiB;
            emit("%-26s %12.1f %12.1f %12.1f", c.name.c_str(), c.used_bytes / kMiB,
                 c.free_bytes / kMiB, delta);
            prev_used = c.used_bytes;
            first = false;
        }
    }

    if (!pools_.empty()) {
        std::vector<Pool> sorted = pools_;
        std::sort(sorted.begin(), sorted.end(),
                  [](const Pool& a, const Pool& b) { return a.peak > b.peak; });
        emit("--- per-pool notes (tracked allocation sites) ---");
        emit("%-26s %12s %12s %10s", "pool", "current_MiB", "peak_MiB", "allocs");
        int64_t cur_sum = 0, peak_sum = 0;
        for (const auto& p : sorted) {
            emit("%-26s %12.1f %12.1f %10lld", p.name.c_str(), p.current / kMiB, p.peak / kMiB,
                 (long long)p.alloc_count);
            cur_sum += p.current;
            peak_sum += p.peak;
        }
        emit("%-26s %12.1f %12.1f", "TRACKED TOTAL", cur_sum / kMiB, peak_sum / kMiB);

        // Named charges. These are real device memory that the per-pool notes
        // structurally cannot see — not imp allocations at all in two of the
        // three cases — so folding them into "untracked" made the residual
        // look like unexplained loss. Criterion 6 asks for >=95% accounted
        // "remainder explicitly attributed (context, driver, library
        // internals)"; this is that attribution.
        int64_t named = 0;
        if (named_context_) {
            emit("%-26s %12.1f    CUDA context + driver", "  named: context",
                 named_context_ / kMiB);
            named += static_cast<int64_t>(named_context_);
        }
        if (named_library_) {
            emit("%-26s %12.1f    claimed on first forward (A1.5)", "  named: library reserve",
                 named_library_ / kMiB);
            named += static_cast<int64_t>(named_library_);
        }
        if (named_arena_) {
            emit("%-26s %12.1f    high-water %.1f MiB", "  named: engine arena",
                 named_arena_ / kMiB, named_arena_high_ / kMiB);
            named += static_cast<int64_t>(named_arena_);
        }

        const double accounted = double(cur_sum) + double(named);
        const double residual = double(used) - accounted;
        emit("%-26s %12.1f", "ACCOUNTED (tracked+named)", accounted / kMiB);
        emit("%-26s %12.1f    %.1f%% of device used", "RESIDUAL (unattributed)", residual / kMiB,
             used ? (100.0 * (1.0 - residual / double(used))) : 0.0);
    }
    emit("===== END VRAM AUDIT [%s] =====", phase_label ? phase_label : "?");

    IMP_LOG_INFO("\n%s", out.c_str());

    if (!dump_path_.empty()) {
        FILE* f = std::fopen(dump_path_.c_str(), "a");
        if (f) {
            std::fputs(out.c_str(), f);
            std::fputc('\n', f);
            std::fclose(f);
        }
    }
}

void trim_device_mempool() {
    // Retire pending async frees before trimming, otherwise their blocks are
    // still referenced and survive the trim.
    cudaDeviceSynchronize();
    int dev = 0;
    cudaMemPool_t pool = nullptr;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess && pool != nullptr) {
        unsigned long long rsv_before = 0, used_before = 0, rsv_after = 0, used_after = 0;
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &rsv_before);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_before);
        cudaError_t te = cudaMemPoolTrimTo(pool, 0);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &rsv_after);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_after);
        IMP_LOG_INFO("mempool trim: reserved %.0f->%.0f MiB used %.0f->%.0f MiB (rc=%s)",
                     rsv_before / kMiB, rsv_after / kMiB, used_before / kMiB, used_after / kMiB,
                     cudaGetErrorString(te));
    }
    // Clear any sticky error (e.g. sync/trim during process-exit teardown when
    // the runtime has already torn the pool down) so a later cudaGetLastError
    // in a caller's destructor doesn't misattribute it.
    (void)cudaGetLastError();
}

}  // namespace imp
