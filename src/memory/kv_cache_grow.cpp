// KVCache growth: try_grow_to and the grow-ahead worker (grow-ahead). Split from
// kv_cache.cu for the file-size gate; same class, same invariants.
#include "memory/kv_cache.h"
#include "memory/vram_query.h"
#include "core/cuda_raii.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <utility>

namespace imp {

namespace {
int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

bool KVCache::grow_refused_recently_() const {
    const int64_t at = grow_refused_at_ns_.load(std::memory_order_relaxed);
    return grow_retry_ms_ > 0 && at != 0 && steady_ns() - at < static_cast<int64_t>(grow_retry_ms_) * 1000000;
}

int KVCache::cap_by_free_vram_(int target, int base) {
    // The ceiling was sized from post-weight VRAM at init, before the library reserve and
    // forward scratch were claimed. A VMM commit does not fail when the card is full, it
    // spills into host memory at a fraction of the bandwidth (#1103), so growth is capped at
    // what is free above the headroom NOW; the planned commit stays the floor and nothing is
    // sized from it.
    const bool wants_more = target > base;
    size_t free_now = 0, total_now = 0;
    const size_t per_block = bytes_per_block();
    if (per_block > 0)
        free_vram_queries_.fetch_add(1, std::memory_order_relaxed);
    if (per_block > 0 && vram_budget_mem_get_info(&free_now, &total_now) && total_now > 0) {
        // The reading already excludes what the lazy pools were charged for and
        // have not committed (vram_budget_mem_get_info): this growth is
        // opportunistic, a promised SSM slot or the vision tower is not.
        const size_t headroom = vram_allocator_headroom(total_now);
        const size_t spare = free_now > headroom ? free_now - headroom : 0;
        const size_t affordable = static_cast<size_t>(base) + (spare / per_block);
        if (affordable < static_cast<size_t>(target)) {
            const int capped = static_cast<int>(affordable);
            if (last_capped_logged_.exchange(capped, std::memory_order_relaxed) != capped) {
                IMP_LOG_INFO(
                    "KV cache: growth capped %d -> %d blocks by free VRAM (%.0f MiB free, %.0f MiB "
                    "allocator headroom kept)",
                    target, capped, free_now / (1024.0 * 1024.0), headroom / (1024.0 * 1024.0));
            }
            target = capped;
        }
    }
    // Nothing affordable: open the retry window, the next reading would say the same.
    if (wants_more && target <= base)
        grow_refused_at_ns_.store(steady_ns(), std::memory_order_relaxed);
    return target;
}

int KVCache::try_grow_to(int wanted) { return try_grow_to_(wanted, /*honor_retry=*/true); }

int KVCache::try_grow_to_now(int wanted) { return try_grow_to_(wanted, /*honor_retry=*/false); }

int KVCache::try_grow_to_(int wanted, bool honor_retry) {
    if (!growable_)
        return blocks_.num_blocks();
    int have = adopt_grown();
    if (wanted <= have)
        return have;
    int target = std::min(wanted, max_blocks_);
    if (target <= have)
        return have;
    // A refused growth a moment ago: one cudaMemGetInfo per block on an empty free list
    // was the admission wait (Gemma-4 pp8192 at the cap: 504 blocks, 22.5-38.1 ms queue).
    if (honor_retry && grow_refused_recently_())
        return have;
    // Waits out an in-flight grow-ahead commit, whose blocks may already cover the target.
    std::lock_guard<std::mutex> commit_lock(commit_mu_);
    have = adopt_grown();
    if (target <= have)
        return have;
    target = cap_by_free_vram_(target, have);
    if (target <= have)
        return have;
    const auto grow_t0 = std::chrono::steady_clock::now();
    last_zero_ms_ = 0.0;
    const int got = commit_blocks_(target);
    staged_blocks_.store(committed_blocks_, std::memory_order_release);
    if (got <= have) {
        grow_refused_at_ns_.store(steady_ns(), std::memory_order_relaxed);  // the driver refused
        return have;
    }
    if (blocks_.grow_slots(got) != MemError::Ok)
        return have;
    const double grow_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - grow_t0).count();
    // Published only after the ids exist: a reader that saw the new capacity
    // first would admit a request the pool cannot yet hand blocks for.
    usable_blocks_.store(got, std::memory_order_release);
    growths_.fetch_add(1, std::memory_order_relaxed);
    IMP_LOG_INFO(
        "KV cache: grew %d -> %d blocks (%.0f tokens, %.0f MiB committed of a %d-block "
        "ceiling) in %.1f ms (zero + sync %.1f ms)",
        have, got, static_cast<double>(got) * block_size_, region_.committed() / (1024.0 * 1024.0),
        max_blocks_, grow_ms, last_zero_ms_);
    return got;
}

int KVCache::adopt_grown() {
    const int have = usable_blocks_.load(std::memory_order_acquire);
    const int staged = staged_blocks_.load(std::memory_order_acquire);
    if (staged <= have || blocks_.grow_slots(staged) != MemError::Ok)
        return have;
    // Same order as try_grow_to: ids first, then the capacity admission reads.
    usable_blocks_.store(staged, std::memory_order_release);
    growths_.fetch_add(1, std::memory_order_relaxed);
    return staged;
}

void KVCache::start_grow_ahead() {
    if (accounting_only_ || !growable_ || !region_ || grow_thread_.joinable())
        return;
    if (cudaGetDevice(&grow_device_) != cudaSuccess)
        return;
    staged_blocks_.store(committed_blocks_, std::memory_order_release);
    grow_thread_ = std::thread(&KVCache::grow_ahead_loop_, this);
    IMP_LOG_INFO("KV cache: grow-ahead on (next step committed off the admission path, %d of %d blocks now)",
                 usable_blocks_.load(std::memory_order_relaxed), max_blocks_);
}

void KVCache::grow_ahead_tick() {
    if (!grow_thread_.joinable())
        return;
    const int have = adopt_grown();
    if (have >= max_blocks_)
        return;
    const int step = std::max({64, have / 4, last_admit_blocks_});
    if (blocks_.free_count() >= step || grow_refused_recently_())
        return;
    {
        std::lock_guard<std::mutex> lk(grow_mu_);
        grow_want_ = std::max(grow_want_, have + step);
    }
    grow_cv_.notify_one();
}

void KVCache::grow_ahead_loop_() {
    if (cudaSetDevice(grow_device_) != cudaSuccess)
        return;
    // Own non-blocking stream: the legacy stream would order the zeroing behind the engine's
    // in-flight forward on this WSL2 host (a legacy copy waited for a non-blocking stream, #2734).
    CudaStream s;
    if (!s.create(cudaStreamNonBlocking))
        (void)cudaGetLastError();  // zeroes on the legacy stream then
    for (;;) {
        int want = 0;
        {
            std::unique_lock<std::mutex> lk(grow_mu_);
            grow_cv_.wait(lk, [this] { return grow_stop_ || grow_want_ > 0; });
            if (grow_stop_)
                break;
            want = std::exchange(grow_want_, 0);
        }
        std::lock_guard<std::mutex> commit_lock(commit_mu_);
        const int base = committed_blocks_;
        const int target = cap_by_free_vram_(std::min(want, max_blocks_), base);
        if (target <= base)
            continue;
        const auto t0 = std::chrono::steady_clock::now();
        last_zero_ms_ = 0.0;
        zero_stream_ = s.get();
        const int got = commit_blocks_(target);
        zero_stream_ = nullptr;
        staged_blocks_.store(committed_blocks_, std::memory_order_release);
        if (got <= base) {
            grow_refused_at_ns_.store(steady_ns(), std::memory_order_relaxed);
            continue;
        }
        growths_ahead_.fetch_add(1, std::memory_order_relaxed);
        IMP_LOG_INFO(
            "KV cache: grew ahead %d -> %d blocks (%.0f MiB committed of a %d-block ceiling) in "
            "%.1f ms off the admission path (zero + sync %.1f ms)",
            base, got, region_.committed() / (1024.0 * 1024.0), max_blocks_,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
            last_zero_ms_);
    }
}

void KVCache::stop_grow_ahead_() {
    if (!grow_thread_.joinable())
        return;
    {
        std::lock_guard<std::mutex> lk(grow_mu_);
        grow_stop_ = true;
    }
    grow_cv_.notify_one();
    grow_thread_.join();
}

}  // namespace imp
