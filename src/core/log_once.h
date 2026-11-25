#pragma once

#include <atomic>

namespace imp {

// "Log once" latch that every model load re-arms (AUDIT_ARCH_2026_11_24 C-12, D-10): a plain
// `static bool logged` kept model B silent about a fallback that model A had tripped.
// Use: `static LogOnce s_warned; if (s_warned.first()) IMP_LOG_WARN(...);`
class LogOnce {
public:
    // Pushes itself onto a lock-free process list: no allocation, safe during static init.
    LogOnce() noexcept;
    LogOnce(const LogOnce&) = delete;
    LogOnce& operator=(const LogOnce&) = delete;

    // True exactly once per arming, from any thread; once fired, a relaxed load.
    [[nodiscard]] bool first() noexcept {
        return !fired_.load(std::memory_order_relaxed) && !fired_.exchange(true, std::memory_order_relaxed);
    }
    void rearm() noexcept { fired_.store(false, std::memory_order_relaxed); }

private:
    friend void log_once_rearm_all() noexcept;
    std::atomic<bool> fired_{false};
    LogOnce* next_ = nullptr;
};

// Re-arms every latch constructed so far. Runs in imp_context_create and, through the
// core/cuda_static_reset.h registry, in ~Engine and imp_gpu_release.
void log_once_rearm_all() noexcept;

// Latches constructed so far (an array of N counts N).
[[nodiscard]] int log_once_count() noexcept;

}  // namespace imp
