#include "core/log_once.h"

#include "core/cuda_static_reset.h"

namespace imp {

namespace {

// constinit: set before any dynamic init, so a file-scope latch of another TU can register.
constinit std::atomic<LogOnce*> g_head{nullptr};
constinit std::atomic<int> g_count{0};

}  // namespace

LogOnce::LogOnce() noexcept {
    LogOnce* head = g_head.load(std::memory_order_relaxed);
    do {
        next_ = head;
    } while (!g_head.compare_exchange_weak(head, this, std::memory_order_release, std::memory_order_relaxed));
    g_count.fetch_add(1, std::memory_order_relaxed);
}

void log_once_rearm_all() noexcept {
    // Latches have static storage and a trivial destructor: the list never shrinks or dangles.
    for (LogOnce* p = g_head.load(std::memory_order_acquire); p; p = p->next_)
        p->rearm();
}

int log_once_count() noexcept { return g_count.load(std::memory_order_relaxed); }

namespace {

// ~Engine and imp_gpu_release() run this registry: a model swap re-arms every latch.
IMP_REGISTER_CUDA_STATIC_RESET(log_once_rearm_all);

}  // namespace

}  // namespace imp
