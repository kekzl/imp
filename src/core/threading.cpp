#include "core/threading.h"

namespace imp {

ThreadPool::ThreadPool(size_t n_threads) {
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 4;
    }
    workers_.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this](std::stop_token stoken) {
            // Wake this thread when stop is requested so cv_.wait() re-evaluates
            std::stop_callback on_stop(stoken, [this] { cv_.notify_all(); });
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mu_);
                    cv_.wait(lock, [&] { return stoken.stop_requested() || !tasks_.empty(); });
                    if (stoken.stop_requested() && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    // jthread destructors call request_stop() + join().
    // stop_callback in each worker wakes cv_.wait() on stop request.
}

} // namespace imp
