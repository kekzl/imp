#include "core/threading.h"

namespace imp {

ThreadPool::ThreadPool(size_t n_threads) {
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 4;
    }
    workers_.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mu_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

} // namespace imp
