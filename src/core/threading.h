#pragma once

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <vector>

namespace imp {

class ThreadPool {
public:
    // Create pool with n threads. 0 = hardware_concurrency.
    explicit ThreadPool(size_t n_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Submit a task and get a future for the result.
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return future;
    }

    size_t num_threads() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

} // namespace imp
