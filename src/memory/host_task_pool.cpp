#include "memory/host_task_pool.h"

#include <utility>

namespace imp {

HostTaskPool::HostTaskPool(int n_threads) {
    if (n_threads < 1)
        n_threads = 1;
    threads_.reserve(static_cast<size_t>(n_threads));
    for (int i = 0; i < n_threads; ++i)
        threads_.emplace_back([this] { worker_(); });
}

HostTaskPool::~HostTaskPool() {
    wait();
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_work_.notify_all();
    for (std::thread& t : threads_)
        t.join();
}

void HostTaskPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(std::move(task));
        ++in_flight_;
    }
    cv_work_.notify_one();
}

void HostTaskPool::wait() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_idle_.wait(lk, [this] { return in_flight_ == 0; });
}

void HostTaskPool::worker_() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty())
                return;  // stop_ with nothing left
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        task();
        std::lock_guard<std::mutex> lk(mu_);
        if (--in_flight_ == 0)
            cv_idle_.notify_all();
    }
}

}  // namespace imp
