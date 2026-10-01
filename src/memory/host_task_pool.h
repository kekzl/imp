#pragma once

// Load-time host work on worker threads: submit() queues and returns, wait() joins.
// Used to pin and fill host-expert slabs in parallel (cudaHostAlloc + memcpy from the mmap).

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace imp {

class HostTaskPool {
public:
    explicit HostTaskPool(int n_threads);
    ~HostTaskPool();
    HostTaskPool(const HostTaskPool&) = delete;
    HostTaskPool& operator=(const HostTaskPool&) = delete;

    void submit(std::function<void()> task);
    // Returns once every submitted task has run; the pool stays usable.
    void wait();

private:
    void worker_();

    std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_idle_;
    std::deque<std::function<void()>> queue_;
    size_t in_flight_ = 0;
    bool stop_ = false;
    std::vector<std::thread> threads_;
};

}  // namespace imp
