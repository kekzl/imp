#pragma once

// Pins host-resident NVFP4 expert projections on worker threads: per projection one
// cudaHostAlloc(Mapped) slab + memcpy from the mmap, 8 in parallel (#2336).

#include "core/tensor.h"
#include "memory/host_pinned.h"
#include "memory/host_task_pool.h"

#include <mutex>
#include <optional>
#include <vector>

namespace imp {

class HostExpertPinner {
public:
    explicit HostExpertPinner(std::vector<PinnedBuffer>& sink) : sink_(sink) {}
    // Joins every pin task, then appends the slabs to `sink`.
    ~HostExpertPinner();
    HostExpertPinner(const HostExpertPinner&) = delete;
    HostExpertPinner& operator=(const HostExpertPinner&) = delete;

    // Queues one projection (`total` = sum of its experts' bytes). The tensors' data pointers
    // move to the slab; they must not be read before this object is destroyed.
    void pin(std::vector<Tensor>& expert_vec, size_t total);

private:
    static constexpr int kThreads = 8;
    std::vector<PinnedBuffer>& sink_;
    std::mutex mu_;
    std::vector<PinnedBuffer> done_;
    std::optional<HostTaskPool> pool_;
};

}  // namespace imp
