#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>

namespace imp {

// Wraps cudaMallocAsync memory pool for fast async allocations.
// All pool operations go through CUDA's stream-ordered memory allocator,
// enabling overlap of allocation/deallocation with kernel execution.
class DeviceAllocator {
public:
    explicit DeviceAllocator(size_t initial_pool_size = 0, int device = 0);
    ~DeviceAllocator();

    // Non-copyable, non-movable.
    DeviceAllocator(const DeviceAllocator&) = delete;
    DeviceAllocator& operator=(const DeviceAllocator&) = delete;
    DeviceAllocator(DeviceAllocator&&) = delete;
    DeviceAllocator& operator=(DeviceAllocator&&) = delete;

    // Allocate nbytes of device memory on the given stream.
    // Returns nullptr if nbytes == 0 or on failure.
    void* allocate(size_t nbytes, cudaStream_t stream = nullptr);

    // Free a pointer previously returned by allocate().
    // The deallocation is stream-ordered: the memory may be reused by later
    // work on the same stream without explicit synchronization.
    void deallocate(void* ptr, cudaStream_t stream = nullptr);

    // Current number of bytes logically allocated (not yet freed).
    size_t allocated() const;

    // High-water mark of allocated bytes since construction or the last
    // call to reset_peak_stats().
    size_t peak_allocated() const;

    // Reset the peak-allocation counter to the current allocation level.
    void reset_peak_stats();

    // Release unused physical memory held by the pool back to the OS,
    // retaining at least min_bytes_to_keep bytes.
    void trim(size_t min_bytes_to_keep = 0);

private:
    cudaMemPool_t pool_ = nullptr;
    int device_ = 0;

    // Allocation tracking – atomic for lock-free reads from allocated()/peak.
    std::atomic<size_t> allocated_{0};
    std::atomic<size_t> peak_allocated_{0};

    // Map from pointer -> allocation size so we can accurately decrement
    // allocated_ on deallocation.  Protected by alloc_map_mutex_.
    std::mutex alloc_map_mutex_;
    std::unordered_map<void*, size_t> alloc_map_;
};

} // namespace imp
