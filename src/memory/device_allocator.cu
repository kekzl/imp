#include "memory/device_allocator.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Internal helper: check a CUDA call and throw on failure.
// ---------------------------------------------------------------------------
#define IMP_CUDA_CHECK(call)                                                  \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            char buf[512];                                                    \
            std::snprintf(buf, sizeof(buf),                                   \
                          "CUDA error in %s at %s:%d – %s (%s)",             \
                          #call, __FILE__, __LINE__,                          \
                          cudaGetErrorString(err_),                           \
                          cudaGetErrorName(err_));                            \
            throw std::runtime_error(buf);                                    \
        }                                                                     \
    } while (0)

namespace imp {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
DeviceAllocator::DeviceAllocator(size_t initial_pool_size, int device)
    : device_(device) {
    // Ensure the requested device is active so that pool creation targets it.
    IMP_CUDA_CHECK(cudaSetDevice(device_));

    // Create a memory pool on the specified device.
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = device_;

    IMP_CUDA_CHECK(cudaMemPoolCreate(&pool_, &pool_props));

    // Set the release threshold.  When initial_pool_size is 0 we use
    // UINT64_MAX so the pool never voluntarily returns memory to the OS;
    // this maximises reuse at the expense of holding physical memory.
    uint64_t threshold = (initial_pool_size == 0)
                             ? UINT64_MAX
                             : static_cast<uint64_t>(initial_pool_size);
    IMP_CUDA_CHECK(cudaMemPoolSetAttribute(
        pool_, cudaMemPoolAttrReleaseThreshold, &threshold));

    // Allow the pool to opportunistically reuse memory that has been freed
    // but whose stream-ordering dependencies have already been satisfied.
    int enable = 1;
    IMP_CUDA_CHECK(cudaMemPoolSetAttribute(
        pool_, cudaMemPoolReuseAllowOpportunistic, &enable));

    // Allow the pool to insert internal dependencies (events) between streams
    // so it can reuse memory across streams when safe.
    IMP_CUDA_CHECK(cudaMemPoolSetAttribute(
        pool_, cudaMemPoolReuseAllowInternalDependencies, &enable));
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
DeviceAllocator::~DeviceAllocator() {
    if (pool_) {
        // cudaMemPoolDestroy marks the pool for destruction.  Outstanding
        // allocations keep it alive until they are freed.
        cudaMemPoolDestroy(pool_);
        pool_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// allocate
// ---------------------------------------------------------------------------
void* DeviceAllocator::allocate(size_t nbytes, cudaStream_t stream) {
    if (nbytes == 0) return nullptr;

    void* ptr = nullptr;
    IMP_CUDA_CHECK(cudaMallocAsync(&ptr, nbytes, pool_, stream));

    // Record the mapping so we can look up the size on deallocation.
    {
        std::lock_guard<std::mutex> lock(alloc_map_mutex_);
        alloc_map_[ptr] = nbytes;
    }

    // Update the running total and the high-water mark.
    size_t current = allocated_.fetch_add(nbytes, std::memory_order_relaxed) + nbytes;

    // Atomically update peak_allocated_ if current exceeds it.
    size_t prev_peak = peak_allocated_.load(std::memory_order_relaxed);
    while (current > prev_peak &&
           !peak_allocated_.compare_exchange_weak(
               prev_peak, current, std::memory_order_relaxed)) {
        // prev_peak is reloaded by compare_exchange_weak on failure.
    }

    return ptr;
}

// ---------------------------------------------------------------------------
// deallocate
// ---------------------------------------------------------------------------
void DeviceAllocator::deallocate(void* ptr, cudaStream_t stream) {
    if (!ptr) return;

    size_t nbytes = 0;
    {
        std::lock_guard<std::mutex> lock(alloc_map_mutex_);
        auto it = alloc_map_.find(ptr);
        if (it != alloc_map_.end()) {
            nbytes = it->second;
            alloc_map_.erase(it);
        }
        // If the pointer was not found we still free it but cannot update the
        // counter – this avoids a hard failure on double-free or foreign ptrs.
    }

    IMP_CUDA_CHECK(cudaFreeAsync(ptr, stream));

    if (nbytes > 0) {
        allocated_.fetch_sub(nbytes, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------
size_t DeviceAllocator::allocated() const {
    return allocated_.load(std::memory_order_relaxed);
}

size_t DeviceAllocator::peak_allocated() const {
    return peak_allocated_.load(std::memory_order_relaxed);
}

void DeviceAllocator::reset_peak_stats() {
    peak_allocated_.store(allocated_.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// trim
// ---------------------------------------------------------------------------
void DeviceAllocator::trim(size_t min_bytes_to_keep) {
    IMP_CUDA_CHECK(cudaMemPoolTrimTo(pool_, min_bytes_to_keep));
}

} // namespace imp
