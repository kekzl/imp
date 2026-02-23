#include "memory/pinned_allocator.h"
#include "core/logging.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

namespace imp {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

size_t PinnedAllocator::round_up_size(size_t nbytes) {
    if (nbytes <= kMinBlockSize) return kMinBlockSize;

    // Next power of two.
    size_t v = nbytes - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

PinnedAllocator::PinnedAllocator(size_t pool_size) {
    pool_size_ = (pool_size == 0) ? kDefaultPoolSize : pool_size;

    cudaError_t err = cudaMallocHost(&pool_base_, pool_size_);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("PinnedAllocator: cudaMallocHost(%zu) failed (%s), "
                     "running without a pool",
                     pool_size_, cudaGetErrorString(err));
        pool_base_ = nullptr;
        pool_size_ = 0;
    } else {
        IMP_LOG_INFO("PinnedAllocator: allocated %zu byte pinned pool at %p",
                     pool_size_, pool_base_);
    }
}

PinnedAllocator::~PinnedAllocator() {
    // Free every outstanding fallback allocation.
    for (void* ptr : fallback_allocs_) {
        cudaFreeHost(ptr);
    }
    fallback_allocs_.clear();

    // Free the pool itself.
    if (pool_base_) {
        cudaFreeHost(pool_base_);
        pool_base_ = nullptr;
    }

    if (used_ != 0) {
        IMP_LOG_WARN("PinnedAllocator: destroyed with %zu bytes still in use",
                     used_);
    }
}

// ---------------------------------------------------------------------------
// Pool internals
// ---------------------------------------------------------------------------

void* PinnedAllocator::alloc_from_free_list(size_t size_class) {
    auto it = free_lists_.find(size_class);
    if (it == free_lists_.end() || it->second.empty()) return nullptr;

    void* ptr = it->second.back();
    it->second.pop_back();
    return ptr;
}

void* PinnedAllocator::alloc_from_bump(size_t size_class) {
    if (!pool_base_) return nullptr;

    // Align the current offset up to kAlignment.
    size_t aligned_offset = (pool_offset_ + kAlignment - 1) & ~(kAlignment - 1);
    if (aligned_offset + size_class > pool_size_) return nullptr;

    void* ptr = static_cast<uint8_t*>(pool_base_) + aligned_offset;
    pool_offset_ = aligned_offset + size_class;
    return ptr;
}

void* PinnedAllocator::alloc_fallback(size_t nbytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, nbytes);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("PinnedAllocator: fallback cudaMallocHost(%zu) failed (%s)",
                      nbytes, cudaGetErrorString(err));
        return nullptr;
    }

    fallback_allocs_.insert(ptr);
    IMP_LOG_DEBUG("PinnedAllocator: fallback allocation of %zu bytes at %p",
                  nbytes, ptr);
    return ptr;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void* PinnedAllocator::allocate(size_t nbytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (nbytes == 0) return nullptr;

    size_t size_class = round_up_size(nbytes);

    // 1. Try the free list for this size class.
    void* ptr = alloc_from_free_list(size_class);

    // 2. Try bumping the pool pointer.
    if (!ptr) {
        ptr = alloc_from_bump(size_class);
    }

    // 3. Fall back to a direct cudaMallocHost.
    if (!ptr) {
        ptr = alloc_fallback(size_class);
        if (!ptr) return nullptr;
    }

    alloc_map_[ptr] = size_class;
    used_ += size_class;
    return ptr;
}

void PinnedAllocator::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ptr) return;

    auto it = alloc_map_.find(ptr);
    if (it == alloc_map_.end()) {
        IMP_LOG_ERROR("PinnedAllocator: deallocate called on unknown pointer %p",
                      ptr);
        return;
    }

    size_t size_class = it->second;
    alloc_map_.erase(it);
    used_ -= size_class;

    // Check whether this pointer falls within the pool range.
    bool in_pool = pool_base_ &&
                   ptr >= pool_base_ &&
                   ptr < static_cast<uint8_t*>(pool_base_) + pool_size_;

    if (in_pool) {
        // Return to the free list for reuse.
        free_lists_[size_class].push_back(ptr);
    } else {
        // Fallback allocation -- release to the OS immediately.
        fallback_allocs_.erase(ptr);
        cudaFreeHost(ptr);
    }
}

size_t PinnedAllocator::pool_size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return pool_size_;
}

size_t PinnedAllocator::used() const {
    std::lock_guard<std::mutex> lock(mu_);
    return used_;
}

} // namespace imp
