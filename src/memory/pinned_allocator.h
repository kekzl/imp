#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace imp {

// Pinned (page-locked) host memory allocator for fast CPU<->GPU transfers.
//
// Pre-allocates a contiguous pool via cudaMallocHost and serves sub-allocations
// from it using a bump pointer with a per-size-class free list for recycled
// blocks.  When the pool is exhausted, falls back to direct cudaMallocHost
// calls.  All operations are mutex-protected.
class PinnedAllocator {
public:
    // If pool_size is 0 a default of 64 MiB is used.
    explicit PinnedAllocator(size_t pool_size = 0);
    ~PinnedAllocator();

    // Non-copyable, non-movable.
    PinnedAllocator(const PinnedAllocator&) = delete;
    PinnedAllocator& operator=(const PinnedAllocator&) = delete;
    PinnedAllocator(PinnedAllocator&&) = delete;
    PinnedAllocator& operator=(PinnedAllocator&&) = delete;

    // Allocate nbytes of pinned host memory (256-byte aligned).
    // Returns nullptr if nbytes == 0 or on failure.
    void* allocate(size_t nbytes);

    // Return a pointer previously obtained from allocate().
    void deallocate(void* ptr);

    // Total pool capacity in bytes.
    size_t pool_size() const;

    // Number of bytes currently in use (pool + fallback).
    size_t used() const;

private:
    static constexpr size_t kDefaultPoolSize = 64ULL * 1024 * 1024;  // 64 MiB
    static constexpr size_t kMinBlockSize    = 256;                   // smallest allocation unit
    static constexpr size_t kAlignment       = 256;

    // Round nbytes up to the next power of two, at least kMinBlockSize.
    static size_t round_up_size(size_t nbytes);

    // Try to allocate from the free list for the given size class.
    void* alloc_from_free_list(size_t size_class);

    // Try to allocate by bumping the pool offset.
    void* alloc_from_bump(size_t size_class);

    // Fall back to a direct cudaMallocHost allocation.
    void* alloc_fallback(size_t nbytes);

    mutable std::mutex mu_;

    // --- Pool state ---
    void*  pool_base_   = nullptr;   // base address of the cudaMallocHost pool
    size_t pool_size_   = 0;         // total pool capacity in bytes
    size_t pool_offset_ = 0;         // bump-pointer offset into the pool

    // Per-size-class free lists (size_class -> list of recycled pointers).
    std::unordered_map<size_t, std::vector<void*>> free_lists_;

    // Maps every outstanding allocation pointer to its effective size (the
    // rounded-up size class for pool allocs, the raw size for fallback allocs).
    std::unordered_map<void*, size_t> alloc_map_;

    // --- Fallback allocations (outside the pool) ---
    std::unordered_set<void*> fallback_allocs_;

    // --- Statistics ---
    size_t used_ = 0;  // currently outstanding bytes
};

} // namespace imp
