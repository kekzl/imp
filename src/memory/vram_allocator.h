#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace imp {

// Centralized GPU memory allocator enforcing a VRAM headroom budget. All GPU memory
// allocations should go through this class to prevent silent spill into shared/system
// memory on WSL2 (not visible via nvidia-smi).
class VRAMAllocator {
public:
    VRAMAllocator() = default;
    ~VRAMAllocator();

    // Non-copyable, non-movable.
    VRAMAllocator(const VRAMAllocator&) = delete;
    VRAMAllocator& operator=(const VRAMAllocator&) = delete;

    // Initialize: query total VRAM and set headroom fraction.
    // Must be called before any allocations.
    [[nodiscard]] bool init(float headroom_fraction = 0.10f);

    // Allocate device memory; returns nullptr if it would violate headroom (unless
    // bypass_headroom) or cudaMalloc fails. tag: descriptive name for reporting.
    // bypass_headroom: skip the cudaMemGetInfo pre-check, only for paths self-tracking a
    // logical budget (e.g. NVFP4 MoE cache, where cudaMemGetInfo doesn't reflect upload-time
    // per-expert frees in time).
    void* allocate(size_t bytes, const char* tag, bool bypass_headroom = false);

    // Free a pointer previously returned by allocate().
    void free(void* ptr);

    // Query: can we allocate `bytes` without violating headroom?
    bool can_allocate(size_t bytes) const;

    // Total physical VRAM on the device.
    size_t total_vram() const { return total_; }

    // Bytes currently tracked as allocated through this allocator.
    size_t allocated() const { return allocated_.load(std::memory_order_relaxed); }

    // Bytes available for allocation (total - allocated - headroom - external).
    // 'external' = VRAM used by other processes / driver overhead.
    size_t available() const;

    // Reserved headroom bytes (not available for allocation).
    size_t headroom() const { return headroom_; }

    // Log a summary of all allocations grouped by tag.
    void report() const;

private:
    size_t total_ = 0;
    size_t headroom_ = 0;
    std::atomic<size_t> allocated_{0};

    struct Allocation {
        size_t bytes;
        std::string tag;
    };
    mutable std::mutex map_mutex_;
    std::unordered_map<void*, Allocation> alloc_map_;
    bool initialized_ = false;
};

}  // namespace imp
