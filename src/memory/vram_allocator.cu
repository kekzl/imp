#include "memory/vram_allocator.h"
#include "core/logging.h"

#include <algorithm>
#include <map>

namespace imp {

VRAMAllocator::~VRAMAllocator() {
    // Don't free tracked allocations here — they may already be freed
    // by their owners. The allocator is a tracker, not an owner.
}

bool VRAMAllocator::init(float headroom_fraction) {
    size_t free_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("VRAMAllocator: cudaMemGetInfo failed: %s", cudaGetErrorString(err));
        return false;
    }

    headroom_ = static_cast<size_t>(total_ * headroom_fraction);
    initialized_ = true;

    IMP_LOG_INFO("VRAMAllocator: total=%.0f MiB, headroom=%.0f MiB (%.0f%%)", total_ / (1024.0 * 1024.0),
                 headroom_ / (1024.0 * 1024.0), headroom_fraction * 100.0f);
    return true;
}

void* VRAMAllocator::allocate(size_t bytes, const char* tag, bool bypass_headroom) {
    if (bytes == 0)
        return nullptr;

    if (initialized_ && !bypass_headroom && !can_allocate(bytes)) {
        // Headroom check failed — but check if physical GPU memory suffices.
        // For models that use nearly all VRAM (e.g. Nemotron-30B at 29+ GiB),
        // the 10% headroom is too conservative. Critical allocations (workspace,
        // SSM state, dequant scratch) should still succeed if CUDA has memory.
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (free_mem >= bytes + (64 << 20)) {  // 64 MiB minimum safety
            IMP_LOG_WARN(
                "VRAMAllocator: %s (%.2f MiB) exceeds headroom, "
                "allowing (%.0f MiB GPU free)",
                tag, bytes / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN(
                "VRAMAllocator: rejecting %s allocation of %.2f MiB "
                "(%.0f MiB free, need %.0f MiB headroom)",
                tag, bytes / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0), headroom_ / (1024.0 * 1024.0));
            return nullptr;
        }
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("VRAMAllocator: cudaMalloc failed for %s (%.2f MiB): %s", tag,
                      bytes / (1024.0 * 1024.0), cudaGetErrorString(err));
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        alloc_map_[ptr] = {bytes, tag ? tag : "unknown"};
    }
    allocated_.fetch_add(bytes, std::memory_order_relaxed);

    return ptr;
}

void VRAMAllocator::free(void* ptr) {
    if (!ptr)
        return;

    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = alloc_map_.find(ptr);
        if (it != alloc_map_.end()) {
            bytes = it->second.bytes;
            alloc_map_.erase(it);
        }
    }

    IMP_CUDA_CHECK_LOG(cudaFree(ptr));

    if (bytes > 0) {
        allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    }
}

bool VRAMAllocator::can_allocate(size_t bytes) const {
    if (!initialized_)
        return true;

    // Small allocations (<16 MiB) always allowed — these are essential
    // runtime buffers (batch pool, penalty tokens, etc.) that must succeed.
    // Headroom enforcement targets large allocations (weight caches, KV cache).
    constexpr size_t kSmallAllocThreshold = 16ULL * 1024 * 1024;
    if (bytes < kSmallAllocThreshold)
        return true;

    // Check against actual free VRAM (not just our tracking)
    // to account for external allocations (driver, other processes).
    size_t free_mem = 0, total = 0;
    cudaMemGetInfo(&free_mem, &total);

    return free_mem >= bytes + headroom_;
}

size_t VRAMAllocator::available() const {
    if (!initialized_)
        return 0;

    size_t free_mem = 0, total = 0;
    cudaMemGetInfo(&free_mem, &total);

    return (free_mem > headroom_) ? (free_mem - headroom_) : 0;
}

void VRAMAllocator::report() const {
    std::lock_guard<std::mutex> lock(map_mutex_);

    // Aggregate by tag
    std::map<std::string, std::pair<size_t, int>> by_tag;  // tag -> (bytes, count)
    for (const auto& [ptr, alloc] : alloc_map_) {
        auto& entry = by_tag[alloc.tag];
        entry.first += alloc.bytes;
        entry.second++;
    }

    size_t tracked = allocated_.load(std::memory_order_relaxed);
    size_t free_mem = 0, total = 0;
    cudaMemGetInfo(&free_mem, &total);

    IMP_LOG_INFO(
        "VRAMAllocator report: tracked=%.0f MiB, free=%.0f MiB, "
        "headroom=%.0f MiB, total=%.0f MiB",
        tracked / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0), headroom_ / (1024.0 * 1024.0),
        total / (1024.0 * 1024.0));

    for (const auto& [tag, info] : by_tag) {
        IMP_LOG_INFO("  %-24s %8.1f MiB  (%d allocs)", tag.c_str(), info.first / (1024.0 * 1024.0),
                     info.second);
    }
}

}  // namespace imp
