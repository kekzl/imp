#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <mutex>

namespace imp {

// Arena allocator: pre-allocates a large chunk of GPU or CPU memory and
// bump-allocates from it. Only supports bulk reset (free everything at once).
class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacity, bool on_device = true);
    ~ArenaAllocator();

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Allocate nbytes with given alignment. Returns nullptr if out of space.
    void* allocate(size_t nbytes, size_t alignment = 256);

    // Reset offset to 0 (logically frees all allocations)
    void reset();

    size_t used() const noexcept { return offset_; }
    size_t capacity() const noexcept { return capacity_; }
    size_t remaining() const noexcept { return capacity_ - offset_; }
    bool on_device() const noexcept { return on_device_; }

private:
    void* base_      = nullptr;
    size_t capacity_ = 0;
    size_t offset_   = 0;
    bool on_device_  = false;
    mutable std::mutex mu_;
};

// Pool allocator: manages fixed-size blocks. Suitable for KV cache blocks
// or other uniform-sized allocations.
class PoolAllocator {
public:
    PoolAllocator(size_t block_size, size_t num_blocks, bool on_device = true);
    ~PoolAllocator();

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    // Allocate one block. Returns nullptr if pool is exhausted.
    void* allocate();

    // Return a block to the pool.
    void deallocate(void* ptr);

    size_t block_size() const noexcept { return block_size_; }
    size_t num_blocks() const noexcept { return num_blocks_; }
    size_t free_count() const;
    bool on_device() const noexcept { return on_device_; }

private:
    void* base_         = nullptr;
    size_t block_size_  = 0;
    size_t num_blocks_  = 0;
    bool on_device_     = false;
    std::vector<void*> free_list_;
    mutable std::mutex mu_;
};

} // namespace imp
