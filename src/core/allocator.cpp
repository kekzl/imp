#include "core/allocator.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

namespace imp {

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

static size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

// --- ArenaAllocator ---

ArenaAllocator::ArenaAllocator(size_t capacity, bool on_device)
    : capacity_(capacity), on_device_(on_device) {
    if (capacity == 0) return;
    if (on_device) {
        check_cuda(cudaMalloc(&base_, capacity), "ArenaAllocator cudaMalloc");
    } else {
        base_ = std::malloc(capacity);
        if (!base_) throw std::bad_alloc();
    }
}

ArenaAllocator::~ArenaAllocator() {
    if (!base_) return;
    if (on_device_) {
        cudaFree(base_);
    } else {
        std::free(base_);
    }
}

void* ArenaAllocator::allocate(size_t nbytes, size_t alignment) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t aligned = align_up(offset_, alignment);
    if (aligned + nbytes > capacity_) {
        return nullptr;
    }
    void* ptr = static_cast<char*>(base_) + aligned;
    offset_ = aligned + nbytes;
    return ptr;
}

void ArenaAllocator::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    offset_ = 0;
}

// --- PoolAllocator ---

PoolAllocator::PoolAllocator(size_t block_size, size_t num_blocks, bool on_device)
    : block_size_(block_size), num_blocks_(num_blocks), on_device_(on_device) {
    if (num_blocks == 0) return;
    size_t total = block_size * num_blocks;
    if (on_device) {
        check_cuda(cudaMalloc(&base_, total), "PoolAllocator cudaMalloc");
    } else {
        base_ = std::malloc(total);
        if (!base_) throw std::bad_alloc();
    }
    free_list_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        free_list_.push_back(static_cast<char*>(base_) + i * block_size);
    }
}

PoolAllocator::~PoolAllocator() {
    if (!base_) return;
    if (on_device_) {
        cudaFree(base_);
    } else {
        std::free(base_);
    }
}

void* PoolAllocator::allocate() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_list_.empty()) return nullptr;
    void* ptr = free_list_.back();
    free_list_.pop_back();
    return ptr;
}

void PoolAllocator::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    free_list_.push_back(ptr);
}

size_t PoolAllocator::free_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return free_list_.size();
}

} // namespace imp
