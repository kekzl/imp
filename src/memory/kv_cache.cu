#include "memory/kv_cache.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>
#include <cstring>

namespace imp {

// ---------------------------------------------------------------------------
// Constructor: allocate one contiguous GPU buffer for all layers, all blocks,
// K+V slots.
//
// Memory layout (byte offsets):
//   Per layer: K blocks contiguous, then V blocks contiguous.
//
//   K offset(layer, block_id) = (layer * 2 * max_blocks + block_id) * block_bytes_
//   V offset(layer, block_id) = (layer * 2 * max_blocks + max_blocks + block_id) * block_bytes_
//
// This ensures K (and V) blocks are contiguous within a layer, allowing
// kernels to stride by block_bytes_ between physical blocks.
//
// Total = n_layers * max_blocks * 2 * block_bytes_
// ---------------------------------------------------------------------------

KVCache::KVCache(int n_layers, int n_kv_heads, int head_dim, DType dtype,
                 int max_blocks)
    : n_layers_(n_layers)
    , n_kv_heads_(n_kv_heads)
    , head_dim_(head_dim)
    , max_blocks_(max_blocks)
    , dtype_(dtype)
    , block_bytes_(static_cast<size_t>(kKVBlockSize) * n_kv_heads * head_dim *
                   dtype_size(dtype)) {

    // Allocate contiguous GPU pool
    size_t total = static_cast<size_t>(n_layers_) * max_blocks_ * 2 * block_bytes_;
    cudaError_t err = cudaMalloc(&pool_, total);
    if (err != cudaSuccess) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "KVCache: cudaMalloc failed for %.2f MiB (%s)",
                      static_cast<double>(total) / (1024.0 * 1024.0),
                      cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }

    // Zero-initialize the pool so fresh blocks start clean
    cudaMemset(pool_, 0, total);

    // Initialise per-block ref counts (0 = free) and build free list
    ref_counts_.resize(max_blocks_, 0);
    free_list_.reserve(max_blocks_);
    for (int i = max_blocks_ - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

KVCache::~KVCache() {
    if (pool_) {
        cudaFree(pool_);
        pool_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Block allocation
// ---------------------------------------------------------------------------

int KVCache::allocate_block() {
    if (free_list_.empty()) {
        return -1;
    }

    int block_id = free_list_.back();
    free_list_.pop_back();
    ref_counts_[block_id] = 1;
    return block_id;
}

void KVCache::free_block(int block_id) {
    if (block_id < 0 || block_id >= max_blocks_) return;
    if (ref_counts_[block_id] <= 0) return;

    --ref_counts_[block_id];
    if (ref_counts_[block_id] == 0) {
        free_list_.push_back(block_id);
    }
}

// ---------------------------------------------------------------------------
// Reference counting
// ---------------------------------------------------------------------------

int KVCache::ref_count(int block_id) const {
    if (block_id < 0 || block_id >= max_blocks_) return 0;
    return ref_counts_[block_id];
}

void KVCache::inc_ref(int block_id) {
    if (block_id < 0 || block_id >= max_blocks_) return;
    ++ref_counts_[block_id];
}

// ---------------------------------------------------------------------------
// Pointer computation into the contiguous pool
// ---------------------------------------------------------------------------

void* KVCache::k_ptr(int layer, int block_id) {
    // K blocks: [layer * 2 * max_blocks + block_id] * block_bytes
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ +
                     static_cast<size_t>(block_id)) * block_bytes_;
    return static_cast<char*>(pool_) + offset;
}

void* KVCache::v_ptr(int layer, int block_id) {
    // V blocks: [layer * 2 * max_blocks + max_blocks + block_id] * block_bytes
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ +
                     max_blocks_ + static_cast<size_t>(block_id)) * block_bytes_;
    return static_cast<char*>(pool_) + offset;
}

// ---------------------------------------------------------------------------
// Capacity queries
// ---------------------------------------------------------------------------

int KVCache::num_free_blocks() const {
    return static_cast<int>(free_list_.size());
}

int KVCache::total_blocks() const {
    return max_blocks_;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

size_t KVCache::block_bytes() const {
    return block_bytes_;
}

int KVCache::n_layers() const {
    return n_layers_;
}

int KVCache::n_kv_heads() const {
    return n_kv_heads_;
}

int KVCache::head_dim() const {
    return head_dim_;
}

DType KVCache::dtype() const {
    return dtype_;
}

} // namespace imp
