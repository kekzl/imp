#pragma once

#include "core/tensor.h"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace imp {

static constexpr int kKVBlockSize = 16; // tokens per block

class KVCache {
public:
    KVCache(int n_layers, int n_kv_heads, int head_dim, DType dtype,
            int max_blocks);
    ~KVCache();

    // Block allocation / deallocation
    int allocate_block();
    void free_block(int block_id);

    // Reference counting (for copy-on-write / prefix caching)
    int ref_count(int block_id) const;
    void inc_ref(int block_id);

    // Pointer access into the contiguous pool
    void* k_ptr(int layer, int block_id);
    void* v_ptr(int layer, int block_id);

    // INT8 per-head scale access (nullptr if dtype != INT8)
    void* k_scale_ptr(int layer, int block_id);
    void* v_scale_ptr(int layer, int block_id);
    size_t scale_block_bytes() const;

    // Capacity queries
    int num_free_blocks() const;
    int total_blocks() const;

    // Accessors
    size_t block_bytes() const;
    int n_layers() const;
    int n_kv_heads() const;
    int head_dim() const;
    DType dtype() const;

private:
    int n_layers_;
    int n_kv_heads_;
    int head_dim_;
    int max_blocks_;
    DType dtype_;
    size_t block_bytes_;            // cached: kKVBlockSize * n_kv_heads * head_dim * dtype_size(dtype)

    std::vector<int> ref_counts_;   // per-block reference count
    std::vector<int> free_list_;
    void* pool_ = nullptr;          // single contiguous GPU allocation

    // INT8 per-head scales: one half per head per token slot.
    // Layout mirrors pool_ but with scale_block_bytes_ per block.
    void* scale_pool_ = nullptr;
    size_t scale_block_bytes_ = 0;  // kKVBlockSize * n_kv_heads * sizeof(half)
};

} // namespace imp
