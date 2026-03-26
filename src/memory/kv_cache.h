#pragma once

#include "core/tensor.h"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace imp {

class VRAMAllocator;  // forward declaration

static constexpr int kKVBlockSize = 16; // default tokens per block

class KVCache {
public:
    KVCache(int n_layers, int n_kv_heads, int head_dim, DType dtype,
            int max_blocks, int block_size = kKVBlockSize,
            VRAMAllocator* alloc = nullptr);
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

    // INT8/INT4/TURBOQUANT per-head scale access (nullptr if not applicable)
    void* k_scale_ptr(int layer, int block_id);
    void* v_scale_ptr(int layer, int block_id);
    size_t scale_block_bytes() const;

    // TurboQuant QJL sketch access (nullptr if dtype != TURBOQUANT)
    void* k_sketch_ptr(int layer, int block_id);
    size_t sketch_block_bytes() const;

    // Capacity queries
    int num_free_blocks() const;
    int total_blocks() const;

    // Accessors
    size_t block_bytes() const;
    int block_size() const { return block_size_; }
    int n_layers() const;
    int n_kv_heads() const;
    int head_dim() const;
    DType dtype() const;

private:
    int n_layers_;
    int n_kv_heads_;
    int head_dim_;
    int max_blocks_;
    int block_size_;                // tokens per block (default 16)
    DType dtype_;
    VRAMAllocator* alloc_ = nullptr;
    size_t block_bytes_;            // cached: block_size * n_kv_heads * head_dim * dtype_size(dtype)

    std::vector<int> ref_counts_;   // per-block reference count
    std::vector<int> free_list_;
    void* pool_ = nullptr;          // single contiguous GPU allocation

    // INT8/INT4/TURBOQUANT per-head scales: one half per head per token slot.
    // Layout mirrors pool_ but with scale_block_bytes_ per block.
    // For TURBOQUANT: K scales store PolarQuant norms, V scales store INT4 per-head scales.
    void* scale_pool_ = nullptr;
    size_t scale_block_bytes_ = 0;  // block_size * n_kv_heads * sizeof(half)

    // TurboQuant QJL 1-bit sketch storage.
    // Layout mirrors K portion of pool_ but with sketch_block_bytes_ per block.
    // Only allocated when dtype == TURBOQUANT.
    void* sketch_pool_ = nullptr;
    size_t sketch_block_bytes_ = 0;  // block_size * n_kv_heads * (head_dim / 8)
};

} // namespace imp
