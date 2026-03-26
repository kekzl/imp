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
    // sketch_dim: QJL sketch dimension (only used for TURBOQUANT / TURBOQUANT_LITE).
    //   TURBOQUANT: defaults to head_dim if sketch_dim <= 0.
    //   TURBOQUANT_LITE: should be multiplier * head_dim for quality (e.g. 2*head_dim).
    KVCache(int n_layers, int n_kv_heads, int head_dim, DType dtype,
            int max_blocks, int block_size = kKVBlockSize,
            VRAMAllocator* alloc = nullptr, int sketch_dim = 0);
    ~KVCache();

    // Block allocation / deallocation
    int allocate_block();
    void free_block(int block_id);

    // Reference counting (for copy-on-write / prefix caching)
    int ref_count(int block_id) const;
    void inc_ref(int block_id);

    // Pointer access into the contiguous pool
    // For TURBOQUANT_LITE, k_ptr returns nullptr (K stored only as sketches).
    void* k_ptr(int layer, int block_id);
    void* v_ptr(int layer, int block_id);

    // INT8/INT4/TURBOQUANT/TURBOQUANT_LITE per-head scale access (nullptr if not applicable)
    // For TURBOQUANT_LITE: K scales store FP16 norms, V scales store INT4 per-head scales.
    void* k_scale_ptr(int layer, int block_id);
    void* v_scale_ptr(int layer, int block_id);
    size_t scale_block_bytes() const;

    // TurboQuant/TurboQuant Lite QJL sketch access (nullptr if not applicable)
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
    int sketch_dim() const { return sketch_dim_; }
    DType dtype() const;

private:
    int n_layers_;
    int n_kv_heads_;
    int head_dim_;
    int max_blocks_;
    int block_size_;                // tokens per block (default 16)
    int sketch_dim_ = 0;           // QJL sketch dimension (0 if not TurboQuant)
    DType dtype_;
    VRAMAllocator* alloc_ = nullptr;
    size_t block_bytes_;            // cached: block_size * n_kv_heads * head_dim * dtype_size(dtype)

    std::vector<int> ref_counts_;   // per-block reference count
    std::vector<int> free_list_;
    void* pool_ = nullptr;          // single contiguous GPU allocation
                                    // For TURBOQUANT_LITE: V-only (no K directions in pool)

    // INT8/INT4/TURBOQUANT/TURBOQUANT_LITE per-head scales: one half per head per token slot.
    // Layout: 2x blocks per layer (K scales region + V scales region).
    // For TURBOQUANT: K scales store PolarQuant norms, V scales store INT4 per-head scales.
    // For TURBOQUANT_LITE: K scales store FP16 norms, V scales store INT4 per-head scales.
    void* scale_pool_ = nullptr;
    size_t scale_block_bytes_ = 0;  // block_size * n_kv_heads * sizeof(half)

    // TurboQuant / TurboQuant Lite QJL 1-bit sketch storage.
    // Layout: [layer, block_id] * sketch_block_bytes_ (K only, no V sketches).
    // For TURBOQUANT: sketch_dim = head_dim (error correction).
    // For TURBOQUANT_LITE: sketch_dim = multiplier * head_dim (primary K representation).
    void* sketch_pool_ = nullptr;
    size_t sketch_block_bytes_ = 0;  // block_size * n_kv_heads * (sketch_dim / 8)
};

} // namespace imp
