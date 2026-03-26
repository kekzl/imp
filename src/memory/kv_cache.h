#pragma once

#include "core/tensor.h"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace imp {

class VRAMAllocator;  // forward declaration

static constexpr int kKVBlockSize = 16; // default tokens per block

// MXFP4 micro-scale group size (matching CUTLASS MXFP4 SfVecSize)
static constexpr int kTQMicroScaleGroup = 32;

class KVCache {
public:
    // sketch_dim: QJL sketch dimension (only used for TURBOQUANT / TURBOQUANT_LITE).
    //   TURBOQUANT: defaults to head_dim if sketch_dim <= 0.
    //   TURBOQUANT_LITE: should be multiplier * head_dim for quality (e.g. 2*head_dim).
    // use_mxfp4: if true and dtype==TURBOQUANT, use FP4 E2M1 + UE8M0 micro-scales
    //   instead of uniform INT4 for K direction quantization (sm_120 path).
    KVCache(int n_layers, int n_kv_heads, int head_dim, DType dtype,
            int max_blocks, int block_size = kKVBlockSize,
            VRAMAllocator* alloc = nullptr, int sketch_dim = 0,
            bool use_mxfp4 = false);
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
    // For TURBOQUANT: K scales store PolarQuant FP16 norms, V scales store INT4 per-head scales.
    // For TURBOQUANT_LITE: K scales store FP16 norms, V scales store INT4 per-head scales.
    void* k_scale_ptr(int layer, int block_id);
    void* v_scale_ptr(int layer, int block_id);
    size_t scale_block_bytes() const;

    // TurboQuant/TurboQuant Lite QJL sketch access (nullptr if not applicable)
    void* k_sketch_ptr(int layer, int block_id);
    size_t sketch_block_bytes() const;

    // TurboQuant MXFP4 per-32-element UE8M0 micro-scale access for K directions.
    // nullptr if not TURBOQUANT or mxfp4 not enabled. K-only (no V micro-scales).
    // Layout: [layer, block_id] * mscale_block_bytes_ (same indexing as sketch_pool_).
    void* k_mscale_ptr(int layer, int block_id);
    size_t mscale_block_bytes() const;

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
    bool use_mxfp4() const { return use_mxfp4_; }
    DType dtype() const;

private:
    int n_layers_;
    int n_kv_heads_;
    int head_dim_;
    int max_blocks_;
    int block_size_;                // tokens per block (default 16)
    int sketch_dim_ = 0;           // QJL sketch dimension (0 if not TurboQuant)
    bool use_mxfp4_ = false;       // FP4 E2M1 + UE8M0 micro-scales for K dirs (sm_120)
    DType dtype_;
    VRAMAllocator* alloc_ = nullptr;
    size_t block_bytes_;            // cached: block_size * n_kv_heads * head_dim * dtype_size(dtype)

    std::vector<int> ref_counts_;   // per-block reference count
    std::vector<int> free_list_;
    void* pool_ = nullptr;          // single contiguous GPU allocation
                                    // For TURBOQUANT_LITE: V-only (no K directions in pool)

    // INT8/INT4/TURBOQUANT/TURBOQUANT_LITE per-head scales: one half per head per token slot.
    // Layout: 2x blocks per layer (K scales region + V scales region).
    // For TURBOQUANT: K scales store PolarQuant norms, V scales store INT4/FP4 per-head scales.
    // For TURBOQUANT_LITE: K scales store FP16 norms, V scales store INT4 per-head scales.
    void* scale_pool_ = nullptr;
    size_t scale_block_bytes_ = 0;  // block_size * n_kv_heads * sizeof(half)

    // TurboQuant / TurboQuant Lite QJL 1-bit sketch storage.
    // Layout: [layer, block_id] * sketch_block_bytes_ (K only, no V sketches).
    void* sketch_pool_ = nullptr;
    size_t sketch_block_bytes_ = 0;  // block_size * n_kv_heads * (sketch_dim / 8)

    // TurboQuant MXFP4 per-32-element UE8M0 micro-scales for K direction vectors.
    // Only allocated when dtype == TURBOQUANT && use_mxfp4_ == true.
    // Layout: [layer, block_id] * mscale_block_bytes_ (K only, same as sketch indexing).
    // Each token-head stores head_dim/32 UE8M0 bytes.
    void* mscale_pool_ = nullptr;
    size_t mscale_block_bytes_ = 0;  // block_size * n_kv_heads * (head_dim / kTQMicroScaleGroup)
};

} // namespace imp
