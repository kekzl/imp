#include "memory/kv_cache.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <cstdio>
#include <cstring>

namespace imp {

// ---------------------------------------------------------------------------
// Constructor: allocate one contiguous GPU buffer for all layers, all blocks,
// K+V slots.
//
// Memory layout (byte offsets) — standard modes (FP16/FP8/INT8/INT4/TURBOQUANT):
//   Per layer: K blocks contiguous, then V blocks contiguous.
//
//   K offset(layer, block_id) = (layer * 2 * max_blocks + block_id) * block_bytes_
//   V offset(layer, block_id) = (layer * 2 * max_blocks + max_blocks + block_id) * block_bytes_
//
//   Total = n_layers * max_blocks * 2 * block_bytes_
//
// Memory layout — TURBOQUANT_LITE:
//   Pool stores V blocks only (K is represented entirely via sketch pool + scale pool).
//   V offset(layer, block_id) = (layer * max_blocks + block_id) * block_bytes_
//   k_ptr() returns nullptr.
//
//   Total = n_layers * max_blocks * block_bytes_  (1x, not 2x)
// ---------------------------------------------------------------------------

KVCache::KVCache(int n_layers, int n_kv_heads, int head_dim, DType dtype,
                 int max_blocks, int block_size, VRAMAllocator* alloc, int sketch_dim,
                 bool use_mxfp4)
    : n_layers_(n_layers)
    , n_kv_heads_(n_kv_heads)
    , head_dim_(head_dim)
    , max_blocks_(max_blocks)
    , block_size_(block_size)
    , sketch_dim_(sketch_dim)
    , use_mxfp4_(use_mxfp4)
    , dtype_(dtype)
    , alloc_(alloc)
    , block_bytes_((dtype == DType::INT4 || dtype == DType::TURBOQUANT
                    || dtype == DType::TURBOQUANT_LITE)
                   ? (static_cast<size_t>(block_size_) * n_kv_heads * head_dim / 2)
                   : (static_cast<size_t>(block_size_) * n_kv_heads * head_dim *
                      dtype_size(dtype))) {

    bool lite = (dtype == DType::TURBOQUANT_LITE);

    // Resolve sketch_dim default: head_dim for TURBOQUANT, 0 for non-TQ modes
    if (dtype == DType::TURBOQUANT && sketch_dim_ <= 0) sketch_dim_ = head_dim;
    if (dtype == DType::TURBOQUANT_LITE && sketch_dim_ <= 0) sketch_dim_ = 2 * head_dim;

    // Allocate contiguous GPU pool
    // TURBOQUANT_LITE: V-only (1x), all others: K+V (2x)
    size_t pool_multiplier = lite ? 1 : 2;
    size_t total = static_cast<size_t>(n_layers_) * max_blocks_ * pool_multiplier * block_bytes_;
    if (alloc_) {
        pool_ = alloc_->allocate(total, "kv_cache");
    } else {
        cudaError_t err = cudaMalloc(&pool_, total);
        if (err != cudaSuccess) pool_ = nullptr;
    }
    if (!pool_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "KVCache: cudaMalloc failed for %.2f MiB (out of memory)",
                      static_cast<double>(total) / (1024.0 * 1024.0));
        throw std::runtime_error(msg);
    }

    // Zero-initialize the pool so fresh blocks start clean
    IMP_CUDA_CHECK_LOG(cudaMemset(pool_, 0, total));

    // Allocate separate scale buffer for INT8/INT4/TURBOQUANT/TURBOQUANT_LITE KV cache
    // For TURBOQUANT_LITE: K scales = FP16 norms, V scales = INT4 per-head scales
    if (dtype == DType::INT8 || dtype == DType::INT4
        || dtype == DType::TURBOQUANT || dtype == DType::TURBOQUANT_LITE) {
        scale_block_bytes_ = static_cast<size_t>(block_size_) * n_kv_heads * sizeof(half);
        // Always 2x: K scales region + V scales region (even for TURBOQUANT_LITE)
        size_t scale_total = static_cast<size_t>(n_layers_) * max_blocks_ * 2 * scale_block_bytes_;
        if (alloc_) {
            scale_pool_ = alloc_->allocate(scale_total, "kv_cache_scales");
        } else {
            cudaError_t serr = cudaMalloc(&scale_pool_, scale_total);
            if (serr != cudaSuccess) scale_pool_ = nullptr;
        }
        if (!scale_pool_) {
            if (alloc_) alloc_->free(pool_); else IMP_CUDA_CHECK_LOG(cudaFree(pool_));
            pool_ = nullptr;
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "KVCache: allocation failed for %s scale pool %.2f MiB",
                          dtype_name(dtype),
                          static_cast<double>(scale_total) / (1024.0 * 1024.0));
            throw std::runtime_error(msg);
        }
        IMP_CUDA_CHECK_LOG(cudaMemset(scale_pool_, 0, scale_total));
    }

    // Allocate QJL 1-bit sketch buffer for TurboQuant / TurboQuant Lite K-cache
    if (dtype == DType::TURBOQUANT || dtype == DType::TURBOQUANT_LITE) {
        sketch_block_bytes_ = static_cast<size_t>(block_size_) * n_kv_heads * (sketch_dim_ / 8);
        // Only K needs sketches, so n_layers * max_blocks * sketch_block_bytes (no 2x)
        size_t sketch_total = static_cast<size_t>(n_layers_) * max_blocks_ * sketch_block_bytes_;
        if (alloc_) {
            sketch_pool_ = alloc_->allocate(sketch_total, "kv_cache_sketches");
        } else {
            cudaError_t serr = cudaMalloc(&sketch_pool_, sketch_total);
            if (serr != cudaSuccess) sketch_pool_ = nullptr;
        }
        if (!sketch_pool_) {
            if (scale_pool_) { if (alloc_) alloc_->free(scale_pool_); else IMP_CUDA_CHECK_LOG(cudaFree(scale_pool_)); scale_pool_ = nullptr; }
            if (alloc_) alloc_->free(pool_); else IMP_CUDA_CHECK_LOG(cudaFree(pool_));
            pool_ = nullptr;
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "KVCache: allocation failed for %s sketch pool %.2f MiB",
                          dtype_name(dtype),
                          static_cast<double>(sketch_total) / (1024.0 * 1024.0));
            throw std::runtime_error(msg);
        }
        IMP_CUDA_CHECK_LOG(cudaMemset(sketch_pool_, 0, sketch_total));

        IMP_LOG_INFO("KVCache: %s sketch_dim=%d, sketch pool %.2f MiB (%d layers, %d blocks)",
                     dtype_name(dtype), sketch_dim_,
                     static_cast<double>(sketch_total) / (1024.0 * 1024.0),
                     n_layers_, max_blocks_);
    }

    // Allocate MXFP4 UE8M0 micro-scale pool for TurboQuant K directions (sm_120 path).
    // One UE8M0 byte per 32 direction elements per head per token.
    if (dtype == DType::TURBOQUANT && use_mxfp4_ && head_dim >= kTQMicroScaleGroup) {
        int n_groups_per_head = head_dim / kTQMicroScaleGroup;
        mscale_block_bytes_ = static_cast<size_t>(block_size_) * n_kv_heads * n_groups_per_head;
        // K-only (same indexing as sketch pool)
        size_t mscale_total = static_cast<size_t>(n_layers_) * max_blocks_ * mscale_block_bytes_;
        if (alloc_) {
            mscale_pool_ = alloc_->allocate(mscale_total, "kv_cache_mscales");
        } else {
            cudaError_t merr = cudaMalloc(&mscale_pool_, mscale_total);
            if (merr != cudaSuccess) mscale_pool_ = nullptr;
        }
        if (!mscale_pool_) {
            IMP_LOG_WARN("KVCache: MXFP4 micro-scale pool allocation failed (%.2f KiB), "
                         "falling back to uniform INT4 quantization",
                         static_cast<double>(mscale_total) / 1024.0);
            use_mxfp4_ = false;
        } else {
            IMP_CUDA_CHECK_LOG(cudaMemset(mscale_pool_, 0, mscale_total));
            IMP_LOG_INFO("KVCache: MXFP4 micro-scales enabled for K directions "
                         "(%d groups/head, %.1f KiB)",
                         n_groups_per_head,
                         static_cast<double>(mscale_total) / 1024.0);
        }
    }

    // Initialise per-block ref counts (0 = free) and build free list
    ref_counts_.resize(max_blocks_, 0);
    free_list_.reserve(max_blocks_);
    for (int i = max_blocks_ - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

KVCache::~KVCache() {
    if (mscale_pool_) {
        if (alloc_) alloc_->free(mscale_pool_); else IMP_CUDA_CHECK_LOG(cudaFree(mscale_pool_));
        mscale_pool_ = nullptr;
    }
    if (sketch_pool_) {
        if (alloc_) alloc_->free(sketch_pool_); else IMP_CUDA_CHECK_LOG(cudaFree(sketch_pool_));
        sketch_pool_ = nullptr;
    }
    if (scale_pool_) {
        if (alloc_) alloc_->free(scale_pool_); else IMP_CUDA_CHECK_LOG(cudaFree(scale_pool_));
        scale_pool_ = nullptr;
    }
    if (pool_) {
        if (alloc_) alloc_->free(pool_); else IMP_CUDA_CHECK_LOG(cudaFree(pool_));
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
    // TURBOQUANT_LITE: no K directions in pool, K is represented by sketches only
    if (dtype_ == DType::TURBOQUANT_LITE) return nullptr;

#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache k_ptr bounds violation: layer=%d/%d, block=%d/%d",
                      layer, n_layers_, block_id, max_blocks_);
    }
#endif
    // K blocks: [layer * 2 * max_blocks + block_id] * block_bytes
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ +
                     static_cast<size_t>(block_id)) * block_bytes_;
    return static_cast<char*>(pool_) + offset;
}

void* KVCache::v_ptr(int layer, int block_id) {
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache v_ptr bounds violation: layer=%d/%d, block=%d/%d",
                      layer, n_layers_, block_id, max_blocks_);
    }
#endif
    if (dtype_ == DType::TURBOQUANT_LITE) {
        // V-only pool: [layer * max_blocks + block_id] * block_bytes
        size_t offset = (static_cast<size_t>(layer) * max_blocks_ +
                         static_cast<size_t>(block_id)) * block_bytes_;
        return static_cast<char*>(pool_) + offset;
    }
    // Standard K+V pool: V blocks follow K blocks per layer
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

// ---------------------------------------------------------------------------
// Scale pointer computation (same layout for all quantized modes)
// ---------------------------------------------------------------------------

void* KVCache::k_scale_ptr(int layer, int block_id) {
    if (!scale_pool_) return nullptr;
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache k_scale_ptr bounds violation: layer=%d/%d, block=%d/%d",
                      layer, n_layers_, block_id, max_blocks_);
    }
#endif
    // Scale pool always uses 2x layout (K scales region + V scales region)
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ +
                     static_cast<size_t>(block_id)) * scale_block_bytes_;
    return static_cast<char*>(scale_pool_) + offset;
}

void* KVCache::v_scale_ptr(int layer, int block_id) {
    if (!scale_pool_) return nullptr;
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache v_scale_ptr bounds violation: layer=%d/%d, block=%d/%d",
                      layer, n_layers_, block_id, max_blocks_);
    }
#endif
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ +
                     max_blocks_ + static_cast<size_t>(block_id)) * scale_block_bytes_;
    return static_cast<char*>(scale_pool_) + offset;
}

size_t KVCache::scale_block_bytes() const {
    return scale_block_bytes_;
}

// ---------------------------------------------------------------------------
// TurboQuant / TurboQuant Lite QJL sketch pointer computation
// ---------------------------------------------------------------------------

void* KVCache::k_sketch_ptr(int layer, int block_id) {
    if (!sketch_pool_) return nullptr;
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache k_sketch_ptr bounds violation: layer=%d/%d, block=%d/%d",
                      layer, n_layers_, block_id, max_blocks_);
    }
#endif
    // Sketch pool layout: [layer, block_id] * sketch_block_bytes_ (K only, no V sketches)
    size_t offset = (static_cast<size_t>(layer) * max_blocks_ +
                     static_cast<size_t>(block_id)) * sketch_block_bytes_;
    return static_cast<char*>(sketch_pool_) + offset;
}

size_t KVCache::sketch_block_bytes() const {
    return sketch_block_bytes_;
}

// ---------------------------------------------------------------------------
// TurboQuant MXFP4 micro-scale pointer computation
// ---------------------------------------------------------------------------

void* KVCache::k_mscale_ptr(int layer, int block_id) {
    if (!mscale_pool_) return nullptr;
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache k_mscale_ptr bounds violation: layer=%d/%d, block=%d/%d",
                      layer, n_layers_, block_id, max_blocks_);
    }
#endif
    // Same indexing as sketch pool: [layer, block_id] * mscale_block_bytes_ (K only)
    size_t offset = (static_cast<size_t>(layer) * max_blocks_ +
                     static_cast<size_t>(block_id)) * mscale_block_bytes_;
    return static_cast<char*>(mscale_pool_) + offset;
}

size_t KVCache::mscale_block_bytes() const {
    return mscale_block_bytes_;
}

} // namespace imp
