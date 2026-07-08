#include "memory/kv_cache.h"
#include "memory/vram_allocator.h"
#include "memory/mem_account.h"
#include "runtime/graph_diag.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
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
//   Total = n_layers * max_blocks * 2 * block_bytes_
// ---------------------------------------------------------------------------

KVCache::KVCache(int n_layers, int n_kv_heads, int head_dim, QType dtype, int max_blocks, int block_size,
                 VRAMAllocator* alloc)
    : n_layers_(n_layers),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      max_blocks_(max_blocks),
      block_size_(block_size),
      dtype_(dtype),
      alloc_(alloc),
      block_bytes_((dtype == QType::INT4 || dtype == QType::NVFP4 || dtype == QType::MXFP4_KV)
                       ? (static_cast<size_t>(block_size_) * n_kv_heads * head_dim / 2)
                       : (static_cast<size_t>(block_size_) * n_kv_heads * head_dim * dtype_size(dtype))) {

    // Allocate contiguous GPU pool — K+V (2x)
    size_t total = static_cast<size_t>(n_layers_) * max_blocks_ * 2 * block_bytes_;
    if (alloc_) {
        pool_ = alloc_->allocate(total, "kv_cache");
    } else {
        cudaError_t err = cudaMalloc(&pool_, total);
        if (err != cudaSuccess)
            pool_ = nullptr;
    }
    if (!pool_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "KVCache: cudaMalloc failed for %.2f MiB (out of memory)",
                      static_cast<double>(total) / (1024.0 * 1024.0));
        throw std::runtime_error(msg);
    }

    // Zero-initialize the pool so fresh blocks start clean
    IMP_CUDA_CHECK_LOG(cudaMemset(pool_, 0, total));
    MemAccount::instance().note("KV_BLOCK_POOL", static_cast<std::ptrdiff_t>(total));

    // Allocate separate scale buffer for quantized KV cache modes.
    //   INT8/INT4: 1 half (FP16) per head per token slot.
    //   NVFP4/MXFP4_KV: 1 scale byte per kNVFP4Group=16 FP4 elems along head_dim.
    bool needs_scales = (dtype == QType::INT8 || dtype == QType::INT4 || dtype == QType::NVFP4 ||
                         dtype == QType::MXFP4_KV);
    if (needs_scales) {
        if (dtype == QType::NVFP4 || dtype == QType::MXFP4_KV) {
            if (head_dim % kNVFP4Group != 0) {
                throw std::runtime_error("KVCache NVFP4: head_dim must be a multiple of 16");
            }
            int n_groups_per_head = head_dim / kNVFP4Group;
            scale_block_bytes_ = static_cast<size_t>(block_size_) * n_kv_heads * n_groups_per_head;
        } else {
            scale_block_bytes_ = static_cast<size_t>(block_size_) * n_kv_heads * sizeof(half);
        }
        // Always 2x: K scales region + V scales region (even for TURBOQUANT_LITE)
        size_t scale_total = static_cast<size_t>(n_layers_) * max_blocks_ * 2 * scale_block_bytes_;
        if (alloc_) {
            scale_pool_ = alloc_->allocate(scale_total, "kv_cache_scales");
        } else {
            cudaError_t serr = cudaMalloc(&scale_pool_, scale_total);
            if (serr != cudaSuccess)
                scale_pool_ = nullptr;
        }
        if (!scale_pool_) {
            if (alloc_)
                alloc_->free(pool_);
            else
                IMP_CUDA_CHECK_LOG(cudaFree(pool_));
            pool_ = nullptr;
            char msg[256];
            std::snprintf(msg, sizeof(msg), "KVCache: allocation failed for %s scale pool %.2f MiB",
                          dtype_name(dtype), static_cast<double>(scale_total) / (1024.0 * 1024.0));
            throw std::runtime_error(msg);
        }
        IMP_CUDA_CHECK_LOG(cudaMemset(scale_pool_, 0, scale_total));
        MemAccount::instance().note("KV_BLOCK_POOL", static_cast<std::ptrdiff_t>(scale_total));
    }

    // Initialise per-block ref counts (0 = free) and build free list
    ref_counts_.resize(max_blocks_, 0);
    free_list_.reserve(max_blocks_);
    for (int i = max_blocks_ - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

// ---------------------------------------------------------------------------
// Per-layer shape constructor (Gemma 4 dual attention geometry)
// ---------------------------------------------------------------------------
KVCache::KVCache(int n_layers, const std::vector<int>& n_kv_heads_per_layer,
                 const std::vector<int>& head_dim_per_layer, QType dtype, int max_blocks, int block_size,
                 VRAMAllocator* alloc, const std::vector<char>& layer_is_swa, int swa_max_blocks)
    : n_layers_(n_layers), max_blocks_(max_blocks), block_size_(block_size), dtype_(dtype), alloc_(alloc),
      layer_is_swa_(layer_is_swa), swa_max_blocks_(swa_max_blocks) {
    bool packed_4bit = (dtype == QType::INT4 || dtype == QType::NVFP4 || dtype == QType::MXFP4_KV);
    size_t elem_size = packed_4bit ? 0  // 4-bit modes use /2 below
                                   : dtype_size(dtype);

    // SWA group only meaningful with both a flag vector and a capacity.
    if (layer_is_swa_.empty() || swa_max_blocks_ <= 0) {
        layer_is_swa_.clear();
        swa_max_blocks_ = 0;
    }
    // Per-layer region capacity: SWA layers hold only the trailing window.
    auto layer_capacity = [&](int l) -> size_t {
        return (swa_max_blocks_ > 0 && l < static_cast<int>(layer_is_swa_.size()) && layer_is_swa_[l])
                   ? static_cast<size_t>(swa_max_blocks_)
                   : static_cast<size_t>(max_blocks_);
    };

    // Compute per-layer block bytes and offsets.
    layer_block_bytes_.resize(n_layers_);
    layer_k_offset_.resize(n_layers_);
    layer_v_offset_.resize(n_layers_);
    size_t running = 0;
    int max_nkv = 0, max_hd = 0;
    for (int l = 0; l < n_layers_; l++) {
        int nkv = (l < (int)n_kv_heads_per_layer.size()) ? n_kv_heads_per_layer[l] : 0;
        int hd = (l < (int)head_dim_per_layer.size()) ? head_dim_per_layer[l] : 0;
        if (nkv <= 0 || hd <= 0) {
            // Layer has no attention (e.g. hybrid SSM layer). Still reserve 0 bytes.
            layer_block_bytes_[l] = 0;
            layer_k_offset_[l] = running;
            layer_v_offset_[l] = running;
            continue;
        }
        max_nkv = std::max(max_nkv, nkv);
        max_hd = std::max(max_hd, hd);

        size_t bb = packed_4bit ? (static_cast<size_t>(block_size_) * nkv * hd / 2)
                                : (static_cast<size_t>(block_size_) * nkv * hd * elem_size);
        layer_block_bytes_[l] = bb;

        // Layout: K region then V region for this layer.
        layer_k_offset_[l] = running;
        running += layer_capacity(l) * bb;
        layer_v_offset_[l] = running;
        running += layer_capacity(l) * bb;
    }
    size_t total = running;

    // Populate scalar fallback fields with max values (for external queries)
    n_kv_heads_ = max_nkv;
    head_dim_ = max_hd;
    block_bytes_ = packed_4bit ? (static_cast<size_t>(block_size_) * max_nkv * max_hd / 2)
                               : (static_cast<size_t>(block_size_) * max_nkv * max_hd * elem_size);

    // Allocate single contiguous pool
    if (alloc_) {
        pool_ = alloc_->allocate(total, "kv_cache");
    } else {
        cudaError_t err = cudaMalloc(&pool_, total);
        if (err != cudaSuccess)
            pool_ = nullptr;
    }
    if (!pool_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "KVCache(per-layer): cudaMalloc failed for %.2f MiB",
                      static_cast<double>(total) / (1024.0 * 1024.0));
        throw std::runtime_error(msg);
    }
    IMP_CUDA_CHECK_LOG(cudaMemset(pool_, 0, total));

    // INT8/INT4 per-layer scales not yet supported in per-layer mode.
    if (dtype == QType::INT8 || dtype == QType::INT4) {
        throw std::runtime_error("KVCache per-layer shape: INT8/INT4 scale pools not yet supported");
    }

    // NVFP4 / MXFP4_KV per-layer scales: each layer's scale-block-bytes derived from its own
    // (nkv * hd / kNVFP4Group) so that layers with different head_dim (Gemma 4
    // SWA vs full-attention layers) get correctly-sized scale storage.
    // MXFP4_KV uses identical layout — same per-16-element group size; only the
    // scale byte semantics differ (UE8M0 vs E4M3), which is transparent here.
    if (dtype == QType::NVFP4 || dtype == QType::MXFP4_KV) {
        layer_scale_block_bytes_.resize(n_layers_);
        layer_k_scale_offset_.resize(n_layers_);
        layer_v_scale_offset_.resize(n_layers_);
        size_t srunning = 0;
        for (int l = 0; l < n_layers_; l++) {
            int nkv = (l < (int)n_kv_heads_per_layer.size()) ? n_kv_heads_per_layer[l] : 0;
            int hd = (l < (int)head_dim_per_layer.size()) ? head_dim_per_layer[l] : 0;
            if (nkv <= 0 || hd <= 0) {
                layer_scale_block_bytes_[l] = 0;
                layer_k_scale_offset_[l] = srunning;
                layer_v_scale_offset_[l] = srunning;
                continue;
            }
            if (hd % kNVFP4Group != 0) {
                throw std::runtime_error(
                    "KVCache per-layer NVFP4: head_dim must be a multiple of 16");
            }
            size_t sbb = static_cast<size_t>(block_size_) * nkv * (hd / kNVFP4Group);
            layer_scale_block_bytes_[l] = sbb;
            layer_k_scale_offset_[l] = srunning;
            srunning += layer_capacity(l) * sbb;
            layer_v_scale_offset_[l] = srunning;
            srunning += layer_capacity(l) * sbb;
        }
        size_t sc_total = srunning;
        if (alloc_) {
            scale_pool_ = alloc_->allocate(sc_total, "kv_cache_scales");
        } else {
            cudaError_t serr = cudaMalloc(&scale_pool_, sc_total);
            if (serr != cudaSuccess)
                scale_pool_ = nullptr;
        }
        if (!scale_pool_) {
            if (alloc_)
                alloc_->free(pool_);
            else
                IMP_CUDA_CHECK_LOG(cudaFree(pool_));
            pool_ = nullptr;
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "KVCache(per-layer NVFP4): scale pool alloc failed for %.2f MiB",
                          static_cast<double>(sc_total) / (1024.0 * 1024.0));
            throw std::runtime_error(msg);
        }
        IMP_CUDA_CHECK_LOG(cudaMemset(scale_pool_, 0, sc_total));
        // For external queries, scalar fallback uses max-layer block bytes so
        // sizeof checks see a non-zero value.
        scale_block_bytes_ = static_cast<size_t>(block_size_) * max_nkv * (max_hd / kNVFP4Group);
    }

    // Initialize ref counts and free list
    ref_counts_.resize(max_blocks_, 0);
    free_list_.reserve(max_blocks_);
    for (int i = max_blocks_ - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }

    // SWA group: separate id space with its own free list.
    if (swa_max_blocks_ > 0) {
        swa_ref_counts_.resize(swa_max_blocks_, 0);
        swa_free_list_.reserve(swa_max_blocks_);
        for (int i = swa_max_blocks_ - 1; i >= 0; --i) {
            swa_free_list_.push_back(i);
        }
        int n_swa_layers = 0;
        for (char f : layer_is_swa_)
            if (f)
                n_swa_layers++;
        IMP_LOG_INFO("KVCache (per-layer): SWA group %d blocks × %d windowed layers "
                     "(global group %d blocks × %d layers)",
                     swa_max_blocks_, n_swa_layers, max_blocks_, n_layers_ - n_swa_layers);
    }

    IMP_LOG_INFO("KVCache (per-layer): %zu layers, pool %.2f MiB, max nkv=%d, max hd=%d", (size_t)n_layers_,
                 total / (1024.0 * 1024.0), max_nkv, max_hd);
}

KVCache::~KVCache() {
    if (scale_pool_) {
        if (alloc_)
            alloc_->free(scale_pool_);
        else
            IMP_CUDA_CHECK_LOG(cudaFree(scale_pool_));
        scale_pool_ = nullptr;
    }
    if (pool_) {
        if (alloc_)
            alloc_->free(pool_);
        else
            IMP_CUDA_CHECK_LOG(cudaFree(pool_));
        pool_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Block allocation
// ---------------------------------------------------------------------------

int KVCache::allocate_block() {
    if (free_list_.empty()) {
        if (graph_diag::enabled()) {
            IMP_LOG_ERROR("[graph_diag:kv_alloc] OOM (phase=%s, 0 free blocks left)",
                          graph_diag::phase_name(graph_diag::phase()));
        }
        return -1;
    }

    int block_id = free_list_.back();
    free_list_.pop_back();
    ref_counts_[block_id] = 1;

    if (graph_diag::enabled()) {
        auto p = graph_diag::phase();
        // Allocations during replay are the smoking gun for Hypothesis H1
        // (KV-block boundary crossed mid-replay with stale block_tables).
        if (p == graph_diag::Phase::REPLAY) {
            IMP_LOG_ERROR(
                "[graph_diag:kv_alloc] allocate_block id=%d free_left=%zu "
                "phase=REPLAY  <-- H1 smoking gun",
                block_id, free_list_.size());
        } else {
            IMP_LOG_INFO("[graph_diag:kv_alloc] allocate_block id=%d free_left=%zu phase=%s", block_id,
                         free_list_.size(), graph_diag::phase_name(p));
        }
    }
    return block_id;
}

void KVCache::free_block(int block_id) {
    if (block_id < 0 || block_id >= max_blocks_)
        return;
    if (ref_counts_[block_id] <= 0)
        return;

    --ref_counts_[block_id];
    if (ref_counts_[block_id] == 0) {
        free_list_.push_back(block_id);
    }
}

// ---------------------------------------------------------------------------
// SWA block group (kv_cache.swa_sizing): separate id space, no sharing —
// SWA blocks are never prefix-cached or pinned, so ref counts stay 0/1.
// ---------------------------------------------------------------------------

int KVCache::allocate_swa_block() {
    if (swa_free_list_.empty())
        return -1;
    int block_id = swa_free_list_.back();
    swa_free_list_.pop_back();
    swa_ref_counts_[block_id] = 1;
    return block_id;
}

void KVCache::free_swa_block(int block_id) {
    if (block_id < 0 || block_id >= swa_max_blocks_)
        return;
    if (swa_ref_counts_[block_id] <= 0)
        return;
    if (--swa_ref_counts_[block_id] == 0)
        swa_free_list_.push_back(block_id);
}

// ---------------------------------------------------------------------------
// Reference counting
// ---------------------------------------------------------------------------

int KVCache::ref_count(int block_id) const {
    if (block_id < 0 || block_id >= max_blocks_)
        return 0;
    return ref_counts_[block_id];
}

void KVCache::inc_ref(int block_id) {
    if (block_id < 0 || block_id >= max_blocks_)
        return;
    ++ref_counts_[block_id];
}

// ---------------------------------------------------------------------------
// Pointer computation into the contiguous pool
// ---------------------------------------------------------------------------

void* KVCache::k_ptr(int layer, int block_id) {
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache k_ptr bounds violation: layer=%d/%d, block=%d/%d", layer, n_layers_, block_id,
                      max_blocks_);
    }
#endif
    // Per-layer shape path: use precomputed per-layer offsets and block bytes.
    if (!layer_block_bytes_.empty()) {
        size_t offset = layer_k_offset_[layer] + static_cast<size_t>(block_id) * layer_block_bytes_[layer];
        return static_cast<char*>(pool_) + offset;
    }
    // K blocks: [layer * 2 * max_blocks + block_id] * block_bytes
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ + static_cast<size_t>(block_id)) *
                    block_bytes_;
    return static_cast<char*>(pool_) + offset;
}

void* KVCache::v_ptr(int layer, int block_id) {
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache v_ptr bounds violation: layer=%d/%d, block=%d/%d", layer, n_layers_, block_id,
                      max_blocks_);
    }
#endif
    // Per-layer shape path
    if (!layer_block_bytes_.empty()) {
        size_t offset = layer_v_offset_[layer] + static_cast<size_t>(block_id) * layer_block_bytes_[layer];
        return static_cast<char*>(pool_) + offset;
    }
    // Standard K+V pool: V blocks follow K blocks per layer
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ + max_blocks_ +
                     static_cast<size_t>(block_id)) *
                    block_bytes_;
    return static_cast<char*>(pool_) + offset;
}

// ---------------------------------------------------------------------------
// Capacity queries
// ---------------------------------------------------------------------------

int KVCache::num_free_blocks() const { return static_cast<int>(free_list_.size()); }

int KVCache::total_blocks() const { return max_blocks_; }

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

size_t KVCache::block_bytes() const { return block_bytes_; }

int KVCache::n_layers() const { return n_layers_; }

int KVCache::n_kv_heads() const { return n_kv_heads_; }

int KVCache::head_dim() const { return head_dim_; }

QType KVCache::qtype() const { return dtype_; }

// ---------------------------------------------------------------------------
// Scale pointer computation (same layout for all quantized modes)
// ---------------------------------------------------------------------------

void* KVCache::k_scale_ptr(int layer, int block_id) {
    if (!scale_pool_)
        return nullptr;
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache k_scale_ptr bounds violation: layer=%d/%d, block=%d/%d", layer, n_layers_,
                      block_id, max_blocks_);
    }
#endif
    // Per-layer NVFP4 path
    if (!layer_scale_block_bytes_.empty()) {
        size_t off = layer_k_scale_offset_[layer] +
                     static_cast<size_t>(block_id) * layer_scale_block_bytes_[layer];
        return static_cast<char*>(scale_pool_) + off;
    }
    // Scale pool always uses 2x layout (K scales region + V scales region)
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ + static_cast<size_t>(block_id)) *
                    scale_block_bytes_;
    return static_cast<char*>(scale_pool_) + offset;
}

void* KVCache::v_scale_ptr(int layer, int block_id) {
    if (!scale_pool_)
        return nullptr;
#ifdef IMP_DEBUG
    if (layer < 0 || layer >= n_layers_ || block_id < 0 || block_id >= max_blocks_) {
        IMP_LOG_ERROR("KV cache v_scale_ptr bounds violation: layer=%d/%d, block=%d/%d", layer, n_layers_,
                      block_id, max_blocks_);
    }
#endif
    if (!layer_scale_block_bytes_.empty()) {
        size_t off = layer_v_scale_offset_[layer] +
                     static_cast<size_t>(block_id) * layer_scale_block_bytes_[layer];
        return static_cast<char*>(scale_pool_) + off;
    }
    size_t offset = (static_cast<size_t>(layer) * 2 * max_blocks_ + max_blocks_ +
                     static_cast<size_t>(block_id)) *
                    scale_block_bytes_;
    return static_cast<char*>(scale_pool_) + offset;
}

size_t KVCache::scale_block_bytes() const { return scale_block_bytes_; }

size_t KVCache::scale_block_bytes(int layer) const {
    if (!layer_scale_block_bytes_.empty()) {
        if (layer < 0 || layer >= n_layers_)
            return 0;
        return layer_scale_block_bytes_[layer];
    }
    return scale_block_bytes_;
}

}  // namespace imp
