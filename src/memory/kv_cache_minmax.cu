// KVCache sparse decode key min/max metadata pool (#1808, #2360). Split from kv_cache.cu
// for the file-size gate; same class, same invariants.
#include "memory/kv_cache.h"
#include "memory/backend.h"
#include "memory/mem_account.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

// ---------------------------------------------------------------------------
// Sparse decode attention: per-block key min/max metadata pool
// ---------------------------------------------------------------------------

bool KVCache::enable_key_minmax() {
    if (minmax_pool_)
        return true;
    // Scalar geometry only: the per-layer ctor leaves n_kv_heads_/head_dim_
    // unset and the offset math below assumes a uniform stride.
    if (!layer_block_bytes_.empty())
        return false;
    minmax_block_bytes_ = minmax_block_bytes(n_kv_heads_, head_dim_);
    size_t total = static_cast<size_t>(n_layers_) * max_blocks_ * minmax_block_bytes_;
    if (growable_) {
        // Grows with the pool (#2360): reserve the ceiling, commit what the KV pool has
        // committed; commit_blocks_ extends both. A fixed metadata pool here capped sparse
        // decode at the start size or cost the whole ceiling up front.
        Backend* be = vmm_backend();
        auto res = be ? be->acquire_growable(total, 0, 256, RegionTag::KvBlockPool)
                      : AcquireResult{Region{}, MemError::NotGrowable};
        if (!res) {
            minmax_block_bytes_ = 0;
            IMP_LOG_WARN(
                "KVCache: key min/max reservation failed (%.1f MiB, %s) - sparse decode "
                "attention disabled",
                static_cast<double>(total) / (1024.0 * 1024.0), mem_error_name(res.error));
            return false;
        }
        minmax_region_ = std::move(res.region);
        minmax_pool_ = minmax_region_.base();
        if (!commit_minmax_(0, committed_blocks_)) {
            MemAccount::instance().note("kv_cache_minmax",
                                        -static_cast<std::ptrdiff_t>(minmax_region_.committed()));
            minmax_region_.reset();
            minmax_pool_ = nullptr;
            minmax_block_bytes_ = 0;
            IMP_LOG_WARN(
                "KVCache: key min/max commit for %d blocks refused - sparse decode "
                "attention disabled",
                committed_blocks_);
            return false;
        }
        IMP_LOG_INFO(
            "KV cache key min/max metadata: %.1f MiB committed (%d layers x %d blocks), "
            "grows with the pool to %.1f MiB",
            minmax_region_.committed() / (1024.0 * 1024.0), n_layers_, committed_blocks_,
            static_cast<double>(total) / (1024.0 * 1024.0));
        return true;
    }
    if (alloc_) {
        minmax_pool_ = alloc_->allocate(total, "kv_cache_minmax");
    } else {
        cudaError_t err = cudaMalloc(&minmax_pool_, total);
        if (err != cudaSuccess)
            minmax_pool_ = nullptr;
    }
    if (!minmax_pool_) {
        minmax_block_bytes_ = 0;
        IMP_LOG_WARN(
            "KVCache: key min/max pool allocation failed (%.1f MiB) - sparse decode "
            "attention disabled",
            static_cast<double>(total) / (1024.0 * 1024.0));
        return false;
    }
    // Zero-init: never read before the slot-0 write initializes a block, but a
    // deterministic pattern keeps a metadata-indexing bug loud instead of UB.
    IMP_CUDA_CHECK_LOG(cudaMemset(minmax_pool_, 0, total));
    IMP_LOG_INFO("KV cache key min/max metadata: %.1f MiB (%d layers x %d blocks)",
                 static_cast<double>(total) / (1024.0 * 1024.0), n_layers_, max_blocks_);
    return true;
}

// Back metadata blocks [0, to) in every layer and zero [from, to). Layer stride is the
// ceiling (max_blocks_), as in key_minmax_ptr; commit_range rounds out into the next
// layer's range, which belongs to the same reservation. Zeroing runs on the legacy stream:
// the caller synchronizes before publishing the blocks (commit_blocks_, #1652).
bool KVCache::commit_minmax_(int from, int to) {
    if (!minmax_region_ || to <= 0)
        return true;
    Backend* be = vmm_backend();
    const size_t before = minmax_region_.committed();
    const size_t stride = static_cast<size_t>(max_blocks_) * minmax_block_bytes_;
    bool ok = be != nullptr;
    for (int l = 0; ok && l < n_layers_; l++)
        ok = be->commit_range(minmax_region_, l * stride, static_cast<size_t>(to) * minmax_block_bytes_) ==
             MemError::Ok;
    const size_t added = minmax_region_.committed() - before;
    if (added > 0)
        MemAccount::instance().note("kv_cache_minmax", static_cast<std::ptrdiff_t>(added));
    if (!ok)
        return false;
    for (int l = 0; l < n_layers_ && to > from; l++) {
        char* dst = static_cast<char*>(minmax_pool_) + l * stride + from * minmax_block_bytes_;
        IMP_CUDA_CHECK_LOG(cudaMemset(dst, 0, static_cast<size_t>(to - from) * minmax_block_bytes_));
    }
    if (to > from)
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(0));
    return true;
}

void* KVCache::key_minmax_ptr(int layer, int block_id) {
    IMP_CHECK(!accounting_only_,
              "KVCache::key_minmax_ptr on an accounting-only cache: it holds no memory. "
              "Build it with the normal constructor if the test needs bytes.");
    if (!minmax_pool_)
        return nullptr;
    size_t offset = (static_cast<size_t>(layer) * max_blocks_ + static_cast<size_t>(block_id)) *
                    minmax_block_bytes_;
    return static_cast<char*>(minmax_pool_) + offset;
}

}  // namespace imp
