// KVCache quantized-KV scale plane (INT8/INT4/NVFP4/MXFP4_KV). Split from kv_cache.cu for
// the file-size gate; same class, same invariants.
#include "memory/kv_cache.h"
#include "memory/backend.h"
#include "memory/mem_account.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace imp {

// Growable KV pool: reserve the ceiling, commit what the KV pool committed; commit_blocks_
// extends both (#2483: eager planes cost 643.8 MiB at a 20603-block ceiling on Qwen3.8-27B).
// Fixed pool or refused reservation: one zeroed allocation, as before.
void KVCache::alloc_scale_pool_(size_t total) {
    if (region_) {
        Backend* be = vmm_backend();
        auto res = be ? be->acquire_growable(total, 0, 256, RegionTag::KvBlockPool)
                      : AcquireResult{Region{}, MemError::NotGrowable};
        if (res) {
            scale_region_ = std::move(res.region);
            scale_pool_ = scale_region_.base();
            if (commit_scales_(0, committed_blocks_)) {
                size_t per_block = 0;  // K + V scale bytes one block adds across the layers
                for (int l = 0; l < n_layers_; l++)
                    per_block += layer_is_swa(l) ? 0 : 2 * scale_block_bytes(l);
                IMP_LOG_INFO(
                    "KV cache scales (kv_cache_scales): %d blocks x %zu B = %.2f MiB, %.1f MiB "
                    "committed; grows with the pool to %.1f MiB at the %d-block ceiling",
                    committed_blocks_, per_block,
                    static_cast<double>(committed_blocks_) * per_block / (1024.0 * 1024.0),
                    scale_region_.committed() / (1024.0 * 1024.0),
                    static_cast<double>(total) / (1024.0 * 1024.0), max_blocks_);
                return;
            }
            MemAccount::instance().note("kv_cache_scales",
                                        -static_cast<std::ptrdiff_t>(scale_region_.committed()));
            scale_region_.reset();
            scale_pool_ = nullptr;
        } else {
            IMP_LOG_WARN("KV cache: scale plane reservation failed (%.1f MiB, %s); allocating it whole",
                         static_cast<double>(total) / (1024.0 * 1024.0), mem_error_name(res.error));
        }
    }
    if (!scale_pool_) {
        if (alloc_) {
            scale_pool_ = alloc_->allocate(total, "kv_cache_scales");
        } else if (cudaMalloc(&scale_pool_, total) != cudaSuccess) {
            scale_pool_ = nullptr;
        }
    }
    if (!scale_pool_) {
        if (region_) {
            MemAccount::instance().note("kv_cache", -static_cast<std::ptrdiff_t>(region_.committed()));
            region_.reset();
        } else if (alloc_) {
            alloc_->free(pool_);
        } else {
            IMP_CUDA_CHECK_LOG(cudaFree(pool_));
        }
        pool_ = nullptr;
        char msg[256];
        std::snprintf(msg, sizeof(msg), "KVCache: allocation failed for %s scale pool %.2f MiB",
                      dtype_name(dtype_), static_cast<double>(total) / (1024.0 * 1024.0));
        throw std::runtime_error(msg);
    }
    IMP_CUDA_CHECK_LOG(cudaMemset(scale_pool_, 0, total));
}

// Back scale blocks [0, to) in every layer's K and V plane and zero [from, to). A windowed
// layer's id space opens whole (open_swa_group_), so its plane commits whole on the first call.
// Zeroing runs on the legacy stream, synchronized before the blocks are published (#1652).
bool KVCache::commit_scales_(int from, int to) {
    if (!scale_region_ || to <= 0)
        return true;
    Backend* be = vmm_backend();
    char* base = static_cast<char*>(scale_pool_);
    const size_t before = scale_region_.committed();
    bool ok = be != nullptr;
    bool zeroed = false;
    for (int l = 0; ok && l < n_layers_; l++) {
        const size_t sbb = scale_block_bytes(l);
        if (sbb == 0)
            continue;  // a non-attention layer in a hybrid holds no KV
        const size_t cap = layer_capacity_(l);
        const bool swa = layer_is_swa(l);
        const size_t hi = swa ? cap : std::min(static_cast<size_t>(to), cap);
        const size_t lo = swa ? (from > 0 ? cap : 0) : std::min(static_cast<size_t>(std::max(from, 0)), hi);
        for (char* p : {static_cast<char*>(k_scale_ptr(l, 0)), static_cast<char*>(v_scale_ptr(l, 0))}) {
            ok = ok &&
                 be->commit_range(scale_region_, static_cast<size_t>(p - base), hi * sbb) == MemError::Ok;
            if (ok && hi > lo) {
                IMP_CUDA_CHECK_LOG(cudaMemset(p + lo * sbb, 0, (hi - lo) * sbb));
                zeroed = true;
            }
        }
    }
    const size_t added = scale_region_.committed() - before;
    if (added > 0)
        MemAccount::instance().note("kv_cache_scales", static_cast<std::ptrdiff_t>(added));
    if (zeroed)
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(0));
    return ok;
}

}  // namespace imp
