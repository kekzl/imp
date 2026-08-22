#include "memory/kv_cache.h"
#include "memory/backend.h"
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
                 VRAMAllocator* alloc, int ceiling_blocks)
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
    // Growable: every offset below is computed from max_blocks_, so the CEILING
    // has to be the stride. What is initially usable is the block count the
    // caller could afford, tracked by the id space rather than by the layout.
    const int usable = plan_growth_(max_blocks, ceiling_blocks);
    size_t total = static_cast<size_t>(n_layers_) * max_blocks_ * 2 * block_bytes_;
    if (!reserve_pool_(total, usable)) {
        // max_blocks_ is back to the fixed size, so the pool below is the one
        // that was actually affordable.
        total = static_cast<size_t>(n_layers_) * max_blocks_ * 2 * block_bytes_;
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
    } else if (commit_blocks_(usable) != usable) {
        throw std::runtime_error("KVCache: growable pool could not commit its initial blocks");
    }
    // Attribution comes from VRAMAllocator::allocate() now, under the "kv_cache"
    // tag — noting it here as well double-counted the pool.

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
    }

    // Block ids + refcounts. Slots-only: this class keeps the memory (the
    // layout is layer-major and cannot be expressed as a uniform stride).
    if (blocks_.open_slots(usable) != MemError::Ok)
        throw std::runtime_error("KVCache: block id space init failed");
    usable_blocks_.store(usable, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Per-layer shape constructor (Gemma 4 dual attention geometry)
// ---------------------------------------------------------------------------
KVCache::KVCache(int n_layers, const std::vector<int>& n_kv_heads_per_layer,
                 const std::vector<int>& head_dim_per_layer, QType dtype, int max_blocks, int block_size,
                 VRAMAllocator* alloc, const std::vector<char>& layer_is_swa, int swa_max_blocks,
                 int ceiling_blocks)
    : n_layers_(n_layers),
      max_blocks_(max_blocks),
      block_size_(block_size),
      dtype_(dtype),
      alloc_(alloc),
      layer_is_swa_(layer_is_swa),
      swa_max_blocks_(swa_max_blocks) {
    // Before the offset loop below: it strides by layer_capacity(l), which is
    // max_blocks_ for a full-attention layer, and the stride has to be the
    // ceiling or growth would move every layer's data.
    const int usable = plan_growth_(max_blocks, ceiling_blocks);
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
    //
    // In a lambda because it strides by layer_capacity(l), which reads
    // max_blocks_ — the CEILING while a growable pool is intended. If the
    // reservation is then declined, the stride has to go back to the affordable
    // size and the whole layout with it, or the offsets would describe a pool
    // that was never allocated.
    layer_block_bytes_.resize(n_layers_);
    layer_k_offset_.resize(n_layers_);
    layer_v_offset_.resize(n_layers_);
    int max_nkv = 0, max_hd = 0;
    auto compute_layout = [&]() -> size_t {
    size_t running = 0;
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
    return running;
    };
    size_t total = compute_layout();

    // Populate scalar fallback fields with max values (for external queries)
    n_kv_heads_ = max_nkv;
    head_dim_ = max_hd;
    block_bytes_ = packed_4bit ? (static_cast<size_t>(block_size_) * max_nkv * max_hd / 2)
                               : (static_cast<size_t>(block_size_) * max_nkv * max_hd * elem_size);

    // Allocate single contiguous pool
    if (!reserve_pool_(total, usable)) {
        total = compute_layout();  // max_blocks_ is back to the affordable size
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
    } else if (commit_blocks_(usable) != usable) {
        throw std::runtime_error("KVCache(per-layer): growable pool could not commit its initial blocks");
    }

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

    // `usable`, never max_blocks_: on a growable pool max_blocks_ is the
    // CEILING, and handing out an id whose memory is not committed yet faults
    // on the first write into that block.
    if (blocks_.open_slots(usable) != MemError::Ok)
        throw std::runtime_error("KVCache(per-layer): block id space init failed");
    usable_blocks_.store(usable, std::memory_order_relaxed);

    // SWA group: separate id space, separate pool. Not a partition of the
    // global one — a ref must not be able to cross between them.
    if (swa_max_blocks_ > 0) {
        if (swa_blocks_.open_slots(swa_max_blocks_) != MemError::Ok)
            throw std::runtime_error("KVCache(per-layer): SWA block id space init failed");
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
    // The manager's referents still hold UNTRACKED references at this point
    // (they store ints). abandon() skips the outstanding-ref check; it goes
    // away with the last int-based caller in A7 step 3's final commit.
    blocks_.abandon();
    swa_blocks_.abandon();
    if (d_copy_meta_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_copy_meta_));
        d_copy_meta_ = nullptr;
    }
    if (scale_pool_) {
        if (alloc_)
            alloc_->free(scale_pool_);
        else
            IMP_CUDA_CHECK_LOG(cudaFree(scale_pool_));
        scale_pool_ = nullptr;
    }
    if (region_) {
        MemAccount::instance().note("kv_cache", -static_cast<std::ptrdiff_t>(region_.committed()));
        region_.reset();  // unmaps and releases every committed chunk
        pool_ = nullptr;
    } else if (pool_) {
        if (alloc_)
            alloc_->free(pool_);
        else
            IMP_CUDA_CHECK_LOG(cudaFree(pool_));
        pool_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Growable pool
// ---------------------------------------------------------------------------

// Decide the layout stride. Returns how many blocks are usable to begin with.
//
// Growth is refused rather than faked when the device has no virtual memory
// management: a pool that reports a ceiling it can never reach would make
// admission wait for memory that is not coming.
int KVCache::plan_growth_(int max_blocks, int ceiling_blocks) {
    if (ceiling_blocks <= max_blocks || vmm_backend() == nullptr)
        return max_blocks;
    max_blocks_ = ceiling_blocks;  // the stride, and the ceiling
    growable_ = true;
    return max_blocks;
}

// Reserve address space for the ceiling. False means "not growable", and the
// caller falls back to the fixed allocation it would have made anyway — a
// reservation failure must not turn into a load failure, because the fixed
// path can still serve.
bool KVCache::reserve_pool_(size_t total_bytes, int fixed_blocks) {
    if (!growable_)
        return false;
    Backend* be = vmm_backend();
    if (be == nullptr)
        return false;
    auto res = be->acquire_growable(total_bytes, 0, 256, RegionTag::KvBlockPool);
    if (!res) {
        IMP_LOG_WARN(
            "KV cache: could not reserve %.0f MiB of address space for a growable pool "
            "(%s); falling back to a fixed pool of %d blocks",
            total_bytes / (1024.0 * 1024.0), mem_error_name(res.error), fixed_blocks);
        // Put the stride back before the caller sizes its fixed allocation from
        // it. Leaving the ceiling in place would allocate the pool this machine
        // has just said it could not even reserve address space for.
        growable_ = false;
        max_blocks_ = fixed_blocks;
        return false;
    }
    region_ = std::move(res.region);
    pool_ = region_.base();
    return true;
}

// Commit the memory that blocks [0, blocks) occupy, in every layer's region.
// Returns the number of blocks actually backed, which is what the caller must
// believe: a growth that got halfway is half a pool, not a failure.
//
// One call per layer region, because the pool is laid out per layer and what
// grows is every layer at once, not the end of one buffer.
int KVCache::commit_blocks_(int blocks) {
    if (!region_ || blocks <= 0)
        return 0;
    Backend* be = vmm_backend();
    const char* base = static_cast<const char*>(pool_);
    const size_t before = region_.committed();
    const int first_new = committed_blocks_;  // what to zero, and only that
    for (int l = 0; l < n_layers_; l++) {
        const size_t bb = layer_block_bytes_.empty() ? block_bytes_ : layer_block_bytes_[l];
        if (bb == 0)
            continue;  // a non-attention layer in a hybrid holds no KV
        // A sliding-window layer's region is smaller than `blocks` implies, so
        // this commits past it into the next layer's range. Deliberate: that
        // range belongs to the same reservation and is wanted anyway, the tail
        // is clamped to the reservation, and capping it per layer was measured
        // to change nothing (64.00 vs 34.00 MiB committed either way, the
        // difference coming from the layout rather than from the cap).
        const size_t want = static_cast<size_t>(blocks) * bb;
        const size_t k_off = static_cast<const char*>(k_ptr(l, 0)) - base;
        const size_t v_off = static_cast<const char*>(v_ptr(l, 0)) - base;
        if (be->commit_range(region_, k_off, want) != MemError::Ok ||
            be->commit_range(region_, v_off, want) != MemError::Ok) {
            // Stop at the first layer that cannot be backed. Blocks are only
            // usable when EVERY layer has memory for them, so the honest
            // capacity is the one that was fully committed before this.
            const size_t added = region_.committed() - before;
            MemAccount::instance().note("kv_cache", static_cast<std::ptrdiff_t>(added));
            return committed_blocks_;
        }
    }
    const size_t added = region_.committed() - before;
    if (added > 0)
        MemAccount::instance().note("kv_cache", static_cast<std::ptrdiff_t>(added));
    // Fresh blocks start clean, exactly as the fixed pool's one big memset
    // guarantees. Driver-committed pages carry no such promise, and a block
    // handed out with stale bytes in its unused tail is the kind of difference
    // that shows up as rare, unreproducible output rather than as an error.
    // Only the new range: rezeroing the old one would erase live KV.
    //
    // The invariant this keeps is "a block is clean the first time it is handed
    // out", and it is tracked by committed_blocks_. A future shrink has to
    // lower that counter when it decommits, or a range that is decommitted and
    // recommitted would come back with whatever the driver hands over and skip
    // the memset below. Reuse of an already-committed block does NOT re-zero,
    // which is the fixed pool's behaviour too: attention reads only the slots
    // the sequence wrote.
    for (int l = 0; l < n_layers_ && blocks > first_new; l++) {
        const size_t bb = layer_block_bytes_.empty() ? block_bytes_ : layer_block_bytes_[l];
        if (bb == 0)
            continue;
        // Clamp to THIS layer's region. The commit loop above deliberately
        // over-commits past a sliding-window layer (same reservation, wanted
        // anyway, and the tail is clamped there); this loop had the same
        // arithmetic and no clamp, so on a pool with windowed layers it wrote
        // past the end of the reservation - `cudaMemset ... an illegal memory
        // access`, which is sticky and took the whole test binary down with it.
        //
        // Two bugs, not one: for a windowed layer that is NOT last, the same
        // overrun lands inside the next layer's live KV and zeroes it.
        const size_t cap = layer_block_bytes_.empty() ? static_cast<size_t>(max_blocks_)
                                                      : layer_capacity_(l);
        const size_t hi = std::min(static_cast<size_t>(blocks), cap);
        if (hi <= static_cast<size_t>(first_new))
            continue;  // this layer's region is already fully zeroed
        const size_t bytes = (hi - static_cast<size_t>(first_new)) * bb;
        IMP_CUDA_CHECK_LOG(cudaMemset(k_ptr(l, first_new), 0, bytes));
        IMP_CUDA_CHECK_LOG(cudaMemset(v_ptr(l, first_new), 0, bytes));
    }
    committed_blocks_ = blocks;
    return blocks;
}

size_t KVCache::committed_bytes() const { return region_ ? region_.committed() : 0; }

int KVCache::try_grow_to(int wanted) {
    const int have = blocks_.num_blocks();
    if (!growable_ || wanted <= have)
        return have;
    const int target = std::min(wanted, max_blocks_);
    if (target <= have)
        return have;
    const int got = commit_blocks_(target);
    if (got <= have)
        return have;
    if (blocks_.grow_slots(got) != MemError::Ok)
        return have;
    // Published only after the ids exist: a reader that saw the new capacity
    // first would admit a request the pool cannot yet hand blocks for.
    usable_blocks_.store(got, std::memory_order_release);
    growths_.fetch_add(1, std::memory_order_relaxed);
    IMP_LOG_INFO(
        "KV cache: grew %d -> %d blocks (%.0f tokens, %.0f MiB committed of a %d-block "
        "ceiling)",
        have, got, static_cast<double>(got) * block_size_, region_.committed() / (1024.0 * 1024.0),
        max_blocks_);
    return got;
}

// ---------------------------------------------------------------------------
// Block allocation
// ---------------------------------------------------------------------------

int KVCache::allocate_block() {
    BlockRef ref = blocks_.acquire();
    if (!ref) {
        if (graph_diag::enabled()) {
            IMP_LOG_ERROR("[graph_diag:kv_alloc] OOM (phase=%s, 0 free blocks left)",
                          graph_diag::phase_name(graph_diag::phase()));
        }
        return -1;
    }
    // The int API keeps the reference untracked, exactly as before.
    const int block_id = ref.release();

    if (graph_diag::enabled()) {
        auto p = graph_diag::phase();
        // Allocations during replay are the smoking gun for Hypothesis H1
        // (KV-block boundary crossed mid-replay with stale block_tables).
        if (p == graph_diag::Phase::REPLAY) {
            IMP_LOG_ERROR(
                "[graph_diag:kv_alloc] allocate_block id=%d free_left=%d "
                "phase=REPLAY  <-- H1 smoking gun",
                block_id, blocks_.free_count());
        } else {
            IMP_LOG_INFO("[graph_diag:kv_alloc] allocate_block id=%d free_left=%d phase=%s", block_id,
                         blocks_.free_count(), graph_diag::phase_name(p));
        }
    }
    return block_id;
}

void KVCache::free_block(int block_id) { blocks_.release_raw(block_id); }

BlockRef KVCache::acquire_block_ref() { return blocks_.acquire(); }

BlockRef KVCache::share_block(int block_id) { return blocks_.share_by_id(block_id); }

// ---------------------------------------------------------------------------
// SWA block group (kv_cache.swa_sizing): separate id space, no sharing —
// SWA blocks are never prefix-cached or pinned, so ref counts stay 0/1.
// ---------------------------------------------------------------------------

int KVCache::allocate_swa_block() {
    BlockRef ref = swa_blocks_.acquire();
    return ref ? ref.release() : -1;
}

void KVCache::free_swa_block(int block_id) { swa_blocks_.release_raw(block_id); }

// ---------------------------------------------------------------------------
// Reference counting
// ---------------------------------------------------------------------------

int KVCache::ref_count(int block_id) const { return blocks_.ref_count(block_id); }

void KVCache::inc_ref(int block_id) { blocks_.acquire_raw(block_id); }

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

int KVCache::num_free_blocks() const { return blocks_.free_count(); }

// What may be handed out, which is the whole pool unless it is growable and
// has not reached its ceiling yet. Admission reads this, so it must never
// report memory that is reserved but not committed.
int KVCache::total_blocks() const { return usable_blocks_.load(std::memory_order_relaxed); }

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

// ── copy_blocks_device ───────────────────────────────────────────────
namespace {

struct KVCopyPairs {
    int src[KVCache::kCopyMaxPairs];
    int dst[KVCache::kCopyMaxPairs];
};

// grid: (n_layers, 4, n_pairs); blockIdx.y section: 0=K, 1=V, 2=K-scales,
// 3=V-scales. meta layout per layer: {k_off, v_off, block_bytes,
// k_scale_off, v_scale_off, scale_bytes}.
__global__ void kv_block_copy_kernel(char* pool, char* spool, const size_t* __restrict__ meta,
                                     KVCopyPairs pairs) {
    const size_t* m = meta + 6ull * blockIdx.x;
    const int sec = blockIdx.y;
    char* base = (sec < 2) ? pool : spool;
    if (base == nullptr)
        return;
    const size_t bytes = (sec < 2) ? m[2] : m[5];
    if (bytes == 0)
        return;
    const size_t off = (sec == 0) ? m[0] : (sec == 1) ? m[1] : (sec == 2) ? m[3] : m[4];
    const char* s = base + off + static_cast<size_t>(pairs.src[blockIdx.z]) * bytes;
    char* d = base + off + static_cast<size_t>(pairs.dst[blockIdx.z]) * bytes;
    const bool vec_ok = bytes % 16 == 0 && (reinterpret_cast<uintptr_t>(s) % 16 == 0) &&
                        (reinterpret_cast<uintptr_t>(d) % 16 == 0);
    if (vec_ok) {
        const uint4* s4 = reinterpret_cast<const uint4*>(s);
        uint4* d4 = reinterpret_cast<uint4*>(d);
        for (size_t i = threadIdx.x; i < bytes / 16; i += blockDim.x)
            d4[i] = s4[i];
    } else {
        for (size_t i = threadIdx.x; i < bytes; i += blockDim.x)
            d[i] = s[i];
    }
}

}  // namespace

void KVCache::copy_blocks_device(const int* srcs, const int* dsts, int n_pairs,
                                 cudaStream_t stream) {
    if (n_pairs <= 0 || !srcs || !dsts)
        return;
    if (n_pairs > kCopyMaxPairs)
        n_pairs = kCopyMaxPairs;
    if (!d_copy_meta_) {
        std::vector<size_t> meta(6ull * n_layers_);
        for (int l = 0; l < n_layers_; ++l) {
            size_t* m = meta.data() + 6ull * l;
            if (layer_is_swa(l)) {
                // Separate id space — never copied here (route excludes SWA).
                m[2] = m[5] = 0;
                continue;
            }
            m[0] = static_cast<char*>(k_ptr(l, 0)) - static_cast<char*>(pool_);
            m[1] = static_cast<char*>(v_ptr(l, 0)) - static_cast<char*>(pool_);
            m[2] = layer_block_bytes_.empty() ? block_bytes_ : layer_block_bytes_[l];
            if (scale_pool_ && k_scale_ptr(l, 0)) {
                m[3] = static_cast<char*>(k_scale_ptr(l, 0)) - static_cast<char*>(scale_pool_);
                m[4] = static_cast<char*>(v_scale_ptr(l, 0)) - static_cast<char*>(scale_pool_);
                m[5] = scale_block_bytes(l);
            } else {
                m[3] = m[4] = m[5] = 0;
            }
        }
        if (cudaMalloc(&d_copy_meta_, meta.size() * sizeof(size_t)) != cudaSuccess) {
            d_copy_meta_ = nullptr;
            return;
        }
        IMP_CUDA_CHECK_LOG(cudaMemcpy(d_copy_meta_, meta.data(), meta.size() * sizeof(size_t),
                                      cudaMemcpyHostToDevice));
    }
    KVCopyPairs pairs{};
    for (int i = 0; i < n_pairs; ++i) {
        pairs.src[i] = srcs[i];
        pairs.dst[i] = dsts[i];
    }
    dim3 grid(n_layers_, scale_pool_ ? 4 : 2, n_pairs);
    kv_block_copy_kernel<<<grid, 256, 0, stream>>>(static_cast<char*>(pool_),
                                                   static_cast<char*>(scale_pool_),
                                                   static_cast<const size_t*>(d_copy_meta_), pairs);
    IMP_CUDA_CHECK_LAUNCH();
}

// ── batched_copy_device ──────────────────────────────────────────────
// One CTA per desc; uint4 fast path when src/dst/bytes are 16-byte aligned
// (block and scale regions are — pool offsets are multiples of the block
// byte sizes), byte loop otherwise.

static __global__ void kv_batched_copy_kernel(const KVCache::CopyDesc* descs, int n) {
    const int d = blockIdx.x;
    if (d >= n)
        return;
    const char* src = static_cast<const char*>(descs[d].src);
    char* dst = static_cast<char*>(descs[d].dst);
    const size_t bytes = descs[d].bytes;
    const bool aligned = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst) |
                           bytes) & 15u) == 0;
    if (aligned) {
        const uint4* s = reinterpret_cast<const uint4*>(src);
        uint4* t = reinterpret_cast<uint4*>(dst);
        const size_t n16 = bytes / 16;
        for (size_t i = threadIdx.x; i < n16; i += blockDim.x)
            t[i] = s[i];
    } else {
        for (size_t i = threadIdx.x; i < bytes; i += blockDim.x)
            dst[i] = src[i];
    }
}

void KVCache::batched_copy_device(const CopyDesc* d_descs, int n, cudaStream_t stream) {
    if (!d_descs || n <= 0)
        return;
    kv_batched_copy_kernel<<<n, 256, 0, stream>>>(d_descs, n);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
