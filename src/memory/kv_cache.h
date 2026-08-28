#pragma once

#include "core/tensor.h"
#include "memory/block_pool.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <vector>

namespace imp {

class VRAMAllocator;  // forward declaration

static constexpr int kKVBlockSize = 16;  // default tokens per block

// NVFP4 / MXFP4_KV micro-block size: 16 FP4 elements share one scale byte.
// All imp model head_dims (64/128/256/512) are multiples of 16.
static constexpr int kNVFP4Group = 16;

class KVCache {
public:
    // `ceiling_blocks` > `max_blocks` asks for a GROWABLE pool: address space
    // is reserved for the ceiling, physical memory is committed for
    // `max_blocks`, and try_grow_to() commits more later. Address space costs
    // nothing, so the ceiling is what the configuration wants and `max_blocks`
    // is only what fits right now.
    //
    // The point is a pool that is no longer sized once, at the moment when the
    // free-VRAM reading is least trustworthy. A server started while another
    // process still holds the card lands on the rescue floor and stays there
    // for its whole life; a growable pool heals when the card frees.
    //
    // 0 (the default) keeps the fixed pool, as does a device or build without
    // virtual memory management. Growth is then simply never available and
    // every other behaviour is bit-identical.
    KVCache(int n_layers, int n_kv_heads, int head_dim, QType dtype, int max_blocks,
            int block_size = kKVBlockSize, VRAMAllocator* alloc = nullptr, int ceiling_blocks = 0);

    // Per-layer-shape constructor (Gemma 4 dual attention geometry).
    // n_kv_heads_per_layer[l] and head_dim_per_layer[l] define layer l's
    // KV shape. The scale/sketch pools are sized using max across layers.
    //
    // SWA-aware sizing (kv_cache.swa_sizing): when `layer_is_swa` is non-empty,
    // layers flagged 1 get a small dedicated block group of `swa_max_blocks`
    // capacity instead of the full `max_blocks` — sliding-window layers only
    // ever hold the trailing window, so their regions shrink accordingly.
    // SWA blocks live in a SEPARATE block-id space [0, swa_max_blocks) with
    // their own free list (allocate_swa_block/free_swa_block); k_ptr/v_ptr
    // interpret block_id in the layer's group space (per-layer offsets).
    KVCache(int n_layers, const std::vector<int>& n_kv_heads_per_layer,
            const std::vector<int>& head_dim_per_layer, QType dtype, int max_blocks, int block_size,
            VRAMAllocator* alloc, const std::vector<char>& layer_is_swa = {}, int swa_max_blocks = 0,
            int ceiling_blocks = 0);
    ~KVCache();

    // How many blocks this pool could grow to. Equals total_blocks() unless it
    // was built growable and has not reached its ceiling.
    int ceiling_blocks() const { return max_blocks_; }
    // Whether the ceiling can actually be reached. Without this, a client
    // reading ceiling == total cannot tell a fixed pool from a growable one
    // sitting at its ceiling, and those want opposite reactions: wait for the
    // card to free, or stop waiting.
    bool growable() const { return growable_; }
    // How many times try_grow_to() actually committed more memory. Exposed for
    // /metrics: a pool that keeps growing under load is the signal an operator
    // wants before the pool stops being able to (#1641).
    uint64_t growths() const { return growths_.load(std::memory_order_relaxed); }

    // Physical memory currently committed by a growable pool, 0 for a fixed
    // one. What the pool actually costs right now, as opposed to the address
    // space it reserved.
    size_t committed_bytes() const;

    // Commit memory for at least `wanted` blocks and make them allocatable.
    // Returns the capacity afterwards, which is what the caller must believe:
    // a partial growth is a real, servable capacity and not a failure.
    //
    // Costs one driver mapping call per layer region, measured at 1.18 ms per
    // 256 MiB, so callers grow in coarse steps rather than per block.
    int try_grow_to(int wanted);

    // Block allocation / deallocation
    int allocate_block();
    void free_block(int block_id);

    // ── SWA block group (separate id space, no ref-count sharing) ────
    bool swa_enabled() const { return swa_max_blocks_ > 0; }
    bool layer_is_swa(int layer) const {
        return layer >= 0 && layer < static_cast<int>(layer_is_swa_.size()) && layer_is_swa_[layer];
    }
    int allocate_swa_block();
    void free_swa_block(int block_id);
    int num_free_swa_blocks() const { return swa_blocks_.free_count(); }
    int swa_total_blocks() const { return swa_max_blocks_; }

    // Reference counting (for copy-on-write / prefix caching)
    int ref_count(int block_id) const;
    void inc_ref(int block_id);

    // RAII handles over the same id space. These are what KVCacheManager
    // holds; allocate_block()/free_block()/inc_ref() above are the untracked
    // int-based equivalents, kept for KVCache's own direct API and its tests.
    [[nodiscard]] BlockRef acquire_block_ref();
    // Take an additional tracked reference to a block that is already held
    // (prefix reuse of a block a live sequence still owns).
    [[nodiscard]] BlockRef share_block(int block_id);

    // Pointer access into the contiguous pool
    void* k_ptr(int layer, int block_id);
    void* v_ptr(int layer, int block_id);

    // Sparse decode attention (attention.sparse_topk_tokens): optional per-block
    // key min/max metadata pool. Layout per (layer, block): n_kv_heads * head_dim
    // half2 pairs, (min, max) interleaved. Scalar-geometry pools only (the
    // per-layer ctor refuses). Returns false when ineligible or the allocation
    // fails; every other behaviour is then unchanged.
    bool enable_key_minmax();
    bool key_minmax_enabled() const { return minmax_pool_ != nullptr; }
    void* key_minmax_ptr(int layer, int block_id);

    // INT8/INT4/NVFP4/MXFP4_KV per-head scale access (nullptr if not applicable).
    // NVFP4/MXFP4_KV: 1 scale byte per kNVFP4Group=16 FP4 elems along head_dim.
    void* k_scale_ptr(int layer, int block_id);
    void* v_scale_ptr(int layer, int block_id);
    size_t scale_block_bytes() const;
    // Per-layer scale block bytes (used by NVFP4 in per-layer mode where head_dim varies).
    // Returns scale_block_bytes_ for the standard path.
    size_t scale_block_bytes(int layer) const;

    // Whole-block D2D copy across all layers (+ scale regions when present):
    // dsts[i] becomes a byte-identical copy of srcs[i]. One kernel launch,
    // pairs passed by value (no H2D staging). Multi-candidate spec-verify
    // staging (speculative.token_recycling, route (a)): each candidate gets
    // a private copy of the committed partial block, the winner's block is
    // copied back. SWA layer groups are skipped (separate id space; the
    // multi-candidate route excludes SWA models).
    static constexpr int kCopyMaxPairs = 16;
    void copy_blocks_device(const int* srcs, const int* dsts, int n_pairs, cudaStream_t stream);

    // Generic batched D2D copy: one kernel launch executes n independent
    // {src, dst, bytes} copies (SWA window snapshot pack/restore). Descs must
    // be device-resident; 16-byte-aligned fast path, byte fallback otherwise.
    struct CopyDesc {
        const void* src;
        void* dst;
        size_t bytes;
    };
    static void batched_copy_device(const CopyDesc* d_descs, int n, cudaStream_t stream);

    // Capacity queries
    int num_free_blocks() const;
    int total_blocks() const;

    // Accessors
    size_t block_bytes() const;
    // Per-layer block bytes (per-layer ctor); falls back to the scalar.
    size_t block_bytes(int layer) const {
        return layer_block_bytes_.empty() ? block_bytes_
                                          : layer_block_bytes_[static_cast<size_t>(layer)];
    }
    int block_size() const { return block_size_; }
    int n_layers() const;
    int n_kv_heads() const;
    int head_dim() const;
    QType qtype() const;

private:
    int n_layers_;
    int n_kv_heads_;
    int head_dim_;
    int max_blocks_;
    int block_size_;          // tokens per block (default 16)
    QType dtype_;
    VRAMAllocator* alloc_ = nullptr;
    size_t block_bytes_;  // cached: block_size * n_kv_heads * head_dim * dtype_size(dtype)

    // Block ids + refcounts (A7 step 3). The pool owns the id space; the
    // MEMORY stays here, because the layout is layer-major — one id's bytes
    // are scattered across per-layer K/V regions of differing size, which a
    // uniform stride cannot express (BlockPool::open_slots).
    BlockPool blocks_;
    void* pool_ = nullptr;  // single contiguous GPU allocation (K+V)
    // Held only by a growable pool: the reservation whose committed prefix per
    // layer region grows. Empty for the fixed path, which keeps using pool_
    // from the allocator exactly as before.
    Region region_;
    bool growable_ = false;
    std::atomic<uint64_t> growths_{0};
    int committed_blocks_ = 0;  // blocks whose memory is backed in every layer
    // What may be handed out. Mirrored here rather than read from the block
    // pool because admission asks per pending request per step, and the pool's
    // accessor takes the lock that allocate/free contend for.
    std::atomic<int> usable_blocks_{0};

    int plan_growth_(int max_blocks, int ceiling_blocks);
    bool reserve_pool_(size_t total_bytes, int fixed_blocks);
    int commit_blocks_(int blocks);

    // Per-layer KV shapes and offsets (for Gemma 4 dual attention geometry).
    // If empty, all layers use the scalar n_kv_heads_/head_dim_/block_bytes_.
    std::vector<size_t> layer_block_bytes_;  // block_size * nkv[l] * hd[l] * dtype_size
    // Blocks layer l's own region holds. A sliding-window layer's region is
    // swa_max_blocks_, not max_blocks_, and anything that strides or writes
    // per layer has to respect that or it lands in the next layer's range.
    size_t layer_capacity_(int l) const {
        return (swa_max_blocks_ > 0 && l < static_cast<int>(layer_is_swa_.size()) && layer_is_swa_[l])
                   ? static_cast<size_t>(swa_max_blocks_)
                   : static_cast<size_t>(max_blocks_);
    }
    std::vector<size_t> layer_k_offset_;     // byte offset of layer l's K region in pool
    std::vector<size_t> layer_v_offset_;     // byte offset of layer l's V region in pool

    // Per-layer NVFP4 scale offsets (only populated when dtype==NVFP4 in per-layer ctor).
    // Each layer's scales region holds K scales then V scales, byte counts per block
    // computed from the layer's nkv*hd/16.
    std::vector<size_t> layer_scale_block_bytes_;
    std::vector<size_t> layer_k_scale_offset_;
    std::vector<size_t> layer_v_scale_offset_;

    // INT8/INT4/NVFP4/MXFP4_KV per-head scales.
    // Layout: 2x blocks per layer (K scales region + V scales region).
    void* scale_pool_ = nullptr;
    size_t scale_block_bytes_ = 0;  // block_size * n_kv_heads * sizeof(half)

    // Sparse decode attention key min/max metadata (see enable_key_minmax).
    void* minmax_pool_ = nullptr;
    size_t minmax_block_bytes_ = 0;  // n_kv_heads * head_dim * 2 * sizeof(half)

    // copy_blocks_device per-layer offset table (lazy device upload):
    // 6 size_t per layer {k_off, v_off, block_bytes, k_scale_off,
    // v_scale_off, scale_bytes}; offsets relative to pool_/scale_pool_.
    void* d_copy_meta_ = nullptr;

    // ── SWA block group state (per-layer ctor only) ──────────────────
    // layer_is_swa_[l] = 1 → layer l's region has swa_max_blocks_ capacity and
    // block ids passed to k_ptr/v_ptr for it come from the SWA id space.
    std::vector<char> layer_is_swa_;
    int swa_max_blocks_ = 0;
    BlockPool swa_blocks_;  // separate id space; SWA blocks are never shared
};

}  // namespace imp
