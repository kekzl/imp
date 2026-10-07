#pragma once

// Device-driven expert cache for host-resident NVFP4 MoE experts (decode, n = 1).
//
// The host LRU path (expert_cache.h) needs the routing on the host: one D2H + stream sync
// per MoE layer, then host bookkeeping, then the miss copies. Here the whole step runs on the
// device: a resolve kernel reads the routed expert ids, looks them up in per-layer tables
// (slot_of[expert], key_of_slot[slot], LRU stamps), picks victims for the misses and writes
// the slot indices and per-slot scales the fused decode kernels read; a gather kernel then
// copies the missing experts from MAPPED pinned host memory (zero-copy over PCIe,
// 51 GB/s in tools/analysis/h2d_gather_probe.cu). No host round trip, capture-safe.
//
// The pool, its slot layout and the per-slot scale mirror are the host cache's; the two
// paths hand the pool back and forth through generations: whoever finds the other's
// generation moved invalidates its own view (every slot empty) before touching a slot.
//
// Tiered layers (experts not pinned: host RAM short, #2621): sources live in a HostExpertTier,
// an LRU of pinned mapped units read from the checkpoint with parallel pread. The resolve
// kernel writes misses with their key and posts a mailbox request, a host thread fills the
// source pointers and acks, the gather runs. Exclusive: a unit lives in VRAM or in the tier;
// the gather writes each victim slot back into the tier before it overwrites it.

#include "exec/expert_cache.h"
#include "exec/moe_workspace.h"
#include "memory/host_expert_tier.h"
#include "memory/host_pinned.h"

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace imp {

class Model;
class VRAMAllocator;

struct DevExpertSrc {
    const char* packed;  // device view of the pinned packed FP4 block
    const char* ms;      // device view of the pinned FP8 micro-scale block
    float scale;         // per-tensor scale
};

struct DevMissEntry {
    const char* packed;
    const char* ms;
    char* dst;        // slot base in the pool
    char* wb_packed;  // tiered: host unit the slot's old occupant is written back to, or null
    char* wb_ms;
    int32_t key;         // proj * n_experts + expert (tiered layers: the tier thread fills packed/ms)
    int32_t victim_key;  // old occupant of dst, -1 = empty slot
};

// One layer's tables, passed to the kernels by value.
struct DevExpertLayer {
    const DevExpertSrc* src;     // [3 * n_experts]
    int32_t* slot_of;            // [3 * n_experts], -1 = not resident
    int32_t* key_of_slot;        // [slots], -1 = empty
    uint32_t* stamp;             // [slots], LRU stamp (step clock at last use)
    uint32_t* clock;             // [1]
    unsigned long long* stats;   // [2]: hits, misses
    char* pool;                  // layer pool base
    float* slot_scales;          // [slots]
    size_t slot_bytes;
    int n_experts;
    int slots;
};

inline constexpr int kDevExpertMaxTopK = 32;  // 3 * top_k <= 96 entries per step

// GPU <-> host handshake for tiered decode layers, in mapped pinned memory. The resolve kernel
// posts req_seq; the tier thread fills the misses' sources and acks; a one-thread kernel spins
// on ack_seq before the gather. Replaces a graph host node (~1 ms per layer under WDDM).
struct TierMailbox {
    uint32_t req_seq;  // GPU-written
    int32_t layer;
    uint32_t pad0[14];
    uint32_t ack_seq;  // host-written, own cache line
    uint32_t pad1[15];
};

class DeviceExpertCache {
public:
    DeviceExpertCache() = default;
    DeviceExpertCache(const DeviceExpertCache&) = delete;
    DeviceExpertCache& operator=(const DeviceExpertCache&) = delete;
    ~DeviceExpertCache() { destroy(); }

    // Builds the per-layer tables for every MoE layer whose experts are host-resident,
    // NVFP4-promoted and pinned in a MAPPED slab of `model`. Unpinned layers go to the host
    // expert tier when set_host_pool_mib() >= 0 (0 = host RAM available minus max(6 GiB, 10 % RAM), else the
    // cap). False (and no layer ready) when no layer qualifies; the host path then serves.
    [[nodiscard]] bool init(const Model& model, ExpertLRUCache& cache, VRAMAllocator* alloc, int top_k);
    // moe.host_expert_pool_mib, read by the next init(): -1 = no tier (default).
    void set_host_pool_mib(int mib) { host_pool_mib_ = mib; }
    void destroy();

    [[nodiscard]] bool layer_ready(int layer) const {
        return layer >= 0 && layer < static_cast<int>(layers_.size()) && layers_[layer].src;
    }
    // Resolves the layer's routed experts to slots (staging the misses), on `stream`:
    // writes 3 * top_k slot indices to `slot_idx_out` (gate, up, down blocks) and the
    // per-slot scales, then copies the misses into the pool. Takes the pool over from the
    // host path if that one moved since the last call.
    void resolve_and_stage(int layer, const int32_t* expert_indices, int top_k,
                           int32_t* slot_idx_out, cudaStream_t stream);
    // Takes the pool over from the host LRU path if that one filled slots since the last
    // call (forgets every slot, on `stream`). resolve_and_stage does this itself when it
    // runs on the host; a graph replay does not, so the engine calls it before each replayed
    // decode step. No-op when nothing moved or no layer is ready.
    void take_over(cudaStream_t stream);

    // Prefill staging on `stream`: copies every expert the routing touched
    // (expert_offsets[e + 1] > expert_offsets[e]) for all three projections from the mapped
    // pinned slabs into the layer stage buffer (projection p: packed at
    // buf + p * proj_bytes + e * pb, micro-scales at buf + p * proj_bytes + n_experts * pb + e * mb).
    // Untouched experts keep stale bytes no grouped-GEMM group reads. False (nothing copied)
    // when the layer is not ready or the layout differs from the cache's.
    [[nodiscard]] bool stage_touched(int layer, const int32_t* expert_offsets, char* stage_buf, size_t proj_bytes,
                       size_t pb, size_t mb, cudaStream_t stream);
    // Block form: experts [e0, e0 + n) of projection proj0 into slot 0 and proj1 (-1 = none) into
    // slot 1, block-local index (packed at slot + i * pb, micro-scales at slot + n * pb + i * mb).
    [[nodiscard]] bool stage_touched_range(int layer, const int32_t* expert_offsets, char* stage_buf, size_t proj_bytes,
                             size_t pb, size_t mb, int e0, int n, int proj0, int proj1, cudaStream_t stream);

    // Per-projection micro-scale offset within a slot (same layout the host path uses).
    size_t ms_off(int layer, int proj) const { return ms_off_[static_cast<size_t>(layer) * 3 + proj]; }

    [[nodiscard]] bool layer_tiered(int layer) const {
        return layer >= 0 && layer < static_cast<int>(tiered_.size()) && tiered_[layer];
    }

private:
    void invalidate_(cudaStream_t stream);
    enum class LayerSource { None, Pinned, Tier };
    LayerSource layer_sources_(const Model& model, int l, size_t slot_size, std::vector<DevExpertSrc>& src);
    [[nodiscard]] bool layer_layout_uniform_(size_t b) const;
    [[nodiscard]] bool init_tier_(const Model& model, const std::vector<char>& tiered, int top_k,
                                  int host_pool_mib);
    void tier_fill_(int layer);
    void tier_serve_();  // mailbox thread: one request per tiered decode layer, in order
    // Tiered prefill staging: makes the touched units of [e0, e0 + n) x {p0, p1} (p1 = -1: none;
    // p0 = -1: all three) resident and returns the device view of a source table, or null.
    const DevExpertSrc* tier_stage_src_(int layer, const int32_t* expert_offsets, int e0, int n, int p0,
                                        int p1, cudaStream_t stream);

    ExpertLRUCache* cache_ = nullptr;
    VRAMAllocator* alloc_ = nullptr;
    std::vector<DevExpertLayer> layers_;   // [n_layers], src == nullptr where not ready
    std::vector<size_t> ms_off_;           // [n_layers * 3]
    std::vector<size_t> packed_bytes_;     // [n_layers * 3]
    std::vector<size_t> ms_bytes_;         // [n_layers * 3]
    void* tables_ = nullptr;               // one device allocation for every table
    size_t tables_bytes_ = 0;
    DevMissEntry* d_misses_ = nullptr;     // [3 * kDevExpertMaxTopK]
    int* d_n_miss_ = nullptr;
    unsigned long long* d_stats_ = nullptr;  // [2]
    uint64_t seen_host_generation_ = 0;
    int layers_ready_ = 0;

    // Host expert tier (tiered layers only).
    std::vector<char> tiered_;  // [n_layers]
    std::unique_ptr<HostExpertTier> tier_;
    PinnedBuffer tier_arena_;              // mapped units
    PinnedBuffer tier_io_;                 // mapped: misses, n_miss, stage source table, mailbox
    DevMissEntry* tier_misses_ = nullptr;  // host view
    int* tier_n_miss_ = nullptr;
    DevExpertSrc* tier_stage_src_h_ = nullptr;
    TierMailbox* tier_mb_ = nullptr;
    DevMissEntry* tier_misses_d_ = nullptr;  // device views
    int* tier_n_miss_d_ = nullptr;
    DevExpertSrc* tier_stage_src_d_ = nullptr;
    TierMailbox* tier_mb_d_ = nullptr;
    uint32_t* d_tier_seq_ = nullptr;  // device: last posted request
    std::thread tier_thread_;
    std::atomic<bool> tier_stop_{false};
    int n_experts_ = 0;
    int host_pool_mib_ = -1;
    struct TierTiming {
        unsigned long long calls = 0;
        double busy_ms = 0, gap_ms = 0;  // serving requests / between consecutive requests
    };
    TierTiming tier_timing_;
};

}  // namespace imp
