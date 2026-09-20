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

#include "exec/expert_cache.h"
#include "exec/moe_workspace.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
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
    char* dst;  // slot base in the pool
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

class DeviceExpertCache {
public:
    DeviceExpertCache() = default;
    DeviceExpertCache(const DeviceExpertCache&) = delete;
    DeviceExpertCache& operator=(const DeviceExpertCache&) = delete;
    ~DeviceExpertCache() { destroy(); }

    // Builds the per-layer tables for every MoE layer whose experts are host-resident,
    // NVFP4-promoted and pinned in a MAPPED slab of `model`. False (and no layer ready)
    // when any of that does not hold; the host path then serves the layer.
    bool init(const Model& model, ExpertLRUCache& cache, VRAMAllocator* alloc, int top_k);
    void destroy();

    bool layer_ready(int layer) const {
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
    bool stage_touched(int layer, const int32_t* expert_offsets, char* stage_buf, size_t proj_bytes,
                       size_t pb, size_t mb, cudaStream_t stream);

    // Per-projection micro-scale offset within a slot (same layout the host path uses).
    size_t ms_off(int layer, int proj) const { return ms_off_[static_cast<size_t>(layer) * 3 + proj]; }

private:
    void invalidate_(cudaStream_t stream);

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
};

}  // namespace imp
