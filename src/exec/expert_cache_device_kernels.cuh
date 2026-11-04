#pragma once

// Device helpers shared by the expert cache resolve kernel (expert_cache_device.cu) and the
// next-layer prefetch resolve (expert_cache_prefetch.cu): LRU victim scan and miss commit.

#include "exec/expert_cache_device.h"

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

namespace imp::expert_cache_kernels {

constexpr int kResolveThreads = 256;
constexpr int kMaxEntries = 3 * kDevExpertMaxTopK * kDevExpertMaxRows;
constexpr int kResolveSmemSlots = 4096;  // 16 KiB of stamps; larger layers scan global memory

// (stamp, slot) lexicographic min: the least recently used slot, ties to the lower index.
__device__ __forceinline__ void argmin_merge(uint32_t& bestv, int& best, uint32_t ov, int ob) {
    if (ob >= 0 && (best < 0 || ov < bestv || (ov == bestv && ob < best))) {
        bestv = ov;
        best = ob;
    }
}

__device__ __forceinline__ void warp_argmin(uint32_t& bestv, int& best) {
    for (int off = 16; off > 0; off >>= 1) {
        const uint32_t ov = __shfl_xor_sync(0xffffffffu, bestv, off);
        const int ob = __shfl_xor_sync(0xffffffffu, best, off);
        argmin_merge(bestv, best, ov, ob);
    }
}

// Block-wide LRU victim among slots not stamped this step; the result is valid on thread 0.
// Warp shuffles, then one 8-entry pass: 1 barrier (the shared-memory tree took 9, 11.8 us per
// Flash-Next layer).
__device__ __forceinline__ int lru_victim(const DevExpertLayer& L, const uint32_t* s_stamp,
                                          bool stamps_in_smem, uint32_t clock, int* s_best,
                                          uint32_t* s_bestv) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int best = -1;
    uint32_t bestv = UINT_MAX;
    for (int s = tid; s < L.slots; s += kResolveThreads) {
        const uint32_t v = stamps_in_smem ? s_stamp[s] : L.stamp[s];
        if (v != clock && v < bestv) {
            bestv = v;
            best = s;
        }
    }
    warp_argmin(bestv, best);
    if (lane == 0) {
        s_best[warp] = best;
        s_bestv[warp] = bestv;
    }
    __syncthreads();
    if (warp == 0) {
        best = lane < kResolveThreads / 32 ? s_best[lane] : -1;
        bestv = lane < kResolveThreads / 32 ? s_bestv[lane] : UINT_MAX;
        warp_argmin(bestv, best);
    }
    return best;
}

// Miss bookkeeping for entry i, one thread per miss: victims are distinct slots and their old keys
// are resident, so no miss key aliases another's old key, and the dependent loads (key_of_slot,
// src) overlap instead of queueing on thread 0.
__device__ __forceinline__ void commit_miss(const DevExpertLayer& L, int key, int victim, int ord,
                                            uint32_t clock, DevMissEntry* misses) {
    const int old = L.key_of_slot[victim];
    const DevExpertSrc src = L.src[key];
    if (old >= 0)
        L.slot_of[old] = -1;
    L.slot_of[key] = victim;
    L.key_of_slot[victim] = key;
    L.stamp[victim] = clock;
    if (L.slot_scales)
        L.slot_scales[victim] = src.scale;
    misses[ord] = DevMissEntry{src.packed, src.ms,  L.pool + (static_cast<size_t>(victim) * L.slot_bytes),
                               nullptr,    nullptr, key,
                               old};
}

}  // namespace imp::expert_cache_kernels
