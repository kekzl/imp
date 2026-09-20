#pragma once

#include <cfloat>

namespace imp {

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARP_SIZE = 32;
// Top-k selection: thread `tid` owns experts tid + j*BLOCK_SIZE (kMaxModelExperts = 4096 slots
// per block). One expert per thread capped the search at index 255, so a 512-expert router
// (Qwen3.8-Flash-Next) never proposed the upper half.
static constexpr int kTopkSlotsPerThread = 4096 / BLOCK_SIZE;

__device__ __forceinline__ void topk_load_slots(const float* __restrict__ sel, int n_experts, int tid,
                                                float* vals) {
#pragma unroll
    for (int j = 0; j < kTopkSlotsPerThread; j++)
        vals[j] = (tid + j * BLOCK_SIZE < n_experts) ? sel[tid + j * BLOCK_SIZE] : -FLT_MAX;
}

// Thread-local argmax over the owned slots; the caller reduces across the warp/block.
__device__ __forceinline__ void topk_owned_argmax(const float* vals, int tid, float& wmax, int& widx) {
    wmax = vals[0];
    widx = tid;
#pragma unroll
    for (int j = 1; j < kTopkSlotsPerThread; j++) {
        if (vals[j] > wmax) {
            wmax = vals[j];
            widx = tid + j * BLOCK_SIZE;
        }
    }
}

}  // namespace imp
