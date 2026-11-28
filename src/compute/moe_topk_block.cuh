#pragma once

// One block's softmax + top-k of one router row (BLOCK_SIZE threads): the code topk_gating_kernel runs,
// shared with the device expert cache resolve kernel (expert_cache_device.cu), which routes a decode row
// itself so the top-k launch drops out (B-59). One source, so both produce the same bits.

#include "compute/moe_routing_internal.cuh"
#include "compute/warp_reduce.cuh"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace imp {

// Softmax gating (Mixtral, DeepSeek, etc.) into s_probs; s_warp holds BLOCK_SIZE / WARP_SIZE floats.
__device__ __forceinline__ void gating_softmax_probs(const float* __restrict__ logits, int n_experts,
                                                     float* s_probs, float* s_warp) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    // Step 1: Find max for numerical stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < n_experts; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[i]);
    }
    local_max = warp_reduce_max(local_max);

    if (lane_id == 0)
        s_warp[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0)
            s_warp[0] = val;
    }
    __syncthreads();
    float gmax = s_warp[0];
    // Every warp must read the max before warp 0 overwrites slot 0 with its partial sum:
    // without this barrier a late warp took a sum as the max and top-k picked wrong experts.
    __syncthreads();

    // Step 2: Compute exp and sum for softmax
    float local_sum = 0.0f;
    for (int i = tid; i < n_experts; i += blockDim.x) {
        local_sum += expf(logits[i] - gmax);
    }
    local_sum = warp_reduce_sum(local_sum);

    if (lane_id == 0)
        s_warp[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0)
            s_warp[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_warp[0];

    // Step 3: Compute softmax probabilities
    for (int i = tid; i < n_experts; i += blockDim.x) {
        s_probs[i] = expf(logits[i] - gmax) * inv_sum;
    }
}

// Parallel top-k over sel[n_experts] (shared memory, written before a barrier): block-wide argmax,
// record, mask out, top_k times. s_topk_val[k] = unbiased[idx] when unbiased is set (score_bias:
// select on biased scores, weight with unbiased ones), else the selected score.
__device__ __forceinline__ void topk_select_block(const float* sel, const float* unbiased, int n_experts,
                                                  int top_k, int32_t* s_topk_idx, float* s_topk_val) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp_max[NUM_WARPS];
    __shared__ int s_warp_argmax[NUM_WARPS];
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Strided ownership tid + j*BLOCK_SIZE (moe_routing_internal.cuh): one expert per thread
    // capped the search at 255 on the 512-expert Qwen3.8-Flash-Next router.
    float my_vals[kTopkSlotsPerThread];
    topk_load_slots(sel, n_experts, tid, my_vals);

    for (int k = 0; k < top_k; k++) {
        float wmax;
        int widx;
        topk_owned_argmax(my_vals, tid, wmax, widx);
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float other_val = __shfl_xor_sync(0xFFFFFFFF, wmax, off);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, widx, off);
            if (other_val > wmax) {
                wmax = other_val;
                widx = other_idx;
            }
        }

        // Write per-warp results
        if (lane_id == 0) {
            s_warp_max[warp_id] = wmax;
            s_warp_argmax[warp_id] = widx;
        }
        __syncthreads();

        // First warp reduces across all warps
        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? s_warp_max[lane_id] : -FLT_MAX;
            int ix = (lane_id < NUM_WARPS) ? s_warp_argmax[lane_id] : -1;
#pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                float ov = __shfl_xor_sync(0xFFFFFFFF, v, off);
                int oi = __shfl_xor_sync(0xFFFFFFFF, ix, off);
                if (ov > v) {
                    v = ov;
                    ix = oi;
                }
            }
            if (lane_id == 0) {
                s_topk_idx[k] = ix;
                s_topk_val[k] = unbiased ? unbiased[ix] : v;
            }
        }
        __syncthreads();

        // Mask out the selected expert so it won't be picked again
        topk_clear_slot(my_vals, s_topk_idx[k], tid);
    }
}

// Thread 0: normalise the selected weights and write one token's row.
__device__ __forceinline__ void gating_write_topk(const int32_t* s_topk_idx, const float* s_topk_val,
                                                  int top_k, bool normalize_weights, int32_t* indices_row,
                                                  float* weights_row) {
    float multiplier = 1.0f;
    if (normalize_weights) {
        float norm = 0.0f;
        for (int j = 0; j < top_k; ++j)
            norm += s_topk_val[j];
        multiplier = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
    }
    for (int j = 0; j < top_k; ++j) {
        indices_row[j] = s_topk_idx[j];
        weights_row[j] = s_topk_val[j] * multiplier;
    }
}

}  // namespace imp
