#pragma once

#include "compute/warp_reduce.cuh"
#include <cuda_runtime.h>
#include <cfloat>

namespace imp {

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARP_SIZE = 32;

static constexpr int MAX_TOP_K = 128;

// Per-TU cleanup helpers for file-scope persistent scratch that is split across
// translation units. sampling_cleanup() (public) calls both.
void sampling_cleanup_cub();  // frees CUB sort scratch (sampling_topk_topp.cu)
void sampling_cleanup_dry();  // frees DRY penalty buffers (sampling_penalties.cu)

// Warp-level argmax reduction: returns the (value, index) of the maximum
// across all lanes in the warp.
__device__ __forceinline__ void warp_argmax(float& val, int& idx) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// Simple LCG random number generator for device code.
__device__ __forceinline__ unsigned int lcg_rand(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

// Convert LCG output to a float in [0, 1).
__device__ __forceinline__ float lcg_rand_float(unsigned int& state) {
    return static_cast<float>(lcg_rand(state)) / 4294967296.0f;
}

// Block-cooperative top-k selection. Each thread passes its own (unsorted) local
// candidate list; produces the block's global top_k (sorted desc, tie-break by
// smaller index) in out_val/out_idx (may be smem or global, written by thread 0).
// s_warp_vals/idxs: smem scratch of NUM_WARPS*top_k each. Caller must have all
// threads reach this with their local arrays populated.
__device__ __forceinline__ void block_reduce_topk(float* local_vals, int* local_idxs, int local_count,
                                                  int top_k, float* s_warp_vals, int* s_warp_idxs,
                                                  float* out_val, int* out_idx) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    // sort this thread's candidates descending (local_count is small)
    for (int i = 0; i < local_count - 1; ++i)
        for (int j = i + 1; j < local_count; ++j)
            if (local_vals[j] > local_vals[i]) {
                float tv = local_vals[i];
                local_vals[i] = local_vals[j];
                local_vals[j] = tv;
                int ti = local_idxs[i];
                local_idxs[i] = local_idxs[j];
                local_idxs[j] = ti;
            }

    // each warp produces a sorted top_k list via repeated warp-max extraction
    float* my_warp_vals = s_warp_vals + warp_id * top_k;
    int* my_warp_idxs = s_warp_idxs + warp_id * top_k;
    int my_ptr = 0;
    for (int ki = 0; ki < top_k; ++ki) {
        float bv = (my_ptr < local_count) ? local_vals[my_ptr] : -FLT_MAX;
        int bi = (my_ptr < local_count) ? local_idxs[my_ptr] : -1;
        int bl = lane_id;
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFF, bv, off);
            int oi = __shfl_xor_sync(0xFFFFFFFF, bi, off);
            int ol = __shfl_xor_sync(0xFFFFFFFF, bl, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
                bv = ov;
                bi = oi;
                bl = ol;
            }
        }
        if (lane_id == bl && my_ptr < local_count)
            my_ptr++;
        if (lane_id == 0) {
            my_warp_vals[ki] = bv;
            my_warp_idxs[ki] = bi;
        }
    }
    __syncthreads();

    // thread 0: k-way merge of the NUM_WARPS sorted lists into the block top_k
    if (tid == 0) {
        int ptrs[NUM_WARPS];
        for (int w = 0; w < NUM_WARPS; ++w)
            ptrs[w] = 0;
        for (int ki = 0; ki < top_k; ++ki) {
            float bv = -FLT_MAX;
            int bi = -1;
            int bw = 0;
            for (int w = 0; w < NUM_WARPS; ++w) {
                if (ptrs[w] < top_k) {
                    float v = s_warp_vals[w * top_k + ptrs[w]];
                    int idx = s_warp_idxs[w * top_k + ptrs[w]];
                    if (idx >= 0 && (v > bv || (v == bv && (bi < 0 || idx < bi)))) {
                        bv = v;
                        bi = idx;
                        bw = w;
                    }
                }
            }
            out_val[ki] = bv;
            out_idx[ki] = bi;
            if (bi >= 0)
                ptrs[bw]++;
        }
    }
}

}  // namespace imp
