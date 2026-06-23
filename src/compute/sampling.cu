#include "compute/sampling.h"
#include "compute/warp_reduce.cuh"
#include "core/logging.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cub/device/device_topk.cuh>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <vector>

namespace imp {

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARP_SIZE = 32;

// ============================================================================
// Greedy sampling (argmax)
// ============================================================================

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

// Single-block argmax kernel (fallback for paths without pre-allocated scratch).
__global__ void argmax_kernel(const float* __restrict__ logits, int vocab_size,
                              int32_t* __restrict__ d_result) {
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    warp_argmax(local_max, local_idx);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_val[NUM_WARPS];
    __shared__ int s_idx[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_val[lane_id] : -FLT_MAX;
        int idx = (lane_id < NUM_WARPS) ? s_idx[lane_id] : 0;

#pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
            if (other_val > val || (other_val == val && other_idx < idx)) {
                val = other_val;
                idx = other_idx;
            }
        }

        if (lane_id == 0) {
            d_result[0] = static_cast<int32_t>(idx);
        }
    }
}

// Multi-block argmax: distributes work across ARGMAX_NBLOCKS blocks so all SMs
// participate.  The single-block kernel above uses 1 SM and takes ~190 us for
// vocab=152K; this version takes ~10 us.
//
// Scratch layout (passed as d_scratch, ARGMAX_SCRATCH_BYTES total):
//   float    partial_vals [ARGMAX_NBLOCKS]
//   int32_t  partial_idxs [ARGMAX_NBLOCKS]

// Phase 1: each block scans its stripe and writes its local max to partials.
__global__ void argmax_partial_kernel(const float* __restrict__ logits, int vocab_size,
                                      float* __restrict__ partial_vals, int32_t* __restrict__ partial_idxs) {
    float local_max = -FLT_MAX;
    int local_idx = 0;

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < vocab_size; i += stride) {
        float v = logits[i];
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    warp_argmax(local_max, local_idx);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_val[NUM_WARPS];
    __shared__ int s_idx[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_val[lane_id] : -FLT_MAX;
        int idx = (lane_id < NUM_WARPS) ? s_idx[lane_id] : 0;

#pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
            if (other_val > val || (other_val == val && other_idx < idx)) {
                val = other_val;
                idx = other_idx;
            }
        }

        if (lane_id == 0) {
            partial_vals[blockIdx.x] = val;
            partial_idxs[blockIdx.x] = idx;
        }
    }
}

// Phase 2: single block reduces ARGMAX_NBLOCKS partial results.
__global__ void argmax_reduce_kernel(const float* __restrict__ partial_vals,
                                     const int32_t* __restrict__ partial_idxs, int n_blocks,
                                     int32_t* __restrict__ d_result) {
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        float v = partial_vals[i];
        int idx = partial_idxs[i];
        if (v > local_max || (v == local_max && idx < local_idx)) {
            local_max = v;
            local_idx = idx;
        }
    }

    warp_argmax(local_max, local_idx);

    if (threadIdx.x == 0) {
        d_result[0] = static_cast<int32_t>(local_idx);
    }
}

int32_t sample_greedy(const Tensor& logits, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    int32_t* d_result = nullptr;
    if (cudaMalloc(&d_result, sizeof(int32_t)) != cudaSuccess) {
        IMP_LOG_ERROR("sample_greedy: cudaMalloc failed");
        return 0;
    }

    argmax_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, d_result);

    int32_t h_result = 0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    IMP_CUDA_CHECK_LOG(cudaFree(d_result));
    return h_result;
}

int32_t sample_greedy(const Tensor& logits, int32_t* d_result, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    // Use multi-block argmax: scratch lives right after d_result.
    // Layout: [result(4B)] [partial_vals(ARGMAX_NBLOCKS*4B)] [partial_idxs(ARGMAX_NBLOCKS*4B)]
    auto* base = reinterpret_cast<char*>(d_result);
    auto* partial_vals = reinterpret_cast<float*>(base + sizeof(int32_t));
    auto* partial_idxs = reinterpret_cast<int32_t*>(base + sizeof(int32_t) + ARGMAX_NBLOCKS * sizeof(float));

    argmax_partial_kernel<<<ARGMAX_NBLOCKS, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, partial_vals,
                                                                     partial_idxs);
    argmax_reduce_kernel<<<1, WARP_SIZE, 0, stream>>>(partial_vals, partial_idxs, ARGMAX_NBLOCKS, d_result);

    int32_t h_result = 0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    return h_result;
}

// ============================================================================
// Top-k / Top-p (nucleus) sampling with temperature
// ============================================================================

// Simple LCG random number generator for device code.
__device__ __forceinline__ unsigned int lcg_rand(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

// Convert LCG output to a float in [0, 1).
__device__ __forceinline__ float lcg_rand_float(unsigned int& state) {
    return static_cast<float>(lcg_rand(state)) / 4294967296.0f;
}

// ============================================================================
// Multi-block top-k + top-p sampling (two phases).
//
// The original single-block kernel ran <<<1, BLOCK_SIZE>>>, using 1 of 170 SMs,
// and scanned the full vocab (~150k) three times — ~737 us/call, the #1 GPU
// consumer in batched server decode (profiled 2026-06-23). This splits the
// full-vocab work across SAMPLE_NBLOCKS blocks.
//
// Phase 1 (multi-block): each block scans a strided vocab subset and emits
//   block_max, block_sum (= sum exp((logit-block_max)*invT)), and the block's
//   top_k *logits* (sorted desc) into candidate scratch. Candidates store
//   logits, not probabilities, so the global softmax can be applied in phase 2.
// Phase 2 (single block): merges block partials into the global max/sum via the
//   online-softmax rescale, k-way merges the SAMPLE_NBLOCKS candidate lists into
//   the global top_k, converts to probabilities, applies top-p and samples with
//   the same LCG as before. Distribution-identical to the old kernel (not
//   bit-identical: reduction order differs).
// ============================================================================

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

// Phase 1: per-block max, sum, and top_k logit candidates over a strided subset.
__global__ void topk_partial_kernel(const float* __restrict__ logits, int vocab_size, int top_k,
                                    float inv_temperature, float* __restrict__ block_max_out,
                                    float* __restrict__ block_sum_out, float* __restrict__ cand_val_out,
                                    int* __restrict__ cand_idx_out) {
    extern __shared__ char smem_raw[];
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    float* s_reduce = reinterpret_cast<float*>(smem_raw);  // BLOCK_SIZE
    float* s_gmax = s_reduce + BLOCK_SIZE;                 // 1
    float* s_gsum = s_gmax + 1;                            // 1
    float* s_warp_vals = s_gsum + 1;                       // NUM_WARPS * top_k
    int* s_warp_idxs = reinterpret_cast<int*>(s_warp_vals + NUM_WARPS * top_k);

    const int tid = threadIdx.x;
    const int gstride = blockDim.x * gridDim.x;
    const int gstart = blockIdx.x * blockDim.x + tid;

    // ---- block max over strided subset ----
    float local_max = -FLT_MAX;
    for (int i = gstart; i < vocab_size; i += gstride) {
        float v = logits[i];
        if (v > local_max)
            local_max = v;
    }
    local_max = warp_reduce_max(local_max);
    s_reduce[tid] = -FLT_MAX;
    __syncthreads();
    if (tid % WARP_SIZE == 0)
        s_reduce[tid / WARP_SIZE] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; ++w)
            if (s_reduce[w] > mx)
                mx = s_reduce[w];
        s_gmax[0] = mx;
    }
    __syncthreads();
    float bmax = s_gmax[0];

    // ---- block sum exp((logit-bmax)*invT) ----
    float local_sum = 0.0f;
    if (bmax > -FLT_MAX) {
        for (int i = gstart; i < vocab_size; i += gstride)
            local_sum += expf((logits[i] - bmax) * inv_temperature);
    }
    local_sum = warp_reduce_sum(local_sum);
    s_reduce[tid] = 0.0f;
    __syncthreads();
    if (tid % WARP_SIZE == 0)
        s_reduce[tid / WARP_SIZE] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < NUM_WARPS; ++w)
            sm += s_reduce[w];
        block_max_out[blockIdx.x] = bmax;
        block_sum_out[blockIdx.x] = sm;
    }

    // ---- per-thread top_k by logit over the subset ----
    int local_k = min(top_k, SAMPLE_MAX_TOP_K);
    float local_vals[SAMPLE_MAX_TOP_K];
    int local_idxs[SAMPLE_MAX_TOP_K];
    int local_count = 0;
    float local_min_val = -FLT_MAX;
    int local_min_pos = 0;
    for (int i = gstart; i < vocab_size; i += gstride) {
        float v = logits[i];
        if (local_count < local_k) {
            local_vals[local_count] = v;
            local_idxs[local_count] = i;
            local_count++;
            if (local_count == local_k) {
                local_min_val = local_vals[0];
                local_min_pos = 0;
                for (int j = 1; j < local_k; ++j)
                    if (local_vals[j] < local_min_val) {
                        local_min_val = local_vals[j];
                        local_min_pos = j;
                    }
            }
        } else if (v > local_min_val) {
            local_vals[local_min_pos] = v;
            local_idxs[local_min_pos] = i;
            local_min_val = local_vals[0];
            local_min_pos = 0;
            for (int j = 1; j < local_k; ++j)
                if (local_vals[j] < local_min_val) {
                    local_min_val = local_vals[j];
                    local_min_pos = j;
                }
        }
    }
    // reduce per-thread candidates into the block's top_k (sorted desc) → scratch
    block_reduce_topk(local_vals, local_idxs, local_count, top_k, s_warp_vals, s_warp_idxs,
                      cand_val_out + static_cast<size_t>(blockIdx.x) * top_k,
                      cand_idx_out + static_cast<size_t>(blockIdx.x) * top_k);
}

// Phase 2: merge the per-block candidate lists into the global top_k, apply
// top-p and sample. All threads cooperate to select the global top_k from the
// candidate pool (block_reduce_topk over the SAMPLE_NBLOCKS*top_k candidates read
// straight from global — coalesced, no big smem staging); only the final
// top-p/sample over top_k entries is serial. Runs inside graph capture.
__global__ void topk_finalize_kernel(int top_k, float top_p, float inv_temperature, unsigned int seed,
                                     int n_blocks, const float* __restrict__ block_max_in,
                                     const float* __restrict__ block_sum_in,
                                     const float* __restrict__ cand_val_in,
                                     const int* __restrict__ cand_idx_in, int32_t* __restrict__ d_result) {
    extern __shared__ char smem_raw[];
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    float* s_warp_vals = reinterpret_cast<float*>(smem_raw);  // NUM_WARPS * top_k
    int* s_warp_idxs = reinterpret_cast<int*>(s_warp_vals + NUM_WARPS * top_k);
    float* s_val = reinterpret_cast<float*>(s_warp_idxs + NUM_WARPS * top_k);  // top_k
    int* s_idx = reinterpret_cast<int*>(s_val + top_k);                        // top_k

    const int tid = threadIdx.x;
    const int n_cand = n_blocks * top_k;

    // each thread builds a local top_k over its slice of the candidate pool
    int local_k = min(top_k, SAMPLE_MAX_TOP_K);
    float local_vals[SAMPLE_MAX_TOP_K];
    int local_idxs[SAMPLE_MAX_TOP_K];
    int local_count = 0;
    float local_min_val = -FLT_MAX;
    int local_min_pos = 0;
    for (int i = tid; i < n_cand; i += blockDim.x) {
        int idx = cand_idx_in[i];
        if (idx < 0)
            continue;
        float v = cand_val_in[i];
        if (local_count < local_k) {
            local_vals[local_count] = v;
            local_idxs[local_count] = idx;
            local_count++;
            if (local_count == local_k) {
                local_min_val = local_vals[0];
                local_min_pos = 0;
                for (int j = 1; j < local_k; ++j)
                    if (local_vals[j] < local_min_val) {
                        local_min_val = local_vals[j];
                        local_min_pos = j;
                    }
            }
        } else if (v > local_min_val) {
            local_vals[local_min_pos] = v;
            local_idxs[local_min_pos] = idx;
            local_min_val = local_vals[0];
            local_min_pos = 0;
            for (int j = 1; j < local_k; ++j)
                if (local_vals[j] < local_min_val) {
                    local_min_val = local_vals[j];
                    local_min_pos = j;
                }
        }
    }
    block_reduce_topk(local_vals, local_idxs, local_count, top_k, s_warp_vals, s_warp_idxs, s_val, s_idx);

    if (tid != 0)
        return;

    // global max + online-softmax-merged global sum (block partials are tiny)
    float gmax = -FLT_MAX;
    for (int b = 0; b < n_blocks; ++b)
        if (block_max_in[b] > gmax)
            gmax = block_max_in[b];
    float gsum = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        float bm = block_max_in[b];
        if (bm > -FLT_MAX)
            gsum += block_sum_in[b] * expf((bm - gmax) * inv_temperature);
    }
    float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 1.0f;

    // logits -> probabilities (global softmax), top-p cutoff, renormalize, sample
    for (int i = 0; i < top_k; ++i) {
        if (s_idx[i] >= 0)
            s_val[i] = expf((s_val[i] - gmax) * inv_temperature) * inv_sum;
        else
            s_val[i] = 0.0f;
    }
    float cumsum = 0.0f;
    int cutoff = top_k;
    for (int i = 0; i < top_k; ++i) {
        cumsum += s_val[i];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    float norm = 0.0f;
    for (int i = 0; i < cutoff; ++i)
        norm += s_val[i];
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;

    unsigned int rng_state = seed;
    float r = lcg_rand_float(rng_state);
    float acc = 0.0f;
    int chosen = s_idx[0];
    for (int i = 0; i < cutoff; ++i) {
        acc += s_val[i] * inv_norm;
        if (r < acc) {
            chosen = s_idx[i];
            break;
        }
    }
    d_result[0] = static_cast<int32_t>(chosen);
}

// Launch the two-phase multi-block sampler. Scratch lives right after d_result
// (caller guarantees >= SAMPLE_SCRATCH_BYTES). Both kernels are graph-capturable
// (no allocation, fixed topology for a given vocab_size/top_k).
static void launch_topk_topp_multiblock(const float* d_logits, int vocab_size, int top_k, float top_p,
                                        float inv_temperature, unsigned int seed, int32_t* d_result,
                                        cudaStream_t stream) {
    char* base = reinterpret_cast<char*>(d_result);
    float* block_max = reinterpret_cast<float*>(base + sizeof(int32_t));
    float* block_sum = block_max + SAMPLE_NBLOCKS;
    float* cand_val = block_sum + SAMPLE_NBLOCKS;
    int* cand_idx = reinterpret_cast<int*>(cand_val + static_cast<size_t>(SAMPLE_NBLOCKS) * top_k);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    // both phases use NUM_WARPS*top_k warp-merge scratch; phase 1 also needs the
    // block reduction scratch, phase 2 the merged top_k output.
    size_t smem1 = static_cast<size_t>(BLOCK_SIZE) * sizeof(float) + 2 * sizeof(float) +
                   static_cast<size_t>(NUM_WARPS) * top_k * (sizeof(float) + sizeof(int));
    size_t smem2 = static_cast<size_t>(NUM_WARPS) * top_k * (sizeof(float) + sizeof(int)) +
                   static_cast<size_t>(top_k) * (sizeof(float) + sizeof(int));

    topk_partial_kernel<<<SAMPLE_NBLOCKS, BLOCK_SIZE, smem1, stream>>>(
        d_logits, vocab_size, top_k, inv_temperature, block_max, block_sum, cand_val, cand_idx);
    topk_finalize_kernel<<<1, BLOCK_SIZE, smem2, stream>>>(top_k, top_p, inv_temperature, seed,
                                                           SAMPLE_NBLOCKS, block_max, block_sum, cand_val,
                                                           cand_idx, d_result);
}

static constexpr int MAX_TOP_K = 128;

// ============================================================================
// CUB-based top-k sampling for k > MAX_TOP_K (128).
//
// Strategy:
//   1. Compute softmax probabilities with temperature scaling.
//   2. Sort (probability, vocab_index) pairs descending via CUB RadixSort.
//   3. Take first k elements, apply top-p cutoff, sample.
//
// This path is NOT used inside CUDA graph capture (CUB launches internal
// kernels). The single-block kernel above handles the graph-captured path.
// ============================================================================

// Kernel: compute softmax probabilities reading max/sum from device memory.
// d_max_sum[0] = global_max, d_max_sum[1] = sum. Avoids 2 D2H syncs.
__global__ void softmax_to_pairs_device_kernel(const float* __restrict__ logits, int vocab_size,
                                               float inv_temperature, const float* __restrict__ d_max_sum,
                                               float* __restrict__ d_keys, int32_t* __restrict__ d_values) {
    float global_max = d_max_sum[0];
    float inv_sum = 1.0f / d_max_sum[1];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;
    float prob = expf((logits[idx] - global_max) * inv_temperature) * inv_sum;
    d_keys[idx] = prob;
    d_values[idx] = idx;
}

// Kernel: find global max of logits (Phase 1)
__global__ void softmax_max_kernel(const float* __restrict__ logits, int vocab_size,
                                   float* __restrict__ d_max) {
    __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];

    float local_max = -FLT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab_size; i += blockDim.x * gridDim.x) {
        float v = logits[i];
        if (v > local_max)
            local_max = v;
    }
    local_max = warp_reduce_max(local_max);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0)
        s_max[warp_id] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; w++)
            if (s_max[w] > mx)
                mx = s_max[w];
        atomicMax(reinterpret_cast<int*>(d_max), __float_as_int(mx));
    }
}

// Kernel: compute sum of exp(logits - max) reading max from device memory.
// Avoids D2H sync between max and sum phases.
__global__ void softmax_sum_device_max_kernel(const float* __restrict__ logits, int vocab_size,
                                              float inv_temperature, const float* __restrict__ d_max,
                                              float* __restrict__ d_sum) {
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];

    float global_max = *d_max;
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab_size; i += blockDim.x * gridDim.x) {
        local_sum += expf((logits[i] - global_max) * inv_temperature);
    }
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0)
        s_sum[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sm = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; w++)
            sm += s_sum[w];
        atomicAdd(d_sum, sm);
    }
}

// Deterministic single-block variant of softmax_sum_device_max_kernel.
// A single block strides the whole vocab and reduces via a fixed-order
// shared-memory tree, writing the sum directly. This removes the cross-block
// FP atomicAdd of the multi-block path whose accumulation order varies
// run-to-run. Opt-in (deterministic mode) only.
__global__ void softmax_sum_device_max_single_block_kernel(const float* __restrict__ logits, int vocab_size,
                                                          float inv_temperature,
                                                          const float* __restrict__ d_max,
                                                          float* __restrict__ d_sum) {
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];

    float global_max = *d_max;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_sum += expf((logits[i] - global_max) * inv_temperature);
    }
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0)
        s_sum[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sm = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; w++)
            sm += s_sum[w];
        d_sum[0] = sm;
    }
}

// Kernel: top-p filter + sample from the first k sorted candidates
__global__ void topp_sample_from_sorted_kernel(const float* __restrict__ sorted_probs,
                                               const int32_t* __restrict__ sorted_indices, int top_k,
                                               float top_p, unsigned int seed,
                                               int32_t* __restrict__ d_result) {
    if (threadIdx.x != 0)
        return;

    float cumsum = 0.0f;
    int cutoff = top_k;
    for (int i = 0; i < top_k; i++) {
        cumsum += sorted_probs[i];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    float norm = 0.0f;
    for (int i = 0; i < cutoff; i++)
        norm += sorted_probs[i];
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;

    unsigned int rng_state = seed;
    float r = lcg_rand_float(rng_state);

    float acc = 0.0f;
    int32_t chosen = sorted_indices[0];
    for (int i = 0; i < cutoff; i++) {
        acc += sorted_probs[i] * inv_norm;
        if (r < acc) {
            chosen = sorted_indices[i];
            break;
        }
    }
    d_result[0] = chosen;
}

// Persistent scratch for CUB sort (lazily allocated, grows only)
struct CubSortScratch {
    float* d_keys_in = nullptr;
    float* d_keys_out = nullptr;
    int32_t* d_vals_in = nullptr;
    int32_t* d_vals_out = nullptr;
    float* d_max_sum = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    int capacity = 0;  // max vocab_size allocated for

    bool ensure(int vocab_size, cudaStream_t stream) {
        if (vocab_size <= capacity)
            return true;
        free();
        size_t elem_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
        size_t idx_bytes = static_cast<size_t>(vocab_size) * sizeof(int32_t);
        if (cudaMalloc(&d_keys_in, elem_bytes) != cudaSuccess ||
            cudaMalloc(&d_keys_out, elem_bytes) != cudaSuccess ||
            cudaMalloc(&d_vals_in, idx_bytes) != cudaSuccess ||
            cudaMalloc(&d_vals_out, idx_bytes) != cudaSuccess ||
            cudaMalloc(&d_max_sum, 2 * sizeof(float)) != cudaSuccess) {
            free();
            return false;
        }
        // Query CUB temp storage requirements: take max of full RadixSort
        // (fallback) and DeviceTopK (preferred) + a small RadixSort over the
        // top-K results (to produce sorted output for top-p).
        temp_bytes = 0;
        size_t rs_full_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, rs_full_bytes, d_keys_in, d_keys_out, d_vals_in,
                                                  d_vals_out, vocab_size, 0, 32, stream);
        size_t topk_bytes = 0;
        cub::DeviceTopK::MaxPairs(nullptr, topk_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out,
                                  vocab_size, vocab_size,
                                  ::cuda::execution::require(::cuda::execution::determinism::not_guaranteed,
                                                             ::cuda::execution::output_ordering::unsorted));
        size_t rs_topk_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, rs_topk_bytes, d_keys_in, d_keys_out, d_vals_in,
                                                  d_vals_out, vocab_size, 0, 32, stream);
        temp_bytes = std::max({rs_full_bytes, topk_bytes, rs_topk_bytes});
        if (cudaMalloc(&d_temp, temp_bytes) != cudaSuccess) {
            free();
            return false;
        }
        capacity = vocab_size;
        return true;
    }

    void free() {
        if (d_keys_in) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_keys_in));
            d_keys_in = nullptr;
        }
        if (d_keys_out) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_keys_out));
            d_keys_out = nullptr;
        }
        if (d_vals_in) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_vals_in));
            d_vals_in = nullptr;
        }
        if (d_vals_out) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_vals_out));
            d_vals_out = nullptr;
        }
        if (d_max_sum) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_max_sum));
            d_max_sum = nullptr;
        }
        if (d_temp) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_temp));
            d_temp = nullptr;
        }
        temp_bytes = 0;
        capacity = 0;
    }
};

static CubSortScratch s_cub_scratch;

// CUB-based top-k sampling for k > MAX_TOP_K.
static int32_t sample_topk_topp_cub(const float* d_logits, int vocab_size, int top_k, float top_p,
                                    float inv_temperature, unsigned int seed, int32_t* d_result,
                                    cudaStream_t stream) {
    if (!s_cub_scratch.ensure(vocab_size, stream)) {
        IMP_LOG_ERROR("CUB sort scratch allocation failed for vocab_size=%d", vocab_size);
        return 0;
    }

    auto& sc = s_cub_scratch;

    // Step 1: Compute softmax stats (max, then sum) entirely on device.
    // All intermediate values stay in d_max_sum — no D2H syncs needed.
    // d_max_sum[0] = global max, d_max_sum[1] = sum of exp.
    float neg_inf_val = -FLT_MAX;
    int neg_inf_bits;
    std::memcpy(&neg_inf_bits, &neg_inf_val, sizeof(int));
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(sc.d_max_sum, &neg_inf_bits, sizeof(int), cudaMemcpyHostToDevice, stream));
    float zero = 0.0f;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(sc.d_max_sum + 1, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));

    int stats_blocks = std::min((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 128);

    const bool deterministic = process_diag_deterministic_gemm();

    // Phase 1: global max (result in d_max_sum[0]). atomicMax on the int-bitcast
    // of the max is order-independent and exact, so it is already deterministic.
    softmax_max_kernel<<<stats_blocks, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, sc.d_max_sum);

    // Phase 2: sum of exp — reads max from device memory (no D2H sync).
    // The default multi-block kernel sums via cross-block FP atomicAdd, whose
    // accumulation order varies run-to-run. In deterministic mode use a single
    // block with a fixed-order tree reduction instead.
    if (deterministic) {
        softmax_sum_device_max_single_block_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            d_logits, vocab_size, inv_temperature, sc.d_max_sum, sc.d_max_sum + 1);
    } else {
        softmax_sum_device_max_kernel<<<stats_blocks, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size,
                                                                               inv_temperature, sc.d_max_sum,
                                                                               sc.d_max_sum + 1);
    }

    // Step 2: Compute probabilities reading max/sum from device memory (no D2H sync)
    int pair_blocks = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_to_pairs_device_kernel<<<pair_blocks, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size,
                                                                           inv_temperature, sc.d_max_sum,
                                                                           sc.d_keys_in, sc.d_vals_in);

    // Step 3: extract top_k via DeviceTopK (unsorted), then sort just those k.
    // Much faster than a full radix sort over the whole vocab when k << vocab.
    //
    // TODO(determinism): even in deterministic mode the candidate SET and the
    // subsequent descending radix sort by probability are reproducible (the
    // FP sum above is now fixed-order, and the probs are a pure function of the
    // logits), so the sampled token is reproducible for distinct
    // probabilities. The one residual gap is exact ties: DeviceTopK is invoked
    // with determinism::not_guaranteed and SortPairsDescending on equal keys is
    // not guaranteed stable on the int32 value, so two tokens with bit-identical
    // probability could swap order between runs. For a fully tie-stable top-k
    // here, fold the vocab index into the sort key (e.g. sort by (prob, -index))
    // or request cub determinism::guaranteed when this stochastic path needs
    // bit-exact reproducibility under ties. The single-block path
    // (top_k <= MAX_TOP_K) already tie-breaks by index and is fully
    // deterministic; this CUB path only runs for top_k > MAX_TOP_K (128).
    {
        size_t tk_bytes = sc.temp_bytes;
        cub::DeviceTopK::MaxPairs(sc.d_temp, tk_bytes, sc.d_keys_in, sc.d_keys_out, sc.d_vals_in,
                                  sc.d_vals_out, vocab_size, top_k,
                                  ::cuda::execution::require(::cuda::execution::determinism::not_guaranteed,
                                                             ::cuda::execution::output_ordering::unsorted));
        size_t rs_bytes = sc.temp_bytes;
        // In-place sort: copy outputs back to inputs as the source for the
        // small sort, write sorted result to outputs.
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(sc.d_keys_in, sc.d_keys_out, top_k * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(sc.d_vals_in, sc.d_vals_out, top_k * sizeof(int32_t),
                                           cudaMemcpyDeviceToDevice, stream));
        cub::DeviceRadixSort::SortPairsDescending(sc.d_temp, rs_bytes, sc.d_keys_in, sc.d_keys_out,
                                                  sc.d_vals_in, sc.d_vals_out, top_k, 0, 32, stream);
    }

    // Step 4: Top-p filter + sample from sorted top-k
    topp_sample_from_sorted_kernel<<<1, 1, 0, stream>>>(sc.d_keys_out, sc.d_vals_out, top_k, top_p, seed,
                                                        d_result);

    int32_t h_result = 0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    return h_result;
}

// Shared implementation for both sample_topk_topp overloads.
// When owns_result is true, d_result was allocated internally and will be freed.
static int32_t sample_topk_topp_impl(const float* d_logits, int vocab_size, int top_k, float top_p,
                                     float inv_temperature, unsigned int seed, int32_t* d_result,
                                     bool owns_result, cudaStream_t stream) {
    // For large top_k, use CUB radix sort path (no MAX_TOP_K limit)
    if (top_k > MAX_TOP_K) {
        int32_t result = sample_topk_topp_cub(d_logits, vocab_size, top_k, top_p, inv_temperature, seed,
                                              d_result, stream);
        if (owns_result)
            IMP_CUDA_CHECK_LOG(cudaFree(d_result));
        return result;
    }

    launch_topk_topp_multiblock(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, stream);

    int32_t h_result = 0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    if (owns_result)
        IMP_CUDA_CHECK_LOG(cudaFree(d_result));
    return h_result;
}

int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p, float temperature, unsigned int seed,
                         cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size)
        top_k = vocab_size;
    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    int32_t* d_result = nullptr;
    if (cudaMalloc(&d_result, SAMPLE_SCRATCH_BYTES) != cudaSuccess) {
        IMP_LOG_ERROR("sample_topk_topp: cudaMalloc failed");
        return 0;
    }

    return sample_topk_topp_impl(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, true,
                                 stream);
}

int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p, float temperature, unsigned int seed,
                         int32_t* d_result, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size)
        top_k = vocab_size;
    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    return sample_topk_topp_impl(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, false,
                                 stream);
}

// ===========================================================================
// Async (device-side) sampling — no host sync
// ===========================================================================

void sample_greedy_device(const Tensor& logits, int32_t* d_result, int32_t* h_mapped, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    // Multi-block argmax: scratch lives right after d_result.
    auto* base = reinterpret_cast<char*>(d_result);
    auto* partial_vals = reinterpret_cast<float*>(base + sizeof(int32_t));
    auto* partial_idxs = reinterpret_cast<int32_t*>(base + sizeof(int32_t) + ARGMAX_NBLOCKS * sizeof(float));

    argmax_partial_kernel<<<ARGMAX_NBLOCKS, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, partial_vals,
                                                                     partial_idxs);
    argmax_reduce_kernel<<<1, WARP_SIZE, 0, stream>>>(partial_vals, partial_idxs, ARGMAX_NBLOCKS, d_result);

    // Async copy to mapped pinned memory — no sync needed.
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_mapped, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
}

void sample_topk_topp_device(const Tensor& logits, int top_k, float top_p, float temperature,
                             unsigned int seed, int32_t* d_result, int32_t* h_mapped, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size)
        top_k = vocab_size;
    if (top_k > MAX_TOP_K) {
        IMP_LOG_WARN("top_k=%d exceeds MAX_TOP_K=%d, clamping", top_k, MAX_TOP_K);
        top_k = MAX_TOP_K;
    }
    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    launch_topk_topp_multiblock(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, stream);

    // Async copy to mapped pinned memory — no sync needed.
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_mapped, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
}

// ===========================================================================
// Repetition / frequency / presence penalties
// ===========================================================================

// Kernel: for each token in history, adjust its logit.
// Uses atomics to handle tokens appearing multiple times.
// Strategy: first count occurrences, then apply penalties.
// For simplicity with small history, we iterate the history per thread.
__global__ void apply_penalties_kernel(float* __restrict__ logits, const int32_t* __restrict__ token_ids,
                                       int n_tokens, int vocab_size, float repetition_penalty,
                                       float frequency_penalty, float presence_penalty) {
    // Each thread handles one vocab entry
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;

    // Count occurrences of this token in history
    int count = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (token_ids[i] == idx)
            count++;
    }
    if (count == 0)
        return;

    float logit = logits[idx];

    // Repetition penalty (multiplicative): divide positive, multiply negative
    if (repetition_penalty != 1.0f) {
        if (logit > 0.0f)
            logit /= repetition_penalty;
        else
            logit *= repetition_penalty;
    }

    // Frequency penalty (subtractive per-occurrence)
    logit -= frequency_penalty * static_cast<float>(count);

    // Presence penalty (subtractive binary)
    logit -= presence_penalty;

    logits[idx] = logit;
}

// Variant: reads n_tokens from a device pointer (for CUDA graph loop where count changes).
// repeat_last_n: when > 0, only scan the last N tokens in the history.
__global__ void apply_penalties_device_count_kernel(
    float* __restrict__ logits, const int32_t* __restrict__ token_ids,
    const int* __restrict__ d_n_tokens,  // [1] device-side token count
    int vocab_size, int repeat_last_n, float repetition_penalty, float frequency_penalty,
    float presence_penalty) {
    int n_tokens = *d_n_tokens;
    if (n_tokens <= 0)
        return;

    // Apply repeat_last_n window
    int start = 0;
    if (repeat_last_n > 0 && n_tokens > repeat_last_n) {
        start = n_tokens - repeat_last_n;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;

    int count = 0;
    for (int i = start; i < n_tokens; i++) {
        if (token_ids[i] == idx)
            count++;
    }
    if (count == 0)
        return;

    float logit = logits[idx];

    if (repetition_penalty != 1.0f) {
        if (logit > 0.0f)
            logit /= repetition_penalty;
        else
            logit *= repetition_penalty;
    }

    logit -= frequency_penalty * static_cast<float>(count);
    logit -= presence_penalty;

    logits[idx] = logit;
}

// Force a single token: set all logits to -inf except the given token.
// Used by think-budget to force </think> generation via logit manipulation.
__global__ void force_single_token_kernel(float* logits, int vocab_size, int32_t keep_token) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;
    logits[idx] = (idx == keep_token) ? 0.0f : -1e30f;
}

void force_single_token(float* logits, int vocab_size, int32_t keep_token, cudaStream_t stream) {
    int blocks = (vocab_size + 255) / 256;
    force_single_token_kernel<<<blocks, 256, 0, stream>>>(logits, vocab_size, keep_token);
}

void apply_penalties(float* logits, int vocab_size, const int32_t* token_ids, int n_tokens,
                     float repetition_penalty, float frequency_penalty, float presence_penalty,
                     cudaStream_t stream) {
    if (n_tokens == 0)
        return;
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return;

    int blocks = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_penalties_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(logits, token_ids, n_tokens, vocab_size,
                                                              repetition_penalty, frequency_penalty,
                                                              presence_penalty);
}

void apply_penalties_device_count(float* logits, int vocab_size, const int32_t* token_ids,
                                  const int* d_n_tokens, int repeat_last_n, float repetition_penalty,
                                  float frequency_penalty, float presence_penalty, cudaStream_t stream) {
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return;

    int blocks = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_penalties_device_count_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(logits, token_ids, d_n_tokens,
                                                                           vocab_size, repeat_last_n,
                                                                           repetition_penalty,
                                                                           frequency_penalty,
                                                                           presence_penalty);
}

// ===========================================================================
// min_p filtering
// ===========================================================================

// Single-kernel min_p: finds max logit via cooperative reduction, then
// filters tokens in logit space.  threshold = max_logit + log(min_p).
// No host sync or temp allocation needed.
__global__ void apply_min_p_kernel(float* __restrict__ logits, int vocab_size, float log_min_p) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_max[NUM_WARPS];
    __shared__ float s_threshold;

    const int tid = threadIdx.x;

    // Pass 1: find max logit (cooperative reduction)
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max)
            local_max = v;
    }
    local_max = warp_reduce_max(local_max);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0)
        s_max[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            if (s_max[w] > mx)
                mx = s_max[w];
        s_threshold = mx + log_min_p;
    }
    __syncthreads();

    // Pass 2: filter tokens below threshold
    float threshold = s_threshold;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[i] < threshold)
            logits[i] = -FLT_MAX;
    }
}

void apply_min_p(float* logits, int vocab_size, float min_p, cudaStream_t stream) {
    if (min_p <= 0.0f)
        return;

    float log_min_p = logf(min_p);
    apply_min_p_kernel<<<1, BLOCK_SIZE, 0, stream>>>(logits, vocab_size, log_min_p);
}

// ===========================================================================
// DRY (Don't Repeat Yourself) repetition penalty
// ===========================================================================

// File-scope persistent GPU buffers for DRY penalty application.
// Promoted from function-local statics so sampling_preallocate_dry() can
// pre-allocate them at engine init time and avoid cudaStreamSynchronize on
// first use during inference.
static int32_t* s_dry_tokens_buf = nullptr;
static float* s_dry_values_buf = nullptr;
static size_t s_dry_buf_cap = 0;

// Sparse penalty application kernel: subtracts penalty from each listed token.
__global__ void apply_dry_sparse_kernel(float* __restrict__ logits,
                                        const int32_t* __restrict__ penalty_tokens,
                                        const float* __restrict__ penalty_values, int n_penalties) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_penalties) {
        logits[penalty_tokens[idx]] -= penalty_values[idx];
    }
}

void apply_dry_penalty(float* d_logits, int vocab_size, const int32_t* host_token_ids, int n_tokens,
                       float multiplier, float base, int allowed_length, int penalty_last_n,
                       cudaStream_t stream) {
    if (multiplier <= 0.0f || n_tokens < 2)
        return;

    int search_start = (penalty_last_n > 0) ? std::max(0, n_tokens - penalty_last_n) : 0;

    // CPU: scan history for suffix matches, compute max match length per token.
    // Use a flat array indexed by token ID (no heap allocation) instead of unordered_map.
    // Reuse a static buffer to avoid per-call allocation for large vocab.
    static thread_local std::vector<int> match_buf;
    if (static_cast<int>(match_buf.size()) < vocab_size) {
        match_buf.assign(vocab_size, 0);
    }
    // Zero only entries we write (sparse clear is faster than memset for large vocab)
    std::vector<int32_t> touched_tokens;

    for (int pos = search_start; pos < n_tokens; pos++) {
        int match_len = 0;
        int a = pos - 1;
        int b = n_tokens - 1;
        while (a >= search_start && b >= 0 && host_token_ids[a] == host_token_ids[b]) {
            match_len++;
            a--;
            b--;
        }

        if (match_len > allowed_length) {
            int32_t token = host_token_ids[pos];
            if (token >= 0 && token < vocab_size) {
                if (match_buf[token] == 0)
                    touched_tokens.push_back(token);
                if (match_len > match_buf[token])
                    match_buf[token] = match_len;
            }
        }
    }

    if (touched_tokens.empty())
        return;

    // Build sparse penalty arrays
    int n = static_cast<int>(touched_tokens.size());
    std::vector<int32_t> h_tokens(n);
    std::vector<float> h_values(n);
    for (int i = 0; i < n; i++) {
        int32_t tok = touched_tokens[i];
        h_tokens[i] = tok;
        h_values[i] = multiplier * std::pow(base, static_cast<float>(match_buf[tok] - allowed_length));
        match_buf[tok] = 0;  // sparse clear
    }

    // Upload to GPU and apply — reuse persistent buffers to avoid per-call cudaMalloc
    // (buffers are file-scope; pre-allocated by sampling_preallocate_dry at engine init)
    size_t needed = static_cast<size_t>(n);
    if (needed > s_dry_buf_cap) {
        // Grow buffers (sync stream first to ensure previous work is done)
        cudaStreamSynchronize(stream);
        if (s_dry_tokens_buf)
            IMP_CUDA_CHECK_LOG(cudaFree(s_dry_tokens_buf));
        if (s_dry_values_buf)
            IMP_CUDA_CHECK_LOG(cudaFree(s_dry_values_buf));
        // Over-allocate to reduce future reallocations
        size_t new_cap = std::max(needed, s_dry_buf_cap * 2);
        new_cap = std::max(new_cap, static_cast<size_t>(256));
        if (cudaMalloc(&s_dry_tokens_buf, new_cap * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc(&s_dry_values_buf, new_cap * sizeof(float)) != cudaSuccess) {
            IMP_LOG_ERROR("apply_dry_penalty: cudaMalloc failed");
            if (s_dry_tokens_buf) {
                IMP_CUDA_CHECK_LOG(cudaFree(s_dry_tokens_buf));
                s_dry_tokens_buf = nullptr;
            }
            if (s_dry_values_buf) {
                IMP_CUDA_CHECK_LOG(cudaFree(s_dry_values_buf));
                s_dry_values_buf = nullptr;
            }
            s_dry_buf_cap = 0;
            return;
        }
        s_dry_buf_cap = new_cap;
    }

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s_dry_tokens_buf, h_tokens.data(), n * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s_dry_values_buf, h_values.data(), n * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_dry_sparse_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_logits, s_dry_tokens_buf, s_dry_values_buf, n);
}

// ===========================================================================
// Typical-P (locally typical) filtering
// ===========================================================================

// Single-block kernel: computes entropy, deviation histogram, finds threshold,
// and filters tokens with deviation > threshold.
static constexpr int TYPICAL_NBUCKETS = 256;

__global__ void apply_typical_p_kernel(float* __restrict__ logits, int vocab_size, float typical_p) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];
    __shared__ float s_max, s_sum, s_entropy, s_max_dev, s_threshold;
    __shared__ float s_buckets[TYPICAL_NBUCKETS];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // --- Pass 1: max logit ---
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_max = fmaxf(local_max, logits[i]);
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0)
        s_warp[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            mx = fmaxf(mx, s_warp[w]);
        s_max = mx;
    }
    __syncthreads();
    float gmax = s_max;

    // --- Pass 2: sum_exp ---
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_sum += expf(logits[i] - gmax);
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0)
        s_warp[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            sm += s_warp[w];
        s_sum = sm;
    }
    __syncthreads();

    float sum_exp = s_sum;
    float log_sum_exp = gmax + logf(sum_exp);
    float inv_log2 = 1.4426950408889634f;  // 1/ln(2)

    // --- Pass 3: entropy H = -sum(p_i * log2(p_i)) ---
    float local_ent = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float p = expf(logits[i] - gmax) / sum_exp;
        if (p > 1e-30f)
            local_ent -= p * log2f(p);
    }
    local_ent = warp_reduce_sum(local_ent);
    if (lane_id == 0)
        s_warp[warp_id] = local_ent;
    __syncthreads();
    if (tid == 0) {
        float e = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            e += s_warp[w];
        s_entropy = e;
    }
    __syncthreads();
    float H = s_entropy;

    // --- Pass 4: max deviation ---
    float local_md = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float surprise = -(logits[i] - log_sum_exp) * inv_log2;
        local_md = fmaxf(local_md, fabsf(surprise - H));
    }
    local_md = warp_reduce_max(local_md);
    if (lane_id == 0)
        s_warp[warp_id] = local_md;
    __syncthreads();
    if (tid == 0) {
        float md = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            md = fmaxf(md, s_warp[w]);
        s_max_dev = md;
    }
    __syncthreads();

    // --- Pass 5: build deviation histogram ---
    // Initialize buckets
    for (int b = tid; b < TYPICAL_NBUCKETS; b += blockDim.x)
        s_buckets[b] = 0.0f;
    __syncthreads();

    float bucket_scale = (s_max_dev > 1e-8f) ? (static_cast<float>(TYPICAL_NBUCKETS) / s_max_dev) : 1.0f;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float surprise = -(logits[i] - log_sum_exp) * inv_log2;
        float dev = fabsf(surprise - H);
        int bucket = min(static_cast<int>(dev * bucket_scale), TYPICAL_NBUCKETS - 1);
        float p = expf(logits[i] - gmax) / sum_exp;
        // TODO(determinism): this shared-memory FP atomicAdd accumulates bucket
        // mass in scheduling-dependent order, so the cumulative cutoff bucket
        // can flip when typical_p lands near a bucket boundary. typical_p is a
        // sampling FILTER (not the greedy / top-k core covered by the
        // deterministic flag); make this an ordered per-bucket reduction if
        // typical_p ever needs bit-exact reproducibility.
        atomicAdd(&s_buckets[bucket], p);
    }
    __syncthreads();

    // --- Pass 6: scan histogram to find threshold (thread 0) ---
    if (tid == 0) {
        float cum = 0.0f;
        s_threshold = s_max_dev + 1.0f;  // default: keep all
        for (int b = 0; b < TYPICAL_NBUCKETS; b++) {
            cum += s_buckets[b];
            if (cum >= typical_p) {
                // Threshold = upper bound of this bucket
                s_threshold = static_cast<float>(b + 1) / bucket_scale;
                break;
            }
        }
    }
    __syncthreads();

    // --- Pass 7: filter tokens with deviation > threshold ---
    float thr = s_threshold;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float surprise = -(logits[i] - log_sum_exp) * inv_log2;
        float dev = fabsf(surprise - H);
        if (dev > thr)
            logits[i] = -FLT_MAX;
    }
}

void apply_typical_p(float* logits, int vocab_size, float typical_p, cudaStream_t stream) {
    if (typical_p <= 0.0f || typical_p >= 1.0f)
        return;

    apply_typical_p_kernel<<<1, BLOCK_SIZE, 0, stream>>>(logits, vocab_size, typical_p);
}

// ===========================================================================
// Mirostat v2 sampling
// ===========================================================================

// Single-block kernel: computes log-sum-exp, filters by surprise threshold,
// samples from filtered set, and outputs token + surprise.
__global__ void mirostat_v2_sample_kernel(const float* __restrict__ logits, int vocab_size, float mu,
                                          float inv_temperature, unsigned int seed,
                                          int32_t* __restrict__ d_result, float* __restrict__ d_surprise) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_fsum;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // --- Step 1: Find max logit ---
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_max = fmaxf(local_max, logits[i]);

    local_max = warp_reduce_max(local_max);
    if (lane_id == 0)
        s_warp[warp_id] = local_max;
    __syncthreads();

    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            mx = fmaxf(mx, s_warp[w]);
        s_max = mx;
    }
    __syncthreads();
    float gmax = s_max;

    // --- Step 2: Compute sum of exp((logit - max) * inv_temperature) ---
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_sum += expf((logits[i] - gmax) * inv_temperature);

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0)
        s_warp[warp_id] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            sm += s_warp[w];
        s_sum = sm;
    }
    __syncthreads();

    // Mirostat threshold: keep tokens with surprise ≤ mu
    // With temperature T, p_i = exp((l_i - max)/T) / sum_exp
    // surprise_i = -log2(p_i) ≤ mu
    // ⟺ (l_i - max)/T ≥ log(sum_exp) - mu * ln(2)
    // ⟺ l_i ≥ max + T * (log(sum_exp) - mu * ln(2))
    float temperature = (inv_temperature > 0.0f) ? (1.0f / inv_temperature) : 1.0f;
    float log_sum_exp = logf(s_sum);
    float threshold = gmax + temperature * (log_sum_exp - mu * 0.6931471805599453f);

    // --- Step 3: Compute filtered probability sum ---
    float local_fsum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[i] >= threshold)
            local_fsum += expf((logits[i] - gmax) * inv_temperature);
    }

    local_fsum = warp_reduce_sum(local_fsum);
    if (lane_id == 0)
        s_warp[warp_id] = local_fsum;
    __syncthreads();

    if (tid == 0) {
        float fs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            fs += s_warp[w];
        // Fallback: if no tokens pass threshold, use entire distribution
        s_fsum = (fs > 0.0f) ? fs : s_sum;
    }
    __syncthreads();

    float fsum = s_fsum;
    bool use_threshold = (fsum < s_sum * 0.9999f);

    // --- Step 4: Sample from filtered distribution ---
    // Thread 0 scans through vocab, accumulating filtered probabilities.
    if (tid == 0) {
        float inv_fsum = 1.0f / fsum;
        unsigned int rng = seed;
        float r = lcg_rand_float(rng);

        float acc = 0.0f;
        int chosen = 0;
        bool found = false;

        for (int i = 0; i < vocab_size; i++) {
            if (!use_threshold || logits[i] >= threshold) {
                float p = expf((logits[i] - gmax) * inv_temperature) * inv_fsum;
                acc += p;
                if (r < acc) {
                    chosen = i;
                    found = true;
                    break;
                }
            }
        }

        // Fallback: pick highest-logit token
        if (!found) {
            float best = -FLT_MAX;
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] > best) {
                    best = logits[i];
                    chosen = i;
                }
            }
        }

        // Compute surprise using temperature-adjusted probability
        float chosen_prob = expf((logits[chosen] - gmax) * inv_temperature) / s_sum;
        float surprise = -log2f(fmaxf(chosen_prob, 1e-30f));

        d_result[0] = chosen;
        d_surprise[0] = surprise;
    }
}

static int32_t sample_mirostat_v2_impl(const Tensor& logits, float temperature, float tau, float eta,
                                       float* mu, unsigned int seed, int32_t* d_result, bool owns_result,
                                       cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    // Surprise value stored right after the token result
    float* d_surprise = reinterpret_cast<float*>(d_result + 1);

    mirostat_v2_sample_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, *mu, inv_temperature, seed,
                                                            d_result, d_surprise);

    // Read results
    int32_t h_result = 0;
    float h_surprise = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(&h_surprise, d_surprise, sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    if (owns_result)
        IMP_CUDA_CHECK_LOG(cudaFree(d_result));

    // Update mu: mu = mu - eta * (surprise - tau)
    *mu = *mu - eta * (h_surprise - tau);

    return h_result;
}

int32_t sample_mirostat_v2(const Tensor& logits, float temperature, float tau, float eta, float* mu,
                           unsigned int seed, cudaStream_t stream) {
    // Allocate temp buffer: 4 bytes for token + 4 bytes for surprise
    int32_t* d_result = nullptr;
    if (cudaMalloc(&d_result, 2 * sizeof(int32_t)) != cudaSuccess) {
        IMP_LOG_ERROR("sample_mirostat_v2: cudaMalloc failed");
        return 0;
    }
    return sample_mirostat_v2_impl(logits, temperature, tau, eta, mu, seed, d_result, true, stream);
}

int32_t sample_mirostat_v2(const Tensor& logits, float temperature, float tau, float eta, float* mu,
                           unsigned int seed, int32_t* d_result, cudaStream_t stream) {
    return sample_mirostat_v2_impl(logits, temperature, tau, eta, mu, seed, d_result, false, stream);
}

// ============================================================================
// CPU-side logprob computation
// ============================================================================

void compute_logprobs_cpu(const float* logits, int vocab_size, int32_t sampled_token, int top_n,
                          LogprobResult* out) {
    // 1. Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_val)
            max_val = logits[i];
    }

    // 2. Compute log-sum-exp
    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += std::exp(static_cast<double>(logits[i]) - static_cast<double>(max_val));
    }
    float log_sum_exp = static_cast<float>(std::log(sum_exp)) + max_val;

    // 3. Extract sampled token's logprob
    out->sampled_logprob = logits[sampled_token] - log_sum_exp;

    // 4. Top-N via partial sort with min-heap
    out->top.clear();
    if (top_n <= 0)
        return;

    // Use a simple approach: collect all (logprob, token) and partial sort
    // For vocab ~150K and top_n <= 20, this is fast enough (~0.3ms)
    struct Entry {
        float logprob;
        int32_t token;
        bool operator<(const Entry& o) const { return logprob > o.logprob; }  // max-heap order
    };

    // Min-heap of size top_n to track the top-N largest
    std::vector<Entry> heap;
    heap.reserve(top_n + 1);

    for (int i = 0; i < vocab_size; i++) {
        float lp = logits[i] - log_sum_exp;
        if (static_cast<int>(heap.size()) < top_n) {
            heap.push_back({lp, i});
            std::push_heap(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) {
                return a.logprob > b.logprob;  // min-heap: smallest logprob at top
            });
        } else if (lp > heap[0].logprob) {
            std::pop_heap(heap.begin(), heap.end(),
                          [](const Entry& a, const Entry& b) { return a.logprob > b.logprob; });
            heap.back() = {lp, i};
            std::push_heap(heap.begin(), heap.end(),
                           [](const Entry& a, const Entry& b) { return a.logprob > b.logprob; });
        }
    }

    // Sort descending by logprob
    std::sort(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) { return a.logprob > b.logprob; });

    out->top.reserve(heap.size());
    for (const auto& e : heap) {
        out->top.push_back({e.token, e.logprob});
    }
}

void sampling_preallocate_dry(int max_seq_len, cudaStream_t /*stream*/) {
    if (max_seq_len <= 0)
        return;
    size_t cap = static_cast<size_t>(max_seq_len);
    if (cap <= s_dry_buf_cap)
        return;  // already large enough

    // Free existing (if any) before re-allocating
    if (s_dry_tokens_buf) {
        cudaFree(s_dry_tokens_buf);
        s_dry_tokens_buf = nullptr;
    }
    if (s_dry_values_buf) {
        cudaFree(s_dry_values_buf);
        s_dry_values_buf = nullptr;
    }
    s_dry_buf_cap = 0;

    if (cudaMalloc(&s_dry_tokens_buf, cap * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&s_dry_values_buf, cap * sizeof(float)) != cudaSuccess) {
        IMP_LOG_ERROR("sampling_preallocate_dry: cudaMalloc failed for %zu elements", cap);
        if (s_dry_tokens_buf) {
            cudaFree(s_dry_tokens_buf);
            s_dry_tokens_buf = nullptr;
        }
        if (s_dry_values_buf) {
            cudaFree(s_dry_values_buf);
            s_dry_values_buf = nullptr;
        }
        return;
    }
    s_dry_buf_cap = cap;
    IMP_LOG_DEBUG("sampling_preallocate_dry: pre-allocated %zu DRY penalty slots", cap);
}

void sampling_cleanup() {
    s_cub_scratch.free();
    if (s_dry_tokens_buf) {
        cudaFree(s_dry_tokens_buf);
        s_dry_tokens_buf = nullptr;
    }
    if (s_dry_values_buf) {
        cudaFree(s_dry_values_buf);
        s_dry_values_buf = nullptr;
    }
    s_dry_buf_cap = 0;
}

}  // namespace imp
