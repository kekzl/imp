#include "compute/moe_routing.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>
#include <cstring>

namespace imp {

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARP_SIZE = 32;

// ============================================================================
// Warp-level reduction helpers
// ============================================================================

__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// ============================================================================
// Kernel 1: Top-k gating
//
// One block per token.  Each block processes one row of gate_logits
// [n_experts], computes softmax, selects top-k experts, normalizes weights.
//
// Outputs:
//   expert_indices[token * top_k + j]  -- j-th selected expert for token
//   expert_weights[token * top_k + j]  -- normalized weight for j-th expert
// ============================================================================

__global__ void topk_gating_kernel(const float* __restrict__ gate_logits,
                                   int n_experts,
                                   int top_k,
                                   int32_t* __restrict__ expert_indices,
                                   float* __restrict__ expert_weights,
                                   bool use_sigmoid,
                                   bool normalize_weights,
                                   const half* __restrict__ score_bias) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const float* logits = gate_logits + static_cast<int64_t>(token) * n_experts;

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    extern __shared__ char smem_raw[];
    float* s_probs   = reinterpret_cast<float*>(smem_raw);
    // When using score_bias, we need a second array for selection scores
    // s_probs holds unbiased scores (used for weight values)
    // s_sel_probs holds biased scores (used for top-k selection)
    float* s_sel_probs = s_probs + (score_bias ? n_experts : 0);
    float* s_topk_val  = s_probs + (score_bias ? 2 * n_experts : n_experts);
    int32_t* s_topk_idx = reinterpret_cast<int32_t*>(s_topk_val + top_k);

    if (use_sigmoid) {
        // --- Sigmoid gating (Nemotron-H): prob_i = sigmoid(logit_i) ---
        for (int i = tid; i < n_experts; i += blockDim.x) {
            float p = 1.0f / (1.0f + expf(-logits[i]));
            s_probs[i] = p;
            if (score_bias) {
                // Bias is added to sigmoid outputs for selection only
                s_sel_probs[i] = p + __half2float(score_bias[i]);
            }
        }
    } else {
        // --- Softmax gating (Mixtral, DeepSeek, etc.) ---
        // Step 1: Find max for numerical stability
        float local_max = -FLT_MAX;
        for (int i = tid; i < n_experts; i += blockDim.x) {
            local_max = fmaxf(local_max, logits[i]);
        }
        local_max = warp_reduce_max(local_max);

        if (lane_id == 0) s_warp[warp_id] = local_max;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
            val = warp_reduce_max(val);
            if (lane_id == 0) s_warp[0] = val;
        }
        __syncthreads();
        float gmax = s_warp[0];

        // Step 2: Compute exp and sum for softmax
        float local_sum = 0.0f;
        for (int i = tid; i < n_experts; i += blockDim.x) {
            local_sum += expf(logits[i] - gmax);
        }
        local_sum = warp_reduce_sum(local_sum);

        if (lane_id == 0) s_warp[warp_id] = local_sum;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane_id == 0) s_warp[0] = val;
        }
        __syncthreads();
        float inv_sum = 1.0f / s_warp[0];

        // Step 3: Compute softmax probabilities
        for (int i = tid; i < n_experts; i += blockDim.x) {
            s_probs[i] = expf(logits[i] - gmax) * inv_sum;
        }
    }
    __syncthreads();

    // Parallel top-k selection: find top_k experts using block-wide argmax reduction.
    // Each iteration finds the global max, records it, and masks it out.
    // When score_bias is provided, select based on biased scores (s_sel_probs)
    // but use UNBIASED scores (s_probs) for weight values.
    {
        // Shared memory for argmax reduction across warps
        __shared__ float s_warp_max[NUM_WARPS];
        __shared__ int   s_warp_argmax[NUM_WARPS];

        const float* sel = score_bias ? s_sel_probs : s_probs;

        // Each thread owns one element (tid < n_experts), or -FLT_MAX if out of range
        float my_val = (tid < n_experts) ? sel[tid] : -FLT_MAX;
        int my_idx = tid;

        for (int k = 0; k < top_k; k++) {
            // Warp-level argmax
            float wmax = my_val;
            int widx = my_idx;
            #pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                float other_val = __shfl_xor_sync(0xFFFFFFFF, wmax, off);
                int other_idx = __shfl_xor_sync(0xFFFFFFFF, widx, off);
                if (other_val > wmax) { wmax = other_val; widx = other_idx; }
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
                    if (ov > v) { v = ov; ix = oi; }
                }
                if (lane_id == 0) {
                    s_topk_idx[k] = ix;
                    // Store UNBIASED weight for the selected expert
                    s_topk_val[k] = score_bias ? s_probs[ix] : v;
                }
            }
            __syncthreads();

            // Mask out the selected expert so it won't be picked again
            if (tid == s_topk_idx[k]) my_val = -FLT_MAX;
        }

        // Thread 0: normalize weights and write output
        if (tid == 0) {
            float multiplier = 1.0f;
            if (normalize_weights) {
                float norm = 0.0f;
                for (int j = 0; j < top_k; ++j) norm += s_topk_val[j];
                multiplier = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
            }
            int base = token * top_k;
            for (int j = 0; j < top_k; ++j) {
                expert_indices[base + j] = s_topk_idx[j];
                expert_weights[base + j] = s_topk_val[j] * multiplier;
            }
        }
    }
}

// ============================================================================
// Fused kernel: Gate GEMV + softmax/sigmoid + top-k selection
//
// For n=1 decode: combines gate weight dot-products with routing in a single
// kernel, eliminating the intermediate FP32 logits buffer and 1 kernel launch.
//
// 1 block × 256 threads (8 warps). Each warp computes ceil(n_experts/8) dot
// products, stores logits to shared memory, then all threads cooperate on
// softmax/sigmoid + top-k selection (same algorithm as topk_gating_kernel).
// ============================================================================

__global__ void gemv_gate_topk_fused_kernel(
        const half* __restrict__ W_gate,        // [n_experts, d_model] FP16
        const half* __restrict__ x,             // [d_model] FP16 input
        int n_experts,
        int d_model,
        int top_k,
        int32_t* __restrict__ expert_indices,   // [top_k] output
        float* __restrict__ expert_weights,     // [top_k] output
        bool use_sigmoid,
        bool normalize_weights,
        const half* __restrict__ score_bias) {

    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Shared memory layout:
    //   s_logits[n_experts]       — gate logit / probability values
    //   s_sel_probs[n_experts]    — biased selection probs (only if score_bias)
    //   s_topk_val[top_k]        — selected top-k values
    //   s_topk_idx[top_k]        — selected top-k indices
    //   s_warp_max[NUM_WARPS]    — warp reduction scratch
    //   s_warp_argmax[NUM_WARPS] — warp reduction scratch
    extern __shared__ char smem_raw[];
    float* s_logits = reinterpret_cast<float*>(smem_raw);
    float* s_sel_probs = s_logits + (score_bias ? n_experts : 0);
    float* s_topk_val  = s_logits + (score_bias ? 2 * n_experts : n_experts);
    int32_t* s_topk_idx = reinterpret_cast<int32_t*>(s_topk_val + top_k);
    float* s_warp_scratch = reinterpret_cast<float*>(s_topk_idx + top_k);
    int*   s_warp_argmax  = reinterpret_cast<int*>(s_warp_scratch + NUM_WARPS);

    // ---- Phase 1: Gate GEMV — compute dot(W_gate[e], x) for all experts ----
    const int K2 = d_model / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x);

    for (int e = warp_id; e < n_experts; e += NUM_WARPS) {
        const half2* W2 = reinterpret_cast<const half2*>(W_gate + static_cast<size_t>(e) * d_model);
        float sum = 0.0f;

        for (int i = lane_id; i < K2; i += WARP_SIZE) {
            half2 w = W2[i];
            half2 v = x2[i];
            sum += __half2float(w.x) * __half2float(v.x);
            sum += __half2float(w.y) * __half2float(v.y);
        }

        // Warp shuffle reduction
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

        if (lane_id == 0) s_logits[e] = sum;
    }
    __syncthreads();

    // ---- Phase 2: Softmax or sigmoid ----
    if (use_sigmoid) {
        for (int i = tid; i < n_experts; i += blockDim.x) {
            float p = 1.0f / (1.0f + expf(-s_logits[i]));
            s_logits[i] = p;  // overwrite logit with prob
            if (score_bias) {
                s_sel_probs[i] = p + __half2float(score_bias[i]);
            }
        }
    } else {
        // Softmax: find max
        float local_max = -FLT_MAX;
        for (int i = tid; i < n_experts; i += blockDim.x) {
            local_max = fmaxf(local_max, s_logits[i]);
        }
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) s_warp_scratch[warp_id] = local_max;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp_scratch[lane_id] : -FLT_MAX;
            val = warp_reduce_max(val);
            if (lane_id == 0) s_warp_scratch[0] = val;
        }
        __syncthreads();
        float gmax = s_warp_scratch[0];

        // Compute exp and sum
        float local_sum = 0.0f;
        for (int i = tid; i < n_experts; i += blockDim.x) {
            local_sum += expf(s_logits[i] - gmax);
        }
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) s_warp_scratch[warp_id] = local_sum;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp_scratch[lane_id] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane_id == 0) s_warp_scratch[0] = val;
        }
        __syncthreads();
        float inv_sum = 1.0f / s_warp_scratch[0];

        // Normalize
        for (int i = tid; i < n_experts; i += blockDim.x) {
            s_logits[i] = expf(s_logits[i] - gmax) * inv_sum;
        }
    }
    __syncthreads();

    // ---- Phase 3: Top-k selection (same algorithm as topk_gating_kernel) ----
    {
        const float* sel = score_bias ? s_sel_probs : s_logits;
        float my_val = (tid < n_experts) ? sel[tid] : -FLT_MAX;
        int my_idx = tid;

        for (int k = 0; k < top_k; k++) {
            float wmax = my_val;
            int widx = my_idx;
            #pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                float other_val = __shfl_xor_sync(0xFFFFFFFF, wmax, off);
                int other_idx = __shfl_xor_sync(0xFFFFFFFF, widx, off);
                if (other_val > wmax) { wmax = other_val; widx = other_idx; }
            }

            if (lane_id == 0) {
                s_warp_scratch[warp_id] = wmax;
                s_warp_argmax[warp_id] = widx;
            }
            __syncthreads();

            if (warp_id == 0) {
                float v = (lane_id < NUM_WARPS) ? s_warp_scratch[lane_id] : -FLT_MAX;
                int ix = (lane_id < NUM_WARPS) ? s_warp_argmax[lane_id] : -1;
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                    float ov = __shfl_xor_sync(0xFFFFFFFF, v, off);
                    int oi = __shfl_xor_sync(0xFFFFFFFF, ix, off);
                    if (ov > v) { v = ov; ix = oi; }
                }
                if (lane_id == 0) {
                    s_topk_idx[k] = ix;
                    s_topk_val[k] = score_bias ? s_logits[ix] : v;
                }
            }
            __syncthreads();

            if (tid == s_topk_idx[k]) my_val = -FLT_MAX;
        }

        // Thread 0: normalize weights and write output
        if (tid == 0) {
            float multiplier = 1.0f;
            if (normalize_weights) {
                float norm = 0.0f;
                for (int j = 0; j < top_k; ++j) norm += s_topk_val[j];
                multiplier = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
            }
            for (int j = 0; j < top_k; ++j) {
                expert_indices[j] = s_topk_idx[j];
                expert_weights[j] = s_topk_val[j] * multiplier;
            }
        }
    }
}

// ============================================================================
// Kernel 2: Count tokens per expert
//
// Each token contributes top_k entries.  We atomically increment per-expert
// counts.  expert_counts has n_experts elements, initialized to zero.
// ============================================================================

__global__ void count_tokens_per_expert_kernel(
        const int32_t* __restrict__ expert_indices,
        int n_tokens,
        int top_k,
        int32_t* __restrict__ expert_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * top_k;
    if (idx < total) {
        int expert = expert_indices[idx];
        atomicAdd(&expert_counts[expert], 1);
    }
}

// ============================================================================
// Kernel 3: Exclusive prefix sum (scan) on expert_counts to produce
// expert_offsets.  Single block, single thread (n_experts is small).
// ============================================================================

__global__ void exclusive_scan_kernel(const int32_t* __restrict__ counts,
                                      int32_t* __restrict__ offsets,
                                      int n_experts) {
    if (threadIdx.x == 0) {
        int32_t running = 0;
        for (int i = 0; i < n_experts; ++i) {
            offsets[i] = running;
            running += counts[i];
        }
        offsets[n_experts] = running;  // total
    }
}

// ============================================================================
// Kernel 4: Scatter token IDs into sorted_token_ids based on expert assignment.
//
// For each (token, j) pair where expert_indices[token*top_k + j] == e,
// place the token into the e-th bucket of sorted_token_ids.
// We use atomicAdd on a write-position counter per expert.
// ============================================================================

__global__ void scatter_token_ids_kernel(
        const int32_t* __restrict__ expert_indices,
        const int32_t* __restrict__ expert_offsets,
        int n_tokens,
        int top_k,
        int32_t* __restrict__ sorted_token_ids,
        int32_t* __restrict__ expert_write_pos) {  // [n_experts], init to 0
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * top_k;
    if (idx < total) {
        int token = idx / top_k;
        int expert = expert_indices[idx];
        int pos = atomicAdd(&expert_write_pos[expert], 1);
        sorted_token_ids[expert_offsets[expert] + pos] = token;
    }
}

// ============================================================================
// Kernel 5: Gather -- reorder tokens by expert assignment
//
// For each position i in sorted_token_ids:
//   gathered[i, :] = input[sorted_token_ids[i], :]
// ============================================================================

__global__ void moe_gather_kernel(const float* __restrict__ input,
                                  const int32_t* __restrict__ sorted_token_ids,
                                  float* __restrict__ gathered,
                                  int total_tokens,
                                  int d_model) {
    int row = blockIdx.x;
    if (row >= total_tokens) return;

    int src_token = sorted_token_ids[row];
    const float* src = input + static_cast<int64_t>(src_token) * d_model;
    float* dst = gathered + static_cast<int64_t>(row) * d_model;

    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        dst[col] = src[col];
    }
}

__global__ void moe_gather_fp16_kernel(const half* __restrict__ input,
                                       const int32_t* __restrict__ sorted_token_ids,
                                       half* __restrict__ gathered,
                                       int total_tokens,
                                       int d_model) {
    int row = blockIdx.x;
    if (row >= total_tokens) return;

    int src_token = sorted_token_ids[row];
    const half* src = input + static_cast<int64_t>(src_token) * d_model;
    half* dst = gathered + static_cast<int64_t>(row) * d_model;

    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        dst[col] = src[col];
    }
}

// ============================================================================
// Kernel 6: Scatter -- weighted scatter-add of expert outputs back to tokens
//
// For each position i in sorted_token_ids:
//   token_id = sorted_token_ids[i]
//   weight   = expert_weights[<corresponding index>]
//   output[token_id, :] += weight * expert_output[i, :]
//
// We need to figure out which (token, j) pair position i corresponds to.
// Since sorted_token_ids[i] = token, and a token may appear top_k times,
// we need the weight for this specific assignment.
//
// Approach: store a parallel array "sorted_weights" during the scatter
// phase of the routing, or recompute.  For simplicity, we build a
// sorted_weights array alongside sorted_token_ids during routing.
// But the MoeRoutingResult struct doesn't have this field.
//
// Alternative: for each sorted position i, we know the token_id and the
// expert.  We can look up the weight from expert_weights by scanning
// expert_indices for the matching (token_id, expert) pair.
//
// Better: during the scatter_token_ids_kernel, also write the weight to a
// parallel "sorted_weights" array, and store it as auxiliary data alongside
// sorted_token_ids.  We'll extend the approach by writing the flat index
// (idx = token*top_k + j) into a parallel "sorted_flat_idx" array.
// Then we can look up expert_weights[sorted_flat_idx[i]].
//
// We'll store this auxiliary array right after sorted_token_ids in memory.
// ============================================================================

// Extended scatter kernel that also writes flat indices.
__global__ void scatter_token_ids_with_flat_idx_kernel(
        const int32_t* __restrict__ expert_indices,
        const int32_t* __restrict__ expert_offsets,
        int n_tokens,
        int top_k,
        int32_t* __restrict__ sorted_token_ids,
        int32_t* __restrict__ sorted_flat_idx,
        int32_t* __restrict__ expert_write_pos,
        int32_t* __restrict__ token_to_expanded) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * top_k;
    if (idx < total) {
        int token = idx / top_k;
        int expert = expert_indices[idx];
        int pos = atomicAdd(&expert_write_pos[expert], 1);
        int dest = expert_offsets[expert] + pos;
        sorted_token_ids[dest] = token;
        sorted_flat_idx[dest]  = idx;  // flat index into expert_weights
        if (token_to_expanded) token_to_expanded[idx] = dest;  // inverse map
    }
}

// Scatter-add kernel using the flat index to look up weights.
__global__ void moe_scatter_kernel(const float* __restrict__ expert_output,
                                   const int32_t* __restrict__ sorted_token_ids,
                                   const int32_t* __restrict__ sorted_flat_idx,
                                   const float* __restrict__ expert_weights,
                                   float* __restrict__ output,
                                   int total_tokens,
                                   int d_model) {
    int row = blockIdx.x;
    if (row >= total_tokens) return;

    int token_id = sorted_token_ids[row];
    int flat_idx = sorted_flat_idx[row];
    float weight = expert_weights[flat_idx];

    const float* src = expert_output + static_cast<int64_t>(row) * d_model;
    float* dst = output + static_cast<int64_t>(token_id) * d_model;

    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        atomicAdd(&dst[col], weight * src[col]);
    }
}

__global__ void moe_scatter_fp16_kernel(const half* __restrict__ expert_output,
                                        const int32_t* __restrict__ sorted_token_ids,
                                        const int32_t* __restrict__ sorted_flat_idx,
                                        const float* __restrict__ expert_weights,
                                        float* __restrict__ output,
                                        int total_tokens,
                                        int d_model) {
    int row = blockIdx.x;
    if (row >= total_tokens) return;

    int token_id = sorted_token_ids[row];
    int flat_idx = sorted_flat_idx[row];
    float weight = expert_weights[flat_idx];

    const half* src = expert_output + static_cast<int64_t>(row) * d_model;
    float* dst = output + static_cast<int64_t>(token_id) * d_model;

    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        atomicAdd(&dst[col], weight * __half2float(src[col]));
    }
}

// ============================================================================
// Utility: zero-initialize device memory
// ============================================================================

__global__ void zero_int32_kernel(int32_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 0;
}

__global__ void zero_float_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 0.0f;
}

// ============================================================================
// Helper to set up a Tensor descriptor
// ============================================================================

static Tensor make_tensor_1d(void* data, DType dtype, int64_t size, bool on_device) {
    Tensor t;
    t.data = data;
    t.dtype = dtype;
    t.ndim = 1;
    t.shape[0] = size;
    t.shape[1] = 0;
    t.shape[2] = 0;
    t.shape[3] = 0;
    t.stride[0] = 1;
    t.stride[1] = 0;
    t.stride[2] = 0;
    t.stride[3] = 0;
    t.on_device = on_device;
    return t;
}

static Tensor make_tensor_2d(void* data, DType dtype, int64_t d0, int64_t d1, bool on_device) {
    Tensor t;
    t.data = data;
    t.dtype = dtype;
    t.ndim = 2;
    t.shape[0] = d0;
    t.shape[1] = d1;
    t.shape[2] = 0;
    t.shape[3] = 0;
    t.stride[0] = d1;
    t.stride[1] = 1;
    t.stride[2] = 0;
    t.stride[3] = 0;
    t.on_device = on_device;
    return t;
}

// ============================================================================
// Public API: moe_topk_gating
// ============================================================================

void moe_topk_gating(const Tensor& gate_logits, int top_k,
                     MoeRoutingResult& result,
                     cudaStream_t stream,
                     bool use_sigmoid,
                     bool normalize_weights,
                     const void* score_bias) {
    const int n_tokens  = static_cast<int>(gate_logits.shape[0]);
    const int n_experts = static_cast<int>(gate_logits.shape[1]);
    const float* d_logits = static_cast<const float*>(gate_logits.data);
    const int total_assignments = n_tokens * top_k;

    // ---- Allocate result tensors ----

    auto check_alloc = [](cudaError_t err, const char* name) -> bool {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("moe_topk_gating: cudaMalloc failed for %s: %s",
                          name, cudaGetErrorString(err));
            return false;
        }
        return true;
    };

    // expert_indices: [n_tokens, top_k] int32
    int32_t* d_expert_indices = nullptr;
    if (!check_alloc(cudaMalloc(&d_expert_indices,
            static_cast<size_t>(total_assignments) * sizeof(int32_t)), "expert_indices"))
        return;

    // expert_weights: [n_tokens, top_k] float
    float* d_expert_weights = nullptr;
    if (!check_alloc(cudaMalloc(&d_expert_weights,
            static_cast<size_t>(total_assignments) * sizeof(float)), "expert_weights")) {
        cudaFree(d_expert_indices);
        return;
    }

    // sorted_token_ids: [total_assignments] int32
    // We allocate 2x to hold a parallel sorted_flat_idx array right after.
    int32_t* d_sorted_token_ids = nullptr;
    if (!check_alloc(cudaMalloc(&d_sorted_token_ids,
            static_cast<size_t>(total_assignments) * 2 * sizeof(int32_t)), "sorted_token_ids")) {
        cudaFree(d_expert_indices);
        cudaFree(d_expert_weights);
        return;
    }
    int32_t* d_sorted_flat_idx = d_sorted_token_ids + total_assignments;

    // expert_offsets: [n_experts + 1] int32
    int32_t* d_expert_offsets = nullptr;
    if (!check_alloc(cudaMalloc(&d_expert_offsets,
            static_cast<size_t>(n_experts + 1) * sizeof(int32_t)), "expert_offsets")) {
        cudaFree(d_expert_indices);
        cudaFree(d_expert_weights);
        cudaFree(d_sorted_token_ids);
        return;
    }

    // Temporary: expert_counts [n_experts], expert_write_pos [n_experts]
    int32_t* d_expert_counts = nullptr;
    int32_t* d_expert_write_pos = nullptr;
    if (!check_alloc(cudaMalloc(&d_expert_counts,
            static_cast<size_t>(n_experts) * sizeof(int32_t)), "expert_counts") ||
        !check_alloc(cudaMalloc(&d_expert_write_pos,
            static_cast<size_t>(n_experts) * sizeof(int32_t)), "expert_write_pos")) {
        cudaFree(d_expert_indices);
        cudaFree(d_expert_weights);
        cudaFree(d_sorted_token_ids);
        cudaFree(d_expert_offsets);
        cudaFree(d_expert_counts);  // safe even if null
        return;
    }

    // Zero-initialize counts and write positions
    int grid_z = (n_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_int32_kernel<<<grid_z, BLOCK_SIZE, 0, stream>>>(d_expert_counts, n_experts);
    zero_int32_kernel<<<grid_z, BLOCK_SIZE, 0, stream>>>(d_expert_write_pos, n_experts);

    // ---- Kernel 1: Softmax + top-k selection per token ----
    // Shared memory: n_experts floats (s_probs) + top_k floats + top_k ints
    //                + NUM_WARPS floats (s_warp, reused from register area)
    // The s_warp area is separate from the extern shared, it's __shared__ inside
    // the kernel.  Actually we declared it inside the kernel as extern __shared__.
    // Let's compute the needed shared memory.
    // Shared memory: s_probs (n_experts) + optionally s_sel_probs (n_experts) + s_topk_val (top_k) + s_topk_idx (top_k)
    int probs_arrays = score_bias ? 2 : 1;  // need extra array for biased selection scores
    size_t smem_gating = static_cast<size_t>(n_experts) * probs_arrays * sizeof(float)
                       + static_cast<size_t>(top_k) * sizeof(float)
                       + static_cast<size_t>(top_k) * sizeof(int32_t);

    topk_gating_kernel<<<n_tokens, BLOCK_SIZE, smem_gating, stream>>>(
        d_logits, n_experts, top_k, d_expert_indices, d_expert_weights, use_sigmoid, normalize_weights,
        static_cast<const half*>(score_bias));

    // ---- Kernel 2: Count tokens per expert ----
    int grid_count = (total_assignments + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_tokens_per_expert_kernel<<<grid_count, BLOCK_SIZE, 0, stream>>>(
        d_expert_indices, n_tokens, top_k, d_expert_counts);

    // ---- Kernel 3: Exclusive prefix sum for offsets ----
    exclusive_scan_kernel<<<1, 1, 0, stream>>>(d_expert_counts, d_expert_offsets, n_experts);

    // ---- Kernel 4: Scatter token IDs (and flat indices) into sorted order ----
    scatter_token_ids_with_flat_idx_kernel<<<grid_count, BLOCK_SIZE, 0, stream>>>(
        d_expert_indices, d_expert_offsets, n_tokens, top_k,
        d_sorted_token_ids, d_sorted_flat_idx, d_expert_write_pos, nullptr);

    // ---- Free temporaries ----
    // We defer freeing to after stream sync, but since these are only needed
    // during gating, we can use cudaFreeAsync if available.  For compatibility,
    // we'll record an event and free after.  For simplicity, free synchronously.
    // In production code, use memory pools.
    cudaStreamSynchronize(stream);
    cudaFree(d_expert_counts);
    cudaFree(d_expert_write_pos);

    // ---- Fill result struct ----
    result.expert_indices = make_tensor_2d(d_expert_indices, DType::INT32,
                                           n_tokens, top_k, true);
    result.expert_weights = make_tensor_2d(d_expert_weights, DType::FP32,
                                           n_tokens, top_k, true);
    // sorted_token_ids: we expose the full allocation (includes flat_idx)
    // but the tensor shape only covers the token IDs part.
    result.sorted_token_ids = make_tensor_1d(d_sorted_token_ids, DType::INT32,
                                             total_assignments, true);
    result.expert_offsets = make_tensor_1d(d_expert_offsets, DType::INT32,
                                           n_experts + 1, true);
}

// ============================================================================
// Public API: moe_gather
// ============================================================================

void moe_gather(const Tensor& input, const MoeRoutingResult& routing,
                Tensor& gathered, cudaStream_t stream) {
    const int n_tokens_orig = static_cast<int>(input.shape[0]);
    const int d_model       = static_cast<int>(input.shape[1]);
    const int total_tokens  = static_cast<int>(routing.sorted_token_ids.shape[0]);
    const int32_t* d_sorted = static_cast<const int32_t*>(routing.sorted_token_ids.data);

    (void)n_tokens_orig;

    if (input.dtype == DType::FP16) {
        const half* d_input = static_cast<const half*>(input.data);
        half* d_gathered    = static_cast<half*>(gathered.data);
        moe_gather_fp16_kernel<<<total_tokens, BLOCK_SIZE, 0, stream>>>(
            d_input, d_sorted, d_gathered, total_tokens, d_model);
    } else {
        const float* d_input = static_cast<const float*>(input.data);
        float* d_gathered    = static_cast<float*>(gathered.data);
        moe_gather_kernel<<<total_tokens, BLOCK_SIZE, 0, stream>>>(
            d_input, d_sorted, d_gathered, total_tokens, d_model);
    }
}

// ============================================================================
// Public API: moe_scatter
// ============================================================================

void moe_scatter(const Tensor& expert_output, const MoeRoutingResult& routing,
                 Tensor& output, cudaStream_t stream) {
    const int d_model      = static_cast<int>(expert_output.shape[1]);
    const int total_tokens = static_cast<int>(routing.sorted_token_ids.shape[0]);
    const int n_tokens     = static_cast<int>(output.shape[0]);

    const int32_t* d_sorted_token_ids = static_cast<const int32_t*>(routing.sorted_token_ids.data);
    // The sorted_flat_idx array is stored immediately after sorted_token_ids
    // in the same allocation (see moe_topk_gating).
    const int32_t* d_sorted_flat_idx = d_sorted_token_ids + total_tokens;
    const float* d_expert_weights = static_cast<const float*>(routing.expert_weights.data);

    // Zero the output first (scatter-add accumulates into it)
    int total_out_elems = n_tokens * d_model;
    int grid_z = (total_out_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Output is always FP32 for the scatter-add (atomicAdd on float)
    float* d_output = static_cast<float*>(output.data);
    zero_float_kernel<<<grid_z, BLOCK_SIZE, 0, stream>>>(d_output, total_out_elems);

    if (expert_output.dtype == DType::FP16) {
        const half* d_expert_out = static_cast<const half*>(expert_output.data);
        moe_scatter_fp16_kernel<<<total_tokens, BLOCK_SIZE, 0, stream>>>(
            d_expert_out, d_sorted_token_ids, d_sorted_flat_idx,
            d_expert_weights, d_output, total_tokens, d_model);
    } else {
        const float* d_expert_out = static_cast<const float*>(expert_output.data);
        moe_scatter_kernel<<<total_tokens, BLOCK_SIZE, 0, stream>>>(
            d_expert_out, d_sorted_token_ids, d_sorted_flat_idx,
            d_expert_weights, d_output, total_tokens, d_model);
    }
}

// ============================================================================
// MoeRoutingBuffers -- pre-allocated pool
// ============================================================================

MoeRoutingBuffers::~MoeRoutingBuffers() {
    free();
}

void MoeRoutingBuffers::allocate(int max_tok, int max_exp, int top_k_val) {
    free();
    max_tokens = max_tok;
    max_experts = max_exp;
    top_k = top_k_val;

    int total_assignments = max_tokens * top_k;
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    size_t indices_sz  = align256(static_cast<size_t>(total_assignments) * sizeof(int32_t));
    size_t weights_sz  = align256(static_cast<size_t>(total_assignments) * sizeof(float));
    size_t sorted_sz   = align256(static_cast<size_t>(total_assignments) * 2 * sizeof(int32_t));
    size_t offsets_sz  = align256(static_cast<size_t>(max_experts + 1) * sizeof(int32_t));
    size_t counts_sz   = align256(static_cast<size_t>(max_experts) * sizeof(int32_t));
    size_t wpos_sz     = align256(static_cast<size_t>(max_experts) * sizeof(int32_t));
    size_t t2e_sz      = align256(static_cast<size_t>(total_assignments) * sizeof(int32_t));

    pool_size = indices_sz + weights_sz + sorted_sz + offsets_sz + counts_sz + wpos_sz + t2e_sz;
    cudaError_t err = cudaMalloc(&pool, pool_size);
    if (err != cudaSuccess) {
        pool = nullptr;
        pool_size = 0;
        return;
    }

    char* ptr = static_cast<char*>(pool);
    expert_indices    = reinterpret_cast<int32_t*>(ptr); ptr += indices_sz;
    expert_weights    = reinterpret_cast<float*>(ptr);   ptr += weights_sz;
    sorted_token_ids  = reinterpret_cast<int32_t*>(ptr); ptr += sorted_sz;
    expert_offsets    = reinterpret_cast<int32_t*>(ptr);  ptr += offsets_sz;
    expert_counts     = reinterpret_cast<int32_t*>(ptr);  ptr += counts_sz;
    expert_write_pos  = reinterpret_cast<int32_t*>(ptr);  ptr += wpos_sz;
    token_to_expanded = reinterpret_cast<int32_t*>(ptr);  ptr += t2e_sz;
}

void MoeRoutingBuffers::free() {
    if (pool) {
        cudaFree(pool);
        pool = nullptr;
    }
    pool_size = 0;
    expert_indices = nullptr;
    expert_weights = nullptr;
    sorted_token_ids = nullptr;
    expert_offsets = nullptr;
    expert_counts = nullptr;
    expert_write_pos = nullptr;
    token_to_expanded = nullptr;
}

// ============================================================================
// Kernel 7: Weighted sum for single-token decode (replaces gather+scatter)
//
// output[i] = Σ_k expert_weights[k] * expert_outputs[k * d_model + i]
// Each thread handles one output element, looping over top_k.
// ============================================================================

__global__ void moe_weighted_sum_kernel(const half* __restrict__ expert_outputs,
                                         const float* __restrict__ expert_weights,
                                         float* __restrict__ output,
                                         int d_model, int top_k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_model) return;

    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        sum += expert_weights[k] * __half2float(expert_outputs[k * d_model + i]);
    }
    output[i] = sum;
}

// ============================================================================
// moe_topk_gating with pre-allocated buffers
// ============================================================================

void moe_topk_gating(const Tensor& gate_logits, int top_k,
                     MoeRoutingBuffers& buffers,
                     MoeRoutingResult& result,
                     cudaStream_t stream,
                     bool use_sigmoid,
                     bool normalize_weights,
                     const void* score_bias,
                     bool skip_sorting) {
    const int n_tokens  = static_cast<int>(gate_logits.shape[0]);
    const int n_experts = static_cast<int>(gate_logits.shape[1]);
    const float* d_logits = static_cast<const float*>(gate_logits.data);
    const int total_assignments = n_tokens * top_k;

    int32_t* d_expert_indices    = buffers.expert_indices;
    float*   d_expert_weights    = buffers.expert_weights;
    int32_t* d_sorted_token_ids  = buffers.sorted_token_ids;
    int32_t* d_expert_offsets    = buffers.expert_offsets;

    // Kernel 1: Softmax + top-k selection per token
    int probs_arrays = score_bias ? 2 : 1;
    size_t smem_gating = static_cast<size_t>(n_experts) * probs_arrays * sizeof(float)
                       + static_cast<size_t>(top_k) * sizeof(float)
                       + static_cast<size_t>(top_k) * sizeof(int32_t);

    topk_gating_kernel<<<n_tokens, BLOCK_SIZE, smem_gating, stream>>>(
        d_logits, n_experts, top_k, d_expert_indices, d_expert_weights, use_sigmoid, normalize_weights,
        static_cast<const half*>(score_bias));

    if (!skip_sorting) {
        int32_t* d_sorted_flat_idx   = d_sorted_token_ids + total_assignments;
        int32_t* d_expert_counts     = buffers.expert_counts;
        int32_t* d_expert_write_pos  = buffers.expert_write_pos;

        // Zero-initialize counts and write positions
        int grid_z = (n_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        zero_int32_kernel<<<grid_z, BLOCK_SIZE, 0, stream>>>(d_expert_counts, n_experts);
        zero_int32_kernel<<<grid_z, BLOCK_SIZE, 0, stream>>>(d_expert_write_pos, n_experts);

        // Kernel 2: Count tokens per expert
        int grid_count = (total_assignments + BLOCK_SIZE - 1) / BLOCK_SIZE;
        count_tokens_per_expert_kernel<<<grid_count, BLOCK_SIZE, 0, stream>>>(
            d_expert_indices, n_tokens, top_k, d_expert_counts);

        // Kernel 3: Exclusive prefix sum for offsets
        exclusive_scan_kernel<<<1, 1, 0, stream>>>(d_expert_counts, d_expert_offsets, n_experts);

        // Kernel 4: Scatter token IDs (and flat indices) into sorted order
        scatter_token_ids_with_flat_idx_kernel<<<grid_count, BLOCK_SIZE, 0, stream>>>(
            d_expert_indices, d_expert_offsets, n_tokens, top_k,
            d_sorted_token_ids, d_sorted_flat_idx, d_expert_write_pos,
            buffers.token_to_expanded);
    }

    // Fill result struct (no ownership -- memory belongs to buffers)
    result.owns_memory = false;
    result.token_to_expanded = buffers.token_to_expanded;
    result.expert_indices = make_tensor_2d(d_expert_indices, DType::INT32,
                                           n_tokens, top_k, true);
    result.expert_weights = make_tensor_2d(d_expert_weights, DType::FP32,
                                           n_tokens, top_k, true);
    result.sorted_token_ids = make_tensor_1d(d_sorted_token_ids, DType::INT32,
                                             total_assignments, true);
    result.expert_offsets = make_tensor_1d(d_expert_offsets, DType::INT32,
                                           n_experts + 1, true);
}

// ============================================================================
// Public API: moe_weighted_sum
// ============================================================================

void moe_weighted_sum(const void* expert_outputs, const float* expert_weights,
                      float* output, int d_model, int top_k,
                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    moe_weighted_sum_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(expert_outputs), expert_weights, output, d_model, top_k);
}

// ============================================================================
// Fused weighted sum + FP16 output + optional residual add.
// Eliminates the FP32 intermediate buffer and fp32_to_fp16 conversion kernel.
// ============================================================================

__global__ void moe_weighted_sum_residual_kernel(
        const half* __restrict__ expert_outputs,
        const float* __restrict__ expert_weights,
        const half* residual,  // may be nullptr, may alias output
        half* output,          // no __restrict__ — may alias residual
        int d_model, int top_k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_model) return;

    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        sum += expert_weights[k] * __half2float(expert_outputs[k * d_model + i]);
    }
    if (residual) sum += __half2float(residual[i]);
    output[i] = __float2half(sum);
}

void moe_weighted_sum_residual(const void* expert_outputs, const float* expert_weights,
                               const void* residual, void* output,
                               int d_model, int top_k, cudaStream_t stream) {
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    moe_weighted_sum_residual_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(expert_outputs), expert_weights,
        static_cast<const half*>(residual),
        static_cast<half*>(output), d_model, top_k);
}

// ============================================================================
// Fused token-centric scatter + FP32->FP16 + residual add (prefill).
//
// One block per output token. Each block reads top_k expert output rows via
// token_to_expanded inverse map, accumulates weighted sum in FP32 registers,
// converts to FP16, optionally adds residual, writes to output.
// No atomicAdd, no output zeroing, no intermediate FP32 buffer.
// ============================================================================

__global__ void moe_scatter_fused_residual_kernel(
        const half* __restrict__ expert_output,  // [expanded, d_model]
        const int32_t* __restrict__ token_to_expanded, // [n_tokens * top_k]
        const float* __restrict__ expert_weights,      // [n_tokens * top_k]
        const half* residual,   // [n_tokens, d_model] or nullptr
        half* output,           // [n_tokens, d_model]
        int d_model, int top_k) {
    const int token = blockIdx.x;
    const int base_flat = token * top_k;

    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            int expanded_row = token_to_expanded[base_flat + k];
            float w = expert_weights[base_flat + k];
            sum += w * __half2float(expert_output[static_cast<int64_t>(expanded_row) * d_model + col]);
        }
        if (residual) sum += __half2float(residual[static_cast<int64_t>(token) * d_model + col]);
        output[static_cast<int64_t>(token) * d_model + col] = __float2half(sum);
    }
}

void moe_scatter_fused_residual(const void* expert_output,
                                 const int32_t* token_to_expanded,
                                 const float* expert_weights,
                                 const void* residual, void* output,
                                 int n_tokens, int d_model, int top_k,
                                 cudaStream_t stream) {
    int threads = 256;
    moe_scatter_fused_residual_kernel<<<n_tokens, threads, 0, stream>>>(
        static_cast<const half*>(expert_output), token_to_expanded, expert_weights,
        static_cast<const half*>(residual), static_cast<half*>(output),
        d_model, top_k);
}

// ============================================================================
// Fused gate GEMV + topk routing launcher
// ============================================================================

void moe_gate_topk_fused(const void* W_gate, const void* x,
                         int n_experts, int d_model, int top_k,
                         MoeRoutingBuffers& buffers,
                         MoeRoutingResult& result,
                         cudaStream_t stream,
                         bool use_sigmoid,
                         bool normalize_weights,
                         const void* score_bias) {
    // Shared memory: logits + optional sel_probs + topk_val + topk_idx
    //                + warp scratch (float[8] + int[8])
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    int probs_arrays = score_bias ? 2 : 1;
    size_t smem = static_cast<size_t>(n_experts) * probs_arrays * sizeof(float)
                + static_cast<size_t>(top_k) * sizeof(float)
                + static_cast<size_t>(top_k) * sizeof(int32_t)
                + NUM_WARPS * sizeof(float)
                + NUM_WARPS * sizeof(int);

    gemv_gate_topk_fused_kernel<<<1, BLOCK_SIZE, smem, stream>>>(
        static_cast<const half*>(W_gate), static_cast<const half*>(x),
        n_experts, d_model, top_k,
        buffers.expert_indices, buffers.expert_weights,
        use_sigmoid, normalize_weights,
        static_cast<const half*>(score_bias));

    // Fill result struct (no ownership — memory belongs to buffers)
    result.owns_memory = false;
    result.expert_indices = make_tensor_2d(buffers.expert_indices, DType::INT32,
                                           1, top_k, true);
    result.expert_weights = make_tensor_2d(buffers.expert_weights, DType::FP32,
                                           1, top_k, true);
    result.sorted_token_ids = make_tensor_1d(buffers.sorted_token_ids, DType::INT32,
                                             top_k, true);
    result.expert_offsets = make_tensor_1d(buffers.expert_offsets, DType::INT32,
                                           n_experts + 1, true);
}

} // namespace imp
