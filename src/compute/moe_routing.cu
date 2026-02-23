#include "compute/moe_routing.h"
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
                                   float* __restrict__ expert_weights) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const float* logits = gate_logits + static_cast<int64_t>(token) * n_experts;

    // --- Step 1: Find max for numerical stability (softmax) ---
    float local_max = -FLT_MAX;
    for (int i = tid; i < n_experts; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[i]);
    }
    local_max = warp_reduce_max(local_max);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) s_warp[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) s_warp[0] = val;
    }
    __syncthreads();
    float gmax = s_warp[0];

    // --- Step 2: Compute exp and sum for softmax ---
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

    // --- Step 3: Compute softmax probabilities and find top-k ---
    // We use shared memory to store the softmax probabilities for all experts.
    // For typical MoE, n_experts is 8-64, which fits easily in shared memory.
    extern __shared__ char smem_raw[];
    float* s_probs   = reinterpret_cast<float*>(smem_raw);
    // s_topk_val and s_topk_idx placed after s_probs
    float*   s_topk_val = s_probs + n_experts;
    int32_t* s_topk_idx = reinterpret_cast<int32_t*>(s_topk_val + top_k);

    // Compute softmax probabilities collaboratively
    for (int i = tid; i < n_experts; i += blockDim.x) {
        s_probs[i] = expf(logits[i] - gmax) * inv_sum;
    }
    __syncthreads();

    // Thread 0 finds top-k using insertion sort (n_experts is small)
    if (tid == 0) {
        for (int j = 0; j < top_k; ++j) {
            s_topk_val[j] = -1.0f;
            s_topk_idx[j] = -1;
        }

        for (int i = 0; i < n_experts; ++i) {
            float p = s_probs[i];
            // Find position to insert (keep sorted descending)
            int pos = -1;
            for (int j = 0; j < top_k; ++j) {
                if (p > s_topk_val[j]) {
                    pos = j;
                    break;
                }
            }
            if (pos >= 0) {
                // Shift elements down
                for (int j = top_k - 1; j > pos; --j) {
                    s_topk_val[j] = s_topk_val[j - 1];
                    s_topk_idx[j] = s_topk_idx[j - 1];
                }
                s_topk_val[pos] = p;
                s_topk_idx[pos] = i;
            }
        }

        // Normalize top-k weights to sum to 1
        float norm = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            norm += s_topk_val[j];
        }
        float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;

        // Write output
        int base = token * top_k;
        for (int j = 0; j < top_k; ++j) {
            expert_indices[base + j] = s_topk_idx[j];
            expert_weights[base + j] = s_topk_val[j] * inv_norm;
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
        int32_t* __restrict__ expert_write_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * top_k;
    if (idx < total) {
        int token = idx / top_k;
        int expert = expert_indices[idx];
        int pos = atomicAdd(&expert_write_pos[expert], 1);
        int dest = expert_offsets[expert] + pos;
        sorted_token_ids[dest] = token;
        sorted_flat_idx[dest]  = idx;  // flat index into expert_weights
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
                     cudaStream_t stream) {
    const int n_tokens  = static_cast<int>(gate_logits.shape[0]);
    const int n_experts = static_cast<int>(gate_logits.shape[1]);
    const float* d_logits = static_cast<const float*>(gate_logits.data);
    const int total_assignments = n_tokens * top_k;

    // ---- Allocate result tensors ----

    // expert_indices: [n_tokens, top_k] int32
    int32_t* d_expert_indices = nullptr;
    cudaMalloc(&d_expert_indices, static_cast<size_t>(total_assignments) * sizeof(int32_t));

    // expert_weights: [n_tokens, top_k] float
    float* d_expert_weights = nullptr;
    cudaMalloc(&d_expert_weights, static_cast<size_t>(total_assignments) * sizeof(float));

    // sorted_token_ids: [total_assignments] int32
    // We allocate 2x to hold a parallel sorted_flat_idx array right after.
    int32_t* d_sorted_token_ids = nullptr;
    cudaMalloc(&d_sorted_token_ids, static_cast<size_t>(total_assignments) * 2 * sizeof(int32_t));
    int32_t* d_sorted_flat_idx = d_sorted_token_ids + total_assignments;

    // expert_offsets: [n_experts + 1] int32
    int32_t* d_expert_offsets = nullptr;
    cudaMalloc(&d_expert_offsets, static_cast<size_t>(n_experts + 1) * sizeof(int32_t));

    // Temporary: expert_counts [n_experts], expert_write_pos [n_experts]
    int32_t* d_expert_counts = nullptr;
    int32_t* d_expert_write_pos = nullptr;
    cudaMalloc(&d_expert_counts, static_cast<size_t>(n_experts) * sizeof(int32_t));
    cudaMalloc(&d_expert_write_pos, static_cast<size_t>(n_experts) * sizeof(int32_t));

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
    size_t smem_gating = static_cast<size_t>(n_experts) * sizeof(float)  // s_probs
                       + static_cast<size_t>(top_k) * sizeof(float)       // s_topk_val
                       + static_cast<size_t>(top_k) * sizeof(int32_t);    // s_topk_idx

    topk_gating_kernel<<<n_tokens, BLOCK_SIZE, smem_gating, stream>>>(
        d_logits, n_experts, top_k, d_expert_indices, d_expert_weights);

    // ---- Kernel 2: Count tokens per expert ----
    int grid_count = (total_assignments + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_tokens_per_expert_kernel<<<grid_count, BLOCK_SIZE, 0, stream>>>(
        d_expert_indices, n_tokens, top_k, d_expert_counts);

    // ---- Kernel 3: Exclusive prefix sum for offsets ----
    exclusive_scan_kernel<<<1, 1, 0, stream>>>(d_expert_counts, d_expert_offsets, n_experts);

    // ---- Kernel 4: Scatter token IDs (and flat indices) into sorted order ----
    scatter_token_ids_with_flat_idx_kernel<<<grid_count, BLOCK_SIZE, 0, stream>>>(
        d_expert_indices, d_expert_offsets, n_tokens, top_k,
        d_sorted_token_ids, d_sorted_flat_idx, d_expert_write_pos);

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

    pool_size = indices_sz + weights_sz + sorted_sz + offsets_sz + counts_sz + wpos_sz;
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
}

// ============================================================================
// moe_topk_gating with pre-allocated buffers
// ============================================================================

void moe_topk_gating(const Tensor& gate_logits, int top_k,
                     MoeRoutingBuffers& buffers,
                     MoeRoutingResult& result,
                     cudaStream_t stream) {
    const int n_tokens  = static_cast<int>(gate_logits.shape[0]);
    const int n_experts = static_cast<int>(gate_logits.shape[1]);
    const float* d_logits = static_cast<const float*>(gate_logits.data);
    const int total_assignments = n_tokens * top_k;

    int32_t* d_expert_indices    = buffers.expert_indices;
    float*   d_expert_weights    = buffers.expert_weights;
    int32_t* d_sorted_token_ids  = buffers.sorted_token_ids;
    int32_t* d_sorted_flat_idx   = d_sorted_token_ids + total_assignments;
    int32_t* d_expert_offsets    = buffers.expert_offsets;
    int32_t* d_expert_counts     = buffers.expert_counts;
    int32_t* d_expert_write_pos  = buffers.expert_write_pos;

    // Zero-initialize counts and write positions
    int grid_z = (n_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_int32_kernel<<<grid_z, BLOCK_SIZE, 0, stream>>>(d_expert_counts, n_experts);
    zero_int32_kernel<<<grid_z, BLOCK_SIZE, 0, stream>>>(d_expert_write_pos, n_experts);

    // Kernel 1: Softmax + top-k selection per token
    size_t smem_gating = static_cast<size_t>(n_experts) * sizeof(float)
                       + static_cast<size_t>(top_k) * sizeof(float)
                       + static_cast<size_t>(top_k) * sizeof(int32_t);

    topk_gating_kernel<<<n_tokens, BLOCK_SIZE, smem_gating, stream>>>(
        d_logits, n_experts, top_k, d_expert_indices, d_expert_weights);

    // Kernel 2: Count tokens per expert
    int grid_count = (total_assignments + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_tokens_per_expert_kernel<<<grid_count, BLOCK_SIZE, 0, stream>>>(
        d_expert_indices, n_tokens, top_k, d_expert_counts);

    // Kernel 3: Exclusive prefix sum for offsets
    exclusive_scan_kernel<<<1, 1, 0, stream>>>(d_expert_counts, d_expert_offsets, n_experts);

    // Kernel 4: Scatter token IDs (and flat indices) into sorted order
    scatter_token_ids_with_flat_idx_kernel<<<grid_count, BLOCK_SIZE, 0, stream>>>(
        d_expert_indices, d_expert_offsets, n_tokens, top_k,
        d_sorted_token_ids, d_sorted_flat_idx, d_expert_write_pos);

    // Fill result struct (no ownership -- memory belongs to buffers)
    result.owns_memory = false;
    result.expert_indices = make_tensor_2d(d_expert_indices, DType::INT32,
                                           n_tokens, top_k, true);
    result.expert_weights = make_tensor_2d(d_expert_weights, DType::FP32,
                                           n_tokens, top_k, true);
    result.sorted_token_ids = make_tensor_1d(d_sorted_token_ids, DType::INT32,
                                             total_assignments, true);
    result.expert_offsets = make_tensor_1d(d_expert_offsets, DType::INT32,
                                           n_experts + 1, true);
}

} // namespace imp
