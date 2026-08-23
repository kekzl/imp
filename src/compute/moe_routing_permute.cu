#include "compute/moe_routing.h"
#include "compute/moe_routing_internal.cuh"
#include "core/logging.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ============================================================================
// Kernel 5: Gather -- reorder tokens by expert assignment
//
// For each position i in sorted_token_ids:
//   gathered[i, :] = input[sorted_token_ids[i], :]
// ============================================================================

template <typename T>
__global__ void moe_gather_kernel_impl(const T* __restrict__ input,
                                       const int32_t* __restrict__ sorted_token_ids, T* __restrict__ gathered,
                                       int total_tokens, int d_model) {
    int row = blockIdx.x;
    if (row >= total_tokens)
        return;

    int src_token = sorted_token_ids[row];
    const T* src = input + static_cast<int64_t>(src_token) * d_model;
    T* dst = gathered + static_cast<int64_t>(row) * d_model;

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

// Scatter-add kernel using the flat index to look up weights.
// Reads from T* expert output (float or half), always accumulates into float* output.
// The to_float helper handles the FP16→FP32 conversion when T=half.
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(half v) { return __half2float(v); }

template <typename T>
__global__ void moe_scatter_kernel_impl(const T* __restrict__ expert_output,
                                        const int32_t* __restrict__ sorted_token_ids,
                                        const int32_t* __restrict__ sorted_flat_idx,
                                        const float* __restrict__ expert_weights, float* __restrict__ output,
                                        int total_tokens, int d_model) {
    int row = blockIdx.x;
    if (row >= total_tokens)
        return;

    int token_id = sorted_token_ids[row];
    int flat_idx = sorted_flat_idx[row];
    float weight = expert_weights[flat_idx];

    const T* src = expert_output + static_cast<int64_t>(row) * d_model;
    float* dst = output + static_cast<int64_t>(token_id) * d_model;

    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        atomicAdd(&dst[col], weight * to_float(src[col]));
    }
}

// Deterministic scatter-add: one block per OUTPUT token. Each block gathers the
// sorted rows belonging to its token, accumulates them (in FP32 registers, in
// ascending row order) and writes once. Avoids the FP atomicAdd of
// moe_scatter_kernel_impl whose accumulation order is scheduling-dependent
// (non-reproducible). Opt-in only (deterministic mode); the default path keeps
// the faster atomic scatter.
//
// The gather used to sit INSIDE the column loop, so the O(total_rows) row scan
// ran ceil(d_model / blockDim) times per token rather than once: 8 times over
// 4096 rows for a 2048-wide model at 512 tokens, to find the same 8 rows every
// time (#1546). It runs once now, and the column loop reads a shared-memory
// list of length top_k.
//
// `cap` is the caller's upper bound on rows per token (0 = unknown). Exceeding
// it falls back to the original scan, which produces the same numbers.
constexpr int kMaxDetRowsPerToken = 64;

template <typename T>
__global__ void moe_scatter_deterministic_kernel_impl(const T* __restrict__ expert_output,
                                                      const int32_t* __restrict__ sorted_token_ids,
                                                      const int32_t* __restrict__ sorted_flat_idx,
                                                      const float* __restrict__ expert_weights,
                                                      float* __restrict__ output, int total_rows,
                                                      int n_tokens, int d_model, int cap) {
    const int token = blockIdx.x;
    if (token >= n_tokens)
        return;

    extern __shared__ int32_t s_rows[];
    __shared__ int s_count;
    if (threadIdx.x == 0)
        s_count = 0;
    __syncthreads();

    if (cap > 0) {
        for (int row = threadIdx.x; row < total_rows; row += blockDim.x) {
            if (sorted_token_ids[row] != token)
                continue;
            const int slot = atomicAdd(&s_count, 1);
            if (slot < cap)
                s_rows[slot] = row;
        }
        __syncthreads();
        // Ascending row order IS the contract of this kernel: it is what makes
        // the FP32 accumulation reproducible. atomicAdd appended in scheduling
        // order, so sort the handful of entries back. top_k is 4-8 on every
        // shipped MoE checkpoint, so one thread insertion-sorting beats a
        // barrier-heavy parallel sort.
        if (threadIdx.x == 0 && s_count <= cap) {
            for (int i = 1; i < s_count; ++i) {
                const int32_t v = s_rows[i];
                int j = i - 1;
                while (j >= 0 && s_rows[j] > v) {
                    s_rows[j + 1] = s_rows[j];
                    --j;
                }
                s_rows[j + 1] = v;
            }
        }
        __syncthreads();
    }

    if (cap > 0 && s_count <= cap) {
        const int k = s_count;
        for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                const int row = s_rows[i];
                const float weight = expert_weights[sorted_flat_idx[row]];
                sum += weight * to_float(expert_output[static_cast<int64_t>(row) * d_model + col]);
            }
            output[static_cast<int64_t>(token) * d_model + col] = sum;
        }
        return;
    }

    // Fallback: a token carrying more rows than the caller's bound allowed for.
    // Same ascending-row accumulation, so the same numbers, just slower.
    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        float sum = 0.0f;
        for (int row = 0; row < total_rows; ++row) {
            if (sorted_token_ids[row] != token)
                continue;
            const float weight = expert_weights[sorted_flat_idx[row]];
            sum += weight * to_float(expert_output[static_cast<int64_t>(row) * d_model + col]);
        }
        output[static_cast<int64_t>(token) * d_model + col] = sum;
    }
}

// ============================================================================
// Utility: zero-initialize device memory
// ============================================================================

__global__ void zero_float_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = 0.0f;
}

// ============================================================================
// Public API: moe_gather
// ============================================================================

void moe_gather(const Tensor& input, const MoeRoutingResult& routing, Tensor& gathered, cudaStream_t stream) {
    const int d_model = static_cast<int>(input.shape[1]);
    const int total_tokens = static_cast<int>(routing.sorted_token_ids.shape[0]);
    const int32_t* d_sorted = static_cast<const int32_t*>(routing.sorted_token_ids.data);

    if (input.qtype == QType::F16) {
        const half* d_input = static_cast<const half*>(input.data);
        half* d_gathered = static_cast<half*>(gathered.data);
        moe_gather_kernel_impl<<<total_tokens, BLOCK_SIZE, 0, stream>>>(d_input, d_sorted, d_gathered,
                                                                        total_tokens, d_model);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        const float* d_input = static_cast<const float*>(input.data);
        float* d_gathered = static_cast<float*>(gathered.data);
        moe_gather_kernel_impl<<<total_tokens, BLOCK_SIZE, 0, stream>>>(d_input, d_sorted, d_gathered,
                                                                        total_tokens, d_model);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

// ============================================================================
// Public API: moe_scatter
// ============================================================================

void moe_scatter(const Tensor& expert_output, const MoeRoutingResult& routing, Tensor& output,
                 cudaStream_t stream) {
    const int d_model = static_cast<int>(expert_output.shape[1]);
    const int total_tokens = static_cast<int>(routing.sorted_token_ids.shape[0]);
    const int n_tokens = static_cast<int>(output.shape[0]);

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

    if (process_diag_deterministic_gemm()) {
        // Deterministic mode: one block per output token, fixed-order FP32
        // accumulation over its rows. Writes output directly (no atomics, no
        // pre-zero needed). total_tokens here is the number of expanded rows.
        //
        // Rows per token is exactly top_k: top-k gating gives every token that
        // many assignments and the permute assigns each one a slot, so the
        // division is exact. Anything else (a routing path that drops or
        // duplicates) leaves cap at 0 and the kernel takes its fallback scan.
        int cap = 0;
        if (n_tokens > 0 && total_tokens % n_tokens == 0) {
            const int rows_per_token = total_tokens / n_tokens;
            if (rows_per_token > 0 && rows_per_token <= kMaxDetRowsPerToken)
                cap = rows_per_token;
        }
        const size_t smem = static_cast<size_t>(cap) * sizeof(int32_t);
        if (expert_output.qtype == QType::F16) {
            const half* d_expert_out = static_cast<const half*>(expert_output.data);
            moe_scatter_deterministic_kernel_impl<<<n_tokens, BLOCK_SIZE, smem, stream>>>(
                d_expert_out, d_sorted_token_ids, d_sorted_flat_idx, d_expert_weights, d_output, total_tokens,
                n_tokens, d_model, cap);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            const float* d_expert_out = static_cast<const float*>(expert_output.data);
            moe_scatter_deterministic_kernel_impl<<<n_tokens, BLOCK_SIZE, smem, stream>>>(
                d_expert_out, d_sorted_token_ids, d_sorted_flat_idx, d_expert_weights, d_output, total_tokens,
                n_tokens, d_model, cap);
            IMP_CUDA_CHECK_LAUNCH();
        }
        return;
    }

    // Zero the output first (scatter-add accumulates into it)
    zero_float_kernel<<<grid_z, BLOCK_SIZE, 0, stream>>>(d_output, total_out_elems);
    IMP_CUDA_CHECK_LAUNCH();

    if (expert_output.qtype == QType::F16) {
        const half* d_expert_out = static_cast<const half*>(expert_output.data);
        moe_scatter_kernel_impl<<<total_tokens, BLOCK_SIZE, 0, stream>>>(d_expert_out, d_sorted_token_ids,
                                                                         d_sorted_flat_idx, d_expert_weights,
                                                                         d_output, total_tokens, d_model);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        const float* d_expert_out = static_cast<const float*>(expert_output.data);
        moe_scatter_kernel_impl<<<total_tokens, BLOCK_SIZE, 0, stream>>>(d_expert_out, d_sorted_token_ids,
                                                                         d_sorted_flat_idx, d_expert_weights,
                                                                         d_output, total_tokens, d_model);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

// ============================================================================
// Fused weighted sum + FP16 output + optional residual add.
// Eliminates the FP32 intermediate buffer and fp32_to_fp16 conversion kernel.
// ============================================================================

__global__ void moe_weighted_sum_residual_kernel(const half* __restrict__ expert_outputs,
                                                 const float* __restrict__ expert_weights,
                                                 const half* residual,  // may be nullptr, may alias output
                                                 half* output,  // no __restrict__ — may alias residual
                                                 int d_model, int top_k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_model)
        return;

    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        sum += expert_weights[k] * __half2float(expert_outputs[k * d_model + i]);
    }
    if (residual)
        sum += __half2float(residual[i]);
    output[i] = __float2half(sum);
}

void moe_weighted_sum_residual(const void* expert_outputs, const float* expert_weights, const void* residual,
                               void* output, int d_model, int top_k, cudaStream_t stream) {
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    moe_weighted_sum_residual_kernel<<<blocks, threads, 0, stream>>>(static_cast<const half*>(expert_outputs),
                                                                     expert_weights,
                                                                     static_cast<const half*>(residual),
                                                                     static_cast<half*>(output), d_model,
                                                                     top_k);
    IMP_CUDA_CHECK_LAUNCH();
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
    const half* __restrict__ expert_output,         // [expanded, d_model]
    const int32_t* __restrict__ token_to_expanded,  // [n_tokens * top_k]
    const float* __restrict__ expert_weights,       // [n_tokens * top_k]
    const half* residual,                           // [n_tokens, d_model] or nullptr
    half* output,                                   // [n_tokens, d_model]
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
        if (residual)
            sum += __half2float(residual[static_cast<int64_t>(token) * d_model + col]);
        output[static_cast<int64_t>(token) * d_model + col] = __float2half(sum);
    }
}

void moe_scatter_fused_residual(const void* expert_output, const int32_t* token_to_expanded,
                                const float* expert_weights, const void* residual, void* output, int n_tokens,
                                int d_model, int top_k, cudaStream_t stream) {
    int threads = 256;
    moe_scatter_fused_residual_kernel<<<n_tokens, threads, 0, stream>>>(
        static_cast<const half*>(expert_output), token_to_expanded, expert_weights,
        static_cast<const half*>(residual), static_cast<half*>(output), d_model, top_k);
    IMP_CUDA_CHECK_LAUNCH();
}

// FP32-input variant: expert_output is float* (not half*).
// Used by IMP_GEMMA4_FP32_EXPERT_DOWN to test if FP16 truncation at down GEMM
// output is the source of the L3+ precision drift.
__global__ void moe_scatter_fused_residual_fp32in_kernel(
    const float* __restrict__ expert_output_fp32,  // [expanded, d_model] FP32
    const int32_t* __restrict__ token_to_expanded, const float* __restrict__ expert_weights,
    const half* residual, half* output, int d_model, int top_k) {
    const int token = blockIdx.x;
    const int base_flat = token * top_k;

    for (int col = threadIdx.x; col < d_model; col += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            int expanded_row = token_to_expanded[base_flat + k];
            float w = expert_weights[base_flat + k];
            sum += w * expert_output_fp32[static_cast<int64_t>(expanded_row) * d_model + col];
        }
        if (residual)
            sum += __half2float(residual[static_cast<int64_t>(token) * d_model + col]);
        output[static_cast<int64_t>(token) * d_model + col] = __float2half(sum);
    }
}

void moe_scatter_fused_residual_fp32in(const void* expert_output_fp32, const int32_t* token_to_expanded,
                                       const float* expert_weights, const void* residual, void* output,
                                       int n_tokens, int d_model, int top_k, cudaStream_t stream) {
    int threads = 256;
    moe_scatter_fused_residual_fp32in_kernel<<<n_tokens, threads, 0, stream>>>(
        static_cast<const float*>(expert_output_fp32), token_to_expanded, expert_weights,
        static_cast<const half*>(residual), static_cast<half*>(output), d_model, top_k);
    IMP_CUDA_CHECK_LAUNCH();
}

// ============================================================================
// gpt-oss bias kernels (issue #547)
// ============================================================================

__global__ void moe_add_logit_bias_kernel(float* __restrict__ logits, const half* __restrict__ bias,
                                          int64_t total, int ne) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
        logits[idx] += __half2float(bias[idx % ne]);
}

__global__ void moe_add_expert_bias_indexed_kernel(half* __restrict__ buf, const half* __restrict__ bias,
                                                   const int32_t* __restrict__ expert_indices,
                                                   int64_t total, int dim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        int row = static_cast<int>(idx / dim);
        int e = expert_indices[row];
        buf[idx] = __hadd(buf[idx], bias[static_cast<int64_t>(e) * dim + idx % dim]);
    }
}

__global__ void moe_add_expert_bias_sorted_kernel(half* __restrict__ buf, const half* __restrict__ bias,
                                                  const int32_t* __restrict__ expert_offsets, int ne,
                                                  int64_t total, int dim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    int row = static_cast<int>(idx / dim);
    // Binary search: expert e with offsets[e] <= row < offsets[e+1]. ne is
    // small (32 for gpt-oss-20b) — 5 iterations, all from L2/__ldg.
    int lo = 0, hi = ne - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (__ldg(&expert_offsets[mid]) <= row)
            lo = mid;
        else
            hi = mid - 1;
    }
    buf[idx] = __hadd(buf[idx], bias[static_cast<int64_t>(lo) * dim + idx % dim]);
}

void moe_add_logit_bias(float* logits_f32, const void* bias_fp16, int n, int ne, cudaStream_t stream) {
    int64_t total = static_cast<int64_t>(n) * ne;
    if (total == 0)
        return;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    moe_add_logit_bias_kernel<<<blocks, threads, 0, stream>>>(
        logits_f32, static_cast<const half*>(bias_fp16), total, ne);
    IMP_CUDA_CHECK_LAUNCH();
}

void moe_add_expert_bias_indexed(void* buf_fp16, const void* bias_fp16, const int32_t* expert_indices,
                                 int n_rows, int dim, cudaStream_t stream) {
    int64_t total = static_cast<int64_t>(n_rows) * dim;
    if (total == 0)
        return;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    moe_add_expert_bias_indexed_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<half*>(buf_fp16), static_cast<const half*>(bias_fp16), expert_indices, total, dim);
    IMP_CUDA_CHECK_LAUNCH();
}

void moe_add_expert_bias_sorted(void* buf_fp16, const void* bias_fp16, const int32_t* expert_offsets,
                                int ne, int n_rows, int dim, cudaStream_t stream) {
    int64_t total = static_cast<int64_t>(n_rows) * dim;
    if (total == 0)
        return;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    moe_add_expert_bias_sorted_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<half*>(buf_fp16), static_cast<const half*>(bias_fp16), expert_offsets, ne, total, dim);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
