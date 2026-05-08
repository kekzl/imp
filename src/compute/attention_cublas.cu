#include "compute/attention_cublas.h"
#include "core/logging.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

// Forward declarations for kernels defined in executor_kernels.cu
namespace imp {
__global__ __launch_bounds__(256) void fp32_to_fp16_kernel(const float* __restrict__ in,
                                                           half* __restrict__ out, int64_t n);
}  // namespace imp

namespace imp {

static constexpr auto kGemmAlgo = CUBLAS_GEMM_AUTOTUNE;

// ---------------------------------------------------------------------------
// cuBLAS handle (reuse global — same as gemm.cu)
// ---------------------------------------------------------------------------
static cublasHandle_t get_attn_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "imp::attention_cublas: cublasCreate failed (status %d)\n", (int)st);
            abort();
        }
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return handle;
}

// ---------------------------------------------------------------------------
// FP32 causal softmax: reads/writes FP32 S matrix.
// Used when QK^T scores are stored as FP32 (Gemma-4 with scale=1.0).
// ---------------------------------------------------------------------------
__global__ void causal_softmax_fp32_inplace_kernel(float* __restrict__ S, int q_len, int kv_len,
                                                    int q_offset, bool causal) {
    int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;
    float* row_ptr = S + (static_cast<int64_t>(head) * q_len + row) * kv_len;
    int abs_row = q_offset + row;

    float max_val = -FLT_MAX;
    for (int j = tid; j < kv_len; j += blockDim.x)
        max_val = fmaxf(max_val, (causal && j > abs_row) ? -FLT_MAX : row_ptr[j]);

    __shared__ float s_max[32];
    for (int m = 16; m > 0; m >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, m));
    if (lane_id == 0)
        s_max[warp_id] = max_val;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < n_warps) ? s_max[tid] : -FLT_MAX;
        for (int m = 16; m > 0; m >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
        s_max[0] = v;
    }
    __syncthreads();
    max_val = s_max[0];

    float sum_val = 0.0f;
    for (int j = tid; j < kv_len; j += blockDim.x)
        sum_val += (causal && j > abs_row) ? 0.0f : expf(row_ptr[j] - max_val);

    __shared__ float s_sum[32];
    for (int m = 16; m > 0; m >>= 1)
        sum_val += __shfl_xor_sync(0xffffffff, sum_val, m);
    if (lane_id == 0)
        s_sum[warp_id] = sum_val;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < n_warps) ? s_sum[tid] : 0.0f;
        for (int m = 16; m > 0; m >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, m);
        s_sum[0] = v;
    }
    __syncthreads();
    float inv_sum = (s_sum[0] > 0.0f) ? (1.0f / s_sum[0]) : 0.0f;

    for (int j = tid; j < kv_len; j += blockDim.x)
        row_ptr[j] = (causal && j > abs_row) ? 0.0f : expf(row_ptr[j] - max_val) * inv_sum;
}

// ---------------------------------------------------------------------------
// Fused causal mask + in-place softmax kernel
//
// S: [n_heads, seq_len, seq_len] FP16, row-major
// Each block handles one (head, row) pair.
// Algorithm:
//   1. Apply causal mask: S[h][i][j] = -inf for j > i
//   2. Row-wise softmax: max -> exp -> sum -> normalize
//
// Warp-level reductions for max and sum using __shfl_xor_sync.
// ---------------------------------------------------------------------------
__global__ void causal_softmax_inplace_kernel(half* __restrict__ S, int q_len, int kv_len,
                                               int q_offset, bool causal) {
    // Each block processes one row: blockIdx.x = row, blockIdx.y = head
    int row = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;

    half* row_ptr = S + (static_cast<int64_t>(head) * q_len + row) * kv_len;
    int abs_row = q_offset + row;

    // Step 1: Find max (for numerical stability)
    float max_val = -FLT_MAX;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float val;
        if (causal && j > abs_row) {
            val = -FLT_MAX;
        } else {
            val = __half2float(row_ptr[j]);
        }
        max_val = fmaxf(max_val, val);
    }

    // Warp reduction for max
    for (int mask = 16; mask > 0; mask >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_max[32];  // up to 32 warps
    if (lane_id == 0)
        s_max[warp_id] = max_val;
    __syncthreads();

    if (tid < 32) {
        float v = (tid < n_warps) ? s_max[tid] : -FLT_MAX;
        for (int mask = 16; mask > 0; mask >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
        }
        s_max[0] = v;
    }
    __syncthreads();
    max_val = s_max[0];

    // Step 2: Compute exp and sum
    float sum_val = 0.0f;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float val;
        if (causal && j > abs_row) {
            val = 0.0f;
        } else {
            val = expf(__half2float(row_ptr[j]) - max_val);
        }
        sum_val += val;
    }

    // Warp reduction for sum
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_val += __shfl_xor_sync(0xffffffff, sum_val, mask);
    }

    __shared__ float s_sum[32];
    if (lane_id == 0)
        s_sum[warp_id] = sum_val;
    __syncthreads();

    if (tid < 32) {
        float v = (tid < n_warps) ? s_sum[tid] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, mask);
        }
        s_sum[0] = v;
    }
    __syncthreads();
    float inv_sum = (s_sum[0] > 0.0f) ? (1.0f / s_sum[0]) : 0.0f;

    // Step 3: Normalize and write back
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float val;
        if (causal && j > abs_row) {
            val = 0.0f;
        } else {
            val = expf(__half2float(row_ptr[j]) - max_val) * inv_sum;
        }
        row_ptr[j] = __float2half(val);
    }
}

// ---------------------------------------------------------------------------
// Softcap kernel: S[i] = softcap * tanh(S[i] / softcap)
// Applied in-place to the S matrix (FP16) between GEMM and softmax.
// ---------------------------------------------------------------------------
__global__ void softcap_fp16_kernel(half* S, int64_t n, float softcap) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(S[idx]);
        val = softcap * tanhf(val / softcap);
        S[idx] = __float2half(val);
    }
}

// ---------------------------------------------------------------------------
// Static device pointer arrays for cublasGemmBatchedEx (GQA attention).
// Allocated once, grown as needed. Layout: [A_ptrs..., B_ptrs..., C_ptrs...]
// ---------------------------------------------------------------------------
static void** s_attn_d_ptrs = nullptr;
static int s_attn_d_ptrs_capacity = 0;  // in number of pointers

static void ensure_attn_ptr_arrays(int n_heads) {
    int needed = 3 * n_heads;
    if (needed <= s_attn_d_ptrs_capacity)
        return;
    if (s_attn_d_ptrs)
        IMP_CUDA_CHECK_LOG(cudaFree(s_attn_d_ptrs));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&s_attn_d_ptrs, needed * sizeof(void*)));
    s_attn_d_ptrs_capacity = needed;
}

// ---------------------------------------------------------------------------
// cuBLAS batched attention for prefill
//
// Q: [seq, n_heads * hd], K: [seq, n_kv * hd], V: [seq, n_kv * hd]
// O: [seq, n_heads * hd], S: [n_heads, seq, seq] workspace
//
// For GQA (n_kv_heads < n_heads): uses cublasGemmBatchedEx with explicit
// pointer arrays so that multiple Q heads map to the same K/V head in a
// single cuBLAS call. This reduces n_kv_heads calls per direction to 1.
//
// For MHA (n_kv_heads == n_heads): uses cublasGemmStridedBatchedEx for
// maximum efficiency (single call, no pointer arrays needed).
// ---------------------------------------------------------------------------
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim, float scale, bool causal,
                              float softcap, cudaStream_t stream) {
    int seq_len = static_cast<int>(Q.shape[0]);
    if (seq_len == 0)
        return;

    int gqa_ratio = n_heads / n_kv_heads;

    cublasHandle_t handle = get_attn_cublas_handle();
    cublasSetStream(handle, stream);

    const half* Q_base = static_cast<const half*>(Q.data);
    const half* K_base = static_cast<const half*>(K.data);
    const half* V_base = static_cast<const half*>(V.data);
    half* O_base = static_cast<half*>(O.data);
    half* S_base = static_cast<half*>(S.data);

    int ld_q = n_heads * head_dim;
    int ld_k = n_kv_heads * head_dim;
    int ld_s = seq_len;
    int ld_o = n_heads * head_dim;

    long long strideS = static_cast<long long>(seq_len) * seq_len;

    float alpha_f = scale;
    float beta_f = 0.0f;
    float one_f = 1.0f;
    float zero_f = 0.0f;

    // FP32 S matrix: avoids FP16 truncation of attention scores before softmax.
    // Originally gated on scale==1.0 (Gemma-4, where the lack of 1/sqrt(hd)
    // scaling makes QK^T scores large enough for FP16 to truncate). But the
    // FP16-S path also fails for Qwen3.5-27B (head_dim=512, scale=1/sqrt(512))
    // at deeper layers — the residual stream values grow ~5× from L0 to L58
    // and the post-RMSNorm Q/K projections produce attention scores that
    // accumulate FP16 round-off into NaN by L59. Symptom: cuBLAS attention
    // emits all-NaN output at one specific layer and silently produces NaN
    // logits downstream. Use FP32 S whenever the scratch buffer fits — FP16
    // only when forced by buffer constraints.
    int64_t s_buf_fp16_elems = static_cast<int64_t>(S.shape[0]) * S.shape[1];
    if (S.ndim >= 3)
        s_buf_fp16_elems *= S.shape[2];
    int64_t s_fp32_elems = static_cast<int64_t>(n_heads) * seq_len * seq_len;
    bool use_fp32_s = (s_fp32_elems * 2 <= s_buf_fp16_elems);
    float* S_f32 = use_fp32_s ? reinterpret_cast<float*>(S.data) : nullptr;

    if (gqa_ratio == 1) {
        // ---------------------------------------------------------------
        // MHA path: single strided batched call per direction
        // ---------------------------------------------------------------

        // S = scale * Q × K^T (FP32 output when use_fp32_s)
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, head_dim, &alpha_f,
                                   K_base, CUDA_R_16F, ld_k, static_cast<long long>(head_dim), Q_base,
                                   CUDA_R_16F, ld_q, static_cast<long long>(head_dim), &beta_f,
                                   use_fp32_s ? static_cast<void*>(S_f32) : static_cast<void*>(S_base),
                                   use_fp32_s ? CUDA_R_32F : CUDA_R_16F, ld_s, strideS, n_heads,
                                   CUBLAS_COMPUTE_32F, kGemmAlgo);

        // Softcap (if enabled — only for FP16 path, Gemma-4 has no attn softcap)
        if (softcap > 0.0f && !use_fp32_s) {
            int64_t total = static_cast<int64_t>(n_heads) * seq_len * seq_len;
            int block = 256;
            int grid_sc = static_cast<int>((total + block - 1) / block);
            softcap_fp16_kernel<<<grid_sc, block, 0, stream>>>(S_base, total, softcap);
        }

        // Softmax (FP32 or FP16)
        {
            int threads = (seq_len <= 128) ? 128 : ((seq_len <= 256) ? 256 : 512);
            if (threads > 1024)
                threads = 1024;
            dim3 grid(seq_len, n_heads);
            if (use_fp32_s) {
                // FP32 softmax: reads FP32 scores, writes FP32 probabilities
                causal_softmax_fp32_inplace_kernel<<<grid, threads, 0, stream>>>(S_f32, seq_len, seq_len, /*q_offset=*/0, causal);
                // Convert FP32 → FP16 for P@V GEMM
                int64_t total = static_cast<int64_t>(n_heads) * seq_len * seq_len;
                int thr = 256;
                int blk = static_cast<int>((total + thr - 1) / thr);
                fp32_to_fp16_kernel<<<blk, thr, 0, stream>>>(S_f32, S_base, total);
            } else {
                causal_softmax_inplace_kernel<<<grid, threads, 0, stream>>>(S_base, seq_len, seq_len, /*q_offset=*/0, causal);
            }
        }

        // O = P × V (always FP16)
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len, &one_f,
                                   V_base, CUDA_R_16F, ld_k, static_cast<long long>(head_dim), S_base,
                                   CUDA_R_16F, ld_s, strideS, &zero_f, O_base, CUDA_R_16F, ld_o,
                                   static_cast<long long>(head_dim), n_heads, CUBLAS_COMPUTE_32F, kGemmAlgo);

    } else {
        // ---------------------------------------------------------------
        // GQA path: single batched call with explicit pointer arrays
        // Multiple Q heads share the same K/V head.
        // ---------------------------------------------------------------
        ensure_attn_ptr_arrays(n_heads);

        // Build host pointer arrays (stack-allocated, max 256 heads)
        const void* h_A[256];
        const void* h_B[256];
        void* h_C[256];

        // Step 1: S = scale * Q × K^T (FP32 S for Gemma-4)
        for (int h = 0; h < n_heads; h++) {
            int g = h / gqa_ratio;
            h_A[h] = K_base + g * head_dim;
            h_B[h] = Q_base + h * head_dim;
            h_C[h] = use_fp32_s ? static_cast<void*>(S_f32 + h * strideS)
                                : static_cast<void*>(S_base + h * strideS);
        }
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(s_attn_d_ptrs, h_A, n_heads * sizeof(void*), cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s_attn_d_ptrs + n_heads, h_B, n_heads * sizeof(void*),
                                           cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s_attn_d_ptrs + 2 * n_heads, h_C, n_heads * sizeof(void*),
                                           cudaMemcpyHostToDevice, stream));

        cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, head_dim, &alpha_f,
                            (const void**)s_attn_d_ptrs, CUDA_R_16F, ld_k,
                            (const void**)(s_attn_d_ptrs + n_heads), CUDA_R_16F, ld_q, &beta_f,
                            (void**)(s_attn_d_ptrs + 2 * n_heads), use_fp32_s ? CUDA_R_32F : CUDA_R_16F, ld_s,
                            n_heads, CUBLAS_COMPUTE_32F, kGemmAlgo);

        // Softcap (only FP16 path — Gemma-4 has no attn softcap)
        if (softcap > 0.0f && !use_fp32_s) {
            int64_t total = static_cast<int64_t>(n_heads) * seq_len * seq_len;
            int block = 256;
            int grid_sc = static_cast<int>((total + block - 1) / block);
            softcap_fp16_kernel<<<grid_sc, block, 0, stream>>>(S_base, total, softcap);
        }

        // Softmax
        {
            int threads = (seq_len <= 128) ? 128 : ((seq_len <= 256) ? 256 : 512);
            if (threads > 1024)
                threads = 1024;
            dim3 grid(seq_len, n_heads);
            if (use_fp32_s) {
                causal_softmax_fp32_inplace_kernel<<<grid, threads, 0, stream>>>(S_f32, seq_len, seq_len, /*q_offset=*/0, causal);
                // Convert FP32 → FP16 for P@V
                int64_t total = static_cast<int64_t>(n_heads) * seq_len * seq_len;
                int thr = 256;
                int blk = static_cast<int>((total + thr - 1) / thr);
                imp::fp32_to_fp16_kernel<<<blk, thr, 0, stream>>>(S_f32, S_base, total);
            } else {
                causal_softmax_inplace_kernel<<<grid, threads, 0, stream>>>(S_base, seq_len, seq_len, /*q_offset=*/0, causal);
            }
        }

        // Step 3: O = P × V
        // cuBLAS: C = alpha * A * B where A=V (OP_N), B=P (OP_N)
        for (int h = 0; h < n_heads; h++) {
            int g = h / gqa_ratio;
            h_A[h] = V_base + g * head_dim;  // V head (GQA: shared)
            h_B[h] = S_base + h * strideS;   // P head
            h_C[h] = O_base + h * head_dim;  // O head
        }
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(s_attn_d_ptrs, h_A, n_heads * sizeof(void*), cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s_attn_d_ptrs + n_heads, h_B, n_heads * sizeof(void*),
                                           cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s_attn_d_ptrs + 2 * n_heads, h_C, n_heads * sizeof(void*),
                                           cudaMemcpyHostToDevice, stream));

        cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len, &one_f,
                            (const void**)s_attn_d_ptrs, CUDA_R_16F, ld_k,
                            (const void**)(s_attn_d_ptrs + n_heads), CUDA_R_16F, ld_s, &zero_f,
                            (void**)(s_attn_d_ptrs + 2 * n_heads), CUDA_R_16F, ld_o, n_heads,
                            CUBLAS_COMPUTE_32F, kGemmAlgo);
    }
}

}  // namespace imp
