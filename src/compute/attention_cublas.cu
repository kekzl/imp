#include "compute/attention_cublas.h"
#include "compute/gemm_lt_ex.h"
#include "core/cuda_errors.h"
#include "core/cublas_check.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

// Forward declarations for kernels defined in executor_kernels.cu
namespace imp {
__global__ __launch_bounds__(256) void fp32_to_fp16_kernel(const float* __restrict__ in,
                                                           half* __restrict__ out, int64_t n);
}  // namespace imp

namespace imp {


// Coverage instrumentation (FA2-coverage dispatch): counts materialized-cuBLAS prefill launches.
// A Gemma-4 prefill test asserts this stays 0, proving the legacy path is unreachable for the
// target set. Relaxed atomic, diagnostic only; never gates behavior.
static std::atomic<uint64_t> s_cublas_prefill_calls{0};
uint64_t attention_cublas_prefill_call_count() {
    return s_cublas_prefill_calls.load(std::memory_order_relaxed);
}
void attention_cublas_prefill_reset_count() {
    s_cublas_prefill_calls.store(0, std::memory_order_relaxed);
}

// Fused causal softmax FP32->FP16: reads FP32 S, writes FP16 probs to a separate buffer, saving
// one pass over [n_heads, q_len, kv_len]. FP32 reduction internal; only the final value is downcast.
__global__ void causal_softmax_fp32_to_fp16_kernel(const float* __restrict__ S_in,
                                                    half* __restrict__ S_out, int q_len,
                                                    int kv_len, int q_offset, bool causal,
                                                    int sliding_window,
                                                    const half* __restrict__ sinks) {
    int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;
    int64_t row_base = ((static_cast<int64_t>(head) * q_len) + row) * kv_len;
    const float* row_in = S_in + row_base;
    half* row_out = S_out + row_base;
    int abs_row = q_offset + row;
    // sliding_window > 0: mask j where (abs_row - j) >= sliding_window.
    auto masked = [abs_row, causal, sliding_window](int j) {
        if (causal && j > abs_row) return true;
        if (sliding_window > 0 && (abs_row - j) >= sliding_window) return true;
        return false;
    };

    float max_val = -FLT_MAX;
    for (int j = tid; j < kv_len; j += blockDim.x)
        max_val = fmaxf(max_val, masked(j) ? -FLT_MAX : row_in[j]);

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

    // gpt-oss sink (#547): virtual extra column — joins max + denominator,
    // dropped from the output (probabilities sum to < 1).
    float sink_val = sinks ? __half2float(sinks[head]) : -FLT_MAX;
    if (sinks)
        max_val = fmaxf(max_val, sink_val);

    float sum_val = 0.0f;
    for (int j = tid; j < kv_len; j += blockDim.x)
        sum_val += masked(j) ? 0.0f : expf(row_in[j] - max_val);

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
    float denom = s_sum[0] + (sinks ? expf(sink_val - max_val) : 0.0f);
    float inv_sum = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    for (int j = tid; j < kv_len; j += blockDim.x) {
        float v = masked(j) ? 0.0f : expf(row_in[j] - max_val) * inv_sum;
        row_out[j] = __float2half(v);
    }
}

// Fused causal mask + in-place softmax. S: [n_heads, seq_len, seq_len] FP16. Each block handles
// one (head, row) pair: mask j>i, then row-wise softmax (max -> exp -> sum -> normalize) using
// warp-level __shfl_xor_sync reductions.
__global__ void causal_softmax_inplace_kernel(half* __restrict__ S, int q_len, int kv_len,
                                               int q_offset, bool causal, int sliding_window,
                                               const half* __restrict__ sinks) {
    // Each block processes one row: blockIdx.x = row, blockIdx.y = head
    int row = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;

    half* row_ptr = S + (((static_cast<int64_t>(head) * q_len) + row) * kv_len);
    int abs_row = q_offset + row;
    auto masked = [abs_row, causal, sliding_window](int j) {
        if (causal && j > abs_row) return true;
        if (sliding_window > 0 && (abs_row - j) >= sliding_window) return true;
        return false;
    };

    // Step 1: Find max (for numerical stability)
    float max_val = -FLT_MAX;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float val;
        if (masked(j)) {
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

    // gpt-oss sink (#547): virtual extra column — joins max + denominator,
    // dropped from the output (probabilities sum to < 1).
    float sink_val = sinks ? __half2float(sinks[head]) : -FLT_MAX;
    if (sinks)
        max_val = fmaxf(max_val, sink_val);

    // Step 2: Compute exp and sum
    float sum_val = 0.0f;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float val;
        if (masked(j)) {
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
    float denom = s_sum[0] + (sinks ? expf(sink_val - max_val) : 0.0f);
    float inv_sum = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    // Step 3: Normalize and write back
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float val;
        if (masked(j)) {
            val = 0.0f;
        } else {
            val = expf(__half2float(row_ptr[j]) - max_val) * inv_sum;
        }
        row_ptr[j] = __float2half(val);
    }
}

// Softcap: S[i] = softcap*tanh(S[i]/softcap), in-place on FP16 S between GEMM and softmax.
__global__ void softcap_fp16_kernel(half* S, int64_t n, float softcap) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (idx < n) {
        float val = __half2float(S[idx]);
        val = softcap * tanhf(val / softcap);
        S[idx] = __float2half(val);
    }
}

// FP32 counterpart: the use_fp32_s path keeps scores in a float scratch and the
// FP32 softmax reads them directly, so softcap must be applied to the float
// buffer here (Gemma-2 attn_logit_softcap=50 was silently dropped on this path).
__global__ void softcap_fp32_kernel(float* S, int64_t n, float softcap) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (idx < n) {
        float val = S[idx];
        S[idx] = softcap * tanhf(val / softcap);
    }
}

// cuBLAS batched attention for prefill. Q:[seq,n_heads*hd], K/V:[seq,n_kv*hd], O:[seq,n_heads*hd],
// S:[n_heads,seq,seq] workspace. cuBLASLt strided batches, one per KV head under GQA.
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim, float scale, bool causal,
                              float softcap, int q_offset, cudaStream_t stream, int sliding_window,
                              const void* sinks) {
    s_cublas_prefill_calls.fetch_add(1, std::memory_order_relaxed);
    const half* sinks_h = static_cast<const half*>(sinks);
    int q_len = static_cast<int>(Q.shape[0]);
    int kv_len = static_cast<int>(K.shape[0]);
    if (q_len == 0)
        return;

    int gqa_ratio = n_heads / n_kv_heads;

    const half* Q_base = static_cast<const half*>(Q.data);
    const half* K_base = static_cast<const half*>(K.data);
    const half* V_base = static_cast<const half*>(V.data);
    half* O_base = static_cast<half*>(O.data);
    half* S_base = static_cast<half*>(S.data);

    int ld_q = n_heads * head_dim;
    int ld_k = n_kv_heads * head_dim;
    int ld_s = kv_len;
    int ld_o = n_heads * head_dim;

    long long strideS = static_cast<long long>(q_len) * kv_len;

    float alpha_f = scale;
    float beta_f = 0.0f;
    float one_f = 1.0f;
    float zero_f = 0.0f;

    // FP32 S avoids FP16 truncation of attention scores before softmax: FP16 S can accumulate
    // round-off into NaN at deep layers (seen on Qwen3.5-27B, hd=512) and emit all-NaN output.
    // Use FP32 S whenever the scratch buffer fits; FP16 only when forced by buffer constraints.
    int64_t s_buf_fp16_elems = static_cast<int64_t>(S.shape[0]) * S.shape[1];
    if (S.ndim >= 3)
        s_buf_fp16_elems *= S.shape[2];
    int64_t s_fp32_elems = static_cast<int64_t>(n_heads) * q_len * kv_len;
    // FP32-S needs 3x the score element count, not 2x (#677): FP32 QK^T scores occupy the front
    // 2x half-slots, FP16 probs go to a separate non-overlapping region (aliasing causes a
    // cross-block WAR hazard, nondeterministic across calls). 2x<=buf<3x falls back to FP16-S softmax.
    bool use_fp32_s = (s_fp32_elems * 3 <= s_buf_fp16_elems);
    float* S_f32 = use_fp32_s ? reinterpret_cast<float*>(S.data) : nullptr;
    // FP16 probabilities: non-overlapping with the FP32 scores on the FP32-S
    // path; identical to S_base on the FP16-S path (in-place, safe).
    half* S_prob = use_fp32_s ? (S_base + (2 * s_fp32_elems)) : S_base;

    // One strided batch per KV head: its gqa_ratio Q heads against a shared K / V (stride 0);
    // MHA is one batch over all heads.
    const int groups = gqa_ratio == 1 ? 1 : n_kv_heads;
    const int per_group = n_heads / groups;
    const long long kv_stride = gqa_ratio == 1 ? head_dim : 0;
    void* S_qk = use_fp32_s ? static_cast<void*>(S_f32) : static_cast<void*>(S_base);
    const cudaDataType_t s_type = use_fp32_s ? CUDA_R_32F : CUDA_R_16F;
    const size_t s_elem = use_fp32_s ? sizeof(float) : sizeof(half);

    // S = scale * Q x K^T
    for (int g = 0; g < groups; g++) {
        const long long h0 = static_cast<long long>(g) * per_group;
        IMP_CUBLAS_CHECK_LOG(
            lt_gemm_strided_batched(CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, head_dim, &alpha_f,
                                    K_base + (static_cast<ptrdiff_t>(g) * head_dim), CUDA_R_16F, ld_k,
                                    kv_stride, Q_base + (h0 * head_dim), CUDA_R_16F, ld_q, head_dim, &beta_f,
                                    static_cast<char*>(S_qk) +
                                        (h0 * strideS * static_cast<long long>(s_elem)),
                                    s_type, ld_s, strideS, per_group, CUBLAS_COMPUTE_32F, stream));
    }

    // Softcap (if enabled) on the buffer the softmax reads; Gemma-2 sets softcap=50.
    if (softcap > 0.0f) {
        int64_t total = static_cast<int64_t>(n_heads) * q_len * kv_len;
        int block = 256;
        int grid_sc = static_cast<int>((total + block - 1) / block);
        if (use_fp32_s) {
            softcap_fp32_kernel<<<grid_sc, block, 0, stream>>>(S_f32, total, softcap);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            softcap_fp16_kernel<<<grid_sc, block, 0, stream>>>(S_base, total, softcap);
            IMP_CUDA_CHECK_LAUNCH();
        }
    }

    // Softmax: FP32 scores -> FP16 probabilities (fused downcast), or FP16 in place.
    {
        int threads = (kv_len <= 128) ? 128 : ((kv_len <= 256) ? 256 : 512);
        threads = std::min(threads, 1024);
        dim3 grid(q_len, n_heads);
        if (use_fp32_s) {
            causal_softmax_fp32_to_fp16_kernel<<<grid, threads, 0, stream>>>(S_f32, S_prob, q_len, kv_len,
                                                                             q_offset, causal, sliding_window,
                                                                             sinks_h);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            causal_softmax_inplace_kernel<<<grid, threads, 0, stream>>>(S_base, q_len, kv_len, q_offset,
                                                                        causal, sliding_window, sinks_h);
            IMP_CUDA_CHECK_LAUNCH();
        }
    }

    // O = P x V (FP16), P read from S_prob (#677).
    for (int g = 0; g < groups; g++) {
        const long long h0 = static_cast<long long>(g) * per_group;
        IMP_CUBLAS_CHECK_LOG(lt_gemm_strided_batched(CUBLAS_OP_N, CUBLAS_OP_N, head_dim, q_len, kv_len,
                                                     &one_f, V_base + (static_cast<ptrdiff_t>(g) * head_dim),
                                                     CUDA_R_16F, ld_k, kv_stride, S_prob + (h0 * strideS),
                                                     CUDA_R_16F, ld_s, strideS, &zero_f,
                                                     O_base + (h0 * head_dim), CUDA_R_16F, ld_o, head_dim,
                                                     per_group, CUBLAS_COMPUTE_32F, stream));
    }
}

bool attention_cublas_prefill_sliced(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                                     Tensor& S, int n_heads, int n_kv_heads, int head_dim, float scale,
                                     bool causal, float softcap, int q_offset, cudaStream_t stream,
                                     int sliding_window, const void* sinks) {
    const int n = static_cast<int>(Q.shape[0]);
    const int kv_len = static_cast<int>(K.shape[0]);
    if (n == 0)
        return true;
    int64_t s_buf_fp16_elems = static_cast<int64_t>(S.shape[0]) * S.shape[1];
    if (S.ndim >= 3)
        s_buf_fp16_elems *= S.shape[2];
    // Largest slice keeping every call on the FP32-S path (use_fp32_s gate above). kv_len is fixed
    // across slices, so the solve is linear, not quadratic.
    int64_t ns64 = s_buf_fp16_elems / (3LL * n_heads * kv_len);
    int ns = static_cast<int>(std::min<int64_t>(ns64, n));
    ns = (ns / 16) * 16;
    if (ns < 16)
        return false;  // workspace too degraded — caller picks the O(n) FMHA fallback
    const half* Q_base = static_cast<const half*>(Q.data);
    half* O_base = static_cast<half*>(O.data);
    const int64_t row = static_cast<int64_t>(n_heads) * head_dim;
    for (int off = 0; off < n; off += ns) {
        const int m = std::min(ns, n - off);
        int64_t q2[2] = {m, row};
        Tensor Qs(const_cast<half*>(Q_base) + (off * row), QType::F16, 2, q2, true);
        Tensor Os(O_base + (off * row), QType::F16, 2, q2, true);
        attention_cublas_prefill(Qs, K, V, Os, S, n_heads, n_kv_heads, head_dim, scale, causal, softcap,
                                 q_offset + off, stream, sliding_window, sinks);
    }
    return true;
}

}  // namespace imp
