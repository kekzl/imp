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

// Forward-decl: definition lower in the file alongside s_attn_d_ptrs.
static void ensure_attn_ptr_arrays(int n_heads);

void attention_cublas_prewarm() {
    // Force lazy-init of the static cuBLAS handle AND issue a dummy
    // GemmBatchedEx so cuBLAS allocates its internal workspace + selects
    // an algorithm. Also pre-size the s_attn_d_ptrs device buffer for
    // the largest n_heads any current model uses (256, matching the host
    // stack array bound). All cudaMallocs happen eagerly here; subsequent
    // calls inside captured streams find everything ready and don't
    // trigger any cudaMalloc (illegal under capture).
    cublasHandle_t h = get_attn_cublas_handle();
    // The handle is process-global and outlives engines. Its last
    // cublasSetStream() may reference a stream the previous engine destroyed
    // (server model auto-swap) — issuing the dummy GEMM there segfaults inside
    // cuBLAS algo selection (cuStreamGetGreenCtx on the dangling stream).
    // Rebind to the default stream; real callers set their own stream before
    // every use.
    cublasSetStream(h, nullptr);
    ensure_attn_ptr_arrays(/*n_heads=*/256);

    constexpr int kM = 8, kN = 8, kK = 8;
    half *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    void *d_ap = nullptr, *d_bp = nullptr, *d_cp = nullptr;
    if (cudaMalloc(&d_a, kM * kK * sizeof(half)) != cudaSuccess) return;
    if (cudaMalloc(&d_b, kK * kN * sizeof(half)) != cudaSuccess) { cudaFree(d_a); return; }
    if (cudaMalloc(&d_c, kM * kN * sizeof(half)) != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); return;
    }
    if (cudaMalloc(&d_ap, sizeof(void*)) != cudaSuccess ||
        cudaMalloc(&d_bp, sizeof(void*)) != cudaSuccess ||
        cudaMalloc(&d_cp, sizeof(void*)) != cudaSuccess) {
        if (d_ap) cudaFree(d_ap); if (d_bp) cudaFree(d_bp); if (d_cp) cudaFree(d_cp);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return;
    }
    cudaMemset(d_a, 0, kM * kK * sizeof(half));
    cudaMemset(d_b, 0, kK * kN * sizeof(half));
    cudaMemcpy(d_ap, &d_a, sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bp, &d_b, sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cp, &d_c, sizeof(void*), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    (void)cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kN, kM, kK, &alpha,
                              (const void**)d_ap, CUDA_R_16F, kK,
                              (const void**)d_bp, CUDA_R_16F, kK, &beta,
                              (void**)d_cp, CUDA_R_16F, kN, 1, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_ap); cudaFree(d_bp); cudaFree(d_cp);
}

// ---------------------------------------------------------------------------
// Fused causal softmax FP32 → FP16: reads FP32 S matrix, writes FP16 probs
// to a separate output buffer. Replaces causal_softmax_fp32_inplace_kernel
// + fp32_to_fp16_kernel for the cuBLAS prefill path. Saves one full pass
// over the [n_heads × q_len × kv_len] tensor (~36% memory traffic on the
// softmax+cast block, ~6-8% prefill on dense Q8). FP32 reduction internal,
// only the final normalized value is downcast.
// ---------------------------------------------------------------------------
__global__ void causal_softmax_fp32_to_fp16_kernel(const float* __restrict__ S_in,
                                                    half* __restrict__ S_out, int q_len,
                                                    int kv_len, int q_offset, bool causal,
                                                    int sliding_window,
                                                    const half* __restrict__ sinks) {
    int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;
    int64_t row_base = (static_cast<int64_t>(head) * q_len + row) * kv_len;
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

// ---------------------------------------------------------------------------
// FP32 causal softmax: reads/writes FP32 S matrix.
// Used when QK^T scores are stored as FP32 (Gemma-4 with scale=1.0).
// ---------------------------------------------------------------------------
__global__ void causal_softmax_fp32_inplace_kernel(float* __restrict__ S, int q_len, int kv_len,
                                                    int q_offset, bool causal, int sliding_window) {
    int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;
    float* row_ptr = S + (static_cast<int64_t>(head) * q_len + row) * kv_len;
    int abs_row = q_offset + row;
    auto masked = [abs_row, causal, sliding_window](int j) {
        if (causal && j > abs_row) return true;
        if (sliding_window > 0 && (abs_row - j) >= sliding_window) return true;
        return false;
    };

    float max_val = -FLT_MAX;
    for (int j = tid; j < kv_len; j += blockDim.x)
        max_val = fmaxf(max_val, masked(j) ? -FLT_MAX : row_ptr[j]);

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
        sum_val += masked(j) ? 0.0f : expf(row_ptr[j] - max_val);

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
        row_ptr[j] = masked(j) ? 0.0f : expf(row_ptr[j] - max_val) * inv_sum;
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
                                               int q_offset, bool causal, int sliding_window,
                                               const half* __restrict__ sinks) {
    // Each block processes one row: blockIdx.x = row, blockIdx.y = head
    int row = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;

    half* row_ptr = S + (static_cast<int64_t>(head) * q_len + row) * kv_len;
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

// FP32 counterpart: the use_fp32_s path keeps scores in a float scratch and the
// FP32 softmax reads them directly, so softcap must be applied to the float
// buffer here (Gemma-2 attn_logit_softcap=50 was silently dropped on this path).
__global__ void softcap_fp32_kernel(float* S, int64_t n, float softcap) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = S[idx];
        S[idx] = softcap * tanhf(val / softcap);
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

// Fill s_attn_d_ptrs device-side so the GQA cuBLAS batched path is
// graph-capturable. The previous implementation built host stack arrays and
// issued cudaMemcpyAsync — host pointers have no stable identity across
// graph replays and the H2D copies abort capture. Pointer pattern:
//   A: GQA-shared,   ptr = base_A + (h / gqa_ratio) * stride_A_bytes
//   B: per-head,     ptr = base_B + h * stride_B_bytes
//   C: per-head,     ptr = base_C + h * stride_C_bytes
__global__ void build_attn_ptr_arrays_kernel(const void** d_A, const void** d_B, void** d_C,
                                              const char* base_A, int64_t stride_A_bytes,
                                              const char* base_B, int64_t stride_B_bytes,
                                              char* base_C, int64_t stride_C_bytes, int gqa_ratio,
                                              int n_heads) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads)
        return;
    int g = h / gqa_ratio;
    d_A[h] = base_A + g * stride_A_bytes;
    d_B[h] = base_B + h * stride_B_bytes;
    d_C[h] = base_C + h * stride_C_bytes;
}

static inline void launch_build_attn_ptrs(void** d_ptrs, int n_heads, const void* base_A,
                                          int64_t stride_A_bytes, const void* base_B,
                                          int64_t stride_B_bytes, void* base_C,
                                          int64_t stride_C_bytes, int gqa_ratio,
                                          cudaStream_t stream) {
    const int block = 64;
    int grid = (n_heads + block - 1) / block;
    build_attn_ptr_arrays_kernel<<<grid, block, 0, stream>>>(
        const_cast<const void**>(d_ptrs), const_cast<const void**>(d_ptrs + n_heads),
        d_ptrs + 2 * n_heads, reinterpret_cast<const char*>(base_A), stride_A_bytes,
        reinterpret_cast<const char*>(base_B), stride_B_bytes, reinterpret_cast<char*>(base_C),
        stride_C_bytes, gqa_ratio, n_heads);
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
                              float softcap, int q_offset, cudaStream_t stream, int sliding_window,
                              const void* sinks) {
    const half* sinks_h = static_cast<const half*>(sinks);
    int q_len = static_cast<int>(Q.shape[0]);
    int kv_len = static_cast<int>(K.shape[0]);
    if (q_len == 0)
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
    int ld_s = kv_len;
    int ld_o = n_heads * head_dim;

    long long strideS = static_cast<long long>(q_len) * kv_len;

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
    int64_t s_fp32_elems = static_cast<int64_t>(n_heads) * q_len * kv_len;
    bool use_fp32_s = (s_fp32_elems * 2 <= s_buf_fp16_elems);
    float* S_f32 = use_fp32_s ? reinterpret_cast<float*>(S.data) : nullptr;

    if (gqa_ratio == 1) {
        // ---------------------------------------------------------------
        // MHA path: single strided batched call per direction
        // ---------------------------------------------------------------

        // S = scale * Q × K^T (FP32 output when use_fp32_s)
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, head_dim, &alpha_f,
                                   K_base, CUDA_R_16F, ld_k, static_cast<long long>(head_dim), Q_base,
                                   CUDA_R_16F, ld_q, static_cast<long long>(head_dim), &beta_f,
                                   use_fp32_s ? static_cast<void*>(S_f32) : static_cast<void*>(S_base),
                                   use_fp32_s ? CUDA_R_32F : CUDA_R_16F, ld_s, strideS, n_heads,
                                   CUBLAS_COMPUTE_32F, kGemmAlgo);

        // Softcap (if enabled): applied to whichever buffer the softmax reads —
        // S_f32 on the use_fp32_s path, S_base otherwise. Gemma-2 sets softcap=50;
        // skipping it on the FP32 path sharpened the softmax incorrectly.
        if (softcap > 0.0f) {
            int64_t total = static_cast<int64_t>(n_heads) * q_len * kv_len;
            int block = 256;
            int grid_sc = static_cast<int>((total + block - 1) / block);
            if (use_fp32_s)
                softcap_fp32_kernel<<<grid_sc, block, 0, stream>>>(S_f32, total, softcap);
            else
                softcap_fp16_kernel<<<grid_sc, block, 0, stream>>>(S_base, total, softcap);
        }

        // Softmax (FP32 or FP16)
        {
            int threads = (kv_len <= 128) ? 128 : ((kv_len <= 256) ? 256 : 512);
            if (threads > 1024)
                threads = 1024;
            dim3 grid(q_len, n_heads);
            if (use_fp32_s) {
                // Fused FP32 softmax + downcast: reads FP32 scores, writes FP16
                // probabilities to S_base directly. Replaces softmax_fp32_inplace
                // + fp32_to_fp16_kernel pair (saves one full pass over the tensor).
                causal_softmax_fp32_to_fp16_kernel<<<grid, threads, 0, stream>>>(
                    S_f32, S_base, q_len, kv_len, q_offset, causal, sliding_window, sinks_h);
            } else {
                causal_softmax_inplace_kernel<<<grid, threads, 0, stream>>>(
                    S_base, q_len, kv_len, q_offset, causal, sliding_window, sinks_h);
            }
        }

        // O = P × V (always FP16)
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, q_len, kv_len, &one_f,
                                   V_base, CUDA_R_16F, ld_k, static_cast<long long>(head_dim), S_base,
                                   CUDA_R_16F, ld_s, strideS, &zero_f, O_base, CUDA_R_16F, ld_o,
                                   static_cast<long long>(head_dim), n_heads, CUBLAS_COMPUTE_32F, kGemmAlgo);

    } else {
        // ---------------------------------------------------------------
        // GQA path: single batched call with explicit pointer arrays
        // Multiple Q heads share the same K/V head.
        // ---------------------------------------------------------------
        ensure_attn_ptr_arrays(n_heads);

        // Step 1: S = scale * Q × K^T — fill pointer arrays device-side.
        launch_build_attn_ptrs(s_attn_d_ptrs, n_heads, K_base,
                               static_cast<int64_t>(head_dim) * sizeof(half), Q_base,
                               static_cast<int64_t>(head_dim) * sizeof(half),
                               use_fp32_s ? static_cast<void*>(S_f32) : static_cast<void*>(S_base),
                               static_cast<int64_t>(strideS) *
                                   (use_fp32_s ? sizeof(float) : sizeof(half)),
                               gqa_ratio, stream);

        cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kv_len, q_len, head_dim, &alpha_f,
                            (const void**)s_attn_d_ptrs, CUDA_R_16F, ld_k,
                            (const void**)(s_attn_d_ptrs + n_heads), CUDA_R_16F, ld_q, &beta_f,
                            (void**)(s_attn_d_ptrs + 2 * n_heads), use_fp32_s ? CUDA_R_32F : CUDA_R_16F, ld_s,
                            n_heads, CUBLAS_COMPUTE_32F, kGemmAlgo);

        // Softcap (if enabled): applied to S_f32 on the use_fp32_s path, else S_base
        // (same fix as the MHA path above — was dropped on FP32-S for Gemma-2).
        if (softcap > 0.0f) {
            int64_t total = static_cast<int64_t>(n_heads) * q_len * kv_len;
            int block = 256;
            int grid_sc = static_cast<int>((total + block - 1) / block);
            if (use_fp32_s)
                softcap_fp32_kernel<<<grid_sc, block, 0, stream>>>(S_f32, total, softcap);
            else
                softcap_fp16_kernel<<<grid_sc, block, 0, stream>>>(S_base, total, softcap);
        }

        // Softmax
        {
            int threads = (kv_len <= 128) ? 128 : ((kv_len <= 256) ? 256 : 512);
            if (threads > 1024)
                threads = 1024;
            dim3 grid(q_len, n_heads);
            if (use_fp32_s) {
                // Fused FP32 softmax + downcast (see MHA path above).
                causal_softmax_fp32_to_fp16_kernel<<<grid, threads, 0, stream>>>(
                    S_f32, S_base, q_len, kv_len, q_offset, causal, sliding_window, sinks_h);
            } else {
                causal_softmax_inplace_kernel<<<grid, threads, 0, stream>>>(
                    S_base, q_len, kv_len, q_offset, causal, sliding_window, sinks_h);
            }
        }

        // Step 3: O = P × V — re-fill pointer arrays device-side for the
        // second cuBLAS call. cuBLAS: C = alpha * A * B with A=V (OP_N),
        // B=P (OP_N). A is GQA-shared, B and C are per-head.
        launch_build_attn_ptrs(s_attn_d_ptrs, n_heads, V_base,
                               static_cast<int64_t>(head_dim) * sizeof(half), S_base,
                               static_cast<int64_t>(strideS) * sizeof(half), O_base,
                               static_cast<int64_t>(head_dim) * sizeof(half), gqa_ratio, stream);

        cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, q_len, kv_len, &one_f,
                            (const void**)s_attn_d_ptrs, CUDA_R_16F, ld_k,
                            (const void**)(s_attn_d_ptrs + n_heads), CUDA_R_16F, ld_s, &zero_f,
                            (void**)(s_attn_d_ptrs + 2 * n_heads), CUDA_R_16F, ld_o, n_heads,
                            CUBLAS_COMPUTE_32F, kGemmAlgo);
    }
}

}  // namespace imp
