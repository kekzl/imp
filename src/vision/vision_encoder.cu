#include "vision/vision_encoder.h"
#include "core/logging.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cfloat>
#include <cmath>

namespace imp {

// ---- cuBLAS handle for vision encoder ----
static cublasHandle_t get_vision_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return handle;
}

// ---- Helper: cuBLAS GEMM  C = alpha * A @ B^T ----
// A: [M, K], B: [N, K], C: [M, N]  (row-major)
static void vision_gemm(const half* A, const half* B, half* C,
                         int M, int N, int K,
                         float alpha, float beta,
                         cudaStream_t stream) {
    auto handle = get_vision_cublas_handle();
    cublasSetStream(handle, stream);

    half h_alpha = __float2half(alpha);
    half h_beta  = __float2half(beta);

    // cuBLAS uses column-major, so we compute C^T = B @ A^T
    // C^T [N, M] = B [N, K] @ A^T [K, M]
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,    // A^T
        CUBLAS_OP_N,    // B
        N, M, K,        // m, n, k in col-major terms
        &h_alpha,
        B, CUDA_R_16F, K,   // B [N, K] col-major stride = K
        A, CUDA_R_16F, K,   // A [M, K] col-major stride = K
        &h_beta,
        C, CUDA_R_16F, N,   // C [M, N] col-major stride = N
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
}

// ---- Helper: Batched strided GEMM for attention ----
// scores = Q @ K^T: [B, M, K] @ [B, K, N]^T -> [B, M, N]
// where B = num_heads, M = N = num_patches, K = head_dim
static void vision_batched_gemm(const half* A, const half* B, half* C,
                                 int batch, int M, int N, int K,
                                 float alpha, float beta,
                                 long long strideA, long long strideB, long long strideC,
                                 cublasOperation_t transB,
                                 cudaStream_t stream) {
    auto handle = get_vision_cublas_handle();
    cublasSetStream(handle, stream);

    half h_alpha = __float2half(alpha);
    half h_beta  = __float2half(beta);

    // Row-major: C [M, N] = A [M, K] @ B^T [K, N] or similar
    // In col-major: C^T [N, M] = B_col @ A_col^T
    // For Q@K^T: A=[M,K], B=[N,K], want [M,N] = A @ B^T
    //   Col-major: [N,M] = B[N,K] @ A[M,K]^T  => op(B)=N, op(A)=T
    if (transB == CUBLAS_OP_T) {
        // C = A @ B^T:  A [M, K], B [N, K], C [M, N]
        // Col-major: C^T[N,M] = B[N,K] @ A^T[K,M]
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &h_alpha,
            B, CUDA_R_16F, K, strideB,
            A, CUDA_R_16F, K, strideA,
            &h_beta,
            C, CUDA_R_16F, N, strideC,
            batch,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT);
    } else {
        // C = A @ B:  A [M, K], B [K, N], C [M, N]
        // Col-major: C^T[N,M] = B^T[N,K] @ A^T[K,M]
        cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &h_alpha,
            B, CUDA_R_16F, N, strideB,
            A, CUDA_R_16F, K, strideA,
            &h_beta,
            C, CUDA_R_16F, N, strideC,
            batch,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT);
    }
}

// ======================================================================
//  CUDA Kernels
// ======================================================================

// Extract patches: [3, H, W] -> [num_patches, patch_size*patch_size*3]
__global__ void extract_patches_kernel(
    const half* __restrict__ pixels,   // [3, H, W]
    half* __restrict__ patches,         // [num_patches, patch_dim]
    int H, int W, int patch_size, int grid_h, int grid_w, int patch_dim
) {
    int patch_idx = blockIdx.x;
    int tid = threadIdx.x;

    int py = patch_idx / grid_w;
    int px = patch_idx % grid_w;
    int y0 = py * patch_size;
    int x0 = px * patch_size;

    // Each thread copies one element of the flattened patch
    for (int i = tid; i < patch_dim; i += blockDim.x) {
        int c = i / (patch_size * patch_size);
        int rem = i % (patch_size * patch_size);
        int dy = rem / patch_size;
        int dx = rem % patch_size;

        int y = y0 + dy;
        int x = x0 + dx;
        half val = pixels[c * H * W + y * W + x];
        patches[patch_idx * patch_dim + i] = val;
    }
}

// Standard LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
__global__ void vision_layernorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int D, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const half* x_row = x + row * D;
    half* o_row = out + row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x)
        sum += __half2float(x_row[i]);
    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    // Cross-warp reduction
    __shared__ float s_buf[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += s_buf[w];
        s_buf[0] = total;
    }
    __syncthreads();
    float mean = s_buf[0] / D;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]) - mean;
        var_sum += v * v;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        var_sum += __shfl_xor_sync(0xffffffff, var_sum, mask);
    if (lane == 0) s_buf[warp_id] = var_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += s_buf[w];
        s_buf[0] = total;
    }
    __syncthreads();
    float inv_std = rsqrtf(s_buf[0] / D + eps);

    // Normalize + scale + bias
    for (int i = tid; i < D; i += blockDim.x) {
        float v = (__half2float(x_row[i]) - mean) * inv_std;
        v = v * __half2float(weight[i]) + __half2float(bias[i]);
        o_row[i] = __float2half(v);
    }
}

// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
__global__ void vision_rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int D, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const half* x_row = x + row * D;
    half* o_row = out + row * D;

    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        ss += v * v;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        ss += __shfl_xor_sync(0xffffffff, ss, mask);

    __shared__ float s_buf[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;
    if (lane == 0) s_buf[warp_id] = ss;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += s_buf[w];
        s_buf[0] = total;
    }
    __syncthreads();
    float inv_rms = rsqrtf(s_buf[0] / D + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]) * inv_rms * __half2float(weight[i]);
        o_row[i] = __float2half(v);
    }
}

// Add bias: x[row, i] += bias[i]
__global__ void add_bias_kernel(half* __restrict__ x, const half* __restrict__ bias,
                                 int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    int col = idx % D;
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(bias[col]));
}

// Element-wise add: out = a + b
__global__ void add_tensors_kernel(const half* __restrict__ a,
                                    const half* __restrict__ b,
                                    half* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

// GELU tanh approximation: gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
__global__ void gelu_tanh_kernel(half* __restrict__ x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = __half2float(x[idx]);
    float c = 0.7978845608f;  // sqrt(2/pi)
    float inner = c * (v + 0.044715f * v * v * v);
    float gelu = 0.5f * v * (1.0f + tanhf(inner));
    x[idx] = __float2half(gelu);
}

// Non-causal row-wise softmax for attention scores
// scores: [num_heads, num_patches, num_patches]
// Each block handles one row.
__global__ void softmax_2d_kernel(half* __restrict__ scores, int cols) {
    int row = blockIdx.x;  // flattened: head * num_patches + patch
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int n_warps = (blockDim.x + 31) / 32;

    half* row_ptr = scores + static_cast<int64_t>(row) * cols;

    // Find max
    float max_val = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x)
        max_val = fmaxf(max_val, __half2float(row_ptr[j]));
    for (int mask = 16; mask > 0; mask >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));

    __shared__ float s_buf[32];
    if (lane == 0) s_buf[warp_id] = max_val;
    __syncthreads();
    if (tid == 0) {
        float m = -FLT_MAX;
        for (int w = 0; w < n_warps; w++) m = fmaxf(m, s_buf[w]);
        s_buf[0] = m;
    }
    __syncthreads();
    max_val = s_buf[0];

    // Exp and sum
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(__half2float(row_ptr[j]) - max_val);
        row_ptr[j] = __float2half(e);
        sum += e;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += s_buf[w];
        s_buf[0] = total;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_buf[0];

    // Normalize
    for (int j = tid; j < cols; j += blockDim.x)
        row_ptr[j] = __float2half(__half2float(row_ptr[j]) * inv_sum);
}

// Average pool: [grid_h, grid_w, D] -> [grid_h/pool, grid_w/pool, D]
// Input: [num_patches, D] interpreted as [grid_h, grid_w, D]
__global__ void avg_pool_spatial_kernel(
    const half* __restrict__ in,  // [grid_h, grid_w, D]
    half* __restrict__ out,        // [out_h, out_w, D]
    int grid_h, int grid_w, int D,
    int pool, int out_h, int out_w
) {
    int out_idx = blockIdx.x;  // output spatial index
    int tid = threadIdx.x;

    int oy = out_idx / out_w;
    int ox = out_idx % out_w;

    for (int d = tid; d < D; d += blockDim.x) {
        float sum = 0.0f;
        int count = 0;
        for (int py = 0; py < pool; py++) {
            for (int px = 0; px < pool; px++) {
                int iy = oy * pool + py;
                int ix = ox * pool + px;
                if (iy < grid_h && ix < grid_w) {
                    sum += __half2float(in[(iy * grid_w + ix) * D + d]);
                    count++;
                }
            }
        }
        out[out_idx * D + d] = __float2half(sum / count);
    }
}

// Replace vision token embeddings in the hidden state
__global__ void replace_vision_embeddings_kernel(
    half* __restrict__ hidden,           // [n_tokens, d_model]
    const int32_t* __restrict__ token_ids, // [n_tokens]
    const half* __restrict__ vision_emb,   // [n_vision_tokens, d_model]
    int vision_token_id,
    int n_tokens, int d_model, int n_vision_tokens
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tokens * d_model) return;

    int token_pos = tid / d_model;
    int dim = tid % d_model;

    if (token_ids[token_pos] != vision_token_id) return;

    // Count how many vision tokens precede this position
    int vision_idx = 0;
    for (int i = 0; i < token_pos; i++) {
        if (token_ids[i] == vision_token_id)
            vision_idx++;
    }

    if (vision_idx < n_vision_tokens) {
        hidden[token_pos * d_model + dim] = vision_emb[vision_idx * d_model + dim];
    }
}

// Optimized version: one block per vision token position
__global__ void replace_vision_embeddings_v2_kernel(
    half* __restrict__ hidden,
    const int32_t* __restrict__ token_ids,
    const half* __restrict__ vision_emb,
    int vision_token_id,
    int n_tokens, int d_model, int n_vision_tokens
) {
    // blockIdx.x = sequential vision token index
    int vision_idx = blockIdx.x;
    if (vision_idx >= n_vision_tokens) return;

    // Find the vision_idx-th occurrence of vision_token_id
    int count = 0;
    int token_pos = -1;
    for (int i = 0; i < n_tokens; i++) {
        if (token_ids[i] == vision_token_id) {
            if (count == vision_idx) {
                token_pos = i;
                break;
            }
            count++;
        }
    }
    if (token_pos < 0) return;

    // Copy vision embedding into hidden state
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        hidden[token_pos * d_model + d] = vision_emb[vision_idx * d_model + d];
    }
}

// ======================================================================
//  VisionEncoder implementation
// ======================================================================

VisionEncoder::~VisionEncoder() {
    free_buffers();
}

void VisionEncoder::free_buffers() {
    auto safe_free = [](half*& p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free(d_patches_);
    safe_free(d_hidden_);
    safe_free(d_residual_);
    safe_free(d_q_);
    safe_free(d_k_);
    safe_free(d_v_);
    safe_free(d_attn_out_);
    safe_free(d_attn_scores_);
    safe_free(d_ffn_);
    safe_free(d_pooled_);
}

bool VisionEncoder::init(const VisionModel& model, int lm_d_model, cudaStream_t stream) {
    model_ = &model;
    lm_d_model_ = lm_d_model;

    const auto& cfg = model.config;
    int np = cfg.num_patches;        // 4096
    int hd = cfg.hidden_size;        // 1152
    int ff = cfg.intermediate_size;  // 4304
    int nh = cfg.num_heads;          // 16
    int pd = cfg.patch_size * cfg.patch_size * 3;  // 588

    auto alloc = [](half*& ptr, size_t n) -> bool {
        return cudaMalloc(&ptr, n * sizeof(half)) == cudaSuccess;
    };

    if (!alloc(d_patches_, np * pd) ||
        !alloc(d_hidden_, np * hd) ||
        !alloc(d_residual_, np * hd) ||
        !alloc(d_q_, np * hd) ||
        !alloc(d_k_, np * hd) ||
        !alloc(d_v_, np * hd) ||
        !alloc(d_attn_out_, np * hd) ||
        !alloc(d_attn_scores_, static_cast<size_t>(nh) * np * np) ||
        !alloc(d_ffn_, np * ff) ||
        !alloc(d_pooled_, cfg.num_image_tokens * hd)) {
        IMP_LOG_ERROR("Vision encoder: workspace allocation failed");
        free_buffers();
        return false;
    }

    size_t total_mb = (
        np * pd + np * hd * 4 + np * hd +      // patches + hidden/residual/q/attn_out + k/v overlap
        static_cast<size_t>(nh) * np * np +      // attention scores
        np * ff +                                 // ffn
        cfg.num_image_tokens * hd                 // pooled
    ) * sizeof(half) / (1024 * 1024);

    IMP_LOG_INFO("Vision encoder: workspace %.0f MiB "
                 "(patches=%d, hidden=%d, ffn=%d, attn_scores=%dx%dx%d)",
                 static_cast<double>(total_mb),
                 np, hd, ff, nh, np, np);

    return true;
}

bool VisionEncoder::encode(const half* d_pixels, half* d_output, cudaStream_t stream) {
    const auto& cfg = model_->config;
    int np = cfg.num_patches;
    int hd = cfg.hidden_size;
    int ff = cfg.intermediate_size;
    int nh = cfg.num_heads;
    int head_dim = cfg.head_dim;
    int ps = cfg.patch_size;
    int img = cfg.image_size;
    int grid = img / ps;
    int patch_dim = ps * ps * 3;  // 588
    float eps = 1e-6f;

    // ---- Step 1: Extract patches ----
    extract_patches_kernel<<<np, 256, 0, stream>>>(
        d_pixels, d_patches_, img, img, ps, grid, grid, patch_dim);

    // ---- Step 2: Patch embedding: patches @ patch_embd_w^T + bias -> hidden ----
    // patch_embd_w: [hidden_size, patch_dim]
    // patches: [num_patches, patch_dim]
    // hidden: [num_patches, hidden_size]
    vision_gemm(d_patches_,
                static_cast<const half*>(model_->patch_embd_w.data),
                d_hidden_,
                np, hd, patch_dim,
                1.0f, 0.0f, stream);

    if (model_->patch_embd_b.data) {
        int total = np * hd;
        add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(model_->patch_embd_b.data), np, hd);
    }

    // ---- Step 3: Add position embeddings ----
    if (model_->position_embd.data) {
        int total = np * hd;
        add_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(model_->position_embd.data),
            d_hidden_, total);
    }

    // ---- Step 4: Transformer layers ----
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        const auto& lw = model_->layers[layer];

        // Pre-attention LayerNorm
        vision_layernorm_kernel<<<np, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(lw.ln1_w.data),
            static_cast<const half*>(lw.ln1_b.data),
            d_residual_, hd, eps);

        // Q, K, V projections
        vision_gemm(d_residual_, static_cast<const half*>(lw.wq.data), d_q_,
                     np, hd, hd, 1.0f, 0.0f, stream);
        vision_gemm(d_residual_, static_cast<const half*>(lw.wk.data), d_k_,
                     np, hd, hd, 1.0f, 0.0f, stream);
        vision_gemm(d_residual_, static_cast<const half*>(lw.wv.data), d_v_,
                     np, hd, hd, 1.0f, 0.0f, stream);

        // Add biases
        if (lw.bq.data) {
            int total = np * hd;
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_q_, static_cast<const half*>(lw.bq.data), np, hd);
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_k_, static_cast<const half*>(lw.bk.data), np, hd);
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_v_, static_cast<const half*>(lw.bv.data), np, hd);
        }

        // Multi-head attention via batched GEMM
        // Reshape: [np, nh, head_dim] -> batched [nh, np, head_dim]
        // Q, K, V are stored as [np, hd] = [np, nh * head_dim]
        // For strided batched GEMM: treat as [nh, np, head_dim] with stride np*head_dim between heads
        // But data is actually [np, nh, head_dim], so stride between heads = head_dim,
        // and stride between rows = nh * head_dim.
        // We need to transpose to [nh, np, head_dim] for standard batched GEMM.
        // Instead, use the fact that cuBLAS supports arbitrary strides:
        //   Q[h, i, :] = Q_flat[i * nh * head_dim + h * head_dim ... + head_dim-1]
        //   stride_batch = head_dim (between heads within same row)
        //   stride_row = nh * head_dim (between rows for same head)

        // scores = Q @ K^T: for each head h, scores[h] = Q_h @ K_h^T
        // Q_h: [np, head_dim] with stride nh*head_dim, batch stride head_dim
        // K_h: [np, head_dim] with stride nh*head_dim, batch stride head_dim
        // scores: [nh, np, np]

        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        {
            auto handle = get_vision_cublas_handle();
            cublasSetStream(handle, stream);

            half h_alpha = __float2half(scale);
            half h_beta  = __float2half(0.0f);

            // Q_h[i, d] = d_q_[i * nh*hd + h*hd + d],  lda = nh*hd, batch_stride = hd
            // K_h[j, d] = d_k_[j * nh*hd + h*hd + d],  same layout
            // S_h[i, j] = scores[h * np*np + i*np + j]

            // In col-major for cuBLAS: computing S^T[np, np] = K @ Q^T
            // where Q, K have leading dim nh*hd and batch stride hd
            cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_T,  // K^T
                CUBLAS_OP_N,  // Q
                np, np, head_dim,  // m, n, k
                &h_alpha,
                d_k_, CUDA_R_16F, nh * head_dim,  // lda for K
                head_dim,                           // batch stride for K (next head)
                d_q_, CUDA_R_16F, nh * head_dim,   // ldb for Q
                head_dim,                           // batch stride for Q
                &h_beta,
                d_attn_scores_, CUDA_R_16F, np,    // ldc for scores
                static_cast<long long>(np) * np,   // batch stride for scores
                nh,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT);
        }

        // Non-causal softmax
        int total_rows = nh * np;
        softmax_2d_kernel<<<total_rows, 256, 0, stream>>>(d_attn_scores_, np);

        // attn_out = scores @ V
        {
            auto handle = get_vision_cublas_handle();
            cublasSetStream(handle, stream);

            half h_one  = __float2half(1.0f);
            half h_zero = __float2half(0.0f);

            // scores_h: [np, np] at scores + h*np*np
            // V_h[j, d] = d_v_[j * nh*hd + h*hd + d]
            // out_h[i, d] = d_attn_out_[i * nh*hd + h*hd + d]

            // Col-major: out^T[hd, np] = V^T[hd, np] @ scores^T[np, np]
            cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N,  // V^T is what we want, but V stored row-major...
                CUBLAS_OP_N,  // scores
                head_dim, np, np,  // m, n, k
                &h_one,
                d_v_, CUDA_R_16F, nh * head_dim,   // V: lda = nh*hd
                head_dim,                            // batch stride
                d_attn_scores_, CUDA_R_16F, np,     // scores: lda = np
                static_cast<long long>(np) * np,    // batch stride
                &h_zero,
                d_attn_out_, CUDA_R_16F, nh * head_dim,  // out: lda = nh*hd
                head_dim,                                   // batch stride
                nh,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT);
        }

        // Output projection: attn_out @ wo^T + bo
        vision_gemm(d_attn_out_, static_cast<const half*>(lw.wo.data), d_residual_,
                     np, hd, hd, 1.0f, 0.0f, stream);

        if (lw.bo.data) {
            int total = np * hd;
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_residual_, static_cast<const half*>(lw.bo.data), np, hd);
        }

        // Residual add: hidden += attn_output
        {
            int total = np * hd;
            add_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_hidden_, d_residual_, d_hidden_, total);
        }

        // Pre-FFN LayerNorm
        vision_layernorm_kernel<<<np, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(lw.ln2_w.data),
            static_cast<const half*>(lw.ln2_b.data),
            d_residual_, hd, eps);

        // FFN up: residual @ ffn_up_w^T + bias -> ffn
        vision_gemm(d_residual_, static_cast<const half*>(lw.ffn_up_w.data), d_ffn_,
                     np, ff, hd, 1.0f, 0.0f, stream);
        if (lw.ffn_up_b.data) {
            int total = np * ff;
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_ffn_, static_cast<const half*>(lw.ffn_up_b.data), np, ff);
        }

        // GELU activation
        {
            int total = np * ff;
            gelu_tanh_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_ffn_, total);
        }

        // FFN down: ffn @ ffn_down_w^T + bias -> residual
        vision_gemm(d_ffn_, static_cast<const half*>(lw.ffn_down_w.data), d_residual_,
                     np, hd, ff, 1.0f, 0.0f, stream);
        if (lw.ffn_down_b.data) {
            int total = np * hd;
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_residual_, static_cast<const half*>(lw.ffn_down_b.data), np, hd);
        }

        // Residual add: hidden += ffn_output
        {
            int total = np * hd;
            add_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_hidden_, d_residual_, d_hidden_, total);
        }
    }

    // ---- Step 5: Post-encoder LayerNorm ----
    if (model_->post_norm_w.data) {
        vision_layernorm_kernel<<<np, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(model_->post_norm_w.data),
            static_cast<const half*>(model_->post_norm_b.data),
            d_hidden_, hd, eps);
    }

    // ---- Step 6: Average pool 4x4 spatial ----
    // hidden: [grid_h, grid_w, hidden_size] = [64, 64, 1152]
    // pooled: [16, 16, 1152] = [256, 1152]
    int pool_factor = 4;
    int out_h = grid / pool_factor;
    int out_w = grid / pool_factor;
    int n_pooled = out_h * out_w;

    avg_pool_spatial_kernel<<<n_pooled, 256, 0, stream>>>(
        d_hidden_, d_pooled_, grid, grid, hd, pool_factor, out_h, out_w);

    // ---- Step 7: Multimodal projector ----
    // RMSNorm -> Linear -> RMSNorm

    // Pre-projection RMSNorm
    if (model_->mm_pre_norm_w.data) {
        vision_rmsnorm_kernel<<<n_pooled, 256, 0, stream>>>(
            d_pooled_, static_cast<const half*>(model_->mm_pre_norm_w.data),
            d_pooled_, hd, eps);
    }

    // Linear projection: [256, 1152] @ mm_proj_w^T + bias -> [256, d_model]
    vision_gemm(d_pooled_,
                static_cast<const half*>(model_->mm_proj_w.data),
                d_output,
                n_pooled, lm_d_model_, hd,
                1.0f, 0.0f, stream);

    if (model_->mm_proj_b.data) {
        int total = n_pooled * lm_d_model_;
        add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_output, static_cast<const half*>(model_->mm_proj_b.data),
            n_pooled, lm_d_model_);
    }

    // Post-projection RMSNorm
    if (model_->mm_post_norm_w.data) {
        vision_rmsnorm_kernel<<<n_pooled, 256, 0, stream>>>(
            d_output, static_cast<const half*>(model_->mm_post_norm_w.data),
            d_output, lm_d_model_, eps);
    }

    return true;
}

// ---- Public kernel launcher for embedding replacement ----
void launch_replace_vision_embeddings(
    half* hidden, const int32_t* token_ids, const half* vision_emb,
    int vision_token_id, int n_tokens, int d_model, int n_vision_tokens,
    cudaStream_t stream)
{
    if (n_vision_tokens <= 0) return;
    replace_vision_embeddings_v2_kernel<<<n_vision_tokens, 256, 0, stream>>>(
        hidden, token_ids, vision_emb,
        vision_token_id, n_tokens, d_model, n_vision_tokens);
}

} // namespace imp
