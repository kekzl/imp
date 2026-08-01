#include "vision/vision_encoder.h"
#include "compute/warp_reduce.cuh"
#include "core/cuda_static_reset.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"
#include "runtime/process_diag.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <vector>

namespace imp {

// ---- cuBLAS handle for vision encoder ----
static cublasHandle_t s_vision_cublas_handle = nullptr;  // file-scope so the reset hook can reach it

static cublasHandle_t get_vision_cublas_handle() {
    if (!s_vision_cublas_handle) {
        cublasCreate(&s_vision_cublas_handle);
        cublasSetMathMode(s_vision_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return s_vision_cublas_handle;
}

// Pre-cudaDeviceReset hook (see core/cuda_static_reset.h).
void vision_encoder_reset_static_cuda_state() {
    if (s_vision_cublas_handle) {
        (void)cublasDestroy(s_vision_cublas_handle);
        s_vision_cublas_handle = nullptr;
    }
}

// ---- Helper: cuBLAS GEMM  C = alpha * A @ B^T ----
// A: [M, K], B: [N, K], C: [M, N]  (row-major)
static void vision_gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta,
                        cudaStream_t stream) {
    auto handle = get_vision_cublas_handle();
    cublasSetStream(handle, stream);

    // FP32 accumulation (FP16 in/out): the FFN down-projection sums thousands of
    // terms; FP16 accumulation overflows for high-magnitude tokens (→ inf, then
    // RMSNorm turns inf×0 into NaN). FP32 compute keeps the encoder numerically
    // robust. alpha/beta must be float for COMPUTE_32F.
    float f_alpha = alpha;
    float f_beta = beta;

    // cuBLAS uses column-major, so we compute C^T = B @ A^T
    // C^T [N, M] = B [N, K] @ A^T [K, M]
    cublasGemmEx(handle,
                 CUBLAS_OP_T,                 // A^T
                 CUBLAS_OP_N,                 // B
                 N, M, K,                     // m, n, k in col-major terms
                 &f_alpha, B, CUDA_R_16F, K,  // B [N, K] col-major stride = K
                 A, CUDA_R_16F, K,            // A [M, K] col-major stride = K
                 &f_beta, C, CUDA_R_16F, N,   // C [M, N] col-major stride = N
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ======================================================================
//  CUDA Kernels
// ======================================================================

// Extract patches: [3, H, W] -> [num_patches, patch_size*patch_size*3]
__global__ void extract_patches_kernel(const half* __restrict__ pixels,  // [3, H, W]
                                       half* __restrict__ patches,       // [num_patches, patch_dim]
                                       int H, int W, int patch_size, int grid_h, int grid_w, int patch_dim) {
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
__global__ void vision_layernorm_kernel(const half* __restrict__ x, const half* __restrict__ weight,
                                        const half* __restrict__ bias, half* __restrict__ out, int D,
                                        float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const half* x_row = x + row * D;
    half* o_row = out + row * D;

    // Compute mean
    __shared__ float s_buf[32];
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x)
        sum += __half2float(x_row[i]);
    float mean = block_reduce_sum(sum, s_buf) / D;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]) - mean;
        var_sum += v * v;
    }
    float inv_std = rsqrtf(block_reduce_sum(var_sum, s_buf) / D + eps);

    // Normalize + scale + bias
    for (int i = tid; i < D; i += blockDim.x) {
        float v = (__half2float(x_row[i]) - mean) * inv_std;
        v = v * __half2float(weight[i]) + __half2float(bias[i]);
        o_row[i] = __float2half(v);
    }
}

// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
__global__ void vision_rmsnorm_kernel(const half* __restrict__ x, const half* __restrict__ weight,
                                      half* __restrict__ out, int D, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const half* x_row = x + row * D;
    half* o_row = out + row * D;

    __shared__ float s_buf[32];
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        ss += v * v;
    }
    float inv_rms = rsqrtf(block_reduce_sum(ss, s_buf) / D + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]) * inv_rms * __half2float(weight[i]);
        o_row[i] = __float2half(v);
    }
}

// Add bias: x[row, i] += bias[i]
__global__ void add_bias_kernel(half* __restrict__ x, const half* __restrict__ bias, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D)
        return;
    int col = idx % D;
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(bias[col]));
}

// Element-wise add: out = a + b
__global__ void add_tensors_kernel(const half* __restrict__ a, const half* __restrict__ b,
                                   half* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

// GELU tanh approximation: gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
__global__ void gelu_tanh_kernel(half* __restrict__ x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
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

    half* row_ptr = scores + static_cast<int64_t>(row) * cols;

    __shared__ float s_buf[32];

    // Find max
    float max_val = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x)
        max_val = fmaxf(max_val, __half2float(row_ptr[j]));
    max_val = block_reduce_max(max_val, s_buf);

    // Exp and sum
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(__half2float(row_ptr[j]) - max_val);
        row_ptr[j] = __float2half(e);
        sum += e;
    }
    float inv_sum = 1.0f / block_reduce_sum(sum, s_buf);

    // Normalize
    for (int j = tid; j < cols; j += blockDim.x)
        row_ptr[j] = __float2half(__half2float(row_ptr[j]) * inv_sum);
}

// Average pool: [grid_h, grid_w, D] -> [grid_h/pool, grid_w/pool, D]
// Input: [num_patches, D] interpreted as [grid_h, grid_w, D]
__global__ void avg_pool_spatial_kernel(const half* __restrict__ in,  // [grid_h, grid_w, D]
                                        half* __restrict__ out,       // [out_h, out_w, D]
                                        int grid_h, int grid_w, int D, int pool, int out_h, int out_w) {
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

// RMSNorm over contiguous rows of width D, optional weight (null = no weight).
// Used by gemma4v for per-head q/k/v norm (D=head_dim, n_rows=np*nh) and the
// weightless pre-projection norm (D=hidden, n_rows=n_tokens).
__global__ void vision_rmsnorm_opt_kernel(const half* __restrict__ x, const half* __restrict__ weight,
                                          half* __restrict__ out, int D, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const half* x_row = x + static_cast<int64_t>(row) * D;
    half* o_row = out + static_cast<int64_t>(row) * D;

    __shared__ float s_buf[32];
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        ss += v * v;
    }
    float inv_rms = rsqrtf(block_reduce_sum(ss, s_buf) / D + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]) * inv_rms;
        if (weight)
            v *= __half2float(weight[i]);
        o_row[i] = __float2half(v);
    }
}

// 2D axial NEOX RoPE for gemma4v vision attention. qk: [np, nh, head_dim].
// The head_dim is split in half: first half rotated by the patch column (pos_x),
// second half by the patch row (pos_y); each half is NEOX (pairs i, i+half/2).
__global__ void vision_rope2d_neox_kernel(half* __restrict__ qk, const int* __restrict__ pos_x,
                                          const int* __restrict__ pos_y, int np, int nh, int head_dim,
                                          float base) {
    int half_dim = head_dim / 2;       // dims rotated per axis
    int pairs = half_dim / 2;          // rotation pairs per axis
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = np * nh * pairs;
    if (idx >= total)
        return;
    int j = idx % pairs;          // 0..pairs-1
    int hp = idx / pairs;         // patch*nh + head
    int p = hp / nh;
    half* vec = qk + static_cast<int64_t>(hp) * head_dim;

    float inv = powf(base, -2.0f * j / half_dim);
    // first half: column position
    {
        float ang = pos_x[p] * inv;
        float c = cosf(ang), s = sinf(ang);
        float a = __half2float(vec[j]);
        float b = __half2float(vec[j + pairs]);
        vec[j] = __float2half(a * c - b * s);
        vec[j + pairs] = __float2half(a * s + b * c);
    }
    // second half: row position
    {
        int o = half_dim;
        float ang = pos_y[p] * inv;
        float c = cosf(ang), s = sinf(ang);
        float a = __half2float(vec[o + j]);
        float b = __half2float(vec[o + j + pairs]);
        vec[o + j] = __float2half(a * c - b * s);
        vec[o + j + pairs] = __float2half(a * s + b * c);
    }
}

// Add the two axial learned position tables (x=col, y=row) to the patch grid.
// tbl is v.position_embd laid out [2, pos_size, D]: x-table at offset 0, y-table
// at offset pos_size*D.
__global__ void axial_pos_add_kernel(half* __restrict__ x, const half* __restrict__ tbl,
                                     const int* __restrict__ pos_x, const int* __restrict__ pos_y, int np,
                                     int D, int pos_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np * D)
        return;
    int p = idx / D;
    int d = idx % D;
    float ex = __half2float(tbl[static_cast<int64_t>(pos_x[p]) * D + d]);
    float ey = __half2float(tbl[(static_cast<int64_t>(pos_size) + pos_y[p]) * D + d]);
    x[idx] = __float2half(__half2float(x[idx]) + ex + ey);
}

// gemma4v projector tail fused in FP32: out = rmsnorm((x*scale_factor - std_bias) * std_scale).
// Gemma vision activations are large (absmax ~3000); the ×√D scale would overflow
// FP16 (→ inf → NaN) if materialized, so the whole tail runs in FP32 registers and
// only the RMS-normalized (small) result is written back as FP16.
__global__ void gemma4v_tail_norm_kernel(const half* __restrict__ x, const half* __restrict__ std_bias,
                                         const half* __restrict__ std_scale, half* __restrict__ out, int D,
                                         float scale_factor, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const half* x_row = x + static_cast<int64_t>(row) * D;
    half* o_row = out + static_cast<int64_t>(row) * D;
    extern __shared__ float sv[];  // [D] FP32 scratch
    __shared__ float s_buf[32];
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]) * scale_factor;
        if (std_bias && std_scale)
            v = (v - __half2float(std_bias[i])) * __half2float(std_scale[i]);
        sv[i] = v;
        ss += v * v;
    }
    float inv = rsqrtf(block_reduce_sum(ss, s_buf) / D + eps);
    for (int i = tid; i < D; i += blockDim.x)
        o_row[i] = __float2half(sv[i] * inv);
}

// Element-wise multiply: out = a * b (GeGLU gate combine).
__global__ void mul_tensors_kernel(const half* __restrict__ a, const half* __restrict__ b,
                                   half* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    out[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx]));
}

// Optimized version: one block per vision token position
__global__ void replace_vision_embeddings_v2_kernel(half* __restrict__ hidden,
                                                    const int32_t* __restrict__ token_ids,
                                                    const half* __restrict__ vision_emb, int vision_token_id,
                                                    int n_tokens, int d_model, int n_vision_tokens,
                                                    int emb_offset) {
    // blockIdx.x = vision token index within THIS call's token span. Under
    // chunked prefill that span is one chunk, so the embedding this position
    // wants sits `emb_offset` further into the buffer.
    int vision_idx = blockIdx.x;
    int emb_idx = vision_idx + emb_offset;
    if (emb_idx >= n_vision_tokens)
        return;

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
    if (token_pos < 0)
        return;

    // Copy vision embedding into hidden state
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        hidden[token_pos * d_model + d] = vision_emb[emb_idx * d_model + d];
    }
}

// ======================================================================
//  VisionEncoder implementation
// ======================================================================

VisionEncoder::~VisionEncoder() { free_buffers(); }

void VisionEncoder::free_buffers() {
    auto safe_free = [this](half*& p) {
        if (p) {
            if (alloc_)
                alloc_->free(p);
            else
                cudaFree(p);
            p = nullptr;
        }
    };
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
    safe_free(d_gate_);
    if (d_pos_x_) {
        cudaFree(d_pos_x_);
        d_pos_x_ = nullptr;
    }
    if (d_pos_y_) {
        cudaFree(d_pos_y_);
        d_pos_y_ = nullptr;
    }
}

bool VisionEncoder::init(const VisionModel& model, int lm_d_model, cudaStream_t stream,
                         VRAMAllocator* alloc_in) {
    model_ = &model;
    lm_d_model_ = lm_d_model;
    alloc_ = alloc_in;

    const auto& cfg = model.config;
    int np = cfg.num_patches;                      // 4096
    int hd = cfg.hidden_size;                      // 1152
    int ff = cfg.intermediate_size;                // 4304
    int nh = cfg.num_heads;                        // 16
    int pd = cfg.patch_size * cfg.patch_size * 3;  // 588

    auto alloc = [this](half*& ptr, size_t n) -> bool {
        size_t bytes = n * sizeof(half);
        if (alloc_) {
            ptr = static_cast<half*>(alloc_->allocate(bytes, "vision_encoder"));
            return ptr != nullptr;
        }
        return cudaMalloc(&ptr, bytes) == cudaSuccess;
    };

    if (!alloc(d_patches_, np * pd) || !alloc(d_hidden_, np * hd) || !alloc(d_residual_, np * hd) ||
        !alloc(d_q_, np * hd) || !alloc(d_k_, np * hd) || !alloc(d_v_, np * hd) ||
        !alloc(d_attn_out_, np * hd) || !alloc(d_attn_scores_, static_cast<size_t>(nh) * np * np) ||
        !alloc(d_ffn_, np * ff) || !alloc(d_pooled_, cfg.num_image_tokens * hd)) {
        IMP_LOG_ERROR("Vision encoder: workspace allocation failed");
        free_buffers();
        return false;
    }

    // gemma4v needs a separate GeGLU gate buffer and precomputed axial position
    // indices (column = patch % grid, row = patch / grid).
    if (cfg.is_gemma4v) {
        int grid = cfg.image_size / cfg.patch_size;
        if (!alloc(d_gate_, np * ff) ||
            cudaMalloc(&d_pos_x_, np * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_pos_y_, np * sizeof(int)) != cudaSuccess) {
            IMP_LOG_ERROR("Vision encoder: gemma4v workspace allocation failed");
            free_buffers();
            return false;
        }
        std::vector<int> hx(np), hy(np);
        for (int i = 0; i < np; i++) {
            hx[i] = i % grid;
            hy[i] = i / grid;
        }
        cudaMemcpy(d_pos_x_, hx.data(), np * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y_, hy.data(), np * sizeof(int), cudaMemcpyHostToDevice);
    }

    size_t total_mb = (np * pd + np * hd * 4 +
                       np * hd +  // patches + hidden/residual/q/attn_out + k/v overlap
                       static_cast<size_t>(nh) * np * np +  // attention scores
                       np * ff +                            // ffn
                       cfg.num_image_tokens * hd            // pooled
                       ) *
                      sizeof(half) / (1024 * 1024);

    IMP_LOG_INFO(
        "Vision encoder: workspace %.0f MiB "
        "(patches=%d, hidden=%d, ffn=%d, attn_scores=%dx%dx%d)",
        static_cast<double>(total_mb), np, hd, ff, nh, np, np);

    return true;
}

bool VisionEncoder::encode(const half* d_pixels, half* d_output, cudaStream_t stream) {
    // Capture/replay the full encoder forward as a CUDA graph. The 27-layer
    // SigLIP ViT launches ~200+ kernels per image — graph replay eliminates
    // the per-kernel launch overhead. The graph is keyed on (d_pixels,
    // d_output); any pointer change invalidates the captured slot.
    //
    // [runtime] no_vision_graph = true forces the eager path (debugging).
    const bool disable_graph = process_diag_no_vision_graph();
    if (disable_graph) {
        return encode_impl(d_pixels, d_output, stream);
    }

    if (d_pixels != graph_d_pixels_ || d_output != graph_d_output_) {
        encode_graph_.invalidate();
        graph_d_pixels_ = d_pixels;
        graph_d_output_ = d_output;
    }

    bool ok = true;
    encode_graph_.set_decode_fn(
        [this, d_pixels, d_output, &ok](cudaStream_t s) { ok = encode_impl(d_pixels, d_output, s); });
    encode_graph_.execute(stream);
    return ok;
}

bool VisionEncoder::encode_impl(const half* d_pixels, half* d_output, cudaStream_t stream) {
    if (model_->config.is_gemma4v)
        return encode_impl_gemma4v(d_pixels, d_output, stream);

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
    extract_patches_kernel<<<np, 256, 0, stream>>>(d_pixels, d_patches_, img, img, ps, grid, grid, patch_dim);
    IMP_CUDA_CHECK_LAUNCH();

    // ---- Step 2: Patch embedding: patches @ patch_embd_w^T + bias -> hidden ----
    // patch_embd_w: [hidden_size, patch_dim]
    // patches: [num_patches, patch_dim]
    // hidden: [num_patches, hidden_size]
    vision_gemm(d_patches_, static_cast<const half*>(model_->patch_embd_w.data), d_hidden_, np, hd, patch_dim,
                1.0f, 0.0f, stream);

    if (model_->patch_embd_b.data) {
        int total = np * hd;
        add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(model_->patch_embd_b.data), np, hd);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // ---- Step 3: Add position embeddings ----
    if (model_->position_embd.data) {
        int total = np * hd;
        add_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(model_->position_embd.data), d_hidden_, total);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // ---- Step 4: Transformer layers ----
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        const auto& lw = model_->layers[layer];

        // Pre-attention LayerNorm
        vision_layernorm_kernel<<<np, 256, 0, stream>>>(d_hidden_, static_cast<const half*>(lw.ln1_w.data),
                                                        static_cast<const half*>(lw.ln1_b.data), d_residual_,
                                                        hd, eps);
        IMP_CUDA_CHECK_LAUNCH();

        // Q, K, V projections
        vision_gemm(d_residual_, static_cast<const half*>(lw.wq.data), d_q_, np, hd, hd, 1.0f, 0.0f, stream);
        vision_gemm(d_residual_, static_cast<const half*>(lw.wk.data), d_k_, np, hd, hd, 1.0f, 0.0f, stream);
        vision_gemm(d_residual_, static_cast<const half*>(lw.wv.data), d_v_, np, hd, hd, 1.0f, 0.0f, stream);

        // Add biases
        if (lw.bq.data) {
            int total = np * hd;
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_q_,
                                                                     static_cast<const half*>(lw.bq.data), np,
                                                                     hd);
            IMP_CUDA_CHECK_LAUNCH();
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_k_,
                                                                     static_cast<const half*>(lw.bk.data), np,
                                                                     hd);
            IMP_CUDA_CHECK_LAUNCH();
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_v_,
                                                                     static_cast<const half*>(lw.bv.data), np,
                                                                     hd);
            IMP_CUDA_CHECK_LAUNCH();
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
            half h_beta = __float2half(0.0f);

            // Q_h[i, d] = d_q_[i * nh*hd + h*hd + d],  lda = nh*hd, batch_stride = hd
            // K_h[j, d] = d_k_[j * nh*hd + h*hd + d],  same layout
            // S_h[i, j] = scores[h * np*np + i*np + j]

            // In col-major for cuBLAS: computing S^T[np, np] = K @ Q^T
            // where Q, K have leading dim nh*hd and batch stride hd
            cublasGemmStridedBatchedEx(handle,
                                       CUBLAS_OP_T,                                // K^T
                                       CUBLAS_OP_N,                                // Q
                                       np, np, head_dim,                           // m, n, k
                                       &h_alpha, d_k_, CUDA_R_16F, nh * head_dim,  // lda for K
                                       head_dim,                         // batch stride for K (next head)
                                       d_q_, CUDA_R_16F, nh * head_dim,  // ldb for Q
                                       head_dim,                         // batch stride for Q
                                       &h_beta, d_attn_scores_, CUDA_R_16F, np,  // ldc for scores
                                       static_cast<long long>(np) * np,          // batch stride for scores
                                       nh, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        }

        // Non-causal softmax
        int total_rows = nh * np;
        softmax_2d_kernel<<<total_rows, 256, 0, stream>>>(d_attn_scores_, np);
        IMP_CUDA_CHECK_LAUNCH();

        // attn_out = scores @ V
        {
            auto handle = get_vision_cublas_handle();
            cublasSetStream(handle, stream);

            half h_one = __float2half(1.0f);
            half h_zero = __float2half(0.0f);

            // scores_h: [np, np] at scores + h*np*np
            // V_h[j, d] = d_v_[j * nh*hd + h*hd + d]
            // out_h[i, d] = d_attn_out_[i * nh*hd + h*hd + d]

            // Col-major: out^T[hd, np] = V^T[hd, np] @ scores^T[np, np]
            cublasGemmStridedBatchedEx(handle,
                                       CUBLAS_OP_N,       // V^T is what we want, but V stored row-major...
                                       CUBLAS_OP_N,       // scores
                                       head_dim, np, np,  // m, n, k
                                       &h_one, d_v_, CUDA_R_16F, nh * head_dim,          // V: lda = nh*hd
                                       head_dim,                                         // batch stride
                                       d_attn_scores_, CUDA_R_16F, np,                   // scores: lda = np
                                       static_cast<long long>(np) * np,                  // batch stride
                                       &h_zero, d_attn_out_, CUDA_R_16F, nh * head_dim,  // out: lda = nh*hd
                                       head_dim,                                         // batch stride
                                       nh, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        }

        // Output projection: attn_out @ wo^T + bo
        vision_gemm(d_attn_out_, static_cast<const half*>(lw.wo.data), d_residual_, np, hd, hd, 1.0f, 0.0f,
                    stream);

        if (lw.bo.data) {
            int total = np * hd;
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_residual_,
                                                                     static_cast<const half*>(lw.bo.data), np,
                                                                     hd);
            IMP_CUDA_CHECK_LAUNCH();
        }

        // Residual add: hidden += attn_output
        {
            int total = np * hd;
            add_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_hidden_, d_residual_, d_hidden_,
                                                                        total);
            IMP_CUDA_CHECK_LAUNCH();
        }

        // Pre-FFN LayerNorm
        vision_layernorm_kernel<<<np, 256, 0, stream>>>(d_hidden_, static_cast<const half*>(lw.ln2_w.data),
                                                        static_cast<const half*>(lw.ln2_b.data), d_residual_,
                                                        hd, eps);
        IMP_CUDA_CHECK_LAUNCH();

        // FFN up: residual @ ffn_up_w^T + bias -> ffn
        vision_gemm(d_residual_, static_cast<const half*>(lw.ffn_up_w.data), d_ffn_, np, ff, hd, 1.0f, 0.0f,
                    stream);
        if (lw.ffn_up_b.data) {
            int total = np * ff;
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_ffn_, static_cast<const half*>(lw.ffn_up_b.data), np, ff);
            IMP_CUDA_CHECK_LAUNCH();
        }

        // GELU activation
        {
            int total = np * ff;
            gelu_tanh_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_ffn_, total);
            IMP_CUDA_CHECK_LAUNCH();
        }

        // FFN down: ffn @ ffn_down_w^T + bias -> residual
        vision_gemm(d_ffn_, static_cast<const half*>(lw.ffn_down_w.data), d_residual_, np, hd, ff, 1.0f, 0.0f,
                    stream);
        if (lw.ffn_down_b.data) {
            int total = np * hd;
            add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                d_residual_, static_cast<const half*>(lw.ffn_down_b.data), np, hd);
            IMP_CUDA_CHECK_LAUNCH();
        }

        // Residual add: hidden += ffn_output
        {
            int total = np * hd;
            add_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_hidden_, d_residual_, d_hidden_,
                                                                        total);
            IMP_CUDA_CHECK_LAUNCH();
        }
    }

    // ---- Step 5: Post-encoder LayerNorm ----
    if (model_->post_norm_w.data) {
        vision_layernorm_kernel<<<np, 256, 0, stream>>>(d_hidden_,
                                                        static_cast<const half*>(model_->post_norm_w.data),
                                                        static_cast<const half*>(model_->post_norm_b.data),
                                                        d_hidden_, hd, eps);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // ---- Step 6: Average pool 4x4 spatial ----
    // hidden: [grid_h, grid_w, hidden_size] = [64, 64, 1152]
    // pooled: [16, 16, 1152] = [256, 1152]
    int pool_factor = 4;
    int out_h = grid / pool_factor;
    int out_w = grid / pool_factor;
    int n_pooled = out_h * out_w;

    avg_pool_spatial_kernel<<<n_pooled, 256, 0, stream>>>(d_hidden_, d_pooled_, grid, grid, hd, pool_factor,
                                                          out_h, out_w);
    IMP_CUDA_CHECK_LAUNCH();

    // ---- Step 7: Multimodal projector ----
    // RMSNorm -> Linear -> RMSNorm

    // Pre-projection RMSNorm
    if (model_->mm_pre_norm_w.data) {
        vision_rmsnorm_kernel<<<n_pooled, 256, 0, stream>>>(
            d_pooled_, static_cast<const half*>(model_->mm_pre_norm_w.data), d_pooled_, hd, eps);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Linear projection: [256, 1152] @ mm_proj_w^T + bias -> [256, d_model]
    vision_gemm(d_pooled_, static_cast<const half*>(model_->mm_proj_w.data), d_output, n_pooled, lm_d_model_,
                hd, 1.0f, 0.0f, stream);

    if (model_->mm_proj_b.data) {
        int total = n_pooled * lm_d_model_;
        add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_output, static_cast<const half*>(model_->mm_proj_b.data), n_pooled, lm_d_model_);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Post-projection RMSNorm
    if (model_->mm_post_norm_w.data) {
        vision_rmsnorm_kernel<<<n_pooled, 256, 0, stream>>>(
            d_output, static_cast<const half*>(model_->mm_post_norm_w.data), d_output, lm_d_model_, eps);
        IMP_CUDA_CHECK_LAUNCH();
    }

    return true;
}

// ======================================================================
//  gemma4v encoder — RMSNorm blocks, per-head q/k/v norm, 2D axial NEOX
//  RoPE, sandwich post-norms, GeGLU FFN, scale-1 attention, avg-pool(3)
//  + ×√D + std-affine + pre-proj RMSNorm + linear projector.
//  See docs/vision_gemma4v_spec.md.
// ======================================================================
bool VisionEncoder::encode_impl_gemma4v(const half* d_pixels, half* d_output, cudaStream_t stream) {
    const auto& cfg = model_->config;
    int np = cfg.num_patches;
    int hd = cfg.hidden_size;
    int ff = cfg.intermediate_size;
    int nh = cfg.num_heads;
    int head_dim = cfg.head_dim;
    int ps = cfg.patch_size;
    int img = cfg.image_size;
    int grid = img / ps;
    int patch_dim = ps * ps * 3;
    float eps = 1e-6f;
    float rope_base = cfg.rope_theta;

    // ---- Patch embed (no bias) ----
    extract_patches_kernel<<<np, 256, 0, stream>>>(d_pixels, d_patches_, img, img, ps, grid, grid, patch_dim);
    IMP_CUDA_CHECK_LAUNCH();
    vision_gemm(d_patches_, static_cast<const half*>(model_->patch_embd_w.data), d_hidden_, np, hd, patch_dim,
                1.0f, 0.0f, stream);

    // ---- Axial learned position embeddings (x=col, y=row tables) ----
    {
        int pos_size = static_cast<int>(model_->position_embd.shape[1]);  // 10240
        int total = np * hd;
        axial_pos_add_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(model_->position_embd.data), d_pos_x_, d_pos_y_, np, hd,
            pos_size);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // ---- 27 transformer layers ----
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        const auto& lw = model_->layers[layer];
        int rows = np * nh;

        // pre-attention RMSNorm (full-width, weighted)
        vision_rmsnorm_opt_kernel<<<np, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(lw.ln1_w.data), d_residual_, hd, eps);
        IMP_CUDA_CHECK_LAUNCH();

        // Q, K, V projections (no bias)
        vision_gemm(d_residual_, static_cast<const half*>(lw.wq.data), d_q_, np, hd, hd, 1.0f, 0.0f, stream);
        vision_gemm(d_residual_, static_cast<const half*>(lw.wk.data), d_k_, np, hd, hd, 1.0f, 0.0f, stream);
        vision_gemm(d_residual_, static_cast<const half*>(lw.wv.data), d_v_, np, hd, hd, 1.0f, 0.0f, stream);

        // per-head q/k RMSNorm (weighted [head_dim]) → 2D RoPE; per-head v RMSNorm (weightless)
        vision_rmsnorm_opt_kernel<<<rows, 64, 0, stream>>>(
            d_q_, static_cast<const half*>(lw.q_norm.data), d_q_, head_dim, eps);
        IMP_CUDA_CHECK_LAUNCH();
        vision_rmsnorm_opt_kernel<<<rows, 64, 0, stream>>>(
            d_k_, static_cast<const half*>(lw.k_norm.data), d_k_, head_dim, eps);
        IMP_CUDA_CHECK_LAUNCH();
        {
            int pairs = (head_dim / 2) / 2;
            int total = np * nh * pairs;
            int blocks = (total + 127) / 128;
            vision_rope2d_neox_kernel<<<blocks, 128, 0, stream>>>(d_q_, d_pos_x_, d_pos_y_, np, nh, head_dim,
                                                                  rope_base);
            IMP_CUDA_CHECK_LAUNCH();
            vision_rope2d_neox_kernel<<<blocks, 128, 0, stream>>>(d_k_, d_pos_x_, d_pos_y_, np, nh, head_dim,
                                                                  rope_base);
            IMP_CUDA_CHECK_LAUNCH();
        }
        vision_rmsnorm_opt_kernel<<<rows, 64, 0, stream>>>(d_v_, nullptr, d_v_, head_dim, eps);
        IMP_CUDA_CHECK_LAUNCH();

        // attention: scores = Q @ K^T, kq_scale = 1.0 (q/k-norm controls magnitude)
        {
            auto handle = get_vision_cublas_handle();
            cublasSetStream(handle, stream);
            float f_alpha = 1.0f;  // kq_scale = 1.0; FP32 accumulate to avoid FP16 overflow
            float f_beta = 0.0f;
            cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, np, np, head_dim, &f_alpha, d_k_,
                                       CUDA_R_16F, nh * head_dim, head_dim, d_q_, CUDA_R_16F, nh * head_dim,
                                       head_dim, &f_beta, d_attn_scores_, CUDA_R_16F, np,
                                       static_cast<long long>(np) * np, nh, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT);
        }
        softmax_2d_kernel<<<nh * np, 256, 0, stream>>>(d_attn_scores_, np);
        IMP_CUDA_CHECK_LAUNCH();
        {
            auto handle = get_vision_cublas_handle();
            cublasSetStream(handle, stream);
            float f_one = 1.0f;
            float f_zero = 0.0f;
            cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, np, np, &f_one, d_v_,
                                       CUDA_R_16F, nh * head_dim, head_dim, d_attn_scores_, CUDA_R_16F, np,
                                       static_cast<long long>(np) * np, &f_zero, d_attn_out_, CUDA_R_16F,
                                       nh * head_dim, head_dim, nh, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }

        // output projection (no bias) → d_residual_
        vision_gemm(d_attn_out_, static_cast<const half*>(lw.wo.data), d_residual_, np, hd, hd, 1.0f, 0.0f,
                    stream);
        // sandwich: post-attention RMSNorm BEFORE residual add
        vision_rmsnorm_opt_kernel<<<np, 256, 0, stream>>>(
            d_residual_, static_cast<const half*>(lw.attn_post_norm.data), d_residual_, hd, eps);
        IMP_CUDA_CHECK_LAUNCH();
        {
            int total = np * hd;
            add_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_hidden_, d_residual_, d_hidden_,
                                                                        total);
            IMP_CUDA_CHECK_LAUNCH();
        }

        // pre-FFN RMSNorm → GeGLU(up, gate) → down
        vision_rmsnorm_opt_kernel<<<np, 256, 0, stream>>>(
            d_hidden_, static_cast<const half*>(lw.ln2_w.data), d_residual_, hd, eps);
        IMP_CUDA_CHECK_LAUNCH();
        vision_gemm(d_residual_, static_cast<const half*>(lw.ffn_up_w.data), d_ffn_, np, ff, hd, 1.0f, 0.0f,
                    stream);
        vision_gemm(d_residual_, static_cast<const half*>(lw.ffn_gate_w.data), d_gate_, np, ff, hd, 1.0f, 0.0f,
                    stream);
        {
            int total = np * ff;
            gelu_tanh_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_gate_, total);
            IMP_CUDA_CHECK_LAUNCH();
            mul_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_gate_, d_ffn_, d_ffn_, total);
            IMP_CUDA_CHECK_LAUNCH();
        }
        vision_gemm(d_ffn_, static_cast<const half*>(lw.ffn_down_w.data), d_residual_, np, hd, ff, 1.0f, 0.0f,
                    stream);
        // sandwich: post-FFN RMSNorm BEFORE residual add
        vision_rmsnorm_opt_kernel<<<np, 256, 0, stream>>>(
            d_residual_, static_cast<const half*>(lw.ffn_post_norm.data), d_residual_, hd, eps);
        IMP_CUDA_CHECK_LAUNCH();
        {
            int total = np * hd;
            add_tensors_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_hidden_, d_residual_, d_hidden_,
                                                                        total);
            IMP_CUDA_CHECK_LAUNCH();
        }
    }

    // ---- Pooler: avg-pool(n_merge) ----
    int pool = cfg.n_merge;
    int out_h = grid / pool, out_w = grid / pool;
    int n_pooled = out_h * out_w;
    avg_pool_spatial_kernel<<<n_pooled, 256, 0, stream>>>(d_hidden_, d_pooled_, grid, grid, hd, pool, out_h,
                                                          out_w);
    IMP_CUDA_CHECK_LAUNCH();

    // ---- Fused FP32 tail: ×√hd → (x-std_bias)*std_scale → pre-projection RMSNorm.
    // Gemma vision activations reach ~3000; ×√hd would overflow FP16 if stored, so
    // the whole tail runs in FP32 and only the RMS-normalized result is written. ----
    gemma4v_tail_norm_kernel<<<n_pooled, 256, hd * sizeof(float), stream>>>(
        d_pooled_, static_cast<const half*>(model_->std_bias.data),
        static_cast<const half*>(model_->std_scale.data), d_pooled_, hd, sqrtf(static_cast<float>(hd)), eps);
    IMP_CUDA_CHECK_LAUNCH();

    // ---- Linear projection ----
    vision_gemm(d_pooled_, static_cast<const half*>(model_->mm_proj_w.data), d_output, n_pooled, lm_d_model_,
                hd, 1.0f, 0.0f, stream);

    return true;
}

// ---- Public kernel launcher for embedding replacement ----
// `emb_offset` is how many image tokens earlier chunks already placed; see
// vision/deepstack_inject.h, whose kernel has to agree with this one.
void launch_replace_vision_embeddings(half* hidden, const int32_t* token_ids, const half* vision_emb,
                                      int vision_token_id, int n_tokens, int d_model, int n_vision_tokens,
                                      int emb_offset, cudaStream_t stream) {
    if (n_vision_tokens <= 0)
        return;
    const int remaining = n_vision_tokens - emb_offset;
    if (emb_offset < 0 || remaining <= 0)
        return;
    replace_vision_embeddings_v2_kernel<<<remaining, 256, 0, stream>>>(hidden, token_ids, vision_emb,
                                                                       vision_token_id, n_tokens, d_model,
                                                                       n_vision_tokens, emb_offset);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
