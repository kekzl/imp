// =============================================================================
// encoder_forward.cu — encoder-only embedder forward (#836, nomic-bert)
// =============================================================================
// See header for the pass structure. Everything runs on one stream; the only
// syncs are the final D2H of the pooled vector (encoder_embed) and the
// one-time dequant in encoder_workspace_init.
// =============================================================================

#include "compute/encoder_forward.h"
#include "compute/activation.h"
#include "compute/attention_cublas.h"
#include "compute/embedding.h"
#include "compute/gemm.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "core/logging.h"
#include "model/model.h"
#include "quant/dequant_gpu.h"
#include <cuda_fp16.h>
#include <numeric>
#include <utility>

namespace imp {

namespace {

// h[i, :] += type_row[:]  (token_type embedding, row 0 for plain text)
__global__ void encoder_add_type_kernel(__half* __restrict__ h, const __half* __restrict__ type_row,
                                        int n, int d) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<int64_t>(n) * d)
        h[idx] = __float2half(__half2float(h[idx]) + __half2float(type_row[idx % d]));
}

// out[j] = mean_i h[i, j]; then L2-normalize out in a second kernel.
__global__ void encoder_mean_pool_kernel(const __half* __restrict__ h, float* __restrict__ out,
                                         int n, int d) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) return;
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += __half2float(h[static_cast<int64_t>(i) * d + j]);
    out[j] = s / n;
}

__global__ void encoder_l2_normalize_kernel(float* __restrict__ v, int d) {
    __shared__ float warp_sums[32];
    float s = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += v[i] * v[i];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffff, s, m);
    if (lane == 0) warp_sums[warp] = s;
    __syncthreads();
    const int n_warps = (blockDim.x + 31) / 32;
    float total = (threadIdx.x < n_warps) ? warp_sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32)
        for (int m = 16; m > 0; m >>= 1) total += __shfl_xor_sync(0xffffffff, total, m);
    if (threadIdx.x == 0) warp_sums[0] = total;
    __syncthreads();
    const float inv = rsqrtf(warp_sums[0] + 1e-12f);
    for (int i = threadIdx.x; i < d; i += blockDim.x) v[i] *= inv;
}

// Dequantize one weight tensor to a fresh FP16 device buffer. F16 sources are
// device-copied as-is.
void* dequant_to_fp16(const Tensor& w, cudaStream_t stream) {
    if (!w.data) return nullptr;
    const int rows = static_cast<int>(w.shape[0]);
    const int cols = static_cast<int>(w.shape[1]);
    void* dst = nullptr;
    if (cudaMalloc(&dst, static_cast<size_t>(rows) * cols * sizeof(__half)) != cudaSuccess)
        return nullptr;
    if (w.qtype == QType::F16) {
        cudaMemcpyAsync(dst, w.data, static_cast<size_t>(rows) * cols * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    } else if (dequant_gpu_supported(w.qtype)) {
        dequant_gpu(w.data, dst, w.qtype, rows, cols, stream);
    } else {
        IMP_LOG_ERROR("encoder: unsupported weight qtype %d", std::to_underlying(w.qtype));
        cudaFree(dst);
        return nullptr;
    }
    return dst;
}

Tensor fp16_view(void* p, int64_t r, int64_t c) {
    int64_t s[2] = {r, c};
    return Tensor(p, QType::F16, 2, s, /*on_device=*/true);
}

}  // namespace

bool encoder_workspace_init(EncoderWorkspace& ws, const Model& model, int max_tokens,
                            cudaStream_t stream) {
    const ModelConfig& cfg = model.config();
    ws.max_tokens = max_tokens;
    ws.d_model = cfg.d_model;
    ws.d_ff = cfg.d_ff;
    ws.n_layers = cfg.n_layers;
    ws.n_heads = cfg.n_heads;
    ws.head_dim = cfg.head_dim > 0 ? cfg.head_dim : cfg.d_model / cfg.n_heads;
    ws.rope_theta = cfg.rope_theta;
    ws.ln_eps = cfg.rms_norm_eps;  // GGUF layer_norm_epsilon lands here

    ws.layers.resize(ws.n_layers);
    for (int i = 0; i < ws.n_layers; ++i) {
        const auto& L = model.layer(i);
        auto& E = ws.layers[i];
        E.wq = dequant_to_fp16(L.wq, stream);
        E.wk = dequant_to_fp16(L.wk, stream);
        E.wv = dequant_to_fp16(L.wv, stream);
        E.wo = dequant_to_fp16(L.wo, stream);
        E.wg = dequant_to_fp16(L.w_gate, stream);
        E.wu = dequant_to_fp16(L.w_up, stream);
        E.wd = dequant_to_fp16(L.w_down, stream);
        if (!E.wq || !E.wk || !E.wv || !E.wo || !E.wg || !E.wu || !E.wd) {
            IMP_LOG_ERROR("encoder: weight dequant failed at layer %d", i);
            encoder_workspace_free(ws);
            return false;
        }
    }

    const size_t nd = static_cast<size_t>(max_tokens) * ws.d_model * sizeof(__half);
    const size_t nff = static_cast<size_t>(max_tokens) * ws.d_ff * sizeof(__half);
    const size_t ns = static_cast<size_t>(ws.n_heads) * max_tokens * max_tokens * sizeof(float);
    bool ok = true;
    auto alloc = [&](void** p, size_t bytes) { ok &= cudaMalloc(p, bytes) == cudaSuccess; };
    alloc(reinterpret_cast<void**>(&ws.d_tokens), max_tokens * sizeof(int32_t));
    alloc(reinterpret_cast<void**>(&ws.d_positions), max_tokens * sizeof(int32_t));
    alloc(&ws.d_h, nd);
    alloc(&ws.d_q, nd);
    alloc(&ws.d_k, nd);
    alloc(&ws.d_v, nd);
    alloc(&ws.d_attn, nd);
    alloc(&ws.d_proj, nd);
    alloc(&ws.d_gate, nff);
    alloc(&ws.d_up, nff);
    alloc(&ws.d_act, nff);
    alloc(&ws.d_scores, ns);
    alloc(reinterpret_cast<void**>(&ws.d_pooled), ws.d_model * sizeof(float));
    if (!ok) {
        IMP_LOG_ERROR("encoder: scratch alloc failed (max_tokens=%d)", max_tokens);
        encoder_workspace_free(ws);
        return false;
    }

    // Positions 0..max_n-1 once (rope reads per-row).
    std::vector<int32_t> pos(max_tokens);
    std::iota(pos.begin(), pos.end(), 0);
    cudaMemcpyAsync(ws.d_positions, pos.data(), max_tokens * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return true;
}

void encoder_workspace_free(EncoderWorkspace& ws) {
    for (auto& E : ws.layers)
        for (void* p : {E.wq, E.wk, E.wv, E.wo, E.wg, E.wu, E.wd})
            if (p) cudaFree(p);
    ws.layers.clear();
    for (void* p : {static_cast<void*>(ws.d_tokens), static_cast<void*>(ws.d_positions), ws.d_h,
                    ws.d_q, ws.d_k, ws.d_v, ws.d_attn, ws.d_proj, ws.d_gate, ws.d_up, ws.d_act,
                    ws.d_scores, static_cast<void*>(ws.d_pooled)})
        if (p) cudaFree(p);
    ws.d_tokens = nullptr;
    ws.d_positions = nullptr;
    ws.d_h = ws.d_q = ws.d_k = ws.d_v = ws.d_attn = ws.d_proj = nullptr;
    ws.d_gate = ws.d_up = ws.d_act = ws.d_scores = nullptr;
    ws.d_pooled = nullptr;
    ws.max_tokens = 0;
}

bool encoder_embed(const Model& model, EncoderWorkspace& ws, const int32_t* tokens, int n,
                   float* out_host, cudaStream_t stream) {
    if (ws.max_tokens == 0 || n <= 0 || n > ws.max_tokens || !out_host)
        return false;
    const int d = ws.d_model;
    const int dff = ws.d_ff;

    cudaMemcpyAsync(ws.d_tokens, tokens, n * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // Embedding + token-type row 0 + post-embedding LayerNorm.
    Tensor h = fp16_view(ws.d_h, n, d);
    embedding_lookup(model.tok_emb_, ws.d_tokens, n, h, model.tok_emb_.qtype, stream);
    if (model.token_types_.data) {
        const int block = 256;
        const int64_t total = static_cast<int64_t>(n) * d;
        encoder_add_type_kernel<<<static_cast<int>((total + block - 1) / block), block, 0, stream>>>(
            static_cast<__half*>(ws.d_h), static_cast<const __half*>(model.token_types_.data), n, d);
    }
    Tensor none{};
    layernorm_residual(h, none, model.tok_emb_norm_, model.tok_emb_norm_bias_, h, ws.ln_eps, stream);

    const float scale = 1.0f / sqrtf(static_cast<float>(ws.head_dim));
    for (int i = 0; i < ws.n_layers; ++i) {
        const auto& L = model.layer(i);
        const auto& E = ws.layers[i];
        Tensor q = fp16_view(ws.d_q, n, d), k = fp16_view(ws.d_k, n, d), v = fp16_view(ws.d_v, n, d);
        gemm(h, fp16_view(E.wq, d, d), q, 1.0f, 0.0f, stream);
        gemm(h, fp16_view(E.wk, d, d), k, 1.0f, 0.0f, stream);
        gemm(h, fp16_view(E.wv, d, d), v, 1.0f, 0.0f, stream);
        int64_t r4[4] = {1, n, ws.n_heads, ws.head_dim};
        Tensor q4 = q.reshape(4, r4), k4 = k.reshape(4, r4);
        rope_forward(q4, k4, ws.d_positions, ws.head_dim, ws.rope_theta, 1.0f, /*rope_dim=*/0,
                     /*neox=*/true, 0.0f, 1.0f, nullptr, stream);

        Tensor attn = fp16_view(ws.d_attn, n, d);
        int64_t s_shape[2] = {static_cast<int64_t>(ws.n_heads), static_cast<int64_t>(n) * n};
        Tensor S(ws.d_scores, QType::F32, 2, s_shape, true);
        attention_cublas_prefill(q, k, v, attn, S, ws.n_heads, ws.n_heads, ws.head_dim, scale,
                                 /*causal=*/false, 0.0f, /*q_offset=*/0, stream);

        Tensor proj = fp16_view(ws.d_proj, n, d);
        gemm(attn, fp16_view(E.wo, d, d), proj, 1.0f, 0.0f, stream);
        layernorm_residual(proj, h, L.post_attn_norm, L.post_attn_norm_bias, h, ws.ln_eps, stream);

        Tensor g = fp16_view(ws.d_gate, n, dff), u = fp16_view(ws.d_up, n, dff),
               a = fp16_view(ws.d_act, n, dff);
        gemm(h, fp16_view(E.wg, dff, d), g, 1.0f, 0.0f, stream);
        gemm(h, fp16_view(E.wu, dff, d), u, 1.0f, 0.0f, stream);
        swiglu(g, u, a, stream);
        gemm(a, fp16_view(E.wd, d, dff), proj, 1.0f, 0.0f, stream);
        layernorm_residual(proj, h, L.post_ffn_norm, L.post_ffn_norm_bias, h, ws.ln_eps, stream);
    }

    // Mean pool + L2 normalize + D2H.
    encoder_mean_pool_kernel<<<(d + 255) / 256, 256, 0, stream>>>(
        static_cast<const __half*>(ws.d_h), ws.d_pooled, n, d);
    encoder_l2_normalize_kernel<<<1, 256, 0, stream>>>(ws.d_pooled, d);
    cudaMemcpyAsync(out_host, ws.d_pooled, d * sizeof(float), cudaMemcpyDeviceToHost, stream);
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

}  // namespace imp
