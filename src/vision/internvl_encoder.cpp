#include "vision/internvl_encoder.h"

#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include "vision/internvl_encoder_kernels.h"
#include "vision/qwen3vl_encoder_kernels.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>

namespace imp {

namespace {

constexpr int kAttnChunk = 512;  // query rows per score pass; scores are the only tokens^2 buffer

cublasHandle_t s_handle = nullptr;

cublasHandle_t handle() {
    if (!s_handle) {
        cublasCreate(&s_handle);
        cublasSetMathMode(s_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return s_handle;
}

// C[M, N] = A[M, K] @ W[N, K]^T + b, row-major, FP32 accumulation (fc2 sums 4096 terms). The bias
// is written into C first and accumulated with beta = 1, so the output is rounded to FP16 once,
// not twice (GEMM, then bias add): the per-stage bound against HF FP32 is 1e-3 relL2.
void linear(const half* A, const half* W, half* C, int M, int N, int K, const half* b, cudaStream_t stream) {
    const float one = 1.0f;
    launch_internvl_fill_rows(C, b, M, N, stream);
    cublasSetStream(handle(), stream);
    cublasGemmEx(handle(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &one, C,
                 CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

const half* hp(const Tensor& t) { return static_cast<const half*>(t.data); }

void internvl_encoder_reset_static_cuda_state() {
    if (s_handle) {
        (void)cublasDestroy(s_handle);
        s_handle = nullptr;
    }
}
IMP_REGISTER_CUDA_STATIC_RESET(internvl_encoder_reset_static_cuda_state);

}  // namespace

InternVLEncoder::~InternVLEncoder() { free_buffers(); }

size_t InternVLEncoder::demand_bytes(const VisionConfig& c) {
    const int64_t n = 1 + c.num_patches, H = c.hidden_size, m = c.num_patches / 4;
    int64_t elems = 0;
    elems += n * H * 3;  // hidden, normed, proj
    elems += n * 3 * H;  // qkv
    elems += n * H * 4;  // q, k, v, attn
    elems += static_cast<int64_t>(c.num_heads) * std::min<int64_t>(kAttnChunk, n) * n;  // scores
    elems += n * c.intermediate_size;                                                   // ffn
    elems += m * 4 * H * 2;          // shuffled, merge_norm
    elems += m * c.out_hidden_size;  // merge_fc
    return static_cast<size_t>(elems) * sizeof(half);
}

void InternVLEncoder::free_buffers() {
    taken_bytes_ = 0;
    d_hidden_ = d_normed_ = d_proj_ = d_qkv_ = d_q_ = d_k_ = d_v_ = d_attn_ = d_scores_ = nullptr;
    d_ffn_ = d_shuffled_ = d_merge_norm_ = d_merge_fc_ = nullptr;
}

bool InternVLEncoder::init(const VisionModel& model) {
    free_buffers();
    const VisionConfig& c = model.config;
    if (!c.is_internvl || c.num_patches <= 0 || c.pos_embed_grid * c.pos_embed_grid != c.num_patches) {
        IMP_LOG_ERROR("InternVL encoder: config is not a parsed InternVL vision config");
        return false;
    }
    model_ = &model;
    const int64_t n = 1 + c.num_patches, H = c.hidden_size, m = c.num_patches / 4;
    bool ok = true;
    auto take = [&](int64_t elems, const char* tag) -> half* {
        if (!ok)
            return nullptr;
        const size_t bytes = static_cast<size_t>(elems) * sizeof(half);
        auto slab = engine_arena().take_bytes(bytes);
        if (slab.empty()) {
            IMP_LOG_ERROR("InternVL encoder: engine arena exhausted for %s (%zu bytes)", tag, bytes);
            ok = false;
            return nullptr;
        }
        taken_bytes_ += bytes;
        return reinterpret_cast<half*>(slab.data());
    };
    d_hidden_ = take(n * H, "internvl_enc_hidden");
    d_normed_ = take(n * H, "internvl_enc_normed");
    d_proj_ = take(n * H, "internvl_enc_proj");
    d_qkv_ = take(n * 3 * H, "internvl_enc_qkv");
    d_q_ = take(n * H, "internvl_enc_q");
    d_k_ = take(n * H, "internvl_enc_k");
    d_v_ = take(n * H, "internvl_enc_v");
    d_attn_ = take(n * H, "internvl_enc_attn");
    d_scores_ = take(static_cast<int64_t>(c.num_heads) * std::min<int64_t>(kAttnChunk, n) * n,
                     "internvl_enc_scores");
    d_ffn_ = take(n * c.intermediate_size, "internvl_enc_ffn");
    d_shuffled_ = take(m * 4 * H, "internvl_enc_shuffled");
    d_merge_norm_ = take(m * 4 * H, "internvl_enc_merge_norm");
    d_merge_fc_ = take(m * c.out_hidden_size, "internvl_enc_merge_fc");
    if (!ok) {
        free_buffers();
        return false;
    }
    IMP_LOG_INFO("InternVL encoder ready: %d patches -> %d image tokens", c.num_patches, c.num_image_tokens);
    return true;
}

bool InternVLEncoder::attention(int tokens, cudaStream_t stream) {
    const VisionConfig& c = model_->config;
    const int heads = c.num_heads, hd = c.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd)), zero = 0.0f, one = 1.0f;
    cublasSetStream(handle(), stream);
    for (int c0 = 0; c0 < tokens; c0 += kAttnChunk) {
        const int rows = std::min(kAttnChunk, tokens - c0);
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            handle(), CUBLAS_OP_T, CUBLAS_OP_N, tokens, rows, hd, &scale, d_k_, CUDA_R_16F, hd,
            static_cast<long long>(tokens) * hd, d_q_ + static_cast<int64_t>(c0) * hd, CUDA_R_16F, hd,
            static_cast<long long>(tokens) * hd, &zero, d_scores_, CUDA_R_16F, tokens,
            static_cast<long long>(rows) * tokens, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            IMP_LOG_ERROR("InternVL encoder: QK^T failed (%d)", static_cast<int>(st));
            return false;
        }
        launch_qwen3vl_softmax_rows(d_scores_, heads * rows, tokens, stream);
        st = cublasGemmStridedBatchedEx(handle(), CUBLAS_OP_N, CUBLAS_OP_N, hd, rows, tokens, &one, d_v_,
                                        CUDA_R_16F, hd, static_cast<long long>(tokens) * hd, d_scores_,
                                        CUDA_R_16F, tokens, static_cast<long long>(rows) * tokens, &zero,
                                        d_attn_ + static_cast<int64_t>(c0) * hd, CUDA_R_16F, hd,
                                        static_cast<long long>(tokens) * hd, heads, CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            IMP_LOG_ERROR("InternVL encoder: attn@V failed (%d)", static_cast<int>(st));
            return false;
        }
    }
    return true;
}

bool InternVLEncoder::encode(const half* d_patches, half* d_out, cudaStream_t stream,
                             const InternVLStageTaps* taps) {
    if (!model_ || !d_hidden_) {
        IMP_LOG_ERROR("InternVL encoder: encode before init");
        return false;
    }
    const VisionConfig& c = model_->config;
    const VisionModel& M = *model_;
    const int np = c.num_patches, n = 1 + np, H = c.hidden_size, I = c.intermediate_size;
    const int m = np / 4, W = 4 * H, D = c.out_hidden_size;
    const float eps = c.layer_norm_eps;
    const int features = static_cast<int>(M.patch_embd_w.shape[1]);
    auto tap = [&](half* dst, const half* src, int64_t elems) {
        if (dst)
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dst, src, static_cast<size_t>(elems) * sizeof(half),
                                               cudaMemcpyDeviceToDevice, stream));
    };

    // Conv2d with kernel == stride == patch is a GEMM against the flattened [H, 3*P*P] weight.
    linear(d_patches, hp(M.patch_embd_w), d_hidden_ + H, np, H, features, hp(M.patch_embd_b), stream);
    launch_internvl_cls_pos(d_hidden_, hp(M.cls_token), hp(M.position_embd), n, H, stream);
    if (taps)
        tap(taps->embeddings, d_hidden_, static_cast<int64_t>(n) * H);

    for (int l = 0; l < c.num_layers; ++l) {
        const VisionLayerWeights& L = M.layers[static_cast<size_t>(l)];
        launch_qwen3vl_layernorm(d_hidden_, hp(L.ln1_w), hp(L.ln1_b), d_normed_, n, H, eps, stream);
        linear(d_normed_, hp(L.wq), d_qkv_, n, 3 * H, H, hp(L.bq), stream);
        launch_internvl_split_heads(d_qkv_, d_q_, d_k_, d_v_, n, c.num_heads, c.head_dim, stream);
        if (!attention(n, stream))
            return false;
        launch_qwen3vl_merge_heads(d_attn_, d_normed_, n, c.num_heads, c.head_dim, stream);
        linear(d_normed_, hp(L.wo), d_proj_, n, H, H, hp(L.bo), stream);
        launch_internvl_scaled_residual(d_hidden_, d_proj_, hp(L.ls1), n, H, stream);

        launch_qwen3vl_layernorm(d_hidden_, hp(L.ln2_w), hp(L.ln2_b), d_normed_, n, H, eps, stream);
        linear(d_normed_, hp(L.ffn_up_w), d_ffn_, n, I, H, hp(L.ffn_up_b), stream);
        launch_qwen3vl_gelu_erf(d_ffn_, static_cast<int64_t>(n) * I, stream);  // hidden_act "gelu"
        linear(d_ffn_, hp(L.ffn_down_w), d_proj_, n, H, I, hp(L.ffn_down_b), stream);
        launch_internvl_scaled_residual(d_hidden_, d_proj_, hp(L.ls2), n, H, stream);
        if (taps && static_cast<size_t>(l) < taps->layers.size())
            tap(taps->layers[static_cast<size_t>(l)], d_hidden_, static_cast<int64_t>(n) * H);
    }

    // vision_feature_layer -1, select "default" (drop CLS), pixel shuffle 0.5, projector.
    launch_internvl_pixel_shuffle(d_hidden_, d_shuffled_, c.pos_embed_grid, H, stream);
    if (taps)
        tap(taps->shuffled, d_shuffled_, static_cast<int64_t>(m) * W);
    launch_qwen3vl_layernorm(d_shuffled_, hp(M.merger.norm_w), hp(M.merger.norm_b), d_merge_norm_, m, W,
                             1e-5f,
                             stream);  // nn.LayerNorm default eps
    linear(d_merge_norm_, hp(M.merger.fc1_w), d_merge_fc_, m, D, W, hp(M.merger.fc1_b), stream);
    launch_qwen3vl_gelu_erf(d_merge_fc_, static_cast<int64_t>(m) * D, stream);  // projector_hidden_act "gelu"
    linear(d_merge_fc_, hp(M.merger.fc2_w), d_out, m, D, D, hp(M.merger.fc2_b), stream);
    return true;
}

}  // namespace imp
