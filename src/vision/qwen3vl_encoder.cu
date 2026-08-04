#include "vision/qwen3vl_encoder.h"
#include "core/cuda_static_reset.h"

#include "core/logging.h"
#include "memory/engine_arena.h"
#include "vision/qwen3vl_encoder_kernels.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>

namespace imp {

namespace {

// Query rows per attention pass. The score matrix is the only buffer that grows
// with tokens^2, so it is the one that gets chunked rather than sized for the
// worst case.
constexpr int kAttnChunk = 512;
constexpr float kLayerNormEps = 1e-6f;
constexpr float kVisionRopeTheta = 10000.0f;

cublasHandle_t s_handle = nullptr;

cublasHandle_t handle() {
    if (!s_handle) {
        cublasCreate(&s_handle);
        cublasSetMathMode(s_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return s_handle;
}

// C[M, N] = alpha * A[M, K] @ B[N, K]^T, all row-major. Weights come off the
// checkpoint as [out, in], which is exactly B.
void gemm_nt(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta,
             cudaStream_t stream) {
    cublasSetStream(handle(), stream);
    // FP32 accumulation: the FFN down-projection sums 4096 terms, and FP16
    // accumulation overflows there for high-magnitude tokens.
    cublasGemmEx(handle(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
                 &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

}  // namespace

// Pre-cudaDeviceReset hook (see core/cuda_static_reset.h).
void qwen3vl_encoder_reset_static_cuda_state() {
    if (s_handle) {
        (void)cublasDestroy(s_handle);
        s_handle = nullptr;
    }
}

// Registered as a pre-cudaDeviceReset hook (#1207); see core/cuda_static_reset.h.
namespace {
IMP_REGISTER_CUDA_STATIC_RESET(qwen3vl_encoder_reset_static_cuda_state);
}  // namespace

Qwen3VLEncoder::~Qwen3VLEncoder() { free_buffers(); }

size_t Qwen3VLEncoder::demand_bytes(const VisionConfig& c, int max_tokens) {
    const int64_t n = max_tokens;
    const int64_t H = c.hidden_size;
    const size_t h = sizeof(half);
    size_t total = 0;
    total += static_cast<size_t>(n * H) * h;                       // hidden
    total += static_cast<size_t>(n * H) * h;                       // normed
    total += static_cast<size_t>(n * H) * h;                       // proj
    total += static_cast<size_t>(n * 3 * H) * h;                   // qkv
    total += static_cast<size_t>(n * H) * h * 4;                   // q, k, v, attn
    total += static_cast<size_t>(c.num_heads * std::min<int64_t>(kAttnChunk, n) * n) * h;  // scores
    total += static_cast<size_t>(n * c.intermediate_size) * h;     // ffn
    total += static_cast<size_t>(n * H) * h * 2;                   // merge_norm, merge_fc
    total += static_cast<size_t>(n) * sizeof(int32_t) * 2;         // row, col
    total += static_cast<size_t>(n * kQwenVisionPosTaps) * sizeof(int32_t);  // taps
    total += static_cast<size_t>(n * kQwenVisionPosTaps) * sizeof(float);    // weights
    return total;
}

void Qwen3VLEncoder::free_buffers() {
    // The buffers are arena slices; the arena releases wholesale on close, so
    // there is nothing to hand back here. Dropping the pointers is what keeps a
    // re-init from reading slices that a closed arena no longer backs.
    taken_bytes_ = 0;
    d_hidden_ = d_normed_ = d_proj_ = d_qkv_ = nullptr;
    d_q_ = d_k_ = d_v_ = d_attn_ = d_scores_ = d_ffn_ = nullptr;
    d_merge_norm_ = d_merge_fc_ = nullptr;
    d_row_ = d_col_ = d_taps_ = nullptr;
    d_weights_ = nullptr;
    max_tokens_ = 0;
}

int Qwen3VLEncoder::merged_tokens(int tokens) const {
    if (!model_)
        return 0;
    const int unit = model_->config.merge_size * model_->config.merge_size;
    return tokens / unit;
}

bool Qwen3VLEncoder::init(const VisionModel& model, int max_tokens) {
    free_buffers();
    if (max_tokens <= 0) {
        IMP_LOG_ERROR("Qwen3-VL encoder: needs a positive token budget");
        return false;
    }
    const VisionConfig& c = model.config;
    if (!c.is_qwen3vl) {
        IMP_LOG_ERROR("Qwen3-VL encoder: config is not a Qwen3-VL vision config");
        return false;
    }
    const int unit = c.merge_size * c.merge_size;
    if (max_tokens % unit != 0) {
        IMP_LOG_ERROR("Qwen3-VL encoder: token budget %d is not a multiple of the merge unit %d", max_tokens,
                      unit);
        return false;
    }
    model_ = &model;
    max_tokens_ = max_tokens;
    taken_bytes_ = 0;

    const int64_t n = max_tokens;
    const int64_t H = c.hidden_size;
    bool ok = true;
    auto take = [&](int64_t elems, size_t elem_bytes, const char* tag) -> void* {
        if (!ok)
            return nullptr;
        const size_t bytes = static_cast<size_t>(elems) * elem_bytes;
        auto slab = engine_arena().take_bytes(bytes);
        if (slab.empty()) {
            IMP_LOG_ERROR("Qwen3-VL encoder: engine arena exhausted for %s (%zu bytes) — the arena "
                          "was reserved without this encoder",
                          tag, bytes);
            ok = false;
            return nullptr;
        }
        taken_bytes_ += bytes;
        return slab.data();
    };
    auto take_half = [&](int64_t elems, const char* tag) {
        return static_cast<half*>(take(elems, sizeof(half), tag));
    };

    d_hidden_ = take_half(n * H, "vision_enc_hidden");
    d_normed_ = take_half(n * H, "vision_enc_normed");
    d_proj_ = take_half(n * H, "vision_enc_proj");
    d_qkv_ = take_half(n * 3 * H, "vision_enc_qkv");
    d_q_ = take_half(n * H, "vision_enc_q");
    d_k_ = take_half(n * H, "vision_enc_k");
    d_v_ = take_half(n * H, "vision_enc_v");
    d_attn_ = take_half(n * H, "vision_enc_attn");
    d_scores_ = take_half(static_cast<int64_t>(c.num_heads) * std::min<int64_t>(kAttnChunk, n) * n,
                          "vision_enc_scores");
    d_ffn_ = take_half(n * c.intermediate_size, "vision_enc_ffn");
    d_merge_norm_ = take_half(n * H, "vision_enc_merge_norm");
    d_merge_fc_ = take_half(n * H, "vision_enc_merge_fc");
    d_row_ = static_cast<int32_t*>(take(n, sizeof(int32_t), "vision_enc_row"));
    d_col_ = static_cast<int32_t*>(take(n, sizeof(int32_t), "vision_enc_col"));
    d_taps_ = static_cast<int32_t*>(take(n * kQwenVisionPosTaps, sizeof(int32_t), "vision_enc_taps"));
    d_weights_ = static_cast<float*>(take(n * kQwenVisionPosTaps, sizeof(float), "vision_enc_weights"));

    if (!ok) {
        free_buffers();
        return false;
    }
    IMP_LOG_INFO("Qwen3-VL encoder ready: up to %d patches (%d merged tokens)", max_tokens,
                 merged_tokens(max_tokens));
    return true;
}

bool Qwen3VLEncoder::attention(int tokens, cudaStream_t stream) {
    const VisionConfig& c = model_->config;
    const int heads = c.num_heads;
    const int hd = c.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const float zero = 0.0f;
    cublasSetStream(handle(), stream);

    for (int c0 = 0; c0 < tokens; c0 += kAttnChunk) {
        const int rows = std::min(kAttnChunk, tokens - c0);

        // scores[h] = Q[h][c0 : c0+rows] @ K[h]^T * scale   -> [rows, tokens]
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            handle(), CUBLAS_OP_T, CUBLAS_OP_N, tokens, rows, hd, &scale, d_k_, CUDA_R_16F, hd,
            static_cast<long long>(tokens) * hd, d_q_ + static_cast<int64_t>(c0) * hd, CUDA_R_16F, hd,
            static_cast<long long>(tokens) * hd, &zero, d_scores_, CUDA_R_16F, tokens,
            static_cast<long long>(rows) * tokens, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            IMP_LOG_ERROR("Qwen3-VL encoder: QK^T failed (%d)", static_cast<int>(st));
            return false;
        }

        // Every head's chunk is contiguous, so one launch normalises them all.
        launch_qwen3vl_softmax_rows(d_scores_, heads * rows, tokens, stream);

        // out[h][c0 : c0+rows] = scores[h] @ V[h]   -> [rows, head_dim]
        const float one = 1.0f;
        st = cublasGemmStridedBatchedEx(handle(), CUBLAS_OP_N, CUBLAS_OP_N, hd, rows, tokens, &one, d_v_,
                                        CUDA_R_16F, hd, static_cast<long long>(tokens) * hd, d_scores_,
                                        CUDA_R_16F, tokens, static_cast<long long>(rows) * tokens, &zero,
                                        d_attn_ + static_cast<int64_t>(c0) * hd, CUDA_R_16F, hd,
                                        static_cast<long long>(tokens) * hd, heads, CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            IMP_LOG_ERROR("Qwen3-VL encoder: attn@V failed (%d)", static_cast<int>(st));
            return false;
        }
    }
    return true;
}

bool Qwen3VLEncoder::run_merger(const VisionMergerWeights& m, const half* d_hidden, int tokens, half* d_out,
                                cudaStream_t stream) {
    const VisionConfig& c = model_->config;
    const int unit = c.merge_size * c.merge_size;
    const int merged = tokens / unit;
    const int wide = c.hidden_size * unit;

    // The norm's own width says where it sits relative to the 2x2 concat: the
    // main merger normalises each patch (hidden_size), the DeepStack mergers
    // normalise the concatenated token (merge^2 * hidden_size). Same bytes
    // either way — only the row length differs.
    const bool postshuffle = m.norm_w.shape[0] == wide;
    if (postshuffle)
        launch_qwen3vl_layernorm(d_hidden, static_cast<const half*>(m.norm_w.data),
                                 static_cast<const half*>(m.norm_b.data), d_merge_norm_, merged, wide,
                                 kLayerNormEps, stream);
    else
        launch_qwen3vl_layernorm(d_hidden, static_cast<const half*>(m.norm_w.data),
                                 static_cast<const half*>(m.norm_b.data), d_merge_norm_, tokens,
                                 c.hidden_size, kLayerNormEps, stream);

    gemm_nt(d_merge_norm_, static_cast<const half*>(m.fc1_w.data), d_merge_fc_, merged, wide, wide, 1.0f,
            0.0f, stream);
    launch_qwen3vl_add_bias(d_merge_fc_, static_cast<const half*>(m.fc1_b.data), merged, wide, stream);
    // nn.GELU(), i.e. the exact erf form — NOT the tanh approximation the block
    // MLP uses. Same symbol upstream, different function.
    launch_qwen3vl_gelu_erf(d_merge_fc_, static_cast<int64_t>(merged) * wide, stream);

    gemm_nt(d_merge_fc_, static_cast<const half*>(m.fc2_w.data), d_out, merged, c.out_hidden_size, wide, 1.0f,
            0.0f, stream);
    launch_qwen3vl_add_bias(d_out, static_cast<const half*>(m.fc2_b.data), merged, c.out_hidden_size, stream);
    return true;
}

bool Qwen3VLEncoder::encode(const half* d_patches, const QwenVisionGrid& grid, half* d_out,
                            const std::vector<half*>& d_deepstack_out, cudaStream_t stream) {
    if (!model_ || !d_hidden_) {
        IMP_LOG_ERROR("Qwen3-VL encoder: encode before init");
        return false;
    }
    const VisionConfig& c = model_->config;
    const int n = grid.tokens;
    if (n <= 0 || n > max_tokens_) {
        IMP_LOG_ERROR("Qwen3-VL encoder: %d patches exceeds the %d-patch budget", n, max_tokens_);
        return false;
    }
    if (!d_deepstack_out.empty() && d_deepstack_out.size() != c.deepstack_indexes.size()) {
        IMP_LOG_ERROR("Qwen3-VL encoder: %zu DeepStack outputs for %zu taps", d_deepstack_out.size(),
                      c.deepstack_indexes.size());
        return false;
    }

    const int H = c.hidden_size;
    const int features = static_cast<int>(model_->patch_embd_w.shape[1]);

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_row_, grid.row.data(), static_cast<size_t>(n) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_col_, grid.col.data(), static_cast<size_t>(n) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_taps_, grid.pos_taps.data(), grid.pos_taps.size() * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_weights_, grid.pos_weights.data(),
                                       grid.pos_weights.size() * sizeof(float), cudaMemcpyHostToDevice,
                                       stream));

    // Patch embedding. The checkpoint's Conv3d has kernel == stride == the whole
    // patch, so it is a plain matrix product against the flattened weight.
    gemm_nt(d_patches, static_cast<const half*>(model_->patch_embd_w.data), d_hidden_, n, H, features, 1.0f,
            0.0f, stream);
    launch_qwen3vl_add_bias(d_hidden_, static_cast<const half*>(model_->patch_embd_b.data), n, H, stream);
    launch_qwen3vl_pos_embed_add(d_hidden_, static_cast<const half*>(model_->position_embd.data), d_taps_,
                                 d_weights_, n, H, kQwenVisionPosTaps, stream);

    size_t next_tap = 0;
    for (int l = 0; l < c.num_layers; ++l) {
        const VisionLayerWeights& L = model_->layers[static_cast<size_t>(l)];

        launch_qwen3vl_layernorm(d_hidden_, static_cast<const half*>(L.ln1_w.data),
                                 static_cast<const half*>(L.ln1_b.data), d_normed_, n, H, kLayerNormEps,
                                 stream);
        gemm_nt(d_normed_, static_cast<const half*>(L.wq.data), d_qkv_, n, 3 * H, H, 1.0f, 0.0f, stream);
        launch_qwen3vl_add_bias(d_qkv_, static_cast<const half*>(L.bq.data), n, 3 * H, stream);
        launch_qwen3vl_split_qkv_rope(d_qkv_, d_row_, d_col_, d_q_, d_k_, d_v_, n, c.num_heads, c.head_dim,
                                      kVisionRopeTheta, stream);
        if (!attention(n, stream))
            return false;
        launch_qwen3vl_merge_heads(d_attn_, d_normed_, n, c.num_heads, c.head_dim, stream);
        gemm_nt(d_normed_, static_cast<const half*>(L.wo.data), d_proj_, n, H, H, 1.0f, 0.0f, stream);
        launch_qwen3vl_add_bias(d_proj_, static_cast<const half*>(L.bo.data), n, H, stream);
        launch_qwen3vl_residual_add(d_hidden_, d_proj_, static_cast<int64_t>(n) * H, stream);

        launch_qwen3vl_layernorm(d_hidden_, static_cast<const half*>(L.ln2_w.data),
                                 static_cast<const half*>(L.ln2_b.data), d_normed_, n, H, kLayerNormEps,
                                 stream);
        gemm_nt(d_normed_, static_cast<const half*>(L.ffn_up_w.data), d_ffn_, n, c.intermediate_size, H, 1.0f,
                0.0f, stream);
        launch_qwen3vl_add_bias(d_ffn_, static_cast<const half*>(L.ffn_up_b.data), n, c.intermediate_size,
                                stream);
        launch_qwen3vl_gelu_tanh(d_ffn_, static_cast<int64_t>(n) * c.intermediate_size, stream);
        gemm_nt(d_ffn_, static_cast<const half*>(L.ffn_down_w.data), d_proj_, n, H, c.intermediate_size, 1.0f,
                0.0f, stream);
        launch_qwen3vl_add_bias(d_proj_, static_cast<const half*>(L.ffn_down_b.data), n, H, stream);
        launch_qwen3vl_residual_add(d_hidden_, d_proj_, static_cast<int64_t>(n) * H, stream);

        // DeepStack taps read the block's OUTPUT, and `deepstack_indexes` is
        // sorted, so a running cursor is enough.
        if (next_tap < c.deepstack_indexes.size() && c.deepstack_indexes[next_tap] == l) {
            if (!d_deepstack_out.empty() && !run_merger(model_->deepstack_mergers[next_tap], d_hidden_, n,
                                                        d_deepstack_out[next_tap], stream))
                return false;
            ++next_tap;
        }
    }

    return run_merger(model_->merger, d_hidden_, n, d_out, stream);
}

}  // namespace imp
