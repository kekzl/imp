#pragma once

// Shared synthetic model builders for GPU tests.
// Used by test_forward_pass.cu, test_engine_integration.cu, test_moe_executor.cu.

#include "model/model.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <memory>

namespace imp {
namespace test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_CUDA()                                                     \
    do {                                                                       \
        if (!::imp::test::HasCudaDevice()) {                                   \
            GTEST_SKIP() << "No CUDA device available";                        \
        }                                                                      \
    } while (0)

inline Tensor make_random_weight(int64_t rows, int64_t cols,
                                  std::mt19937& rng, float scale = 0.02f) {
    std::normal_distribution<float> dist(0.0f, scale);
    int64_t n = rows * cols;
    std::vector<half> h_data(n);
    for (int64_t i = 0; i < n; i++) {
        h_data[i] = __float2half(dist(rng));
    }

    Tensor t;
    t.qtype = QType::F16;
    t.ndim = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.compute_strides();
    t.on_device = true;

    size_t bytes = n * sizeof(half);
    cudaMalloc(&t.data, bytes);
    cudaMemcpy(t.data, h_data.data(), bytes, cudaMemcpyHostToDevice);
    return t;
}

inline Tensor make_norm_weight(int64_t dim) {
    std::vector<half> h_data(dim, __float2half(1.0f));

    Tensor t;
    t.qtype = QType::F16;
    t.ndim = 1;
    t.shape[0] = dim;
    t.compute_strides();
    t.on_device = true;

    size_t bytes = dim * sizeof(half);
    cudaMalloc(&t.data, bytes);
    cudaMemcpy(t.data, h_data.data(), bytes, cudaMemcpyHostToDevice);
    return t;
}

inline void free_tensor(Tensor& t) {
    if (t.data && t.on_device) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Verify logits are finite (no NaN/Inf) and not all-zero
// ---------------------------------------------------------------------------
inline void verify_logits_finite(const Tensor& logits, int vocab_size) {
    std::vector<float> h_logits(vocab_size);
    if (logits.qtype == QType::F32) {
        cudaMemcpy(h_logits.data(), logits.data, vocab_size * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        std::vector<half> tmp(vocab_size);
        cudaMemcpy(tmp.data(), logits.data, vocab_size * sizeof(half),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < vocab_size; i++) h_logits[i] = __half2float(tmp[i]);
    }

    int nan_count = 0, inf_count = 0, zero_count = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (std::isnan(h_logits[i])) nan_count++;
        if (std::isinf(h_logits[i])) inf_count++;
        if (h_logits[i] == 0.0f) zero_count++;
    }

    EXPECT_EQ(nan_count, 0) << "Found " << nan_count << " NaN logits";
    EXPECT_EQ(inf_count, 0) << "Found " << inf_count << " Inf logits";
    EXPECT_LT(zero_count, vocab_size) << "All logits are zero — no signal";
}

// ---------------------------------------------------------------------------
// Read FP32 logits from device
// ---------------------------------------------------------------------------
inline std::vector<float> read_logits(const Tensor& logits, int count) {
    std::vector<float> h(count);
    if (logits.qtype == QType::F32) {
        cudaMemcpy(h.data(), logits.data, count * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        std::vector<half> tmp(count);
        cudaMemcpy(tmp.data(), logits.data, count * sizeof(half),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < count; i++) h[i] = __half2float(tmp[i]);
    }
    return h;
}

// ---------------------------------------------------------------------------
// Dense model builder
// ---------------------------------------------------------------------------
struct DenseTestModel {
    std::shared_ptr<Model> model;
    std::vector<Tensor> all_tensors;

    static DenseTestModel create(int d_model, int d_ff, int vocab_size,
                                  int n_layers, int n_heads, int n_kv_heads,
                                  int max_seq_len = 512, int seed = 42,
                                  float weight_scale = 0.02f) {
        DenseTestModel result;
        result.model = std::make_shared<Model>();
        auto& cfg = result.model->config_;
        cfg.arch = ModelArch::LLAMA;
        cfg.n_layers = n_layers;
        cfg.n_heads = n_heads;
        cfg.n_kv_heads = n_kv_heads;
        cfg.d_model = d_model;
        cfg.d_ff = d_ff;
        cfg.vocab_size = vocab_size;
        cfg.max_seq_len = max_seq_len;
        cfg.rope_theta = 10000.0f;
        cfg.rms_norm_eps = 1e-5f;
        cfg.n_experts = 0;
        cfg.n_experts_active = 0;
        cfg.expert_d_ff = 0;

        std::mt19937 rng(seed);
        int head_dim = d_model / n_heads;

        result.model->tok_emb_ = make_random_weight(vocab_size, d_model, rng, weight_scale);
        result.all_tensors.push_back(result.model->tok_emb_);

        result.model->out_norm_ = make_norm_weight(d_model);
        result.all_tensors.push_back(result.model->out_norm_);

        result.model->out_proj_ = make_random_weight(vocab_size, d_model, rng, weight_scale);
        result.all_tensors.push_back(result.model->out_proj_);

        result.model->layers_.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            auto& ly = result.model->layers_[i];
            ly.wq = make_random_weight(n_heads * head_dim, d_model, rng, weight_scale);
            ly.wk = make_random_weight(n_kv_heads * head_dim, d_model, rng, weight_scale);
            ly.wv = make_random_weight(n_kv_heads * head_dim, d_model, rng, weight_scale);
            ly.wo = make_random_weight(d_model, n_heads * head_dim, rng, weight_scale);
            ly.attn_norm = make_norm_weight(d_model);
            ly.ffn_norm = make_norm_weight(d_model);
            ly.w_gate = make_random_weight(d_ff, d_model, rng, weight_scale);
            ly.w_up = make_random_weight(d_ff, d_model, rng, weight_scale);
            ly.w_down = make_random_weight(d_model, d_ff, rng, weight_scale);

            result.all_tensors.push_back(ly.wq);
            result.all_tensors.push_back(ly.wk);
            result.all_tensors.push_back(ly.wv);
            result.all_tensors.push_back(ly.wo);
            result.all_tensors.push_back(ly.attn_norm);
            result.all_tensors.push_back(ly.ffn_norm);
            result.all_tensors.push_back(ly.w_gate);
            result.all_tensors.push_back(ly.w_up);
            result.all_tensors.push_back(ly.w_down);
        }

        result.model->gpu_weights_ready_ = true;
        return result;
    }

    void cleanup() {
        for (auto& t : all_tensors) free_tensor(t);
        all_tensors.clear();
    }
};

// ---------------------------------------------------------------------------
// MoE model builder
// ---------------------------------------------------------------------------
struct MoETestModel {
    std::shared_ptr<Model> model;
    std::vector<Tensor> all_tensors;

    static MoETestModel create(int d_model, int d_ff, int vocab_size,
                                int n_layers, int n_heads, int n_kv_heads,
                                int n_experts, int n_experts_active,
                                int expert_d_ff, int max_seq_len = 512,
                                int seed = 42, float weight_scale = 0.02f) {
        MoETestModel result;
        result.model = std::make_shared<Model>();
        auto& cfg = result.model->config_;
        cfg.arch = ModelArch::MIXTRAL;
        cfg.n_layers = n_layers;
        cfg.n_heads = n_heads;
        cfg.n_kv_heads = n_kv_heads;
        cfg.d_model = d_model;
        cfg.d_ff = d_ff;
        cfg.vocab_size = vocab_size;
        cfg.max_seq_len = max_seq_len;
        cfg.rope_theta = 10000.0f;
        cfg.rms_norm_eps = 1e-5f;
        cfg.n_experts = n_experts;
        cfg.n_experts_active = n_experts_active;
        cfg.expert_d_ff = expert_d_ff;

        std::mt19937 rng(seed);

        result.model->tok_emb_ = make_random_weight(vocab_size, d_model, rng, weight_scale);
        result.all_tensors.push_back(result.model->tok_emb_);

        result.model->out_norm_ = make_norm_weight(d_model);
        result.all_tensors.push_back(result.model->out_norm_);

        result.model->out_proj_ = make_random_weight(vocab_size, d_model, rng, weight_scale);
        result.all_tensors.push_back(result.model->out_proj_);

        int head_dim = d_model / n_heads;

        result.model->layers_.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            auto& ly = result.model->layers_[i];

            ly.wq = make_random_weight(n_heads * head_dim, d_model, rng, weight_scale);
            ly.wk = make_random_weight(n_kv_heads * head_dim, d_model, rng, weight_scale);
            ly.wv = make_random_weight(n_kv_heads * head_dim, d_model, rng, weight_scale);
            ly.wo = make_random_weight(d_model, n_heads * head_dim, rng, weight_scale);
            ly.attn_norm = make_norm_weight(d_model);

            result.all_tensors.push_back(ly.wq);
            result.all_tensors.push_back(ly.wk);
            result.all_tensors.push_back(ly.wv);
            result.all_tensors.push_back(ly.wo);
            result.all_tensors.push_back(ly.attn_norm);

            ly.ffn_norm = make_norm_weight(d_model);
            result.all_tensors.push_back(ly.ffn_norm);

            ly.moe_gate = make_random_weight(n_experts, d_model, rng, weight_scale);
            result.all_tensors.push_back(ly.moe_gate);

            ly.expert_w_gate.resize(n_experts);
            ly.expert_w_up.resize(n_experts);
            ly.expert_w_down.resize(n_experts);

            for (int e = 0; e < n_experts; e++) {
                ly.expert_w_gate[e] = make_random_weight(expert_d_ff, d_model, rng, weight_scale);
                ly.expert_w_up[e] = make_random_weight(expert_d_ff, d_model, rng, weight_scale);
                ly.expert_w_down[e] = make_random_weight(d_model, expert_d_ff, rng, weight_scale);

                result.all_tensors.push_back(ly.expert_w_gate[e]);
                result.all_tensors.push_back(ly.expert_w_up[e]);
                result.all_tensors.push_back(ly.expert_w_down[e]);
            }
        }

        result.model->gpu_weights_ready_ = true;
        return result;
    }

    void cleanup() {
        for (auto& t : all_tensors) free_tensor(t);
        all_tensors.clear();
    }
};

// ---------------------------------------------------------------------------
// Q8_0 weight creation (for FP8/NVFP4 pre-dequant tests)
// Q8_0 format: 34 bytes per 32 elements = half(scale) + int8_t[32]
// cols must be divisible by 32.
// ---------------------------------------------------------------------------
inline Tensor make_q8_0_weight(int64_t rows, int64_t cols,
                                std::mt19937& rng, float scale = 0.5f) {
    assert(cols % 32 == 0);
    std::normal_distribution<float> dist(0.0f, scale);
    int64_t n_blocks_per_row = cols / 32;
    int64_t total_blocks = rows * n_blocks_per_row;
    size_t block_bytes = 34;  // 2 (fp16 scale) + 32 (int8 quants)
    size_t total_bytes = total_blocks * block_bytes;

    std::vector<uint8_t> data(total_bytes);
    for (int64_t b = 0; b < total_blocks; b++) {
        // Generate 32 random float values for this block
        float vals[32];
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            vals[j] = dist(rng);
            amax = std::max(amax, std::abs(vals[j]));
        }
        float d = amax / 127.0f;
        half d_fp16 = __float2half(d);

        uint8_t* block_ptr = data.data() + b * block_bytes;
        // Write scale (2 bytes, little-endian half)
        memcpy(block_ptr, &d_fp16, 2);
        // Write quantized int8 values
        int8_t* qs = reinterpret_cast<int8_t*>(block_ptr + 2);
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        for (int j = 0; j < 32; j++) {
            int v = static_cast<int>(roundf(vals[j] * id));
            qs[j] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
        }
    }

    Tensor t;
    t.qtype = QType::F16;  // logical dtype for shape computation
    t.ndim = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.compute_strides();
    t.on_device = true;

    cudaMalloc(&t.data, total_bytes);
    cudaMemcpy(t.data, data.data(), total_bytes, cudaMemcpyHostToDevice);
    return t;
}

// ---------------------------------------------------------------------------
// Dense model with Q8_0 weights (for testing FP8/NVFP4 pre-dequant paths)
// Embedding, norms, and output projection stay FP16 (like real models).
// Attention and FFN weights are Q8_0.
// ---------------------------------------------------------------------------
struct Q8DenseTestModel {
    std::shared_ptr<Model> model;
    std::vector<Tensor> all_tensors;

    static Q8DenseTestModel create(int d_model, int d_ff, int vocab_size,
                                    int n_layers, int n_heads, int n_kv_heads,
                                    int max_seq_len = 512, int seed = 42) {
        Q8DenseTestModel result;
        result.model = std::make_shared<Model>();
        auto& cfg = result.model->config_;
        cfg.arch = ModelArch::LLAMA;
        cfg.n_layers = n_layers;
        cfg.n_heads = n_heads;
        cfg.n_kv_heads = n_kv_heads;
        cfg.d_model = d_model;
        cfg.d_ff = d_ff;
        cfg.vocab_size = vocab_size;
        cfg.max_seq_len = max_seq_len;
        cfg.rope_theta = 10000.0f;
        cfg.rms_norm_eps = 1e-5f;
        cfg.n_experts = 0;
        cfg.n_experts_active = 0;
        cfg.expert_d_ff = 0;

        std::mt19937 rng(seed);
        int head_dim = d_model / n_heads;

        // Embedding and output stay FP16 (not quantized in real models either)
        result.model->tok_emb_ = make_random_weight(vocab_size, d_model, rng);
        result.all_tensors.push_back(result.model->tok_emb_);

        result.model->out_norm_ = make_norm_weight(d_model);
        result.all_tensors.push_back(result.model->out_norm_);

        result.model->out_proj_ = make_random_weight(vocab_size, d_model, rng);
        result.model->out_proj_.qtype = QType::NONE;
        result.all_tensors.push_back(result.model->out_proj_);

        result.model->layers_.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            auto& ly = result.model->layers_[i];

            // Attention weights: Q8_0
            ly.wq = make_q8_0_weight(n_heads * head_dim, d_model, rng);
            ly.wk = make_q8_0_weight(n_kv_heads * head_dim, d_model, rng);
            ly.wv = make_q8_0_weight(n_kv_heads * head_dim, d_model, rng);
            ly.wo = make_q8_0_weight(d_model, n_heads * head_dim, rng);
            ly.wq.qtype = QType::Q8_0;
            ly.wk.qtype = QType::Q8_0;
            ly.wv.qtype = QType::Q8_0;
            ly.wo.qtype = QType::Q8_0;

            // Norms stay FP16
            ly.attn_norm = make_norm_weight(d_model);
            ly.ffn_norm = make_norm_weight(d_model);

            // FFN weights: Q8_0
            ly.w_gate = make_q8_0_weight(d_ff, d_model, rng);
            ly.w_up = make_q8_0_weight(d_ff, d_model, rng);
            ly.w_down = make_q8_0_weight(d_model, d_ff, rng);
            ly.w_gate.qtype = QType::Q8_0;
            ly.w_up.qtype = QType::Q8_0;
            ly.w_down.qtype = QType::Q8_0;

            result.all_tensors.push_back(ly.wq);
            result.all_tensors.push_back(ly.wk);
            result.all_tensors.push_back(ly.wv);
            result.all_tensors.push_back(ly.wo);
            result.all_tensors.push_back(ly.attn_norm);
            result.all_tensors.push_back(ly.ffn_norm);
            result.all_tensors.push_back(ly.w_gate);
            result.all_tensors.push_back(ly.w_up);
            result.all_tensors.push_back(ly.w_down);
        }

        result.model->gpu_weights_ready_ = true;
        return result;
    }

    void cleanup() {
        for (auto& t : all_tensors) free_tensor(t);
        all_tensors.clear();
    }
};

} // namespace test
} // namespace imp
