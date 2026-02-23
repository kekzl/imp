#include <gtest/gtest.h>
#include "graph/executor.h"
#include "model/model.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <random>
#include <memory>

namespace imp {
namespace {

// ============================================================================
// Helper: skip if no CUDA device
// ============================================================================
static bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_CUDA()                                                     \
    do {                                                                       \
        if (!HasCudaDevice()) {                                                \
            GTEST_SKIP() << "No CUDA device available";                        \
        }                                                                      \
    } while (0)

// ============================================================================
// Helper: create a device tensor with random FP16 weights
// ============================================================================
static Tensor make_random_weight(int64_t rows, int64_t cols,
                                  std::mt19937& rng, float scale = 0.02f) {
    std::normal_distribution<float> dist(0.0f, scale);
    int64_t n = rows * cols;
    std::vector<half> h_data(n);
    for (int64_t i = 0; i < n; i++) {
        h_data[i] = __float2half(dist(rng));
    }

    Tensor t;
    t.dtype = DType::FP16;
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

// Helper: create a 1-D norm weight (all ones initially)
static Tensor make_norm_weight(int64_t dim) {
    std::vector<half> h_data(dim);
    for (int64_t i = 0; i < dim; i++) {
        h_data[i] = __float2half(1.0f);
    }

    Tensor t;
    t.dtype = DType::FP16;
    t.ndim = 1;
    t.shape[0] = dim;
    t.compute_strides();
    t.on_device = true;

    size_t bytes = dim * sizeof(half);
    cudaMalloc(&t.data, bytes);
    cudaMemcpy(t.data, h_data.data(), bytes, cudaMemcpyHostToDevice);
    return t;
}

// Helper: free a device tensor
static void free_tensor(Tensor& t) {
    if (t.data && t.on_device) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// ============================================================================
// Build a minimal MoE model for testing
// ============================================================================
struct MoETestModel {
    std::shared_ptr<Model> model;
    std::vector<Tensor> all_tensors;  // for cleanup

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

        // Token embedding [vocab_size, d_model]
        result.model->tok_emb_ = make_random_weight(vocab_size, d_model, rng, weight_scale);
        result.all_tensors.push_back(result.model->tok_emb_);

        // Output norm [d_model]
        result.model->out_norm_ = make_norm_weight(d_model);
        result.all_tensors.push_back(result.model->out_norm_);

        // Output projection [vocab_size, d_model]
        result.model->out_proj_ = make_random_weight(vocab_size, d_model, rng, weight_scale);
        result.all_tensors.push_back(result.model->out_proj_);

        int head_dim = d_model / n_heads;

        result.model->layers_.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            auto& ly = result.model->layers_[i];

            // Attention weights
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

            // FFN norm
            ly.ffn_norm = make_norm_weight(d_model);
            result.all_tensors.push_back(ly.ffn_norm);

            // MoE gate: [n_experts, d_model]
            ly.moe_gate = make_random_weight(n_experts, d_model, rng, weight_scale);
            result.all_tensors.push_back(ly.moe_gate);

            // Per-expert FFN weights
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

        return result;
    }

    ~MoETestModel() = default;

    void cleanup() {
        for (auto& t : all_tensors) {
            free_tensor(t);
        }
        all_tensors.clear();
    }
};

// Build a minimal dense model (no MoE)
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

        return result;
    }

    void cleanup() {
        for (auto& t : all_tensors) {
            free_tensor(t);
        }
        all_tensors.clear();
    }
};

// ============================================================================
// Test 1: MoE executor initializes successfully
// ============================================================================
TEST(MoEExecutorTest, InitSucceeds) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    tm.cleanup();
}

// ============================================================================
// Test 2: MoE forward produces valid output (no NaN/Inf)
// ============================================================================
TEST(MoEExecutorTest, ForwardProducesValidOutput) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    // Create input tokens
    const int n_tokens = 4;
    std::vector<int32_t> h_tokens = {1, 5, 10, 20};
    std::vector<int> h_positions = {0, 1, 2, 3};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.temperature = 0.0f;  // greedy

    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    // Check logits shape (prefill: only last token projected)
    ASSERT_EQ(logits.ndim, 2);
    ASSERT_EQ(logits.shape[0], 1);
    ASSERT_EQ(logits.shape[1], 256);  // vocab_size

    // Read back logits and check for NaN/Inf (logits are FP32)
    int64_t numel = logits.numel();
    std::vector<float> h_logits(numel);
    cudaMemcpy(h_logits.data(), logits.data, numel * sizeof(float), cudaMemcpyDeviceToHost);

    int nan_count = 0;
    int inf_count = 0;
    for (int64_t i = 0; i < numel; i++) {
        float v = h_logits[i];
        if (std::isnan(v)) nan_count++;
        if (std::isinf(v)) inf_count++;
    }
    EXPECT_EQ(nan_count, 0) << "Found NaN values in logits";
    EXPECT_EQ(inf_count, 0) << "Found Inf values in logits";

    cudaFree(d_tokens);
    cudaFree(d_positions);
    tm.cleanup();
}

// ============================================================================
// Test 3: MoE forward samples a token (full pipeline)
// ============================================================================
TEST(MoEExecutorTest, ForwardSamplesToken) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    const int n_tokens = 3;
    std::vector<int32_t> h_tokens = {1, 2, 3};
    std::vector<int> h_positions = {0, 1, 2};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.temperature = 0.0f;  // greedy

    int32_t token = executor.forward(state, nullptr);
    cudaDeviceSynchronize();

    // Token should be a valid vocab index
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 256);

    cudaFree(d_tokens);
    cudaFree(d_positions);
    tm.cleanup();
}

// ============================================================================
// Test 4: MoE forward is deterministic with same input
// ============================================================================
TEST(MoEExecutorTest, Deterministic) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    const int n_tokens = 4;
    std::vector<int32_t> h_tokens = {5, 10, 15, 20};
    std::vector<int> h_positions = {0, 1, 2, 3};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.temperature = 0.0f;

    int32_t token1 = executor.forward(state, nullptr);
    cudaDeviceSynchronize();

    int32_t token2 = executor.forward(state, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(token1, token2) << "MoE forward should be deterministic with greedy sampling";

    cudaFree(d_tokens);
    cudaFree(d_positions);
    tm.cleanup();
}

// ============================================================================
// Test 5: MoE with 2 layers
// ============================================================================
TEST(MoEExecutorTest, MultiLayer) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/2, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    const int n_tokens = 4;
    std::vector<int32_t> h_tokens = {1, 2, 3, 4};
    std::vector<int> h_positions = {0, 1, 2, 3};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    cudaDeviceSynchronize();

    EXPECT_GE(token, 0);
    EXPECT_LT(token, 256);

    cudaFree(d_tokens);
    cudaFree(d_positions);
    tm.cleanup();
}

// ============================================================================
// Test 6: MoE with 8 experts (Mixtral-like)
// ============================================================================
TEST(MoEExecutorTest, EightExperts) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/8, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    const int n_tokens = 8;
    std::vector<int32_t> h_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> h_positions = {0, 1, 2, 3, 4, 5, 6, 7};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    cudaDeviceSynchronize();

    EXPECT_GE(token, 0);
    EXPECT_LT(token, 256);

    cudaFree(d_tokens);
    cudaFree(d_positions);
    tm.cleanup();
}

// ============================================================================
// Test 7: Single token input (edge case)
// ============================================================================
TEST(MoEExecutorTest, SingleToken) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    int32_t h_token = 42;
    int h_position = 0;

    int32_t* d_token = nullptr;
    int* d_position = nullptr;
    cudaMalloc(&d_token, sizeof(int32_t));
    cudaMalloc(&d_position, sizeof(int));
    cudaMemcpy(d_token, &h_token, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_position, &h_position, sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_token;
    state.positions = d_position;
    state.n_tokens = 1;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.temperature = 0.0f;

    int32_t token = executor.forward(state, nullptr);
    cudaDeviceSynchronize();

    EXPECT_GE(token, 0);
    EXPECT_LT(token, 256);

    cudaFree(d_token);
    cudaFree(d_position);
    tm.cleanup();
}

// ============================================================================
// Test 8: MoE vs Dense produce different logits (different FFN path)
// ============================================================================
TEST(MoEExecutorTest, MoEVsDenseDiffer) {
    SKIP_IF_NO_CUDA();

    // Dense model -- use seed 100 with larger weight scale
    auto dense = DenseTestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*max_seq_len=*/512, /*seed=*/100, /*weight_scale=*/0.5f);

    // MoE model -- use seed 200 with larger weight scale
    auto moe = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128,
        /*max_seq_len=*/512, /*seed=*/200, /*weight_scale=*/0.5f);

    GraphExecutor dense_exec, moe_exec;
    ASSERT_TRUE(dense_exec.init(*dense.model, DType::FP16, false));
    ASSERT_TRUE(moe_exec.init(*moe.model, DType::FP16, false));

    const int n_tokens = 4;
    std::vector<int32_t> h_tokens = {1, 2, 3, 4};
    std::vector<int> h_positions = {0, 1, 2, 3};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.temperature = 0.0f;

    Tensor dense_logits, moe_logits;
    dense_exec.forward_logits(state, dense_logits, nullptr);
    moe_exec.forward_logits(state, moe_logits, nullptr);
    cudaDeviceSynchronize();

    // Both should produce valid logits (prefill: 1 token projected)
    ASSERT_EQ(dense_logits.shape[0], 1);
    ASSERT_EQ(moe_logits.shape[0], 1);

    // Read back logits (last token only) - logits are FP32
    int vocab = 256;
    int total = 1 * vocab;
    std::vector<float> h_dense(total), h_moe(total);

    cudaMemcpy(h_dense.data(), dense_logits.data,
               total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_moe.data(), moe_logits.data,
               total * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that both outputs are non-zero (sanity check)
    float dense_sum = 0.0f, moe_sum = 0.0f;
    for (int i = 0; i < total; i++) {
        dense_sum += std::abs(h_dense[i]);
        moe_sum += std::abs(h_moe[i]);
    }
    EXPECT_GT(dense_sum, 0.0f) << "Dense logits are all zero";
    EXPECT_GT(moe_sum, 0.0f) << "MoE logits are all zero";

    // Count how many logits differ between dense and MoE
    int diff_count = 0;
    for (int i = 0; i < total; i++) {
        float dv = h_dense[i];
        float mv = h_moe[i];
        if (std::abs(dv - mv) > 0.01f) diff_count++;
    }
    // With different random seeds, the embedding + projection weights differ,
    // so the logits must differ. Even a small fraction differing proves the
    // paths are distinct.
    EXPECT_GT(diff_count, total / 10)
        << "MoE and Dense should produce substantially different logits"
        << " (dense_sum=" << dense_sum << ", moe_sum=" << moe_sum
        << ", diff_count=" << diff_count << "/" << total << ")";

    cudaFree(d_tokens);
    cudaFree(d_positions);
    dense.cleanup();
    moe.cleanup();
}

// ============================================================================
// Test 9: MoE forward_logits output shape is correct
// ============================================================================
TEST(MoEExecutorTest, LogitsShape) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    for (int n_tokens : {1, 2, 4, 8, 16}) {
        std::vector<int32_t> h_tokens(n_tokens);
        std::vector<int> h_positions(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            h_tokens[i] = i + 1;
            h_positions[i] = i;
        }

        int32_t* d_tokens = nullptr;
        int* d_positions = nullptr;
        cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
        cudaMalloc(&d_positions, n_tokens * sizeof(int));
        cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

        InferenceState state;
        state.token_ids = d_tokens;
        state.positions = d_positions;
        state.n_tokens = n_tokens;
        state.is_prefill = true;
        state.n_sequences = 1;

        Tensor logits;
        executor.forward_logits(state, logits, nullptr);
        cudaDeviceSynchronize();

        EXPECT_EQ(logits.ndim, 2) << "n_tokens=" << n_tokens;
        EXPECT_EQ(logits.shape[0], 1) << "n_tokens=" << n_tokens;  // prefill: last token only
        EXPECT_EQ(logits.shape[1], 256) << "n_tokens=" << n_tokens;

        cudaFree(d_tokens);
        cudaFree(d_positions);
    }

    tm.cleanup();
}

// ============================================================================
// Test 10: Mixed model - some layers MoE, some dense (DeepSeek-like)
// ============================================================================
TEST(MoEExecutorTest, MixedMoEDense) {
    SKIP_IF_NO_CUDA();

    // Create a 2-layer model where:
    //   Layer 0: Dense FFN (no expert weights)
    //   Layer 1: MoE FFN (has expert weights)
    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/2, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    // Make layer 0 a dense layer by clearing expert weights and adding dense FFN weights
    auto& ly0 = tm.model->layers_[0];
    // Free expert weights for layer 0
    for (auto& t : ly0.expert_w_gate) free_tensor(t);
    for (auto& t : ly0.expert_w_up) free_tensor(t);
    for (auto& t : ly0.expert_w_down) free_tensor(t);
    ly0.expert_w_gate.clear();
    ly0.expert_w_up.clear();
    ly0.expert_w_down.clear();
    free_tensor(ly0.moe_gate);
    ly0.moe_gate = Tensor();

    // Add dense FFN weights for layer 0
    std::mt19937 rng(123);
    ly0.w_gate = make_random_weight(128, 64, rng);
    ly0.w_up = make_random_weight(128, 64, rng);
    ly0.w_down = make_random_weight(64, 128, rng);
    tm.all_tensors.push_back(ly0.w_gate);
    tm.all_tensors.push_back(ly0.w_up);
    tm.all_tensors.push_back(ly0.w_down);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false));

    const int n_tokens = 4;
    std::vector<int32_t> h_tokens = {1, 2, 3, 4};
    std::vector<int> h_positions = {0, 1, 2, 3};

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, n_tokens * sizeof(int));
    cudaMemcpy(d_tokens, h_tokens.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_tokens;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.temperature = 0.0f;

    // Should use dense FFN for layer 0 and MoE FFN for layer 1
    int32_t token = executor.forward(state, nullptr);
    cudaDeviceSynchronize();

    EXPECT_GE(token, 0);
    EXPECT_LT(token, 256);

    cudaFree(d_tokens);
    cudaFree(d_positions);
    tm.cleanup();
}

} // namespace
} // namespace imp
