#include <gtest/gtest.h>
#include "exec/executor.h"
#include "runtime/vram_budget.h"  // VRAMBudget: executor.h forward-declares it
#include "model/model.h"
#include "core/tensor.h"
#include "compute/gemm.h"
#include "test_model_builder.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <memory>
#include <cstring>
#include <random>

namespace imp {
namespace {

using test::DenseTestModel;
using test::free_tensor;
using test::make_random_weight;
using test::MoETestModel;

static void init_executor_moe(GraphExecutor& executor, Model& model) {
    ASSERT_TRUE(executor.init(model, QType::F16, false));
    gemm_init();
    ASSERT_TRUE(executor.allocate_workspaces(false));
    VRAMBudget budget;
    budget.strategy = VRAMBudget::FP16_ONLY;
    executor.pre_dequant_weights(nullptr, budget);
    cudaDeviceSynchronize();
}

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
    ASSERT_TRUE(executor.init(*tm.model, QType::F16, false));

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
    init_executor_moe(executor, *tm.model);

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
        if (std::isnan(v))
            nan_count++;
        if (std::isinf(v))
            inf_count++;
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
    init_executor_moe(executor, *tm.model);

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
    init_executor_moe(executor, *tm.model);

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

// Token-equality hides logit drift below the argmax margin. MoE grouped-GEMM expert-scatter
// float atomics cause nondeterminism (NVFP4 greedy A/B flip on Qwen3-30B-A3B, ModelProfile D1);
// measures per-logit drift over K forwards and bounds it; genuine case is model-level, gated e2e.
TEST(MoEExecutorTest, LogitDriftBounded) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/2, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/8, /*n_experts_active=*/2, /*expert_d_ff=*/128);

    GraphExecutor executor;
    init_executor_moe(executor, *tm.model);

    const int n_tokens = 4;
    std::vector<int32_t> h_tokens = {3, 7, 11, 23};
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

    const int K = 5;
    std::vector<float> ref;
    double max_abs_drift = 0.0, ref_rms = 0.0;
    for (int k = 0; k < K; ++k) {
        Tensor logits;
        executor.forward_logits(state, logits, nullptr);
        cudaDeviceSynchronize();
        int64_t numel = logits.numel();
        std::vector<float> h(numel);
        cudaMemcpy(h.data(), logits.data, numel * sizeof(float), cudaMemcpyDeviceToHost);
        if (k == 0) {
            ref = h;
            double ss = 0.0;
            for (float v : ref)
                ss += (double)v * v;
            ref_rms = std::sqrt(ss / std::max<int64_t>(1, numel));
        } else {
            for (int64_t i = 0; i < numel; ++i)
                max_abs_drift = std::max(max_abs_drift, (double)std::fabs(h[i] - ref[i]));
        }
    }
    double rel_drift = ref_rms > 1e-9 ? max_abs_drift / ref_rms : max_abs_drift;
    printf("[moe logit drift] K=%d max_abs=%.3e rel(rms)=%.3e ref_rms=%.4f\n", K, max_abs_drift,
           rel_drift, ref_rms);
    // Documented epsilon: the synthetic FP16 MoE path is deterministic, so drift
    // must be negligible (1e-3 of the logit rms is generous and catches any
    // newly-introduced nondeterminism while tolerating benign fp reassociation).
    EXPECT_LT(rel_drift, 1e-3) << "MoE run-to-run logit drift exceeds the documented bound";

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
    init_executor_moe(executor, *tm.model);

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
    init_executor_moe(executor, *tm.model);

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
    init_executor_moe(executor, *tm.model);

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
    init_executor_moe(dense_exec, *dense.model);
    init_executor_moe(moe_exec, *moe.model);

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

    cudaMemcpy(h_dense.data(), dense_logits.data, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_moe.data(), moe_logits.data, total * sizeof(float), cudaMemcpyDeviceToHost);

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
        if (std::abs(dv - mv) > 0.01f)
            diff_count++;
    }
    // With different random seeds, the embedding + projection weights differ,
    // so the logits must differ. Even a small fraction differing proves the
    // paths are distinct.
    EXPECT_GT(diff_count, total / 10) << "MoE and Dense should produce substantially different logits"
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
    init_executor_moe(executor, *tm.model);

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
    for (auto& t : ly0.expert_w_gate)
        free_tensor(t);
    for (auto& t : ly0.expert_w_up)
        free_tensor(t);
    for (auto& t : ly0.expert_w_down)
        free_tensor(t);
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
    init_executor_moe(executor, *tm.model);

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

// #2447: host-resident Q8_0 experts + no dp4a scratch -> FP16 decode arm. It must read the staged slot
// pool, not the host tensor: forward 2 runs after the host bytes are zeroed, so a host read changes it.
namespace host_experts {

constexpr int kD = 256, kEff = 256, kNe = 4, kTopK = 2, kVocab = 256;
constexpr size_t kQ8Block = 34;  // fp16 scale + 32 int8

size_t q8_bytes(int64_t rows, int64_t cols) { return static_cast<size_t>(rows) * (cols / 32) * kQ8Block; }

// Pinned + mapped, so a device read of the host tensor is legal and shows up as wrong logits, not a fault.
Tensor host_q8_experts(int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bytes = kNe * q8_bytes(rows, cols);
    void* p = nullptr;
    EXPECT_EQ(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), cudaSuccess);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    auto* b = static_cast<uint8_t*>(p);
    for (size_t off = 0; off < bytes; off += kQ8Block) {
        const half s = __float2half(0.01f);
        std::memcpy(b + off, &s, sizeof(s));
        for (int j = 0; j < 32; ++j)
            b[off + 2 + j] = static_cast<uint8_t>(static_cast<int8_t>(std::clamp(dist(rng) * 64.0f, -127.0f, 127.0f)));
    }
    int64_t shape[3] = {kNe, rows, cols};
    return Tensor(p, QType::Q8_0, 3, shape, /*on_device=*/false);
}

void zero_scales(const Tensor& t) {
    auto* b = static_cast<uint8_t*>(t.data);
    const size_t bytes = kNe * q8_bytes(t.shape[1], t.shape[2]);
    for (size_t off = 0; off < bytes; off += kQ8Block)
        std::memset(b + off, 0, 2);
}

std::vector<float> decode_logits(GraphExecutor& ex) {
    int32_t* d_tok = nullptr;
    int* d_pos = nullptr;
    const int32_t tok = 42;
    const int pos = 0;
    EXPECT_EQ(cudaMalloc(&d_tok, sizeof(tok)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_pos, sizeof(pos)), cudaSuccess);
    cudaMemcpy(d_tok, &tok, sizeof(tok), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, &pos, sizeof(pos), cudaMemcpyHostToDevice);
    InferenceState st;
    st.token_ids = d_tok;
    st.positions = d_pos;
    st.n_tokens = 1;
    st.is_prefill = true;
    st.n_sequences = 1;
    st.temperature = 0.0f;
    Tensor logits;
    ex.forward_logits(st, logits, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> h(static_cast<size_t>(logits.numel()));
    cudaMemcpy(h.data(), logits.data, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_tok);
    cudaFree(d_pos);
    return h;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

void init_host_expert_executor(GraphExecutor& ex, Model& model) {
    init_executor_moe(ex, model);
    ex.drop_dp4a_scratch_for_test();
}

}  // namespace host_experts

TEST(MoEExecutorTest, HostExpertFp16DecodeReadsTheSlotPool) {
    SKIP_IF_NO_CUDA();
    using namespace host_experts;
    auto tm = MoETestModel::create(kD, /*d_ff=*/kEff, kVocab, /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
                                   kNe, kTopK, kEff);
    std::mt19937 rng(7);
    auto& ly = tm.model->layers_[0];
    ly.expert_gate_packed = host_q8_experts(kEff, kD, rng);
    ly.expert_up_packed = host_q8_experts(kEff, kD, rng);
    ly.expert_down_packed = host_q8_experts(kD, kEff, rng);

    std::vector<float> ref, after;
    {
        GraphExecutor ex;
        ASSERT_NO_FATAL_FAILURE(init_host_expert_executor(ex, *tm.model));
        ASSERT_GT(ex.expert_cache_slots_per_layer(), 0) << "no expert LRU cache: the host path is not reached";
        ref = decode_logits(ex);
        const int64_t misses = ex.expert_cache_misses();
        ASSERT_EQ(misses, 3 * kTopK) << "decode did not stage the routed experts into the slot pool";
        for (const Tensor* t : {&ly.expert_gate_packed, &ly.expert_up_packed, &ly.expert_down_packed})
            zero_scales(*t);
        after = decode_logits(ex);
        EXPECT_EQ(ex.expert_cache_misses(), misses) << "second decode reloaded experts from the host";
    }
    // Control: zeroed host experts, fresh cache. The zeroing must be visible, else forward 2 proves nothing.
    std::vector<float> zeroed;
    {
        GraphExecutor ex;
        ASSERT_NO_FATAL_FAILURE(init_host_expert_executor(ex, *tm.model));
        zeroed = decode_logits(ex);
    }
    ASSERT_EQ(ref.size(), static_cast<size_t>(kVocab));
    EXPECT_GT(max_abs_diff(ref, zeroed), 1e-2f) << "control: zeroed experts did not change the logits";
    EXPECT_LE(max_abs_diff(ref, after), 1e-6f) << "FP16 decode arm read the host tensor, not the slot pool";

    for (const Tensor* t : {&ly.expert_gate_packed, &ly.expert_up_packed, &ly.expert_down_packed})
        cudaFreeHost(t->data);
    tm.cleanup();
}

}  // namespace
}  // namespace imp
