#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "graph/executor.h"
#include "memory/kv_cache.h"
#include "compute/gemm.h"
#include "test_model_builder.h"

#include <vector>
#include <cmath>

namespace imp {
namespace {

using test::DenseTestModel;
using test::verify_logits_finite;
using test::read_logits;

// ---------------------------------------------------------------------------
// Helper: run prefill and return logits
// ---------------------------------------------------------------------------
static Tensor run_prefill(GraphExecutor& executor, KVCache& cache,
                          const std::vector<int32_t>& tokens,
                          int max_blocks_per_seq = 1) {
    int n = static_cast<int>(tokens.size());
    std::vector<int> positions(n);
    for (int i = 0; i < n; i++) positions[i] = i;

    int32_t* d_tokens;
    int* d_positions;
    cudaMalloc(&d_tokens, n * sizeof(int32_t));
    cudaMalloc(&d_positions, n * sizeof(int));
    cudaMemcpy(d_tokens, tokens.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> h_bt(max_blocks_per_seq);
    for (int i = 0; i < max_blocks_per_seq; i++) h_bt[i] = i;
    int* d_bt;
    cudaMalloc(&d_bt, max_blocks_per_seq * sizeof(int));
    cudaMemcpy(d_bt, h_bt.data(), max_blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice);

    int h_ctx = n;
    int* d_ctx;
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_ctx, &h_ctx, sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.kv_cache = &cache;
    state.block_tables = d_bt;
    state.max_blocks_per_seq = max_blocks_per_seq;
    state.max_context_len = n;
    state.context_lens = d_ctx;

    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    cudaFree(d_tokens);
    cudaFree(d_positions);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    return logits;
}

// ---------------------------------------------------------------------------
// Helper: run a single decode step and return logits
// ---------------------------------------------------------------------------
static Tensor run_decode(GraphExecutor& executor, KVCache& cache,
                         int32_t token, int position, int context_len,
                         int max_blocks_per_seq = 1) {
    int32_t* d_token;
    int* d_pos;
    cudaMalloc(&d_token, sizeof(int32_t));
    cudaMalloc(&d_pos, sizeof(int));
    cudaMemcpy(d_token, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, &position, sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> h_bt(max_blocks_per_seq);
    for (int i = 0; i < max_blocks_per_seq; i++) h_bt[i] = i;
    int* d_bt;
    cudaMalloc(&d_bt, max_blocks_per_seq * sizeof(int));
    cudaMemcpy(d_bt, h_bt.data(), max_blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice);

    int* d_ctx;
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_ctx, &context_len, sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_token;
    state.positions = d_pos;
    state.n_tokens = 1;
    state.is_prefill = false;
    state.n_sequences = 1;
    state.kv_cache = &cache;
    state.block_tables = d_bt;
    state.max_blocks_per_seq = max_blocks_per_seq;
    state.max_context_len = context_len;
    state.context_lens = d_ctx;

    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    cudaFree(d_token);
    cudaFree(d_pos);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    return logits;
}

// ---------------------------------------------------------------------------
// Test 1: Prefill with 1-layer model (original test)
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, SyntheticModelForwardLogits) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 1, 4, 4, 64);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    KVCache cache(1, 4, 32, DType::FP16, 8);
    Tensor logits = run_prefill(executor, cache, {1, 42, 100, 200});

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_NE(logits.data, nullptr);
    ASSERT_GE(logits.numel(), 256);
    verify_logits_finite(logits, 256);

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 2: Decode after prefill (original test)
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, SyntheticModelDecodeAfterPrefill) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 1, 4, 4, 64);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    KVCache cache(1, 4, 32, DType::FP16, 8);

    // Prefill 3 tokens
    run_prefill(executor, cache, {1, 2, 3});
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Decode 1 token at position 3
    Tensor logits = run_decode(executor, cache, 50, 3, 4);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    verify_logits_finite(logits, 256);

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 3: Multi-layer prefill (4 layers — catches residual accumulation bugs)
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, MultiLayerPrefill) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 4, 4, 4, 64);

    GraphExecutor executor;
    gemm_init();
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    KVCache cache(4, 4, 32, DType::FP16, 8);
    Tensor logits = run_prefill(executor, cache, {1, 2, 3, 4, 5, 6, 7, 8});

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_NE(logits.data, nullptr);
    verify_logits_finite(logits, 256);

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 4: GQA forward pass (n_heads=8, n_kv_heads=2 — catches KV broadcast bugs)
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, GQAForwardPass) {
    SKIP_IF_NO_CUDA();

    // d_model=256, n_heads=8, n_kv_heads=2 → head_dim=32, 4:1 GQA ratio
    auto tm = DenseTestModel::create(256, 512, 256, 2, 8, 2, 64);

    GraphExecutor executor;
    gemm_init();
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    // n_kv_heads=2 for KVCache
    KVCache cache(2, 2, 32, DType::FP16, 8);

    // Prefill
    run_prefill(executor, cache, {1, 2, 3, 4});
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Decode
    Tensor logits = run_decode(executor, cache, 50, 4, 5);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    verify_logits_finite(logits, 256);

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 5: Multi-step decode (prefill 4, then decode 4 sequentially)
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, MultiStepDecode) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 1, 4, 4, 64);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    KVCache cache(1, 4, 32, DType::FP16, 8);

    // Prefill 4 tokens
    run_prefill(executor, cache, {1, 2, 3, 4});
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Decode 4 tokens sequentially
    int32_t next_token = 50;
    for (int step = 0; step < 4; step++) {
        int position = 4 + step;
        int context_len = 5 + step;
        Tensor logits = run_decode(executor, cache, next_token, position, context_len);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess)
            << "Decode step " << step << " failed";
        verify_logits_finite(logits, 256);

        // Use argmax for next token (deterministic)
        auto h = read_logits(logits, 256);
        next_token = static_cast<int32_t>(
            std::max_element(h.begin(), h.end()) - h.begin());
    }

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 6: Deterministic logits (same input → same output)
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, DeterministicLogits) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 1, 4, 4, 64);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    std::vector<int32_t> tokens = {1, 42, 100};

    // Run 1
    KVCache cache1(1, 4, 32, DType::FP16, 8);
    Tensor logits1 = run_prefill(executor, cache1, tokens);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    auto h1 = read_logits(logits1, 256);

    // Run 2 (fresh KV cache)
    KVCache cache2(1, 4, 32, DType::FP16, 8);
    Tensor logits2 = run_prefill(executor, cache2, tokens);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    auto h2 = read_logits(logits2, 256);

    // Compare bitwise
    for (int i = 0; i < 256; i++) {
        EXPECT_EQ(h1[i], h2[i]) << "Logit mismatch at index " << i
            << " (run1=" << h1[i] << ", run2=" << h2[i] << ")";
    }

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 7: Long sequence prefill (32 tokens, near max_seq_len=64)
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, LongSequencePrefill) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 1, 4, 4, 64);

    GraphExecutor executor;
    ASSERT_TRUE(executor.init(*tm.model, DType::FP16, false, 1, 64));
    ASSERT_TRUE(executor.allocate_workspaces());

    // 32 tokens need 2 KV blocks (block_size=16)
    KVCache cache(1, 4, 32, DType::FP16, 8);

    std::vector<int32_t> tokens(32);
    for (int i = 0; i < 32; i++) tokens[i] = (i + 1) % 256;

    Tensor logits = run_prefill(executor, cache, tokens, /*max_blocks_per_seq=*/2);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_NE(logits.data, nullptr);
    verify_logits_finite(logits, 256);

    tm.cleanup();
}

} // anonymous namespace
} // namespace imp
