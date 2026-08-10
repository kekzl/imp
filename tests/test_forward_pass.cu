#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "exec/executor.h"
#include "memory/kv_cache.h"
#include "compute/gemm.h"
#include "test_model_builder.h"

#include <vector>
#include <cmath>
#include <algorithm>

namespace imp {
namespace {

using test::DenseTestModel;
using test::read_logits;
using test::verify_logits_finite;

static void init_executor(GraphExecutor& executor, Model& model, int n_sequences = 1,
                           int max_seq_len = 64) {
    ASSERT_TRUE(executor.init(model, QType::F16, false, n_sequences, max_seq_len));
    ASSERT_TRUE(executor.allocate_workspaces());
    VRAMBudget budget;
    budget.strategy = VRAMBudget::FP16_ONLY;
    executor.pre_dequant_weights(nullptr, budget);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Helper: run prefill and return logits
// ---------------------------------------------------------------------------
static Tensor run_prefill(GraphExecutor& executor, KVCache& cache, const std::vector<int32_t>& tokens,
                          int max_blocks_per_seq = 1) {
    int n = static_cast<int>(tokens.size());
    std::vector<int> positions(n);
    for (int i = 0; i < n; i++)
        positions[i] = i;

    int32_t* d_tokens;
    int* d_positions;
    cudaMalloc(&d_tokens, n * sizeof(int32_t));
    cudaMalloc(&d_positions, n * sizeof(int));
    cudaMemcpy(d_tokens, tokens.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> h_bt(max_blocks_per_seq);
    for (int i = 0; i < max_blocks_per_seq; i++)
        h_bt[i] = i;
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
static Tensor run_decode(GraphExecutor& executor, KVCache& cache, int32_t token, int position,
                         int context_len, int max_blocks_per_seq = 1) {
    int32_t* d_token;
    int* d_pos;
    cudaMalloc(&d_token, sizeof(int32_t));
    cudaMalloc(&d_pos, sizeof(int));
    cudaMemcpy(d_token, &token, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, &position, sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> h_bt(max_blocks_per_seq);
    for (int i = 0; i < max_blocks_per_seq; i++)
        h_bt[i] = i;
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
    init_executor(executor, *tm.model);

    KVCache cache(1, 4, 32, QType::F16, 8);
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
    init_executor(executor, *tm.model);

    KVCache cache(1, 4, 32, QType::F16, 8);

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
    init_executor(executor, *tm.model);

    KVCache cache(4, 4, 32, QType::F16, 8);
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
    init_executor(executor, *tm.model);

    // n_kv_heads=2 for KVCache
    KVCache cache(2, 2, 32, QType::F16, 8);

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
    init_executor(executor, *tm.model);

    KVCache cache(1, 4, 32, QType::F16, 8);

    // Prefill 4 tokens
    run_prefill(executor, cache, {1, 2, 3, 4});
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Decode 4 tokens sequentially
    int32_t next_token = 50;
    for (int step = 0; step < 4; step++) {
        int position = 4 + step;
        int context_len = 5 + step;
        Tensor logits = run_decode(executor, cache, next_token, position, context_len);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Decode step " << step << " failed";
        verify_logits_finite(logits, 256);

        // Use argmax for next token (deterministic)
        auto h = read_logits(logits, 256);
        next_token = static_cast<int32_t>(std::max_element(h.begin(), h.end()) - h.begin());
    }

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Batch-composition invariance (#1314)
//
// A sequence's logits must not depend on which unrelated sequences share its
// batch. #1314 measured greedy output changing under 45 concurrent requests,
// and its escape analysis is class E1: "no test asserts batch invariance
// anywhere". The scheduler tests cover bookkeeping — table shapes, queue
// transitions, slot counts — and never compare a solo run's output with a
// batched one's.
//
// This is the logit-level oracle the issue asks for, and it is deliberately
// NOT a bit-equality assert: the same sequence in a wider batch legitimately
// takes different GEMM shapes, so the property is agreement within tolerance
// plus an identical argmax. Padding leakage and mask bugs — a row reading
// another sequence's KV, a context_len applied to the wrong slot — move logits
// far more than rounding does and die here.
// ---------------------------------------------------------------------------
namespace {

// Prefill one sequence into an explicit set of physical KV blocks.
void prefill_into_blocks(GraphExecutor& executor, KVCache& cache, const std::vector<int32_t>& tokens,
                         const std::vector<int>& blocks) {
    int n = static_cast<int>(tokens.size());
    std::vector<int> positions(n);
    for (int i = 0; i < n; i++)
        positions[i] = i;

    int32_t* d_tokens = nullptr;
    int* d_positions = nullptr;
    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_tokens, n * sizeof(int32_t));
    cudaMalloc(&d_positions, n * sizeof(int));
    cudaMalloc(&d_bt, blocks.size() * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_tokens, tokens.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx, &n, sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n;
    state.is_prefill = true;
    state.n_sequences = 1;
    state.kv_cache = &cache;
    state.block_tables = d_bt;
    state.max_blocks_per_seq = static_cast<int>(blocks.size());
    state.max_context_len = n;
    state.context_lens = d_ctx;

    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    cudaFree(d_tokens);
    cudaFree(d_positions);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

// One decode step for `n_seq` sequences at once. block_tables is the
// [n_seq, max_blocks_per_seq] row-major form the paged kernels index when
// n_sequences > 1. Returns the [n_seq, vocab] logits read back to host.
std::vector<float> decode_batch(GraphExecutor& executor, KVCache& cache, const std::vector<int32_t>& tokens,
                                const std::vector<int>& positions, const std::vector<int>& ctx_lens,
                                const std::vector<int>& block_tables, int max_blocks_per_seq, int vocab) {
    const int n_seq = static_cast<int>(tokens.size());
    int32_t* d_tokens = nullptr;
    int *d_pos = nullptr, *d_ctx = nullptr, *d_bt = nullptr;
    cudaMalloc(&d_tokens, n_seq * sizeof(int32_t));
    cudaMalloc(&d_pos, n_seq * sizeof(int));
    cudaMalloc(&d_ctx, n_seq * sizeof(int));
    cudaMalloc(&d_bt, block_tables.size() * sizeof(int));
    cudaMemcpy(d_tokens, tokens.data(), n_seq * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, positions.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx, ctx_lens.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, block_tables.data(), block_tables.size() * sizeof(int), cudaMemcpyHostToDevice);

    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_pos;
    state.n_tokens = n_seq;
    state.is_prefill = false;
    state.n_sequences = n_seq;
    state.kv_cache = &cache;
    state.block_tables = d_bt;
    state.max_blocks_per_seq = max_blocks_per_seq;
    state.max_context_len = *std::max_element(ctx_lens.begin(), ctx_lens.end());
    state.context_lens = d_ctx;

    Tensor logits;
    executor.forward_logits(state, logits, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> host(static_cast<size_t>(n_seq) * vocab);
    cudaMemcpy(host.data(), logits.data, host.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_tokens);
    cudaFree(d_pos);
    cudaFree(d_ctx);
    cudaFree(d_bt);
    return host;
}

}  // namespace

TEST(ForwardPassTest, DecodeLogitsInvariantToBatchComposition) {
    SKIP_IF_NO_CUDA();

    constexpr int kVocab = 256;
    constexpr int kMaxBlocks = 2;  // 2 blocks x 16 slots = 32 positions per sequence
    auto tm = DenseTestModel::create(128, 512, kVocab, 2, 4, 4, 64);

    // One executor for both arms: the question is whether the SAME engine
    // answers differently depending on who else is in the batch.
    GraphExecutor executor;
    init_executor(executor, *tm.model, /*n_sequences=*/3, /*max_seq_len=*/64);

    // The sequence under test, and two unrelated ones of DIFFERENT lengths —
    // equal lengths would hide any bug that keys off a per-row context_len.
    const std::vector<int32_t> seq_a = {1, 42, 100, 7, 33};
    const std::vector<int32_t> seq_b = {9, 9, 9};
    const std::vector<int32_t> seq_c = {200, 11, 55, 77, 88, 12, 3};
    // Same SHAPES as b/c, different content — the neighbours-changed arm.
    const std::vector<int32_t> seq_b2 = {123, 4, 250};
    const std::vector<int32_t> seq_c2 = {5, 5, 5, 5, 5, 5, 5};
    const int32_t next_a = 50;
    const int pos_a = static_cast<int>(seq_a.size());
    const int ctx_a = pos_a + 1;
    const std::vector<int> bt3 = {0, 1, 2, 3, 4, 5};  // [3, kMaxBlocks] row-major

    auto run_batched = [&](const std::vector<int32_t>& b, const std::vector<int32_t>& c, int32_t b_next,
                           int32_t c_next) {
        KVCache cache(1, 4, 32, QType::F16, 8);
        prefill_into_blocks(executor, cache, seq_a, {0, 1});
        prefill_into_blocks(executor, cache, b, {2, 3});
        prefill_into_blocks(executor, cache, c, {4, 5});
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        return decode_batch(executor, cache, {next_a, b_next, c_next},
                            {pos_a, static_cast<int>(b.size()), static_cast<int>(c.size())},
                            {ctx_a, static_cast<int>(b.size()) + 1, static_cast<int>(c.size()) + 1}, bt3,
                            kMaxBlocks, kVocab);
    };

    // --- Property 1: neighbours' CONTENT must not reach this row at all -----
    // Both arms have the identical batch shape, the identical row lengths and
    // the identical physical blocks for row 0. Every kernel therefore runs at
    // exactly the same dimensions, so there is no rounding excuse: any
    // difference is one sequence's KV or mask reaching another's row. Bit
    // equality is the right bar here, and it is the assert that would have
    // caught the #1044/#1045 cross-request KV class.
    std::vector<float> batched = run_batched(seq_b, seq_c, 9, 12);
    std::vector<float> batched_other = run_batched(seq_b2, seq_c2, 250, 5);
    ASSERT_GE(batched.size(), static_cast<size_t>(kVocab));
    ASSERT_GE(batched_other.size(), static_cast<size_t>(kVocab));

    int content_diffs = 0;
    double content_max = 0.0;
    for (int i = 0; i < kVocab; i++) {
        if (batched[i] != batched_other[i]) {
            content_diffs++;
            content_max = std::max(content_max, static_cast<double>(std::abs(batched[i] - batched_other[i])));
        }
    }
    EXPECT_EQ(content_diffs, 0) << "a sequence's decode logits changed when only its BATCH NEIGHBOURS' "
                                   "token content changed — same batch shape, same row lengths, same "
                                   "physical blocks, so this is leakage, not rounding ("
                                << content_diffs << " of " << kVocab
                                << " logits differ, max |delta| = " << content_max << ", #1314)";

    // --- Property 2: joining a batch may only cost rounding -----------------
    // Solo vs batched DOES change GEMM shapes (M=1 vs M=3), so bit equality is
    // not the property; agreement within tolerance and an identical greedy
    // argmax are. #1314 is the end-to-end symptom of this margin landing on a
    // near-tie.
    std::vector<float> solo;
    {
        KVCache cache(1, 4, 32, QType::F16, 8);
        prefill_into_blocks(executor, cache, seq_a, {0, 1});
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        solo = decode_batch(executor, cache, {next_a}, {pos_a}, {ctx_a}, {0, 1}, kMaxBlocks, kVocab);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }
    ASSERT_EQ(solo.size(), static_cast<size_t>(kVocab));

    auto mm = std::minmax_element(solo.begin(), solo.end());
    const double range = static_cast<double>(*mm.second - *mm.first);
    double max_abs = 0.0;
    for (int i = 0; i < kVocab; i++)
        max_abs = std::max(max_abs, static_cast<double>(std::abs(solo[i] - batched[i])));

    const int argmax_solo = static_cast<int>(std::max_element(solo.begin(), solo.begin() + kVocab) -
                                             solo.begin());
    const int argmax_batched = static_cast<int>(std::max_element(batched.begin(), batched.begin() + kVocab) -
                                                batched.begin());

    // Measured 2026-08-10 on this shape: max |delta| = 3.1e-3 over a logit
    // range of 1.41, i.e. 0.22 % of the range. The bound is 1 % of the range —
    // ~4.5x headroom over the measurement, and far below what a mask or
    // padding fault costs. Property 1 above is the strict half; this one is
    // deliberately not tightened to the measurement, because the arms
    // genuinely run different GEMM shapes.
    const double tol = 1e-2 * std::max(range, 1.0);
    EXPECT_LE(max_abs, tol) << "decode logits for one sequence moved too far when unrelated sequences "
                               "joined its batch: max |delta| = "
                            << max_abs << " over a logit range of " << range << " (#1314)";
    EXPECT_EQ(argmax_solo, argmax_batched)
        << "greedy argmax changed with batch composition (solo=" << argmax_solo
        << ", batched=" << argmax_batched << ", max |delta| = " << max_abs << ")";

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 6: Deterministic logits (same input → same output)
// ---------------------------------------------------------------------------
TEST(ForwardPassTest, DeterministicLogits) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 1, 4, 4, 64);

    GraphExecutor executor;
    init_executor(executor, *tm.model);

    std::vector<int32_t> tokens = {1, 42, 100};

    // Run 1
    KVCache cache1(1, 4, 32, QType::F16, 8);
    Tensor logits1 = run_prefill(executor, cache1, tokens);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    auto h1 = read_logits(logits1, 256);

    // Run 2 (fresh KV cache)
    KVCache cache2(1, 4, 32, QType::F16, 8);
    Tensor logits2 = run_prefill(executor, cache2, tokens);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    auto h2 = read_logits(logits2, 256);

    // Compare bitwise
    for (int i = 0; i < 256; i++) {
        EXPECT_EQ(h1[i], h2[i]) << "Logit mismatch at index " << i << " (run1=" << h1[i] << ", run2=" << h2[i]
                                << ")";
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
    init_executor(executor, *tm.model);

    // 32 tokens need 2 KV blocks (block_size=16)
    KVCache cache(1, 4, 32, QType::F16, 8);

    std::vector<int32_t> tokens(32);
    for (int i = 0; i < 32; i++)
        tokens[i] = (i + 1) % 256;

    Tensor logits = run_prefill(executor, cache, tokens, /*max_blocks_per_seq=*/2);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_NE(logits.data, nullptr);
    verify_logits_finite(logits, 256);

    tm.cleanup();
}

}  // anonymous namespace
}  // namespace imp
