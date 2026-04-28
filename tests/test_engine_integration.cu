#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "runtime/engine.h"
#include "compute/gemm.h"
#include "test_model_builder.h"

#include <memory>

namespace imp {
namespace {

using test::DenseTestModel;
using test::MoETestModel;
using test::Q8DenseTestModel;

// Minimal EngineConfig with all auto-detection disabled for deterministic tests.
static EngineConfig test_engine_config() {
    EngineConfig cfg;
    cfg.max_batch_size = 1;
    cfg.max_seq_len = 64;
    cfg.use_cuda_graphs = false;
    cfg.use_pdl = false;
    cfg.use_fp8_prefill = false;
    cfg.use_nvfp4_decode = 0;
    cfg.kv_cache_dtype = QType::F16;
    cfg.compute_dtype = QType::F16;
    cfg.kv_block_size = 16;
    cfg.use_green_contexts = false;
    cfg.gpu_layers = -1;
    cfg.enable_speculative = false;
    cfg.enable_self_speculative = false;
    cfg.enable_ngram_spec = false;
    cfg.use_prefix_caching = false;
    cfg.use_mxfp4_prefill = false;
    cfg.dual_path_quant = false;
    return cfg;
}

// ---------------------------------------------------------------------------
// Test 1: Engine initializes with synthetic dense model
// ---------------------------------------------------------------------------
TEST(EngineIntegrationTest, InitSucceeds) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
    gemm_init();

    Engine engine;
    ASSERT_TRUE(engine.init(tm.model, test_engine_config()));

    EXPECT_NE(engine.executor(), nullptr);
    EXPECT_NE(engine.kv_cache(), nullptr);
    EXPECT_NE(engine.scheduler(), nullptr);
    EXPECT_NE(engine.model(), nullptr);

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 2: Engine step loop — prefill + decode a short sequence
// ---------------------------------------------------------------------------
TEST(EngineIntegrationTest, StepPrefillDecode) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
    gemm_init();

    Engine engine;
    ASSERT_TRUE(engine.init(tm.model, test_engine_config()));

    // Create a request
    auto req = std::make_shared<Request>();
    req->input_tokens = {1, 2, 3};
    req->max_tokens = 4;
    req->temperature = 0.0f;  // greedy for determinism
    req->seed = 42;

    engine.add_request(req);

    // Run step loop until request finishes (with safety limit)
    int steps = 0;
    while (req->status != RequestStatus::FINISHED && steps < 20) {
        bool more = engine.step();
        steps++;
        if (!more && req->status != RequestStatus::FINISHED) break;
    }

    EXPECT_EQ(req->status, RequestStatus::FINISHED)
        << "Request did not finish after " << steps << " steps";
    EXPECT_GE(static_cast<int>(req->output_tokens.size()), 1)
        << "Expected at least 1 output token";
    EXPECT_LE(static_cast<int>(req->output_tokens.size()), 4)
        << "Expected at most max_tokens=4 output tokens";

    // All output tokens should be valid vocab indices
    for (int32_t tok : req->output_tokens) {
        EXPECT_GE(tok, 0);
        EXPECT_LT(tok, 256);
    }

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 3: Multiple sequential requests (no state leak between them)
// ---------------------------------------------------------------------------
TEST(EngineIntegrationTest, MultipleRequestsSequential) {
    SKIP_IF_NO_CUDA();

    auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
    gemm_init();

    Engine engine;
    ASSERT_TRUE(engine.init(tm.model, test_engine_config()));

    for (int r = 0; r < 2; r++) {
        auto req = std::make_shared<Request>();
        req->input_tokens = {10 + r, 20 + r, 30 + r};
        req->max_tokens = 3;
        req->temperature = 0.0f;
        req->seed = 42;

        engine.add_request(req);

        int steps = 0;
        while (req->status != RequestStatus::FINISHED && steps < 20) {
            (void)engine.step();
            steps++;
        }

        EXPECT_EQ(req->status, RequestStatus::FINISHED)
            << "Request " << r << " did not finish";
        EXPECT_GE(static_cast<int>(req->output_tokens.size()), 1)
            << "Request " << r << " produced no output";

        for (int32_t tok : req->output_tokens) {
            EXPECT_GE(tok, 0);
            EXPECT_LT(tok, 256);
        }
    }

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 4: Engine init with MoE model
// ---------------------------------------------------------------------------
TEST(EngineIntegrationTest, MoEInitSucceeds) {
    SKIP_IF_NO_CUDA();

    auto tm = MoETestModel::create(
        /*d_model=*/64, /*d_ff=*/128, /*vocab_size=*/256,
        /*n_layers=*/1, /*n_heads=*/4, /*n_kv_heads=*/4,
        /*n_experts=*/4, /*n_experts_active=*/2, /*expert_d_ff=*/128,
        /*max_seq_len=*/64);
    gemm_init();

    Engine engine;
    ASSERT_TRUE(engine.init(tm.model, test_engine_config()));

    EXPECT_NE(engine.executor(), nullptr);
    EXPECT_NE(engine.kv_cache(), nullptr);

    // Run a short request to verify forward pass works
    auto req = std::make_shared<Request>();
    req->input_tokens = {1, 2, 3};
    req->max_tokens = 2;
    req->temperature = 0.0f;

    engine.add_request(req);

    int steps = 0;
    while (req->status != RequestStatus::FINISHED && steps < 20) {
        (void)engine.step();
        steps++;
    }

    EXPECT_EQ(req->status, RequestStatus::FINISHED);
    EXPECT_GE(static_cast<int>(req->output_tokens.size()), 1);

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Helper: run a request through engine and verify completion
// ---------------------------------------------------------------------------
static void run_request(Engine& engine, std::vector<int32_t> input, int max_tokens,
                        int vocab_size = 256) {
    auto req = std::make_shared<Request>();
    req->input_tokens = std::move(input);
    req->max_tokens = max_tokens;
    req->temperature = 0.0f;
    req->seed = 42;

    engine.add_request(req);

    int steps = 0;
    while (req->status != RequestStatus::FINISHED && steps < 30) {
        (void)engine.step();
        steps++;
    }

    ASSERT_EQ(req->status, RequestStatus::FINISHED)
        << "Request did not finish after " << steps << " steps";
    EXPECT_GE(static_cast<int>(req->output_tokens.size()), 1);
    EXPECT_LE(static_cast<int>(req->output_tokens.size()), max_tokens);
    for (int32_t tok : req->output_tokens) {
        EXPECT_GE(tok, 0);
        EXPECT_LT(tok, vocab_size);
    }
}

// ---------------------------------------------------------------------------
// Test 5: Q8_0 weights with FP8 prefill cache
// ---------------------------------------------------------------------------
TEST(EngineIntegrationTest, FP8PrefillWithQ8Weights) {
    SKIP_IF_NO_CUDA();

    // d_model and d_ff must be divisible by 32 for Q8_0
    auto tm = Q8DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
    gemm_init();

    auto cfg = test_engine_config();
    cfg.use_fp8_prefill = true;  // enable FP8 weight cache

    Engine engine;
    ASSERT_TRUE(engine.init(tm.model, cfg));

    // Prefill + decode through FP8 path
    run_request(engine, {1, 2, 3}, 3);

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 6: Q8_0 weights with NVFP4 decode cache
// ---------------------------------------------------------------------------
TEST(EngineIntegrationTest, NVFP4DecodeWithQ8Weights) {
    SKIP_IF_NO_CUDA();

    auto tm = Q8DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
    gemm_init();

    auto cfg = test_engine_config();
    cfg.use_nvfp4_decode = 2;  // incremental NVFP4

    Engine engine;
    ASSERT_TRUE(engine.init(tm.model, cfg));

    run_request(engine, {1, 2, 3}, 3);

    tm.cleanup();
}

// ---------------------------------------------------------------------------
// Test 7: FP8 prefill + NVFP4 decode combined (dual-path)
// ---------------------------------------------------------------------------
TEST(EngineIntegrationTest, FP8PrefillNVFP4Decode) {
    SKIP_IF_NO_CUDA();

    auto tm = Q8DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
    gemm_init();

    auto cfg = test_engine_config();
    cfg.use_fp8_prefill = true;
    cfg.use_nvfp4_decode = 2;

    Engine engine;
    ASSERT_TRUE(engine.init(tm.model, cfg));

    run_request(engine, {1, 2, 3, 4, 5}, 4);

    tm.cleanup();
}

} // namespace
} // namespace imp
