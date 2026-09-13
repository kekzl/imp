// Cross-sequence ragged prefill (runtime.prefill_batch, roadmap 0(d)): concatenates chunks of
// several requests into one forward instead of one sequence per forward.
// MatchesSerialDense: batched vs serial greedy (runtime.deterministic) must agree despite
// different GEMM M-shape accumulation order; RepeatBatchIdentical: no state leak between
// rounds; ChunkedRagged: chunked continuation (q_offset>0) still matches serial.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "runtime/engine.h"
#include "runtime/config.h"
#include "compute/gemm.h"
#include "test_model_builder.h"

#include <memory>
#include <vector>

namespace imp {
namespace {

using test::DenseTestModel;

static EngineConfig ragged_engine_config() {
    EngineConfig cfg;
    cfg.max_batch_size = 4;
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
    cfg.use_prefix_caching = false;
    cfg.use_mxfp4_prefill = false;
    cfg.dual_path_quant = false;
    return cfg;
}

static RuntimeConfig ragged_runtime_config(bool prefill_batch, int chunk_size = -1) {
    RuntimeConfig rc;
    rc.runtime.prefill_batch = prefill_batch;
    rc.runtime.deterministic = true;
    if (chunk_size > 0)
        rc.runtime.prefill_chunk_size = chunk_size;
    return rc;
}

static std::vector<std::vector<int32_t>> ragged_prompts() {
    return {
        {5, 17, 42},
        {101, 7, 88, 23, 54, 9, 111, 6, 40},
        {200, 3},
        {60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73},
    };
}

// Submit all prompts at once, step to completion, return per-request outputs.
static std::vector<std::vector<int32_t>> run_batch(Engine& engine,
                                                   const std::vector<std::vector<int32_t>>& prompts,
                                                   int max_tokens) {
    std::vector<std::shared_ptr<Request>> reqs;
    for (const auto& p : prompts) {
        auto req = std::make_shared<Request>();
        req->input_tokens = p;
        req->max_tokens = max_tokens;
        req->temperature = 0.0f;
        req->seed = 42;
        engine.add_request(req);
        reqs.push_back(req);
    }
    auto all_done = [&]() {
        for (const auto& r : reqs)
            if (r->status != RequestStatus::FINISHED && r->status != RequestStatus::CANCELLED)
                return false;
        return true;
    };
    int steps = 0;
    while (!all_done() && steps < 200) {
        (void)engine.step();
        steps++;
    }
    std::vector<std::vector<int32_t>> out;
    for (const auto& r : reqs) {
        EXPECT_EQ(r->status, RequestStatus::FINISHED)
            << "request with " << r->input_tokens.size() << " prompt tokens did not finish";
        EXPECT_GE(static_cast<int>(r->output_tokens.size()), 1);
        out.push_back(r->output_tokens);
    }
    return out;
}

class RaggedPrefillTest : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

TEST_F(RaggedPrefillTest, MatchesSerialDense) {
    auto prompts = ragged_prompts();
    std::vector<std::vector<int32_t>> ragged_out, serial_out;
    {
        auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
        gemm_init();
        set_pending_runtime_config(ragged_runtime_config(/*prefill_batch=*/true));
        Engine engine;
        ASSERT_TRUE(engine.init(tm.model, ragged_engine_config()));
        ragged_out = run_batch(engine, prompts, 6);
        tm.cleanup();
    }
    {
        auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
        gemm_init();
        set_pending_runtime_config(ragged_runtime_config(/*prefill_batch=*/false));
        Engine engine;
        ASSERT_TRUE(engine.init(tm.model, ragged_engine_config()));
        serial_out = run_batch(engine, prompts, 6);
        tm.cleanup();
    }
    ASSERT_EQ(ragged_out.size(), serial_out.size());
    for (size_t i = 0; i < ragged_out.size(); ++i)
        EXPECT_EQ(ragged_out[i], serial_out[i]) << "request " << i << " diverged (ragged vs serial)";
}

TEST_F(RaggedPrefillTest, RepeatBatchIdentical) {
    auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
    gemm_init();
    set_pending_runtime_config(ragged_runtime_config(/*prefill_batch=*/true));
    Engine engine;
    ASSERT_TRUE(engine.init(tm.model, ragged_engine_config()));

    auto prompts = ragged_prompts();
    auto round1 = run_batch(engine, prompts, 6);
    auto round2 = run_batch(engine, prompts, 6);
    ASSERT_EQ(round1.size(), round2.size());
    for (size_t i = 0; i < round1.size(); ++i)
        EXPECT_EQ(round1[i], round2[i]) << "request " << i << " diverged across rounds (state leak?)";
    tm.cleanup();
}

TEST_F(RaggedPrefillTest, ChunkedRagged) {
    // 40-token prompts with a 16-token chunk force multi-chunk ragged prefill:
    // continuation chunks run the per-seq attention loop with q_offset > 0.
    std::vector<std::vector<int32_t>> prompts(3);
    for (int r = 0; r < 3; ++r)
        for (int i = 0; i < 40 - 8 * r; ++i)
            prompts[r].push_back((r * 97 + i * 13) % 256);

    std::vector<std::vector<int32_t>> ragged_out, serial_out;
    {
        auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
        gemm_init();
        set_pending_runtime_config(ragged_runtime_config(/*prefill_batch=*/true, /*chunk_size=*/16));
        Engine engine;
        ASSERT_TRUE(engine.init(tm.model, ragged_engine_config()));
        ragged_out = run_batch(engine, prompts, 4);
        tm.cleanup();
    }
    {
        auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
        gemm_init();
        set_pending_runtime_config(ragged_runtime_config(/*prefill_batch=*/false, /*chunk_size=*/16));
        Engine engine;
        ASSERT_TRUE(engine.init(tm.model, ragged_engine_config()));
        serial_out = run_batch(engine, prompts, 4);
        tm.cleanup();
    }
    ASSERT_EQ(ragged_out.size(), serial_out.size());
    for (size_t i = 0; i < ragged_out.size(); ++i)
        EXPECT_EQ(ragged_out[i], serial_out[i]) << "request " << i << " diverged (chunked ragged)";
}

// Staggered arrival: two prompts decode while two more prefill, so the late pair's ragged
// forward carries the early pair as one-row riders (runtime.prefill_mixed_decode). Must match
// the separate-step arm's tokens, and the engine must report mixed steps actually ran.
static std::vector<std::vector<int32_t>> run_staggered(Engine& engine,
                                                       const std::vector<std::vector<int32_t>>& prompts,
                                                       int max_tokens) {
    std::vector<std::shared_ptr<Request>> reqs;
    auto submit = [&](size_t i) {
        auto req = std::make_shared<Request>();
        req->input_tokens = prompts[i];
        req->max_tokens = max_tokens;
        req->temperature = 0.0f;
        req->seed = 42;
        engine.add_request(req);
        reqs.push_back(req);
    };
    submit(0);
    submit(1);
    for (int s = 0; s < 3; ++s)
        (void)engine.step();
    submit(2);
    submit(3);
    auto all_done = [&]() {
        for (const auto& r : reqs)
            if (r->status != RequestStatus::FINISHED && r->status != RequestStatus::CANCELLED)
                return false;
        return true;
    };
    int steps = 0;
    while (!all_done() && steps < 200) {
        (void)engine.step();
        steps++;
    }
    std::vector<std::vector<int32_t>> out;
    for (const auto& r : reqs) {
        EXPECT_EQ(r->status, RequestStatus::FINISHED);
        out.push_back(r->output_tokens);
    }
    return out;
}

TEST_F(RaggedPrefillTest, MixedDecodeRidersMatchSeparateSteps) {
    auto prompts = ragged_prompts();
    std::vector<std::vector<int32_t>> mixed_out, plain_out;
    uint64_t mixed_steps = 0;
    {
        auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
        gemm_init();
        auto rc = ragged_runtime_config(/*prefill_batch=*/true);
        rc.runtime.prefill_mixed_decode = true;
        set_pending_runtime_config(rc);
        Engine engine;
        ASSERT_TRUE(engine.init(tm.model, ragged_engine_config()));
        mixed_out = run_staggered(engine, prompts, 8);
        mixed_steps = engine.mixed_decode_steps();
        tm.cleanup();
    }
    {
        auto tm = DenseTestModel::create(128, 512, 256, 2, 4, 4, 64);
        gemm_init();
        auto rc = ragged_runtime_config(/*prefill_batch=*/true);
        rc.runtime.prefill_mixed_decode = false;
        set_pending_runtime_config(rc);
        Engine engine;
        ASSERT_TRUE(engine.init(tm.model, ragged_engine_config()));
        plain_out = run_staggered(engine, prompts, 8);
        EXPECT_EQ(engine.mixed_decode_steps(), 0u);
        tm.cleanup();
    }
    EXPECT_GE(mixed_steps, 1u) << "no step carried riders: the late pair's prefill ran without the decoders";
    ASSERT_EQ(mixed_out.size(), plain_out.size());
    for (size_t i = 0; i < mixed_out.size(); ++i)
        EXPECT_EQ(mixed_out[i], plain_out[i]) << "request " << i << " diverged (mixed vs separate steps)";
}

}  // namespace
}  // namespace imp
