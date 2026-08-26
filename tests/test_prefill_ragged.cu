// Cross-sequence ragged prefill (runtime.prefill_batch, roadmap 0(d)).
//
// A burst of prompts used to prefill one sequence per forward; the ragged path
// concatenates the chunks of several requests into one forward (attention and
// the GDN conv loop per sequence inside the executor). These tests drive the
// full engine on the synthetic dense model:
//
//   - MatchesSerialDense: the same four greedy requests, batched vs serial,
//     produce the same tokens under runtime.deterministic. The GEMM M-shape
//     differs between the arms (concatenated rows vs per-request rows), so
//     this asserts that the low-bit accumulation-order difference does not
//     reach the argmax on this model — the property the serving path relies
//     on. If a toolchain bump ever flips a single near-tie token here, the
//     right fix is a looser comparison, not disabling the path.
//   - RepeatBatchIdentical: two identical ragged rounds on one engine match
//     byte for byte (recurrent/KV state does not leak between rounds).
//   - ChunkedRagged: prompts longer than prefill_chunk_size run the ragged
//     path with q_offset > 0 (chunked continuation per sequence) and still
//     match the serial arm.
//
// GDN-model coverage: the ragged scan itself is bit-tested in test_gdn.cu
// (seq_row_offsets); the engine-level GDN validation runs against the real
// checkpoint (docs/plans/2026-08-24-qwen38-port.md, phase-1 measurement).

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

}  // namespace
}  // namespace imp
