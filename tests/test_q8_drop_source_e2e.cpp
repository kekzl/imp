// Phase 4c on a dense Q8_0 GGUF: the sources go, the planner credits them, greedy decode runs 48 tokens
// (the keep-sources arm left with gemm.q8_drop_source, 2026-11-14). GTEST_SKIPs without the model.

#include "imp/imp.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <regex>
#include <string>
#include <vector>

namespace {

constexpr const char kModel[] = "/models/Qwen3-4B-Instruct-2507-Q8_0.gguf";

struct Arm {
    std::string log;
    std::vector<int32_t> tokens;
};

// Loads the model, returns the init log and 48 greedy tokens.
Arm run_arm() {
    Arm arm;
    imp::RuntimeConfig rc;
    rc.speculative.ngram = false;
    rc.speculative.suffix = false;
    rc.speculative.token_recycling = false;
    rc.speculative.mtp_k = 0;
    imp::set_pending_runtime_config(rc);
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    EXPECT_EQ(imp_model_load(kModel, IMP_FORMAT_GGUF, &model), IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 512;
    cfg.max_batch_size = 1;
    testing::internal::CaptureStdout();
    const ImpError rc_ctx = imp_context_create(model, &cfg, &ctx);
    arm.log = testing::internal::GetCapturedStdout();
    EXPECT_EQ(rc_ctx, IMP_SUCCESS) << arm.log;
    if (rc_ctx == IMP_SUCCESS) {
        ImpGenerateParams params = imp_generate_params_default();
        params.temperature = 0.0f;
        params.top_k = 1;
        params.max_tokens = 48;
        std::vector<int32_t> prompt(256);
        int n_prompt = 0;
        EXPECT_EQ(imp_tokenize(model, "Write a short poem about the sea and the wind.", prompt.data(),
                               &n_prompt, 256),
                  IMP_SUCCESS);
        EXPECT_EQ(imp_prefill_with_params(ctx, prompt.data(), n_prompt, &params), IMP_SUCCESS);
        int32_t tok = -1;
        EXPECT_EQ(imp_prefill_token(ctx, &tok), IMP_SUCCESS);
        if (tok >= 0)
            arm.tokens.push_back(tok);
        while (arm.tokens.size() < 48 && imp_decode_step(ctx, &params, &tok) == IMP_SUCCESS && tok >= 0)
            arm.tokens.push_back(tok);
        imp_context_free(ctx);
    }
    imp_model_free(model);
    return arm;
}

TEST(Q8DropSourceE2E, SourcesGoGreedyDecodes) {
    if (!std::filesystem::exists(kModel))
        GTEST_SKIP() << "model not present at " << kModel;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || (free_b >> 20) < 14000)
        GTEST_SKIP() << "needs ~14000 MiB free";
    const Arm drop = run_arm();
    std::smatch m;
    ASSERT_TRUE(
        std::regex_search(drop.log, m, std::regex(R"(Phase-4c: freed (\d+) Q8_0 sources \(([0-9.]+) MiB)")))
        << drop.log;
    EXPECT_GT(std::stoi(m[1].str()), 0);
    EXPECT_GT(std::stod(m[2].str()), 0.0);
    EXPECT_NE(drop.log.find("of Q8_0 sources freed for their planes"), std::string::npos)
        << "the planner charged the planes on top of sources Phase 4c frees";
    EXPECT_EQ(drop.tokens.size(), 48u) << "greedy decode on the planes-only weights stopped early";
}

}  // namespace
