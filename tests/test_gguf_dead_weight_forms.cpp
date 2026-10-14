// Weight forms no default path reads stay out of VRAM on GGUF models:
// - a dense GGUF weight keeps its CUTLASS_NVFP4 tier without an SfAtom (its M>1 GEMM reads the source);
// - an LM head served by the NVFP4 decode cache frees its GGUF source.
// Each test checks the load log, then that greedy decode still answers. GTEST_SKIPs without the model.

#include "imp/imp.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <regex>
#include <string>
#include <vector>

namespace {

size_t device_free_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return free_b >> 20;
}

class GgufDeadWeightForms : public ::testing::Test {
protected:
    // Loads `path` with greedy-only speculation off; returns the captured init log.
    std::string load(const char* path, size_t need_mib) {
        if (!std::filesystem::exists(path)) {
            skip_ = std::string("model not present at ") + path;
            return {};
        }
        if (device_free_mib() < need_mib) {
            skip_ = "needs ~" + std::to_string(need_mib) + " MiB free";
            return {};
        }
        imp::RuntimeConfig rc;
        rc.speculative.ngram = false;
        rc.speculative.suffix = false;
        rc.speculative.token_recycling = false;
        rc.speculative.mtp_k = 0;
        imp::set_pending_runtime_config(rc);
        EXPECT_EQ(imp_model_load(path, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        testing::internal::CaptureStdout();
        const ImpError rc_ctx = imp_context_create(model_, &cfg, &ctx_);
        const std::string log = testing::internal::GetCapturedStdout();
        EXPECT_EQ(rc_ctx, IMP_SUCCESS) << log;
        return log;
    }

    // Greedy continuation of `prompt`, detokenized.
    std::string greedy(const char* prompt, int n_gen) {
        ImpGenerateParams params = imp_generate_params_default();
        params.temperature = 0.0f;
        params.top_k = 1;
        params.max_tokens = n_gen;
        std::vector<int32_t> toks(256);
        int n_prompt = 0;
        EXPECT_EQ(imp_tokenize(model_, prompt, toks.data(), &n_prompt, 256), IMP_SUCCESS);
        toks.resize(n_prompt);
        EXPECT_EQ(imp_context_reset(ctx_), IMP_SUCCESS);
        EXPECT_EQ(imp_prefill_with_params(ctx_, toks.data(), n_prompt, &params), IMP_SUCCESS);
        std::vector<int32_t> out;
        int32_t tok = -1;
        EXPECT_EQ(imp_prefill_token(ctx_, &tok), IMP_SUCCESS);
        if (tok >= 0)
            out.push_back(tok);
        while (static_cast<int>(out.size()) < n_gen && imp_decode_step(ctx_, &params, &tok) == IMP_SUCCESS &&
               tok >= 0)
            out.push_back(tok);
        std::string buf(4096, '\0');
        EXPECT_EQ(imp_detokenize(model_, out.data(), static_cast<int>(out.size()), buf.data(), buf.size()),
                  IMP_SUCCESS);
        buf.resize(std::strlen(buf.c_str()));
        return buf;
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    std::string skip_;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(GgufDeadWeightForms, DenseGgufWeightsCarryNoSfAtom) {
    const std::string log = load("/models/Qwen3-14B-Q6_K.gguf", 16000);
    if (!skip_.empty())
        GTEST_SKIP() << skip_;
    std::smatch m;
    const std::regex line(
        R"(CUTLASS sm_120 NVFP4 weight cache: (\d+) tensors, ([0-9.]+) MiB; (\d+) GGUF-source)");
    ASSERT_TRUE(std::regex_search(log, m, line)) << "no CUTLASS cache line in the init log:\n" << log;
    EXPECT_GT(std::stoi(m[3].str()), 0) << "no dense GGUF weight kept its tier without an SfAtom";
    EXPECT_EQ(m[2].str(), "0.00") << "a dense GGUF weight still built an SfAtom no GEMM reads";
    EXPECT_NE(greedy("The capital of France is", 8).find("Paris"), std::string::npos);
}

// Qwen3.8-27B Q4_K_M: Q6_K head, NVFP4 decode-cache head (nvfp4_lm_head_gdn), 994.6 MiB source.
TEST_F(GgufDeadWeightForms, Nvfp4ServedHeadFreesItsSource) {
    const std::string log = load("/models/Qwen3.8-27B-Q4_K_M.gguf", 26000);
    if (!skip_.empty())
        GTEST_SKIP() << skip_;
    std::smatch m;
    const std::regex line(
        R"(NVFP4 LM head: freed the \w+ source \(([0-9.]+) MiB.*after the ([0-9.]+) MiB served head)");
    ASSERT_TRUE(std::regex_search(log, m, line)) << "the NVFP4-served head kept its source:\n" << log;
    EXPECT_GT(std::stod(m[1].str()), 0.0);
    EXPECT_GT(std::stod(m[2].str()), 0.0) << "the served NVFP4 head was not priced against the freed bytes";
    EXPECT_NE(greedy("The capital of France is", 8).find("Paris"), std::string::npos);
}

}  // namespace
