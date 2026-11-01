// Gemma-4 E4B (PLE + KV sharing) greedy against the HF fp32 CPU reference.
// IMP_TEST_MODEL_GEMMA4_E4B = checkpoint dir (google/gemma-4-E4B-it, BF16),
// IMP_TEST_GEMMA4_E4B_REF = reference dir (ids.txt, greedy_ids.txt, prompt.txt; long/ the same for
// a 20-token answer), written by gemma4_e4b_ref.py / gemma4_e4b_greedy2.py (transformers 5.20).
// Reference: "The capital of France is **Paris**.<turn|>" (9 ids, min top-2 margin 12.47).
// Skips without both variables; CUDA graphs on (production path), FP16 KV, no drafters.

#include <gtest/gtest.h>

#include "imp/imp.h"
#include "runtime/config.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> read_ids(const std::string& path) {
    std::vector<int32_t> ids;
    std::ifstream in(path);
    int64_t v = 0;
    while (in >> v)
        ids.push_back(static_cast<int32_t>(v));
    return ids;
}

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

class Gemma4E4BRef : public ::testing::Test {
protected:
    void SetUp() override {
        const char* model = std::getenv("IMP_TEST_MODEL_GEMMA4_E4B");
        const char* ref = std::getenv("IMP_TEST_GEMMA4_E4B_REF");
        if (model == nullptr || ref == nullptr)
            GTEST_SKIP() << "Set IMP_TEST_MODEL_GEMMA4_E4B and IMP_TEST_GEMMA4_E4B_REF";
        ref_ = ref;
        imp::RuntimeConfig rc;
        rc.speculative.ngram = false;
        rc.speculative.suffix = false;
        rc.speculative.token_recycling = false;
        rc.speculative.mtp_k = 0;
        rc.kv_cache.dtype = "fp16";
        imp::set_pending_runtime_config(rc);
        ASSERT_EQ(imp_model_load(model, IMP_FORMAT_SAFETENSORS, &model_), IMP_SUCCESS);
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 1024;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 1;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }
    void TearDown() override {
        if (ctx_ != nullptr)
            imp_context_free(ctx_);
        if (model_ != nullptr)
            imp_model_free(model_);
    }

    // Greedy from the reference prompt ids (tokenizer bypassed: the forward is under test).
    std::vector<int32_t> greedy(const std::vector<int32_t>& prompt, int n_gen) {
        ImpGenerateParams p = imp_generate_params_default();
        p.temperature = 0.0f;
        p.top_k = 1;
        p.top_p = 1.0f;
        p.max_tokens = n_gen;
        EXPECT_EQ(imp_context_reset(ctx_), IMP_SUCCESS);
        EXPECT_EQ(imp_prefill_with_params(ctx_, prompt.data(), static_cast<int>(prompt.size()), &p),
                  IMP_SUCCESS);
        std::vector<int32_t> out;
        int32_t tok = -1;
        EXPECT_EQ(imp_prefill_token(ctx_, &tok), IMP_SUCCESS);
        while (tok >= 0 && static_cast<int>(out.size()) < n_gen) {
            out.push_back(tok);
            if (tok == 1 || tok == 106 || tok == 50)  // generation_config eos ids
                break;
            if (imp_decode_step(ctx_, &p, &tok) != IMP_SUCCESS)
                break;
        }
        return out;
    }

    void expect_matches(const std::string& sub) {
        const std::vector<int32_t> prompt = read_ids(ref_ + sub + "/ids.txt");
        const std::vector<int32_t> want = read_ids(ref_ + sub + "/greedy_ids.txt");
        ASSERT_FALSE(prompt.empty());
        ASSERT_FALSE(want.empty());
        const std::vector<int32_t> got = greedy(prompt, static_cast<int>(want.size()));
        EXPECT_EQ(got, want) << "HF: " << read_text(ref_ + sub + "/greedy.txt");
    }

    std::string ref_;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(Gemma4E4BRef, CapitalOfFranceMatchesHf) { expect_matches(""); }

// 20 tokens without EOS; HF min top-2 margin 0.396 at step 13, so a late flip is a near-tie.
TEST_F(Gemma4E4BRef, SkyIsBlueTwentyTokensMatchHf) { expect_matches("/long"); }

}  // namespace
