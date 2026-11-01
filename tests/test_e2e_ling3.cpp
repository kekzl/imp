// Ling-3.0 (bailing_hybrid: KDA + MLA) greedy against the HF fp32 CPU reference.
// IMP_TEST_MODEL_LING3 = checkpoint dir (inclusionAI/Ling-3.0-tiny, BF16),
// IMP_TEST_LING3_REF = reference dir (ids.txt, greedy_ids.txt, greedy.txt) from ling_ref.py.
// Reference: "The capital of France is Paris.<|role_end|>" (8 ids, 28-token prompt).
// Decode reads the KDA state the prefill wrote; a chunked prefill hands it across forwards.

#include <gtest/gtest.h>

#include "imp/imp.h"
#include "runtime/config.h"

#include <cstdint>
#include <cstdlib>
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

class Ling3Ref : public ::testing::Test {
protected:
    void load(int prefill_chunk) {
        const char* model = std::getenv("IMP_TEST_MODEL_LING3");
        const char* ref = std::getenv("IMP_TEST_LING3_REF");
        if (model == nullptr || ref == nullptr)
            GTEST_SKIP() << "Set IMP_TEST_MODEL_LING3 and IMP_TEST_LING3_REF";
        ref_ = ref;
        imp::RuntimeConfig rc;
        rc.speculative.ngram = false;
        rc.speculative.suffix = false;
        rc.speculative.token_recycling = false;
        rc.speculative.mtp_k = 0;
        rc.kv_cache.dtype = "fp16";
        rc.runtime.prefill_chunk_size = prefill_chunk;
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
            if (tok == 156895)  // <|role_end|>
                break;
            if (imp_decode_step(ctx_, &p, &tok) != IMP_SUCCESS)
                break;
        }
        return out;
    }

    void expect_matches() {
        const std::vector<int32_t> prompt = read_ids(ref_ + "/ids.txt");
        const std::vector<int32_t> want = read_ids(ref_ + "/greedy_ids.txt");
        ASSERT_FALSE(prompt.empty());
        ASSERT_FALSE(want.empty());
        const std::vector<int32_t> got = greedy(prompt, static_cast<int>(want.size()));
        EXPECT_EQ(got, want) << "HF: " << read_text(ref_ + "/greedy.txt");
    }

    std::string ref_;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(Ling3Ref, CapitalOfFranceMatchesHf) {
    load(-1);
    if (!IsSkipped() && !HasFatalFailure())
        expect_matches();
}

// 28 prompt tokens in chunks of 9: four forwards, the last one token (the n == 1 prefill path).
TEST_F(Ling3Ref, ChunkedPrefillMatchesHf) {
    load(9);
    if (!IsSkipped() && !HasFatalFailure())
        expect_matches();
}

}  // namespace
