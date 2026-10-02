// Llama 3 GGUF specifics (#2520): rope_freqs.weight reaches the RoPE tables; llama-bpe adds BOS.
#include <gtest/gtest.h>

#include "gguf_stub.h"
#include "model/gguf_loader.h"
#include "model/model.h"
#include "model/tokenizer.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

// Stub geometry: rope.dimension_count=32, no rope.freq_base (theta 10000).
constexpr int kRopeDim = 32;
constexpr int kPairs = kRopeDim / 2;
constexpr double kTheta = 10000.0;

// HF transformers _compute_llama3_parameters, closed form per pair.
double hf_llama3_factor(int i, double factor, double low, double high, double orig) {
    const double base = 1.0 / std::pow(kTheta, 2.0 * i / kRopeDim);
    const double wavelen = 2.0 * M_PI / base;
    if (wavelen < orig / high)
        return 1.0;
    if (wavelen > orig / low)
        return factor;
    const double smooth = (orig / wavelen - low) / (high - low);
    return factor / (1.0 - smooth + smooth * factor);
}

struct StubModel {
    std::string path;
    std::unique_ptr<imp::Model> model;
    explicit StubModel(const std::vector<float>& rope_freqs, const std::string& pre = "") {
        path = imp::test::generate_gguf_stub("llama", rope_freqs, pre);
        if (!path.empty())
            model = imp::load_gguf(path);
    }
    ~StubModel() {
        model.reset();
        if (!path.empty())
            imp::test::remove_gguf_stub(path);
    }
};

}  // namespace

TEST(GgufRopeFreqs, NoRopeFreqsTensorLeavesTablesEmpty) {
    StubModel s({});
    ASSERT_NE(s.model, nullptr);
    EXPECT_TRUE(s.model->config().rope_short_factor.empty());
    EXPECT_TRUE(s.model->config().rope_long_factor.empty());
}

// Llama-3.2 rope_scaling: factor 32, low 1, high 4, orig 8192. With theta 1e4 and rd 32 the
// table spans all three HF regions: unscaled (i<=10), smooth (11,12), fully scaled (i>=13).
TEST(GgufRopeFreqs, Llama3TableMatchesHfClosedForm) {
    std::vector<float> table(kPairs);
    for (int i = 0; i < kPairs; ++i)
        table[i] = static_cast<float>(hf_llama3_factor(i, 32.0, 1.0, 4.0, 8192.0));
    ASSERT_EQ(table[0], 1.0f);
    ASSERT_GT(table[11], 1.0f);
    ASSERT_LT(table[12], 32.0f);
    ASSERT_EQ(table[kPairs - 1], 32.0f);

    StubModel s(table);
    ASSERT_NE(s.model, nullptr);
    const imp::ModelConfig& cfg = s.model->config();
    ASSERT_EQ(static_cast<int>(cfg.rope_long_factor.size()), kPairs);
    ASSERT_EQ(static_cast<int>(cfg.rope_short_factor.size()), kPairs);
    for (int i = 0; i < kPairs; ++i) {
        // Effective frequency the executor builds (executor_workspace.cpp) vs HF inv_freq_llama.
        const double hf_freq = (1.0 / std::pow(kTheta, 2.0 * i / kRopeDim)) /
                               hf_llama3_factor(i, 32.0, 1.0, 4.0, 8192.0);
        const double imp_freq = (1.0 / std::pow(static_cast<double>(cfg.rope_theta), 2.0 * i / kRopeDim)) /
                                cfg.rope_long_factor[i];
        EXPECT_NEAR(imp_freq / hf_freq, 1.0, 1e-6) << "pair " << i;
        EXPECT_EQ(cfg.rope_short_factor[i], cfg.rope_long_factor[i]) << "pair " << i;
    }
}

// A table whose length does not match the rope pairs is ignored, not misapplied.
TEST(GgufRopeFreqs, WrongLengthTableIgnored) {
    StubModel s(std::vector<float>(kPairs + 1, 2.0f));
    ASSERT_NE(s.model, nullptr);
    EXPECT_TRUE(s.model->config().rope_long_factor.empty());
}

// No tokenizer.ggml.add_bos_token key: llama-bpe adds BOS (llama.cpp: 12806 = 12805 + BOS on
// ppl_corpus_45k), other gpt2 pre-tokenizers keep the no-BOS default.
TEST(GgufLlama3Bos, LlamaBpeDefaultsToBos) {
    for (const char* pre : {"llama-bpe", "llama3", "llama-v3"}) {
        StubModel s({}, pre);
        ASSERT_NE(s.model, nullptr) << pre;
        EXPECT_TRUE(s.model->tokenizer()->add_bos()) << pre;
    }
    for (const char* pre : {"", "qwen2"}) {
        StubModel s({}, pre);
        ASSERT_NE(s.model, nullptr) << pre;
        EXPECT_FALSE(s.model->tokenizer()->add_bos()) << pre;
    }
}
