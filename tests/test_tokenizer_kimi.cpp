// Kimi tiktoken tokenizer (tiktoken.model + tokenizer_config.json, no tokenizer.json) vs HF
// transformers TikTokenTokenizer: golden tests/refs/tokenizer_golden_kimi.h from
// gen_tokenizer_golden.py (extended corpus + rendered chat template). Reads only tokenizer files.
#include "model/tokenizer.h"
#include "refs/tokenizer_golden_kimi.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Directory holding tiktoken.model and tokenizer_config.json (any Kimi-Linear export).
constexpr const char* kEnvKimi = "IMP_TEST_MODEL_KIMI";

std::string ids_to_string(const std::vector<int32_t>& ids) {
    std::string s;
    for (size_t i = 0; i < ids.size(); i++)
        s += (i ? " " : "") + std::to_string(ids[i]);
    return s;
}

class KimiTiktokenParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        const char* env = std::getenv(kEnvKimi);
        if (!env)
            GTEST_SKIP() << "Set " << kEnvKimi << " to a directory with tiktoken.model and "
                         << "tokenizer_config.json; the golden is " << tokenizer_golden_kimi::k_model;
        // A set-but-wrong path fails, never skips.
        const std::filesystem::path dir(env);
        ASSERT_TRUE(std::filesystem::exists(dir / "tiktoken.model")) << env << " has no tiktoken.model";
        ASSERT_TRUE(
            tok_.load_tiktoken((dir / "tiktoken.model").string(), (dir / "tokenizer_config.json").string()));
    }
    imp::Tokenizer tok_;
};

TEST_F(KimiTiktokenParityTest, FingerprintIdentifiesTheTokenizer) {
    EXPECT_EQ(tok_.encode(tokenizer_golden_kimi::k_fingerprint_text),
              tokenizer_golden_kimi::k_fingerprint_ids);
}

TEST_F(KimiTiktokenParityTest, EncodeMatchesHuggingFaceOnEveryCase) {
    int mismatches = 0;
    size_t n_tokens = 0;
    for (const auto& c : tokenizer_golden_kimi::cases()) {
        const auto ids = tok_.encode(c.text);
        n_tokens += c.ids.size();
        if (ids != c.ids) {
            mismatches++;
            ADD_FAILURE() << "encode mismatch for \"" << c.text << "\"\n  imp: [" << ids_to_string(ids)
                          << "]\n  HF : [" << ids_to_string(c.ids) << "]";
        }
    }
    EXPECT_EQ(mismatches, 0);
    std::printf("kimi tiktoken parity: %zu strings, %zu tokens, %d mismatches\n",
                tokenizer_golden_kimi::cases().size(), n_tokens, mismatches);
}

TEST_F(KimiTiktokenParityTest, RoundTripIsByteIdentical) {
    for (const auto& c : tokenizer_golden_kimi::cases()) {
        const auto ids = tok_.encode(c.text);
        EXPECT_EQ(tok_.decode(ids), std::string(c.decoded)) << "HF decode differs for \"" << c.text << "\"";
        EXPECT_EQ(tok_.decode(ids), std::string(c.text)) << "decode(encode(s)) != s for \"" << c.text << "\"";
    }
}

TEST_F(KimiTiktokenParityTest, VocabSizeMatchesTheCheckpoint) {
    EXPECT_EQ(tok_.vocab_size(), tokenizer_golden_kimi::k_n_tokens);
}

}  // namespace
