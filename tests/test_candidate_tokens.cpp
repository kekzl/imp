// /v1/decide + /v1/score candidate guard (#2198) on the real Qwen3 tokenizer.json: letters
// A..P are single tokens after the chat generation prompt, and ":A" / ">A" merge into one
// token, which the boundary check must refuse. Reads only tokenizer.json, no GPU.

#include "candidate_tokens.h"
#include "model/tokenizer.h"

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace {

// Qwen3 model DIRECTORY (e.g. /models/Qwen3-0.6B) or a tokenizer.json path.
constexpr const char* kEnvQwen3Tok = "IMP_TEST_TOKENIZER_QWEN3";

class CandidateTokenGuardTest : public ::testing::Test {
protected:
    void SetUp() override {
        const char* env = std::getenv(kEnvQwen3Tok);
        if (!env)
            GTEST_SKIP() << "Set " << kEnvQwen3Tok << " to a Qwen3 model directory or its tokenizer.json";
        std::filesystem::path p(env);
        if (std::filesystem::is_directory(p))
            p /= "tokenizer.json";
        // A set-but-wrong path must FAIL, not skip.
        ASSERT_TRUE(std::filesystem::exists(p))
            << kEnvQwen3Tok << "=" << env << " but " << p << " is missing";
        ASSERT_TRUE(tok_.load(p.string())) << "Tokenizer::load failed for " << p;
    }

    std::vector<std::string> letters(int n) {
        std::vector<std::string> out;
        for (int i = 0; i < n; i++)
            out.emplace_back(1, static_cast<char>('A' + i));
        return out;
    }

    imp::Tokenizer tok_;
};

TEST_F(CandidateTokenGuardTest, SixteenLettersAfterGenerationPromptAreSingleTokens) {
    // Qwen3 generation prompt with thinking suppressed ends on "</think>\n\n".
    const auto res = imp::server::resolve_candidate_tokens(tok_, "\n\n", letters(16));
    ASSERT_TRUE(res.error.empty()) << res.error;
    ASSERT_EQ(res.ids.size(), 16u);
    EXPECT_EQ(std::set<int32_t>(res.ids.begin(), res.ids.end()).size(), 16u);
    for (int i = 0; i < 16; i++) {
        const auto alone = tok_.encode(std::string(1, static_cast<char>('A' + i)), true);
        ASSERT_EQ(alone.size(), 1u);
        EXPECT_EQ(res.ids[static_cast<size_t>(i)], alone[0]);
    }
}

TEST_F(CandidateTokenGuardTest, ColonPrefixMergesAndIsRefused) {
    // Precondition the guard exists for: ":A" is one token in this vocabulary.
    ASSERT_EQ(tok_.encode(":A", true).size(), 1u);
    const auto res = imp::server::resolve_candidate_tokens(tok_, "Answer:", letters(2));
    EXPECT_TRUE(res.ids.empty());
    EXPECT_NE(res.error.find("merges with the end of the prompt"), std::string::npos) << res.error;
}

TEST_F(CandidateTokenGuardTest, AngleBracketPrefixMergesAndIsRefused) {
    ASSERT_EQ(tok_.encode(">A", true).size(), 1u);
    const auto res = imp::server::resolve_candidate_tokens(tok_, "<answer>", letters(4));
    EXPECT_TRUE(res.ids.empty());
    EXPECT_NE(res.error.find("merges with the end of the prompt"), std::string::npos) << res.error;
}

TEST_F(CandidateTokenGuardTest, SpaceBeforeColonPrefixIsAccepted) {
    // Control arm for the two refusals: same letters, prefix ends in a separator that splits.
    const auto res = imp::server::resolve_candidate_tokens(tok_, "Answer:\n", letters(4));
    EXPECT_TRUE(res.error.empty()) << res.error;
    EXPECT_EQ(res.ids.size(), 4u);
}

TEST_F(CandidateTokenGuardTest, MultiTokenCandidateIsRefused) {
    const auto res = imp::server::resolve_candidate_tokens(tok_, "\n\n", {"A", "Xylophonequarkzz"});
    EXPECT_TRUE(res.ids.empty());
    EXPECT_NE(res.error.find("scoring needs exactly one"), std::string::npos) << res.error;
}

TEST_F(CandidateTokenGuardTest, DuplicateCandidateIsRefused) {
    const auto res = imp::server::resolve_candidate_tokens(tok_, "\n\n", {"A", "B", "A"});
    EXPECT_TRUE(res.ids.empty());
    EXPECT_NE(res.error.find("another candidate already uses"), std::string::npos) << res.error;
}

TEST_F(CandidateTokenGuardTest, TailStopsAtTheLastControlToken) {
    const auto prompt = tok_.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", true);
    const std::string tail = imp::server::prompt_tail_text(tok_, prompt);
    EXPECT_EQ(tail, "assistant\n");
    const auto ok = imp::server::resolve_candidate_tokens(tok_, tail, letters(16));
    EXPECT_TRUE(ok.error.empty()) << ok.error;

    const auto colon = tok_.encode("<|im_start|>assistant\nAnswer:", true);
    const auto bad = imp::server::resolve_candidate_tokens(tok_, imp::server::prompt_tail_text(tok_, colon),
                                                           letters(2));
    EXPECT_FALSE(bad.error.empty()) << "prompt ending in ':' must fail the boundary guard";
}

TEST_F(CandidateTokenGuardTest, CandidateIdsRangeAndDuplicates) {
    EXPECT_TRUE(imp::server::validate_candidate_ids(tok_, {1, 2, 3}).empty());
    EXPECT_FALSE(imp::server::validate_candidate_ids(tok_, {-1, 2}).empty());
    EXPECT_FALSE(imp::server::validate_candidate_ids(tok_, {tok_.vocab_size(), 2}).empty());
    EXPECT_FALSE(imp::server::validate_candidate_ids(tok_, {5, 5}).empty());
}

}  // namespace
