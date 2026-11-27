// Engine-wide sampler ban list (runtime/banned_tokens.h) per template family. The list is one device
// copy for every request, with or without `tools`, so a tool dialect token it bans can never be
// sampled: Gemma-4 <|tool_call> (id 48) banned = 0 of 20 auto tool calls (B-143).
#include "runtime/banned_tokens.h"

#include "model/tokenizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using imp::Tokenizer;

constexpr int kNormal = 1;
constexpr int kControl = 3;
constexpr int kUserDefined = 4;

// Vocab of `n` plain pieces with the given (id, text, type) markers placed at their real ids.
Tokenizer synthetic(int n, const std::vector<std::tuple<int, std::string, int>>& markers, int eos,
                    bool with_types = true) {
    std::vector<std::string> v;
    for (int i = 0; i < n; ++i)
        v.push_back("p" + std::to_string(i));
    std::vector<int32_t> types(n, kNormal);
    for (const auto& [id, text, type] : markers) {
        v[id] = text;
        types[id] = type;
    }
    Tokenizer t;
    EXPECT_TRUE(t.load_vocab(v, std::vector<float>(v.size(), 0.0f), 2, eos));
    if (with_types)
        t.load_token_types(types);
    return t;
}

bool banned(const std::vector<int32_t>& list, int32_t id) {
    return std::binary_search(list.begin(), list.end(), id);
}

// Gemma-4-26B-A4B-it / gemma-4-E4B-it tokenizer.json: every marker below is `special: true`.
// Stops: <eos> 1, <|tool_response> 50 (generation_config eos), <turn|> 106.
Tokenizer gemma4(bool with_types = true) {
    return synthetic(110,
                     {{0, "<pad>", kControl},
                      {1, "<eos>", kControl},
                      {2, "<bos>", kControl},
                      {3, "<unk>", kControl},
                      {4, "<mask>", kControl},
                      {46, "<|tool>", kControl},
                      {47, "<tool|>", kControl},
                      {48, "<|tool_call>", kControl},
                      {49, "<tool_call|>", kControl},
                      {50, "<|tool_response>", kControl},
                      {51, "<tool_response|>", kControl},
                      {52, "<|\"|>", kControl},
                      {98, "<|think|>", kControl},
                      {100, "<|channel>", kControl},
                      {101, "<channel|>", kControl},
                      {105, "<|turn>", kControl},
                      {106, "<turn|>", kControl}},
                     1, with_types);
}

TEST(BannedTokens, Gemma4ToolCallDialectIsSampleable) {
    const Tokenizer tok = gemma4();
    const auto list = imp::build_banned_token_ids(&tok, {106, 50}, {105, -1, -1});
    // The call a model writes: opener, closer, string delimiter.
    for (int32_t id : {48, 49, 52})
        EXPECT_FALSE(banned(list, id)) << id;
    // Stops and the reasoning channel.
    for (int32_t id : {1, 50, 106, 98, 100, 101})
        EXPECT_FALSE(banned(list, id)) << id;
    // Only the template writes these: declaration block, tool response close, turn opener.
    for (int32_t id : {0, 2, 3, 4, 46, 47, 51, 105})
        EXPECT_TRUE(banned(list, id)) << id;
    EXPECT_EQ(list.size(), 8u);
}

// A GGUF without token types takes the `<|..|>` heuristic, which matches <|"|> but not <|tool_call>.
TEST(BannedTokens, Gemma4HeuristicScanKeepsTheStringDelimiter) {
    const Tokenizer tok = gemma4(/*with_types=*/false);
    const auto list = imp::build_banned_token_ids(&tok, {106, 50}, {105, -1, -1});
    for (int32_t id : {48, 49, 52})
        EXPECT_FALSE(banned(list, id)) << id;
    EXPECT_TRUE(banned(list, 0));
    EXPECT_FALSE(banned(list, 98));
}

// ChatML (Qwen3.x): <tool_call> is a non-special added token, the delimiter <|im_start|> is CONTROL.
// Nemotron [TOOL_CALLS] (CONTROL) stays banned: that family calls through <tool_call>.
TEST(BannedTokens, OtherFamiliesUnchanged) {
    const Tokenizer tok = synthetic(20,
                                    {{0, "<|endoftext|>", kControl},
                                     {1, "<|im_start|>", kControl},
                                     {2, "<|im_end|>", kControl},
                                     {3, "<tool_call>", kUserDefined},
                                     {4, "</tool_call>", kUserDefined},
                                     {5, "<think>", kUserDefined},
                                     {9, "[TOOL_CALLS]", kControl}},
                                    0);
    const auto list = imp::build_banned_token_ids(&tok, {2}, {1, -1, -1});
    EXPECT_EQ(list, (std::vector<int32_t>{1, 9}));
}

TEST(BannedTokens, NoTokenizerBansOnlyTheDelimiters) {
    EXPECT_EQ(imp::build_banned_token_ids(nullptr, {7}, {7, 3, -1, 3}), (std::vector<int32_t>{3}));
}

}  // namespace
