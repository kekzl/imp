// Prompt logprobs (#2207): request parsing and host-side assembly of the vLLM and echo shapes.
// Row p of PromptLogprobs scores prompt[p + 1]; prompt[0] has no logprob (null in both shapes).

#include "prompt_logprobs.h"
#include "runtime/prompt_lp_chunk.h"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

const std::map<int32_t, std::string> kText = {{7, "a"},    {8, "b"},   {9, "c"},
                                              {10, "Hel"}, {11, "lo"}, {12, " world"}};

std::string token_text(int32_t id) {
    const auto it = kText.find(id);
    return it == kText.end() ? "?" : it->second;
}

// prompt {10, 11, 12}: row 0 scores 11 ("lo"), row 1 scores 12 (" world").
imp::PromptLogprobs two_rows() {
    imp::PromptLogprobs p;
    p.top_n = 2;
    p.rows = 2;
    p.token_lp = {-1.5f, -0.25f};
    p.rank = {3, 1};
    p.top_ids = {7, 8, 12, 9};
    p.top_lp = {-0.1f, -0.2f, -0.25f, -2.0f};
    return p;
}

const std::vector<int32_t> kPrompt = {10, 11, 12};

json body(const char* s) { return json::parse(s); }

}  // namespace

TEST(PromptLogprobsParse, AbsentAndNullAreOff) {
    PromptLogprobsRequest r;
    EXPECT_EQ(parse_prompt_logprobs(body("{}"), false, false, false, 0, r), "");
    EXPECT_EQ(r.prompt_logprobs, -1);
    EXPECT_EQ(r.engine_top_n, -1);
    EXPECT_EQ(parse_prompt_logprobs(body(R"({"prompt_logprobs": null})"), true, false, false, 0, r), "");
    EXPECT_EQ(r.engine_top_n, -1);
}

TEST(PromptLogprobsParse, RangeIsZeroToTwenty) {
    PromptLogprobsRequest r;
    EXPECT_EQ(parse_prompt_logprobs(body(R"({"prompt_logprobs": 0})"), false, false, false, 0, r), "");
    EXPECT_EQ(r.engine_top_n, 0);
    EXPECT_EQ(parse_prompt_logprobs(body(R"({"prompt_logprobs": 20})"), false, false, false, 0, r), "");
    EXPECT_EQ(r.engine_top_n, 20);
    for (const char* bad :
         {R"({"prompt_logprobs": 21})", R"({"prompt_logprobs": -1})", R"({"prompt_logprobs": true})",
          R"({"prompt_logprobs": 1.5})", R"({"prompt_logprobs": "3"})"}) {
        const std::string err = parse_prompt_logprobs(body(bad), false, false, false, 0, r);
        EXPECT_NE(err.find("\"prompt_logprobs\" must be an integer in [0, 20]"), std::string::npos) << bad;
    }
}

TEST(PromptLogprobsParse, StreamIsRefusedForBothForms) {
    PromptLogprobsRequest r;
    EXPECT_NE(parse_prompt_logprobs(body(R"({"prompt_logprobs": 1})"), true, false, false, 0, r), "");
    EXPECT_NE(parse_prompt_logprobs(body("{}"), true, /*echo=*/true, /*logprobs=*/true, 0, r), "");
    // echo alone or logprobs alone stream as before
    EXPECT_EQ(parse_prompt_logprobs(body("{}"), true, true, false, 0, r), "");
    EXPECT_EQ(parse_prompt_logprobs(body("{}"), true, false, true, 3, r), "");
    EXPECT_EQ(r.engine_top_n, -1);
}

TEST(PromptLogprobsParse, EngineTopNIsTheLargerRequest) {
    PromptLogprobsRequest r;
    EXPECT_EQ(parse_prompt_logprobs(body("{}"), false, true, true, 3, r), "");
    EXPECT_TRUE(r.echo_logprobs);
    EXPECT_EQ(r.echo_top, 3);
    EXPECT_EQ(r.engine_top_n, 3);
    EXPECT_EQ(parse_prompt_logprobs(body(R"({"prompt_logprobs": 5})"), false, true, true, 2, r), "");
    EXPECT_EQ(r.engine_top_n, 5);
    EXPECT_EQ(r.echo_top, 2);
}

TEST(PromptLogprobsParse, CompletionsLogprobsIntegerOrBoolean) {
    bool lp = false;
    int top = -1;
    PromptLogprobsRequest r;
    EXPECT_EQ(parse_completions_logprobs(body(R"({"logprobs": 5})"), false, false, lp, top, r), "");
    EXPECT_TRUE(lp);
    EXPECT_EQ(top, 5);
    EXPECT_EQ(parse_completions_logprobs(body(R"({"logprobs": 0})"), false, false, lp, top, r), "");
    EXPECT_FALSE(lp);
    EXPECT_EQ(parse_completions_logprobs(body(R"({"logprobs": true, "top_logprobs": 50})"), false, false, lp,
                                         top, r),
              "");
    EXPECT_TRUE(lp);
    EXPECT_EQ(top, 20);
    EXPECT_NE(parse_completions_logprobs(body(R"({"logprobs": "5"})"), false, false, lp, top, r), "");
    // echo + logprobs: 3 alternatives per echoed prompt token
    EXPECT_EQ(parse_completions_logprobs(body(R"({"logprobs": 3})"), false, true, lp, top, r), "");
    EXPECT_EQ(r.engine_top_n, 3);
}

TEST(PromptLogprobsAssembly, CompleteNeedsEveryRow) {
    imp::PromptLogprobs p = two_rows();
    EXPECT_TRUE(prompt_logprobs_complete(p, 3));
    EXPECT_FALSE(prompt_logprobs_complete(p, 4));
    p.rows = 1;
    EXPECT_FALSE(prompt_logprobs_complete(p, 3));
    EXPECT_FALSE(prompt_logprobs_complete(imp::PromptLogprobs{}, 0));
    EXPECT_TRUE(prompt_logprobs_complete(imp::PromptLogprobs{}, 1));  // one-token prompt: zero rows
}

TEST(PromptLogprobsAssembly, VllmShapeFirstNullThenTargetPlusTopN) {
    const json a = vllm_prompt_logprobs_json(kPrompt, two_rows(), 2, token_text);
    ASSERT_EQ(a.size(), 3u);
    EXPECT_TRUE(a[0].is_null());
    // position 1 = token 11 scored by row 0; 11 is not in the top 2, so 3 entries
    ASSERT_EQ(a[1].size(), 3u);
    EXPECT_FLOAT_EQ(a[1]["11"]["logprob"].get<float>(), -1.5f);
    EXPECT_EQ(a[1]["11"]["rank"].get<int>(), 3);
    EXPECT_EQ(a[1]["11"]["decoded_token"].get<std::string>(), "lo");
    EXPECT_FLOAT_EQ(a[1]["7"]["logprob"].get<float>(), -0.1f);
    EXPECT_EQ(a[1]["7"]["rank"].get<int>(), 1);
    EXPECT_EQ(a[1]["8"]["rank"].get<int>(), 2);
    // position 2 = token 12 scored by row 1; 12 is its own top-1, so 2 entries
    ASSERT_EQ(a[2].size(), 2u);
    EXPECT_FLOAT_EQ(a[2]["12"]["logprob"].get<float>(), -0.25f);
    EXPECT_EQ(a[2]["12"]["rank"].get<int>(), 1);
    EXPECT_FLOAT_EQ(a[2]["9"]["logprob"].get<float>(), -2.0f);
}

TEST(PromptLogprobsAssembly, VllmLimitCutsAlternativesNotTheTarget) {
    const json a = vllm_prompt_logprobs_json(kPrompt, two_rows(), 0, token_text);
    ASSERT_EQ(a.size(), 3u);
    ASSERT_EQ(a[1].size(), 1u);
    EXPECT_TRUE(a[1].contains("11"));
    ASSERT_EQ(a[2].size(), 1u);
    EXPECT_TRUE(a[2].contains("12"));
}

TEST(PromptLogprobsAssembly, OneTokenPromptIsASingleNull) {
    const json a = vllm_prompt_logprobs_json({10}, imp::PromptLogprobs{}, 5, token_text);
    ASSERT_EQ(a.size(), 1u);
    EXPECT_TRUE(a[0].is_null());
}

TEST(PromptLogprobsAssembly, EchoShapePromptThenCompletionWithOffsets) {
    imp::TokenLogprobInfo out;
    out.text = "!";
    out.logprob = -0.5f;
    out.top.push_back({9, -0.5f, "!"});
    const std::string full = "Hello world!";
    const json lp = echo_completions_logprobs_json(kPrompt, two_rows(), 1, token_text, {out}, 1, full);

    ASSERT_EQ(lp["tokens"], json({"Hel", "lo", " world", "!"}));
    ASSERT_EQ(lp["token_logprobs"].size(), 4u);
    EXPECT_TRUE(lp["token_logprobs"][0].is_null());
    EXPECT_FLOAT_EQ(lp["token_logprobs"][1].get<float>(), -1.5f);
    EXPECT_FLOAT_EQ(lp["token_logprobs"][2].get<float>(), -0.25f);
    EXPECT_FLOAT_EQ(lp["token_logprobs"][3].get<float>(), -0.5f);
    EXPECT_EQ(lp["text_offset"], json({0, 3, 5, 11}));
    EXPECT_TRUE(lp["top_logprobs"][0].is_null());
    EXPECT_EQ(lp["top_logprobs"][1], json({{"a", -0.1f}}));
    EXPECT_EQ(lp["top_logprobs"][2], json({{" world", -0.25f}}));
    EXPECT_EQ(lp["top_logprobs"][3], json({{"!", -0.5f}}));
}

TEST(PromptLogprobsAssembly, EchoOffsetsStayInsideTheText) {
    // A decoded piece that is not in the text (e.g. an inserted BOS) keeps the offset where it is.
    const json lp = echo_completions_logprobs_json({7, 10, 11}, two_rows(), 0, token_text, {}, 0, "Hello");
    EXPECT_EQ(lp["text_offset"], json({0, 0, 3}));
}

TEST(PromptLogprobsAssembly, AttachRefusesAPartialResult) {
    imp::Request req;
    req.input_tokens = kPrompt;
    req.prompt_lp = two_rows();
    PromptLogprobsRequest plp;
    plp.prompt_logprobs = 2;
    plp.engine_top_n = 2;
    json choice = json::object();
    ASSERT_TRUE(attach_prompt_logprobs(choice, plp, req, token_text, 0, "Hello world"));
    EXPECT_EQ(choice["prompt_logprobs"].size(), 3u);
    EXPECT_FALSE(choice.contains("logprobs"));

    req.prompt_lp.rows = 1;
    json partial = json::object();
    EXPECT_FALSE(attach_prompt_logprobs(partial, plp, req, token_text, 0, "Hello world"));
    EXPECT_FALSE(partial.contains("prompt_logprobs"));

    PromptLogprobsRequest off;
    json untouched = json::object();
    EXPECT_TRUE(attach_prompt_logprobs(untouched, off, req, token_text, 0, ""));
    EXPECT_TRUE(untouched.empty());
}

// #2257: logits chunk rows = min(n_rows, 1024, (avail / 2) / (4 * vocab)).
TEST(PromptLogprobs, LogitsChunkRowsFormula) {
    constexpr int kV = 151936;
    constexpr size_t kRowBytes = sizeof(float) * kV;
    EXPECT_EQ(imp::prompt_lp_chunk_rows(2047, kV, size_t{32} << 30), imp::kPromptLpMaxChunkRows);
    EXPECT_EQ(imp::prompt_lp_chunk_rows(300, kV, size_t{32} << 30), 300);
    EXPECT_EQ(imp::prompt_lp_chunk_rows(2047, kV, 2 * 100 * kRowBytes), 100);
    EXPECT_EQ(imp::prompt_lp_chunk_rows(2047, kV, 2 * 100 * kRowBytes - 1), 99);
    EXPECT_EQ(imp::prompt_lp_chunk_rows(2047, kV, kRowBytes), 0);
    EXPECT_EQ(imp::prompt_lp_chunk_rows(0, kV, size_t{32} << 30), 0);
    EXPECT_EQ(imp::prompt_lp_chunk_rows(5, 0, size_t{32} << 30), 0);
}
