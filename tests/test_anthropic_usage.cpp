// CPU unit tests for Anthropic /v1/messages usage accounting (anthropic.h, anthropic.cpp): the three
// prompt fields are disjoint (#3055), cache read/creation and the speculation counters pass through,
// reasoning tokens become output_tokens_details.

#include "anthropic.h"
#include "spec_usage_keys.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using imp_server::anthropic::openai_to_anthropic_response;
using json = nlohmann::ordered_json;

TEST(AnthropicCacheUsage, SplitSumsToPromptOnEveryTurnShape) {
    struct Case {
        int prompt, cached, creation, input, read, written;
    };
    const Case cases[] = {
        {1550, 0, 1536, 14, 0, 1536},    // cold pinned turn: block-rounded write, tail stays input
        {1600, 1536, 48, 16, 1536, 48},  // warm turn
        {100, 0, 0, 100, 0, 0},          // nothing pinned
        {64, 64, 0, 0, 64, 0},           // full hit
        {40, 500, 0, 0, 40, 0},          // cached past the prompt
        {100, 90, 50, 0, 90, 10},        // creation past what is left
        {100, -1, -1, 100, 0, 0},        // negative counters
    };
    for (const Case& c : cases) {
        const auto s = imp_server::anthropic::split_prompt_usage(c.prompt, c.cached, c.creation);
        EXPECT_EQ(s.input, c.input) << c.prompt << "/" << c.cached << "/" << c.creation;
        EXPECT_EQ(s.cache_read, c.read) << c.prompt << "/" << c.cached << "/" << c.creation;
        EXPECT_EQ(s.cache_creation, c.written) << c.prompt << "/" << c.cached << "/" << c.creation;
        EXPECT_EQ(s.input + s.cache_read + s.cache_creation, c.prompt);
    }
}

// --- usage accounting -------------------------------------------------------

json oai_response_with_usage(json usage) {
    return json{
        {"id", "chatcmpl-1"},
        {"created", 1},
        {"choices", json::array({json{
                        {"index", 0},
                        {"message", json{{"role", "assistant"}, {"content", "ok"}}},
                        {"finish_reason", "stop"},
                    }})},
        {"usage", std::move(usage)},
    };
}

// C-6: the per-request speculation counters reach the Anthropic usage as
// top-level vendor-prefixed keys, and only when the chat usage carried them.
TEST(AnthropicCacheUsage, SpecCountersPassThrough) {
    json oai = oai_response_with_usage(json{
        {"prompt_tokens", 10},
        {"completion_tokens", 5},
        {"total_tokens", 15},
        {"completion_tokens_details",
         json{{"imp_spec_drafted", 40}, {"imp_spec_accepted", 31}, {"imp_spec_verify_steps", 5}}},
    });
    json anth = openai_to_anthropic_response(oai, "claude-x");
    EXPECT_EQ(anth["usage"].value("imp_spec_drafted", -1), 40);
    EXPECT_EQ(anth["usage"].value("imp_spec_accepted", -1), 31);
    EXPECT_EQ(anth["usage"].value("imp_spec_verify_steps", -1), 5);
    json plain = openai_to_anthropic_response(
        oai_response_with_usage(json{{"prompt_tokens", 10}, {"completion_tokens", 5}, {"total_tokens", 15}}),
        "claude-x");
    EXPECT_FALSE(plain["usage"].contains("imp_spec_drafted"));
}

// Asserts every key the chat shape writes survives the shim, not a hand-copied subset: a
// 3-name literal list once missed imp_spec_emitted and the decline reason on non-chat
// endpoints. Now driven by one shared table, tools/imp-server/spec_usage_keys.h.
TEST(AnthropicCacheUsage, EverySpecUsageKeyPassesThrough) {
    json details = json::object();
    for (const char* k : imp_server::kSpecUsageKeys)
        details[k] = "sentinel";
    json anth = openai_to_anthropic_response(oai_response_with_usage(
                                                 json{{"prompt_tokens", 10},
                                                      {"completion_tokens", 5},
                                                      {"total_tokens", 15},
                                                      {"completion_tokens_details", details}}),
                                             "claude-x");
    for (const char* k : imp_server::kSpecUsageKeys)
        EXPECT_EQ(anth["usage"].value(k, std::string("MISSING")), "sentinel")
            << k << " was dropped by the Anthropic transform";
}

// The decline is the one a caller actually needs: it is the ONLY thing that
// distinguishes "the server declined your MTP request" from "speculation ran
// and drafted nothing", and it arrives with zero counters.
TEST(AnthropicCacheUsage, DeclineReachesTheAnthropicUsage) {
    json anth = openai_to_anthropic_response(oai_response_with_usage(
                                                 json{{"prompt_tokens", 10},
                                                      {"completion_tokens", 5},
                                                      {"total_tokens", 15},
                                                      {"completion_tokens_details",
                                                       json{{"imp_spec_drafted", 0},
                                                            {"imp_spec_emitted", 0},
                                                            {"imp_spec_declined", "mtp_head_not_loaded"}}}}),
                                             "claude-x");
    EXPECT_EQ(anth["usage"].value("imp_spec_declined", std::string()), "mtp_head_not_loaded");
    EXPECT_EQ(anth["usage"].value("imp_spec_emitted", -1), 0);
}

// Reads the OpenAI body the chat handler produced (both directions): guards against a mutant
// that emits the exhaustion detail unconditionally instead of only when the answer was lost.
TEST(AnthropicReasoningReport, ReasoningTokensBecomeOutputTokensDetails) {
    json oai = oai_response_with_usage(json{
        {"prompt_tokens", 10},
        {"completion_tokens", 40},
        {"total_tokens", 50},
        {"completion_tokens_details", json{{"reasoning_tokens", 33}}},
    });
    json anth = openai_to_anthropic_response(oai, "claude-x");
    // Anthropic's own name for the number, so it is not imp-namespaced.
    EXPECT_EQ(anth["usage"]["output_tokens_details"].value("reasoning_tokens", -1), 33);
    json plain = openai_to_anthropic_response(
        oai_response_with_usage(json{{"prompt_tokens", 10}, {"completion_tokens", 5}, {"total_tokens", 15}}),
        "claude-x");
    EXPECT_FALSE(plain["usage"].contains("output_tokens_details"));
}

TEST(AnthropicReasoningReport, FinishDetailIsCopiedOnlyWhenTheChoiceCarriesIt) {
    json oai = oai_response_with_usage(json{{"prompt_tokens", 10}, {"completion_tokens", 5}});
    oai["choices"][0]["imp_finish_detail"] = "reasoning_budget_exhausted";
    json anth = openai_to_anthropic_response(oai, "claude-x");
    EXPECT_EQ(anth.value("imp_finish_detail", ""), "reasoning_budget_exhausted");
    // stop_reason keeps the Anthropic enum: a client switching on it sees no
    // new member.
    EXPECT_EQ(anth.value("stop_reason", ""), "end_turn");

    json answered = oai_response_with_usage(json{{"prompt_tokens", 10}, {"completion_tokens", 5}});
    EXPECT_FALSE(openai_to_anthropic_response(answered, "claude-x").contains("imp_finish_detail"));
}

TEST(AnthropicCacheUsage, CacheReadAndCreationMapped) {
    json oai = oai_response_with_usage(json{
        {"prompt_tokens", 100},
        {"completion_tokens", 5},
        {"total_tokens", 105},
        {"prompt_tokens_details", json{{"cached_tokens", 32}, {"cache_creation_tokens", 64}}},
    });
    json anth = openai_to_anthropic_response(oai, "claude-x");
    const auto& u = anth["usage"];
    // Disjoint fields: input excludes cache reads and cache writes, the three sum to the prompt (#3055).
    EXPECT_EQ(u.value("input_tokens", -1), 4);
    EXPECT_EQ(u.value("cache_read_input_tokens", -1), 32);
    EXPECT_EQ(u.value("cache_creation_input_tokens", -1), 64);
    EXPECT_EQ(u.value("output_tokens", -1), 5);
}

TEST(AnthropicCacheUsage, NoDetailsMeansZeroCacheFields) {
    json oai = oai_response_with_usage(json{
        {"prompt_tokens", 10},
        {"completion_tokens", 2},
        {"total_tokens", 12},
    });
    json anth = openai_to_anthropic_response(oai, "claude-x");
    const auto& u = anth["usage"];
    EXPECT_EQ(u.value("input_tokens", -1), 10);
    EXPECT_EQ(u.value("cache_read_input_tokens", -1), 0);
    EXPECT_EQ(u.value("cache_creation_input_tokens", -1), 0);
}

}  // namespace
