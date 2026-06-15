// CPU unit tests for the Anthropic /v1/messages transforms (anthropic.cpp) —
// cache_control mapping and cache usage accounting (issue #522 item 1).
//
// cache_control contract: ANY cache_control marker in the request (system
// blocks, message content blocks, tool definitions) sets the internal
// "cache_prompt" flag on the converted OpenAI body, which the server maps to
// prompt-KV pinning. Position is irrelevant — imp's prefix cache is
// block-granular and automatic; the marker only requests eviction protection.

#include "anthropic.h"

#include <gtest/gtest.h>

namespace {

using imp_server::anthropic::anthropic_to_openai_body;
using imp_server::anthropic::openai_to_anthropic_response;
using json = nlohmann::json;

json base_request() {
    return json{
        {"model", "claude-x"},
        {"max_tokens", 64},
        {"messages", json::array({json{{"role", "user"}, {"content", "hi"}}})},
    };
}

TEST(AnthropicCacheControl, NoMarkerNoCachePrompt) {
    json oai = anthropic_to_openai_body(base_request());
    EXPECT_FALSE(oai.contains("cache_prompt"));
}

TEST(AnthropicCacheControl, SystemBlockMarkerSetsCachePrompt) {
    json req = base_request();
    req["system"] = json::array({
        json{{"type", "text"}, {"text", "You are helpful."}, {"cache_control", json{{"type", "ephemeral"}}}},
    });
    json oai = anthropic_to_openai_body(req);
    EXPECT_TRUE(oai.value("cache_prompt", false));
}

TEST(AnthropicCacheControl, MessageContentBlockMarkerSetsCachePrompt) {
    json req = base_request();
    req["messages"] = json::array({
        json{{"role", "user"},
             {"content", json::array({
                             json{{"type", "text"},
                                  {"text", "long context"},
                                  {"cache_control", json{{"type", "ephemeral"}}}},
                         })}},
    });
    json oai = anthropic_to_openai_body(req);
    EXPECT_TRUE(oai.value("cache_prompt", false));
}

TEST(AnthropicCacheControl, ToolDefinitionMarkerSetsCachePrompt) {
    json req = base_request();
    req["tools"] = json::array({
        json{{"name", "get_weather"},
             {"description", "d"},
             {"input_schema", json{{"type", "object"}}},
             {"cache_control", json{{"type", "ephemeral"}}}},
    });
    json oai = anthropic_to_openai_body(req);
    EXPECT_TRUE(oai.value("cache_prompt", false));
}

TEST(AnthropicCacheControl, PlainStringSystemNoCachePrompt) {
    json req = base_request();
    req["system"] = "plain string system";
    json oai = anthropic_to_openai_body(req);
    EXPECT_FALSE(oai.contains("cache_prompt"));
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

TEST(AnthropicCacheUsage, CacheReadAndCreationMapped) {
    json oai = oai_response_with_usage(json{
        {"prompt_tokens", 100},
        {"completion_tokens", 5},
        {"total_tokens", 105},
        {"prompt_tokens_details", json{{"cached_tokens", 32}, {"cache_creation_tokens", 64}}},
    });
    json anth = openai_to_anthropic_response(oai, "claude-x");
    const auto& u = anth["usage"];
    // Anthropic splits the prompt: input excludes cache reads.
    EXPECT_EQ(u.value("input_tokens", -1), 68);
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

// --- extended-thinking control ---------------------------------------------
// Anthropic carries thinking in a `thinking` object
// ({type:"enabled",budget_tokens:N} | {type:"disabled"}); imp's orchestrator
// reads `enable_thinking` (bool) + `think_budget` on the OpenAI body. Note imp's
// `think_budget` is a FRACTION of max_tokens (default 0.5), whereas Anthropic's
// `budget_tokens` is absolute — so we map it to budget_tokens/max_tokens,
// clamped to [0,1]. Without this mapping /v1/messages on a think-model could
// never be told NOT to reason (the request's intent was silently dropped).

TEST(AnthropicThinking, NoFieldLeavesThinkingUnset) {
    json oai = anthropic_to_openai_body(base_request());
    EXPECT_FALSE(oai.contains("enable_thinking"));
    EXPECT_FALSE(oai.contains("think_budget"));
}

TEST(AnthropicThinking, DisabledMapsToEnableThinkingFalseAndZeroBudget) {
    json req = base_request();
    req["thinking"] = json{{"type", "disabled"}};
    json oai = anthropic_to_openai_body(req);
    ASSERT_TRUE(oai.contains("enable_thinking"));
    EXPECT_TRUE(oai["enable_thinking"].is_boolean());
    EXPECT_FALSE(oai.value("enable_thinking", true));
    // Both signals are required for suppress_thinking — see anthropic.cpp.
    EXPECT_FLOAT_EQ(oai.value("think_budget", -1.0f), 0.0f);
}

TEST(AnthropicThinking, EnabledMapsToTrueAndFractionalBudget) {
    json req = base_request();
    req["max_tokens"] = 1024;
    req["thinking"] = json{{"type", "enabled"}, {"budget_tokens", 256}};
    json oai = anthropic_to_openai_body(req);
    EXPECT_TRUE(oai.value("enable_thinking", false));
    EXPECT_FLOAT_EQ(oai.value("think_budget", -1.0f), 0.25f);  // 256 / 1024
}

TEST(AnthropicThinking, EnabledBudgetClampedToOne) {
    json req = base_request();  // max_tokens = 64
    req["thinking"] = json{{"type", "enabled"}, {"budget_tokens", 4096}};
    json oai = anthropic_to_openai_body(req);
    EXPECT_TRUE(oai.value("enable_thinking", false));
    EXPECT_FLOAT_EQ(oai.value("think_budget", -1.0f), 1.0f);  // 4096/64 clamped
}

TEST(AnthropicThinking, EnabledWithoutBudgetSetsNoBudget) {
    json req = base_request();
    req["thinking"] = json{{"type", "enabled"}};
    json oai = anthropic_to_openai_body(req);
    EXPECT_TRUE(oai.value("enable_thinking", false));
    EXPECT_FALSE(oai.contains("think_budget"));
}

TEST(AnthropicThinking, UnknownTypeOrMalformedIsIgnored) {
    json req = base_request();
    req["thinking"] = json{{"type", "weird"}};
    EXPECT_FALSE(anthropic_to_openai_body(req).contains("enable_thinking"));

    json req2 = base_request();
    req2["thinking"] = "not-an-object";
    EXPECT_FALSE(anthropic_to_openai_body(req2).contains("enable_thinking"));
}

}  // namespace
