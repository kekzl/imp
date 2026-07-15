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

// ---- tool_choice conversion + parallel_tool_calls (issue #892) -------------

TEST(AnthropicToolChoice, AbsentLeavesToolChoiceUnset) {
    json oai = anthropic_to_openai_body(base_request());
    EXPECT_FALSE(oai.contains("tool_choice"));
    EXPECT_FALSE(oai.contains("parallel_tool_calls"));
}

TEST(AnthropicToolChoice, AutoNoneAnyMap) {
    json req = base_request();
    req["tool_choice"] = json{{"type", "auto"}};
    EXPECT_EQ(anthropic_to_openai_body(req)["tool_choice"], "auto");

    req["tool_choice"] = json{{"type", "none"}};
    EXPECT_EQ(anthropic_to_openai_body(req)["tool_choice"], "none");

    // Anthropic "any" (must call some tool) → OpenAI "required".
    req["tool_choice"] = json{{"type", "any"}};
    EXPECT_EQ(anthropic_to_openai_body(req)["tool_choice"], "required");
}

TEST(AnthropicToolChoice, NamedToolMapsToFunction) {
    json req = base_request();
    req["tool_choice"] = json{{"type", "tool"}, {"name", "get_weather"}};
    json tc = anthropic_to_openai_body(req)["tool_choice"];
    EXPECT_EQ(tc.value("type", ""), "function");
    EXPECT_EQ(tc["function"].value("name", ""), "get_weather");
}

TEST(AnthropicToolChoice, StringPassthrough) {
    // A client that already supplied the OpenAI string form is kept as-is.
    json req = base_request();
    req["tool_choice"] = "required";
    EXPECT_EQ(anthropic_to_openai_body(req)["tool_choice"], "required");
}

TEST(AnthropicToolChoice, DisableParallelToolUseSetsFlag) {
    json req = base_request();
    req["tool_choice"] = json{{"type", "auto"}, {"disable_parallel_tool_use", true}};
    json oai = anthropic_to_openai_body(req);
    EXPECT_EQ(oai["tool_choice"], "auto");
    EXPECT_FALSE(oai.value("parallel_tool_calls", true));  // mapped to false
}

TEST(AnthropicToolChoice, ParallelDefaultLeavesFlagUnset) {
    // Without disable_parallel_tool_use the flag must not be forced (default is
    // parallel-allowed on the OpenAI side).
    json req = base_request();
    req["tool_choice"] = json{{"type", "auto"}};
    EXPECT_FALSE(anthropic_to_openai_body(req).contains("parallel_tool_calls"));

    req["tool_choice"] = json{{"type", "auto"}, {"disable_parallel_tool_use", false}};
    EXPECT_FALSE(anthropic_to_openai_body(req).contains("parallel_tool_calls"));
}

// ---- message / content-block conversion ------------------------------------

// Small builders to keep the nested Anthropic payloads readable.
json tblk(const std::string& text) {
    return json{{"type", "text"}, {"text", text}};
}
json one_user(const json& content) {
    json r = base_request();
    r["messages"] = json::array({json{{"role", "user"}, {"content", content}}});
    return r;
}
json one_assistant(const json& content) {
    json r = base_request();
    r["messages"] = json::array({json{{"role", "assistant"}, {"content", content}}});
    return r;
}

TEST(AnthropicMessages, SystemStringPrependedAsSystemRole) {
    json req = base_request();
    req["system"] = "You are helpful.";
    json msgs = anthropic_to_openai_body(req)["messages"];
    ASSERT_FALSE(msgs.empty());
    EXPECT_EQ(msgs[0]["role"], "system");
    EXPECT_EQ(msgs[0]["content"], "You are helpful.");
}

TEST(AnthropicMessages, SystemBlockArrayFlattenedWithNewlines) {
    json req = base_request();
    req["system"] = json::array({tblk("A"), tblk("B")});
    json msgs = anthropic_to_openai_body(req)["messages"];
    EXPECT_EQ(msgs[0]["role"], "system");
    EXPECT_EQ(msgs[0]["content"], "A\nB");
}

TEST(AnthropicMessages, SingleUserTextBlockCollapsesToString) {
    json msgs = anthropic_to_openai_body(one_user(json::array({tblk("hi")})))["messages"];
    ASSERT_EQ(msgs.size(), 1u);
    EXPECT_EQ(msgs[0]["role"], "user");
    EXPECT_EQ(msgs[0]["content"], "hi");  // collapsed, not an array
}

TEST(AnthropicMessages, UserImageBase64BecomesDataUrl) {
    json src = json{{"type", "base64"}, {"media_type", "image/png"}, {"data", "AAA"}};
    json img = json{{"type", "image"}, {"source", src}};
    json oai = anthropic_to_openai_body(one_user(json::array({tblk("look"), img})));
    json content = oai["messages"][0]["content"];
    ASSERT_TRUE(content.is_array());
    ASSERT_EQ(content.size(), 2u);
    EXPECT_EQ(content[0]["type"], "text");
    EXPECT_EQ(content[1]["type"], "image_url");
    EXPECT_EQ(content[1]["image_url"]["url"], "data:image/png;base64,AAA");
}

TEST(AnthropicMessages, UserImageUrlSourcePassedThrough) {
    json img = json{{"type", "image"}, {"source", json{{"type", "url"}, {"url", "https://x/y.png"}}}};
    json content = anthropic_to_openai_body(one_user(json::array({img})))["messages"][0]["content"];
    ASSERT_TRUE(content.is_array());
    EXPECT_EQ(content[0]["type"], "image_url");
    EXPECT_EQ(content[0]["image_url"]["url"], "https://x/y.png");
}

TEST(AnthropicMessages, AssistantToolUseBecomesToolCalls) {
    json tu = json{{"type", "tool_use"}, {"id", "tu_1"}, {"name", "foo"}, {"input", json{{"a", 1}}}};
    json msg = anthropic_to_openai_body(one_assistant(json::array({tblk("ok"), tu})))["messages"][0];
    EXPECT_EQ(msg["role"], "assistant");
    EXPECT_EQ(msg["content"], "ok");
    ASSERT_TRUE(msg.contains("tool_calls"));
    ASSERT_EQ(msg["tool_calls"].size(), 1u);
    EXPECT_EQ(msg["tool_calls"][0]["id"], "tu_1");
    EXPECT_EQ(msg["tool_calls"][0]["type"], "function");
    EXPECT_EQ(msg["tool_calls"][0]["function"]["name"], "foo");
    // arguments is a serialized JSON string — parse it back to compare stably.
    std::string raw_args = msg["tool_calls"][0]["function"]["arguments"].get<std::string>();
    EXPECT_EQ(json::parse(raw_args), (json{{"a", 1}}));
}

TEST(AnthropicMessages, AssistantOnlyToolUseHasNullContent) {
    json tu = json{{"type", "tool_use"}, {"id", "tu_2"}, {"name", "f"}, {"input", json::object()}};
    json msg = anthropic_to_openai_body(one_assistant(json::array({tu})))["messages"][0];
    EXPECT_TRUE(msg["content"].is_null());
    EXPECT_EQ(msg["tool_calls"][0]["id"], "tu_2");
}

TEST(AnthropicMessages, AssistantThinkingBecomesReasoningContent) {
    json think = json{{"type", "thinking"}, {"thinking", "hmm"}};
    json msg = anthropic_to_openai_body(one_assistant(json::array({think, tblk("answer")})))["messages"][0];
    EXPECT_EQ(msg["content"], "answer");
    EXPECT_EQ(msg.value("reasoning_content", ""), "hmm");
}

TEST(AnthropicMessages, ToolResultBecomesToolRoleMessage) {
    json tr = json{{"type", "tool_result"}, {"tool_use_id", "tu_1"}, {"content", "42"}};
    json msgs = anthropic_to_openai_body(one_user(json::array({tr})))["messages"];
    ASSERT_EQ(msgs.size(), 1u);
    EXPECT_EQ(msgs[0]["role"], "tool");
    EXPECT_EQ(msgs[0]["tool_call_id"], "tu_1");
    EXPECT_EQ(msgs[0]["content"], "42");
}

TEST(AnthropicMessages, ToolResultArrayContentFlattened) {
    json tr = json{{"type", "tool_result"},
                   {"tool_use_id", "tu_9"},
                   {"content", json::array({tblk("a"), tblk("b")})}};
    json msgs = anthropic_to_openai_body(one_user(json::array({tr})))["messages"];
    EXPECT_EQ(msgs[0]["role"], "tool");
    EXPECT_EQ(msgs[0]["content"], "a\nb");
}

// #1006: images inside tool_result blocks (screenshot-returning tools) must
// not be dropped — they are re-homed onto a trailing user turn (the
// multimodal path the engine serves) with a marker in the tool body.
TEST(AnthropicMessages, ToolResultImageRehomedToUserTurn) {
    json img = json{{"type", "image"},
                    {"source", json{{"type", "base64"},
                                    {"media_type", "image/png"},
                                    {"data", "aGk="}}}};
    json tr = json{{"type", "tool_result"},
                   {"tool_use_id", "tu_img"},
                   {"content", json::array({tblk("screenshot taken"), img})}};
    json msgs = anthropic_to_openai_body(one_user(json::array({tr})))["messages"];
    ASSERT_EQ(msgs.size(), 2u);
    EXPECT_EQ(msgs[0]["role"], "tool");
    const std::string body = msgs[0]["content"].get<std::string>();
    EXPECT_NE(body.find("screenshot taken"), std::string::npos);
    EXPECT_NE(body.find("1 image(s)"), std::string::npos) << "marker must tie image to result";
    EXPECT_EQ(msgs[1]["role"], "user");
    ASSERT_TRUE(msgs[1]["content"].is_array());
    bool has_image = false;
    for (const auto& part : msgs[1]["content"])
        if (part.value("type", "") == "image_url")
            has_image = true;
    EXPECT_TRUE(has_image) << "the image must survive on the user turn";
}

// ---- openai_to_anthropic_response (reverse transform) ----------------------

// Wrap an OpenAI assistant message + finish_reason into a chat.completion.
json oai_response(json message, const std::string& finish = "stop") {
    json choice = json{{"index", 0}, {"message", std::move(message)}, {"finish_reason", finish}};
    return json{{"id", "chatcmpl-abc"}, {"choices", json::array({choice})}};
}

TEST(AnthropicResponse, ErrorEnvelopeFlipsType) {
    json err = json{{"error", json{{"type", "invalid_request_error"}, {"message", "bad"}}}};
    json out = openai_to_anthropic_response(err, "claude-x");
    EXPECT_EQ(out["type"], "error");
    EXPECT_EQ(out["error"]["type"], "invalid_request_error");
    EXPECT_EQ(out["error"]["message"], "bad");
}

TEST(AnthropicResponse, TextContentBecomesTextBlock) {
    json out = openai_to_anthropic_response(oai_response(json{{"content", "hello"}}), "claude-x");
    EXPECT_EQ(out["type"], "message");
    EXPECT_EQ(out["role"], "assistant");
    EXPECT_EQ(out["model"], "claude-x");
    ASSERT_EQ(out["content"].size(), 1u);
    EXPECT_EQ(out["content"][0]["type"], "text");
    EXPECT_EQ(out["content"][0]["text"], "hello");
    EXPECT_EQ(out["stop_reason"], "end_turn");
}

TEST(AnthropicResponse, ThinkingEmittedBeforeText) {
    json msg = json{{"reasoning_content", "hmm"}, {"content", "answer"}};
    json content = openai_to_anthropic_response(oai_response(msg), "m")["content"];
    ASSERT_EQ(content.size(), 2u);
    EXPECT_EQ(content[0]["type"], "thinking");
    EXPECT_EQ(content[0]["thinking"], "hmm");
    EXPECT_EQ(content[1]["type"], "text");
}

TEST(AnthropicResponse, EmptyTextProducesNoBlock) {
    json content = openai_to_anthropic_response(oai_response(json{{"content", ""}}), "m")["content"];
    EXPECT_TRUE(content.empty());
}

TEST(AnthropicResponse, ToolCallsBecomeToolUse) {
    json fn = json{{"name", "foo"}, {"arguments", "{\"a\":1}"}};
    json tc = json{{"id", "call_imp_7"}, {"type", "function"}, {"function", fn}};
    json msg = json{{"content", nullptr}, {"tool_calls", json::array({tc})}};
    json out = openai_to_anthropic_response(oai_response(msg, "tool_calls"), "m");
    EXPECT_EQ(out["stop_reason"], "tool_use");
    ASSERT_EQ(out["content"].size(), 1u);
    EXPECT_EQ(out["content"][0]["type"], "tool_use");
    EXPECT_EQ(out["content"][0]["id"], "toolu_7");  // call_imp_ -> toolu_
    EXPECT_EQ(out["content"][0]["name"], "foo");
    EXPECT_EQ(out["content"][0]["input"], (json{{"a", 1}}));
}

TEST(AnthropicResponse, MalformedToolArgsBecomeEmptyObject) {
    json fn = json{{"name", "f"}, {"arguments", "{not json"}};
    json tc = json{{"id", "call_imp_1"}, {"function", fn}};
    json msg = json{{"tool_calls", json::array({tc})}};
    json out = openai_to_anthropic_response(oai_response(msg, "tool_calls"), "m");
    EXPECT_EQ(out["content"][0]["input"], json::object());
}

TEST(AnthropicResponse, FinishReasonMapping) {
    auto stop_for = [](const std::string& finish) {
        return openai_to_anthropic_response(oai_response(json{{"content", "x"}}, finish), "m")
            .value("stop_reason", "");
    };
    EXPECT_EQ(stop_for("stop"), "end_turn");
    EXPECT_EQ(stop_for("length"), "max_tokens");
    EXPECT_EQ(stop_for("tool_calls"), "tool_use");
    EXPECT_EQ(stop_for("cancelled"), "end_turn");
    EXPECT_EQ(stop_for("content_filter"), "content_filter");  // unknown → passthrough
}

TEST(AnthropicResponse, UsageSplitsCacheReadFromInput) {
    json details = json{{"cached_tokens", 30}, {"cache_creation_tokens", 5}};
    json usage = json{{"prompt_tokens", 100}, {"completion_tokens", 20}, {"prompt_tokens_details", details}};
    json oai = oai_response(json{{"content", "x"}});
    oai["usage"] = usage;
    json u = openai_to_anthropic_response(oai, "m")["usage"];
    EXPECT_EQ(u["input_tokens"], 70);  // prompt - cached
    EXPECT_EQ(u["cache_read_input_tokens"], 30);
    EXPECT_EQ(u["cache_creation_input_tokens"], 5);
    EXPECT_EQ(u["output_tokens"], 20);
}

TEST(AnthropicResponse, UsageCachedClampedToPrompt) {
    json details = json{{"cached_tokens", 500}};  // absurdly larger than prompt
    json usage = json{{"prompt_tokens", 40}, {"completion_tokens", 0}, {"prompt_tokens_details", details}};
    json oai = oai_response(json{{"content", "x"}});
    oai["usage"] = usage;
    json u = openai_to_anthropic_response(oai, "m")["usage"];
    EXPECT_EQ(u["input_tokens"], 0);
    EXPECT_EQ(u["cache_read_input_tokens"], 40);  // clamped to prompt_tokens
}

TEST(AnthropicResponse, ChatcmplIdRewrittenToMsg) {
    json out = openai_to_anthropic_response(oai_response(json{{"content", "x"}}), "m");
    // "chatcmpl-abc" -> "msg_-abc"
    EXPECT_EQ(out["id"].get<std::string>().rfind("msg_", 0), 0u);
}

}  // namespace
