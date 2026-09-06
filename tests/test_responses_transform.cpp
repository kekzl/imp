// Unit tests for tools/imp-server/responses.{h,cpp} — the /v1/responses
// (OpenAI Responses API) <-> chat-completions transforms. Oracle:
// hand-written request/response pairs per the published API shapes (what the
// OpenAI Agents SDK and Codex CLI send/expect) — no imp-vs-imp.

#include <gtest/gtest.h>
#include "responses.h"

#include <stdexcept>

using imp_server::responses::json;
using imp_server::responses::openai_to_responses_response;
using imp_server::responses::responses_to_openai_body;

// ---------------------------------------------------------------------------
// Request transform
// ---------------------------------------------------------------------------

TEST(ResponsesTransform, StringInputBecomesUserMessage) {
    json body = {{"model", "m"}, {"input", "hello"}, {"instructions", "be brief"}};
    json oai = responses_to_openai_body(body);
    ASSERT_EQ(oai["messages"].size(), 2u);
    EXPECT_EQ(oai["messages"][0]["role"], "system");
    EXPECT_EQ(oai["messages"][0]["content"], "be brief");
    EXPECT_EQ(oai["messages"][1]["role"], "user");
    EXPECT_EQ(oai["messages"][1]["content"], "hello");
    EXPECT_EQ(oai["model"], "m");
}

TEST(ResponsesTransform, ItemArrayWithToolRoundTrip) {
    // The Agents SDK transcript shape: message items (content parts),
    // a replayed function_call + its function_call_output, a reasoning item.
    json body = {
        {"model", "m"},
        {"input",
         json::array(
             {{{"role", "user"},
               {"content", json::array({{{"type", "input_text"}, {"text", "weather?"}}})}},
              {{"type", "reasoning"}, {"id", "rs_1"}, {"summary", json::array()}},
              {{"type", "function_call"},
               {"call_id", "call_1"},
               {"name", "get_weather"},
               {"arguments", "{\"city\":\"Paris\"}"}},
              {{"type", "function_call_output"}, {"call_id", "call_1"}, {"output", "sunny"}}})}};
    json oai = responses_to_openai_body(body);
    ASSERT_EQ(oai["messages"].size(), 3u);  // reasoning skipped
    EXPECT_EQ(oai["messages"][0]["role"], "user");
    EXPECT_EQ(oai["messages"][0]["content"], "weather?");
    EXPECT_EQ(oai["messages"][1]["role"], "assistant");
    ASSERT_TRUE(oai["messages"][1].contains("tool_calls"));
    EXPECT_EQ(oai["messages"][1]["tool_calls"][0]["id"], "call_1");
    EXPECT_EQ(oai["messages"][1]["tool_calls"][0]["function"]["name"], "get_weather");
    EXPECT_EQ(oai["messages"][2]["role"], "tool");
    EXPECT_EQ(oai["messages"][2]["tool_call_id"], "call_1");
    EXPECT_EQ(oai["messages"][2]["content"], "sunny");
}

TEST(ResponsesTransform, FlatToolsBecomeNested) {
    json body = {{"model", "m"},
                 {"input", "x"},
                 {"tools", json::array({{{"type", "function"},
                                         {"name", "f"},
                                         {"description", "d"},
                                         {"parameters", {{"type", "object"}}},
                                         {"strict", true}}})},
                 {"tool_choice", json{{"type", "function"}, {"name", "f"}}}};
    json oai = responses_to_openai_body(body);
    ASSERT_EQ(oai["tools"].size(), 1u);
    EXPECT_EQ(oai["tools"][0]["type"], "function");
    EXPECT_EQ(oai["tools"][0]["function"]["name"], "f");
    EXPECT_EQ(oai["tools"][0]["function"]["description"], "d");
    EXPECT_EQ(oai["tools"][0]["function"]["strict"], true);
    EXPECT_EQ(oai["tool_choice"]["function"]["name"], "f");
}

TEST(ResponsesTransform, TextFormatAndKnobs) {
    json body = {
        {"model", "m"},
        {"input", "x"},
        {"max_output_tokens", 321},
        {"temperature", 0.3},
        {"reasoning", {{"effort", "low"}}},
        {"priority", 3},
        {"speculative", false},
        {"text",
         {{"format",
           {{"type", "json_schema"}, {"name", "out"}, {"schema", {{"type", "object"}}}, {"strict", true}}}}}};
    json oai = responses_to_openai_body(body);
    EXPECT_EQ(oai["max_tokens"], 321);
    EXPECT_DOUBLE_EQ(oai["temperature"].get<double>(), 0.3);
    EXPECT_DOUBLE_EQ(oai["think_budget"].get<double>(), 0.25);
    // Both imp extensions reach the chat body (C-9: `speculative` used to be
    // dropped while `priority` was bridged).
    EXPECT_EQ(oai["priority"], 3);
    EXPECT_EQ(oai["speculative"], false);
    EXPECT_EQ(oai["response_format"]["type"], "json_schema");
    EXPECT_EQ(oai["response_format"]["json_schema"]["name"], "out");
    EXPECT_EQ(oai["response_format"]["json_schema"]["strict"], true);
}

// `regex` and `grammar` are imp's own response_format extensions and both work
// on /v1/chat/completions. This transform carried neither, so the SAME request
// was constrained on one endpoint and free text on the other, at 200 with no
// reason. Measured on the model-less binary: `{"type":"nonsense_value"}` was 400
// on /v1/chat/completions (naming the known set) and 503 on /v1/responses, i.e.
// the chat parser's own check never saw the field the transform had dropped.
TEST(ResponsesTransform, RegexAndGrammarFormatsReachTheChatBody) {
    json rx = responses_to_openai_body(
        json{{"model", "m"}, {"input", "x"}, {"text", {{"format", {{"type", "regex"}, {"regex", "a+"}}}}}});
    ASSERT_TRUE(rx.contains("response_format")) << "regex was dropped, so nothing constrained the reply";
    EXPECT_EQ(rx["response_format"]["type"], "regex");
    EXPECT_EQ(rx["response_format"]["regex"], "a+");

    json gr = responses_to_openai_body(
        json{{"model", "m"},
             {"input", "x"},
             {"text", {{"format", {{"type", "grammar"}, {"grammar", "root ::= \"a\""}}}}}});
    ASSERT_TRUE(gr.contains("response_format"));
    EXPECT_EQ(gr["response_format"]["type"], "grammar");
    EXPECT_EQ(gr["response_format"]["grammar"], "root ::= \"a\"");
}

// "text" and an absent format both mean unconstrained, which chat/completions
// expresses by having no response_format at all. Refusing here would break the
// SDKs that always send the field.
TEST(ResponsesTransform, PlainTextFormatWritesNoResponseFormat) {
    json oai = responses_to_openai_body(
        json{{"model", "m"}, {"input", "x"}, {"text", {{"format", {{"type", "text"}}}}}});
    EXPECT_FALSE(oai.contains("response_format"));
    json bare = responses_to_openai_body(json{{"model", "m"}, {"input", "x"}});
    EXPECT_FALSE(bare.contains("response_format"));
}

TEST(ResponsesTransform, UnknownTextFormatIsRefused) {
    EXPECT_THROW(responses_to_openai_body(json{{"model", "m"},
                                               {"input", "x"},
                                               {"text", {{"format", {{"type", "nonsense_value"}}}}}}),
                 std::invalid_argument);
}

// A tool_choice object the transform cannot map wrote nothing at all, so the
// chat parser applied its own default "auto". A caller demanding a call -
// `{"type":"allowed_tools","mode":"required"}`, the shape the Agents SDK emits -
// got a fluent 200 with no call. `validate_tool_choice` could not catch it: it
// runs on the transformed body, where the field no longer existed.
TEST(ResponsesTransform, UnmappableToolChoiceIsRefused) {
    auto with = [](const json& tc) {
        return json{{"model", "m"},
                    {"input", "x"},
                    {"tools", json::array({{{"type", "function"}, {"name", "f"}}})},
                    {"tool_choice", tc}};
    };
    for (const char* t : {"allowed_tools", "mcp", "custom", "file_search", "nonsense_value"})
        EXPECT_THROW(responses_to_openai_body(with(json{{"type", t}})), std::invalid_argument) << t;

    // The two shapes that do map must keep working.
    EXPECT_EQ(responses_to_openai_body(with(json("required")))["tool_choice"], "required");
    EXPECT_EQ(responses_to_openai_body(
                  with(json{{"type", "function"}, {"name", "f"}}))["tool_choice"]["function"]["name"],
              "f");
}

TEST(ResponsesTransform, StatefulFieldsRejected) {
    EXPECT_THROW(
        responses_to_openai_body(json{{"input", "x"}, {"previous_response_id", "resp_1"}}),
        std::invalid_argument);
    EXPECT_THROW(responses_to_openai_body(json{{"input", "x"}, {"store", true}}),
                 std::invalid_argument);
    // store=false (what Codex/Agents SDK send) is fine.
    EXPECT_NO_THROW(responses_to_openai_body(json{{"input", "x"}, {"store", false}}));
}

TEST(ResponsesTransform, UnsupportedInputPartsRejected) {
    json body = {{"input", json::array({{{"role", "user"},
                                         {"content", json::array({{{"type", "input_image"},
                                                                   {"image_url", "http://x"}}})}}})}};
    EXPECT_THROW(responses_to_openai_body(body), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Response transform
// ---------------------------------------------------------------------------

static json make_oai(const char* content, json tool_calls = nullptr,
                     const char* reasoning = nullptr, const char* finish = "stop") {
    json msg = {{"role", "assistant"}};
    if (content)
        msg["content"] = content;
    if (!tool_calls.is_null())
        msg["tool_calls"] = tool_calls;
    if (reasoning)
        msg["reasoning_content"] = reasoning;
    return {{"id", "chatcmpl-1"},
            {"object", "chat.completion"},
            {"created", 123},
            {"model", "m"},
            {"choices", json::array({{{"index", 0}, {"message", msg}, {"finish_reason", finish}}})},
            {"usage",
             {{"prompt_tokens", 10},
              {"completion_tokens", 5},
              {"total_tokens", 15},
              {"prompt_tokens_details", {{"cached_tokens", 4}}}}}};
}

// C-6: the per-request speculation counters ride from the chat usage into
// output_tokens_details under the same vendor-prefixed keys.
TEST(ResponsesTransform, SpecCountersPassThrough) {
    json oai = make_oai("hi");
    oai["usage"]["completion_tokens_details"] = {
        {"reasoning_tokens", 2}, {"imp_spec_drafted", 40}, {"imp_spec_accepted", 31}, {"imp_spec_verify_steps", 5}};
    json out = openai_to_responses_response(oai, "m", "resp_s");
    const json& d = out["usage"]["output_tokens_details"];
    EXPECT_EQ(d["reasoning_tokens"], 2);
    EXPECT_EQ(d["imp_spec_drafted"], 40);
    EXPECT_EQ(d["imp_spec_accepted"], 31);
    EXPECT_EQ(d["imp_spec_verify_steps"], 5);
    // Absent upstream stays absent downstream.
    json plain = openai_to_responses_response(make_oai("hi"), "m", "resp_p");
    EXPECT_FALSE(plain["usage"]["output_tokens_details"].contains("imp_spec_drafted"));
}

TEST(ResponsesTransform, TextResponse) {
    json out = openai_to_responses_response(make_oai("hi there"), "m", "resp_x");
    EXPECT_EQ(out["id"], "resp_x");
    EXPECT_EQ(out["object"], "response");
    EXPECT_EQ(out["status"], "completed");
    ASSERT_EQ(out["output"].size(), 1u);
    const json& item = out["output"][0];
    EXPECT_EQ(item["type"], "message");
    EXPECT_EQ(item["role"], "assistant");
    EXPECT_EQ(item["content"][0]["type"], "output_text");
    EXPECT_EQ(item["content"][0]["text"], "hi there");
    EXPECT_EQ(out["usage"]["input_tokens"], 10);
    EXPECT_EQ(out["usage"]["output_tokens"], 5);
    EXPECT_EQ(out["usage"]["input_tokens_details"]["cached_tokens"], 4);
}

TEST(ResponsesTransform, ToolCallAndReasoningItems) {
    json tcs = json::array({{{"id", "call_9"},
                             {"type", "function"},
                             {"function", {{"name", "f"}, {"arguments", "{\"a\":1}"}}}}});
    json out = openai_to_responses_response(make_oai(nullptr, tcs, "thinking...", "tool_calls"),
                                            "m", "resp_y");
    ASSERT_EQ(out["output"].size(), 2u);
    EXPECT_EQ(out["output"][0]["type"], "reasoning");
    EXPECT_EQ(out["output"][0]["summary"][0]["text"], "thinking...");
    const json& fc = out["output"][1];
    EXPECT_EQ(fc["type"], "function_call");
    EXPECT_EQ(fc["call_id"], "call_9");
    EXPECT_EQ(fc["name"], "f");
    EXPECT_EQ(fc["arguments"], "{\"a\":1}");
    EXPECT_EQ(out["status"], "completed");
}

TEST(ResponsesTransform, LengthFinishBecomesIncomplete) {
    json out = openai_to_responses_response(make_oai("partial", nullptr, nullptr, "length"), "m",
                                            "resp_z");
    EXPECT_EQ(out["status"], "incomplete");
    EXPECT_EQ(out["incomplete_details"]["reason"], "max_output_tokens");
}

// Context lost to StreamingLLM eviction must survive the transform. A caller
// on /v1/responses is exactly as entitled to know its context was truncated
// mid-generation as a caller on chat-completions, and the field is absent
// unless it happened — so its presence is the signal, not its value.
TEST(ResponsesTransform, EvictedTokensForwarded) {
    json oai = make_oai("hi");
    oai["usage"]["prompt_tokens_details"]["evicted_tokens"] = 384;
    json out = openai_to_responses_response(oai, "m", "resp_e");
    EXPECT_EQ(out["usage"]["input_tokens_details"]["evicted_tokens"], 384);
    EXPECT_EQ(out["usage"]["input_tokens_details"]["cached_tokens"], 4);
}

TEST(ResponsesTransform, NoEvictionMeansNoKey) {
    json out = openai_to_responses_response(make_oai("hi"), "m", "resp_n");
    EXPECT_FALSE(out["usage"]["input_tokens_details"].contains("evicted_tokens"));
    // A zero must not be forwarded as a key either — "0 evicted" and "never
    // evicted" would then be indistinguishable from "the field exists".
    json oai0 = make_oai("hi");
    oai0["usage"]["prompt_tokens_details"]["evicted_tokens"] = 0;
    json out0 = openai_to_responses_response(oai0, "m", "resp_z");
    EXPECT_FALSE(out0["usage"]["input_tokens_details"].contains("evicted_tokens"));
}
