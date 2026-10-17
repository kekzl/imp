// Assistant prefill on the wire transforms: /v1/messages sets continue_final_message for a
// trailing assistant turn with text; /v1/responses forwards the field.

#include "anthropic.h"
#include "responses.h"

#include <gtest/gtest.h>

namespace {

using imp_server::anthropic::anthropic_to_openai_body;
using imp_server::responses::responses_to_openai_body;
using json = nlohmann::json;

json messages_request() {
    return json{
        {"model", "claude-x"},
        {"max_tokens", 64},
        {"messages", json::array({json{{"role", "user"}, {"content", "hi"}}})},
    };
}

TEST(AnthropicPrefill, TrailingAssistantTextSetsContinueFinal) {
    json req = messages_request();
    req["messages"].push_back(json{{"role", "assistant"}, {"content", "The answer is"}});
    EXPECT_TRUE(anthropic_to_openai_body(req).value("continue_final_message", false));
    req["messages"].back()["content"] = json::array({json{{"type", "text"}, {"text", "The answer is"}}});
    EXPECT_TRUE(anthropic_to_openai_body(req).value("continue_final_message", false));
}

TEST(AnthropicPrefill, NoPrefillWithoutAssistantText) {
    EXPECT_FALSE(anthropic_to_openai_body(messages_request()).contains("continue_final_message"));
    json req = messages_request();
    req["messages"].push_back(json{{"role", "assistant"}, {"content", " \n"}});
    EXPECT_FALSE(anthropic_to_openai_body(req).contains("continue_final_message"));
    req["messages"].back()["content"] = json::array(
        {json{{"type", "tool_use"}, {"id", "t1"}, {"name", "f"}, {"input", json::object()}}});
    EXPECT_FALSE(anthropic_to_openai_body(req).contains("continue_final_message"));
}

TEST(ResponsesPrefill, ContinueFinalMessageForwarded) {
    json body = {{"model", "m"},
                 {"input", json::array({{{"role", "user"}, {"content", "q"}},
                                        {{"role", "assistant"}, {"content", "The answer is"}}})},
                 {"continue_final_message", true}};
    EXPECT_TRUE(responses_to_openai_body(body).value("continue_final_message", false));
    body.erase("continue_final_message");
    EXPECT_FALSE(responses_to_openai_body(body).contains("continue_final_message"));
}

}  // namespace
