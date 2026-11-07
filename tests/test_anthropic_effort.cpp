// /v1/messages output_config.effort: the body field, and the level the loaded template gets.
#include "anthropic.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

using imp_server::anthropic::anthropic_to_openai_body;
using imp_server::anthropic::nearest_reasoning_effort;
using json = nlohmann::json;

namespace {
json base_request() {
    return json{{"model", "claude-x"},
                {"max_tokens", 64},
                {"messages", json::array({json{{"role", "user"}, {"content", "hi"}}})}};
}
}  // namespace

TEST(AnthropicEffort, OutputConfigEffortReachesTheBody) {
    json req = base_request();
    req["output_config"] = json{{"effort", "low"}};
    EXPECT_EQ(anthropic_to_openai_body(req).value("reasoning_effort", ""), "low");
    EXPECT_FALSE(anthropic_to_openai_body(base_request()).contains("reasoning_effort"));
    req["output_config"] = json{{"effort", 3}};
    EXPECT_FALSE(anthropic_to_openai_body(req).contains("reasoning_effort"));
    req["output_config"] = "low";
    EXPECT_FALSE(anthropic_to_openai_body(req).contains("reasoning_effort"));
}

TEST(AnthropicEffort, NearestLevelInTheTemplateList) {
    const std::vector<std::string> qwen = {"xhigh", "medium", "low"};
    EXPECT_EQ(nearest_reasoning_effort("low", qwen), "low");
    EXPECT_EQ(nearest_reasoning_effort("medium", qwen), "medium");
    EXPECT_EQ(nearest_reasoning_effort("high", qwen), "xhigh");
    EXPECT_EQ(nearest_reasoning_effort("xhigh", qwen), "xhigh");
    EXPECT_EQ(nearest_reasoning_effort("max", qwen), "xhigh");
    const std::vector<std::string> three = {"low", "medium", "high"};
    EXPECT_EQ(nearest_reasoning_effort("max", three), "high");
    EXPECT_EQ(nearest_reasoning_effort("xhigh", three), "high");
    EXPECT_EQ(nearest_reasoning_effort("low", {"medium", "high"}), "medium");
    EXPECT_EQ(nearest_reasoning_effort("high", {}), "high");
    EXPECT_EQ(nearest_reasoning_effort("weird", qwen), "weird");
    EXPECT_EQ(nearest_reasoning_effort("high", {"custom"}), "high");
}
