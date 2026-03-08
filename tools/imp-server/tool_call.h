#pragma once

#include "model/chat_template.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

struct ParsedToolCall {
    std::string id;         // "call_imp_0", "call_imp_1", ...
    std::string name;       // Function name
    std::string arguments;  // JSON string
};

std::string build_tool_prompt(imp::ChatTemplateFamily family,
                              const json& tools,
                              const json& tool_choice);

std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls_chatml(const std::string& text, std::atomic<int>& next_tool_call_id);

std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls_llama3(const std::string& text, std::atomic<int>& next_tool_call_id);

std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls(imp::ChatTemplateFamily family, const std::string& text,
                 std::atomic<int>& next_tool_call_id);

std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family,
                                         const json& tool_calls,
                                         const std::string& content);

std::string format_tool_response(imp::ChatTemplateFamily family,
                                 const json& msg);
