#pragma once

#include "model/chat_template.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

struct ParsedToolCall {
    std::string id;            // "call_imp_0", "call_imp_1", ...
    std::string name;          // Function name
    std::string arguments;     // JSON string
    bool valid = true;         // false if arguments failed schema validation
    std::string error;         // human-readable reason when !valid
};

// Validate a tool call's parsed arguments against the matching tool's JSON
// schema (the OpenAI `tools[].function.parameters` object). Checks that the
// arguments parse as a JSON object, that all `required` properties are
// present, and that present top-level properties roughly match their declared
// `type`. On failure sets tc.valid=false and tc.error. Self-contained: does
// not depend on the engine-side constraint/FSM code. A no-op (leaves the call
// valid) when the tool or its schema can't be located.
void validate_tool_call(ParsedToolCall& tc, const json& tools);

std::string build_tool_prompt(imp::ChatTemplateFamily family, const json& tools, const json& tool_choice);

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_chatml(
    const std::string& text, std::atomic<int>& next_tool_call_id);

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_llama3(
    const std::string& text, std::atomic<int>& next_tool_call_id);

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_gemma(
    const std::string& text, std::atomic<int>& next_tool_call_id);

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls(imp::ChatTemplateFamily family,
                                                                     const std::string& text,
                                                                     std::atomic<int>& next_tool_call_id);

std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family, const json& tool_calls,
                                         const std::string& content);

std::string format_tool_response(imp::ChatTemplateFamily family, const json& msg);
