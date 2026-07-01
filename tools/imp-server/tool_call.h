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

// Parse Qwen3.6's XML-flavored tool-call body (<function=NAME><parameter=K>V
// </parameter>...); exported so the streaming paths can fall back to it when
// json::parse fails on a <tool_call> body. Fills tc.name/tc.arguments only.
bool parse_qwen36_xml_call(const std::string& body, ParsedToolCall& tc);

// Parse a single Gemma-4 tool-call body ("call:NAME{key:value,...}", the
// <|tool_call>/<tool_call|> markers already stripped). Fills tc.name and
// tc.arguments (re-emitted as JSON); tc.id is left for the caller.
bool parse_gemma_tool_call_body(const std::string& body, ParsedToolCall& tc);

// ---------------------------------------------------------------------------
// Streaming tool-call tag scanning (shared by the OpenAI + Anthropic SSE
// paths; see tool_stream_filter.h for the full state machine).
// ---------------------------------------------------------------------------

// Result of scanning an accumulated buffer (starting at a '<') for a
// tool-call open marker of the given chat-template family.
struct ToolTagScan {
    enum class Kind {
        NONE,     // provably not a tool tag anywhere in the buffer — emit as content
        PARTIAL,  // could still become a tag — keep buffering
        OPEN,     // complete open marker found
    };
    Kind kind = Kind::NONE;
    size_t content_len = 0;     // OPEN: bytes before the open marker (plain content)
    size_t body_start = 0;      // OPEN: offset where the tool-call body begins
    const char* close_tag = ""; // OPEN: expected close marker
    bool gemma_body = false;    // OPEN: body uses the Gemma "call:NAME{...}" syntax
    std::string fn_name;        // OPEN, Llama3: function name from the <function=NAME> tag
};

// Family markers: LLAMA3 -> "<function=NAME>"; GEMMA -> "<|tool_call>" (native)
// or "<tool_call>" (ChatML fallback prompt); everything else -> "<tool_call>".
ToolTagScan scan_tool_tag(const std::string& buf, imp::ChatTemplateFamily family);

// Parse a completed streaming tool-call body (close marker already stripped).
// gemma_body selects the Gemma syntax; a non-empty fn_name means Llama3 (name
// came from the open tag, body is the bare JSON args). The ChatML path tries
// JSON first and falls back to the Qwen3.6 XML layout. Returns false when the
// body cannot be parsed (caller should restore the raw text to the stream).
bool parse_stream_tool_body(const std::string& body, bool gemma_body, const std::string& fn_name,
                            ParsedToolCall& tc);

std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family, const json& tool_calls,
                                         const std::string& content);

std::string format_tool_response(imp::ChatTemplateFamily family, const json& msg);
