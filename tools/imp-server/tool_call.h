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
    bool valid = true;      // false if arguments failed schema validation
    std::string error;      // human-readable reason when !valid
};

// validate_tool_call: checks parsed arguments are a JSON object, all `required` properties are
// present, and present properties roughly match their declared `type`. Sets tc.valid=false +
// tc.error on failure; a no-op when the tool or its schema can't be located.
void validate_tool_call(ParsedToolCall& tc, const json& tools);

std::string build_tool_prompt(imp::ChatTemplateFamily family, const json& tools, const json& tool_choice);

// tool_choice_is_enforceable (#1002): whether the loaded template family can enforce this
// tool_choice through the decode FSM rather than degrade to a prompt sentence (#1592).
// "required" -> CHATML only; {"function":{..}} -> CHATML or LLAMA3 (name-in-tag); "auto"/"none"/
// absent -> always true. ToolChoiceEnforcement.HelperAgreesWithTheCollectors asserts this
// matches the collectors below.
bool tool_choice_is_enforceable(imp::ChatTemplateFamily family, const json& tool_choice);

std::vector<std::pair<std::string, std::string>> collect_tool_constraint(imp::ChatTemplateFamily family,
                                                                         const json& tools,
                                                                         const json& tool_choice);

// collect_strict_tool_constraint (#1002): strict optional tool calling - collects (name, schema)
// for ALL callable tools only when the model is free to choose AND every tool declares
// strict:true with enforceable params. Non-empty only on ChatML; the envelope itself isn't forced.
std::vector<std::pair<std::string, std::string>> collect_strict_tool_constraint(
    imp::ChatTemplateFamily family, const json& tools, const json& tool_choice);

// collect_llama3_forced_tool (#1002): Llama3's body IS the arguments object, so a forced single
// function constrains the bare parameter schema (per-tool envelope), not a TOOL_CALL wrapper.
// Returns {} when not a forced enforceable Llama3 function.
std::pair<std::string, std::string> collect_llama3_forced_tool(imp::ChatTemplateFamily family,
                                                               const json& tools, const json& tool_choice);

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_chatml(
    const std::string& text, std::atomic<int>& next_tool_call_id);

// known_tool_names: the names the request actually offered. Llama 3.2's bare-JSON-object form is
// only treated as a call when its name is one of these - otherwise a model can fabricate a tool
// call nobody asked for (observed on Llama-3.2-3B). Empty list = envelope-form only (pre-2026-07-26).
std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_llama3(
    const std::string& text, std::atomic<int>& next_tool_call_id,
    const std::vector<std::string>& known_tool_names = {});

// Gemma-4's quote-escape sequence. Shared by the dialect parser
// (tool_call_gemma.cpp) and the streaming / re-emission paths (tool_call.cpp).
inline constexpr const char* kGemmaQuote = "<|\"|>";
inline constexpr size_t kGemmaQuoteLen = 5;

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_gemma(
    const std::string& text, std::atomic<int>& next_tool_call_id);

// Harmony (gpt-oss) tool calls are a channel, not a tag
// (<|channel|>commentary to=functions.NAME...<|message|>{args}<|call|>); split_harmony_channels
// does the channel walk, this only turns its recipients into ParsedToolCall.
std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_harmony(
    const std::string& text, std::atomic<int>& next_tool_call_id);

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls(
    imp::ChatTemplateFamily family, const std::string& text, std::atomic<int>& next_tool_call_id,
    const std::vector<std::string>& known_tool_names = {});

// Names of the functions a request offered, for the check above.
std::vector<std::string> tool_names_from_request(const json& tools);

// Parse Qwen3.6's XML-flavored tool-call body (<function=NAME><parameter=K>V
// </parameter>...); exported so the streaming paths can fall back to it when
// json::parse fails on a <tool_call> body. Fills tc.name/tc.arguments only.
bool parse_qwen36_xml_call(const std::string& body, ParsedToolCall& tc);

// Parse a single Gemma-4 tool-call body ("call:NAME{key:value,...}", the
// <|tool_call>/<tool_call|> markers already stripped). Fills tc.name and
// tc.arguments (re-emitted as JSON); tc.id is left for the caller.
bool parse_gemma_tool_call_body(const std::string& body, ParsedToolCall& tc);

// Streaming tool-call tag scanning shared by the OpenAI and Anthropic SSE paths; full state
// machine in tool_stream_filter.h.

// Result of scanning an accumulated buffer (starting at a '<') for a
// tool-call open marker of the given chat-template family.
struct ToolTagScan {
    enum class Kind {
        NONE,     // provably not a tool tag anywhere in the buffer — emit as content
        PARTIAL,  // could still become a tag — keep buffering
        OPEN,     // complete open marker found
    };
    Kind kind = Kind::NONE;
    size_t content_len = 0;      // OPEN: bytes before the open marker (plain content)
    size_t body_start = 0;       // OPEN: offset where the tool-call body begins
    const char* close_tag = "";  // OPEN: expected close marker
    bool gemma_body = false;     // OPEN: body uses the Gemma "call:NAME{...}" syntax
    std::string fn_name;         // OPEN, Llama3: function name from the <function=NAME> tag
};

// Family markers: LLAMA3 -> "<function=NAME>"; GEMMA -> "<|tool_call>" (native)
// or "<tool_call>" (ChatML fallback prompt); everything else -> "<tool_call>".
ToolTagScan scan_tool_tag(const std::string& buf, imp::ChatTemplateFamily family);

// parse_stream_tool_body: body has its close marker already stripped. gemma_body selects Gemma
// syntax; non-empty fn_name means Llama3 (bare JSON args). ChatML tries JSON first, then the
// Qwen3.6 XML layout. Returns false when unparseable (caller restores the raw text to the stream).
bool parse_stream_tool_body(const std::string& body, bool gemma_body, const std::string& fn_name,
                            ParsedToolCall& tc);

// xml (ChatML family only): render prior calls in the Qwen-Coder XML shape
// (<function=NAME><parameter=KEY> raw-text values) the template teaches,
// instead of the ChatML JSON body.
std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family, const json& tool_calls,
                                         const std::string& content, bool xml = false);

std::string format_tool_response(imp::ChatTemplateFamily family, const json& msg);
