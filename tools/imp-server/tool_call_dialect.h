#pragma once

// Tool-call dialect registry (#2464): one ToolCallDialect per wire format (prompt, parser, stream
// scanner, reconstruct, tool response, constraint envelope); tool_call.h entry points dispatch here.
// New dialect = its descriptor file + one row in kRegistry (tool_call_dialect.cpp).

#include "tool_call.h"
#include "runtime/tool_call_gate.h"

#include <atomic>
#include <string>
#include <utility>
#include <vector>

using ToolCallParseFn = std::pair<std::string, std::vector<ParsedToolCall>> (*)(
    const std::string& text, std::atomic<int>& next_tool_call_id,
    const std::vector<std::string>& known_tool_names);
// Tool-definition block of the system prompt; the tool_choice sentence is appended by the caller.
using ToolPromptFn = std::string (*)(const json& tools);
// Streaming open-marker scan over a buffer that starts at a '<'.
using ToolScanFn = ToolTagScan (*)(const std::string& buf);
// Appends one prior assistant call (args = JSON text) in the dialect's wire format.
using ToolRenderCallFn = void (*)(std::string& out, const std::string& name, const std::string& args,
                                  bool xml);
// role=tool message text (content already flattened to a string).
using ToolResponseFn = std::string (*)(const json& msg, const std::string& content);
// Literals around a forced function's bare parameter schema (the envelope names the function).
using ToolForcedEnvelopeFn = void (*)(const std::string& name, std::string& open, std::string& close);

struct ToolCallDialect {
    const char* name;
    ToolPromptFn prompt;
    ToolCallParseFn parse;
    ToolScanFn scan;
    ToolRenderCallFn render_call;
    ToolResponseFn tool_response;  // nullptr = content verbatim
    // Non-null: tool_choice required / forced / strict constrain {"name","arguments"} between these.
    const char* json_envelope_open;
    const char* json_envelope_close;
    bool xml_body;  // json envelope may carry the Qwen-Coder XML body (template's tool_xml_dialect)
    ToolForcedEnvelopeFn forced_envelope;  // nullptr = forced function not enforceable
    const char* native_token;              // non-null: forced_envelope needs this vocab token
    bool tool_response_joins_assistant;    // template renders role=tool only inside the assistant turn
    const char* gate_open;                 // constraint preamble gate (imp::ToolCallGate)
    const char* gate_close;
    bool gate_tokens;
};

const ToolCallDialect& tool_call_dialect(imp::ChatTemplateFamily family);
imp::ToolCallGate tool_call_gate(imp::ChatTemplateFamily family);

// Shared building blocks for dialect descriptors.
std::string chatml_tool_prompt(const json& tools);
void chatml_render_call(std::string& out, const std::string& name, const std::string& args, bool xml);
// <tool_call> scan; gemma_native also opens on <|tool_call> with a Gemma "call:NAME{...}" body.
ToolTagScan scan_tool_call_tag(const std::string& buf, bool gemma_native);
// True iff some suffix of buf is a proper prefix of marker m (more bytes could complete it).
bool suffix_is_marker_prefix(const std::string& buf, const char* m, size_t mlen);
