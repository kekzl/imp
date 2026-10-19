// OpenAI Responses API <-> Chat Completions transform (same shim pattern as anthropic.h). Scope
// v1: input (string or item array: message/function_call/custom_tool_call and their outputs;
// reasoning items skipped), instructions, flat tools (function, custom, namespace; hosted dropped)
// +tool_choice, text.format, temperature/top_p/
// max_output_tokens, reasoning.effort->think_budget, stream. store / previous_response_id:
// resolved by the handler against responses_store.h (#2206).

#pragma once

#include <nlohmann/json.hpp>
#include <map>
#include <set>
#include <string>

namespace imp_server::responses {

using json = nlohmann::json;

// responses_to_openai_body: throws std::invalid_argument with a client-facing message on
// unsupported fields (unresolved previous_response_id, non-text input parts).
json responses_to_openai_body(const json& rsp);

// `input` as an item array: a bare string becomes one user message, absent becomes [].
json normalize_input_items(const json& input);

// previous_response_id: stored input + stored output + new input, one flat item array. Fed
// through responses_to_openai_body it yields the messages a stateless resend would.
json continue_conversation(const json& prev_input_items, const json& prev_output_items,
                           const json& new_input);

// Request tools that change the shape of a call item: custom (freeform) tool names, and the
// namespace each namespaced tool came from (Codex routes a call by name + namespace).
struct ToolKinds {
    std::set<std::string> custom;
    std::map<std::string, std::string> ns;
};
ToolKinds tool_kinds(const json& rsp);

// custom tool call arguments {"input": "..."} -> the raw input; anything else verbatim.
std::string custom_tool_input(const std::string& arguments);

// One output item for a tool call: custom_tool_call {input} for a custom tool, else
// function_call {arguments}; `namespace` set when the tool came from one.
json tool_call_item(const ToolKinds& kinds, const std::string& item_id, const std::string& call_id,
                    const std::string& name, const std::string& arguments, const char* status);

// Inverse: translate a non-streaming chat.completion response into a
// Responses API `response` object. `req_model` is echoed verbatim;
// `response_id` is the caller-generated resp_... id.
json openai_to_responses_response(const json& oai, const std::string& req_model,
                                  const std::string& response_id, const ToolKinds& kinds = {});

// id helpers (resp_/msg_/fc_ prefixes, hex counter — same shape the OpenAI
// SDKs expect to treat as opaque strings).
// Response ids append 64 random bits: GET /v1/responses/{id} serves stored content, so ids
// must not be enumerable from the counter.
std::string make_response_id(uint64_t counter);
std::string make_item_id(const char* prefix, uint64_t counter);

}  // namespace imp_server::responses
