// OpenAI Responses API <-> Chat Completions transform (same shim pattern as anthropic.h). Scope
// v1, stateless: input (string or item array: message/function_call/function_call_output;
// reasoning items skipped), instructions, flat tools+tool_choice, text.format, temperature/top_p/
// max_output_tokens, reasoning.effort->think_budget, stream. previous_response_id/store=true rejected.

#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace imp_server::responses {

using json = nlohmann::json;

// responses_to_openai_body: throws std::invalid_argument with a client-facing message on
// unsupported fields (previous_response_id, store=true, non-text input parts).
json responses_to_openai_body(const json& rsp);

// Inverse: translate a non-streaming chat.completion response into a
// Responses API `response` object. `req_model` is echoed verbatim;
// `response_id` is the caller-generated resp_... id.
json openai_to_responses_response(const json& oai, const std::string& req_model,
                                  const std::string& response_id);

// id helpers (resp_/msg_/fc_ prefixes, hex counter — same shape the OpenAI
// SDKs expect to treat as opaque strings).
std::string make_response_id(uint64_t counter);
std::string make_item_id(const char* prefix, uint64_t counter);

}  // namespace imp_server::responses
