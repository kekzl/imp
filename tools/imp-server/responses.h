// OpenAI Responses API <-> Chat Completions transform (same shim pattern as anthropic.h). Scope
// v1: input (string or item array: message/function_call/function_call_output;
// reasoning items skipped), instructions, flat tools+tool_choice, text.format, temperature/top_p/
// max_output_tokens, reasoning.effort->think_budget, stream. store / previous_response_id:
// resolved by the handler against responses_store.h (#2206).

#pragma once

#include <nlohmann/json.hpp>
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

// Inverse: translate a non-streaming chat.completion response into a
// Responses API `response` object. `req_model` is echoed verbatim;
// `response_id` is the caller-generated resp_... id.
json openai_to_responses_response(const json& oai, const std::string& req_model,
                                  const std::string& response_id);

// id helpers (resp_/msg_/fc_ prefixes, hex counter — same shape the OpenAI
// SDKs expect to treat as opaque strings).
// Response ids append 64 random bits: GET /v1/responses/{id} serves stored content, so ids
// must not be enumerable from the counter.
std::string make_response_id(uint64_t counter);
std::string make_item_id(const char* prefix, uint64_t counter);

}  // namespace imp_server::responses
