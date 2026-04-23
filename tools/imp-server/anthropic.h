// Anthropic Messages API (/v1/messages) <-> OpenAI Chat Completions transforms.
// Allows imp-server to expose a Claude-compatible endpoint by reusing the
// existing OpenAI code path for generation.

#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace imp_server::anthropic {

using json = nlohmann::json;

// Transform an Anthropic /v1/messages request body into an equivalent
// OpenAI /v1/chat/completions body. The returned body can be fed back
// into handle_chat_completions unchanged.
//
// Throws nothing; malformed fields default to OpenAI-sensible values.
json anthropic_to_openai_body(const json& anth);

// Inverse: translate an OpenAI non-streaming chat.completion response
// into an Anthropic Messages response.
//
// `anth_model` is the model name passed in the original Anthropic request
// (used verbatim in the response — Anthropic clients expect the name they
// sent, not whatever alias the OpenAI side resolved to).
json openai_to_anthropic_response(const json& oai, const std::string& anth_model);

// Generate an Anthropic message id (msg_XXXX). Uses the same atomic counter
// semantics as make_completion_id; callers pass in a fresh integer.
std::string make_message_id(uint64_t counter);

// Generate an Anthropic tool_use id (toolu_XXXX) from an internal tool-call id.
// If the source id already has an anthropic-style prefix, pass it through.
std::string tool_call_id_to_anthropic(const std::string& openai_id);

}  // namespace imp_server::anthropic
