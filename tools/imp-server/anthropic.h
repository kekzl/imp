// Anthropic Messages API (/v1/messages) <-> OpenAI Chat Completions transforms.
// Allows imp-server to expose a Claude-compatible endpoint by reusing the
// existing OpenAI code path for generation.

#pragma once

#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace imp_server::anthropic {

using json = nlohmann::json;

// Transforms an Anthropic /v1/messages request body into an equivalent OpenAI
// /v1/chat/completions body, feedable back into handle_chat_completions unchanged. Throws
// nothing; malformed fields default to OpenAI-sensible values.
json anthropic_to_openai_body(const json& anth);

// output_config.effort levels against the loaded template's reasoning_effort list: the
// level itself, else the next higher one listed, else the highest listed. An unranked
// value or an empty list returns `want`.
std::string nearest_reasoning_effort(const std::string& want, const std::vector<std::string>& values);

// Inverse: translates an OpenAI non-streaming chat.completion response into an Anthropic
// Messages response. anth_model is used verbatim (the name the client sent, not a resolved
// alias). stop_sequence names the matched text when OpenAI's finish_reason="stop" meant a
// stop sequence rather than end_turn (#1550), read off the shim (g_shim_stop_sequence) or the
// stream loop result. omit_thinking drops the thinking block for thinking.display:"omitted"
// (#1560); the model still reasons, the client just isn't shown it.
json openai_to_anthropic_response(const json& oai, const std::string& anth_model,
                                  const std::string& stop_sequence = {}, bool omit_thinking = false);

// Anthropic's three prompt fields are disjoint: input + cache_read + cache_creation == prompt (#3055).
struct PromptUsage {
    int input = 0;
    int cache_read = 0;
    int cache_creation = 0;
};

// `cached` and `creation` count from the same prompt: each is clamped to what is left of it.
inline PromptUsage split_prompt_usage(int prompt_tokens, int cached, int creation) {
    PromptUsage u;
    u.cache_read = std::clamp(cached, 0, prompt_tokens);
    u.cache_creation = std::clamp(creation, 0, prompt_tokens - u.cache_read);
    u.input = prompt_tokens - u.cache_read - u.cache_creation;
    return u;
}

// The `signature` a thinking block carries: Anthropic SDKs round-trip and expect the field;
// imp cannot attest anything, so this is a deterministic digest of the block text (#1555),
// proving only that it came back unedited.
std::string thinking_signature(const std::string& thinking);

// True when the request asked for `thinking.display: "omitted"` or did not opt into thinking
// (enabled/adaptive). Display is not a generation setting, so it is read off the request on the
// way out rather than transformed into the OpenAI body.
bool thinking_display_omitted(const json& anth_body);

// OpenAI finish_reason -> Anthropic stop_reason, in one place: the streaming and
// non-streaming paths had two copies that disagreed (streaming passed "capacity" straight
// through, not in Anthropic's enum, #1552; neither produced "stop_sequence", #1550).
const char* anthropic_stop_reason(const std::string& openai_finish, bool stop_sequence_matched);

// Generate an Anthropic message id (msg_XXXX). Uses the same atomic counter
// semantics as make_completion_id; callers pass in a fresh integer.
std::string make_message_id(uint64_t counter);

// Generate an Anthropic tool_use id (toolu_XXXX) from an internal tool-call id.
// If the source id already has an anthropic-style prefix, pass it through.
std::string tool_call_id_to_anthropic(const std::string& openai_id);

}  // namespace imp_server::anthropic
