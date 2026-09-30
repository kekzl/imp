#pragma once

// Prompt logprobs on /v1/completions (#2207): request parsing and the two response shapes.
// vLLM `prompt_logprobs`: one entry per prompt token, entry 0 null. OpenAI `echo` + `logprobs`:
// prompt tokens precede completion tokens in choices[].logprobs, first token_logprobs null.

#include <nlohmann/json.hpp>

#include "runtime/request.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct PromptLogprobsRequest {
    int prompt_logprobs = -1;    // vLLM field, -1 = absent
    bool echo_logprobs = false;  // echo && logprobs
    int echo_top = -1;           // alternatives per echoed prompt token
    int engine_top_n = -1;       // imp::Request::prompt_logprobs
};

// "" or the 400 message. prompt_logprobs: integer in [0, kMaxPromptLogprobs]; both forms refuse stream.
std::string parse_prompt_logprobs(const nlohmann::json& body, bool stream, bool echo, bool req_logprobs,
                                  int top_logprobs, PromptLogprobsRequest& out);

// /v1/completions `logprobs` (integer top-N, or the Chat boolean) and `top_logprobs` (clamped to
// 0..20), then parse_prompt_logprobs. "" or the 400 message.
std::string parse_completions_logprobs(const nlohmann::json& body, bool stream, bool echo, bool& req_logprobs,
                                       int& top_logprobs, PromptLogprobsRequest& plp);

using TokenTextFn = std::function<std::string(int32_t)>;

// True when the engine scored every prompt row (n_prompt - 1 rows).
bool prompt_logprobs_complete(const imp::PromptLogprobs& p, size_t n_prompt);

// vLLM shape: [null, {"<id>": {"logprob", "rank", "decoded_token"}}, ...]; prompt token plus top `limit`.
nlohmann::json vllm_prompt_logprobs_json(const std::vector<int32_t>& prompt, const imp::PromptLogprobs& p,
                                         int limit, const TokenTextFn& text);

// OpenAI Completions shape over prompt then completion tokens; text_offset is a byte offset into
// full_text (= prompt text + completion text). Prompt entries carry top `limit` alternatives.
nlohmann::json echo_completions_logprobs_json(const std::vector<int32_t>& prompt,
                                              const imp::PromptLogprobs& p, int limit,
                                              const TokenTextFn& text,
                                              const std::vector<imp::TokenLogprobInfo>& out, size_t out_limit,
                                              const std::string& full_text);

// Writes choice["prompt_logprobs"] (vLLM) and/or choice["logprobs"] (echo shape) from req.prompt_lp.
// false = the engine did not score every prompt row: the caller answers 500, never a partial list.
bool attach_prompt_logprobs(nlohmann::json& choice, const PromptLogprobsRequest& plp, const imp::Request& req,
                            const TokenTextFn& text, size_t out_limit, const std::string& full_text);
